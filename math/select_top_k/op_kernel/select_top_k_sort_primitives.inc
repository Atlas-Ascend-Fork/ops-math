/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 来源：R20 已验证的 TopKMerge 排序、位编码和流式归并指令。
// 不包含 R20 分核、启发式选路或跨核逐层同步；由 基础调度复用。
    // 流式加载并归并多路有序段，输出指定数量的结果。
    template <bool FinalOutput = false>
    __aicore__ inline void Merge(const uint64_t starts[MERGE_WAYS], const uint64_t lengths[MERGE_WAYS], uint32_t routes,
                                 uint64_t dstBase, uint64_t target,
                                 uint64_t outputRow = 0)
    {
        uint64_t pos[MERGE_WAYS] = {0, 0, 0, 0};
        uint32_t h[MERGE_WAYS] = {0, 0, 0, 0};
        uint64_t written = 0;
        bool outputInFlight = false;
        // 仅补充耗尽窗口；每轮按硬件实际消耗推进各路，达到目标 K 后停止。
        while (written < target) {
            uint32_t active[MERGE_WAYS] = {0, 0, 0, 0};
            uint32_t q = 0;
            for (uint32_t i = 0; i < routes; ++i) {
                if (pos[i] == lengths[i]) {
                    continue;
                }
                if (!h[i]) {
                    h[i] = Min<uint64_t>(d_->mergeWindow, lengths[i] - pos[i]);
                    DataCopyExtParams load{1, h[i] * PAIR_BYTES, 0, 0, 0};
                    DataCopyPad(windows_[i], gm_[srcRole_][(starts[i] + pos[i]) * PAIR_COMPONENTS], load,
                                DataCopyPadExtParams<float>{false, 0, 0, 0});
                }
                active[q++] = i;
            }
            Fence<HardEvent::MTE2_V>();
            uint32_t consumed[MERGE_WAYS], m = 0;
            if (outputInFlight) {
                // 窗口 DMA 可与上次归并结果的写回重叠；下次归并前等待输出缓冲释放。
                Fence<HardEvent::MTE3_V>();
                outputInFlight = false;
            }
            auto chunk = MergeWindow(windows_, merged_, active, q, h, consumed, m);
            uint32_t keep = Min<uint64_t>(m, target - written);
            if constexpr (FinalOutput) {
                if (q == 1) {
                    // Emit 会复用输入窗口；将唯一剩余的有序段保存在独立归并输出区。
                    DataCopy(merged_.ReinterpretCast<uint32_t>(), chunk.template ReinterpretCast<uint32_t>(),
                             static_cast<uint32_t>(DivUp(keep * PAIR_COMPONENTS, FLOAT_BLOCK_ELEMENTS) *
                                                   FLOAT_BLOCK_ELEMENTS));
                    PipeBarrier<PIPE_V>();
                    chunk = merged_;
                }
                for (uint32_t start = 0; start < keep; start += d_->mergeWindow) {
                    const uint32_t count = Min<uint32_t>(d_->mergeWindow, keep - start);
                    Emit(outputRow, dstBase + written + start, chunk[start * PAIR_COMPONENTS], count);
                }
            } else {
                if (q > 1) {
                    DataCopyExtParams store{1, keep * PAIR_BYTES, 0, 0, 0};
                    Fence<HardEvent::V_MTE3>();
                    DataCopyPad(gm_[1 - srcRole_][(dstBase + written) * PAIR_COMPONENTS], chunk, store);
                    outputInFlight = true;
                } else {
                    StoreToGm(gm_[1 - srcRole_][(dstBase + written) * PAIR_COMPONENTS], chunk, keep * PAIR_COMPONENTS);
                }
            }
            written += keep;
            for (uint32_t i = 0; i < routes; ++i) {
                pos[i] += consumed[i];
                h[i] -= consumed[i];
                if constexpr (FinalOutput) {
                    h[i] = 0; // Emit 可能覆盖尚未消耗的窗口。
                } else if (written < target && h[i] && consumed[i]) {
                    h[i] = 0; // 从已推进的 GM 位置重新填充窗口。
                }
            }
        }
        if (outputInFlight)
            Fence<HardEvent::MTE3_S>();
    }

    // 加载并排序一个输入 tile，生成带原轴索引的有序 pair。
    __aicore__ inline LocalTensor<float> SortTile(uint64_t row, uint32_t start, uint32_t valid)
    {
        uint32_t padded = static_cast<uint32_t>(DivUp(valid, SORT32_ELEMENTS) * SORT32_ELEMENTS);
        const uint32_t absoluteStart = start;
        DataCopyExtParams load{1, static_cast<uint32_t>(valid * sizeof(T)), 0, 0, 0};
        DataCopyPad(raw_, x_[row * d_->axis + absoluteStart], load, DataCopyPadExtParams<T>{false, 0, 0, 0});
        // 先初始化填充值，避免对未对齐短尾单独执行 Duplicate。
        if (valid != padded) {
            Duplicate(key_, SortPadding(), padded);
            PipeBarrier<PIPE_V>();
        }
        CreateLinearIndex(index_.ReinterpretCast<int32_t>(), absoluteStart, padded);
        Fence<HardEvent::MTE2_V>();
        EncodeSortKeys(key_, raw_, valid, d_->largest);
        PipeBarrier<PIPE_V>();
        Sort32(ping_, key_, index_, Min<uint32_t>(padded / SORT32_ELEMENTS, VECTOR_MAX_REPEATS));
        if (padded > SORT32_MAX_ELEMENTS) {
            Sort32(ping_[SORT32_MAX_ELEMENTS * PAIR_COMPONENTS], key_[SORT32_MAX_ELEMENTS], index_[SORT32_MAX_ELEMENTS],
                   1);
        }
        PipeBarrier<PIPE_V>();
        LocalTensor<float> src = ping_, dst = pong_;
        for (uint32_t span = SORT32_ELEMENTS; span < padded; span *= MERGE_WAYS) {
            MergeTileStage(dst, src, padded, span);
            auto tmp = src;
            src = dst;
            dst = tmp;
        }
        return src;
    }
    // 解码有序 pair 并向 GM 输出值和索引。
    __aicore__ inline void Emit(uint64_t row, uint64_t offset, const LocalTensor<float>& pairs, uint32_t count)
    {
        uint64_t selectedCount = 0;
        GatherMaskParams params(1, 1, VECTOR_REPEAT_BLOCKS, 0);
        GatherMask(key_, pairs, uint8_t(1), true, count * PAIR_COMPONENTS, params, selectedCount);
        // 驻留归并的键和索引目的区独立，两个GatherMask只读同一pair，可一起等待。
        if constexpr (!(Storage == MergeStorage::ALL_CANDIDATES || Storage == MergeStorage::GROUPED_CANDIDATES))
            PipeBarrier<PIPE_V>();
        GatherMask(index_, pairs.ReinterpretCast<uint32_t>(), uint8_t(GATHER_INDEX_PATTERN), true,
                   count * PAIR_COMPONENTS, params, selectedCount);
        PipeBarrier<PIPE_V>();
        if (!d_->largest) {
            Adds(key_.template ReinterpretCast<int32_t>(), key_.template ReinterpretCast<int32_t>(), INT32_MIN, count);
            // FP32直接输出键，后面的V_S等待即可；窄类型Cast仍依赖修改后的键。
            if constexpr (!(Storage == MergeStorage::ALL_CANDIDATES || Storage == MergeStorage::GROUPED_CANDIDATES) || !std::is_same<T, float>::value)
                PipeBarrier<PIPE_V>();
        }
        LocalTensor<T> output = raw_;
        if constexpr (std::is_same<T, float>::value) {
            output = key_.template ReinterpretCast<T>();
        } else {
            Cast(output, key_, RoundMode::CAST_RINT, count);
            if constexpr (!(Storage == MergeStorage::ALL_CANDIDATES || Storage == MergeStorage::GROUPED_CANDIDATES))
                PipeBarrier<PIPE_V>();
        }
        // RestoreNaNPayload自身的V_S/S_V等待保留，确保位恢复及其标量写入完成。
        RestoreNaNPayload<T>(output, key_, index_, x_.GetPhyAddr(), row * d_->axis, count, d_->largest);
        DataCopyExtParams valuesCopy{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        if constexpr (WideIndices) {
            // 归并结果写出时：Gather 已消耗 pair 输入，可复用 ping 存放 INT64 索引，保持值和 INT32 索引独立。
            auto wide = ping_.template ReinterpretCast<int64_t>();
            Cast(wide, index_.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, count);
            // 驻留路径的V_MTE3覆盖完整向量结果，无需在它之前重复等待PIPE_V。
            if constexpr (!(Storage == MergeStorage::ALL_CANDIDATES || Storage == MergeStorage::GROUPED_CANDIDATES))
                PipeBarrier<PIPE_V>();
            DataCopyExtParams indexCopy{1, count * INDEX64_BYTES, 0, 0, 0};
            Fence<HardEvent::V_MTE3>();
            DataCopyPad(values_[row * d_->k + offset], output, valuesCopy);
            DataCopyPad(indices64_[row * d_->k + offset], wide, indexCopy);
        } else {
            DataCopyExtParams indexCopy{1, count * INDEX32_BYTES, 0, 0, 0};
            Fence<HardEvent::V_MTE3>();
            DataCopyPad(values_[row * d_->k + offset], output, valuesCopy);
            DataCopyPad(indices_[row * d_->k + offset], index_.ReinterpretCast<int32_t>(), indexCopy);
        }
        Fence<HardEvent::MTE3_S>();
    }
