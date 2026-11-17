/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SELECT_TOP_K_BASE_MERGE_H
#define SELECT_TOP_K_BASE_MERGE_H
#include "select_top_k_base_common.h"
#include "select_top_k_base_index.h"
#include "select_top_k_base_tiling_data.h"
#include "select_top_k_base_select_layout.h"
#include <type_traits>

namespace SelectTopKBase {
// 区间窗口归并：输出有效项及 INT32 原轴索引，扩宽和填充由统一收尾阶段完成。
template <typename T>
class TopKMerge {
public:
    // 绑定每行有效边界；缺省端点分别使用轴起点和轴长度。
    __aicore__ inline void SetBounds(GM_ADDR begin, GM_ADDR end, uint32_t flags)
    {
        begin_.SetGlobalBuffer((__gm__ int32_t*)begin);
        end_.SetGlobalBuffer((__gm__ int32_t*)end);
        boundsFlags_ = flags;
    }

    // 绑定 GM 张量，依据通用布局建立排序、归并和输出缓冲。
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices, GM_ADDR workspace,
                                const SelectTopKBaseTilingData* data, TPipe* pipe)
    {
        d_ = data;
        x_.SetGlobalBuffer((__gm__ T*)x);
        values_.SetGlobalBuffer((__gm__ T*)values);
        indices_.SetGlobalBuffer((__gm__ int32_t*)indices);
        if (d_->route == ROUTE_EMPTY) {
            return;
        }
        // 共用 arena，使 Gather 能保留目标前缀并追加归并结果；2201 不依赖 Scatter。
        pipe->InitBuffer(arenaBuf_, d_->ubBytes);
        const GeneralMergeLayout layout(d_->tile, d_->mergeWindow, d_->k, sizeof(T), d_->route);
        raw_ = View<T>(layout.raw, d_->tile);
        index_ = View<uint32_t>(layout.index, d_->tile);
        if (d_->route == ROUTE_COPY) {
            return;
        }
        key_ = View<float>(layout.key, d_->tile);
        ping_ = View<float>(layout.ping, d_->tile * PAIR_COMPONENTS);
        pong_ = View<float>(layout.pong, d_->tile * PAIR_COMPONENTS);
        for (uint32_t i = 0; i < MERGE_WAYS; ++i) {
            windows_[i] = View<float>(layout.windows + i * d_->mergeWindow * PAIR_BYTES,
                                      d_->mergeWindow * PAIR_COMPONENTS);
        }
        merged_ = View<float>(layout.merged, d_->mergeWindow * MERGE_ALL_SOURCE_COMPONENTS);
        copy_ = View<float>(layout.appendScratch, d_->mergeWindow * PAIR_COMPONENTS);
        offsets_ = View<int32_t>(
            layout.offsets,
            d_->mergeWindow * (d_->route == ROUTE_GM_MERGE ? PAIR_COMPONENTS : MERGE_ALL_SOURCE_COMPONENTS));
        if (d_->route == ROUTE_UB_ACCUMULATE) {
            const uint32_t words = static_cast<uint32_t>(DivUp(d_->k * PAIR_BYTES, UB_BLOCK_BYTES) *
                                                         FLOAT_BLOCK_ELEMENTS);
            acc_ = View<float>(layout.accumulator, words);
            next_ = View<float>(layout.nextAccumulator, words);
        }
        if (d_->route == ROUTE_GM_MERGE) {
            gm_[0].SetGlobalBuffer((__gm__ float*)workspace);
            gm_[1].SetGlobalBuffer((__gm__ float*)workspace + d_->rows * d_->rowWorkspacePairs * PAIR_COMPONENTS);
        }
    }

    // 根据 Host 路径执行拷贝、单行排序、UB 累积或 GM 多层归并。
    __aicore__ inline void Process()
    {
        if (d_->route == ROUTE_COPY) {
            CopyOnly();
        } else if (d_->route == ROUTE_ROW_SORT || d_->route == ROUTE_COMPACT_ROW_SORT) {
            SortRows();
        } else if (d_->route == ROUTE_UB_ACCUMULATE) {
            AccumulateRows();
        } else if (d_->route == ROUTE_GM_MERGE) {
            MergeGlobalRows();
        }
    }

private:
    // 返回当前行裁剪后的区间起点。
    __aicore__ inline uint32_t RowStart(uint64_t row)
    {
        if (boundsFlags_ & BOUND_HAS_BEGIN) {
            const int32_t value = begin_.GetValue((boundsFlags_ & BOUND_BROADCAST_BEGIN) ? 0 : row);
            return value < 0 ? 0 : Min<uint64_t>(value, d_->axis);
        }
        return 0;
    }

    // 返回当前行有效区间长度，空区间返回零。
    __aicore__ inline uint32_t RowLength(uint64_t row)
    {
        const uint32_t start = RowStart(row);
        uint32_t end = d_->axis;
        if (boundsFlags_ & BOUND_HAS_END) {
            const int32_t value = end_.GetValue((boundsFlags_ & BOUND_BROADCAST_END) ? 0 : row);
            end = value < 0 ? 0 : Min<uint64_t>(value, d_->axis);
        }
        return end > start ? end - start : 0;
    }

    // 按字节偏移取得指定类型的 UB 局部张量视图。
    template <typename U>
    __aicore__ inline LocalTensor<U> View(uint32_t byteOffset, uint32_t elements)
    {
        auto view = arenaBuf_.Get<U>()[byteOffset / sizeof(U)];
        view.SetSize(elements);
        return view;
    }
    // 从 GM 读取有序段片段并处理块对齐尾部。
    template <typename U>
    __aicore__ inline void Read(const LocalTensor<U>& dst, const GlobalTensor<U>& src, uint32_t count)
    {
        DataCopyExtParams p{1, static_cast<uint32_t>(count * sizeof(U)), 0, 0, 0};
        DataCopyPad(dst, src, p, DataCopyPadExtParams<U>{false, 0, 0, 0});
        Fence<HardEvent::MTE2_S>();
    }

    // 生成指定起点和步长的向量序列。
    __aicore__ inline void Sequence(const LocalTensor<int32_t>& dst, int32_t first, int32_t step, uint32_t count)
    {
        // 2201 的 ArithProgression 首块使用标量写；两侧同步并恢复向量 mask 后再复用。
        Fence<HardEvent::V_S>();
        ArithProgression<int32_t>(dst, first, step, count);
        Fence<HardEvent::S_V>();
        SetMaskNorm();
        ResetMask();
        PipeBarrier<PIPE_V>();
    }
    // 为 pair 分量生成 Gather 所需的字节偏移。
    __aicore__ inline void OffsetSequence(uint32_t byteBase, uint32_t words)
    {
        Sequence(offsets_, static_cast<int32_t>(byteBase), INDEX32_BYTES, words);
    }
    // 从有序 pair 段中提取指定区间到独立缓冲。
    __aicore__ inline void Slice(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t pairOffset,
                                 uint32_t pairs)
    {
        OffsetSequence(pairOffset * PAIR_BYTES, pairs * PAIR_COMPONENTS);
        Gather(dst.ReinterpretCast<uint32_t>(), src.ReinterpretCast<uint32_t>(), offsets_.ReinterpretCast<uint32_t>(),
               0, pairs * PAIR_COMPONENTS);
        PipeBarrier<PIPE_V>();
    }
    // 将新片段追加到目标段，处理未对齐边界。
    __aicore__ inline void Append(const LocalTensor<float>& dst, uint32_t pairOffset, const LocalTensor<float>& src,
                                  uint32_t pairs)
    {
        // 2201 不使用 Scatter：保留目标前缀至多 3 个 pair，再用 Gather 追加候选。
        // 按完整 32B 块写回；Host 为目标尾部预留对齐空间，不能越过已核验的缓冲边界。
        auto arena = arenaBuf_.Get<uint32_t>();
        uint32_t arenaAddr = static_cast<uint32_t>(reinterpret_cast<uint64_t>(arena.GetPhyAddr()));
        uint32_t srcAddr = static_cast<uint32_t>(reinterpret_cast<uint64_t>(src.GetPhyAddr())) - arenaAddr;
        uint32_t dstAddr = static_cast<uint32_t>(reinterpret_cast<uint64_t>(dst.GetPhyAddr())) - arenaAddr;
        uint32_t done = 0;
        while (done < pairs) {
            uint32_t head = (pairOffset + done) % PAIR_ALIGNMENT;
            uint32_t alignedPair = pairOffset + done - head;
            uint32_t take = Min<uint32_t>(pairs - done, d_->mergeWindow - head);
            uint32_t words = (head + take) * PAIR_COMPONENTS;
            uint32_t sourceStart = srcAddr + done * PAIR_BYTES - head * PAIR_BYTES;
            OffsetSequence(sourceStart, words);
            if (head) {
                int32_t delta = static_cast<int32_t>(dstAddr + alignedPair * PAIR_BYTES) -
                                static_cast<int32_t>(sourceStart);
                Adds(offsets_, offsets_, delta, head * PAIR_COMPONENTS);
                PipeBarrier<PIPE_V>();
            }
            Gather(copy_.ReinterpretCast<uint32_t>(), arena, offsets_.ReinterpretCast<uint32_t>(), 0, words);
            PipeBarrier<PIPE_V>();
            DataCopy(dst[alignedPair * PAIR_COMPONENTS].ReinterpretCast<uint32_t>(), copy_.ReinterpretCast<uint32_t>(),
                     static_cast<uint32_t>(DivUp(words, FLOAT_BLOCK_ELEMENTS) * FLOAT_BLOCK_ELEMENTS));
            PipeBarrier<PIPE_V>();
            done += take;
        }
    }
    // 移除窗口已消耗的前缀，保留后续有效 pair。
    __aicore__ inline void Compact(uint32_t route, uint32_t consumed, uint32_t remaining)
    {
        Slice(copy_, windows_[route], consumed, remaining);
        // 两侧基址均对齐，补齐部分限制在 H 容量内；下一轮只暴露 remaining 个有效 pair。
        DataCopy(
            windows_[route].ReinterpretCast<uint32_t>(), copy_.ReinterpretCast<uint32_t>(),
            static_cast<uint32_t>(DivUp(remaining * PAIR_COMPONENTS, FLOAT_BLOCK_ELEMENTS) * FLOAT_BLOCK_ELEMENTS));
        PipeBarrier<PIPE_V>();
    }
    // 流式加载并归并多路有序段，输出指定数量的结果。
    __aicore__ inline void Merge(bool global, const LocalTensor<float> localSrc[MERGE_WAYS],
                                 const uint64_t starts[MERGE_WAYS], const uint64_t lengths[MERGE_WAYS], uint32_t routes,
                                 const LocalTensor<float>& localDst, uint64_t dstBase, uint64_t target)
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
                    if (global) {
                        DataCopyExtParams load{1, h[i] * PAIR_BYTES, 0, 0, 0};
                        DataCopyPad(windows_[i], gm_[srcRole_][(starts[i] + pos[i]) * PAIR_COMPONENTS], load,
                                    DataCopyPadExtParams<float>{false, 0, 0, 0});
                    } else {
                        Slice(windows_[i], localSrc[i], starts[i] + pos[i], h[i]);
                    }
                }
                active[q++] = i;
            }
            if (global)
                Fence<HardEvent::MTE2_V>();
            uint32_t consumed[MERGE_WAYS], m = 0;
            if (outputInFlight) {
                // 窗口 DMA 可与上次归并结果的写回重叠；下次归并前等待输出缓冲释放。
                Fence<HardEvent::MTE3_V>();
                outputInFlight = false;
            }
            auto chunk = MergeWindow(windows_, merged_, active, q, h, consumed, m);
            uint32_t keep = Min<uint64_t>(m, target - written);
            if (global) {
                if (q > 1) {
                    DataCopyExtParams store{1, keep * PAIR_BYTES, 0, 0, 0};
                    Fence<HardEvent::V_MTE3>();
                    DataCopyPad(gm_[1 - srcRole_][(dstBase + written) * PAIR_COMPONENTS], chunk, store);
                    outputInFlight = true;
                } else {
                    StoreToGm(gm_[1 - srcRole_][(dstBase + written) * PAIR_COMPONENTS], chunk, keep * PAIR_COMPONENTS);
                }
            } else {
                Append(localDst, dstBase + written, chunk, keep);
            }
            written += keep;
            for (uint32_t i = 0; i < routes; ++i) {
                pos[i] += consumed[i];
                h[i] -= consumed[i];
                if (written < target && h[i] && consumed[i]) {
                    if (global) {
                        h[i] = 0; // 从已推进的 GM 位置重新填充窗口。
                    } else {
                        Compact(i, consumed[i], h[i]);
                    }
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
        const uint32_t absoluteStart = RowStart(row) + start;
        DataCopyExtParams load{1, static_cast<uint32_t>(valid * sizeof(T)), 0, 0, 0};
        DataCopyPad(raw_, x_[row * d_->axis + absoluteStart], load, DataCopyPadExtParams<T>{false, 0, 0, 0});
        // 先初始化填充值，避免对未对齐短尾单独执行 Duplicate。
        Duplicate(key_, SortPadding(), padded);
        PipeBarrier<PIPE_V>();
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
        PipeBarrier<PIPE_V>();
        GatherMask(index_, pairs.ReinterpretCast<uint32_t>(), uint8_t(GATHER_INDEX_PATTERN), true,
                   count * PAIR_COMPONENTS, params, selectedCount);
        PipeBarrier<PIPE_V>();
        if (!d_->largest) {
            Adds(key_.template ReinterpretCast<int32_t>(), key_.template ReinterpretCast<int32_t>(), INT32_MIN, count);
            PipeBarrier<PIPE_V>();
        }
        LocalTensor<T> output = raw_;
        if constexpr (std::is_same<T, float>::value) {
            output = key_.template ReinterpretCast<T>();
        } else {
            Cast(output, key_, RoundMode::CAST_RINT, count);
            PipeBarrier<PIPE_V>();
        }
        RestoreNaNPayload<T>(output, key_, index_, x_.GetPhyAddr(), row * d_->axis, count, d_->largest);
        DataCopyExtParams valuesCopy{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyExtParams indexCopy{1, count * INDEX32_BYTES, 0, 0, 0};
        Fence<HardEvent::V_MTE3>();
        DataCopyPad(values_[row * d_->k + offset], output, valuesCopy);
        DataCopyPad(indices_[row * d_->k + offset], index_.ReinterpretCast<int32_t>(), indexCopy);
        Fence<HardEvent::MTE3_S>();
    }
    // 从本地排序结果恢复原值，写出当前行的前 K 项。
    __aicore__ inline void EmitLocal(uint64_t row, const LocalTensor<float>& result)
    {
        const uint64_t count = Min<uint64_t>(d_->k, RowLength(row));
        for (uint64_t o = 0; o < count; o += d_->outputTile) {
            uint32_t f = Min<uint64_t>(d_->outputTile, count - o);
            Slice(copy_, result, o, f);
            Emit(row, o, copy_, f);
        }
    }
    // 对无需选择排序的路径拷贝输入并生成原轴索引。
    __aicore__ inline void CopyOnly()
    {
        uint64_t total = d_->rows * d_->axis;
        uint64_t tasks = DivUp(total, d_->tile);
        for (uint64_t task = GetBlockIdx(); task < tasks; task += d_->cores) {
            uint64_t end = Min<uint64_t>((task + 1) * d_->tile, total);
            if (d_->axis == 1) {
                // 单元素排序行的索引恒为 0；整块搬运，避免每个元素各触发一次 DMA 和同步。
                const uint64_t start = task * d_->tile;
                const uint32_t count = static_cast<uint32_t>(end - start);
                Read(raw_, x_[start], count);
                Duplicate(index_, uint32_t(0), count);
                PipeBarrier<PIPE_V>();
                StoreToGm(values_[start], raw_, count);
                StoreToGm(indices_[start], index_.ReinterpretCast<int32_t>(), count);
                continue;
            }
            for (uint64_t s = task * d_->tile; s < end;) {
                uint32_t count = Min<uint64_t>(end - s, d_->axis - s % d_->axis);
                Read(raw_, x_[s], count);
                Sequence(index_.ReinterpretCast<int32_t>(), static_cast<int32_t>(s % d_->axis), 1, count);
                PipeBarrier<PIPE_V>();
                StoreToGm(values_[s], raw_, count);
                StoreToGm(indices_[s], index_.ReinterpretCast<int32_t>(), count);
                s += count;
            }
        }
    }
    // 每行在 UB 内完整排序并直接输出前 K 项。
    __aicore__ inline void SortRows()
    {
        for (uint64_t row = GetBlockIdx(); row < d_->rows; row += d_->cores) {
            const uint32_t length = RowLength(row);
            if (length == 0) {
                continue;
            }
            auto result = SortTile(row, 0, length);
            if (d_->route == ROUTE_COMPACT_ROW_SORT) {
                if (result.GetPhyAddr() != ping_.GetPhyAddr()) {
                    DataCopy(ping_.ReinterpretCast<uint32_t>(), result.template ReinterpretCast<uint32_t>(),
                             static_cast<uint32_t>(DivUp(length, SORT32_ELEMENTS) * SORT32_PAIR_COMPONENTS));
                    PipeBarrier<PIPE_V>();
                }
                Emit(row, 0, ping_, Min<uint64_t>(d_->k, length));
            } else {
                EmitLocal(row, result);
            }
        }
    }
    // 逐 tile 排序并在 UB 中持续归并，维护每行前 K 项。
    __aicore__ inline void AccumulateRows()
    {
        for (uint64_t row = GetBlockIdx(); row < d_->rows; row += d_->cores) {
            LocalTensor<float> acc = acc_, next = next_;
            uint64_t committed = 0;
            const uint32_t length = RowLength(row);
            for (uint64_t t = 0; t < DivUp(length, d_->tile); ++t) {
                uint32_t valid = Min<uint64_t>(d_->tile, length - t * d_->tile);
                auto ct = SortTile(row, t * d_->tile, valid);
                uint64_t count = Min<uint64_t>(d_->k, valid);
                LocalTensor<float> inputs[MERGE_WAYS] = {acc, ct, ct, ct};
                uint64_t starts[MERGE_WAYS] = {0, 0, 0, 0}, lengths[MERGE_WAYS] = {committed, count, 0, 0};
                uint64_t nextCount = Min<uint64_t>(d_->k, committed + count);
                Merge(false, inputs, starts, lengths, TWO_MERGE_RUNS, next, 0, nextCount);
                auto tmp = acc;
                acc = next;
                next = tmp;
                committed = nextCount;
            }
            EmitLocal(row, acc);
        }
    }
    // 计算指定归并层中有序段实际保留的项数。
    __aicore__ inline uint64_t RunLength(uint64_t row, uint64_t leaf, uint64_t span)
    {
        const uint32_t length = RowLength(row);
        if (leaf * d_->tile >= length) {
            return 0;
        }
        return Min<uint64_t>(d_->k, Min<uint64_t>(length - leaf * d_->tile, span * d_->tile));
    }
    // 计算有序段在行工作区中的起始位置。
    __aicore__ inline uint64_t RunBase(uint64_t row, uint64_t leaf)
    {
        return row * d_->rowWorkspacePairs + leaf * Min<uint64_t>(d_->k, d_->tile);
    }
    // 等待工作区回写并同步所有参与归并的核。
    __aicore__ inline void StageBarrier()
    {
        Fence<HardEvent::MTE3_S>();
        SyncAll(); // 全部启动核都必须参与，包括本层没有任务的核。
    }
    // 按编码后的排序规则比较两项，包含 NaN 与相等键的处理。
    __aicore__ inline bool SortsBefore(float a, float b)
    {
        if (IsNaN(a)) {
            union {
                float value;
                uint32_t bits;
            } data{a};
            return !(data.bits & FP32_SIGN_BIT) && !IsNaN(b);
        }
        if (IsNaN(b)) {
            union {
                float value;
                uint32_t bits;
            } data{b};
            return data.bits & FP32_SIGN_BIT;
        }
        return a > b;
    }
    // 稳定协同排名将最后两路切分为互不重叠的输出区间；同键优先第一路，NaN 保持既定排序关系。
    __aicore__ inline uint64_t CoRank(uint64_t rank, uint64_t a, uint64_t na, uint64_t b, uint64_t nb)
    {
        uint64_t lo = rank > nb ? rank - nb : 0;
        uint64_t hi = Min<uint64_t>(rank, na);
        while (lo <= hi) {
            const uint64_t i = (lo + hi) / HALF_DIVISOR, j = rank - i;
            if (i && j < nb &&
                SortsBefore(gm_[srcRole_].GetValue((b + j) * PAIR_COMPONENTS),
                            gm_[srcRole_].GetValue((a + i - 1) * PAIR_COMPONENTS))) {
                hi = i - 1;
            } else if (j && i < na &&
                       !SortsBefore(gm_[srcRole_].GetValue((b + j - 1) * PAIR_COMPONENTS),
                                    gm_[srcRole_].GetValue((a + i) * PAIR_COMPONENTS))) {
                lo = i + 1;
            } else {
                return i;
            }
        }
        return lo;
    }
    // 并行生成叶子段并逐层 GM 归并，最终直接解码输出。
    __aicore__ inline void MergeGlobalRows()
    {
        // 叶子与每层任务按全体 AIV 分摊；每层读取一份 GM 平面、写另一份，屏障后交换。
        for (uint64_t task = GetBlockIdx(); task < d_->rows * d_->tiles; task += d_->cores) {
            uint64_t row = task / d_->tiles, t = task % d_->tiles;
            const uint32_t length = RowLength(row);
            if (t * d_->tile >= length) {
                continue;
            }
            uint32_t valid = Min<uint64_t>(d_->tile, length - t * d_->tile);
            auto ct = SortTile(row, t * d_->tile, valid);
            StoreToGm(gm_[0][RunBase(row, t) * PAIR_COMPONENTS], ct, Min<uint64_t>(d_->k, valid) * PAIR_COMPONENTS);
        }
        StageBarrier(); // 即使只有一段，也要等写回完成。
        srcRole_ = 0;
        uint64_t n = d_->tiles, span = 1;
        while (n > 1) {
            uint64_t groups = DivUp(n, MERGE_WAYS);
            const uint32_t partitions = n == TWO_MERGE_RUNS && d_->rows < d_->cores &&
                                                d_->k >= Tuning::PARALLEL_FINAL_MIN_K ?
                                            Min<uint64_t>(MERGE_WAYS, d_->cores / d_->rows) :
                                            1;
            for (uint64_t task = GetBlockIdx(); task < d_->rows * groups * partitions; task += d_->cores) {
                const uint32_t part = task % partitions;
                uint64_t row = task / partitions / groups, group = task / partitions % groups;
                uint64_t firstLeaf = group * MERGE_WAYS * span;
                uint64_t starts[MERGE_WAYS] = {0, 0, 0, 0}, lengths[MERGE_WAYS] = {0, 0, 0, 0};
                LocalTensor<float> unused[MERGE_WAYS] = {ping_, ping_, ping_, ping_};
                uint32_t routes = 0;
                uint64_t sum = 0;
                for (uint32_t i = 0; i < MERGE_WAYS && firstLeaf + i * span < d_->tiles; ++i) {
                    uint64_t leaf = firstLeaf + i * span;
                    starts[i] = RunBase(row, leaf);
                    lengths[i] = RunLength(row, leaf, span);
                    sum += lengths[i];
                    ++routes;
                }
                const uint64_t target = Min<uint64_t>(d_->k, sum);
                uint64_t outputStart = 0, outputCount = target;
                if (partitions > 1) {
                    const uint64_t chunk = DivUp(DivUp(target, partitions), SORT32_ELEMENTS) * SORT32_ELEMENTS;
                    outputStart = part * chunk;
                    if (outputStart >= target)
                        continue;
                    const uint64_t outputEnd = Min<uint64_t>(target, outputStart + chunk);
                    const uint64_t i0 = CoRank(outputStart, starts[0], lengths[0], starts[1], lengths[1]);
                    const uint64_t i1 = CoRank(outputEnd, starts[0], lengths[0], starts[1], lengths[1]);
                    starts[0] += i0;
                    starts[1] += outputStart - i0;
                    lengths[0] = i1 - i0;
                    lengths[1] = outputEnd - outputStart - lengths[0];
                    outputCount = outputEnd - outputStart;
                }
                Merge(true, unused, starts, lengths, routes, ping_, RunBase(row, firstLeaf) + outputStart, outputCount);
            }
            StageBarrier();
            srcRole_ = 1 - srcRole_;
            n = groups;
            span = Min<uint64_t>(d_->tiles, span * MERGE_WAYS);
        }
        const uint64_t chunks = DivUp(d_->k, d_->outputTile);
        for (uint64_t task = GetBlockIdx(); task < d_->rows * chunks; task += d_->cores) {
            const uint64_t row = task / chunks, o = (task % chunks) * d_->outputTile;
            const uint64_t count = Min<uint64_t>(d_->k, RowLength(row));
            if (o < count) {
                const uint32_t f = Min<uint64_t>(d_->outputTile, count - o);
                Read(ping_, gm_[srcRole_][(RunBase(row, 0) + o) * PAIR_COMPONENTS], f * PAIR_COMPONENTS);
                Emit(row, o, ping_, f);
            }
        }
    }

    const SelectTopKBaseTilingData* d_ = nullptr;
    uint32_t srcRole_ = 0;
    uint32_t boundsFlags_ = 0;
    GlobalTensor<int32_t> begin_, end_;
    GlobalTensor<T> x_;
    GlobalTensor<T> values_;
    GlobalTensor<int32_t> indices_;
    GlobalTensor<float> gm_[DOUBLE_BUFFER_COUNT];
    TBuf<TPosition::VECCALC> arenaBuf_;
    LocalTensor<T> raw_;
    LocalTensor<uint32_t> index_;
    LocalTensor<int32_t> offsets_;
    LocalTensor<float> key_;
    LocalTensor<float> ping_;
    LocalTensor<float> pong_;
    LocalTensor<float> windows_[MERGE_WAYS];
    LocalTensor<float> merged_;
    LocalTensor<float> copy_;
    LocalTensor<float> acc_;
    LocalTensor<float> next_;
};
} // namespace SelectTopKBase
#endif
