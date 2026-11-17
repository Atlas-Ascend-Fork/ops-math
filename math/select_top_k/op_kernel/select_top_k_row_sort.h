/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SELECT_TOP_K_ROW_SORT_H_
#define SELECT_TOP_K_ROW_SORT_H_
#include "select_top_k_merge_sort.h"

namespace SelectTopK {
// 整行批处理：连续行块分核、Sort32 对齐、双输入缓冲，批量搬运并逐行核内归并。
template <typename T, bool WideIndices>
class RowSort {
    using Bits = typename std::conditional<sizeof(T) == VALUE16_BYTES, uint16_t, uint32_t>::type;
    using Index = typename std::conditional<WideIndices, int64_t, int32_t>::type;

public:
    // 按共享容量公式划分两个原值区、键、索引和两份排序结果。
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices, const SelectTopKTilingData& data,
                                TPipe* pipe)
    {
        d_ = &data.base;
        schedule_ = &data.schedule;
        width_ = schedule_->rowWidth;
        x_.SetGlobalBuffer((__gm__ Bits*)x);
        values_.SetGlobalBuffer((__gm__ T*)values);
        indices_.SetGlobalBuffer((__gm__ Index*)indices);
        pipe->InitBuffer(arena_, d_->ubBytes);
        uint32_t offset = 0;
        for (uint32_t i = 0; i < ROW_INPUT_BUFFER_COUNT; ++i) {
            raw_[i] = UbSubview<T>(arena_, offset, d_->tile);
            offset += d_->tile * sizeof(T);
        }
        key_ = UbSubview<float>(arena_, offset, d_->tile);
        offset += d_->tile * VALUE32_BYTES;
        index_ = UbSubview<uint32_t>(arena_, offset, d_->tile);
        offset += d_->tile * INDEX32_BYTES;
        ping_ = UbSubview<float>(arena_, offset, d_->tile * PAIR_COMPONENTS);
        offset += d_->tile * PAIR_BYTES;
        pong_ = UbSubview<float>(arena_, offset, d_->tile * PAIR_COMPONENTS);
    }

    // 只处理当前核的连续行块；下一批输入预取与当前批排序使用互不重叠的原值区。
    __aicore__ inline void Process()
    {
        const uint64_t first = GetBlockIdx() * schedule_->rowsPerCore;
        const uint64_t end = Min<uint64_t>(d_->rows, first + schedule_->rowsPerCore);
        if (first >= end)
            return;
        uint32_t role = 0;
        LoadRows(first, Min<uint64_t>(d_->batchRows, end - first), role);
        for (uint64_t row = first; row < end; row += d_->batchRows) {
            const uint32_t rows = Min<uint64_t>(d_->batchRows, end - row);
            const uint32_t slots = rows * width_;
            Fence<HardEvent::MTE2_V>();
            const uint64_t next = row + d_->batchRows;
            if (next < end)
                LoadRows(next, Min<uint64_t>(d_->batchRows, end - next), 1 - role);
            PrepareKeys(role, slots, rows);
            PrepareIndices(rows);
            Sort32(ping_, key_, index_, slots / SORT32_ELEMENTS);
            PipeBarrier<PIPE_V>();
            auto source = ping_, destination = pong_;
            for (uint32_t span = SORT32_ELEMENTS; span < width_; span *= MERGE_WAYS) {
                // 每行含完整四路组时按整批 repeat，归并不会跨行，并减少逐行的等待。
                if (width_ % (span * MERGE_WAYS) == 0) {
                    MergeTileStage(destination, source, slots, span);
                } else {
                    for (uint32_t r = 0; r < rows; ++r)
                        MergeTileStage(destination[r * width_ * PAIR_COMPONENTS], source[r * width_ * PAIR_COMPONENTS],
                                       width_, span);
                }
                auto old = source;
                source = destination;
                destination = old;
            }
            EmitRows(row, rows, slots, role, source);
            role = 1 - role;
        }
    }

private:
    // 发起一批连续输入的二维拷贝；完整对齐行可以直接使用连续 DataCopy。
    __aicore__ inline void LoadRows(uint64_t first, uint32_t rows, uint32_t role)
    {
        auto bits = raw_[role].template ReinterpretCast<Bits>();
        if (d_->axis == width_) {
            DataCopy(bits, x_[first * d_->axis], rows * width_);
            return;
        }
        Duplicate(bits, Bits(0), rows * width_);
        Fence<HardEvent::V_MTE2>();
        LoadPaddedRows(bits, x_[first * d_->axis], rows, d_->axis, width_, Bits(0));
    }

    // 保留输入原值，单独编码 NaN 和升降序键，最后屏蔽每行无效尾部。
    __aicore__ inline void PrepareKeys(uint32_t role, uint32_t slots, uint32_t rows)
    {
        EncodeSortKeys(key_, raw_[role], pong_.ReinterpretCast<uint8_t>(), slots, d_->largest);
        const uint32_t validTail = d_->axis % SORT32_ELEMENTS;
        if (validTail != 0) {
            uint64_t mask[VECTOR_MASK_WORDS] = {PACKED_INDEX_LOW_BITS & ~((1ULL << validTail) - 1), 0};
            for (uint32_t row = 0; row < rows; ++row)
                Duplicate(key_[row * width_ + width_ - SORT32_ELEMENTS], SortPadding(), mask, 1, 1,
                          SHORT_ROW_REPEAT_BLOCKS);
            PipeBarrier<PIPE_V>();
        }
    }

    // 生成一行轴内索引，再复制到同批各行；排序索引始终不包含行偏移。
    __aicore__ inline void PrepareIndices(uint32_t rows)
    {
        CreateLinearIndex(index_.ReinterpretCast<int32_t>(), 0, width_);
        for (uint32_t row = 1; row < rows; ++row)
            DataCopy(index_[row * width_], index_, width_);
        PipeBarrier<PIPE_V>();
    }

    // 先提取索引，再从原始位数组恢复值；排序双缓冲复用为偏移和 INT64 输出。
    __aicore__ inline void EmitRows(uint64_t first, uint32_t rows, uint32_t slots, uint32_t role,
                                    const LocalTensor<float>& sorted)
    {
        uint64_t selected = 0;
        GatherMaskParams params(1, 1, VECTOR_REPEAT_BLOCKS, 0);
        GatherMask(index_, sorted.template ReinterpretCast<uint32_t>(), uint8_t(GATHER_INDEX_PATTERN), true,
                   slots * PAIR_COMPONENTS, params, selected);
        PipeBarrier<PIPE_V>();
        auto offsets = ping_.ReinterpretCast<uint32_t>();
        Muls(offsets.ReinterpretCast<int32_t>(), index_.ReinterpretCast<int32_t>(), static_cast<int32_t>(sizeof(T)),
             slots);
        PipeBarrier<PIPE_V>();
        auto restored = key_.ReinterpretCast<Bits>();
        auto original = raw_[role].template ReinterpretCast<Bits>();
        for (uint32_t row = 0; row < rows; ++row)
            Gather(restored[row * width_], original[row * width_], offsets[row * width_], 0, width_);
        PipeBarrier<PIPE_V>();
        if constexpr (WideIndices) {
            auto wide = ping_.ReinterpretCast<int64_t>();
            Cast(wide, index_.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, slots);
            Fence<HardEvent::V_MTE3>();
            WriteRowsToGm(values_[first * d_->k], restored.template ReinterpretCast<T>(), rows, d_->k, width_);
            WriteRowsToGm(indices_[first * d_->k], wide, rows, d_->k, width_);
        } else {
            Fence<HardEvent::V_MTE3>();
            WriteRowsToGm(values_[first * d_->k], restored.template ReinterpretCast<T>(), rows, d_->k, width_);
            WriteRowsToGm(indices_[first * d_->k], index_.ReinterpretCast<int32_t>(), rows, d_->k, width_);
        }
        // 值和索引已各自准备好，两次DMA共用一个完成等待，之后才复用UB。
        Fence<HardEvent::MTE3_S>();
    }

    const SelectTopKBaseTilingData* d_ = nullptr;
    const SortSchedule* schedule_ = nullptr;
    uint32_t width_ = 0;
    TBuf<TPosition::VECCALC> arena_;
    GlobalTensor<Bits> x_;
    GlobalTensor<T> values_;
    GlobalTensor<Index> indices_;
    LocalTensor<T> raw_[ROW_INPUT_BUFFER_COUNT];
    LocalTensor<float> key_, ping_, pong_;
    LocalTensor<uint32_t> index_;
};
} // namespace SelectTopK
#endif
