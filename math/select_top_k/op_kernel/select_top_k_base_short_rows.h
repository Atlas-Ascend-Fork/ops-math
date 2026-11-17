/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SELECT_TOP_K_BASE_SHORT_ROWS_H
#define SELECT_TOP_K_BASE_SHORT_ROWS_H

#include "select_top_k_base_common.h"

namespace SelectTopKBase {

// R1 的短轴子路径。核数、每批行数及 UB 容量由平台资源和实际 shape 决定。
template <typename T, bool DirectIndices64 = false>
class TopKShortRows {
    using Bits = typename std::conditional<sizeof(T) == VALUE16_BYTES, uint16_t, uint32_t>::type;
    using Index = typename std::conditional<DirectIndices64, int64_t, int32_t>::type;

public:
    // 绑定短轴输入输出，划分批量行的排序与索引缓冲。
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices, const SelectTopKBaseTilingData* data,
                                TPipe* pipe)
    {
        d_ = data;
        xBits_.SetGlobalBuffer((__gm__ Bits*)x);
        values_.SetGlobalBuffer((__gm__ T*)values);
        indices_.SetGlobalBuffer((__gm__ Index*)indices);
        pipe->InitBuffer(arena_, d_->ubBytes);
        uint32_t cursor = 0;
        raw_ = UbSubview<T>(arena_, cursor, d_->tile);
        cursor += d_->tile * sizeof(T);
        key_ = UbSubview<float>(arena_, cursor, d_->tile);
        cursor += d_->tile * sizeof(float);
        index_ = UbSubview<uint32_t>(arena_, cursor, d_->tile);
        cursor += d_->tile * sizeof(uint32_t);
        pairs_ = UbSubview<float>(arena_, cursor, d_->tile * PAIR_COMPONENTS);
    }

    // 按批读取短行，每行由一个 Sort32 repeat 排序并写回。
    __aicore__ inline void Process()
    {
        const uint32_t batchRows = d_->batchRows;
        const uint64_t step = static_cast<uint64_t>(batchRows) * d_->cores;
        for (uint64_t first = static_cast<uint64_t>(GetBlockIdx()) * batchRows; first < d_->rows; first += step) {
            const uint32_t rows = static_cast<uint32_t>(Min<uint64_t>(batchRows, d_->rows - first));
            const uint32_t slots = rows * SORT32_ELEMENTS;
            PrepareIndices(slots);
            ReadRows(first, rows, slots);
            if constexpr (DirectIndices64) {
                EncodeSortKeys(key_, raw_, pairs_.ReinterpretCast<uint8_t>(), slots, d_->largest);
            } else {
                EncodeSortKeys(key_, raw_, slots, d_->largest);
                PipeBarrier<PIPE_V>();
            }
            if (d_->axis < SORT32_ELEMENTS) {
                uint64_t paddingMask[VECTOR_MASK_WORDS] = {PACKED_INDEX_LOW_BITS & ~((1ULL << d_->axis) - 1), 0};
                Duplicate(key_, SortPadding(), paddingMask, static_cast<uint8_t>(rows), 1, SHORT_ROW_REPEAT_BLOCKS);
                PipeBarrier<PIPE_V>();
            }
            // 每个 repeat 只比较一条补齐后的 32 元素行，行与行之间不做归并。
            Sort32(pairs_, key_, index_, rows);
            PipeBarrier<PIPE_V>();
            EmitRows(first, rows, slots);
        }
    }

private:
    // 从保留的 UB 原始值按排序索引 Gather，保留 NaN 载荷并省去逐行标量检查和 GM 回读。
    __aicore__ inline void GatherOriginalValues(uint64_t first, uint32_t rows, uint32_t slots)
    {
        auto offsets = pairs_.ReinterpretCast<uint32_t>();
        Muls(offsets.ReinterpretCast<int32_t>(), index_.ReinterpretCast<int32_t>(), static_cast<int32_t>(sizeof(T)),
             slots);
        PipeBarrier<PIPE_V>();
        auto restored = key_.ReinterpretCast<Bits>();
        auto original = raw_.template ReinterpretCast<Bits>();
        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t start = row * SORT32_ELEMENTS;
            Gather(restored[start], original[start], offsets[start], 0, SORT32_ELEMENTS);
        }
        PipeBarrier<PIPE_V>();
        WriteRows(values_[first * d_->k], restored.template ReinterpretCast<T>(), rows);
    }

    // 为每行构造原轴索引及无效尾部标记。
    __aicore__ inline void PrepareIndices(uint32_t slots)
    {
        Fence<HardEvent::V_S>();
        // 只生成一行 0..31，再用源 repeat 步长为 0 的 Adds 广播到其余各行。
        // 避免先生成 0..(rows*32-1) 再按位与；长序列 API 内部有逐段依赖链。
        auto rowIndices = index_.ReinterpretCast<int32_t>();
        ArithProgression<int32_t>(rowIndices, 0, 1, SORT32_ELEMENTS);
        Fence<HardEvent::S_V>();
        SetMaskNorm();
        ResetMask();
        PipeBarrier<PIPE_V>();
        const uint32_t rows = slots / SORT32_ELEMENTS;
        if (rows > 1) {
            const UnaryRepeatParams repeatParams(1, 1, SHORT_ROW_REPEAT_BLOCKS, 0);
            Adds(rowIndices[SORT32_ELEMENTS], rowIndices, int32_t(0), uint64_t(SORT32_ELEMENTS),
                 static_cast<uint8_t>(rows - 1), repeatParams);
            PipeBarrier<PIPE_V>();
        }
    }

    // 批量加载各短行并填充对齐尾部。
    __aicore__ inline void ReadRows(uint64_t first, uint32_t rows, uint32_t slots)
    {
        if constexpr (DirectIndices64) {
            if (d_->axis == SORT32_ELEMENTS) {
                // 完整 Sort32 行在 GM/UB 均连续且满足块对齐，无需初始化尾部或描述逐行 padding。
                DataCopy(raw_.template ReinterpretCast<Bits>(), xBits_[first * d_->axis], slots);
                Fence<HardEvent::MTE2_V>();
                return;
            }
        }
        Bits padding;
        if constexpr (std::is_same<T, float>::value) {
            padding = d_->largest ? uint32_t(FP32_NEG_INF_BITS) : uint32_t(FP32_POS_INF_BITS);
        } else if constexpr (std::is_same<T, half>::value) {
            padding = d_->largest ? uint16_t(FP16_NEG_INF_BITS) : uint16_t(FP16_POS_INF_BITS);
        } else {
            padding = d_->largest ? uint16_t(BF16_NEG_INF_BITS) : uint16_t(BF16_POS_INF_BITS);
        }
        // 先初始化整行间隙；DMA 只补齐本行末尾的一个 32B 块，其余间隙保留哨兵。
        auto rawBits = raw_.template ReinterpretCast<Bits>();
        Duplicate(rawBits, padding, slots);
        Fence<HardEvent::V_MTE2>();
        LoadPaddedRows(rawBits, xBits_[first * d_->axis], rows, d_->axis, SORT32_ELEMENTS, padding);
        Fence<HardEvent::MTE2_V>();
    }

    // 短行模板逐次等待值和索引写回；公共搬运函数不引入或合并同步。
    template <typename U>
    __aicore__ inline void WriteRows(const GlobalTensor<U>& destination, const LocalTensor<U>& source, uint32_t rows)
    {
        Fence<HardEvent::V_MTE3>();
        WriteRowsToGm(destination, source, rows, d_->k, SORT32_ELEMENTS);
        Fence<HardEvent::MTE3_S>();
    }

    // 提取各行前 K 个值和索引，并恢复原值位模式。
    __aicore__ inline void EmitRows(uint64_t first, uint32_t rows, uint32_t slots)
    {
        uint64_t selectedCount = 0;
        GatherMaskParams params(1, 1, VECTOR_REPEAT_BLOCKS, 0);
        if constexpr (!DirectIndices64) {
            GatherMask(key_, pairs_, uint8_t(1), true, slots * PAIR_COMPONENTS, params, selectedCount);
            PipeBarrier<PIPE_V>();
        }
        GatherMask(index_, pairs_.ReinterpretCast<uint32_t>(), uint8_t(GATHER_INDEX_PATTERN), true,
                   slots * PAIR_COMPONENTS, params, selectedCount);
        PipeBarrier<PIPE_V>();
        if constexpr (!DirectIndices64) {
            if (!d_->largest) {
                Adds(key_.ReinterpretCast<int32_t>(), key_.ReinterpretCast<int32_t>(), INT32_MIN, slots);
                PipeBarrier<PIPE_V>();
            }
        }
        if constexpr (DirectIndices64) {
            GatherOriginalValues(first, rows, slots);
        } else if constexpr (std::is_same<T, float>::value) {
            for (uint32_t r = 0; r < rows; ++r) {
                RestoreNaNPayload<float>(key_[r * SORT32_ELEMENTS], key_[r * SORT32_ELEMENTS],
                                         index_[r * SORT32_ELEMENTS],
                                         reinterpret_cast<const __gm__ float*>(xBits_.GetPhyAddr()),
                                         (first + r) * d_->axis, d_->k, d_->largest);
            }
            WriteRows(values_[first * d_->k], key_, rows);
        } else {
            Cast(raw_, key_, RoundMode::CAST_RINT, slots);
            PipeBarrier<PIPE_V>();
            for (uint32_t r = 0; r < rows; ++r) {
                RestoreNaNPayload<T>(raw_[r * SORT32_ELEMENTS], key_[r * SORT32_ELEMENTS], index_[r * SORT32_ELEMENTS],
                                     reinterpret_cast<const __gm__ T*>(xBits_.GetPhyAddr()), (first + r) * d_->axis,
                                     d_->k, d_->largest);
            }
            WriteRows(values_[first * d_->k], raw_, rows);
        }
        if constexpr (DirectIndices64) {
            // 值与索引已提取，pairs_ 不再存活；其每项 8B 容量恰好复用为 INT64 索引。
            auto wide = pairs_.ReinterpretCast<int64_t>();
            Cast(wide, index_.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, slots);
            PipeBarrier<PIPE_V>();
            WriteRows(indices_[first * d_->k], wide, rows);
        } else {
            WriteRows(indices_[first * d_->k], index_.ReinterpretCast<int32_t>(), rows);
        }
    }

    const SelectTopKBaseTilingData* d_ = nullptr;
    TBuf<TPosition::VECCALC> arena_;
    GlobalTensor<Bits> xBits_;
    GlobalTensor<T> values_;
    GlobalTensor<Index> indices_;
    LocalTensor<T> raw_;
    LocalTensor<float> key_;
    LocalTensor<uint32_t> index_;
    LocalTensor<float> pairs_;
};
} // namespace SelectTopKBase
#endif
