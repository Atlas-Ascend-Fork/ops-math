/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SELECT_TOP_K_PREFIX_ROWS_H_
#define SELECT_TOP_K_PREFIX_ROWS_H_
#include "select_top_k_tiling_data.h"
#include "select_top_k_base_common.h"

namespace SelectTopK {
using namespace SelectTopKBase;

// 大批量行的小 K 归并：每段保留 64 项，逐层紧凑化，原始位值始终单独保存。
template <typename T>
class PrefixRows {
    using Bits = typename std::conditional<sizeof(T) == VALUE16_BYTES, uint16_t, uint32_t>::type;
    static constexpr uint32_t FIRST_RUN = SORT32_ELEMENTS * MERGE_WAYS;
    static constexpr uint32_t NEXT_RUN = PREFIX_ROW_ITEMS * MERGE_WAYS;
    static constexpr uint32_t PACKED_INDEX_ITEMS = sizeof(uint64_t) / sizeof(uint32_t); // 一次标量写两个索引。
    static constexpr uint32_t PARALLEL_INDEX_GROWTH = 4; // 每层独立生成三段，减少串行向量等待。
public:
    // 划分两个原值区及排序工作区；固定轴内索引只初始化一次。
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices, const SelectTopKTilingData& data,
                                TPipe* pipe)
    {
        d_ = &data.base;
        rowsPerCore_ = data.schedule.rowsPerCore;
        width_ = data.schedule.rowWidth;
        x_.SetGlobalBuffer((__gm__ Bits*)x);
        values_.SetGlobalBuffer((__gm__ Bits*)values);
        indices_.SetGlobalBuffer((__gm__ int64_t*)indices);
        pipe->InitBuffer(arena_, d_->ubBytes);
        uint32_t offset = 0;
        for (uint32_t role = 0; role < ROW_INPUT_BUFFER_COUNT; ++role) {
            raw_[role] = UbSubview<T>(arena_, offset, d_->tile);
            offset += d_->tile * sizeof(T);
            // 一批即可完成的核不读取第二份原值区，避免初始化永远不会使用的缓冲。
            if (role == 0 || rowsPerCore_ > d_->batchRows)
                Duplicate(raw_[role].template ReinterpretCast<Bits>(), Bits(0), d_->tile);
        }
        key_ = UbSubview<float>(arena_, offset, d_->tile);
        offset += d_->tile * VALUE32_BYTES;
        index_ = UbSubview<uint32_t>(arena_, offset, d_->tile);
        offset += d_->tile * INDEX32_BYTES;
        ping_ = UbSubview<float>(arena_, offset, d_->tile * PAIR_COMPONENTS);
        offset += d_->tile * PAIR_BYTES;
        pong_ = UbSubview<float>(arena_, offset, d_->tile * PAIR_COMPONENTS);
        InitializeRowIndex();
        for (uint32_t row = 1; row < d_->batchRows; ++row)
            DataCopy(index_[row * width_], index_, width_);
        PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE2>();
    }

    // 连续行块分核，交替预取下一批；每次只恢复及写回 K 对有效输出。
    __aicore__ inline void Process()
    {
        const uint64_t first = GetBlockIdx() * rowsPerCore_;
        const uint64_t end = Min<uint64_t>(d_->rows, first + rowsPerCore_);
        if (first >= end)
            return;
        uint32_t role = 0;
        LoadRows(first, Min<uint64_t>(d_->batchRows, end - first), role);
        for (uint64_t firstRow = first; firstRow < end; firstRow += d_->batchRows) {
            const uint32_t rows = Min<uint64_t>(d_->batchRows, end - firstRow);
            Fence<HardEvent::MTE2_V>();
            const uint64_t next = firstRow + d_->batchRows;
            if (next < end)
                LoadRows(next, Min<uint64_t>(d_->batchRows, end - next), 1 - role);
            EncodeKeys(rows, role);
            Sort32(ping_, key_, index_, rows * width_ / SORT32_ELEMENTS);
            PipeBarrier<PIPE_V>();
            Merge<SORT32_ELEMENTS>(pong_, ping_, rows * width_ / FIRST_RUN);
            Compact<FIRST_RUN>(ping_, pong_, rows * width_ / FIRST_RUN);
            MergeSecondStage(rows);
            Compact<NEXT_RUN>(ping_, pong_, rows * DivUp(width_, PREFIX_ROW_GROUP_ITEMS));
            MergeRowPrefixes(rows);
            EmitRows(firstRow, rows, role);
            role = 1 - role;
        }
    }

private:
    // 新索引区没有在途写者；成对写首块，再并行扩展连续的三段，只在层间等待。
    __aicore__ inline void InitializeRowIndex()
    {
        auto packed = index_.ReinterpretCast<uint64_t>();
        for (uint32_t pair = 0; pair < INDEX_SEQUENCE_SEED / PACKED_INDEX_ITEMS; ++pair) {
            const uint32_t first = pair * PACKED_INDEX_ITEMS;
            packed.SetValue(pair, uint64_t{first} | (uint64_t{first + 1} << MASK_WORD32_BITS));
        }
        Fence<HardEvent::S_V>();
        auto index = index_.ReinterpretCast<int32_t>();
        for (uint32_t generated = INDEX_SEQUENCE_SEED; generated < width_; generated *= PARALLEL_INDEX_GROWTH) {
            for (uint32_t part = 1; part < PARALLEL_INDEX_GROWTH && part * generated < width_; ++part) {
                const uint32_t first = part * generated;
                Adds(index[first], index, static_cast<int32_t>(first), Min<uint32_t>(generated, width_ - first));
            }
            PipeBarrier<PIPE_V>();
        }
    }

    // 二维 DMA 仅覆盖有效输入及一个对齐块，余下行尾保持初始化的零值。
    __aicore__ inline void LoadRows(uint64_t first, uint32_t rows, uint32_t role)
    {
        LoadPaddedRows(raw_[role].template ReinterpretCast<Bits>(), x_[first * d_->axis], rows, d_->axis, width_,
                       Bits(0));
    }

    // 排序键规范化 NaN 与方向；所有无效尾项都低于有效键，原始位值不变。
    __aicore__ inline void EncodeKeys(uint32_t rows, uint32_t role)
    {
        const uint32_t count = rows * width_;
        EncodeSortKeys(key_, raw_[role], pong_.ReinterpretCast<uint8_t>(), count, d_->largest);
        // 首个不完整向量使用位掩码，保证所有向量指令的 UB 起点按块对齐。
        const uint32_t valid = d_->axis % FLOAT_VECTOR_ELEMENTS;
        const uint32_t aligned = DivUp(d_->axis, FLOAT_VECTOR_ELEMENTS) * FLOAT_VECTOR_ELEMENTS;
        for (uint32_t row = 0; row < rows; ++row) {
            if (valid != 0) {
                uint64_t mask[VECTOR_MASK_WORDS] = {~((1ULL << valid) - 1), 0};
                Duplicate(key_[row * width_ + aligned - FLOAT_VECTOR_ELEMENTS], SortPadding(), mask, 1, 1,
                          VECTOR_REPEAT_BLOCKS);
            }
            if (aligned < width_)
                Duplicate(key_[row * width_ + aligned], SortPadding(), width_ - aligned);
        }
        PipeBarrier<PIPE_V>();
    }

    // 末级仅归并当前行的有效组；三路行不补第四路，输出仍保留统一的四路槽距。
    __aicore__ inline void MergeRowPrefixes(uint32_t rows)
    {
        const uint32_t groups = DivUp(width_, PREFIX_ROW_GROUP_ITEMS);
        if (groups == MERGE_WAYS) {
            Merge<PREFIX_ROW_ITEMS>(pong_, ping_, rows);
            return;
        }
        uint16_t lengths[MERGE_WAYS] = {PREFIX_ROW_ITEMS, PREFIX_ROW_ITEMS, PREFIX_ROW_ITEMS, 0};
        for (uint32_t row = 0; row < rows; ++row) {
            auto src = ping_[row * groups * PREFIX_ROW_ITEMS * PAIR_COMPONENTS];
            MrgSortSrcList<float> sources(src, src[PREFIX_ROW_ITEMS * PAIR_COMPONENTS],
                                          src[PREFIX_ROW_ITEMS * MERGE_THIRD_SOURCE * PAIR_COMPONENTS], src);
            MrgSort4Info info(lengths, false, static_cast<uint16_t>((1U << groups) - 1), 1);
            MrgSort(pong_[row * NEXT_RUN * PAIR_COMPONENTS], sources, info);
        }
        PipeBarrier<PIPE_V>();
    }

    // 首级组数非四的倍数时，各行先批量合并满组，再单独处理真实尾组，禁止跨行归并。
    __aicore__ inline void MergeSecondStage(uint32_t rows)
    {
        const uint32_t sourceGroups = width_ / FIRST_RUN;
        const uint32_t fullGroups = sourceGroups / MERGE_WAYS;
        const uint32_t tailGroups = sourceGroups % MERGE_WAYS;
        if (tailGroups == 0) {
            Merge<PREFIX_ROW_ITEMS>(pong_, ping_, rows * fullGroups);
            return;
        }
        const uint32_t sourceStride = sourceGroups * PREFIX_ROW_ITEMS;
        const uint32_t outputStride = (fullGroups + 1) * NEXT_RUN;
        uint16_t lengths[MERGE_WAYS] = {PREFIX_ROW_ITEMS, PREFIX_ROW_ITEMS, PREFIX_ROW_ITEMS, PREFIX_ROW_ITEMS};
        for (uint32_t row = 0; row < rows; ++row) {
            auto source = ping_[row * sourceStride * PAIR_COMPONENTS];
            auto output = pong_[row * outputStride * PAIR_COMPONENTS];
            MrgSortSrcList<float> full(source, source[PREFIX_ROW_ITEMS * PAIR_COMPONENTS],
                                       source[PREFIX_ROW_ITEMS * MERGE_THIRD_SOURCE * PAIR_COMPONENTS],
                                       source[PREFIX_ROW_ITEMS * MERGE_FOURTH_SOURCE * PAIR_COMPONENTS]);
            MrgSort4Info fullInfo(lengths, false, MERGE_ALL_SOURCES_MASK, static_cast<uint8_t>(fullGroups));
            MrgSort(output, full, fullInfo);
            auto tail = source[fullGroups * NEXT_RUN * PAIR_COMPONENTS];
            auto tailOutput = output[fullGroups * NEXT_RUN * PAIR_COMPONENTS];
            if (tailGroups == 1) {
                // 单个64项前缀已经有序，直接放入尾槽的有效部分。
                DataCopy(tailOutput.ReinterpretCast<uint32_t>(), tail.ReinterpretCast<uint32_t>(),
                         PREFIX_ROW_ITEMS * PAIR_COMPONENTS);
            } else {
                MrgSortSrcList<float> partial(tail, tail[PREFIX_ROW_ITEMS * PAIR_COMPONENTS],
                                              tailGroups > MERGE_THIRD_SOURCE ?
                                                  tail[PREFIX_ROW_ITEMS * MERGE_THIRD_SOURCE * PAIR_COMPONENTS] :
                                                  tail,
                                              tail);
                MrgSort4Info tailInfo(lengths, false, static_cast<uint16_t>((1U << tailGroups) - 1), 1);
                MrgSort(tailOutput, partial, tailInfo);
            }
        }
        // 每行目标槽互不重叠；所有尾槽前64项有效后才统一裁剪。
        PipeBarrier<PIPE_V>();
    }

    // 所有有序段等长且连续，硬件 repeat 一次推进整批互相独立的四路组。
    template <uint32_t Span>
    __aicore__ inline void Merge(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t groups)
    {
        uint16_t lengths[MERGE_WAYS] = {Span, Span, Span, Span};
        MrgSortSrcList<float> sources(src, src[Span * PAIR_COMPONENTS],
                                      src[Span * MERGE_THIRD_SOURCE * PAIR_COMPONENTS],
                                      src[Span * MERGE_FOURTH_SOURCE * PAIR_COMPONENTS]);
        MrgSort4Info info(lengths, false, MERGE_ALL_SOURCES_MASK, static_cast<uint8_t>(groups));
        MrgSort(dst, sources, info);
        PipeBarrier<PIPE_V>();
    }

    // 每个已排序组只保留前 64 项；被丢弃项不可能进入后续任意 K<=64 的结果。
    template <uint32_t Span>
    __aicore__ inline void Compact(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t groups)
    {
        DataCopyParams copy{static_cast<uint16_t>(groups),
                            static_cast<uint16_t>(PREFIX_ROW_ITEMS * PAIR_BYTES / UB_BLOCK_BYTES),
                            static_cast<uint16_t>((Span - PREFIX_ROW_ITEMS) * PAIR_BYTES / UB_BLOCK_BYTES), 0};
        DataCopy(dst.ReinterpretCast<uint32_t>(), src.ReinterpretCast<uint32_t>(), copy);
        PipeBarrier<PIPE_V>();
    }

    // 只从前缀提取索引，按原值位宽 Gather；两份输出共享一次写回同步。
    __aicore__ inline void EmitRows(uint64_t first, uint32_t rows, uint32_t role)
    {
        auto selected = key_.ReinterpretCast<uint32_t>();
        uint64_t kept = 0;
        GatherMaskParams gather(1, 1, VECTOR_REPEAT_BLOCKS, 0);
        for (uint32_t row = 0; row < rows; ++row)
            GatherMask(selected[row * PREFIX_ROW_ITEMS],
                       pong_[row * NEXT_RUN * PAIR_COMPONENTS].ReinterpretCast<uint32_t>(),
                       uint8_t(GATHER_INDEX_PATTERN), true, PREFIX_ROW_ITEMS * PAIR_COMPONENTS, gather, kept);
        PipeBarrier<PIPE_V>();
        auto offsets = ping_.ReinterpretCast<uint32_t>();
        Muls(offsets.ReinterpretCast<int32_t>(), selected.ReinterpretCast<int32_t>(), static_cast<int32_t>(sizeof(T)),
             rows * PREFIX_ROW_ITEMS);
        PipeBarrier<PIPE_V>();
        auto restored = pong_.ReinterpretCast<Bits>();
        auto original = raw_[role].template ReinterpretCast<Bits>();
        for (uint32_t row = 0; row < rows; ++row)
            Gather(restored[row * PREFIX_ROW_ITEMS], original[row * width_], offsets[row * PREFIX_ROW_ITEMS], 0,
                   PREFIX_ROW_ITEMS);
        PipeBarrier<PIPE_V>();
        auto wide = ping_.ReinterpretCast<int64_t>();
        Cast(wide, selected.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, rows * PREFIX_ROW_ITEMS);
        PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE3>();
        WriteRowsToGm(values_[first * d_->k], restored, rows, d_->k, PREFIX_ROW_ITEMS);
        WriteRowsToGm(indices_[first * d_->k], wide, rows, d_->k, PREFIX_ROW_ITEMS);
        Fence<HardEvent::MTE3_S>();
    }

    const SelectTopKBaseTilingData* d_ = nullptr;
    uint64_t rowsPerCore_ = 0;
    uint32_t width_ = 0;
    TBuf<TPosition::VECCALC> arena_;
    GlobalTensor<Bits> x_, values_;
    GlobalTensor<int64_t> indices_;
    LocalTensor<T> raw_[ROW_INPUT_BUFFER_COUNT];
    LocalTensor<float> key_, ping_, pong_;
    LocalTensor<uint32_t> index_;
};
} // namespace SelectTopK
#endif
