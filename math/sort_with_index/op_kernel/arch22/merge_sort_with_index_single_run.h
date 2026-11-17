/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SORT_WITH_INDEX_MERGE_SINGLE_RUN_ARCH22_H
#define SORT_WITH_INDEX_MERGE_SINGLE_RUN_ARCH22_H

#include "merge_sort_with_index_native.h"

namespace SortWithIndexA2 {
// 整行的两份排序数组均放入 UB。有限值记录直接携带 INT32 索引；
// 含 NaN 时按规范化键排序位置，再取回原值和索引，无需将记录写入 GM。
template <typename T>
class MergeSortWithIndexSingleRun : public NativeMergeSortWithIndex<T> {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR outputIndex, GM_ADDR workspace,
                                const SortWithIndexTilingDataA2& data)
    {
        NativeMergeSortWithIndex<T>::template Init<false>(x, index, y, outputIndex, workspace, data);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t wave = 0; wave < this->rowsPerCore_; ++wave) {
            const uint64_t row = uint64_t(GetBlockIdx()) * this->rowsPerCore_ + wave;
            if (row >= this->rows_)
                continue;
            this->nativeValues_ = this->UseFiniteValues(row);
            this->carryPayload_ = this->nativeValues_;
            this->halfRecords_ = this->nativeValues_ && AscendC::IsSameType<T, half>::value;
            this->overrideOff_ = 0;
            this->overrideCount_ = this->length_;
            auto sorted = this->halfRecords_ ? this->HalfLocalSort(row, 0) :
                                               this->NativeLocalSort(row, 0, false, false);
            auto records = this->template At<float>(this->chunk_ * MERGE_PROPOSAL_BYTES);
            if (sorted.GetPhyAddr() != records.GetPhyAddr()) {
                DataCopy(records, sorted, this->chunk_ * MERGE_PROPOSAL_ELEMENTS);
                PipeBarrier<PIPE_V>();
            }
            if (this->carryPayload_) {
                WriteCarriedRow(row, records);
            } else {
                WriteRow(row, records);
            }
        }
    }

private:
    __aicore__ inline void WriteCarriedRow(uint64_t row, LocalTensor<float> records)
    {
        const uint32_t count = this->length_;
        const uint32_t sortCount = AlignUp(count, MERGE_SORT_RUN_ELEMENTS);
        // 排序记录占 [8*A,16*A) 字节；低地址排序数组复用为值和索引区，BF16 转换使用 [16*A,18*A)，避免覆盖记录。
        auto tokens = this->template At<uint32_t>(this->chunk_ * INDEX_BYTES);
        auto values = this->template At<uint16_t>();
        if constexpr (AscendC::IsSameType<T, half>::value) {
            uint64_t selected = 0;
            // 每条 half 记录含四个半字；改用限定长度的 GatherMask，避免 half Extract 扩大索引写入范围。
            GatherMask(values, records.template ReinterpretCast<uint16_t>(), uint8_t(3), true,
                       sortCount * MERGE_PROPOSAL_BYTES / VALUE_BYTES, {1, 1, 0, 0}, selected);
            PipeBarrier<PIPE_V>();
            GatherMask(tokens, records.template ReinterpretCast<uint32_t>(), uint8_t(2), true,
                       sortCount * MERGE_PROPOSAL_ELEMENTS, {1, 1, 0, 0}, selected);
            PipeBarrier<PIPE_V>();
            if (!this->descending_) {
                Adds(values.template ReinterpretCast<int16_t>(), values.template ReinterpretCast<int16_t>(),
                     int16_t(-32768), count);
                PipeBarrier<PIPE_V>();
            }
        } else {
            auto scores = this->template At<float>();
            Extract(scores, tokens, records, sortCount / MERGE_SORT_RUN_ELEMENTS);
            PipeBarrier<PIPE_V>();
            if (!this->descending_) {
                Adds(scores.template ReinterpretCast<int32_t>(), scores.template ReinterpretCast<int32_t>(),
                     int32_t(-2147483647 - 1), count);
                PipeBarrier<PIPE_V>();
            }
            values = this->template At<uint16_t>(this->chunk_ * MERGE_PROPOSAL_BYTES * MERGE_WORKSPACE_BUFFERS);
            Cast(values.template ReinterpretCast<T>(), scores, RoundMode::CAST_RINT, count);
            PipeBarrier<PIPE_V>();
        }
        EventSync<HardEvent::V_MTE3>();
        DataCopyPad(this->output_[row * count], values, {1, count * VALUE_BYTES, 0, 0, 0});
        DataCopyPad(this->outputIndex_[row * count], tokens.template ReinterpretCast<int32_t>(),
                    {1, count * INDEX_BYTES, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void WriteRow(uint64_t row, LocalTensor<float> records)
    {
        const uint32_t count = this->length_;
        const uint32_t sortCount = AlignUp(count, MERGE_SORT_RUN_ELEMENTS);
        auto data = this->template At<int32_t>();
        auto offsets = this->template At<uint32_t>(this->chunk_ * INDEX_BYTES);
        auto output = this->template At<int32_t>(this->chunk_ * MERGE_PROPOSAL_BYTES * MERGE_WORKSPACE_BUFFERS);
        uint64_t selected = 0;
        GatherMask(offsets, records.template ReinterpretCast<uint32_t>(), uint8_t(2), true,
                   sortCount * MERGE_PROPOSAL_ELEMENTS, {1, 1, 0, 0}, selected);
        PipeBarrier<PIPE_V>();
        DataCopyPad(data.template ReinterpretCast<uint16_t>(), this->input_[row * count],
                    {1, count * VALUE_BYTES, 0, 0, 0}, {false, 0, 0, 0});
        EventSync<HardEvent::MTE2_V>();
        ShiftLeft(offsets, offsets, uint32_t(VALUE_BYTE_SHIFT), count);
        PipeBarrier<PIPE_V>();
        Gather(output.template ReinterpretCast<uint16_t>(), data.template ReinterpretCast<uint16_t>(), offsets, 0,
               count);
        PipeBarrier<PIPE_V>();
        if (!this->nativeValues_) {
            CanonicalizeSortValues<T>(output.template ReinterpretCast<uint16_t>(),
                                      records.template ReinterpretCast<uint16_t>(),
                                      this->small_.template Get<uint8_t>(), count);
        }
        EventSync<HardEvent::V_MTE3>();
        DataCopyPad(this->output_[row * count], output.template ReinterpretCast<uint16_t>(),
                    {1, count * VALUE_BYTES, 0, 0, 0});
        EventSync<HardEvent::MTE3_V>();
        // MTE2 将复用原值所在区域；仅等待 MTE3_V 无法保证前面的向量 Gather 已完成，需额外同步。
        EventSync<HardEvent::V_MTE2>();
        DataCopyPad(data, this->inputIndex_[row * count], {1, count * INDEX_BYTES, 0, 0, 0}, {false, 0, 0, 0});
        EventSync<HardEvent::MTE2_V>();
        ShiftLeft(offsets, offsets, uint32_t(VALUE_BYTE_SHIFT), count);
        PipeBarrier<PIPE_V>();
        Gather(output, data, offsets, 0, count);
        EventSync<HardEvent::V_MTE3>();
        DataCopyPad(this->outputIndex_[row * count], output, {1, count * INDEX_BYTES, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
    }
};
} // namespace SortWithIndexA2
#endif // SORT_WITH_INDEX_MERGE_SINGLE_RUN_ARCH22_H
