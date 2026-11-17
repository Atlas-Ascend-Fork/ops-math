/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SORT_WITH_INDEX_MERGE_ARCH22_H
#define SORT_WITH_INDEX_MERGE_ARCH22_H

#include "radix_sort_with_index_base.h"
#include "merge_sort_with_index_common.h"

namespace SortWithIndexA2 {

// 短行在单核 UB 内先做 Sort32，再四路归并。排序键组合值的编码和位置，保证等键稳定；
// 排序后按位置取回原值和调用方传入的索引。
template <typename T>
class MergeSortWithIndex : public RadixSortWithIndexBase<T, MERGE_LOCAL_MODE> {
    using Base = RadixSortWithIndexBase<T, MERGE_LOCAL_MODE>;
    using Base::alignedLength_;
    using Base::indexOutputBuf_;
    using Base::indexScratchBuf_;
    using Base::lastAxisNum_;
    using Base::mergePingBuf_;
    using Base::mergePongBuf_;
    using Base::mergePositionsBuf_;
    using Base::valueKeyBuf_;

public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR outputIndex, GM_ADDR workspace,
                                const SortWithIndexTilingDataA2& data)
    {
        Base::Init(x, index, y, outputIndex, workspace, data);
        batchRows_ = data.tileCount;
        rowStride_ = AlignUp(lastAxisNum_, MERGE_SORT_RUN_ELEMENTS);
        if (batchRows_ > 1 && lastAxisNum_ <= MERGE_BATCH_POWER_OF_TWO_LIMIT) {
            rowStride_ = MERGE_SORT_RUN_ELEMENTS;
            while (rowStride_ < lastAxisNum_) {
                rowStride_ *= MERGE_PAIR_LIST_COUNT;
            }
        }
    }

    __aicore__ inline void Process()
    {
        auto positions = mergePositionsBuf_.template Get<float>();
        this->FastPositions(positions, alignedLength_);
        PipeBarrier<PIPE_V>();
        Cast(mergePositionsBuf_.template Get<int32_t>(), positions, RoundMode::CAST_RINT, alignedLength_);
        PipeBarrier<PIPE_V>();
        if (batchRows_ == 1) {
            for (uint64_t row = GetBlockIdx(); row < this->rowNum_; row += GetBlockNum()) {
                ProcessRow(row);
            }
            return;
        }
        // 先均衡各核的实际行数，再划分 UB 批次，避免按整批分配导致部分核多处理一整批。
        const uint32_t rowsPerCore = this->rowNum_ / GetBlockNum();
        const uint32_t extraRows = this->rowNum_ % GetBlockNum();
        const uint64_t begin = uint64_t(GetBlockIdx()) * rowsPerCore +
                               (GetBlockIdx() < extraRows ? GetBlockIdx() : extraRows);
        const uint64_t end = begin + rowsPerCore + (GetBlockIdx() < extraRows ? 1 : 0);
        for (uint64_t row = begin; row < end; row += batchRows_) {
            uint32_t remaining = end - row;
            ProcessBatch(row, remaining < batchRows_ ? remaining : batchRows_);
        }
    }

private:
    __aicore__ inline void MakeStableScores(uint32_t count, bool padTail = true)
    {
        auto scores = indexScratchBuf_.template Get<float>();
        auto scoreBits = indexScratchBuf_.template Get<uint32_t>();
        auto mask = indexOutputBuf_.template Get<uint32_t>();
        // 补齐项排在所有有效编码之后，不参与最终取数。
        if (padTail && count < alignedLength_) {
            Duplicate(scores, MERGE_PADDING_SCORE, alignedLength_);
            PipeBarrier<PIPE_V>();
        }
        Cast(scores, valueKeyBuf_.template Get<int16_t>()[alignedLength_], RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Cast(indexScratchBuf_.template Get<int32_t>(), scores, RoundMode::CAST_RINT, count);
        Duplicate(mask, MERGE_KEY_MASK, count);
        PipeBarrier<PIPE_V>();
        // A2 的 And(count) 按 16 位通道计数；显式使用半字视图，确保每个 INT32 的高低半字都被掩码处理。
        And(scoreBits.template ReinterpretCast<uint16_t>(), scoreBits.template ReinterpretCast<uint16_t>(),
            mask.template ReinterpretCast<uint16_t>(), count * HALFWORDS_PER_INDEX);
        PipeBarrier<PIPE_V>();
        ShiftLeft(scoreBits, scoreBits, uint32_t(MERGE_POSITION_BITS), count);
        PipeBarrier<PIPE_V>();
        Add(indexScratchBuf_.template Get<int32_t>(), indexScratchBuf_.template Get<int32_t>(),
            mergePositionsBuf_.template Get<int32_t>(), count);
        PipeBarrier<PIPE_V>();
        Adds(indexScratchBuf_.template Get<int32_t>(), indexScratchBuf_.template Get<int32_t>(), MERGE_SCORE_BASE_BITS,
             count);
        PipeBarrier<PIPE_V>();
        // 将组合编码按位解释为正规 FP32 后取负，使 Sort32/MrgSort 的降序结果对应键和位置的升序。
        Muls(scores, scores, -1.0f, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline LocalTensor<float> SortPositions(uint32_t sortLength)
    {
        auto ping = mergePingBuf_.template Get<float>();
        auto pong = mergePongBuf_.template Get<float>();
        uint32_t runs = sortLength / MERGE_SORT_RUN_ELEMENTS;
        Sort32(ping, indexScratchBuf_.template Get<float>(), mergePositionsBuf_.template Get<uint32_t>(), runs);
        PipeBarrier<PIPE_V>();
        uint32_t runLength = MERGE_SORT_RUN_ELEMENTS;
        bool inPing = true;
        while (runs > 1) {
            MergeStage(inPing ? pong : ping, inPing ? ping : pong, sortLength, runLength, runs);
            PipeBarrier<PIPE_ALL>();
            inPing = !inPing;
            runLength *= MERGE_LIST_COUNT;
            runs = (runs + MERGE_LIST_COUNT - 1) / MERGE_LIST_COUNT;
        }
        return inPing ? ping : pong;
    }

    __aicore__ inline void WriteResult(uint64_t row, LocalTensor<float> sorted, uint32_t sortLength)
    {
        auto offsets = indexOutputBuf_.template Get<uint32_t>();
        Extract(indexScratchBuf_.template Get<float>(), offsets, sorted, sortLength / MERGE_SORT_RUN_ELEMENTS);
        PipeBarrier<PIPE_V>();
        Muls(indexOutputBuf_.template Get<int32_t>(), indexOutputBuf_.template Get<int32_t>(), int32_t(VALUE_BYTES),
             lastAxisNum_);
        PipeBarrier<PIPE_V>();
        Gather(indexScratchBuf_.template Get<uint16_t>(), valueKeyBuf_.template Get<uint16_t>(), offsets, 0,
               lastAxisNum_);
        this->template EventSync<HardEvent::V_MTE3>();
        uint64_t rowOffset = row * lastAxisNum_;
        DataCopyPad(this->outputGm_[rowOffset], indexScratchBuf_.template Get<uint16_t>(),
                    {1, lastAxisNum_ * VALUE_BYTES, 0, 0, 0});
        this->template EventSync<HardEvent::MTE3_MTE2>();
        DataCopyPad(indexScratchBuf_.template Get<int32_t>(), this->inputIndexGm_[rowOffset],
                    {1, lastAxisNum_ * INDEX_BYTES, 0, 0, 0}, {false, 0, 0, 0});
        this->template EventSync<HardEvent::MTE2_V>();
        Muls(indexOutputBuf_.template Get<int32_t>(), indexOutputBuf_.template Get<int32_t>(),
             int32_t(INDEX_BYTES / VALUE_BYTES), lastAxisNum_);
        PipeBarrier<PIPE_V>();
        Gather(valueKeyBuf_.template Get<int32_t>(), indexScratchBuf_.template Get<int32_t>(), offsets, 0,
               lastAxisNum_);
        this->template EventSync<HardEvent::V_MTE3>();
        DataCopyPad(this->outputIndexGm_[rowOffset], valueKeyBuf_.template Get<int32_t>(),
                    {1, lastAxisNum_ * INDEX_BYTES, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessRow(uint64_t row)
    {
        DataCopyPad(valueKeyBuf_.template Get<uint16_t>(), this->inputGm_[row * lastAxisNum_],
                    {1, lastAxisNum_ * VALUE_BYTES, 0, 0, 0}, {false, 0, 0, 0});
        this->template EventSync<HardEvent::MTE2_V>();
        this->MakeKeys(lastAxisNum_);
        MakeStableScores(lastAxisNum_);
        uint32_t sortLength = AlignUp(lastAxisNum_, MERGE_SORT_RUN_ELEMENTS);
        auto sorted = SortPositions(sortLength);
        WriteResult(row, sorted, sortLength);
    }

    __aicore__ inline void ProcessBatch(uint64_t row, uint32_t rows);

    uint32_t batchRows_ = 1;
    uint32_t rowStride_ = 0;
};
} // namespace SortWithIndexA2

#include "merge_sort_with_index_batch.h"

#endif // SORT_WITH_INDEX_MERGE_ARCH22_H
