/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SORT_WITH_INDEX_MERGE_BATCH_ARCH22_H
#define SORT_WITH_INDEX_MERGE_BATCH_ARCH22_H

#include "merge_sort_with_index_arch22.h"

namespace SortWithIndexA2 {
// GM 中各行连续存放，UB 行跨度按 Sort32 对齐，排序和归并均不跨行。
// 批内位置用于从正确的行取回原值及 INT32 索引。
template <typename T>
__aicore__ inline void MergeSortWithIndex<T>::ProcessBatch(uint64_t row, uint32_t rows)
{
    const uint32_t count = rows * rowStride_;
    const uint32_t valueStride = (rowStride_ * VALUE_BYTES - AlignUp(lastAxisNum_ * VALUE_BYTES, UB_BLOCK_BYTES)) /
                                 UB_BLOCK_BYTES;
    const uint32_t indexStride = (rowStride_ * INDEX_BYTES - AlignUp(lastAxisNum_ * INDEX_BYTES, UB_BLOCK_BYTES)) /
                                 UB_BLOCK_BYTES;
    const uint64_t offset = row * lastAxisNum_;
    DataCopyPad(valueKeyBuf_.template Get<uint16_t>(), this->inputGm_[offset],
                {uint16_t(rows), lastAxisNum_ * VALUE_BYTES, 0, valueStride, 0}, {false, 0, 0, 0});
    this->template EventSync<HardEvent::MTE2_V>();
    this->MakeKeys(count);
    MakeStableScores(count, false);
    if (rowStride_ != lastAxisNum_ && lastAxisNum_ > MERGE_BATCH_POWER_OF_TWO_LIMIT) {
        // 仅最后一个不满 Sort32 的段需要补齐；掩码写入避免覆盖前面的有效编码。
        constexpr uint32_t scoresPerBlock = UB_BLOCK_BYTES / sizeof(float);
        const uint32_t begin = lastAxisNum_ / scoresPerBlock * scoresPerBlock;
        const uint32_t tail = rowStride_ - begin;
        const uint32_t valid = lastAxisNum_ - begin;
        uint64_t mask[2] = {((uint64_t(1) << tail) - 1) & ~((uint64_t(1) << valid) - 1), 0};
        const uint32_t repeatStride = rowStride_ / scoresPerBlock;
        if (repeatStride <= UINT8_MAX) {
            Duplicate(indexScratchBuf_.template Get<float>()[begin], MERGE_PADDING_SCORE, mask,
                      static_cast<uint8_t>(rows), 1, static_cast<uint8_t>(repeatStride));
        } else {
            for (uint32_t r = 0; r < rows; ++r) {
                Duplicate(indexScratchBuf_.template Get<float>()[r * rowStride_ + begin], MERGE_PADDING_SCORE, mask, 1,
                          1, 8);
            }
        }
        PipeBarrier<PIPE_V>();
    } else if (rowStride_ != lastAxisNum_) {
        auto localPositions = indexOutputBuf_.template Get<uint32_t>();
        Duplicate(localPositions, rowStride_ - 1, count);
        PipeBarrier<PIPE_V>();
        And(localPositions.template ReinterpretCast<uint16_t>(), mergePositionsBuf_.template Get<uint16_t>(),
            localPositions.template ReinterpretCast<uint16_t>(), count * HALFWORDS_PER_INDEX);
        PipeBarrier<PIPE_V>();
        Cast(indexOutputBuf_.template Get<float>(), indexOutputBuf_.template Get<int32_t>(), RoundMode::CAST_NONE,
             count);
        PipeBarrier<PIPE_V>();
        Compares<float, uint8_t>(this->maskBuf_.template Get<uint8_t>(), indexOutputBuf_.template Get<float>(),
                                 static_cast<float>(static_cast<int32_t>(lastAxisNum_)), CMPMODE::GE,
                                 AlignUp(count, SIMD_ALIGN_ELEMENTS));
        PipeBarrier<PIPE_V>();
        Duplicate(indexOutputBuf_.template Get<float>(), MERGE_PADDING_SCORE, count);
        PipeBarrier<PIPE_V>();
        Select(indexScratchBuf_.template Get<float>(), this->maskBuf_.template Get<uint8_t>(),
               indexOutputBuf_.template Get<float>(), indexScratchBuf_.template Get<float>(),
               SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        PipeBarrier<PIPE_V>();
    }
    auto ping = mergePingBuf_.template Get<float>();
    auto pong = mergePongBuf_.template Get<float>();
    Sort32(ping, indexScratchBuf_.template Get<float>(), mergePositionsBuf_.template Get<uint32_t>(),
           count / MERGE_SORT_RUN_ELEMENTS);
    PipeBarrier<PIPE_V>();
    bool inPing = true;
    for (uint32_t run = MERGE_SORT_RUN_ELEMENTS; run < rowStride_; run *= MERGE_LIST_COUNT) {
        if (rowStride_ % (run * MERGE_LIST_COUNT) == 0) {
            // 每个硬件归并组都限定在本行内，重复执行时不会合并不同行的数据。
            MergeStage(inPing ? pong : ping, inPing ? ping : pong, count, run, count / run);
        } else {
            for (uint32_t r = 0; r < rows; ++r) {
                uint32_t start = GetSortOffset<float>(r * rowStride_);
                MergeStage((inPing ? pong : ping)[start], (inPing ? ping : pong)[start], rowStride_, run,
                           (rowStride_ + run - 1) / run);
            }
        }
        PipeBarrier<PIPE_ALL>();
        inPing = !inPing;
    }
    auto offsets = indexOutputBuf_.template Get<uint32_t>();
    Extract(indexScratchBuf_.template Get<float>(), offsets, inPing ? ping : pong, count / MERGE_SORT_RUN_ELEMENTS);
    PipeBarrier<PIPE_V>();
    Muls(indexOutputBuf_.template Get<int32_t>(), indexOutputBuf_.template Get<int32_t>(), int32_t(VALUE_BYTES), count);
    PipeBarrier<PIPE_V>();
    Gather(indexScratchBuf_.template Get<uint16_t>(), valueKeyBuf_.template Get<uint16_t>(), offsets, 0, count);
    this->template EventSync<HardEvent::V_MTE3>();
    DataCopyPad(this->outputGm_[offset], indexScratchBuf_.template Get<uint16_t>(),
                {uint16_t(rows), lastAxisNum_ * VALUE_BYTES, valueStride, 0, 0});
    this->template EventSync<HardEvent::MTE3_MTE2>();
    DataCopyPad(indexScratchBuf_.template Get<int32_t>(), this->inputIndexGm_[offset],
                {uint16_t(rows), lastAxisNum_ * INDEX_BYTES, 0, indexStride, 0}, {false, 0, 0, 0});
    this->template EventSync<HardEvent::MTE2_V>();
    Muls(indexOutputBuf_.template Get<int32_t>(), indexOutputBuf_.template Get<int32_t>(),
         int32_t(INDEX_BYTES / VALUE_BYTES), count);
    PipeBarrier<PIPE_V>();
    Gather(valueKeyBuf_.template Get<int32_t>(), indexScratchBuf_.template Get<int32_t>(), offsets, 0, count);
    this->template EventSync<HardEvent::V_MTE3>();
    DataCopyPad(this->outputIndexGm_[offset], valueKeyBuf_.template Get<int32_t>(),
                {uint16_t(rows), lastAxisNum_ * INDEX_BYTES, indexStride, 0, 0});
    PipeBarrier<PIPE_ALL>();
}
} // namespace SortWithIndexA2

#endif // SORT_WITH_INDEX_MERGE_BATCH_ARCH22_H
