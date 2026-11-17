/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RADIX_SORT_WITH_INDEX_BASE_H
#define RADIX_SORT_WITH_INDEX_BASE_H

#include "sort_with_index_common_arch22.h"

namespace SortWithIndexA2 {
using namespace AscendC;

// 全局基数排序与单核短行归并共用的存储。
template <typename T, uint32_t Mode>
class RadixSortWithIndexBase {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR outputIndex, GM_ADDR workspace,
                                const SortWithIndexTilingDataA2& data)
    {
        lastAxisNum_ = data.length;
        alignedLength_ = data.alignedLength;
        rowNum_ = data.rows;
        isDescending_ = data.descending;
        tileNum_ = Mode == MULTI_BLOCK_MODE ? data.tileCount : 1;
        coresPerRow_ = data.localFanout;
        inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(x));
        inputIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(y));
        outputIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(outputIndex));
        InitWorkspace(workspace);
        InitBuffers();
    }

protected:
    __aicore__ inline void InitWorkspace(GM_ADDR workspace)
    {
        if constexpr (Mode == MULTI_BLOCK_MODE) {
            // 生成的入口传入用户 workspace 起点，不能回退指针覆盖系统预留区。
            uint64_t valueBytes = AlignUp(uint64_t(lastAxisNum_) * VALUE_BYTES, UB_BLOCK_BYTES);
            uint64_t indexBytes = AlignUp(uint64_t(lastAxisNum_) * INDEX_BYTES, UB_BLOCK_BYTES);
            uint64_t rowBytes = valueBytes + indexBytes + uint64_t(tileNum_ + coresPerRow_) * HISTOGRAM_RECORD_BYTES;
            histogramRowStride_ = rowBytes / INDEX_BYTES;
            histogramRowsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace + valueBytes + indexBytes));
            workspace += uint64_t(GetBlockIdx() / coresPerRow_) * rowBytes;
            scratchValueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(workspace));
            scratchIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace + valueBytes));
            histogramGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ int32_t*>(workspace + (tileNum_ == 1 ? 0 : valueBytes + indexBytes)));
        }
    }
    __aicore__ inline void InitBuffers()
    {
        pipe_.InitBuffer(valueKeyBuf_, alignedLength_ * VALUE_KEY_BYTES);
        if constexpr (Mode == MERGE_LOCAL_MODE) {
            pipe_.InitBuffer(indexScratchBuf_, (alignedLength_ + WORK_GUARD_ELEMENTS) * INDEX_BYTES);
        }
        pipe_.InitBuffer(indexOutputBuf_, (alignedLength_ + WORK_GUARD_ELEMENTS) * INDEX_BYTES);
        if constexpr (Mode == MULTI_BLOCK_MODE) {
            pipe_.InitBuffer(valueOutputBuf_, (alignedLength_ + WORK_GUARD_ELEMENTS) * VALUE_BYTES);
            pipe_.InitBuffer(keyOutputBuf_, (alignedLength_ + WORK_GUARD_ELEMENTS) * VALUE_BYTES);
            pipe_.InitBuffer(digitScratchBuf_, alignedLength_ * INDEX_BYTES);
            pipe_.InitBuffer(prefixRecordsBuf_,
                             (tileNum_ < PREFIX_BATCH_TILES ? tileNum_ : PREFIX_BATCH_TILES) * HISTOGRAM_RECORD_BYTES);
        }
        if constexpr (Mode == MERGE_LOCAL_MODE) {
            pipe_.InitBuffer(mergePingBuf_, alignedLength_ * MERGE_PROPOSAL_BYTES);
            pipe_.InitBuffer(mergePongBuf_, alignedLength_ * MERGE_PROPOSAL_BYTES);
            pipe_.InitBuffer(mergePositionsBuf_, alignedLength_ * INDEX_BYTES);
        }
        pipe_.InitBuffer(maskBuf_, AlignUp(alignedLength_ / BITS_PER_BYTE, UB_BLOCK_BYTES));
    }

    template <HardEvent event>
    __aicore__ inline void EventSync()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
        SetFlag<event>(id);
        WaitFlag<event>(id);
    }
    __aicore__ inline void MakeKeys(uint32_t count) { MakeKeys(valueKeyBuf_.template Get<int16_t>(), count); }
    __aicore__ inline void MakeKeys(LocalTensor<int16_t> bits, uint32_t count)
    {
        auto key = valueKeyBuf_.template Get<int16_t>()[alignedLength_];
        LocalTensor<int16_t> a, b;
        if constexpr (Mode == MULTI_BLOCK_MODE) {
            a = valueOutputBuf_.template Get<int16_t>();
            b = keyOutputBuf_.template Get<int16_t>();
        } else {
            a = indexScratchBuf_.template Get<int16_t>();
            b = indexScratchBuf_.template Get<int16_t>()[alignedLength_];
        }
        auto fp = indexOutputBuf_.template Get<float>();
        auto cmp = maskBuf_.template Get<uint8_t>();
        MakeSortKeys<T>(bits, key, a, b, fp, cmp, isDescending_, count);
    }
    __aicore__ inline void FastPositions(LocalTensor<float> positions, uint32_t count)
    {
        SortWithIndexA2::FastPositions(positions, count);
    }
    TPipe pipe_;
    TBuf<TPosition::VECCALC> valueKeyBuf_, indexScratchBuf_, valueOutputBuf_, indexOutputBuf_, keyOutputBuf_,
        digitScratchBuf_, maskBuf_, prefixRecordsBuf_;
    TBuf<TPosition::VECCALC> mergePingBuf_, mergePongBuf_, mergePositionsBuf_;
    GlobalTensor<uint16_t> inputGm_, outputGm_, scratchValueGm_;
    GlobalTensor<int32_t> inputIndexGm_, outputIndexGm_, scratchIndexGm_, histogramGm_, histogramRowsGm_;
    uint64_t histogramRowStride_;
    uint32_t lastAxisNum_, alignedLength_, rowNum_, tileNum_, coresPerRow_;
    bool isDescending_;
};
} // namespace SortWithIndexA2

#endif // RADIX_SORT_WITH_INDEX_BASE_H
