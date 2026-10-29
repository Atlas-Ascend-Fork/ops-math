/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file radix_sort_top_k_multi_core_optimization.h
 * \brief radix_sort_top_k_multi_core_optimization impl
 */

#ifndef RADIX_SORT_TOP_K_INTER_CORE_TEMPLATE_OPTIMIZATION_H
#define RADIX_SORT_TOP_K_INTER_CORE_TEMPLATE_OPTIMIZATION_H

#include "kernel_operator.h"
#include "top_k_small_size_bitonic_sort.h"
#include "top_k_util_type_simd.h"
#include "radix_topk_util.h"
#include "top_k_small_bitonic_packed_simt.h"
#include <cstdint>
#include <type_traits>
#include <algorithm>

using namespace AscendC;
using namespace topkV2;

// Small-k UB threshold selection belongs to the inter-core template below.
namespace topkV2 {
template <typename T>
__simt_callee__ inline uint64_t TopKUbIntegerKey(T value)
{
    uint64_t key = static_cast<uint64_t>(value);
    if constexpr (std::is_signed_v<T>) {
        key ^= uint64_t(1) << 63;
    }
    return key;
}

// The existing multi-core template owns all copying and synchronization.
// Replace only the per-tile and final small-k selection, retaining exact Bitonic output.
template <typename T, bool Largest>
__simt_callee__ inline uint64_t FindTopKUbThreshold(__ubuf__ const T* input, uint32_t n, uint32_t k, uint32_t lane,
                                                    uint64_t low, uint64_t high)
{
    uint32_t kth = Largest ? n - k : k - 1;
    if (k == n) {
        low = Largest ? low : high;
        high = low;
    }
    while (low < high) {
        uint64_t mid = low + (high - low) / 2;
        uint32_t count = 0;
        for (uint32_t i = lane; i < n; i += BITONIC_SMALL_TOPK_SIZE) {
            count += TopKUbIntegerKey(input[i]) <= mid ? 1 : 0;
        }
        count = asc_reduce_add(count);
        if (count > kth) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    return low;
}

template <typename T, bool Largest, bool HasIndices>
__simt_callee__ inline void FinalizeTopKUbSelection(__ubuf__ const T* input, __ubuf__ const int32_t* inputIndices,
                                                    __ubuf__ T* output, __ubuf__ int32_t* outputIndices, uint32_t k,
                                                    uint32_t lane, uint32_t selected)
{
    bool valid = lane < k;
    T value = valid ? input[selected] : T(0);
    int32_t index = valid ? (HasIndices ? inputIndices[selected] : static_cast<int32_t>(selected)) : -1;
    if constexpr (HasIndices) {
        BitonicSmallFinalizeExactSelection<T, int32_t, Largest>(value, index, valid, k);
    }
    if (valid) {
        output[lane] = value;
        outputIndices[lane] = index;
    }
}

template <typename T, bool Largest, bool HasIndices>
__simt_vf__ __aicore__ LAUNCH_BOUND(BITONIC_SMALL_TOPK_SIZE) void TopKUbThresholdSelect(
    uint32_t n, uint32_t k, __ubuf__ const T* input, __ubuf__ const int32_t* inputIndices, __ubuf__ T* output,
    __ubuf__ int32_t* outputIndices, __ubuf__ uint32_t* selectedPath)
{
    uint32_t lane = threadIdx.x;
    uint64_t low = UINT64_MAX;
    uint64_t high = 0;
    for (uint32_t i = lane; i < n; i += BITONIC_SMALL_TOPK_SIZE) {
        uint64_t key = TopKUbIntegerKey(input[i]);
        low = key < low ? key : low;
        high = key > high ? key : high;
    }
#pragma unroll
    for (uint32_t shift = BITONIC_SMALL_TOPK_SIZE / 2U; shift > 0; shift /= 2U) {
        uint64_t peerLow = asc_shfl_xor(low, shift, BITONIC_SMALL_TOPK_SIZE);
        uint64_t peerHigh = asc_shfl_xor(high, shift, BITONIC_SMALL_TOPK_SIZE);
        low = peerLow < low ? peerLow : low;
        high = peerHigh > high ? peerHigh : high;
    }
    // Native radix selection is preferable for wide 64-bit ranges. Report the
    // decision through the existing scratch buffer; the caller retains fallback.
    bool narrow = high - low <= UINT16_MAX;
    if (lane == 0) {
        selectedPath[0] = narrow ? 1U : 0U;
    }
    if (!narrow) {
        return;
    }
    // With one distinct value, the two gather passes would select the first k
    // elements in source order. Copy that prefix directly.
    if (low == high) {
        FinalizeTopKUbSelection<T, Largest, HasIndices>(input, inputIndices, output, outputIndices, k, lane, lane);
        return;
    }
    low = FindTopKUbThreshold<T, Largest>(input, n, k, lane, low, high);
    uint32_t selected = UINT32_MAX;
    uint32_t gathered = 0;
    for (uint32_t pass = 0; pass < 2; ++pass) {
        for (uint32_t base = 0; base < n && gathered < k; base += BITONIC_SMALL_TOPK_SIZE) {
            uint32_t i = base + lane;
            uint64_t key = i < n ? TopKUbIntegerKey(input[i]) : 0;
            bool chosen = i < n && (pass == 0 ? (Largest ? key > low : key < low) : key == low);
            uint32_t mask = asc_ballot(chosen);
            uint32_t count = __popc(mask);
            if (lane >= gathered && lane < gathered + count && lane < k) {
                selected = base + BitonicSmallNthSetBit(mask, lane - gathered);
            }
            gathered += count;
        }
    }
    FinalizeTopKUbSelection<T, Largest, HasIndices>(input, inputIndices, output, outputIndices, k, lane, selected);
}
} // namespace topkV2

// 类比当前文件
template <typename T, bool IS_LARGEST, bool IS_SORT, typename T_INDEX_TO, bool IS_BITONIC_SORT = false>
struct RadixSortTopKMultiCoreOptimization {
    __aicore__ inline RadixSortTopKMultiCoreOptimization(){};
    __aicore__ inline void Init(GM_ADDR inputValue, GM_ADDR k, GM_ADDR value, GM_ADDR indices, GM_ADDR workSpace,
                                const TopKV2TilingDataSimd* tilingData);
    __aicore__ inline void InitPara(GM_ADDR inputValue, GM_ADDR k, GM_ADDR value, GM_ADDR indices, GM_ADDR workSpace,
                                    const TopKV2TilingDataSimd* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessSingleLoop(GlobalTensor<T> inputX, int32_t sortLoopRound);
    __aicore__ inline void ProcessTileTopK(GlobalTensor<T> inputX, uint32_t unsortedAxisId);
    __aicore__ inline void ProcessOneTile(GlobalTensor<T> inputX, uint32_t unsortedAxisId, uint32_t tileId,
                                          int32_t remainTileDataNum);
    __aicore__ inline void ComputeTileTopK(LocalTensor<T> topkOutValue, LocalTensor<int32_t> topkOutIndexValue,
                                           LocalTensor<T> xLocal, LocalTensor<uint8_t> shareTmpBuffer,
                                           int32_t currTileNum);
    __aicore__ inline void ProcessFinalTopK(int32_t sortLoopRound);
    __aicore__ inline void ComputeFinalTopK(LocalTensor<T> topkOutValue, LocalTensor<int32_t> topkOutIndexValue,
                                            LocalTensor<T> xLocal, LocalTensor<int32_t> xIndexLocal,
                                            LocalTensor<uint8_t> shareTmpBuffer);
    template <bool HAS_INDICES>
    __aicore__ inline bool TryUbThresholdSelect(LocalTensor<T> input, LocalTensor<int32_t> inputIndices,
                                                LocalTensor<T> output, LocalTensor<int32_t> outputIndices,
                                                LocalTensor<uint8_t> shareTmpBuffer, uint32_t n, uint32_t k);
    __aicore__ inline LocalTensor<T_INDEX_TO> ConvertIndexType(LocalTensor<int32_t> topkOutIndexValue);
    __aicore__ inline void CopyDataIn(GlobalTensor<T> inputX, uint64_t tileOffset, uint32_t currTileSize);
    __aicore__ inline void CopyIndexIn(GlobalTensor<int32_t> inputX, uint64_t tileOffset, uint32_t currTileSize);
    __aicore__ inline void CopyFinalResultToGm(GlobalTensor<T> dataTensor, uint64_t dataOffset,
                                               GlobalTensor<T_INDEX_TO> indexTensor, uint64_t indexOffset);
    __aicore__ inline void CopyIndexOutWithOffset(GlobalTensor<T> dataTensor, uint64_t dataOffset,
                                                  GlobalTensor<int32_t> indexTensor, uint64_t indexOffset);

public:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECIN, 1> inQueueIndexX_;
    TQue<QuePosition::VECOUT, 1> topkOutValueQueue_;
    TQue<QuePosition::VECOUT, 1> tempIndexConversionQueue_;
    TQue<QuePosition::VECOUT, 1> topkOutIndexQueue_;

    GlobalTensor<T> tempSortResultDataGm_;
    GlobalTensor<int32_t> tempSortIndexDataGm_;
    LocalTensor<int32_t> srcIndexLocal;
    TBuf<TPosition::VECCALC> topKApiTmpTBuf_;

    GM_ADDR workspace_;
    uint32_t lastDimTileNum_ = 0;
    uint32_t totalDataNum_ = 0;
    uint32_t numTileData_ = 0;
    uint32_t nowTileSize_ = 0;
    uint32_t unsortedDimNum_ = 0;
    uint32_t unsortedDimParallel_ = 0;
    uint32_t sortLoopTimes_ = 0;
    uint32_t lastDimRealCore_ = 0;
    uint32_t topKApiTmpSize_ = 0;
    uint32_t topkValueInput_ = 0;
    uint32_t tailDataSize_ = 0;
    GlobalTensor<T> inputValueGm_;
    // output
    GlobalTensor<T> topkValueGm_;
    GlobalTensor<T_INDEX_TO> topkValueIndexGm_;
};

template <typename T, bool IS_LARGEST, bool IS_SORT, typename T_INDEX_TO, bool IS_BITONIC_SORT>
__aicore__ inline void RadixSortTopKMultiCoreOptimization<T, IS_LARGEST, IS_SORT, T_INDEX_TO, IS_BITONIC_SORT>::Init(
    GM_ADDR inputValue, GM_ADDR k, GM_ADDR value, GM_ADDR indices, GM_ADDR workSpace,
    const TopKV2TilingDataSimd* tilingData)
{
    // init para
    uint32_t workSpaceOffset = 0;
    InitPara(inputValue, k, value, indices, workSpace, tilingData);
    pipe.InitBuffer(inQueueX_, 1, ROUND_UP_AGLIN(numTileData_) * sizeof(T));
    pipe.InitBuffer(inQueueIndexX_, 1, ROUND_UP_AGLIN(numTileData_) * sizeof(int32_t));
    pipe.InitBuffer(topkOutIndexQueue_, 1, ROUND_UP_AGLIN(topkValueInput_ * sizeof(int32_t)));
    pipe.InitBuffer(topkOutValueQueue_, 1, ROUND_UP_AGLIN(topkValueInput_ * sizeof(T)));
    pipe.InitBuffer(tempIndexConversionQueue_, 1, ROUND_UP_AGLIN(topkValueInput_ * sizeof(T_INDEX_TO)));
    pipe.InitBuffer(topKApiTmpTBuf_, ROUND_UP_AGLIN(topKApiTmpSize_));
    uint32_t oneBlockNum = topkV2::UB_AGLIN_VALUE / static_cast<uint32_t>(sizeof(T));
    uint32_t oneBlockNumB32 = topkV2::UB_AGLIN_VALUE / static_cast<uint32_t>(sizeof(int32_t));
    uint32_t sortResultOffset = CeilDivMul(unsortedDimParallel_ * topkValueInput_ * lastDimTileNum_, oneBlockNum);
    tempSortResultDataGm_.SetGlobalBuffer((__gm__ T*)workspace_, sortResultOffset);
    workSpaceOffset += sortResultOffset * static_cast<uint32_t>(sizeof(T)) / static_cast<uint32_t>(sizeof(int32_t));
    uint32_t sortIndexOffset = CeilDivMul(unsortedDimParallel_ * topkValueInput_ * lastDimTileNum_, oneBlockNumB32);
    tempSortIndexDataGm_.SetGlobalBuffer((__gm__ int32_t*)workspace_ + workSpaceOffset, sortIndexOffset);
}

template <typename T, bool IS_LARGEST, bool IS_SORT, typename T_INDEX_TO, bool IS_BITONIC_SORT>
__aicore__ inline void
RadixSortTopKMultiCoreOptimization<T, IS_LARGEST, IS_SORT, T_INDEX_TO, IS_BITONIC_SORT>::InitPara(
    GM_ADDR inputValue, GM_ADDR k, GM_ADDR value, GM_ADDR indices, GM_ADDR workSpace,
    const TopKV2TilingDataSimd* tilingData)
{
    workspace_ = workSpace;
    numTileData_ = tilingData->numTileDataSize;
    lastDimRealCore_ = tilingData->lastDimNeedCore;
    totalDataNum_ = tilingData->lastAxisNum;
    lastDimTileNum_ = tilingData->lastDimTileNum;
    unsortedDimParallel_ = tilingData->unsortedDimParallel;
    unsortedDimNum_ = tilingData->unsortedDimNum;
    sortLoopTimes_ = tilingData->sortLoopTimes;
    topkValueInput_ = tilingData->topKRealValue;
    nowTileSize_ = tilingData->numTileDataSize;
    topKApiTmpSize_ = tilingData->topkAcApiTmpBufferSize;
    inputValueGm_.SetGlobalBuffer((__gm__ T*)(inputValue));
    // output
    topkValueGm_.SetGlobalBuffer((__gm__ T*)(value));
    topkValueIndexGm_.SetGlobalBuffer((__gm__ T_INDEX_TO*)(indices));
}

template <typename T, bool IS_LARGEST, bool IS_SORT, typename T_INDEX_TO, bool IS_BITONIC_SORT>
__aicore__ inline void
RadixSortTopKMultiCoreOptimization<T, IS_LARGEST, IS_SORT, T_INDEX_TO, IS_BITONIC_SORT>::Process()
{
    for (int32_t i = 0; i < this->sortLoopTimes_; i++) {
        uint64_t loopOffset = i * unsortedDimParallel_ * totalDataNum_;
        ProcessSingleLoop(inputValueGm_[loopOffset], i);
    }
}

template <typename T, bool IS_LARGEST, bool IS_SORT, typename T_INDEX_TO, bool IS_BITONIC_SORT>
__aicore__ inline void RadixSortTopKMultiCoreOptimization<T, IS_LARGEST, IS_SORT, T_INDEX_TO,
                                                          IS_BITONIC_SORT>::ProcessSingleLoop(GlobalTensor<T> inputX,
                                                                                              int32_t sortLoopRound)
{
    uint32_t unsortedAxisId = GetBlockIdx() / lastDimRealCore_;
    uint32_t unsortedDimIndex = unsortedAxisId + sortLoopRound * unsortedDimParallel_;
    // inUnsortedDimRange决定是否执行本次计算流程，超出Batch轴数量范围不执行
    bool inUnsortedDimRange = unsortedDimIndex >= unsortedDimNum_ ? false : true;
    // 超出需要的核心数不执行
    if (GetBlockIdx() >= lastDimTileNum_ * unsortedDimParallel_) {
        inUnsortedDimRange = false;
    }
    tailDataSize_ = totalDataNum_ % numTileData_;

    if (inUnsortedDimRange) {
        ProcessTileTopK(inputX, unsortedAxisId);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll();
    if (inUnsortedDimRange) {
        ProcessFinalTopK(sortLoopRound);
    }
}

template <typename T, bool IS_LARGEST, bool IS_SORT, typename T_INDEX_TO, bool IS_BITONIC_SORT>
__aicore__ inline void RadixSortTopKMultiCoreOptimization<T, IS_LARGEST, IS_SORT, T_INDEX_TO,
                                                          IS_BITONIC_SORT>::ProcessTileTopK(GlobalTensor<T> inputX,
                                                                                            uint32_t unsortedAxisId)
{
    int tileCount = (totalDataNum_ + numTileData_ - 1) / numTileData_;
    uint32_t startTileId = GetBlockIdx() % lastDimRealCore_;
    for (uint32_t tileId = startTileId; tileId < tileCount; tileId += lastDimRealCore_) {
        int32_t remainTileDataNum = totalDataNum_ - static_cast<int32_t>(tileId * numTileData_);
        if (remainTileDataNum < 0) {
            break;
        }
        ProcessOneTile(inputX, unsortedAxisId, tileId, remainTileDataNum);
    }
}

template <typename T, bool IS_LARGEST, bool IS_SORT, typename T_INDEX_TO, bool IS_BITONIC_SORT>
__aicore__ inline void
RadixSortTopKMultiCoreOptimization<T, IS_LARGEST, IS_SORT, T_INDEX_TO, IS_BITONIC_SORT>::ProcessOneTile(
    GlobalTensor<T> inputX, uint32_t unsortedAxisId, uint32_t tileId, int32_t remainTileDataNum)
{
    uint64_t tileOffset = tileId * numTileData_;
    int32_t currTileNum = TopkGetMin<int32_t>(remainTileDataNum, static_cast<int32_t>(numTileData_));
    uint32_t inputXUnsortedAxisOffset = unsortedAxisId * totalDataNum_;
    CopyDataIn(inputX[inputXUnsortedAxisOffset], tileOffset, currTileNum);
    LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
    LocalTensor<T> topkOutValue = topkOutValueQueue_.AllocTensor<T>();
    LocalTensor<int32_t> topkOutIndexValue = topkOutIndexQueue_.AllocTensor<int32_t>();
    LocalTensor<uint8_t> shareTmpBuffer = topKApiTmpTBuf_.Get<uint8_t>();
    ComputeTileTopK(topkOutValue, topkOutIndexValue, xLocal, shareTmpBuffer, currTileNum);

    int32_t offsetValue = static_cast<int32_t>(tileOffset);
    AscendC::Adds(topkOutIndexValue, topkOutIndexValue, offsetValue, this->topkValueInput_);

    topkOutValueQueue_.EnQue<T>(topkOutValue);
    topkOutIndexQueue_.EnQue<int32_t>(topkOutIndexValue);

    uint64_t outPutInlineOffset = tileId * this->topkValueInput_;
    uint64_t outPutInterLineOffset = unsortedAxisId * topkValueInput_ * lastDimTileNum_;
    uint64_t outOffset = outPutInterLineOffset + outPutInlineOffset;
    CopyIndexOutWithOffset(tempSortResultDataGm_, outOffset, tempSortIndexDataGm_, outOffset);
    inQueueX_.FreeTensor(xLocal);
    topkOutValueQueue_.FreeTensor(topkOutValue);
    topkOutIndexQueue_.FreeTensor(topkOutIndexValue);
}

template <typename T, bool IS_LARGEST, bool IS_SORT, typename T_INDEX_TO, bool IS_BITONIC_SORT>
template <bool HAS_INDICES>
__aicore__ inline bool
RadixSortTopKMultiCoreOptimization<T, IS_LARGEST, IS_SORT, T_INDEX_TO, IS_BITONIC_SORT>::TryUbThresholdSelect(
    LocalTensor<T> input, LocalTensor<int32_t> inputIndices, LocalTensor<T> output, LocalTensor<int32_t> outputIndices,
    LocalTensor<uint8_t> shareTmpBuffer, uint32_t n, uint32_t k)
{
    if constexpr (IS_BITONIC_SORT && sizeof(T) == sizeof(uint64_t) && std::is_integral_v<T>) {
        if (topkValueInput_ <= BITONIC_SMALL_TOPK_SIZE) {
            uint32_t selectedCount = HAS_INDICES ? k : TopkGetMin<uint32_t>(n, k);
            asc_vf_call<TopKUbThresholdSelect<T, IS_LARGEST, HAS_INDICES>>(
                dim3(BITONIC_SMALL_TOPK_SIZE), n, selectedCount, (__ubuf__ T*)input.GetPhyAddr(),
                (__ubuf__ int32_t*)inputIndices.GetPhyAddr(), (__ubuf__ T*)output.GetPhyAddr(),
                (__ubuf__ int32_t*)outputIndices.GetPhyAddr(), (__ubuf__ uint32_t*)shareTmpBuffer.GetPhyAddr());
            event_t rangeEvent = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(rangeEvent);
            WaitFlag<HardEvent::V_S>(rangeEvent);
            if (shareTmpBuffer.ReinterpretCast<uint32_t>().GetValue(0) != 0) {
                return true;
            }
            event_t reuseEvent = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(reuseEvent);
            WaitFlag<HardEvent::S_V>(reuseEvent);
        }
    }
    return false;
}

template <typename T, bool IS_LARGEST, bool IS_SORT, typename T_INDEX_TO, bool IS_BITONIC_SORT>
__aicore__ inline void
RadixSortTopKMultiCoreOptimization<T, IS_LARGEST, IS_SORT, T_INDEX_TO, IS_BITONIC_SORT>::ComputeTileTopK(
    LocalTensor<T> topkOutValue, LocalTensor<int32_t> topkOutIndexValue, LocalTensor<T> xLocal,
    LocalTensor<uint8_t> shareTmpBuffer, int32_t currTileNum)
{
    if (TryUbThresholdSelect<false>(xLocal, topkOutIndexValue, topkOutValue, topkOutIndexValue, shareTmpBuffer,
                                    static_cast<uint32_t>(currTileNum), topkValueInput_)) {
        return;
    }
    // Normalize only the UB copy; workspace candidates retain canonical comparison bits.
    if constexpr (IS_BITONIC_SORT) {
        NormalizeRadixFloatInput<T>(xLocal, 1U, static_cast<uint32_t>(currTileNum), static_cast<uint32_t>(currTileNum));
    }
    // topk
    static constexpr TopKConfig topkConfig{TopKAlgo::RADIX_SELECT, TopKOrder::UNSET, IS_SORT && !IS_BITONIC_SORT};
    TopKInfo topKInfo;
    topKInfo.outter = 1;
    topKInfo.inner = ROUND_UP_AGLIN(currTileNum);
    topKInfo.n = currTileNum;
    LocalTensor<bool> emptyFinishLocal;
    TopkTiling emptyTopkTiling;
    // topk高阶api调用需要满足topKInfo.n(实际数据长度) >= k >= 1, 要比较尾块和topk的大小，取较小者
    int32_t aglinMinTopkValue = TopkGetMin<int32_t>(currTileNum, static_cast<int32_t>(this->topkValueInput_));
    AscendC::TopK<T, false, false, false, TopKMode::TOPK_NORMAL, topkConfig>(
        topkOutValue, topkOutIndexValue, xLocal, srcIndexLocal, emptyFinishLocal, shareTmpBuffer, aglinMinTopkValue,
        emptyTopkTiling, topKInfo, IS_LARGEST);
}

template <typename T, bool IS_LARGEST, bool IS_SORT, typename T_INDEX_TO, bool IS_BITONIC_SORT>
__aicore__ inline void RadixSortTopKMultiCoreOptimization<T, IS_LARGEST, IS_SORT, T_INDEX_TO,
                                                          IS_BITONIC_SORT>::ProcessFinalTopK(int32_t sortLoopRound)
{
    // copy data from gm to ub
    uint32_t realLastAxisDim = unsortedDimParallel_;
    uint32_t tailLastDimNum = this->unsortedDimNum_ % unsortedDimParallel_;
    if (tailLastDimNum != 0 && sortLoopRound == this->sortLoopTimes_ - 1) {
        realLastAxisDim = tailLastDimNum;
    }
    if (GetBlockIdx() < realLastAxisDim) {
        uint64_t offset = GetBlockIdx() * topkValueInput_ * lastDimTileNum_;
        CopyDataIn(tempSortResultDataGm_, offset, topkValueInput_ * lastDimTileNum_);
        CopyIndexIn(tempSortIndexDataGm_, offset, topkValueInput_ * lastDimTileNum_);
        LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
        LocalTensor<int32_t> xIndexLocal = inQueueIndexX_.DeQue<int32_t>();
        LocalTensor<T> topkOutValue = topkOutValueQueue_.AllocTensor<T>();
        LocalTensor<int32_t> topkOutIndexValue = topkOutIndexQueue_.AllocTensor<int32_t>();
        LocalTensor<uint8_t> shareTmpBuffer = topKApiTmpTBuf_.Get<uint8_t>();
        ComputeFinalTopK(topkOutValue, topkOutIndexValue, xLocal, xIndexLocal, shareTmpBuffer);
        if constexpr (IS_BITONIC_SORT) {
            uint64_t sourceRow = static_cast<uint64_t>(sortLoopRound) * unsortedDimParallel_ + GetBlockIdx();
            RestoreRadixFloatValues<T, int32_t>(topkOutValue, topkOutIndexValue,
                                                inputValueGm_[sourceRow * totalDataNum_], topkValueInput_, 1U,
                                                topkValueInput_, topkValueInput_, totalDataNum_, 1U);
        }

        // convert index from int32_t to int64_t if needed
        LocalTensor<T_INDEX_TO> tempConversionLocal = ConvertIndexType(topkOutIndexValue);
        topkOutValueQueue_.EnQue<T>(topkOutValue);
        tempIndexConversionQueue_.EnQue<T_INDEX_TO>(tempConversionLocal);

        // copy final result to gm
        uint64_t timesLoopOffset = sortLoopRound * unsortedDimParallel_ * topkValueInput_;
        uint64_t outValueOffset = GetBlockIdx() * topkValueInput_ + timesLoopOffset;
        CopyFinalResultToGm(topkValueGm_, outValueOffset, topkValueIndexGm_, outValueOffset);

        inQueueX_.FreeTensor(xLocal);
        inQueueIndexX_.FreeTensor(xIndexLocal);
        topkOutValueQueue_.FreeTensor(topkOutValue);
        topkOutIndexQueue_.FreeTensor(topkOutIndexValue);
    }
}

template <typename T, bool IS_LARGEST, bool IS_SORT, typename T_INDEX_TO, bool IS_BITONIC_SORT>
__aicore__ inline void
RadixSortTopKMultiCoreOptimization<T, IS_LARGEST, IS_SORT, T_INDEX_TO, IS_BITONIC_SORT>::ComputeFinalTopK(
    LocalTensor<T> topkOutValue, LocalTensor<int32_t> topkOutIndexValue, LocalTensor<T> xLocal,
    LocalTensor<int32_t> xIndexLocal, LocalTensor<uint8_t> shareTmpBuffer)
{
    uint32_t count = topkValueInput_ * (lastDimTileNum_ - 1) +
                     (tailDataSize_ != 0 && tailDataSize_ < topkValueInput_ ? tailDataSize_ : topkValueInput_);
    if (TryUbThresholdSelect<true>(xLocal, xIndexLocal, topkOutValue, topkOutIndexValue, shareTmpBuffer, count,
                                   topkValueInput_)) {
        return;
    }
    // Unified processing of temporary results
    static constexpr TopKConfig topkConfig{TopKAlgo::RADIX_SELECT, TopKOrder::UNSET, IS_SORT && !IS_BITONIC_SORT};
    TopKInfo topKInfo;
    topKInfo.outter = 1;
    topKInfo.inner = ROUND_UP_AGLIN(topkValueInput_ * lastDimTileNum_);
    topKInfo.n = count;
    LocalTensor<bool> emptyFinishLocal;
    TopkTiling emptyTopkTiling;
    AscendC::TopK<T, true, false, false, TopKMode::TOPK_NORMAL, topkConfig>(
        topkOutValue, topkOutIndexValue, xLocal, xIndexLocal, emptyFinishLocal, shareTmpBuffer,
        static_cast<int32_t>(this->topkValueInput_), emptyTopkTiling, topKInfo, IS_LARGEST);
    if constexpr (IS_BITONIC_SORT) {
        RunRadixCandidatesFullBitonic<T, int32_t, IS_LARGEST>(topkOutValue, topkOutIndexValue, topkValueInput_, 1U,
                                                              topkValueInput_, topkValueInput_);
    }
}

template <typename T, bool IS_LARGEST, bool IS_SORT, typename T_INDEX_TO, bool IS_BITONIC_SORT>
__aicore__ inline LocalTensor<T_INDEX_TO>
RadixSortTopKMultiCoreOptimization<T, IS_LARGEST, IS_SORT, T_INDEX_TO, IS_BITONIC_SORT>::ConvertIndexType(
    LocalTensor<int32_t> topkOutIndexValue)
{
    bool isLongIndex = IsSameType<T_INDEX_TO, int64_t>::value;
    if (isLongIndex) {
        // convert index from int32 to int64
        LocalTensor<T_INDEX_TO> tempConversionLocal = tempIndexConversionQueue_.AllocTensor<T_INDEX_TO>();
        AscendC::Cast(tempConversionLocal, topkOutIndexValue, RoundMode::CAST_NONE,
                      static_cast<int32_t>(topkValueInput_));
        return tempConversionLocal;
    }
    // reconvert index from int32 to int32
    return topkOutIndexValue.ReinterpretCast<T_INDEX_TO>();
}

template <typename T, bool IS_LARGEST, bool IS_SORT, typename T_INDEX_TO, bool IS_BITONIC_SORT>
__aicore__ inline void RadixSortTopKMultiCoreOptimization<T, IS_LARGEST, IS_SORT, T_INDEX_TO,
                                                          IS_BITONIC_SORT>::CopyDataIn(GlobalTensor<T> inputX,
                                                                                       uint64_t tileOffset,
                                                                                       uint32_t currTileSize)
{
    LocalTensor<T> xLocal = inQueueX_.AllocTensor<T>();
    uint32_t countAlign = ROUND_UP_AGLIN(currTileSize);
    T defaultValue = IS_LARGEST ? GetTypeMinValue<T>() : GetTypeMaxValue<T>();
    Duplicate(xLocal, defaultValue, countAlign);
    event_t copyDataEventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(copyDataEventId);
    WaitFlag<HardEvent::V_MTE2>(copyDataEventId);
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;
    DataCopyExtParams dataCopyParam;
    dataCopyParam.blockCount = 1;
    dataCopyParam.blockLen = currTileSize * sizeof(T);
    dataCopyParam.srcStride = 0;
    dataCopyParam.dstStride = 0;
    DataCopyPad(xLocal, inputX[tileOffset], dataCopyParam, padParams);
    inQueueX_.EnQue(xLocal);
}

template <typename T, bool IS_LARGEST, bool IS_SORT, typename T_INDEX_TO, bool IS_BITONIC_SORT>
__aicore__ inline void RadixSortTopKMultiCoreOptimization<T, IS_LARGEST, IS_SORT, T_INDEX_TO,
                                                          IS_BITONIC_SORT>::CopyIndexIn(GlobalTensor<int32_t> inputX,
                                                                                        uint64_t tileOffset,
                                                                                        uint32_t currTileSize)
{
    LocalTensor<int32_t> xLocal = inQueueIndexX_.AllocTensor<int32_t>();
    uint32_t countAlign = ROUND_UP_AGLIN(currTileSize);
    int32_t defaultValue = IS_LARGEST ? GetTypeMinValue<int32_t>() : GetTypeMaxValue<int32_t>();
    Duplicate(xLocal, defaultValue, countAlign);
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventId);
    WaitFlag<HardEvent::V_MTE2>(eventId);
    DataCopyPadExtParams<int32_t> padParams;
    padParams.isPad = false;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;
    DataCopyExtParams dataCopyParam;
    dataCopyParam.blockCount = 1;
    dataCopyParam.blockLen = currTileSize * sizeof(int32_t);
    dataCopyParam.srcStride = 0;
    dataCopyParam.dstStride = 0;
    DataCopyPad(xLocal, inputX[tileOffset], dataCopyParam, padParams);
    inQueueIndexX_.EnQue(xLocal);
}

template <typename T, bool IS_LARGEST, bool IS_SORT, typename T_INDEX_TO, bool IS_BITONIC_SORT>
__aicore__ inline void
RadixSortTopKMultiCoreOptimization<T, IS_LARGEST, IS_SORT, T_INDEX_TO, IS_BITONIC_SORT>::CopyFinalResultToGm(
    GlobalTensor<T> valueGm, uint64_t valueOffset, GlobalTensor<T_INDEX_TO> indexGm, uint64_t indexOffset)
{
    // copy sorted value
    AscendC::LocalTensor<T> valuesLocal = topkOutValueQueue_.DeQue<T>();
    DataCopyExtParams dataCopyParamValue;
    dataCopyParamValue.blockCount = 1;
    dataCopyParamValue.blockLen = topkValueInput_ * sizeof(T);
    dataCopyParamValue.srcStride = 0;
    dataCopyParamValue.dstStride = 0;
    AscendC::DataCopyPad(valueGm[valueOffset], valuesLocal, dataCopyParamValue);
    topkOutValueQueue_.FreeTensor(valuesLocal);

    // copy index to gm
    AscendC::LocalTensor<T_INDEX_TO> indexLocal = tempIndexConversionQueue_.DeQue<T_INDEX_TO>();
    DataCopyExtParams dataCopyParamIndex;
    dataCopyParamIndex.blockCount = 1;
    dataCopyParamIndex.blockLen = topkValueInput_ * sizeof(T_INDEX_TO);
    dataCopyParamIndex.srcStride = 0;
    dataCopyParamIndex.dstStride = 0;
    AscendC::DataCopyPad(indexGm[indexOffset], indexLocal, dataCopyParamIndex);
    tempIndexConversionQueue_.FreeTensor(indexLocal);
}

/**
 * @brief
 * 将多核topk的中间输出值和索引拷贝到GM中存放，由于是多核topk，第k个核输出的索引值需要加上第k个核的所有元素在整个尾轴的偏移量
 * @param [in] valueOffset：排序出的k个数应该在GM的存放位置
 * @param [in] indexOffset：排序出的k个数的索引应该在GM的存放位置
 * @param [in] indexValueOffset：排序出的k个数的索引应该加上的偏移量
 */
template <typename T, bool IS_LARGEST, bool IS_SORT, typename T_INDEX_TO, bool IS_BITONIC_SORT>
__aicore__ inline void
RadixSortTopKMultiCoreOptimization<T, IS_LARGEST, IS_SORT, T_INDEX_TO, IS_BITONIC_SORT>::CopyIndexOutWithOffset(
    GlobalTensor<T> valueGm, uint64_t valueOffset, GlobalTensor<int32_t> indexGm, uint64_t indexOffset)
{
    // copy sorted value
    AscendC::LocalTensor<T> valuesLocal = topkOutValueQueue_.DeQue<T>();
    DataCopyExtParams dataCopyParamValue;
    dataCopyParamValue.blockCount = 1;
    dataCopyParamValue.blockLen = topkValueInput_ * sizeof(T);
    dataCopyParamValue.srcStride = 0;
    dataCopyParamValue.dstStride = 0;
    AscendC::DataCopyPad(valueGm[valueOffset], valuesLocal, dataCopyParamValue);
    topkOutValueQueue_.FreeTensor(valuesLocal);

    // copy sorted value index
    AscendC::LocalTensor<int32_t> indexLocal = topkOutIndexQueue_.DeQue<int32_t>();
    DataCopyExtParams dataCopyParamIndex;
    dataCopyParamIndex.blockCount = 1;
    dataCopyParamIndex.blockLen = topkValueInput_ * sizeof(int32_t);
    dataCopyParamIndex.srcStride = 0;
    dataCopyParamIndex.dstStride = 0;
    AscendC::DataCopyPad(indexGm[indexOffset], indexLocal, dataCopyParamIndex);
    topkOutIndexQueue_.FreeTensor(indexLocal);
}
#endif // RADIX_SORT_TOP_K_INTER_CORE_TEMPLATE_OPTIMIZATION_H
