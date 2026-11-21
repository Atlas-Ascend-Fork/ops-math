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
 * \file one_axis_concat_no_align_same_shape_gather.h
 * \brief one axis concat no align same shape gather
 */

#ifndef ONE_AXIS_CONCAT_NO_ALIGN_SAME_SHAPE_GATHER
#define ONE_AXIS_CONCAT_NO_ALIGN_SAME_SHAPE_GATHER

#include "concat_base.h"
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

namespace Concat {
using namespace AscendC;

template <typename T, typename U, typename TILINGDATA = ConcatTilingData>
class OneAxisConcatNoAlignGather {
public:
    __aicore__ inline OneAxisConcatNoAlignGather(__tiling_data_ptr__ TILINGDATA* tilingData)
        : tilingData_(tilingData){};
    static __aicore__ inline void PreloadTiling(__tiling_data_ptr__ TILINGDATA* tilingData)
    {
        PreloadMergedInfo(tilingData);
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dst);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInNoSplitDim1(int64_t srcRowsOffset, int64_t rows, int64_t startIdx, int64_t endIdx,
                                             LocalTensor<T>& srcLocal);
    __aicore__ inline __gm__ T* GetTensorAddr(uint32_t index, int64_t offset);
    __aicore__ inline void CopyOut(int64_t dstOffset, int64_t rows, int64_t cols, LocalTensor<T>& dstLocal);
    __aicore__ inline void CopyOut(int64_t dstOffset, int64_t dataLen, LocalTensor<T>& dstLocal);
    __aicore__ inline void SetCopyInparam(int64_t rows, int64_t cols, int64_t srcStride, int64_t dstStride);
    __aicore__ inline void ProcessBlockSplitDim0NoSplitDim1();
    __aicore__ inline void ProcessBlockSplitDim1();
    __aicore__ inline void ProcessBlockSplitDim0SplitDim1();
    __aicore__ inline void GenGatherIndex(uint32_t cols, uint32_t tensorStride, uint32_t inputCols,
                                          LocalTensor<U>& indexLocal);
    __aicore__ inline void ComputeNoSplitDim1(uint32_t rows, uint32_t cols, uint32_t tensorStride, uint32_t inputCols,
                                              LocalTensor<T>& srcLocal, LocalTensor<T>& dstLocal);
    __aicore__ inline void ComputeNoSplitDim1B8(uint32_t rows, uint32_t cols, uint32_t tensorStride, uint32_t inputCols,
                                                LocalTensor<T>& srcLocal, LocalTensor<T>& dstLocal);
    __aicore__ inline void ComputeNoSplitDim1Norm(uint32_t rows, uint32_t cols, uint32_t tensorStride,
                                                  uint32_t inputCols, LocalTensor<T>& srcLocal,
                                                  LocalTensor<T>& dstLocal);
    template <const bool needColAlign = true>
    __aicore__ inline void ComputeSplitDim1(uint32_t rows, uint32_t cols, uint32_t tensorStride, uint32_t inputCols,
                                            LocalTensor<T>& srcLocal, LocalTensor<T>& dstLocal);
    template <const bool needColAlign = true>
    __aicore__ inline void ComputeSplitDim1B8(uint32_t rows, uint32_t cols, uint32_t tensorStride, uint32_t inputCols,
                                              LocalTensor<T>& srcLocal, LocalTensor<T>& dstLocal);
    template <const bool needColAlign = true>
    __aicore__ inline void ComputeSplitDim1Norm(uint32_t rows, uint32_t cols, uint32_t tensorStride, uint32_t inputCols,
                                                LocalTensor<T>& srcLocal, LocalTensor<T>& dstLocal);
    __aicore__ inline void ProcessPerLoop(int64_t srcRowsOffset, int64_t rows, const SplitLoopParam& params,
                                          uint16_t& bufIdX, uint16_t& bufIdY);

private:
    constexpr static event_t EVT_ID0 = static_cast<event_t>(0);
    constexpr static event_t EVT_ID1 = static_cast<event_t>(1);

    GlobalTensor<T> dstGlobal_;
    GlobalTensor<T> srcGlobal_;
    __tiling_data_ptr__ TILINGDATA* tilingData_;
    int64_t blockOffset_ = 0;
    int64_t blockIdx_ = 0;

    int64_t numPerBlock_ = BYTES_PER_BLOCK / sizeof(T);
    int64_t colsUsedCoreNum_ = 1;
    int64_t rowsUsedCoreNum_ = 1;

    DataCopyExtParams copyInParam_ = {0, 0, 0, 0, 0};

    int64_t startTensorIdx_ = 0;
    int64_t startTensorOffset_ = 0;
    int64_t endTensorIdx_ = 0;
    ListTensorDesc inputList_;

    uint32_t vfLen_ = GetVRegSize() / sizeof(U);

    uint32_t bufSize_{0};
    uint32_t bufElems_{0};
    uint32_t inOffset0_{0};
    uint32_t inOffset1_{0};
    uint32_t outOffset0_{0};
    uint32_t outOffset1_{0};
    uint32_t idxOffset_{0};
    uint32_t idxElems_{0};

    int16_t ubSplitDim1_{0};
    int16_t tensorNum_{0};
    int16_t isNonContiguous_{0};
    int32_t ubFactorDim0_{0};
    int32_t ubFactorDim1_{0};
    int32_t tailUbFactorDim0_{0};
    int32_t tailUbFactorDim1_{0};
    int32_t bufferSize_{0};
    int64_t blockFactor_{0};
    int64_t tailBlockFactor_{0};
    int64_t uoDim0_{0};
    int64_t catDim1_{0};
    int64_t sameShapeTensorDim1_{0};
};

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::Init(GM_ADDR x, GM_ADDR dst)
{
    blockIdx_ = GetBlockIdx();
    if constexpr (IsSame<TILINGDATA, ConcatTilingData>::value || IsSame<TILINGDATA, ConcatTilingDataCompact>::value) {
        int64_t colsUsedCoreNum = GetBlockNum() / tilingData_->uoDim0;
        if (blockIdx_ % colsUsedCoreNum != 0) {
            startTensorIdx_ = tilingData_->arrays.mergedInfo[CORE_INFO_BASE + 2 * (blockIdx_ - 1)];
            startTensorOffset_ = tilingData_->arrays.mergedInfo[CORE_INFO_BASE + 2 * (blockIdx_ - 1) + 1];
        }
        endTensorIdx_ = tilingData_->arrays.mergedInfo[CORE_INFO_BASE + 2 * blockIdx_];
    }
    if (startTensorOffset_ == tilingData_->sameShapeTensorDim1) {
        startTensorIdx_ += 1;
        startTensorOffset_ = 0;
    }
    if constexpr (IsSame<TILINGDATA, ConcatTilingDataNoArray>::value ||
                  IsSame<TILINGDATA, ConcatTilingDataNoArrayCompact>::value) {
        blockOffset_ = blockIdx_ * tilingData_->ubFactorDim0 * tilingData_->blockFactor;
        dstGlobal_.SetGlobalBuffer((__gm__ T*)dst + blockOffset_ * tilingData_->catDim1);
    } else {
        rowsUsedCoreNum_ = tilingData_->uoDim0;
        colsUsedCoreNum_ = GetBlockNum() / rowsUsedCoreNum_;
        blockOffset_ = tilingData_->ubFactorDim0 * (blockIdx_ / colsUsedCoreNum_);
        int64_t colOffset = (blockIdx_ % colsUsedCoreNum_) * tilingData_->blockFactor * tilingData_->ubFactorDim1;
        dstGlobal_.SetGlobalBuffer((__gm__ T*)dst + blockOffset_ * tilingData_->catDim1 + colOffset);
    }

    bufSize_ = static_cast<uint32_t>(tilingData_->bufferSize * sizeof(T));
    bufElems_ = static_cast<uint32_t>(tilingData_->bufferSize);
    idxOffset_ = 0;
    idxElems_ = INDEX_SIZE / sizeof(U);
    inOffset0_ = static_cast<uint32_t>((INDEX_SIZE + 31) / 32 * 32);
    inOffset1_ = inOffset0_ + bufSize_;
    outOffset0_ = inOffset1_ + bufSize_;
    outOffset1_ = outOffset0_ + bufSize_;

    ubSplitDim1_ = tilingData_->ubSplitDim1;
    tensorNum_ = tilingData_->tensorNum;
    isNonContiguous_ = tilingData_->isNonContiguous;
    ubFactorDim0_ = tilingData_->ubFactorDim0;
    ubFactorDim1_ = tilingData_->ubFactorDim1;
    tailUbFactorDim0_ = tilingData_->tailUbFactorDim0;
    tailUbFactorDim1_ = tilingData_->tailUbFactorDim1;
    bufferSize_ = tilingData_->bufferSize;
    blockFactor_ = tilingData_->blockFactor;
    tailBlockFactor_ = tilingData_->tailBlockFactor;
    uoDim0_ = tilingData_->uoDim0;
    catDim1_ = tilingData_->catDim1;
    sameShapeTensorDim1_ = tilingData_->sameShapeTensorDim1;

    inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline __gm__ T* OneAxisConcatNoAlignGather<T, U, TILINGDATA>::GetTensorAddr(uint32_t index, int64_t offset)
{
    return inputList_.GetDataPtr<T>(index) + offset;
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::Process()
{
    if (blockIdx_ >= GetBlockNum()) {
        return;
    }
    if constexpr (IsSame<TILINGDATA, ConcatTilingData>::value || IsSame<TILINGDATA, ConcatTilingDataCompact>::value) {
        ProcessBlockSplitDim1();
    } else {
        if (ubSplitDim1_ == 0) {
            ProcessBlockSplitDim0NoSplitDim1();
        } else {
            ProcessBlockSplitDim0SplitDim1();
        }
    }
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::ProcessBlockSplitDim1()
{
    int64_t blockIdxInCol = blockIdx_ % colsUsedCoreNum_;
    int64_t blockIdxInRow = blockIdx_ / colsUsedCoreNum_;
    int64_t copyRowsNum = ubFactorDim0_;

    if (blockIdxInRow == rowsUsedCoreNum_ - 1) {
        copyRowsNum = tailUbFactorDim0_;
    }
    int64_t handleTensorNum = (endTensorIdx_ - startTensorIdx_) + 1;
    int64_t perLoopDealTensorNum = (bufferSize_ - copyRowsNum * numPerBlock_) /
                                   (copyRowsNum * sameShapeTensorDim1_ + numPerBlock_);
    int64_t tensorStride = (copyRowsNum * sameShapeTensorDim1_ + numPerBlock_ - 1) / numPerBlock_ * numPerBlock_;
    int64_t dim1Loop = handleTensorNum / perLoopDealTensorNum;
    int64_t tailLoopDealTensorNum = handleTensorNum - dim1Loop * perLoopDealTensorNum;
    if (tailLoopDealTensorNum == 0) {
        dim1Loop -= 1;
        tailLoopDealTensorNum = perLoopDealTensorNum;
    }
    LocalTensor<U> indexLocal(TPosition::VECCALC, idxOffset_, idxElems_);
    GenGatherIndex(perLoopDealTensorNum * sameShapeTensorDim1_, tensorStride, sameShapeTensorDim1_, indexLocal);
    SplitLoopParam param = {startTensorIdx_,       handleTensorNum, perLoopDealTensorNum,
                            tailLoopDealTensorNum, dim1Loop,        tensorStride};
    SetFlag<HardEvent::V_MTE2>(EVT_ID0);
    SetFlag<HardEvent::V_MTE2>(EVT_ID1);
    SetFlag<HardEvent::MTE3_V>(EVT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVT_ID1);
    uint16_t bufIdX = 0;
    uint16_t bufIdY = 0;
    ProcessPerLoop(0, copyRowsNum, param, bufIdX, bufIdY);
    WaitFlag<HardEvent::V_MTE2>(EVT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVT_ID1);
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::ProcessPerLoop(int64_t srcRowsOffset, int64_t rows,
                                                                                    const SplitLoopParam& params,
                                                                                    uint16_t& bufIdX, uint16_t& bufIdY)
{
    for (int64_t i = 0; i < params.loopSize; i++) {
        event_t evtX = (bufIdX == 0) ? EVT_ID0 : EVT_ID1;
        WaitFlag<HardEvent::V_MTE2>(evtX);
        LocalTensor<T> srcLocal(TPosition::VECIN, (bufIdX == 0) ? inOffset0_ : inOffset1_, bufElems_);
        CopyInNoSplitDim1(srcRowsOffset, rows, params.startTensorIdx + i * params.perLoopDealTensorNum,
                          params.startTensorIdx + (i + 1) * params.perLoopDealTensorNum, srcLocal);
        SetFlag<HardEvent::MTE2_V>(evtX);
        WaitFlag<HardEvent::MTE2_V>(evtX);

        event_t evtY = (bufIdY == 0) ? EVT_ID0 : EVT_ID1;
        WaitFlag<HardEvent::MTE3_V>(evtY);
        LocalTensor<T> dstLocal(TPosition::VECOUT, (bufIdY == 0) ? outOffset0_ : outOffset1_, bufElems_);
        ComputeSplitDim1(rows, params.perLoopDealTensorNum * sameShapeTensorDim1_, params.tensorStride,
                         sameShapeTensorDim1_, srcLocal, dstLocal);
        SetFlag<HardEvent::V_MTE2>(evtX);
        SetFlag<HardEvent::V_MTE3>(evtY);

        WaitFlag<HardEvent::V_MTE3>(evtY);
        CopyOut(srcRowsOffset * catDim1_ + i * params.perLoopDealTensorNum * sameShapeTensorDim1_, rows,
                params.perLoopDealTensorNum * sameShapeTensorDim1_, dstLocal);
        SetFlag<HardEvent::MTE3_V>(evtY);
        bufIdX ^= 1;
        bufIdY ^= 1;
    }
    event_t evtX = (bufIdX == 0) ? EVT_ID0 : EVT_ID1;
    WaitFlag<HardEvent::V_MTE2>(evtX);
    LocalTensor<T> srcLocal(TPosition::VECIN, (bufIdX == 0) ? inOffset0_ : inOffset1_, bufElems_);
    CopyInNoSplitDim1(srcRowsOffset, rows, params.startTensorIdx + params.loopSize * params.perLoopDealTensorNum,
                      params.startTensorIdx + params.handleTensorNum, srcLocal);
    SetFlag<HardEvent::MTE2_V>(evtX);
    WaitFlag<HardEvent::MTE2_V>(evtX);

    event_t evtY = (bufIdY == 0) ? EVT_ID0 : EVT_ID1;
    WaitFlag<HardEvent::MTE3_V>(evtY);
    LocalTensor<T> dstLocal(TPosition::VECOUT, (bufIdY == 0) ? outOffset0_ : outOffset1_, bufElems_);
    ComputeSplitDim1(rows, params.tailLoopDealTensorNum * sameShapeTensorDim1_, params.tensorStride,
                     sameShapeTensorDim1_, srcLocal, dstLocal);
    SetFlag<HardEvent::V_MTE2>(evtX);
    SetFlag<HardEvent::V_MTE3>(evtY);

    WaitFlag<HardEvent::V_MTE3>(evtY);
    CopyOut(srcRowsOffset * catDim1_ + params.loopSize * params.perLoopDealTensorNum * sameShapeTensorDim1_, rows,
            params.tailLoopDealTensorNum * sameShapeTensorDim1_, dstLocal);
    SetFlag<HardEvent::MTE3_V>(evtY);
    bufIdX ^= 1;
    bufIdY ^= 1;
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::ProcessBlockSplitDim0SplitDim1()
{
    int64_t loopSizeDim0 = blockFactor_;
    if (blockIdx_ == GetBlockNum() - 1) {
        loopSizeDim0 = tailBlockFactor_ - 1;
    }

    int64_t perLoopDealTensorNum = (bufferSize_ - ubFactorDim0_ * numPerBlock_) /
                                   (ubFactorDim0_ * sameShapeTensorDim1_ + numPerBlock_);
    int64_t tensorStride = (ubFactorDim0_ * sameShapeTensorDim1_ + numPerBlock_ - 1) / numPerBlock_ * numPerBlock_;
    int64_t dim1Loop = tensorNum_ / perLoopDealTensorNum;
    int64_t tailLoopDealTensorNum = tensorNum_ - dim1Loop * perLoopDealTensorNum;
    if (tailLoopDealTensorNum == 0) {
        dim1Loop -= 1;
        tailLoopDealTensorNum = perLoopDealTensorNum;
    }

    LocalTensor<U> indexLocal(TPosition::VECCALC, idxOffset_, idxElems_);
    if (loopSizeDim0 > 0) {
        GenGatherIndex(perLoopDealTensorNum * sameShapeTensorDim1_, tensorStride, sameShapeTensorDim1_, indexLocal);
    }
    SplitLoopParam param = {0, tensorNum_, perLoopDealTensorNum, tailLoopDealTensorNum, dim1Loop, tensorStride};
    SetFlag<HardEvent::V_MTE2>(EVT_ID0);
    SetFlag<HardEvent::V_MTE2>(EVT_ID1);
    SetFlag<HardEvent::MTE3_V>(EVT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVT_ID1);
    uint16_t bufIdX = 0;
    uint16_t bufIdY = 0;
    for (int64_t i = 0; i < loopSizeDim0; i++) {
        ProcessPerLoop(i * ubFactorDim0_, ubFactorDim0_, param, bufIdX, bufIdY);
    }
    if (blockIdx_ == GetBlockNum() - 1) {
        tensorStride = (tailUbFactorDim0_ * sameShapeTensorDim1_ + numPerBlock_ - 1) / numPerBlock_ * numPerBlock_;
        GenGatherIndex(perLoopDealTensorNum * sameShapeTensorDim1_, tensorStride, sameShapeTensorDim1_, indexLocal);
        param.tensorStride = tensorStride;
        ProcessPerLoop(loopSizeDim0 * ubFactorDim0_, tailUbFactorDim0_, param, bufIdX, bufIdY);
    }
    WaitFlag<HardEvent::V_MTE2>(EVT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVT_ID1);
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::ProcessBlockSplitDim0NoSplitDim1()
{
    int64_t loopSize = (blockIdx_ == GetBlockNum() - 1) ? tailBlockFactor_ - 1 : blockFactor_;
    int64_t tensorStride = (ubFactorDim0_ * sameShapeTensorDim1_ + numPerBlock_ - 1) / numPerBlock_ * numPerBlock_;
    LocalTensor<U> indexLocal(TPosition::VECCALC, idxOffset_, idxElems_);
    if (loopSize > 0) {
        GenGatherIndex(catDim1_, tensorStride, sameShapeTensorDim1_, indexLocal);
    }
    SetFlag<HardEvent::V_MTE2>(EVT_ID0);
    SetFlag<HardEvent::V_MTE2>(EVT_ID1);
    SetFlag<HardEvent::MTE3_V>(EVT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVT_ID1);
    uint16_t bufIdX = 0;
    uint16_t bufIdY = 0;
    if (catDim1_ * sizeof(U) <= GetVRegSize()) {
        for (int64_t i = 0; i < loopSize; i++) {
            event_t evtX = (bufIdX == 0) ? EVT_ID0 : EVT_ID1;
            WaitFlag<HardEvent::V_MTE2>(evtX);
            LocalTensor<T> srcLocal(TPosition::VECIN, (bufIdX == 0) ? inOffset0_ : inOffset1_, bufElems_);
            CopyInNoSplitDim1(i * ubFactorDim0_, ubFactorDim0_, 0, tensorNum_, srcLocal);
            SetFlag<HardEvent::MTE2_V>(evtX);
            WaitFlag<HardEvent::MTE2_V>(evtX);

            event_t evtY = (bufIdY == 0) ? EVT_ID0 : EVT_ID1;
            WaitFlag<HardEvent::MTE3_V>(evtY);
            LocalTensor<T> dstLocal(TPosition::VECOUT, (bufIdY == 0) ? outOffset0_ : outOffset1_, bufElems_);
            ComputeNoSplitDim1(ubFactorDim0_, catDim1_, tensorStride, sameShapeTensorDim1_, srcLocal, dstLocal);
            SetFlag<HardEvent::V_MTE2>(evtX);
            SetFlag<HardEvent::V_MTE3>(evtY);

            WaitFlag<HardEvent::V_MTE3>(evtY);
            CopyOut(i * ubFactorDim0_ * catDim1_, ubFactorDim0_ * catDim1_, dstLocal);
            SetFlag<HardEvent::MTE3_V>(evtY);
            bufIdX ^= 1;
            bufIdY ^= 1;
        }
        tensorStride = (tailUbFactorDim0_ * sameShapeTensorDim1_ + numPerBlock_ - 1) / numPerBlock_ * numPerBlock_;
        if (blockIdx_ == GetBlockNum() - 1) {
            GenGatherIndex(catDim1_, tensorStride, sameShapeTensorDim1_, indexLocal);
            event_t evtX = (bufIdX == 0) ? EVT_ID0 : EVT_ID1;
            WaitFlag<HardEvent::V_MTE2>(evtX);
            LocalTensor<T> srcLocal(TPosition::VECIN, (bufIdX == 0) ? inOffset0_ : inOffset1_, bufElems_);
            CopyInNoSplitDim1(loopSize * ubFactorDim0_, tailUbFactorDim0_, 0, tensorNum_, srcLocal);
            SetFlag<HardEvent::MTE2_V>(evtX);
            WaitFlag<HardEvent::MTE2_V>(evtX);

            event_t evtY = (bufIdY == 0) ? EVT_ID0 : EVT_ID1;
            WaitFlag<HardEvent::MTE3_V>(evtY);
            LocalTensor<T> dstLocal(TPosition::VECOUT, (bufIdY == 0) ? outOffset0_ : outOffset1_, bufElems_);
            ComputeNoSplitDim1(tailUbFactorDim0_, catDim1_, tensorStride, sameShapeTensorDim1_, srcLocal, dstLocal);
            SetFlag<HardEvent::V_MTE2>(evtX);
            SetFlag<HardEvent::V_MTE3>(evtY);

            WaitFlag<HardEvent::V_MTE3>(evtY);
            CopyOut(loopSize * ubFactorDim0_ * catDim1_, tailUbFactorDim0_ * catDim1_, dstLocal);
            SetFlag<HardEvent::MTE3_V>(evtY);
            bufIdX ^= 1;
            bufIdY ^= 1;
        }
    } else {
        for (int64_t i = 0; i < loopSize; i++) {
            event_t evtX = (bufIdX == 0) ? EVT_ID0 : EVT_ID1;
            WaitFlag<HardEvent::V_MTE2>(evtX);
            LocalTensor<T> srcLocal(TPosition::VECIN, (bufIdX == 0) ? inOffset0_ : inOffset1_, bufElems_);
            CopyInNoSplitDim1(i * ubFactorDim0_, ubFactorDim0_, 0, tensorNum_, srcLocal);
            SetFlag<HardEvent::MTE2_V>(evtX);
            WaitFlag<HardEvent::MTE2_V>(evtX);

            event_t evtY = (bufIdY == 0) ? EVT_ID0 : EVT_ID1;
            WaitFlag<HardEvent::MTE3_V>(evtY);
            LocalTensor<T> dstLocal(TPosition::VECOUT, (bufIdY == 0) ? outOffset0_ : outOffset1_, bufElems_);
            ComputeSplitDim1<false>(ubFactorDim0_, catDim1_, tensorStride, sameShapeTensorDim1_, srcLocal, dstLocal);
            SetFlag<HardEvent::V_MTE2>(evtX);
            SetFlag<HardEvent::V_MTE3>(evtY);

            WaitFlag<HardEvent::V_MTE3>(evtY);
            CopyOut(i * ubFactorDim0_ * catDim1_, ubFactorDim0_ * catDim1_, dstLocal);
            SetFlag<HardEvent::MTE3_V>(evtY);
            bufIdX ^= 1;
            bufIdY ^= 1;
        }
        tensorStride = (sameShapeTensorDim1_ * tailUbFactorDim0_ + numPerBlock_ - 1) / numPerBlock_ * numPerBlock_;
        if (blockIdx_ == GetBlockNum() - 1) {
            GenGatherIndex(catDim1_, tensorStride, sameShapeTensorDim1_, indexLocal);
            event_t evtX = (bufIdX == 0) ? EVT_ID0 : EVT_ID1;
            WaitFlag<HardEvent::V_MTE2>(evtX);
            LocalTensor<T> srcLocal(TPosition::VECIN, (bufIdX == 0) ? inOffset0_ : inOffset1_, bufElems_);
            CopyInNoSplitDim1(ubFactorDim0_ * loopSize, tailUbFactorDim0_, 0, tensorNum_, srcLocal);
            SetFlag<HardEvent::MTE2_V>(evtX);
            WaitFlag<HardEvent::MTE2_V>(evtX);

            event_t evtY = (bufIdY == 0) ? EVT_ID0 : EVT_ID1;
            WaitFlag<HardEvent::MTE3_V>(evtY);
            LocalTensor<T> dstLocal(TPosition::VECOUT, (bufIdY == 0) ? outOffset0_ : outOffset1_, bufElems_);
            ComputeSplitDim1<false>(tailUbFactorDim0_, catDim1_, tensorStride, sameShapeTensorDim1_, srcLocal,
                                    dstLocal);
            SetFlag<HardEvent::V_MTE2>(evtX);
            SetFlag<HardEvent::V_MTE3>(evtY);

            WaitFlag<HardEvent::V_MTE3>(evtY);
            CopyOut(loopSize * ubFactorDim0_ * catDim1_, tailUbFactorDim0_ * catDim1_, dstLocal);
            SetFlag<HardEvent::MTE3_V>(evtY);
            bufIdX ^= 1;
            bufIdY ^= 1;
        }
    }
    WaitFlag<HardEvent::V_MTE2>(EVT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVT_ID1);
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::GenGatherIndex(uint32_t cols,
                                                                                    uint32_t tensorStride,
                                                                                    uint32_t inputCols,
                                                                                    LocalTensor<U>& indexLocal)
{
    auto dstAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        using regType = typename VciTypeGet<U>::T;
        AscendC::Reg::RegTensor<regType> tmp;
        AscendC::Reg::RegTensor<U> v0;
        AscendC::Reg::RegTensor<U> v1;
        AscendC::Reg::RegTensor<U> v2;
        AscendC::Reg::RegTensor<U> v3;

        AscendC::Reg::RegTensor<U> vd0;
        AscendC::Reg::RegTensor<U> vd1;
        AscendC::Reg::RegTensor<U> vd2;
        AscendC::Reg::RegTensor<U> vd3;
        AscendC::Reg::RegTensor<U> vd4;
        AscendC::Reg::RegTensor<U> vd5;
        AscendC::Reg::RegTensor<U> vd6;
        AscendC::Reg::RegTensor<U> vd7;
        AscendC::Reg::RegTensor<U> vd8;
        AscendC::Reg::RegTensor<U> vd9;
        AscendC::Reg::RegTensor<U> vd10;
        uint32_t num = vfLen_;
        AscendC::Reg::MaskReg p0 = AscendC::Reg::UpdateMask<U>(num);
        AscendC::Reg::Arange(tmp, 0);
        if constexpr (sizeof(U) == sizeof(uint32_t) || sizeof(U) == sizeof(uint16_t)) {
            v0 = (AscendC::Reg::RegTensor<U>&)tmp;
        }
        AscendC::Reg::Duplicate(v1, (U)inputCols, p0);
        AscendC::Reg::Duplicate(v2, (U)tensorStride, p0);
        AscendC::Reg::Duplicate(v3, (U)cols, p0);

        AscendC::Reg::Div(vd2, v0, v3, p0);
        AscendC::Reg::Mul(vd6, vd2, v3, p0);
        AscendC::Reg::Sub(vd7, v0, vd6, p0);

        AscendC::Reg::Div(vd0, vd7, v1, p0);
        AscendC::Reg::Mul(vd1, vd0, v2, p0);

        AscendC::Reg::Div(vd3, vd7, v1, p0);
        AscendC::Reg::Mul(vd4, vd3, v1, p0);
        AscendC::Reg::Sub(vd5, vd7, vd4, p0);

        AscendC::Reg::Mul(vd8, vd2, v1, p0);

        AscendC::Reg::Add(vd9, vd1, vd5, p0);
        AscendC::Reg::Add(vd10, vd8, vd9, p0);
        AscendC::Reg::StoreAlign(dstAddr, vd10, p0);
    }
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::ComputeNoSplitDim1(uint32_t rows, uint32_t cols,
                                                                                        uint32_t tensorStride,
                                                                                        uint32_t inputCols,
                                                                                        LocalTensor<T>& srcLocal,
                                                                                        LocalTensor<T>& dstLocal)
{
    if constexpr (sizeof(T) == 1) {
        ComputeNoSplitDim1B8(rows, cols, tensorStride, inputCols, srcLocal, dstLocal);
    } else {
        ComputeNoSplitDim1Norm(rows, cols, tensorStride, inputCols, srcLocal, dstLocal);
    }
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::ComputeNoSplitDim1Norm(
    uint32_t rows, uint32_t cols, uint32_t tensorStride, uint32_t inputCols, LocalTensor<T>& srcLocal,
    LocalTensor<T>& dstLocal)
{
    uint16_t regFactorDim0 = vfLen_ / cols;
    uint16_t size0 = rows / regFactorDim0;
    uint16_t tailRegFactorDim0 = rows - size0 * regFactorDim0;
    uint32_t stride = inputCols * regFactorDim0;
    auto dstAddr = (__ubuf__ T*)dstLocal.GetPhyAddr();
    auto srcAddr = (__ubuf__ T*)srcLocal.GetPhyAddr();
    LocalTensor<U> indexLocal(TPosition::VECCALC, idxOffset_, idxElems_);
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<U> vd0;
        AscendC::Reg::RegTensor<U> vd1;
        AscendC::Reg::RegTensor<T> vd2;
        AscendC::Reg::UnalignRegForStore u0;

        uint32_t num = static_cast<uint32_t>(regFactorDim0 * cols);
        uint32_t tailNum = static_cast<uint32_t>(tailRegFactorDim0 * cols);
        uint32_t main = num;
        uint32_t tail = tailNum;
        AscendC::Reg::MaskReg p0 = AscendC::Reg::UpdateMask<U>(num);
        AscendC::Reg::MaskReg p1 = AscendC::Reg::UpdateMask<U>(tailNum);

        AscendC::Reg::LoadAlign(vd0, indexAddr);
        for (uint16_t i = 0; i < size0; i++) {
            AscendC::Reg::Adds(vd1, vd0, (U)(i * stride), p0);
            AscendC::Reg::Gather(vd2, srcAddr, vd1, p0);
            AscendC::Reg::StoreUnAlign(dstAddr, vd2, u0, main);
            AscendC::Reg::StoreUnAlignPost(dstAddr, u0, 0);
        }
        AscendC::Reg::Adds(vd1, vd0, (U)(size0 * stride), p1);
        AscendC::Reg::Gather(vd2, srcAddr, vd1, p1);
        AscendC::Reg::StoreUnAlign(dstAddr, vd2, u0, tail);
        AscendC::Reg::StoreUnAlignPost(dstAddr, u0, 0);
    }
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::ComputeNoSplitDim1B8(uint32_t rows, uint32_t cols,
                                                                                          uint32_t tensorStride,
                                                                                          uint32_t inputCols,
                                                                                          LocalTensor<T>& srcLocal,
                                                                                          LocalTensor<T>& dstLocal)
{
    uint16_t regFactorDim0 = vfLen_ / cols;
    uint16_t size0 = rows / regFactorDim0;
    uint16_t tailRegFactorDim0 = rows - size0 * regFactorDim0;
    uint32_t stride = inputCols * regFactorDim0;
    auto dstAddr = reinterpret_cast<__ubuf__ uint8_t*>(dstLocal.GetPhyAddr());
    auto srcAddr = reinterpret_cast<__ubuf__ uint8_t*>(srcLocal.GetPhyAddr());
    LocalTensor<U> indexLocal(TPosition::VECCALC, idxOffset_, idxElems_);
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<uint16_t> vd0;
        AscendC::Reg::RegTensor<uint16_t> vd1;
        AscendC::Reg::RegTensor<uint16_t> vd2;
        AscendC::Reg::RegTensor<uint8_t> vd3;
        AscendC::Reg::RegTensor<uint16_t> vd4;
        AscendC::Reg::RegTensor<uint8_t> vd5;

        AscendC::Reg::UnalignRegForStore u0;

        uint32_t num = static_cast<uint32_t>(regFactorDim0 * cols);
        uint32_t tailNum = static_cast<uint32_t>(tailRegFactorDim0 * cols);
        uint32_t main = num;
        uint32_t tail = tailNum;
        AscendC::Reg::MaskReg p0 = AscendC::Reg::UpdateMask<U>(num);
        AscendC::Reg::MaskReg p1 = AscendC::Reg::UpdateMask<U>(tailNum);

        AscendC::Reg::LoadAlign(vd0, indexAddr);
        for (uint16_t i = 0; i < size0; i++) {
            AscendC::Reg::Adds(vd1, vd0, (U)(i * stride), p0);
            AscendC::Reg::Gather(vd2, srcAddr, vd1, p0);
            AscendC::Reg::Pack(vd3, vd2);
            AscendC::Reg::StoreUnAlign(dstAddr, vd3, u0, main);
            AscendC::Reg::StoreUnAlignPost(dstAddr, u0, 0);
        }
        AscendC::Reg::Adds(vd1, vd0, (U)(size0 * stride), p1);
        AscendC::Reg::Gather(vd4, srcAddr, vd1, p1);
        AscendC::Reg::Pack(vd5, vd4);
        AscendC::Reg::StoreUnAlign(dstAddr, vd5, u0, tail);
        AscendC::Reg::StoreUnAlignPost(dstAddr, u0, 0);
    }
}

template <typename T, typename U, typename TILINGDATA>
template <const bool needColAlign>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::ComputeSplitDim1(uint32_t rows, uint32_t cols,
                                                                                      uint32_t tensorStride,
                                                                                      uint32_t inputCols,
                                                                                      LocalTensor<T>& srcLocal,
                                                                                      LocalTensor<T>& dstLocal)
{
    if constexpr (sizeof(T) == 1) {
        ComputeSplitDim1B8<needColAlign>(rows, cols, tensorStride, inputCols, srcLocal, dstLocal);
    } else {
        ComputeSplitDim1Norm<needColAlign>(rows, cols, tensorStride, inputCols, srcLocal, dstLocal);
    }
}

template <typename T, typename U, typename TILINGDATA>
template <const bool needColAlign>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::ComputeSplitDim1Norm(uint32_t rows, uint32_t cols,
                                                                                          uint32_t tensorStride,
                                                                                          uint32_t inputCols,
                                                                                          LocalTensor<T>& srcLocal,
                                                                                          LocalTensor<T>& dstLocal)
{
    uint16_t size0 = rows;
    uint16_t everyRegCalcTensorNum = vfLen_ / inputCols;
    uint16_t regFactorDim1 = everyRegCalcTensorNum * inputCols;
    uint16_t size1 = cols / regFactorDim1;
    uint16_t tailRegFactorDim1 = cols - size1 * regFactorDim1;

    uint32_t padCols = cols;
    if constexpr (needColAlign) {
        padCols = (cols + numPerBlock_ - 1) / numPerBlock_ * numPerBlock_;
    }
    auto dstAddr = (__ubuf__ T*)dstLocal.GetPhyAddr();
    auto srcAddr = (__ubuf__ T*)srcLocal.GetPhyAddr();
    LocalTensor<U> indexLocal(TPosition::VECCALC, idxOffset_, idxElems_);
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __ubuf__ T* curDstAddr = dstAddr;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<U> vd0;
        AscendC::Reg::RegTensor<U> vd1;
        AscendC::Reg::RegTensor<T> vd2;
        AscendC::Reg::RegTensor<U> vd3;
        AscendC::Reg::RegTensor<T> vd4;
        AscendC::Reg::UnalignRegForStore u0;
        AscendC::Reg::UnalignRegForStore u1;

        uint32_t num = static_cast<uint32_t>(regFactorDim1);
        uint32_t tailNum = static_cast<uint32_t>(tailRegFactorDim1);
        uint32_t pnum = num;
        uint32_t tailPnum = tailNum;

        uint32_t main = num;
        uint32_t tail = tailNum;

        AscendC::Reg::LoadAlign(vd0, indexAddr);
        AscendC::Reg::MaskReg p0 = AscendC::Reg::UpdateMask<U>(pnum);
        AscendC::Reg::MaskReg p1 = AscendC::Reg::UpdateMask<U>(tailPnum);
        uint32_t stride = everyRegCalcTensorNum * tensorStride;
        uint32_t rowStride = 0;
        for (uint16_t i = 0; i < size0; i++) {
            main = num;
            curDstAddr = dstAddr + i * padCols;
            rowStride = i * inputCols;
            for (uint16_t j = 0; j < size1; j++) {
                AscendC::Reg::Adds(vd1, vd0, (U)(rowStride + j * stride), p0);
                AscendC::Reg::Gather(vd2, srcAddr, vd1, p0);
                AscendC::Reg::StoreUnAlign(curDstAddr, vd2, u0, main);
                AscendC::Reg::StoreUnAlignPost(curDstAddr, u0, 0);
            }
            tail = tailNum;
            AscendC::Reg::Adds(vd3, vd0, (U)(rowStride + size1 * stride), p1);
            AscendC::Reg::Gather(vd4, srcAddr, vd3, p1);
            AscendC::Reg::StoreUnAlign(curDstAddr, vd4, u1, tail);
            AscendC::Reg::StoreUnAlignPost(curDstAddr, u1, 0);
        }
    }
}

template <typename T, typename U, typename TILINGDATA>
template <const bool needColAlign>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::ComputeSplitDim1B8(uint32_t rows, uint32_t cols,
                                                                                        uint32_t tensorStride,
                                                                                        uint32_t inputCols,
                                                                                        LocalTensor<T>& srcLocal,
                                                                                        LocalTensor<T>& dstLocal)
{
    uint16_t everyRegCalcTensorNum = vfLen_ / inputCols;
    uint16_t size0 = rows;
    uint16_t regFactorDim1 = everyRegCalcTensorNum * inputCols;
    uint16_t size1 = cols / regFactorDim1;
    uint16_t tailRegFactorDim1 = cols - size1 * regFactorDim1;

    uint32_t padCols = cols;
    if constexpr (needColAlign) {
        padCols = (cols + numPerBlock_ - 1) / numPerBlock_ * numPerBlock_;
    }
    auto dstAddr = reinterpret_cast<__ubuf__ uint8_t*>(dstLocal.GetPhyAddr());
    auto srcAddr = reinterpret_cast<__ubuf__ uint8_t*>(srcLocal.GetPhyAddr());
    LocalTensor<U> indexLocal(TPosition::VECCALC, idxOffset_, idxElems_);
    auto indexAddr = reinterpret_cast<__ubuf__ U*>(indexLocal.GetPhyAddr());
    __ubuf__ uint8_t* curDstAddr;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<uint16_t> vd0;
        AscendC::Reg::RegTensor<uint16_t> vd1;
        AscendC::Reg::RegTensor<uint16_t> vd2;
        AscendC::Reg::RegTensor<uint8_t> vd3;
        AscendC::Reg::RegTensor<uint16_t> vd4;
        AscendC::Reg::RegTensor<uint16_t> vd5;
        AscendC::Reg::RegTensor<uint8_t> vd6;
        AscendC::Reg::UnalignRegForStore u0;
        AscendC::Reg::UnalignRegForStore u1;

        uint32_t num = static_cast<uint32_t>(regFactorDim1);
        uint32_t tailNum = static_cast<uint32_t>(tailRegFactorDim1);
        uint32_t tailPnum = tailNum;
        uint32_t pnum = num;

        uint32_t main = num;
        uint32_t tail = tailNum;

        AscendC::Reg::LoadAlign(vd0, indexAddr);
        AscendC::Reg::MaskReg p1 = AscendC::Reg::UpdateMask<U>(tailPnum);
        AscendC::Reg::MaskReg p0 = AscendC::Reg::UpdateMask<U>(pnum);
        uint32_t stride = everyRegCalcTensorNum * tensorStride;
        uint32_t rowStride = 0;
        for (uint16_t i = 0; i < size0; i++) {
            main = num;
            curDstAddr = dstAddr + i * padCols;
            rowStride = inputCols * i;
            for (uint16_t j = 0; j < size1; j++) {
                AscendC::Reg::Adds(vd1, vd0, (U)(rowStride + j * stride), p0);
                AscendC::Reg::Gather(vd2, srcAddr, vd1, p0);
                AscendC::Reg::Pack(vd3, vd2);
                AscendC::Reg::StoreUnAlign(curDstAddr, vd3, u0, main);
                AscendC::Reg::StoreUnAlignPost(curDstAddr, u0, 0);
            }
            tail = tailNum;
            AscendC::Reg::Adds(vd4, vd0, (U)(rowStride + size1 * stride), p1);
            AscendC::Reg::Gather(vd5, srcAddr, vd4, p1);
            AscendC::Reg::Pack(vd6, vd5);
            AscendC::Reg::StoreUnAlign(curDstAddr, vd6, u1, tail);
            AscendC::Reg::StoreUnAlignPost(curDstAddr, u1, 0);
        }
    }
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::CopyInNoSplitDim1(int64_t srcRowsOffset,
                                                                                       int64_t rows, int64_t startIdx,
                                                                                       int64_t endIdx,
                                                                                       LocalTensor<T>& srcLocal)
{
    int64_t curDim1Offset = 0;
    int64_t tensorStride = (rows * sameShapeTensorDim1_ + numPerBlock_ - 1) / numPerBlock_ * numPerBlock_;
    DataCopyPadExtParams<T> padParam = {false, 0, 0, 0};
    for (int64_t tensorIdx = startIdx; tensorIdx < endIdx; tensorIdx++) {
        int64_t dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, tensorIdx, sameShapeTensorDim1_);
        if (isNonContiguous_) {
            SetCopyInparam(rows, sameShapeTensorDim1_, dim0stride - sameShapeTensorDim1_, 0);
        } else {
            SetCopyInparam(1, rows * sameShapeTensorDim1_, 0, 0);
        }
        srcGlobal_.SetGlobalBuffer(GetTensorAddr(tensorIdx, blockOffset_ * dim0stride));
#if defined(__DAV_920R1__) || defined(__DAV_920R2__)
        DataCopyPad<T, PaddingMode::Compact>(srcLocal[curDim1Offset], srcGlobal_[srcRowsOffset * dim0stride],
                                             copyInParam_, padParam, true);
#else
        DataCopyPad<T, PaddingMode::Compact>(srcLocal[curDim1Offset], srcGlobal_[srcRowsOffset * dim0stride],
                                             copyInParam_, padParam);
#endif
        curDim1Offset += tensorStride;
    }
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::SetCopyInparam(int64_t rows, int64_t cols,
                                                                                    int64_t srcStride,
                                                                                    int64_t dstStride)
{
    copyInParam_.blockCount = rows;
    copyInParam_.blockLen = cols * sizeof(T);
    copyInParam_.srcStride = srcStride * sizeof(T);
    copyInParam_.dstStride = dstStride * sizeof(T);
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::CopyOut(int64_t dstOffset, int64_t rows,
                                                                             int64_t cols, LocalTensor<T>& dstLocal)
{
    DataCopyExtParams copyOutParam = {0, 0, 0, 0, 0};
    copyOutParam.blockCount = rows;
    copyOutParam.blockLen = cols * sizeof(T);
    copyOutParam.srcStride = 0;
    copyOutParam.dstStride = (catDim1_ - cols) * sizeof(T);
    DataCopyPad(dstGlobal_[dstOffset], dstLocal, copyOutParam);
}

template <typename T, typename U, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignGather<T, U, TILINGDATA>::CopyOut(int64_t dstOffset, int64_t dataLen,
                                                                             LocalTensor<T>& dstLocal)
{
    DataCopyExtParams copyOutParam = {0, 0, 0, 0, 0};
    copyOutParam.blockCount = 1;
    copyOutParam.blockLen = dataLen * sizeof(T);
    copyOutParam.srcStride = 0;
    copyOutParam.dstStride = 0;
    DataCopyPad(dstGlobal_[dstOffset], dstLocal, copyOutParam);
}

} // namespace Concat

#endif // ONE_AXIS_CONCAT_NO_ALIGN_SAME_SHAPE_GATHER
