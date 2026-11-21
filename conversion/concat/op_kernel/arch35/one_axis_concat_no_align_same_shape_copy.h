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
 * \file one_axis_concat_no_align_same_shape_copy.h
 * \brief one axis concat no align same shape copy
 */

#ifndef ONE_AXIS_CONCAT_NO_ALIGN_SAME_SHAPE_COPY
#define ONE_AXIS_CONCAT_NO_ALIGN_SAME_SHAPE_COPY

#include "concat_base.h"
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

namespace Concat {
using namespace AscendC;

template <typename T, typename TILINGDATA = ConcatTilingData>
class OneAxisConcatNoAlignCopy {
public:
    __aicore__ inline OneAxisConcatNoAlignCopy(__tiling_data_ptr__ TILINGDATA* tilingData) : tilingData_(tilingData){};
    static __aicore__ inline void PreloadTiling(__tiling_data_ptr__ TILINGDATA* tilingData)
    {
        PreloadMergedInfo(tilingData);
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dst);
    __aicore__ inline void Process();

private:
    __aicore__ inline __gm__ T* GetTensorAddr(uint32_t index, int64_t offset);
    __aicore__ inline void CopyInNoSplitDim1(int64_t srcRowsOffset, int64_t rows, int64_t startIdx, int64_t endIdx,
                                             LocalTensor<T>& srcLocal);
    __aicore__ inline void CopyOut(int64_t dstOffset, int64_t rows, int64_t cols, LocalTensor<T>& dstLocal);
    __aicore__ inline void CopyOut(int64_t dstOffset, int64_t dataLen, LocalTensor<T>& dstLocal);
    __aicore__ inline void SetCopyInparam(int64_t rows, int64_t cols, int64_t srcStride, int64_t dstStride);
    __aicore__ inline void ProcessBlockSplitDim0NoSplitDim1();
    __aicore__ inline void ProcessBlockSplitDim0SplitDim1();
    __aicore__ inline void ProcessBlockSplitDim1();
    __aicore__ inline void ComputeNoSplitDim1Copy(uint32_t rows, uint32_t cols, uint32_t tensorStride,
                                                  uint32_t inputCols, uint32_t dstColStride, LocalTensor<T>& srcLocal,
                                                  LocalTensor<T>& dstLocal);
    __aicore__ inline void ComputeSplitDim1Copy(uint32_t rows, uint32_t cols, SplitInfo& splitInfo,
                                                LocalTensor<T>& srcLocal, LocalTensor<T>& dstLocal);
    __aicore__ inline void CopyInSingleTensor(const LocalTensor<T>& dstLocal, int64_t tensorIdx, int64_t srcOffset,
                                              const DataCopyExtParams& params);
    __aicore__ inline int64_t CopyInSplitDim1(int64_t endIdx, int64_t endOffset, int64_t srcRowsOffset, int64_t rows,
                                              SplitInfo& splitInfo, LocalTensor<T>& srcLocal);

private:
    constexpr static event_t EVT_ID0 = static_cast<event_t>(0);
    constexpr static event_t EVT_ID1 = static_cast<event_t>(1);

    GlobalTensor<T> dstGlobal_;
    GlobalTensor<T> srcGlobal_;
    __tiling_data_ptr__ TILINGDATA* tilingData_;
    int64_t blockIdx_ = 0;
    int64_t blockOffset_ = 0;
    int64_t numPerBlock_ = BYTES_PER_BLOCK / sizeof(T);
    int64_t colsUsedCoreNum_ = 1;
    int64_t rowsUsedCoreNum_ = 1;

    DataCopyExtParams copyInParam_ = {0, 0, 0, 0, 0};

    int64_t startTensorIdx_ = 0;
    int64_t startTensorOffset_ = 0;
    int64_t endTensorIdx_ = 0;
    int64_t endTensorOffset_ = 0;
    ListTensorDesc inputList_;

    uint32_t vfLen_ = GetVRegSize() / sizeof(T);

    uint32_t bufSize_{0};
    uint32_t bufElems_{0};
    uint32_t inOffset0_{0};
    uint32_t inOffset1_{0};
    uint32_t outOffset0_{0};
    uint32_t outOffset1_{0};

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

template <typename T, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignCopy<T, TILINGDATA>::Init(GM_ADDR x, GM_ADDR dst)
{
    blockIdx_ = GetBlockIdx();
    if constexpr (IsSame<TILINGDATA, ConcatTilingData>::value || IsSame<TILINGDATA, ConcatTilingDataCompact>::value) {
        int64_t colsUsedCoreNum = GetBlockNum() / tilingData_->uoDim0;
        if (blockIdx_ % colsUsedCoreNum != 0) {
            // 数组按核行重复展开，blockIdx-1 即同列上一核定界，与本核定界对相邻
            startTensorIdx_ = tilingData_->arrays.mergedInfo[CORE_INFO_BASE + 2 * (blockIdx_ - 1)];
            startTensorOffset_ = tilingData_->arrays.mergedInfo[CORE_INFO_BASE + 2 * (blockIdx_ - 1) + 1];
        }
        endTensorIdx_ = tilingData_->arrays.mergedInfo[CORE_INFO_BASE + 2 * blockIdx_];
        endTensorOffset_ = tilingData_->arrays.mergedInfo[CORE_INFO_BASE + 2 * blockIdx_ + 1];
    }
    if (startTensorOffset_ == tilingData_->sameShapeTensorDim1) {
        startTensorIdx_ += 1;
        startTensorOffset_ = 0;
    }
    if constexpr (IsSame<TILINGDATA, ConcatTilingData>::value || IsSame<TILINGDATA, ConcatTilingDataCompact>::value) {
        rowsUsedCoreNum_ = tilingData_->uoDim0;
        colsUsedCoreNum_ = GetBlockNum() / rowsUsedCoreNum_;
        blockOffset_ = (blockIdx_ / colsUsedCoreNum_) * tilingData_->ubFactorDim0;
        int64_t colOffset = (blockIdx_ % colsUsedCoreNum_) * tilingData_->blockFactor * tilingData_->ubFactorDim1;
        dstGlobal_.SetGlobalBuffer((__gm__ T*)dst + blockOffset_ * tilingData_->catDim1 + colOffset);
    } else {
        blockOffset_ = blockIdx_ * tilingData_->blockFactor * tilingData_->ubFactorDim0;
        dstGlobal_.SetGlobalBuffer((__gm__ T*)dst + blockOffset_ * tilingData_->catDim1);
    }

    bufSize_ = static_cast<uint32_t>(tilingData_->bufferSize * sizeof(T));
    bufElems_ = static_cast<uint32_t>(tilingData_->bufferSize);
    inOffset0_ = 0;
    inOffset1_ = bufSize_;
    outOffset0_ = bufSize_ * BUFFER_NUM;
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

template <typename T, typename TILINGDATA>
__aicore__ inline __gm__ T* OneAxisConcatNoAlignCopy<T, TILINGDATA>::GetTensorAddr(uint32_t index, int64_t offset)
{
    return inputList_.GetDataPtr<T>(index) + offset;
}

template <typename T, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignCopy<T, TILINGDATA>::Process()
{
    if (blockIdx_ >= GetBlockNum()) {
        return;
    }
    if constexpr (IsSame<TILINGDATA, ConcatTilingData>::value || IsSame<TILINGDATA, ConcatTilingDataCompact>::value) {
        ProcessBlockSplitDim1();
    } else {
        if (0 == ubSplitDim1_) {
            ProcessBlockSplitDim0NoSplitDim1();
        } else {
            ProcessBlockSplitDim0SplitDim1();
        }
    }
}

template <typename T, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignCopy<T, TILINGDATA>::ProcessBlockSplitDim1()
{
    int64_t blockIdxInCol = blockIdx_ % colsUsedCoreNum_;
    int64_t blockIdxInRow = blockIdx_ / colsUsedCoreNum_;
    int64_t copyRowsNum = ubFactorDim0_;

    if (blockIdxInRow == rowsUsedCoreNum_ - 1) {
        copyRowsNum = tailUbFactorDim0_;
    }
    int64_t curColOffset = 0;
    int64_t curColNum = 0;
    int64_t totalCols = 0;
    SplitInfo splitInfo = {0, 0, startTensorIdx_, startTensorOffset_};
    if (startTensorIdx_ == endTensorIdx_) {
        totalCols = endTensorOffset_ - startTensorOffset_;
    } else {
        totalCols = (sameShapeTensorDim1_ - startTensorOffset_) + endTensorOffset_ +
                    (endTensorIdx_ - startTensorIdx_ - 1) * sameShapeTensorDim1_;
    }
    SetFlag<HardEvent::V_MTE2>(EVT_ID0);
    SetFlag<HardEvent::V_MTE2>(EVT_ID1);
    SetFlag<HardEvent::MTE3_V>(EVT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVT_ID1);
    uint16_t bufIdX = 0;
    uint16_t bufIdY = 0;
    while (curColOffset < totalCols) {
        event_t evtX = (bufIdX == 0) ? EVT_ID0 : EVT_ID1;
        WaitFlag<HardEvent::V_MTE2>(evtX);
        LocalTensor<T> srcLocal(TPosition::VECIN, (bufIdX == 0) ? inOffset0_ : inOffset1_, bufElems_);
        curColNum = CopyInSplitDim1(endTensorIdx_, endTensorOffset_, 0, copyRowsNum, splitInfo, srcLocal);
        SetFlag<HardEvent::MTE2_V>(evtX);
        WaitFlag<HardEvent::MTE2_V>(evtX);

        event_t evtY = (bufIdY == 0) ? EVT_ID0 : EVT_ID1;
        WaitFlag<HardEvent::MTE3_V>(evtY);
        LocalTensor<T> dstLocal(TPosition::VECOUT, (bufIdY == 0) ? outOffset0_ : outOffset1_, bufElems_);
        ComputeSplitDim1Copy(copyRowsNum, curColNum, splitInfo, srcLocal, dstLocal);
        SetFlag<HardEvent::V_MTE2>(evtX);
        SetFlag<HardEvent::V_MTE3>(evtY);

        WaitFlag<HardEvent::V_MTE3>(evtY);
        CopyOut(curColOffset, copyRowsNum, curColNum, dstLocal);
        SetFlag<HardEvent::MTE3_V>(evtY);
        bufIdX ^= 1;
        bufIdY ^= 1;
        curColOffset += curColNum;
    }
    WaitFlag<HardEvent::V_MTE2>(EVT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVT_ID1);
}

template <typename T, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignCopy<T, TILINGDATA>::ProcessBlockSplitDim0SplitDim1()
{
    int64_t loopSizeDim0 = blockFactor_;
    if (blockIdx_ == GetBlockNum() - 1) {
        loopSizeDim0 = tailBlockFactor_ - 1;
    }

    int64_t curColOffset = 0;
    int64_t curColNum = 0;
    SplitInfo splitInfo = {0, 0, 0, 0};

    SetFlag<HardEvent::V_MTE2>(EVT_ID0);
    SetFlag<HardEvent::V_MTE2>(EVT_ID1);
    SetFlag<HardEvent::MTE3_V>(EVT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVT_ID1);
    uint16_t bufIdX = 0;
    uint16_t bufIdY = 0;
    for (int64_t i = 0; i < loopSizeDim0; i++) {
        curColOffset = 0;
        curColNum = 0;
        splitInfo = {0, 0, 0, 0};
        while (curColOffset < catDim1_) {
            event_t evtX = (bufIdX == 0) ? EVT_ID0 : EVT_ID1;
            WaitFlag<HardEvent::V_MTE2>(evtX);
            LocalTensor<T> srcLocal(TPosition::VECIN, (bufIdX == 0) ? inOffset0_ : inOffset1_, bufElems_);
            curColNum = CopyInSplitDim1(tensorNum_ - 1, sameShapeTensorDim1_, i * ubFactorDim0_, ubFactorDim0_,
                                        splitInfo, srcLocal);
            SetFlag<HardEvent::MTE2_V>(evtX);
            WaitFlag<HardEvent::MTE2_V>(evtX);

            event_t evtY = (bufIdY == 0) ? EVT_ID0 : EVT_ID1;
            WaitFlag<HardEvent::MTE3_V>(evtY);
            LocalTensor<T> dstLocal(TPosition::VECOUT, (bufIdY == 0) ? outOffset0_ : outOffset1_, bufElems_);
            ComputeSplitDim1Copy(ubFactorDim0_, curColNum, splitInfo, srcLocal, dstLocal);
            SetFlag<HardEvent::V_MTE2>(evtX);
            SetFlag<HardEvent::V_MTE3>(evtY);

            WaitFlag<HardEvent::V_MTE3>(evtY);
            CopyOut(i * ubFactorDim0_ * catDim1_ + curColOffset, ubFactorDim0_, curColNum, dstLocal);
            SetFlag<HardEvent::MTE3_V>(evtY);
            bufIdX ^= 1;
            bufIdY ^= 1;
            curColOffset += curColNum;
        }
    }

    if (blockIdx_ == GetBlockNum() - 1) {
        curColOffset = 0;
        curColNum = 0;
        splitInfo = {0, 0, 0, 0};
        while (curColOffset < catDim1_) {
            event_t evtX = (bufIdX == 0) ? EVT_ID0 : EVT_ID1;
            WaitFlag<HardEvent::V_MTE2>(evtX);
            LocalTensor<T> srcLocal(TPosition::VECIN, (bufIdX == 0) ? inOffset0_ : inOffset1_, bufElems_);
            curColNum = CopyInSplitDim1(tensorNum_ - 1, sameShapeTensorDim1_, loopSizeDim0 * ubFactorDim0_,
                                        tailUbFactorDim0_, splitInfo, srcLocal);
            SetFlag<HardEvent::MTE2_V>(evtX);
            WaitFlag<HardEvent::MTE2_V>(evtX);

            event_t evtY = (bufIdY == 0) ? EVT_ID0 : EVT_ID1;
            WaitFlag<HardEvent::MTE3_V>(evtY);
            LocalTensor<T> dstLocal(TPosition::VECOUT, (bufIdY == 0) ? outOffset0_ : outOffset1_, bufElems_);
            ComputeSplitDim1Copy(tailUbFactorDim0_, curColNum, splitInfo, srcLocal, dstLocal);
            SetFlag<HardEvent::V_MTE2>(evtX);
            SetFlag<HardEvent::V_MTE3>(evtY);

            WaitFlag<HardEvent::V_MTE3>(evtY);
            CopyOut(loopSizeDim0 * ubFactorDim0_ * catDim1_ + curColOffset, tailUbFactorDim0_, curColNum, dstLocal);
            SetFlag<HardEvent::MTE3_V>(evtY);
            bufIdX ^= 1;
            bufIdY ^= 1;
            curColOffset += curColNum;
        }
    }
    WaitFlag<HardEvent::V_MTE2>(EVT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVT_ID1);
}

template <typename T, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignCopy<T, TILINGDATA>::ProcessBlockSplitDim0NoSplitDim1()
{
    int64_t loopSize = blockFactor_;
    if (blockIdx_ == GetBlockNum() - 1) {
        loopSize = tailBlockFactor_ - 1;
    }
    int64_t tensorStride = (ubFactorDim0_ * sameShapeTensorDim1_ + numPerBlock_ - 1) / numPerBlock_ * numPerBlock_;

    SetFlag<HardEvent::V_MTE2>(EVT_ID0);
    SetFlag<HardEvent::V_MTE2>(EVT_ID1);
    SetFlag<HardEvent::MTE3_V>(EVT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVT_ID1);
    uint16_t bufIdX = 0;
    uint16_t bufIdY = 0;
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
        ComputeNoSplitDim1Copy(ubFactorDim0_, catDim1_, tensorStride, sameShapeTensorDim1_, catDim1_, srcLocal,
                               dstLocal);
        SetFlag<HardEvent::V_MTE2>(evtX);
        SetFlag<HardEvent::V_MTE3>(evtY);

        WaitFlag<HardEvent::V_MTE3>(evtY);
        CopyOut(i * ubFactorDim0_ * catDim1_, ubFactorDim0_ * catDim1_, dstLocal);
        SetFlag<HardEvent::MTE3_V>(evtY);
        bufIdX ^= 1;
        bufIdY ^= 1;
    }
    if (blockIdx_ == GetBlockNum() - 1) {
        tensorStride = (tailUbFactorDim0_ * sameShapeTensorDim1_ + numPerBlock_ - 1) / numPerBlock_ * numPerBlock_;
        event_t evtX = (bufIdX == 0) ? EVT_ID0 : EVT_ID1;
        WaitFlag<HardEvent::V_MTE2>(evtX);
        LocalTensor<T> srcLocal(TPosition::VECIN, (bufIdX == 0) ? inOffset0_ : inOffset1_, bufElems_);
        CopyInNoSplitDim1(loopSize * ubFactorDim0_, tailUbFactorDim0_, 0, tensorNum_, srcLocal);
        SetFlag<HardEvent::MTE2_V>(evtX);
        WaitFlag<HardEvent::MTE2_V>(evtX);

        event_t evtY = (bufIdY == 0) ? EVT_ID0 : EVT_ID1;
        WaitFlag<HardEvent::MTE3_V>(evtY);
        LocalTensor<T> dstLocal(TPosition::VECOUT, (bufIdY == 0) ? outOffset0_ : outOffset1_, bufElems_);
        ComputeNoSplitDim1Copy(tailUbFactorDim0_, catDim1_, tensorStride, sameShapeTensorDim1_, catDim1_, srcLocal,
                               dstLocal);
        SetFlag<HardEvent::V_MTE2>(evtX);
        SetFlag<HardEvent::V_MTE3>(evtY);

        WaitFlag<HardEvent::V_MTE3>(evtY);
        CopyOut(loopSize * ubFactorDim0_ * catDim1_, tailUbFactorDim0_ * catDim1_, dstLocal);
        SetFlag<HardEvent::MTE3_V>(evtY);
        bufIdX ^= 1;
        bufIdY ^= 1;
    }
    WaitFlag<HardEvent::V_MTE2>(EVT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVT_ID1);
}

template <typename T, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignCopy<T, TILINGDATA>::ComputeNoSplitDim1Copy(
    uint32_t rows, uint32_t cols, uint32_t tensorStride, uint32_t inputCols, uint32_t dstColStride,
    LocalTensor<T>& srcLocal, LocalTensor<T>& dstLocal)
{
    uint16_t inputNum = cols / inputCols;
    uint16_t size0 = rows;
    uint16_t size1 = inputCols / vfLen_;

    uint32_t main = vfLen_;
    uint32_t tail = inputCols - size1 * vfLen_;

    auto dstAddr = (__ubuf__ T*)dstLocal.GetPhyAddr();
    auto srcAddr = (__ubuf__ T*)srcLocal.GetPhyAddr();

    __ubuf__ T* curDstAddr;
    __ubuf__ T* curSrcAddr;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T> vd0;
        AscendC::Reg::RegTensor<T> vd1;
        AscendC::Reg::UnalignRegForLoad u0;
        AscendC::Reg::UnalignRegForStore u1;
        for (uint16_t i = 0; i < size0; i++) {
            for (uint16_t j = 0; j < inputNum; j++) {
                curSrcAddr = srcAddr + j * tensorStride + i * inputCols;
                curDstAddr = dstAddr + i * dstColStride + j * inputCols;
                AscendC::Reg::LoadUnAlignPre(u0, curSrcAddr);
                for (uint16_t k = 0; k < size1; k++) {
                    AscendC::Reg::LoadUnAlign(vd0, u0, curSrcAddr, main);
                    AscendC::Reg::StoreUnAlign(curDstAddr, vd0, u1, main);
                }
                AscendC::Reg::LoadUnAlign(vd0, u0, curSrcAddr, tail);
                AscendC::Reg::StoreUnAlign(curDstAddr, vd0, u1, tail);
                AscendC::Reg::StoreUnAlignPost(curDstAddr, u1, 0);
            }
        }
    }
}

template <typename T, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignCopy<T, TILINGDATA>::ComputeSplitDim1Copy(uint32_t rows, uint32_t cols,
                                                                                     SplitInfo& splitInfo,
                                                                                     LocalTensor<T>& srcLocal,
                                                                                     LocalTensor<T>& dstLocal)
{
    int64_t copyColsNum = 0;
    int64_t srcOffset = 0;
    int64_t dstOffset = 0;
    uint32_t rowStride = (cols + numPerBlock_ - 1) / numPerBlock_ * numPerBlock_;
    uint32_t tensorOffset = (rows * sameShapeTensorDim1_ + numPerBlock_ - 1) / numPerBlock_ * numPerBlock_;

    if (splitInfo.startIdx == splitInfo.endIdx) {
        copyColsNum = splitInfo.endOffset - splitInfo.startOffset;
    } else {
        copyColsNum = sameShapeTensorDim1_ - splitInfo.startOffset;
    }
    Copy(dstLocal, srcLocal, rows, copyColsNum, rowStride);
    srcOffset += (copyColsNum * rows + numPerBlock_ - 1) / numPerBlock_ * numPerBlock_;
    dstOffset += copyColsNum;

    for (int64_t i = splitInfo.startIdx + 1; i < splitInfo.endIdx; i++) {
        Copy(dstLocal[dstOffset], srcLocal[srcOffset], rows, sameShapeTensorDim1_, rowStride);
        dstOffset += sameShapeTensorDim1_;
        srcOffset += tensorOffset;
    }

    if (splitInfo.startIdx != splitInfo.endIdx) {
        copyColsNum = splitInfo.endOffset;
        Copy(dstLocal[dstOffset], srcLocal[srcOffset], rows, copyColsNum, rowStride);
    }
}

template <typename T, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignCopy<T, TILINGDATA>::CopyInNoSplitDim1(int64_t srcRowsOffset, int64_t rows,
                                                                                  int64_t startIdx, int64_t endIdx,
                                                                                  LocalTensor<T>& srcLocal)
{
    int64_t curDim1Offset = 0;
    int64_t tensorStride = (rows * sameShapeTensorDim1_ + numPerBlock_ - 1) / numPerBlock_ * numPerBlock_;
    SetCopyInparam(1, rows * sameShapeTensorDim1_, 0, 0);
    DataCopyPadExtParams<T> padParam = {false, 0, 0, 0};
    for (int64_t i = startIdx; i < endIdx; i++) {
        int64_t dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, i, sameShapeTensorDim1_);
        if (isNonContiguous_) {
            SetCopyInparam(rows, sameShapeTensorDim1_, dim0stride - sameShapeTensorDim1_, 0);
        } else {
            SetCopyInparam(1, rows * sameShapeTensorDim1_, 0, 0);
        }
        srcGlobal_.SetGlobalBuffer(GetTensorAddr(i, blockOffset_ * dim0stride));
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

template <typename T, typename TILINGDATA>
__aicore__ inline int64_t OneAxisConcatNoAlignCopy<T, TILINGDATA>::CopyInSplitDim1(int64_t endIdx, int64_t endOffset,
                                                                                   int64_t srcRowsOffset, int64_t rows,
                                                                                   SplitInfo& splitInfo,
                                                                                   LocalTensor<T>& srcLocal)
{
    int64_t curOffset = 0;
    int64_t limit = bufferSize_ - rows * numPerBlock_ - numPerBlock_;
    int64_t limitCols = limit / rows;
    int64_t copyCols = 0;
    int64_t colOffset = 0;
    int64_t dim0stride = 0;

    splitInfo.startIdx = splitInfo.endIdx;
    splitInfo.startOffset = splitInfo.endOffset;

    if (splitInfo.startOffset == sameShapeTensorDim1_) {
        splitInfo.startIdx += 1;
        splitInfo.startOffset = 0;
    }
    if (splitInfo.startIdx == endIdx) {
        copyCols = splitInfo.startOffset + limitCols > endOffset ? endOffset - splitInfo.startOffset : limitCols;
        dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, splitInfo.startIdx, sameShapeTensorDim1_);
        SetCopyInparam(rows, copyCols, dim0stride - copyCols, copyCols);
        splitInfo.endOffset = splitInfo.startOffset + copyCols;
        splitInfo.endIdx = splitInfo.startIdx;
        CopyInSingleTensor(srcLocal, splitInfo.startIdx, srcRowsOffset * dim0stride + splitInfo.startOffset,
                           copyInParam_);
        return copyCols;
    }
    if ((sameShapeTensorDim1_ - splitInfo.startOffset) >= limitCols) {
        copyCols = limitCols;
        dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, splitInfo.startIdx, sameShapeTensorDim1_);
        SetCopyInparam(rows, copyCols, dim0stride - copyCols, copyCols);
        splitInfo.endOffset = splitInfo.startOffset + copyCols;
        splitInfo.endIdx = splitInfo.startIdx;
        CopyInSingleTensor(srcLocal, splitInfo.startIdx, srcRowsOffset * dim0stride + splitInfo.startOffset,
                           copyInParam_);
        return copyCols;
    }
    copyCols = sameShapeTensorDim1_ - splitInfo.startOffset;
    dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, splitInfo.startIdx, sameShapeTensorDim1_);
    SetCopyInparam(rows, copyCols, dim0stride - copyCols, copyCols);
    CopyInSingleTensor(srcLocal, splitInfo.startIdx, srcRowsOffset * dim0stride + splitInfo.startOffset, copyInParam_);
    colOffset += copyCols;
    curOffset += (copyCols * rows + numPerBlock_ - 1) / numPerBlock_ * numPerBlock_;

    int64_t tensorStride = (sameShapeTensorDim1_ * rows + numPerBlock_ - 1) / numPerBlock_ * numPerBlock_;
    for (int64_t i = splitInfo.startIdx + 1; i < endIdx; i++) {
        if (curOffset + tensorStride >= limit) {
            copyCols = min((limit - curOffset) / rows, sameShapeTensorDim1_);
            splitInfo.endIdx = i;
            splitInfo.endOffset = copyCols;
            dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, i, sameShapeTensorDim1_);
            SetCopyInparam(rows, copyCols, dim0stride - copyCols, copyCols);
            CopyInSingleTensor(srcLocal[curOffset], i, srcRowsOffset * dim0stride, copyInParam_);
            colOffset += copyCols;
            return colOffset;
        }
        dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, i, sameShapeTensorDim1_);
        SetCopyInparam(rows, sameShapeTensorDim1_, dim0stride - sameShapeTensorDim1_, sameShapeTensorDim1_);
        CopyInSingleTensor(srcLocal[curOffset], i, srcRowsOffset * dim0stride, copyInParam_);
        colOffset += sameShapeTensorDim1_;
        curOffset += tensorStride;
    }

    if (curOffset + endOffset * rows < limit) {
        copyCols = endOffset;
    } else {
        copyCols = (limit - curOffset) / rows;
    }
    splitInfo.endIdx = endIdx;
    splitInfo.endOffset = copyCols;
    dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, splitInfo.endIdx, tilingData_->sameShapeTensorDim1);
    SetCopyInparam(rows, copyCols, dim0stride - copyCols, copyCols);
    CopyInSingleTensor(srcLocal[curOffset], endIdx, srcRowsOffset * dim0stride, copyInParam_);
    colOffset += copyCols;
    return colOffset;
}

template <typename T, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignCopy<T, TILINGDATA>::CopyInSingleTensor(const LocalTensor<T>& dstLocal,
                                                                                   int64_t tensorIdx, int64_t srcOffset,
                                                                                   const DataCopyExtParams& params)
{
    int64_t dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, tensorIdx, sameShapeTensorDim1_);
    srcGlobal_.SetGlobalBuffer(GetTensorAddr(tensorIdx, blockOffset_ * dim0stride));
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
#if defined(__DAV_920R1__) || defined(__DAV_920R2__)
    DataCopyPad<T, PaddingMode::Compact>(dstLocal, srcGlobal_[srcOffset], params, padParams, true);
#else
    DataCopyPad<T, PaddingMode::Compact>(dstLocal, srcGlobal_[srcOffset], params, padParams);
#endif
}

template <typename T, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignCopy<T, TILINGDATA>::SetCopyInparam(int64_t rows, int64_t cols,
                                                                               int64_t srcStride, int64_t dstStride)
{
    copyInParam_.blockCount = rows;
    copyInParam_.blockLen = cols * sizeof(T);
    copyInParam_.srcStride = srcStride * sizeof(T);
    copyInParam_.dstStride = dstStride * sizeof(T);
}

template <typename T, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignCopy<T, TILINGDATA>::CopyOut(int64_t dstOffset, int64_t rows, int64_t cols,
                                                                        LocalTensor<T>& dstLocal)
{
    DataCopyExtParams copyOutParam = {0, 0, 0, 0, 0};
    copyOutParam.blockCount = rows;
    copyOutParam.blockLen = cols * sizeof(T);
    copyOutParam.dstStride = (catDim1_ - cols) * sizeof(T);
    copyOutParam.srcStride = 0;
    DataCopyPad(dstGlobal_[dstOffset], dstLocal, copyOutParam);
}

template <typename T, typename TILINGDATA>
__aicore__ inline void OneAxisConcatNoAlignCopy<T, TILINGDATA>::CopyOut(int64_t dstOffset, int64_t dataLen,
                                                                        LocalTensor<T>& dstLocal)
{
    DataCopyExtParams copyOutParam = {0, 0, 0, 0, 0};
    copyOutParam.blockCount = 1;
    copyOutParam.blockLen = dataLen * sizeof(T);
    copyOutParam.dstStride = 0;
    copyOutParam.srcStride = 0;
    DataCopyPad(dstGlobal_[dstOffset], dstLocal, copyOutParam);
}

} // namespace Concat

#endif // ONE_AXIS_CONCAT_NO_ALIGN_SAME_SHAPE_COPY
