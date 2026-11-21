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
 * \file one_axis_concat_all_align.h
 * \brief one axis concat all align
 */

#ifndef ONE_AXIS_CONCAT_ALL_ALIGN_
#define ONE_AXIS_CONCAT_ALL_ALIGN_

#include "concat_base.h"
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

namespace Concat {
using namespace AscendC;

template <typename T, const bool SAMESHAPE, typename TILINGDATA = ConcatTilingData>
class OneAxisConcatAllAlign {
public:
    __aicore__ inline OneAxisConcatAllAlign(__tiling_data_ptr__ TILINGDATA* tilingData) : tilingData_(tilingData){};
    static __aicore__ inline void PreloadTiling(__tiling_data_ptr__ TILINGDATA* tilingData)
    {
        PreloadMergedInfo(tilingData);
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dst);
    __aicore__ inline void Process();

private:
    __aicore__ inline __gm__ T* GetTensorAddr(uint32_t index, int64_t offset);
    __aicore__ inline void CopyInNoSplitDim1(int64_t srcRowsOffset, int64_t rows, int64_t cols,
                                             LocalTensor<T>& srcLocal);
    __aicore__ inline void CopyOut(int64_t dstOffset, int64_t rows, int64_t cols, LocalTensor<T>& dstLocal);
    __aicore__ inline void ProcessBlockSplitDim0NoSplitDim1();
    __aicore__ inline void SetCopyInparam(int64_t rows, int64_t cols, int64_t srcStride, int64_t dstStride);
    __aicore__ inline void CopyInSplitDim1(int64_t srcRowsOffset, int64_t rows, int64_t cols, SplitInfo& splitInfo,
                                           LocalTensor<T>& srcLocal);
    __aicore__ inline void ProcessBlockSplitDim0SplitDim1();
    __aicore__ inline void ProcessBlockSplitDim0SplitDim1PerLoop(int64_t loopIdxInCols, int64_t cols,
                                                                 int64_t loopSizeDim0, SplitInfo& splitInfo);
    __aicore__ inline void ProcessBlockSplitDim1();
    __aicore__ inline void ProcessBlockSplitDim1PerLoop(int64_t loopIdx, int64_t rows, int64_t cols,
                                                        int64_t blockIdxInCol, SplitInfo& splitInfo, uint16_t bufId);
    __aicore__ inline void CalcSplitInfo(int64_t endIdx, int64_t endOffset, int64_t length, SplitInfo& splitInfo);
    __aicore__ inline int64_t GetTensorDim1(int64_t idx);
    __aicore__ inline void UpdateSplitInfo();

private:
    constexpr static event_t EVT_ID0 = static_cast<event_t>(0);
    constexpr static event_t EVT_ID1 = static_cast<event_t>(1);

    GlobalTensor<T> dstGlobal_;
    GlobalTensor<T> srcGlobal_;
    __tiling_data_ptr__ TILINGDATA* tilingData_;
    uint32_t blockIdx_{0};
    int64_t blockOffset_{0};
    int64_t numPerBlock_{BYTES_PER_BLOCK / sizeof(T)};
    uint32_t colsUsedCoreNum_{1};
    uint32_t rowsUsedCoreNum_{1};

    DataCopyExtParams copyInParam_{0, 0, 0, 0, 0};

    int64_t endTensorIdx_{0};
    int64_t endTensorOffset_{0};
    int64_t startTensorIdx_{0};
    int64_t startTensorOffset_{0};
    ListTensorDesc inputList_;
    uint64_t buf_[ARRAY_SIZE];
    TensorDesc<T> desc_;

    uint32_t bufSize_{0};
    uint32_t bufElems_{0};

    int16_t ubSplitDim1_{0};
    int16_t dim_{0};
    int16_t tensorNum_{0};
    int16_t isNonContiguous_{0};
    int16_t isFP4Type_{0};
    int32_t ubFactorDim0_{0};
    int32_t ubFactorDim1_{0};
    int32_t tailUbFactorDim0_{0};
    int32_t tailUbFactorDim1_{0};
    int64_t blockFactor_{0};
    int64_t tailBlockFactor_{0};
    int64_t uoDim0_{0};
    int64_t uoDim1_{0};
    int64_t catDim1_{0};
    int64_t sameShapeTensorDim1_{0};
};

template <typename T, const bool SAMESHAPE, typename TILINGDATA>
__aicore__ inline void OneAxisConcatAllAlign<T, SAMESHAPE, TILINGDATA>::Init(GM_ADDR x, GM_ADDR dst)
{
    blockIdx_ = GetBlockIdx();
    if constexpr (IsSame<TILINGDATA, ConcatTilingData>::value || IsSame<TILINGDATA, ConcatTilingDataCompact>::value) {
        int64_t colsUsedCoreNum = GetBlockNum() / tilingData_->uoDim0;
        if (blockIdx_ % colsUsedCoreNum != 0) {
            startTensorIdx_ = tilingData_->arrays.mergedInfo[CORE_INFO_BASE + 2 * (blockIdx_ - 1)];
            startTensorOffset_ = tilingData_->arrays.mergedInfo[CORE_INFO_BASE + 2 * (blockIdx_ - 1) + 1];
        }
        endTensorOffset_ = tilingData_->arrays.mergedInfo[CORE_INFO_BASE + 2 * blockIdx_ + 1];
        endTensorIdx_ = tilingData_->arrays.mergedInfo[CORE_INFO_BASE + 2 * blockIdx_];
    }
    if constexpr (IsSame<TILINGDATA, ConcatTilingDataNoArray>::value ||
                  IsSame<TILINGDATA, ConcatTilingDataNoArrayCompact>::value) {
        blockOffset_ = blockIdx_ * tilingData_->blockFactor * tilingData_->ubFactorDim0;
        dstGlobal_.SetGlobalBuffer((__gm__ T*)dst + tilingData_->catDim1 * blockOffset_);
    } else {
        rowsUsedCoreNum_ = tilingData_->uoDim0;
        colsUsedCoreNum_ = GetBlockNum() / rowsUsedCoreNum_;
        blockOffset_ = (blockIdx_ / colsUsedCoreNum_) * tilingData_->ubFactorDim0;
        int64_t colOffset = (blockIdx_ % colsUsedCoreNum_) * tilingData_->blockFactor * tilingData_->ubFactorDim1;
        dstGlobal_.SetGlobalBuffer((__gm__ T*)dst + blockOffset_ * tilingData_->catDim1 + colOffset);
    }

    bufSize_ = static_cast<uint32_t>(tilingData_->ubFactorDim0 * tilingData_->ubFactorDim1 * sizeof(T));
    bufElems_ = static_cast<uint32_t>(tilingData_->ubFactorDim0 * tilingData_->ubFactorDim1);

    ubSplitDim1_ = tilingData_->ubSplitDim1;
    dim_ = tilingData_->dim;
    tensorNum_ = tilingData_->tensorNum;
    isNonContiguous_ = tilingData_->isNonContiguous;
    isFP4Type_ = tilingData_->isFP4Type;
    ubFactorDim0_ = tilingData_->ubFactorDim0;
    ubFactorDim1_ = tilingData_->ubFactorDim1;
    tailUbFactorDim0_ = tilingData_->tailUbFactorDim0;
    tailUbFactorDim1_ = tilingData_->tailUbFactorDim1;
    blockFactor_ = tilingData_->blockFactor;
    tailBlockFactor_ = tilingData_->tailBlockFactor;
    uoDim0_ = tilingData_->uoDim0;
    uoDim1_ = tilingData_->uoDim1;
    catDim1_ = tilingData_->catDim1;
    sameShapeTensorDim1_ = tilingData_->sameShapeTensorDim1;

    inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));
    desc_.SetShapeAddr(&buf_[0]);
    if constexpr (IsSame<TILINGDATA, ConcatTilingData>::value || IsSame<TILINGDATA, ConcatTilingDataCompact>::value) {
        UpdateSplitInfo();
    }
}

template <typename T, const bool SAMESHAPE, typename TILINGDATA>
__aicore__ inline void OneAxisConcatAllAlign<T, SAMESHAPE, TILINGDATA>::UpdateSplitInfo()
{
    if (startTensorOffset_ == GetTensorDim1(startTensorIdx_)) {
        startTensorIdx_ += 1;
        startTensorOffset_ = 0;
    }
}

template <typename T, const bool SAMESHAPE, typename TILINGDATA>
__aicore__ inline __gm__ T* OneAxisConcatAllAlign<T, SAMESHAPE, TILINGDATA>::GetTensorAddr(uint32_t index,
                                                                                           int64_t offset)
{
    return inputList_.GetDataPtr<T>(index) + offset;
}

template <typename T, const bool SAMESHAPE, typename TILINGDATA>
__aicore__ inline void OneAxisConcatAllAlign<T, SAMESHAPE, TILINGDATA>::Process()
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

template <typename T, const bool SAMESHAPE, typename TILINGDATA>
__aicore__ inline void OneAxisConcatAllAlign<T, SAMESHAPE, TILINGDATA>::ProcessBlockSplitDim0NoSplitDim1()
{
    int64_t loopSize = blockFactor_;
    if (blockIdx_ == GetBlockNum() - 1) {
        loopSize = tailBlockFactor_ - 1;
    }
    SetFlag<HardEvent::MTE3_MTE2>(EVT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVT_ID1);
    uint16_t bufId = 0;
    for (int64_t i = 0; i < loopSize; i++) {
        event_t evt = (bufId == 0) ? EVT_ID0 : EVT_ID1;
        WaitFlag<HardEvent::MTE3_MTE2>(evt);
        LocalTensor<T> srcLocal(TPosition::VECIN, bufId * bufSize_, bufElems_);
        CopyInNoSplitDim1(i * ubFactorDim0_, ubFactorDim0_, ubFactorDim1_, srcLocal);
        SetFlag<HardEvent::MTE2_MTE3>(evt);
        WaitFlag<HardEvent::MTE2_MTE3>(evt);
        CopyOut(i * ubFactorDim0_ * catDim1_, ubFactorDim0_, ubFactorDim1_, srcLocal);
        SetFlag<HardEvent::MTE3_MTE2>(evt);
        bufId ^= 1;
    }
    if (blockIdx_ == GetBlockNum() - 1) {
        event_t evt = (bufId == 0) ? EVT_ID0 : EVT_ID1;
        WaitFlag<HardEvent::MTE3_MTE2>(evt);
        LocalTensor<T> srcLocal(TPosition::VECIN, bufId * bufSize_, bufElems_);
        CopyInNoSplitDim1(loopSize * ubFactorDim0_, tailUbFactorDim0_, ubFactorDim1_, srcLocal);
        SetFlag<HardEvent::MTE2_MTE3>(evt);
        WaitFlag<HardEvent::MTE2_MTE3>(evt);
        CopyOut(loopSize * ubFactorDim0_ * catDim1_, tailUbFactorDim0_, ubFactorDim1_, srcLocal);
        SetFlag<HardEvent::MTE3_MTE2>(evt);
        bufId ^= 1;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVT_ID1);
}

template <typename T, const bool SAMESHAPE, typename TILINGDATA>
__aicore__ inline void OneAxisConcatAllAlign<T, SAMESHAPE, TILINGDATA>::ProcessBlockSplitDim0SplitDim1()
{
    int64_t loopSizeDim0 = blockFactor_;
    if (blockIdx_ == GetBlockNum() - 1) {
        loopSizeDim0 = tailBlockFactor_ - 1;
    }
    SplitInfo splitInfo = {0, 0, 0, 0};
    int64_t lastTensorDim1 = GetTensorDim1(tensorNum_ - 1);
    for (int64_t i = 0; i < uoDim1_ - 1; i++) {
        CalcSplitInfo(tensorNum_ - 1, lastTensorDim1, ubFactorDim1_, splitInfo);
        ProcessBlockSplitDim0SplitDim1PerLoop(i, ubFactorDim1_, loopSizeDim0, splitInfo);
    }

    CalcSplitInfo(tensorNum_ - 1, lastTensorDim1, tailUbFactorDim1_, splitInfo);
    ProcessBlockSplitDim0SplitDim1PerLoop(uoDim1_ - 1, tailUbFactorDim1_, loopSizeDim0, splitInfo);
}

template <typename T, const bool SAMESHAPE, typename TILINGDATA>
__aicore__ inline void OneAxisConcatAllAlign<T, SAMESHAPE, TILINGDATA>::ProcessBlockSplitDim1()
{
    int64_t loopSizeDim1 = blockFactor_ - 1;
    int64_t blockIdxInCol = blockIdx_ % colsUsedCoreNum_;
    int64_t blockIdxInRow = blockIdx_ / colsUsedCoreNum_;
    int64_t tailLoopCopyColsNum = ubFactorDim1_;
    int64_t copyRowsNum = ubFactorDim0_;

    if (blockIdxInCol == colsUsedCoreNum_ - 1) {
        loopSizeDim1 = tailBlockFactor_ - 1;
        tailLoopCopyColsNum = tailUbFactorDim1_;
    }
    if (blockIdxInRow == rowsUsedCoreNum_ - 1) {
        copyRowsNum = tailUbFactorDim0_;
    }

    SetFlag<HardEvent::MTE3_MTE2>(EVT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVT_ID1);
    SplitInfo splitInfo = {0, 0, startTensorIdx_, startTensorOffset_};
    uint16_t bufId = 0;
    for (int64_t i = 0; i < loopSizeDim1; i++) {
        ProcessBlockSplitDim1PerLoop(i, copyRowsNum, ubFactorDim1_, blockIdxInCol, splitInfo, bufId);
        bufId ^= 1;
    }

    ProcessBlockSplitDim1PerLoop(loopSizeDim1, copyRowsNum, tailLoopCopyColsNum, blockIdxInCol, splitInfo, bufId);
    WaitFlag<HardEvent::MTE3_MTE2>(EVT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVT_ID1);
}

template <typename T, const bool SAMESHAPE, typename TILINGDATA>
__aicore__ inline void OneAxisConcatAllAlign<T, SAMESHAPE, TILINGDATA>::CalcSplitInfo(int64_t endIdx, int64_t endOffset,
                                                                                      int64_t length,
                                                                                      SplitInfo& splitInfo)
{
    splitInfo.startIdx = splitInfo.endIdx;
    splitInfo.startOffset = splitInfo.endOffset;

    if (splitInfo.startOffset == GetTensorDim1(splitInfo.startIdx)) {
        splitInfo.startIdx += 1;
        splitInfo.startOffset = 0;
    }

    if (splitInfo.startIdx == endIdx || splitInfo.startOffset + length <= GetTensorDim1(splitInfo.startIdx)) {
        splitInfo.endIdx = splitInfo.startIdx;
        splitInfo.endOffset = splitInfo.startOffset + length;
        return;
    }

    int64_t curOffset = GetTensorDim1(splitInfo.startIdx) - splitInfo.startOffset;

    for (int64_t i = splitInfo.startIdx + 1; i <= endIdx; i++) {
        curOffset += GetTensorDim1(i);
        if (curOffset >= length) {
            splitInfo.endIdx = i;
            splitInfo.endOffset = GetTensorDim1(i) + length - curOffset;
            return;
        }
    }
    splitInfo.endIdx = endIdx;
    splitInfo.endOffset = endOffset;
}

template <typename T, const bool SAMESHAPE, typename TILINGDATA>
__aicore__ inline void OneAxisConcatAllAlign<T, SAMESHAPE, TILINGDATA>::ProcessBlockSplitDim0SplitDim1PerLoop(
    int64_t loopIdxInCols, int64_t copyColsNum, int64_t loopSizeDim0, SplitInfo& splitInfo)
{
    SetFlag<HardEvent::MTE3_MTE2>(EVT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVT_ID1);
    uint16_t bufId = 0;
    for (int64_t i = 0; i < loopSizeDim0; i++) {
        event_t evt = (bufId == 0) ? EVT_ID0 : EVT_ID1;
        WaitFlag<HardEvent::MTE3_MTE2>(evt);
        LocalTensor<T> srcLocal(TPosition::VECIN, bufId * bufSize_, bufElems_);
        CopyInSplitDim1(i * ubFactorDim0_, ubFactorDim0_, copyColsNum, splitInfo, srcLocal);
        SetFlag<HardEvent::MTE2_MTE3>(evt);
        WaitFlag<HardEvent::MTE2_MTE3>(evt);
        CopyOut(i * ubFactorDim0_ * catDim1_ + loopIdxInCols * ubFactorDim1_, ubFactorDim0_, copyColsNum, srcLocal);
        SetFlag<HardEvent::MTE3_MTE2>(evt);
        bufId ^= 1;
    }
    if (blockIdx_ == GetBlockNum() - 1) {
        event_t evt = (bufId == 0) ? EVT_ID0 : EVT_ID1;
        WaitFlag<HardEvent::MTE3_MTE2>(evt);
        LocalTensor<T> srcLocal(TPosition::VECIN, bufId * bufSize_, bufElems_);
        CopyInSplitDim1(loopSizeDim0 * ubFactorDim0_, tailUbFactorDim0_, copyColsNum, splitInfo, srcLocal);
        SetFlag<HardEvent::MTE2_MTE3>(evt);
        WaitFlag<HardEvent::MTE2_MTE3>(evt);
        CopyOut(loopSizeDim0 * ubFactorDim0_ * catDim1_ + loopIdxInCols * ubFactorDim1_, tailUbFactorDim0_, copyColsNum,
                srcLocal);
        SetFlag<HardEvent::MTE3_MTE2>(evt);
        bufId ^= 1;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVT_ID1);
}

template <typename T, const bool SAMESHAPE, typename TILINGDATA>
__aicore__ inline void OneAxisConcatAllAlign<T, SAMESHAPE, TILINGDATA>::ProcessBlockSplitDim1PerLoop(
    int64_t loopIdx, int64_t rows, int64_t cols, int64_t blockIdxInCol, SplitInfo& splitInfo, uint16_t bufId)
{
    event_t evt = (bufId == 0) ? EVT_ID0 : EVT_ID1;
    WaitFlag<HardEvent::MTE3_MTE2>(evt);
    LocalTensor<T> srcLocal(TPosition::VECIN, bufId * bufSize_, bufElems_);
    CalcSplitInfo(endTensorIdx_, endTensorOffset_, cols, splitInfo);
    CopyInSplitDim1(0, rows, cols, splitInfo, srcLocal);
    SetFlag<HardEvent::MTE2_MTE3>(evt);
    WaitFlag<HardEvent::MTE2_MTE3>(evt);
    CopyOut(loopIdx * ubFactorDim1_, rows, cols, srcLocal);
    SetFlag<HardEvent::MTE3_MTE2>(evt);
}

template <typename T, const bool SAMESHAPE, typename TILINGDATA>
__aicore__ inline void OneAxisConcatAllAlign<T, SAMESHAPE, TILINGDATA>::CopyInNoSplitDim1(int64_t srcRowsOffset,
                                                                                          int64_t rows, int64_t cols,
                                                                                          LocalTensor<T>& srcLocal)
{
    int64_t curDim1Offset = 0;
    DataCopyPadExtParams<T> padParam{false, 0, 0, 0};
    for (int64_t i = 0; i < tensorNum_; i++) {
        int64_t dim1 = GetTensorDim1(i);
        int64_t dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, i, dim1);
        srcGlobal_.SetGlobalBuffer(GetTensorAddr(i, blockOffset_ * dim0stride));
        SetCopyInparam(rows, dim1, dim0stride - dim1, (cols - dim1) / numPerBlock_);
#if defined(__DAV_920R1__) || defined(__DAV_920R2__)
        DataCopyPad(srcLocal[curDim1Offset], srcGlobal_[srcRowsOffset * dim0stride], copyInParam_, padParam, true);
#else
        DataCopyPad(srcLocal[curDim1Offset], srcGlobal_[srcRowsOffset * dim0stride], copyInParam_, padParam);
#endif
        curDim1Offset += dim1;
    }
}

template <typename T, const bool SAMESHAPE, typename TILINGDATA>
__aicore__ inline void OneAxisConcatAllAlign<T, SAMESHAPE, TILINGDATA>::CopyInSplitDim1(int64_t srcRowsOffset,
                                                                                        int64_t rows, int64_t cols,
                                                                                        SplitInfo& splitInfo,
                                                                                        LocalTensor<T>& srcLocal)
{
    int64_t curDim1Offset = 0;

    int64_t dim1 = GetTensorDim1(splitInfo.startIdx);
    int64_t dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, splitInfo.startIdx, dim1);
    int64_t copyColsNum = 0;
    DataCopyPadExtParams<T> padParam{false, 0, 0, 0};
    if (splitInfo.startIdx == splitInfo.endIdx) {
        copyColsNum = splitInfo.endOffset - splitInfo.startOffset;
    } else {
        copyColsNum = dim1 - splitInfo.startOffset;
    }
    srcGlobal_.SetGlobalBuffer(GetTensorAddr(splitInfo.startIdx, blockOffset_ * dim0stride + splitInfo.startOffset));
    SetCopyInparam(rows, copyColsNum, dim0stride - copyColsNum, (cols - copyColsNum) / numPerBlock_);
#if defined(__DAV_920R1__) || defined(__DAV_920R2__)
    DataCopyPad(srcLocal[curDim1Offset], srcGlobal_[srcRowsOffset * dim0stride], copyInParam_, padParam, true);
#else
    DataCopyPad(srcLocal[curDim1Offset], srcGlobal_[srcRowsOffset * dim0stride], copyInParam_, padParam);
#endif
    curDim1Offset += copyColsNum;

    for (int64_t i = splitInfo.startIdx + 1; i < splitInfo.endIdx; i++) {
        dim1 = GetTensorDim1(i);
        dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, i, dim1);
        srcGlobal_.SetGlobalBuffer(GetTensorAddr(i, blockOffset_ * dim0stride));
        SetCopyInparam(rows, dim1, dim0stride - dim1, (cols - dim1) / numPerBlock_);
#if defined(__DAV_920R1__) || defined(__DAV_920R2__)
        DataCopyPad(srcLocal[curDim1Offset], srcGlobal_[srcRowsOffset * dim0stride], copyInParam_, padParam, true);
#else
        DataCopyPad(srcLocal[curDim1Offset], srcGlobal_[srcRowsOffset * dim0stride], copyInParam_, padParam);
#endif
        curDim1Offset += dim1;
    }

    if (splitInfo.startIdx != splitInfo.endIdx) {
        dim1 = GetTensorDim1(splitInfo.endIdx);
        dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, splitInfo.endIdx, dim1);
        copyColsNum = splitInfo.endOffset;
        srcGlobal_.SetGlobalBuffer(GetTensorAddr(splitInfo.endIdx, blockOffset_ * dim0stride));
        SetCopyInparam(rows, copyColsNum, dim0stride - copyColsNum, (cols - copyColsNum) / numPerBlock_);
#if defined(__DAV_920R1__) || defined(__DAV_920R2__)
        DataCopyPad(srcLocal[curDim1Offset], srcGlobal_[srcRowsOffset * dim0stride], copyInParam_, padParam, true);
#else
        DataCopyPad(srcLocal[curDim1Offset], srcGlobal_[srcRowsOffset * dim0stride], copyInParam_, padParam);
#endif
    }
}

template <typename T, const bool SAMESHAPE, typename TILINGDATA>
__aicore__ inline void OneAxisConcatAllAlign<T, SAMESHAPE, TILINGDATA>::SetCopyInparam(int64_t rows, int64_t cols,
                                                                                       int64_t srcStride,
                                                                                       int64_t dstStride)
{
    copyInParam_.blockCount = rows;
    copyInParam_.blockLen = cols * sizeof(T);
    copyInParam_.srcStride = srcStride * sizeof(T);
    copyInParam_.dstStride = dstStride;
}

template <typename T, const bool SAMESHAPE, typename TILINGDATA>
__aicore__ inline void OneAxisConcatAllAlign<T, SAMESHAPE, TILINGDATA>::CopyOut(int64_t dstOffset, int64_t rows,
                                                                                int64_t cols, LocalTensor<T>& dstLocal)
{
    DataCopyExtParams copyOutParam{0, 0, 0, 0, 0};
    copyOutParam.blockCount = rows;
    copyOutParam.blockLen = cols * sizeof(T);
    copyOutParam.dstStride = (catDim1_ - cols) * sizeof(T);
    copyOutParam.srcStride = 0;
    DataCopyPad(dstGlobal_[dstOffset], dstLocal, copyOutParam);
}

template <typename T, const bool SAMESHAPE, typename TILINGDATA>
__aicore__ inline int64_t OneAxisConcatAllAlign<T, SAMESHAPE, TILINGDATA>::GetTensorDim1(int64_t idx)
{
    if constexpr (SAMESHAPE) {
        return sameShapeTensorDim1_;
    } else {
        if (idx < PRELOAD_DIM1_SIZE) {
            if constexpr (IsSame<TILINGDATA, ConcatTilingData>::value ||
                          IsSame<TILINGDATA, ConcatTilingDataNoArray>::value) {
                return tilingData_->preLoadDim1[idx];
            } else {
                return tilingData_->arrays.preLoadDim1[idx];
            }
        }
        int64_t dim1 = GetNonConDimSize<TILINGDATA, T>(tilingData_, idx, inputList_, desc_) * sameShapeTensorDim1_;
        if (isFP4Type_) {
            dim1 /= 2;
        }
        return dim1;
    }
}
} // namespace Concat

#endif // ONE_AXIS_CONCAT_ALL_ALIGN_
