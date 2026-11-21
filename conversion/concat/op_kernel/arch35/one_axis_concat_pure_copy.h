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
 * \file one_axis_concat_pure_copy.h
 * \brief one axis concat pure copy
 */

#ifndef ONE_AXIS_CONCAT_PURE_COPY_
#define ONE_AXIS_CONCAT_PURE_COPY_

#include "concat_base.h"
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

namespace Concat {
using namespace AscendC;

template <typename TILINGDATA = ConcatTilingData>
class OneAxisConcatPureCopy {
public:
    __aicore__ inline OneAxisConcatPureCopy(__tiling_data_ptr__ TILINGDATA* tilingData) : tilingData_(tilingData){};
    static __aicore__ inline void PreloadTiling(__tiling_data_ptr__ TILINGDATA* tilingData)
    {
        PreloadMergedInfo(tilingData);
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dst);
    __aicore__ inline void Process();

private:
    __aicore__ inline __gm__ int8_t* GetTensorAddr(uint32_t index, int64_t offset);
    __aicore__ inline void CopyInSingleTensor(const GlobalTensor<int8_t> srcGm, int64_t copyRows, int64_t copyCols,
                                              int64_t srcOffset, int64_t tensorDim1, LocalTensor<int8_t>& srcLocal);
    __aicore__ inline void CopyOutSingleTensor(int64_t copyRows, int64_t copyCols, int64_t dstOffset,
                                               LocalTensor<int8_t>& dstLocal);
    __aicore__ inline void ProcessSingleTensor(int64_t tensorIdx, int64_t tensorDim1, int64_t totalRows,
                                               int64_t totalCols, int64_t globalSrcOffset, int64_t globalDstOffset);
    __aicore__ inline void ProcessNoSplitDim1();
    __aicore__ inline void SetCopyInparam(int64_t rows, int64_t cols, int64_t srcStride, int64_t dstStride);
    __aicore__ inline void ProcessSplitDim1();
    __aicore__ inline int64_t GetTensorDim1(int64_t idx);
    __aicore__ inline void UpdateSplitInfo();

private:
    constexpr static event_t EVT_ID0 = static_cast<event_t>(0);
    constexpr static event_t EVT_ID1 = static_cast<event_t>(1);

    GlobalTensor<int8_t> dstGlobal_;
    GlobalTensor<int8_t> srcGlobal_;
    __tiling_data_ptr__ TILINGDATA* tilingData_;
    uint32_t blockIdx_{0};
    int64_t blockOffset_{0};

    uint32_t colsUsedCoreNum_{1};
    uint32_t rowsUsedCoreNum_{1};
    uint32_t blockIdxInCol_{0};
    uint32_t blockIdxInRow_{0};

    DataCopyExtParams copyInParam_{0, 0, 0, 0, 0};

    int64_t endTensorIdx_{0};
    int64_t endTensorOffset_{0};
    int64_t startTensorIdx_{0};
    int64_t startTensorOffset_{0};
    ListTensorDesc inputList_;
    uint64_t buf_[ARRAY_SIZE];
    TensorDesc<int8_t> desc_;

    uint32_t bufSize_{0};

    int16_t tensorNum_{0};
    int16_t isNonContiguous_{0};
    int16_t isFP4Type_{0};
    int32_t ubFactorDim0_{0};
    int32_t tailUbFactorDim0_{0};
    int32_t bufferSize_{0};
    int32_t dtypeSize_{0};
    int64_t uoDim0_{0};
    int64_t uoDim1_{0};
    int64_t catDim1_{0};
    int64_t sameShapeTensorDim1_{0};
};

template <typename TILINGDATA>
__aicore__ inline void OneAxisConcatPureCopy<TILINGDATA>::Init(GM_ADDR x, GM_ADDR dst)
{
    blockIdx_ = GetBlockIdx();
    if constexpr (IsSame<TILINGDATA, ConcatTilingData>::value || IsSame<TILINGDATA, ConcatTilingDataCompact>::value) {
        rowsUsedCoreNum_ = tilingData_->uoDim0;
        colsUsedCoreNum_ = tilingData_->uoDim1;
        blockIdxInCol_ = blockIdx_ % colsUsedCoreNum_;
        blockIdxInRow_ = blockIdx_ / colsUsedCoreNum_;
        if (blockIdxInCol_ != 0) {
            startTensorIdx_ = tilingData_->arrays.mergedInfo[CORE_INFO_BASE + 2 * (blockIdx_ - 1)];
            startTensorOffset_ = tilingData_->arrays.mergedInfo[CORE_INFO_BASE + 2 * (blockIdx_ - 1) + 1];
        }
        endTensorOffset_ = tilingData_->arrays.mergedInfo[CORE_INFO_BASE + 2 * blockIdx_ + 1];
        endTensorIdx_ = tilingData_->arrays.mergedInfo[CORE_INFO_BASE + 2 * blockIdx_];
    }
    if constexpr (IsSame<TILINGDATA, ConcatTilingDataNoArray>::value ||
                  IsSame<TILINGDATA, ConcatTilingDataNoArrayCompact>::value) {
        blockOffset_ = static_cast<int64_t>(blockIdx_) * tilingData_->ubFactorDim0;
        dstGlobal_.SetGlobalBuffer((__gm__ int8_t*)dst +
                                   tilingData_->catDim1 * tilingData_->dtypeSize * static_cast<int64_t>(blockOffset_));
    } else {
        blockOffset_ = blockIdxInRow_ * tilingData_->ubFactorDim0;
        int64_t colOffset = static_cast<int64_t>(blockIdxInCol_) * tilingData_->ubFactorDim1 * tilingData_->dtypeSize;
        dstGlobal_.SetGlobalBuffer((__gm__ int8_t*)dst + blockOffset_ * tilingData_->catDim1 * tilingData_->dtypeSize +
                                   colOffset);
    }

    bufSize_ = static_cast<uint32_t>(tilingData_->bufferSize * tilingData_->dtypeSize);

    tensorNum_ = tilingData_->tensorNum;
    isNonContiguous_ = tilingData_->isNonContiguous;
    isFP4Type_ = tilingData_->isFP4Type;
    ubFactorDim0_ = tilingData_->ubFactorDim0;
    tailUbFactorDim0_ = tilingData_->tailUbFactorDim0;
    bufferSize_ = tilingData_->bufferSize;
    dtypeSize_ = tilingData_->dtypeSize;
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

template <typename TILINGDATA>
__aicore__ inline void OneAxisConcatPureCopy<TILINGDATA>::UpdateSplitInfo()
{
    if (startTensorOffset_ == GetTensorDim1(startTensorIdx_)) {
        startTensorIdx_ += 1;
        startTensorOffset_ = 0;
    }
}

template <typename TILINGDATA>
__aicore__ inline __gm__ int8_t* OneAxisConcatPureCopy<TILINGDATA>::GetTensorAddr(uint32_t index, int64_t offset)
{
    return inputList_.GetDataPtr<int8_t>(index) + offset;
}

template <typename TILINGDATA>
__aicore__ inline void OneAxisConcatPureCopy<TILINGDATA>::Process()
{
    if (blockIdx_ >= GetBlockNum()) {
        return;
    }
    if constexpr (IsSame<TILINGDATA, ConcatTilingData>::value || IsSame<TILINGDATA, ConcatTilingDataCompact>::value) {
        ProcessSplitDim1();
    } else {
        ProcessNoSplitDim1();
    }
}

template <typename TILINGDATA>
__aicore__ inline void OneAxisConcatPureCopy<TILINGDATA>::ProcessNoSplitDim1()
{
    int64_t totalRows = ubFactorDim0_;
    if (blockIdx_ == GetBlockNum() - 1) {
        totalRows = tailUbFactorDim0_;
    }
    int64_t totalColOffset = 0;
    for (int64_t i = 0; i < tensorNum_; i++) {
        int64_t dim1 = GetTensorDim1(i);
        int64_t dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, i, dim1);
        int64_t globalSrcOffset = blockOffset_ * dim0stride;
        ProcessSingleTensor(i, dim0stride, totalRows, dim1, globalSrcOffset, totalColOffset);
        totalColOffset += dim1;
    }
}

template <typename TILINGDATA>
__aicore__ inline void OneAxisConcatPureCopy<TILINGDATA>::ProcessSplitDim1()
{
    int64_t totalRows = ubFactorDim0_;
    if (blockIdxInRow_ == rowsUsedCoreNum_ - 1) {
        totalRows = tailUbFactorDim0_;
    }
    int64_t totalColOffset = 0;
    int64_t globalSrcOffset = 0;
    int64_t tensorDim1 = GetTensorDim1(startTensorIdx_);
    int64_t dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, startTensorIdx_, tensorDim1);
    globalSrcOffset = blockOffset_ * dim0stride + startTensorOffset_;
    if (startTensorIdx_ == endTensorIdx_) {
        ProcessSingleTensor(startTensorIdx_, dim0stride, totalRows, endTensorOffset_ - startTensorOffset_,
                            globalSrcOffset, totalColOffset);
        return;
    }
    ProcessSingleTensor(startTensorIdx_, dim0stride, totalRows, tensorDim1 - startTensorOffset_, globalSrcOffset,
                        totalColOffset);
    totalColOffset += tensorDim1 - startTensorOffset_;
    for (int64_t i = startTensorIdx_ + 1; i < endTensorIdx_; i++) {
        tensorDim1 = GetTensorDim1(i);
        dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, i, tensorDim1);
        globalSrcOffset = blockOffset_ * dim0stride;
        ProcessSingleTensor(i, dim0stride, totalRows, tensorDim1, globalSrcOffset, totalColOffset);
        totalColOffset += tensorDim1;
    }
    tensorDim1 = GetTensorDim1(endTensorIdx_);
    dim0stride = GetTensorDim0Stride<TILINGDATA>(tilingData_, endTensorIdx_, tensorDim1);
    globalSrcOffset = blockOffset_ * dim0stride;
    ProcessSingleTensor(endTensorIdx_, dim0stride, totalRows, endTensorOffset_, globalSrcOffset, totalColOffset);
}

template <typename TILINGDATA>
__aicore__ inline void OneAxisConcatPureCopy<TILINGDATA>::ProcessSingleTensor(int64_t tensorIdx, int64_t tensorDim1,
                                                                              int64_t totalRows, int64_t totalCols,
                                                                              int64_t globalSrcOffset,
                                                                              int64_t globalDstOffset)
{
    if (totalRows == 0 || totalCols == 0) {
        return;
    }
    srcGlobal_.SetGlobalBuffer(GetTensorAddr(tensorIdx, globalSrcOffset * dtypeSize_));
    int64_t colFactor = min(totalCols, static_cast<int64_t>(bufferSize_));
    int64_t rowFactor = bufferSize_ / colFactor;
    if (rowFactor > UINT16_MAX) {
        rowFactor = UINT16_MAX; // blockCount 为 uint16
    }
    int64_t colLoopSize = (totalCols + colFactor - 1) / colFactor;
    int64_t rowLoopSize = (totalRows + rowFactor - 1) / rowFactor;
    int64_t tailColFactor = totalCols - (colLoopSize - 1) * colFactor;
    int64_t tailRowFactor = totalRows - (rowLoopSize - 1) * rowFactor;
    int64_t srcOffset = 0;
    int64_t dstOffset = 0;
    SetFlag<HardEvent::MTE3_MTE2>(EVT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVT_ID1);
    uint16_t bufId = 0;
    for (int64_t m = 0; m < rowLoopSize; m++) {
        int64_t copyRows = m == rowLoopSize - 1 ? tailRowFactor : rowFactor;
        dstOffset = globalDstOffset + m * rowFactor * catDim1_;
        for (int64_t n = 0; n < colLoopSize - 1; n++) {
            srcOffset = m * rowFactor * tensorDim1 + n * colFactor;
            event_t evt = (bufId == 0) ? EVT_ID0 : EVT_ID1;
            WaitFlag<HardEvent::MTE3_MTE2>(evt);
            LocalTensor<int8_t> srcLocal(TPosition::VECIN, bufId * bufSize_, static_cast<uint32_t>(bufferSize_));
            CopyInSingleTensor(srcGlobal_, copyRows, colFactor, srcOffset, tensorDim1, srcLocal);
            SetFlag<HardEvent::MTE2_MTE3>(evt);
            WaitFlag<HardEvent::MTE2_MTE3>(evt);
            CopyOutSingleTensor(copyRows, colFactor, dstOffset, srcLocal);
            SetFlag<HardEvent::MTE3_MTE2>(evt);
            bufId ^= 1;
            dstOffset += colFactor;
        }
        srcOffset = m * rowFactor * tensorDim1 + (colLoopSize - 1) * colFactor;
        event_t evt = (bufId == 0) ? EVT_ID0 : EVT_ID1;
        WaitFlag<HardEvent::MTE3_MTE2>(evt);
        LocalTensor<int8_t> srcLocal(TPosition::VECIN, bufId * bufSize_, static_cast<uint32_t>(bufferSize_));
        CopyInSingleTensor(srcGlobal_, copyRows, tailColFactor, srcOffset, tensorDim1, srcLocal);
        SetFlag<HardEvent::MTE2_MTE3>(evt);
        WaitFlag<HardEvent::MTE2_MTE3>(evt);
        CopyOutSingleTensor(copyRows, tailColFactor, dstOffset, srcLocal);
        SetFlag<HardEvent::MTE3_MTE2>(evt);
        bufId ^= 1;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVT_ID1);
}

template <typename TILINGDATA>
__aicore__ inline void OneAxisConcatPureCopy<TILINGDATA>::CopyInSingleTensor(const GlobalTensor<int8_t> srcGm,
                                                                             int64_t copyRows, int64_t copyCols,
                                                                             int64_t srcOffset, int64_t tensorDim1,
                                                                             LocalTensor<int8_t>& srcLocal)
{
    DataCopyPadExtParams<int8_t> padParam{false, 0, 0, 0};
    SetCopyInparam(copyRows, copyCols, tensorDim1 - copyCols, 0);
#if defined(__DAV_920R1__) || defined(__DAV_920R2__)
    DataCopyPad<int8_t, PaddingMode::Compact>(srcLocal, srcGm[srcOffset * dtypeSize_], copyInParam_, padParam, true);

#else
    DataCopyPad<int8_t, PaddingMode::Compact>(srcLocal, srcGm[srcOffset * dtypeSize_], copyInParam_, padParam);
#endif
}

template <typename TILINGDATA>
__aicore__ inline void OneAxisConcatPureCopy<TILINGDATA>::CopyOutSingleTensor(int64_t copyRows, int64_t copyCols,
                                                                              int64_t dstOffset,
                                                                              LocalTensor<int8_t>& dstLocal)
{
    DataCopyExtParams copyOutParam{0, 0, 0, 0, 0};
    copyOutParam.blockCount = copyRows;
    copyOutParam.blockLen = copyCols * dtypeSize_;
    copyOutParam.dstStride = (catDim1_ - copyCols) * dtypeSize_;
    copyOutParam.srcStride = 0;
    DataCopyPad<int8_t, PaddingMode::Compact>(dstGlobal_[dstOffset * dtypeSize_], dstLocal, copyOutParam);
}

template <typename TILINGDATA>
__aicore__ inline void OneAxisConcatPureCopy<TILINGDATA>::SetCopyInparam(int64_t rows, int64_t cols, int64_t srcStride,
                                                                         int64_t dstStride)
{
    copyInParam_.blockCount = rows;
    copyInParam_.blockLen = cols * dtypeSize_;
    copyInParam_.srcStride = srcStride * dtypeSize_;
    copyInParam_.dstStride = dstStride;
}

template <typename TILINGDATA>
__aicore__ inline int64_t OneAxisConcatPureCopy<TILINGDATA>::GetTensorDim1(int64_t idx)
{
    if (idx < PRELOAD_DIM1_SIZE) {
        if constexpr (IsSame<TILINGDATA, ConcatTilingData>::value ||
                      IsSame<TILINGDATA, ConcatTilingDataNoArray>::value) {
            return tilingData_->preLoadDim1[idx];
        } else {
            return tilingData_->arrays.preLoadDim1[idx];
        }
    }
    int64_t dim1 = GetNonConDimSize<TILINGDATA, int8_t>(tilingData_, idx, inputList_, desc_) * sameShapeTensorDim1_;
    if (isFP4Type_) {
        dim1 /= 2;
    }
    return dim1;
}
} // namespace Concat

#endif // ONE_AXIS_CONCAT_PURE_COPY_
