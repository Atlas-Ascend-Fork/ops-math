/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file broadcast_to_with_dma.h
 * \brief kernel of broadcast_to with nddma (静态Tensor编程 + Mutex同步)
 */

#ifndef BROADCAST_TO_WITH_NDDMA_H_
#define BROADCAST_TO_WITH_NDDMA_H_

#include "broadcast_to_base.h"
#include "kernel_operator.h"

namespace BrcTo {
using namespace AscendC;

constexpr NdDmaConfig copyCfg{false, 0, 0, false};
constexpr MutexID NDDMA_MUTEX_ID = 0; // 单buffer: MTE2↔MTE3同步

template <typename T, typename U, uint8_t maxDim = 4>
class BrcToWithNDDMA : public BrcToBase<U> {
public:
    __aicore__ inline BrcToWithNDDMA(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const U* tilingDataPtr);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SetDMAParams();
    __aicore__ inline void CalcOutSize();
    __aicore__ inline void CopyDataInWithDMA();
    __aicore__ inline void CopyDataOut();

private:
    const U* tdPtr_;
    int64_t blockIdx = 0;
    AxesLpInfo lpInfo;
    int64_t aBaseIdx = 0;
    int64_t bBaseIdx = 0;
    GlobalTensor<T> inGM;
    GlobalTensor<T> outGM;
    LocalTensor<T> ubBuf_;
    int64_t gmInOffset;
    int64_t gmOutOffset;
    int64_t inBlockOffset = 0;
    int64_t outBlockOffset = 0;
    AscendC::NdDmaLoopInfo<maxDim> copyLpInfo;
    AscendC::NdDmaParams<T, maxDim> mCopyParams;
    AscendC::DataCopyExtParams copyParams{1, 0, 0, 0, 0};
    uint32_t outLen = sizeof(T);
    uint8_t copySwitch = 1; // to avoid repeat copy in for U is broadcast axis
};

template <typename T, typename U, uint8_t maxDim>
__aicore__ inline void BrcToWithNDDMA<T, U, maxDim>::Init(GM_ADDR x, GM_ADDR y, const U* tilingDataPtr)
{
    inGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
    tdPtr_ = tilingDataPtr;

    // 静态Tensor: 每核按核编号偏移UB地址
    int64_t realBlockIdx = GetBlockIdx();
    blockIdx = (tdPtr_->doubleMode != 0) ? realBlockIdx % tdPtr_->usedCoreCnt : realBlockIdx;
    ubBuf_ = LocalTensor<T>(TPosition::VECCALC, 0, tdPtr_->tensorSize);
}

template <typename T, typename U, uint8_t maxDim>
__aicore__ inline void BrcToWithNDDMA<T, U, maxDim>::SetDMAParams()
{
    for (uint8_t i = 0; i < maxDim; i++) {
        copyLpInfo.loopSrcStride[i] = tdPtr_->xSrcStride[i];
        copyLpInfo.loopDstStride[i] = tdPtr_->xDstStride[i];
        copyLpInfo.loopSize[i] = tdPtr_->xSize[i];
    }
    mCopyParams.loopInfo = copyLpInfo;
}

template <typename T, typename U, uint8_t maxDim>
__aicore__ inline void BrcToWithNDDMA<T, U, maxDim>::CopyDataInWithDMA()
{
    if (copySwitch > 0) {
        // 获取buffer(等待MTE3释放上一笔), 搬入后释放给MTE3
        Mutex::Lock<PIPE_MTE2>(NDDMA_MUTEX_ID);
        DataCopy<T, maxDim, copyCfg>(ubBuf_, inGM[gmInOffset], mCopyParams);
        Mutex::Unlock<PIPE_MTE2>(NDDMA_MUTEX_ID);
        copySwitch = copySwitch * uint8_t(tdPtr_->isUNotB);
    }
}

template <typename T, typename U, uint8_t maxDim>
__aicore__ inline void BrcToWithNDDMA<T, U, maxDim>::CalcOutSize()
{
    for (uint8_t i = 1; i < maxDim; i++) {
        outLen *= tdPtr_->xSize[i];
    }
}

template <typename T, typename U, uint8_t maxDim>
__aicore__ inline void BrcToWithNDDMA<T, U, maxDim>::CopyDataOut()
{
    int64_t bOffset = 0;
    // 获取buffer(等待MTE2写入完成), 搬出后释放给下一笔MTE2复用
    Mutex::Lock<PIPE_MTE3>(NDDMA_MUTEX_ID);
    for (int64_t bIdx = lpInfo.bLpBegIdx; bIdx < lpInfo.bLpEndIdx; bIdx++) {
        bOffset = this->CalcBAxesOffset(tdPtr_, bBaseIdx + bIdx);
        DataCopyPad(outGM[gmOutOffset + bOffset], ubBuf_, copyParams);
    }
    Mutex::Unlock<PIPE_MTE3>(NDDMA_MUTEX_ID);
}

template <typename T, typename U, uint8_t maxDim>
__aicore__ inline void BrcToWithNDDMA<T, U, maxDim>::Process()
{
    this->CalcABBaseIdx(tdPtr_, blockIdx, aBaseIdx, bBaseIdx);
    this->CalcAxesLoopInfo(tdPtr_, blockIdx, lpInfo);
    this->CalcInBlockOffset(tdPtr_, blockIdx, inBlockOffset);
    this->CalcOutBlockOffset(tdPtr_, blockIdx, outBlockOffset);

    SetDMAParams();
    CalcOutSize();
    for (int64_t aIdx = 0; aIdx < lpInfo.aLpCnt; aIdx++) {
        mCopyParams.loopInfo.loopSize[0] = tdPtr_->uLpUnit;
        copyParams.blockLen = tdPtr_->uLpUnit * outLen;
        copySwitch = 1;
        int64_t aInOffset = this->CalcAInAxesOffset(tdPtr_, aBaseIdx + aIdx);
        int64_t aOutOffset = this->CalcAOutAxesOffset(tdPtr_, aBaseIdx + aIdx);
        for (int64_t uLpIdx = lpInfo.uLpBegIdx; uLpIdx < lpInfo.uLpEndIdx; uLpIdx++) {
            gmInOffset = (aInOffset + inBlockOffset + uLpIdx * tdPtr_->uLpUnit * tdPtr_->uInOffset * tdPtr_->isUNotB);
            CopyDataInWithDMA();
            gmOutOffset = (aOutOffset + outBlockOffset + uLpIdx * tdPtr_->uLpUnit * tdPtr_->uOutOffset);
            CopyDataOut();
        }
        if (lpInfo.uLeft > 0) {
            mCopyParams.loopInfo.loopSize[0] = lpInfo.uLeft;
            copyParams.blockLen = lpInfo.uLeft * outLen;
            gmInOffset = (aInOffset + inBlockOffset +
                          lpInfo.uLpCnt * tdPtr_->uLpUnit * tdPtr_->uInOffset * tdPtr_->isUNotB);
            CopyDataInWithDMA();
            gmOutOffset = (aOutOffset + outBlockOffset + lpInfo.uLpCnt * tdPtr_->uLpUnit * tdPtr_->uOutOffset);
            CopyDataOut();
        }
    }
}

} // namespace BrcTo

#endif // BROADCAST_TO_WITH_NDDMA_H_
