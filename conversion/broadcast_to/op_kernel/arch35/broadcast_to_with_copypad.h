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
 * \file broadcast_to_with_copypad.h
 * \brief kernel of broadcast_to with copy_pad (静态Tensor编程 + Mutex同步, 双buffer乒乓)
 */

#ifndef BROADCAST_TO_WITH_DATACOPYPAD_H_
#define BROADCAST_TO_WITH_DATACOPYPAD_H_

#include "broadcast_to_base.h"
#include "kernel_operator.h"

namespace BrcTo {
using namespace AscendC;

constexpr uint8_t bufferNum_datacopypad = 2;
constexpr MutexID CP_MUTEX_ID_BUF0 = 0; // buffer0: MTE2↔MTE3同步
constexpr MutexID CP_MUTEX_ID_BUF1 = 1; // buffer1: MTE2↔MTE3同步(与BUF0按buffer乒乓配对)

template <typename T, typename U>
class BrcToDataCopyPad : public BrcToBase<U> {
public:
    __aicore__ inline BrcToDataCopyPad(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const U* tilingDataPtr);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyDataIn(int64_t bufIdx);
    __aicore__ inline void CopyDataOut(int64_t bufIdx);
    __aicore__ inline LocalTensor<T>& GetBuf(int64_t bufIdx);
    __aicore__ inline MutexID GetBufMutexId(int64_t bufIdx);

private:
    const U* tdPtr_;
    int64_t blockIdx = 0;
    AxesLpInfo lpInfo;
    int64_t aBaseIdx = 0;
    int64_t bBaseIdx = 0;
    // 静态Tensor: 不通过TPipe/TQue, UB地址在Init时直接指定
    LocalTensor<T> buf0_;
    LocalTensor<T> buf1_;
    GlobalTensor<T> inGM;
    GlobalTensor<T> outGM;
    int64_t gmInOffset;
    int64_t gmOutOffset;
    int64_t inBlockOffset = 0;
    int64_t outBlockOffset = 0;
    int64_t cpIter_ = 0; // 核内搬运序号(跨a/u循环连续递增), buffer按其奇偶乒乓
    AscendC::DataCopyPadExtParams<T> copyInPadParams_{false, 0, 0, 0};
    AscendC::DataCopyExtParams copyParams{1, 0, 0, 0, 0};
    uint32_t outLen = sizeof(T);
    uint32_t tensorSize_ = 0;
};

template <typename T, typename U>
__aicore__ inline LocalTensor<T>& BrcToDataCopyPad<T, U>::GetBuf(int64_t bufIdx)
{
    return (bufIdx & 1) ? buf1_ : buf0_;
}

template <typename T, typename U>
__aicore__ inline MutexID BrcToDataCopyPad<T, U>::GetBufMutexId(int64_t bufIdx)
{
    return (bufIdx & 1) ? CP_MUTEX_ID_BUF1 : CP_MUTEX_ID_BUF0;
}

template <typename T, typename U>
__aicore__ inline void BrcToDataCopyPad<T, U>::Init(GM_ADDR x, GM_ADDR y, const U* tilingDataPtr)
{
    inGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
    tdPtr_ = tilingDataPtr;
    tensorSize_ = tilingDataPtr->tensorSize;

    // 静态Tensor: 双buffer乒乓(MTE2写入/MTE3读出同一块), 各buffer地址32B对齐
    constexpr uint32_t alignBytes = 32;
    uint32_t bufBytes = (tensorSize_ * sizeof(T) + alignBytes - 1) / alignBytes * alignBytes;
    buf0_ = LocalTensor<T>(TPosition::VECCALC, 0, tensorSize_);
    buf1_ = LocalTensor<T>(TPosition::VECCALC, bufBytes, tensorSize_);
    blockIdx = GetBlockIdx() % tdPtr_->usedCoreCnt;
}

template <typename T, typename U>
__aicore__ inline void BrcToDataCopyPad<T, U>::CopyDataIn(int64_t bufIdx)
{
    MutexID mutexId = GetBufMutexId(bufIdx);
    // 获取buffer(等待MTE3释放该buffer), 搬入后释放给MTE3
    Mutex::Lock<PIPE_MTE2>(mutexId);
    DataCopyPad(GetBuf(bufIdx), inGM[gmInOffset], copyParams, copyInPadParams_);
    Mutex::Unlock<PIPE_MTE2>(mutexId);
}

template <typename T, typename U>
__aicore__ inline void BrcToDataCopyPad<T, U>::CopyDataOut(int64_t bufIdx)
{
    MutexID mutexId = GetBufMutexId(bufIdx);
    // 获取buffer(等待MTE2写入完成), 搬出后释放给下一笔MTE2复用
    Mutex::Lock<PIPE_MTE3>(mutexId);
    int64_t bOffset = 0;
    for (int64_t bIdx = lpInfo.bLpBegIdx; bIdx < lpInfo.bLpEndIdx; bIdx++) {
        bOffset = this->CalcBAxesOffset(tdPtr_, bBaseIdx + bIdx);
        DataCopyPad(outGM[gmOutOffset + bOffset], GetBuf(bufIdx), copyParams);
    }
    // 搬出完成, 释放buffer给下一笔MTE2复用
    Mutex::Unlock<PIPE_MTE3>(mutexId);
}

template <typename T, typename U>
__aicore__ inline void BrcToDataCopyPad<T, U>::Process()
{
    this->CalcABBaseIdx(tdPtr_, blockIdx, aBaseIdx, bBaseIdx);
    this->CalcAxesLoopInfo(tdPtr_, blockIdx, lpInfo);
    // LARGE_A下uAxis恒为尾轴(TrySetULastDim命中): uInOffset=uOutOffset=1, 出入块偏移相同, 仅U轴切分时非零
    if (tdPtr_->blockAxis == uAxisPos) {
        inBlockOffset = blockIdx * tdPtr_->ntcULen;
        outBlockOffset = inBlockOffset;
    }

    for (int64_t aIdx = 0; aIdx < lpInfo.aLpCnt; aIdx++) {
        copyParams.blockLen = tdPtr_->uLpUnit * outLen;
        int64_t aInOffset = this->CalcAInAxesOffset(tdPtr_, aBaseIdx + aIdx);
        int64_t aOutOffset = this->CalcAOutAxesOffset(tdPtr_, aBaseIdx + aIdx);
        for (int64_t uLpIdx = lpInfo.uLpBegIdx; uLpIdx < lpInfo.uLpEndIdx; uLpIdx++) {
            gmInOffset = (aInOffset + inBlockOffset + uLpIdx * tdPtr_->uLpUnit);
            CopyDataIn(cpIter_);
            gmOutOffset = (aOutOffset + outBlockOffset + uLpIdx * tdPtr_->uLpUnit);
            CopyDataOut(cpIter_);
            cpIter_++;
        }
        if (lpInfo.uLeft > 0) {
            copyParams.blockLen = lpInfo.uLeft * outLen;
            gmInOffset = (aInOffset + inBlockOffset + lpInfo.uLpCnt * tdPtr_->uLpUnit);
            CopyDataIn(cpIter_);
            gmOutOffset = (aOutOffset + outBlockOffset + lpInfo.uLpCnt * tdPtr_->uLpUnit);
            CopyDataOut(cpIter_);
            cpIter_++;
        }
    }
}

} // namespace BrcTo

#endif // BROADCAST_TO_WITH_DATACOPYPAD_H_
