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
 * \file broadcast_to_with_tailAxis.h
 * \brief broadcst_to schedule (静态Tensor编程 + Mutex同步, 支持双buffer)
 */

#ifndef BROADCAST_TO_WITH_TAILAXIS_H_
#define BROADCAST_TO_WITH_TAILAXIS_H_

#include "broadcast_to_base.h"

namespace BrcTo {
using namespace AscendC;

using AscendC::Reg::CreateMask;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
using AscendC::Reg::UpdateMask;

constexpr int64_t DataCopyAlignUnit = 32;
constexpr MutexID MUTEX_IN0 = 0;  // 输入buffer0: MTE2↔V同步
constexpr MutexID MUTEX_OUT0 = 1; // 输出buffer0: V↔MTE3同步
constexpr MutexID MUTEX_OUT1 = 2; // 输出buffer1: V↔MTE3同步(与OUT0按buffer乒乓配对)
constexpr MutexID MUTEX_IN1 = 3;  // 输入buffer1: MTE2↔V同步(与IN0按buffer乒乓配对)

template <typename T, typename U>
class BrcToWithTailAxis : public BrcToBase<U> {
public:
    __aicore__ inline BrcToWithTailAxis(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const U* tilingDataPtr);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyDataIn(int64_t aIdx, int64_t curDataCnt);
    __aicore__ inline void ProcessBrc(int64_t aIdx, int64_t dataCnt, int64_t bufIdx);
    __aicore__ inline void BroadcastUb(LocalTensor<T>& inBuf, LocalTensor<T>& outBuf, int64_t inputOffset,
                                       int64_t elemIdx, int64_t brcIdx);
    __aicore__ inline void VFBrcTo(LocalTensor<T> outTensor, LocalTensor<T> inTensor, int64_t inputOffset,
                                   int64_t elemIdx, uint32_t brcCnt);
    __aicore__ inline void CopyDataOut(int64_t aIdx, int64_t brcIdx, LocalTensor<T>& outBuf);
    __aicore__ inline LocalTensor<T>& GetInBuf(int64_t bufIdx);
    __aicore__ inline LocalTensor<T>& GetOutBuf(int64_t brcIdx);
    __aicore__ inline MutexID GetInMutexId(int64_t bufIdx);
    __aicore__ inline MutexID GetOutMutexId(int64_t brcIdx);
    __aicore__ inline void CalcBlockOffset();

private:
    const U* tdPtr_;
    int64_t blockIdx = 0;
    AxesLpInfo lpInfo;
    int64_t aBaseIdx = 0;
    int64_t bBaseIdx = 0;
    GlobalTensor<T> inGM;
    GlobalTensor<T> outGM;
    LocalTensor<T> inBuf0_;
    LocalTensor<T> inBuf1_;
    LocalTensor<T> outBuf0_;
    LocalTensor<T> outBuf1_;
    uint8_t bufferCnt_ = 1;
    int64_t gmInOffset;
    int64_t gmOutOffset;
    int64_t inBlockOffset = 0;
    int64_t outBlockOffset = 0;
    int64_t aLpUnit_ = 0;
    int64_t uLpUnit_ = 0;
    int64_t uOutOffset_ = 1; // 恒为1(LAST_DIM_LARGE_B下尾轴是U轴, TrySetULastDim硬编码)
    uint32_t tensorSize_ = 0;
    AscendC::DataCopyExtParams copyInParams_{1, 0, 0, 0, 0};
};

template <typename T, typename U>
__aicore__ inline LocalTensor<T>& BrcToWithTailAxis<T, U>::GetInBuf(int64_t bufIdx)
{
    return (bufferCnt_ > 1 && (bufIdx & 1)) ? inBuf1_ : inBuf0_;
}

template <typename T, typename U>
__aicore__ inline LocalTensor<T>& BrcToWithTailAxis<T, U>::GetOutBuf(int64_t brcIdx)
{
    return (bufferCnt_ > 1 && (brcIdx & 1)) ? outBuf1_ : outBuf0_;
}

template <typename T, typename U>
__aicore__ inline MutexID BrcToWithTailAxis<T, U>::GetInMutexId(int64_t bufIdx)
{
    return (bufferCnt_ > 1 && (bufIdx & 1)) ? MUTEX_IN1 : MUTEX_IN0;
}

template <typename T, typename U>
__aicore__ inline MutexID BrcToWithTailAxis<T, U>::GetOutMutexId(int64_t brcIdx)
{
    return (bufferCnt_ > 1 && (brcIdx & 1)) ? MUTEX_OUT1 : MUTEX_OUT0;
}

template <typename T, typename U>
__aicore__ inline void BrcToWithTailAxis<T, U>::Init(GM_ADDR x, GM_ADDR y, const U* tilingDataPtr)
{
    inGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
    tdPtr_ = tilingDataPtr;

    // 读取高频字段到成员变量, 减少GM读取次数
    tensorSize_ = tilingDataPtr->tensorSize;
    aLpUnit_ = tilingDataPtr->aLpUnit;
    bufferCnt_ = tilingDataPtr->bufferCnt;
    // 推导值(LAST_DIM_LARGE_B下由tiling保证): uLpUnit=tensorSize, uOutOffset=1(声明处初始化)
    uLpUnit_ = tensorSize_;

    // 静态Tensor: 每核UB独立, outBuf_地址需32B对齐; 双buffer时ping/pong各一套
    int64_t realBlockIdx = GetBlockIdx();
    constexpr uint32_t alignBytes = 32;
    uint32_t bufBytes = (tensorSize_ * sizeof(T) + alignBytes - 1) / alignBytes * alignBytes;
    int64_t usedCoreCnt = tilingDataPtr->usedCoreCnt;
    bool needMod = tilingDataPtr->doubleMode != 0;
    inBuf0_ = LocalTensor<T>(TPosition::VECCALC, 0, tensorSize_);
    outBuf0_ = LocalTensor<T>(TPosition::VECCALC, bufBytes, tensorSize_);
    if (bufferCnt_ > 1) {
        inBuf1_ = LocalTensor<T>(TPosition::VECCALC, bufBytes * 2, tensorSize_);
        outBuf1_ = LocalTensor<T>(TPosition::VECCALC, bufBytes * 3, tensorSize_);
    }
    blockIdx = needMod ? realBlockIdx % usedCoreCnt : realBlockIdx;
}

template <typename T, typename U>
__aicore__ inline void BrcToWithTailAxis<T, U>::CalcBlockOffset()
{
    // blockAxis==uAxisPos(U轴)时需算偏移; LAST_DIM_LARGE_B下uOutOffset=1, uInOffset*isUNotB=0
    if (tdPtr_->blockAxis == uAxisPos) {
        outBlockOffset = blockIdx * tdPtr_->ntcULen;
    }
}

template <typename T, typename U>
__aicore__ inline void BrcToWithTailAxis<T, U>::Process()
{
    this->CalcABBaseIdx(tdPtr_, blockIdx, aBaseIdx, bBaseIdx);
    CalcBlockOffset();
    this->CalcAxesLoopInfo(tdPtr_, blockIdx, lpInfo);
    int64_t aLpCnt = lpInfo.aLpCnt / aLpUnit_;
    int64_t aLpLeft = lpInfo.aLpCnt % aLpUnit_;

    // 双buffer: inBuf按aIdx奇偶交替, outBuf按核内广播序号奇偶乒乓(MTE3搬第k笔时V广播第k+1笔)
    // 软件流水: 整轮循环里先预取下一轮CopyIn再执行本轮ProcessBrc, MTE2(k+1)与V(k)并行, 隐藏MTE2传输延迟
    // (bufferCnt_=1时无预取轮次, 行为不变)
    int64_t firstCnt = (aLpCnt > 0) ? aLpUnit_ : aLpLeft;
    CopyDataIn(0, firstCnt);
    for (int64_t aIdx = 0; aIdx < aLpCnt; aIdx++) {
        if (aIdx + 1 < aLpCnt) {
            CopyDataIn(aIdx + 1, aLpUnit_);
        } else if (aLpLeft > 0) {
            CopyDataIn(aLpCnt, aLpLeft); // 整轮已尽, 预取余量轮
        }
        ProcessBrc(aIdx, aLpUnit_, aIdx);
    }
    // 余量轮: 最后一轮, 无下一轮可预取
    if (aLpLeft > 0) {
        ProcessBrc(aLpCnt, aLpLeft, aLpCnt);
    }
}

template <typename T, typename U>
__aicore__ inline void BrcToWithTailAxis<T, U>::CopyDataIn(int64_t aIdx, int64_t curDataCnt)
{
    MutexID inMutexId = GetInMutexId(aIdx);
    // 获取输入buffer(等待V消费完同buffer的上一轮数据)
    Mutex::Lock<PIPE_MTE2>(inMutexId);
    gmInOffset = (this->CalcAInAxesOffset(tdPtr_, aBaseIdx + aIdx * aLpUnit_) + inBlockOffset);
    copyInParams_.blockLen = curDataCnt * sizeof(T);
    AscendC::DataCopyPadExtParams<T> copyInPadParams{false, 0, 0, 0};
    DataCopyPad(GetInBuf(aIdx), inGM[gmInOffset], copyInParams_, copyInPadParams);
    // 释放输入buffer, 通知V数据就绪
    Mutex::Unlock<PIPE_MTE2>(inMutexId);
}

template <typename T, typename U>
__aicore__ inline void BrcToWithTailAxis<T, U>::ProcessBrc(int64_t aIdx, int64_t dataCnt, int64_t bufIdx)
{
    MutexID inMutexId = GetInMutexId(bufIdx);
    // 获取输入buffer, 等待MTE2写入完成
    Mutex::Lock<PIPE_V>(inMutexId);

    int64_t copyUnit = DataCopyAlignUnit / sizeof(T);
    int64_t copyLpCnt = dataCnt / copyUnit;
    int64_t copyLpLeft = dataCnt % copyUnit;
    int64_t inputOffset = 0;

    // 循环不变量提到循环外
    LocalTensor<T>& inBuf = GetInBuf(bufIdx);
    int64_t aOutBase = aBaseIdx + aIdx * aLpUnit_;

    for (int64_t copyIdx = 0; copyIdx < copyLpCnt; copyIdx++) {
        inputOffset = copyIdx * copyUnit;
        for (int64_t elemIdx = 0; elemIdx < copyUnit; elemIdx++) {
            // brcIdx为核内广播序号(从0连续递增, 跨aIdx不重置), outBuf按其奇偶乒乓:
            // MTE3搬第k笔(out偶)与V广播第k+1笔(out奇)写不同buffer, 可流水并行
            int64_t brcIdx = aIdx * aLpUnit_ + inputOffset + elemIdx;
            LocalTensor<T>& outBuf = GetOutBuf(brcIdx);
            BroadcastUb(inBuf, outBuf, inputOffset, elemIdx, brcIdx);
            CopyDataOut(aOutBase + inputOffset + elemIdx, brcIdx, outBuf);
        }
    }
    if (copyLpLeft > 0) {
        inputOffset = copyLpCnt * copyUnit;
        for (int64_t elemIdx = 0; elemIdx < copyLpLeft; elemIdx++) {
            int64_t brcIdx = aIdx * aLpUnit_ + inputOffset + elemIdx;
            LocalTensor<T>& outBuf = GetOutBuf(brcIdx);
            BroadcastUb(inBuf, outBuf, inputOffset, elemIdx, brcIdx);
            CopyDataOut(aOutBase + inputOffset + elemIdx, brcIdx, outBuf);
        }
    }

    // 释放输入buffer, 允许MTE2复用
    Mutex::Unlock<PIPE_V>(inMutexId);
}

template <typename T, typename U>
__aicore__ inline void BrcToWithTailAxis<T, U>::BroadcastUb(LocalTensor<T>& inBuf, LocalTensor<T>& outBuf,
                                                            int64_t inputOffset, int64_t elemIdx, int64_t brcIdx)
{
    MutexID outMutexId = GetOutMutexId(brcIdx);
    // 获取输出buffer(等待MTE3搬完同buffer的上一笔), 填充后释放给MTE3
    Mutex::Lock<PIPE_V>(outMutexId);
    VFBrcTo(outBuf, inBuf, inputOffset, elemIdx, static_cast<uint32_t>(tensorSize_));
    Mutex::Unlock<PIPE_V>(outMutexId);
}

template <typename T, typename U>
__aicore__ inline void BrcToWithTailAxis<T, U>::CopyDataOut(int64_t aIdx, int64_t brcIdx, LocalTensor<T>& outBuf)
{
    MutexID outMutexId = GetOutMutexId(brcIdx);
    // 获取输出buffer, 等待V填充完成
    Mutex::Lock<PIPE_MTE3>(outMutexId);

    // 标量计算提前到加锁后, S流水线与V流水并行
    AscendC::DataCopyExtParams copyOutParams_{1, 0, 0, 0, 0};
    int64_t aOutOffset = this->CalcAOutAxesOffset(tdPtr_, aIdx);
    int64_t bOffset = 0;

    for (int64_t bIdx = lpInfo.bLpBegIdx; bIdx < lpInfo.bLpEndIdx; bIdx++) {
        bOffset = this->CalcBAxesOffset(tdPtr_, bBaseIdx + bIdx);
        copyOutParams_.blockLen = uLpUnit_ * uOutOffset_ * sizeof(T);
        for (int64_t uLpIdx = lpInfo.uLpBegIdx; uLpIdx < lpInfo.uLpEndIdx; uLpIdx++) {
            gmOutOffset = (aOutOffset + outBlockOffset + uLpIdx * uLpUnit_ * uOutOffset_);
            DataCopyPad(outGM[gmOutOffset + bOffset], outBuf, copyOutParams_);
        }
        if (lpInfo.uLeft > 0) {
            copyOutParams_.blockLen = lpInfo.uLeft * uOutOffset_ * sizeof(T);
            gmOutOffset = (aOutOffset + outBlockOffset + lpInfo.uLpCnt * uLpUnit_ * uOutOffset_);
            DataCopyPad(outGM[gmOutOffset + bOffset], outBuf, copyOutParams_);
        }
    }

    // 搬出完成, 释放输出buffer给下一轮V复用
    Mutex::Unlock<PIPE_MTE3>(outMutexId);
}

template <typename T, typename U>
__aicore__ inline void BrcToWithTailAxis<T, U>::VFBrcTo(LocalTensor<T> outTensor, LocalTensor<T> inTensor,
                                                        int64_t inputOffset, int64_t elemIdx, uint32_t brcCnt)
{
    __ubuf__ T* inputAddr = (__ubuf__ T*)inTensor.GetPhyAddr();
    __ubuf__ T* outputAddr = (__ubuf__ T*)outTensor.GetPhyAddr();

    uint32_t VL_CNT = Ops::Base::GetVRegSize() / sizeof(T);
    uint16_t brcLoopCnt = Ops::Base::CeilDiv(brcCnt, VL_CNT);
    int64_t outputOffset = 0;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T> tmpIn;
        AscendC::Reg::RegTensor<T> tmpOut;
        MaskReg pregAll = CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
        MaskReg pregAllB8 = CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
        MaskReg pregAllFB8 = CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALLF>();

        MaskReg pregTmp;
        MaskReg pregGather;
        MaskReg pregLoopB;

        Reg::LoadAlign(tmpIn, inputAddr + inputOffset);
        uint32_t sregTmp = uint32_t(elemIdx * sizeof(T));
        pregTmp = UpdateMask<uint8_t>(sregTmp);
        AscendC::Reg::Select(pregGather, pregAllFB8, pregAllB8, pregTmp);
        AscendC::Reg::Squeeze((Reg::RegTensor<uint8_t>&)tmpOut, (Reg::RegTensor<uint8_t>&)tmpIn, pregGather);
        Duplicate(tmpOut, tmpOut, pregAll);

        uint32_t sregB = brcCnt;
        for (uint16_t vIdx = 0; vIdx < brcLoopCnt; vIdx++) {
            pregLoopB = AscendC::Reg::UpdateMask<T>(sregB);
            AscendC::Reg::StoreAlign(outputAddr + outputOffset, tmpOut, pregLoopB);
            outputOffset += VL_CNT;
        }
    }
}

} // namespace BrcTo

#endif
