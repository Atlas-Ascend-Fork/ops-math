/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPLIT_V_UB_SPLIT_SAME_LEN_SMALL_G_H
#define SPLIT_V_UB_SPLIT_SAME_LEN_SMALL_G_H
#include "kernel_operator_list_tensor_intf.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace SplitV {
using namespace AscendC;
template <typename T, typename U, typename Y> // T原始数据类型  U做datacopygather的数据类型 Y是做vci的数据类型
class SplitVUbSplitSameLenSmallG {
public:
    __aicore__ inline SplitVUbSplitSameLenSmallG(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SplitVScalarTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline __gm__ T* GetTensorAddr(int64_t index);
    __aicore__ inline void CopyIn(int64_t blockCount, int64_t blockLen, int64_t srcOffset, int64_t srcStride,
                                  int64_t dstStride, LocalTensor<T>& xUb);
    __aicore__ inline void ComputeIdx(int64_t processGNum, int64_t processMNNum, uint32_t gnAlignSize,
                                      uint32_t mnAlignSize);
    __aicore__ inline void Compute(uint32_t g, int64_t processMNNum, uint32_t gnAlignSize, uint32_t mnAlignSize,
                                   LocalTensor<T>& srcUb);
    __aicore__ inline void CopyOut(int64_t localOffset, int64_t blockCount, int64_t blockLen, int64_t dstOffset,
                                   int64_t srcStride, int64_t dstStride);

private:
    constexpr static uint32_t BUFFER_NUM = 2;
    constexpr static uint32_t BLOCK_BYTES = Ops::Base::GetUbBlockSize();
    constexpr static int64_t BLOCK_ELENUM = Ops::Base::GetUbBlockSize() / sizeof(T);
    constexpr static int64_t VL_LEN = Ops::Base::GetVRegSize();
    constexpr static int64_t DTYPE_SIZE = sizeof(T);
    constexpr static int32_t U_VL = Ops::Base::GetVRegSize() / sizeof(U);
    // 手动双缓冲同步(免TPipe/TQue): in 复用(V_MTE2)、in 就绪(MTE2_V)、out 复用(MTE3_V)、
    // out 就绪(V_MTE3); 不同 HardEvent 类型同号 id 物理独立
    constexpr static event_t EVT_ID0 = static_cast<event_t>(0);
    constexpr static event_t EVT_ID1 = static_cast<event_t>(1);
    constexpr static event_t EVT_ID2 = static_cast<event_t>(2);

    const SplitVScalarTilingData* tilingData_ = nullptr;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    LocalTensor<T> yLocal_;
    LocalTensor<U> idxLocal_;
    ListTensorDesc inputList_;
    int32_t blockIdx_ = 0;
    int64_t gSize_ = 0;
    int64_t nSize_ = 0;
    int64_t mUBCount_ = 0;
    int64_t mUBFactor_ = 0;
    int64_t mUBFactorTail_ = 0;
    int64_t gUBCount_ = 0;
    int64_t gUBFactor_ = 0;
    int64_t gUBFactorTail_ = 0;
    int64_t blockFactor_ = 0;
    int64_t blockFactorTail_ = 0;
    int64_t blkProcessNum_ = 0;
    int64_t realCoreNum_ = 0;

    // 手动UB布局: [idx: VL_LEN][in0][in1][out0][out1], 各子块32B对齐; idx 单块由 V 每轮覆写(V序内自洽)
    uint32_t idxOffset_ = 0;
    uint32_t inXOffset0_ = 0;
    uint32_t inXOffset1_ = 0;
    uint32_t outYOffset0_ = 0;
    uint32_t outYOffset1_ = 0;
    uint32_t inBufElems_ = 0;
    uint32_t idxBufElems_ = 0;

    int64_t curIdx_ = 0;
    int64_t tmpLoop_ = 0;
    int64_t nRegSize_ = 0;

    DataCopyExtParams copyInParam_{0, 0, 0, 0, 0};
    DataCopyPadExtParams<T> padParam_{false, 0, 0, 0};
    DataCopyExtParams copyOutParam_{0, 0, 0, 0, 0};
};

template <typename T, typename U, typename Y>
__aicore__ inline void SplitVUbSplitSameLenSmallG<T, U, Y>::Init(GM_ADDR x, GM_ADDR y,
                                                                 const SplitVScalarTilingData* tilingData)
{
    AscendC::InitSocState();
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;
    const int64_t ubSize = tilingData_->ubSize;
    gSize_ = tilingData_->gSize;
    nSize_ = tilingData_->nBlockFactorNum;
    mUBFactor_ = tilingData_->mBlockFactor;
    mUBFactorTail_ = tilingData_->mBlockFactorTail;
    mUBCount_ = tilingData_->mBlockCount;
    gUBFactor_ = tilingData_->gUBFactor;
    gUBFactorTail_ = tilingData_->gUBFactorTail;
    gUBCount_ = tilingData_->gUBCount;
    blockFactor_ = tilingData_->blockFactor;
    blockFactorTail_ = tilingData_->blockFactorTail;
    realCoreNum_ = tilingData_->realCoreNum;

    const uint32_t idxSpaceBytes = static_cast<uint32_t>(VL_LEN);
    const uint32_t halfUbBytes = static_cast<uint32_t>(ubSize) / BUFFER_NUM;
    const uint32_t inputSpaceBytes = (halfUbBytes - idxSpaceBytes) / BUFFER_NUM;
    const uint32_t alignedInputSpace = Ops::Base::CeilAlign(inputSpaceBytes, BLOCK_BYTES);

    idxOffset_ = 0;
    inXOffset0_ = Ops::Base::CeilAlign(idxSpaceBytes, BLOCK_BYTES);
    inXOffset1_ = inXOffset0_ + alignedInputSpace;
    outYOffset0_ = inXOffset1_ + alignedInputSpace;
    outYOffset1_ = outYOffset0_ + alignedInputSpace;
    inBufElems_ = alignedInputSpace / static_cast<uint32_t>(DTYPE_SIZE);
    idxBufElems_ = idxSpaceBytes / static_cast<uint32_t>(sizeof(U));

    xGm_.SetGlobalBuffer((__gm__ T*)x);
    inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(y));

    // Calc start idx per core
    blkProcessNum_ = blockFactor_;
    curIdx_ = blockIdx_ * blockFactor_;
    if (blockIdx_ < blockFactorTail_) {
        blkProcessNum_ += 1;
        curIdx_ += blockIdx_;
    } else {
        curIdx_ += blockFactorTail_;
    }
    if constexpr (sizeof(T) == sizeof(int64_t)) {
        tmpLoop_ = U_VL / (nSize_ * BUFFER_NUM);
    } else {
        tmpLoop_ = U_VL / nSize_;
    }
}

template <typename T, typename U, typename Y>
__aicore__ inline void SplitVUbSplitSameLenSmallG<T, U, Y>::Process()
{
    if (blockIdx_ >= realCoreNum_) {
        return;
    }

    SetFlag<HardEvent::V_MTE2>(EVT_ID0);
    SetFlag<HardEvent::V_MTE2>(EVT_ID1);
    SetFlag<HardEvent::MTE3_V>(EVT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVT_ID1);

    LocalTensor<U> idxUbStatic(TPosition::VECCALC, idxOffset_, idxBufElems_);

    const int64_t mUBCount = mUBCount_;
    const int64_t mUBFactor = mUBFactor_;
    const int64_t mUBFactorTail = mUBFactorTail_;
    const int64_t gUBCount = gUBCount_;
    const int64_t gUBFactor = gUBFactor_;
    const int64_t gUBFactorTail = gUBFactorTail_;
    const int64_t gSize = gSize_;
    const int64_t nSize = nSize_;
    const int64_t blkProcessNum = blkProcessNum_;

    int64_t curIdx = curIdx_;
    int64_t tmpLoop = tmpLoop_;
    uint16_t bufIdX = 0;
    uint16_t bufIdY = 0;

    for (int64_t i = 0; i < blkProcessNum; i++) {
        int64_t mIdx = curIdx / gUBCount;
        int64_t gIdx = curIdx % gUBCount;
        // Calc 4 types processNum
        int64_t processMNum;
        int64_t processGNum;
        if (mUBCount > 1 && gUBCount > 1 && mIdx == mUBCount - 1 && gIdx == gUBCount - 1) {
            // Process last tail
            processMNum = mUBFactorTail;
            processGNum = gUBFactorTail;
        } else if (mUBCount > 1 && mIdx == mUBCount - 1) {
            // Process Mtail
            processMNum = mUBFactorTail;
            processGNum = gUBFactor;
        } else if (gUBCount > 1 && gIdx == gUBCount - 1) {
            // Process Gtail
            processMNum = mUBFactor;
            processGNum = gUBFactorTail;
        } else {
            // Process mainFactor
            processMNum = mUBFactor;
            processGNum = gUBFactor;
        }

        // Copy in
        int64_t gmOffset = gIdx * gUBFactor * nSize + mIdx * mUBFactor * gSize * nSize;
        int64_t srcStride = gSize * nSize - processGNum * nSize;
        if (processMNum < tmpLoop) {
            tmpLoop = processMNum;
            tmpLoop_ = tmpLoop;
        }

        // Comput and Copy out
        int64_t processNum = processMNum * nSize;
        uint32_t gnAlignSize = processGNum * nSize;
        uint32_t mnAlignSize = CeilDivision(processMNum * nSize, BLOCK_ELENUM) * BLOCK_ELENUM;
        nRegSize_ = nSize * tmpLoop;

        idxLocal_ = idxUbStatic;
        ComputeIdx(processGNum, processNum, gnAlignSize, mnAlignSize);

        // out 复用: 上一轮同块 MTE3(拷出)读完 -> 本轮 V(gather) 可覆盖写
        event_t evtY = (bufIdY == 0) ? EVT_ID0 : EVT_ID1;
        WaitFlag<HardEvent::MTE3_V>(evtY);
        uint32_t outYOffset = (bufIdY == 0) ? outYOffset0_ : outYOffset1_;
        yLocal_ = LocalTensor<T>(TPosition::VECOUT, outYOffset, inBufElems_);

        // in 复用: 上一轮同块 V 读完 -> 本轮 MTE2 可覆盖写
        event_t evtX = (bufIdX == 0) ? EVT_ID0 : EVT_ID1;
        WaitFlag<HardEvent::V_MTE2>(evtX);
        uint32_t inXOffset = (bufIdX == 0) ? inXOffset0_ : inXOffset1_;
        LocalTensor<T> xUb(TPosition::VECIN, inXOffset, inBufElems_);

        CopyIn(1, processMNum * processGNum * nSize, gmOffset, srcStride, 0, xUb);

        // in 就绪: MTE2 写完 -> V(gather) 读
        SetFlag<HardEvent::MTE2_V>(evtX);
        WaitFlag<HardEvent::MTE2_V>(evtX);

        LocalTensor<T> srcUb = xUb;
        for (int64_t g = 0; g < processGNum; g++) {
            Compute(g, processNum, gnAlignSize, mnAlignSize, srcUb);
            int64_t outOffset = mIdx * mUBFactor * nSize;
            int64_t yGMIdx = gIdx * gUBFactor + g;
            yGm_.SetGlobalBuffer(GetTensorAddr(yGMIdx));
            // out 就绪: V 写完 yLocal_ -> MTE3 读出
            SetFlag<HardEvent::V_MTE3>(EVT_ID2);
            WaitFlag<HardEvent::V_MTE3>(EVT_ID2);
            int64_t ubOffset = g * mnAlignSize;
            CopyOut(ubOffset, 1, processMNum * nSize, outOffset, 0, 0);
        }

        // in 释放: V 读完 -> 下一轮同块 MTE2 可覆盖写
        SetFlag<HardEvent::V_MTE2>(evtX);
        // out 释放: MTE3 读完 -> 下一轮同块 V 可覆盖写
        SetFlag<HardEvent::MTE3_V>(evtY);

        bufIdX ^= 1;
        bufIdY ^= 1;
        curIdx++;
    }

    // 收尾: 排空仍在途的复用事件
    WaitFlag<HardEvent::V_MTE2>(EVT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVT_ID1);

    curIdx_ = curIdx;
    tmpLoop_ = tmpLoop;
}

template <typename T, typename U, typename Y>
__aicore__ inline void SplitVUbSplitSameLenSmallG<T, U, Y>::CopyIn(int64_t blockCount, int64_t blockLen,
                                                                   int64_t srcOffset, int64_t srcStride,
                                                                   int64_t dstStride, LocalTensor<T>& xUb)
{
    copyInParam_.blockCount = blockCount;
    copyInParam_.blockLen = blockLen * DTYPE_SIZE;
    copyInParam_.srcStride = srcStride * DTYPE_SIZE;
    copyInParam_.dstStride = dstStride / BLOCK_ELENUM;
#if defined(__DAV_920R1__) || defined(__DAV_920R2__)
    DataCopyPad<T, PaddingMode::Compact>(xUb, xGm_[srcOffset], copyInParam_, padParam_, true);
#else
    DataCopyPad<T, PaddingMode::Compact>(xUb, xGm_[srcOffset], copyInParam_, padParam_);
#endif
}

template <typename T, typename U, typename Y>
__aicore__ inline void SplitVUbSplitSameLenSmallG<T, U, Y>::CopyOut(int64_t localOffset, int64_t blockCount,
                                                                    int64_t blockLen, int64_t dstOffset,
                                                                    int64_t srcStride, int64_t dstStride)
{
    copyOutParam_.blockCount = blockCount;
    copyOutParam_.blockLen = blockLen * DTYPE_SIZE;
    copyOutParam_.srcStride = srcStride / BLOCK_ELENUM;
    copyOutParam_.dstStride = dstStride * DTYPE_SIZE;
    DataCopyPad(yGm_[dstOffset], yLocal_[localOffset], copyOutParam_);
}

template <typename T, typename U, typename Y>
__aicore__ inline void SplitVUbSplitSameLenSmallG<T, U, Y>::ComputeIdx(int64_t processGNum, int64_t processMNNum,
                                                                       uint32_t gnAlignSize, uint32_t mnAlignSize)
{
    LocalTensor<U>& idxUb = idxLocal_;
    uint32_t processNum = processMNNum;
    uint16_t loopNum = CeilDivision(processNum, nRegSize_);
    uint32_t processLastNum = nRegSize_ * loopNum == processNum ? nRegSize_ : processNum % nRegSize_;
    uint32_t mnSize = tmpLoop_ * nSize_;
    uint32_t mnSizeLast = processLastNum;
    uint32_t nRegSize = nRegSize_;
    int32_t uVL = U_VL;
    int64_t nSize = nSize_;
    int64_t tmpLoop = tmpLoop_;
    uint16_t gLoopNum = processGNum;

    if constexpr (sizeof(T) == sizeof(int64_t)) {
        uint32_t nSizeInt64 = nSize * BUFFER_NUM;
        uint32_t tmpLoop = uVL / nSizeInt64;
        nRegSize = nRegSize_ * BUFFER_NUM;
        gnAlignSize = gnAlignSize * BUFFER_NUM;
        mnAlignSize = mnAlignSize * BUFFER_NUM;
        processNum = processNum * BUFFER_NUM;
        processLastNum = processNum % nRegSize == 0 ? nRegSize : processNum % nRegSize;
        loopNum = CeilDivision(processNum, tmpLoop * nSizeInt64);
        __ubuf__ U* idxPtr = (__ubuf__ U*)idxUb.GetPhyAddr();

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<Y> indexRegB64;
            AscendC::Reg::RegTensor<U> tmpB64;
            AscendC::Reg::RegTensor<U> tmp1B64;
            AscendC::Reg::RegTensor<U> tmp2B64;
            AscendC::Reg::RegTensor<U> addRegB64;
            AscendC::Reg::RegTensor<U> niRegB64;
            AscendC::Reg::RegTensor<U> subRegB64;
            AscendC::Reg::MaskReg maskB64;
            AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<U, AscendC::Reg::MaskPattern::ALL>();

            maskB64 = AscendC::Reg::UpdateMask<U>(processNum);

            Y startIdx = (Y)0;
            AscendC::Reg::Arange(indexRegB64, startIdx);
            AscendC::Reg::Duplicate(niRegB64, (U)nSizeInt64, maskAll);
            AscendC::Reg::Div(tmpB64, (AscendC::Reg::RegTensor<U>&)indexRegB64, niRegB64, maskB64);
            AscendC::Reg::Muls(tmp1B64, tmpB64, (U)gnAlignSize, maskB64);
            AscendC::Reg::Mul(subRegB64, tmpB64, niRegB64, maskB64);
            AscendC::Reg::Sub(tmp2B64, (AscendC::Reg::RegTensor<U>&)indexRegB64, subRegB64, maskB64);
            AscendC::Reg::Add(addRegB64, tmp1B64, tmp2B64, maskB64);

            AscendC::Reg::StoreAlign(idxPtr, addRegB64, maskB64);
        }
    } else {
        __ubuf__ U* idxPtr = (__ubuf__ U*)idxUb.GetPhyAddr();

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<Y> indexReg;
            AscendC::Reg::RegTensor<U> tmp;
            AscendC::Reg::RegTensor<U> tmp1;
            AscendC::Reg::RegTensor<U> tmp2;
            AscendC::Reg::RegTensor<U> addReg;
            AscendC::Reg::RegTensor<U> niReg;
            AscendC::Reg::RegTensor<U> subReg;
            AscendC::Reg::MaskReg mask;
            AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<U, AscendC::Reg::MaskPattern::ALL>();

            mask = AscendC::Reg::UpdateMask<U>(processNum);
            Y startIdx = (Y)0;
            AscendC::Reg::Arange(indexReg, startIdx);
            AscendC::Reg::Duplicate(niReg, (U)nSize, maskAll);
            AscendC::Reg::Div(tmp, (AscendC::Reg::RegTensor<U>&)indexReg, niReg, mask);
            AscendC::Reg::Muls(tmp1, tmp, (U)gnAlignSize, mask);
            AscendC::Reg::Mul(subReg, tmp, niReg, mask);
            AscendC::Reg::Sub(tmp2, (AscendC::Reg::RegTensor<U>&)indexReg, subReg, mask);
            AscendC::Reg::Add(addReg, tmp1, tmp2, mask);

            AscendC::Reg::StoreAlign(idxPtr, addReg, mask);
        }
    }
}

template <typename T, typename U, typename Y>
__aicore__ inline void SplitVUbSplitSameLenSmallG<T, U, Y>::Compute(uint32_t g, int64_t processMNNum,
                                                                    uint32_t gnAlignSize, uint32_t mnAlignSize,
                                                                    LocalTensor<T>& srcUb)
{
    uint32_t processNum = processMNNum;
    uint16_t loopNum = CeilDivision(processNum, nRegSize_);
    uint32_t gnSize = g * nSize_;
    uint32_t processLastNum = nRegSize_ * loopNum == processNum ? nRegSize_ : processNum % nRegSize_;
    uint32_t mnSize = tmpLoop_ * nSize_;
    uint32_t mnSizeLast = processLastNum;
    uint32_t nRegSize = nRegSize_;
    int32_t uVL = U_VL;
    int64_t nSize = nSize_;
    int64_t tmpLoop = tmpLoop_;

    if constexpr (sizeof(T) == sizeof(int64_t)) {
        LocalTensor<U> srcUbU32 = srcUb.template ReinterpretCast<U>();
        // As 2 uint32 data concatenated to process
        uint32_t nSizeInt64 = nSize_ * BUFFER_NUM;
        gnSize = g * nSizeInt64;
        mnSize = mnSize * BUFFER_NUM;
        mnSizeLast = mnSizeLast * BUFFER_NUM;
        tmpLoop = uVL / nSizeInt64;
        nRegSize = nRegSize_ * BUFFER_NUM;
        gnAlignSize = gnAlignSize * BUFFER_NUM;
        mnAlignSize = mnAlignSize * BUFFER_NUM;
        loopNum = loopNum * BUFFER_NUM;
        processNum = processNum * BUFFER_NUM;
        processLastNum = processNum % nRegSize == 0 ? nRegSize : processNum % nRegSize;
        loopNum = CeilDivision(processNum, tmpLoop * nSizeInt64);
        __ubuf__ U* srcPtr = (__ubuf__ U*)srcUbU32.GetPhyAddr();
        __ubuf__ U* dstPtr = (__ubuf__ U*)yLocal_.GetPhyAddr() + g * mnAlignSize;
        __ubuf__ U* idxPtr = (__ubuf__ U*)idxLocal_.GetPhyAddr();

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<U> addReg;
            AscendC::Reg::RegTensor<U> dstReg;
            AscendC::Reg::UnalignRegForStore uDst;
            AscendC::Reg::MaskReg mask;
            AscendC::Reg::MaskReg maskTmp;

            mask = AscendC::Reg::UpdateMask<U>(processNum);
            maskTmp = AscendC::Reg::UpdateMask<U>(nRegSize);
            AscendC::Reg::LoadAlign(addReg, idxPtr);
            AscendC::Reg::Adds(addReg, addReg, (U)gnSize, mask);

            for (uint16_t i = 0; i < loopNum; i++) {
                AscendC::Reg::Gather(dstReg, srcPtr, addReg, maskTmp);
                // Copy out
                AscendC::Reg::StoreUnAlign(dstPtr, dstReg, uDst, mnSize);
                AscendC::Reg::Adds(addReg, addReg, (U)(tmpLoop * gnAlignSize), mask);
            }
            maskTmp = AscendC::Reg::UpdateMask<U>(processLastNum);
            AscendC::Reg::Gather(dstReg, srcPtr, addReg, maskTmp);
            AscendC::Reg::StoreUnAlign(dstPtr, dstReg, uDst, mnSizeLast);
            AscendC::Reg::StoreUnAlignPost(dstPtr, uDst, 0);
        }
    } else {
        __ubuf__ T* srcPtr = (__ubuf__ T*)srcUb.GetPhyAddr();
        __ubuf__ U* dstPtrU = (__ubuf__ U*)yLocal_.GetPhyAddr() + g * mnAlignSize;
        __ubuf__ T* dstPtrT = (__ubuf__ T*)yLocal_.GetPhyAddr() + g * mnAlignSize;
        __ubuf__ U* idxPtr = (__ubuf__ U*)idxLocal_.GetPhyAddr();

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<U> addReg;
            AscendC::Reg::RegTensor<U> dstReg;
            AscendC::Reg::RegTensor<T> dstRegT;
            AscendC::Reg::UnalignRegForStore uDst;
            AscendC::Reg::MaskReg mask;
            AscendC::Reg::MaskReg maskTmp;

            mask = AscendC::Reg::UpdateMask<U>(processNum);
            maskTmp = AscendC::Reg::UpdateMask<U>(nRegSize);
            AscendC::Reg::LoadAlign(addReg, idxPtr);
            AscendC::Reg::Adds(addReg, addReg, (U)gnSize, mask);

            for (uint16_t i = 0; i < loopNum; i++) {
                AscendC::Reg::Gather(dstReg, srcPtr, addReg, maskTmp);
                // Copy out
                if constexpr (sizeof(T) == sizeof(int8_t)) {
                    // Convert B16 to B8
                    AscendC::Reg::Pack(dstRegT, dstReg);
                    AscendC::Reg::StoreUnAlign(dstPtrT, dstRegT, uDst, mnSize);
                    AscendC::Reg::StoreUnAlignPost(dstPtrT, uDst, 0);
                } else {
                    AscendC::Reg::StoreUnAlign(dstPtrU, dstReg, uDst, mnSize);
                    AscendC::Reg::StoreUnAlignPost(dstPtrU, uDst, 0);
                }
                AscendC::Reg::Adds(addReg, addReg, (U)(tmpLoop * gnAlignSize), mask);
            }
            maskTmp = AscendC::Reg::UpdateMask<U>(processLastNum);
            AscendC::Reg::Gather(dstReg, srcPtr, addReg, maskTmp);
            if constexpr (sizeof(T) == sizeof(int8_t)) {
                // Convert B16 to B8
                AscendC::Reg::Pack(dstRegT, dstReg);
                AscendC::Reg::StoreUnAlign(dstPtrT, dstRegT, uDst, mnSizeLast);
                AscendC::Reg::StoreUnAlignPost(dstPtrT, uDst, 0);
            } else {
                AscendC::Reg::StoreUnAlign(dstPtrU, dstReg, uDst, mnSizeLast);
                AscendC::Reg::StoreUnAlignPost(dstPtrU, uDst, 0);
            }
        }
    }
}

template <typename T, typename U, typename Y>
__aicore__ inline __gm__ T* SplitVUbSplitSameLenSmallG<T, U, Y>::GetTensorAddr(int64_t index)
{
    return inputList_.GetDataPtr<T>(index);
}

} // namespace SplitV
#endif // SPLIT_V_UB_SPLIT_SAME_LEN_SMALL_G_H
