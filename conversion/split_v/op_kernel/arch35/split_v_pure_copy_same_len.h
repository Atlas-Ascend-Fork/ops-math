/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPLIT_V_PURE_COPY_SAME_LEN_H
#define SPLIT_V_PURE_COPY_SAME_LEN_H
#include "kernel_operator_list_tensor_intf.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace SplitV {
using namespace AscendC;

template <typename T>
class SplitVPureCopyModeSameLen {
public:
    __aicore__ inline SplitVPureCopyModeSameLen(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SplitVSlimTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline __gm__ T* GetTensorAddr(int64_t index);
    __aicore__ inline void ProcessWithCutN();
    __aicore__ inline void ProcessWithCutMG();
    __aicore__ inline int64_t CalcGMOffset(int64_t curIdx) const;

private:
    constexpr static uint32_t BUFFER_NUM = 2;
    constexpr static uint32_t BLOCK_SIZE = Ops::Base::GetUbBlockSize();
    constexpr static int64_t BLOCK_NUM = BLOCK_SIZE / sizeof(T);
    constexpr static int64_t DTYPE_SIZE = sizeof(T);
    constexpr static event_t EVT_ID0 = static_cast<event_t>(0);
    constexpr static event_t EVT_ID1 = static_cast<event_t>(1);

    const SplitVSlimTilingData* tilingData_ = nullptr;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
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
    int64_t nUBCount_ = 0;
    int64_t nUBFactor_ = 0;
    int64_t nUBFactorTail_ = 0;
    int64_t blockFactor_ = 0;
    int64_t blockFactorTail_ = 0;
    int64_t blkProcessNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t realCoreNum_ = 0;

    int64_t curIdx_ = 0;

    DataCopyExtParams copyInParam_{0, 0, 0, 0, 0};
    DataCopyPadExtParams<T> padParam_{false, 0, 0, 0};
    DataCopyExtParams copyOutParam_{0, 0, 0, 0, 0};
    LoopModeParams loopMode_{};
};

template <typename T>
__aicore__ inline void SplitVPureCopyModeSameLen<T>::Init(GM_ADDR x, GM_ADDR y, const SplitVSlimTilingData* tilingData)
{
    AscendC::InitSocState();
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;

    const int64_t ubSize = tilingData->ubSize;
    const int64_t gSize = tilingData->gSize;
    const int64_t nSize = tilingData->nBlockFactorNum;
    const int64_t mUBFactor = tilingData->mBlockFactor;
    const int64_t mUBFactorTail = tilingData->mBlockFactorTail;
    const int64_t mUBCount = tilingData->mBlockCount;
    const int64_t gUBFactor = tilingData->gUBFactor;
    const int64_t gUBFactorTail = tilingData->gUBFactorTail;
    const int64_t gUBCount = tilingData->gUBCount;
    const int64_t nUBFactor = tilingData->nBlockFactor;
    const int64_t nUBFactorTail = tilingData->nBlockFactorTail;
    const int64_t nUBCount = tilingData->nBlockCount;
    const int64_t blockFactor = tilingData->blockFactor;
    const int64_t blockFactorTail = tilingData->blockFactorTail;
    const int64_t realCoreNum = tilingData->realCoreNum;

    gSize_ = gSize;
    nSize_ = nSize;
    mUBFactor_ = mUBFactor;
    mUBFactorTail_ = mUBFactorTail;
    mUBCount_ = mUBCount;
    gUBFactor_ = gUBFactor;
    gUBFactorTail_ = gUBFactorTail;
    gUBCount_ = gUBCount;
    nUBFactor_ = nUBFactor;
    nUBFactorTail_ = nUBFactorTail;
    nUBCount_ = nUBCount;
    blockFactor_ = blockFactor;
    blockFactorTail_ = blockFactorTail;
    ubSize_ = ubSize;
    realCoreNum_ = realCoreNum;

    xGm_.SetGlobalBuffer((__gm__ T*)x);
    inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(y));

    blkProcessNum_ = blockFactor;
    curIdx_ = blockIdx_ * blockFactor;
    if (blockIdx_ < blockFactorTail) {
        blkProcessNum_ += 1;
        curIdx_ += blockIdx_;
    } else {
        curIdx_ += blockFactorTail;
    }
}

template <typename T>
__aicore__ inline void SplitVPureCopyModeSameLen<T>::Process()
{
    if (blockIdx_ >= realCoreNum_) {
        return;
    }

    if (nUBCount_ > 1) {
        ProcessWithCutN();
    } else {
        ProcessWithCutMG();
    }
}

template <typename T>
__aicore__ inline int64_t SplitVPureCopyModeSameLen<T>::CalcGMOffset(int64_t curIdx) const
{
    int64_t gRSize = nUBCount_;
    int64_t mRSize = gUBCount_ * nUBCount_;
    int64_t gASize = nSize_;
    int64_t mASize = gSize_ * nSize_;
    return curIdx % nUBCount_ * nUBFactor_ + curIdx / gRSize % gUBCount_ * gUBFactor_ * gASize +
           curIdx / mRSize * mUBFactor_ * mASize;
}

template <typename T>
__aicore__ inline void SplitVPureCopyModeSameLen<T>::ProcessWithCutN()
{
    SetFlag<HardEvent::MTE3_MTE2>(EVT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVT_ID1);

    loopMode_.loop1Size = 1;
    loopMode_.loop2Size = 1;
    const int64_t nUBCount = nUBCount_;
    const int64_t nUBFactor = nUBFactor_;
    const int64_t nUBFactorTail = nUBFactorTail_;
    const int64_t gUBCount = gUBCount_;
    const int64_t gSize = gSize_;
    const int64_t nSize = nSize_;
    const int64_t blkProcessNum = blkProcessNum_;
    const int64_t halfUbBytes = ubSize_ / BUFFER_NUM;
    const int64_t halfUbElems = halfUbBytes / DTYPE_SIZE;

    int64_t curIdx = curIdx_;
    uint16_t bufId = 0;

    for (int64_t i = 0; i < blkProcessNum; i++) {
        event_t evt = (bufId == 0) ? EVT_ID0 : EVT_ID1;

        WaitFlag<HardEvent::MTE3_MTE2>(evt);

        LocalTensor<T> srcLocal(TPosition::VECIN, static_cast<uint32_t>(bufId * halfUbBytes),
                                static_cast<uint32_t>(halfUbElems));

        int64_t nIdx = curIdx % nUBCount;
        int64_t processNNum = (nIdx == nUBCount - 1) ? nUBFactorTail : nUBFactor;
        int64_t gmOffset = CalcGMOffset(curIdx);

        SetLoopModePara(loopMode_, DataCopyMVType::OUT_TO_UB);
        copyInParam_.blockCount = 1;
        copyInParam_.blockLen = processNNum * DTYPE_SIZE;
        copyInParam_.srcStride = 0;
        copyInParam_.dstStride = 0;
#if defined(__DAV_920R1__) || defined(__DAV_920R2__)
        DataCopyPad(srcLocal, xGm_[gmOffset], copyInParam_, padParam_, true);
#else
        DataCopyPad(srcLocal, xGm_[gmOffset], copyInParam_, padParam_);
#endif
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);

        SetFlag<HardEvent::MTE2_MTE3>(evt);
        WaitFlag<HardEvent::MTE2_MTE3>(evt);

        int64_t mgIdx = curIdx / nUBCount;
        int64_t yGMIdx = mgIdx % gSize;
        int64_t dstOffset = mgIdx / gUBCount * nSize + nIdx * nUBFactor;
        yGm_.SetGlobalBuffer(GetTensorAddr(yGMIdx));
        copyOutParam_.blockCount = 1;
        copyOutParam_.blockLen = processNNum * DTYPE_SIZE;
        copyOutParam_.srcStride = 0;
        copyOutParam_.dstStride = 0;
        DataCopyPad(yGm_[dstOffset], srcLocal, copyOutParam_);

        SetFlag<HardEvent::MTE3_MTE2>(evt);

        bufId ^= 1;
        curIdx++;
    }

    WaitFlag<HardEvent::MTE3_MTE2>(EVT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVT_ID1);
    curIdx_ = curIdx;
}

template <typename T>
__aicore__ inline void SplitVPureCopyModeSameLen<T>::ProcessWithCutMG()
{
    SetFlag<HardEvent::MTE3_MTE2>(EVT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVT_ID1);

    const int64_t mUBCount = mUBCount_;
    const int64_t mUBFactor = mUBFactor_;
    const int64_t mUBFactorTail = mUBFactorTail_;
    const int64_t gUBCount = gUBCount_;
    const int64_t gUBFactor = gUBFactor_;
    const int64_t gUBFactorTail = gUBFactorTail_;
    const int64_t gSize = gSize_;
    const int64_t nSize = nSize_;
    const int64_t blkProcessNum = blkProcessNum_;
    const int64_t halfUbBytes = ubSize_ / BUFFER_NUM;
    const int64_t halfUbElems = halfUbBytes / DTYPE_SIZE;
    const int64_t nAlignSize = Ops::Base::CeilAlign(nSize, BLOCK_NUM);

    int64_t curIdx = curIdx_;
    uint16_t bufId = 0;

    for (int64_t i = 0; i < blkProcessNum; i++) {
        event_t evt = (bufId == 0) ? EVT_ID0 : EVT_ID1;

        WaitFlag<HardEvent::MTE3_MTE2>(evt);

        int64_t mIdx = curIdx / gUBCount;
        int64_t gIdx = curIdx - mIdx * gUBCount;

        int64_t processMNum;
        int64_t processGNum;
        if (mUBCount > 1 && gUBCount > 1 && mIdx == mUBCount - 1 && gIdx == gUBCount - 1) {
            processMNum = mUBFactorTail;
            processGNum = gUBFactorTail;
        } else if (mUBCount > 1 && mIdx == mUBCount - 1) {
            processMNum = mUBFactorTail;
            processGNum = gUBFactor;
        } else if (gUBCount > 1 && gIdx == gUBCount - 1) {
            processMNum = mUBFactor;
            processGNum = gUBFactorTail;
        } else {
            processMNum = mUBFactor;
            processGNum = gUBFactor;
        }

        int64_t gmOffset = gIdx * gUBFactor * nSize + mIdx * mUBFactor * gSize * nSize;

        LocalTensor<T> srcLocal(TPosition::VECIN, static_cast<uint32_t>(bufId * halfUbBytes),
                                static_cast<uint32_t>(halfUbElems));

        int64_t srcStride = 0;
        int64_t dstStride = nAlignSize * (processMNum - 1);
        loopMode_.loop1Size = processMNum;
        loopMode_.loop2Size = 1;
        loopMode_.loop1SrcStride = gSize * nSize * DTYPE_SIZE;
        loopMode_.loop1DstStride = nAlignSize * DTYPE_SIZE;
        SetLoopModePara(loopMode_, DataCopyMVType::OUT_TO_UB);
        copyInParam_.blockCount = processGNum;
        copyInParam_.blockLen = nSize * DTYPE_SIZE;
        copyInParam_.srcStride = srcStride * DTYPE_SIZE;
        copyInParam_.dstStride = dstStride * DTYPE_SIZE / BLOCK_SIZE;
#if defined(__DAV_920R1__) || defined(__DAV_920R2__)
        DataCopyPad(srcLocal, xGm_[gmOffset], copyInParam_, padParam_, true);
#else
        DataCopyPad(srcLocal, xGm_[gmOffset], copyInParam_, padParam_);
#endif
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);

        SetFlag<HardEvent::MTE2_MTE3>(evt);
        WaitFlag<HardEvent::MTE2_MTE3>(evt);

        for (int64_t g = 0; g < processGNum; g++) {
            int64_t outOffset = mIdx * mUBFactor * nSize;
            int64_t yGMIdx = gIdx * gUBFactor + g;
            int64_t ubOffset = nAlignSize * processMNum * g;
            yGm_.SetGlobalBuffer(GetTensorAddr(yGMIdx));
            copyOutParam_.blockCount = processMNum;
            copyOutParam_.blockLen = nSize * DTYPE_SIZE;
            copyOutParam_.srcStride = 0;
            copyOutParam_.dstStride = 0;
            DataCopyPad(yGm_[outOffset], srcLocal[ubOffset], copyOutParam_);
        }

        SetFlag<HardEvent::MTE3_MTE2>(evt);

        bufId ^= 1;
        curIdx++;
    }

    WaitFlag<HardEvent::MTE3_MTE2>(EVT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVT_ID1);
    curIdx_ = curIdx;
}

template <typename T>
__aicore__ inline __gm__ T* SplitVPureCopyModeSameLen<T>::GetTensorAddr(int64_t index)
{
    return inputList_.GetDataPtr<T>(index);
}

} // namespace SplitV
#endif // SPLIT_V_PURE_COPY_SAME_LEN_H
