/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPLIT_V_UB_SPLIT_SAME_LEN_DEINTERLEAVE_H
#define SPLIT_V_UB_SPLIT_SAME_LEN_DEINTERLEAVE_H
#include "kernel_operator_list_tensor_intf.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace SplitV {
using namespace AscendC;
template <typename T>
class SplitVUbSplitSameLenDeinterleave {
public:
    __aicore__ inline SplitVUbSplitSameLenDeinterleave(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SplitVScalarTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline __gm__ T* GetTensorAddr(int64_t index);
    __aicore__ inline void CopyIn(int64_t blockLen, int64_t srcOffset, LocalTensor<T>& xUb);
    __aicore__ inline void CopyOut(LocalTensor<T>& yLocal, int64_t blockLen, int64_t dstOffset);

private:
    constexpr static uint32_t BUFFER_NUM = 2;
    constexpr static int64_t BLOCK_ELENUM = Ops::Base::GetUbBlockSize() / sizeof(T);
    constexpr static uint32_t BLOCK_BYTES = Ops::Base::GetUbBlockSize();
    constexpr static int64_t DTYPE_SIZE = sizeof(T);
    // 手动双缓冲同步(免TPipe/TQue): in 复用(V_MTE2)、in 就绪(MTE2_V)、out slot 复用(MTE3_V)、
    // out 就绪(V_MTE3); 不同 HardEvent 类型同号 id 物理独立
    constexpr static event_t EVT_ID0 = static_cast<event_t>(0);
    constexpr static event_t EVT_ID1 = static_cast<event_t>(1);
    constexpr static event_t EVT_ID2 = static_cast<event_t>(2);

    const SplitVScalarTilingData* tilingData_ = nullptr;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    ListTensorDesc inputList_;
    int32_t blockIdx_ = 0;
    int64_t gSize_ = 0;
    int64_t nSize_ = 0;
    int64_t mUBCount_ = 0;
    int64_t mUBFactor_ = 0;
    int64_t mUBFactorTail_ = 0;
    int64_t blockFactor_ = 0;
    int64_t blockFactorTail_ = 0;
    int64_t blkProcessNum_ = 0;
    int64_t curIdx_ = 0;

    // 手动UB布局: [in0][in1][slot0: y0|y1][slot1: y0|y1], 各子块32B对齐
    uint32_t inXOffset0_ = 0;
    uint32_t inXOffset1_ = 0;
    uint32_t outY0Offset0_ = 0; // slot0 中 y0/y1 起始偏移
    uint32_t outY1Offset0_ = 0;
    uint32_t outY0Offset1_ = 0; // slot1 中 y0/y1 起始偏移
    uint32_t outY1Offset1_ = 0;
    uint32_t inBufElems_ = 0;
    uint32_t outBufElems_ = 0;
};

template <typename T>
__aicore__ inline void SplitVUbSplitSameLenDeinterleave<T>::Init(GM_ADDR x, GM_ADDR y,
                                                                 const SplitVScalarTilingData* tilingData)
{
    AscendC::InitSocState();
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;
    gSize_ = tilingData_->gSize;
    nSize_ = tilingData_->nBlockFactorNum;
    mUBFactor_ = tilingData_->mBlockFactor;
    mUBFactorTail_ = tilingData_->mBlockFactorTail;
    mUBCount_ = tilingData_->mBlockCount;
    blockFactor_ = tilingData_->blockFactor;
    blockFactorTail_ = tilingData_->blockFactorTail;

    const uint32_t inSpaceBytes = static_cast<uint32_t>(gSize_ * mUBFactor_ * DTYPE_SIZE);
    const uint32_t inSpaceAlign = Ops::Base::CeilAlign(inSpaceBytes, BLOCK_BYTES);
    const uint32_t outSpaceBytes = static_cast<uint32_t>(CeilDivision(mUBFactor_, BLOCK_ELENUM) * BLOCK_ELENUM *
                                                         DTYPE_SIZE);
    inXOffset0_ = 0;
    inXOffset1_ = inXOffset0_ + inSpaceAlign;
    outY0Offset0_ = inXOffset1_ + inSpaceAlign;
    outY1Offset0_ = outY0Offset0_ + outSpaceBytes;
    outY0Offset1_ = outY1Offset0_ + outSpaceBytes;
    outY1Offset1_ = outY0Offset1_ + outSpaceBytes;
    inBufElems_ = inSpaceBytes / static_cast<uint32_t>(DTYPE_SIZE);
    outBufElems_ = outSpaceBytes / static_cast<uint32_t>(DTYPE_SIZE);

    xGm_.SetGlobalBuffer((__gm__ T*)x);
    inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(y));

    blkProcessNum_ = blockFactor_;
    curIdx_ = blockIdx_ * blockFactor_;
    if (blockIdx_ < blockFactorTail_) {
        blkProcessNum_ += 1;
        curIdx_ += blockIdx_;
    } else {
        curIdx_ += blockFactorTail_;
    }
}

template <typename T>
__aicore__ inline void SplitVUbSplitSameLenDeinterleave<T>::Process()
{
    if (blockIdx_ >= tilingData_->realCoreNum) {
        return;
    }

    SetFlag<HardEvent::V_MTE2>(EVT_ID0);
    SetFlag<HardEvent::V_MTE2>(EVT_ID1);
    SetFlag<HardEvent::MTE3_V>(EVT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVT_ID1);

    const int64_t mUBCount = mUBCount_;
    const int64_t mUBFactor = mUBFactor_;
    const int64_t mUBFactorTail = mUBFactorTail_;
    const int64_t gSize = gSize_;
    const int64_t nSize = nSize_;
    const int64_t blkProcessNum = blkProcessNum_;

    int64_t curIdx = curIdx_;
    uint16_t bufId = 0;

    for (int64_t i = 0; i < blkProcessNum; i++) {
        event_t evt = (bufId == 0) ? EVT_ID0 : EVT_ID1;

        int64_t mIdx = curIdx;
        int64_t processMNum = (mIdx == mUBCount - 1) ? mUBFactorTail : mUBFactor;
        int64_t srcCount = processMNum * gSize;
        int64_t gmOffset = mIdx * mUBFactor * gSize * nSize;
        int64_t dstOffset = mIdx * mUBFactor * nSize;

        // out slot 复用: 上一轮同 slot MTE3(拷出)读完 -> 本轮 V(DeInterleave) 可覆盖写
        WaitFlag<HardEvent::MTE3_V>(evt);
        uint32_t y0Offset = (bufId == 0) ? outY0Offset0_ : outY0Offset1_;
        uint32_t y1Offset = (bufId == 0) ? outY1Offset0_ : outY1Offset1_;
        LocalTensor<T> yLocal0(TPosition::VECOUT, y0Offset, outBufElems_);
        LocalTensor<T> yLocal1(TPosition::VECOUT, y1Offset, outBufElems_);

        // in 复用: 上一轮同块 V 读完 -> 本轮 MTE2 可覆盖写
        WaitFlag<HardEvent::V_MTE2>(evt);
        uint32_t inXOffset = (bufId == 0) ? inXOffset0_ : inXOffset1_;
        LocalTensor<T> srcLocal(TPosition::VECIN, inXOffset, inBufElems_);

        CopyIn(srcCount, gmOffset, srcLocal);

        // in 就绪: MTE2 写完 -> V(DeInterleave) 读
        SetFlag<HardEvent::MTE2_V>(evt);
        WaitFlag<HardEvent::MTE2_V>(evt);

        AscendC::DeInterleave(yLocal0, yLocal1, srcLocal, srcCount);

        // V 读完 in -> 下一轮同块 MTE2 可覆盖写
        SetFlag<HardEvent::V_MTE2>(evt);
        // out 就绪: V 写完 y0/y1 -> MTE3 读出
        SetFlag<HardEvent::V_MTE3>(EVT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVT_ID2);

        yGm_.SetGlobalBuffer(GetTensorAddr(0));
        CopyOut(yLocal0, processMNum, dstOffset);
        yGm_.SetGlobalBuffer(GetTensorAddr(1));
        CopyOut(yLocal1, processMNum, dstOffset);

        // out slot 释放: MTE3 读完 -> 下一轮同 slot V 可覆盖写
        SetFlag<HardEvent::MTE3_V>(evt);

        bufId ^= 1;
        curIdx++;
    }

    // 收尾: 排空仍在途的复用事件
    WaitFlag<HardEvent::V_MTE2>(EVT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVT_ID1);
    curIdx_ = curIdx;
}

template <typename T>
__aicore__ inline void SplitVUbSplitSameLenDeinterleave<T>::CopyIn(int64_t blockLen, int64_t srcOffset,
                                                                   LocalTensor<T>& xUb)
{
    DataCopyExtParams copyInParam{1, static_cast<uint32_t>(blockLen * DTYPE_SIZE), 0, 0, 0};
    DataCopyPadExtParams<T> padParam{false, 0, 0, 0};
    DataCopyPad(xUb, xGm_[srcOffset], copyInParam, padParam);
}

template <typename T>
__aicore__ inline void SplitVUbSplitSameLenDeinterleave<T>::CopyOut(LocalTensor<T>& yLocal, int64_t blockLen,
                                                                    int64_t dstOffset)
{
    DataCopyExtParams copyOutParam{1, static_cast<uint32_t>(blockLen * DTYPE_SIZE), 0, 0, 0};
    DataCopyPad(yGm_[dstOffset], yLocal, copyOutParam);
}

template <typename T>
__aicore__ inline __gm__ T* SplitVUbSplitSameLenDeinterleave<T>::GetTensorAddr(int64_t index)
{
    return inputList_.GetDataPtr<T>(index);
}

} // namespace SplitV
#endif // SPLIT_V_UB_SPLIT_SAME_LEN_DEINTERLEAVE_H
