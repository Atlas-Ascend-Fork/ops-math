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
 * \file transpose_nddma_big_dim.h
 * \brief TilingKey=10005 NDDMA_BIG_DIM 策略实现
 *
 * 适用场景：维度 > 5（超过 NDDMA 5维限制），需要将原始 shape 压缩到5维 NDDMA 格式。
 *
 * 核心策略：
 *   - Host 侧 FlushBaseNumForBigDim() 预计算 nddmaIdx 映射：
 *     将原始轴索引压缩到5维 NDDMA 格式，nddmaIdx[i] 记录5维中第i维对应的原始轴
 *   - Kernel 侧 SetLoopInfo() 根据 nddmaIdx 和 perm[outCutIndex] 设置5维 NDDMA 参数
 *   - nddmaFlag[i] 标记每维的状态：
 *     = 1  → 该维保持原始大小（位于输出切分轴之后，UB内完整保留）
 *     = -1 → 该维被切分（即为输出切分轴对应的输入轴）
 *     = 0  → 该维大小为1（位于输出切分轴之前，已合并到更高级别）
 *
 * 数据流：GM → DataCopy<T,5>(UB, NDDMA) → DataCopyPad(GM, 整块写出)
 * Main/Tail 判断同 NDDMA_CUT_1，按 outCutLoopSize 判断。
 */
#ifndef TRANSPOSE_NDDMA_BIG_DIM_H
#define TRANSPOSE_NDDMA_BIG_DIM_H

#include "transpose_base.h"

namespace Transpose {
using namespace AscendC;

/**
 * @brief NDDMA_BIG_DIM 策略类，实现 >5维压缩到5维 NDDMA 的转置
 *
 * @tparam T 数据元素类型
 */
template <typename T>
class TransposeNddmaBigDim : public TransposeBase<T> {
public:
    __aicore__ inline TransposeNddmaBigDim(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TransposeNddmaBigDimTilingData* data);
    __aicore__ inline void Process();

private:
    __aicore__ inline void DecimalToMixed(int64_t num, int64_t bases[], int64_t mixedBase[]);
    __aicore__ inline void SetLoopInfo(NdDmaLoopInfo<TP_NDDMA_MAX_DIM_NUM>& loopInfo);
    __aicore__ inline void CopyIn(int64_t loopIdx, NdDmaParams<T, TP_NDDMA_MAX_DIM_NUM>& params);
    __aicore__ inline void CopyOut(int64_t loopIdx, DataCopyExtParams& copyOutParams);
    __aicore__ inline void ProcessPerCore(NdDmaParams<T, TP_NDDMA_MAX_DIM_NUM>& paramsMain,
                                          NdDmaParams<T, TP_NDDMA_MAX_DIM_NUM>& paramsTail,
                                          DataCopyExtParams& copyOutParamsMain, DataCopyExtParams& copyOutParamsTail);

private:
    int64_t blockIdx_;

    // buffer
    LocalTensor<T> local_[2];
    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;

    // tiling params
    const TransposeNddmaBigDimTilingData* tiling_;

    // core params
    int64_t blkProcessNum_ = 0;
    int64_t blkProcessIdxStart_ = 0;
    int64_t blkProcessIdxEnd_ = 0;

    // addressOffset params
    int64_t dstAddressOffsetMixedBase_[TP_MAX_AXIS_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t dstAddressOffsetMixedBaseRes_[TP_MAX_AXIS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t dstLoopStride_[TP_MAX_AXIS_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t srcLoopStride_[TP_MAX_AXIS_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};

    int64_t cutIdx_ = 0;
    uint32_t bufferNum_ = 2;
};

template <typename T>
__aicore__ inline void TransposeNddmaBigDim<T>::Init(GM_ADDR x, GM_ADDR y, const TransposeNddmaBigDimTilingData* data)
{
    blockIdx_ = GetBlockIdx();
    tiling_ = data;
    bufferNum_ = data->bufferNum == 0 ? 1 : data->bufferNum;
    inputGM_.SetGlobalBuffer((__gm__ T*)x);
    outputGM_.SetGlobalBuffer((__gm__ T*)y);
    const uint32_t ubBytes = data->ubBytes;
    for (uint32_t i = 0; i < bufferNum_; ++i) {
        local_[i] = LocalTensor<T>(TPosition::VECIN, i * ubBytes, ubBytes / sizeof(T));
    }
}

template <typename T>
__aicore__ inline void TransposeNddmaBigDim<T>::DecimalToMixed(int64_t num, int64_t bases[], int64_t mixedBase[])
{
    if (num == 0) {
        return;
    }
    for (int64_t i = 0; i < tiling_->permSize; i++) {
        mixedBase[i] = num % bases[i];
        num /= bases[i];
        if (num == 0) {
            break;
        }
    }
}

/**
 * @brief 设置 NDDMA 5维搬运的 LoopInfo 参数
 *
 * 根据 nddmaIdx 映射和 perm[outCutIndex] 设置5维 NDDMA 参数。
 * nddmaFlag[i] 标记每维的状态：
 *   - nddmaFlag[i] == 1：该维在输出切分轴之后，保持原始大小
 *   - nddmaFlag[i] == -1：该维是输出切分轴对应的输入轴，使用 outUbFactor 作为 loopSize
 *   - nddmaFlag[i] == 0：该维在输出切分轴之前，大小为1（不参与循环）
 */
template <typename T>
__aicore__ inline void TransposeNddmaBigDim<T>::SetLoopInfo(NdDmaLoopInfo<TP_NDDMA_MAX_DIM_NUM>& loopInfo)
{
    int64_t nddmaFlag[TP_NDDMA_MAX_DIM_NUM] = {0};
    for (int64_t i = 0; i < TP_NDDMA_MAX_DIM_NUM; i++) {
        if (tiling_->nddmaIdx[i] == tiling_->perm[tiling_->outCutIndex]) {
            nddmaFlag[i] = -1;
        }
        for (int64_t j = tiling_->outCutIndex + 1; j < tiling_->permSize; j++) {
            if (tiling_->perm[j] == tiling_->nddmaIdx[i]) {
                nddmaFlag[i] = 1;
            }
        }
    }

    for (int64_t i = TP_NDDMA_MAX_DIM_NUM - 1; i >= 0; i--) {
        if (nddmaFlag[TP_NDDMA_MAX_DIM_NUM - 1 - i] == 1) {
            loopInfo.loopSize[i] = tiling_->inputShape[tiling_->nddmaIdx[TP_NDDMA_MAX_DIM_NUM - 1 - i]];
            loopInfo.loopSrcStride[i] = tiling_->baseInShape[tiling_->nddmaIdx[TP_NDDMA_MAX_DIM_NUM - 1 - i]];
            loopInfo.loopDstStride[i] = tiling_->baseNddmaShape[TP_NDDMA_MAX_DIM_NUM - 1 - i];
        } else if (nddmaFlag[TP_NDDMA_MAX_DIM_NUM - 1 - i] == -1) {
            cutIdx_ = i;
            loopInfo.loopSize[i] = tiling_->outUbFactor; // not tail
            loopInfo.loopSrcStride[i] = tiling_->baseInShape[tiling_->nddmaIdx[TP_NDDMA_MAX_DIM_NUM - 1 - i]];
            loopInfo.loopDstStride[i] = tiling_->baseNddmaShape[TP_NDDMA_MAX_DIM_NUM - 1 - i];
        } else {
            loopInfo.loopSize[i] = 1;
            loopInfo.loopSrcStride[i] = 1;
            loopInfo.loopDstStride[i] = tiling_->baseNddmaShape[TP_NDDMA_MAX_DIM_NUM - 1 - i];
        }

        loopInfo.loopLpSize[i] = 0;
        loopInfo.loopRpSize[i] = 0;
    }
}

template <typename T>
__aicore__ inline void TransposeNddmaBigDim<T>::Process()
{
    if (!ParseMultiCoreRange(blockIdx_, static_cast<int64_t>(GetBlockNum()), tiling_->blkFactor, tiling_->blkTailFactor,
                             blkProcessNum_, blkProcessIdxStart_, blkProcessIdxEnd_)) {
        return;
    }

    NdDmaLoopInfo<TP_NDDMA_MAX_DIM_NUM> loopInfo;
    SetLoopInfo(loopInfo);

    // calculate blockCount
    DataCopyExtParams copyOutParamsMain{1, 0, 0, 0, 0};
    copyOutParamsMain.blockLen = tiling_->totalNddmaNum * sizeof(T);

    DataCopyExtParams copyOutParamsTail{1, 0, 0, 0, 0};
    copyOutParamsTail.blockLen = tiling_->totalNddmaNum / tiling_->outUbFactor * tiling_->outTailFactor * sizeof(T);

    T constValue = 0;
    NdDmaParams<T, TP_NDDMA_MAX_DIM_NUM> paramsMain = {loopInfo, constValue};
    if (tiling_->outTailFactor != 0) {
        loopInfo.loopSize[cutIdx_] = tiling_->outTailFactor;
        loopInfo.loopDstStride[cutIdx_] = tiling_->baseNddmaShape[TP_NDDMA_MAX_DIM_NUM - 1 - cutIdx_];
    }
    NdDmaParams<T, TP_NDDMA_MAX_DIM_NUM> paramsTail = {loopInfo, constValue};
    ProcessPerCore(paramsMain, paramsTail, copyOutParamsMain, copyOutParamsTail);
}

template <typename T>
__aicore__ inline void TransposeNddmaBigDim<T>::CopyIn(int64_t loopIdx, NdDmaParams<T, TP_NDDMA_MAX_DIM_NUM>& params)
{
    int64_t srcAddressOffset = 0;
    DecimalToMixed(loopIdx, dstAddressOffsetMixedBase_, dstAddressOffsetMixedBaseRes_);
    for (int64_t i = 0; i < tiling_->permSize; i++) {
        srcAddressOffset += dstAddressOffsetMixedBaseRes_[i] *
                            srcLoopStride_[tiling_->permSize - 1 - tiling_->perm[tiling_->permSize - 1 - i]];
    }
    const uint32_t bufIdx = static_cast<uint32_t>(loopIdx % static_cast<int64_t>(bufferNum_));
    LocalTensor<T> bindLocalIn = local_[bufIdx];
    Mutex::Lock<PIPE_MTE2>(static_cast<MutexID>(bufIdx));
#if defined(__DAV_920R1__) || defined(__DAV_920R2__)
    DataCopy<T, TP_NDDMA_MAX_DIM_NUM, config>(bindLocalIn, inputGM_[srcAddressOffset], params, true);
#else
    DataCopy<T, TP_NDDMA_MAX_DIM_NUM, config>(bindLocalIn, inputGM_[srcAddressOffset], params);
#endif
    Mutex::Unlock<PIPE_MTE2>(static_cast<MutexID>(bufIdx));
}

template <typename T>
__aicore__ inline void TransposeNddmaBigDim<T>::CopyOut(int64_t loopIdx, DataCopyExtParams& copyOutParams)
{
    int64_t dstAddressOffset = 0;
    for (int64_t i = 0; i < tiling_->permSize; i++) {
        dstAddressOffset += dstAddressOffsetMixedBaseRes_[i] * dstLoopStride_[i];
    }
    const uint32_t bufIdx = static_cast<uint32_t>(loopIdx % static_cast<int64_t>(bufferNum_));
    LocalTensor<T> bindLocalOut = local_[bufIdx];
    Mutex::Lock<PIPE_MTE3>(static_cast<MutexID>(bufIdx));
    DataCopyPad(outputGM_[dstAddressOffset], bindLocalOut, copyOutParams);
    Mutex::Unlock<PIPE_MTE3>(static_cast<MutexID>(bufIdx));
}

template <typename T>
__aicore__ inline void TransposeNddmaBigDim<T>::ProcessPerCore(NdDmaParams<T, TP_NDDMA_MAX_DIM_NUM>& paramsMain,
                                                               NdDmaParams<T, TP_NDDMA_MAX_DIM_NUM>& paramsTail,
                                                               DataCopyExtParams& copyOutParamsMain,
                                                               DataCopyExtParams& copyOutParamsTail)
{
    int64_t outCutLoopSize = Ops::Base::CeilDiv(tiling_->outputShape[tiling_->outCutIndex],
                                                static_cast<int64_t>(tiling_->outUbFactor));
    const int64_t permSize = tiling_->permSize;
    for (int64_t i = 0; i < permSize; i++) {
        if (permSize - 1 - i == tiling_->outCutIndex) {
            dstAddressOffsetMixedBase_[i] = outCutLoopSize;
        }
        if (permSize - 1 - i < tiling_->outCutIndex) {
            dstAddressOffsetMixedBase_[i] = tiling_->outputShape[permSize - 1 - i];
        }

        if (i >= 1) {
            dstLoopStride_[i] = dstLoopStride_[i - 1] * tiling_->outputShape[permSize - i];
            srcLoopStride_[i] = srcLoopStride_[i - 1] * tiling_->inputShape[permSize - i];
        }
    }
    dstLoopStride_[permSize - 1 - tiling_->outCutIndex] *= tiling_->outUbFactor;
    srcLoopStride_[permSize - 1 - tiling_->perm[tiling_->outCutIndex]] *= tiling_->outUbFactor;

    for (int64_t loopIdx = blkProcessIdxStart_; loopIdx < blkProcessIdxEnd_; loopIdx++) {
        if (tiling_->outTailFactor != 0 && (loopIdx + 1) % outCutLoopSize == 0) { // tail
            CopyIn(loopIdx, paramsTail);
            CopyOut(loopIdx, copyOutParamsTail);
        } else { // main
            CopyIn(loopIdx, paramsMain);
            CopyOut(loopIdx, copyOutParamsMain);
        }
    }
}

} // namespace Transpose

#endif // TRANSPOSE_NDDMA_BIG_DIM_H
