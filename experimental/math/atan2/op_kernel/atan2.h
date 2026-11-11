/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file atan2.h
 * \brief
 */
#ifndef ATAN2_H
#define ATAN2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "atan2_tiling_data.h"
#include "atan2_tiling_key.h"

namespace NsAtan2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

constexpr float CONST_PI = 3.14159265358979323846f;
constexpr uint32_t COMPARE_ALIGN_ELEMS = 256 / sizeof(float);

template <typename T>
class Atan2 {
public:
    __aicore__ inline Atan2(){};

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const Atan2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX1; // y arg of atan2(y,x)
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX2; // x arg
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    // float32 scratch buffers
    TBuf<QuePosition::VECCALC> tmpBuf0, tmpBuf1, tmpBuf2, tmpBuf3, tmpBuf4;
    // uint8_t temp space for AscendC::Atan internal use
    TBuf<QuePosition::VECCALC> atanTmpBuf;

    GlobalTensor<T> inputGMX1;
    GlobalTensor<T> inputGMX2;
    GlobalTensor<T> outputGMY;

    uint32_t coreDataNum = 0;
    uint32_t tileNum = 0;
    uint32_t tileDataNum = 0;
    uint32_t tailDataNum = 0;
    uint32_t processDataNum = 0;
    uint32_t atanTmpSize = 0; // byte size of atanTmpBuf per tile, from tiling
};

// ── Init ───────────────────────────────────────────────────────────────────

template <typename T>
__aicore__ inline void Atan2<T>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const Atan2TilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t coreIdx = AscendC::GetBlockIdx();
    uint32_t globalBufferIndex = tilingData->bigCoreDataNum * coreIdx;

    this->tileDataNum = tilingData->tileDataNum;
    this->atanTmpSize = tilingData->atanTmpSize;

    if (coreIdx < (uint32_t)tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) *
                             (coreIdx - (uint32_t)tilingData->tailBlockNum);
    }

    inputGMX1.SetGlobalBuffer((__gm__ T*)x1 + globalBufferIndex, this->coreDataNum);
    inputGMX2.SetGlobalBuffer((__gm__ T*)x2 + globalBufferIndex, this->coreDataNum);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + globalBufferIndex, this->coreDataNum);

    pipe.InitBuffer(inputQueueX1, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(inputQueueX2, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(tmpBuf0, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(tmpBuf1, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(tmpBuf2, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(tmpBuf3, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(tmpBuf4, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(atanTmpBuf, this->atanTmpSize);
}

// ── CopyIn / CopyOut ───────────────────────────────────────────────────────

template <typename T>
__aicore__ inline void Atan2<T>::CopyIn(int32_t progress)
{
    LocalTensor<T> x1Local = inputQueueX1.AllocTensor<T>();
    LocalTensor<T> x2Local = inputQueueX2.AllocTensor<T>();
    AscendC::DataCopy(x1Local, inputGMX1[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(x2Local, inputGMX2[progress * this->tileDataNum], this->processDataNum);
    inputQueueX1.EnQue(x1Local);
    inputQueueX2.EnQue(x2Local);
}

template <typename T>
__aicore__ inline void Atan2<T>::CopyOut(int32_t progress)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    AscendC::DataCopy(outputGMY[progress * this->tileDataNum], yLocal, this->processDataNum);
    outputQueueY.FreeTensor(yLocal);
}

// ── Compute ────────────────────────────────────────────────────────────────
//
// Reuse float scratch as comparison masks after computing the base angle;
// no additional UB allocation is required for the zero boundary handling.

template <typename T>
__aicore__ inline void Atan2<T>::Compute(int32_t progress)
{
    LocalTensor<T> yLocal = inputQueueX1.DeQue<T>();
    LocalTensor<T> xLocal = inputQueueX2.DeQue<T>();
    LocalTensor<T> outLocal = outputQueueY.AllocTensor<T>();

    LocalTensor<float> tmp0 = tmpBuf0.Get<float>();
    LocalTensor<float> tmp1 = tmpBuf1.Get<float>();
    LocalTensor<float> tmp2 = tmpBuf2.Get<float>();
    LocalTensor<float> tmp3 = tmpBuf3.Get<float>();
    LocalTensor<float> tmp4 = tmpBuf4.Get<float>();
    LocalTensor<uint8_t> atanTmp = atanTmpBuf.Get<uint8_t>();

    uint32_t n = this->processDataNum;
    uint32_t compareNum = (n + COMPARE_ALIGN_ELEMS - 1U) / COMPARE_ALIGN_ELEMS * COMPARE_ALIGN_ELEMS;
    if (compareNum > n) {
        // CompareScalar requires a complete 256-byte input repeat. Initialize
        // that final repeat before copying/casting the valid input elements.
        uint32_t lastRepeat = compareNum - COMPARE_ALIGN_ELEMS;
        AscendC::Duplicate(tmp0[lastRepeat], 0.0f, COMPARE_ALIGN_ELEMS);
        AscendC::Duplicate(tmp1[lastRepeat], 0.0f, COMPARE_ALIGN_ELEMS);
        if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value) {
            AscendC::Duplicate(tmp4[lastRepeat], 0.0f, COMPARE_ALIGN_ELEMS);
        }
    }

    // ── Step 1: cast inputs to float32 → tmp0(yF), tmp1(xF) ──────────────
    if constexpr (AscendC::Std::is_same<T, half>::value || AscendC::Std::is_same<T, bfloat16_t>::value) {
        AscendC::Cast(tmp0, yLocal, RoundMode::CAST_NONE, n);
        AscendC::Cast(tmp1, xLocal, RoundMode::CAST_NONE, n);
    } else {
        // Copy the bits so that the sign of a negative zero is preserved.
        AscendC::DataCopy(tmp0, yLocal.template ReinterpretCast<float>(), n);
        AscendC::DataCopy(tmp1, xLocal.template ReinterpretCast<float>(), n);
    }

    // Atan requires distinct source and destination tensors.
    AscendC::Div(tmp2, tmp0, tmp1, n);
    AscendC::Abs(tmp2, tmp2, n);
    AscendC::Atan(tmp4, tmp2, atanTmp, n);

    LocalTensor<uint8_t> mask = tmp2.ReinterpretCast<uint8_t>();
    LocalTensor<uint8_t> xZeroMask = tmp3.ReinterpretCast<uint8_t>();
    AscendC::CompareScalar(mask, tmp1, 0.0f, CMPMODE::LT, compareNum);
    AscendC::Muls(tmp3, tmp4, -1.0f, n);
    AscendC::Adds(tmp3, tmp3, CONST_PI, n);
    AscendC::Select(tmp4, mask, tmp3, tmp4, SELMODE::VSEL_TENSOR_TENSOR_MODE, n);

    // Save the origin mask before reusing xF to form an exact +/-1 sign.
    AscendC::CompareScalar(mask, tmp0, 0.0f, CMPMODE::EQ, compareNum);
    AscendC::CompareScalar(xZeroMask, tmp1, 0.0f, CMPMODE::EQ, compareNum);
    AscendC::And(mask.ReinterpretCast<uint16_t>(), mask.ReinterpretCast<uint16_t>(),
                 xZeroMask.ReinterpretCast<uint16_t>(), (n + 15U) / 16U);
    AscendC::ShiftRight(tmp1.ReinterpretCast<uint32_t>(), tmp0.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(31),
                        n);
    AscendC::Cast(tmp1, tmp1.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, n);
    AscendC::Muls(tmp1, tmp1, -2.0f, n);
    AscendC::Adds(tmp1, tmp1, 1.0f, n);
    AscendC::Mul(tmp4, tmp4, tmp1, n);

    // Select the documented origin result: multiplying NaN by zero would
    // leave the 0/0 from the division above as NaN.
    AscendC::Duplicate(tmp0, 0.0f, n);
    AscendC::Select(tmp4, mask, tmp0, tmp4, SELMODE::VSEL_TENSOR_TENSOR_MODE, n);

    // Cast the float32 result back to the input dtype.
    if constexpr (AscendC::Std::is_same<T, half>::value || AscendC::Std::is_same<T, bfloat16_t>::value) {
        AscendC::Cast(outLocal, tmp4, RoundMode::CAST_ROUND, n);
    } else {
        AscendC::DataCopy(outLocal.template ReinterpretCast<float>(), tmp4, n);
    }

    if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value) {
        // Preserve the NaN result independently of float-to-BF16 conversion.
        // Select only copies bits: treating BF16 storage as half here does
        // not convert the quiet-NaN encoding or the rounded finite values.
        AscendC::Compare(mask, tmp4, tmp4, CMPMODE::EQ, compareNum);
        AscendC::Not(mask.ReinterpretCast<uint16_t>(), mask.ReinterpretCast<uint16_t>(), (compareNum + 15U) / 16U);
        AscendC::Duplicate(tmp0.ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0x7fc0), n);
        AscendC::Select(outLocal.template ReinterpretCast<half>(), mask, tmp0.ReinterpretCast<half>(),
                        outLocal.template ReinterpretCast<half>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
    }

    outputQueueY.EnQue<T>(outLocal);
    inputQueueX1.FreeTensor(yLocal);
    inputQueueX2.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Atan2<T>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;

    for (int32_t i = 0; i < loopCount; i++) {
        if (i == this->tileNum - 1) {
            this->processDataNum = this->tailDataNum;
        }
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

} // namespace NsAtan2
#endif // ATAN2_H
