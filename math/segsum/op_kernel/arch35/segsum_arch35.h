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
 * \file segsum_arch35.h (arch35 / Ascend950)
 * \brief arch35 implementation of Segsum.
 *
 * y[i][j] = exp(sum_{p = j + 1}^{i} x_p) for j <= i, y[i][j] = 0 for j > i.
 * Column-wise the value follows the recurrence c_i[j] = c_{i-1}[j] + x_i (j < i), c_i[i] = 0,
 * so one fp32 carry vector walks the rows top down and no cross core state is needed.
 */

#ifndef SEGSUM_ARCH35_H
#define SEGSUM_ARCH35_H

#include <type_traits>
#include "kernel_operator.h"
#include "segsum_tiling_data.h"

namespace SegsumArch35 {
using namespace AscendC;

constexpr float ZERO_FLOAT = 0;
constexpr int32_t NO_BUFFER_NUM = 1;

constexpr Reg::CastTrait EXP_FP32_TO_S32_RINT = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                 Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr Reg::CastTrait EXP_S32_TO_FP32 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING,
                                            RoundMode::CAST_NONE};

// Preserve the carry input. Normal range reduction keeps the polynomial near
// one; exceptional, overflow and subnormal ranges retain the original Exp.
__simd_callee__ inline void ShortExpAccurate(Reg::RegTensor<float>& result, Reg::RegTensor<float>& input,
                                             Reg::MaskReg& mask)
{
    Reg::RegTensor<float> absolute, zero, safe, scaled, powerFloat, negativePower, reduced, coefficient, polynomial;
    Reg::RegTensor<int32_t> power;
    Reg::MaskReg normal;
    Reg::Exp(result, input, mask);
    Reg::Abs(absolute, input, mask);
    Reg::CompareScalar<float, CMPMODE::LE>(normal, absolute, 80.0f, mask);
    Reg::Duplicate(zero, 0.0f, mask);
    Reg::Select(safe, input, zero, normal);
    Reg::Muls(scaled, safe, 1.4426950216293335f, mask);
    Reg::Cast<int32_t, float, EXP_FP32_TO_S32_RINT>(power, scaled, mask);
    Reg::Cast<float, int32_t, EXP_S32_TO_FP32>(powerFloat, power, mask);
    Reg::Muls(negativePower, powerFloat, -1.0f, mask);
    Reg::Muls(reduced, safe, 1.0f, mask);
    Reg::Duplicate(coefficient, 0.693145751953125f, mask);
    Reg::MulAddDst(reduced, negativePower, coefficient, mask);
    Reg::Duplicate(coefficient, 1.428606765330187e-6f, mask);
    Reg::MulAddDst(reduced, negativePower, coefficient, mask);
    Reg::Duplicate(polynomial, 2.4801587642286904e-5f, mask);
    Reg::Duplicate(coefficient, 1.9841270113829523e-4f, mask);
    Reg::MulDstAdd(polynomial, reduced, coefficient, mask);
    Reg::Duplicate(coefficient, 1.3888889225199819e-3f, mask);
    Reg::MulDstAdd(polynomial, reduced, coefficient, mask);
    Reg::Duplicate(coefficient, 8.333333767950535e-3f, mask);
    Reg::MulDstAdd(polynomial, reduced, coefficient, mask);
    Reg::Duplicate(coefficient, 4.16666679084301e-2f, mask);
    Reg::MulDstAdd(polynomial, reduced, coefficient, mask);
    Reg::Duplicate(coefficient, 1.666666716337204e-1f, mask);
    Reg::MulDstAdd(polynomial, reduced, coefficient, mask);
    Reg::Duplicate(coefficient, 0.5f, mask);
    Reg::MulDstAdd(polynomial, reduced, coefficient, mask);
    Reg::Duplicate(coefficient, 1.0f, mask);
    Reg::MulDstAdd(polynomial, reduced, coefficient, mask);
    Reg::Duplicate(coefficient, 1.0f, mask);
    Reg::MulDstAdd(polynomial, reduced, coefficient, mask);
    Reg::Adds(power, power, 127, mask);
    Reg::ShiftLefts(power, power, static_cast<int16_t>(23), mask);
    Reg::Mul(polynomial, polynomial, (Reg::RegTensor<float>&)power, mask);
    Reg::Select(result, polynomial, result, normal);
}

__simd_vf__ inline void ComputeAccurateExpVf(__ubuf__ float* output, __ubuf__ float* input, uint32_t count)
{
    constexpr uint32_t lanes = VECTOR_REG_WIDTH / sizeof(float);
    const uint16_t loops = static_cast<uint16_t>((count + lanes - 1) / lanes);
    Reg::RegTensor<int32_t> signedIndex;
    Reg::RegTensor<float> value, result;
    Reg::Arange(signedIndex, 0);
    for (uint16_t block = 0; block < loops; ++block) {
        auto mask = Reg::UpdateMask<float>(count);
        Reg::Gather(value, input + block * lanes, (Reg::RegTensor<uint32_t>&)signedIndex, mask);
        ShortExpAccurate(result, value, mask);
        Reg::Scatter(output + block * lanes, result, (Reg::RegTensor<uint32_t>&)signedIndex, mask);
    }
}

// Each lane owns one sequence. Preserve the original per-column addition order.
template <bool PRECISE_EXP>
__simd_vf__ inline void ComputeShortBatchVf(__ubuf__ float* x, __ubuf__ float* y, uint32_t count, uint32_t length)
{
    constexpr uint32_t lanes = VECTOR_REG_WIDTH / sizeof(float);
    const uint16_t loops = static_cast<uint16_t>((count + lanes - 1) / lanes);
    Reg::RegTensor<int32_t> lane;
    Reg::RegTensor<uint32_t> inputBase, outputBase, inputIndex, outputIndex;
    Reg::RegTensor<float> one, sum, value, result;
    auto full = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::Arange(lane, 0);
    Reg::Muls(inputBase, (Reg::RegTensor<uint32_t>&)lane, length, full);
    Reg::Muls(outputBase, (Reg::RegTensor<uint32_t>&)lane, length * length, full);
    Reg::Duplicate(one, 1.0f, full);
    for (uint16_t i = 0; i < loops; ++i) {
        auto mask = Reg::UpdateMask<float>(count);
        auto xTile = x + i * lanes * length;
        auto yTile = y + i * lanes * length * length;
        for (uint16_t col = 0; col < length; ++col) {
            Reg::Adds(outputIndex, outputBase, col * (length + 1), mask);
            Reg::Scatter(yTile, one, outputIndex, mask);
            Reg::Duplicate(sum, 0.0f, mask);
            for (uint16_t row = col + 1; row < length; ++row) {
                Reg::Adds(inputIndex, inputBase, static_cast<uint32_t>(row), mask);
                Reg::Gather(value, xTile, inputIndex, mask);
                Reg::Add(sum, sum, value, mask);
                if constexpr (PRECISE_EXP) {
                    ShortExpAccurate(result, sum, mask);
                } else {
                    Reg::Exp(result, sum, mask);
                }
                Reg::Adds(outputIndex, outputBase, row * length + col, mask);
                Reg::Scatter(yTile, result, outputIndex, mask);
            }
        }
    }
}

template <typename T, int32_t MODE>
class Segsum {
public:
    TPipe pipe;

    __aicore__ inline Segsum(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SegsumTilingDataArch35* tilingData);
    __aicore__ inline void Process();

private:
    static constexpr bool IS_FLOAT = std::is_same<T, float>::value;

    template <typename T1>
    __aicore__ inline T1 Min(T1 a, T1 b) const
    {
        return a < b ? a : b;
    };

    // The fp32 view of x aliases the input buffer for float and the cast buffer otherwise.
    __aicore__ inline LocalTensor<float> GetXFloat()
    {
        if constexpr (IS_FLOAT) {
            return xBuf.Get<float>();
        } else {
            return xFloatBuf.Get<float>();
        }
    }

    // For float the output queue tensor is the fp32 exp buffer itself, no extra buffer is needed.
    __aicore__ inline LocalTensor<float> GetExpTensor(LocalTensor<T> yTensor)
    {
        if constexpr (IS_FLOAT) {
            return yTensor.template ReinterpretCast<float>();
        } else {
            return expBuf.Get<float>();
        }
    }

    __aicore__ inline void ParseTilingData(const SegsumTilingDataArch35* tilingData);
    __aicore__ inline void LoadX(int64_t offset, int64_t num);
    __aicore__ inline void ComputeRowBlocks();
    __aicore__ inline void ComputeShortBatch();
    __aicore__ inline void ComputeColumnStripes();
    __aicore__ inline void CopyOutRows(int64_t offset, int64_t rows, uint32_t bytesPerRow);

private:
    TBuf<QuePosition::VECCALC> xBuf;
    TBuf<QuePosition::VECCALC> xFloatBuf;
    TBuf<QuePosition::VECCALC> lastBuf;
    TBuf<QuePosition::VECCALC> curBuf;
    TBuf<QuePosition::VECCALC> expBuf;
    TQue<QuePosition::VECOUT, NO_BUFFER_NUM> yQue;

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;

    int64_t blockIdx = 0;
    int64_t batchStart = 0;
    int64_t batchEnd = 0;
    int64_t needCoreNum = 0;
    int64_t tailDimLength = 0;
    int64_t rowLen = 0;
    int64_t rowNum = 0;
    int64_t stripeLen = 0;
};

template <typename T, int32_t MODE>
__aicore__ inline void Segsum<T, MODE>::ParseTilingData(const SegsumTilingDataArch35* tilingData)
{
    needCoreNum = tilingData->needCoreNum;
    tailDimLength = tilingData->tailDimLength;
    rowLen = tilingData->rowLen;
    rowNum = tilingData->rowNum;
    stripeLen = tilingData->stripeLen;

    int64_t averageBatches = tilingData->averageBatches;
    batchStart = blockIdx * averageBatches;
    batchEnd = Min(batchStart + averageBatches, tilingData->batches);
}

template <typename T, int32_t MODE>
__aicore__ inline void Segsum<T, MODE>::Init(GM_ADDR x, GM_ADDR y, const SegsumTilingDataArch35* tilingData)
{
    blockIdx = GetBlockIdx();
    ParseTilingData(tilingData);
    if (blockIdx >= needCoreNum || tailDimLength == 0 || batchStart >= batchEnd) {
        return;
    }

    xGm.SetGlobalBuffer((__gm__ T*)x);
    yGm.SetGlobalBuffer((__gm__ T*)y);

    if constexpr (MODE == 1) {
        if (rowNum > tailDimLength) {
            if (tailDimLength > 1) {
                pipe.InitBuffer(xBuf, rowNum * sizeof(T));
                if constexpr (!IS_FLOAT) {
                    pipe.InitBuffer(xFloatBuf, rowNum * sizeof(float));
                    pipe.InitBuffer(expBuf, rowNum * tailDimLength * sizeof(float));
                }
            }
            pipe.InitBuffer(yQue, NO_BUFFER_NUM, rowNum * tailDimLength * sizeof(T));
            return;
        }
        pipe.InitBuffer(xBuf, rowLen * sizeof(T));
        if constexpr (!IS_FLOAT) {
            pipe.InitBuffer(xFloatBuf, rowLen * sizeof(float));
            pipe.InitBuffer(expBuf, rowNum * rowLen * sizeof(float));
        }
        pipe.InitBuffer(lastBuf, rowLen * sizeof(float));
        pipe.InitBuffer(curBuf, rowNum * rowLen * sizeof(float));
        pipe.InitBuffer(yQue, NO_BUFFER_NUM, rowNum * rowLen * sizeof(T));
    } else {
        pipe.InitBuffer(xBuf, SEGSUM_X_CHUNK_ARCH35 * sizeof(T));
        if constexpr (!IS_FLOAT) {
            pipe.InitBuffer(xFloatBuf, SEGSUM_X_CHUNK_ARCH35 * sizeof(float));
            pipe.InitBuffer(expBuf, stripeLen * sizeof(float));
        }
        pipe.InitBuffer(lastBuf, stripeLen * sizeof(float));
        pipe.InitBuffer(curBuf, stripeLen * sizeof(float));
        pipe.InitBuffer(yQue, NO_BUFFER_NUM, stripeLen * sizeof(T));
    }
}

template <typename T, int32_t MODE>
__aicore__ inline void Segsum<T, MODE>::Process()
{
    if (blockIdx >= needCoreNum || tailDimLength == 0 || batchStart >= batchEnd) {
        return;
    }
    if constexpr (MODE == 1) {
        if (rowNum > tailDimLength) {
            ComputeShortBatch();
            return;
        }
        ComputeRowBlocks();
    } else {
        ComputeColumnStripes();
    }
}

/*
 * Loads num elements of x starting at offset and, for half/bfloat16, casts them to the fp32 view.
 * The scalar reads that follow need the data visible to the scalar unit, hence the explicit flags.
 */
template <typename T, int32_t MODE>
__aicore__ inline void Segsum<T, MODE>::LoadX(int64_t offset, int64_t num)
{
    LocalTensor<T> xTensor = xBuf.Get<T>();
    // Guard the refill against readers of the previous chunk.
    event_t eventVMte2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventVMte2);
    WaitFlag<HardEvent::V_MTE2>(eventVMte2);
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(num * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xTensor, xGm[offset], copyParams, padParams);

    if constexpr (IS_FLOAT) {
        event_t eventMte2S = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventMte2S);
        WaitFlag<HardEvent::MTE2_S>(eventMte2S);
    } else {
        event_t eventMte2V = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2V);
        WaitFlag<HardEvent::MTE2_V>(eventMte2V);
        LocalTensor<float> xFloat = xFloatBuf.Get<float>();
        Cast(xFloat, xTensor, RoundMode::CAST_NONE, num);
        event_t eventVS = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
    }
}

template <typename T, int32_t MODE>
__aicore__ inline void Segsum<T, MODE>::CopyOutRows(int64_t offset, int64_t rows, uint32_t bytesPerRow)
{
    LocalTensor<T> yTensor = yQue.DeQue<T>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(rows), bytesPerRow, 0, 0, 0};
    DataCopyPad(yGm[offset], yTensor, copyParams);
    yQue.FreeTensor(yTensor);
}

template <typename T, int32_t MODE>
__aicore__ inline void Segsum<T, MODE>::ComputeShortBatch()
{
    const int64_t tile = rowNum / tailDimLength;
    if (tailDimLength == 1) {
        // The 1x1 output is exp(0) regardless of the input, including NaN/Inf.
        LocalTensor<T> ones = yQue.AllocTensor<T>();
        Duplicate(ones, static_cast<T>(1.0f), static_cast<int32_t>(tile));
        yQue.EnQue(ones);
        ones = yQue.DeQue<T>();
        for (int64_t start = batchStart; start < batchEnd; start += tile) {
            const uint32_t count = static_cast<uint32_t>(Min(tile, batchEnd - start));
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm[start], ones, copyParams);
        }
        yQue.FreeTensor(ones);
        return;
    }
    LocalTensor<float> xFloat = GetXFloat();
    for (int64_t start = batchStart; start < batchEnd; start += tile) {
        const uint32_t count = static_cast<uint32_t>(Min(tile, batchEnd - start));
        LoadX(tailDimLength * start, tailDimLength * count);
        LocalTensor<T> yTensor = yQue.AllocTensor<T>();
        LocalTensor<float> yFloat = GetExpTensor(yTensor);
        const uint32_t outputCount = static_cast<uint32_t>(tailDimLength * tailDimLength * count);
        Duplicate(yFloat, 0.0f, outputCount);
        PipeBarrier<PIPE_V>();
        asc_vf_call<ComputeShortBatchVf<IS_FLOAT>>((__ubuf__ float*)xFloat.GetPhyAddr(),
                                                   (__ubuf__ float*)yFloat.GetPhyAddr(), count,
                                                   static_cast<uint32_t>(tailDimLength));
        PipeBarrier<PIPE_V>();
        if constexpr (!IS_FLOAT) {
            Cast(yTensor, yFloat, RoundMode::CAST_RINT, outputCount);
            PipeBarrier<PIPE_V>();
        }
        yQue.EnQue(yTensor);
        CopyOutRows(tailDimLength * tailDimLength * start, 1, outputCount * sizeof(T));
    }
}

/*
 * TilingKey 1: a whole row fits in UB. rowNum consecutive rows are chained inside the fp32
 * accumulator buffer, exponentiated in place and written back with a single contiguous copy.
 */
template <typename T, int32_t MODE>
__aicore__ inline void Segsum<T, MODE>::ComputeRowBlocks()
{
    LocalTensor<float> lastRow = lastBuf.Get<float>();
    LocalTensor<float> curRows = curBuf.Get<float>();
    const uint32_t bytesPerRow = static_cast<uint32_t>(tailDimLength * sizeof(T));

    for (int64_t batchIdx = batchStart; batchIdx < batchEnd; batchIdx++) {
        LoadX(batchIdx * tailDimLength, tailDimLength);
        LocalTensor<float> xFloat = GetXFloat();

        Duplicate(lastRow, ZERO_FLOAT, rowLen);
        PipeBarrier<PIPE_V>();

        for (int64_t rowStart = 0; rowStart < tailDimLength; rowStart += rowNum) {
            int64_t rows = Min(rowNum, tailDimLength - rowStart);
            LocalTensor<T> yTensor = yQue.AllocTensor<T>();
            LocalTensor<float> expRows = GetExpTensor(yTensor);

            // Zeroing the accumulator supplies the diagonal zero of every row, zeroing the exp
            // buffer supplies the strict upper triangle zeros of the output block.
            Duplicate(curRows, ZERO_FLOAT, rows * rowLen);
            Duplicate(expRows, ZERO_FLOAT, rows * rowLen);
            PipeBarrier<PIPE_V>();

            for (int64_t r = 0; r < rows; r++) {
                int64_t rowIdx = rowStart + r;
                LocalTensor<float> curRow = curRows[r * rowLen];
                if (rowIdx > 0) {
                    LocalTensor<float> prevRow = (r == 0) ? lastRow : curRows[(r - 1) * rowLen];
                    Adds(curRow, prevRow, xFloat.GetValue(rowIdx), rowIdx);
                    PipeBarrier<PIPE_V>();
                }
                if constexpr (IS_FLOAT) {
                    asc_vf_call<ComputeAccurateExpVf>((__ubuf__ float*)expRows[r * rowLen].GetPhyAddr(),
                                                      (__ubuf__ float*)curRow.GetPhyAddr(), rowIdx + 1);
                } else {
                    Exp(expRows[r * rowLen], curRow, rowIdx + 1);
                }
                PipeBarrier<PIPE_V>();
            }

            if constexpr (!IS_FLOAT) {
                Cast(yTensor, expRows, RoundMode::CAST_RINT, rows * rowLen);
                PipeBarrier<PIPE_V>();
            }
            yQue.EnQue(yTensor);
            CopyOutRows((batchIdx * tailDimLength + rowStart) * tailDimLength, rows, bytesPerRow);

            // Carry the last computed row over to the next block.
            DataCopy(lastRow, curRows[(rows - 1) * rowLen], rowLen);
            PipeBarrier<PIPE_V>();
        }
    }
}

/*
 * TilingKey 0: a row does not fit in UB. Columns are cut into stripes; every stripe keeps its own
 * fp32 carry and walks all rows top down, so stripes stay independent and no GM state is needed.
 */
template <typename T, int32_t MODE>
__aicore__ inline void Segsum<T, MODE>::ComputeColumnStripes()
{
    LocalTensor<float> lastRow = lastBuf.Get<float>();
    LocalTensor<float> curRow = curBuf.Get<float>();

    for (int64_t batchIdx = batchStart; batchIdx < batchEnd; batchIdx++) {
        for (int64_t colStart = 0; colStart < tailDimLength; colStart += stripeLen) {
            int64_t cols = Min(stripeLen, tailDimLength - colStart);
            int64_t chunkBase = -1;

            Duplicate(lastRow, ZERO_FLOAT, cols);
            PipeBarrier<PIPE_V>();

            for (int64_t rowIdx = 0; rowIdx < tailDimLength; rowIdx++) {
                // Columns of this stripe that sit strictly left of the diagonal.
                int64_t accCols = rowIdx > colStart ? Min(rowIdx - colStart, cols) : 0;
                // Valid prefix of the output: accumulated columns plus the diagonal when it is inside.
                int64_t validCols = rowIdx < colStart ? 0 : Min(rowIdx - colStart + 1, cols);

                LocalTensor<T> yTensor = yQue.AllocTensor<T>();
                LocalTensor<float> expRow = GetExpTensor(yTensor);

                Duplicate(curRow, ZERO_FLOAT, cols);
                Duplicate(expRow, ZERO_FLOAT, cols);
                PipeBarrier<PIPE_V>();

                if (accCols > 0) {
                    if (chunkBase < 0 || rowIdx >= chunkBase + SEGSUM_X_CHUNK_ARCH35) {
                        chunkBase = rowIdx / SEGSUM_X_CHUNK_ARCH35 * SEGSUM_X_CHUNK_ARCH35;
                        LoadX(batchIdx * tailDimLength + chunkBase,
                              Min(SEGSUM_X_CHUNK_ARCH35, tailDimLength - chunkBase));
                    }
                    LocalTensor<float> xFloat = IS_FLOAT ? xBuf.Get<float>() : xFloatBuf.Get<float>();
                    Adds(curRow, lastRow, xFloat.GetValue(rowIdx - chunkBase), accCols);
                    PipeBarrier<PIPE_V>();
                }
                if (validCols > 0) {
                    if constexpr (IS_FLOAT) {
                        asc_vf_call<ComputeAccurateExpVf>((__ubuf__ float*)expRow.GetPhyAddr(),
                                                          (__ubuf__ float*)curRow.GetPhyAddr(), validCols);
                    } else {
                        Exp(expRow, curRow, validCols);
                    }
                    PipeBarrier<PIPE_V>();
                }

                if constexpr (!IS_FLOAT) {
                    Cast(yTensor, expRow, RoundMode::CAST_RINT, cols);
                    PipeBarrier<PIPE_V>();
                }
                yQue.EnQue(yTensor);
                CopyOutRows((batchIdx * tailDimLength + rowIdx) * tailDimLength + colStart, 1,
                            static_cast<uint32_t>(cols * sizeof(T)));

                // stripeLen is 32B aligned, cols may not be; copying the whole stripe keeps the
                // UB to UB copy aligned and the trailing values are never read back.
                DataCopy(lastRow, curRow, stripeLen);
                PipeBarrier<PIPE_V>();
            }
        }
    }
}
} // namespace SegsumArch35
#endif // SEGSUM_ARCH35_H
