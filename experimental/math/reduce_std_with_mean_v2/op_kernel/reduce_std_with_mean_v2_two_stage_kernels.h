/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef REDUCE_STD_WITH_MEAN_WELFORD_TWO_STAGE_KERNELS_H
#define REDUCE_STD_WITH_MEAN_WELFORD_TWO_STAGE_KERNELS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "reduce_std_with_mean_v2_tiling_data.h"

/*!
 * two-stage cross-core reduction :
 *
 * Stage 1 (ReduceStdWithMeanV2Partial): blockDim = K. Core k reduces
 * every row's R segment [k*partialSeg, min((k+1)*partialSeg, R)) with the
 * same in-tile two-pass Welford as the serial path (tileSum -> tileMean ->
 * diff^2 -> tileM2 -> scalar Chan merge across tiles) and writes the local
 * (n, mean, m2) triple to partial[m*K*3 + k*3 + {0,1,2}] (float32).
 *
 * Stage 2 (ReduceStdWithMeanV2Merge): reads the K triples of each row
 * and Chan-merges them (the same formula the chunked Welford uses between
 * tiles -- exact for any partition of the reduce dim), then applies
 * correction/eps/invert and writes std/mean as float32 (the L2 layer Casts
 * to the original input dtype). blockDim = min(M, coreNum); each core
 * handles a contiguous row range.
 *
 * Math: Chan parallel merge is exact for arbitrary subset splits, so the
 * two-stage result equals the single-kernel chunked Welford up to fp32
 * summation order (~1e-7 relative).
 */

namespace NsReduceStdWithMeanV2 {
using namespace AscendC;

template <typename T, int schMode>
class ReduceStdWithMeanV2Partial {
    static constexpr int32_t BUFFER_NUM = 2;

public:
    __aicore__ inline ReduceStdWithMeanV2Partial() {}
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR partial, GM_ADDR workspace,
                                const ReduceStdWithMeanV2TilingData* tiling)
    {
        totalNonReduce_ = tiling->totalNonReduce;
        reduceLength_ = tiling->reduceLength;
        ubLength_ = tiling->ubLength;
        partialK_ = tiling->partialK;
        partialSeg_ = tiling->partialSeg;
        if (ubLength_ < 1) {
            ubLength_ = 1;
        }
        if (partialK_ < 1) {
            partialK_ = 1;
        }
        if (partialSeg_ < 1) {
            partialSeg_ = reduceLength_;
        }

        selfGM.SetGlobalBuffer((__gm__ T*)self, totalNonReduce_ * reduceLength_);
        partialGM.SetGlobalBuffer((__gm__ float*)partial, totalNonReduce_ * partialK_ * 3);

        pipe.InitBuffer(inQueueSelf, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(reduceBuf, 8 * sizeof(float));
        pipe.InitBuffer(scratchBuf, ubLength_ * sizeof(float));
        pipe.InitBuffer(workBuf, ubLength_ * sizeof(float));
        if constexpr (!std::is_same_v<T, float>)
            pipe.InitBuffer(castBuf, ubLength_ * sizeof(float));
        // (n, mean, m2) staging for the UB->GM partial write (12 bytes, GM
        // dst only needs 1-byte alignment; UB src must be 32B aligned).
        pipe.InitBuffer(outQueuePartial, BUFFER_NUM, 16 * sizeof(float));
    }
    __aicore__ inline void Process();

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueSelf;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueuePartial;
    TBuf<TPosition::VECCALC> reduceBuf, scratchBuf, workBuf, castBuf;
    GlobalTensor<T> selfGM;
    GlobalTensor<float> partialGM;
    int64_t totalNonReduce_, reduceLength_, ubLength_, partialK_, partialSeg_;
};

template <typename T, int schMode>
__aicore__ inline void ReduceStdWithMeanV2Partial<T, schMode>::Process()
{
    if (totalNonReduce_ <= 0 || reduceLength_ <= 0) {
        return;
    }
    const int64_t k = AscendC::GetBlockIdx();
    const int64_t segStart = k * partialSeg_;
    if (segStart >= reduceLength_) {
        return;
    }
    int64_t segLen = partialSeg_;
    if (segStart + segLen > reduceLength_) {
        segLen = reduceLength_ - segStart;
    }
    const int64_t numTiles = (segLen + ubLength_ - 1) / ubLength_;

    for (int64_t m = 0; m < totalNonReduce_; ++m) {
        int64_t count = 0;
        float mean = 0.0f;
        float m2 = 0.0f;

        for (int64_t t = 0; t < numTiles; ++t) {
            int64_t curLen = ((t + 1) * ubLength_ <= segLen) ? ubLength_ : (segLen - t * ubLength_);
            uint32_t uCurLen = static_cast<uint32_t>(curLen);

            LocalTensor<T> sLocal = inQueueSelf.AllocTensor<T>();
            AscendC::DataCopyPad(sLocal, selfGM[m * reduceLength_ + segStart + t * ubLength_],
                                 {1, static_cast<uint16_t>(uCurLen * sizeof(T)), 0, 0}, {false, 0, 0, 0});
            inQueueSelf.EnQue(sLocal);
            sLocal = inQueueSelf.DeQue<T>();

            LocalTensor<float> dBuf = reduceBuf.Get<float>();
            LocalTensor<float> tBuf = scratchBuf.Get<float>();
            LocalTensor<float> wBuf = workBuf.Get<float>();

            float tileSum;
            float tileM2;
            if constexpr (std::is_same_v<T, float>) {
                // tileSum = sum(tile)
                ReduceSum(dBuf, sLocal, tBuf, uCurLen);
                PipeBarrier<PIPE_V>();
                tileSum = dBuf.GetValue(0);
                // tBuf = tileMean - x (sign-free; squared below)
                Muls(tBuf, sLocal, -1.0f, uCurLen);
                PipeBarrier<PIPE_V>();
                Adds(tBuf, tBuf, tileSum / static_cast<float>(curLen), uCurLen);
                PipeBarrier<PIPE_V>();
                // tileM2 = sum(diff^2)
                Mul(wBuf, tBuf, tBuf, uCurLen);
                PipeBarrier<PIPE_V>();
                ReduceSum(dBuf, wBuf, tBuf, uCurLen);
                PipeBarrier<PIPE_V>();
                tileM2 = dBuf.GetValue(0);
            } else {
                // fp16/bf16: upcast to fp32 before compute
                LocalTensor<float> fBuf = castBuf.Get<float>();
                Cast(fBuf, sLocal, RoundMode::CAST_NONE, uCurLen);
                PipeBarrier<PIPE_V>();
                ReduceSum(dBuf, fBuf, tBuf, uCurLen);
                PipeBarrier<PIPE_V>();
                tileSum = dBuf.GetValue(0);
                Muls(tBuf, fBuf, -1.0f, uCurLen);
                PipeBarrier<PIPE_V>();
                Adds(tBuf, tBuf, tileSum / static_cast<float>(curLen), uCurLen);
                PipeBarrier<PIPE_V>();
                Mul(wBuf, tBuf, tBuf, uCurLen);
                PipeBarrier<PIPE_V>();
                ReduceSum(dBuf, wBuf, tBuf, uCurLen);
                PipeBarrier<PIPE_V>();
                tileM2 = dBuf.GetValue(0);
            }

            // scalar Chan parallel merge (same formula as the serial path)
            float tileMean = tileSum / static_cast<float>(curLen);
            float delta = tileMean - mean;
            count += curLen;
            mean += delta * (static_cast<float>(curLen) / static_cast<float>(count));
            m2 += tileM2 + delta * delta * static_cast<float>(curLen) * static_cast<float>(count - curLen) /
                               static_cast<float>(count);

            inQueueSelf.FreeTensor(sLocal);
        }

        // write (n=segLen, mean, m2) of row m, segment k
        LocalTensor<float> oLocal = outQueuePartial.AllocTensor<float>();
        oLocal.SetValue(0, static_cast<float>(segLen));
        oLocal.SetValue(1, mean);
        oLocal.SetValue(2, m2);
        outQueuePartial.EnQue(oLocal);
        LocalTensor<float> oData = outQueuePartial.DeQue<float>();
        AscendC::DataCopyExtParams outCp{1, 12, 0, 0, 0};
        AscendC::DataCopyPad(partialGM[m * partialK_ * 3 + k * 3], oData, outCp);
        outQueuePartial.FreeTensor(oData);
    }
}

template <typename T, int schMode>
class ReduceStdWithMeanV2Merge {
    static constexpr int32_t BUFFER_NUM = 2;
    // K <= coreNum (40 on 910B); 64 leaves room for future chips.
    static constexpr int32_t MAX_K = 64;

public:
    __aicore__ inline ReduceStdWithMeanV2Merge() {}
    __aicore__ inline void Init(GM_ADDR partial, GM_ADDR output, GM_ADDR meanOut, GM_ADDR workspace,
                                const ReduceStdWithMeanV2TilingData* tiling)
    {
        totalNonReduce_ = tiling->totalNonReduce;
        reduceLength_ = tiling->reduceLength;
        partialK_ = tiling->partialK;
        correction_ = tiling->correction;
        eps_ = tiling->eps;
        invert_ = tiling->invert;
        if (partialK_ < 1) {
            partialK_ = 1;
        }
        if (partialK_ > MAX_K) {
            partialK_ = MAX_K;
        }

        int64_t blockIdx = AscendC::GetBlockIdx();
        int64_t blockNum = AscendC::GetBlockNum();
        if (blockNum < 1) {
            blockNum = 1;
        }
        rowsPerCore_ = (totalNonReduce_ + blockNum - 1) / blockNum;
        coreStartM_ = blockIdx * rowsPerCore_;
        int64_t rem = totalNonReduce_ - coreStartM_;
        coreM_ = (rem > rowsPerCore_) ? rowsPerCore_ : rem;
        if (coreM_ < 0) {
            coreM_ = 0;
        }

        partialGM.SetGlobalBuffer((__gm__ float*)partial, totalNonReduce_ * partialK_ * 3);
        outputGM.SetGlobalBuffer((__gm__ float*)output + coreStartM_, coreM_);
        meanOutGM.SetGlobalBuffer((__gm__ float*)meanOut + coreStartM_, coreM_);

        pipe.InitBuffer(outQueueY, BUFFER_NUM, sizeof(T));
        pipe.InitBuffer(outQueueMean, BUFFER_NUM, sizeof(T));
        pipe.InitBuffer(reduceBuf, 8 * sizeof(float));
        pipe.InitBuffer(tripleBuf, MAX_K * 3 * sizeof(float));
    }
    __aicore__ inline void Process();

private:
    __aicore__ inline void WriteOutput(int64_t m, float stdVal);
    __aicore__ inline void WriteMeanOutput(int64_t m, float meanVal);

    TPipe pipe;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueY;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueMean;
    TBuf<TPosition::VECCALC> reduceBuf, tripleBuf;
    GlobalTensor<float> partialGM;
    GlobalTensor<float> outputGM, meanOutGM;
    int64_t totalNonReduce_, reduceLength_, partialK_;
    int64_t correction_;
    float eps_;
    bool invert_;
    int64_t rowsPerCore_, coreStartM_, coreM_;
};

template <typename T, int schMode>
__aicore__ inline void ReduceStdWithMeanV2Merge<T, schMode>::Process()
{
    if (coreM_ <= 0 || partialK_ <= 0) {
        return;
    }
    const int64_t K = partialK_;
    LocalTensor<float> triples = tripleBuf.Get<float>();

    for (int64_t m = 0; m < coreM_; ++m) {
        // load row (coreStartM_ + m)'s K triples: K*3 floats contiguous
        AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(K * 12), 0, 0, 0};
        AscendC::DataCopyPad(triples, partialGM[(coreStartM_ + m) * K * 3], cp, {false, 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        // Chan parallel merge over the K segment partials (exact partition)
        int64_t count = 0;
        float mean = 0.0f;
        float m2 = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
            float nF = triples.GetValue(3 * k);
            float meanI = triples.GetValue(3 * k + 1);
            float m2I = triples.GetValue(3 * k + 2);
            if (!(nF > 0.0f)) {
                continue;
            }
            float delta = meanI - mean;
            count += static_cast<int64_t>(nF);
            float countF = static_cast<float>(count);
            mean += delta * (nF / countF);
            m2 += m2I + delta * delta * nF * (countF - nF) / countF;
        }

        float denom = static_cast<float>(count) - static_cast<float>(correction_);
        if (denom < 0.0f) {
            denom = 0.0f;
        }
        float var = (denom > 0.0f) ? (m2 / denom) : 0.0f;
        float stdVal;
        if (invert_) {
            float stdTmp = sqrt(var + eps_);
            stdVal = (stdTmp > 0.0f) ? (1.0f / stdTmp) : 0.0f;
        } else {
            stdVal = sqrt(var + eps_);
        }
        WriteOutput(m, stdVal);
        WriteMeanOutput(m, mean);
    }
}

// The Merge kernel always outputs float32 (the partial tensor is fp32 and
// DTYPE_SELF would be float anyway); the L2 layer Casts to the original
// input dtype where needed (same as the RegBase path).
template <typename T, int schMode>
__aicore__ inline void ReduceStdWithMeanV2Merge<T, schMode>::WriteOutput(int64_t m, float stdVal)
{
    LocalTensor<float> outLocal = outQueueY.AllocTensor<float>();
    outLocal.SetValue(0, stdVal);
    outQueueY.EnQue(outLocal);
    LocalTensor<float> outData = outQueueY.DeQue<float>();
    AscendC::DataCopyExtParams outCp{1, 4, 0, 0, 0};
    AscendC::DataCopyPad(outputGM[m], outData, outCp);
    outQueueY.FreeTensor(outData);
}

template <typename T, int schMode>
__aicore__ inline void ReduceStdWithMeanV2Merge<T, schMode>::WriteMeanOutput(int64_t m, float meanVal)
{
    LocalTensor<float> outLocal = outQueueMean.AllocTensor<float>();
    outLocal.SetValue(0, meanVal);
    outQueueMean.EnQue(outLocal);
    LocalTensor<float> outData = outQueueMean.DeQue<float>();
    AscendC::DataCopyExtParams outCp{1, 4, 0, 0, 0};
    AscendC::DataCopyPad(meanOutGM[m], outData, outCp);
    outQueueMean.FreeTensor(outData);
}

} // namespace NsReduceStdWithMeanV2
#endif
