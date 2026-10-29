/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef REDUCE_STD_WITH_MEAN_WELFORD_H
#define REDUCE_STD_WITH_MEAN_WELFORD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "reduce_std_with_mean_v2_tiling_data.h"
#include "reduce_std_with_mean_v2_tiling_key.h"

namespace NsReduceStdWithMeanV2 {
using namespace AscendC;

template <typename T, int schMode>
class ReduceStdWithMeanV2 {
    static constexpr int32_t BUFFER_NUM = 2;

public:
    __aicore__ inline ReduceStdWithMeanV2() {}
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR output, GM_ADDR meanOut, GM_ADDR workspace,
                                const ReduceStdWithMeanV2TilingData* tiling)
    {
        totalNonReduce_ = tiling->totalNonReduce;
        reduceLength_ = tiling->reduceLength;
        blockFactor_ = tiling->blockFactor;
        ubLength_ = tiling->ubLength;
        correction_ = tiling->correction;
        eps_ = tiling->eps;
        invert_ = tiling->invert;
        vectorize_ = tiling->vectorize;
        ubM_ = tiling->ubM;

        int64_t blockIdx = AscendC::GetBlockIdx();
        coreStartM_ = blockIdx * blockFactor_;
        int64_t rem = totalNonReduce_ - coreStartM_;
        coreM_ = (rem > blockFactor_) ? blockFactor_ : rem;
        if (coreM_ < 0) {
            coreM_ = 0;
        }

        selfGM.SetGlobalBuffer((__gm__ T*)self + coreStartM_ * reduceLength_, coreM_ * reduceLength_);
        outputGM.SetGlobalBuffer((__gm__ T*)output + coreStartM_, coreM_);
        meanOutGM.SetGlobalBuffer((__gm__ T*)meanOut + coreStartM_, coreM_);

        // Output queues are shared by both paths (WriteOutput/WriteMeanOutput).
        pipe.InitBuffer(outQueueY, BUFFER_NUM, sizeof(T));
        pipe.InitBuffer(outQueueMean, BUFFER_NUM, sizeof(T));

        if (vectorize_) {
            // M-vectorized path buffers (DESIGN_MVECTOR.md §4), fp32 compute
            // domain. Every batch is [ubM_, reduceLength_] (single tile per
            // row, P1). +64 elements slack: the last BlockReduceSum repeat
            // reads up to 64-R elements past the batch (masked off, but keep
            // slack).
            // tmpBuf holds the pass1 block sums of one batch: one 32B slot
            // (8 floats) per segment, 8*S per row (S = R/64 segments for
            // R > 64, S = 1 for R <= 64), plus the same 64-element slack.
            // Note 8*S = max(R/8, 8): R <= 64 rows still occupy a full 32B
            // slot, so the size is NOT R/8 there.
            const int64_t rowElems = ubM_ * reduceLength_;
            const int64_t segElems = ubM_ * ((reduceLength_ > 64) ? (reduceLength_ / 8) : 8);
            pipe.InitBuffer(vecDataBuf, (rowElems + 64) * sizeof(float));
            pipe.InitBuffer(vecMeanBuf, (rowElems + 64) * sizeof(float));
            if constexpr (!std::is_same_v<T, float>)
                pipe.InitBuffer(vecRawBuf, rowElems * sizeof(T));
            pipe.InitBuffer(tmpBuf, (segElems + 64) * sizeof(float));
            pipe.InitBuffer(vecSumBuf, ubM_ * 8 * sizeof(float));
            pipe.InitBuffer(vecM2Buf, ubM_ * 8 * sizeof(float));
            pipe.InitBuffer(vecStateBuf, ubM_ * 8 * sizeof(float));
            // WriteOutput/WriteMeanOutput use reduceBuf as the fp32 staging
            // buffer for the half/bf16 CAST_ROUND output path.
            pipe.InitBuffer(reduceBuf, 8 * sizeof(float));
        } else {
            pipe.InitBuffer(inQueueSelf, BUFFER_NUM, ubLength_ * sizeof(T));
            pipe.InitBuffer(reduceBuf, 8 * sizeof(float));
            pipe.InitBuffer(scratchBuf, ubLength_ * sizeof(float));
            pipe.InitBuffer(workBuf, ubLength_ * sizeof(float));
            if constexpr (!std::is_same_v<T, float>)
                pipe.InitBuffer(castBuf, ubLength_ * sizeof(float));
        }
    }
    __aicore__ inline void Process();
    __aicore__ inline void ProcessVectorized();

private:
    __aicore__ inline void WriteOutput(int64_t m, float stdVal);
    __aicore__ inline void WriteMeanOutput(int64_t m, float meanVal);

    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueSelf;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueY;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueMean;
    TBuf<TPosition::VECCALC> reduceBuf, scratchBuf, workBuf, castBuf;
    // M-vectorized path buffers (allocated only when vectorize_ is true).
    TBuf<TPosition::VECCALC> vecDataBuf, vecMeanBuf, vecRawBuf;
    TBuf<TPosition::VECCALC> tmpBuf, vecSumBuf, vecM2Buf, vecStateBuf;
    GlobalTensor<T> selfGM, outputGM, meanOutGM;

    int64_t totalNonReduce_, reduceLength_, blockFactor_, ubLength_;
    int64_t correction_;
    float eps_;
    bool invert_;
    bool vectorize_;
    int64_t ubM_;
    int64_t coreStartM_, coreM_;
};

template <typename T, int schMode>
__aicore__ inline void ReduceStdWithMeanV2<T, schMode>::Process()
{
    if (vectorize_) {
        ProcessVectorized();
        return;
    }
    if (coreM_ <= 0 || reduceLength_ <= 0) {
        return;
    }
    int64_t numTiles = (reduceLength_ + ubLength_ - 1) / ubLength_;

    // Chunked Welford (Chan parallel merge): single GM pass over self,
    // per-row scalar state (count/mean/m2) updated per tile.
    for (int64_t m = 0; m < coreM_; ++m) {
        int64_t count = 0;
        float mean = 0.0f;
        float m2 = 0.0f;

        for (int64_t i = 0; i < numTiles; ++i) {
            int64_t curLen = ((i + 1) * ubLength_ <= reduceLength_) ? ubLength_ : (reduceLength_ - i * ubLength_);
            uint32_t uCurLen = static_cast<uint32_t>(curLen);

            LocalTensor<T> sLocal = inQueueSelf.AllocTensor<T>();
            AscendC::DataCopyPad(sLocal, selfGM[m * reduceLength_ + i * ubLength_],
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
                // diff = tile - tileMean (scalar broadcast).
                // Subs is RegBase-only; use Muls(-1)+Adds(tileMean) instead:
                // tBuf = -(x) + tileMean = tileMean - x, squared below (sign-free).
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

            // Chan parallel merge:
            //   delta = tileMean - mean
            //   count += curLen
            //   mean  += delta * (curLen / count)
            //   m2    += tileM2 + delta^2 * curLen * (count - curLen) / count
            float tileMean = tileSum / static_cast<float>(curLen);
            float delta = tileMean - mean;
            count += curLen;
            mean += delta * (static_cast<float>(curLen) / static_cast<float>(count));
            m2 += tileM2 + delta * delta * static_cast<float>(curLen) * static_cast<float>(count - curLen) /
                               static_cast<float>(count);

            inQueueSelf.FreeTensor(sLocal);
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

/*!
 * M-vectorized path :
 * process M_tile rows per batch with BlockReduceSum (multi-segment vcgadd)
 * instead of the serial per-row loop. Each row is a single tile of the full
 * reduce length R (tiling guarantees R <= ubLength and R <= 1920 here). For
 * R <= 64 one vcgadd repeat covers each row (v1); for R > 64 each row is
 * split into S = R/64 segments of 64 elements (P1 segmented pass1) and the
 * segment sums are merged per row with a standard ReduceSum (pass2, no
 * mask<=64 limit).
 *
 * Data flow per batch of [curM, R] fp32 rows (R = reduceLength,
 * S = R/64 segments per row for R > 64, else S = 1):
 *   1. load (fp16/bf16: rawBuf -> Cast fp32; fp32: direct). The batch
 *      [curM, R] is one contiguous GM region (rows of length R adjacent),
 *      loaded with one strided DataCopyPad (blockCount = curM,
 *      blockLen = R*typeSize bytes, srcStride = 0).
 *   2. tileSum pass1: BlockReduceSum(tmp, data, repeat=curM*S, mask=64,
 *        dstRepStride=1, srcBlkStride=1, srcRepStride=8)
 *        -> tmp[8*(m*S+s)+j] = block j partial sum of row m, segment s
 *        (R <= 64: repeat=curM, mask=R, srcRepStride=R/8, v1 shape).
 *        Tiling caps ubM <= 255/S, so repeat = curM*S <= 255 (single call).
 *   3. tileSum pass2: per-row ReduceSum(sumBuf[8m], tmp+8m*S, work, 8*S)
 *        -> row m's tileSum at the 32B slot sumBuf[8m], then packed dense
 *        into sumBuf[m] for the curM-wide steps below
 *   4. tileMean = sumBuf * (1/R)                     (curM-wide Muls)
 *   5. meanBlock: Dup per row from sumBuf[m]         (curM Dup calls)
 *   6. diff = data - meanBlock; data = diff^2        (full-batch Sub/Mul)
 *   7. tileM2: pass1' + pass2' -> m2Buf (dense)      (same as 2/3)
 *   8. Chan merge (curM-wide vectors, DESIGN §6): single tile, count = R,
 *        w1 = R/count = 1, w2 = R*(count-R)/count = 0
 *   9. per-row scalar output: std/mean -> WriteOutput/WriteMeanOutput
 */
template <typename T, int schMode>
__aicore__ inline void ReduceStdWithMeanV2<T, schMode>::ProcessVectorized()
{
    if (coreM_ <= 0 || reduceLength_ <= 0) {
        return;
    }
    const int64_t R = reduceLength_;
    // Segmentation: S = R/64 segments per row for R > 64 (tiling guarantees
    // R % 64 == 0 there), S = 1 for R <= 64 (v1 single-repeat rows).
    const int32_t S = (R > 64) ? static_cast<int32_t>(R / 64) : 1;
    const int32_t rowSegElems = 8 * S; // fp32 block sums of one row in tmp
    // pass1 params: mask = elements per repeat, srcRepStride = 32B blocks
    // between consecutive repeats. R > 64: each repeat is one 64-element
    // segment (8 blocks, srcRepStride=8). R <= 64: each repeat is one row
    // (mask=R, srcRepStride=R/8, v1 behavior -- inactive blocks pre-zeroed).
    const int32_t pass1Mask = (R > 64) ? 64 : static_cast<int32_t>(R);
    const int32_t pass1RepStride = (R > 64) ? 8 : static_cast<int32_t>(R / 8);
    const int32_t mCap = static_cast<int32_t>(ubM_);

    LocalTensor<float> dataBuf = vecDataBuf.Get<float>();
    LocalTensor<float> meanBuf = vecMeanBuf.Get<float>();
    LocalTensor<float> tmp = tmpBuf.Get<float>();
    LocalTensor<float> sumBuf = vecSumBuf.Get<float>();
    LocalTensor<float> m2Buf = vecM2Buf.Get<float>();
    LocalTensor<float> state = vecStateBuf.Get<float>(); // [0..ubM) meanVec, [ubM..2ubM) m2Vec

    for (int64_t mStart = 0; mStart < coreM_; mStart += ubM_) {
        int64_t curM64 = ((mStart + ubM_) <= coreM_) ? ubM_ : (coreM_ - mStart);
        int32_t curM = static_cast<int32_t>(curM64);
        uint32_t batchElems = static_cast<uint32_t>(curM64 * R);

        // 0. zero the per-batch merge state (meanVec at [0,mCap), m2Vec at
        //    [mCap,2mCap)). The Chan merge relies on exact arithmetic: stale
        //    state would poison delta via fp32 catastrophic cancellation, and
        //    the m2Vec region is reused by later (tail) batches, so the full
        //    capacity is zeroed every batch.
        AscendC::Duplicate(state, static_cast<float>(0.0f), 2 * mCap);
        PipeBarrier<PIPE_V>();

        // 1. load the batch [curM, R]: contiguous rows (pitch R), one strided
        //    DataCopyPad. fp16/bf16 land in rawBuf first, then Cast to fp32.
        if constexpr (std::is_same_v<T, float>) {
            AscendC::DataCopyExtParams cp{static_cast<uint16_t>(curM), static_cast<uint32_t>(R * 4), 0, 0, 0};
            AscendC::DataCopyPad(dataBuf, selfGM[mStart * R], cp, {false, 0, 0, 0});
        } else {
            LocalTensor<T> rawBuf = vecRawBuf.Get<T>();
            AscendC::DataCopyExtParams cp{static_cast<uint16_t>(curM), static_cast<uint32_t>(R * 2), 0, 0, 0};
            AscendC::DataCopyPad(rawBuf, selfGM[mStart * R], cp, {false, 0, 0, 0});
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Cast(dataBuf, rawBuf, RoundMode::CAST_NONE, batchElems);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        // 2. tileSum: pass1 (per-segment block sums) + pass2 (per-row merge).
        //    pass1 runs one vcgadd repeat per segment across the batch
        //    (repeat = curM*S <= 255 by the tiling's ubM <= 255/S cap,
        //    mask = 64), so tmp[8*(m*S+s)+j] = sum of block j of row m,
        //    segment s (rows/segments adjacent in UB -> srcRepStride=8).
        //    pass2 merges the 8*S block sums of each row with a standard
        //    ReduceSum (8*S <= 240 elements, no mask<=64 constraint); its
        //    sharedTmpBuffer is unused on 2201, meanBuf is free here.
        AscendC::Duplicate(tmp, static_cast<float>(0.0f), 8 * curM * S);
        PipeBarrier<PIPE_V>();
        AscendC::BlockReduceSum(tmp, dataBuf, curM * S, pass1Mask, 1, 1, pass1RepStride);
        PipeBarrier<PIPE_V>();
        for (int32_t m = 0; m < curM; ++m) {
            AscendC::ReduceSum(sumBuf[8 * m], tmp[8 * m * S], meanBuf, rowSegElems);
        }
        PipeBarrier<PIPE_V>();
        // dense pack: pass2 scatters the row sums at 32B slots [8m]; the
        // tileMean and the Chan merge below operate on curM-wide dense
        // vectors, so gather them in place. Forward order is safe: dense
        // slot m overlaps an unread sparse slot 8*m' only for m' = m/8 < m.
        for (int32_t m = 0; m < curM; ++m) {
            sumBuf.SetValue(m, sumBuf.GetValue(8 * m));
        }
        PipeBarrier<PIPE_V>();

        // 3. tileMean (curM-wide)
        const float invR = 1.0f / static_cast<float>(R);
        AscendC::Muls(sumBuf, sumBuf, invR, curM);
        PipeBarrier<PIPE_V>();

        // 4. meanBlock: Dup per row (no Broadcast API, DESIGN §3 step 4)
        for (int32_t m = 0; m < curM; ++m) {
            float mv = sumBuf.GetValue(m);
            AscendC::Duplicate(meanBuf[m * R], mv, static_cast<uint32_t>(R));
        }
        PipeBarrier<PIPE_V>();

        // 5. diff = data - meanBlock; data = diff^2 (full [curM, R])
        AscendC::Sub(dataBuf, dataBuf, meanBuf, batchElems);
        PipeBarrier<PIPE_V>();
        AscendC::Mul(dataBuf, dataBuf, dataBuf, batchElems);
        PipeBarrier<PIPE_V>();

        // 6. tileM2: pass1' + pass2' (tmp re-zeroed; meanBuf is free again
        //    after step 5 consumed the meanBlock, reused as ReduceSum work)
        AscendC::Duplicate(tmp, static_cast<float>(0.0f), 8 * curM * S);
        PipeBarrier<PIPE_V>();
        AscendC::BlockReduceSum(tmp, dataBuf, curM * S, pass1Mask, 1, 1, pass1RepStride);
        PipeBarrier<PIPE_V>();
        for (int32_t m = 0; m < curM; ++m) {
            AscendC::ReduceSum(m2Buf[8 * m], tmp[8 * m * S], meanBuf, rowSegElems);
        }
        PipeBarrier<PIPE_V>();
        for (int32_t m = 0; m < curM; ++m) {
            m2Buf.SetValue(m, m2Buf.GetValue(8 * m));
        }
        PipeBarrier<PIPE_V>();

        // 7. Chan merge, curM-wide vectors (DESIGN §6): single tile per row,
        //    count = R, so w1 = R/count = 1, w2 = R*(count-R)/count = 0:
        //    meanVec = tileMean, m2Vec = tileM2.
        const int64_t count = R;
        const float w1 = static_cast<float>(R) / static_cast<float>(count);
        const float w2 = static_cast<float>(R) * static_cast<float>(count - R) / static_cast<float>(count);
        // delta = tileMean - meanVec   (sumBuf holds tileMean)
        AscendC::Sub(sumBuf, sumBuf, state, curM);
        PipeBarrier<PIPE_V>();
        // m2Vec += tileM2 + delta^2 * w2
        AscendC::Mul(meanBuf, sumBuf, sumBuf, curM); // delta^2
        PipeBarrier<PIPE_V>();
        AscendC::Muls(meanBuf, meanBuf, w2, curM);
        PipeBarrier<PIPE_V>();
        AscendC::Add(meanBuf, meanBuf, m2Buf, curM); // + tileM2
        PipeBarrier<PIPE_V>();
        AscendC::Add(state[mCap], state[mCap], meanBuf, curM);
        PipeBarrier<PIPE_V>();
        // meanVec += delta * w1
        AscendC::Muls(sumBuf, sumBuf, w1, curM);
        PipeBarrier<PIPE_V>();
        AscendC::Add(state, state, sumBuf, curM);
        PipeBarrier<PIPE_V>();

        // 8. per-row output (denom over the full R)
        float denom = static_cast<float>(R) - static_cast<float>(correction_);
        if (denom < 0.0f) {
            denom = 0.0f;
        }
        for (int64_t m = 0; m < curM; ++m) {
            float meanVal = state.GetValue(m);
            float m2Val = state.GetValue(mCap + m);
            float var = (denom > 0.0f) ? (m2Val / denom) : 0.0f;
            float stdVal;
            if (invert_) {
                float stdTmp = sqrt(var + eps_);
                stdVal = (stdTmp > 0.0f) ? (1.0f / stdTmp) : 0.0f;
            } else {
                stdVal = sqrt(var + eps_);
            }
            WriteOutput(mStart + m, stdVal);
            WriteMeanOutput(mStart + m, meanVal);
        }
    }
}

template <typename T, int schMode>
__aicore__ inline void ReduceStdWithMeanV2<T, schMode>::WriteOutput(int64_t m, float stdVal)
{
    LocalTensor<T> outLocal = outQueueY.AllocTensor<T>();
    if constexpr (std::is_same_v<T, float>) {
        outLocal.SetValue(0, stdVal);
    } else {
        LocalTensor<float> fBuf = reduceBuf.Get<float>();
        fBuf.SetValue(0, stdVal);
        // CAST_RINT (RNE) matches torch's fp32->bf16/fp16 conversion (IEEE
        // round-to-nearest-even). CAST_ROUND (away-from-zero) flips 1 ulp at
        // quantization tie boundaries, which ATK flags on bf16/fp16 means.
        Cast(outLocal, fBuf, RoundMode::CAST_RINT, 1);
    }
    outQueueY.EnQue(outLocal);
    LocalTensor<T> outData = outQueueY.DeQue<T>();
    AscendC::DataCopyExtParams outCp{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(outputGM[m], outData, outCp);
    outQueueY.FreeTensor(outData);
}

template <typename T, int schMode>
__aicore__ inline void ReduceStdWithMeanV2<T, schMode>::WriteMeanOutput(int64_t m, float meanVal)
{
    LocalTensor<T> outLocal = outQueueMean.AllocTensor<T>();
    if constexpr (std::is_same_v<T, float>) {
        outLocal.SetValue(0, meanVal);
    } else {
        LocalTensor<float> fBuf = reduceBuf.Get<float>();
        fBuf.SetValue(0, meanVal);
        Cast(outLocal, fBuf, RoundMode::CAST_RINT, 1);
    }
    outQueueMean.EnQue(outLocal);
    LocalTensor<T> outData = outQueueMean.DeQue<T>();
    AscendC::DataCopyExtParams outCp{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(meanOutGM[m], outData, outCp);
    outQueueMean.FreeTensor(outData);
}

} // namespace NsReduceStdWithMeanV2
#endif
