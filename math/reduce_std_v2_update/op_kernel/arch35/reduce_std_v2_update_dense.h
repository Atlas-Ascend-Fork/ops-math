/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_STD_V2_UPDATE_DENSE_H
#define REDUCE_STD_V2_UPDATE_DENSE_H
#include "reduce_std_v2_update.h"
#if __NPU_ARCH__ == 3510
namespace NsReduceStdV2Update {
// Contiguous input tiles avoid padding every short inner A/R group. Separate
// specializations leave the existing kernel instruction streams unchanged.
template <typename DType, bool ReduceRows>
class ReduceStdV2UpdateDenseKernel : public ReduceStdV2UpdateKernel<DType, true> {
public:
    using D_T = DType;
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR mean, GM_ADDR output, GM_ADDR workspace,
                                const ReduceStdV2UpdateTilingData* td)
    {
        this->axisNum_ = td->axisNum;
        fullOut_ = 1;
        for (int32_t i = this->axisNum_ - 1; i >= 0; --i) {
            this->axisShape_[i] = td->axisShape[i];
            this->axisStride_[i] = td->axisStride[i];
            this->outStride_[i] = fullOut_;
            if (i % 2 == 0)
                fullOut_ *= td->axisShape[i];
        }
        alignedOut_ = FastCeil8Elems(fullOut_);
        mainAxis_ = td->fastDb;
        block_ = td->fastLaneElems;
        innerOut_ = td->fastUnitOutLanes;
        prefixes_ = td->fastUnitCnt;
        chunks_ = td->rGroupCnt;
        rowCount_ = td->axisShape[mainAxis_];
        tileRows_ = td->fastRowsPerTile;
        laneSlot_ = FastCeil8Elems((block_ > alignedOut_ ? block_ : alignedOut_) + 64);
        this->usedCoreNum_ = td->usedCoreNum;
        this->cof_ = td->cof;
        this->ifStd_ = td->ifStd;
        this->tmpSlotElems_ = td->tmpBufUbSize / sizeof(float);
        this->xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(x));
        this->meanGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(mean));
        this->outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(output));
        this->pipe_.InitBuffer(this->preInXQue_, 1, td->preReduceUbSize / 2);
        this->pipe_.InitBuffer(this->preInMeanQue_, 1, td->preReduceUbSize / 2);
        const int64_t extra = ReduceRows ? 3 * laneSlot_ : FastCeil8Elems(tileRows_ * innerOut_ + 64);
        this->pipe_.InitBuffer(this->tmpBuf_, td->tmpBufUbSize + extra * sizeof(float));
        this->pipe_.InitBuffer(this->outQue_, 1, td->postReduceUbSize);
        if constexpr (ReduceRows) {
            this->wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
        } else {
            this->pipe_.InitBuffer(indexBuf_, 256);
            auto indices = indexBuf_.template Get<uint32_t>();
            for (int32_t lane = 0; lane < 64; ++lane) {
                int64_t a = lane % innerOut_;
                int64_t off = (lane / innerOut_) * block_;
                for (int32_t i = this->axisNum_ - 1; i > mainAxis_; --i) {
                    if (i % 2 == 0) {
                        off += (a % this->axisShape_[i]) * this->axisStride_[i];
                        a /= this->axisShape_[i];
                    }
                }
                indices.SetValue(lane, static_cast<uint32_t>(off));
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void Process()
    {
        if constexpr (ReduceRows)
            ProcessReduceRows();
        else
            ProcessOutputRows();
    }

private:
    __aicore__ inline void SquareTile(int64_t start, uint32_t elems)
    {
        auto x = this->preInXQue_.template AllocTensor<D_T>();
        auto m = this->preInMeanQue_.template AllocTensor<D_T>();
        this->FastCopyContig(x, this->xGm_, start, elems);
        this->FastCopyContig(m, this->meanGm_, start, elems);
        this->preInXQue_.EnQue(x);
        this->preInMeanQue_.EnQue(m);
        auto dx = this->preInXQue_.template DeQue<D_T>();
        auto dm = this->preInMeanQue_.template DeQue<D_T>();
        this->FastSubSquareCompact(dx, dm, elems);
        this->preInXQue_.FreeTensor(dx);
        this->preInMeanQue_.FreeTensor(dm);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void ProcessReduceRows()
    {
        const int64_t core = GetBlockIdx();
        const int64_t prefix = core / chunks_;
        const int64_t chunk = core % chunks_;
        const int64_t begin = rowCount_ * chunk / chunks_;
        const int64_t end = rowCount_ * (chunk + 1) / chunks_;
        auto all = this->tmpBuf_.template Get<float>();
        auto sq = all;
        auto part = all[static_cast<int32_t>(this->tmpSlotElems_)];
        auto acc = all[static_cast<int32_t>(this->tmpSlotElems_ + laneSlot_)];
        auto out = all[static_cast<int32_t>(this->tmpSlotElems_ + 2 * laneSlot_)];
        this->FastZeroLanes(acc, block_);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int64_t row = begin; row < end; row += tileRows_) {
            const uint32_t nRows = (row + tileRows_ <= end) ? tileRows_ : end - row;
            SquareTile((prefix * rowCount_ + row) * block_, nRows * block_);
            if (block_ % 8 == 0)
                this->FastReduceRowsRA(part, sq, nRows, block_);
            else
                this->FastReduceRowsRAFlat(part, sq, nRows, block_);
            this->FastAccumLanes(acc, part, block_);
        }
        this->FastZeroLanes(out, alignedOut_);
        AscendC::PipeBarrier<PIPE_ALL>();
        int64_t p = prefix, base = 0;
        for (int32_t i = mainAxis_ - 1; i >= 0; --i) {
            const int64_t coordinate = p % this->axisShape_[i];
            p /= this->axisShape_[i];
            if (i % 2 == 0)
                base += coordinate * this->outStride_[i];
        }
        for (int64_t lane = 0; lane < block_; ++lane) {
            int64_t k = lane, off = base;
            for (int32_t i = this->axisNum_ - 1; i > mainAxis_; --i) {
                const int64_t coordinate = k % this->axisShape_[i];
                k /= this->axisShape_[i];
                if (i % 2 == 0)
                    off += coordinate * this->outStride_[i];
            }
            out.SetValue(off, out.GetValue(off) + acc.GetValue(lane));
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        this->FastStageCacheLanes(part, out, 0, alignedOut_);
        const auto event = this->pipe_.template FetchEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::V_MTE3>(event);
        WaitFlag<HardEvent::V_MTE3>(event);
        DataCopy(this->wsGm_[core * alignedOut_], part, static_cast<uint32_t>(alignedOut_));
        SyncAll();
        if (core != 0)
            return;
        auto input = this->preInXQue_.template AllocTensor<D_T>();
        auto f = input.template ReinterpretCast<float>();
        DataCopy(f, this->wsGm_, static_cast<uint32_t>(this->usedCoreNum_ * alignedOut_));
        this->preInXQue_.EnQue(input);
        auto deq = this->preInXQue_.template DeQue<D_T>();
        auto sums = deq.template ReinterpretCast<float>();
        this->FastReduceRowsRA(out, sums, this->usedCoreNum_, alignedOut_);
        this->FastPostOut(out, 0, fullOut_);
        this->preInXQue_.FreeTensor(deq);
    }
    __aicore__ inline void AccumulateInner(AscendC::LocalTensor<float>& acc, AscendC::LocalTensor<float>& sq,
                                           uint32_t outputs)
    {
        auto indices = indexBuf_.template Get<uint32_t>();
        __ubuf__ uint32_t* indexPtr = reinterpret_cast<__ubuf__ uint32_t*>(indices.GetPhyAddr());
        __ubuf__ float* sqPtr = reinterpret_cast<__ubuf__ float*>(sq.GetPhyAddr());
        __ubuf__ float* accPtr = reinterpret_cast<__ubuf__ float*>(acc.GetPhyAddr());
        const uint16_t repeats = (outputs + 63) / 64;
        const int32_t srcStep = (64 / innerOut_) * block_;
        for (int64_t r = 0; r < block_ / innerOut_; ++r) {
            int64_t value = r, rOff = 0;
            for (int32_t i = this->axisNum_ - 1; i > mainAxis_; --i) {
                if (i % 2 != 0) {
                    rOff += (value % this->axisShape_[i]) * this->axisStride_[i];
                    value /= this->axisShape_[i];
                }
            }
            uint32_t remaining = outputs;
            const uint32_t offset = rOff;
            __VEC_SCOPE__
            {
                AscendC::Reg::RegTensor<uint32_t> baseIndex, index;
                AscendC::Reg::RegTensor<float> v, a;
                AscendC::Reg::LoadAlign(baseIndex, indexPtr);
                for (uint16_t vId = 0; vId < repeats; ++vId) {
                    auto mask = AscendC::Reg::UpdateMask<float>(remaining);
                    AscendC::Reg::Adds(index, baseIndex, offset, mask);
                    AscendC::Reg::Gather(v, sqPtr + vId * srcStep, index, mask);
                    AscendC::Reg::LoadAlign(a, accPtr + vId * 64);
                    AscendC::Reg::Add(a, a, v, mask);
                    AscendC::Reg::StoreAlign(accPtr + vId * 64, a, mask);
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void ProcessOutputRows()
    {
        const int64_t core = GetBlockIdx();
        const int64_t begin = rowCount_ * core / this->usedCoreNum_;
        const int64_t end = rowCount_ * (core + 1) / this->usedCoreNum_;
        auto all = this->tmpBuf_.template Get<float>();
        auto sq = all;
        auto acc = all[static_cast<int32_t>(this->tmpSlotElems_)];
        for (int64_t row = begin; row < end; row += tileRows_) {
            const uint32_t nRows = (row + tileRows_ <= end) ? tileRows_ : end - row;
            const uint32_t outputs = nRows * innerOut_;
            this->FastZeroLanes(acc, outputs);
            AscendC::PipeBarrier<PIPE_ALL>();
            for (int64_t prefix = 0; prefix < prefixes_; ++prefix) {
                SquareTile((prefix * rowCount_ + row) * block_, nRows * block_);
                AccumulateInner(acc, sq, outputs);
            }
            this->FastPostOut(acc, row * innerOut_, outputs);
        }
    }
    int32_t mainAxis_ = 0;
    int64_t block_ = 0, innerOut_ = 0, prefixes_ = 0, chunks_ = 0;
    int64_t rowCount_ = 0, tileRows_ = 0, fullOut_ = 0, alignedOut_ = 0, laneSlot_ = 0;
    TBuf<QuePosition::VECCALC> indexBuf_;
};
} // namespace NsReduceStdV2Update
#endif
#endif
