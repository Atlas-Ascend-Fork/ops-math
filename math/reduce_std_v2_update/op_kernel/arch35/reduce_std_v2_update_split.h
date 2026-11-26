/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_STD_V2_UPDATE_SPLIT_H
#define REDUCE_STD_V2_UPDATE_SPLIT_H
#include "reduce_std_v2_update.h"
#if __NPU_ARCH__ == 3510
namespace NsReduceStdV2Update {
// A separate tiling specialization keeps legacy kernels free of new runtime
// branches and register allocation changes. Shared numerical helpers retain
// the PR's dtype conversions and postprocessing semantics.
template <typename DType>
class ReduceStdV2UpdateSplitKernel : public ReduceStdV2UpdateKernel<DType, true> {
public:
    using D_T = DType;
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR mean, GM_ADDR output, const ReduceStdV2UpdateTilingData* td)
    {
        for (int32_t i = 0; i < 4; ++i) {
            this->axisShape_[i] = td->axisShape[i];
            this->axisStride_[i] = td->axisStride[i];
        }
        this->usedCoreNum_ = td->usedCoreNum;
        this->fastLaneElems_ = td->fastLaneElems;
        this->fastRowsPerTile_ = td->fastRowsPerTile;
        this->fastUnitCnt_ = td->fastUnitCnt;
        this->fastUnitOutLanes_ = td->fastUnitOutLanes;
        this->fastDb_ = td->fastDb;
        this->tmpSlotElems_ = td->tmpBufUbSize / sizeof(float);
        this->cof_ = td->cof;
        this->ifStd_ = td->ifStd;
        this->xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(x));
        this->meanGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(mean));
        this->outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T*>(output));
        this->pipe_.InitBuffer(this->preInXQue_, 1, td->preReduceUbSize / 2);
        this->pipe_.InitBuffer(this->preInMeanQue_, 1, td->preReduceUbSize / 2);
        const int64_t sumBytes = 3 * FastCeil8Elems(td->fastLaneElems + 64) * sizeof(float);
        this->pipe_.InitBuffer(this->tmpBuf_, td->tmpBufUbSize + sumBytes);
        this->pipe_.InitBuffer(this->outQue_, 1, td->postReduceUbSize);
    }
    __aicore__ inline void Process()
    {
        const int64_t midA = this->axisShape_[2];
        const int64_t innerR = this->axisShape_[3];
        const int64_t lanes = this->fastUnitOutLanes_;
        const int64_t chunks = this->fastUnitCnt_ / this->axisShape_[0];
        const int64_t block = static_cast<int64_t>(GetBlockIdx());
        const int64_t begin = this->fastUnitCnt_ * block / this->usedCoreNum_;
        const int64_t end = this->fastUnitCnt_ * (block + 1) / this->usedCoreNum_;
        const int64_t rowsPerTile = this->fastRowsPerTile_;
        auto tmpAll = this->tmpBuf_.template Get<float>();
        auto sq = tmpAll;
        auto dst1 = tmpAll[static_cast<int32_t>(this->tmpSlotElems_)];
        const int64_t dst1Elems = FastCeil8Elems(this->fastLaneElems_ + 64);
        const int64_t dst2Elems = FastCeil8Elems(this->fastLaneElems_ + 64);
        auto dst2 = tmpAll[static_cast<int32_t>(this->tmpSlotElems_ + dst1Elems)];
        auto acc = tmpAll[static_cast<int32_t>(this->tmpSlotElems_ + dst1Elems + dst2Elems)];
        for (int64_t u = begin; u < end; ++u) {
            const int64_t a0 = u / chunks;
            const int64_t chunk = u % chunks;
            const int64_t outStart = (this->fastDb_ == 2) ? midA * chunk / chunks : chunk * lanes;
            const int64_t outLanes = (this->fastDb_ == 2) ? midA * (chunk + 1) / chunks - outStart :
                                                            ((outStart + lanes <= midA) ? lanes : midA - outStart);
            const int64_t copyStart = (outStart + lanes <= midA) ? outStart : midA - lanes;
            this->FastZeroLanes(acc, static_cast<uint32_t>(this->fastLaneElems_));
            AscendC::PipeBarrier<PIPE_ALL>();
            for (int64_t r0 = 0; r0 < this->axisShape_[1]; r0 += rowsPerTile) {
                const int64_t rows = (r0 + rowsPerTile < this->axisShape_[1]) ? rowsPerTile : this->axisShape_[1] - r0;
                const int64_t start = a0 * this->axisStride_[0] + r0 * midA * innerR + copyStart * innerR;
                auto x = this->preInXQue_.template AllocTensor<D_T>();
                auto mean = this->preInMeanQue_.template AllocTensor<D_T>();
                DataCopyExtParams ext;
                ext.blockCount = static_cast<uint16_t>(rows);
                ext.blockLen = static_cast<uint32_t>(lanes * innerR * sizeof(D_T));
                ext.srcStride = static_cast<uint32_t>((midA - lanes) * innerR * sizeof(D_T));
                ext.dstStride = 0;
                DataCopyPadExtParams<D_T> pad{true, 0, static_cast<uint8_t>(this->fastLaneElems_ - lanes * innerR),
                                              static_cast<D_T>(0)};
                DataCopyPad(x, this->xGm_[start], ext, pad);
                DataCopyPad(mean, this->meanGm_[start], ext, pad);
                this->preInXQue_.EnQue(x);
                this->preInMeanQue_.EnQue(mean);
                auto dx = this->preInXQue_.template DeQue<D_T>();
                auto dm = this->preInMeanQue_.template DeQue<D_T>();
                this->FastSubSquareCompact(dx, dm, static_cast<uint32_t>(rows * this->fastLaneElems_));
                this->preInXQue_.FreeTensor(dx);
                this->preInMeanQue_.FreeTensor(dm);
                // Sum R0 while all R1 lanes remain aligned. Reduce R1 only once
                // after the R0 tiles, avoiding an unaligned AR reduction per tile.
                this->FastReduceRowsRA(dst2, sq, static_cast<uint32_t>(rows),
                                       static_cast<uint32_t>(this->fastLaneElems_));
                this->FastAccumLanes(acc, dst2, static_cast<uint32_t>(this->fastLaneElems_));
            }
            this->FastReduceInnerAR(dst1, acc, static_cast<uint32_t>(lanes), static_cast<uint32_t>(innerR));
            if (copyStart == outStart) {
                this->FastPostOut(dst1, a0 * midA + outStart, static_cast<uint32_t>(outLanes));
            } else {
                AscendC::PipeBarrier<PIPE_ALL>();
                for (int64_t l = 0; l < outLanes; ++l) {
                    dst2.SetValue(l, dst1.GetValue(outStart - copyStart + l));
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                this->FastPostOut(dst2, a0 * midA + outStart, static_cast<uint32_t>(outLanes));
            }
        }
    }
};
} // namespace NsReduceStdV2Update
#endif
#endif
