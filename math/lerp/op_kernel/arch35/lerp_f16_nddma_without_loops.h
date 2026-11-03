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
 * \file lerp_f16_nddma_without_loops.h
 * \brief
 */
#ifndef ASCENDC_LERP_F16_NDDMA_WITHOUT_LOOPS_H_
#define ASCENDC_LERP_F16_NDDMA_WITHOUT_LOOPS_H_

#include "kernel_operator.h"
#include "atvoss/util/broadcast_utils.h"
#include "vf/lerp_vf.h"

namespace Lerp {
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TQue;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;

// start is float16, end is float16, weight is float16, y is float16, max dims in ub is 5 and nddma does not need loops
class LerpF16NddmaWithoutLoops {
public:
    __aicore__ inline LerpF16NddmaWithoutLoops(){};
    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace,
                                const LerpTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipePtr_ = pipePtr;
        tilingDataPtr_ = tilingDataPtr;
        inputGmStart_.SetGlobalBuffer((__gm__ half*)start);
        inputGmEnd_.SetGlobalBuffer((__gm__ half*)end);
        inputGmWeight_.SetGlobalBuffer((__gm__ half*)weight);
        outputGmY_.SetGlobalBuffer((__gm__ half*)y);
        constexpr int64_t DOUBLE_BUFFER = 2;
        int64_t BUFFER_SIZE_0 = tilingDataPtr_->elemNum * sizeof(half);
        pipePtr_->InitBuffer(queIn0_, DOUBLE_BUFFER, BUFFER_SIZE_0);
        pipePtr_->InitBuffer(queIn1_, DOUBLE_BUFFER, BUFFER_SIZE_0);
        pipePtr_->InitBuffer(queIn2_, DOUBLE_BUFFER, BUFFER_SIZE_0);
        pipePtr_->InitBuffer(queOut0_, DOUBLE_BUFFER, BUFFER_SIZE_0);
    }

    __aicore__ inline void Process()
    {
        int64_t ubLoopNum = AscendC::GetBlockIdx() == AscendC::GetBlockNum() - 1 ? tilingDataPtr_->blockTail :
                                                                                   tilingDataPtr_->blockFormer;
        int64_t axesIndices[Ops::Base::BROADCAST_MAX_DIMS] = {0};
        Ops::Base::BroadcastGetAxesIndices(axesIndices, tilingDataPtr_->blockFormer * AscendC::GetBlockIdx(),
                                           tilingDataPtr_->outputDims, tilingDataPtr_->ubSplitAxis,
                                           tilingDataPtr_->dimProductBeforeUbInner);
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopNum; ubLoopIdx += 1) {
            if (ubLoopIdx != 0) {
                Ops::Base::BroadcastUpdateAxesIndices(axesIndices, tilingDataPtr_->outputDims,
                                                      tilingDataPtr_->ubSplitAxis, tilingDataPtr_->ubOuter);
            }
            int64_t ubSplitSize = axesIndices[tilingDataPtr_->ubSplitAxis] == tilingDataPtr_->ubOuter - 1 ?
                                      tilingDataPtr_->ubTail :
                                      tilingDataPtr_->ubFormer;
            CopyIn0(ubSplitSize, axesIndices, ubLoopIdx);
            CopyIn1(ubSplitSize, axesIndices, ubLoopIdx);
            CopyIn2(ubSplitSize, axesIndices, ubLoopIdx);
            Compute3(ubSplitSize, axesIndices, ubLoopIdx);
            CopyOut4(ubSplitSize, axesIndices, ubLoopIdx);
        }
    }

private:
    __aicore__ inline void CopyIn0(int64_t ubSplitSize, const int64_t (&axesIndices)[Ops::Base::BROADCAST_MAX_DIMS],
                                   int64_t ubLoopIdx)
    {
        bufferIn0_ = queIn0_.AllocTensor<half>();
        if ((tilingDataPtr_->input2Strides[tilingDataPtr_->ubSplitAxis] != 0) ||
            (ubLoopIdx <= 1 ||
             (AscendC::GetBlockIdx() * tilingDataPtr_->blockFormer + ubLoopIdx) % tilingDataPtr_->ubOuter <= 1)) {
            Ops::Base::BroadcastNddmaWithoutLoop(inputGmWeight_, bufferIn0_, tilingDataPtr_->outputDims,
                                                 tilingDataPtr_->outputStrides, tilingDataPtr_->input2Strides,
                                                 axesIndices, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->shapeLen,
                                                 ubSplitSize, tilingDataPtr_->ubFormer);
        }
        queIn0_.EnQue<half>(bufferIn0_);
    }

    __aicore__ inline void CopyIn1(int64_t ubSplitSize, const int64_t (&axesIndices)[Ops::Base::BROADCAST_MAX_DIMS],
                                   int64_t ubLoopIdx)
    {
        bufferIn1_ = queIn1_.AllocTensor<half>();
        if ((tilingDataPtr_->input1Strides[tilingDataPtr_->ubSplitAxis] != 0) ||
            (ubLoopIdx <= 1 ||
             (AscendC::GetBlockIdx() * tilingDataPtr_->blockFormer + ubLoopIdx) % tilingDataPtr_->ubOuter <= 1)) {
            Ops::Base::BroadcastNddmaWithoutLoop(inputGmEnd_, bufferIn1_, tilingDataPtr_->outputDims,
                                                 tilingDataPtr_->outputStrides, tilingDataPtr_->input1Strides,
                                                 axesIndices, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->shapeLen,
                                                 ubSplitSize, tilingDataPtr_->ubFormer);
        }
        queIn1_.EnQue<half>(bufferIn1_);
    }

    __aicore__ inline void CopyIn2(int64_t ubSplitSize, const int64_t (&axesIndices)[Ops::Base::BROADCAST_MAX_DIMS],
                                   int64_t ubLoopIdx)
    {
        bufferIn2_ = queIn2_.AllocTensor<half>();
        if ((tilingDataPtr_->input0Strides[tilingDataPtr_->ubSplitAxis] != 0) ||
            (ubLoopIdx <= 1 ||
             (AscendC::GetBlockIdx() * tilingDataPtr_->blockFormer + ubLoopIdx) % tilingDataPtr_->ubOuter <= 1)) {
            Ops::Base::BroadcastNddmaWithoutLoop(inputGmStart_, bufferIn2_, tilingDataPtr_->outputDims,
                                                 tilingDataPtr_->outputStrides, tilingDataPtr_->input0Strides,
                                                 axesIndices, tilingDataPtr_->ubSplitAxis, tilingDataPtr_->shapeLen,
                                                 ubSplitSize, tilingDataPtr_->ubFormer);
        }
        queIn2_.EnQue<half>(bufferIn2_);
    }

    __aicore__ inline void Compute3(int64_t ubSplitSize, const int64_t (&axesIndices)[Ops::Base::BROADCAST_MAX_DIMS],
                                    int64_t ubLoopIdx)
    {
        bufferIn0_ = queIn0_.DeQue<half>();
        bufferIn1_ = queIn1_.DeQue<half>();
        bufferIn2_ = queIn2_.DeQue<half>();
        bufferOut0_ = queOut0_.AllocTensor<half>();
        int64_t dataLength = ubSplitSize * tilingDataPtr_->outputStrides[tilingDataPtr_->ubSplitAxis];
        __ubuf__ half* bufferIn0Addr = (__ubuf__ half*)bufferIn0_.GetPhyAddr();
        __ubuf__ half* bufferIn1Addr = (__ubuf__ half*)bufferIn1_.GetPhyAddr();
        __ubuf__ half* bufferIn2Addr = (__ubuf__ half*)bufferIn2_.GetPhyAddr();
        __ubuf__ half* bufferOut0Addr = (__ubuf__ half*)bufferOut0_.GetPhyAddr();
        asc_vf_call<LerpF16VF>(bufferIn0Addr, bufferIn1Addr, bufferIn2Addr, bufferOut0Addr, dataLength);
        queIn0_.FreeTensor(bufferIn0_);
        queIn1_.FreeTensor(bufferIn1_);
        queIn2_.FreeTensor(bufferIn2_);
        queOut0_.EnQue<half>(bufferOut0_);
    }

    __aicore__ inline void CopyOut4(int64_t ubSplitSize, const int64_t (&axesIndices)[Ops::Base::BROADCAST_MAX_DIMS],
                                    int64_t ubLoopIdx)
    {
        bufferOut0_ = queOut0_.DeQue<half>();
        AscendC::DataCopyExtParams dataCopyExtParams;
        dataCopyExtParams.blockCount = 1;
        dataCopyExtParams.blockLen = ubSplitSize * tilingDataPtr_->outputStrides[tilingDataPtr_->ubSplitAxis] *
                                     sizeof(half);
        int64_t gmOffset = Ops::Base::BroadcastGetGmOffset(axesIndices, tilingDataPtr_->outputStrides,
                                                           tilingDataPtr_->ubSplitAxis, tilingDataPtr_->ubFormer);
        AscendC::DataCopyPad(outputGmY_[gmOffset], bufferOut0_[0], dataCopyExtParams);
        queOut0_.FreeTensor(bufferOut0_);
    }

private:
    TPipe* pipePtr_;
    const LerpTilingData* tilingDataPtr_;
    GlobalTensor<half> inputGmStart_;
    GlobalTensor<half> inputGmEnd_;
    GlobalTensor<half> inputGmWeight_;
    GlobalTensor<half> outputGmY_;
    TQue<AscendC::QuePosition::VECIN, 1> queIn0_;
    TQue<AscendC::QuePosition::VECIN, 1> queIn1_;
    TQue<AscendC::QuePosition::VECIN, 1> queIn2_;
    TQue<AscendC::QuePosition::VECOUT, 1> queOut0_;
    LocalTensor<half> bufferIn0_;
    LocalTensor<half> bufferIn1_;
    LocalTensor<half> bufferIn2_;
    LocalTensor<half> bufferOut0_;
};

} // namespace Lerp
#endif // ASCENDC_LERP_F16_NDDMA_WITHOUT_LOOPS_H_
