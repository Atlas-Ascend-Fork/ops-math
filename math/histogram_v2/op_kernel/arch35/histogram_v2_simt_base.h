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
 * \file histogram_v2_simt_base.h
 * \brief Base class of the unified arch35 HistogramV2 SIMT kernel: GM tensors,
 *        tiling parsing, min/max load + expansion, and the SyncAll UT guard.
 */
#ifndef HISTOGRAM_V2_SIMT_BASE_H
#define HISTOGRAM_V2_SIMT_BASE_H

#include "histogram_v2_simt_common.h"

namespace HistogramV2SIMT {
using namespace AscendC;

template <typename X_TYPE, typename COMPUTE_TYPE, typename OUT_TYPE>
class HistogramV2SimtBase {
public:
    __aicore__ inline HistogramV2SimtBase(){};

protected:
    // Init common global tensors and tiling fields. The int32/float views of y are
    // only dereferenced on the deterministic fp32-output path; setting them up
    // unconditionally is just address bookkeeping.
    __aicore__ inline void BaseInit(GM_ADDR x, GM_ADDR min, GM_ADDR max, GM_ADDR y,
                                    const HistogramV2SimtTilingData* __restrict tilingData, TPipe* tPipe)
    {
        pipe_ = tPipe;
        blockIdx_ = static_cast<int32_t>(GetBlockIdx());
        bins_ = tilingData->bins;
        ubNumCanUse_ = tilingData->ubNumCanUse;
        ubLoopNum_ = tilingData->ubLoopNum;
        needXCoreNum_ = tilingData->needXCoreNum;
        formerLength_ = tilingData->formerLength;
        tailLength_ = tilingData->tailLength;
        clearYCoreNum_ = tilingData->clearYCoreNum;
        clearYFactor_ = tilingData->clearYFactor;
        clearYTail_ = tilingData->clearYTail;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_TYPE*>(x));
        minGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_TYPE*>(min));
        maxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_TYPE*>(max));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ OUT_TYPE*>(y));
        yGmInt_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(y));
        yGmFloat_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
    }

    // Load min/max, expand when equal (CPU #163506 semantics via nextafter for float),
    // and precompute the range length. Input GM values are never modified, so calling
    // this before the clear phase is value-identical to the legacy per-variant timing.
    __aicore__ inline void ProcessMinMaxValue()
    {
        minValue_ = HistogramV2Common::ScalarCastTo<COMPUTE_TYPE>(minGm_.GetValue(0));
        maxValue_ = HistogramV2Common::ScalarCastTo<COMPUTE_TYPE>(maxGm_.GetValue(0));
        HistogramV2Common::ExpandMinMaxIfEqual(minValue_, maxValue_);
        minMaxLength_ = maxValue_ - minValue_;
    }

    __aicore__ inline void SyncAllIfNotUt()
    {
#ifndef __CCE_UT_TEST__
        SyncAll();
#endif
    }

    __aicore__ inline int64_t ClearYIndexBase() const { return blockIdx_ * clearYFactor_; }

    __aicore__ inline int64_t ClearYDataLength() const
    {
        return (blockIdx_ == clearYCoreNum_ - 1) ? clearYTail_ : clearYFactor_;
    }

    __aicore__ inline int64_t XIndexBase() const { return blockIdx_ * formerLength_; }

    __aicore__ inline int64_t CoreDataLength() const
    {
        return (blockIdx_ == needXCoreNum_ - 1) ? tailLength_ : formerLength_;
    }

protected:
    GlobalTensor<X_TYPE> xGm_;
    GlobalTensor<X_TYPE> minGm_;
    GlobalTensor<X_TYPE> maxGm_;
    GlobalTensor<OUT_TYPE> yGm_;
    GlobalTensor<int32_t> yGmInt_; // alias of y, deterministic fp32-out Phase 1/2
    GlobalTensor<float> yGmFloat_; // alias of y, deterministic fp32-out Phase 3

    TPipe* pipe_ = nullptr;
    TQue<TPosition::VECOUT, 1> yQue_;

    COMPUTE_TYPE minValue_ = 0;
    COMPUTE_TYPE maxValue_ = 0;
    COMPUTE_TYPE minMaxLength_ = 0;

    int32_t blockIdx_ = 0;
    int64_t bins_ = 0;
    int64_t ubNumCanUse_ = 0;
    int64_t ubLoopNum_ = 0;
    int64_t needXCoreNum_ = 0;
    int64_t formerLength_ = 0;
    int64_t tailLength_ = 0;
    int64_t clearYCoreNum_ = 0;
    int64_t clearYFactor_ = 0;
    int64_t clearYTail_ = 0;
};

} // namespace HistogramV2SIMT

#endif // HISTOGRAM_V2_SIMT_BASE_H
