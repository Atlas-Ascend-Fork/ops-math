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
 * \file strided_slice_arch31.h
 * \brief
 */
#ifndef STRIDED_SLICE_H
#define STRIDED_SLICE_H

#include "strided_slice_base_arch31.h"
#include "strided_slice_move_align_arch31.h"
#include "strided_slice_move_align_two_dim_arch31.h"
#include "strided_slice_move_align_last_dim_arch31.h"
#include "strided_slice_move_align_gather_arch31.h"
#include "strided_slice_move_align_negative_strides_ub2ub_arch31.h"

using namespace StridedSlice;

extern "C" __aicore__ inline void StridedSliceMoveAlignProcess(GM_ADDR x, GM_ADDR begin, GM_ADDR end, GM_ADDR strides,
                                                               GM_ADDR y, const StridedSliceMATilingData* tilingData)
{
    TPipe pipe;
    // ub不切最后一根轴
    // complex32/uint32/int32/float32
    if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        StridedSlice::StridedSliceMoveAlign<int32_t, DTYPE_BEGIN> op;
        op.Init(x, begin, end, strides, y, tilingData, &pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        // complex64/uint64/int64
        StridedSlice::StridedSliceMoveAlign<int64_t, DTYPE_BEGIN> op;
        op.Init(x, begin, end, strides, y, tilingData, &pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        StridedSlice::StridedSliceMoveAlign<int8_t, DTYPE_BEGIN> op;
        op.Init(x, begin, end, strides, y, tilingData, &pipe);
        op.Process();
    } else {
        StridedSlice::StridedSliceMoveAlign<DTYPE_X, DTYPE_BEGIN> op;
        op.Init(x, begin, end, strides, y, tilingData, &pipe);
        op.Process();
    }
}

extern "C" __aicore__ inline void StridedSliceMoveAlignLastDimProcess(GM_ADDR x, GM_ADDR begin, GM_ADDR end,
                                                                      GM_ADDR strides, GM_ADDR y,
                                                                      const StridedSliceMALastDimTilingData* tilingData)
{
    TPipe pipe;
    // ub切最后一根轴 or 纯搬运模板
    // complex32/uint32/int32/float32
    if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        StridedSlice::StridedSliceMoveAlignLastDim<int32_t, DTYPE_BEGIN> op;
        op.Init(x, begin, end, strides, y, tilingData, &pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        // complex64/uint64/int64
        StridedSlice::StridedSliceMoveAlignLastDim<int64_t, DTYPE_BEGIN> op;
        op.Init(x, begin, end, strides, y, tilingData, &pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        StridedSlice::StridedSliceMoveAlignLastDim<int8_t, DTYPE_BEGIN> op;
        op.Init(x, begin, end, strides, y, tilingData, &pipe);
        op.Process();
    } else {
        StridedSlice::StridedSliceMoveAlignLastDim<DTYPE_X, DTYPE_BEGIN> op;
        op.Init(x, begin, end, strides, y, tilingData, &pipe);
        op.Process();
    }
}

extern "C" __aicore__ inline void StridedSliceMoveAlignTwoDimProcess(GM_ADDR x, GM_ADDR begin, GM_ADDR end,
                                                                     GM_ADDR strides, GM_ADDR y,
                                                                     const StridedSliceMALast2DimTilingData* tilingData)
{
    TPipe pipe;
    if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        StridedSlice::StridedSliceMoveAlignTwoDim<int32_t, DTYPE_BEGIN> op;
        op.Init(x, begin, end, strides, y, tilingData, &pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        StridedSlice::StridedSliceMoveAlignTwoDim<int64_t, DTYPE_BEGIN> op;
        op.Init(x, begin, end, strides, y, tilingData, &pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        StridedSlice::StridedSliceMoveAlignTwoDim<int8_t, DTYPE_BEGIN> op;
        op.Init(x, begin, end, strides, y, tilingData, &pipe);
        op.Process();
    } else {
        StridedSlice::StridedSliceMoveAlignTwoDim<DTYPE_X, DTYPE_BEGIN> op;
        op.Init(x, begin, end, strides, y, tilingData, &pipe);
        op.Process();
    }
}

extern "C" __aicore__ inline void StridedSliceMoveAlignGatherProcess(GM_ADDR x, GM_ADDR begin, GM_ADDR end,
                                                                     GM_ADDR strides, GM_ADDR y,
                                                                     const StridedSliceMAGatherTilingData* tilingData)
{
    TPipe pipe;
    if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        StridedSlice::StridedSliceMoveAlignGather<int32_t, DTYPE_BEGIN> op;
        op.Init(x, begin, y, tilingData, &pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        StridedSlice::StridedSliceMoveAlignGather<int64_t, DTYPE_BEGIN> op;
        op.Init(x, begin, y, tilingData, &pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        StridedSlice::StridedSliceMoveAlignGather<int8_t, DTYPE_BEGIN> op;
        op.Init(x, begin, y, tilingData, &pipe);
        op.Process();
    } else {
        StridedSlice::StridedSliceMoveAlignGather<int16_t, DTYPE_BEGIN> op;
        op.Init(x, begin, y, tilingData, &pipe);
        op.Process();
    }
}

extern "C" __aicore__ inline void StridedSliceMoveAlignUb2UbProcess(GM_ADDR x, GM_ADDR begin, GM_ADDR end,
                                                                    GM_ADDR strides, GM_ADDR y,
                                                                    const StridedSliceMAUB2UBTilingData* tilingData)
{
    TPipe pipe;
    if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        StridedSlice::StridedSliceMoveAlignUb2Ub<int32_t, DTYPE_BEGIN> op;
        op.Init(x, begin, y, tilingData, &pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        StridedSlice::StridedSliceMoveAlignUb2Ub<int64_t, DTYPE_BEGIN> op;
        op.Init(x, begin, y, tilingData, &pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        StridedSlice::StridedSliceMoveAlignUb2Ub<int8_t, DTYPE_BEGIN> op;
        op.Init(x, begin, y, tilingData, &pipe);
        op.Process();
    } else {
        StridedSlice::StridedSliceMoveAlignUb2Ub<int16_t, DTYPE_BEGIN> op;
        op.Init(x, begin, y, tilingData, &pipe);
        op.Process();
    }
}

#endif // STRIDED_SLICE_H
