/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transpose_generic_datacopy.h"
#include "transpose_strided_datacopy.h"
#include "transpose_tensor_move_arch31.h"
#include "transpose_transdata_5hd_arch31.h"
#include "transpose_transdata_5hd_021_arch31.h"

#define TENSOR_MOVE 10000
#define VCONV_TRANSPOSE 10007
#define VCONV_021_TRANSPOSE 10008
#define GENERIC_DATACOPY 10009
#define STRIDED_DATACOPY_TRANSPOSE 10010

using namespace Transpose;

template <typename T>
__aicore__ inline void RunTensorMove(GM_ADDR x, GM_ADDR y, const KirinTransposeOpTilingData* opTiling, TPipe* pipe)
{
    TransposeTensorMove<T> kernel;
    kernel.Init(x, y, opTiling, pipe);
    kernel.Process();
}

template <typename T>
__aicore__ inline void RunGenericDataCopy(GM_ADDR x, GM_ADDR y, const KirinTransposeOpTilingData* opTiling, TPipe* pipe)
{
    TransposeGenericDataCopy<T> kernel;
    kernel.Init(x, y, opTiling, pipe);
    kernel.Process();
}

template <typename T>
__aicore__ inline void RunVConvTranspose(GM_ADDR x, GM_ADDR y, const KirinTransposeVConvOpTilingData* opTiling,
                                         TPipe* pipe)
{
    KirinTransposeTransDataTo5HD<T> kernel;
    kernel.Init(x, y, opTiling, pipe);
    kernel.Process();
}

template <typename T>
__aicore__ inline void Run021VConvTranspose(GM_ADDR x, GM_ADDR y, const KirinTranspose021VConvOpTilingData* opTiling,
                                            TPipe* pipe)
{
    KirinKernelTransDataTo5HD021<T> kernel;
    kernel.Init(x, y, opTiling, pipe);
    kernel.Process();
}

template <typename T>
__aicore__ inline void RunStridedDataCopyTranspose(GM_ADDR x, GM_ADDR y,
                                                   const KirinTransposeStridedDataCopyOpTilingData* opTiling,
                                                   TPipe* pipe)
{
    Transpose::TransposeStridedDataCopy<T> kernel;
    kernel.Init(x, y, opTiling, pipe);
    kernel.Process();
}

extern "C" __global__ __aicore__ void transpose(GM_ADDR x, GM_ADDR perm, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    (void)perm;
    (void)workspace;

    if (TILING_KEY_IS(TENSOR_MOVE)) {
        GET_TILING_DATA_WITH_STRUCT(KirinTransposeTensorMoveTilingData, tilingData, tiling);
        const KirinTransposeOpTilingData* opTiling = &tilingData.transposeOpTiling;
        switch (opTiling->elementBytes) {
            case 1:
                RunTensorMove<int8_t>(x, y, opTiling, &pipe);
                break;
            case 2:
                RunTensorMove<int16_t>(x, y, opTiling, &pipe);
                break;
            case 4:
                RunTensorMove<int32_t>(x, y, opTiling, &pipe);
                break;
            default:
                break;
        }
    } else if (TILING_KEY_IS(VCONV_TRANSPOSE)) {
        GET_TILING_DATA_WITH_STRUCT(KirinTransposeVConvTilingData, tilingData, tiling);
        RunVConvTranspose<int16_t>(x, y, &tilingData.transposeVConvTiling, &pipe);
    } else if (TILING_KEY_IS(VCONV_021_TRANSPOSE)) {
        GET_TILING_DATA_WITH_STRUCT(KirinTranspose021VConvTilingData, tilingData, tiling);
        const KirinTranspose021VConvOpTilingData* opTiling = &tilingData.transpose021VConvOpTiling;
        if (opTiling->elementBytes == 1) {
            Run021VConvTranspose<int8_t>(x, y, opTiling, &pipe);
        } else if (opTiling->elementBytes == 2) {
            Run021VConvTranspose<int16_t>(x, y, opTiling, &pipe);
        } else if (opTiling->elementBytes == 4) {
            Run021VConvTranspose<int32_t>(x, y, opTiling, &pipe);
        }
    } else if (TILING_KEY_IS(STRIDED_DATACOPY_TRANSPOSE)) {
        GET_TILING_DATA_WITH_STRUCT(KirinTransposeStridedDataCopyTilingData, tilingData, tiling);
        const KirinTransposeStridedDataCopyOpTilingData* opTiling = &tilingData.transposeStridedDataCopyOpTiling;
        switch (opTiling->elementBytes) {
            case 1:
                RunStridedDataCopyTranspose<int8_t>(x, y, opTiling, &pipe);
                break;
            case 2:
                RunStridedDataCopyTranspose<int16_t>(x, y, opTiling, &pipe);
                break;
            case 4:
                RunStridedDataCopyTranspose<int32_t>(x, y, opTiling, &pipe);
                break;
            case 8:
                RunStridedDataCopyTranspose<int64_t>(x, y, opTiling, &pipe);
                break;
            default:
                break;
        }
    } else if (TILING_KEY_IS(GENERIC_DATACOPY)) {
        GET_TILING_DATA_WITH_STRUCT(KirinTransposeGenericDataCopyTilingData, tilingData, tiling);
        const KirinTransposeOpTilingData* opTiling = &tilingData.transposeOpTiling;
        switch (opTiling->elementBytes) {
            case 1:
                RunGenericDataCopy<int8_t>(x, y, opTiling, &pipe);
                break;
            case 2:
                RunGenericDataCopy<int16_t>(x, y, opTiling, &pipe);
                break;
            case 4:
                RunGenericDataCopy<int32_t>(x, y, opTiling, &pipe);
                break;
            default:
                break;
        }
    }
}
