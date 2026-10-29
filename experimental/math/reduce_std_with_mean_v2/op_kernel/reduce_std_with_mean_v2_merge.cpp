/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reduce_std_with_mean_v2_merge_tiling_key.h"
#include "reduce_std_with_mean_v2_two_stage_kernels.h"

template <uint32_t schMode>
__global__ __aicore__ void reduce_std_with_mean_v2_merge(GM_ADDR partial, GM_ADDR output, GM_ADDR meanOut,
                                                         GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReduceStdWithMeanV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(ReduceStdWithMeanV2TilingData, tilingData, tiling);
    // The Merge kernel is float32-only (partial input and outputs are fp32;
    // the L2 layer Casts to the original dtype), so T is hardcoded instead of
    // DTYPE_SELF (which follows the fp32 input here and is not defined by the
    // compile driver).
    NsReduceStdWithMeanV2::ReduceStdWithMeanV2Merge<float, schMode> op;
    op.Init(partial, output, meanOut, workspace, &tilingData);
    op.Process();
}
