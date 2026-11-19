/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file histogram_v2_apt.cpp
 * \brief arch35 entry of HistogramV2: single unified SIMT kernel, load mode and
 *        deterministic scheme selected by the schId template parameter
 *        (see arch35/histogram_v2_tiling_key.h). Input/output dtypes come from
 *        the DTYPE_X / DTYPE_Y compile-time macros generated from the op def.
 */

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

#include "arch35/histogram_v2_tiling_key.h"
#include "arch35/histogram_v2_simt.h"

using namespace AscendC;

template <uint64_t schId>
__global__ __aicore__ void histogram_v2(GM_ADDR x, GM_ADDR min, GM_ADDR max, GM_ADDR y, GM_ADDR workspace,
                                        GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (g_coreType == AIC) {
        return;
    }

    // 注意：不要使用"注册默认 tiling 结构"的那个宏——仓内 kernel UT 的
    // gen_tiling_head_file.py 一旦在本算子目录 grep 到它就会切换生成模式，
    // 不再为 arch22 kernel UT 生成 HistogramV2TilingData。
    GET_TILING_DATA_WITH_STRUCT(HistogramV2SimtTilingData, tilingDataIn, tiling);
    const HistogramV2SimtTilingData* __restrict tilingData = &tilingDataIn;

    AscendC::TPipe tpipe;
    using X_TYPE = DTYPE_X;
    using OUT_TYPE = DTYPE_Y;
    using COMPUTE_TYPE = typename HistogramV2SIMT::ComputeTypeOf<X_TYPE>::Type;
    HistogramV2SIMT::HistogramV2Simt<X_TYPE, COMPUTE_TYPE, OUT_TYPE, schId> op;
    op.Init(x, min, max, y, tilingData, &tpipe);
    op.Process();
}
