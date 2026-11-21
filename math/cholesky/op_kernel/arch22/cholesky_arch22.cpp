/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "cholesky_arch22.h"
#include "cholesky_arch22_tiling_key.h"

template <bool upper>
__global__ __aicore__ void cholesky(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if constexpr (upper) {
        Cholesky::Cholesky<float> op;
        op.InitTriu(self, out, workspace, &tilingData, &pipe);
        op.ProcessTriu();
    } else {
        Cholesky::Cholesky<float> op;
        op.InitTril(self, out, workspace, &tilingData, &pipe);
        op.ProcessTril();
    }
}

// 设备编译由构建系统生成的入口包装调用 cholesky<TEMPLATE_PARAMS>；
// op_kernel UT 直接取两个实例的地址，需显式实例化才能链到本 TU 的定义
#ifdef __CCE_UT_TEST__
template __global__ __aicore__ void cholesky<false>(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling);
template __global__ __aicore__ void cholesky<true>(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling);
#endif
