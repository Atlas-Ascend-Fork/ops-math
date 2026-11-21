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
#include "cholesky_arch35.h"
#include "cholesky_arch35_tiling_key.h"

template <bool upper, bool cube = false>
__global__ __aicore__ void cholesky(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    // TilingKey = upper | (cube << 1)（template_argument.h: FastEncodeTilingKeyDirect
    // 按声明顺序低位优先累加）。cube==true 的两把键（2/3）以 MIX(AIC:AIV=1:2) 下发，
    // 其余键（0/1）保持 AIV_ONLY ⇒ 非 cube case 的 launch 形态不变。
    // 编译器要求必须声明 default kernel type（kernel_info_infer.py:557），default 为 AIV_ONLY。
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(3, KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if constexpr (upper) {
        Cholesky::Cholesky<float> op;
        op.InitTriu(self, out, workspace, &tilingData, &pipe, cube);
        if constexpr (cube) {
            op.ProcessTriuCube();
        } else {
            op.ProcessTriu();
        }
    } else {
        Cholesky::Cholesky<float> op;
        op.InitTril(self, out, workspace, &tilingData, &pipe, cube);
        if constexpr (cube) {
            op.ProcessTrilCube();
        } else {
            op.ProcessTril();
        }
    }
}

// 设备编译由构建系统生成的入口包装调用 cholesky<TEMPLATE_PARAMS>；
// op_kernel UT 直接取两个实例的地址，需显式实例化才能链到本 TU 的定义
#ifdef __CCE_UT_TEST__
template __global__ __aicore__ void cholesky<false, false>(GM_ADDR self, GM_ADDR out, GM_ADDR workspace,
                                                           GM_ADDR tiling);
template __global__ __aicore__ void cholesky<true, false>(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling);
template __global__ __aicore__ void cholesky<false, true>(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling);
template __global__ __aicore__ void cholesky<true, true>(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling);
#endif
