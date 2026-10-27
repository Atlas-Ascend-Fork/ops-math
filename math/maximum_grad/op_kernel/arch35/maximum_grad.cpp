/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file maximum_grad.cpp
 * \brief MaximumGrad 算子 Kernel 入口
 *
 * 支持三种输出模式：
 *   outputCount=1: 只输出 y1 (grad_x=true, grad_y=False)
 *   outputCount=2: 只输出 y2 (grad_x=False, grad_y=True)
 *   outputCount=3: 输出 y1 和 y2 (grad_x=True, grad_y=True)
 *
 */

#include "kernel_operator.h"
#include "maximum_grad_dag.h"
#include "maximum_grad_struct.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace AscendC;
using namespace NsMaximumGrad;

template <uint64_t schMode, uint64_t outputCount>
__global__ __aicore__ void maximum_grad(GM_ADDR grads, GM_ADDR x1, GM_ADDR x2, GM_ADDR y1, GM_ADDR y2,
                                        GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // 使用 DTYPE_GRADS 编译期宏获取数据类型
    using T = DTYPE_GRADS;

    // 空 Tensor：元素个数为 0，无数据搬运/写回。Tiling 侧已把 blockDim 设为 1，
    // 这里单核空转返回。GET_TILING_DATA_WITH_STRUCT 不能省
    if constexpr (schMode == TPL_SCH_MODE_EMPTY) {
        GET_TILING_DATA_WITH_STRUCT(MaximumGradEmptyTilingData, emptyTilingData, tiling);
        return;
    } else if constexpr (outputCount == TPL_OUTPUT_Y1_ONLY) {
        // 只输出 y1
        using OpDag = DagTypes::OutputY1<T>;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(grads, x1, x2, y1);
    } else if constexpr (outputCount == TPL_OUTPUT_Y2_ONLY) {
        // 只输出 y2
        using OpDag = DagTypes::OutputY2<T>;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(grads, x1, x2, y2);
    } else {
        // 输出 y1 和 y2
        using OpDag = DagTypes::TwoOutput<T>;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(grads, x1, x2, y1, y2);
    }
}
