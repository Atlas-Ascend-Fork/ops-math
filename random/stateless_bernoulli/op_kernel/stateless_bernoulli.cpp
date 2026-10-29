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
 * \file stateless_bernoulli.cpp
 * \brief
 */

#include "arch35/stateless_bernoulli_simt.h"

using namespace StatelessBernoulli;

// 场景化 tilingKey: 100=标量(交织), 101=张量非广播(直接索引), 102=张量广播(stride 寻址)。
// 注: TILING_KEY_IS 的预编译解析(kernel_info_infer)仅识别数字与宏, 仓内随机数族算子
// (stateless_random_normal_v2 等)均以 #define 承载 tilingKey, 对齐先例
#define STATELESS_BERNOULLI_SCALAR_KEY 100
#define STATELESS_BERNOULLI_TENSOR_DENSE_KEY 101
#define STATELESS_BERNOULLI_TENSOR_BCAST_KEY 102

// 场景化入口的分发型辅助: 与 tilingKey 分支一一对应, 保持每个 tilingKey 的
// kernel symbol 只编对应路径(kernel 内不做语义分支判断); bool 输出以 int8 承载
template <typename Tp, typename To>
__aicore__ inline void RunScalar(GM_ADDR prob, GM_ADDR y, const RandomUnifiedSimtTilingDataStruct* tilingData)
{
    StatelessBernoulliKernel<Tp, To> op;
    op.ProcessScalar(prob, y, tilingData);
}

template <typename Tp, typename To>
__aicore__ inline void RunTensorDense(GM_ADDR prob, GM_ADDR y, const RandomUnifiedSimtTilingDataStruct* tilingData)
{
    StatelessBernoulliKernel<Tp, To> op;
    op.ProcessTensorDense(prob, y, tilingData);
}

template <typename Tp, typename To>
__aicore__ inline void RunTensorBcast(GM_ADDR prob, GM_ADDR y, const RandomUnifiedSimtTilingDataStruct* tilingData)
{
    StatelessBernoulliKernel<Tp, To> op;
    op.ProcessTensorBcast(prob, y, tilingData);
}

__global__ __aicore__ void stateless_bernoulli(GM_ADDR shape, GM_ADDR prob, GM_ADDR seed, GM_ADDR offset, GM_ADDR y,
                                               GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(RandomUnifiedSimtTilingDataStruct);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GET_TILING_DATA(tilingData, tiling);

    // prob dtype 三型分发: def 层 probDataType = {FLOAT16, FLOAT, BFLOAT16}
    using To = typename AscendC::Conditional<AscendC::IsSameType<DTYPE_Y, bool>::value, int8_t, DTYPE_Y>::type;
    if (TILING_KEY_IS(STATELESS_BERNOULLI_SCALAR_KEY)) {
        if constexpr (AscendC::IsSameType<DTYPE_PROB, float>::value) {
            RunScalar<float, To>(prob, y, &tilingData);
        } else if constexpr (AscendC::IsSameType<DTYPE_PROB, half>::value) {
            RunScalar<half, To>(prob, y, &tilingData);
        } else if constexpr (AscendC::IsSameType<DTYPE_PROB, bfloat16_t>::value) {
            RunScalar<bfloat16_t, To>(prob, y, &tilingData);
        }
    } else if (TILING_KEY_IS(STATELESS_BERNOULLI_TENSOR_DENSE_KEY)) {
        if constexpr (AscendC::IsSameType<DTYPE_PROB, float>::value) {
            RunTensorDense<float, To>(prob, y, &tilingData);
        } else if constexpr (AscendC::IsSameType<DTYPE_PROB, half>::value) {
            RunTensorDense<half, To>(prob, y, &tilingData);
        } else if constexpr (AscendC::IsSameType<DTYPE_PROB, bfloat16_t>::value) {
            RunTensorDense<bfloat16_t, To>(prob, y, &tilingData);
        }
    } else if (TILING_KEY_IS(STATELESS_BERNOULLI_TENSOR_BCAST_KEY)) {
        if constexpr (AscendC::IsSameType<DTYPE_PROB, float>::value) {
            RunTensorBcast<float, To>(prob, y, &tilingData);
        } else if constexpr (AscendC::IsSameType<DTYPE_PROB, half>::value) {
            RunTensorBcast<half, To>(prob, y, &tilingData);
        } else if constexpr (AscendC::IsSameType<DTYPE_PROB, bfloat16_t>::value) {
            RunTensorBcast<bfloat16_t, To>(prob, y, &tilingData);
        }
    }
}
