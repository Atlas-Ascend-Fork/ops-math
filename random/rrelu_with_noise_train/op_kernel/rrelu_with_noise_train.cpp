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
 * \file rrelu_with_noise_train.cpp
 * \brief rrelu_with_noise kernel entry
 */

#include "arch35/rrelu_with_noise_train_impl.h"

// 单 tiling key + 编译期 dtype 变体(与 stateless_uniform 等同仓随机算子一致):
// def 按 {FP16, FP, BF16} 对齐注册 x/y/noise, 构建系统为每种 dtype 组合
// 各产一个变体并定义 DTYPE_X; 运行时按实际输入 dtype 选择变体。
// key=100 与共享 tiling 基类 RandomTilingArch35::CalcTilingKeyAndWorkspace 对齐。
#define RRELU_WITH_NOISE_TRAIN_DEFAULT_TILING_KEY 100

extern "C" __global__ __aicore__ void rrelu_with_noise_train(GM_ADDR x, GM_ADDR seed, GM_ADDR offset, GM_ADDR y,
                                                             GM_ADDR noise, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(RandomUnifiedSimtTilingDataStruct);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GET_TILING_DATA_WITH_STRUCT(RandomUnifiedSimtTilingDataStruct, tilingData, tiling);

    if (TILING_KEY_IS(RRELU_WITH_NOISE_TRAIN_DEFAULT_TILING_KEY)) {
        RreluWithNoise::Process<DTYPE_X>(x, seed, offset, y, noise, &tilingData);
    }
}
