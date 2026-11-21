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
 * \file test_transpose_tiling_strategy_unity.cpp
 * \brief 策略 UT unity 编译单元（仅提供策略实现符号，无测试用例）
 *
 * 生产 op_host .so 符号为隐藏可见性，UT 可执行体无法直接链接生产 .cpp 的策略实现。
 * 各策略 UT 文件只 include 策略头文件；策略实现的符号统一由本 unity 编译单元提供
 * （与生产同一源码，行为一致），避免多文件重复定义。include 按文件名排序，同类模板相邻。
 */

#include "../../../../op_host/arch35/transpose_tiling_strategy.cpp"
#include "../../../../op_host/arch35/transpose_tiling_strategy_b4.cpp"
#include "../../../../op_host/arch35/transpose_tiling_strategy_gather.cpp"
#include "../../../../op_host/arch35/transpose_tiling_strategy_n_last.cpp"
#include "../../../../op_host/arch35/transpose_tiling_strategy_nddma.cpp"
#include "../../../../op_host/arch35/transpose_tiling_strategy_nddma_big_dim.cpp"
#include "../../../../op_host/arch35/transpose_tiling_strategy_small_shape.cpp"
#include "../../../../op_host/arch35/transpose_tiling_strategy_tensor_move.cpp"
#include "../../../../op_host/arch35/transpose_tiling_strategy_transto5hd_10.cpp"
#include "../../../../op_host/arch35/transpose_tiling_strategy_transto5hd_10_perf.cpp"
#include "../../../../op_host/arch35/transpose_tiling_strategy_transto5hd_021.cpp"
