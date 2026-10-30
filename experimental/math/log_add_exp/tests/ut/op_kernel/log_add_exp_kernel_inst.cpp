/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "../../../op_kernel/log_add_exp.h"

// Keep kernel headers separate from gtest's host headers.
template <typename T>
__aicore__ inline void RunLogAddExp(GM_ADDR x, GM_ADDR y, GM_ADDR out, GM_ADDR tiling)
{
    NsLogAddExp::LogAddExp<T, float> op;
    op.Init(x, y, out, nullptr, reinterpret_cast<LogAddExpTilingData*>(tiling));
    op.Process();
}

extern "C" __global__ __aicore__ void log_add_exp_fp16_ut(GM_ADDR x, GM_ADDR y, GM_ADDR out, GM_ADDR tiling)
{
    RunLogAddExp<half>(x, y, out, tiling);
}

extern "C" __global__ __aicore__ void log_add_exp_bf16_ut(GM_ADDR x, GM_ADDR y, GM_ADDR out, GM_ADDR tiling)
{
    RunLogAddExp<bfloat16_t>(x, y, out, tiling);
}
