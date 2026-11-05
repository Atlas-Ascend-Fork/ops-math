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
 * \file fused_mul_add_struct.h
 * \brief
 */
#ifndef FUSED_MUL_ADD_STRUCT_H_
#define FUSED_MUL_ADD_STRUCT_H_

#include "atvoss/broadcast/broadcast_base_struct.h"

// 空 Tensor 自定义分支：ATVOSS BroadcastBaseTiling 在合轴阶段拒绝 0 元素，
// 空 Tensor 不能复用常规 schMode，需单独走自定义 schMode 999。
#define FUSED_MUL_ADD_TPL_SCH_MODE_EMPTY 999

// 空 Tensor 分支的 tiling data
struct FusedMulAddEmptyTilingData {
    int64_t numel; // 恒为 0，仅占位与调试
};

ASCENDC_TPL_ARGS_DECL(FusedMulAdd, BRC_TEMP_SCH_MODE_KEY_DECL(schMode));

// 常规 broadcast 通路 + 空 Tensor 自定义分支（schMode=999）。
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_MODE_KEY_SEL(schMode)),
                ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_CUSTOM_MODE_KEY_SEL(schMode)));
#endif // FUSED_MUL_ADD_STRUCT_H_
