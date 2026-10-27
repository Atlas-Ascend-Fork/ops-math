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
 * \file minimum_grad_struct.h
 * \brief MinimumGrad 算子 TilingData 和 TilingKey 定义（atvoss 框架 - Broadcast 模式）
 *
 * TilingKey 参数：
 *   schMode: 调度模式（NDDMA/UB 等）
 *   outputCount: 输出数量（1=只y1, 2=只y2, 3=双输出）
 */

#ifndef MINIMUM_GRAD_STRUCT_H
#define MINIMUM_GRAD_STRUCT_H

#include "atvoss/broadcast/broadcast_base_struct.h"

// outputCount 编码：区分单输出和双输出模式
#define TPL_OUTPUT_Y1_ONLY 1   // grad_x=True, grad_y=False → 只输出 y1
#define TPL_OUTPUT_Y2_ONLY 2   // grad_x=False, grad_y=True → 只输出 y2
#define TPL_OUTPUT_BOTH 3      // grad_x=True, grad_y=True → 输出 y1 和 y2
#define TPL_SCH_MODE_EMPTY 999 // 空 Tensor 自定义分支

namespace NsMinimumGrad {
// 空 Tensor 分支的 tiling data
struct MinimumGradEmptyTilingData {
    int64_t numel; // 恒为 0，仅占位与调试
};

ASCENDC_TPL_ARGS_DECL(MinimumGrad, BRC_TEMP_SCH_MODE_KEY_DECL(schMode),
                      ASCENDC_TPL_UINT_DECL(outputCount, 2, ASCENDC_TPL_UI_LIST, TPL_OUTPUT_Y1_ONLY, TPL_OUTPUT_Y2_ONLY,
                                            TPL_OUTPUT_BOTH));

// 常规 broadcast 通路 + 空 Tensor 自定义分支（schMode=999）。
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_MODE_KEY_SEL(schMode),
                                     ASCENDC_TPL_UINT_SEL(outputCount, ASCENDC_TPL_UI_LIST, TPL_OUTPUT_Y1_ONLY,
                                                          TPL_OUTPUT_Y2_ONLY, TPL_OUTPUT_BOTH)),
                ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_CUSTOM_MODE_KEY_SEL(schMode),
                                     ASCENDC_TPL_UINT_SEL(outputCount, ASCENDC_TPL_UI_LIST, TPL_OUTPUT_Y1_ONLY,
                                                          TPL_OUTPUT_Y2_ONLY, TPL_OUTPUT_BOTH)));
} // namespace NsMinimumGrad

#endif // MINIMUM_GRAD_STRUCT_H
