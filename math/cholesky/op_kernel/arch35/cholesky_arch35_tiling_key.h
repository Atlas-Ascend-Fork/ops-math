/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * A5 encodes upper in bit 0 and cube in bit 1.
 * Keys 0/1 select AIV-only execution; keys 2/3 select mixed AIC:AIV=1:2 execution.
 * Host tiling and kernel entries use this declaration from the same build.
 */

#ifndef CHOLESKY_ARCH35_TILING_KEY_H
#define CHOLESKY_ARCH35_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

// cube：是否走 AIC Cube MMAD 的 Schur 更新路径（仅 arch35 单矩阵大 M 命中）。
// 位布局按声明顺序低位优先：key = upper | (cube << 1) ⇒ 0/1 = AIV_ONLY，2/3 = MIX_AIC_1_2。
ASCENDC_TPL_ARGS_DECL(Cholesky, ASCENDC_TPL_BOOL_DECL(upper, 0, 1), ASCENDC_TPL_BOOL_DECL(cube, 0, 1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(upper, 0, 1), ASCENDC_TPL_BOOL_SEL(cube, 0, 1)), );

#endif // CHOLESKY_ARCH35_TILING_KEY_H
