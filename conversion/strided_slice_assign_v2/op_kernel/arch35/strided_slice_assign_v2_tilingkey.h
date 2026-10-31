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
 * \file strided_slice_assign_v2_tilingkey.h
 * \brief StridedSliceAssignV2 模板参数声明（替代原 TILING_KEY_IS 分派的 tilingKey）
 */

#ifndef STRIDED_SLICE_ASSIGN_V2_TILINGKEY_H_
#define STRIDED_SLICE_ASSIGN_V2_TILINGKEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

// TemplateNum 取值：模板编号（位宽 2，覆盖 0..3）。
// 方向（全正 / 全负 / 混合）已随 axisStep 符号落入各 TilingData，kernel 侧不依赖方向模板维度。
#define SSA_TPL_HYBRID 0  // T1 StridedSliceAssignV2Hybrid
#define SSA_TPL_CONTIG 1  // T2 StridedSliceAssignV2Contig
#define SSA_TPL_SCATTER 2 // T3 StridedSliceAssignV2Scatter
#define SSA_TPL_EMPTY 3   // T4 StridedSliceAssignV2Empty

ASCENDC_TPL_ARGS_DECL(StridedSliceAssignV2, ASCENDC_TPL_UINT_DECL(TemplateNum, 2, ASCENDC_TPL_UI_RANGE, 1, 0, 3));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(TemplateNum, ASCENDC_TPL_UI_RANGE, 1, 0, 3)));

#endif // STRIDED_SLICE_ASSIGN_V2_TILINGKEY_H_
