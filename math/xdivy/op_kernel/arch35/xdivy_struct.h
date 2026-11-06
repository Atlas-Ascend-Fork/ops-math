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
 * \file xdivy_struct.h
 * \brief Xdivy TilingKey — 模板参数：RANK 按实际 rank 映射到 4 或 8
 */
#ifndef XDIVY_STRUCT_H_
#define XDIVY_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define XDIVY_RANK_4 4
#define XDIVY_RANK_8 8

// userDef 模板参数：0 = 常规通路；1 = 空 Tensor 通路（输出 0 元素，kernel 侧直接返回）。
// 空 Tensor 必须单独开模板分支：kernel 侧 GET_TILING_DATA_WITH_STRUCT 引用的结构体
// 不同（XdivyTilingData vs XdivyEmptyTilingData），opc 按模板实例反推 tiling 结构体大小。
// 写法与 math/add_v2 的 userDef 空分支一致。
#define XDIVY_USER_DEF_NORMAL 0
#define XDIVY_USER_DEF_EMPTY 1

ASCENDC_TPL_ARGS_DECL(Xdivy, ASCENDC_TPL_UINT_DECL(RANK, 8, ASCENDC_TPL_UI_LIST, XDIVY_RANK_4, XDIVY_RANK_8),
                      ASCENDC_TPL_UINT_DECL(userDef, 8, ASCENDC_TPL_UI_LIST, XDIVY_USER_DEF_NORMAL,
                                            XDIVY_USER_DEF_EMPTY));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(RANK, ASCENDC_TPL_UI_LIST, XDIVY_RANK_4),
                                     ASCENDC_TPL_UINT_SEL(userDef, ASCENDC_TPL_UI_LIST, XDIVY_USER_DEF_NORMAL)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(RANK, ASCENDC_TPL_UI_LIST, XDIVY_RANK_8),
                                     ASCENDC_TPL_UINT_SEL(userDef, ASCENDC_TPL_UI_LIST, XDIVY_USER_DEF_NORMAL)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(RANK, ASCENDC_TPL_UI_LIST, XDIVY_RANK_4),
                                     ASCENDC_TPL_UINT_SEL(userDef, ASCENDC_TPL_UI_LIST, XDIVY_USER_DEF_EMPTY)));

#endif // XDIVY_STRUCT_H_
