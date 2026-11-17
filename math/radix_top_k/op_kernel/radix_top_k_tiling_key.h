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
 * \file radix_top_k_tiling_key.h
 * \brief Radix TopK TilingKey 模板参数声明
 *        定义 sorted/largest/isLargeShape/isInt64Shape/isSelectTopK 组合，
 *        编译期选择模板实例化，避免运行时分支
 */

#ifndef RADIX_TOP_K_TILING_KEY_H
#define RADIX_TOP_K_TILING_KEY_H
#include "ascendc/host_api/tiling/template_argument.h"

/**
 * @brief 声明 isSelectTopK 为最高位；取 0 时现网 16 个 radix 键编码不变，取 1 时收敛为 4 键
 */
ASCENDC_TPL_ARGS_DECL(RadixTopK, ASCENDC_TPL_BOOL_DECL(sorted, 0, 1), ASCENDC_TPL_BOOL_DECL(largest, 0, 1),
                      ASCENDC_TPL_BOOL_DECL(isLargeShape, 0, 1), ASCENDC_TPL_BOOL_DECL(isInt64Shape, 0, 1),
                      ASCENDC_TPL_BOOL_DECL(isSelectTopK, 0, 1));

/** @brief 组合选择：isSelectTopK=0 全组合 16 键，=1 收敛 4 键（供 GET_TPL_TILING_KEY 校验） */
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(sorted, 0, 1), ASCENDC_TPL_BOOL_SEL(largest, 0, 1),
                                     ASCENDC_TPL_BOOL_SEL(isLargeShape, 0, 1), ASCENDC_TPL_BOOL_SEL(isInt64Shape, 0, 1),
                                     ASCENDC_TPL_BOOL_SEL(isSelectTopK, 0)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(sorted, 0, 1), ASCENDC_TPL_BOOL_SEL(largest, 0, 1),
                                     ASCENDC_TPL_BOOL_SEL(isLargeShape, 0), ASCENDC_TPL_BOOL_SEL(isInt64Shape, 0),
                                     ASCENDC_TPL_BOOL_SEL(isSelectTopK, 1)));

#endif // RADIX_TOP_K_TILING_KEY_H
