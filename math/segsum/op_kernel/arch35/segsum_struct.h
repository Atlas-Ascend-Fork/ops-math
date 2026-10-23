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
 * \file segsum_struct.h (arch35 / Ascend950)
 * \brief TilingKey template declaration for Segsum.
 */

#ifndef SEGSUM_STRUCT_ARCH35_H
#define SEGSUM_STRUCT_ARCH35_H

#include "ascendc/host_api/tiling/template_argument.h"

#define SEGSUM_TPL_SCH_MODE_STRIPE 0
#define SEGSUM_TPL_SCH_MODE_ROW_BLOCK 1

namespace SegsumArch35 {
ASCENDC_TPL_ARGS_DECL(Segsum, ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, SEGSUM_TPL_SCH_MODE_STRIPE,
                                                    SEGSUM_TPL_SCH_MODE_ROW_BLOCK));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, SEGSUM_TPL_SCH_MODE_STRIPE,
                                                          SEGSUM_TPL_SCH_MODE_ROW_BLOCK)));
} // namespace SegsumArch35

#endif // SEGSUM_STRUCT_ARCH35_H
