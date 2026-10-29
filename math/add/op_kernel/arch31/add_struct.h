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
 * \file add_struct.h
 * \brief add struct
 */
#ifndef ADD_STRUCT_H_
#define ADD_STRUCT_H_

#include "atvoss/broadcast/broadcast_base_struct.h"

// 算子自定义的tiling key字段
ASCENDC_TPL_ARGS_DECL(Add,
                      ASCENDC_TPL_UINT_DECL(schMode, 16, ASCENDC_TPL_UI_LIST, 101, 102, 103, 104, 109, 201, 202, 999));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, 101, 102, 103, 104, 109, 201,
                                                          202)));
#endif // ADD_STRUCT_H_
