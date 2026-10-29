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
 * \file mul_struct.h
 * \brief mul struct
 */
#ifndef MUL_ARCH31_STRUCT_H_
#define MUL_ARCH31_STRUCT_H_

#include "atvoss/broadcast/broadcast_base_struct.h"

ASCENDC_TPL_ARGS_DECL(Mul,
                      ASCENDC_TPL_UINT_DECL(schMode, 16, ASCENDC_TPL_UI_LIST, 101, 102, 103, 104, 109, 201, 202, 999));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, 101, 102, 103, 104, 109, 201,
                                                          202)));
#endif // MUL_ARCH31_STRUCT_H_
