/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_MATH_NO_OP_PROTO_H_
#define OPS_MATH_NO_OP_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
 * @brief Does nothing. Only useful as a placeholder for control edges . \n

 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator NoOp.
 */
#ifndef OPS_PROTO_DEF_NOOP
#define OPS_PROTO_DEF_NOOP
REG_OP(NoOp).OP_END_FACTORY_REG(NoOp)
#endif
} // namespace ge

#endif // OPS_MATH_NO_OP_PROTO_H_
