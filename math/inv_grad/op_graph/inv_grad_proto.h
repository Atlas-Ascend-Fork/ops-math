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
 * \file inv_grad_proto.h
 * \brief
 */
#ifndef OPS_OP_INV_GRAD_PROTO_H_
#define OPS_OP_INV_GRAD_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief Computes "x" reciprocal grad, y = -grad * x * x, where "x" is the output of
* the forward Inv operator (i.e., "x = 1 / original_x"), "grad" is the corresponding
* input gradient, and "y" is the gradient with respect to the original input.
* Inputs "x" and "grad" must have the
* same shape; broadcasting is not supported, pre-broadcast the inputs externally
* before calling this operator.

* @par Inputs:
* Two inputs, including:
* @li x: A ND Tensor. Must be one of the following types: float16, float32, bfloat16, int32, int8.
* @li grad: A ND Tensor. Has the same dtype as "x". \n

* @par Outputs:
* y: A ND Tensor, Has the same dtype as "x". \n

* @par Third-party framework compatibility:
* Compatible with the TensorFlow operator InvGrad.
*/
#ifndef OPS_PROTO_DEF_INVGRAD
#define OPS_PROTO_DEF_INVGRAD
REG_OP(InvGrad)
    .INPUT(x, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16, DT_INT32, DT_INT8}))
    .INPUT(grad, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16, DT_INT32, DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16, DT_INT32, DT_INT8}))
    .OP_END_FACTORY_REG(InvGrad)
#endif
} // namespace ge
#endif
