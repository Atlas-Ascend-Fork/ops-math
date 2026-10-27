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
 * \file maximum_grad_proto.h
 * \brief MaximumGrad算子原型定义
 */

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Computes the backpropagation of the maximum operation.

* @par Inputs:
* Three inputs, including:
* @li grads: Gradient input tensor. Must be one of the following types: float16, float, bfloat16, int32.
* Empty tensors (any dim = 0) are not supported.
* @li x1: First input tensor. Must be the same dtype as grads.
* @li x2: Second input tensor. Must be the same dtype as grads.

* @par Outputs:
* Two outputs, including:
* @li y1: Gradient for x1 (when grad_x=True). Same dtype as inputs.
* @li y2: Gradient for x2 (when grad_y=True). Same dtype as inputs.

* @par Attributes:
* @li grad_x: Boolean, whether to output y1. Default: true.
* @li grad_y: Boolean, whether to output y2. Default: true.

* @par Third-party framework compatibility:
* Compatible with the TensorFlow operator MaximumGrad.
*/
#ifndef OPS_PROTO_DEF_MAXIMUMGRAD
#define OPS_PROTO_DEF_MAXIMUMGRAD
REG_OP(MaximumGrad)
    .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT32}))
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT32}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT32}))
    .OUTPUT(y1, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT32}))
    .OUTPUT(y2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT32}))
    .ATTR(grad_x, Bool, true)
    .ATTR(grad_y, Bool, true)
    .OP_END_FACTORY_REG(MaximumGrad)
#endif
} // namespace ge
