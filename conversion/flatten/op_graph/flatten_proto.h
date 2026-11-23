/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_MATH_FLATTEN_PROTO_H_
#define OPS_MATH_FLATTEN_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
* @brief Flattens the inputs tensor into a 2D matrix. If input tensor has shape (d_0, d_1,..., d_n),
*        then the output will have shape (d_0 X d_1 ... d_(axis-1), d_axis X d_(axis + 1)...X d_n)\n

* @par Inputs:
* One input:
* x: A multi-dimensional tensor. All data types are supported.

* @par Outputs:
* y: A 2D flattened tensor with the contents of the input tensor, with input dimensions up to axis flattened
* to the outer dimension of the output and remaining input dimensions flattened into the inner dimension of the
* output.
* Has the same type as "x".

* @par Attributes:
* axis: A optional int32, default value is 1. Indicate up to which input dimensions (exclusive) should be flattened
* to the outer dimension of the output. The value for axis must be in the range [-r, r], where r is the rank of
* the input tensor. Negative value means counting dimensions from the back. When axis = 0, the shape of
* the output tensor is (1, (d_0 X d_1 ... d_n), where the shape of the input tensor is (d_0, d_1, ... d_n).

* @par Third-party framework compatibility
* Compatible with TensorFlow / ONNX operator Flatten.
*/
REG_OP(Flatten).INPUT(x, TensorType::ALL()).OUTPUT(y, TensorType::ALL()).ATTR(axis, Int, 1).OP_END_FACTORY_REG(Flatten)
} // namespace ge
#endif // OPS_MATH_FLATTEN_PROTO_H_
