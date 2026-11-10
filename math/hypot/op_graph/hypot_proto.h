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
 * \file hypot_proto.h
 * \brief hypot graph-mode operator proto
 */
#ifndef OPS_OP_PROTO_INC_HYPOT_H_
#define OPS_OP_PROTO_INC_HYPOT_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Computes the hypotenuse sqrt(self^2 + other^2) element-wise with a numerically
*stable implementation. Supports broadcasting. \n

*@par Inputs:
*Two inputs, including:
* @li self: A ND Tensor. Must be float16, float32 or bfloat16. \n
* @li other: A ND Tensor. Must be float16, float32 or bfloat16, and must be broadcast-compatible
* with "self". Its dtype is promoted with "self" to determine the output dtype. \n

*@par Outputs:
*out: A ND Tensor. Has the promoted dtype of "self" and "other". \n

*@par Third-party framework compatibility
*Compatible with the PyTorch operator hypot.
*/
REG_OP(Hypot)
    .INPUT(self, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(other, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(out, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(Hypot)

} // namespace ge

#endif // OPS_OP_PROTO_INC_HYPOT_H_
