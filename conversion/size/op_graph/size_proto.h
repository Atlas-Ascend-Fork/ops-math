/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_MATH_SIZE_PROTO_H_
#define OPS_MATH_SIZE_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Returns the size of a tensor, that is, an integer of the number of elements of the tensor. \n

 * @par Inputs:
 * x: A tensor. Must be one of the following types: float32, float16, int8,
 * int16, uint16, uint8, int32, int64, uint32, uint64, bool, double, string. \n

 * @par Attributes:
 * dtype: An optional int32 or int64. The output data type. Defaults to "int32". \n

 * @par Outputs:
 * y: A tensor. The size of the input tensor. \n

 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator Size.
 */
#ifndef OPS_PROTO_DEF_SIZE
#define OPS_PROTO_DEF_SIZE
REG_OP(Size)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(dtype, Int, DT_INT32)
    .OP_END_FACTORY_REG(Size)
#endif
} // namespace ge
#endif // OPS_MATH_SIZE_PROTO_H_
