/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_MATH_TRANS_SHAPE_PROTO_H_
#define OPS_MATH_TRANS_SHAPE_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
*@brief Change the shape of output according to the attr outShape
*

*@par Inputs:
*x: A Tensor. \n

*@par Outputs:
*y: A Tensor. Has the same type as "x".It's required and the value should equal to output_num. \n

*@par Attributes:
*outShape: The shape of output will be inferred according to the attribute
*/
#ifndef OPS_PROTO_DEF_TRANSSHAPE
#define OPS_PROTO_DEF_TRANSSHAPE
REG_OP(TransShape)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(outShape, ListInt, {})
    .OP_END_FACTORY_REG(TransShape);
#endif
} // namespace ge
#endif // OPS_MATH_TRANS_SHAPE_PROTO_H_
