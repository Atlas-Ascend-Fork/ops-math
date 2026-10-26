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
 * \file gather_shapes_proto.h
 * \brief Definition of the GatherShapes operator.
 */
#ifndef GATHER_SHAPES_PROTO_H_
#define GATHER_SHAPES_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 *@brief Gather selected dims of input which returns the shape of tensor shape after gathershapes.\n

 *@par Inputs:
 *x: A list of input tensors. All data types are supported. It's a dynamic input. \n

 *@par Attributes:
 *@li axes: An 2-D list of int32 or int64 required. Select some dims of input.
 *@li dtype: An optional int32, which indicates the data type of output. Defaults to DT_INT32. \n

 *@par Outputs:
 *shape: The shape of tensor shape after gathershapes. Must be one of the following types: int32、int64. \n
 */
REG_OP(GatherShapes)
    .DYNAMIC_INPUT(x, TensorType::ALL())
    .OUTPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(axes, ListListInt)
    .ATTR(dtype, Int, DT_INT32)
    .OP_END_FACTORY_REG(GatherShapes)
} // namespace ge
#endif // GATHER_SHAPES_PROTO_H_
