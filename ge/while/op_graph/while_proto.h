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
 * \file while_proto.h
 * \brief Definition of the While operator.
 */
#ifndef WHILE_PROTO_H_
#define WHILE_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
 * @brief Cyclic execute the "body" subgraph until the return tensor of "cond" subgraph means False . \n

 * @par Inputs:
 * input: The input tensors . It's a dynamic input. \n

 * @par Graphs:
 * @li cond: A subgraph takes 'input' and returns a tensor.
 *          If the tensor is not a scalar of boolean type,
 *          it will be converted to a boolean according to the following rule:
 *          if it is a numerical scalar, non-zero means True and zero means False;
 *          if it is a string scalar, non-empty means True and empty means False;
 *          if it is not a scalar, non-empty means True and empty means False.
 * @li body: A subgraph takes 'input' and returns another list of tensors . \n

 * @par Attributes:
 * parallel_iterations: An optional int, default as 10 . \n

 * @par Outputs:
 * output: The output tensors returned by "body". Has the same type as "input" . It's a dynamic output. \n

 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator While.
 */
#ifndef OPS_PROTO_DEF_WHILE
#define OPS_PROTO_DEF_WHILE
REG_OP(While)
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .GRAPH(cond)
    .GRAPH(body)
    .ATTR(parallel_iterations, Int, 10)
    .OP_END_FACTORY_REG(While)
#endif
} // namespace ge
#endif // WHILE_PROTO_H_
