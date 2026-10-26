/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FOR_PROTO_H_
#define FOR_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
 * @brief Cyclic execute the "body" subgraph until the first input of For op exceed upper bound . \n

 * @par Inputs:
 * @li start: An int32 scalar. The lower bound.
 * @li limit: An int32 scalar. The upper bound.
 * @li delta: An int32 scalar. The step size.
 * @li input: The input tensors, which will be passed to "body" . It's a dynamic input. \n

 * @par Graphs:
 * body: A subgraph takes 'input' and returns another list of tensors . \n

 * @par Outputs:
 * output: The output tensors returned by "body". Has the same type as "input" . It's a dynamic output. \n

 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator For.
 */
#ifndef OPS_PROTO_DEF_FOR
#define OPS_PROTO_DEF_FOR
REG_OP(For)
    .INPUT(start, DT_INT32)
    .INPUT(limit, DT_INT32)
    .INPUT(delta, DT_INT32)
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .GRAPH(body)
    .OP_END_FACTORY_REG(For)
#endif
} // namespace ge

#endif // FOR_PROTO_H_
