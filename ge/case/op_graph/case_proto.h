/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CASE_PROTO_H_
#define CASE_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
 * @brief Select one of the subgraphs to pass the input tensors and return the output tensors . \n

 * @par Inputs:
 * @li branch_index: A int32 scalar which determines the selected subgraph.
 * @li input: The input tensors, which will be passed to the subgraph . It's a dynamic input. \n

 * @par Graphs:
 * branches: A list of subgraphs, each of which takes 'input' and returns a list of tensors,
 *          whose types are the same as what every other subgraph returns . \n

 * @par Outputs:
 * output: The output tensors returned by one of branches . It's a dynamic output.

 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator Case.
 */
#ifndef OPS_PROTO_DEF_CASE
#define OPS_PROTO_DEF_CASE
REG_OP(Case)
    .INPUT(branch_index, DT_INT32)
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .DYNAMIC_GRAPH(branches)
    .OP_END_FACTORY_REG(Case)
#endif

} // namespace ge

#endif // CASE_PROTO_H_
