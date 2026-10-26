/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARTITIONEDCALL_PROTO_H_
#define PARTITIONEDCALL_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
 * @brief Pass the input tensors to the subgraph "f" and return the output tensors . \n

 * @par Inputs:
 * args: The input tensors, which will be passed to "f" . It's a dynamic input. \n

 * @par Graphs:
 * f: A subgraph takes 'args' and returns another list of tensors . \n

 * @par Attributes:
 * @li config: An optional string, default as "".
 * @li config_proto: An optional string, default as "".
 * @li executor_type: An optional string, default as "" . \n

 * @par Outputs:
 * output: The output tensors returned by "f" . It's a dynamic output. \n

 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator PartitionedCall.
 */
#ifndef OPS_PROTO_DEF_PARTITIONEDCALL
#define OPS_PROTO_DEF_PARTITIONEDCALL
REG_OP(PartitionedCall)
    .DYNAMIC_INPUT(args, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .GRAPH(f)
    .ATTR(config, String, "")
    .ATTR(config_proto, String, "")
    .ATTR(executor_type, String, "")
    .OP_END_FACTORY_REG(PartitionedCall)
#endif
} // namespace ge

#endif // PARTITIONEDCALL_PROTO_H_
