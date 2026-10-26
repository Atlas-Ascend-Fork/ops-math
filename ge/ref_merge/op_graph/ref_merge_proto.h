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
 * \file ref_merge_proto.h
 * \brief Definition of the RefMerge operator.
 */
#ifndef REF_MERGE_PROTO_H_
#define REF_MERGE_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 *@brief Forwards the value of an available tensor from input "x" to output "y".
 *       Merge waits for at least one of the input tensors to become available.
 *       It is usually combined with Switch to implement branching.
 *       Merge forwards the first tensor to become available to output "y",
 *       and sets "value_index" the index of the tensor in inputs .

 *@par Inputs:
 *x: The input tensors, one of which will become available.
 *   Must be one of the following types: float16, float32, float64, int8,
 *   int16, int32, int64, uint8, uint16, uint32, uint64, bool, string . It's a dynamic input. \n

 *@par Outputs:
 *@li y: The available tensor. Has the same type as "x".
 *@li value_index: A scalar of type int32, for the index of the chosen input
 *                 tensor . \n

 *@see Switch() | Merge()

 *@par Third-party framework compatibility
 *Compatible with the TensorFlow operator RefMerge.
 */
REG_OP(RefMerge)
    .DYNAMIC_INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .OUTPUT(value_index, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(RefMerge)
} // namespace ge
#endif // REF_MERGE_PROTO_H_
