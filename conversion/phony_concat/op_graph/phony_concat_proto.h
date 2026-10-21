/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_MATH_PHONY_CONCAT_PROTO_H_
#define OPS_MATH_PHONY_CONCAT_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
 * @brief Concatenates a list of "N" tensors along "concat_dim".
 *        All input of the PhonyConcat are allocated with the same memory block.
 *        GE calculates the memory offset of each input,
 *        and the custom operators before write the memory by offset.
 *        Warning: This operator is used only to identify that the GE allocates continuous
 *        memory and does not perform any calculation . \n

 * @par Inputs:
 * Dynamic input: A list of input tensors. Has the same type and format as "x" .
 * @li x:An ND Tensor.
 * Must be one of the following types: float16, float32, int32, int8, int16,
 * int64, uint8, uint16, uint32, uint64, bool, bfloat16.

 * @par Attributes:
 * @li concat_dim: A required list of int32. Specifies the dimensions along which to concat. No default value.
 * @li N: A required list of int32. Specifies the number of concat tensors. No default value .
 * @li keep_input_offset: A optional Bool. Specifies whether calculate the memory offset of input tensors. Default True
 .

 * @par Outputs:
 * @li y:One output.

 * @attention Constraints:
 * @li "concat_dim" is in the range [-len(x.shape), (x.shape)-1] .
 * @li "N" is greater than or equals to 1.

 * @par Third-party framework support
 * Support ONNX  framework.
 */
REG_OP(PhonyConcat)
    .DYNAMIC_INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .REQUIRED_ATTR(concat_dim, ListInt)
    .REQUIRED_ATTR(N, ListInt)
    .ATTR(keep_input_offset, Bool, true)
    .OP_END_FACTORY_REG(PhonyConcat)
} // namespace ge

#endif // OPS_MATH_PHONY_CONCAT_PROTO_H_
