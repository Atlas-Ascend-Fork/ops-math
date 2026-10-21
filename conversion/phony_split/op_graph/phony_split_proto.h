/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_MATH_PHONY_SPLIT_PROTO_H_
#define OPS_MATH_PHONY_SPLIT_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
* @brief Splits a tensor along dimension "split_dim" into "num_split" smaller tensors.
        All outputs of the PhonySplit are allocated with the same memory block.
        GE calculates the memory offset of each output,
        and the custom operators next read the memory by offset.
* Warning: This operator is used only to identify that the GE allocates continuous memory and does not perform any
calculation.

* @par Inputs:
* One input:
* @li x:An ND Tensor.
* Must be one of the following types: float16, float32, int32, int8, int16,
  int64, uint8, uint16, uint32, uint64, bool, bfloat16.

* @par Attributes:
* @li split_dim: A required int32. Specifies the dimension along which to split. No default value.
* @li num_split: A required int32. Specifies the number of output tensors. No default value .
* @li keep_output_offset: A optional Bool. Specifies whether calculate the memory offset of outpute tensors. Default
True .

* @par Outputs:
* @li y:Dynamic output. A list of output tensors. Has the same type and format as "x" .

* @attention Constraints:
* @li "num_split" is greater than or equals to 1.
* @li "num_split" is divisible by the size of dimension "split_dim".
* @li "split_dim" is in the range [-len(x.shape), (x.shape)-1] .

* @par Third-party framework support
* Support ONNX  framework.
*/
REG_OP(PhonySplit)
    .INPUT(x, TensorType::ALL())
    .DYNAMIC_OUTPUT(y, TensorType::ALL())
    .REQUIRED_ATTR(split_dim, ListInt)
    .REQUIRED_ATTR(num_split, ListInt)
    .ATTR(keep_output_offset, Bool, true)
    .OP_END_FACTORY_REG(PhonySplit)
} // namespace ge
#endif // OPS_MATH_PHONY_SPLIT_PROTO_H_
