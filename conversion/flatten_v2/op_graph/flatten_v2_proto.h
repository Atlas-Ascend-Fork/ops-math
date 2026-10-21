/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_MATH_FLATTEN_V2_PROTO_H_
#define OPS_MATH_FLATTEN_V2_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
*@brief Flattens the input tensor to one-dimensional . \n

*@par Inputs:
*x: An ND tensor. All data types are supported. \n

*@par Attributes:
*@li axis: An optional int32, specifying the first axis to flatten. All preceding axes are retained in the output.
Defaults to "1".
*@li end_axis: An optional int32, specifying the last axis to flatten. All following axes are retained in the output.
Defaults to "-1" . \n

*@par Outputs:
*y: The flattened ND tensor. All data types are supported. \n

*@attention Constraints:
* "axis" and "end_axis" must be within the dimension range of the input. This operator cannot be directly called by the
acllopExecute API.
*@par Third-party framework compatibility
* Compatible with the Caffe operator Flatten.
*/
#ifndef OPS_PROTO_DEF_FLATTENV2
#define OPS_PROTO_DEF_FLATTENV2
REG_OP(FlattenV2)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axis, Int, 1)
    .ATTR(end_axis, Int, -1)
    .OP_END_FACTORY_REG(FlattenV2)
#endif
} // namespace ge
#endif // OPS_MATH_FLATTEN_V2_PROTO_H_
