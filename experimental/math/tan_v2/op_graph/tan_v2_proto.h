/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Tu Yuanhang <@TuYHAAAAAA>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tan_v2_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_TANV2_H_
#define OPS_OP_PROTO_INC_TANV2_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Returns tan(x).
*@par Inputs:
*One input, including:
* @li x: An ND Tensor. Must be one of the following types: float32, float16. \n

*@par Outputs:
*y: An ND Tensor with the same shape as "x". Must be one of the following types: float32, float16.
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Tan.
*/
REG_OP(TanV2)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(TanV2)

} // namespace ge

#endif // OPS_OP_PROTO_INC_TanV2_H_
