/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Li Wen <@liwenkkklll>
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
 * \file add_v2_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_ADDV2_H_
#define OPS_OP_PROTO_INC_ADDV2_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Returns x1 + x2.
*@par Inputs:
*Two inputs, including:
* @li x1: An ND Tensor. Must be one of the following types: float32, float16, int16, int32.
* @li x2: An ND Tensor. Must be one of the following types: float32, float16, int16, int32. \n

*@par Outputs:
*y: An ND Tensor. Must be one of the following types: float32, float16, int16, int32.
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator AddV2.
*/
REG_OP(AddV2)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32}))
    .OP_END_FACTORY_REG(AddV2)

} // namespace ge

#endif // OPS_OP_PROTO_INC_AddV2_H_
