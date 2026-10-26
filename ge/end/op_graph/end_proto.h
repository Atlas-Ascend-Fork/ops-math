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
 * \file end_proto.h
 * \brief Definition of the End operator.
 */
#ifndef END_PROTO_H_
#define END_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 *@brief Mark outputs of one sub graph which partitioned by engine type.

*@par Inputs:
*x: A tensor. \n

*@par Outputs:
*y: A tensor. \n

*@par Attributes:
*@li peerIndex: The index of the corresponding 'placeholder' node it's connected to.
*@li parentOpType: Op type of original node.

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
#ifndef OPS_PROTO_DEF_END
#define OPS_PROTO_DEF_END
REG_OP(End)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(peerIndex, Int, 0)
    .ATTR(parentOpType, String, "")
    .OP_END_FACTORY_REG(End)
#endif
} // namespace ge
#endif // END_PROTO_H_
