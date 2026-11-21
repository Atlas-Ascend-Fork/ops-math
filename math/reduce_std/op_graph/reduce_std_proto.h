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
 * \file reduce_std_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_REDUCE_STD_H_
#define OPS_OP_PROTO_INC_REDUCE_STD_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Calculates the standard deviation and average value of Tensors.

* @par Inputs:
* x: A Tensor. Must be one of the following types:
*     float16, float32. \n

* @par Attributes:
* Three Attributes, including:
* @li dim: An optional listint, Defaults to "[]". \n

* @li unbiased: An optional bool. Defaults to "True".
*     If "True", Use Bessel Correction.
*     If "False", Do not use Bessel Correction. \n

* @li keepdim: An optional bool. Defaults to "False".
*     If "True", Keep the original tensor dimension.
*     If "False", Do not keep the original tensor dimension. \n

* @par Outputs:
* Two Outputs, including:
* @li y1: A Tensor. Has the same type as "x".
* @li y2: A Tensor. Has the same type as "x". \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator ReduceStd.
*/
#ifndef OPS_PROTO_DEF_REDUCESTD
#define OPS_PROTO_DEF_REDUCESTD
REG_OP(ReduceStd)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y1, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y2, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(dim, ListInt, {})
    .ATTR(unbiased, Bool, true)
    .ATTR(keepdim, Bool, false)
    .OP_END_FACTORY_REG(ReduceStd)
#endif
} // namespace ge

#endif // OPS_OP_PROTO_INC_REDUCE_STD_H_
