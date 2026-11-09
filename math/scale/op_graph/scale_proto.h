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
 * \file scale_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_SCALE_H_
#define OPS_OP_PROTO_INC_SCALE_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Scales the input.

 * @par Inputs:
 * Three inputs, including:
 * @li x: An ND tensor of type float16 or float32 or bfloat16.
 * @li scale: An ND tensor of type float16 or float32 or bfloat16.
 * @li bias: An optional ND tensor of type float16 or float32 or bfloat16. \n

 * @par Attributes:
 * @li axis: An optional int64 used to compute the shape of scale and bias input from the online bottoms.
 *     Defaults to "1".
 * @li num_axes: An optional int64 used to compute the shape of scale and bias input from a Caffe model trained offline.
 *     Defaults to "1".
 * @li scale_from_blob: An optional bool. If "true", scale and bias are input from a Caffe model trained offline.
 *     If "false", scale and bias are input from online bottoms. Defaults to "true" . \n

 * @par Outputs:
 * y: An ND tensor of type float16 or float32 or bfloat16. \n

 * @attention Constraints:
 * Assume that the shape length of "x" is "n" and that of "scale" is "m".
 * @li "axis" is within the range [-n, n-1]. num_axes >= -1.
 * @li If "scale_from_blob = true", "num_axes = -1", and "axis >= 0",
 *     the ith axis of "scale" and the (i+"axis")th axis of "x" must have the same size (0 <= i < n-axis).
 * If "axis < 0", the ith axis of "scale" and the (i+n+"axis")th axis of "x" must have the same size
 *     (0 <= i < -axis).
 * @li If "scale_from_blob = true" and "num_axes = 0", "scale" must have scalar shape ().
 * @li If "scale_from_blob = true", "num_axes > 0", and "axis >= 0", "axis + num_axes" must be less than or equal
 *     to "n" and the ith axis of "scale" and the (i+"axis")th axis of "x" must have the same size
 *     (0 <= i < num_axes).
 * If "axis < 0", "n + axis + num_axes" must be less than or equal to "n" and the ith axis of "scale"
 *     and the (i+n+"axis")th axis of "x" must have the same size (0 <= i < num_axes).
 * @li If "scale_from_blob = false", "scale" is not a scalar, and "axis >= 0", "axis + m" must be less than or
 *     equal to "n" and the ith axis of "scale" and the (i+"axis")th axis of "x" must have the same size
 *     (0 <= i < m), except for the singleton window rule below.
 * If "axis < 0", "n + axis + m" must be less than or equal to "n" and the ith axis of "scale" and
 *     the (i+n+"axis")th axis of "x" must have the same size (0 <= i < m), except for the singleton window rule below.
 * @li With scale_from_blob=false, the window must fit inside x; expanding x is not allowed.
 *     For static 4D descriptors, trailing singleton dimensions of the scale window may be broadcast.
 *     Dynamic aligned 4D runtime descriptors allow each scale dimension to be 1 or match x.
 *     Scalar shapes () and [1] use scalar broadcast.
 * @li Nonempty bias must have the same shape as scale. Tiling treats a zero-element bias as omitted.
 * @li Each graph input may independently use float16, float32 or bfloat16; y follows x.
 *     GE normalizes graph inputs before dispatch to the single-dtype kernel.
 * @li The dynamic host path accepts x ranks 1 through 8 and scale/bias ranks 0 through 8.
 * @li y has the same shape and dtype as x. Concrete dimensions must be nonnegative.
 * @li Graph descriptors may use -1 for unknown dimensions or [-2] for unknown rank, independently for each input.
 *     Runtime inputs must satisfy the concrete rank and window constraints. The same graph may be reused with
 *     different valid concrete ranks; negative descriptor values are not runtime dimensions.
 * @li Empty x is valid after dtype, rank, axis and window validation; y is empty with the same shape.
 *     A zero-element scale must still satisfy its shape constraints.

 * @par Third-party framework compatibility
 * Compatible with the Caffe operator Scale.
 */
#ifndef OPS_PROTO_DEF_SCALE
#define OPS_PROTO_DEF_SCALE
REG_OP(Scale)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(scale, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(axis, Int, 1)
    .ATTR(num_axes, Int, 1)
    .ATTR(scale_from_blob, Bool, true)
    .OP_END_FACTORY_REG(Scale)
#endif

} // namespace ge

#endif // OPS_OP_PROTO_INC_SCALE_H_
