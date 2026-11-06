/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file bincount_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_BINCOUNT_H_
#define OPS_OP_PROTO_INC_BINCOUNT_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Counts the number of occurrences of each value in an integer array.
Outputs a vector with length size and the same dtype as weights. If weights
are empty, then index i stores the number of times the value i is counted in
arr. If weights are non-empty, then index i stores the sum of the value in
weights at each index.

*@par Inputs:
*The input size must be a non-negative int32 scalar Tensor. Inputs include:
*@li array:1D Tensor for each value in the range [0, size). Every element of
array must satisfy 0 <= value < size. This is a caller obligation: the
operator does not currently clamp or reject out-of-range elements, and
passing them leads to out-of-bounds accumulation.
*@li size:non-negative int32 scalar Tensor. Its constant value must be
resolvable at graph compile time, otherwise tiling fails.
*@li weights: is an int32, int64, float32, or double Tensor with the same
shape as arr, or a length-0 Tensor, in which case it acts as all weights
equal to 1. The dtype of weights must be identical to the dtype of bins. \n

*@par Outputs:
*bins:1D Tensor with length equal to size. The counts or summed weights for
each value in the range [0, size). The length of bins must be exactly equal
to the value of size. \n

*@attention The declared weights/bins dtype domain below is wider than the
dtype domain currently registered in the operator information library and
accepted by tiling. Unifying the dtype domain across layers is pending a
separate decision; refer to README.md for the dtype range that is executable
today.

*@par Third-party framework compatibility
*Compatible with tensorflow Bincount operator. A TensorFlow graph parser
plugin is provided in framework/bincount_tf_plugin.cpp to map TensorFlow
Bincount nodes onto this operator IR. No end-to-end TensorFlow or PyTorch
binding is delivered for this operator.
*/

REG_OP(Bincount)
    .INPUT(array, TensorType(DT_INT32))
    .INPUT(size, TensorType(DT_INT32))
    .INPUT(weights, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_DOUBLE}))
    .OUTPUT(bins, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_DOUBLE}))
    .OP_END_FACTORY_REG(Bincount)

} // namespace ge

#endif // OPS_OP_PROTO_INC_BINCOUNT_H_
