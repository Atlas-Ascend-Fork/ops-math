/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file asinh_grad_proto.h
 * \brief AsinhGrad GEIR prototype and parameter constraints.
 */
#ifndef ASINH_GRAD_PROTO_H_
#define ASINH_GRAD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Computes the Asinh gradient z = dy / cosh(y).
 *
 * @par Inputs:
 * @li y: Required ND tensor of type float16, float32 or bfloat16.
 * The unscaled forward output y = asinh(x).
 * @li dy: Required ND tensor with the same dtype as y. Shapes are compared
 * after treating a rank-0 scalar as [1], for A2 compatibility. Other nonempty
 * shapes must match; general broadcasting is not supported.
 *
 * @par Outputs:
 * z: Tensor with the same dtype and original logical shape as y.
 * The gradient with respect to the forward input x.
 *
 * @par Constraints:
 * Normal computation supports ranks 0 through 8. Empty tensors follow GE/A2
 * compatibility behavior, without additional empty-input rejection.
 *
 * @par Third-party framework compatibility:
 * Equivalent in real arithmetic to the TensorFlow Asinh gradient dy / cosh(y).
 * TensorFlow automatic differentiation may express it as a composite graph.
 */
REG_OP(AsinhGrad)
    .INPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(z, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(AsinhGrad)
} // namespace ge
#endif // ASINH_GRAD_PROTO_H_
