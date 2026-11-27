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
 * \file masked_fill_range_proto.h
 * \brief MaskedFillRange 算子 GE graph 侧 IR 定义
 *
 * 原型照抄 canndev TIK 版算子注册定义（op_proto/inc/selection_ops.h 中的
 * MaskedFillRange 条目），dtype 集与 op_host/masked_fill_range_def.cpp 一致，
 * 形态对齐 conversion 仓既有算子（masked_fill/matrix_diag 等 op_host +
 * op_graph 并存形态）。注意：本文件注释中不得出现算子注册宏调用字样，
 * merge_proto.py 按正则截取注册块，注释内出现会破坏合并产物。
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_SELECTION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SELECTION_OPS_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief masked fill tensor along with one axis by range.
* boxes. It is a customized masked fill range operator . \n

* @par Inputs:
* Four inputs, including:
* @li x: input tensor. A ND Tensor of float32/float16/int32/int8 with shapes
* 1-D (D,), 2-D(N, D), 3-D(N, C, D)
* @li start: masked fill start pos. A 2D Tensor of int32 with
* shape (num, N). "num" indicates the number of loop masked fill, and the value N
* indicates the batch of ND Tensor: for 1-D x, N = 1; for 2-D x (N, D), N is the
* row count; for 3-D x (N, C, D), N equals x.shape[0] and all C rows of the same
* n share window column n.
* @li end: masked fill end pos. A 2D Tensor of int32 with
* shape (num, N). "num" indicates the number of loop masked fill, and the value N
* indicates the batch of ND Tensor (same as start: 1-D N = 1, 2-D N = row count,
* 3-D N = x.shape[0] with all C rows of the same n sharing window column n).
* @li value: masked fill value. A 1D Tensor of float32/float16/int32/int8 with
* shape (num,). "num" indicates the number of loop masked fill. \n

* @par Attributes:
* @li axis: axis with masked fill of int32. Defaults to -1.

* @par Outputs:
* y: A ND Tensor of float32/float16/int32/int8 with shapes 1-D (D,), 2-D(N, D), 3-D(N, C, D)

* @attention Constraints:
* Warning: input shape's length must not be bigger than 1024 * 1024 * 1024.
*/
#ifndef OPS_PROTO_DEF_MASKEDFILLRANGE
#define OPS_PROTO_DEF_MASKEDFILLRANGE
REG_OP(MaskedFillRange)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32}))
    .INPUT(start, TensorType({DT_INT32}))
    .INPUT(end, TensorType({DT_INT32}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32}))
    .REQUIRED_ATTR(axis, Int)
    .OP_END_FACTORY_REG(MaskedFillRange)
#endif
} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_SELECTION_OPS_H_
