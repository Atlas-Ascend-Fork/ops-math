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
 * \file exit_proto.h
 * \brief
 */
#ifndef EXIT_PROTO_H_
#define EXIT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 *@brief Exits the current frame to its parent frame .

 *@par Inputs:
 *x: The tensor to be made available to the parent frame.
 *   Must be one of the following types: float16, float32, float64, int8,
 *   int16, int32, int64, uint8, uint16, uint32, uint64, bool . \n

 *@par Outputs:
 *y: A Tensor. Has the same type as "x" . \n

 *@see Enter()

 *@par Third-party framework compatibility
 *Compatible with the TensorFlow operator Exit.
 */
#ifndef OPS_PROTO_DEF_EXIT
#define OPS_PROTO_DEF_EXIT
REG_OP(Exit)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16,
                          DT_UINT32, DT_UINT64, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16,
                           DT_UINT32, DT_UINT64, DT_BOOL}))
    .OP_END_FACTORY_REG(Exit)
#endif
} // namespace ge
#endif // EXIT_PROTO_H_
