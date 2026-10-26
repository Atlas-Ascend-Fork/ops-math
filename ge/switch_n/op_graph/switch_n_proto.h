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
 * \file switch_n_proto.h
 * \brief Definition of the SwitchN operator.
 */
#ifndef SWITCH_N_PROTO_H_
#define SWITCH_N_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 *@brief Forwards "data" to the output port determined by "pred_value" .

 *@par Inputs:
 *@li data: The tensor to be forwarded.
 *          Must be one of the following types: float16, float32, float64,
 *          int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool.
 *@li pred_value: An int64 tensor which determines the output port that will receive data . \n

 *@par Outputs:
 *output: The output tensors, one of which will become available.
 *        Has the same type as "data". It's a dynamic output.
 */
REG_OP(SwitchN)
    .INPUT(data, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                             DT_UINT16, DT_UINT32, DT_UINT64, DT_BOOL}))
    .INPUT(pred_value, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(output, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64,
                                        DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_BOOL}))
    .OP_END_FACTORY_REG(SwitchN)
} // namespace ge
#endif // SWITCH_N_PROTO_H_
