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
 * \file refswitch_proto.h
 * \brief Definition of the RefSwitch operator.
 */
#ifndef REFSWITCH_PROTO_H_
#define REFSWITCH_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 *@brief Forwards "data" to the output port determined by "pred".
 *       If "pred" is "true", the data input is forwarded to "output_true".
 *       Otherwise, the data is forwarded to "output_false" .

 *@par Inputs:
 *@li data: The ref tensor to be forwarded.
 *          Must be one of the following types: float16, float32, float64,
 *          int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool, string.
 *@li pred: A boolean scalar. The output port that will receive data . \n

 *@par Outputs:
 *@li output_false: If "pred" is "false", data will be forwarded to this output.
 *                  Has the same type as "data".
 *@li output_true: If "pred" is "true", data will be forwarded to this output.
 *                 Has the same type as "data" . \n

 *@see Merge() | Switch()

 *@par Third-party framework compatibility
 *Compatible with the TensorFlow operator RefSwitch.
 */
REG_OP(RefSwitch)
    .INPUT(data, TensorType::ALL())
    .INPUT(pred, TensorType({DT_BOOL}))
    .OUTPUT(output_false, TensorType::ALL())
    .OUTPUT(output_true, TensorType::ALL())
    .OP_END_FACTORY_REG(RefSwitch)
} // namespace ge
#endif // REFSWITCH_PROTO_H_
