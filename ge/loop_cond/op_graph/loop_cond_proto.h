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
 * \file loop_cond_proto.h
 * \brief Definition of the LoopCond operator.
 */
#ifndef LOOP_COND_PROTO_H_
#define LOOP_COND_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 *@brief Forwards the input to the output. This op represents the loop
 *       termination condition . \n

 *@par Inputs:
 *x: A boolean scalar. The condition of the Switch op . \n

 *@par Outputs:
 *y: The tensor "x" . \n

 *@see Switch()

 *@par Third-party framework compatibility
 *Compatible with the TensorFlow operator LoopCond.
 */
#ifndef OPS_PROTO_DEF_LOOPCOND
#define OPS_PROTO_DEF_LOOPCOND
REG_OP(LoopCond).INPUT(x, TensorType({DT_BOOL})).OUTPUT(y, TensorType({DT_BOOL})).OP_END_FACTORY_REG(LoopCond)
#endif
} // namespace ge
#endif // LOOP_COND_PROTO_H_
