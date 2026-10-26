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
 * \file switch_by_index_proto.h
 * \brief Definition of the SwitchByIndex operator.
 */
#ifndef SWITCH_BY_INDEX_PROTO_H_
#define SWITCH_BY_INDEX_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Writes the input data of the corresponding subscript to the specified register.
 * @par Inputs:
 * Two inputs:
 * @li x1: A 1D tensor, dtype is int32, format is ND, shape is (1,).
 * @li x2: A 1D tensor, dtype is uint64, the format is ND.
 */
REG_OP(SwitchByIndex)
    .INPUT(x1, TensorType({DT_INT32}))
    .INPUT(x2, TensorType({DT_UINT64}))
    .OP_END_FACTORY_REG(SwitchByIndex)
} // namespace ge
#endif // SWITCH_BY_INDEX_PROTO_H_
