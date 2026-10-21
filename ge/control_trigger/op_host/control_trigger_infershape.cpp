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
 * \file control_trigger_infershape.cpp
 * \brief Runtime shape and data-type inference for ControlTrigger.
 */
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"

namespace ops {
// ControlTrigger is similar to NoOp — only useful as a placeholder for control
// edges. It has no inputs and no outputs, so there is nothing to infer. Both
// functions return GRAPH_SUCCESS. InferDataType is still registered to
// avoid "Cannot find custom infer_datatype func" on some GE versions.
static ge::graphStatus InferShapeForControlTrigger(gert::InferShapeContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForControlTrigger(gert::InferDataTypeContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ControlTrigger)
    .InferShape(InferShapeForControlTrigger)
    .InferDataType(InferDataTypeForControlTrigger);
} // namespace ops
