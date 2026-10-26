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
 * \file if_infershape.cpp
 * \brief Runtime shape and data-type inference for If.
 */
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"

namespace ops {
// If is a control-flow operator whose output shape and data type are determined at execution
// time by the subgraph selected through `cond` (then_branch or else_branch). At InferShape
// stage no concrete shape/dtype can be derived, so both functions return GRAPH_SUCCESS and
// rely on the runtime subgraph execution to produce the actual outputs.
// RT1.0 IfInfer/IfVerify are both empty (no validation), so no cond checks are added here.
static ge::graphStatus InferShapeForIf(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForIf(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(If).InputsDataDependency({0}).InferShape(InferShapeForIf).InferDataType(InferDataTypeForIf);
} // namespace ops
