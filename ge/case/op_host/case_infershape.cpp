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
 * \file case_infershape.cpp
 * \brief Runtime shape and data-type inference for Case.
 */
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"

namespace ops {
// Case is a control-flow operator whose output shape and data type are determined at
// execution time by the subgraph selected through `branch_index`. At InferShape stage no
// concrete shape/dtype can be derived, so both functions return GRAPH_SUCCESS and rely on
// the runtime subgraph execution to produce the actual outputs.
// branch_index (input 0) must be an INT32 scalar — migrated from RT1.0 VerifyInt32Scalar.
static ge::graphStatus InferShapeForCase(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto* branch_shape = context->GetInputShape(0U);
    if (branch_shape == nullptr || !branch_shape->IsScalar()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForCase(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputDataType(0U) != ge::DT_INT32) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Case).InputsDataDependency({0}).InferShape(InferShapeForCase).InferDataType(InferDataTypeForCase);
} // namespace ops
