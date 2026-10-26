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
 * \file for_infershape.cpp
 * \brief Runtime shape and data-type inference for For.
 */
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"

namespace ops {
// For is a control-flow operator that cyclically executes the "body" subgraph
// while the loop counter (initialized from "start", stepped by "delta") does
// not exceed "limit". The output shape and data type are determined at
// execution time by the "body" subgraph. At InferShape stage no concrete
// shape/dtype can be derived, so both functions return GRAPH_SUCCESS and rely
// on the runtime subgraph execution to produce the actual outputs.
// start (input 0), limit (input 1), delta (input 2) must be INT32 scalars —
// migrated from RT1.0 VerifyInt32Scalar.
static ge::graphStatus InferShapeForFor(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0U; i < 3U; i++) {
        const auto* shape = context->GetInputShape(i);
        if (shape == nullptr || !shape->IsScalar()) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForFor(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0U; i < 3U; i++) {
        if (context->GetInputDataType(i) != ge::DT_INT32) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(For).InferShape(InferShapeForFor).InferDataType(InferDataTypeForFor);
} // namespace ops
