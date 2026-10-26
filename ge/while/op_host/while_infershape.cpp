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
 * \file while_infershape.cpp
 * \brief Runtime shape and data-type inference registration for While.
 */
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"

namespace ops {
// While cyclically executes the "body" subgraph until the "cond" subgraph returns false;
// the output shape and data type are determined at execution time by the loop iterations.
// No concrete shape/dtype derivation is implemented at the InferShape stage, consistent
// with the V2 registration delivered by canndev: both functions return GRAPH_SUCCESS and
// the legacy V1 subset keeps the shape-refinement (REPASS) logic.
static ge::graphStatus InferShapeForWhile(gert::InferShapeContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForWhile(gert::InferDataTypeContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(While).InferShape(InferShapeForWhile).InferDataType(InferDataTypeForWhile);
} // namespace ops
