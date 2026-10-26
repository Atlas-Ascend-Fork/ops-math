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
 * \file next_iteration_infershape.cpp
 * \brief InferShape and InferDataType implementation for NextIteration operator.
 */
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "log/log.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t kInputIndex = 0U;
constexpr size_t kOutputIndex = 0U;
} // namespace

static ge::graphStatus InferShape4NextIteration(gert::InferShapeContext* context)
{
    const gert::Shape* inShape = context->GetInputShape(kInputIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, inShape);
    gert::Shape* outShape = context->GetOutputShape(kOutputIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    *outShape = *inShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4NextIteration(gert::InferDataTypeContext* context)
{
    return context->SetOutputDataType(kOutputIndex, context->GetInputDataType(kInputIndex));
}

IMPL_OP_INFERSHAPE(NextIteration).InferShape(InferShape4NextIteration).InferDataType(InferDataType4NextIteration);

} // namespace ops
