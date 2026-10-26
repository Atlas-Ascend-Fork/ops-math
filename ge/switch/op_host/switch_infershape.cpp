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
 * \file switch_infershape.cpp
 * \brief InferShape and InferDataType implementation for Switch operator.
 */
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "log/log.h"

using namespace ge;

namespace ops {
namespace {
constexpr int64_t kUnknownRankDim = -2;

bool IsUnknownRankShape(const gert::Shape& shape)
{
    return shape.GetDimNum() == 1U && shape.GetDim(0) == kUnknownRankDim;
}
} // namespace

static ge::graphStatus InferShape4Switch(gert::InferShapeContext* context)
{
    const auto* data_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, data_shape);
    const auto* pred_shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, pred_shape);

    if (!IsUnknownRankShape(*pred_shape) && pred_shape->GetDimNum() != 0U) {
        OP_LOGE(context->GetNodeName(), "input pred should be a scalar, but its rank is %zu", pred_shape->GetDimNum());
        return ge::GRAPH_FAILED;
    }

    auto* output_false_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_false_shape);
    auto* output_true_shape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_true_shape);

    *output_false_shape = *data_shape;
    *output_true_shape = *data_shape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4Switch(gert::InferDataTypeContext* context)
{
    const auto pred_dtype = context->GetInputDataType(1);
    if (pred_dtype != ge::DT_BOOL) {
        OP_LOGE(context->GetNodeName(), "input pred should be DT_BOOL, but got %d", static_cast<int>(pred_dtype));
        return ge::GRAPH_FAILED;
    }
    const auto data_dtype = context->GetInputDataType(0);
    (void)context->SetOutputDataType(0, data_dtype);
    (void)context->SetOutputDataType(1, data_dtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Switch).InferShape(InferShape4Switch).InferDataType(InferDataType4Switch);

} // namespace ops
