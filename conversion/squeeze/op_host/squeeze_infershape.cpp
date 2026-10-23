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
 * \file squeeze_infershape.cpp
 * \brief InferShape and InferDataType implementation for Squeeze operator.
 */
#include <string>
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "log/log.h"

using namespace ge;
namespace ops {
namespace {
bool IsUnknownRank(const gert::Shape& shape)
{
    return (shape.GetDimNum() == 1U) && (shape.GetDim(0U) == ge::UNKNOWN_DIM_NUM);
}

bool IsUnknownDim(const gert::Shape& shape)
{
    for (size_t i = 0U; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) == ge::UNKNOWN_DIM) {
            return true;
        }
    }
    return false;
}

ge::graphStatus SetOutputShapeAsUnknownRank(gert::Shape* y_shape)
{
    y_shape->SetDimNum(1U);
    y_shape->SetDim(0U, ge::UNKNOWN_DIM_NUM);
    return ge::GRAPH_SUCCESS;
}

// squeeze x by squeeze_dims; when squeeze_dims is nullptr, squeeze all dims whose value is 1;
// unknown dim(-1) pointed by axis is treated as 1, as no method to verify it on compile stage
ge::graphStatus SqueezeImpl(const gert::Shape* x_shape, const bool* squeeze_dims, gert::Shape* y_shape)
{
    y_shape->SetDimNum(0U);
    for (size_t idx = 0U; idx < x_shape->GetDimNum(); ++idx) {
        const auto x_shape_dim = x_shape->GetDim(idx);
        const bool squeeze = (squeeze_dims == nullptr) ? (x_shape_dim == 1) : squeeze_dims[idx];
        if (!squeeze) {
            y_shape->AppendDim(x_shape_dim);
        } else if ((x_shape_dim != 1) && (x_shape_dim != ge::UNKNOWN_DIM)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                "Squeeze", "x",
                ("dimension " + std::to_string(idx) + " has size " + std::to_string(x_shape_dim)).c_str(),
                "dimensions selected by axis must have size 1 or be unknown (-1)");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// validate axes in range [-x_dim_num, x_dim_num), mark the squeeze positions on squeeze_dims
bool GetSqueezeDims(const gert::Shape* x_shape, const gert::TypedContinuousVector<int64_t>* axes, bool* squeeze_dims)
{
    const auto x_dim_num = static_cast<int64_t>(x_shape->GetDimNum());
    for (size_t idx = 0U; idx < axes->GetSize(); ++idx) {
        const auto raw_axis = axes->GetData()[idx];
        const auto real_axis = (raw_axis < 0) ? (raw_axis + x_dim_num) : raw_axis;
        if ((real_axis < 0) || (real_axis >= x_dim_num)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                "Squeeze", "axis", std::to_string(raw_axis).c_str(),
                ("axis must be in [" + std::to_string(-x_dim_num) + ", " + std::to_string(x_dim_num) + ")").c_str());
            return false;
        }
        squeeze_dims[real_axis] = true;
    }
    return true;
}
} // namespace

static ge::graphStatus InferShapeForSqueeze(gert::InferShapeContext* context)
{
    const auto x_shape = context->GetInputShape(0U);
    auto y_shape = context->GetOutputShape(0U);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (x_shape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "x.shape");
        return ge::GRAPH_FAILED;
    }
    if (y_shape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "y.shape");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto axes = attrs->GetListInt(0);
    // the axis attr is optional with an empty-list default: a missing attr is the same as an
    // explicitly empty one, both squeeze all 1-dims
    const auto axes_size = (axes == nullptr) ? 0U : axes->GetSize();

    // case 1: input shape is unknown rank(-2), or has unknown dims(-1) while axes is empty,
    // cannot determine which unknown dim is 1, set output shape as unknown rank
    if (IsUnknownRank(*x_shape) || (IsUnknownDim(*x_shape) && (axes_size == 0U))) {
        OP_LOGD(context, "Input shape is unknown rank, or has unknown dims while axes is empty. Set output shape as "
                         "unknown rank.");
        return SetOutputShapeAsUnknownRank(y_shape);
    }

    // case 2: axes is empty, squeeze all dims whose value is 1
    if (axes_size == 0U) {
        OP_LOGD(context, "Axes is empty. Squeeze all dims whose value is 1.");
        return SqueezeImpl(x_shape, nullptr, y_shape);
    }

    // case 3: squeeze by axes, unknown dim(-1) pointed by axis is treated as 1
    bool squeeze_dims[gert::Shape::kMaxDimNum] = {false};
    if (!GetSqueezeDims(x_shape, axes, squeeze_dims)) {
        return ge::GRAPH_FAILED;
    }
    return SqueezeImpl(x_shape, squeeze_dims, y_shape);
}

static ge::graphStatus InferDataTypeForSqueeze(gert::InferDataTypeContext* context)
{
    return context->SetOutputDataType(0U, context->GetInputDataType(0U));
}

IMPL_OP_INFERSHAPE(Squeeze).InferShape(InferShapeForSqueeze).InferDataType(InferDataTypeForSqueeze);
} // namespace ops
