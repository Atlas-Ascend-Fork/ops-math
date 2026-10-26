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
 * \file squeezev2_infershape.cpp
 * \brief InferShape and InferDataType implementation for SqueezeV2 operator.
 */
#include <algorithm>
#include <string>
#include <vector>

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
                "SqueezeV2", "x",
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
            OP_LOGW("SqueezeV2", "SqueezeV2 axis[%ld] is not in range [-%ld, %ld)!", raw_axis, x_dim_num, x_dim_num);
            return false;
        }
        squeeze_dims[real_axis] = true;
    }
    return true;
}

// count axes by raw values after sort and unique, equivalent to the RT1.0 rule: axes like {0,-2} on
// a 2-dim x dedups to size 2 >= dim num, squeeze falls back; not the normalized-position count
bool IsAxesNumNotLessThanDimNum(const gert::TypedContinuousVector<int64_t>* axes, int64_t x_dim_num)
{
    std::vector<int64_t> raw_axes(axes->GetData(), axes->GetData() + axes->GetSize());
    std::sort(raw_axes.begin(), raw_axes.end());
    raw_axes.erase(std::unique(raw_axes.begin(), raw_axes.end()), raw_axes.end());
    return raw_axes.size() >= static_cast<size_t>(x_dim_num);
}
} // namespace

static ge::graphStatus InferShapeForSqueezeV2(gert::InferShapeContext* context)
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
    const auto axes = attrs->GetListInt(0U);
    // the axis attr is not set, GetListInt returns nullptr: same as an explicit empty list,
    // as the RT1.0 GetAttr failure is ignored
    const auto axes_size = (axes == nullptr) ? 0U : axes->GetSize();

    // case 1: input shape is unknown rank(-2), output shape is unknown rank
    if (IsUnknownRank(*x_shape)) {
        OP_LOGD(context, "Input shape is unknown rank. Set output shape as unknown rank.");
        return SetOutputShapeAsUnknownRank(y_shape);
    }

    // case 2: axis is not set or empty: squeeze all dims whose value is 1;
    // x has unknown dims(-1) while axis is empty, cannot determine which unknown dim is 1
    if (axes_size == 0U) {
        if (IsUnknownDim(*x_shape)) {
            OP_LOGD(context, "Input shape has unknown dims and axis is empty. Set output shape as unknown rank.");
            return SetOutputShapeAsUnknownRank(y_shape);
        }
        OP_LOGD(context, "Axis is not set or empty. Squeeze all dims whose value is 1.");
        return SqueezeImpl(x_shape, nullptr, y_shape);
    }

    // case 3: V2 special logic of RT1.0: axis out of range, or deduped axis num not less than the
    // x dim num: the output shape falls back to the input shape
    bool squeeze_dims[gert::Shape::kMaxDimNum] = {false};
    if (!GetSqueezeDims(x_shape, axes, squeeze_dims)) {
        OP_LOGW(context, "SqueezeV2 axis is not valid. Set output shape as input shape.");
        *y_shape = *x_shape;
        return ge::GRAPH_SUCCESS;
    }
    if (IsAxesNumNotLessThanDimNum(axes, static_cast<int64_t>(x_shape->GetDimNum()))) {
        OP_LOGW(context, "SqueezeV2 deduped axis num is not less than x dim num. Set output shape as input shape.");
        *y_shape = *x_shape;
        return ge::GRAPH_SUCCESS;
    }

    // case 4: squeeze by axis, unknown dim(-1) pointed by axis is treated as 1
    return SqueezeImpl(x_shape, squeeze_dims, y_shape);
}

static ge::graphStatus InferDataTypeForSqueezeV2(gert::InferDataTypeContext* context)
{
    return context->SetOutputDataType(0, context->GetInputDataType(0));
}

IMPL_OP_INFERSHAPE(SqueezeV2).InferShape(InferShapeForSqueezeV2).InferDataType(InferDataTypeForSqueezeV2);
} // namespace ops
