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
 * \file unsqueezev2_infershape.cpp
 * \brief InferShape and InferDataType implementation for UnsqueezeV2 operator.
 */
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

ge::graphStatus SetOutputShapeAsUnknownRank(gert::Shape* y_shape)
{
    y_shape->SetDimNum(1U);
    y_shape->SetDim(0U, ge::UNKNOWN_DIM_NUM);
    return ge::GRAPH_SUCCESS;
}

// insert 1-dims at the axes positions into x, negative axis is normalized by the y dim num;
// repeated axes (on normalized positions, same as the RT1.0 normalize-sort-unique check) are rejected
ge::graphStatus UnsqueezeImpl(const gert::Shape* x_shape, const gert::TypedContinuousVector<int64_t>* axes,
                              gert::Shape* y_shape)
{
    const auto x_dim_num = x_shape->GetDimNum();
    const auto y_dim_num = x_dim_num + axes->GetSize();
    if (y_dim_num > gert::Shape::kMaxDimNum) {
        OP_LOGE("UnsqueezeV2", "DimNum of output shape is %zu, larger than kMaxDimNum which is %zu!", y_dim_num,
                gert::Shape::kMaxDimNum);
        return ge::GRAPH_FAILED;
    }
    y_shape->SetDimNum(y_dim_num);
    for (size_t i = 0U; i < y_dim_num; ++i) {
        y_shape->SetDim(i, 0);
    }
    const auto signed_y_dim_num = static_cast<int64_t>(y_dim_num);
    for (size_t i = 0U; i < axes->GetSize(); ++i) {
        const auto raw_axis = axes->GetData()[i];
        const auto real_axis = (raw_axis < 0) ? (raw_axis + signed_y_dim_num) : raw_axis;
        if ((real_axis < 0) || (real_axis >= signed_y_dim_num)) {
            OP_LOGE("UnsqueezeV2", "UnsqueezeV2 axis[%ld] is not in range [-%ld, %ld)!", raw_axis, signed_y_dim_num,
                    signed_y_dim_num);
            return ge::GRAPH_FAILED;
        }
        if (y_shape->GetDim(real_axis) == 1) {
            OP_LOGE("UnsqueezeV2", "UnsqueezeV2 axis repeated!");
            return ge::GRAPH_FAILED;
        }
        y_shape->SetDim(real_axis, 1);
    }

    size_t idx = 0U;
    for (size_t i = 0U; i < y_dim_num; ++i) {
        if (y_shape->GetDim(i) != 1) {
            y_shape->SetDim(i, x_shape->GetDim(idx));
            ++idx;
        }
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

static ge::graphStatus InferShapeForUnsqueezeV2(gert::InferShapeContext* context)
{
    const auto x_shape = context->GetInputShape(0U);
    auto y_shape = context->GetOutputShape(0U);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
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

    // case 2: axis is not set or empty: no dim is inserted, the output shape is the input shape
    if (axes_size == 0U) {
        OP_LOGD(context, "Axis is not set or empty. Set output shape as input shape.");
        *y_shape = *x_shape;
        return ge::GRAPH_SUCCESS;
    }

    // case 3: insert 1-dims at the axes positions
    return UnsqueezeImpl(x_shape, axes, y_shape);
}

static ge::graphStatus InferDataTypeForUnsqueezeV2(gert::InferDataTypeContext* context)
{
    return context->SetOutputDataType(0, context->GetInputDataType(0));
}

IMPL_OP_INFERSHAPE(UnsqueezeV2).InferShape(InferShapeForUnsqueezeV2).InferDataType(InferDataTypeForUnsqueezeV2);
} // namespace ops
