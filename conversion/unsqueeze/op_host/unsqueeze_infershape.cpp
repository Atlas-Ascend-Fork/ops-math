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
 * \file unsqueeze_infershape.cpp
 * \brief InferShape and InferDataType implementation for Unsqueeze operator.
 */
#include <string>
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "log/log.h"

using namespace ge;
namespace ops {
namespace {
// validate axes in range [-out_dim_size, out_dim_size) without duplication, then insert dim 1 at
// axes positions and copy x dims in order elsewhere; unknown dim(-1) of input is copied as-is;
// axes is empty means output shape is the same as input
ge::graphStatus UnsqueezeImpl(const gert::Shape* x_shape, const gert::TypedContinuousVector<int64_t>* axes,
                              gert::Shape* y_shape)
{
    const auto out_dim_size = x_shape->GetDimNum() + axes->GetSize();
    if (out_dim_size > gert::Shape::kMaxDimNum) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("Unsqueeze", "y", std::to_string(out_dim_size).c_str(),
                                                 "DimNum of output shape must be less than or equal to kMaxDimNum");
        return ge::GRAPH_FAILED;
    }
    bool insert_dims[gert::Shape::kMaxDimNum] = {false};
    for (size_t idx = 0U; idx < axes->GetSize(); ++idx) {
        const auto raw_axis = (axes->GetData())[idx];
        const auto real_axis = (raw_axis < 0) ? (raw_axis + static_cast<int64_t>(out_dim_size)) : raw_axis;
        if ((real_axis < 0) || (real_axis >= static_cast<int64_t>(out_dim_size))) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                "Unsqueeze", "axes", std::to_string(raw_axis).c_str(),
                ("Each value of axes must be in the range of [-" + std::to_string(out_dim_size) + ", " +
                 std::to_string(out_dim_size) + ")")
                    .c_str());
            return ge::GRAPH_FAILED;
        }
        if (insert_dims[real_axis]) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("Unsqueeze", "axes", std::to_string(raw_axis).c_str(),
                                                  "The values of axes must not be duplicated");
            return ge::GRAPH_FAILED;
        }
        insert_dims[real_axis] = true;
    }
    y_shape->SetDimNum(out_dim_size);
    size_t x_pos = 0U;
    for (size_t idx = 0U; idx < out_dim_size; ++idx) {
        if (insert_dims[idx]) {
            y_shape->SetDim(idx, 1);
        } else {
            y_shape->SetDim(idx, x_shape->GetDim(x_pos));
            ++x_pos;
        }
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

static ge::graphStatus InferShapeForUnsqueeze(gert::InferShapeContext* context)
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
    const auto axes = attrs->GetListInt(0);

    // case 1: input shape is unknown rank(-2), output shape is unknown rank
    if ((x_shape->GetDimNum() == 1U) && (x_shape->GetDim(0U) == ge::UNKNOWN_DIM_NUM)) {
        OP_LOGD(context, "Input shape is unknown rank. Set output shape as unknown rank.");
        y_shape->SetDimNum(1U);
        y_shape->SetDim(0U, ge::UNKNOWN_DIM_NUM);
        return ge::GRAPH_SUCCESS;
    }

    // case 2: the axes attr is optional with an empty-list default: a missing attr is the same
    // as an explicitly empty one, no dim is inserted and the output keeps the input shape
    if (axes == nullptr) {
        y_shape->SetDimNum(x_shape->GetDimNum());
        for (size_t i = 0U; i < x_shape->GetDimNum(); ++i) {
            y_shape->SetDim(i, x_shape->GetDim(i));
        }
        return ge::GRAPH_SUCCESS;
    }

    // case 3: insert dim 1 at axes positions
    return UnsqueezeImpl(x_shape, axes, y_shape);
}

static ge::graphStatus InferDataTypeForUnsqueeze(gert::InferDataTypeContext* context)
{
    return context->SetOutputDataType(0, context->GetInputDataType(0));
}

IMPL_OP_INFERSHAPE(Unsqueeze).InferShape(InferShapeForUnsqueeze).InferDataType(InferDataTypeForUnsqueeze);
} // namespace ops
