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
 * \file unsqueezev3_infershape.cpp
 * \brief InferShape and InferDataType implementation for UnsqueezeV3 operator.
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
ge::graphStatus UnsqueezeImpl(const gert::Shape* x_shape, const int64_t* axes_data, const size_t axes_size,
                              gert::Shape* y_shape)
{
    const auto out_dim_size = x_shape->GetDimNum() + axes_size;
    if (out_dim_size > gert::Shape::kMaxDimNum) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("UnsqueezeV3", "y", std::to_string(out_dim_size).c_str(),
                                                 "DimNum of output shape must be less than or equal to kMaxDimNum");
        return ge::GRAPH_FAILED;
    }
    bool insert_dims[gert::Shape::kMaxDimNum] = {false};
    for (size_t idx = 0U; idx < axes_size; ++idx) {
        const auto raw_axis = axes_data[idx];
        const auto real_axis = (raw_axis < 0) ? (raw_axis + static_cast<int64_t>(out_dim_size)) : raw_axis;
        if ((real_axis < 0) || (real_axis >= static_cast<int64_t>(out_dim_size))) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                "UnsqueezeV3", "axes", std::to_string(raw_axis).c_str(),
                ("Each value of axes must be in the range of [-" + std::to_string(out_dim_size) + ", " +
                 std::to_string(out_dim_size) + ")")
                    .c_str());
            return ge::GRAPH_FAILED;
        }
        if (insert_dims[real_axis]) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("UnsqueezeV3", "axes", std::to_string(raw_axis).c_str(),
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

static ge::graphStatus InferShapeForUnsqueezeV3(gert::InferShapeContext* context)
{
    const auto x_shape = context->GetInputShape(0U);
    auto y_shape = context->GetOutputShape(0U);
    const auto axes = context->GetInputTensor(1U);
    if (x_shape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "x.shape");
        return ge::GRAPH_FAILED;
    }
    if (y_shape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "y.shape");
        return ge::GRAPH_FAILED;
    }
    if (axes == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "axes");
        return ge::GRAPH_FAILED;
    }

    // case 1: input shape is unknown rank(-2), output shape is unknown rank
    if ((x_shape->GetDimNum() == 1U) && (x_shape->GetDim(0U) == ge::UNKNOWN_DIM_NUM)) {
        OP_LOGD(context, "Input shape is unknown rank. Set output shape as unknown rank.");
        y_shape->SetDimNum(1U);
        y_shape->SetDim(0U, ge::UNKNOWN_DIM_NUM);
        return ge::GRAPH_SUCCESS;
    }

    // the axes input is constrained to DT_INT64 by the proto, other dtypes would be misread as
    // int64 and cause misparse or out of range access
    if (axes->GetDataType() != ge::DT_INT64) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON("UnsqueezeV3", "axes",
                                              std::to_string(static_cast<int32_t>(axes->GetDataType())).c_str(),
                                              "dtype of the axes input must be DT_INT64");
        return ge::GRAPH_FAILED;
    }

    // case 2: axes data is not got on compile stage (e.g. axes comes from upstream
    // computation), set output shape as unknown rank
    const auto axes_data = static_cast<const int64_t*>(axes->GetTensorData().GetAddr());
    if (axes_data == nullptr) {
        OP_LOGD(context, "Axes data is not got on compile stage. Set output shape as unknown rank.");
        y_shape->SetDimNum(1U);
        y_shape->SetDim(0U, ge::UNKNOWN_DIM_NUM);
        return ge::GRAPH_SUCCESS;
    }

    // case 3: insert dim 1 at axes positions
    return UnsqueezeImpl(x_shape, axes_data, static_cast<size_t>(axes->GetShapeSize()), y_shape);
}

static ge::graphStatus InferDataTypeForUnsqueezeV3(gert::InferDataTypeContext* context)
{
    return context->SetOutputDataType(0, context->GetInputDataType(0));
}

IMPL_OP_INFERSHAPE(UnsqueezeV3)
    .InferShape(InferShapeForUnsqueezeV3)
    .InferDataType(InferDataTypeForUnsqueezeV3)
    .InputsDataDependency({1U});
} // namespace ops
