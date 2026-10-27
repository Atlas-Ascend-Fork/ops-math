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
 * \file topk_infershape.cpp
 * \brief InferShape for TopK operator: output shape = input shape with dim-th axis replaced by k
 */
#include "register/op_impl_registry.h"
#include "util/shape_util.h"
#include "log/log.h"

static constexpr int OUTPUT_VALUES_INDEX = 0;
static constexpr int OUTPUT_INDICES_INDEX = 1;
static constexpr int INPUT_X_INDEX = 0;

using namespace ge;
namespace ops {

static graphStatus InferShapeForTopKCommon(gert::InferShapeContext* context, int64_t k, const int64_t* dim)
{
    const gert::Shape* input_x_shape = context->GetInputShape(INPUT_X_INDEX);
    if (input_x_shape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "x");
        return ge::GRAPH_FAILED;
    }
    size_t dim_size = input_x_shape->GetDimNum();
    if (dim_size <= 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE(context->GetNodeName(), "x", std::to_string(dim_size).c_str(), "greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (k < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "k", std::to_string(k).c_str(), "k must be >= 0");
        return ge::GRAPH_FAILED;
    }
    int64_t sorted_axis = dim_size - 1;

    if (dim != nullptr) {
        sorted_axis = *dim;
        if (sorted_axis < 0) {
            sorted_axis += dim_size;
        }
        if (sorted_axis < 0 || sorted_axis >= static_cast<int64_t>(dim_size)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "dim", std::to_string(sorted_axis).c_str(),
                                                  "Dim is out of shape size");
            return ge::GRAPH_FAILED;
        }
    }

    int64_t dim_size_on_axis = input_x_shape->GetDim(sorted_axis);
    if (dim_size_on_axis >= 0 && k > dim_size_on_axis) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "k", std::to_string(k).c_str(),
                                              "k must be <= size of dim");
        return ge::GRAPH_FAILED;
    }

    gert::Shape* output_values_shape = context->GetOutputShape(OUTPUT_VALUES_INDEX);
    if (output_values_shape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "y");
        return ge::GRAPH_FAILED;
    }
    gert::Shape* output_indices_shape = context->GetOutputShape(OUTPUT_INDICES_INDEX);
    if (output_indices_shape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "y");
        return ge::GRAPH_FAILED;
    }

    output_values_shape->SetDimNum(dim_size);
    output_indices_shape->SetDimNum(dim_size);
    for (size_t i = 0; i < dim_size; i++) {
        if (static_cast<int64_t>(i) == sorted_axis) {
            output_values_shape->SetDim(i, k);
            output_indices_shape->SetDim(i, k);
            continue;
        }
        output_values_shape->SetDim(i, input_x_shape->GetDim(i));
        output_indices_shape->SetDim(i, input_x_shape->GetDim(i));
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferShapeForTopK(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do TopKInferShape");
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "attrs");
        return ge::GRAPH_FAILED;
    }
    const gert::Tensor* input_k_tensor = context->GetInputTensor(1U);
    if (input_k_tensor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "k");
        return ge::GRAPH_FAILED;
    }
    const int32_t* k = input_k_tensor->GetData<int32_t>();
    if (k == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "k");
        return ge::GRAPH_FAILED;
    }
    const int64_t* dim = attrs->GetInt(2U);
    return InferShapeForTopKCommon(context, *k, dim);
}

IMPL_OP_INFERSHAPE(TopK).InferShape(InferShapeForTopK).InputsDataDependency({1});

} // namespace ops
