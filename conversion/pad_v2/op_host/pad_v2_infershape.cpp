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
 * \file pad_v2_infershape.cpp
 * \brief PadV2 InferShape implementation
 */
#include <string>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"
#include "op_api/op_util.h"

using namespace ge;

namespace {
constexpr size_t INDEX_X = 0;
constexpr size_t INDEX_PADDINGS = 1;
constexpr size_t INDEX_Y = 0;
constexpr size_t PAIR = 2;
static constexpr int64_t UNKNOWN_DIM_VALUE_ = -1L;
} // namespace

namespace ops {

template <typename T>
static ge::graphStatus PadV2Infershape(const gert::InferShapeContext* context, const gert::Shape* x_shape,
                                       const gert::Tensor* paddings_tensor, gert::Shape* y_shape)
{
    const T* paddings_value = paddings_tensor->GetData<T>();
    const size_t paddings_num = static_cast<size_t>(paddings_tensor->GetShapeSize());

    OP_LOGD(context->GetNodeName(), "Begin to do PadV2Infershape");
    OP_LOGD(context->GetNodeName(), "input x = %s", Ops::Base::ToString(*x_shape).c_str());

    if (paddings_num > 0 && paddings_value == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "paddings");
        return ge::GRAPH_FAILED;
    }

    // input shape check
    size_t input_dim_size = x_shape->GetDimNum();
    if (input_dim_size == 0) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x", std::to_string(input_dim_size).c_str(),
                                                 "input x must have at least one dimension");
        return ge::GRAPH_FAILED;
    }

    // pad size check
    if (input_dim_size * PAIR != paddings_num) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            context->GetNodeName(), "paddings", std::to_string(paddings_num).c_str(),
            ("paddings must contain 2 * rank(x) = " + std::to_string(input_dim_size * PAIR) + " elements").c_str());
        return ge::GRAPH_FAILED;
    }

    // infer output shape (paddings_contiguous is always true for PadV2)
    y_shape->SetDimNum(input_dim_size);
    for (size_t i = 0; i < input_dim_size; ++i) {
        auto pad_front = paddings_value[PAIR * i]; // paddings_contiguous=true
        auto pad_end = paddings_value[PAIR * i + 1];

        int64_t dim_value = x_shape->GetDim(i) == UNKNOWN_DIM_VALUE_ ? UNKNOWN_DIM_VALUE_ :
                                                                       (x_shape->GetDim(i) + pad_front + pad_end);
        if (x_shape->GetDim(i) != UNKNOWN_DIM_VALUE_ && dim_value < 0) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                context->GetNodeName(), "x, paddings", std::to_string(dim_value).c_str(),
                (std::string("x[") + std::to_string(i) + "] + paddings[" + std::to_string(PAIR * i) + "] + paddings[" +
                 std::to_string(PAIR * i + 1) + "] = " + std::to_string(x_shape->GetDim(i)) + " + (" +
                 std::to_string(pad_front) + ") + (" + std::to_string(pad_end) + ") = " + std::to_string(dim_value) +
                 ", but output shape cannot contain negative values")
                    .c_str());
            return ge::GRAPH_FAILED;
        }
        y_shape->SetDim(i, dim_value);
    }

    OP_LOGD(context->GetNodeName(), "output y = %s", Ops::Base::ToString(*y_shape).c_str());
    OP_LOGD(context->GetNodeName(), "End to do PadV2Infershape");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetAllUnknownDim(const int64_t rank, gert::Shape* output_shape)
{
    output_shape->SetDimNum(rank);
    for (int64_t i = 0; i < rank; ++i) {
        output_shape->SetDim(i, UNKNOWN_DIM_VALUE_);
    }
    OP_LOGD("SetAllUnknownDim", "set all dim = -1, output = %s", Ops::Base::ToString(*output_shape).c_str());

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4PadV2(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(INDEX_X);
    if (x_shape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "x");
        return ge::GRAPH_FAILED;
    }
    gert::Shape* y_shape = context->GetOutputShape(INDEX_Y);
    if (y_shape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "y");
        return ge::GRAPH_FAILED;
    }

    // if x_shape is unknown rank [-2] that means cannot know how many ranks,
    // which make output unknown rank.
    if (Ops::Base::IsUnknownRank(*x_shape)) {
        Ops::Base::SetUnknownRank(*y_shape);
        return GRAPH_SUCCESS;
    }

    const gert::Tensor* paddings_tensor = context->GetInputTensor(INDEX_PADDINGS);
    if (paddings_tensor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "paddings");
        return ge::GRAPH_FAILED;
    }

    if (!IsConstTensor(paddings_tensor)) {
        return SetAllUnknownDim(x_shape->GetDimNum(), y_shape);
    }

    ge::DataType paddings_dtype = paddings_tensor->GetDataType();
    switch (paddings_dtype) {
        case ge::DT_INT32: {
            return PadV2Infershape<int32_t>(context, x_shape, paddings_tensor, y_shape);
        }
        case ge::DT_INT64: {
            return PadV2Infershape<int64_t>(context, x_shape, paddings_tensor, y_shape);
        }
        default:
            OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "paddings", Ops::Base::ToString(paddings_dtype).c_str(),
                                      "int32 or int64");
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_FAILED;
}

IMPL_OP_INFERSHAPE(PadV2).InferShape(InferShape4PadV2).InputsDataDependency({INDEX_PADDINGS});
} // namespace ops
