/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "util/const_util.h"
#include "log/log.h"
#include "util/shape_util.h"
#include <limits>
namespace ops {
constexpr int64_t MAX_OUTPUT_ELEMENTS = std::numeric_limits<int64_t>::max() / sizeof(float);

static ge::graphStatus CheckDenseBincountShapes(const gert::InferShapeContext* context, const gert::Shape& input,
                                                const gert::Shape& sizeShape, const gert::Shape& weights)
{
    const bool validSizeShape = sizeShape.IsScalar() ||
                                (sizeShape.GetDimNum() == 1 && (sizeShape.GetDim(0) == -1 || sizeShape.GetDim(0) == 1));
    if (!Ops::Base::IsUnknownRank(sizeShape) && !validSizeShape) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "size", Ops::Base::ToString(sizeShape),
                                              "size must be a scalar or 1D tensor with shape [1]");
        return ge::GRAPH_FAILED;
    }
    if (!Ops::Base::IsUnknownRank(weights) && weights.GetDimNum() > 2) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "weights",
                                                 std::to_string(weights.GetDimNum()).c_str(), "rank at most 2");
        return ge::GRAPH_FAILED;
    }
    if (Ops::Base::IsUnknownRank(input)) {
        return ge::GRAPH_SUCCESS;
    }
    if (input.GetDimNum() < 1 || input.GetDimNum() > 2) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "input",
                                                 std::to_string(input.GetDimNum()).c_str(), "1D or 2D tensor");
        return ge::GRAPH_FAILED;
    }
    const int64_t inputElements = input.GetShapeSize();
    const int64_t weightElements = weights.GetShapeSize();
    if (inputElements >= 0 && weightElements > 0 && weightElements != inputElements) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "input, weights",
                                               Ops::Base::ToString(input) + ", " + Ops::Base::ToString(weights),
                                               "non-empty weights must have the same element count as input");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetDenseBincountOutputShape(const gert::InferShapeContext* context, const gert::Shape& input,
                                                   int64_t size, gert::Shape& output)
{
    const int64_t rows = input.GetDimNum() == 1 ? 1 : input.GetDim(0);
    if (rows >= 0 && rows > 0 && size > MAX_OUTPUT_ELEMENTS / rows) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "output.numel", "overflow",
                                              "output element count exceeds the int64 FLOAT32 storage limit");
        return ge::GRAPH_FAILED;
    }
    output.SetDimNum(input.GetDimNum());
    output.SetDim(0, input.GetDim(0));
    output.SetDim(input.GetDimNum() == 1 ? 0 : 1, size);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeDenseBincount(gert::InferShapeContext* context)
{
    const auto* input = context->GetInputShape(0);
    const auto* sizeShape = context->GetInputShape(1);
    const auto* weights = context->GetInputShape(2);
    const auto* sizeTensor = context->GetInputTensor(1);
    auto* output = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input);
    OP_CHECK_NULL_WITH_CONTEXT(context, sizeShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weights);
    OP_CHECK_NULL_WITH_CONTEXT(context, sizeTensor);
    OP_CHECK_NULL_WITH_CONTEXT(context, output);
    if (CheckDenseBincountShapes(context, *input, *sizeShape, *weights) != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    if (Ops::Base::IsUnknownRank(*input)) {
        Ops::Base::SetUnknownRank(*output);
        return ge::GRAPH_SUCCESS;
    }
    int64_t size = 0;
    if (!Ops::Base::GetConstInt(context, 1, size)) {
        return SetDenseBincountOutputShape(context, *input, -1, *output);
    }
    if (size < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "size", std::to_string(size),
                                              "size must be a non-negative constant");
        return ge::GRAPH_FAILED;
    }
    return SetDenseBincountOutputShape(context, *input, size, *output);
}
IMPL_OP_INFERSHAPE(DenseBincount).InferShape(InferShapeDenseBincount).InputsDataDependency({1});
} // namespace ops
