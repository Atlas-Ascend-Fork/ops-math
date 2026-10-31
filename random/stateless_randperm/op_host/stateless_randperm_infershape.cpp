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
 * \file stateless_randperm_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

using namespace ge;
namespace {
constexpr size_t kOutputRank = 1; // 输出恒为 1-D [n]
} // namespace
namespace ops {
template <typename T>
ge::graphStatus StatelessRandpermInferShapeImpl(const T* shapeDims, gert::Shape& outputShape, size_t shapeSize)
{
    outputShape.SetDimNum(shapeSize);
    outputShape.SetDim(0, shapeDims[0]);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForStatelessRandperm(gert::InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do InferShapeForStatelessRandperm.");
    auto nShapeTensor = context->GetInputTensor(0);
    auto outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, nShapeTensor);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    const gert::Shape& nInputDesc = nShapeTensor->GetShape().GetStorageShape();
    if (Ops::Base::IsUnknownRank(nInputDesc) || Ops::Base::IsUnknownShape(nInputDesc)) {
        OP_LOGI(context->GetNodeName(), "n is unknown rank(-2) or unknown shape(-1), set output to unknown.");
        Ops::Base::SetUnknownShape(kOutputRank, *outputShape);
        return ge::GRAPH_SUCCESS;
    }

    auto nShapeSize = nShapeTensor->GetShapeSize();
    if (nShapeSize != 1) {
        OP_LOGE(context->GetNodeName(), "The shape size of n must be 1, but got %ld.",
                static_cast<int64_t>(nShapeSize));
        return ge::GRAPH_FAILED;
    }
    auto nShapeData = nShapeTensor->GetData<int64_t>();
    if (nShapeData == nullptr) {
        OP_LOGW(context->GetNodeName(), "n const data is null, set output to unknown.");
        Ops::Base::SetUnknownShape(kOutputRank, *outputShape);
        return ge::GRAPH_SUCCESS;
    }
    if (nShapeData[0] < 0) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "n", std::to_string(nShapeData[0]).c_str(),
                                     "non-negative");
        return ge::GRAPH_FAILED;
    }
    OP_LOGI(context->GetNodeName(), "Get n value from input tensor, n = %ld.", nShapeData[0]);
    auto ret = StatelessRandpermInferShapeImpl<int64_t>(nShapeData, *outputShape, static_cast<size_t>(nShapeSize));
    OP_LOGI(context->GetNodeName(), "End to do InferShapeForStatelessRandperm, result = %d.",
            static_cast<int32_t>(ret));
    return ret;
}

IMPL_OP_INFERSHAPE(StatelessRandperm).InputsDataDependency({0}).InferShape(InferShapeForStatelessRandperm);
} // namespace ops
