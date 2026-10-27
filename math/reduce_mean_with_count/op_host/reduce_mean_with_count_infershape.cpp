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
 * \file reduce_mean_with_count_infershape.cpp
 * \brief InferShape for ReduceMeanWithCount operator
 *
 * Prototype: INPUT(x, count, count_sum) -> OUTPUT(y), ATTR(axes[required], keep_dims)
 * axes is a REQUIRED_ATTR(ListInt), not an input tensor.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_host/infershape_reduce_util.h"

using namespace ge;
using namespace Ops::Base;

namespace ops {
static ge::graphStatus InferShape4ReduceMeanWithCount(gert::InferShapeContext* context)
{
    auto inShape = context->GetInputShape(0);
    if (inShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "x");
        return ge::GRAPH_FAILED;
    }
    auto outShape = context->GetOutputShape(0);
    if (outShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "y");
        return ge::GRAPH_FAILED;
    }
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "attrs");
        return ge::GRAPH_FAILED;
    }

    // Read axes from attribute (ListInt), not from input tensor
    auto axesListPtr = attrs->GetListInt(0);               // attr index 0 = axes
    const bool* keepDims = attrs->GetAttrPointer<bool>(1); // attr index 1 = keep_dims
    if (keepDims == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "keep_dims");
        return ge::GRAPH_FAILED;
    }

    // Build axes vector with negative index normalization
    int64_t dimNum = static_cast<int64_t>(inShape->GetDimNum());
    std::vector<int64_t> axesVec;
    if (axesListPtr == nullptr || axesListPtr->GetSize() == 0) {
        // Empty axes means reduce all dims
        for (int64_t i = 0; i < dimNum; i++) {
            axesVec.push_back(i);
        }
    } else {
        for (size_t i = 0; i < axesListPtr->GetSize(); i++) {
            int64_t ax = axesListPtr->GetData()[i];
            if (ax < 0) {
                ax += dimNum;
            }
            axesVec.push_back(ax);
        }
    }

    auto axesSize = static_cast<int32_t>(axesVec.size());
    if (*keepDims) {
        return ReduceDimsWithKeepDims<int64_t>(inShape, axesVec.data(), axesSize, outShape);
    }
    return ReduceDimsWithoutKeepDims<int64_t>(inShape, axesVec.data(), axesSize, outShape);
}

IMPL_OP_INFERSHAPE(ReduceMeanWithCount).InferShape(InferShape4ReduceMeanWithCount);
} // namespace ops
