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
 * \file identity_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"
#include "runtime/storage_shape.h"
#include "log/log.h"

#include <vector>

using namespace gert;
namespace ops {
const std::string OP_NAME = "Identity";
static ge::graphStatus InferShapeForIdentity(InferShapeContext* context)
{
    if (context == nullptr) {
        OP_LOGE(OP_NAME, "InferShape context is nullptr.");
        return ge::GRAPH_FAILED;
    }
    auto input_num = context->GetComputeNodeInputNum();
    if (input_num != context->GetComputeNodeOutputNum()) {
        OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(
            OP_NAME, "x and y",
            (std::to_string(input_num) + " and " + std::to_string(context->GetComputeNodeOutputNum())).c_str(),
            "the number of inputs must be equal to the number of outputs");
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0U; i < input_num; ++i) {
        auto xshape = context->GetInputShape(i);
        auto yshape = context->GetOutputShape(i);
        if ((xshape == nullptr) || (yshape == nullptr)) {
            OP_LOGE(OP_NAME, "Input shape or output shape is nullptr, index %zu.", i);
            return ge::GRAPH_FAILED;
        }
        *yshape = *xshape;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForIdentity(InferDataTypeContext* context)
{
    if (context == nullptr) {
        OP_LOGE(OP_NAME, "InferDataType context is nullptr.");
        return ge::GRAPH_FAILED;
    }
    return context->SetOutputDataType(0U, context->GetInputDataType(0U));
}

IMPL_OP_INFERSHAPE(Identity).InferShape(InferShapeForIdentity).InferDataType(InferDataTypeForIdentity);
} // namespace ops
