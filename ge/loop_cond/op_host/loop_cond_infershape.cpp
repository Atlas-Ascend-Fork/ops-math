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
 * \file loop_cond_infershape.cpp
 * \brief InferShape and InferDataType implementation for LoopCond operator.
 */
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "log/log.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t kInputIndex = 0U;
constexpr size_t kOutputIndex = 0U;
constexpr int64_t kUnknownRankDim = -2;

inline bool IsUnknownRank(const gert::Shape* shape)
{
    return (shape->GetDimNum() == 1U) && (shape->GetDim(0) == kUnknownRankDim);
}
} // namespace

static ge::graphStatus InferShape4LoopCond(gert::InferShapeContext* context)
{
    const gert::Shape* inShape = context->GetInputShape(kInputIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, inShape);
    gert::Shape* outShape = context->GetOutputShape(kOutputIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    if (!IsUnknownRank(inShape) && (inShape->GetDimNum() != 0U)) {
        OP_LOGE(context->GetNodeName(),
                "[InferShape][Check] Check shape rank failed, input x should be a scalar, "
                "actually rank=%zu",
                inShape->GetDimNum());
        return ge::GRAPH_FAILED;
    }
    *outShape = *inShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4LoopCond(gert::InferDataTypeContext* context)
{
    const ge::DataType inDtype = context->GetInputDataType(kInputIndex);
    if (inDtype != ge::DT_BOOL) {
        OP_LOGE(context->GetNodeName(),
                "[InferDataType][Check] Check dtype failed, input x should be DT_BOOL, "
                "actually is %d",
                static_cast<int32_t>(inDtype));
        return ge::GRAPH_FAILED;
    }
    return context->SetOutputDataType(kOutputIndex, inDtype);
}

IMPL_OP_INFERSHAPE(LoopCond).InferShape(InferShape4LoopCond).InferDataType(InferDataType4LoopCond);

} // namespace ops
