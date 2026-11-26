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
 * \file select_v2_infershape.cpp
 * \brief SelectV2 三输入同形形状校验与输出数据类型推导。
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t COND_INPUT_IDX = 0;
constexpr size_t SELF_INPUT_IDX = 1;
constexpr size_t OTHER_INPUT_IDX = 2;
constexpr size_t OUT_IDX = 0;

bool IsSameShape(const gert::Shape& lhs, const gert::Shape& rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t axis = 0; axis < lhs.GetDimNum(); ++axis) {
        if (lhs.GetDim(axis) != rhs.GetDim(axis)) {
            return false;
        }
    }
    return true;
}
} // namespace

static ge::graphStatus InferShapeForSelectV2(gert::InferShapeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    const gert::Shape* condShape = context->GetInputShape(COND_INPUT_IDX);
    const gert::Shape* selfShape = context->GetInputShape(SELF_INPUT_IDX);
    const gert::Shape* otherShape = context->GetInputShape(OTHER_INPUT_IDX);
    gert::Shape* outShape = context->GetOutputShape(OUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, condShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, selfShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, otherShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    if (!IsSameShape(*condShape, *selfShape) || !IsSameShape(*condShape, *otherShape)) {
        OP_LOGE(context->GetNodeName(), "SelectV2 requires condition, self and other to have the exact same shape.");
        return GRAPH_FAILED;
    }

    const size_t rank = condShape->GetDimNum();
    outShape->SetDimNum(rank);
    for (size_t axis = 0; axis < rank; ++axis) {
        outShape->SetDim(axis, condShape->GetDim(axis));
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForSelectV2(gert::InferDataTypeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    context->SetOutputDataType(OUT_IDX, context->GetInputDataType(SELF_INPUT_IDX));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SelectV2).InferShape(InferShapeForSelectV2).InferDataType(InferDataTypeForSelectV2);
} // namespace ops
