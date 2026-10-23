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
 * \file confusion_transpose_d_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

#include <string>

using namespace ge;

namespace {
constexpr size_t CONFUSION_TRANSPOSE_D_IN_X = 0U;
constexpr size_t CONFUSION_TRANSPOSE_D_OUT_Y = 0U;
constexpr size_t CONFUSION_TRANSPOSE_D_ATTR_PERM = 0U;
constexpr size_t CONFUSION_TRANSPOSE_D_ATTR_SHAPE = 1U;
constexpr size_t CONFUSION_TRANSPOSE_D_ATTR_TRANSPOSE_FIRST = 2U;
} // namespace

namespace ops {
static ge::graphStatus InferShape4ConfusionTransposeD(gert::InferShapeContext* context)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const gert::TypedContinuousVector<int64_t>* perm = attrs->GetListInt(CONFUSION_TRANSPOSE_D_ATTR_PERM);
    OP_CHECK_NULL_WITH_CONTEXT(context, perm);
    const gert::TypedContinuousVector<int64_t>* shape = attrs->GetListInt(CONFUSION_TRANSPOSE_D_ATTR_SHAPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape);
    const bool* transposeFirst = attrs->GetAttrPointer<bool>(CONFUSION_TRANSPOSE_D_ATTR_TRANSPOSE_FIRST);
    OP_CHECK_NULL_WITH_CONTEXT(context, transposeFirst);

    const gert::Shape* xShape = context->GetInputShape(CONFUSION_TRANSPOSE_D_IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(CONFUSION_TRANSPOSE_D_OUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        return ge::GRAPH_SUCCESS;
    }

    const size_t dimNum = shape->GetSize();
    const int64_t* shapeData = shape->GetData();
    const int64_t* permData = perm->GetData();
    yShape->SetDimNum(dimNum);
    if (*transposeFirst) {
        for (size_t i = 0; i < dimNum; ++i) {
            yShape->SetDim(i, shapeData[i]);
        }
    } else {
        if (perm->GetSize() != dimNum) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                context->GetNodeName(), "perm, shape",
                (std::to_string(perm->GetSize()) + " and " + std::to_string(dimNum)).c_str(),
                "the size of attr perm must be equal to the size of attr shape.");
            return ge::GRAPH_FAILED;
        }
        for (size_t i = 0; i < dimNum; ++i) {
            if (permData[i] < 0 || static_cast<size_t>(permData[i]) >= dimNum) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "perm", std::to_string(permData[i]).c_str(),
                    "elements of attr perm must be in range [0, " + std::to_string(dimNum) + ").");
                return ge::GRAPH_FAILED;
            }
            yShape->SetDim(i, shapeData[permData[i]]);
        }
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ConfusionTransposeD).InferShape(InferShape4ConfusionTransposeD);
} // namespace ops
