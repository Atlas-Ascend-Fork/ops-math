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
 * \file approximate_equal_infershape.cpp
 * \brief ApproximateEqual shape inference (y shape = x1 shape, x1 and x2 must have the same shape;
 *        output merges unknown info: takes x2 known shape when x1 rank is unknown, x1 as base with
 *        dim-level known absorption from x2 otherwise)
 */

#include "util/shape_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {

// GE shape标记语义: -1=未知维, -2=未知rank; 同形校验中标记按通配处理。
static bool ShapeEqualOrUnknown(const gert::Shape& x1Shape, const gert::Shape& x2Shape)
{
    if (Ops::Base::IsUnknownRank(x1Shape) || Ops::Base::IsUnknownRank(x2Shape)) {
        return true;
    }
    if (x1Shape.GetDimNum() != x2Shape.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < x1Shape.GetDimNum(); ++i) {
        const int64_t dimX1 = x1Shape.GetDim(i);
        const int64_t dimX2 = x2Shape.GetDim(i);
        if (dimX1 != dimX2 && dimX1 != -1 && dimX2 != -1) {
            return false;
        }
    }
    return true;
}

// 用known侧同位值填充out中标记为未知(-1)的维。
static void FillUnknownDims(gert::Shape& outShape, const gert::Shape& knownSide)
{
    for (size_t i = 0; i < outShape.GetDimNum(); ++i) {
        if (outShape.GetDim(i) == -1 && knownSide.GetDim(i) != -1) {
            outShape.SetDim(i, knownSide.GetDim(i));
        }
    }
}

static ge::graphStatus InferShape4ApproximateEqual(gert::InferShapeContext* context)
{
    const gert::Shape* x1Shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    const gert::Shape* x2Shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    gert::Shape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    OP_CHECK_IF(
        !ShapeEqualOrUnknown(*x1Shape, *x2Shape),
        OP_LOGE(context->GetNodeName(), "ApproximateEqual does not support broadcast: x1 shape=%s != x2 shape=%s",
                Ops::Base::ToString(*x1Shape).c_str(), Ops::Base::ToString(*x2Shape).c_str()),
        return ge::GRAPH_FAILED);
    const bool x1UnknownRank = Ops::Base::IsUnknownRank(*x1Shape);
    const bool x2UnknownRank = Ops::Base::IsUnknownRank(*x2Shape);
    if (x1UnknownRank && !x2UnknownRank) {
        *outputShape = *x2Shape;
    } else {
        *outputShape = *x1Shape;
        if (!x2UnknownRank) {
            FillUnknownDims(*outputShape, *x2Shape);
        }
    }
    OP_LOGI(context->GetNodeName(), "[InferShape] output0 shape=%s", Ops::Base::ToString(*outputShape).c_str());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApproximateEqual).InferShape(InferShape4ApproximateEqual);

} // namespace ops
