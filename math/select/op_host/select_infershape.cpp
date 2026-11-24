/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file select_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "util/shape_util.h"
#include "register/op_impl_registry.h"
#include "version/metadef_version.h"

using namespace ge;
namespace ops {
namespace {
constexpr int64_t UNKNOWN_DIM_VALUE = -1;

/*!
 * \brief Fix the unknown dims (-1) of yShape with the corresponding dims of x2Shape.
 * \param nodeName Name of the node for logging.
 * \param yShape Shape of output y, fixed in place.
 * \param x2Shape Shape of input x2, read only.
 */
void FixUnknownDimFromX2(const char* nodeName, gert::Shape* yShape, const gert::Shape* x2Shape)
{
    for (size_t i = 0; i < yShape->GetDimNum(); ++i) {
        int64_t dimY = yShape->GetDim(i);
        int64_t dimX2 = x2Shape->GetDim(i);
        if (dimY == UNKNOWN_DIM_VALUE) {
            yShape->SetDim(i, dimX2);
        } else if (dimX2 != UNKNOWN_DIM_VALUE && dimY != dimX2) {
            OP_LOGW(nodeName, "x1 dim %ld differs from x2 dim %ld on axis %zu.", dimY, dimX2, i);
            return; // conflicting x1/x2 shapes are rejected by tiling; no further fixing is meaningful
        }
    }
}

} // namespace

static graphStatus InferShape4Select(gert::InferShapeContext* context)
{
    const gert::Shape* x1Shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    const gert::Shape* x2Shape = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);

    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    *yShape = *x1Shape;

    if (Ops::Base::IsUnknownRank(*x1Shape)) {
        if (!Ops::Base::IsUnknownRank(*x2Shape)) {
            *yShape = *x2Shape;
        }
        return GRAPH_SUCCESS;
    }
    if (Ops::Base::IsUnknownRank(*x2Shape)) {
        return GRAPH_SUCCESS;
    }
    if (x1Shape->GetDimNum() != x2Shape->GetDimNum()) {
        OP_LOGD(context->GetNodeName(), "Select x1 rank %zu differs from x2 rank %zu, keep x1 shape.",
                x1Shape->GetDimNum(), x2Shape->GetDimNum());
        return GRAPH_SUCCESS;
    }

    FixUnknownDimFromX2(context->GetNodeName(), yShape, x2Shape);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Select)
    .InferShape(InferShape4Select)
#if defined(METADEF_VERSION_NUM) && METADEF_VERSION_NUM >= 90200000
    .SetSupportPcieThrough()
#endif
    ;
} // namespace ops
