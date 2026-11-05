/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "op_host/util/const_util.h"
#include "op_host/util/shape_util.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
static constexpr size_t INPUT_IDX_X = 0;
static constexpr size_t INPUT_IDX_PADDINGS = 1;
static constexpr size_t OUTPUT_IDX_Y = 0;
static constexpr size_t PADDING_PAIR_SIZE = 2;
static constexpr int64_t UNKNOWN_DIM = -1;

static ge::graphStatus InferShape4CircularPadGrad(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_IF(xShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "x"), return ge::GRAPH_FAILED);
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_IDX_Y);
    if (yShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "y");
        return ge::GRAPH_FAILED;
    }

    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        OP_LOGI(context->GetNodeName(), "CircularPadGrad output shape: %s.", Ops::Base::ToString(*yShape).c_str());
        return ge::GRAPH_SUCCESS;
    }

    gert::Shape paddings;
    if (!Ops::Base::GetConstIntToShape<gert::InferShapeContext>(context, INPUT_IDX_PADDINGS, paddings)) {
        yShape->SetDimNum(xShape->GetDimNum());
        for (size_t i = 0; i < xShape->GetDimNum(); ++i) {
            yShape->SetDim(i, UNKNOWN_DIM);
        }
        OP_LOGI(context->GetNodeName(), "CircularPadGrad output shape: %s.", Ops::Base::ToString(*yShape).c_str());
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(paddings.GetDimNum() != xShape->GetDimNum() * PADDING_PAIR_SIZE,
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    context->GetNodeName(), "paddings and x",
                    (std::to_string(paddings.GetDimNum()) + ", " + std::to_string(xShape->GetDimNum())).c_str(),
                    "the element count of paddings must be twice the rank of x"),
                return ge::GRAPH_FAILED);
    *yShape = *xShape;
    for (size_t i = 0; i < xShape->GetDimNum(); ++i) {
        const int64_t xDim = xShape->GetDim(i);
        if (xDim != UNKNOWN_DIM) {
            const int64_t padFront = paddings.GetDim(PADDING_PAIR_SIZE * i);
            const int64_t padBack = paddings.GetDim(PADDING_PAIR_SIZE * i + 1);
            const int64_t yDim = xDim - padFront - padBack;
            OP_CHECK_IF(
                yDim < 0,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context->GetNodeName(), "x, paddings", std::to_string(yDim).c_str(),
                    (std::string("x[") + std::to_string(i) + "] - paddings[" + std::to_string(PADDING_PAIR_SIZE * i) +
                     "] - paddings[" + std::to_string(PADDING_PAIR_SIZE * i + 1) + "] = " + std::to_string(xDim) +
                     " - (" + std::to_string(padFront) + ") - (" + std::to_string(padBack) +
                     ") = " + std::to_string(yDim) + ", but output shape cannot contain negative values")
                        .c_str()),
                return ge::GRAPH_FAILED);
            yShape->SetDim(i, yDim);
        }
    }
    OP_LOGI(context->GetNodeName(), "CircularPadGrad output shape: %s.", Ops::Base::ToString(*yShape).c_str());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4CircularPadGrad(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(OUTPUT_IDX_Y, context->GetInputDataType(INPUT_IDX_X));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CircularPadGrad)
    .InferShape(InferShape4CircularPadGrad)
    .InferDataType(InferDataType4CircularPadGrad)
    .InputsDataDependency({INPUT_IDX_PADDINGS});
} // namespace ops
