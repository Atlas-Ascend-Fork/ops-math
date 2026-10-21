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
 * \file trans_shape_infershape.cpp
 * \brief Runtime shape and data-type inference for TransShape.
 */
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"
#include "log/log.h"

namespace ops {
// TransShape changes the output shape to the value of the "outShape" attribute
// without changing the underlying data. The output data type is the same as
// the input.
static constexpr size_t TRANS_SHAPE_IDX_IN_X = 0;
static constexpr size_t TRANS_SHAPE_IDX_OUT_Y = 0;
static constexpr size_t TRANS_SHAPE_IDX_ATTR_OUT_SHAPE = 0;

static ge::graphStatus InferShapeForTransShape(gert::InferShapeContext* context)
{
    auto* yShape = context->GetOutputShape(TRANS_SHAPE_IDX_OUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    const auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto outShape = attrs->GetAttrPointer<gert::TypedContinuousVector<int64_t>>(TRANS_SHAPE_IDX_ATTR_OUT_SHAPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    yShape->SetDimNum(outShape->GetSize());
    for (size_t i = 0; i < outShape->GetSize(); ++i) {
        yShape->SetDim(i, outShape->GetData()[i]);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForTransShape(gert::InferDataTypeContext* context)
{
    return context->SetOutputDataType(TRANS_SHAPE_IDX_OUT_Y, context->GetInputDataType(TRANS_SHAPE_IDX_IN_X));
}

IMPL_OP_INFERSHAPE(TransShape).InferShape(InferShapeForTransShape).InferDataType(InferDataTypeForTransShape);
} // namespace ops
