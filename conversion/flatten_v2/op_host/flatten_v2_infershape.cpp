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
 * \file flatten_v2_infershape.cpp
 * \brief Runtime shape and data-type inference for FlattenV2.
 */
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"
#include "log/log.h"
#include "op_host/util/shape_util.h"

namespace ops {
// FlattenV2 flattens the dims in [axis, end_axis] of the input into a single
// dim: the dims before axis and after end_axis are retained in the output.
// If any dim in [axis, end_axis] is unknown (-1), the flattened dim is -1.
static constexpr size_t FLATTEN_V2_IDX_IN_X = 0;
static constexpr size_t FLATTEN_V2_IDX_OUT_Y = 0;
static constexpr size_t FLATTEN_V2_IDX_ATTR_AXIS = 0;
static constexpr size_t FLATTEN_V2_IDX_ATTR_END_AXIS = 1;

static ge::graphStatus InferShapeForFlattenV2(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(FLATTEN_V2_IDX_IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(FLATTEN_V2_IDX_OUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    const auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* attrAxis = attrs->GetAttrPointer<int64_t>(FLATTEN_V2_IDX_ATTR_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrAxis);
    const int64_t* attrEndAxis = attrs->GetAttrPointer<int64_t>(FLATTEN_V2_IDX_ATTR_END_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrEndAxis);

    // input shape is unknown rank (-2): rejected (RT1.0 verifier semantics)
    OP_CHECK_IF(Ops::Base::IsUnknownRank(*xShape), OP_LOGE(context, "Input shape is unknown rank, not supported."),
                return ge::GRAPH_FAILED);

    const int64_t dimNum = static_cast<int64_t>(xShape->GetDimNum());
    int64_t axis = *attrAxis;
    int64_t endAxis = *attrEndAxis;
    // convert negative axis/end_axis to positive
    if (axis < 0) {
        axis += dimNum;
    }
    if (endAxis < 0) {
        endAxis += dimNum;
    }

    OP_CHECK_IF((axis < 0) || (axis >= dimNum),
                OP_LOGE(context, "axis[%ld] is invalid, must be in range [0, %ld).", axis, dimNum),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((endAxis < 0) || (endAxis >= dimNum),
                OP_LOGE(context, "end_axis[%ld] is invalid, must be in range [0, %ld).", endAxis, dimNum),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(axis > endAxis, OP_LOGE(context, "axis[%ld] is greater than end_axis[%ld]!", axis, endAxis),
                return ge::GRAPH_FAILED);

    // dims before axis are retained
    yShape->SetDimNum(0);
    for (int64_t i = 0; i < axis; i++) {
        yShape->AppendDim(xShape->GetDim(static_cast<size_t>(i)));
    }

    // dims in [axis, end_axis] are flattened into one; any unknown (-1) dim makes it -1
    bool hasUnknownDim = false;
    for (int64_t i = axis; i <= endAxis; i++) {
        if (xShape->GetDim(static_cast<size_t>(i)) == -1) {
            hasUnknownDim = true;
            break;
        }
    }
    if (hasUnknownDim) {
        yShape->AppendDim(-1);
    } else {
        int64_t dimVal = 1;
        for (int64_t i = axis; i <= endAxis; i++) {
            dimVal *= xShape->GetDim(static_cast<size_t>(i));
        }
        yShape->AppendDim(dimVal);
    }

    // dims after end_axis are retained
    for (int64_t i = endAxis + 1; i < dimNum; i++) {
        yShape->AppendDim(xShape->GetDim(static_cast<size_t>(i)));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForFlattenV2(gert::InferDataTypeContext* context)
{
    return context->SetOutputDataType(FLATTEN_V2_IDX_OUT_Y, context->GetInputDataType(FLATTEN_V2_IDX_IN_X));
}

IMPL_OP_INFERSHAPE(FlattenV2).InferShape(InferShapeForFlattenV2).InferDataType(InferDataTypeForFlattenV2);
} // namespace ops
