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
 * \file flatten_infershape.cpp
 * \brief Runtime shape and data-type inference for Flatten.
 */
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "log/log.h"

#include <vector>
#include "op_host/util/shape_util.h"

namespace ops {
static constexpr size_t FLATTEN_IDX_IN_X = 0;
static constexpr size_t FLATTEN_IDX_OUT_Y = 0;
static constexpr size_t FLATTEN_IDX_ATTR_AXIS = 0;

// Flatten splits the input dims at axis into two groups and multiplies each
// group into one dim: y = {prod(dims[0, axis)), prod(dims[axis, rank))}.
// axis is allowed to equal rank, which makes the second group empty (1).
static int64_t GetPartSize(const gert::Shape& shape, int64_t begin, int64_t end)
{
    int64_t size = 1;
    for (int64_t i = begin; i < end; i++) {
        const int64_t dim = shape.GetDim(static_cast<size_t>(i));
        // any unknown dim(-1) in the group makes the flattened dim unknown
        if (dim == ge::UNKNOWN_DIM) {
            return ge::UNKNOWN_DIM;
        }
        size *= dim;
    }
    return size;
}

static ge::graphStatus InferShapeForFlatten(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(FLATTEN_IDX_IN_X);
    OP_CHECK_IF(
        xShape == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{context->GetNodeName(), "Input shape x is unavailable; check InferShape "
                                                              "context construction and operator registration."})),
        return ge::GRAPH_FAILED);
    gert::Shape* yShape = context->GetOutputShape(FLATTEN_IDX_OUT_Y);
    OP_CHECK_IF(
        yShape == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{context->GetNodeName(), "Output shape y is unavailable; check InferShape "
                                                              "context construction and operator registration."})),
        return ge::GRAPH_FAILED);
    const auto attrs = context->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{context->GetNodeName(), "Runtime attrs is unavailable; check InferShape "
                                                              "context construction and operator registration."})),
        return ge::GRAPH_FAILED);
    const int64_t* attrAxis = attrs->GetAttrPointer<int64_t>(FLATTEN_IDX_ATTR_AXIS);
    OP_CHECK_IF(attrAxis == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "axis"),
                return ge::GRAPH_FAILED);

    // unknown rank(-2) input keeps unknown rank output
    if (Ops::Base::IsUnknownRank(*xShape)) {
        *yShape = *xShape;
        return ge::GRAPH_SUCCESS;
    }

    const int64_t dimNum = static_cast<int64_t>(xShape->GetDimNum());
    int64_t axis = *attrAxis;
    // RT1.0 semantics: axis is allowed in [-rank, rank], both ends inclusive
    if ((axis < -dimNum) || (axis > dimNum)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "axis", std::to_string(axis).c_str(),
            "axis must be in range [-" + std::to_string(dimNum) + ", " + std::to_string(dimNum) + "]");
        return ge::GRAPH_FAILED;
    }
    if (axis < 0) {
        axis += dimNum;
    }

    // RT1.0 semantics: a rank-1 input with an unknown dim keeps rank 1 instead
    // of producing a 2-D output
    if ((dimNum == 1) && (xShape->GetDim(0U) == ge::UNKNOWN_DIM)) {
        yShape->SetDimNum(1U);
        yShape->SetDim(0U, ge::UNKNOWN_DIM);
        return ge::GRAPH_SUCCESS;
    }

    yShape->SetDimNum(0);
    yShape->AppendDim(GetPartSize(*xShape, 0, axis));
    yShape->AppendDim(GetPartSize(*xShape, axis, dimNum));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForFlatten(gert::InferDataTypeContext* context)
{
    return context->SetOutputDataType(FLATTEN_IDX_OUT_Y, context->GetInputDataType(FLATTEN_IDX_IN_X));
}

IMPL_OP_INFERSHAPE(Flatten).InferShape(InferShapeForFlatten).InferDataType(InferDataTypeForFlatten);
} // namespace ops
