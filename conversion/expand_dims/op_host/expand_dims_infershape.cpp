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
 * \file expand_dims_infershape.cpp
 * \brief Runtime shape and data-type inference for ExpandDims.
 */
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"
#include "log/log.h"
#include "op_host/util/shape_util.h"

namespace ops {
// ExpandDims inserts a dim of 1 into the input shape at the position given by
// the "axis" input tensor (int32/int64 scalar, data-dependent). The output
// data type is the same as the input.
static constexpr size_t EXPAND_DIMS_IDX_IN_X = 0;
static constexpr size_t EXPAND_DIMS_IDX_IN_AXIS = 1;
static constexpr size_t EXPAND_DIMS_IDX_OUT_Y = 0;
static constexpr int64_t UNKNOWN_DIM = -1;

template <typename T>
static ge::graphStatus ExpandDimsInferShapeImpl(gert::InferShapeContext* context, const T* expandDim,
                                                const gert::Shape& xShape, gert::Shape& yShape)
{
    T axisData = *expandDim;
    axisData = (axisData < 0) ? (axisData + static_cast<T>(xShape.GetDimNum() + 1)) : axisData;
    if ((axisData < 0) || (axisData > static_cast<T>(xShape.GetDimNum()))) {
        OP_LOGE(context, "ExpandDims failed, axis[%d] is not in [%d, %d].", static_cast<int32_t>(axisData),
                static_cast<int32_t>(-1 - static_cast<T>(xShape.GetDimNum())),
                static_cast<int32_t>(xShape.GetDimNum()));
        return ge::GRAPH_FAILED;
    }
    yShape.SetDimNum(xShape.GetDimNum() + 1U);
    size_t xPos = 0U;
    const size_t unsignedAxisData = static_cast<size_t>(axisData);
    for (size_t idx = 0U; idx < yShape.GetDimNum(); ++idx) {
        if (idx == unsignedAxisData) {
            yShape.SetDim(idx, 1);
        } else {
            yShape.SetDim(idx, xShape.GetDim(xPos++));
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForExpandDims(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(EXPAND_DIMS_IDX_IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(EXPAND_DIMS_IDX_OUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    const gert::Tensor* axisTensor = context->GetInputTensor(EXPAND_DIMS_IDX_IN_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, axisTensor);

    const ge::DataType axisDtype = axisTensor->GetDataType();
    OP_CHECK_IF((axisDtype != ge::DT_INT32) && (axisDtype != ge::DT_INT64),
                OP_LOGE(context, "The dtype of axis must be DT_INT32 or DT_INT64, actually is %d.",
                        static_cast<int32_t>(axisDtype)),
                return ge::GRAPH_FAILED);

    // unknown rank input keeps unknown rank output, regardless of axis data availability (RT1.0 v1 semantics:
    // inserting a dim into an unknown-rank shape is meaningless, so the output rank stays unknown)
    if (Ops::Base::IsUnknownRank(*xShape)) {
        *yShape = *xShape;
        return ge::GRAPH_SUCCESS;
    }

    // axis input must be a scalar or a tensor with a single value (checked before the compile-phase
    // branch, same as RT1.0 v1: the axis shape is known even when its value is not)
    OP_CHECK_IF(axisTensor->GetShapeSize() != 1,
                OP_LOGE(context,
                        "ExpandDims failed, axis input must be a tensor with a single value, "
                        "actually shape size is %ld.",
                        axisTensor->GetShapeSize()),
                return ge::GRAPH_FAILED);

    // axis tensor data addr is nullptr, means the axis value is unknown (compile phase)
    if (axisTensor->GetTensorData().GetAddr() == nullptr) {
        yShape->SetDimNum(xShape->GetDimNum() + 1U);
        for (size_t idx = 0U; idx < yShape->GetDimNum(); ++idx) {
            yShape->SetDim(idx, UNKNOWN_DIM);
        }
        return ge::GRAPH_SUCCESS;
    }

    if (axisDtype == ge::DT_INT32) {
        const auto axisData = axisTensor->GetData<int32_t>();
        return ExpandDimsInferShapeImpl<int32_t>(context, axisData, *xShape, *yShape);
    }
    const auto axisData = axisTensor->GetData<int64_t>();
    return ExpandDimsInferShapeImpl<int64_t>(context, axisData, *xShape, *yShape);
}

static ge::graphStatus InferDataTypeForExpandDims(gert::InferDataTypeContext* context)
{
    return context->SetOutputDataType(EXPAND_DIMS_IDX_OUT_Y, context->GetInputDataType(EXPAND_DIMS_IDX_IN_X));
}

IMPL_OP_INFERSHAPE(ExpandDims)
    .InferShape(InferShapeForExpandDims)
    .InferDataType(InferDataTypeForExpandDims)
    .InputsDataDependency({EXPAND_DIMS_IDX_IN_AXIS});
} // namespace ops
