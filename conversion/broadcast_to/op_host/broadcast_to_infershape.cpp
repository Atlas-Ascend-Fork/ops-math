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
 * \file broadcast_to_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_util.h"
#include "util/shape_util.h"
using namespace ge;
namespace ops {
constexpr size_t INPUT_INDEX_SHAPE = 1;

template <typename T>
static void GetConstValueToShape(const gert::Tensor* tensor, size_t size, gert::Shape* shape)
{
    const T* value = tensor->GetData<T>();
    shape->SetDimNum(size);
    for (size_t i = 0; i < size; i++) {
        shape->SetDim(i, value[i]);
    }
}

static void WarnShapeValues(const gert::InferShapeContext* context, const gert::Shape* shape)
{
    for (size_t i = 0; i < shape->GetDimNum(); ++i) {
        if (shape->GetDim(i) < -1) {
            OP_LOGW(context->GetNodeName(), "target shape dim %ld at index %zu is less than -1; continue infershape.",
                    shape->GetDim(i), i);
        }
    }
}

static ge::graphStatus BroadcastToInferShapeWithShapeValues(const gert::InferShapeContext* context,
                                                            const gert::Shape* x_shape,
                                                            const gert::ContinuousVector* shape_attr,
                                                            gert::Shape* out_shape)
{
    OP_LOGD(context->GetNodeName(), "Begin to do BroadcastToInfershape.");
    const int64_t* shape_value = static_cast<const int64_t*>(shape_attr->GetData());
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_value);
    const size_t dim_num = shape_attr->GetSize();

    OP_LOGD(context->GetNodeName(), "input x = %s", Ops::Base::ToString(*x_shape).c_str());

    out_shape->SetDimNum(dim_num);
    for (size_t i = 0; i < dim_num; ++i) {
        out_shape->SetDim(i, shape_value[i]);
    }
    WarnShapeValues(context, out_shape);

    OP_LOGD(context->GetNodeName(), "output y = %s", Ops::Base::ToString(*out_shape).c_str());
    OP_LOGD(context->GetNodeName(), "End to do BroadcastToInfershape.");

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateShapeDtype(const gert::InferShapeContext* context, DataType data_type)
{
    OP_CHECK_IF(
        (data_type != DT_INT32) && (data_type != DT_INT64),
        OP_LOGE(context->GetNodeName(), "%s",
                ConcatString("shape's dtype ", Ops::Base::ToString(data_type), " must be in (int32,int64)!").c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static void DispatchGetConstValue(const gert::Tensor* shape_tensor, size_t shape_size, DataType data_type,
                                  gert::Shape* out_shape)
{
    if (data_type == DT_INT32) {
        GetConstValueToShape<int32_t>(shape_tensor, shape_size, out_shape);
    } else {
        GetConstValueToShape<int64_t>(shape_tensor, shape_size, out_shape);
    }
}

static ge::graphStatus ResolveBroadcastDim(const gert::InferShapeContext* context, const gert::Shape* x_shape,
                                           gert::Shape* out_shape, size_t i, size_t diff)
{
    if (out_shape->GetDim(i) == -1) {
        if (i >= diff) {
            out_shape->SetDim(i, x_shape->GetDim(i - diff));
        } else {
            out_shape->SetDim(i, 1);
        }
    }
    if (i < diff) {
        return ge::GRAPH_SUCCESS;
    }
    const int64_t x_dim = x_shape->GetDim(i - diff);
    OP_CHECK_IF((x_dim != -1) && (out_shape->GetDim(i) != x_dim) && (1 != x_dim),
                OP_LOGE(context->GetNodeName(), "%s",
                        ConcatString(Ops::Base::ToString(*x_shape).c_str(), " can not broadcast to ",
                                     Ops::Base::ToString(*out_shape).c_str())
                            .c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus BroadcastToInferShapeWithShapeTensor(const gert::InferShapeContext* context,
                                                            const gert::Shape* x_shape,
                                                            const gert::Tensor* shape_tensor, gert::Shape* out_shape)
{
    auto shape_size = static_cast<size_t>(shape_tensor->GetShapeSize());
    DataType data_type = shape_tensor->GetDataType();
    ge::graphStatus status = ValidateShapeDtype(context, data_type);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    if (Ops::Base::IsUnknownRank(*x_shape)) {
        DispatchGetConstValue(shape_tensor, shape_size, data_type, out_shape);
        WarnShapeValues(context, out_shape);
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(shape_size < x_shape->GetDimNum(),
                OP_LOGE(context->GetNodeName(), "%s",
                        ConcatString("shape size ", shape_size, " cannot be less than x size ", x_shape->GetDimNum(),
                                     ", error!")
                            .c_str()),
                return ge::GRAPH_FAILED);
    out_shape->SetDimNum(shape_size);
    OP_LOGD(context->GetNodeName(), "shape_size is %zu", shape_size);
    size_t diff = shape_size - x_shape->GetDimNum();
    DispatchGetConstValue(shape_tensor, shape_size, data_type, out_shape);
    WarnShapeValues(context, out_shape);
    for (size_t i = 0; i < shape_size; i++) {
        status = ResolveBroadcastDim(context, x_shape, out_shape, i, diff);
        if (status != ge::GRAPH_SUCCESS) {
            return status;
        }
    }
    return GRAPH_SUCCESS;
}

// 非 const shape：输出 rank 取 x rank 与 shape 长度的较大值
static ge::graphStatus BroadcastToInferShapeWithUnknownShape(gert::InferShapeContext* context,
                                                             const gert::Shape* x_shape, gert::Shape* out_shape)
{
    const auto* shape_desc = context->GetInputShape(INPUT_INDEX_SHAPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_desc);
    OP_CHECK_IF(shape_desc->GetDimNum() > 1,
                OP_LOGE(context->GetNodeName(), "the rank of shape should not be more than 1"),
                return ge::GRAPH_FAILED);
    if (Ops::Base::IsUnknownRank(*x_shape)) {
        Ops::Base::SetUnknownRank(*out_shape);
        return ge::GRAPH_SUCCESS;
    }
    int64_t rank = static_cast<int64_t>(x_shape->GetDimNum());
    if (shape_desc->GetDimNum() == 1 && shape_desc->GetDim(0) > rank) {
        rank = shape_desc->GetDim(0);
    }
    Ops::Base::SetUnknownShape(rank, *out_shape);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4BroadcastTo(gert::InferShapeContext* context)
{
    auto x_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

    auto shape_tensor = context->GetInputTensor(INPUT_INDEX_SHAPE);
    if (shape_tensor == nullptr || !IsConstTensor(shape_tensor)) {
        auto attrs = context->GetAttrs();
        const gert::ContinuousVector* shape_attr = (attrs != nullptr) ?
                                                       attrs->GetAttrPointer<gert::ContinuousVector>(0) :
                                                       nullptr;
        if (shape_attr != nullptr && shape_attr->GetSize() > 0) {
            return BroadcastToInferShapeWithShapeValues(context, x_shape, shape_attr, out_shape);
        }
        return BroadcastToInferShapeWithUnknownShape(context, x_shape, out_shape);
    }
    return BroadcastToInferShapeWithShapeTensor(context, x_shape, shape_tensor, out_shape);
}

IMPL_OP_INFERSHAPE(BroadcastTo)
    .InferShape(InferShape4BroadcastTo)
    .InputsDataDependency({INPUT_INDEX_SHAPE})
    .PrivateAttr("_shape_values", std::vector<int64_t>{});
} // namespace ops
