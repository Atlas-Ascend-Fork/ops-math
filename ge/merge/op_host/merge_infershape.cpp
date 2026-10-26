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
 * \file merge_infershape.cpp
 * \brief InferShape and InferDataType implementation for Merge operator.
 */
#include <vector>
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "log/log.h"

using namespace ge;

namespace ops {
namespace {
constexpr int64_t kUnknownRankDim = -2;

bool IsUnknownRankShape(const gert::Shape& shape)
{
    return shape.GetDimNum() == 1U && shape.GetDim(0) == kUnknownRankDim;
}

void SetUnknownRankShape(gert::Shape* shape)
{
    shape->SetDimNum(1);
    shape->SetDim(0, kUnknownRankDim);
}
} // namespace

static ge::graphStatus InferShape4Merge(gert::InferShapeContext* context)
{
    const auto* in_info = context->GetIrInputInstanceInfo(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_info);
    const size_t in_num = in_info->GetInstanceNum();
    if (in_num < 1U) {
        OP_LOGE(context->GetNodeName(), "input num of Merge should be >= 1, but got %zu", in_num);
        return ge::GRAPH_FAILED;
    }

    auto* value_index_shape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, value_index_shape);
    value_index_shape->SetDimNum(0);

    const auto* x0 = context->GetDynamicInputShape(0, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x0);
    auto* y_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);

    if (IsUnknownRankShape(*x0)) {
        SetUnknownRankShape(y_shape);
        return ge::GRAPH_SUCCESS;
    }
    const size_t rank = x0->GetDimNum();

    std::vector<int64_t> out_dims(rank, -1);
    std::vector<bool> dim_seen(rank, false);
    std::vector<bool> dim_unknown(rank, false);
    for (size_t i = 0; i < in_num; ++i) {
        const auto* xi = context->GetDynamicInputShape(0, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, xi);
        if (IsUnknownRankShape(*xi) || xi->GetDimNum() != rank) {
            SetUnknownRankShape(y_shape);
            return ge::GRAPH_SUCCESS;
        }
        for (size_t d = 0; d < rank; ++d) {
            const int64_t dim = xi->GetDim(d);
            if (dim < 0) {
                dim_unknown[d] = true;
                continue;
            }
            if (!dim_seen[d]) {
                out_dims[d] = dim;
                dim_seen[d] = true;
            } else if (out_dims[d] != dim) {
                dim_unknown[d] = true;
            }
        }
    }

    y_shape->SetDimNum(rank);
    for (size_t d = 0; d < rank; ++d) {
        y_shape->SetDim(d, dim_unknown[d] ? -1 : out_dims[d]);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4Merge(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto* in_info = context->GetIrInputInstanceInfo(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_info);
    const size_t input_num = in_info->GetInstanceNum();
    if (input_num == 0U) {
        OP_LOGE(context->GetNodeName(), "Merge requires at least one input.");
        return ge::GRAPH_FAILED;
    }
    const auto y_dtype = context->GetDynamicInputDataType(0, 0);
    for (size_t i = 1; i < input_num; ++i) {
        const auto dtype = context->GetDynamicInputDataType(0, i);
        if (dtype != y_dtype) {
            OP_LOGE(context->GetNodeName(), "Input x[%zu] has data type %d, expected %d to match x[0].", i,
                    static_cast<int>(dtype), static_cast<int>(y_dtype));
            return ge::GRAPH_FAILED;
        }
    }
    const auto status = context->SetOutputDataType(0, y_dtype);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    return context->SetOutputDataType(1, ge::DT_INT32);
}

IMPL_OP_INFERSHAPE(Merge).InferShape(InferShape4Merge).InferDataType(InferDataType4Merge);

} // namespace ops
