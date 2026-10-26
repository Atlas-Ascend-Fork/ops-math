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
 * \file gather_shapes_infershape.cpp
 * \brief InferShape and InferDataType implementation for GatherShapes operator.
 */
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "runtime/storage_shape.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;

namespace {
// REG_OP(GatherShapes) 中 axes 为第 0 个 ListListInt 属性，dtype 为第 1 个 Int 属性
constexpr size_t kAttrAxesIndex = 0U;
constexpr size_t kAttrDtypeIndex = 1U;
// V1 CheckAxes 约束：axes 中每个元素为 [输入序号, 维度序号] 二元组
constexpr size_t kAxesPairSize = 2U;
} // namespace

namespace ops {
// 与 V1 CheckAxes 逻辑对齐：axes 非空、每个元素为二元组、输入/维度序号不越界
static ge::graphStatus CheckAxesForGatherShapes(gert::InferShapeContext* context,
                                                const gert::ContinuousVectorVector* axes)
{
    if (axes->GetSize() == 0U) {
        OP_LOGE(context->GetNodeName(), "Attr [axes] is empty");
        return ge::GRAPH_FAILED;
    }
    const auto input_num = context->GetComputeNodeInputNum();
    for (size_t i = 0U; i < axes->GetSize(); ++i) {
        const auto* axis = axes->Get(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, axis);
        if (axis->GetSize() != kAxesPairSize) {
            OP_LOGE(context->GetNodeName(), "Attr [axes] value is invalid: axis size is %zu, where 2 is requested",
                    axis->GetSize());
            return ge::GRAPH_FAILED;
        }
        const auto* axis_data = reinterpret_cast<const int64_t*>(axis->GetData());
        const int64_t input_idx = axis_data[0];
        const int64_t dim_idx = axis_data[1];
        if ((input_idx < 0) || (static_cast<uint64_t>(input_idx) >= input_num)) {
            OP_LOGE(context->GetNodeName(),
                    "Attr [axes] value is invalid: axes[%ld, %ld] input index is out of range, input num is %zu",
                    input_idx, dim_idx, input_num);
            return ge::GRAPH_FAILED;
        }
        const auto* x_shape = context->GetInputShape(static_cast<size_t>(input_idx));
        OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
        if ((dim_idx < 0) || (static_cast<uint64_t>(dim_idx) >= x_shape->GetDimNum())) {
            OP_LOGE(context->GetNodeName(),
                    "Attr [axes] value is invalid: axes[%ld, %ld] dim index is out of range, dim num is %zu", input_idx,
                    dim_idx, x_shape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForGatherShapes(gert::InferShapeContext* context)
{
    auto y_shape = context->GetOutputShape(0U);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* axes = attrs->GetListListInt(kAttrAxesIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, axes);
    // 与 V1 行为对齐：存在未知秩(-2)输入时跳过 axes 越界检查（此时输入维度数不可知）
    bool has_unknown_rank = false;
    const auto input_num = context->GetComputeNodeInputNum();
    for (size_t i = 0U; i < input_num; ++i) {
        const auto* x_shape = context->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
        if (Ops::Base::IsUnknownRank(*x_shape)) {
            has_unknown_rank = true;
            break;
        }
    }
    if (!has_unknown_rank) {
        const auto ret = CheckAxesForGatherShapes(context, axes);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
    }
    y_shape->SetDimNum(1);
    y_shape->SetDim(0, static_cast<int64_t>(axes->GetSize()));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForGatherShapes(gert::InferDataTypeContext* context)
{
    // 与 V1 行为对齐：dtype 为可选属性，缺省/读取失败时沿用 proto 默认值 DT_INT32
    ge::DataType out_type = ge::DT_INT32;
    const auto* attrs = context->GetAttrs();
    const int64_t* dtype = nullptr;
    if (attrs != nullptr) {
        dtype = attrs->GetAttrPointer<int64_t>(kAttrDtypeIndex);
    }
    if (dtype != nullptr) {
        out_type = static_cast<ge::DataType>(*dtype);
    }
    if ((out_type != ge::DT_INT32) && (out_type != ge::DT_INT64)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "dtype",
                                              std::to_string(static_cast<int64_t>(out_type)).c_str(),
                                              "dtype must be DT_INT32 or DT_INT64");
        return ge::GRAPH_FAILED;
    }
    return context->SetOutputDataType(0, out_type);
}

IMPL_OP_INFERSHAPE(GatherShapes).InferShape(InferShapeForGatherShapes).InferDataType(InferDataTypeForGatherShapes);
} // namespace ops
