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
 * \file shape_infershape.cpp
 * \brief InferShape and InferDataType implementation for Shape operator.
 */
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "runtime/storage_shape.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;

namespace {
// REG_OP(Shape) 中 dtype 为第 0 个（且唯一）Int 属性
constexpr size_t kAttrDtypeIndex = 0U;
} // namespace

namespace ops {
static ge::graphStatus InferShapeForShape(gert::InferShapeContext* context)
{
    auto input_num = context->GetComputeNodeInputNum();
    for (size_t i = 0U; i < input_num; ++i) {
        auto x_shape = context->GetDynamicInputShape(0U, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
        auto y_shape = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
        // 与 V1 行为对齐：输入为未知秩(-2)时，输出为一维未知形状 {-1}
        if (Ops::Base::IsUnknownRank(*x_shape)) {
            Ops::Base::SetUnknownShape(1, *y_shape);
            continue;
        }
        y_shape->SetDimNum(1);
        y_shape->SetDim(0, static_cast<int64_t>(x_shape->GetDimNum()));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForShape(gert::InferDataTypeContext* context)
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

IMPL_OP_INFERSHAPE(Shape).InferShape(InferShapeForShape).InferDataType(InferDataTypeForShape);
} // namespace ops
