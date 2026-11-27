/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <string>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_datatype_context.h"

using namespace ge;

namespace {
// REG_OP(SearchSorted) 中 dtype 为第 0 个（Type）属性：输出 dtype 由该属性决定（RT1 语义）
constexpr size_t kAttrDtypeIndex = 0U;
} // namespace

namespace ops {
static graphStatus InferShape4SearchSorted(gert::InferShapeContext* context)
{
    OP_LOGI("Begin InferShape4SearchSorted");
    const gert::Shape* values_shape = context->GetInputShape(1);
    if (values_shape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "values");
        return ge::GRAPH_FAILED;
    }

    gert::Shape* out_shape = context->GetOutputShape(0);
    if (out_shape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "y");
        return ge::GRAPH_FAILED;
    }
    *out_shape = *values_shape;

    return GRAPH_SUCCESS;
}

// 与 V1 行为对齐：输出 dtype 取自 dtype 属性（Type），缺省/读取失败沿用 proto 默认值 DT_INT64
static graphStatus InferDataType4SearchSorted(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType out_type = ge::DT_INT64;
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

IMPL_OP_INFERSHAPE(SearchSorted).InferShape(InferShape4SearchSorted).InferDataType(InferDataType4SearchSorted);
} // namespace ops
