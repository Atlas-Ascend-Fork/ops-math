/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cast_graph_infer.cpp
 * \brief Cast graph InferDataType implementation
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
static constexpr int64_t IDX_0 = 0;

ge::graphStatus InferDataTypeCast(gert::InferDataTypeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeCast");
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* dstType = attrs->GetAttrPointer<int64_t>(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstType);
    // int64 域校验: 直接 static_cast 后大值会截断到枚举底层宽度(int32), 可能落入合法区间绕过校验
    OP_CHECK_IF(*dstType < 0 || *dstType >= static_cast<int64_t>(ge::DT_MAX) ||
                    *dstType == static_cast<int64_t>(ge::DT_UNDEFINED),
                OP_LOGE(context->GetNodeName(), "the value of attr [dst_type] is invalid, got [%ld]", *dstType),
                return ge::GRAPH_FAILED);
    const auto outputDtype = static_cast<ge::DataType>(*dstType);
    context->SetOutputDataType(IDX_0, outputDtype);
    OP_LOGD(context->GetNodeName(), "CastInferDataType end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(Cast).InferDataType(InferDataTypeCast);
} // namespace ops
