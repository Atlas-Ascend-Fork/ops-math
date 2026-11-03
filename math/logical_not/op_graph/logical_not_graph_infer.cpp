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
 * \file logical_not_graph_infer.cpp
 * \brief LogicalNot graph InferDataType implementation
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
static constexpr int64_t IDX_0 = 0;

ge::graphStatus InferDataTypeLogicalNot(gert::InferDataTypeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeLogicalNot");
    // 输出恒为 BOOL(def 输出声明仅 DT_BOOL), 不随输入 dtype
    context->SetOutputDataType(IDX_0, ge::DT_BOOL);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeLogicalNot");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(LogicalNot).InferDataType(InferDataTypeLogicalNot);
} // namespace ops
