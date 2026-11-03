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
 * \file expm1_graph_infer.cpp
 * \brief Expm1 graph InferDataType implementation
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
static constexpr int64_t IDX_0 = 0;

ge::graphStatus InferDataTypeExpm1(gert::InferDataTypeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeExpm1");
    context->SetOutputDataType(IDX_0, context->GetInputDataType(IDX_0));
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeExpm1");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(Expm1).InferDataType(InferDataTypeExpm1);
} // namespace ops
