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
 * \file get_shape_graph_infer.cpp
 * \brief GetShape graph InferDataType implementation, output dtype is fixed to int32
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t kOutputIndexY = 0U;
} // namespace

static ge::graphStatus InferDataType4GetShape(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4GetShape");
    context->SetOutputDataType(kOutputIndexY, ge::DT_INT32);
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4GetShape");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(GetShape).InferDataType(InferDataType4GetShape);
} // namespace ops
