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
 * \file sim_thread_exponential_infershape.cpp
 * \brief Shape and dtype inference for sim_thread_exponential operator
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace {
constexpr uint32_t INPUT_SELF_IDX = 0;
constexpr uint32_t OUTPUT_SELF_IDX = 0;

// 空指针检查仅写日志不注册错误码，GetNodeName为空时回退到字面算子名（参照官方整改基线）
const char* OpNameOf(gert::InferShapeContext* inferContext)
{
    const char* name = inferContext->GetNodeName();
    return name == nullptr ? "SimThreadExponential" : name;
}
} // namespace

namespace ops {
static ge::graphStatus InferShape4SimThreadExponential(gert::InferShapeContext* context)
{
    const gert::Shape* inputSelfShape = context->GetInputShape(INPUT_SELF_IDX);
    if (inputSelfShape == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OpNameOf(context), "inputSelfShape is nullptr");
        return ge::GRAPH_FAILED;
    }
    gert::Shape* outputSelfShape = context->GetOutputShape(OUTPUT_SELF_IDX);
    if (outputSelfShape == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OpNameOf(context), "outputSelfShape is nullptr");
        return ge::GRAPH_FAILED;
    }
    *outputSelfShape = *inputSelfShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4SimThreadExponential(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(INPUT_SELF_IDX);
    context->SetOutputDataType(OUTPUT_SELF_IDX, inputDataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SimThreadExponential)
    .InferShape(InferShape4SimThreadExponential)
    .InferDataType(InferDataType4SimThreadExponential);
} // namespace ops
