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
 * \file switch_n_infershape.cpp
 * \brief InferShape and InferDataType implementation for SwitchN operator.
 */
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4SwitchN(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto* data_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, data_shape);

    // The predicate selects the live output; forwarding does not change its shape.
    const size_t output_num = context->GetComputeNodeOutputNum();
    if (output_num == 0U) {
        OP_LOGE(context->GetNodeName(), "SwitchN requires at least one output.");
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < output_num; ++i) {
        auto* output_shape = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);
        *output_shape = *data_shape;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4SwitchN(gert::InferDataTypeContext* context)
{
    const auto data_dtype = context->GetInputDataType(0);
    const size_t out_num = context->GetComputeNodeOutputNum();
    for (size_t i = 0; i < out_num; ++i) {
        (void)context->SetOutputDataType(i, data_dtype);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SwitchN).InferShape(InferShape4SwitchN).InferDataType(InferDataType4SwitchN);

} // namespace ops
