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
 * \file kl_div_v2_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShape4KLDiv(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do KLDivInferShape");
    auto out_shape = context->GetOutputShape(0);
    if (out_shape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "out_shape");
        return GRAPH_FAILED;
    }
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "attrs");
        return GRAPH_FAILED;
    }

    const char* reduction = attrs->GetAttrPointer<char>(0);
    if (reduction == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "reduction");
        return GRAPH_FAILED;
    }
    if (strcmp(reduction, "none") == 0) {
        auto in_shape = context->GetInputShape(0);
        if (in_shape == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "in_shape");
            return GRAPH_FAILED;
        }
        *out_shape = *in_shape;
    } else {
        // if reduction == "batchmean" or reduction == "sum" , output a scalar
        out_shape->SetDimNum(0);
    }

    OP_LOGD(context->GetNodeName(), "End to do KLDivInferShape");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(KLDivV2).InferShape(InferShape4KLDiv);
} // namespace ops
