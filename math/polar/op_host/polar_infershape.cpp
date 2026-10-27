/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file polar_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "infershape_broadcast_util.h"
#include "log/log.h"

using namespace Ops::Base;
using namespace ge;
namespace ops {
static ge::graphStatus InferShape4Polar(gert::InferShapeContext* context)
{
    auto in_shape1 = context->GetInputShape(0);
    if (in_shape1 == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "abs");
        return ge::GRAPH_FAILED;
    }
    auto in_shape2 = context->GetInputShape(1);
    if (in_shape2 == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "angle");
        return ge::GRAPH_FAILED;
    }
    auto out_shape = context->GetOutputShape(0);
    if (out_shape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "y");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(!BroadcastShape(in_shape1, in_shape2, out_shape),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "abs/angle",
                    ("shape " + ToString(*in_shape1) + " and " + ToString(*in_shape2) + " cannot broadcast").c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(Polar).InferShape(InferShape4Polar);
} // namespace ops
