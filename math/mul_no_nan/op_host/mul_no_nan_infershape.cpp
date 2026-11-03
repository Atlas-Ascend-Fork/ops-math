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
 * \file mul_no_nan_infershape.cpp
 * \brief
 */
#include "op_host/infershape_broadcast_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShape4MulNoNan(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4MulNoNan in ops-math");
    constexpr size_t MAX_INPUT_RANK = 8;
    for (size_t i = 0; i < 2; ++i) {
        const auto* shape = context->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, shape);
        OP_CHECK_IF(
            shape->GetDimNum() > MAX_INPUT_RANK,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), i == 0 ? "x1" : "x2",
                                                     std::to_string(shape->GetDimNum()), "rank must not exceed 8"),
            return ge::GRAPH_FAILED);
    }
    return Ops::Base::InferShape4Broadcast(context);
}

IMPL_OP_INFERSHAPE(MulNoNan).InferShape(InferShape4MulNoNan);

} // namespace ops
