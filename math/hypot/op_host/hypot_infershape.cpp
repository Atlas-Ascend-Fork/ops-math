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
 * \file hypot_infershape.cpp
 * \brief hypot infershape: broadcast infershape
 */

#include "infershape_broadcast_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {

static ge::graphStatus InferShape4Hypot(gert::InferShapeContext* context)
{
    OP_LOGI(context, "Begin InferShape4Hypot");
    return Ops::Base::InferShape4Broadcast(context);
}

static ge::graphStatus InferDataType4Hypot(gert::InferDataTypeContext* context)
{
    const ge::DataType x1Type = context->GetInputDataType(0);
    const ge::DataType x2Type = context->GetInputDataType(1);
    ge::DataType outType = x1Type;
    if (x1Type == ge::DT_FLOAT || x2Type == ge::DT_FLOAT) {
        outType = ge::DT_FLOAT;
    } else if (x1Type != x2Type) {
        outType = ge::DT_FLOAT;
    }
    context->SetOutputDataType(0, outType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Hypot).InferShape(InferShape4Hypot).InferDataType(InferDataType4Hypot);

} // namespace ops
