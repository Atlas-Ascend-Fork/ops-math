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
 * \file as_strided_graph_infer.cpp
 * \brief Data type inference for AsStrided operator.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
ge::graphStatus InferDataType4AsStrided(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferDataType4AsStrided start");
    auto input_x_dtype = context->GetInputDataType(0);
    const auto output_num = context->GetComputeNodeOutputNum();
    for (size_t i = 0UL; i < output_num; ++i) {
        context->SetOutputDataType(i, input_x_dtype);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(AsStrided).InferDataType(InferDataType4AsStrided);
} // namespace ops
