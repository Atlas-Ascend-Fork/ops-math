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
 * \file strided_slice_grad_graph_infer.cpp
 * \brief Data type inference for StridedSliceGrad operator.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static constexpr size_t IN_DY_IDX = 4;

ge::graphStatus InferDataType4StridedSliceGrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferDataType4StridedSliceGrad start");
    auto dy_dtype = context->GetInputDataType(IN_DY_IDX);
    const auto output_num = context->GetComputeNodeOutputNum();
    for (size_t i = 0UL; i < output_num; ++i) {
        context->SetOutputDataType(i, dy_dtype);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(StridedSliceGrad).InferDataType(InferDataType4StridedSliceGrad);
} // namespace ops
