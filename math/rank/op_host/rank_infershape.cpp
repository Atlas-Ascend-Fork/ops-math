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
 * \file rank_infershape.cpp
 * \brief Runtime shape and data-type inference for Rank.
 */
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"
#include "runtime/infer_datatype_context.h"

namespace ops {
// Rank returns the number of dimensions of the input tensor as a scalar.
// Equivalent to the RT1.0 logic in rank_infer_v1.cpp: the output shape is a
// scalar (empty dim vector) and the output dtype is the fixed DT_INT32
// declared in the proto.
static ge::graphStatus InferShapeForRank(gert::InferShapeContext* context)
{
    auto y_shape = context->GetOutputShape(0);
    if (y_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    y_shape->SetScalar();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForRank(gert::InferDataTypeContext* context)
{
    return context->SetOutputDataType(0, ge::DT_INT32);
}

IMPL_OP_INFERSHAPE(Rank).InferShape(InferShapeForRank).InferDataType(InferDataTypeForRank);
} // namespace ops
