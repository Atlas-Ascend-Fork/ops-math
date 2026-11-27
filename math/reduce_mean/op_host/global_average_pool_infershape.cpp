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
 * \file global_average_pool_infershape.cpp
 * \brief GlobalAveragePool infershape, ported from canndev GlobalAveragePoolInferShape
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "log/log.h"
#include "graph/types.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t GLOBAL_AVG_POOL_RANK_MIN = 3;
constexpr size_t GLOBAL_AVG_POOL_RANK_MAX = 5;
constexpr size_t GLOBAL_AVG_POOL_NCHW_RANK = 4;
constexpr int64_t GLOBAL_AVG_POOL_OUT_DIM = 1;

ge::graphStatus InferShape4GlobalAveragePool(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    const auto* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    const size_t inputRank = inputShape->GetDimNum();
    const auto inputPrimaryFormat = static_cast<ge::Format>(
        ge::GetPrimaryFormat(static_cast<int32_t>(inputDesc->GetOriginFormat())));

    const bool isRankFormatSupported = (inputRank == GLOBAL_AVG_POOL_RANK_MIN && inputPrimaryFormat == FORMAT_ND) ||
                                       (inputRank == GLOBAL_AVG_POOL_NCHW_RANK &&
                                        (inputPrimaryFormat == FORMAT_NCHW || inputPrimaryFormat == FORMAT_ND)) ||
                                       (inputRank == GLOBAL_AVG_POOL_RANK_MAX &&
                                        (inputPrimaryFormat == FORMAT_NCDHW || inputPrimaryFormat == FORMAT_ND));
    if (!isRankFormatSupported) {
        OP_LOGE(context, "input rank[%zu] or format[%d] is not supported by GlobalAveragePool.", inputRank,
                static_cast<int32_t>(inputPrimaryFormat));
        return ge::GRAPH_FAILED;
    }

    outputShape->SetDimNum(inputRank);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, inputShape->GetDim(1));
    for (size_t i = 2; i < inputRank; ++i) {
        outputShape->SetDim(i, GLOBAL_AVG_POOL_OUT_DIM);
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_INFERSHAPE(GlobalAveragePool).InferShape(InferShape4GlobalAveragePool);
} // namespace ops
