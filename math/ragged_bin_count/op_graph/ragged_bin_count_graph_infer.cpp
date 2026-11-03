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
 * \file ragged_bin_count_graph_infer.cpp
 * \brief Data type inference for the RaggedBinCount operator.
 */

#include <string>

#include "ragged_bin_count_graph_infer_internal.h"
#include "graph/operator_factory.h"

namespace ops {
namespace {
ge::graphStatus VerifyRaggedBinCountRowSplits(ge::Operator& op)
{
    const auto dims = op.GetInputDesc("splits").GetShape().GetDims();
    // GE can eliminate a zero-row output before the runtime shape/tiling checks.
    // Reject a known short splits tensor before that empty-output optimization.
    OP_CHECK_IF(
        dims.size() == 1U && dims[0] >= 0 && dims[0] < 2,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(op.GetName().c_str(), "splits", "[" + std::to_string(dims[0]) + "]",
                                              "splits must contain at least two elements"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::VerifyFuncRegister registerRaggedBinCountVerifier("RaggedBinCount", VerifyRaggedBinCountRowSplits);
} // namespace

static ge::graphStatus InferDataTypeForRaggedBinCount(gert::InferDataTypeContext* context)
{
    return ragged_bin_count_graph_infer_internal::InferDataType(context);
}

IMPL_OP(RaggedBinCount).InferDataType(InferDataTypeForRaggedBinCount);
} // namespace ops
