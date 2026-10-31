/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "dense_bincount_proto.h"
#include "log/log.h"

namespace ge {
namespace {
constexpr size_t INPUT_INDEX = 0;
constexpr size_t SIZE_INDEX = 1;
constexpr size_t WEIGHTS_INDEX = 2;
constexpr size_t MAX_INPUT_RANK = 2;

bool IsUnknownRank(const ge::Shape& shape) { return shape.GetDims() == ge::UNKNOWN_RANK; }

bool IsRankAtMost(const ge::Shape& shape, size_t maxRank)
{
    return IsUnknownRank(shape) || shape.GetDimNum() <= maxRank;
}
} // namespace

IMPLEMT_VERIFIER(DenseBincount, VerifyDenseBincount)
{
    const auto inputShape = op.GetInputDesc(INPUT_INDEX).GetShape();
    if (!IsRankAtMost(inputShape, MAX_INPUT_RANK)) {
        OP_LOGE("DenseBincount", "input rank must be at most 2.");
        return GRAPH_FAILED;
    }

    const auto sizeShape = op.GetInputDesc(SIZE_INDEX).GetShape();
    if (!IsUnknownRank(sizeShape) && sizeShape.GetDimNum() > 1U) {
        OP_LOGE("DenseBincount", "size must be a scalar or 1D tensor.");
        return GRAPH_FAILED;
    }

    const auto weightsShape = op.GetInputDesc(WEIGHTS_INDEX).GetShape();
    if (!IsRankAtMost(weightsShape, MAX_INPUT_RANK)) {
        OP_LOGE("DenseBincount", "weights rank must be at most 2.");
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

VERIFY_FUNC_REG(DenseBincount, VerifyDenseBincount);
} // namespace ge

namespace ops {
static ge::graphStatus InferDataTypeDenseBincount(gert::InferDataTypeContext* context)
{
    const ge::DataType inputDtype = context->GetInputDataType(0);
    const ge::DataType sizeDtype = context->GetInputDataType(1);
    const ge::DataType weightsDtype = context->GetInputDataType(2);
    OP_CHECK_IF(
        inputDtype != ge::DT_INT32 && inputDtype != ge::DT_INT64,
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "input", Ops::Base::ToString(inputDtype), "int32 or int64"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        sizeDtype != inputDtype,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "input, size",
                                               Ops::Base::ToString(inputDtype) + ", " + Ops::Base::ToString(sizeDtype),
                                               "input and size must have the same dtype"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        weightsDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "weights", Ops::Base::ToString(weightsDtype), "float32"),
        return ge::GRAPH_FAILED);
    return context->SetOutputDataType(0, ge::DT_FLOAT);
}
IMPL_OP(DenseBincount).InferDataType(InferDataTypeDenseBincount);
} // namespace ops
