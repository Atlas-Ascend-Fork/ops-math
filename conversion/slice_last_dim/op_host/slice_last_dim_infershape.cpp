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
#include "log/log.h"
#include "util/shape_util.h"
#include "util/math_util.h"

using namespace ge;
namespace ops {
static constexpr size_t INPUT_IDX_X = 0;
static constexpr size_t OUTPUT_IDX_Y = 0;
static constexpr int64_t UNKNOWN_DIM = -1;
static constexpr int64_t MAX_RANK = 8;

static ge::graphStatus Infershape4SliceLastDim(gert::InferShapeContext* context)
{
    auto xShape = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_IF(xShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "x"), return ge::GRAPH_FAILED);
    auto yShape = context->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "start", "null", "required attribute is missing"),
        return ge::GRAPH_FAILED);
    auto startPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_IF(
        startPtr == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "start", "null", "required attribute is missing"),
        return ge::GRAPH_FAILED);
    auto endPtr = attrs->GetAttrPointer<int64_t>(1);
    OP_CHECK_IF(
        endPtr == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "end", "null", "required attribute is missing"),
        return ge::GRAPH_FAILED);

    int64_t stride = 1;
    auto stridePtr = attrs->GetAttrPointer<int64_t>(2);
    if (stridePtr != nullptr) {
        stride = *stridePtr;
    }
    OP_CHECK_IF(stride < 1,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "stride", std::to_string(stride).c_str(),
                                                      "The value of stride must be >= 1."),
                return ge::GRAPH_FAILED);

    // Unknown rank ([-2]) is passed through: the output keeps an unknown-rank descriptor and the
    // real shape is derived at runtime from the actual input. Dynamic shapes whose rank is known
    // (dims marked as -1) are still supported below.
    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        return ge::GRAPH_SUCCESS;
    }

    int64_t rank = static_cast<int64_t>(xShape->GetDimNum());
    OP_CHECK_IF(rank < 1 || rank > MAX_RANK,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x", std::to_string(rank).c_str(),
                                                         "The shape dim of x must be in [1, 8]."),
                return ge::GRAPH_FAILED);
    yShape->SetDimNum(0);

    for (int64_t i = 0; i < rank - 1; ++i) {
        yShape->AppendDim(xShape->GetDim(i));
    }

    int64_t lastDim = xShape->GetDim(rank - 1);
    if (lastDim == UNKNOWN_DIM) {
        yShape->AppendDim(UNKNOWN_DIM);
    } else {
        int64_t s = *startPtr;
        int64_t e = *endPtr;
        // Negative indices count from the end of the last dim (s += lastDim / e += lastDim).
        // Compare before adding so that the addition never overflows: the addition only runs
        // while the result stays within [0, lastDim].
        if (s < 0 && s >= -lastDim) {
            s += lastDim;
        }
        if (e < 0 && e >= -lastDim) {
            e += lastDim;
        }
        // Clamp both start and end to [0, lastDim] so that (e - s) stays within [-lastDim, lastDim]
        // and the length computation below is overflow-free.
        if (s < 0) {
            s = 0;
        }
        if (s > lastDim) {
            s = lastDim;
        }
        if (e < 0) {
            e = 0;
        }
        if (e > lastDim) {
            e = lastDim;
        }
        // CeilDiv (same as the tiling entry) divides first and adjusts the sign afterwards, which
        // keeps the computation free of signed overflow for extreme start/end/stride values.
        int64_t length = (e > s) ? Ops::Base::CeilDiv(e - s, stride) : 0;
        yShape->AppendDim(length);
    }

    OP_LOGI(context->GetNodeName(), "SliceLastDim output shape: %s.", Ops::Base::ToString(*yShape).c_str());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SliceLastDim).InferShape(Infershape4SliceLastDim);
} // namespace ops
