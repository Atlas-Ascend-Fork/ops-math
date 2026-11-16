/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file histogram_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {

namespace {
constexpr int64_t BINS_IDX = 0;
constexpr int64_t OUTPUT_IDX = 0;
constexpr int64_t DEFAULT_BINS = 100;
} // namespace

static ge::graphStatus HistogramInferShapeFunc(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do HistogramInferShapeFunc");
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "attrs");
        return ge::GRAPH_FAILED;
    }
    auto binsPtr = attrs->GetAttrPointer<int64_t>(BINS_IDX);
    int64_t bins = (binsPtr == nullptr) ? DEFAULT_BINS : *binsPtr;
    OP_LOGD(context, "bins = %ld", bins);
    OP_CHECK_IF(bins <= 0,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "bins", std::to_string(bins).c_str(), ">0"),
                return ge::GRAPH_FAILED);

    gert::Shape* outputShape = context->GetOutputShape(OUTPUT_IDX);
    if (outputShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "y");
        return ge::GRAPH_FAILED;
    }
    outputShape->SetDimNum(1);
    outputShape->SetDim(0, bins);
    OP_LOGD(context, "output shape = [%ld]", bins);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Histogram).InferShape(HistogramInferShapeFunc);

} // namespace ops
