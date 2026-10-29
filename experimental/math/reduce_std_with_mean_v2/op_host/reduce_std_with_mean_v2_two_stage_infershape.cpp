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
 * \file reduce_std_with_mean_v2_two_stage_infershape.cpp
 * \brief InferShape registration for the two-stage kernels
 *
 * ReduceStdWithMeanV2Partial:
 *   self [*] -> partial [M, K, 3] (float32), M = non-reduce element count,
 *   K = partialK attr.
 * ReduceStdWithMeanV2Merge:
 *   partial [M, K, 3] -> out / meanOut, flat [M] (the L2 layer overrides the
 *   view shape to the exact user output shape before ViewCopy).
 */

#include "op_host/infershape_reduce_util.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

using namespace ge;

namespace ops {

const size_t TS_INPUT_INDEX_SELF = 0;
const size_t TS_ATTR_INDEX_DIM = 0;
const size_t TS_ATTR_INDEX_PARTIALK = 1;
const size_t TS_OUTPUT_INDEX_OUT = 0;

static ge::graphStatus InferShape4ReduceStdWithMeanV2Partial(gert::InferShapeContext* context)
{
    auto inputShape = context->GetInputShape(TS_INPUT_INDEX_SELF);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    auto partialShape = context->GetOutputShape(TS_OUTPUT_INDEX_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, partialShape);

    int64_t inputDimNum = inputShape->GetDimNum();
    int64_t totalNum = 1;
    for (int64_t d = 0; d < inputDimNum; d++) {
        totalNum *= inputShape->GetDim(d);
    }

    // reduce length: product of the dims in the dim attr (all dims if empty)
    int64_t reduceLen = 1;
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto dimPtr = attrs->GetAttrPointer<gert::ContinuousVector>(TS_ATTR_INDEX_DIM);
    OP_CHECK_NULL_WITH_CONTEXT(context, dimPtr);
    int64_t dimSize = dimPtr->GetSize();
    if (dimSize == 0) {
        for (int64_t d = 0; d < inputDimNum; d++) {
            reduceLen *= inputShape->GetDim(d);
        }
    } else {
        auto dimData = static_cast<const int64_t*>(dimPtr->GetData());
        for (int64_t i = 0; i < dimSize; i++) {
            int64_t axis = dimData[i];
            if (axis < 0) {
                axis += inputDimNum;
            }
            if (axis >= 0 && axis < inputDimNum) {
                reduceLen *= inputShape->GetDim(axis);
            }
        }
    }
    if (reduceLen <= 0) {
        reduceLen = 1;
    }

    int64_t nonReduce = (totalNum > 0) ? (totalNum / reduceLen) : 0;
    int64_t partialK = 1;
    const int64_t* kPtr = attrs->GetAttrPointer<int64_t>(TS_ATTR_INDEX_PARTIALK);
    if (kPtr != nullptr && *kPtr > 0) {
        partialK = *kPtr;
    }

    partialShape->SetDimNum(3);
    partialShape->SetDim(0, nonReduce);
    partialShape->SetDim(1, partialK);
    partialShape->SetDim(2, 3);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ReduceStdWithMeanV2Partial).InferShape(InferShape4ReduceStdWithMeanV2Partial);

static ge::graphStatus InferShape4ReduceStdWithMeanV2Merge(gert::InferShapeContext* context)
{
    auto partialShape = context->GetInputShape(TS_INPUT_INDEX_SELF);
    OP_CHECK_NULL_WITH_CONTEXT(context, partialShape);
    auto outShape = context->GetOutputShape(TS_OUTPUT_INDEX_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    auto meanOutShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, meanOutShape);

    int64_t M = (partialShape->GetDimNum() > 0) ? partialShape->GetDim(0) : 0;
    // flat [M]; the L2 layer fixes the view shape to the exact user shape
    outShape->SetDimNum(1);
    outShape->SetDim(0, M);
    *meanOutShape = *outShape;

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ReduceStdWithMeanV2Merge).InferShape(InferShape4ReduceStdWithMeanV2Merge);

} // namespace ops
