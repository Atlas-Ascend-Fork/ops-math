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
 * \file phony_concat_infershape.cpp
 * \brief Runtime shape and data-type inference for PhonyConcat.
 */
#include <cstdint>
#include <vector>

#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"
#include "log/log.h"
#include "op_host/util/shape_util.h"

namespace ops {
// PhonyConcat only identifies that GE allocates one continuous memory block for
// all its inputs; it performs no calculation. The output shape is derived from
// the first dynamic input: each concat_dim dimension is multiplied by the
// corresponding N value, unknown (-1) dims are kept as-is, and the output dtype
// is copied from the input.
static constexpr size_t PHONY_CONCAT_IDX_IN_X = 0;
static constexpr size_t PHONY_CONCAT_IDX_OUT_Y = 0;
static constexpr size_t PHONY_CONCAT_IDX_ATTR_CONCAT_DIM = 0;
static constexpr size_t PHONY_CONCAT_IDX_ATTR_N = 1;

static ge::graphStatus InferShapeForPhonyConcat(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(PHONY_CONCAT_IDX_IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(PHONY_CONCAT_IDX_OUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    const auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto concatDim = attrs->GetAttrPointer<gert::TypedContinuousVector<int64_t>>(
        PHONY_CONCAT_IDX_ATTR_CONCAT_DIM);
    OP_CHECK_NULL_WITH_CONTEXT(context, concatDim);
    const auto numConcat = attrs->GetAttrPointer<gert::TypedContinuousVector<int64_t>>(PHONY_CONCAT_IDX_ATTR_N);
    OP_CHECK_NULL_WITH_CONTEXT(context, numConcat);

    OP_CHECK_IF(
        concatDim->GetSize() != numConcat->GetSize(),
        OP_LOGE(context, "The size of concat_dim and N must be the same, concat_dim size is %zu, N size is %zu.",
                concatDim->GetSize(), numConcat->GetSize()),
        return ge::GRAPH_FAILED);

    // the product of N must equal the number of dynamic input instances
    int64_t nTotalNum = 1;
    for (size_t i = 0; i < numConcat->GetSize(); ++i) {
        nTotalNum *= numConcat->GetData()[i];
    }
    OP_CHECK_IF(nTotalNum != static_cast<int64_t>(context->GetComputeNodeInputNum()),
                OP_LOGE(context,
                        "The product of N must be equal to the number of inputs, product of N is %ld, "
                        "input num is %zu.",
                        nTotalNum, context->GetComputeNodeInputNum()),
                return ge::GRAPH_FAILED);

    // input shape is unknown rank (-2), output keeps unknown rank
    if (Ops::Base::IsUnknownRank(*xShape)) {
        *yShape = *xShape;
        return ge::GRAPH_SUCCESS;
    }

    // check attrs and normalize negative concat_dim
    const int64_t dimNum = static_cast<int64_t>(xShape->GetDimNum());
    std::vector<int64_t> concatDims(concatDim->GetData(), concatDim->GetData() + concatDim->GetSize());
    for (size_t i = 0; i < concatDims.size(); ++i) {
        OP_CHECK_IF(
            (concatDims[i] < -dimNum) || (concatDims[i] >= dimNum),
            OP_LOGE(context, "concat_dim[%zu] is %ld, must be in range [-%ld, %ld).", i, concatDims[i], dimNum, dimNum),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(numConcat->GetData()[i] < 1,
                    OP_LOGE(context, "N[%zu] is %ld, must be greater than or equal to 1.", i, numConcat->GetData()[i]),
                    return ge::GRAPH_FAILED);
        if (concatDims[i] < 0) {
            concatDims[i] += dimNum;
        }
    }

    // output shape: dim on each concat_dim is multiplied by N, unknown (-1) dim is kept
    *yShape = *xShape;
    for (size_t i = 0; i < concatDims.size(); ++i) {
        const int64_t dimIdx = concatDims[i];
        if (yShape->GetDim(static_cast<size_t>(dimIdx)) == -1) {
            continue;
        }
        yShape->SetDim(static_cast<size_t>(dimIdx),
                       yShape->GetDim(static_cast<size_t>(dimIdx)) * numConcat->GetData()[i]);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForPhonyConcat(gert::InferDataTypeContext* context)
{
    // all dynamic inputs must have the same dtype as the first input (RT1.0 verifier semantics)
    const ge::DataType xDtype = context->GetInputDataType(PHONY_CONCAT_IDX_IN_X);
    const size_t inputNum = context->GetComputeNodeInputNum();
    for (size_t i = 1; i < inputNum; i++) {
        OP_CHECK_IF(context->GetInputDataType(i) != xDtype,
                    OP_LOGE(context, "The dtype of input[%zu] is %d, must be the same as input[0] which is %d.", i,
                            static_cast<int32_t>(context->GetInputDataType(i)), static_cast<int32_t>(xDtype)),
                    return ge::GRAPH_FAILED);
    }
    return context->SetOutputDataType(PHONY_CONCAT_IDX_OUT_Y, xDtype);
}

IMPL_OP_INFERSHAPE(PhonyConcat).InferShape(InferShapeForPhonyConcat).InferDataType(InferDataTypeForPhonyConcat);
} // namespace ops
