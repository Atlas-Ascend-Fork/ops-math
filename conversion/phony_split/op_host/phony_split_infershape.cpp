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
 * \file phony_split_infershape.cpp
 * \brief Runtime shape and data-type inference for PhonySplit.
 */
#include <cstdint>
#include <vector>

#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"
#include "log/log.h"
#include "op_host/util/shape_util.h"

namespace ops {
// PhonySplit only identifies that GE allocates one continuous memory block for
// all its outputs; it performs no calculation. The output shapes are derived
// from the input: each split_dim dimension is divided by the corresponding
// num_split value (must be divisible, unknown -1 dims are kept as-is), and all
// outputs share the same shape with the input dtype.
static constexpr size_t PHONY_SPLIT_IDX_IN_X = 0;
static constexpr size_t PHONY_SPLIT_IDX_ATTR_SPLIT_DIM = 0;
static constexpr size_t PHONY_SPLIT_IDX_ATTR_NUM_SPLIT = 1;

static ge::graphStatus InferShapeForPhonySplit(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(PHONY_SPLIT_IDX_IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    const auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto splitDim = attrs->GetAttrPointer<gert::TypedContinuousVector<int64_t>>(PHONY_SPLIT_IDX_ATTR_SPLIT_DIM);
    OP_CHECK_NULL_WITH_CONTEXT(context, splitDim);
    const auto numSplit = attrs->GetAttrPointer<gert::TypedContinuousVector<int64_t>>(PHONY_SPLIT_IDX_ATTR_NUM_SPLIT);
    OP_CHECK_NULL_WITH_CONTEXT(context, numSplit);

    OP_CHECK_IF(splitDim->GetSize() != numSplit->GetSize(),
                OP_LOGE(context,
                        "The size of split_dim and num_split must be the same, split_dim size is %zu, "
                        "num_split size is %zu.",
                        splitDim->GetSize(), numSplit->GetSize()),
                return ge::GRAPH_FAILED);

    // the product of num_split must equal the number of dynamic output instances
    int64_t splitTotalNum = 1;
    for (size_t i = 0; i < numSplit->GetSize(); ++i) {
        splitTotalNum *= numSplit->GetData()[i];
    }
    const size_t outputNum = context->GetComputeNodeOutputNum();
    OP_CHECK_IF(splitTotalNum != static_cast<int64_t>(outputNum),
                OP_LOGE(context,
                        "The product of num_split must be equal to the number of outputs, product of "
                        "num_split is %ld, output num is %zu.",
                        splitTotalNum, outputNum),
                return ge::GRAPH_FAILED);

    // input shape is unknown rank (-2), all outputs keep unknown rank
    if (Ops::Base::IsUnknownRank(*xShape)) {
        for (size_t i = 0; i < outputNum; i++) {
            gert::Shape* outShape = context->GetOutputShape(i);
            OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
            *outShape = *xShape;
        }
        return ge::GRAPH_SUCCESS;
    }

    // check attrs and normalize negative split_dim
    const int64_t dimNum = static_cast<int64_t>(xShape->GetDimNum());
    std::vector<int64_t> splitDims(splitDim->GetData(), splitDim->GetData() + splitDim->GetSize());
    for (size_t i = 0; i < splitDims.size(); ++i) {
        OP_CHECK_IF(
            (splitDims[i] < -dimNum) || (splitDims[i] >= dimNum),
            OP_LOGE(context, "split_dim[%zu] is %ld, must be in range [-%ld, %ld).", i, splitDims[i], dimNum, dimNum),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            numSplit->GetData()[i] < 1,
            OP_LOGE(context, "num_split[%zu] is %ld, must be greater than or equal to 1.", i, numSplit->GetData()[i]),
            return ge::GRAPH_FAILED);
        if (splitDims[i] < 0) {
            splitDims[i] += dimNum;
        }
    }

    // output shape: dim on each split_dim is divided by num_split (must be divisible),
    // unknown (-1) dim is kept. Divisions are applied sequentially, same as RT1.0.
    gert::Shape outShape = *xShape;
    for (size_t i = 0; i < splitDims.size(); ++i) {
        const size_t dimIdx = static_cast<size_t>(splitDims[i]);
        if (outShape.GetDim(dimIdx) == -1) {
            continue;
        }
        OP_CHECK_IF(outShape.GetDim(dimIdx) % numSplit->GetData()[i] != 0,
                    OP_LOGE(context, "The dim[%zu] of input is %ld, should be divisible by num_split[%zu]=%ld.", dimIdx,
                            outShape.GetDim(dimIdx), i, numSplit->GetData()[i]),
                    return ge::GRAPH_FAILED);
        outShape.SetDim(dimIdx, outShape.GetDim(dimIdx) / numSplit->GetData()[i]);
    }

    // all outputs share the same shape
    for (size_t i = 0; i < outputNum; i++) {
        gert::Shape* yShape = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
        *yShape = outShape;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForPhonySplit(gert::InferDataTypeContext* context)
{
    const ge::DataType xDtype = context->GetInputDataType(PHONY_SPLIT_IDX_IN_X);
    const size_t outputNum = context->GetComputeNodeOutputNum();
    for (size_t i = 0; i < outputNum; i++) {
        OP_CHECK_IF(context->SetOutputDataType(i, xDtype) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context, "Set dtype of output[%zu] failed.", i), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PhonySplit).InferShape(InferShapeForPhonySplit).InferDataType(InferDataTypeForPhonySplit);
} // namespace ops
