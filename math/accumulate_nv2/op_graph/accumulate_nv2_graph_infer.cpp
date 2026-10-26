/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "log/log.h"
#include "accumulate_nv2_graph_infer.h"

namespace ge {
namespace {

bool IsUnknownRank(const std::vector<int64_t>& dims) { return dims == UNKNOWN_RANK; }

} // namespace

IMPLEMT_COMMON_INFERFUNC(AccumulateNV2LegacyV1InferShape)
{
    int64_t attrN = 0;
    if (op.GetAttr("N", attrN) != GRAPH_SUCCESS || attrN <= 0) {
        return GRAPH_FAILED;
    }

    // GetDynamicInputNum("x") is 0 on the GEIR V1 route even though the
    // flattened dynamic inputs are present. GetInputsSize() is the reliable
    // count used by the legacy registry.
    const size_t inputNum = op.GetInputsSize();
    if (inputNum == 0U || static_cast<int64_t>(inputNum) != attrN) {
        return GRAPH_FAILED;
    }

    const TensorDesc firstInputDesc = op.GetDynamicInputDesc("x", 0U);
    const DataType dataType = firstInputDesc.GetDataType();
    std::vector<int64_t> outputDims = firstInputDesc.GetShape().GetDims();
    // The operator itself does not support broadcast: every input must share the
    // first input's shape. aclnnSum aligns tensors with BroadcastTo in op_api
    // before composing the inner operator, so aligned calls still pass.
    // Dynamic-shape markers keep their inference-time semantics: -2 (unknown
    // rank) inputs are runtime-determined and skipped; -1 (unknown dim) is a
    // wildcard that resolves against the first concrete value seen for that
    // dim (the output keeps the resolved dims), and two different concrete
    // values for the same dim fail.
    const bool firstUnknownRank = IsUnknownRank(outputDims);
    std::vector<int64_t> resolvedDims = outputDims;
    for (size_t i = 1; i < inputNum; ++i) {
        const TensorDesc inputDesc = op.GetDynamicInputDesc("x", static_cast<uint32_t>(i));
        if (inputDesc.GetDataType() != dataType) {
            return GRAPH_FAILED;
        }
        if (firstUnknownRank) {
            continue;
        }
        const std::vector<int64_t> inputDims = inputDesc.GetShape().GetDims();
        if (IsUnknownRank(inputDims)) {
            continue;
        }
        if (inputDims.size() != outputDims.size()) {
            return GRAPH_FAILED;
        }
        for (size_t d = 0; d < outputDims.size(); ++d) {
            if (inputDims[d] == -1) {
                continue;
            }
            if (resolvedDims[d] == -1) {
                resolvedDims[d] = inputDims[d];
                continue;
            }
            if (inputDims[d] != resolvedDims[d]) {
                return GRAPH_FAILED;
            }
        }
    }

    TensorDesc outputDesc = op.GetOutputDesc(0U);
    outputDesc.SetShape(Shape(resolvedDims));
    outputDesc.SetDataType(dataType);
    return op.UpdateOutputDesc(0U, outputDesc);
}

// GEIR still invokes the legacy V1 registry for this built-in op name. Register
// the package implementation before libopsproto.so is loaded so zero-element
// broadcast shapes use the safe implementation above instead of 0 % 0.
COMMON_INFER_FUNC_REG(AccumulateNV2, AccumulateNV2LegacyV1InferShape);

} // namespace ge

namespace ops {

static ge::graphStatus InferDataType4AccumulateNV2(gert::InferDataTypeContext* context)
{
    const size_t inputNum = context->GetComputeNodeInputNum();
    if (inputNum == 0U) {
        OP_LOGE("AccumulateNV2", "input count must not be zero");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType firstDtype = context->GetInputDataType(0);
    for (size_t i = 1U; i < inputNum; ++i) {
        const ge::DataType dtype = context->GetInputDataType(i);
        if (dtype != firstDtype) {
            OP_LOGE("AccumulateNV2",
                    "all input dtypes must be the same as the first input, but input[%zu] dtype %d differs from "
                    "input[0] dtype %d",
                    i, static_cast<int32_t>(dtype), static_cast<int32_t>(firstDtype));
            return ge::GRAPH_FAILED;
        }
    }
    context->SetOutputDataType(0, firstDtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(AccumulateNV2).InferDataType(InferDataType4AccumulateNV2);

} // namespace ops
