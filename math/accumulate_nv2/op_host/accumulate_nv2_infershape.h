/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACCUMULATE_NV2_INFERSHAPE_H_
#define ACCUMULATE_NV2_INFERSHAPE_H_

#include <cstdint>
#include <string>
#include <vector>

#include "graph/ge_error_codes.h"
#include "op_common/log/log.h"

namespace ops {

// Core shape-resolution logic for AccumulateNV2 InferShape. Single source of
// truth, called by the V2 gert wrapper in op_host/accumulate_nv2_infershape.cpp
// (it previously also served the legacy V1 registration in op_graph during
// the GEIR V1-route verification; that registration has been removed).
//
// The operator itself does not support broadcast: every input must share the
// first input's shape. aclnnSum aligns tensors with BroadcastTo in op_api
// before composing the inner operator, so aligned calls still pass.
// Dynamic-shape markers: -2 (unknown rank) inputs are runtime-determined and
// skipped; -1 (unknown dim) resolves to the first concrete value seen for
// that dim, and two different concrete values for the same dim fail (tiling
// independently rejects non-concrete shapes).
//
// Params:
//   opName        - log tag
//   attrN         - the N attribute value; must match the input count
//   inputDimsList - one std::vector<int64_t> per input (dims of that input)
//   outputDims    - [out] the resolved output dims
// Returns GRAPH_SUCCESS or GRAPH_FAILED (with OP_LOGE on failure).
inline ge::graphStatus AccumulateNV2ResolveShape(const char* opName, int64_t attrN,
                                                 const std::vector<std::vector<int64_t>>& inputDimsList,
                                                 std::vector<int64_t>& outputDims)
{
    if (attrN < 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "N", std::to_string(attrN).c_str(), "must be at least 1");
        return ge::GRAPH_FAILED;
    }
    const size_t inputNum = inputDimsList.size();
    if (inputNum == 0U || static_cast<int64_t>(inputNum) != attrN) {
        const std::string reason = "must equal the actual x input count " + std::to_string(inputNum);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "N", std::to_string(attrN).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }

    outputDims = inputDimsList[0];
    const bool firstUnknownRank = outputDims.size() == 1U && outputDims[0] == -2;
    std::vector<int64_t>& resolved = outputDims;
    std::vector<size_t> resolvedFrom(outputDims.size(), 0U);
    for (size_t i = 1; i < inputNum; ++i) {
        const std::vector<int64_t>& inputDims = inputDimsList[i];
        if (firstUnknownRank) {
            continue;
        }
        if (inputDims.size() == 1U && inputDims[0] == -2) {
            continue;
        }
        if (inputDims.size() != outputDims.size()) {
            const std::string inputNames = "x[" + std::to_string(i) + "] and x[0]";
            const std::string incorrectDims = std::to_string(inputDims.size()) + " and " +
                                              std::to_string(outputDims.size());
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(opName, inputNames.c_str(), incorrectDims.c_str(),
                                                      "input ranks must match; broadcasting is not supported");
            return ge::GRAPH_FAILED;
        }
        for (size_t d = 0; d < outputDims.size(); ++d) {
            if (inputDims[d] == -1) {
                continue;
            }
            if (resolved[d] == -1) {
                resolved[d] = inputDims[d];
                resolvedFrom[d] = i;
                continue;
            }
            if (inputDims[d] != resolved[d]) {
                const std::string inputNames = "x[" + std::to_string(i) + "] and x[" + std::to_string(resolvedFrom[d]) +
                                               "]";
                const std::string incorrectDims = std::to_string(inputDims[d]) + " and " + std::to_string(resolved[d]);
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    opName, inputNames.c_str(), incorrectDims.c_str(),
                    ("dimension " + std::to_string(d) + " must match; broadcasting is not supported").c_str());
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace ops

#endif // ACCUMULATE_NV2_INFERSHAPE_H_
