/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transpose_tiling_base_arch31.h"

#include <limits>

namespace optiling {
namespace {
bool CheckedMultiply(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}
} // namespace

bool BuildKirinReducedShape(const KirinTransposeShapeInfo& input, KirinReducedShapeInfo& output)
{
    std::array<uint64_t, TRANSPOSE_MAX_AXIS_NUM> inputStride{};
    uint64_t stride = 1;
    for (int32_t axis = static_cast<int32_t>(input.rank) - 1; axis >= 0; --axis) {
        inputStride[axis] = stride;
        if (!CheckedMultiply(stride, input.inputShape[axis], stride)) {
            return false;
        }
    }

    std::array<uint64_t, TRANSPOSE_MAX_AXIS_NUM> shape{};
    std::array<uint64_t, TRANSPOSE_MAX_AXIS_NUM> srcStride{};
    uint32_t rank = 0;
    for (uint32_t axis = 0; axis < input.rank; ++axis) {
        const uint64_t axisSize = input.outputShape[axis];
        if (axisSize == 1) {
            continue;
        }
        shape[rank] = axisSize;
        srcStride[rank] = inputStride[static_cast<uint32_t>(input.perm[axis])];
        ++rank;
    }

    if (rank == 0) {
        rank = 1;
        shape[0] = 1;
        srcStride[0] = 1;
    }

    uint32_t mergedRank = 0;
    for (uint32_t axis = 0; axis < rank; ++axis) {
        uint64_t expectedOuterStride = 0;
        const bool canMerge = CheckedMultiply(shape[axis], srcStride[axis], expectedOuterStride) && mergedRank > 0 &&
                              srcStride[mergedRank - 1] == expectedOuterStride;
        if (canMerge) {
            if (!CheckedMultiply(shape[mergedRank - 1], shape[axis], shape[mergedRank - 1])) {
                return false;
            }
            srcStride[mergedRank - 1] = srcStride[axis];
        } else {
            shape[mergedRank] = shape[axis];
            srcStride[mergedRank] = srcStride[axis];
            ++mergedRank;
        }
    }

    output.rank = mergedRank;
    uint64_t outStride = 1;
    for (int32_t axis = static_cast<int32_t>(mergedRank) - 1; axis >= 0; --axis) {
        output.outShape[axis] = shape[axis];
        output.srcStride[axis] = srcStride[axis];
        output.outStride[axis] = outStride;
        if (!CheckedMultiply(outStride, shape[axis], outStride)) {
            return false;
        }
    }
    return true;
}
} // namespace optiling
