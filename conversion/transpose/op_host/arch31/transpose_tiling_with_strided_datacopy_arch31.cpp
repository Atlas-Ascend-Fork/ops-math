/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transpose_tiling_with_strided_datacopy_arch31.h"

#include <algorithm>
#include <limits>

#include "log/log.h"

namespace optiling {
namespace {
constexpr uint64_t STRIDED_DATACOPY_UB_RESERVED_BYTES = 256UL;
constexpr uint64_t STRIDED_DATACOPY_ALIGN_BYTES = 128UL;
constexpr uint64_t STRIDED_DATACOPY_MAX_BLOCK_COUNT = 4095UL;
constexpr uint64_t STRIDED_DATACOPY_BUFFER_COUNT = 2UL;
constexpr uint64_t STRIDED_DATACOPY_UINT32_MAX = 0xFFFFFFFFUL;
bool IsStridedDataCopyElementBytesSupported(uint64_t elementBytes)
{
    return elementBytes == 1UL || elementBytes == 2UL || elementBytes == 4UL || elementBytes == 8UL;
}
} // namespace

bool BuildKirinStridedDataCopyTransposeTiling(const TransposeCompilerInfo& compilerInfo,
                                              const KirinTransposeShapeInfo& shapeInfo,
                                              const KirinReducedShapeInfo& reducedShape,
                                              KirinTransposeStridedDataCopyTilingData& tilingData, uint32_t& blockDim)
{
    const uint32_t rank = reducedShape.rank;
    if (rank < 3) {
        return false;
    }
    const int32_t lastIdx = static_cast<int32_t>(rank) - 1;
    const int32_t midIdx = static_cast<int32_t>(rank) - 2;
    const int32_t nIdx = static_cast<int32_t>(rank) - 3;
    // 1. Lowest dim W contiguous in source and output.
    if (reducedShape.srcStride[lastIdx] != 1 || reducedShape.outStride[lastIdx] != 1) {
        return false;
    }

    const uint64_t W = reducedShape.outShape[lastIdx];
    const uint64_t elementBytes = shapeInfo.elementBytes;
    if (W == 0 || elementBytes == 0) {
        return false;
    }

    const uint64_t W_bytes = W * elementBytes;
    // 2. W aligned to 128B
    if (W_bytes % STRIDED_DATACOPY_ALIGN_BYTES != 0) {
        return false;
    }

    // 3. Output mid-axis stride == W (M axis directly precedes W in output).
    if (reducedShape.outStride[midIdx] != W) {
        return false;
    }

    // 4. Source mid-axis stride is an interger multiple of W.
    const uint64_t srcMidStride = reducedShape.srcStride[midIdx];
    if (srcMidStride % W != 0) {
        return false;
    }

    const uint64_t N = srcMidStride / W;
    // 5. N >= 2 (otherwise no transpose needed).
    if (N < 2) {
        return false;
    }

    if (reducedShape.outShape[nIdx] != N || reducedShape.srcStride[nIdx] != W) {
        return false;
    }

    for (int32_t axis = 0; axis < nIdx; ++axis) {
        if (reducedShape.srcStride[axis] != reducedShape.outStride[axis]) {
            return false;
        }
    }

    const uint64_t M = reducedShape.outShape[midIdx];
    if (M == 0) {
        return false;
    }

    const uint64_t MNW = M * N * W;
    if (MNW == 0 || shapeInfo.totalElements % MNW != 0) {
        return false;
    }
    const uint64_t B = shapeInfo.totalElements / MNW;

    if (!IsStridedDataCopyElementBytesSupported(elementBytes)) {
        return false;
    }
    const uint64_t srcStrideBytes = (N - 1UL) * W_bytes;
    if (srcStrideBytes > STRIDED_DATACOPY_UINT32_MAX) {
        return false;
    }

    const uint64_t ubSize = static_cast<uint64_t>(compilerInfo.ubSize);
    if (ubSize <= STRIDED_DATACOPY_UB_RESERVED_BYTES) {
        return false;
    }
    const uint64_t availableUb = ubSize - STRIDED_DATACOPY_UB_RESERVED_BYTES;
    const uint64_t perBufferBytes = W_bytes;
    uint64_t maxMTile = availableUb / STRIDED_DATACOPY_BUFFER_COUNT / perBufferBytes;
    if (maxMTile == 0) {
        return false;
    }
    if (maxMTile > STRIDED_DATACOPY_MAX_BLOCK_COUNT) {
        maxMTile = STRIDED_DATACOPY_MAX_BLOCK_COUNT;
    }
    uint64_t mTile = M < maxMTile ? M : maxMTile;

    // --- Fill tiling data ---
    auto& op = tilingData.transposeStridedDataCopyOpTiling;
    op.set_mLen(M);
    op.set_nLen(N);
    op.set_wLen(W);
    op.set_bLen(B);
    op.set_wBytes(W_bytes);
    op.set_mTile(mTile);
    op.set_elementBytes(static_cast<uint8_t>(elementBytes));
    op.set_batchStride(MNW);
    blockDim = 1;
    return true;
}
} // namespace optiling
