/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transpose_tiling_generic_arch31.h"
#include "transpose_tiling_with_021vconv_arch31.h"
#include "transpose_tiling_with_nchwconv_arch31.h"
#include "transpose_tiling_with_strided_datacopy_arch31.h"

#include <algorithm>
#include <array>
#include <limits>

#include "log/log.h"

namespace optiling {
namespace {
constexpr uint64_t MAX_UB_TILE_BYTES = 64UL * 1024UL;

template <typename T>
ge::graphStatus CommitTiling(gert::TilingContext* context, KirinTransposeKey key, uint32_t blockDim, T& tilingData)
{
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    auto* rawTilingData = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData);
    OP_CHECK_IF(tilingData.GetDataSize() > rawTilingData->GetCapacity(),
                OP_LOGE(context->GetNodeName(), "Kirin transpose tiling buffer is too small"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(blockDim == 0 || context->SetBlockDim(blockDim) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Failed to set Kirin transpose block dim: %u", blockDim),
                return ge::GRAPH_FAILED);

    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());
    workspaces[0] = TRANSPOSE_WORKSPACE_SIZE;
    context->SetTilingKey(static_cast<uint64_t>(key));
    return ge::GRAPH_SUCCESS;
}

void FillOpTiling(KirinTransposeOpTilingData& opTiling, const KirinTransposeShapeInfo& shapeInfo,
                  const KirinReducedShapeInfo& reducedShape, uint64_t blocksPerCore, uint64_t extraBlockCoreCount,
                  uint64_t tailElements, uint64_t usedCoreCount, uint64_t ubTileElements64)
{
    opTiling.set_totalElements(shapeInfo.totalElements);
    opTiling.set_blocksPerCore(blocksPerCore);
    opTiling.set_extraBlockCoreCount(extraBlockCoreCount);
    opTiling.set_tailElements(tailElements);
    opTiling.set_usedCoreCount(static_cast<uint32_t>(usedCoreCount));
    opTiling.set_ubTileElements(static_cast<uint32_t>(ubTileElements64));
    opTiling.set_reducedRank(static_cast<uint8_t>(reducedShape.rank));
    opTiling.set_elementBytes(static_cast<uint8_t>(shapeInfo.elementBytes));
    opTiling.set_reserved(0);
    auto outShape = reducedShape.outShape;
    auto srcStride = reducedShape.srcStride;
    auto outStride = reducedShape.outStride;
    opTiling.set_outShape(outShape.data());
    opTiling.set_srcStride(srcStride.data());
    opTiling.set_outStride(outStride.data());
}
} // namespace

ge::graphStatus DoKirinTransposeTiling(gert::TilingContext* context, const TransposeCompilerInfo& compilerInfo,
                                       const KirinTransposeShapeInfo& shapeInfo,
                                       const KirinReducedShapeInfo& reducedShape)
{
    const bool isTensorMove = reducedShape.rank == 1 && reducedShape.srcStride[0] == 1;
    if (!isTensorMove) {
        KirinTransposeVConvTilingData vconvTiling;
        uint32_t vconvBlockDim = 0;
        if (BuildKirinTransposeVConvTiling(compilerInfo, shapeInfo, reducedShape, vconvTiling, vconvBlockDim)) {
            return CommitTiling(context, KirinTransposeKey::VCONV_TRANSPOSE, vconvBlockDim, vconvTiling);
        }

        KirinTranspose021VConvTilingData vconv021Tiling;
        uint32_t vconv021BlockDim = 0;
        if (BuildKirinTranspose021VConvTiling(compilerInfo, shapeInfo, reducedShape, vconv021Tiling,
                                              vconv021BlockDim)) {
            return CommitTiling(context, KirinTransposeKey::VCONV_021_TRANSPOSE, vconv021BlockDim, vconv021Tiling);
        }

        KirinTransposeStridedDataCopyTilingData stridedDataCopyTiling;
        uint32_t stridedDataCopyBlockDim = 0;
        if (BuildKirinStridedDataCopyTransposeTiling(compilerInfo, shapeInfo, reducedShape, stridedDataCopyTiling,
                                                     stridedDataCopyBlockDim)) {
            return CommitTiling(context, KirinTransposeKey::STRIDED_DATACOPY_TRANSPOSE, stridedDataCopyBlockDim,
                                stridedDataCopyTiling);
        }
    }

    OP_CHECK_IF(shapeInfo.elementBytes == 8,
                OP_LOGE(context->GetNodeName(),
                        "INT64 transpose only supported via strided-datacopy template (W must be 128B-aligned)"),
                return ge::GRAPH_FAILED);

    const uint64_t elementsPerBlock = TRANSPOSE_BLOCK_BYTES / shapeInfo.elementBytes;
    const uint64_t fullBlockCount = shapeInfo.totalElements / elementsPerBlock;
    const uint64_t tailElements = shapeInfo.totalElements % elementsPerBlock;
    const uint64_t usableCoreCount = std::max<int64_t>(compilerInfo.coreNum, 1);
    const uint64_t normalUsedCoreCount = fullBlockCount == 0 ? 1 : std::min(fullBlockCount, usableCoreCount);
    const bool forceSingleCoreDiagnostic = !isTensorMove && shapeInfo.elementBytes == sizeof(float) &&
                                           reducedShape.rank == 2 && reducedShape.outShape[0] == 4096 &&
                                           reducedShape.outShape[1] == 4096;
    const uint64_t usedCoreCount = forceSingleCoreDiagnostic ? 1 : normalUsedCoreCount;
    const uint64_t blocksPerCore = fullBlockCount / usedCoreCount;
    const uint64_t extraBlockCoreCount = fullBlockCount % usedCoreCount;

    const uint64_t availableUbBytes = std::min(static_cast<uint64_t>(compilerInfo.ubSize) / 2, MAX_UB_TILE_BYTES);
    const uint64_t alignedUbBytes = availableUbBytes / TRANSPOSE_BLOCK_BYTES * TRANSPOSE_BLOCK_BYTES;
    const uint64_t ubTileElements64 = alignedUbBytes / shapeInfo.elementBytes;
    OP_CHECK_IF(ubTileElements64 == 0 || ubTileElements64 > std::numeric_limits<uint32_t>::max(),
                OP_LOGE(context->GetNodeName(), "Invalid Kirin transpose UB tile size: %lu", ubTileElements64),
                return ge::GRAPH_FAILED);

    const auto tilingKey = isTensorMove ? KirinTransposeKey::TENSOR_MOVE : KirinTransposeKey::GENERIC_DATACOPY;

    // Use the wrapper type matching the tiling key
    if (tilingKey == KirinTransposeKey::TENSOR_MOVE) {
        KirinTransposeTensorMoveTilingData tilingData;
        FillOpTiling(tilingData.transposeOpTiling, shapeInfo, reducedShape, blocksPerCore, extraBlockCoreCount,
                     tailElements, usedCoreCount, ubTileElements64);
        return CommitTiling(context, tilingKey, static_cast<uint32_t>(usedCoreCount), tilingData);
    } else {
        KirinTransposeGenericDataCopyTilingData tilingData;
        FillOpTiling(tilingData.transposeOpTiling, shapeInfo, reducedShape, blocksPerCore, extraBlockCoreCount,
                     tailElements, usedCoreCount, ubTileElements64);
        return CommitTiling(context, tilingKey, static_cast<uint32_t>(usedCoreCount), tilingData);
    }
}
} // namespace optiling
