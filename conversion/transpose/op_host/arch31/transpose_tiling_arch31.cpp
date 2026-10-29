/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transpose_tiling_arch31.h"

#include <array>
#include <limits>
#include <string>

#include "log/log.h"
#include "transpose_tiling_base_arch31.h"
#include "transpose_tiling_generic_arch31.h"
#include "util/platform_util.h"

namespace optiling {
namespace {
template <typename T>
bool ReadPerm(gert::TilingContext* context, const gert::Tensor* permTensor, KirinTransposeShapeInfo& shapeInfo)
{
    const T* values = permTensor->GetData<T>();
    if (values == nullptr) {
        OP_LOGE(context->GetNodeName(), "Perm data is null");
        return false;
    }
    for (uint32_t axis = 0; axis < shapeInfo.rank; ++axis) {
        const int64_t value = static_cast<int64_t>(values[axis]);
        shapeInfo.perm[axis] = value < 0 ? value + static_cast<int64_t>(shapeInfo.rank) : value;
    }
    return true;
}

ge::graphStatus ReadAndValidateInputs(gert::TilingContext* context, KirinTransposeShapeInfo& shapeInfo)
{
    const gert::Tensor* input = context->GetInputTensor(INPUT_IDX_X);
    const gert::Tensor* permTensor = context->GetInputTensor(INPUT_IDX_PERM);
    const gert::StorageShape* output = context->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, input);
    OP_CHECK_NULL_WITH_CONTEXT(context, permTensor);
    OP_CHECK_NULL_WITH_CONTEXT(context, output);

    const gert::Shape& inputShape = input->GetStorageShape();
    const gert::Shape& outputShape = output->GetStorageShape();
    const size_t rank = inputShape.GetDimNum();
    OP_CHECK_IF(rank == 0 || rank > TRANSPOSE_MAX_AXIS_NUM || outputShape.GetDimNum() != rank ||
                    permTensor->GetShapeSize() != static_cast<int64_t>(rank),
                OP_LOGE(context->GetNodeName(), "Transpose rank must be in [1, 8] and match y/perm"),
                return ge::GRAPH_FAILED);
    shapeInfo.rank = static_cast<uint32_t>(rank);

    const ge::DataType inputDtype = context->GetInputDesc(INPUT_IDX_X)->GetDataType();
    shapeInfo.inputDtype = inputDtype;
    shapeInfo.elementBytes = static_cast<uint32_t>(ge::GetSizeByDataType(inputDtype));
    OP_CHECK_IF(shapeInfo.elementBytes != 1 && shapeInfo.elementBytes != 2 && shapeInfo.elementBytes != 4 &&
                    shapeInfo.elementBytes != 8,
                OP_LOGE(context->GetNodeName(), "Kirin transpose only supports 1/2/4/8-byte element types"),
                return ge::GRAPH_FAILED);

    const ge::DataType permDtype = context->GetInputDesc(INPUT_IDX_PERM)->GetDataType();
    if (permDtype == ge::DT_INT32) {
        OP_CHECK_IF(!ReadPerm<int32_t>(context, permTensor, shapeInfo),
                    OP_LOGE(context->GetNodeName(), "Failed to read int32 perm"), return ge::GRAPH_FAILED);
    } else if (permDtype == ge::DT_INT64) {
        OP_CHECK_IF(!ReadPerm<int64_t>(context, permTensor, shapeInfo),
                    OP_LOGE(context->GetNodeName(), "Failed to read int64 perm"), return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(context->GetNodeName(), "Perm dtype must be int32 or int64");
        return ge::GRAPH_FAILED;
    }

    std::array<bool, TRANSPOSE_MAX_AXIS_NUM> axisSeen{};
    uint64_t totalElements = 1;
    for (uint32_t axis = 0; axis < shapeInfo.rank; ++axis) {
        const int64_t inputDim = inputShape[axis];
        const int64_t outputDim = outputShape[axis];
        OP_CHECK_IF(inputDim <= 0 || outputDim <= 0,
                    OP_LOGE(context->GetNodeName(), "Transpose dimensions must be positive"), return ge::GRAPH_FAILED);
        shapeInfo.inputShape[axis] = static_cast<uint64_t>(inputDim);
        shapeInfo.outputShape[axis] = static_cast<uint64_t>(outputDim);

        const int64_t permAxis = shapeInfo.perm[axis];
        OP_CHECK_IF(permAxis < 0 || permAxis >= static_cast<int64_t>(shapeInfo.rank) || axisSeen[permAxis],
                    OP_LOGE(context->GetNodeName(), "Perm must contain every axis exactly once"),
                    return ge::GRAPH_FAILED);
        axisSeen[permAxis] = true;
        OP_CHECK_IF(outputDim != inputShape[permAxis],
                    OP_LOGE(context->GetNodeName(), "Output shape does not match perm"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(totalElements > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(inputDim),
                    OP_LOGE(context->GetNodeName(), "Transpose element count overflows uint64"),
                    return ge::GRAPH_FAILED);
        totalElements *= static_cast<uint64_t>(inputDim);
    }
    OP_CHECK_IF(totalElements > std::numeric_limits<uint64_t>::max() / shapeInfo.elementBytes,
                OP_LOGE(context->GetNodeName(), "Transpose byte size overflows uint64"), return ge::GRAPH_FAILED);
    shapeInfo.totalElements = totalElements;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeTilingForAscendC(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto* compilerInfo = context->GetCompileInfo<TransposeCompilerInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compilerInfo);

    KirinTransposeShapeInfo shapeInfo;
    const ge::graphStatus status = ReadAndValidateInputs(context, shapeInfo);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    KirinReducedShapeInfo reducedShape;
    OP_CHECK_IF(!BuildKirinReducedShape(shapeInfo, reducedShape),
                OP_LOGE(context->GetNodeName(), "Failed to reduce transpose shape or stride overflowed"),
                return ge::GRAPH_FAILED);
    return DoKirinTransposeTiling(context, *compilerInfo, shapeInfo, reducedShape);
}
} // namespace

ge::graphStatus TilingPrepareTransposeForAscendC(gert::TilingParseContext* context)
{
    auto* compilerInfo = context->GetCompiledInfo<TransposeCompilerInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compilerInfo);
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    compilerInfo->coreNum = platform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compilerInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(compilerInfo->coreNum <= 0 || compilerInfo->ubSize < TRANSPOSE_BLOCK_BYTES,
                OP_LOGE(context->GetNodeName(), "Invalid Kirin AIV/UB platform information"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Transpose)
    .Tiling(TransposeTilingForAscendC)
    .TilingParse<TransposeCompilerInfo>(TilingPrepareTransposeForAscendC);
} // namespace optiling
