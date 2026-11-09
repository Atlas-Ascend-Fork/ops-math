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
 * \file matrix_diag_part_tiling_arch35.cpp
 * \brief Implemention of MatrixDiagPart tiling
 */

#include "matrix_diag_part_tiling_arch35.h"
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_base_util.h"
#include "op_host/math_tiling_templates_registry.h"
#include <limits>

namespace optiling {

using namespace Ops::Math::OpTiling;

constexpr int64_t PER_CORE_MIN = 1024;
constexpr uint32_t DCACHE_SIZE = 128 * 1024;
constexpr uint32_t STATIC_UB_ESTIMATE = 0;
constexpr size_t MATRIX_DIAG_PART_MIN_RANK = 2;
constexpr size_t MATRIX_DIAG_PART_MAX_RANK = 8;

struct MatrixDiagPartCompileInfo {};

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckRank(gert::TilingContext* context, size_t originalRank, size_t rank)
{
    if (rank < MATRIX_DIAG_PART_MIN_RANK) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "x", std::to_string(originalRank).c_str(),
                                                  "input rank must be >= " + std::to_string(MATRIX_DIAG_PART_MIN_RANK));
        return ge::GRAPH_FAILED;
    }
    if (rank > MATRIX_DIAG_PART_MAX_RANK) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "x", std::to_string(originalRank).c_str(),
                                                  "input rank must be <= " + std::to_string(MATRIX_DIAG_PART_MAX_RANK));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckDimsConcrete(gert::TilingContext* context, const gert::Shape& shape, size_t rank)
{
    // Tiling requires every dimension to be a concrete non-negative value. Checking only the
    // last two dims (or the signed product) misses batch-position -1/negative dims whose
    // product can be positive (e.g. [-1,-1,3,4]).
    for (size_t i = 0; i < rank; ++i) {
        if (shape.GetDim(i) < 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x", std::to_string(shape.GetDim(i)).c_str(),
                                                  "all dimensions of x must be concrete and non-negative at tiling");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ComputeElementCounts(gert::TilingContext* context, const gert::Shape& shape, size_t rank,
                                            int64_t& d, int64_t& n, int64_t& matrixSize, int64_t& totalOutputElements)
{
    int64_t M = shape.GetDim(rank - 2);
    n = shape.GetDim(rank - 1);
    d = std::min(M, n);

    constexpr int64_t kInt64Max = std::numeric_limits<int64_t>::max();
    // Guard M * n overflow before evaluating it.
    if (M != 0 && n > kInt64Max / M) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x",
                                              (std::to_string(M) + "x" + std::to_string(n)).c_str(),
                                              "the product of the last two dimensions of x overflows int64");
        return ge::GRAPH_FAILED;
    }
    matrixSize = M * n;

    // GetShapeSize returns kInvalidDimValue (INT64_MIN) on product overflow and a negative
    // product for unknown dims; both must be rejected before they reach tiling/kernel.
    int64_t totalInputElements = shape.GetShapeSize();
    if (totalInputElements < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x", std::to_string(totalInputElements).c_str(),
                                              "the element count of x must be non-negative (unknown dim or overflow)");
        return ge::GRAPH_FAILED;
    }
    int64_t batchTotal = (matrixSize > 0) ? (totalInputElements / matrixSize) : 0;
    // Guard batchTotal * d overflow before evaluating it.
    if (d != 0 && batchTotal > kInt64Max / d) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x",
                                              (std::to_string(batchTotal) + "x" + std::to_string(d)).c_str(),
                                              "the output element count overflows int64");
        return ge::GRAPH_FAILED;
    }
    totalOutputElements = batchTotal * d;

    // inputRowStride (n + 1) is only used when the output is non-empty; zero-element
    // empty tensors (README: supported) must not be rejected for an unused stride.
    if (totalOutputElements > 0 && n == kInt64Max) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x", std::to_string(n).c_str(),
                                              "the last dimension of x is too large: n + 1 overflows int64");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateOutputShape(gert::TilingContext* context, const gert::Shape& inputShape, size_t rank,
                                           int64_t d)
{
    auto outputY = context->GetOutputShape(0);
    if (outputY == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto outputShape = outputY->GetStorageShape();
    if (outputShape.GetDimNum() != rank - 1) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "y",
                                                  std::to_string(outputShape.GetDimNum()).c_str(),
                                                  "output rank must be input rank minus 1");
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i + 2 < rank; ++i) {
        if (outputShape.GetDim(i) >= 0 && outputShape.GetDim(i) != inputShape.GetDim(i)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "y",
                                                  std::to_string(outputShape.GetDim(i)).c_str(),
                                                  "output batch dimensions must match the input");
            return ge::GRAPH_FAILED;
        }
    }
    int64_t outDiag = outputShape.GetDim(rank - 2);
    if (outDiag >= 0 && outDiag != d) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "y", std::to_string(outDiag).c_str(),
                                              "output diagonal length must equal min(last two input dims)");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateInputAndComputeDims(gert::TilingContext* context, int64_t& d, int64_t& n,
                                                   int64_t& matrixSize, int64_t& totalOutputElements)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto rawShape = inputX->GetStorageShape();
    // Report the caller's real rank: EnsureNotScalar rewrites a 0-D scalar to rank 1,
    // which would otherwise hide the true rank in diagnostics.
    auto originalRank = rawShape.GetDimNum();
    auto inputShapeX = Ops::Base::EnsureNotScalar(rawShape);
    auto rank = inputShapeX.GetDimNum();

    OP_CHECK_IF(CheckRank(context, originalRank, rank) != ge::GRAPH_SUCCESS, OP_LOGE(context, "CheckRank error"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckDimsConcrete(context, inputShapeX, rank) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "CheckDimsConcrete error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ComputeElementCounts(context, inputShapeX, rank, d, n, matrixSize, totalOutputElements) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "ComputeElementCounts error"), return ge::GRAPH_FAILED);
    return ValidateOutputShape(context, inputShapeX, rank, d);
}

static ge::graphStatus CheckDtype(gert::TilingContext* context)
{
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8,
                                                   ge::DT_UINT8};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x", Ops::Base::ToString(dataType).c_str(),
                                              "only support float16, float32, int32, int8, uint8");
        return ge::GRAPH_FAILED;
    }

    // README: output y has the same dtype as input x. A mismatched declared output dtype
    // would make the kernel write the input element width into a smaller output buffer.
    auto outputDesc = context->GetOutputDesc(0);
    if (outputDesc != nullptr) {
        auto outputDtype = outputDesc->GetDataType();
        if (outputDtype != ge::DT_UNDEFINED && outputDtype != dataType) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context->GetNodeName(), "x,y",
                (Ops::Base::ToString(dataType) + " and " + Ops::Base::ToString(outputDtype)).c_str(),
                "input and output dtypes must be identical");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetupMemoryAndWorkspace(gert::TilingContext* context, uint64_t ubSize)
{
    auto res = context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS),
                OP_LOGE(context, "SetLocalMemorySize failed, ubSize=%lu, DCACHE_SIZE=%u, STATIC_UB_ESTIMATE=%u", ubSize,
                        DCACHE_SIZE, STATIC_UB_ESTIMATE),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(sysWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MatrixDiagPartTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context, "Enter TilingMatrixDiagPart");
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    int64_t d = 0;
    int64_t n = 0;
    int64_t matrixSize = 0;
    int64_t totalOutputElements = 0;
    OP_CHECK_IF(ValidateInputAndComputeDims(context, d, n, matrixSize, totalOutputElements) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ValidateInputAndComputeDims error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckDtype(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "CheckDtype error"),
                return ge::GRAPH_FAILED);

    MatrixDiagPartTilingData* tiling = context->GetTilingData<MatrixDiagPartTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(MatrixDiagPartTilingData), 0, sizeof(MatrixDiagPartTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    tiling->totalOutputElements = totalOutputElements;
    tiling->diagLen = d;
    // Preserve n + 1 for ordinary shapes; only avoid the overflowing value when n == INT64_MAX
    // (legal only for zero-element empty tensors, which never read the stride).
    tiling->inputRowStride = (n == std::numeric_limits<int64_t>::max()) ? 0 : (n + 1);
    tiling->matrixSize = matrixSize;

    int64_t perCoreElements = Ops::Base::CeilDiv(totalOutputElements, coreNum);
    if (perCoreElements < PER_CORE_MIN) {
        perCoreElements = PER_CORE_MIN;
    }
    int64_t needCoreNum = (totalOutputElements == 0) ? 1 : Ops::Base::CeilDiv(totalOutputElements, perCoreElements);
    context->SetBlockDim(needCoreNum);

    OP_CHECK_IF(SetupMemoryAndWorkspace(context, ubSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "SetupMemoryAndWorkspace error"), return ge::GRAPH_FAILED);

    uint64_t tilingKey = GET_TPL_TILING_KEY(MODE_DEFAULT);
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMatrixDiagPart([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MatrixDiagPart)
    .Tiling(MatrixDiagPartTilingFunc)
    .TilingParse<MatrixDiagPartCompileInfo>(TilingParseForMatrixDiagPart);
} // namespace optiling
