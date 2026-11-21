/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "math/cholesky/op_host/cholesky_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "../../op_kernel/arch22/cholesky_arch22_tiling_key.h"

namespace optiling {
constexpr uint32_t MINIMUM_DIMENSION = 2;
constexpr uint32_t MAXIMUM_DIMENSION = 8;
constexpr uint32_t UPPER_INDEX = 0;
constexpr uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
constexpr uint32_t LOCAL_MEMORY_SIZE = 128U * 1024U;
constexpr uint32_t MAX_BLOCK_SIZE = 256;
constexpr int64_t MAX_MATRIX_SIZE = 8192;

class CholeskyTilingArch22 {
public:
    explicit CholeskyTilingArch22(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus RunBigKernelTiling();

private:
    void PrintTilingData();

private:
    gert::TilingContext* tilingContext = nullptr;
    CholeskyTilingData tilingData;
    uint32_t matSizeN = 0;
    uint64_t matrixNumCount = 1;
    uint32_t needCoreNum = 0;
    bool upper = false;
    uint32_t blockSize = 0;
    uint32_t blockNum = 0;
};

ge::graphStatus CholeskyTilingArch22::Init()
{
    auto inputTensor = tilingContext->GetInputTensor(0);
    if (inputTensor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(tilingContext->GetNodeName(), "self");
        return ge::GRAPH_FAILED;
    }

    auto attrs = tilingContext->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(tilingContext->GetNodeName(), "Get attrs failed.");
        return ge::GRAPH_FAILED;
    }
    const bool* ptrUpper = attrs->GetAttrPointer<bool>(UPPER_INDEX);
    if (ptrUpper == nullptr) {
        OP_LOGE(tilingContext->GetNodeName(), "Get attr upper failed.");
        return ge::GRAPH_FAILED;
    }
    upper = *ptrUpper;

    auto inputDesc = tilingContext->GetInputDesc(0);
    auto outputDesc = tilingContext->GetOutputDesc(0);
    if (inputDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(tilingContext->GetNodeName(), "self");
        return ge::GRAPH_FAILED;
    }
    if (outputDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(tilingContext->GetNodeName(), "out");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(inputDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "self",
                                          ge::TypeUtils::DataTypeToSerialString(inputDesc->GetDataType()), "DT_FLOAT"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(outputDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "out",
                                          ge::TypeUtils::DataTypeToSerialString(outputDesc->GetDataType()), "DT_FLOAT"),
                return ge::GRAPH_FAILED);
    auto inputShape = tilingContext->GetInputShape(0);
    auto outputShape = tilingContext->GetOutputShape(0);
    if (inputShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(tilingContext->GetNodeName(), "self");
        return ge::GRAPH_FAILED;
    }
    if (outputShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(tilingContext->GetNodeName(), "out");
        return ge::GRAPH_FAILED;
    }
    auto matAShape = inputShape->GetOriginShape();
    const std::string actualShapes = Ops::Base::ToString(matAShape) + ", " +
                                     Ops::Base::ToString(outputShape->GetOriginShape());
    OP_CHECK_IF(matAShape != outputShape->GetOriginShape(),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(tilingContext->GetNodeName(), "self, out", actualShapes,
                                                       "input and output shapes must be identical"),
                return ge::GRAPH_FAILED);
    uint32_t inputDim = static_cast<uint32_t>(matAShape.GetDimNum());
    OP_CHECK_IF(
        inputDim < MINIMUM_DIMENSION || inputDim > MAXIMUM_DIMENSION,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(tilingContext->GetNodeName(), "self", std::to_string(inputDim).c_str(),
                                                 "input rank must be in [2, 8]"),
        return ge::GRAPH_FAILED);

    int64_t rowSize = matAShape[inputDim - MINIMUM_DIMENSION];
    int64_t columnSize = matAShape[inputDim - 1];
    OP_CHECK_IF(rowSize != columnSize,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(tilingContext->GetNodeName(), "self",
                                                      Ops::Base::ToString(matAShape).c_str(),
                                                      "the innermost two dimensions must form a square matrix"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(columnSize < 0 || columnSize > MAX_MATRIX_SIZE,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(tilingContext->GetNodeName(), "self",
                                                          std::to_string(columnSize).c_str(),
                                                          "the last dimension must be in [0, 8192]"),
                return ge::GRAPH_FAILED);
    matSizeN = static_cast<uint32_t>(columnSize);
    for (uint32_t i = 0; i < (inputDim - MINIMUM_DIMENSION); i++) {
        // A negative batch dimension carries a GE unknown-dim marker (-1/-2): static_cast<uint64_t>
        // wraps it into a huge value, driving the kernel batch loop and GM offsets far beyond the
        // real tensor, so it must be rejected here.
        OP_CHECK_IF(matAShape[i] < 0,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(tilingContext->GetNodeName(), "self",
                                                              std::to_string(matAShape[i]).c_str(),
                                                              "the batch dimensions must be known and non-negative"),
                    return ge::GRAPH_FAILED);
        matrixNumCount = matrixNumCount * static_cast<uint64_t>(matAShape[i]);
    }

    if (matSizeN <= MAX_BLOCK_SIZE) {
        blockSize = matSizeN;
        blockNum = 1;
    } else {
        blockSize = MAX_BLOCK_SIZE;
        blockNum = (matSizeN + blockSize - 1) / blockSize;
    }

    auto compileInfo = reinterpret_cast<const CholeskyCompileInfo*>(tilingContext->GetCompileInfo());
    if (compileInfo == nullptr) {
        OP_LOGE(tilingContext->GetNodeName(), "Get compile info failed.");
        return ge::GRAPH_FAILED;
    }
    uint32_t coreNumPlatForm = compileInfo->coreNum;
    // A single matrix assigns one core per panel block; batches assign one core per matrix.
    needCoreNum = matrixNumCount == 1 ? (coreNumPlatForm < blockNum ? coreNumPlatForm : blockNum) :
                                        (coreNumPlatForm < matrixNumCount ? coreNumPlatForm : matrixNumCount);
    // A legal empty batch (matrixNumCount == 0, e.g. (0, 4, 4) / (2, 0, 4, 4)) still requires a
    // non-zero block dimension at runtime, mirroring segsum/exp_segsum_grad in this repository.
    // With matrixNumCount_ == 0 the kernel batch loop body is never entered
    // (offsetPrefix(0) < matrixNumCount_(0) is false), so no computation and no GM access happen,
    // matching the aclnn first-phase IsEmpty() early-return contract of dispatching nothing.
    if (needCoreNum == 0) {
        needCoreNum = 1;
        OP_LOGW(tilingContext->GetNodeName(), "Empty batch: matrixNumCount is 0, set blockDim to 1.");
    }

    size_t* currentWorkSpace = tilingContext->GetWorkspaceSizes(1);
    if (currentWorkSpace == nullptr) {
        OP_LOGE(tilingContext->GetNodeName(), "Get workspace sizes failed.");
        return ge::GRAPH_FAILED;
    }
    currentWorkSpace[0] = WS_SYS_SIZE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CholeskyTilingArch22::RunBigKernelTiling()
{
    tilingContext->SetBlockDim(needCoreNum);
    auto result = tilingContext->SetLocalMemorySize(LOCAL_MEMORY_SIZE);
    OP_CHECK_IF(result != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext, "SetLocalMemorySize failed"),
                return ge::GRAPH_FAILED);
    ASCENDC_TPL_SEL_PARAM(tilingContext, upper);
    auto rawTilingData = tilingContext->GetRawTilingData();
    if (rawTilingData == nullptr) {
        OP_LOGE(tilingContext->GetNodeName(), "Get raw tiling data failed.");
        return ge::GRAPH_FAILED;
    }
    rawTilingData->SetDataSize(tilingData.GetDataSize());

    tilingData.set_matrixNumCount(matrixNumCount);
    tilingData.set_matSizeN(matSizeN);
    tilingData.set_blockSize(blockSize);
    tilingData.set_blockNum(blockNum);

    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());

    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

void CholeskyTilingArch22::PrintTilingData()
{
    OP_LOGD(tilingContext, "matSizeN: %u", matSizeN);
    OP_LOGD(tilingContext, "matrixNumCount: %lu", matrixNumCount);
    OP_LOGD(tilingContext, "blockSize: %u", blockSize);
    OP_LOGD(tilingContext, "blockNum: %u", blockNum);
}

static ge::graphStatus CholeskyTilingFuncArch22(gert::TilingContext* context)
{
    CholeskyTilingArch22 tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus TilingPrepareForCholeskyArch22(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<CholeskyCompileInfo>();
    if (compileInfo == nullptr) {
        OP_LOGE(context->GetNodeName(), "Get compiled info failed.");
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();

    OP_CHECK_IF(
        (compileInfo->coreNum <= 0),
        OP_LOGE(context->GetNodeName(), "Cholesky GetHardwareInfo Failed, vectorCoreNum: %u", compileInfo->coreNum),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Cholesky)
    .Tiling(CholeskyTilingFuncArch22)
    .TilingParse<CholeskyCompileInfo>(TilingPrepareForCholeskyArch22);
} // namespace optiling
