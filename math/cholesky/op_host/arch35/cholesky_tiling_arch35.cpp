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
#include <cstdio>
#include "op_common/op_host/util/math_util.h"
#include "../../op_kernel/arch35/cholesky_arch35_tiling_key.h"
#include "../../op_kernel/cholesky_shared_const.h"

namespace optiling {
constexpr uint32_t MINIMUM_DIMENSION = 2;
constexpr uint32_t MAXIMUM_DIMENSION = 8;
constexpr uint32_t UPPER_INDEX = 0;
// 系统/lib-API 保留区，必须与 kernel 侧 RESERVED_WORKSPACE(__NPU_ARCH__==3510) 一致；
// kernel 用 GetUserWorkspace() 跳过这一段后才访问算子私有的用户区
constexpr uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
// 用户区：[0, blockDim) 保留 + 面板分解缩放因子导出区
// （blockDim × CHOLESKY_PANEL_WIDTH 个 float，供并行阶段 A 免重算），按 32B 向上对齐
constexpr uint32_t WS_USER_ALIGN = 32U;
// Schur 更新用 512 列块：matRight=512*64*4=128KB、matA=16*512*4=32KB、matL/matLeft=4KB、
// matResult=2KB ⇒ 约 170KB，必须把申报的 UB 预算同步抬高（平台实测 ub_size=248KB）
constexpr uint32_t LOCAL_MEMORY_SIZE = 192U * 1024U;
constexpr uint32_t MAX_BLOCK_SIZE = 256;
constexpr int64_t MAX_MATRIX_SIZE = 8192;
// Cube MMAD Schur 路径的启用下界。单矩阵 M>=2048 时 Schur 更新（Σ L²/2 × 64 MAC）已占
// 并行段 >95%，才值得付出 mix launch 与每面板 3 次全核屏障的固定成本；M 必须是面板宽度
// 的整数倍（kernel 侧对角 tile 整块写出依赖该前提，见 arch35/cholesky_arch35.h 的 cube 段）。
constexpr int64_t CUBE_MIN_MATRIX_SIZE = 2048;
// 单矩阵多核启用阈值由 ../../op_kernel/cholesky_shared_const.h 提供（与 kernel 共享）：
// CHOLESKY_SINGLE_MATRIX_SINGLE_CORE_MAX(4) —— M 小于等于该值时多核 launch 与面板间同步的固定开销主导，
// 保持单核；阈值不超过 CHOLESKY_PANEL_WIDTH（64），
// 保证单核时面板路径仍为单面板完整分解。
// arch35 的 kernel 恒走协同面板路径（含单核）；CHOLESKY_BATCH_COOP_MIN_M 为未使用的历史阈值，
// M=96/97 不构成当前 host 调度或 kernel cooperative_ 的切换边界。

class CholeskyTilingArch35 {
public:
    explicit CholeskyTilingArch35(gert::TilingContext* context) : tilingContext(context) {};
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
    bool useCube_ = false;
    uint32_t blockSize = 0;
    uint32_t blockNum = 0;
};

ge::graphStatus CholeskyTilingArch35::Init()
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
        // 批量维负值含 GE 未知维标记（-1/-2）：static_cast<uint64_t> 会回绕成巨值，
        // 使 kernel 批量循环次数与 GM 偏移远超实际张量，必须在此显式拒绝
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
    // arch35 恒为协同面板路径（kernel 侧 cooperative_ 恒 true，两侧判据一致是 SyncAll 的 ABI 前提）。
    // 仅「单矩阵且 M<=4」保持单核（多核 launch 与面板间 SyncAll 的固定开销主导）；
    // 批量每核的分组宽度由 kernel 侧的 groupCores_=max(1,blockDim/matrixNumCount) 决定，host 不下发。
    const bool useCube = compileInfo->aicNum > 0 && (matrixNumCount == 1) &&
                         (matSizeN >= static_cast<uint32_t>(CUBE_MIN_MATRIX_SIZE)) &&
                         (matSizeN % CHOLESKY_PANEL_WIDTH == 0U);
    needCoreNum = matrixNumCount == 1 && matSizeN <= CHOLESKY_SINGLE_MATRIX_SINGLE_CORE_MAX ? 1 : coreNumPlatForm;
    // cube 键以 MIX(AIC:AIV=1:2) 下发：blockDim = AIC 数，AIV 实际为 2 倍
    useCube_ = useCube;
    if (useCube) {
        needCoreNum = compileInfo->aicNum;
    }
    // 合法空 batch（matrixNumCount==0，如 (0,4,4)/(2,0,4,4)）：运行时要求非零 blockDim，
    // 与仓内 segsum/exp_segsum_grad 同构处置。kernel 侧 matrixNumCount_==0 使批量循环体一次都不进入
    // （非协同分支 offsetPrefix(0) < matrixNumCount_(0) 为假，协同分支 for m < 0 不迭代），
    // 不产生计算与 GM 访问，与 aclnn 第一段 IsEmpty() 早退「不下发任何计算」的契约一致
    if (needCoreNum == 0) {
        needCoreNum = 1;
        OP_LOGW(tilingContext->GetNodeName(), "Empty batch: matrixNumCount is 0, set blockDim to 1.");
    }

    size_t* currentWorkSpace = tilingContext->GetWorkspaceSizes(1);
    if (currentWorkSpace == nullptr) {
        OP_LOGE(tilingContext->GetNodeName(), "Get workspace sizes failed.");
        return ge::GRAPH_FAILED;
    }
    // 申报 = 系统保留区(16MB) + 用户区。用户区偏移由 kernel 侧显式调用
    // GetUserWorkspace(workspace) 完成，不是由入口 stub 代劳。
    // 用户区按 AIV 核数申报（cube 路径下 AIV 数 = 2 * blockDim，kernel 侧的保留区/缩放因子区
    // 都按 AIV 的 blockDim_ 寻址），needCoreNum 为 uint32 核数，uint64 乘加不会溢出
    uint64_t wsCores = useCube ? 2ULL * static_cast<uint64_t>(needCoreNum) : static_cast<uint64_t>(needCoreNum);
    uint64_t voteBytes = wsCores * sizeof(float);
    // 与 kernel 侧一致：保留区先按 32B 对齐，缩放因子区紧随其后（kernel 用 AlignToSlot(blockDim) 定位）
    uint64_t voteAligned = (voteBytes + WS_USER_ALIGN - 1) / WS_USER_ALIGN * WS_USER_ALIGN;
    uint64_t scaleBytes = wsCores * CHOLESKY_PANEL_WIDTH * sizeof(float);
    // cube 路径追加 -Uᵀ / Uᵀ 两块 (CHOLESKY_PANEL_WIDTH × M) float（kernel 侧同一公式）
    uint64_t utBytes = useCube ? 2ULL * CHOLESKY_PANEL_WIDTH * static_cast<uint64_t>(matSizeN) * sizeof(float) : 0ULL;
    uint64_t userBytes = (voteAligned + scaleBytes + utBytes + WS_USER_ALIGN - 1) / WS_USER_ALIGN * WS_USER_ALIGN;
    currentWorkSpace[0] = static_cast<size_t>(static_cast<uint64_t>(WS_SYS_SIZE) + userBytes);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CholeskyTilingArch35::RunBigKernelTiling()
{
    tilingContext->SetBlockDim(needCoreNum);
    auto result = tilingContext->SetLocalMemorySize(LOCAL_MEMORY_SIZE);
    OP_CHECK_IF(result != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext, "SetLocalMemorySize failed"),
                return ge::GRAPH_FAILED);
    ASCENDC_TPL_SEL_PARAM(tilingContext, upper, useCube_);
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

    // The arch35 cooperative panel path uses SyncAll and requires batch scheduling.
    OP_CHECK_IF(tilingContext->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
                OP_LOGE(tilingContext, "SetScheduleMode failed"), return ge::GRAPH_FAILED);

    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

void CholeskyTilingArch35::PrintTilingData()
{
    OP_LOGD(tilingContext, "matSizeN: %u", matSizeN);
    OP_LOGD(tilingContext, "matrixNumCount: %lu", matrixNumCount);
    OP_LOGD(tilingContext, "blockSize: %u", blockSize);
    OP_LOGD(tilingContext, "blockNum: %u", blockNum);
}

static ge::graphStatus CholeskyTilingFuncArch35(gert::TilingContext* context)
{
    CholeskyTilingArch35 tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus TilingPrepareForCholeskyArch35(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<CholeskyCompileInfo>();
    if (compileInfo == nullptr) {
        OP_LOGE(context->GetNodeName(), "Get compiled info failed.");
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo->aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(
        (compileInfo->coreNum <= 0),
        OP_LOGE(context->GetNodeName(), "Cholesky GetHardwareInfo Failed, vectorCoreNum: %u", compileInfo->coreNum),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Cholesky)
    .Tiling(CholeskyTilingFuncArch35)
    .TilingParse<CholeskyCompileInfo>(TilingPrepareForCholeskyArch35);
} // namespace optiling
