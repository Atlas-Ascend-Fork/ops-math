/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tile_with_axis_tiling.cpp
 * \brief TileWithAxis Host 侧 Tiling 实现（arch35, DAV_3510）
 *
 * 设计依据: DESIGN.md v1.6 3.5 节
 *
 * 两步法 Tiling:
 *   第一步: UB 切分 -- 从候选轴中选择 ubAxis + ubFactor
 *   第二步: 多核切分 -- totalCount 个 Block 均分给 coreNum 个核
 *
 * 3D 模型 (v1.6 折叠 axisDim 进 rowLength):
 *   inShape  = [outerDim, 1, rowLength]
 *   outShape = [outerDim, tiles, rowLength]
 *   其中 rowLength = axisDim * innerDim
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch35/tile_with_axis_tiling_data.h"
#include "../../op_kernel/arch35/tile_with_axis_tiling_key.h"
#include <algorithm>

namespace optiling {

using Ops::Base::CeilDiv;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr size_t WORKSPACE_NUM = 1;
constexpr int64_t MIN_PER_CORE_ELEMENTS = 4096;
constexpr double TARGET_CORE_RATIO = 0.8;

static constexpr int32_t TILING_KEY_ELEMENT_COUNT = 2;

// ============================================================================
// Helper: Get platform info (ubSize, coreNum)
// ============================================================================
// ============================================================================
// Helper: guard an int64 multiplication against signed overflow
// ============================================================================
static ge::graphStatus CheckMulNoOverflow(gert::TilingContext* context, int64_t a, int64_t b, const char* product_name)
{
    if (a < 0 || b < 0) {
        OP_LOGE(context, "%s operand is negative: %ld * %ld", product_name, a, b);
        return ge::GRAPH_FAILED;
    }
    if (a != 0 && b != 0 && a > INT64_MAX / b) {
        OP_LOGE(context, "%s overflows int64: %ld * %ld", product_name, a, b);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(*ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Helper: Compute workspace size
// ============================================================================
static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Helper: Handle scalar (rank=0) input
// ============================================================================
static void LogTilingData(gert::TilingContext* context, const TileWithAxisTilingData* tiling)
{
    OP_LOGI(context->GetNodeName(),
            "TileWithAxis TilingData: inShape=[%ld,%ld,%ld] outShape=[%ld,%ld,%ld] totalCount=%lu "
            "perCoreCount=%lu ubAxis=%u ubFactor=%u bufferSize=%u tiles=%ld rowLength=%ld",
            tiling->inShape[0], tiling->inShape[1], tiling->inShape[2], tiling->outShape[0], tiling->outShape[1],
            tiling->outShape[2], tiling->totalCount, tiling->perCoreCount, tiling->ubAxis, tiling->ubFactor,
            tiling->bufferSize, tiling->tiles, tiling->rowLength);
}

static ge::graphStatus HandleScalarInput(gert::TilingContext* context, int64_t tiles, uint64_t ubSize,
                                         int64_t dtypeSize, int64_t coreNum)
{
    TileWithAxisTilingData* tiling = context->GetTilingData<TileWithAxisTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(TileWithAxisTilingData), 0, sizeof(TileWithAxisTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    tiling->inShape[0] = 1;
    tiling->inShape[1] = 1;
    tiling->inShape[2] = 1;
    tiling->outShape[0] = 1;
    tiling->outShape[1] = tiles;
    tiling->outShape[2] = 1;
    tiling->bufferSize = ubSize / 2;
    tiling->tiles = tiles;
    tiling->rowLength = 1;

    // 标量输出共 tiles 个元素；单缓冲容量 bufferSize/dtypeSize 个元素。
    // tiles 超过单缓冲容量时沿 tiles 维分块（ubAxis=1: NDDMA dim1 源 stride=0 广播读，见 kernel
    // InitNddmaParams），每块不超过单缓冲，消除原实现的 UB 越界写。
    int64_t bufferSizeElements = static_cast<int64_t>(tiling->bufferSize) / dtypeSize;
    OP_CHECK_IF(bufferSizeElements <= 0,
                OP_LOGE(context, "UB half buffer %u bytes too small for dtype size %ld", tiling->bufferSize, dtypeSize),
                return ge::GRAPH_FAILED);
    if (tiles <= bufferSizeElements) {
        tiling->totalCount = 1;
        tiling->perCoreCount = 1;
        tiling->ubAxis = 0;
        tiling->ubFactor = 1;
        LogTilingData(context, tiling);
        context->SetBlockDim(1);
        ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(0));
    } else {
        uint32_t ubFactor = static_cast<uint32_t>(bufferSizeElements);
        uint64_t totalCount = static_cast<uint64_t>(Ops::Base::CeilDiv(tiles, static_cast<int64_t>(ubFactor)));
        uint64_t perCoreCount = static_cast<uint64_t>(Ops::Base::CeilDiv(static_cast<int64_t>(totalCount), coreNum));
        int64_t realCoreNum = Ops::Base::CeilDiv(static_cast<int64_t>(totalCount), static_cast<int64_t>(perCoreCount));
        tiling->totalCount = totalCount;
        tiling->perCoreCount = perCoreCount;
        tiling->ubAxis = 1;
        tiling->ubFactor = ubFactor;
        LogTilingData(context, tiling);
        context->SetBlockDim(realCoreNum);
        ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(1));
    }
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Helper: Validate and normalize axis to [0, rank)
// ============================================================================
static ge::graphStatus NormalizeAxis(gert::TilingContext* context, int64_t& axis, int64_t rank)
{
    if (axis < -rank || axis >= rank) {
        OP_LOGE(context, "axis=%ld out of range [-%ld, %ld)", axis, rank, rank);
        return ge::GRAPH_FAILED;
    }
    if (axis < 0) {
        axis = axis + rank;
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// Helper: Compute max blocks on a given axis
// ============================================================================
static int64_t MaxBlocksOnAxis(uint8_t ax, int64_t outerDim, int64_t tiles, int64_t rowLength)
{
    if (ax == 0)
        return outerDim;
    if (ax == 1) {
        // 仅 rowLength>0 时可达（此时 outerDim*tiles 已被入口总乘积检查约束）
        return (rowLength == 0) ? 0 : outerDim * tiles;
    }
    // ax == 2
    return (rowLength == 0) ? 0 : outerDim * tiles * rowLength;
}

// ============================================================================
// Helper: Compute ubFactor from axis + target block count
// ============================================================================
static uint32_t ComputeUbFactor(uint8_t ax, int64_t target, int64_t outerDim, int64_t tiles, int64_t rowLength)
{
    if (ax == 0) {
        return static_cast<uint32_t>(std::max(static_cast<int64_t>(1), CeilDiv(outerDim, target)));
    }
    if (ax == 1) {
        int64_t bpo = CeilDiv(target, outerDim);
        return static_cast<uint32_t>(std::max(static_cast<int64_t>(1), CeilDiv(tiles, bpo)));
    }
    // ax == 2；rowLength==0 为空张量，无需按块计算 ubFactor
    if (rowLength == 0) {
        return 1;
    }
    int64_t bpot = CeilDiv(target, outerDim * tiles);
    return static_cast<uint32_t>(std::max(static_cast<int64_t>(1), CeilDiv(rowLength, bpot)));
}

// ============================================================================
// Helper: Compute totalCount from axis + ubFactor
// ============================================================================
static uint64_t ComputeTotalCount(uint8_t ax, int64_t uf, int64_t outerDim, int64_t tiles, int64_t rowLength)
{
    if (ax == 0)
        return static_cast<uint64_t>(CeilDiv(outerDim, uf));
    if (ax == 1)
        return static_cast<uint64_t>(outerDim * CeilDiv(tiles, uf));
    // ax == 2；rowLength==0 为空张量，块数为 0
    return (rowLength == 0) ? 0U : static_cast<uint64_t>(outerDim * tiles * CeilDiv(rowLength, uf));
}

// ============================================================================
// Step 2: Select initial UB axis by priority (0 > 1 > 2)
// ============================================================================
static void SelectInitialUbAxis(int64_t ubFactor0, int64_t ubFactor1, int64_t ubFactor2, int64_t outerDim,
                                int64_t tiles, int64_t rowLength, uint8_t& ubAxis, uint32_t& ubFactor,
                                uint64_t& totalCount)
{
    if (ubFactor0 >= 1) {
        ubAxis = 0;
        ubFactor = static_cast<uint32_t>(ubFactor0);
        totalCount = static_cast<uint64_t>(CeilDiv(outerDim, ubFactor0));
    } else if (ubFactor1 >= 1) {
        ubAxis = 1;
        ubFactor = static_cast<uint32_t>(ubFactor1);
        totalCount = static_cast<uint64_t>(outerDim * CeilDiv(tiles, ubFactor1));
    } else {
        ubAxis = 2;
        ubFactor = static_cast<uint32_t>(std::max(ubFactor2, static_cast<int64_t>(1)));
        // rowLength==0 为空张量，块数为 0
        totalCount = (rowLength == 0) ? 0U : static_cast<uint64_t>(outerDim * tiles * CeilDiv(rowLength, ubFactor2));
    }
}

// ============================================================================
// Step 2.5: Core-aware ubFactor adjustment + cross-axis fallback
// ============================================================================
static void AdjustForCoreUtilization(uint8_t& ubAxis, uint32_t& ubFactor, uint64_t& totalCount, int64_t ubFactor0,
                                     int64_t ubFactor1, int64_t ubFactor2, int64_t outerDim, int64_t tiles,
                                     int64_t rowLength, int64_t coreNum)
{
    int64_t totalOutElements = outerDim * tiles * rowLength;
    int64_t thresholdBlocks = static_cast<int64_t>(static_cast<double>(coreNum) * TARGET_CORE_RATIO);

    if (static_cast<int64_t>(totalCount) >= thresholdBlocks) {
        return;
    }

    int64_t maxBlocksByData = std::max(totalOutElements / MIN_PER_CORE_ELEMENTS, static_cast<int64_t>(1));

    // 2.5a: 同轴降 ubFactor 增加 block 数
    int64_t maxBlocksOnCurAxis = MaxBlocksOnAxis(ubAxis, outerDim, tiles, rowLength);
    int64_t targetBlocks = std::min({thresholdBlocks, maxBlocksByData, maxBlocksOnCurAxis});

    if (targetBlocks > static_cast<int64_t>(totalCount)) {
        ubFactor = ComputeUbFactor(ubAxis, targetBlocks, outerDim, tiles, rowLength);
        totalCount = ComputeTotalCount(ubAxis, static_cast<int64_t>(ubFactor), outerDim, tiles, rowLength);
    }

    // 2.5b: 同轴调节后仍不足 → 跨轴回退
    if (static_cast<int64_t>(totalCount) >= thresholdBlocks) {
        return;
    }

    uint8_t bestAxis = ubAxis;
    uint64_t bestCount = totalCount;
    uint32_t bestUf = ubFactor;

    const int64_t origUbFactors[] = {ubFactor0, ubFactor1, ubFactor2};
    for (uint8_t cand = 0; cand < 3; cand++) {
        if (cand == ubAxis || origUbFactors[cand] < 1)
            continue;

        int64_t maxOnCand = MaxBlocksOnAxis(cand, outerDim, tiles, rowLength);
        int64_t candTarget = std::min({thresholdBlocks, maxBlocksByData, maxOnCand, coreNum});
        uint32_t candUf = ComputeUbFactor(cand, candTarget, outerDim, tiles, rowLength);
        uint64_t candCount = ComputeTotalCount(cand, static_cast<int64_t>(candUf), outerDim, tiles, rowLength);

        if (candCount > static_cast<uint64_t>(coreNum))
            continue;

        if (candCount > bestCount) {
            bestAxis = cand;
            bestCount = candCount;
            bestUf = candUf;
        }
    }

    if (bestAxis != ubAxis) {
        ubAxis = bestAxis;
        ubFactor = bestUf;
        totalCount = bestCount;
    }
}

// ============================================================================
// Tiling 分发入口
// ============================================================================
static ge::graphStatus TileWithAxisTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin the tiling process for Arch35 architecture");
    // 1. 获取平台运行信息
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2. 获取输入信息
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShape = inputX->GetStorageShape();

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto dataType = inputDesc->GetDataType();

    constexpr size_t ATTR_AXIS_IDX = 0;
    constexpr size_t ATTR_TILES_IDX = 1;
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* axis_ptr = attrs->GetAttrPointer<int64_t>(ATTR_AXIS_IDX);
    const int64_t* tiles_ptr = attrs->GetAttrPointer<int64_t>(ATTR_TILES_IDX);
    // tiles 为必选属性（proto REQUIRED_ATTR / def REQUIRED / README）：缺失时拒绝，不静默回退
    OP_CHECK_IF(tiles_ptr == nullptr, OP_LOGE(context, "tiles is required but missing"), return ge::GRAPH_FAILED);
    int64_t tiles = *tiles_ptr;
    // axis 默认 1（与 TBE/AICPU 缺省一致）；属性缺失时兜底 1
    int64_t axis = (axis_ptr != nullptr) ? *axis_ptr : 1;

    OP_CHECK_IF(tiles <= 0, OP_LOGE(context, "tiles must be > 0, got %ld", tiles), return ge::GRAPH_FAILED);

    int64_t rank = inputShape.GetDimNum();

    // 3. 标量输入特殊处理
    if (rank == 0) {
        int64_t scalarDtypeSize = static_cast<int64_t>(ge::GetSizeByDataType(dataType));
        OP_CHECK_IF(scalarDtypeSize <= 0,
                    OP_LOGE(context, "unsupported dtype, GetSizeByDataType returned %ld", scalarDtypeSize),
                    return ge::GRAPH_FAILED);
        return HandleScalarInput(context, tiles, ubSize, scalarDtypeSize, coreNum);
    }

    // 4. 校验并归一化 axis
    OP_CHECK_IF(NormalizeAxis(context, axis, rank) != ge::GRAPH_SUCCESS, OP_LOGE(context, "NormalizeAxis failed"),
                return ge::GRAPH_FAILED);

    // 5. 计算展平参数 (v1.6 折叠 axisDim 进 rowLength)
    int64_t outerDim = 1;
    for (int64_t i = 0; i < axis; i++) {
        int64_t dim = inputShape.GetDim(i);
        OP_CHECK_IF(CheckMulNoOverflow(context, outerDim, dim, "outerDim") != ge::GRAPH_SUCCESS,
                    OP_LOGE(context, "outerDim overflow"), return ge::GRAPH_FAILED);
        outerDim *= dim;
    }
    int64_t axisDim = inputShape.GetDim(axis);
    int64_t innerDim = 1;
    for (int64_t i = axis + 1; i < rank; i++) {
        int64_t dim = inputShape.GetDim(i);
        OP_CHECK_IF(CheckMulNoOverflow(context, innerDim, dim, "innerDim") != ge::GRAPH_SUCCESS,
                    OP_LOGE(context, "innerDim overflow"), return ge::GRAPH_FAILED);
        innerDim *= dim;
    }
    OP_CHECK_IF(CheckMulNoOverflow(context, axisDim, innerDim, "rowLength") != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "rowLength overflow"), return ge::GRAPH_FAILED);
    int64_t rowLength = axisDim * innerDim;

    // 6. TilingData 空间分配
    TileWithAxisTilingData* tiling = context->GetTilingData<TileWithAxisTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(TileWithAxisTilingData), 0, sizeof(TileWithAxisTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // 7. UB 切分
    int64_t dtypeSize = static_cast<int64_t>(ge::GetSizeByDataType(dataType));
    OP_CHECK_IF(dtypeSize <= 0, OP_LOGE(context, "unsupported dtype, GetSizeByDataType returned %ld", dtypeSize),
                return ge::GRAPH_FAILED);
    int64_t bufferSizeElements = (static_cast<int64_t>(ubSize) / 2) / dtypeSize;
    OP_CHECK_IF(CheckMulNoOverflow(context, tiles, rowLength, "rowOutElements") != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "rowOutElements overflow"), return ge::GRAPH_FAILED);
    int64_t rowOutElements = tiles * rowLength;
    // 总输出元素数 outerDim*tiles*rowLength：下游 AdjustForCoreUtilization 的 totalOutElements、
    // MaxBlocksOnAxis 与 ComputeTotalCount 均为这三个因子的子集/缩放，验证后不再溢出。
    OP_CHECK_IF(CheckMulNoOverflow(context, outerDim, rowOutElements, "totalOutElements") != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "totalOutElements overflow"), return ge::GRAPH_FAILED);

    int64_t ubFactor0 = (rowOutElements > 0) ? std::min(outerDim, bufferSizeElements / rowOutElements) : 0;
    int64_t ubFactor1 = (rowLength > 0) ? std::min(tiles, bufferSizeElements / rowLength) : 0;
    int64_t ubFactor2 = std::min(rowLength, bufferSizeElements);

    uint8_t ubAxis = 0;
    uint32_t ubFactor = 1;
    uint64_t totalCount = 0;

    SelectInitialUbAxis(ubFactor0, ubFactor1, ubFactor2, outerDim, tiles, rowLength, ubAxis, ubFactor, totalCount);

    AdjustForCoreUtilization(ubAxis, ubFactor, totalCount, ubFactor0, ubFactor1, ubFactor2, outerDim, tiles, rowLength,
                             coreNum);

    // 8. 多核切分（空张量兜底：totalCount==0 时 SetBlockDim(1)，kernel 空循环，见 REQUIREMENTS §3.5.2）
    uint64_t perCoreCount = 1;
    int64_t realCoreNum = 1;
    if (totalCount > 0) {
        perCoreCount = static_cast<uint64_t>(CeilDiv(static_cast<int64_t>(totalCount), coreNum));
        realCoreNum = CeilDiv(static_cast<int64_t>(totalCount), static_cast<int64_t>(perCoreCount));
    }

    // 9. 填充 TilingData
    tiling->inShape[0] = outerDim;
    tiling->inShape[1] = 1;
    tiling->inShape[2] = rowLength;
    tiling->outShape[0] = outerDim;
    tiling->outShape[1] = tiles;
    tiling->outShape[2] = rowLength;
    tiling->totalCount = totalCount;
    tiling->perCoreCount = perCoreCount;
    tiling->ubAxis = ubAxis;
    tiling->ubFactor = ubFactor;
    tiling->bufferSize = static_cast<uint32_t>(ubSize) / 2;
    tiling->tiles = tiles;
    tiling->rowLength = rowLength;
    LogTilingData(context, tiling);

    // 10. 设置 BlockDim + workspace + TilingKey
    context->SetBlockDim(realCoreNum);

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);

    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(ubAxis));

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// TilingParse
// ============================================================================
static ge::graphStatus TilingParseForTileWithAxis([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TileWithAxisCompileInfo {};

// ============================================================================
// Tiling 注册入口
// ============================================================================
IMPL_OP_OPTILING(TileWithAxis)
    .Tiling(TileWithAxisTilingFunc)
    .TilingParse<TileWithAxisCompileInfo>(TilingParseForTileWithAxis);

} // namespace optiling
