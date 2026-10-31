/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "op_common/op_host/tiling_base_util.h"
#include "platform/platform_ascendc.h"
#include "util/const_util.h"
#include <algorithm>
#include <limits>
#include "../../op_kernel/arch35/dense_bincount_tiling_data.h"
#include "../../op_kernel/arch35/dense_bincount_tiling_key.h"
namespace optiling {
constexpr int64_t IDX_INPUT = 0;
constexpr int64_t IDX_SIZE = 1;
constexpr int64_t IDX_WEIGHTS = 2;
constexpr int64_t IDX_OUTPUT = 0;
constexpr int64_t ATTR_BINARY = 0;
constexpr int64_t MAX_OUTPUT_ELEMENTS = std::numeric_limits<int64_t>::max() / sizeof(float);
constexpr int64_t PER_CORE_MIN_ELEMENTS = 1024;
constexpr int64_t PRECISE_WEIGHTED_MAX_VALUES = 1024;
constexpr int64_t PRECISE_WEIGHTED_MAX_OUTPUTS = 64;
constexpr int64_t PRECISE_WEIGHTED_MAX_WORK = 4096;
constexpr int64_t MAX_PRIVATE_HIST_TILES = 8;
constexpr int64_t GM_ATOMIC_CONTENTION_FACTOR = 200;
constexpr uint64_t DCACHE_SIZE = 128U * 1024U;
constexpr uint64_t UB_ALIGN_BYTES = 32U;
constexpr int64_t PRIVATE_WRITEBACK_FACTOR = 8;
constexpr uint32_t PRIVATE_HIST_NONE = 0U;
constexpr uint32_t PRIVATE_HIST_FULL = 1U;
constexpr uint32_t PRIVATE_HIST_ROW = 2U;
constexpr uint32_t PRIVATE_HIST_TILED = 3U;
constexpr uint32_t DETERMINISTIC_NONE = 0U;
constexpr uint32_t DETERMINISTIC_PARTIAL = 1U;
constexpr uint32_t DETERMINISTIC_HISTOGRAM = 2U;
constexpr uint64_t DETERMINISTIC_WORKSPACE_LIMIT = 16U * 1024U * 1024U;
constexpr uint64_t DETERMINISTIC_PARTIAL_BYTES = 64U;
constexpr uint64_t DETERMINISTIC_HISTOGRAM_LIMIT = 64U * 1024U * 1024U;
constexpr uint64_t DETERMINISTIC_HISTOGRAM_SLOT_BYTES = 128U;
constexpr uint64_t DETERMINISTIC_PARTIAL_MAX_WORK = 16U * 1024U * 1024U;
constexpr uint32_t DETERMINISTIC_ZERO_BUFFER_BYTES = 32U * 1024U;
constexpr int64_t DETERMINISTIC_PARTITIONS_PER_CORE = 16;
constexpr int64_t DETERMINISTIC_VALUES_PER_PARTITION = 256;
struct DenseBincountCompileInfo {};

static int64_t SafeCeilDiv(int64_t dividend, int64_t divisor)
{
    return dividend / divisor + static_cast<int64_t>(dividend % divisor != 0);
}

static int64_t GetRequiredCoreNum(int64_t workload, int64_t maxCoreNum)
{
    if (workload <= 0) {
        return 1;
    }
    int64_t perCoreElements = std::max(SafeCeilDiv(workload, maxCoreNum), PER_CORE_MIN_ELEMENTS);
    return std::min(maxCoreNum, SafeCeilDiv(workload, perCoreElements));
}

static bool FitsInLocalMemory(int64_t elements, uint64_t localMemorySize)
{
    if (elements <= 0) {
        return false;
    }
    const uint64_t histogramBytes = static_cast<uint64_t>(elements) * sizeof(float);
    const uint64_t alignedBytes = ((histogramBytes + UB_ALIGN_BYTES - 1U) / UB_ALIGN_BYTES) * UB_ALIGN_BYTES;
    return alignedBytes <= localMemorySize;
}

static bool IsPrivateWriteProfitable(int64_t elements, int64_t usedCoreNum, int64_t numValues)
{
    if (elements <= 0 || usedCoreNum <= 0 || elements > std::numeric_limits<int64_t>::max() / usedCoreNum) {
        return false;
    }
    return SafeCeilDiv(elements * usedCoreNum, PRIVATE_WRITEBACK_FACTOR) <= numValues;
}

static void ConfigurePrivateHistogram(int64_t rank, int64_t outputElements, int64_t numValues, int64_t usedCoreNum,
                                      uint64_t localMemorySize, bool useRowPrivate, int64_t rowHistElems,
                                      DenseBincountTilingData* tiling)
{
    tiling->privateHistElems = 0U;
    tiling->privateHistMode = PRIVATE_HIST_NONE;
    if (outputElements <= 0 || numValues <= 0) {
        return;
    }
    if (FitsInLocalMemory(outputElements, localMemorySize) &&
        IsPrivateWriteProfitable(outputElements, usedCoreNum, numValues)) {
        tiling->privateHistElems = static_cast<uint32_t>(outputElements);
        tiling->privateHistMode = PRIVATE_HIST_FULL;
        return;
    }
    if (useRowPrivate && FitsInLocalMemory(rowHistElems, localMemorySize) &&
        IsPrivateWriteProfitable(rowHistElems, usedCoreNum, numValues)) {
        tiling->privateHistElems = static_cast<uint32_t>(rowHistElems);
        tiling->privateHistMode = PRIVATE_HIST_ROW;
        return;
    }
    const int64_t tileElements = static_cast<int64_t>(localMemorySize / sizeof(float) / 8U * 8U);
    if (rank == 1 && tileElements > 0 && outputElements > tileElements &&
        SafeCeilDiv(outputElements, tileElements) <= MAX_PRIVATE_HIST_TILES &&
        numValues > outputElements / GM_ATOMIC_CONTENTION_FACTOR) {
        tiling->privateHistElems = static_cast<uint32_t>(tileElements);
        tiling->privateHistMode = PRIVATE_HIST_TILED;
    }
}

static bool UsePreciseWeightedPath(int64_t numValues, int64_t outputElements)
{
    // The output-owner path scans the input once for every output element. Bound
    // both axes and their product so the automatic precision fallback cannot
    // turn a normal launch into an unbounded quadratic workload.
    if (numValues <= 0 || numValues > PRECISE_WEIGHTED_MAX_VALUES || outputElements <= 0 ||
        outputElements > PRECISE_WEIGHTED_MAX_OUTPUTS) {
        return false;
    }
    return numValues <= PRECISE_WEIGHTED_MAX_WORK / outputElements;
}

static uint32_t GetDeterministicPartitions(int64_t numValues, int64_t outputElements, uint32_t coreNum)
{
    const int64_t perCore = SafeCeilDiv(numValues, static_cast<int64_t>(coreNum));
    const int64_t needed = std::max<int64_t>(1, SafeCeilDiv(perCore, DETERMINISTIC_VALUES_PER_PARTITION));
    const uint64_t capacity = DETERMINISTIC_WORKSPACE_LIMIT / DETERMINISTIC_PARTIAL_BYTES /
                              static_cast<uint64_t>(outputElements) / coreNum;
    const int64_t budget = std::max<int64_t>(1, static_cast<int64_t>(capacity));
    const int64_t base = outputElements < 64 ? SafeCeilDiv(1024, outputElements) : DETERMINISTIC_PARTITIONS_PER_CORE;
    return static_cast<uint32_t>(std::min({needed, budget, base}));
}

static bool UseDeterministicPartials(int64_t numValues, int64_t outputElements, uint32_t coreNum)
{
    if (numValues <= 0 || outputElements <= 0 || coreNum == 0U ||
        static_cast<uint64_t>(outputElements) > DETERMINISTIC_WORKSPACE_LIMIT / DETERMINISTIC_PARTIAL_BYTES / coreNum) {
        return false;
    }
    return static_cast<uint64_t>(numValues) <= DETERMINISTIC_PARTIAL_MAX_WORK / static_cast<uint64_t>(outputElements);
}

static bool UseDeterministicHistogram(int64_t outputElements)
{
    return outputElements > 0 &&
           static_cast<uint64_t>(outputElements) <= DETERMINISTIC_HISTOGRAM_LIMIT / DETERMINISTIC_HISTOGRAM_SLOT_BYTES;
}

static bool ValidateDenseBincountInputs(gert::TilingContext* context)
{
    auto inputShape = context->GetInputShape(IDX_INPUT);
    auto sizeShape = context->GetInputShape(IDX_SIZE);
    auto weightsShape = context->GetInputShape(IDX_WEIGHTS);
    auto inputDesc = context->GetInputDesc(IDX_INPUT);
    auto sizeDesc = context->GetInputDesc(IDX_SIZE);
    auto weightsDesc = context->GetInputDesc(IDX_WEIGHTS);
    auto outputDesc = context->GetOutputDesc(IDX_OUTPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, sizeShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightsShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, sizeDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
    const auto rank = inputShape->GetStorageShape().GetDimNum();
    const auto& sizeStorageShape = Ops::Base::EnsureNotScalar(sizeShape->GetStorageShape());
    const auto weightsRank = weightsShape->GetStorageShape().GetDimNum();
    const bool validDtype = inputDesc->GetDataType() == ge::DT_INT32 || inputDesc->GetDataType() == ge::DT_INT64;
    const bool validWeights = weightsDesc->GetDataType() == ge::DT_FLOAT;
    const bool validSizeShape = sizeStorageShape.GetDimNum() == 1 && sizeStorageShape.GetDim(0) == 1;
    if (rank < 1 || rank > 2 || !validSizeShape || weightsRank > 2 || !validDtype ||
        sizeDesc->GetDataType() != inputDesc->GetDataType() || !validWeights ||
        outputDesc->GetDataType() != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "DenseBincount", "invalid",
                                              "dtype/rank/shape contract failed");
        return false;
    }
    return true;
}

static bool GetDenseBincountSize(gert::TilingContext* context, int64_t& size)
{
    if (Ops::Base::GetConstInt(context, IDX_SIZE, size) && size >= 0) {
        return true;
    }
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "size", "unknown",
                                          "size must be a non-negative constant");
    return false;
}

static bool SetDenseBincountTilingData(gert::TilingContext* context, const gert::Shape& shape,
                                       const gert::Shape& weightsShape, int64_t size, int64_t& weightsNum)
{
    const int64_t rank = shape.GetDimNum();
    const int64_t rows = rank == 1 ? 1 : shape.GetDim(0);
    const int64_t cols = rank == 1 ? shape.GetDim(0) : shape.GetDim(1);
    const int64_t numValues = shape.GetShapeSize();
    weightsNum = weightsShape.GetShapeSize();
    if (rows < 0 || cols < 0 || numValues < 0 || weightsNum < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "shape", "unknown",
                                              "runtime tiling requires concrete non-negative dimensions");
        return false;
    }
    if (rows > 0 && size > MAX_OUTPUT_ELEMENTS / rows) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "output.numel", "overflow",
                                              "output element count exceeds the int64 FLOAT32 storage limit");
        return false;
    }
    if (weightsNum != 0 && weightsNum != numValues) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "weights.numel", std::to_string(weightsNum),
                                              "weights element count must match input");
        return false;
    }
    auto* tiling = context->GetTilingData<DenseBincountTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->numValues = numValues;
    tiling->inputRows = rows;
    tiling->inputCols = cols;
    tiling->size = size;
    return true;
}

static ge::graphStatus ConfigureDenseBincountSchedule(gert::TilingContext* context,
                                                      platform_ascendc::PlatformAscendC& platform, int64_t rank,
                                                      const gert::Shape& shape, bool binary, bool ordered,
                                                      bool deterministic, DenseBincountTilingData* tiling)
{
    const int64_t outputElements = rank == 1 ? tiling->size : shape.GetDim(0) * tiling->size;
    const int64_t maxCoreNum = static_cast<int64_t>(platform.GetCoreNumAiv());
    if (maxCoreNum <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "AIV core count", std::to_string(maxCoreNum),
                                              "platform AIV core count must be positive");
        return ge::GRAPH_FAILED;
    }
    const int64_t workload = ordered ? outputElements : std::max(tiling->numValues, outputElements);
    const int64_t usedCoreNum = GetRequiredCoreNum(workload, maxCoreNum);
    uint64_t ubSize = 0U;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize <= DCACHE_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "UB size", std::to_string(ubSize),
                                              "UB must exceed the SIMT DCache reservation");
        return ge::GRAPH_FAILED;
    }
    const uint64_t localMemorySize = ubSize - DCACHE_SIZE;
    tiling->usedCoreNum = static_cast<uint32_t>(usedCoreNum);
    tiling->deterministicMode = DETERMINISTIC_NONE;
    tiling->deterministicPartitions = 0U;
    tiling->deterministicBufferBytes = 0U;
    if (deterministic) {
        if (UseDeterministicPartials(tiling->numValues, outputElements, tiling->usedCoreNum)) {
            tiling->deterministicMode = DETERMINISTIC_PARTIAL;
            tiling->deterministicPartitions = GetDeterministicPartitions(tiling->numValues, outputElements,
                                                                         tiling->usedCoreNum);
        } else if (UseDeterministicHistogram(outputElements)) {
            if (localMemorySize < DETERMINISTIC_ZERO_BUFFER_BYTES) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "local UB",
                                                      std::to_string(localMemorySize),
                                                      "deterministic histogram requires a 32 KiB zero buffer");
                return ge::GRAPH_FAILED;
            }
            tiling->deterministicMode = DETERMINISTIC_HISTOGRAM;
            tiling->deterministicBufferBytes = DETERMINISTIC_ZERO_BUFFER_BYTES;
        } else {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context->GetNodeName(), "output.numel", std::to_string(outputElements),
                "deterministic weighted accumulation exceeds the bounded workspace policies");
            return ge::GRAPH_FAILED;
        }
    }
    const bool useRowPrivate = rank == 2 && binary && tiling->inputRows > 1 && tiling->inputRows <= usedCoreNum;
    if (ordered || deterministic) {
        tiling->privateHistElems = 0U;
        tiling->privateHistMode = PRIVATE_HIST_NONE;
    } else {
        ConfigurePrivateHistogram(rank, outputElements, tiling->numValues, usedCoreNum, localMemorySize, useRowPrivate,
                                  tiling->size, tiling);
    }
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    // The production Host API accepts these platform-derived values. This
    // branch only protects an injected API failure and is not constructible by
    // TilingContextFaker.
    if (context->SetScheduleMode(1) != ge::GRAPH_SUCCESS ||
        context->SetDynUBufSize(static_cast<uint32_t>(localMemorySize)) != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "failed to configure synchronized SIMT scheduling or dynamic UB");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetDenseBincountWorkspace(gert::TilingContext* context,
                                                 platform_ascendc::PlatformAscendC& platform, int64_t outputElements,
                                                 const DenseBincountTilingData& tiling)
{
    uint64_t userWorkspace = 0U;
    if (tiling.deterministicMode == DETERMINISTIC_PARTIAL) {
        const uint64_t coreNum = tiling.usedCoreNum;
        const uint64_t outputs = static_cast<uint64_t>(outputElements);
        const uint64_t partitions = tiling.deterministicPartitions;
        constexpr uint64_t maxValue = std::numeric_limits<uint64_t>::max();
        // The deterministic policy caps the final product at 16 MiB. Keep the
        // checks local so future policy changes cannot silently overflow.
        if (outputs != 0U && coreNum > maxValue / outputs) {
            OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "deterministic partial entry count overflows uint64");
            return ge::GRAPH_FAILED;
        }
        uint64_t entries = coreNum * outputs;
        if (partitions != 0U && entries > maxValue / partitions) {
            OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "deterministic partial partition count overflows uint64");
            return ge::GRAPH_FAILED;
        }
        entries *= partitions;
        if (entries > maxValue / DETERMINISTIC_PARTIAL_BYTES) {
            OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "deterministic partial workspace overflows uint64");
            return ge::GRAPH_FAILED;
        }
        userWorkspace = entries * DETERMINISTIC_PARTIAL_BYTES;
    } else if (tiling.deterministicMode == DETERMINISTIC_HISTOGRAM) {
        userWorkspace = static_cast<uint64_t>(outputElements) * DETERMINISTIC_HISTOGRAM_SLOT_BYTES;
    }
    const uint64_t systemWorkspace = platform.GetLibApiWorkSpaceSize();
    // Current policies cap user workspace at 64 MiB; this remains a defensive
    // guard for a platform reporting an invalid system workspace size.
    if (userWorkspace > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        systemWorkspace > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) - userWorkspace) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "system and user workspace addition overflows size_t");
        return ge::GRAPH_FAILED;
    }
    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = static_cast<size_t>(systemWorkspace + userWorkspace);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DenseBincountTilingFunc(gert::TilingContext* context)
{
    auto* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    auto inputShape = context->GetInputShape(IDX_INPUT);
    if (!ValidateDenseBincountInputs(context))
        return ge::GRAPH_FAILED;
    auto weightsShape = context->GetInputShape(IDX_WEIGHTS);
    const auto& shape = inputShape->GetStorageShape();
    int64_t rank = shape.GetDimNum();
    int64_t size = 0;
    if (!GetDenseBincountSize(context, size))
        return ge::GRAPH_FAILED;
    int64_t weightsNum = 0;
    if (!SetDenseBincountTilingData(context, shape, weightsShape->GetStorageShape(), size, weightsNum))
        return ge::GRAPH_FAILED;
    auto* tiling = context->GetTilingData<DenseBincountTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    bool binary = false;
    if (const auto* attrs = context->GetAttrs()) {
        if (const bool* v = attrs->GetBool(ATTR_BINARY))
            binary = *v;
    }
    const int64_t outputElements = rank == 1 ? size : shape.GetDim(0) * size;
    const bool weighted = weightsNum > 0 && !binary;
    // The ordered kernel scans all input values for each output bin. Keep this
    // bounded; deterministic requests outside that range use bounded workspace
    // reductions instead of falling back to unordered FP32 atomics.
    const bool ordered = weighted && UsePreciseWeightedPath(tiling->numValues, outputElements);
    const bool deterministic = weighted && !ordered && outputElements > 0 && context->GetDeterministic() == 1;
    if (ConfigureDenseBincountSchedule(context, platform, rank, shape, binary, ordered, deterministic, tiling) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (SetDenseBincountWorkspace(context, platform, outputElements, *tiling) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    int32_t hasWeights = weightsNum > 0 ? 1 : 0;
    int32_t is1D = rank == 1 ? 1 : 0;
    uint32_t mode = static_cast<uint32_t>((is1D << 2) | ((binary ? 1 : 0) << 1) | hasWeights);
    if (ordered) {
        mode = rank == 1 ? SCH_MODE_1D_COUNT_WEIGHT_ORDERED : SCH_MODE_ND_COUNT_WEIGHT_ORDERED;
    } else if (deterministic) {
        mode = rank == 1 ? SCH_MODE_1D_COUNT_WEIGHT_DETERMINISTIC : SCH_MODE_ND_COUNT_WEIGHT_DETERMINISTIC;
    }
    context->SetTilingKey(GET_TPL_TILING_KEY(mode));
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus DenseBincountParse(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(DenseBincount)
    .Tiling(DenseBincountTilingFunc)
    .TilingParse<DenseBincountCompileInfo>(DenseBincountParse);
} // namespace optiling
