/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sort_with_index_tiling_arch22.h"
#include "../../op_kernel/arch22/sort_with_index_constants.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace optiling {
namespace {
using namespace SortWithIndexA2;

// 限制并发基数排序行组使用的 GM 临时空间；该值仅用于调度，不是设备容量或算法正确性的限制。
constexpr uint64_t ROW_PARALLEL_WORKSPACE_LIMIT = 256ULL * 1024 * 1024;

// 910B 原生 Sort 在以下长度内仅使用 UB；改用 SingleRun，避免 GM 记录缓冲使 workspace 超过 Sort+Gather。
constexpr uint32_t NATIVE_SORT_FP16_UB_AXIS_LIMIT = 5600;
constexpr uint32_t NATIVE_SORT_BF16_UB_AXIS_LIMIT = 5088;
static_assert(NATIVE_SORT_FP16_UB_AXIS_LIMIT <= STREAM_SINGLE_RUN_AXIS_LIMIT &&
                  NATIVE_SORT_BF16_UB_AXIS_LIMIT <= STREAM_SINGLE_RUN_AXIS_LIMIT,
              "workspace-preferred rows must fit the SingleRun UB sorter");

struct RadixScheduleCost {
    static constexpr uint64_t ALIGNED_ELEMENTS_NS = 550; // 每 SIMD_ALIGN_ELEMENTS 项
    static constexpr uint64_t TILE_NS = 16400;
    static constexpr uint64_t ROW_NS = 13300;
    static constexpr uint64_t COOPERATION_NS = 8500;
    static constexpr uint64_t CORE_SUMMARY_NS = 600;
    static constexpr uint64_t WAVE_SUMMARY_NS = 550;
    // 预计收益落在模型误差范围内时，优先使用更少的核，减少摘要开销并保留较大的 tile。
    static constexpr uint32_t TOLERANCE_PERCENT = 3;
};

struct SortTileInfo {
    uint32_t rows = 0;
    uint32_t length = 0;
    uint32_t alignedLength = 0;
    uint32_t tileCount = 0;
    uint32_t coresPerRow = 0;
    uint32_t mergeGroupSpan = 0;
    uint32_t rowsPerCore = 1;
    uint32_t mergeBlockDim = 0;
    uint32_t availableCores = 0;
    uint32_t mode = MERGE_LOCAL_MODE;
    bool descending = false;
    ge::DataType dtype = ge::DT_UNDEFINED;
};

template <typename T>
T AlignUp(T value, uint32_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

ge::graphStatus GetInputInfo(gert::TilingContext* context, SortTileInfo& info)
{
    const auto* shape = context->GetInputShape(0);
    const auto* indexShape = context->GetInputShape(1);
    const auto* inputDesc = context->GetInputDesc(0);
    const auto* indexDesc = context->GetInputDesc(1);
    const auto* attrs = context->GetAttrs();
    if (shape == nullptr || indexShape == nullptr || inputDesc == nullptr || indexDesc == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto& dims = shape->GetStorageShape();
    const auto rank = dims.GetDimNum();
    if (rank == 0 || indexShape->GetStorageShape() != dims) {
        return ge::GRAPH_FAILED;
    }
    const auto* axis = attrs->GetAttrPointer<int64_t>(0);
    const auto* descending = attrs->GetAttrPointer<bool>(1);
    if (axis == nullptr || descending == nullptr || (*axis != -1 && *axis != static_cast<int64_t>(rank) - 1)) {
        return ge::GRAPH_FAILED;
    }
    info.dtype = inputDesc->GetDataType();
    if ((info.dtype != ge::DT_FLOAT16 && info.dtype != ge::DT_BF16) || indexDesc->GetDataType() != ge::DT_INT32) {
        return ge::GRAPH_FAILED;
    }
    const int64_t length = dims.GetDim(rank - 1);
    if (length <= 0 || length > INT32_MAX) {
        return ge::GRAPH_FAILED;
    }
    uint64_t rows = 1;
    for (size_t dim = 0; dim + 1 < rank; ++dim) {
        const auto size = dims.GetDim(dim);
        if (size <= 0 || rows > UINT32_MAX / static_cast<uint64_t>(size)) {
            return ge::GRAPH_FAILED;
        }
        rows *= size;
    }
    info.rows = static_cast<uint32_t>(rows);
    info.length = static_cast<uint32_t>(length);
    info.descending = *descending;
    return ge::GRAPH_SUCCESS;
}

uint32_t RadixFirstLength(const SortTileInfo& info)
{
    if (info.rows == 1)
        return RADIX_SINGLE_ROW_FIRST_LENGTH;
    if (info.rows <= RADIX_FEW_ROWS_LIMIT)
        return RADIX_FEW_ROWS_FIRST_LENGTH;
    return RADIX_MANY_ROWS_FIRST_LENGTH;
}

void SelectSortMode(SortTileInfo& info)
{
    if (info.length >= RadixFirstLength(info) || info.length > NATIVE_MERGE_POSITION_LIMIT) {
        info.mode = MULTI_BLOCK_MODE;
    } else if (info.length <= MERGE_AXIS_LIMIT) {
        info.mode = MERGE_LOCAL_MODE;
    } else if (info.length > STREAM_MERGE_AXIS_LIMIT ||
               (info.length == STREAM_NATIVE_AXIS_LIMIT && info.rows > info.availableCores)) {
        info.mode = MERGE_GM_MODE;
    } else {
        info.mode = MERGE_STREAM_MODE;
    }
}

void ComputeTileShape(SortTileInfo& info)
{
    if (info.mode == MERGE_STREAM_MODE) {
        info.alignedLength = AlignUp((info.length + STREAM_MERGE_RUNS - 1) / STREAM_MERGE_RUNS, SIMD_ALIGN_ELEMENTS);
        info.tileCount = STREAM_MERGE_RUNS;
        return;
    }
    const uint32_t tileLength = info.mode == MULTI_BLOCK_MODE ? info.alignedLength : info.length;
    info.alignedLength = AlignUp(tileLength, SIMD_ALIGN_ELEMENTS);
    info.tileCount = info.mode == MULTI_BLOCK_MODE ?
                         static_cast<uint32_t>((static_cast<uint64_t>(info.length) + tileLength - 1) / tileLength) :
                         1;
}

uint64_t ComputeRowWorkspaceSize(const SortTileInfo& info, uint32_t coresPerRow)
{
    const uint64_t valueBytes = AlignUp(static_cast<uint64_t>(info.length) * VALUE_BYTES, UB_BLOCK_BYTES);
    const uint64_t indexBytes = AlignUp(static_cast<uint64_t>(info.length) * INDEX_BYTES, UB_BLOCK_BYTES);
    const uint64_t histogramBytes = (static_cast<uint64_t>(info.tileCount) + coresPerRow) * HISTOGRAM_RECORD_BYTES;
    return valueBytes + indexBytes + histogramBytes;
}

uint64_t ComputeUbSize(const SortTileInfo& info)
{
    if (info.mode == MERGE_GM_MODE || (info.mode == MERGE_STREAM_MODE && info.length <= STREAM_NATIVE_AXIS_LIMIT)) {
        return std::max<uint64_t>(uint64_t(info.alignedLength) * NATIVE_MERGE_LOCAL_BYTES_PER_ELEMENT,
                                  NATIVE_MERGE_ARENA_BYTES) +
               NATIVE_MERGE_SMALL_BYTES + NATIVE_MERGE_POSITION_SEED_BYTES;
    }
    if (info.mode == MERGE_STREAM_MODE) {
        return uint64_t(info.alignedLength) * STREAM_MERGE_RUNS * MERGE_PROPOSAL_BYTES +
               STREAM_MERGE_PAYLOAD_ELEMENTS * INDEX_BYTES;
    }
    const uint64_t length = info.alignedLength;
    const uint64_t valueKeyBytes = length * VALUE_KEY_BYTES;
    const uint64_t indexBytes = (length + WORK_GUARD_ELEMENTS) * INDEX_BYTES;
    const uint64_t maskBytes = AlignUp(length / BITS_PER_BYTE, UB_BLOCK_BYTES);
    uint64_t bytes = valueKeyBytes + indexBytes + maskBytes;
    if (info.mode == MULTI_BLOCK_MODE) {
        const uint64_t valueOutputBytes = (length + WORK_GUARD_ELEMENTS) * VALUE_BYTES;
        const uint64_t keyOutputBytes = valueOutputBytes;
        const uint64_t digitScratchBytes = length * INDEX_BYTES;
        const uint64_t prefixBytes = std::min(info.tileCount, PREFIX_BATCH_TILES) * HISTOGRAM_RECORD_BYTES;
        bytes += valueOutputBytes + keyOutputBytes + digitScratchBytes + prefixBytes;
        const uint64_t inputTileBytes = length * (VALUE_BYTES + INDEX_BYTES);
        const uint64_t outputTileBytes = (length + WORK_GUARD_ELEMENTS) * (VALUE_BYTES + INDEX_BYTES);
        bytes += RADIX_PIPELINE_BUFFERS * (inputTileBytes + outputTileBytes);
    }
    if (info.mode == MERGE_LOCAL_MODE) {
        bytes += indexBytes + length * (MERGE_WORKSPACE_BUFFERS * MERGE_PROPOSAL_BYTES + INDEX_BYTES);
    }
    return bytes;
}

uint32_t FloorPowerOfTwo(uint32_t value)
{
    uint32_t result = 1;
    while (result <= value / MERGE_PAIR_LIST_COUNT) {
        result *= MERGE_PAIR_LIST_COUNT;
    }
    return result;
}

void SelectSmallRowBatch(SortTileInfo& info, uint32_t coreNum, uint64_t ubSize)
{
    if (info.mode != MERGE_LOCAL_MODE ||
        static_cast<uint64_t>(info.rows) < static_cast<uint64_t>(coreNum) * MERGE_PAIR_LIST_COUNT ||
        ubSize <= SDK_RESERVED_UB_BYTES) {
        return;
    }
    // 行跨度按 Sort32 对齐，避免长度为 129、257 等场景补齐到二次幂后工作量翻倍；极短行仍使用二次幂补齐。
    uint32_t stride = AlignUp(info.length, MERGE_SORT_RUN_ELEMENTS);
    if (info.length <= MERGE_BATCH_POWER_OF_TWO_LIMIT) {
        stride = MERGE_SORT_RUN_ELEMENTS;
        while (stride < info.length) {
            stride *= MERGE_PAIR_LIST_COUNT;
        }
    }
    const uint32_t rowsPerCore = (static_cast<uint64_t>(info.rows) + coreNum - 1) / coreNum;
    uint32_t batch = std::min(rowsPerCore, MERGE_BATCH_ELEMENTS / stride);
    for (; batch > 1; --batch) {
        SortTileInfo candidate = info;
        candidate.tileCount = batch;
        candidate.alignedLength = AlignUp(batch * stride, SIMD_ALIGN_ELEMENTS);
        if (ComputeUbSize(candidate) <= ubSize - SDK_RESERVED_UB_BYTES) {
            info = candidate;
            return;
        }
    }
}

void ComputeRadixCandidate(SortTileInfo& info, uint32_t cores, uint32_t capacity)
{
    info.coresPerRow = cores;
    info.alignedLength = capacity;
    ComputeTileShape(info);
    if (cores == 1) {
        return;
    }
    const uint32_t tileWaves = (info.tileCount + cores - 1) / cores;
    const uint64_t targetTiles = static_cast<uint64_t>(tileWaves) * cores;
    const uint32_t balanced = AlignUp(static_cast<uint32_t>((info.length + targetTiles - 1) / targetTiles),
                                      SIMD_ALIGN_ELEMENTS);
    info.alignedLength = std::min(capacity, std::max(RADIX_MIN_TILE_ELEMENTS, balanced));
    ComputeTileShape(info);
}

double EstimateRadixTime(const SortTileInfo& info, uint32_t groups)
{
    const uint64_t tileWaves = (static_cast<uint64_t>(info.tileCount) + info.coresPerRow - 1) / info.coresPerRow;
    const uint64_t elements = std::min<uint64_t>(info.length, tileWaves * info.alignedLength);
    uint64_t rowCost = AlignUp(elements, SIMD_ALIGN_ELEMENTS) / SIMD_ALIGN_ELEMENTS *
                           RadixScheduleCost::ALIGNED_ELEMENTS_NS +
                       tileWaves * RadixScheduleCost::TILE_NS + RadixScheduleCost::ROW_NS;
    if (info.coresPerRow > 1) {
        // 单核处理一行无需 SyncAll；多核协作时，即使每核仅一个 tile，每轮仍需同步并扫描核摘要。
        rowCost += RadixScheduleCost::COOPERATION_NS + uint64_t(info.coresPerRow) * RadixScheduleCost::CORE_SUMMARY_NS;
        if (groups > 1) {
            rowCost += uint64_t(groups) * info.coresPerRow * RadixScheduleCost::WAVE_SUMMARY_NS;
        }
    }
    const uint64_t rowWaves = (static_cast<uint64_t>(info.rows) + groups - 1) / groups;
    return static_cast<double>(rowWaves) * rowCost;
}

// 依据实际 UB 布局，为每个候选核数分别计算 tile 大小，再比较处理全部行的预计耗时。
void SelectRadixTile(SortTileInfo& info, uint32_t coreNum, uint64_t ubSize)
{
    if (info.mode != MULTI_BLOCK_MODE || ubSize <= SDK_RESERVED_UB_BYTES) {
        return;
    }
    const uint64_t budget = ubSize - SDK_RESERVED_UB_BYTES;
    uint32_t maximum = 0;
    SortTileInfo candidate = info;
    for (uint32_t length = SIMD_ALIGN_ELEMENTS; length <= budget / VALUE_KEY_BYTES; length += SIMD_ALIGN_ELEMENTS) {
        candidate.alignedLength = length;
        candidate.tileCount = (static_cast<uint64_t>(info.length) + length - 1) / length;
        if (ComputeUbSize(candidate) <= budget) {
            maximum = length;
        }
    }
    if (maximum == 0) {
        return;
    }
    // 协作 tile 向下取二次幂后再均衡切分，减少向量重复与 DMA 不均；独立行保留较大 tile。
    const uint32_t capacity = (static_cast<uint64_t>(info.length) + maximum - 1) / maximum < coreNum ?
                                  FloorPowerOfTwo(maximum) :
                                  maximum;
    const uint32_t maxCores = std::min<uint64_t>((static_cast<uint64_t>(info.length) + capacity - 1) / capacity,
                                                 coreNum);
    if (info.rows == 1) {
        ComputeRadixCandidate(info, maxCores, maxCores == 1 ? maximum : capacity);
        return;
    }
    bool selected = false;
    double bestTime = 0;
    uint64_t bestTileWaves = 0;
    std::vector<SortTileInfo> feasible;
    for (uint32_t cores = 1; cores <= maxCores; ++cores) {
        SortTileInfo plan = info;
        ComputeRadixCandidate(plan, cores, cores == 1 ? maximum : capacity);
        const uint32_t groups = std::min(info.rows, coreNum / cores);
        if (ComputeUbSize(plan) > budget ||
            (groups > 1 && ComputeRowWorkspaceSize(plan, cores) * groups > ROW_PARALLEL_WORKSPACE_LIMIT)) {
            continue;
        }
        feasible.push_back(plan);
        const double time = EstimateRadixTime(plan, groups);
        if (!selected || time < bestTime) {
            selected = true;
            bestTime = time;
            bestTileWaves = (static_cast<uint64_t>(plan.tileCount) + cores - 1) / cores;
        }
    }
    // 候选按核数递增枚举。误差范围内优先减少核数，但不能增加最忙核串行处理的 tile 批数。
    for (const auto& plan : feasible) {
        const uint32_t groups = std::min(info.rows, coreNum / plan.coresPerRow);
        const uint64_t tileWaves = (static_cast<uint64_t>(plan.tileCount) + plan.coresPerRow - 1) / plan.coresPerRow;
        if (tileWaves <= bestTileWaves &&
            EstimateRadixTime(plan, groups) * 100 <= bestTime * (100 + RadixScheduleCost::TOLERANCE_PERCENT)) {
            info = plan;
            return;
        }
    }
}

bool PreferIndependentMergeRows(const SortTileInfo& info, uint32_t coreNum)
{
    const uint32_t parallelRows = std::min(info.rows, coreNum);
    const uint32_t rowCoreBudget = coreNum / parallelRows;
    const uint32_t workspaceFreeLimit = info.dtype == ge::DT_FLOAT16 ? NATIVE_SORT_FP16_UB_AXIS_LIMIT :
                                                                       NATIVE_SORT_BF16_UB_AXIS_LIMIT;
    if (info.length <= workspaceFreeLimit) {
        return true;
    }
    // 超过原生 Sort 无 workspace 的范围后，中等长度单行也可使用多核归并；SelectGmMergePlan 按源数据分段粒度限制核数。
    if (info.length <= STREAM_SINGLE_RUN_AXIS_LIMIT && rowCoreBudget < NATIVE_MERGE_MIN_COOPERATIVE_CORES) {
        return true;
    }
    return rowCoreBudget == 1;
}

void SelectMergePlan(SortTileInfo& info, uint32_t coreNum, uint64_t ubSize)
{
    if (info.mode != MERGE_STREAM_MODE || ubSize <= SDK_RESERVED_UB_BYTES) {
        return;
    }
    // 协作归并使用 key 8，具体行与核的划分由 SelectGmMergePlan 决定。
    if (!PreferIndependentMergeRows(info, coreNum)) {
        info.mode = MERGE_GM_MODE;
        return;
    }
    info.coresPerRow = 1;
    ComputeTileShape(info);
    if (ComputeUbSize(info) > ubSize - SDK_RESERVED_UB_BYTES) {
        info.mode = MERGE_GM_MODE;
    }
}

uint32_t ComputeBlockDim(const SortTileInfo& info, uint32_t coreNum)
{
    if (info.mode == MERGE_GM_MODE)
        return info.mergeBlockDim;
    const uint32_t rowGroups = std::min(info.rows, std::max(1U, coreNum / info.coresPerRow));
    return info.coresPerRow * rowGroups;
}

void SelectGmMergePlan(SortTileInfo& info, uint32_t coreNum, uint64_t ubSize)
{
    info.alignedLength = NATIVE_MERGE_RUN_ELEMENTS;
    if (ubSize <= SDK_RESERVED_UB_BYTES || ComputeUbSize(info) > ubSize - SDK_RESERVED_UB_BYTES) {
        info.mode = MULTI_BLOCK_MODE;
        SelectRadixTile(info, coreNum, ubSize);
        return;
    }
    if (info.rows >= coreNum) {
        info.rowsPerCore = static_cast<uint32_t>((uint64_t(info.rows) + coreNum - 1) / coreNum);
        info.mergeBlockDim = static_cast<uint32_t>((uint64_t(info.rows) + info.rowsPerCore - 1) / info.rowsPerCore);
        info.coresPerRow = 1;
        info.mergeGroupSpan = info.length;
    } else {
        const uint32_t available = coreNum / info.rows;
        info.mergeGroupSpan = AlignUp(static_cast<uint32_t>((uint64_t(info.length) + available - 1) / available),
                                      NATIVE_MERGE_AXIS_GRAIN);
        info.coresPerRow = static_cast<uint32_t>((uint64_t(info.length) + info.mergeGroupSpan - 1) /
                                                 info.mergeGroupSpan);
        // 五段需要两层四路 GM 归并；若四个稍大的分段仍能各自在 UB 排序，则只保留一层最终归并。
        if (info.coresPerRow == MERGE_LIST_COUNT + 1) {
            const uint32_t fourWaySpan = AlignUp(
                static_cast<uint32_t>((uint64_t(info.length) + MERGE_LIST_COUNT - 1) / MERGE_LIST_COUNT),
                SIMD_ALIGN_ELEMENTS);
            if (fourWaySpan <= NATIVE_MERGE_COOPERATIVE_RUN_LIMIT) {
                info.mergeGroupSpan = fourWaySpan;
                info.coresPerRow = MERGE_LIST_COUNT;
            }
        }
        info.mergeBlockDim = info.coresPerRow * info.rows;
        info.rowsPerCore = 1;
        // 协作分段在单次排序上限和实际 UB 容量内可适当增大，避免按 4096 项拆分后再增加 GM 归并。
        info.alignedLength = std::min(NATIVE_MERGE_COOPERATIVE_RUN_LIMIT, info.mergeGroupSpan);
        while (info.alignedLength > NATIVE_MERGE_RUN_ELEMENTS && ComputeUbSize(info) > ubSize - SDK_RESERVED_UB_BYTES) {
            info.alignedLength -= SIMD_ALIGN_ELEMENTS;
        }
    }
    info.tileCount = static_cast<uint32_t>((uint64_t(info.length) + info.alignedLength - 1) / info.alignedLength);
}

uint64_t ComputeWorkspaceSize(const SortTileInfo& info, uint32_t blockDim)
{
    if (info.mode == MERGE_GM_MODE) {
        return AlignUp(uint64_t(info.length), DMA_VALUE_ELEMENTS) * NATIVE_MERGE_ROW_BYTES_PER_ELEMENT *
               (blockDim / info.coresPerRow);
    }
    if (info.mode == MERGE_STREAM_MODE && info.length <= STREAM_SINGLE_RUN_AXIS_LIMIT) {
        return 0;
    }
    if (info.mode == MERGE_STREAM_MODE && info.length <= STREAM_NATIVE_AXIS_LIMIT) {
        return AlignUp(uint64_t(info.length), DMA_VALUE_ELEMENTS) * NATIVE_MERGE_ROW_BYTES_PER_ELEMENT * blockDim;
    }
    if (info.mode == MERGE_STREAM_MODE) {
        return uint64_t(info.alignedLength) * STREAM_MERGE_RUNS * (MERGE_PROPOSAL_BYTES + INDEX_BYTES) * blockDim;
    }
    if (info.mode != MULTI_BLOCK_MODE) {
        return 0;
    }
    // 并发处理的行各自占用临时数据及直方图空间；输出张量复用为基数排序乒乓缓冲的另一端。
    return ComputeRowWorkspaceSize(info, info.coresPerRow) * (blockDim / info.coresPerRow);
}

uint32_t SelectTilingKey(const SortTileInfo& info)
{
    // SingleRun 和 Native Stream 共用调度与缓冲规划，通过不同编译入口分发。
    if (info.mode == MERGE_STREAM_MODE && info.length > STREAM_SINGLE_RUN_AXIS_LIMIT) {
        return MERGE_STREAM_NATIVE_MODE;
    }
    return info.mode;
}

ge::graphStatus SaveTilingData(gert::TilingContext* context, const SortTileInfo& info)
{
    SortWithIndexTilingDataA2 data;
    data.set_rows(info.rows);
    data.set_length(info.length);
    data.set_alignedLength(info.alignedLength);
    data.set_descending(static_cast<uint32_t>(info.descending));
    data.set_tileCount(info.tileCount);
    data.set_localFanout(info.coresPerRow);
    data.set_mergeGroupSpan(info.mergeGroupSpan);
    data.set_rowsPerCore(info.rowsPerCore);
    auto* rawTilingData = context->GetRawTilingData();
    if (rawTilingData == nullptr || rawTilingData->GetCapacity() < data.GetDataSize()) {
        return ge::GRAPH_FAILED;
    }
    data.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
} // namespace

static ge::graphStatus TilingSortWithIndexA2(gert::TilingContext* context)
{
    const auto* compileInfo = static_cast<const SortWithIndexCompileInfoA2*>(context->GetCompileInfo());
    if (compileInfo == nullptr || compileInfo->core_num == 0) {
        return ge::GRAPH_FAILED;
    }
    SortTileInfo info;
    if (GetInputInfo(context, info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    info.availableCores = compileInfo->core_num;
    SelectSortMode(info);
    if (info.mode == MULTI_BLOCK_MODE) {
        SelectRadixTile(info, compileInfo->core_num, compileInfo->ub_size);
    } else {
        info.coresPerRow = 1;
        ComputeTileShape(info);
    }
    SelectMergePlan(info, compileInfo->core_num, compileInfo->ub_size);
    if (info.mode == MERGE_GM_MODE) {
        SelectGmMergePlan(info, compileInfo->core_num, compileInfo->ub_size);
    }
    if (info.mode == MERGE_STREAM_MODE && info.length <= STREAM_NATIVE_AXIS_LIMIT) {
        // 每段独立限制排序长度；尾段仅将有效数据按 Sort32 对齐，不补齐到完整段长。
        info.alignedLength = info.length <= STREAM_SINGLE_RUN_AXIS_LIMIT ? AlignUp(info.length, SIMD_ALIGN_ELEMENTS) :
                                                                           NATIVE_MERGE_RUN_ELEMENTS;
        info.tileCount = (info.length + info.alignedLength - 1) / info.alignedLength;
        // 流式归并按核分配完整行，各核独立处理。
        info.mergeGroupSpan = info.length;
        info.rowsPerCore = static_cast<uint32_t>((uint64_t(info.rows) + compileInfo->core_num - 1) /
                                                 compileInfo->core_num);
    }
    SelectSmallRowBatch(info, compileInfo->core_num, compileInfo->ub_size);
    if (info.mode == MERGE_STREAM_MODE && info.length > STREAM_SINGLE_RUN_AXIS_LIMIT &&
        info.length <= STREAM_NATIVE_AXIS_LIMIT) {
        // key 8 使用独立编译入口；保留已选定的行、核及 workspace 方案，避免重复规划改变调度。
        info.mergeBlockDim = ComputeBlockDim(info, compileInfo->core_num);
        info.mode = MERGE_GM_MODE;
    }
    if (info.alignedLength == 0 || info.coresPerRow == 0 || compileInfo->ub_size <= SDK_RESERVED_UB_BYTES ||
        ComputeUbSize(info) > compileInfo->ub_size - SDK_RESERVED_UB_BYTES) {
        return ge::GRAPH_FAILED;
    }
    if (SaveTilingData(context, info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(SelectTilingKey(info));
    const uint32_t blockDim = ComputeBlockDim(info, compileInfo->core_num);
    context->SetBlockDim(blockDim);
    const uint64_t userWorkspaceSize = ComputeWorkspaceSize(info, blockDim);
    // 生成的 kernel 入口已传入 GetUserWorkspace(workspace)。用户临时区须位于系统预留区之后。
    // 仅使用 UB 的 LOCAL 和 SingleRun 无需申请这两部分 GM 空间。
    const uint64_t sysWorkspaceSize = userWorkspaceSize == 0 ? 0 : compileInfo->sys_workspace_size;
    if (userWorkspaceSize > std::numeric_limits<size_t>::max() - sysWorkspaceSize) {
        return ge::GRAPH_FAILED;
    }
    auto* workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaceSizes[0] = userWorkspaceSize + sysWorkspaceSize;
    if (info.mode == MULTI_BLOCK_MODE || info.mode == MERGE_GM_MODE) {
        context->SetScheduleMode(BATCH_SCHEDULE_MODE);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus PrepareSortWithIndexA2(gert::TilingParseContext* context)
{
    auto* compileInfo = context->GetCompiledInfo<SortWithIndexCompileInfoA2>();
    auto* platformInfo = context->GetPlatformInfo();
    if (compileInfo == nullptr || platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(platformInfo);
    compileInfo->core_num = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ub_size);
    compileInfo->sys_workspace_size = platform.GetLibApiWorkSpaceSize();
    if (compileInfo->core_num == 0 || compileInfo->ub_size <= SDK_RESERVED_UB_BYTES) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SortWithIndex)
    .Tiling(TilingSortWithIndexA2)
    .TilingParse<SortWithIndexCompileInfoA2>(PrepareSortWithIndexA2);
} // namespace optiling
