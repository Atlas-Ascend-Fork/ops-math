/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SELECT_TOP_K_SORT_PLAN_H_
#define SELECT_TOP_K_SORT_PLAN_H_
#include "../../op_kernel/select_top_k_tiling_data.h"
#include "select_top_k_base_plan.h"

namespace SelectTopK {
// 按连续行块分核；AscendC DataCopyPad 仅写有效字节，无需沿用原生 DMA 的短输出单核限制。
inline void PlanRowBlocks(uint64_t rows, uint32_t cores, SortSchedule& schedule, SelectTopKBaseTilingData& plan)
{
    schedule.rowsPerCore = SelectTopKBase::CeilDiv(rows, cores);
    plan.cores = SelectTopKBase::CeilDiv(rows, schedule.rowsPerCore);
}

// 按 四种模式规划分核和分层归并；容量来自 AscendC 实际缓冲，不使用场景特化阈值。
inline bool BuildSortPlan(uint64_t rows, uint64_t axis, uint64_t k, uint32_t bytes, bool largest, bool sorted,
                          uint64_t ub, uint32_t cores, SelectTopKBaseTilingData& plan, SortSchedule& schedule)
{
    using namespace SelectTopKBase;
    plan = {};
    schedule = {};
    plan.rows = rows;
    plan.axis = axis;
    plan.k = k;
    plan.dtypeBytes = bytes;
    plan.largest = largest;
    plan.sorted = sorted;
    plan.kernel = KernelTemplate::ROW_SORT;
    if (cores == 0 || ub <= UB_RESERVE || (bytes != VALUE16_BYTES && bytes != VALUE32_BYTES))
        return false;
    if (rows == 0 || axis == 0 || k == 0) {
        plan.cores = 1;
        plan.route = ROUTE_EMPTY;
        return true;
    }
    const uint64_t available = ub - UB_RESERVE;
    const uint32_t rowBytes = ROW_SORT_ARRAY_BYTES + ROW_INPUT_BUFFER_COUNT * bytes;
    const uint32_t rowCapacity = std::min<uint64_t>(SORT32_MAX_ELEMENTS,
                                                    available / rowBytes / SORT32_ELEMENTS * SORT32_ELEMENTS);
    const uint32_t leafCapacity = std::min<uint64_t>(
        MAX_LEAF_SORT_ITEMS, available / COMPACT_SORT_BYTES / SHARD_ALIGNMENT * SHARD_ALIGNMENT);
    if (rowCapacity < SORT32_ELEMENTS || leafCapacity == 0)
        return false;

    if (Align(axis, SORT32_ELEMENTS) <= rowCapacity) {
        schedule.pattern = SortPattern::ROWS;
        schedule.rowWidth = Align(axis, SORT32_ELEMENTS);
        PlanRowBlocks(rows, cores, schedule, plan);
        plan.batchRows = std::min<uint64_t>(schedule.rowsPerCore, rowCapacity / schedule.rowWidth);
        plan.tile = plan.batchRows * schedule.rowWidth;
        plan.tiles = 1;
        plan.shardCount = 1;
        plan.outputTile = schedule.rowWidth;
        plan.route = ROUTE_COMPACT_ROW_SORT;
        plan.ubBytes = plan.tile * rowBytes;
        return true;
    }

    // 空闲核按整行数均分，保持每个轴片段连续；不是所有叶子在全核间轮转。
    const uint64_t shardsAvailable = cores / rows;
    schedule.pattern = SortPattern::LOCAL_MERGE;
    schedule.shardElements = axis;
    plan.shardCount = 1;
    if (shardsAvailable > 1) {
        schedule.shardElements = Align(CeilDiv(axis, shardsAvailable), SHARD_ALIGNMENT);
        schedule.pattern = schedule.shardElements <= leafCapacity ? SortPattern::CROSS_MERGE :
                                                                    SortPattern::LOCAL_CROSS_MERGE;
        // 局部前缀工作区仍按 pair 块对齐；行首核的最终输出使用非对齐字节拷贝。
        // 保留真实 K，不额外计算 KFirst32，也不引入裁剪、Gather 或全核同步。
        plan.shardCount = CeilDiv(axis, schedule.shardElements);
        plan.cores = rows * plan.shardCount;
        schedule.rowsPerCore = 1;
        if (plan.shardCount == 1)
            schedule.pattern = SortPattern::LOCAL_MERGE;
    } else {
        PlanRowBlocks(rows, cores, schedule, plan);
    }
    plan.kernel = KernelTemplate::MERGE_SORT;
    plan.tile = std::min<uint64_t>(leafCapacity, Align(schedule.shardElements, SORT32_ELEMENTS));
    schedule.leavesPerShard = CeilDiv(schedule.shardElements, plan.tile);
    schedule.leafStride = Align(std::min<uint64_t>(k, plan.tile), PAIR_ALIGNMENT);
    plan.tiles = plan.shardCount * schedule.leavesPerShard;
    plan.route = ROUTE_GM_MERGE;
    plan.mergeWindow = std::min<uint64_t>(plan.tile, available / MERGE_WORK_BYTES / SORT32_ELEMENTS * SORT32_ELEMENTS);
    if (plan.mergeWindow == 0)
        return false;
    plan.outputTile = plan.mergeWindow;
    const GeneralMergeLayout layout(plan.tile, plan.mergeWindow, k, bytes, plan.route);
    plan.ubBytes = layout.bytes;
    plan.rowWorkspacePairs = plan.tiles * schedule.leafStride;
    uint64_t pairs = 0;
    return layout.bytes <= available && Mul(rows, plan.rowWorkspacePairs, pairs) &&
           Mul(pairs, DOUBLE_PAIR_BYTES, plan.workspaceBytes) && plan.workspaceBytes <= INT64_MAX;
}
} // namespace SelectTopK
#endif
