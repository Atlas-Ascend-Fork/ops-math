/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SELECT_TOP_K_TARGETED_PLAN_H_
#define SELECT_TOP_K_TARGETED_PLAN_H_
#include "select_top_k_select_plan.h"
#include "../../op_kernel/select_top_k_two_leaf_layout.h"
#include "../../op_kernel/select_top_k_scalar_layout.h"
namespace SelectTopK {
// 计算完整候选列表所需的四路归并层数，用于权衡局部排序与最终归并。
inline uint32_t LeafMergeRounds(uint64_t lists)
{
    uint32_t rounds = 0;
    while (lists > 1) {
        lists = SelectTopKBase::CeilDiv(lists, SelectTopKBase::MERGE_WAYS);
        ++rounds;
    }
    return rounds;
}

// 在核数和 UB 允许时重设叶子块长，并同步更新候选工作区与核数。
inline bool RetileLeaf(SelectTopKBaseTilingData& plan, uint32_t tile, uint32_t capacity, uint64_t ub, uint32_t cores)
{
    using namespace SelectTopKBase;
    const uint64_t shards = CeilDiv(plan.axis, tile);
    const LeafSelectLayout layout(tile, plan.k, plan.dtypeBytes, capacity);
    if (plan.k > tile || shards > cores / plan.rows || layout.bytes > ub - UB_RESERVE)
        return false;
    plan.tile = tile;
    plan.tiles = plan.shardCount = shards;
    plan.cores = plan.rows * shards;
    plan.finalCapacity = capacity;
    plan.ubBytes = layout.bytes;
    plan.rowWorkspacePairs = shards * Align(plan.k, PAIR_ALIGNMENT);
    plan.workspaceBytes = plan.rows * plan.rowWorkspacePairs * PAIR_BYTES;
    return true;
}

// 按已验证的UB容量和候选段数均衡叶子，不改变基础排序或逐页过滤的规划。
inline void TuneLeafPlan(SelectTopKBaseTilingData& plan, uint64_t ub, uint32_t cores)
{
    using namespace SelectTopKBase;
    const uint64_t rows = plan.rows, axis = plan.axis, selectedK = plan.k;
    const uint32_t bytes = plan.dtypeBytes;
    // 四个完整前缀加一个可选短尾只需一次全核同步。
    if (plan.candidateMerge == CandidateMerge::FULL_LISTS && selectedK > SelectTopKBase::Tuning::SMALL_K &&
        selectedK <= Tuning::PREFIX_SMALL_K && plan.tile <= SelectTopKBase::Tuning::MAX_LEAF_BLOCK &&
        (plan.shardCount == MERGE_WAYS || plan.shardCount == MERGE_WAYS + 1)) {
        plan.candidateMerge = CandidateMerge::HALF_PREFIX;
    }
    if (plan.candidateMerge == CandidateMerge::FULL_LISTS && plan.shardCount > Tuning::BALANCED_LEAVES &&
        plan.shardCount * SelectTopKBase::Align(selectedK, PAIR_ALIGNMENT) > plan.finalCapacity) {
        const SelectTopKBase::LeafSelectLayout layout(plan.tile, selectedK, bytes, plan.finalCapacity);
        const uint64_t prefix = (selectedK + HALF_PREFIX_ALIGNMENT - 1) / HALF_PREFIX_ALIGNMENT * PAIR_ALIGNMENT;
        if (prefix * MERGE_WORK_BYTES <= layout.accumulator) {
            plan.candidateMerge = CandidateMerge::HALF_PREFIX;
        }
    }

    // 保持独立叶子数量不变，以 Sort32 的粒度均衡每个叶子的工作量。
    if (plan.kernel == KernelTemplate::LEAF_SELECT && plan.tile > SelectTopKBase::Tuning::MAX_LEAF_BLOCK) {
        const uint32_t balanced = SelectTopKBase::Align(SelectTopKBase::CeilDiv(axis, plan.shardCount),
                                                        SORT32_ELEMENTS);
        if (balanced > SelectTopKBase::Tuning::MAX_LEAF_BLOCK && balanced < plan.tile) {
            const SelectTopKBase::LeafSelectLayout layout(balanced, selectedK, bytes, plan.finalCapacity);
            if (layout.bytes <= ub - SelectTopKBase::UB_RESERVE) {
                plan.tile = balanced;
                plan.ubBytes = layout.bytes;
            }
        }
    }
    // 八个均衡叶子可利用空闲核，同时保持批量归并层数。
    // 常规叶子使用固定长度模板；扩展叶子按 Sort32 粒度归并，可细分工作量。
    if (plan.kernel == KernelTemplate::LEAF_SELECT && plan.tile > SelectTopKBase::Tuning::MAX_LEAF_BLOCK) {
        if (selectedK <= SelectTopKBase::Tuning::TINY_LEAF_BLOCK && plan.shardCount < Tuning::BALANCED_LEAVES &&
            cores / rows >= Tuning::BALANCED_LEAVES) {
            const uint32_t tile = SelectTopKBase::Align(SelectTopKBase::CeilDiv(axis, Tuning::BALANCED_LEAVES),
                                                        SORT32_ELEMENTS);
            if (tile > SelectTopKBase::Tuning::MAX_LEAF_BLOCK && tile < plan.tile)
                RetileLeaf(plan, tile, plan.finalCapacity, ub, cores);
        } else if (selectedK >= SelectTopKBase::Tuning::SMALL_LEAF_BLOCK && plan.shardCount > Tuning::BALANCED_LEAVES &&
                   (selectedK <= SelectTopKBase::Tuning::SMALL_LEAF_BLOCK ||
                    SelectTopKBase::CeilDiv(axis, SelectTopKBase::Tuning::MAX_LEAF_BLOCK) <=
                        Tuning::MAX_FIXED_LEAVES)) {
            // 第九个扩展叶子会增加一层前缀归并；常规叶子模板可在相同层数下完成排序。
            RetileLeaf(plan, SelectTopKBase::Tuning::MAX_LEAF_BLOCK, SelectTopKBase::Tuning::MAX_LEAF_BLOCK, ub, cores);
        }
    }
    // 将小叶子容量加倍可保持归并层数，并将小 K 的候选列表和并发 DMA 写入数减半。
    if (plan.candidateMerge == CandidateMerge::FULL_LISTS && plan.tile == SelectTopKBase::Tuning::TINY_LEAF_BLOCK &&
        selectedK <= Tuning::SMALL_LEAF_MAX_K && axis >= SelectTopKBase::Tuning::FILTER_MIN_AXIS) {
        const uint64_t shards = SelectTopKBase::CeilDiv(axis, SelectTopKBase::Tuning::SMALL_LEAF_BLOCK);
        const uint32_t capacity = LeafCandidateCapacity(shards, selectedK, SelectTopKBase::Tuning::SMALL_LEAF_BLOCK);
        RetileLeaf(plan, SelectTopKBase::Tuning::SMALL_LEAF_BLOCK, capacity, ub, cores);
    }
    // 单行小 K 用更多空闲核缩短局部排序，但不因此增加最终候选的归并层数。
    // 核数、候选容量或 UB 不足时，仍保留原计划。
    if (rows == 1 && plan.candidateMerge == CandidateMerge::FULL_LISTS &&
        plan.tile == SelectTopKBase::Tuning::SMALL_LEAF_BLOCK && selectedK <= Tuning::SMALL_LEAF_MAX_K &&
        axis >= SelectTopKBase::Tuning::FILTER_MIN_AXIS) {
        const uint64_t shards = SelectTopKBase::CeilDiv(axis, SelectTopKBase::Tuning::TINY_LEAF_BLOCK);
        const uint32_t capacity = LeafCandidateCapacity(shards, selectedK, SelectTopKBase::Tuning::SMALL_LEAF_BLOCK);
        if (LeafMergeRounds(shards) <= LeafMergeRounds(plan.shardCount)) {
            RetileLeaf(plan, SelectTopKBase::Tuning::TINY_LEAF_BLOCK, capacity, ub, cores);
        }
    }
    // 中小K的候选段较多时，用固定4096叶子减少段数和末级归并工作量。
    // 保留K<=64的小叶子方案；容量、核数或UB不满足时仍使用原计划。
    if (plan.candidateMerge == CandidateMerge::FULL_LISTS && plan.tile == SelectTopKBase::Tuning::SMALL_LEAF_BLOCK &&
        selectedK > Tuning::SMALL_LEAF_MAX_K && selectedK <= SelectTopKBase::Tuning::SMALL_K &&
        plan.shardCount > Tuning::BALANCED_LEAVES) {
        const uint32_t tile = SelectTopKBase::Tuning::MAX_LEAF_BLOCK;
        const uint64_t shards = CeilDiv(axis, tile);
        const uint32_t capacity = LeafCandidateCapacity(shards, selectedK, tile);
        RetileLeaf(plan, tile, capacity, ub, cores);
    }
}

// 轴可拆成每核一个叶子时，候选在UB内完成归并；容不下仍保留扫描过滤或基础排序。
inline void TryLeafOptimization(uint64_t ub, uint32_t cores, SelectTopKBaseTilingData& plan, SortSchedule& schedule)
{
    using namespace SelectTopKBase;
    if (plan.rows == 0 || plan.rows >= cores || plan.k < SelectTopKBase::Tuning::FILTER_MIN_K ||
        plan.k > SelectTopKBase::Tuning::FILTER_MAX_K)
        return;
    uint32_t maxLeafTile = SelectTopKBase::Tuning::MAX_LEAF_BLOCK;
    // 四个常规叶子和一个短尾已能占满该行核预算时，避免扩大单核排序。
    const bool shortAxisLargeK = plan.k >= SelectTopKBase::Tuning::SMALL_LEAF_BLOCK &&
                                 plan.axis <= MERGE_WAYS * SelectTopKBase::Tuning::MAX_LEAF_BLOCK + SORT32_ELEMENTS &&
                                 plan.rows <= Tuning::SHORT_AXIS_MAX_ROWS &&
                                 plan.rows * CeilDiv(plan.axis, SelectTopKBase::Tuning::MAX_LEAF_BLOCK) <= cores;
    if (plan.k > SelectTopKBase::Tuning::SMALL_K && !shortAxisLargeK) {
        for (uint32_t block = SelectTopKBase::Tuning::MAX_EXTENDED_LEAF_BLOCK;
             block > SelectTopKBase::Tuning::MAX_LEAF_BLOCK; block -= SORT32_ELEMENTS) {
            const LeafSelectLayout layout(block, plan.k, plan.dtypeBytes, block);
            if (layout.bytes <= ub - UB_RESERVE) {
                maxLeafTile = block;
                break;
            }
        }
    }
    auto candidate = SelectionInput(plan);
    const PlanResources resources{ub - UB_RESERVE, cores, SelectTopKBase::Tuning::SMALL_LEAF_BLOCK, maxLeafTile};
    if (TryLeafSort(candidate, resources) != PlanAttempt::SUCCESS)
        return;
    // 宽块过滤只需一次扫描时，较大的叶子并不一定更好；仅接受小叶子或原两片路径。
    if (plan.kernel == KernelTemplate::WIDE_SELECT && candidate.tile > SelectTopKBase::Tuning::SMALL_LEAF_BLOCK &&
        plan.shardCount != TWO_MERGE_RUNS)
        return;
    TuneLeafPlan(candidate, ub, cores);
    plan = candidate;
    schedule = {};
}

// K=1只求一个极值；规划原值位恢复及每片一个候选，不执行排序或前缀归并。
inline void TryTop1Optimization(uint64_t ub, uint32_t cores, SelectTopKBaseTilingData& plan, SortSchedule& schedule)
{
    using namespace SelectTopKBase;
    const uint64_t bytes = TOP1_TILE * (plan.dtypeBytes + INDEX32_BYTES_64) + TOP1_FIXED_UB;
    if (plan.rows == 0 || plan.k != 1 || plan.axis <= SORT32_ELEMENTS || bytes > ub - UB_RESERVE)
        return;
    auto candidate = SelectionInput(plan);
    candidate.kernel = KernelTemplate::TOP1;
    candidate.route = ROUTE_TOP1_REDUCE;
    candidate.tile = TOP1_TILE;
    candidate.tiles = CeilDiv(plan.axis, TOP1_TILE);
    candidate.outputTile = 1;
    candidate.ubBytes = bytes;
    candidate.shardCount = std::min<uint64_t>(CeilDiv(plan.axis, SelectTopKBase::Tuning::SMALL_LEAF_BLOCK),
                                              std::max<uint64_t>(1, cores / plan.rows));
    candidate.cores = std::min<uint64_t>(cores, plan.rows * candidate.shardCount);
    candidate.rowWorkspacePairs = candidate.shardCount;
    candidate.workspaceBytes = candidate.shardCount == 1 ?
                                   0 :
                                   Align(plan.rows * candidate.shardCount * PAIR_BYTES, UB_BLOCK_BYTES);
    plan = candidate;
    schedule = {};
}

// 标准连续短行直接生成INT64索引，按一批多个Sort32行降低DMA及初始化成本。
inline void TryShortRowOptimization(uint64_t ub, uint32_t cores, uint32_t indexBytes, int64_t offset, int64_t mode,
                                    uint64_t outputInner, SelectTopKBaseTilingData& plan, SortSchedule& schedule)
{
    using namespace SelectTopKBase;
    if (plan.rows == 0 || plan.k == 0 || indexBytes != INDEX64_BYTES || offset != 0 || mode != OUTPUT_BOTH ||
        outputInner != 1)
        return;
    auto candidate = SelectionInput(plan);
    if (TryShortRowBatch(candidate, {ub - UB_RESERVE, cores}) != PlanAttempt::SUCCESS)
        return;
    candidate.kernel = KernelTemplate::SHORT_ROWS;
    plan = candidate;
    schedule = {};
}

// 完整且无序的标准输出直接复制位模式；连续行内生成原轴INT64索引。
inline void TryCopyAllOptimization(uint64_t ub, uint32_t cores, uint32_t indexBytes, int64_t offset, int64_t mode,
                                   uint64_t outputInner, SelectTopKBaseTilingData& plan, SortSchedule& schedule)
{
    using namespace SelectTopKBase;
    if (plan.rows == 0 || plan.axis <= SORT32_ELEMENTS || plan.k != plan.axis || plan.sorted ||
        indexBytes != INDEX64_BYTES || offset != 0 || mode != OUTPUT_BOTH || outputInner != 1)
        return;
    auto candidate = SelectionInput(plan);
    candidate.tile = Align(std::min<uint64_t>(plan.axis, COPY_ALL_TILE), SORT32_ELEMENTS);
    const uint64_t bytes = uint64_t(candidate.tile) * (plan.dtypeBytes + INDEX32_BYTES + INDEX64_BYTES);
    if (bytes > ub - UB_RESERVE)
        return;
    candidate.kernel = KernelTemplate::COPY_ALL;
    candidate.route = ROUTE_COPY;
    candidate.tiles = CeilDiv(plan.axis, candidate.tile);
    candidate.cores = std::min<uint64_t>(cores, plan.rows * candidate.tiles);
    candidate.ubBytes = bytes;
    plan = candidate;
    schedule = {};
}

// 每核最多一条短行时，复用输入/排序缓冲，省去批处理和预取管理。
inline void TryCompactRowOptimization(uint64_t ub, uint32_t cores, SelectTopKBaseTilingData& plan,
                                      SortSchedule& schedule)
{
    using namespace SelectTopKBase;
    if (plan.rows == 0 || plan.rows > cores || plan.axis <= SORT32_ELEMENTS ||
        plan.k < SelectTopKBase::Tuning::FILTER_MIN_K || plan.axis > COMPACT_ROW_MAX_AXIS ||
        plan.kernel != KernelTemplate::ROW_SORT)
        return;
    auto candidate = SelectionInput(plan);
    candidate.tile = Align(plan.axis, SORT32_ELEMENTS);
    const uint64_t bytes = uint64_t(candidate.tile) * COMPACT_SORT_BYTES;
    if (bytes > ub - UB_RESERVE)
        return;
    candidate.kernel = KernelTemplate::MERGE_SORT;
    candidate.route = ROUTE_COMPACT_ROW_SORT;
    candidate.tiles = candidate.shardCount = 1;
    candidate.cores = plan.rows;
    candidate.outputTile = candidate.tile;
    candidate.ubBytes = bytes;
    plan = candidate;
    schedule = {};
}

// 候选可同时驻留时避免窗口重载；核内两叶减少逐叶DMA，跨核仍用四叶并行。
inline void TryResidentMergeOptimization(uint64_t ub, uint32_t cores, SelectTopKBaseTilingData& plan,
                                         SortSchedule& schedule)
{
    using namespace SelectTopKBase;
    if (plan.rows == 0 || plan.kernel != KernelTemplate::MERGE_SORT || schedule.pattern == SortPattern::ROWS ||
        schedule.storage != MergeStorage::WINDOWS)
        return;
    const uint32_t twoLeafTile = Align(CeilDiv(plan.axis, TWO_MERGE_RUNS), SORT32_ELEMENTS);
    const uint32_t ways = plan.rows >= cores && plan.k >= twoLeafTile ? TWO_MERGE_RUNS : MERGE_WAYS;
    const uint32_t tile = Align(CeilDiv(plan.axis, ways), SORT32_ELEMENTS);
    const uint64_t leaves = CeilDiv(plan.axis, tile);
    const uint64_t slots = leaves * tile;
    const uint32_t tileLimit = ways == TWO_MERGE_RUNS ? TWO_MERGE_RUNS * MERGE_MAX_ALIGNED_ITEMS :
                                                        MERGE_MAX_ALIGNED_ITEMS;
    if (tile > tileLimit || plan.k < tile || slots * COMPACT_SORT_BYTES > ub - UB_RESERVE)
        return;
    const bool parallel = plan.rows * leaves <= cores;
    // 中间核预算保留基础切分；新路径只使用完整的每叶一核或完整的每核多行。
    if (!parallel && plan.rows < cores)
        return;
    auto candidate = SelectionInput(plan);
    SortSchedule candidateSchedule{};
    candidateSchedule.pattern = parallel ? SortPattern::CROSS_MERGE : SortPattern::LOCAL_MERGE;
    candidateSchedule.rowsPerCore = parallel ? 1 : CeilDiv(plan.rows, cores);
    candidateSchedule.shardElements = parallel ? tile : plan.axis;
    candidateSchedule.leavesPerShard = parallel ? 1 : leaves;
    candidateSchedule.leafStride = tile;
    candidate.kernel = KernelTemplate::MERGE_SORT;
    candidateSchedule.storage = MergeStorage::ALL_CANDIDATES;
    candidate.route = ROUTE_GM_MERGE;
    candidate.tile = tile;
    candidate.tiles = leaves;
    candidate.shardCount = parallel ? leaves : 1;
    candidate.cores = parallel ? plan.rows * leaves : CeilDiv(plan.rows, candidateSchedule.rowsPerCore);
    candidate.outputTile = plan.k;
    candidate.rowWorkspacePairs = slots;
    if (!Mul(plan.rows, slots, candidate.workspaceBytes) ||
        !Mul(candidate.workspaceBytes, PAIR_BYTES, candidate.workspaceBytes) || candidate.workspaceBytes > INT64_MAX)
        return;
    candidate.ubBytes = slots * COMPACT_SORT_BYTES;
    plan = candidate;
    schedule = candidateSchedule;
}

// 五到八个常规叶子分两批归并；整批结果和一份TopK同时驻留，避免中间GM回写。
inline void TryGroupedResidentOptimization(uint64_t ub, uint32_t cores, SelectTopKBaseTilingData& plan,
                                           SortSchedule& schedule)
{
    using namespace SelectTopKBase;
    if (plan.rows == 0 || plan.kernel != KernelTemplate::MERGE_SORT || schedule.pattern == SortPattern::ROWS ||
        schedule.storage != MergeStorage::WINDOWS || plan.k < SHARD_ALIGNMENT)
        return;
    const uint64_t leaves = CeilDiv(plan.axis, SHARD_ALIGNMENT);
    const uint64_t alignedK = Align(plan.k, SORT32_ELEMENTS);
    const uint64_t capacity = std::max<uint64_t>(MERGE_WAYS * SHARD_ALIGNMENT, DOUBLE_BUFFER_COUNT * alignedK);
    const uint64_t bytes = capacity * DOUBLE_PAIR_BYTES + alignedK * PAIR_BYTES;
    if (leaves <= MERGE_WAYS || leaves > DOUBLE_BUFFER_COUNT * MERGE_WAYS || plan.rows * leaves > cores ||
        alignedK > DOUBLE_BUFFER_COUNT * MERGE_MAX_ALIGNED_ITEMS || bytes > ub - UB_RESERVE)
        return;
    auto candidate = SelectionInput(plan);
    candidate.kernel = KernelTemplate::MERGE_SORT;
    candidate.route = ROUTE_GM_MERGE;
    candidate.tile = SHARD_ALIGNMENT;
    candidate.tiles = candidate.shardCount = leaves;
    candidate.cores = plan.rows * leaves;
    candidate.outputTile = plan.k;
    candidate.finalCapacity = capacity;
    candidate.rowWorkspacePairs = leaves * SHARD_ALIGNMENT;
    if (!Mul(plan.rows, candidate.rowWorkspacePairs, candidate.workspaceBytes) ||
        !Mul(candidate.workspaceBytes, PAIR_BYTES, candidate.workspaceBytes) || candidate.workspaceBytes > INT64_MAX)
        return;
    candidate.ubBytes = bytes;
    schedule = {};
    schedule.pattern = SortPattern::CROSS_MERGE;
    schedule.storage = MergeStorage::GROUPED_CANDIDATES;
    schedule.rowsPerCore = 1;
    schedule.shardElements = candidate.tile;
    schedule.leavesPerShard = 1;
    schedule.leafStride = candidate.tile;
    plan = candidate;
}

// 多行大K恰需两个叶子时，以叶子任务数判断并行度，扩大两路流式窗口减少重载。
// 行数少于核数仍可用两个叶子填满各核；最终行归并允许部分核没有任务。
inline void TryTwoLeafLocalOptimization(uint64_t ub, uint32_t cores, SelectTopKBaseTilingData& plan,
                                        SortSchedule& schedule)
{
    using namespace SelectTopKBase;
    if (plan.kernel != KernelTemplate::MERGE_SORT || schedule.pattern != SortPattern::LOCAL_MERGE ||
        schedule.storage != MergeStorage::WINDOWS || plan.rows < CeilDiv(cores, TWO_MERGE_RUNS) ||
        schedule.leavesPerShard != TWO_MERGE_RUNS || plan.shardCount != 1 || plan.k <= plan.tile ||
        plan.mergeWindow >= TWO_LEAF_WINDOW)
        return;
    const TwoLeafMergeLayout layout(plan.tile, TWO_LEAF_WINDOW);
    if (layout.bytes > ub - UB_RESERVE)
        return;
    schedule.storage = MergeStorage::TWO_LEAF_WINDOWS;
    plan.mergeWindow = plan.outputTile = TWO_LEAF_WINDOW;
    plan.ubBytes = layout.bytes;
    plan.cores = cores;
    schedule.rowsPerCore = CeilDiv(plan.rows, cores);
}

// 大K的五至七个扩展叶子在有空闲核时均衡为八片，由单核依次处理两个四路组。
inline void TryBalancedLargePrefix(uint64_t ub, uint32_t cores, SelectTopKBaseTilingData& plan)
{
    using namespace SelectTopKBase;
    if (plan.kernel != KernelTemplate::LEAF_SELECT || plan.candidateMerge != CandidateMerge::HALF_PREFIX ||
        plan.k < SelectTopKBase::Tuning::SMALL_LEAF_BLOCK || plan.tile <= SelectTopKBase::Tuning::MAX_LEAF_BLOCK ||
        plan.shardCount <= MERGE_WAYS || plan.shardCount >= Tuning::BALANCED_LEAVES ||
        cores / plan.rows < Tuning::BALANCED_LEAVES)
        return;
    const uint32_t tile = Align(CeilDiv(plan.axis, Tuning::BALANCED_LEAVES), SORT32_ELEMENTS);
    if (tile <= SelectTopKBase::Tuning::MAX_LEAF_BLOCK || tile >= plan.tile)
        return;
    auto candidate = plan;
    if (RetileLeaf(candidate, tile, plan.finalCapacity, ub, cores)) {
        plan = candidate;
    }
}

// 每叶一核的4096项固定排序模板；四路短前缀逐组累积，省去中间GM写回与额外同步。
inline void TryDenseLeafPrefix(uint64_t ub, uint32_t cores, SelectTopKBaseTilingData& plan)
{
    using namespace SelectTopKBase;
    if (plan.kernel != KernelTemplate::LEAF_SELECT || plan.candidateMerge != CandidateMerge::HALF_PREFIX ||
        plan.k <= SelectTopKBase::Tuning::SMALL_LEAF_BLOCK || plan.k > SelectTopKBase::Tuning::MAX_LEAF_BLOCK)
        return;
    const uint64_t leaves = CeilDiv(plan.axis, SelectTopKBase::Tuning::MAX_LEAF_BLOCK);
    if (leaves <= LOCAL_PREFIX_MAX_LISTS || leaves > DENSE_PREFIX_MAX_LEAVES)
        return;
    auto candidate = plan;
    if (RetileLeaf(candidate, SelectTopKBase::Tuning::MAX_LEAF_BLOCK, SelectTopKBase::Tuning::MAX_LEAF_BLOCK, ub,
                   cores)) {
        candidate.candidateMerge = CandidateMerge::DENSE_PREFIX;
        plan = candidate;
    }
}

// 标准单行短轴使用固定小UB入口，避免批处理、输出分段和通用收尾的固定开销。
inline void TryTinyRowOptimization(uint64_t ub, uint32_t indexBytes, int64_t offset, int64_t mode, uint64_t outputInner,
                                   SelectTopKBaseTilingData& plan, SortSchedule& schedule)
{
    using namespace SelectTopKBase;
    if (plan.rows != 1 || plan.axis <= 1 || plan.axis > SINGLE_ROW_MAX_AXIS || plan.k == 0 ||
        (plan.axis > TINY_ROW_ITEMS && plan.kernel == KernelTemplate::COPY_ALL) ||
        plan.kernel == KernelTemplate::TOP1 || indexBytes != INDEX64_BYTES || offset != 0 || mode != OUTPUT_BOTH ||
        outputInner != 1)
        return;
    auto candidate = SelectionInput(plan);
    // 容量作为 tiling 数据，由同一入口在内部选择固定容量实现。
    const uint32_t slots = plan.axis <= SORT32_ELEMENTS ?
                               SORT32_ELEMENTS :
                               (plan.axis <= TINY_ROW_ITEMS ? TINY_ROW_ITEMS : SINGLE_ROW_MAX_AXIS);
    const uint32_t keySlots = Align(slots, FLOAT_VECTOR_ELEMENTS);
    const uint64_t bytes = uint64_t(slots) * (plan.dtypeBytes + INDEX32_BYTES + DOUBLE_PAIR_BYTES) +
                           uint64_t(keySlots) * VALUE32_BYTES;
    if (bytes > ub - UB_RESERVE)
        return;
    candidate.kernel = KernelTemplate::TINY_ROW;
    candidate.route = ROUTE_COMPACT_ROW_SORT;
    candidate.tile = slots;
    candidate.tiles = candidate.shardCount = candidate.cores = 1;
    candidate.outputTile = plan.k;
    candidate.ubBytes = bytes;
    plan = candidate;
    schedule = {};
}

// 已核验的标准单行最多16项时，使用固定标量缓冲，省去向量索引、掩码及归并初始化。
inline void TryScalarRowOptimization(uint64_t ub, SelectTopKBaseTilingData& plan)
{
    using namespace SelectTopKBase;
    if (plan.kernel != KernelTemplate::TINY_ROW || plan.tile != SORT32_ELEMENTS || plan.axis > SCALAR_ROW_ITEMS)
        return;
    // 无序且只舍弃一项继续使用线性压缩；其他较长短行避免标量插入排序的二次比较。
    if (plan.axis > SCALAR_SORT_MAX_ITEMS && (plan.sorted || plan.k + 1 != plan.axis))
        return;
    const ScalarRowLayout layout(plan.dtypeBytes);
    if (layout.bytes > ub - UB_RESERVE)
        return;
    plan.tile = SCALAR_ROW_ITEMS;
    plan.ubBytes = layout.bytes;
}
// 连续标准多行在归并层间只保留64项；容量、行数及范围由输入特征决定。
inline void TryPrefixRowOptimization(uint64_t ub, uint32_t cores, uint32_t indexBytes, int64_t offset, int64_t mode,
                                     uint64_t outputInner, uint64_t inputInner, SelectTopKBaseTilingData& plan,
                                     SortSchedule& schedule)
{
    using namespace SelectTopKBase;
    if (plan.kernel != KernelTemplate::ROW_SORT || indexBytes != INDEX64_BYTES || offset != 0 || mode != OUTPUT_BOTH ||
        outputInner != 1 || inputInner != 1 || plan.axis <= PREFIX_ROW_MIN_AXIS || plan.axis > PREFIX_ROW_WIDTH ||
        plan.k <= SORT32_ELEMENTS || plan.k > PREFIX_ROW_ITEMS ||
        plan.rows < static_cast<uint64_t>(cores) * PREFIX_ROW_MIN_BATCHES)
        return;
    const uint32_t rowBytes = ROW_SORT_ARRAY_BYTES + ROW_INPUT_BUFFER_COUNT * plan.dtypeBytes;
    const uint32_t capacity = std::min<uint64_t>(SORT32_MAX_ELEMENTS, (ub - UB_RESERVE) / rowBytes);
    // 首级四路Sort32组已经可以独立裁剪；末级不足四路由同一模板处理。
    const uint32_t width = Align(plan.axis, SORT32_ELEMENTS * MERGE_WAYS);
    const uint32_t batch = std::min<uint64_t>(schedule.rowsPerCore, capacity / width);
    if (batch == 0)
        return;
    plan.kernel = KernelTemplate::PREFIX_ROWS;
    plan.batchRows = batch;
    plan.tile = batch * width;
    plan.ubBytes = plan.tile * rowBytes;
    plan.outputTile = PREFIX_ROW_ITEMS;
    schedule.rowWidth = width;
}
} // namespace SelectTopK
#endif
