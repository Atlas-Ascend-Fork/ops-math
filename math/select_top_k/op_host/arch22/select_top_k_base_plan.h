/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SELECT_TOP_K_BASE_PLAN_H
#define SELECT_TOP_K_BASE_PLAN_H

#include <algorithm>
#include <cstdint>
#include <limits>

#include "../../op_kernel/select_top_k_base_tiling_data.h"
#include "../../op_kernel/select_top_k_base_select_layout.h"

namespace SelectTopKBase {
using SelectTopK::CandidateMerge;
using SelectTopK::KernelTemplate;
using SelectTopK::SelectPartition;
// 核数及 UB 总容量由本次平台资源查询获得；调优常量见 select_top_k_base_config.h。
inline uint64_t CeilDiv(uint64_t n, uint64_t d)
{
    // 避免使用 (n + d - 1) / d 时，加法先发生溢出。
    return n / d + (n % d != 0);
}

// 将元素数或字节数向上对齐到指定粒度。
inline uint64_t Align(uint64_t n, uint64_t alignment) { return CeilDiv(n, alignment) * alignment; }

// 检查无符号乘法溢出，成功时写出乘积。
inline bool Mul(uint64_t a, uint64_t b, uint64_t& out)
{
    if (b != 0 && a > std::numeric_limits<uint64_t>::max() / b) {
        return false;
    }
    out = a * b;
    return true;
}

// 选择与页数互质的扫描步长，使各页恰好访问一次。
inline uint32_t SelectPageStep(uint32_t pages)
{
    // 与页数互质的步长构成全排列；不依赖输入数值，也不分配 N 规模的页索引表。
    uint32_t step = std::max<uint32_t>(
        1, (static_cast<uint64_t>(pages) * Tuning::PAGE_STEP_MULTIPLIER) >> MASK_WORD32_BITS);
    while (true) {
        uint32_t a = pages;
        uint32_t b = step;
        while (b != 0) {
            const uint32_t remainder = a % b;
            a = b;
            b = remainder;
        }
        if (a == 1) {
            return step;
        }
        ++step;
    }
}

// 根据通用归并布局返回完整 UB 占用字节数。
inline uint64_t UbBytes(uint32_t tile, uint32_t window, uint64_t k, uint32_t dtypeBytes, uint32_t route)
{
    return GeneralMergeLayout(tile, window, k, dtypeBytes, route).bytes;
}

// 从大到小选择满足 UB 预算的排序块与归并窗口。
inline bool SelectTile(uint64_t usableUbBytes, uint32_t dtypeBytes, uint64_t k, uint32_t route, uint32_t& tile,
                       uint32_t& window, uint64_t& usedUbBytes)
{
    // Sort32 每次处理 32 个元素，repeat 最大 255；从大到小选择能容纳全部缓冲的 tile。
    for (tile = SORT32_MAX_ELEMENTS; tile >= SORT32_ELEMENTS; tile -= SORT32_ELEMENTS) {
        // 窗口取 tile 的 1/8，再向下对齐到 4 个 pair，保证每路窗口的 32B 对齐。
        window = std::min<uint32_t>(MERGE_MAX_ALIGNED_ITEMS,
                                    std::max<uint32_t>(PAIR_ALIGNMENT, tile / Tuning::GENERAL_WINDOW_DIVISOR /
                                                                           PAIR_ALIGNMENT * PAIR_ALIGNMENT));
        if (route == ROUTE_GM_MERGE) {
            // 流式归并与叶子排序分时复用 UB；大 K 使用较宽窗口以摊薄 DMA 和归并准备开销。
            window = std::min<uint64_t>(std::min<uint32_t>(tile, MERGE_MAX_ALIGNED_ITEMS),
                                        usableUbBytes / MERGE_WORK_BYTES / PAIR_ALIGNMENT * PAIR_ALIGNMENT);
            if (window < PAIR_ALIGNMENT) {
                return false;
            }
        }
        usedUbBytes = UbBytes(tile, window, k, dtypeBytes, route);
        if (usedUbBytes <= usableUbBytes && usedUbBytes <= UINT32_MAX) {
            return true;
        }
    }
    return false;
}

// 资源预算与算法参数分离；usableUbBytes 已在 BuildPlan 中扣过一次系统预留。
struct PlanResources {
    uint64_t usableUbBytes;
    uint32_t cores;
    uint64_t smallKMinAxis = Tuning::FILTER_MIN_AXIS;
    uint32_t maxLeafTile = Tuning::MAX_LEAF_BLOCK;
    bool allowWideRowBatch = false;
};

enum class PlanAttempt { NOT_APPLICABLE, SUCCESS, INVALID };

// 判断 N/K 是否落在过滤选择允许的范围。
inline bool IsFilterRange(const SelectTopKBaseTilingData& plan, const PlanResources& resources)
{
    const uint64_t minAxis = plan.k <= Tuning::SMALL_K ? resources.smallKMinAxis : Tuning::FILTER_MIN_AXIS;
    return plan.axis >= minAxis && plan.k >= Tuning::FILTER_MIN_K && plan.k <= Tuning::FILTER_MAX_K;
}

// 短轴每行补到 32 项，用一个 Sort32 repeat 独立处理一行。
inline PlanAttempt TryShortRowBatch(SelectTopKBaseTilingData& plan, const PlanResources& resources)
{
    if (!(plan.axis > 1 && plan.axis <= SORT32_ELEMENTS && (plan.sorted || plan.k != plan.axis))) {
        return PlanAttempt::NOT_APPLICABLE;
    }

    // R1 短轴子路径：每行补至 32 项，Sort32 的每个 repeat 排一行，无需归并。
    // raw、FP32 key、UINT32 index 和交错 pair 同时存活；所有行天然按 32B 对齐。
    const uint64_t bytesPerRow = SORT32_ELEMENTS_64 * (plan.dtypeBytes + VALUE32_BYTES + INDEX32_BYTES + PAIR_BYTES);
    const uint64_t maxRowsPerBatch = std::min<uint64_t>(VECTOR_MAX_REPEATS, resources.usableUbBytes / bytesPerRow);
    if (maxRowsPerBatch == 0) {
        return PlanAttempt::INVALID;
    }
    // 短轴优先让每核一次处理至少 16 行，避免少量数据分散到大量核后的小块 DMA 争用。
    // 大输入仍按实际可用核数均分；小 UB 和 repeat 上限始终优先约束批量。
    const uint64_t preferredRows = std::max<uint64_t>(Tuning::SHORT_ROWS_MIN_BATCH,
                                                      CeilDiv(plan.rows, resources.cores));
    const uint64_t rowsPerBatch = std::min(maxRowsPerBatch, std::min(plan.rows, preferredRows));
    plan.route = ROUTE_ROW_SORT;
    plan.tile = static_cast<uint32_t>(rowsPerBatch * SORT32_ELEMENTS);
    plan.batchRows = static_cast<uint32_t>(rowsPerBatch);
    plan.kernel = KernelTemplate::SHORT_ROWS;
    plan.outputTile = SORT32_ELEMENTS;
    plan.tiles = 1;
    plan.ubBytes = static_cast<uint32_t>(rowsPerBatch * bytesPerRow);
    plan.cores = static_cast<uint32_t>(std::min<uint64_t>(resources.cores, CeilDiv(plan.rows, rowsPerBatch)));
    return PlanAttempt::SUCCESS;
}

// 行并行充分时，每核独立维护本行 TopK 阈值和候选。
inline PlanAttempt TryRowFilter(SelectTopKBaseTilingData& plan, const PlanResources& resources)
{
    if (!(IsFilterRange(plan, resources) && plan.rows >= resources.cores)) {
        return PlanAttempt::NOT_APPLICABLE;
    }

    // 分阶段复用候选/排序临时区后，大 K 也尝试 2048；实际放不下时继续逐级减半。
    const uint32_t preferredBlock = Tuning::FILTER_PREFERRED_BLOCK;
    for (uint32_t block = preferredBlock; block >= Tuning::FILTER_MIN_BLOCK; block /= HALF_DIVISOR) {
        const SelectLayout layout(block, static_cast<uint32_t>(plan.k), plan.dtypeBytes);
        if (layout.bytes <= resources.usableUbBytes) {
            plan.route = ROUTE_FILTER_MERGE;
            plan.kernel = KernelTemplate::SELECT;
            plan.partition = SelectPartition::ROWS;
            plan.cores = resources.cores;
            plan.tile = block;
            plan.pendingLimit = block;
            plan.outputTile = std::min<uint32_t>(block, static_cast<uint32_t>(plan.k));
            plan.tiles = CeilDiv(plan.axis, block);
            plan.pageStep = SelectPageStep(static_cast<uint32_t>(plan.tiles));
            plan.ubBytes = layout.bytes;
            return PlanAttempt::SUCCESS;
        }
    }
    // UB 无法容纳最小扫描块及完整 Acc 时，沿用原有 R1/R2/R3 规划与错误处理。

    return PlanAttempt::NOT_APPLICABLE;
}

// 少行小 K 采用宽块过滤；每行按实际核数分片。
inline PlanAttempt TryWideFilter(SelectTopKBaseTilingData& plan, const PlanResources& resources)
{
    if (!(IsFilterRange(plan, resources) && (plan.rows < resources.cores || resources.allowWideRowBatch) &&
          plan.k <= Tuning::SMALL_K && plan.rows > 1)) {
        return PlanAttempt::NOT_APPLICABLE;
    }

    // 小 K 的候选很少，优先扩大一次扫描量；B 由实际 UB 决定，不能固定为某个芯片容量。
    // 每项三份 4B 数组、1bit mask，另留完整 Acc；精确布局检查负责对齐后的边界。
    const uint64_t accumulatorBytes = Align(plan.k, SORT32_ELEMENTS) * PAIR_BYTES;
    uint32_t block = resources.usableUbBytes > accumulatorBytes ?
                         static_cast<uint32_t>(std::min<uint64_t>((resources.usableUbBytes - accumulatorBytes) *
                                                                      BITS_PER_BYTE / WIDE_SCAN_BITS_PER_ITEM /
                                                                      FLOAT_VECTOR_ELEMENTS * FLOAT_VECTOR_ELEMENTS,
                                                                  FLOAT_VECTOR_ELEMENTS * VECTOR_MAX_REPEATS)) :
                         0;
    for (; block >= Tuning::WIDE_MIN_BLOCK; block -= FLOAT_VECTOR_ELEMENTS) {
        const WideSelectLayout layout(block, static_cast<uint32_t>(plan.k));
        if (layout.bytes > resources.usableUbBytes) {
            continue;
        }
        const uint32_t seed = plan.k <= Tuning::TINY_K ? Tuning::WIDE_TINY_SEED : Tuning::WIDE_SMALL_SEED;
        // 仅当每行在建立初始候选后只需扫描一次时，才将多行合并为一个批次。
        if (plan.rows >= resources.cores && plan.axis > block + seed) {
            return PlanAttempt::NOT_APPLICABLE;
        }
        const uint64_t shards = std::min<uint64_t>(std::max<uint64_t>(1, resources.cores / plan.rows),
                                                   CeilDiv(plan.axis, block + seed));
        plan.route = ROUTE_WIDE_FILTER;
        plan.kernel = KernelTemplate::WIDE_SELECT;
        plan.cores = static_cast<uint32_t>(std::min<uint64_t>(resources.cores, plan.rows * shards));
        plan.tile = block;
        plan.seedItems = seed;
        plan.candidateSortTile = Tuning::WIDE_SORT_CHUNK;
        while (plan.candidateSortTile * DOUBLE_PAIR_BYTES > block * VALUE32_BYTES) {
            plan.candidateSortTile /= HALF_DIVISOR;
        }
        plan.tiles = CeilDiv(plan.axis, block);
        plan.shardCount = static_cast<uint32_t>(shards);
        plan.ubBytes = layout.bytes;
        if (shards > 1) {
            plan.rowWorkspacePairs = shards * Align(plan.k, UB_BLOCK_BYTES / PAIR_BYTES);
            if (!Mul(plan.rows, plan.rowWorkspacePairs * PAIR_BYTES, plan.workspaceBytes)) {
                return PlanAttempt::INVALID;
            }
        }
        return PlanAttempt::SUCCESS;
    }

    return PlanAttempt::NOT_APPLICABLE;
}

// 汇总各叶子的对齐候选数，保留调用方的最低容量；上限及 UB 校验仍由各规划规则决定。
inline uint64_t LeafCandidateCapacity(uint64_t shards, uint64_t k, uint32_t minimum)
{
    return std::max<uint64_t>(minimum, Align(shards * Align(k, PAIR_ALIGNMENT), SORT32_ELEMENTS));
}

// 每片能一次排序且独占一核时，直接局部排序再归并候选。
inline PlanAttempt TryLeafSort(SelectTopKBaseTilingData& plan, const PlanResources& resources)
{
    if (!(IsFilterRange(plan, resources) && plan.rows < resources.cores)) {
        return PlanAttempt::NOT_APPLICABLE;
    }
    // 所有片均能在 UB 内一次排序，且每片有独立 AIV 时，直接局部排序并保留 TopK。
    // block 仍受实际 UB/核数约束；条件不满足继续采用下方多页过滤。
    // 单行小 K 用更多叶子并行；多行宽块筛选优先，资源不足时仍能落到本分支。
    const uint32_t firstLeafBlock = plan.k <= Tuning::TINY_K ?
                                        Tuning::TINY_LEAF_BLOCK :
                                        (plan.k <= Tuning::SMALL_K ? Tuning::SMALL_LEAF_BLOCK : Tuning::MAX_LEAF_BLOCK);
    const uint32_t maxLeafBlock = std::min<uint32_t>(resources.maxLeafTile, Tuning::MAX_EXTENDED_LEAF_BLOCK);
    // 大 K 优先减少候选列表数；小 K 保留最小可行叶子，仅在核数不足时尝试扩展叶子。
    const bool preferLarge = maxLeafBlock > Tuning::MAX_LEAF_BLOCK && plan.k >= Tuning::LARGE_LEAF_MIN_K;
    const uint32_t candidates[Tuning::LEAF_CANDIDATE_COUNT] = {
        preferLarge ? maxLeafBlock : 0, firstLeafBlock, firstLeafBlock * HALF_DIVISOR, firstLeafBlock * MERGE_WAYS,
        preferLarge ? 0 : maxLeafBlock};
    for (uint32_t attempt = 0; attempt < Tuning::LEAF_CANDIDATE_COUNT; ++attempt) {
        const uint32_t block = candidates[attempt];
        if (block < firstLeafBlock || block > maxLeafBlock ||
            (attempt > 0 && attempt < Tuning::LAST_REGULAR_LEAF_CANDIDATE && block > Tuning::MAX_LEAF_BLOCK)) {
            continue;
        }
        const uint64_t shards = CeilDiv(plan.axis, block);
        // 优先一次读入全部片候选；受 4096 项归并容量与实际 UB 约束。
        uint32_t finalCapacity = static_cast<uint32_t>(
            std::min<uint64_t>(block > Tuning::MAX_LEAF_BLOCK ? maxLeafBlock : Tuning::MAX_LEAF_BLOCK,
                               LeafCandidateCapacity(shards, plan.k, block)));
        LeafSelectLayout layout(block, static_cast<uint32_t>(plan.k), plan.dtypeBytes, finalCapacity);
        if (layout.bytes > resources.usableUbBytes) {
            finalCapacity = block;
            layout = LeafSelectLayout(block, static_cast<uint32_t>(plan.k), plan.dtypeBytes, finalCapacity);
        }
        if (layout.bytes > resources.usableUbBytes || plan.k > block || shards > resources.cores / plan.rows) {
            continue;
        }
        plan.route = ROUTE_SHARDED_SELECT;
        plan.cores = static_cast<uint32_t>(plan.rows * shards);
        plan.tile = block;
        plan.kernel = KernelTemplate::LEAF_SELECT;
        plan.candidateMerge = plan.k >= Tuning::PREFIX_MERGE_MIN_K ? CandidateMerge::HALF_PREFIX :
                                                                     CandidateMerge::FULL_LISTS;
        plan.finalCapacity = finalCapacity;
        plan.tiles = shards;
        plan.shardCount = static_cast<uint32_t>(shards);
        plan.ubBytes = layout.bytes;
        plan.rowWorkspacePairs = shards * Align(plan.k, UB_BLOCK_BYTES / PAIR_BYTES);
        if (!Mul(plan.rows, plan.rowWorkspacePairs * PAIR_BYTES, plan.workspaceBytes)) {
            return PlanAttempt::INVALID;
        }
        return PlanAttempt::SUCCESS;
    }

    return PlanAttempt::NOT_APPLICABLE;
}

// 其余少行特化按多页分片过滤，最终合并各片有序 TopK。
inline PlanAttempt TryShardedFilter(SelectTopKBaseTilingData& plan, const PlanResources& resources)
{
    if (!(IsFilterRange(plan, resources) && plan.rows < resources.cores)) {
        return PlanAttempt::NOT_APPLICABLE;
    }
    // 少行长轴按页分片，每片独立过滤并保留 TopK；最后只合并各片候选。
    // 小 K 允许单块分片缩短扫描；较大 K 保留至少约 4096 项，限制候选归并量。
    for (uint32_t block = Tuning::FILTER_PREFERRED_BLOCK; block >= Tuning::FILTER_MIN_BLOCK; block /= HALF_DIVISOR) {
        const SelectLayout layout(block, static_cast<uint32_t>(plan.k), plan.dtypeBytes);
        // 最终归并需同时容纳两份 K；ping 至少能读入一份完整局部候选。
        if (layout.bytes > resources.usableUbBytes || layout.capacity < plan.k) {
            continue;
        }
        const uint64_t pages = CeilDiv(plan.axis, block);
        const uint64_t minShardItems = plan.k <= Tuning::SMALL_K ? Tuning::SMALL_K_MIN_SHARD_ITEMS :
                                                                   Tuning::LARGE_K_MIN_SHARD_ITEMS;
        const uint64_t maxShards = std::max<uint64_t>(1, plan.axis / minShardItems);
        const uint64_t shards = std::min<uint64_t>(resources.cores / plan.rows, maxShards);
        plan.route = ROUTE_SHARDED_SELECT;
        plan.cores = static_cast<uint32_t>(std::min<uint64_t>(resources.cores, plan.rows * shards));
        plan.tile = block;
        plan.pendingLimit = block;
        plan.kernel = KernelTemplate::SELECT;
        plan.partition = SelectPartition::SHARDS;
        plan.outputTile = std::min<uint32_t>(block, static_cast<uint32_t>(plan.k));
        plan.tiles = pages;
        plan.shardCount = static_cast<uint32_t>(shards);
        plan.ubBytes = layout.bytes;
        // 扫描结束后复用 arena；四路前缀的输入/输出均须容纳，复制前 K 项不能碰到活跃 Acc。
        const uint64_t prefix = (plan.k + (HALF_PREFIX_ALIGNMENT - 1)) / HALF_PREFIX_ALIGNMENT * PAIR_ALIGNMENT;
        const uint64_t stride = Align(plan.k, PAIR_ALIGNMENT);
        const bool prefixFits = prefix * MERGE_WORK_BYTES <= layout.bytes &&
                                prefix * MERGE_INPUT_BYTES + stride * PAIR_BYTES <= layout.accumulator;
        if (plan.k >= Tuning::PREFIX_MERGE_MIN_K && shards >= Tuning::PREFIX_MIN_SHARDS && prefixFits) {
            plan.candidateMerge = CandidateMerge::HALF_PREFIX;
        }
        if (shards > 1) {
            // 每片槽位按 32B 对齐；仅一份 GM 候选平面，最终归并直接写输出。
            plan.rowWorkspacePairs = shards * Align(plan.k, UB_BLOCK_BYTES / PAIR_BYTES);
            if (!Mul(plan.rows, plan.rowWorkspacePairs * PAIR_BYTES, plan.workspaceBytes)) {
                return PlanAttempt::INVALID;
            }
        }
        return PlanAttempt::SUCCESS;
    }

    return PlanAttempt::NOT_APPLICABLE;
}

// 通用路径：整行排序 -> UB 累积；容量或行并行度不足时回退到 GM 分层归并。
inline bool PlanGeneralMerge(SelectTopKBaseTilingData& plan, const PlanResources& resources, uint64_t elements)
{
    uint64_t usedUbBytes = 0;
    uint32_t tile = 0;
    uint32_t window = 0;
    // R0：全量且不要求排序，或轴长为 1；其余场景先按 R1 的缓冲布局选择 tile。
    const bool copyOnly = (!plan.sorted && plan.k == plan.axis) || plan.axis == 1;
    plan.route = copyOnly ? ROUTE_COPY : ROUTE_ROW_SORT;
    if (!SelectTile(resources.usableUbBytes, plan.dtypeBytes, plan.k, plan.route, tile, window, usedUbBytes)) {
        return false;
    }

    if (!copyOnly && plan.axis > tile) {
        uint32_t accumulateTile = 0;
        uint32_t accumulateWindow = 0;
        uint64_t accumulateUbBytes = 0;
        // R2：行数足以覆盖可用 AIV，且两份 Acc 连同全部临时区都能放入 UB。
        // 加入 Acc 后必须重新选择 tile，不能沿用 R1 的容量计算。
        if (plan.rows >= resources.cores &&
            // 从大到小选择满足 UB 预算的排序块与归并窗口。
            SelectTile(resources.usableUbBytes, plan.dtypeBytes, plan.k, ROUTE_UB_ACCUMULATE, accumulateTile,
                       accumulateWindow, accumulateUbBytes) &&
            accumulateTile >= plan.k) {
            plan.route = ROUTE_UB_ACCUMULATE;
            tile = accumulateTile;
            window = accumulateWindow;
            usedUbBytes = accumulateUbBytes;
        } else {
            // R3：行并行度不足或 UB 放不下 Acc，通过 GM 候选段进行分层归并。
            plan.route = ROUTE_GM_MERGE;
            if (!SelectTile(resources.usableUbBytes, plan.dtypeBytes, plan.k, plan.route, tile, window, usedUbBytes)) {
                return false;
            }
        }
    }

    plan.tile = tile;
    plan.mergeWindow = window;
    plan.outputTile = std::min(tile, window);
    plan.ubBytes = static_cast<uint32_t>(usedUbBytes);
    plan.tiles = CeilDiv(plan.axis, tile);

    // R1/R2 按行分配任务；R0 可以跨行按连续片段分配任务。
    uint64_t tasks = plan.rows;
    if (plan.route == ROUTE_COPY) {
        tasks = CeilDiv(elements, tile);
    }
    if (plan.route == ROUTE_GM_MERGE) {
        // 每个叶子段最多保留 min(plan.k, tile) 个 pair；每行的固定槽位覆盖全部叶子段。
        const uint64_t leafCandidatePairs = std::min<uint64_t>(plan.k, tile);
        uint64_t rowCandidatePairs = 0;
        if (!Mul(plan.tiles, leafCandidatePairs, rowCandidatePairs) ||
            rowCandidatePairs > INT64_MAX / PAIR_BYTES - PAIR_ALIGNMENT) {
            return false;
        }
        // 每行按 4 个 pair 对齐，使两份 GM 平面中的各行起点都满足 32B 对齐。
        plan.rowWorkspacePairs = Align(rowCandidatePairs, UB_BLOCK_BYTES / PAIR_BYTES);
        uint64_t planePairs = 0;
        if (!Mul(plan.rows, plan.rowWorkspacePairs, planePairs)) {
            return false;
        }
        // GM ping/pong 各一份：plan.rows * rowWorkspacePairs * 8B * 2；不含框架系统 workspace。
        if (!Mul(planePairs, DOUBLE_BUFFER_COUNT * PAIR_BYTES, plan.workspaceBytes) ||
            plan.workspaceBytes > INT64_MAX) {
            return false;
        }
        // R3 的叶子排序按“行 × tile”切任务，短行数场景也可以使用多个 AIV。
        if (!Mul(plan.rows, plan.tiles, tasks)) {
            return false;
        }
    }

    // 核数由实际任务量和平台可用 AIV 共同决定，不固定核数。
    plan.cores = static_cast<uint32_t>(std::min<uint64_t>(resources.cores, tasks));
    if (plan.route == ROUTE_GM_MERGE) {
        // 输出阶段可复用完整叶子缓冲；同时保留足够输出块，使工作分布到所有活动核。
        plan.outputTile = std::min<uint64_t>(
            tile, std::max<uint64_t>(window, Align(CeilDiv(plan.rows * plan.k, plan.cores), SORT32_ELEMENTS)));
    }
    return true;
}

} // namespace SelectTopKBase
#endif
