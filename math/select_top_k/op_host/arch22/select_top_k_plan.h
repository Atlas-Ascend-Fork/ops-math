/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SELECT_TOP_K_PLAN_H_
#define SELECT_TOP_K_PLAN_H_
#include "../../op_kernel/select_top_k_tiling_data.h"
#include "../../op_kernel/select_top_k_output_layout.h"
#include "../../op_kernel/select_top_k_output_reuse_layout.h"
#include "select_top_k_targeted_plan.h"
#include <type_traits>

namespace SelectTopK {
using namespace SelectTopKBase;
// 为一个工作区片段计算 32B 对齐后的偏移，并提前检查地址范围溢出。
inline bool AddWorkspace(uint64_t count, uint32_t bytes, uint64_t& total, uint64_t& offset)
{
    uint64_t size = 0;
    if (!SelectTopKBase::Mul(count, bytes, size) || size > INT64_MAX - (UB_BLOCK_BYTES - 1) ||
        total > INT64_MAX - size - (UB_BLOCK_BYTES - 1)) {
        return false;
    }
    offset = total;
    total += SelectTopKBase::Align(size, UB_BLOCK_BYTES);
    return true;
}

// 按原有增长序列搜索输出列块，统一检查并行度、Gather 容量与 UB 预算。
template <bool Reuse>
inline bool PlanOutputLayout(uint64_t ub, uint32_t indicesBytes, SelectTopKTilingData& data)
{
    using Layout = typename std::conditional<Reuse, OutputReuseLayout, OutputLayout>::type;
    const uint64_t inner = data.outputInner, k = data.outputK;
    const uint32_t bytes = data.base.dtypeBytes;
    const uint64_t columnsPerCore = CeilDiv((data.base.rows / inner) * k, std::max<uint32_t>(1, data.base.cores));
    const uint32_t limit = Reuse ? MAX_REUSED_OUTPUT_BLOCK :
                                   (inner >= GM_MIN_INNER ? (k > LARGE_K_THRESHOLD ? WIDE_OUTPUT_BLOCK : GM_MIN_INNER) :
                                                            MAX_OUTPUT_BLOCK);
    const uint32_t capacity = Reuse ? MAX_REUSED_CAPACITY : MAX_OUTPUT_CAPACITY;
    uint32_t block = MIN_LAYOUT_BLOCK;
    auto layout = Layout(inner, block, k, bytes, indicesBytes);
    if (layout.bytes > ub - UB_RESERVE)
        return false;
    while (block < limit && block < columnsPerCore) {
        const uint32_t nextBlock = Reuse && block == WIDE_OUTPUT_BLOCK ?
                                       REUSED_MID_BLOCK :
                                       (Reuse && block == REUSED_MID_BLOCK ? MAX_REUSED_OUTPUT_BLOCK :
                                                                             block * DOUBLE_BUFFER_COUNT);
        const auto next = Layout(inner, nextBlock, k, bytes, indicesBytes);
        if (next.capacity > capacity || next.bytes > ub - UB_RESERVE)
            break;
        block = nextBlock;
        layout = next;
    }
    data.outputBlock = block;
    data.base.ubBytes = std::max<uint32_t>(data.base.ubBytes, layout.bytes);
    return true;
}

// 校验完整接口，依次规划基础排序、过滤选择和已定位特化，再适配扩展收尾与输入输出布局。
inline bool BuildPlan(uint64_t rows, uint64_t axis, uint64_t k, uint32_t bytes, bool largest, bool sorted,
                      uint32_t indicesBytes, int64_t offset, int64_t mode, uint32_t boundsFlags, float valueFill,
                      int64_t indexFill, uint64_t ub, uint32_t cores, SelectTopKTilingData& data,
                      uint64_t outputInner = 1, uint64_t inputInner = 1)
{
    if (cores == 0 || ub <= SelectTopKBase::UB_RESERVE || k > INT32_MAX || axis > INT32_MAX ||
        (indicesBytes != INDEX32_BYTES && indicesBytes != INDEX64_BYTES) || mode < 0 || mode > OUTPUT_BOTH) {
        return false;
    }
    const int64_t upper = indicesBytes == INDEX32_BYTES ? INT32_MAX : INT64_MAX;
    const int64_t lower = indicesBytes == INDEX32_BYTES ? INT32_MIN : INT64_MIN;
    const int64_t maxIndex = axis == 0 ? 0 : static_cast<int64_t>(axis - 1);
    uint64_t inputItems = 0, outputItems = 0;
    if (offset < lower || offset > upper - maxIndex || indexFill < lower || indexFill > upper ||
        !SelectTopKBase::Mul(rows, axis, inputItems) || !SelectTopKBase::Mul(rows, k, outputItems) ||
        inputItems > INT64_MAX / INDEX64_BYTES || outputItems > INT64_MAX / INDEX64_BYTES) {
        return false;
    }
    if (outputInner == 0 || outputInner > MAX_LAYOUT_INNER || rows % outputInner != 0 ||
        (outputInner > 1 && (k <= 1 || k > axis || boundsFlags != 0 || mode != OUTPUT_BOTH || offset != 0))) {
        return false;
    }
    data = {};
    data.outputInner = outputInner;
    if (inputInner == 0 || inputInner > MAX_LAYOUT_INNER ||
        (inputInner > 1 && (inputInner != outputInner || axis <= 1)))
        return false;
    data.inputInner = inputInner;
    data.outputK = k;
    data.indicesOffset = offset;
    data.indexFill = indexFill;
    data.valueFill = valueFill;
    data.flags = boundsFlags;
    const bool ranged = (data.flags & (SelectTopK::HAS_BEGIN | SelectTopK::HAS_END)) != 0 || data.outputK > axis;
    if (ranged) {
        data.flags |= SelectTopK::RANGE;
    }
    const uint64_t selectedK = std::min<uint64_t>(axis, data.outputK);
    if (!BuildSortPlan(rows, axis, selectedK, bytes, largest, sorted, ub, cores, data.base, data.schedule)) {
        return false;
    }
    if (ranged)
        data.base.kernel = KernelTemplate::RANGE_MERGE;
    if (!ranged) {
        TryFilterSelectPlan(ub, cores, data.base, data.schedule);
        TryLeafOptimization(ub, cores, data.base, data.schedule);
        TryBalancedLargePrefix(ub, cores, data.base);
        TryDenseLeafPrefix(ub, cores, data.base);
        TryTop1Optimization(ub, cores, data.base, data.schedule);
        TryShortRowOptimization(ub, cores, indicesBytes, offset, mode, outputInner, data.base, data.schedule);
        TryCompactRowOptimization(ub, cores, data.base, data.schedule);
        TryCopyAllOptimization(ub, cores, indicesBytes, offset, mode, outputInner, data.base, data.schedule);
        TryResidentMergeOptimization(ub, cores, data.base, data.schedule);
        TryGroupedResidentOptimization(ub, cores, data.base, data.schedule);
        TryTwoLeafLocalOptimization(ub, cores, data.base, data.schedule);
        TryTinyRowOptimization(ub, indicesBytes, offset, mode, outputInner, data.base, data.schedule);
        TryScalarRowOptimization(ub, data.base);
        TryPrefixRowOptimization(ub, cores, indicesBytes, offset, mode, outputInner, inputInner, data.base,
                                 data.schedule);
    }
    if (ranged && rows != 0 && selectedK != 0) {
        // 区间长度来自设备输入，按每行真实有效长度做受限归并，避免用数值哨兵表示被排除的元素。
        data.base = {};
        data.base.rows = rows;
        data.base.axis = axis;
        data.base.k = selectedK;
        data.base.dtypeBytes = bytes;
        data.base.largest = largest;
        data.base.sorted = true;
        if (!SelectTopKBase::PlanGeneralMerge(data.base, {ub - SelectTopKBase::UB_RESERVE, cores}, rows * axis)) {
            return false;
        }
        if (data.base.route == SelectTopKBase::ROUTE_COPY) {
            data.base.route = SelectTopKBase::ROUTE_ROW_SORT;
            if (!SelectTopKBase::SelectTile(ub - SelectTopKBase::UB_RESERVE, bytes, selectedK, data.base.route,
                                            data.base.tile, data.base.mergeWindow, data.workspaceBytes)) {
                return false;
            }
            data.base.ubBytes = static_cast<uint32_t>(data.workspaceBytes);
            data.base.outputTile = std::min(data.base.tile, data.base.mergeWindow);
            data.base.cores = std::min<uint64_t>(rows, cores);
        }
    }
    const bool reuseOutput = !ranged && outputInner >= GM_MIN_INNER &&
                             selectedK > SelectTopKBase::Tuning::MAX_LEAF_BLOCK && indicesBytes == INDEX64_BYTES;
    if (outputInner > 1) {
        const bool planned = reuseOutput ? PlanOutputLayout<true>(ub, indicesBytes, data) :
                                           PlanOutputLayout<false>(ub, indicesBytes, data);
        if (!planned)
            return false;
    }
    data.workspaceBytes = SelectTopKBase::Align(data.base.workspaceBytes, UB_BLOCK_BYTES);
    const bool values = mode != 1, indices = mode != 0;
    if (values) {
        data.flags |= SelectTopK::WRITE_VALUES;
    }
    if (indices) {
        data.flags |= SelectTopK::WRITE_INDICES;
    }
    const bool directIndices64 = outputInner == 1 && !ranged && values && indices && indicesBytes == INDEX64_BYTES &&
                                 offset == 0;
    if (directIndices64) {
        data.flags |= SelectTopK::DIRECT_INDICES64;
    }
    const bool tempValues = ranged || !values || outputInner > 1;
    const bool tempIndices = !directIndices64 &&
                             (ranged || !indices || indicesBytes == INDEX64_BYTES || offset != 0 || outputInner > 1);
    uint64_t items = 0;
    if (!SelectTopKBase::Mul(rows, selectedK, items)) {
        return false;
    }
    if (tempValues) {
        data.flags |= SelectTopK::TEMP_VALUES;
        if (!AddWorkspace(items, bytes, data.workspaceBytes, data.valuesWorkspaceOffset)) {
            return false;
        }
    }
    if (tempIndices) {
        data.flags |= SelectTopK::TEMP_INDICES;
        if (!AddWorkspace(items, INDEX32_BYTES, data.workspaceBytes, data.indicesWorkspaceOffset)) {
            return false;
        }
    }
    if (ranged || (indices && tempIndices) || outputInner > 1) {
        data.flags |= SelectTopK::FINALIZE;
    }
    if (data.base.cores == 0) {
        data.base.cores = std::max<uint32_t>(1, std::min<uint64_t>(rows, cores));
    }
    if (inputInner > 1) {
        uint32_t block = MIN_LAYOUT_BLOCK;
        const uint64_t perColumn = inputInner * (DOUBLE_BUFFER_COUNT * bytes + INDEX32_BYTES);
        if (block * perColumn > ub - SelectTopKBase::UB_RESERVE)
            return false;
        while (block < MAX_INPUT_BLOCK && block < axis &&
               block * DOUBLE_BUFFER_COUNT * inputInner <= MAX_REUSED_CAPACITY &&
               block * DOUBLE_BUFFER_COUNT * perColumn <= ub - SelectTopKBase::UB_RESERVE) {
            block *= DOUBLE_BUFFER_COUNT;
        }
        data.inputBlock = block;
        data.base.ubBytes = std::max<uint32_t>(data.base.ubBytes, block * perColumn);
        if (!AddWorkspace(inputItems, bytes, data.workspaceBytes, data.inputWorkspaceOffset))
            return false;
    }
    // 输出布局属于基础路径的适配阶段，不引入性能特化 key。
    if (reuseOutput) {
        data.flags |= SelectTopK::REUSE_LAYOUT;
    }
    return true;
}
} // namespace SelectTopK
#endif
