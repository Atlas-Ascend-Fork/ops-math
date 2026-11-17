/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SELECT_TOP_K_ENTRY_H_
#define SELECT_TOP_K_ENTRY_H_
#include "select_top_k_finalize.h"
#include "select_top_k_reformat.h"
#include "select_top_k_prepare_input.h"
#include "select_top_k_row_sort.h"
#include "select_top_k_prefix_rows.h"
#include "select_top_k_top1.h"
#include "select_top_k_copy_all.h"
#include "select_top_k_tiny_row.h"
#include "select_top_k_scalar_row.h"
#include "select_top_k_base_merge.h"
#include "select_top_k_base_select.h"
#include "select_top_k_base_wide_select.h"
#include "select_top_k_base_leaf_select.h"
#include "select_top_k_base_short_rows.h"

namespace SelectTopK {
using namespace SelectTopKBase;
// 独立单行入口只用固定UB视图和成对硬事件；保留硬件初始化及末尾等待，省去队列管理。
template <typename T, uint32_t MaximumItems>
__aicore__ inline void RunTinyRowCapacity(GM_ADDR x, GM_ADDR values, GM_ADDR indices,
                                          const SelectTopKBaseTilingData& data)
{
#if defined(__ASCENDC_ENABLE_SUPER_KERNEL__) || defined(ASCENDC_DEBUG) || \
    (defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1)
    // 融合或调试模式沿用框架上下文；普通A2入口不使用队列及动态工作缓冲。
    TPipe pipe;
#else
    InitSocState();
#endif
    if constexpr (MaximumItems == SCALAR_ROW_ITEMS) {
        ScalarRow<T> op;
        op.Run(x, values, indices, data);
    } else {
        TinyRow<T, MaximumItems> op;
        op.Run(x, values, indices, data);
    }
#if !defined(__ASCENDC_ENABLE_SUPER_KERNEL__) && !defined(ASCENDC_DEBUG) && \
    !(defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1)
    // 两份输出写回后才结束入口，不依赖自动插入同步或TPipe析构。
    PipeBarrier<PIPE_ALL>();
#endif
}

// 一次读取少量 tiling 字段，再按容量选择内部标量/向量实现；不搬运完整布局计划。
template <typename T>
__aicore__ inline void RunTinyRow(GM_ADDR x, GM_ADDR values, GM_ADDR indices, const __gm__ SelectTopKTilingData* source)
{
    SelectTopKBaseTilingData row{};
    row.axis = source->base.axis;
    row.k = source->base.k;
    row.largest = source->base.largest;
    row.sorted = source->base.sorted;
    row.ubBytes = source->base.ubBytes;
    const uint32_t capacity = source->base.tile;
    if (capacity == SCALAR_ROW_ITEMS)
        RunTinyRowCapacity<T, SCALAR_ROW_ITEMS>(x, values, indices, row);
    else if (capacity == SORT32_ELEMENTS)
        RunTinyRowCapacity<T, SORT32_ELEMENTS>(x, values, indices, row);
    else if (capacity == TINY_ROW_ITEMS)
        RunTinyRowCapacity<T, TINY_ROW_ITEMS>(x, values, indices, row);
    else
        RunTinyRowCapacity<T, SINGLE_ROW_MAX_AXIS>(x, values, indices, row);
}

// 固定核内/跨核调度和驻留布局后执行同一归并算法，模板特化只影响循环内代码。
template <typename T, bool Wide, SortPattern Pattern, MergeStorage Storage = MergeStorage::WINDOWS>
__aicore__ inline void ExecuteMergeSort(GM_ADDR x, GM_ADDR values, GM_ADDR indices, GM_ADDR workspace,
                                        const SelectTopKTilingData& data, TPipe* pipe)
{
    MergeSort<T, Wide, Pattern, Storage> op;
    op.Init(x, values, indices, workspace, data, pipe);
    if constexpr (Pattern == SortPattern::ROWS)
        op.ProcessCompactRows();
    else
        op.Process();
}

// 调度与驻留分别选择；不存在携带旧 key 编号的二次分发。
template <typename T, bool Wide>
__aicore__ inline void RunMergeSort(GM_ADDR x, GM_ADDR values, GM_ADDR indices, GM_ADDR workspace,
                                    const SelectTopKTilingData& data, TPipe* pipe)
{
    const SortPattern pattern = data.schedule.pattern;
    const MergeStorage storage = data.schedule.storage;
    if (pattern == SortPattern::ROWS) {
        ExecuteMergeSort<T, Wide, SortPattern::ROWS>(x, values, indices, workspace, data, pipe);
    } else if (storage == MergeStorage::ALL_CANDIDATES) {
        if (pattern == SortPattern::LOCAL_MERGE)
            ExecuteMergeSort<T, Wide, SortPattern::LOCAL_MERGE, MergeStorage::ALL_CANDIDATES>(x, values, indices,
                                                                                              workspace, data, pipe);
        else
            ExecuteMergeSort<T, Wide, SortPattern::CROSS_MERGE, MergeStorage::ALL_CANDIDATES>(x, values, indices,
                                                                                              workspace, data, pipe);
    } else if (storage == MergeStorage::GROUPED_CANDIDATES) {
        ExecuteMergeSort<T, Wide, SortPattern::CROSS_MERGE, MergeStorage::GROUPED_CANDIDATES>(x, values, indices,
                                                                                              workspace, data, pipe);
    } else if (storage == MergeStorage::TWO_LEAF_WINDOWS) {
        ExecuteMergeSort<T, Wide, SortPattern::LOCAL_MERGE, MergeStorage::TWO_LEAF_WINDOWS>(x, values, indices,
                                                                                            workspace, data, pipe);
    } else if (pattern == SortPattern::LOCAL_MERGE) {
        ExecuteMergeSort<T, Wide, SortPattern::LOCAL_MERGE>(x, values, indices, workspace, data, pipe);
    } else if (pattern == SortPattern::CROSS_MERGE) {
        ExecuteMergeSort<T, Wide, SortPattern::CROSS_MERGE>(x, values, indices, workspace, data, pipe);
    } else {
        ExecuteMergeSort<T, Wide, SortPattern::LOCAL_CROSS_MERGE>(x, values, indices, workspace, data, pipe);
    }
}

// 在内部函数边界固定候选策略，隔离寄存器和栈规划；该函数不生成额外 tiling key。
template <typename T, typename I, bool Direct, CandidateMerge Merge>
__aicore__ __attribute__((noinline)) void ExecuteLeafSelect(GM_ADDR x, GM_ADDR values, GM_ADDR indices,
                                                            GM_ADDR workspace, const SelectTopKTilingData& data,
                                                            TPipe* pipe)
{
    const bool wide = Direct || (std::is_same<I, int64_t>::value && (data.flags & DIRECT_INDICES64));
    if constexpr (Merge == CandidateMerge::FULL_LISTS) {
        TopKLeafSelect<T> op;
        op.Init(x, values, indices, workspace, &data.base, pipe, wide);
        op.Process();
    } else if constexpr (Merge == CandidateMerge::HALF_PREFIX) {
        TopKLeafSelect<T, true, true> op;
        op.Init(x, values, indices, workspace, &data.base, pipe, wide);
        op.Process();
    } else {
        TopKLeafSelect<T, true, Direct, DENSE_PREFIX_DIVISOR, true, Direct> op;
        op.Init(x, values, indices, workspace, &data.base, pipe, wide);
        op.Process();
    }
}

// 直接输出省去布局阶段；候选读取量在完整 LeafSelect 入口内部独立选择。
template <typename T, typename I, bool Direct>
__aicore__ inline void RunLeafSelect(GM_ADDR x, GM_ADDR values, GM_ADDR indices, GM_ADDR workspace,
                                     const SelectTopKTilingData& data, TPipe* pipe)
{
    if (data.base.candidateMerge == CandidateMerge::FULL_LISTS) {
        ExecuteLeafSelect<T, I, Direct, CandidateMerge::FULL_LISTS>(x, values, indices, workspace, data, pipe);
    } else if (data.base.candidateMerge == CandidateMerge::HALF_PREFIX) {
        ExecuteLeafSelect<T, I, Direct, CandidateMerge::HALF_PREFIX>(x, values, indices, workspace, data, pipe);
    } else {
        ExecuteLeafSelect<T, I, Direct, CandidateMerge::DENSE_PREFIX>(x, values, indices, workspace, data, pipe);
    }
}

// 完成计算阶段；每个模板的循环仍可按策略编译特化，公开 key 只标识完整算法族。
template <typename T, typename I, KernelTemplate Family>
__aicore__ inline void RunCore(GM_ADDR x, GM_ADDR begin, GM_ADDR end, GM_ADDR values, GM_ADDR indices,
                               GM_ADDR workspace, const SelectTopKTilingData& data, TPipe* pipe)
{
    const bool wide = std::is_same<I, int64_t>::value && (data.flags & DIRECT_INDICES64);
    if constexpr (Family == KernelTemplate::ROW_SORT) {
        if (wide) {
            RowSort<T, true> op;
            op.Init(x, values, indices, data, pipe);
            op.Process();
        } else {
            RowSort<T, false> op;
            op.Init(x, values, indices, data, pipe);
            op.Process();
        }
    } else if constexpr (Family == KernelTemplate::MERGE_SORT) {
        if (wide)
            RunMergeSort<T, true>(x, values, indices, workspace, data, pipe);
        else
            RunMergeSort<T, false>(x, values, indices, workspace, data, pipe);
    } else if constexpr (Family == KernelTemplate::RANGE_MERGE) {
        TopKMerge<T> op;
        op.Init(x, values, indices, workspace, &data.base, pipe);
        op.SetBounds(begin, end, data.flags);
        op.Process();
    } else if constexpr (Family == KernelTemplate::TOP1) {
        if (wide) {
            Top1<T, true> op;
            op.Run(x, values, indices, workspace, data.base, pipe);
        } else {
            Top1<T, false> op;
            op.Run(x, values, indices, workspace, data.base, pipe);
        }
    } else if constexpr (Family == KernelTemplate::SELECT) {
        TopKSelect<T> op;
        if (data.base.partition == SelectPartition::ROWS) {
            op.Init(x, values, indices, &data.base, pipe, wide);
            op.Process();
        } else {
            op.template Init<true>(x, values, indices, &data.base, pipe, wide);
            if (data.base.candidateMerge == CandidateMerge::HALF_PREFIX)
                op.template ProcessSharded<true>(workspace);
            else
                op.ProcessSharded(workspace);
        }
    } else if constexpr (Family == KernelTemplate::WIDE_SELECT) {
        TopKWideSelect<T> op;
        op.Init(x, values, indices, workspace, &data.base, pipe, wide);
        op.Process();
    } else if constexpr (Family == KernelTemplate::LEAF_SELECT) {
        RunLeafSelect<T, I, false>(x, values, indices, workspace, data, pipe);
    }
}

// 输入适配在计算前统一完成；所有相关模板共用同一套地址、同步及生命周期约定。
template <typename T>
__aicore__ inline GM_ADDR PrepareLayout(GM_ADDR x, GM_ADDR workspace, const SelectTopKTilingData& data, TPipe* pipe)
{
    if (data.inputInner == 1)
        return x;
    GM_ADDR prepared = workspace + data.inputWorkspaceOffset;
    PrepareInput<T>(x, prepared, data, pipe);
    Fence<HardEvent::MTE3_S>();
    SyncAll();
    pipe->Reset();
    return prepared;
}

// 计算结束后统一恢复布局、索引类型及扩展填充；只对有效索引应用偏移。
template <typename T, typename I>
__aicore__ inline void FinishLayout(GM_ADDR begin, GM_ADDR end, GM_ADDR coreValues, GM_ADDR coreIndices, GM_ADDR values,
                                    GM_ADDR indices, const SelectTopKTilingData& data, TPipe* pipe)
{
    if (!(data.flags & FINALIZE))
        return;
    Fence<HardEvent::MTE3_S>();
    SyncAll();
    pipe->Reset();
    if (data.outputInner > 1) {
        if (data.flags & REUSE_LAYOUT)
            ReformatOutput<T, I, true>(coreValues, coreIndices, values, indices, data, pipe);
        else
            ReformatOutput<T, I>(coreValues, coreIndices, values, indices, data, pipe);
    } else {
        Finalize<T, I> output;
        output.Init(begin, end, coreValues, coreIndices, values, indices, &data, pipe);
        output.Process();
    }
}

// 一个 Family 实例是完整的 Kernel 入口：准备、计算和收尾在同一模板内闭合。
template <typename T, typename I, KernelTemplate Family>
__aicore__ inline void RunKernel(GM_ADDR x, GM_ADDR begin, GM_ADDR end, GM_ADDR values, GM_ADDR indices,
                                 GM_ADDR workspace, const SelectTopKTilingData& data)
{
    TPipe pipe;
    if constexpr (Family == KernelTemplate::PREFIX_ROWS) {
        PrefixRows<T> op;
        op.Init(x, values, indices, data, &pipe);
        op.Process();
    } else if constexpr (Family == KernelTemplate::COPY_ALL) {
        CopyAll<T> op;
        op.Run(x, values, indices, data.base, &pipe);
    } else if constexpr (Family == KernelTemplate::SHORT_ROWS) {
        if (data.base.rows == 0 || data.base.k == 0)
            return;
        TopKShortRows<T, std::is_same<I, int64_t>::value> op;
        op.Init(x, values, indices, &data.base, &pipe);
        op.Process();
    } else {
        if constexpr (Family == KernelTemplate::LEAF_SELECT) {
            if ((data.flags & DIRECT_INDICES64) && data.inputInner == 1) {
                RunLeafSelect<T, I, true>(x, values, indices, workspace, data, &pipe);
                return;
            }
        }
        GM_ADDR coreValues = (data.flags & TEMP_VALUES) ? workspace + data.valuesWorkspaceOffset : values;
        GM_ADDR coreIndices = (data.flags & TEMP_INDICES) ? workspace + data.indicesWorkspaceOffset : indices;
        x = PrepareLayout<T>(x, workspace, data, &pipe);
        if (data.base.k != 0 && data.base.rows != 0)
            RunCore<T, I, Family>(x, begin, end, coreValues, coreIndices, workspace, data, &pipe);
        FinishLayout<T, I>(begin, end, coreValues, coreIndices, values, indices, data, &pipe);
    }
}
} // namespace SelectTopK
#endif
