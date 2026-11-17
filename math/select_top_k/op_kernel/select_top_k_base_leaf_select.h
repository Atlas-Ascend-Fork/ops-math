/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SELECT_TOP_K_BASE_LEAF_SELECT_H
#define SELECT_TOP_K_BASE_LEAF_SELECT_H

#include "select_top_k_base_common.h"
#include "select_top_k_base_select_layout.h"
#include "select_top_k_base_index.h"

namespace SelectTopKBase {
// 单块分片排序：各片保留局部 TopK，再按行归并候选。
template <typename T, bool UsePrefixMerge = false, bool ParallelFinalMerge = false,
          uint32_t FourPrefixDivisor = HALF_PREFIX_ALIGNMENT / PAIR_ALIGNMENT, bool AllowPaddedLeafPrefix = false,
          bool UseFinalPrefix = false>
class TopKLeafSelect {
    static constexpr uint32_t PACKED_INDEX_ITEMS = sizeof(uint64_t) / sizeof(uint32_t);
    static constexpr uint32_t INITIAL_INDEX_GROWTH = 4; // 每层从就绪前缀独立生成三段。
public:
    // 绑定输入、输出及候选工作区，并按叶子布局建立 UB 视图。
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices, GM_ADDR workspace,
                                const SelectTopKBaseTilingData* data, TPipe* pipe, bool wideIndices = false)
    {
        data_ = data;
        wideIndices_ = wideIndices;
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        values_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(values));
        indices_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices));
        partials_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
        const LeafSelectLayout layout(data_->tile, static_cast<uint32_t>(data_->k), sizeof(T), data_->finalCapacity);
        pipe->InitBuffer(arena_, data_->ubBytes);
        raw_ = arena_.Get<T>()[layout.raw / sizeof(T)];
        key_ = arena_.Get<float>()[layout.key / VALUE32_BYTES];
        index_ = arena_.Get<uint32_t>()[layout.index / VALUE32_BYTES];
        ping_ = arena_.Get<float>()[layout.ping / VALUE32_BYTES];
        pong_ = arena_.Get<float>()[layout.pong / VALUE32_BYTES];
        accumulator_ = arena_.Get<float>()[layout.accumulator / VALUE32_BYTES];
    }

    // 并行排序各叶子，再按计划归并候选段并输出每行前 K 项。
    __aicore__ inline void Process()
    {
        const uint32_t shards = data_->shardCount;
        const uint64_t task = GetBlockIdx();
        const uint64_t row = task / shards;
        const uint32_t shard = static_cast<uint32_t>(task % shards);
        const uint32_t start = shard * data_->tile;
        const uint32_t valid = Min<uint64_t>(data_->tile, data_->axis - start);
        // 保持 GM 候选段步长；短尾按实际大小选择排序模板，避免排序整个叶子。
        const bool shortTail = valid <= SORT32_ELEMENTS;
        const uint32_t padded = shortTail ? SORT32_ELEMENTS : data_->tile;
        const uint32_t stride = static_cast<uint32_t>((data_->k + (PAIR_ALIGNMENT - 1)) / PAIR_ALIGNMENT *
                                                      PAIR_ALIGNMENT);
        DataCopyExtParams copy{1, static_cast<uint32_t>(valid * sizeof(T)), 0, 0, 0};
        DataCopyPad(raw_, x_[row * data_->axis + start], copy, DataCopyPadExtParams<T>{false, 0, 0, 0});
        InitializeLeafIndex(start, padded);
        Fence<HardEvent::MTE2_V>();
        if (valid < padded) {
            Duplicate(key_, SortPadding(), padded);
            PipeBarrier<PIPE_V>();
        }
        EncodeSortKeys(key_, raw_, valid, data_->largest);
        PipeBarrier<PIPE_V>();
        if (shortTail && stride > SORT32_ELEMENTS) {
            Duplicate(ping_.ReinterpretCast<uint32_t>(), UINT32_MAX, stride * PAIR_COMPONENTS);
            PipeBarrier<PIPE_V>();
        }
        Sort32(ping_, key_, index_, Min<uint32_t>(padded / SORT32_ELEMENTS, VECTOR_MAX_REPEATS));
        if (padded > SORT32_MAX_ELEMENTS) {
            Sort32(ping_[SORT32_MAX_ELEMENTS * PAIR_COMPONENTS], key_[SORT32_MAX_ELEMENTS], index_[SORT32_MAX_ELEMENTS],
                   1);
        }
        PipeBarrier<PIPE_V>();
        LocalTensor<float> sorted = shortTail ? ping_ : SortBlock();
        // 每片最多保留 Align(K,4) 个有序候选，直接写出，省掉 Acc 的初始化和中转拷贝。
        // 比 K 多的至多三项仍是真实候选或 -Inf 尾项，不会丢失全局 TopK。
        Fence<HardEvent::V_MTE3>();
        DataCopyExtParams partialCopy{1, stride * PAIR_BYTES, 0, 0, 0};
        DataCopyPad(partials_[(row * data_->rowWorkspacePairs + shard * stride) * PAIR_COMPONENTS], sorted,
                    partialCopy);
        // 2201：标志排在 MTE3 写回之后；所有核到齐才继续读取候选。
        // V_MTE3 已保证本片排序结果就绪，不再增加全流水屏障。
        // 本 kernel 仅使用一次 flag 0，不占用框架保留的 11~14。
#if defined(__ASCENDC_SUPERKERNEL_AUTO_SYNC_ALL__)
        // 超级核配置需要框架维护同步状态，保留其标准入口。
        SyncAll();
#else
        CrossCoreSetFlag<0, PIPE_MTE3>(0);
        CrossCoreWaitFlag(0);
#endif
        if constexpr (UsePrefixMerge) {
            if (data_->tile > Tuning::MAX_LEAF_BLOCK) {
                uint32_t capacity = static_cast<uint32_t>((data_->k + (SORT32_ELEMENTS - 1)) / SORT32_ELEMENTS *
                                                          SORT32_PAIR_COMPONENTS);
                if (capacity < data_->tile)
                    capacity = data_->tile;
                if (capacity < data_->finalCapacity)
                    capacity = data_->finalCapacity;
                ping_ = arena_.Get<float>();
                pong_ = ping_[capacity * PAIR_COMPONENTS];
            }
            uint32_t lists = shards, step = 1;
            while (ParallelFinalMerge && lists > LOCAL_PREFIX_MAX_LISTS) {
                const uint32_t groups = (lists + (MERGE_WAYS - 1)) / MERGE_WAYS;
                if (task < data_->rows * groups) {
                    const uint64_t mergeRow = task / groups;
                    const uint32_t group = task % groups;
                    const uint32_t count = Min<uint32_t>(MERGE_WAYS, lists - group * MERGE_WAYS);
                    const uint32_t first = group * MERGE_WAYS * step;
                    MergeLargePartials(mergeRow, count, stride, step, first);
                    Fence<HardEvent::V_MTE3>();
                    DataCopyExtParams output{1, stride * PAIR_BYTES, 0, 0, 0};
                    DataCopyPad(partials_[(mergeRow * data_->rowWorkspacePairs + first * stride) * PAIR_COMPONENTS],
                                accumulator_, output);
                }
                SyncAll();
                lists = groups;
                step *= MERGE_WAYS;
            }
            if (task >= data_->rows)
                return;
            MergeLargePartials(task, lists, stride, step);
        } else {
            if (task >= data_->rows)
                return;
            // 每批连续读取尽可能多的有序候选；批内归并后只保留 K，再与 Acc 合并。
            // 分片写回后复用整个 arena 前部，两份最终归并区与仍存活的 Acc 不重叠。
            ping_ = arena_.Get<float>();
            pong_ = ping_[data_->finalCapacity * PAIR_COMPONENTS];
            const uint32_t batchShards = data_->finalCapacity / stride;
            bool haveAccumulator = false;
            for (uint32_t first = 0; first < shards; first += batchShards) {
                const uint32_t sourceCount = Min<uint64_t>(data_->k, data_->axis - first * data_->tile);
                const uint32_t count = sourceCount < data_->k ?
                                           (sourceCount + (PAIR_ALIGNMENT - 1)) / PAIR_ALIGNMENT * PAIR_ALIGNMENT :
                                           Min<uint32_t>(batchShards, shards - first) * stride;
                if (haveAccumulator) {
                    Fence<HardEvent::V_MTE2>();
                }
                DataCopyExtParams params{1, count * PAIR_BYTES, 0, 0, 0};
                DataCopyPad(ping_, partials_[(task * data_->rowWorkspacePairs + first * stride) * PAIR_COMPONENTS],
                            params, DataCopyPadExtParams<float>{false, 0, 0, 0});
                Fence<HardEvent::MTE2_V>();
                LocalTensor<float> merged = MergeRuns(count, stride);
                if (haveAccumulator) {
                    merged = MergeAccumulator(merged, sourceCount);
                }
                DataCopy(accumulator_.ReinterpretCast<uint32_t>(), merged.ReinterpretCast<uint32_t>(),
                         stride * PAIR_COMPONENTS);
                PipeBarrier<PIPE_V>();
                haveAccumulator = true;
            }
        }
        Emit(task);
    }

private:
    // 叶子索引区首次使用且没有在途向量写；成对初始化首块，再按四倍分层扩展原轴索引。
    __aicore__ inline void InitializeLeafIndex(uint32_t first, uint32_t count)
    {
        auto packed = index_.ReinterpretCast<uint64_t>();
        for (uint32_t pair = 0; pair < INDEX_SEQUENCE_SEED / PACKED_INDEX_ITEMS; ++pair) {
            const uint32_t index = first + pair * PACKED_INDEX_ITEMS;
            packed.SetValue(pair, uint64_t{index} | (uint64_t{index + 1} << MASK_WORD32_BITS));
        }
        Fence<HardEvent::S_V>();
        SetMaskNorm();
        ResetMask();
        auto index = index_.ReinterpretCast<int32_t>();
        for (uint32_t generated = INDEX_SEQUENCE_SEED; generated < count; generated *= INITIAL_INDEX_GROWTH) {
            for (uint32_t part = 1; part < INITIAL_INDEX_GROWTH && part * generated < count; ++part) {
                const uint32_t start = part * generated;
                Adds(index[start], index, static_cast<int32_t>(start), Min<uint32_t>(generated, count - start));
            }
            PipeBarrier<PIPE_V>();
        }
    }

    // 三个并行组的最终归并先读各半K前缀；短组或实际产出不足K时保留完整归并。
    __aicore__ inline bool TryMergeFinalThree(uint64_t row, uint32_t first, uint32_t stride, uint32_t step)
    {
        constexpr uint32_t finalWays = MERGE_WAYS - 1;
        const uint32_t prefix = static_cast<uint32_t>((data_->k + HALF_PREFIX_ALIGNMENT - 1) / HALF_PREFIX_ALIGNMENT *
                                                      PAIR_ALIGNMENT);
        const uint64_t lastLeaf = first + (finalWays - 1) * step;
        // 组内只承诺写足有效前缀，短组的其余槽位不作为已初始化候选读取。
        if (data_->axis - lastLeaf * data_->tile < prefix)
            return false;
        auto input = arena_.Get<float>();
        auto output = input[finalWays * prefix * PAIR_COMPONENTS];
        Fence<HardEvent::V_MTE2>();
        DataCopyExtParams copy{finalWays, prefix * PAIR_BYTES, (step * stride - prefix) * PAIR_BYTES, 0, 0};
        const uint64_t slot = row * data_->rowWorkspacePairs + first * stride;
        DataCopyPad(input, partials_[slot * PAIR_COMPONENTS], copy, DataCopyPadExtParams<float>{false, 0, 0, 0});
        Fence<HardEvent::MTE2_V>();
        const uint16_t length = static_cast<uint16_t>(prefix);
        uint16_t lengths[MERGE_WAYS] = {length, length, length, 0};
        MrgSortSrcList<float> sources(input, input[prefix * PAIR_COMPONENTS],
                                      input[prefix * MERGE_THIRD_SOURCE * PAIR_COMPONENTS], input);
        MrgSort4Info info(lengths, true, MERGE_THREE_SOURCES_MASK, 1);
        MrgSort(output, sources, info);
        Fence<HardEvent::V_S>();
        uint16_t consumed[MERGE_WAYS] = {};
        GetMrgSortResult(consumed[0], consumed[1], consumed[MERGE_THIRD_SOURCE], consumed[MERGE_FOURTH_SOURCE]);
        if (uint32_t(consumed[0]) + consumed[1] + consumed[MERGE_THIRD_SOURCE] < data_->k)
            return false;
        DataCopy(accumulator_.ReinterpretCast<uint32_t>(), output.template ReinterpretCast<uint32_t>(),
                 stride * PAIR_COMPONENTS);
        PipeBarrier<PIPE_V>();
        return true;
    }

    // 先用四路短前缀尝试归并，输出不足 K 时返回失败供完整路径接管。
    __aicore__ inline bool TryMergeFour(uint64_t row, uint32_t first, uint32_t stride, bool& haveAccumulator,
                                        uint32_t step = 1)
    {
        // 默认仍为半K；独立优化入口可缩短前缀，正确性始终由真实产出和完整回退保证。
        constexpr uint32_t prefixAlignment = FourPrefixDivisor * PAIR_ALIGNMENT;
        const uint32_t prefix = static_cast<uint32_t>((data_->k + prefixAlignment - 1) / prefixAlignment *
                                                      PAIR_ALIGNMENT);
        // 叶子写回后 raw/key/index 已失效。复用 arena 前部为四路输入及一次归并输出。
        // 两个区域各 4*prefix 个 pair；Host 检查二者均位于仍存活的 Acc 之前。
        LocalTensor<float> input = arena_.Get<float>();
        LocalTensor<float> output = input[prefix * MERGE_ALL_SOURCE_COMPONENTS];
        Fence<HardEvent::V_MTE2>();
        DataCopyExtParams copy{MERGE_WAYS, prefix * PAIR_BYTES, (step * stride - prefix) * PAIR_BYTES, 0, 0};
        const uint64_t slot = row * data_->rowWorkspacePairs + first * stride;
        DataCopyPad(input, partials_[slot * PAIR_COMPONENTS], copy, DataCopyPadExtParams<float>{false, 0, 0, 0});
        Fence<HardEvent::MTE2_V>();
        const uint16_t length = static_cast<uint16_t>(prefix);
        uint16_t lengths[MERGE_WAYS] = {length, length, length, length};
        MrgSortSrcList<float> sources(input, input[prefix * PAIR_COMPONENTS],
                                      input[prefix * MERGE_THIRD_SOURCE * PAIR_COMPONENTS],
                                      input[prefix * MERGE_FOURTH_SOURCE_COMPONENT_OFFSET]);
        MrgSort4Info info(lengths, true, MERGE_ALL_SOURCES_MASK, 1);
        MrgSort(output, sources, info);
        Fence<HardEvent::V_S>();
        uint16_t consumed[MERGE_WAYS] = {0, 0, 0, 0};
        GetMrgSortResult(consumed[0], consumed[1], consumed[MERGE_THIRD_SOURCE], consumed[MERGE_FOURTH_SOURCE]);
        const uint32_t produced = consumed[0] + consumed[1] + consumed[MERGE_THIRD_SOURCE] +
                                  consumed[MERGE_FOURTH_SOURCE];
        if (produced < data_->k) {
            // 只在实际产出至少 K 个全局有序项时采用快速结果，偏斜/重复值安全回退。
            return false;
        }
        LocalTensor<float> result = output;
        if (haveAccumulator) {
            // 将本组 TopK 移至已失效的输入区，再与 Acc 归并。
            // 此时 ping+pong 均可作为输出，允许结果跨越旧 ping 的阶段边界。
            DataCopy(input.ReinterpretCast<uint32_t>(), output.ReinterpretCast<uint32_t>(), stride * PAIR_COMPONENTS);
            PipeBarrier<PIPE_V>();
            result = MergeAccumulator(input, static_cast<uint32_t>(data_->k));
        }
        DataCopy(accumulator_.ReinterpretCast<uint32_t>(), result.ReinterpretCast<uint32_t>(),
                 stride * PAIR_COMPONENTS);
        PipeBarrier<PIPE_V>();
        haveAccumulator = true;
        return true;
    }

    // 累积段来自多个叶子，可能贡献超过半数结果，必须完整保留。
    // 仅对新加入的两路使用短前缀，并检查暂停时输出数量以处理偏斜输入。
    __aicore__ inline bool TryMergeTwoTail(uint64_t row, uint32_t first, uint32_t stride, uint32_t step)
    {
        if (data_->tile <= Tuning::MAX_LEAF_BLOCK)
            return false;
        const uint32_t prefix = static_cast<uint32_t>((data_->k + (HALF_PREFIX_ALIGNMENT - 1)) / HALF_PREFIX_ALIGNMENT *
                                                      PAIR_ALIGNMENT);
        if (data_->axis - static_cast<uint64_t>(first + step) * data_->tile < prefix)
            return false;
        const LeafSelectLayout layout(data_->tile, data_->k, sizeof(T), data_->finalCapacity);
        if (prefix * MERGE_INPUT_BYTES_64 + data_->k * PAIR_BYTES > layout.accumulator)
            return false;
        LocalTensor<float> input = arena_.Get<float>();
        LocalTensor<float> output = input[prefix * MERGE_THIRD_SOURCE * PAIR_COMPONENTS];
        Fence<HardEvent::V_MTE2>();
        for (uint32_t part = 0; part < TWO_MERGE_RUNS; ++part) {
            const uint64_t slot = row * data_->rowWorkspacePairs + (first + part * step) * stride;
            DataCopyExtParams copy{1, prefix * PAIR_BYTES, 0, 0, 0};
            DataCopyPad(input[part * prefix * PAIR_COMPONENTS], partials_[slot * PAIR_COMPONENTS], copy,
                        DataCopyPadExtParams<float>{false, 0, 0, 0});
        }
        Fence<HardEvent::MTE2_V>();
        uint16_t lengths[MERGE_WAYS] = {static_cast<uint16_t>(data_->k), static_cast<uint16_t>(prefix),
                                        static_cast<uint16_t>(prefix), 0};
        MrgSortSrcList<float> sources(accumulator_, input, input[prefix * PAIR_COMPONENTS], input);
        MrgSort4Info info(lengths, true, MERGE_THREE_SOURCES_MASK, 1);
        MrgSort(output, sources, info);
        Fence<HardEvent::V_S>();
        uint16_t consumed[MERGE_WAYS] = {0, 0, 0, 0};
        GetMrgSortResult(consumed[0], consumed[1], consumed[MERGE_THIRD_SOURCE], consumed[MERGE_FOURTH_SOURCE]);
        if (static_cast<uint32_t>(consumed[0] + consumed[1] + consumed[MERGE_THIRD_SOURCE] +
                                  consumed[MERGE_FOURTH_SOURCE]) < stride)
            return false;
        DataCopy(accumulator_.ReinterpretCast<uint32_t>(), output.ReinterpretCast<uint32_t>(),
                 stride * PAIR_COMPONENTS);
        PipeBarrier<PIPE_V>();
        return true;
    }

    // 读取一个分片候选段并与当前累积段归并。
    __aicore__ inline void MergeOnePartial(uint64_t row, uint32_t part, uint32_t stride, bool& haveAccumulator)
    {
        const uint32_t count = Min<uint64_t>(data_->k, data_->axis - part * data_->tile);
        Fence<HardEvent::V_MTE2>();
        DataCopyExtParams copy{
            1, static_cast<uint32_t>((count + (PAIR_ALIGNMENT - 1)) / PAIR_ALIGNMENT * UB_BLOCK_BYTES), 0, 0, 0};
        DataCopyPad(ping_, partials_[(row * data_->rowWorkspacePairs + part * stride) * PAIR_COMPONENTS], copy,
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
        Fence<HardEvent::MTE2_V>();
        if (haveAccumulator && count < data_->k) {
            Fence<HardEvent::V_S>();
            const float best = ping_.GetValue(0);
            const float threshold = accumulator_.GetValue((data_->k - 1) * PAIR_COMPONENTS);
            if (!IsNaN(best) && !IsNaN(threshold) && best <= threshold)
                return;
        }
        LocalTensor<float> result = ping_;
        if (haveAccumulator) {
            result = MergeAccumulator(ping_, count);
        }
        DataCopy(accumulator_.ReinterpretCast<uint32_t>(), result.ReinterpretCast<uint32_t>(),
                 stride * PAIR_COMPONENTS);
        PipeBarrier<PIPE_V>();
        haveAccumulator = true;
    }

    // 分组归并较长分片候选，保留每阶段所需的前 K 项。
    __aicore__ inline void MergeLargePartials(uint64_t row, uint32_t shards, uint32_t stride, uint32_t step = 1,
                                              uint32_t firstLeaf = 0)
    {
        if constexpr (UseFinalPrefix) {
            if (step > 1 && shards == MERGE_WAYS - 1 && TryMergeFinalThree(row, firstLeaf, stride, step))
                return;
        }
        const uint64_t lastLeaf = firstLeaf + (shards - 1) * step;
        if (data_->tile > Tuning::MAX_LEAF_BLOCK && step == 1 && shards * stride <= data_->finalCapacity &&
            data_->axis - lastLeaf * data_->tile >= stride) {
            Fence<HardEvent::V_MTE2>();
            DataCopyExtParams copy{1, shards * stride * PAIR_BYTES, 0, 0, 0};
            DataCopyPad(ping_, partials_[(row * data_->rowWorkspacePairs + firstLeaf * stride) * PAIR_COMPONENTS], copy,
                        DataCopyPadExtParams<float>{false, 0, 0, 0});
            Fence<HardEvent::MTE2_V>();
            const auto merged = MergeRuns(shards * stride, stride);
            DataCopy(accumulator_.ReinterpretCast<uint32_t>(), merged.template ReinterpretCast<uint32_t>(),
                     stride * PAIR_COMPONENTS);
            PipeBarrier<PIPE_V>();
            return;
        }
        bool haveAccumulator = false;
        uint32_t ordinal = 0;
        while (ordinal < shards) {
            const uint32_t first = firstLeaf + ordinal * step;
            const uint32_t count = Min<uint32_t>(MERGE_WAYS, shards - ordinal);
            // 新入口的每叶候选已写足stride，短尾以严格更小的无效键补齐；
            // 暂停时产出不足K仍完整回退，不能把一个短尾直接等同于整个组不适用。
            const bool fourFullLists = count == MERGE_WAYS &&
                                       ((AllowPaddedLeafPrefix && step == 1) ||
                                        data_->axis - static_cast<uint64_t>(first + MERGE_FOURTH_SOURCE * step) *
                                                          data_->tile >=
                                            data_->k);
            if (fourFullLists && TryMergeFour(row, first, stride, haveAccumulator, step)) {
                ordinal += MERGE_WAYS;
                continue;
            }
            if (count == TWO_MERGE_RUNS && haveAccumulator && TryMergeTwoTail(row, first, stride, step)) {
                ordinal += count;
                continue;
            }
            // 每组最多尝试一次前缀归并；不满足充分条件时完整处理该组，避免重复试探。
            for (uint32_t part = 0; part < count; ++part) {
                MergeOnePartial(row, first + part * step, stride, haveAccumulator);
            }
            ordinal += count;
        }
    }

    // 将叶子内的有序小段按四路归并成下一层。
    template <uint16_t Span>
    __aicore__ inline void MergeBlockStage(const LocalTensor<float>& output, const LocalTensor<float>& input,
                                           uint32_t repeats)
    {
        uint16_t lengths[MERGE_WAYS] = {Span, Span, Span, Span};
        MrgSortSrcList<float> parts(input, input[Span * PAIR_COMPONENTS],
                                    input[Span * MERGE_THIRD_SOURCE * PAIR_COMPONENTS],
                                    input[Span * MERGE_FOURTH_SOURCE_COMPONENT_OFFSET]);
        MrgSort4Info info(lengths, false, MERGE_ALL_SOURCES_MASK, static_cast<uint8_t>(repeats));
        MrgSort(output, parts, info);
        PipeBarrier<PIPE_V>();
    }

    // 对叶子执行 Sort32 和后续归并，返回完整有序的 pair 段。
    __aicore__ inline LocalTensor<float> SortBlock()
    {
        if (data_->tile > Tuning::MAX_LEAF_BLOCK)
            return MergeRuns(data_->tile, SORT32_ELEMENTS);
        // Host 选择 1024/2048/4096。末层每路只读取前 Align(K,4) 项；
        // 每路都有这么多项时，任一路耗尽前已输出足够的有序前缀。
        MergeBlockStage<SORT32_ELEMENTS>(pong_, ping_, data_->tile / SORT_STAGE2_ITEMS);
        MergeBlockStage<SORT_STAGE2_ITEMS>(ping_, pong_, data_->tile / SORT_STAGE3_ITEMS);
        const uint16_t needed = static_cast<uint16_t>((data_->k + (PAIR_ALIGNMENT - 1)) / PAIR_ALIGNMENT *
                                                      PAIR_ALIGNMENT);
        if (data_->tile <= SORT_STAGE4_ITEMS) {
            const uint16_t length = Min<uint16_t>(SORT_STAGE3_ITEMS, needed);
            uint16_t lengths[MERGE_WAYS] = {length, length, length, length};
            MrgSortSrcList<float> parts(ping_, ping_[SORT_STAGE3_ITEMS * PAIR_COMPONENTS],
                                        ping_[SORT_STAGE3_ITEMS * MERGE_THIRD_SOURCE * PAIR_COMPONENTS],
                                        ping_[SORT_STAGE3_ITEMS * MERGE_FOURTH_SOURCE * PAIR_COMPONENTS]);
            const uint16_t routes = data_->tile == SORT_LEAF_SMALL_ITEMS ? MERGE_TWO_SOURCES_MASK :
                                                                           MERGE_ALL_SOURCES_MASK;
            MrgSort4Info info(lengths, needed <= SORT_STAGE3_ITEMS, routes, 1);
            MrgSort(pong_, parts, info);
            PipeBarrier<PIPE_V>();
            return pong_;
        }
        if (data_->tile == Tuning::MAX_LEAF_BLOCK && needed <= SORT_STAGE3_ITEMS) {
            // 下阶段每组四路只需 K 项；先读取半 K 前缀，暂停时不足 K 项则回退完整归并。
            const uint16_t prefix = static_cast<uint16_t>((needed + (HALF_PREFIX_ALIGNMENT - 1)) /
                                                          HALF_PREFIX_ALIGNMENT * PAIR_ALIGNMENT);
            for (uint32_t group = 0; group < TWO_MERGE_RUNS; ++group) {
                auto source = ping_[group * SORT_STAGE4_ITEMS * PAIR_COMPONENTS];
                auto output = pong_[group * SORT_STAGE4_ITEMS * PAIR_COMPONENTS];
                uint16_t lengths[MERGE_WAYS] = {prefix, prefix, prefix, prefix};
                MrgSortSrcList<float> sources(source, source[SORT_STAGE3_ITEMS * PAIR_COMPONENTS],
                                              source[SORT_STAGE3_ITEMS * MERGE_THIRD_SOURCE * PAIR_COMPONENTS],
                                              source[SORT_STAGE3_ITEMS * MERGE_FOURTH_SOURCE * PAIR_COMPONENTS]);
                MrgSort4Info info(lengths, true, MERGE_ALL_SOURCES_MASK, 1);
                MrgSort(output, sources, info);
                Fence<HardEvent::V_S>();
                uint16_t consumed[MERGE_WAYS] = {0, 0, 0, 0};
                GetMrgSortResult(consumed[0], consumed[1], consumed[MERGE_THIRD_SOURCE], consumed[MERGE_FOURTH_SOURCE]);
                if (static_cast<uint32_t>(consumed[0] + consumed[1] + consumed[MERGE_THIRD_SOURCE] +
                                          consumed[MERGE_FOURTH_SOURCE]) < needed) {
                    uint16_t full[MERGE_WAYS] = {needed, needed, needed, needed};
                    MrgSort4Info fallback(full, false, MERGE_ALL_SOURCES_MASK, 1);
                    MrgSort(output, sources, fallback);
                    PipeBarrier<PIPE_V>();
                }
            }
        } else {
            MergeBlockStage<SORT_STAGE3_ITEMS>(pong_, ping_, data_->tile / SORT_STAGE4_ITEMS);
        }
        if (data_->tile == Tuning::MAX_LEAF_BLOCK) {
            const uint16_t length = Min<uint16_t>(SORT_STAGE4_ITEMS, needed);
            uint16_t lengths[MERGE_WAYS] = {length, length, 0, 0};
            MrgSortSrcList<float> parts(pong_, pong_[SORT_STAGE4_ITEMS * PAIR_COMPONENTS], pong_, pong_);
            MrgSort4Info info(lengths, needed <= SORT_STAGE4_ITEMS, MERGE_TWO_SOURCES_MASK, 1);
            MrgSort(ping_, parts, info);
            PipeBarrier<PIPE_V>();
            return ping_;
        }
        return pong_;
    }

    // 将两段有序候选归并到输出缓冲，并保留前 K 项。
    __aicore__ inline LocalTensor<float> MergeRuns(uint32_t count, uint32_t span)
    {
        LocalTensor<float> input = ping_;
        LocalTensor<float> output = pong_;
        for (; span < count; span *= MERGE_WAYS) {
            if (span * MERGE_WAYS >= count && span >= data_->k) {
                // 最后一层只需要每路前 K 项；其余项不可能进入全局 TopK。
                // 使用完整归并，因此短尾路提前耗尽也不会丢失候选。
                const uint32_t k = static_cast<uint32_t>(data_->k);
                const uint32_t routes = (count + span - 1) / span;
                uint16_t lengths[MERGE_WAYS] = {
                    static_cast<uint16_t>(k), static_cast<uint16_t>(Min<uint32_t>(k, count - span)),
                    static_cast<uint16_t>(
                        count > PAIR_COMPONENTS * span ? Min<uint32_t>(k, count - PAIR_COMPONENTS * span) : 0),
                    static_cast<uint16_t>(
                        count > MERGE_FOURTH_SOURCE * span ? Min<uint32_t>(k, count - MERGE_FOURTH_SOURCE * span) : 0)};
                // 固定四路地址，避免动态 LocalTensor 数组索引生成大量标量搬运。
                MrgSortSrcList<float> sources(input, input[span * PAIR_COMPONENTS],
                                              input[span * MERGE_THIRD_SOURCE * PAIR_COMPONENTS],
                                              input[span * MERGE_FOURTH_SOURCE_COMPONENT_OFFSET]);
                MrgSort4Info info(lengths, false, static_cast<uint16_t>((1U << routes) - 1), 1);
                MrgSort(output, sources, info);
                PipeBarrier<PIPE_V>();
                return output;
            }
            const uint32_t groups = count / (MERGE_WAYS * span);
            if (groups != 0) {
                const uint16_t length = static_cast<uint16_t>(span);
                uint16_t lengths[MERGE_WAYS] = {length, length, length, length};
                MrgSortSrcList<float> parts(input, input[span * PAIR_COMPONENTS],
                                            input[span * MERGE_THIRD_SOURCE * PAIR_COMPONENTS],
                                            input[span * MERGE_FOURTH_SOURCE_COMPONENT_OFFSET]);
                MrgSort4Info info(lengths, false, MERGE_ALL_SOURCES_MASK, static_cast<uint8_t>(groups));
                MrgSort(output, parts, info);
            }
            const uint32_t start = groups * MERGE_WAYS * span;
            const uint32_t tail = count - start;
            if (tail != 0 && tail <= span) {
                DataCopy(output[start * PAIR_COMPONENTS].ReinterpretCast<uint32_t>(),
                         input[start * PAIR_COMPONENTS].ReinterpretCast<uint32_t>(), tail * PAIR_COMPONENTS);
            } else if (tail > span) {
                LocalTensor<float> parts[MERGE_WAYS] = {input, input, input, input};
                uint16_t lengths[MERGE_WAYS] = {0, 0, 0, 0};
                uint32_t routes = 0;
                for (uint32_t offset = start; offset < count; offset += span) {
                    parts[routes] = input[offset * PAIR_COMPONENTS];
                    lengths[routes] = static_cast<uint16_t>(Min<uint32_t>(span, count - offset));
                    ++routes;
                }
                MrgSortSrcList<float> sources(parts[0], parts[1], parts[MERGE_THIRD_SOURCE],
                                              parts[MERGE_FOURTH_SOURCE]);
                MrgSort4Info info(lengths, false, static_cast<uint16_t>((1U << routes) - 1), 1);
                MrgSort(output[start * PAIR_COMPONENTS], sources, info);
            }
            PipeBarrier<PIPE_V>();
            auto temporary = input;
            input = output;
            output = temporary;
        }
        return input;
    }

    // 将新候选段并入当前累积段。
    __aicore__ inline LocalTensor<float> MergeAccumulator(const LocalTensor<float>& source, uint32_t sourceCount)
    {
        if constexpr (!UsePrefixMerge) {
            // 完整列表路径的两路输入各不超过 K，无需四路切段和动态地址数组。
            LocalTensor<float> output = source.GetPhyAddr() == ping_.GetPhyAddr() ? pong_ : ping_;
            uint16_t lengths[MERGE_WAYS] = {static_cast<uint16_t>(data_->k), static_cast<uint16_t>(sourceCount), 0, 0};
            MrgSortSrcList<float> sources(accumulator_, source, source, source);
            MrgSort4Info info(lengths, false, MERGE_TWO_SOURCES_MASK, 1);
            MrgSort(output, sources, info);
            PipeBarrier<PIPE_V>();
            return output;
        }
        if (data_->k <= SORT_STAGE4_ITEMS && sourceCount <= SORT_STAGE4_ITEMS) {
            auto output = source.GetPhyAddr() == ping_.GetPhyAddr() ? pong_ : ping_;
            uint16_t lengths[MERGE_WAYS] = {static_cast<uint16_t>(data_->k), static_cast<uint16_t>(sourceCount), 0, 0};
            MrgSortSrcList<float> sources(accumulator_, source, source, source);
            MrgSort4Info info(lengths, false, MERGE_TWO_SOURCES_MASK, 1);
            MrgSort(output, sources, info);
            PipeBarrier<PIPE_V>();
            return output;
        }
        LocalTensor<float> parts[MERGE_WAYS] = {source, source, source, source};
        uint16_t lengths[MERGE_WAYS] = {0, 0, 0, 0};
        uint32_t routes = 0;
        // 每路不超过 2048；K=4096 时两份候选恰好拆为四路。
        for (uint32_t offset = 0; offset < data_->k; offset += SORT_STAGE4_ITEMS) {
            parts[routes] = accumulator_[offset * PAIR_COMPONENTS];
            lengths[routes] = static_cast<uint16_t>(Min<uint64_t>(SORT_STAGE4_ITEMS, data_->k - offset));
            ++routes;
        }
        for (uint32_t offset = 0; offset < sourceCount; offset += SORT_STAGE4_ITEMS) {
            parts[routes] = source[offset * PAIR_COMPONENTS];
            lengths[routes] = static_cast<uint16_t>(Min<uint64_t>(SORT_STAGE4_ITEMS, sourceCount - offset));
            ++routes;
        }
        // batchShards>=2 时 2*K<=tile，ping 足以容纳结果；单路批量的结果始终在 ping。
        auto output = source.GetPhyAddr() == ping_.GetPhyAddr() ? pong_ : ping_;
        MrgSortSrcList<float> inputs(parts[0], parts[1], parts[MERGE_THIRD_SOURCE], parts[MERGE_FOURTH_SOURCE]);
        MrgSort4Info info(lengths, false, static_cast<uint16_t>((1U << routes) - 1), 1);
        MrgSort(output, inputs, info);
        PipeBarrier<PIPE_V>();
        return output;
    }

    // 解码最终 pair，恢复输入位模式并写出值及索引。
    __aicore__ inline void Emit(uint64_t row)
    {
        const uint32_t count = static_cast<uint32_t>(data_->k);
        uint64_t selected = 0;
        const GatherMaskParams params(1, 1, VECTOR_REPEAT_BLOCKS, 0);
        GatherMask(key_, accumulator_, uint8_t(1), true, count * PAIR_COMPONENTS, params, selected);
        GatherMask(index_, accumulator_.ReinterpretCast<uint32_t>(), uint8_t(GATHER_INDEX_PATTERN), true,
                   count * PAIR_COMPONENTS, params, selected);
        PipeBarrier<PIPE_V>();
        if (!data_->largest) {
            Adds(key_.template ReinterpretCast<int32_t>(), key_.template ReinterpretCast<int32_t>(), INT32_MIN, count);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (!std::is_same<T, float>::value) {
            Cast(raw_, key_, RoundMode::CAST_RINT, count);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (std::is_same<T, float>::value) {
            RestoreNaNPayload<float>(key_, key_, index_, reinterpret_cast<const __gm__ float*>(x_.GetPhyAddr()),
                                     row * data_->axis, count, data_->largest);
        } else {
            RestoreNaNPayload<T>(raw_, key_, index_, x_.GetPhyAddr(), row * data_->axis, count, data_->largest);
        }
        // 两路 Gather 和 NaN 载荷恢复完成后，pair 累积缓冲即可复用。
        auto wide = accumulator_.ReinterpretCast<int64_t>();
        if (wideIndices_) {
            Cast(wide, index_.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
        }
        DataCopyExtParams valuesCopy{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        Fence<HardEvent::V_MTE3>();
        if constexpr (std::is_same<T, float>::value) {
            DataCopyPad(values_[row * data_->k], key_, valuesCopy);
        } else {
            DataCopyPad(values_[row * data_->k], raw_, valuesCopy);
        }
        if (wideIndices_) {
            GlobalTensor<int64_t> destination;
            destination.SetGlobalBuffer((__gm__ int64_t*)indices_.GetPhyAddr());
            DataCopyExtParams indexCopy{1, count * INDEX64_BYTES, 0, 0, 0};
            DataCopyPad(destination[row * data_->k], wide, indexCopy);
        } else {
            DataCopyExtParams indexCopy{1, count * INDEX32_BYTES, 0, 0, 0};
            DataCopyPad(indices_[row * data_->k], index_.ReinterpretCast<int32_t>(), indexCopy);
        }
        Fence<HardEvent::MTE3_S>();
    }

    const SelectTopKBaseTilingData* data_;
    TBuf<TPosition::VECCALC> arena_;
    GlobalTensor<T> x_;
    GlobalTensor<T> values_;
    GlobalTensor<int32_t> indices_;
    bool wideIndices_ = false;
    GlobalTensor<float> partials_;
    LocalTensor<T> raw_;
    LocalTensor<float> key_;
    LocalTensor<uint32_t> index_;
    LocalTensor<float> ping_;
    LocalTensor<float> pong_;
    LocalTensor<float> accumulator_;
};
} // namespace SelectTopKBase
#endif
