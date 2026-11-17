/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SELECT_TOP_K_MERGE_SORT_H_
#define SELECT_TOP_K_MERGE_SORT_H_
#include "select_top_k_tiling_data.h"
#include "select_top_k_two_leaf_layout.h"
#include "select_top_k_base_common.h"
#include "select_top_k_base_index.h"
#include "select_top_k_base_select_layout.h"

namespace SelectTopK {
using namespace SelectTopKBase;
// 层级：先各核连续片段内归并，再一次同步，最后每行的首核归并片段前缀。
template <typename T, bool WideIndices, SortPattern Pattern, MergeStorage Storage = MergeStorage::WINDOWS>
class MergeSort {
public:
    // 绑定输入输出与两个工作区平面，按排序/流式归并的生命周期复用 UB。
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices, GM_ADDR workspace,
                                const SelectTopKTilingData& data, TPipe* pipe)
    {
        d_ = &data.base;
        schedule_ = &data.schedule;
        x_.SetGlobalBuffer((__gm__ T*)x);
        values_.SetGlobalBuffer((__gm__ T*)values);
        indices_.SetGlobalBuffer((__gm__ int32_t*)indices);
        indices64_.SetGlobalBuffer((__gm__ int64_t*)indices);
        gm_[0].SetGlobalBuffer((__gm__ float*)workspace);
        gm_[1].SetGlobalBuffer((__gm__ float*)workspace + d_->rows * d_->rowWorkspacePairs * PAIR_COMPONENTS);
        pipe->InitBuffer(arenaBuf_, d_->ubBytes);
        const GeneralMergeLayout layout(d_->tile, d_->mergeWindow, d_->k, sizeof(T), ROUTE_GM_MERGE);
        raw_ = View<T>(layout.raw, d_->tile);
        key_ = View<float>(layout.key, d_->tile);
        index_ = View<uint32_t>(layout.index, d_->tile);
        ping_ = View<float>(layout.ping, d_->tile * PAIR_COMPONENTS);
        pong_ = View<float>(layout.pong, d_->tile * PAIR_COMPONENTS);
        if constexpr (Storage == MergeStorage::TWO_LEAF_WINDOWS) {
            const TwoLeafMergeLayout localLayout(d_->tile, d_->mergeWindow);
            for (uint32_t i = 0; i < MERGE_WAYS; ++i)
                windows_[i] = View<float>((i % TWO_MERGE_RUNS) * d_->mergeWindow * PAIR_BYTES,
                                          d_->mergeWindow * PAIR_COMPONENTS);
            merged_ = View<float>(localLayout.merged, d_->mergeWindow * TWO_MERGE_RUNS * PAIR_COMPONENTS);
        } else {
            for (uint32_t i = 0; i < MERGE_WAYS; ++i)
                windows_[i] = View<float>(layout.windows + i * d_->mergeWindow * PAIR_BYTES,
                                          d_->mergeWindow * PAIR_COMPONENTS);
            merged_ = View<float>(layout.merged, d_->mergeWindow * MERGE_ALL_SOURCE_COMPONENTS);
        }
    }

    // 无跨核路径按连续行块运行；跨核路径仅在局部前缀全部就绪后做一次全核同步。
    __aicore__ inline void Process()
    {
        if constexpr (Storage == MergeStorage::TWO_LEAF_WINDOWS) {
            ProcessTwoLeafRows();
            return;
        }
        if constexpr (Pattern == SortPattern::LOCAL_MERGE) {
            const uint64_t first = GetBlockIdx() * schedule_->rowsPerCore;
            const uint64_t end = Min<uint64_t>(d_->rows, first + schedule_->rowsPerCore);
            for (uint64_t row = first; row < end; ++row) {
                if constexpr (Storage == MergeStorage::ALL_CANDIDATES || Storage == MergeStorage::GROUPED_CANDIDATES)
                    SortResidentRow(row);
                else
                    SortShard(row, 0, true);
            }
            return;
        }
        const uint64_t row = GetBlockIdx() / d_->shardCount;
        const uint64_t shard = GetBlockIdx() % d_->shardCount;
        SortShard(row, shard, false);
        Fence<HardEvent::MTE3_S>();
        SyncAll();
        if (shard == 0) {
            if constexpr (Storage == MergeStorage::GROUPED_CANDIDATES)
                MergeResidentGroups(row);
            else if constexpr (Storage == MergeStorage::ALL_CANDIDATES || Storage == MergeStorage::GROUPED_CANDIDATES)
                MergeResident(row);
            else
                MergeShards(row);
        }
    }

    // 少行短轴专用入口：整行排序后直接写出，不启动GM归并和跨核同步。
    __aicore__ inline void ProcessCompactRows()
    {
        for (uint64_t row = GetBlockIdx(); row < d_->rows; row += d_->cores) {
            auto sorted = SortTile(row, 0, static_cast<uint32_t>(d_->axis));
            EmitSorted(row, sorted, static_cast<uint32_t>(d_->k));
        }
    }

private:
    // 叶子任务连续均分给全部核，避免行数尾部让部分核多做两个叶子；同步后独立归并各行。
    __aicore__ inline void ProcessTwoLeafRows()
    {
        const uint64_t core = GetBlockIdx();
        const uint64_t tasks = d_->rows * TWO_MERGE_RUNS;
        const uint64_t leafQuotient = tasks / d_->cores, leafRemainder = tasks % d_->cores;
        const uint64_t firstLeaf = core * leafQuotient + Min<uint64_t>(core, leafRemainder);
        const uint64_t leafCount = leafQuotient + (core < leafRemainder ? 1 : 0);
        BindConversionBuffers(d_->tile);
        for (uint64_t task = firstLeaf; task < firstLeaf + leafCount; ++task) {
            const uint64_t row = task / TWO_MERGE_RUNS;
            const uint32_t leaf = task % TWO_MERGE_RUNS;
            const uint32_t start = leaf * d_->tile;
            const uint32_t valid = Min<uint64_t>(d_->tile, d_->axis - start);
            auto sorted = SortTile(row, start, valid);
            const uint64_t target = row * d_->rowWorkspacePairs + leaf * schedule_->leafStride;
            StoreToGm(gm_[0][target * PAIR_COMPONENTS], sorted, Min<uint64_t>(d_->k, valid) * PAIR_COMPONENTS);
        }
        Fence<HardEvent::MTE3_S>();
        SyncAll();
        const uint64_t rowQuotient = d_->rows / d_->cores, rowRemainder = d_->rows % d_->cores;
        const uint64_t firstRow = core * rowQuotient + Min<uint64_t>(core, rowRemainder);
        const uint64_t rowCount = rowQuotient + (core < rowRemainder ? 1 : 0);
        for (uint64_t row = firstRow; row < firstRow + rowCount; ++row) {
            BindConversionBuffers(d_->tile);
            srcRole_ = 0;
            const uint64_t base = row * d_->rowWorkspacePairs;
            uint64_t starts[MERGE_WAYS] = {}, lengths[MERGE_WAYS] = {};
            for (uint32_t leaf = 0; leaf < TWO_MERGE_RUNS; ++leaf) {
                const uint32_t start = leaf * d_->tile;
                const uint32_t valid = Min<uint64_t>(d_->tile, d_->axis - start);
                starts[leaf] = base + leaf * schedule_->leafStride;
                lengths[leaf] = Min<uint64_t>(d_->k, valid);
            }
            MergeGroup(row, starts, lengths, TWO_MERGE_RUNS, 0, d_->k, true);
        }
    }

    // 返回字节对齐后的 UB 视图，所有范围由 Host 的共享资源账本约束。
    template <typename U>
    __aicore__ inline LocalTensor<U> View(uint32_t offset, uint32_t count)
    {
        auto tensor = arenaBuf_.Get<U>()[offset / sizeof(U)];
        tensor.SetSize(count);
        return tensor;
    }

    // 流式结果仍存活时，输出暂存区只能复用已消耗的输入窗口，不能沿用较大叶子的偏移。
    // 20*slots 的输出暂存位于 32*window 的归并结果之前；叶子排序时恢复 slots=tile。
    __aicore__ inline void BindConversionBuffers(uint32_t slots)
    {
        raw_ = View<T>(0, slots);
        key_ = View<float>(slots * VALUE32_BYTES, slots);
        index_ = View<uint32_t>(slots * KEY_INDEX_BYTES, slots);
        ping_ = View<float>(slots * RAW_KEY_INDEX_BYTES, slots * PAIR_COMPONENTS);
    }

#include "select_top_k_sort_primitives.inc"

    // 核内生成两个或四个有序叶子；每段尾部写足pair对齐的无效排序键。
    __aicore__ inline void SortResidentRow(uint64_t row)
    {
        BindConversionBuffers(d_->tile);
        for (uint32_t leaf = 0; leaf < d_->tiles; ++leaf) {
            const uint64_t start = uint64_t(leaf) * d_->tile;
            const uint32_t valid = Min<uint64_t>(d_->tile, d_->axis - start);
            auto sorted = SortTile(row, start, valid);
            const uint32_t keep = DivUp(valid, PAIR_ALIGNMENT) * PAIR_ALIGNMENT;
            StoreToGm(gm_[0][(row * d_->rowWorkspacePairs + leaf * d_->tile) * PAIR_COMPONENTS], sorted,
                      keep * PAIR_COMPONENTS);
        }
        Fence<HardEvent::MTE3_MTE2>();
        MergeResident(row);
    }

    // 所有输入段和完整归并结果同时驻留；解码只复用已经消耗的输入区。
    __aicore__ inline void MergeResident(uint64_t row)
    {
        const uint32_t slots = d_->rowWorkspacePairs;
        auto input = View<float>(0, slots * PAIR_COMPONENTS);
        auto output = View<float>(slots * PAIR_BYTES, slots * PAIR_COMPONENTS);
        DataCopyPad(input, gm_[0][row * slots * PAIR_COMPONENTS], DataCopyExtParams{1, slots * PAIR_BYTES, 0, 0, 0},
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
        Fence<HardEvent::MTE2_V>();
        LocalTensor<float> sources[MERGE_WAYS] = {input, input, input, input};
        uint16_t lengths[MERGE_WAYS] = {};
        uint32_t routes = 0;
        for (uint32_t leaf = 0; leaf < d_->tiles; ++leaf) {
            const uint32_t start = leaf * d_->tile;
            const uint32_t valid = Min<uint64_t>(d_->tile, d_->axis - start);
            const uint32_t count = static_cast<uint32_t>(DivUp(valid, PAIR_ALIGNMENT) * PAIR_ALIGNMENT);
            if constexpr (Pattern == SortPattern::LOCAL_MERGE) {
                // 两个较长叶子各拆为至多两路，单路不超过硬件4095项限制。
                for (uint32_t offset = 0; offset < count; offset += MERGE_MAX_ALIGNED_ITEMS) {
                    sources[routes] = input[(start + offset) * PAIR_COMPONENTS];
                    lengths[routes++] = static_cast<uint16_t>(Min<uint32_t>(MERGE_MAX_ALIGNED_ITEMS, count - offset));
                }
            } else {
                sources[routes] = input[start * PAIR_COMPONENTS];
                lengths[routes++] = static_cast<uint16_t>(count);
            }
        }
        MrgSortSrcList<float> lists(sources[0], sources[1], sources[MERGE_THIRD_SOURCE], sources[MERGE_FOURTH_SOURCE]);
        MrgSort4Info info(lengths, false, static_cast<uint16_t>((1U << routes) - 1), 1);
        MrgSort(output, lists, info);
        PipeBarrier<PIPE_V>();
        // 键和INT32索引占据已失效输入区；INT64复用已解码pair，原值转换独立存放。
        key_ = View<float>(0, slots);
        index_ = View<uint32_t>(slots * VALUE32_BYTES, slots);
        ping_ = output;
        raw_ = View<T>(slots * DOUBLE_PAIR_BYTES, slots);
        Emit(row, 0, output, static_cast<uint32_t>(d_->k));
    }

    // 装入最多四个叶子并一次完整归并；尾部无效键只在有效候选之后参与。
    __aicore__ inline void MergeResidentLeafGroup(uint64_t row, uint32_t first, uint32_t count,
                                                  const LocalTensor<float>& input, const LocalTensor<float>& output)
    {
        const uint64_t offset = row * d_->rowWorkspacePairs + first * d_->tile;
        Fence<HardEvent::V_MTE2>();
        DataCopyPad(input, gm_[0][offset * PAIR_COMPONENTS],
                    DataCopyExtParams{1, count * d_->tile * PAIR_BYTES, 0, 0, 0},
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
        Fence<HardEvent::MTE2_V>();
        LocalTensor<float> sources[MERGE_WAYS] = {input, input, input, input};
        uint16_t lengths[MERGE_WAYS] = {};
        for (uint32_t leaf = 0; leaf < count; ++leaf) {
            const uint64_t start = uint64_t(first + leaf) * d_->tile;
            const uint32_t valid = Min<uint64_t>(d_->tile, d_->axis - start);
            sources[leaf] = input[leaf * d_->tile * PAIR_COMPONENTS];
            lengths[leaf] = static_cast<uint16_t>(DivUp(valid, PAIR_ALIGNMENT) * PAIR_ALIGNMENT);
        }
        MrgSortSrcList<float> lists(sources[0], sources[1], sources[MERGE_THIRD_SOURCE], sources[MERGE_FOURTH_SOURCE]);
        MrgSort4Info info(lengths, false, static_cast<uint16_t>((1U << count) - 1), 1);
        MrgSort(output, lists, info);
        PipeBarrier<PIPE_V>();
    }

    // 两批TopK各拆为至多两路，每路严格不超过硬件4095项限制；重新四路归并保留全局前缀。
    __aicore__ inline void MergeResidentGroups(uint64_t row)
    {
        const uint32_t capacity = d_->finalCapacity;
        const uint32_t alignedK = DivUp(d_->k, SORT32_ELEMENTS) * SORT32_ELEMENTS;
        const uint32_t prefix = DivUp(d_->k, PAIR_ALIGNMENT) * PAIR_ALIGNMENT;
        auto input = View<float>(0, capacity * PAIR_COMPONENTS);
        auto output = View<float>(capacity * PAIR_BYTES, capacity * PAIR_COMPONENTS);
        auto accumulator = View<float>(capacity * DOUBLE_PAIR_BYTES, alignedK * PAIR_COMPONENTS);
        MergeResidentLeafGroup(row, 0, MERGE_WAYS, input, output);
        DataCopy(accumulator.template ReinterpretCast<uint32_t>(), output.template ReinterpretCast<uint32_t>(),
                 alignedK * PAIR_COMPONENTS);
        PipeBarrier<PIPE_V>();
        MergeResidentLeafGroup(row, MERGE_WAYS, d_->tiles - MERGE_WAYS, input, output);
        const uint32_t tail = Min<uint64_t>(d_->k, d_->axis - uint64_t(MERGE_WAYS) * d_->tile);
        const uint32_t counts[TWO_MERGE_RUNS] = {prefix,
                                                 static_cast<uint32_t>(DivUp(tail, PAIR_ALIGNMENT) * PAIR_ALIGNMENT)};
        LocalTensor<float> candidates[TWO_MERGE_RUNS] = {accumulator, output};
        LocalTensor<float> sources[MERGE_WAYS] = {input, input, input, input};
        uint16_t lengths[MERGE_WAYS] = {};
        uint32_t routes = 0;
        for (uint32_t group = 0; group < TWO_MERGE_RUNS; ++group) {
            for (uint32_t start = 0; start < counts[group]; start += MERGE_MAX_ALIGNED_ITEMS) {
                sources[routes] = candidates[group][start * PAIR_COMPONENTS];
                lengths[routes++] = static_cast<uint16_t>(
                    Min<uint32_t>(MERGE_MAX_ALIGNED_ITEMS, counts[group] - start));
            }
        }
        MrgSortSrcList<float> lists(sources[0], sources[1], sources[MERGE_THIRD_SOURCE], sources[MERGE_FOURTH_SOURCE]);
        MrgSort4Info info(lengths, false, static_cast<uint16_t>((1U << routes) - 1), 1);
        MrgSort(input, lists, info);
        PipeBarrier<PIPE_V>();
        // 最终pair在input；输出转换复用已失效的第二批区域，宽索引复用已解码pair。
        key_ = View<float>(capacity * PAIR_BYTES, alignedK);
        index_ = View<uint32_t>(capacity * PAIR_BYTES + alignedK * VALUE32_BYTES, alignedK);
        raw_ = View<T>(capacity * PAIR_BYTES + alignedK * PAIR_BYTES, alignedK);
        ping_ = input;
        Emit(row, 0, input, static_cast<uint32_t>(d_->k));
    }

    // 给定轴片段在双平面工作区中的首 pair；不同片段永不覆盖。
    __aicore__ inline uint64_t ShardBase(uint64_t row, uint64_t shard)
    {
        return row * d_->rowWorkspacePairs + shard * schedule_->leavesPerShard * schedule_->leafStride;
    }

    // 计算连续输入区间内实际保留的项数，避免将未填充的工作区尾部参与归并。
    __aicore__ inline uint64_t PrefixLength(uint64_t start, uint64_t limit)
    {
        return start < d_->axis ? Min<uint64_t>(d_->k, Min<uint64_t>(d_->axis - start, limit)) : 0;
    }

    // 对当前核负责的连续轴片段先排序叶子，再在本核完成四路归并；不做全核同步。
    __aicore__ inline void SortShard(uint64_t row, uint64_t shard, bool finalOutput)
    {
        BindConversionBuffers(d_->tile);
        srcRole_ = 0;
        const uint64_t first = shard * schedule_->shardElements;
        const uint64_t length = Min<uint64_t>(schedule_->shardElements, d_->axis - first);
        const uint64_t base = ShardBase(row, shard);
        if constexpr (Pattern == SortPattern::CROSS_MERGE) {
            // 单叶子模式直接写出当前核的有序前缀，不编译核内多层归并循环。
            auto sorted = SortTile(row, first, static_cast<uint32_t>(length));
            const uint32_t validKeep = Min<uint64_t>(d_->k, length);
            const uint32_t keep = (Storage == MergeStorage::ALL_CANDIDATES ||
                                   Storage == MergeStorage::GROUPED_CANDIDATES) ?
                                      DivUp(validKeep, PAIR_ALIGNMENT) * PAIR_ALIGNMENT :
                                      validKeep;
            StoreToGm(gm_[0][base * PAIR_COMPONENTS], sorted, keep * PAIR_COMPONENTS);
            return;
        }
        for (uint64_t leaf = 0; leaf < schedule_->leavesPerShard; ++leaf) {
            const uint64_t local = leaf * d_->tile;
            if (local >= length)
                break;
            const uint32_t valid = Min<uint64_t>(d_->tile, length - local);
            auto sorted = SortTile(row, first + local, valid);
            const uint32_t keep = Min<uint64_t>(d_->k, valid);
            if (schedule_->leavesPerShard == 1 && finalOutput) {
                EmitSorted(row, sorted, keep);
                return;
            }
            StoreToGm(gm_[0][(base + leaf * schedule_->leafStride) * PAIR_COMPONENTS], sorted, keep * PAIR_COMPONENTS);
        }
        uint64_t runs = schedule_->leavesPerShard;
        for (uint64_t span = 1; runs > 1; span *= MERGE_WAYS) {
            const uint64_t groups = DivUp(runs, MERGE_WAYS);
            for (uint64_t group = 0; group < groups; ++group) {
                const uint64_t firstLeaf = group * MERGE_WAYS * span;
                uint64_t starts[MERGE_WAYS] = {}, lengths[MERGE_WAYS] = {};
                uint64_t total = 0;
                uint32_t routes = 0;
                for (uint32_t i = 0; i < MERGE_WAYS; ++i) {
                    const uint64_t leaf = firstLeaf + i * span;
                    const uint64_t start = leaf * d_->tile;
                    if (leaf >= schedule_->leavesPerShard || start >= length)
                        break;
                    starts[routes] = base + leaf * schedule_->leafStride;
                    lengths[routes] = Min<uint64_t>(d_->k, Min<uint64_t>(length - start, span * d_->tile));
                    total += lengths[routes++];
                }
                if (routes == 0)
                    continue;
                MergeGroup(row, starts, lengths, routes, base + firstLeaf * schedule_->leafStride,
                           Min<uint64_t>(d_->k, total), finalOutput && groups == 1);
            }
            Fence<HardEvent::MTE3_MTE2>();
            srcRole_ = 1 - srcRole_;
            runs = groups;
        }
    }

    // 局部前缀已完成后由首核负责该行；多层跨片段归并均在同一核内，无额外 SyncAll。
    __aicore__ inline void MergeShards(uint64_t row)
    {
        uint64_t runs = d_->shardCount;
        for (uint64_t span = 1; runs > 1; span *= MERGE_WAYS) {
            const uint64_t groups = DivUp(runs, MERGE_WAYS);
            for (uint64_t group = 0; group < groups; ++group) {
                const uint64_t firstShard = group * MERGE_WAYS * span;
                uint64_t starts[MERGE_WAYS] = {}, lengths[MERGE_WAYS] = {};
                uint64_t total = 0;
                uint32_t routes = 0;
                for (uint32_t i = 0; i < MERGE_WAYS && firstShard + i * span < d_->shardCount; ++i) {
                    const uint64_t shard = firstShard + i * span;
                    starts[i] = ShardBase(row, shard);
                    lengths[i] = PrefixLength(shard * schedule_->shardElements, span * schedule_->shardElements);
                    total += lengths[i];
                    ++routes;
                }
                MergeGroup(row, starts, lengths, routes, ShardBase(row, firstShard), Min<uint64_t>(d_->k, total),
                           groups == 1);
            }
            Fence<HardEvent::MTE3_MTE2>();
            srcRole_ = 1 - srcRole_;
            runs = groups;
        }
    }

    // 选择流式归并的最终输出或下一工作区平面，不改变输入段的排序语义。
    __aicore__ inline void MergeGroup(uint64_t row, const uint64_t starts[MERGE_WAYS],
                                      const uint64_t lengths[MERGE_WAYS], uint32_t routes, uint64_t destination,
                                      uint64_t count, bool finalOutput)
    {
        if (finalOutput) {
            BindConversionBuffers(d_->mergeWindow);
            Merge<true>(starts, lengths, routes, 0, count, row);
        } else {
            Merge(starts, lengths, routes, destination, count);
        }
    }

    // 单叶子最终结果先移到独立 pair 区，再复用键缓冲写出。
    __aicore__ inline void EmitSorted(uint64_t row, const LocalTensor<float>& sorted, uint32_t count)
    {
        if (sorted.GetPhyAddr() != ping_.GetPhyAddr()) {
            DataCopy(ping_.ReinterpretCast<uint32_t>(), sorted.template ReinterpretCast<uint32_t>(),
                     DivUp(count, PAIR_ALIGNMENT) * PAIR_ALIGNMENT * PAIR_COMPONENTS);
            PipeBarrier<PIPE_V>();
        }
        Emit(row, 0, ping_, count);
    }

    const SelectTopKBaseTilingData* d_ = nullptr;
    const SortSchedule* schedule_ = nullptr;
    uint32_t srcRole_ = 0;
    GlobalTensor<T> x_, values_;
    GlobalTensor<int32_t> indices_;
    GlobalTensor<int64_t> indices64_;
    GlobalTensor<float> gm_[DOUBLE_BUFFER_COUNT];
    TBuf<TPosition::VECCALC> arenaBuf_;
    LocalTensor<T> raw_;
    LocalTensor<float> key_, ping_, pong_, windows_[MERGE_WAYS], merged_;
    LocalTensor<uint32_t> index_;
};
} // namespace SelectTopK
#endif
