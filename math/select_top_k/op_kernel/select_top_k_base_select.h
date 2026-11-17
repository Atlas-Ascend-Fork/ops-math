/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SELECT_TOP_K_BASE_SELECT_H
#define SELECT_TOP_K_BASE_SELECT_H

#include "select_top_k_base_common.h"
#include "select_top_k_base_select_layout.h"
#include "select_top_k_base_index.h"

namespace SelectTopKBase {
// 通过扫描、过滤和压缩筛选候选；2201 上用有序候选归并完成压缩。
// Process 按行分配；ProcessSharded 按行内页分片。单输入缓冲在搬运完成后复用。
template <typename T>
class TopKSelect {
public:
    // 绑定输入输出并按逐页过滤布局建立候选与累积段缓冲。
    template <bool FastIndices = false>
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR values, GM_ADDR indices, const SelectTopKBaseTilingData* data,
                                TPipe* pipe, bool wideIndices = false)
    {
        data_ = data;
        wideIndices_ = wideIndices;
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
        values_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(values));
        indices_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices));
        const SelectLayout layout(data_->tile, static_cast<uint32_t>(data_->k), sizeof(T));
        pipe->InitBuffer(arena_, data_->ubBytes);
        raw_ = View<T>(layout.raw);
        key_ = View<float>(layout.key);
        baseIndex_ = View<int32_t>(layout.index);
        mask_ = View<uint8_t>(layout.mask);
        pendingKey_ = View<float>(layout.pendingKey);
        pendingIndex_ = View<uint32_t>(layout.pendingIndex);
        ping_ = View<float>(layout.ping);
        pong_ = View<float>(layout.pong);
        accumulator_ = View<float>(layout.accumulator);
        mergeAfterPing_ = layout.mergeAfterPing;
        mergeAfterPong_ = layout.mergeAfterPong;

        // 页内索引只生成一次。过滤后再加物理页起点，避免每页重新走标量播种流程。
        if constexpr (FastIndices) {
            CreateLinearIndex(baseIndex_, 0, data_->tile);
        } else {
            Fence<HardEvent::V_S>();
            ArithProgression<int32_t>(baseIndex_, 0, 1, data_->tile);
            Fence<HardEvent::S_V>();
        }
        SetMaskNorm();
        ResetMask();
        PipeBarrier<PIPE_V>();
    }

    // 逐行建立初始候选、扫描过滤各页并输出前 K 项。
    __aicore__ inline void Process()
    {
        for (uint64_t row = GetBlockIdx(); row < data_->rows; row += data_->cores) {
            accumulatorCount_ = 0;
            pendingSlots_ = 0;
            pendingValid_ = 0;
            threshold_ = SortPadding();
            const uint32_t pages = static_cast<uint32_t>(data_->tiles);
            uint32_t page = static_cast<uint32_t>((row * Tuning::ROW_PAGE_PHASE) % pages);
            for (uint32_t ordinal = 0; ordinal < pages; ++ordinal) {
                const uint32_t start = page * data_->tile;
                const uint32_t valid = Min<uint64_t>(data_->tile, data_->axis - start);
                // 负无穷是合法输入；用初始负无穷阈值严格比较无法建立完整的首批 K 项。
                if (accumulatorCount_ < data_->k && pendingSlots_ == 0) {
                    SeedPage(row, start, valid);
                } else {
                    FilterPage(row, start, valid);
                }
                if (pendingSlots_ >= data_->pendingLimit || accumulatorCount_ < data_->k || ordinal + 1 == pages) {
                    CompactCandidates();
                }
                // Host 保证步长与页数互质，每页恰好访问一次；尾页按真实长度搬运。
                page += data_->pageStep;
                if (page >= pages) {
                    page -= pages;
                }
            }
            EmitRow(row);
        }
    }

    // 将单行页面分配给多核，收集并归并各分片候选。
    template <bool UsePrefixFinalMerge = false>
    __aicore__ inline void ProcessSharded(GM_ADDR workspace)
    {
        // Host 保证 rows*shards==cores；每核恰好承担一片，不重复展开扫描与输出循环。
        const uint32_t shards = data_->shardCount;
        const uint64_t task = GetBlockIdx();
        const uint64_t row = task / shards;
        const uint32_t shard = static_cast<uint32_t>(task % shards);
        const uint32_t first = static_cast<uint32_t>(data_->tiles * shard / shards);
        const uint32_t end = static_cast<uint32_t>(data_->tiles * (shard + 1) / shards);
        SelectPageRange(row, first, end);

        if (shards > 1) {
            GlobalTensor<float> partials;
            partials.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
            const uint32_t stride = static_cast<uint32_t>((data_->k + (PAIR_ALIGNMENT - 1)) / PAIR_ALIGNMENT *
                                                          PAIR_ALIGNMENT);
            const uint64_t slot = row * data_->rowWorkspacePairs + shard * stride;
            PadPartial(stride);
            StoreToGm(partials[slot * PAIR_COMPONENTS], accumulator_, stride * PAIR_COMPONENTS);
            Fence<HardEvent::MTE3_S>();
            SyncAll();
            if constexpr (UsePrefixFinalMerge) {
                const uint32_t prefix = static_cast<uint32_t>((data_->k + (HALF_PREFIX_ALIGNMENT - 1)) /
                                                              HALF_PREFIX_ALIGNMENT * PAIR_ALIGNMENT);
                // Host 已核验前缀复用区；只需按分片数选择本核收尾或稀疏归并树。
                if (shards <= Tuning::LOCAL_FINAL_MAX_SHARDS) {
                    if (task < data_->rows) {
                        MergeFewPartials(partials, task, stride, prefix);
                        EmitRow(task);
                    }
                } else {
                    MergePartialTree(partials, stride, prefix);
                }
                return;
            }
            // 前 rows 个核分别完成一行最终归并，其余核在唯一一次同步之后退出。
            if (task >= data_->rows) {
                return;
            }
            if (shards * stride <= data_->tile * PAIR_COMPONENTS) {
                MergeSmallPartials(partials[task * data_->rowWorkspacePairs * PAIR_COMPONENTS], shards * stride,
                                   stride);
            } else {
                accumulatorCount_ = 0;
                for (uint32_t part = 0; part < shards; ++part) {
                    const uint64_t begin = data_->tiles * part / shards * data_->tile;
                    const uint64_t finish = Min<uint64_t>(data_->axis,
                                                          data_->tiles * (part + 1) / shards * data_->tile);
                    const uint32_t count = Min<uint64_t>(data_->k, finish - begin);
                    const uint64_t offset = task * data_->rowWorkspacePairs + part * stride;
                    MergePartial(partials[offset * PAIR_COMPONENTS], count);
                }
            }
        }
        // 单片时 task 就是行号；多片时只有负责最终归并的前 rows 个核能到达此处。
        EmitRow(task);
    }

private:
    // 计算给定分片区间需要保留的有效候选数量。
    __aicore__ inline uint32_t PartialRangeCount(uint32_t firstShard, uint32_t shardCount)
    {
        const uint32_t shards = data_->shardCount;
        const uint32_t endShard = Min<uint32_t>(shards, firstShard + shardCount);
        const uint64_t begin = data_->tiles * firstShard / shards * data_->tile;
        const uint64_t end = Min<uint64_t>(data_->axis, data_->tiles * endShard / shards * data_->tile);
        return Min<uint64_t>(data_->k, end - begin);
    }

    // 读取四路候选前缀并归并，必要时使用完整列表。
    __aicore__ inline bool MergeFourPrefixes(const GlobalTensor<float>& partials, uint64_t row, uint32_t first,
                                             uint32_t span, uint32_t stride, uint32_t prefix)
    {
        LocalTensor<float> input = View<float>(0);
        LocalTensor<float> output = input[prefix * MERGE_ALL_SOURCE_COMPONENTS];
        Fence<HardEvent::V_MTE2>();
        const uint64_t slot = row * data_->rowWorkspacePairs + first * stride;
        DataCopyExtParams copy{MERGE_WAYS, prefix * PAIR_BYTES, (span * stride - prefix) * PAIR_BYTES, 0, 0};
        DataCopyPad(input, partials[slot * PAIR_COMPONENTS], copy, DataCopyPadExtParams<float>{false, 0, 0, 0});
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
            // 分布偏斜时前缀不足，必须重读完整列表；不能截断未比较的候选。
            return false;
        }
        accumulatorCount_ = PartialRangeCount(first, span * MERGE_WAYS);
        DataCopy(accumulator_.ReinterpretCast<uint32_t>(), output.ReinterpretCast<uint32_t>(),
                 stride * PAIR_COMPONENTS);
        PipeBarrier<PIPE_V>();
        return true;
    }

    // 按四路树形阶段合并分片候选并同步各核。
    __aicore__ inline void MergePartialTree(const GlobalTensor<float>& partials, uint32_t stride, uint32_t prefix)
    {
        const uint32_t shards = data_->shardCount;
        const uint64_t task = GetBlockIdx();
        for (uint32_t span = 1;; span *= MERGE_WAYS) {
            const uint32_t groups = (shards + span * MERGE_WAYS - 1) / (span * MERGE_WAYS);
            if (task < data_->rows * groups) {
                const uint64_t row = task / groups;
                const uint32_t group = static_cast<uint32_t>(task % groups);
                const uint32_t first = group * span * MERGE_WAYS;
                const uint32_t sourceCount = Min<uint32_t>(MERGE_WAYS, (shards - first + span - 1) / span);
                if (sourceCount > 1 || groups == 1) {
                    const bool mergedPrefix = sourceCount == MERGE_WAYS &&
                                              MergeFourPrefixes(partials, row, first, span, stride, prefix);
                    if (!mergedPrefix) {
                        accumulatorCount_ = 0;
                        for (uint32_t part = 0; part < sourceCount; ++part) {
                            const uint32_t source = first + part * span;
                            const uint64_t slot = row * data_->rowWorkspacePairs + source * stride;
                            MergePartial<true>(partials[slot * PAIR_COMPONENTS], PartialRangeCount(source, span));
                        }
                    }
                    if (groups == 1) {
                        EmitRow(row);
                    } else {
                        // 原地写回本组的第一个槽，绝不写入其他组尚未读取的槽。
                        // 每层保留稀疏位置 0,4,8,...；下一层按 span 读取，避免压缩写回的数据竞争。
                        PadPartial(stride);
                        const uint64_t slot = row * data_->rowWorkspacePairs + first * stride;
                        StoreToGm(partials[slot * PAIR_COMPONENTS], accumulator_, stride * PAIR_COMPONENTS);
                    }
                }
            }
            if (groups == 1) {
                return;
            }
            // 全部初始核都参与每一层同步，空闲核也不能提前退出。
            SyncAll();
        }
    }

    // 由单核归并少量分片候选，减少全核同步。
    __aicore__ inline void MergeFewPartials(const GlobalTensor<float>& partials, uint64_t row, uint32_t stride,
                                            uint32_t prefix)
    {
        // 5~7 片只需一组四路归并和少量尾路，不写中间 GM，也不增加层间同步。
        if (!MergeFourPrefixes(partials, row, 0, 1, stride, prefix)) {
            accumulatorCount_ = 0;
            for (uint32_t part = 0; part < MERGE_WAYS; ++part) {
                const uint64_t slot = row * data_->rowWorkspacePairs + part * stride;
                MergePartial<true>(partials[slot * PAIR_COMPONENTS], PartialRangeCount(part, 1));
            }
        }
        for (uint32_t part = MERGE_WAYS; part < data_->shardCount; ++part) {
            const uint64_t slot = row * data_->rowWorkspacePairs + part * stride;
            MergePartial<true>(partials[slot * PAIR_COMPONENTS], PartialRangeCount(part, 1));
        }
    }

    // 对候选段补齐 DMA 对齐尾部，填入无效排序编码。
    __aicore__ inline void PadPartial(uint32_t stride)
    {
        if (accumulatorCount_ == stride) {
            return;
        }
        // 片间按 32B 对齐；显式填充无效 pair 后，可按等长段进行批量归并。
        Fence<HardEvent::V_S>();
        for (uint32_t i = accumulatorCount_; i < stride; ++i) {
            accumulator_.SetValue(i * PAIR_COMPONENTS, SortPadding());
            accumulator_.ReinterpretCast<uint32_t>().SetValue(i * PAIR_COMPONENTS + 1, UINT32_MAX);
        }
        Fence<HardEvent::S_V>();
    }

    // 一次读取小候选列表并完成最终归并。
    __aicore__ inline void MergeSmallPartials(const GlobalTensor<float>& source, uint32_t count, uint32_t stride)
    {
        Fence<HardEvent::V_MTE2>();
        DataCopyExtParams params{1, count * PAIR_BYTES, 0, 0, 0};
        DataCopyPad(ping_, source, params, DataCopyPadExtParams<float>{false, 0, 0, 0});
        Fence<HardEvent::MTE2_V>();
        LocalTensor<float> input = ping_;
        LocalTensor<float> output = pong_;
        for (uint32_t span = stride; span < count; span *= MERGE_WAYS) {
            const uint32_t groups = count / (MERGE_WAYS * span);
            if (groups != 0) {
                const uint16_t length = static_cast<uint16_t>(span);
                uint16_t lengths[MERGE_WAYS] = {length, length, length, length};
                MrgSortSrcList<float> parts(input, input[PAIR_COMPONENTS * span],
                                            input[MERGE_THIRD_SOURCE * PAIR_COMPONENTS * span],
                                            input[MERGE_FOURTH_SOURCE_COMPONENT_OFFSET * span]);
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
        accumulatorCount_ = static_cast<uint32_t>(data_->k);
        DataCopy(accumulator_.ReinterpretCast<uint32_t>(), input.ReinterpretCast<uint32_t>(),
                 (accumulatorCount_ + (PAIR_ALIGNMENT - 1)) / PAIR_ALIGNMENT * FLOAT_BLOCK_ELEMENTS);
        PipeBarrier<PIPE_V>();
    }

    // 将页面有效项追加为初始候选，建立首个 K 项累积段。
    __aicore__ inline void SeedPage(uint64_t row, uint32_t start, uint32_t valid)
    {
        // 尚未建立阈值的首块全部保留，直接生成排序输入，省去 Compare/GatherMask 的往返。
        DataCopyExtParams params{1, static_cast<uint32_t>(valid * sizeof(T)), 0, 0, 0};
        DataCopyPad(raw_, x_[row * data_->axis + start], params, DataCopyPadExtParams<T>{false, 0, 0, 0});
        Fence<HardEvent::MTE2_V>();
        EncodeSortKeys(pendingKey_, raw_, valid, data_->largest);
        Adds(pendingIndex_.ReinterpretCast<int32_t>(), baseIndex_, static_cast<int32_t>(start), valid);
        PipeBarrier<PIPE_V>();
        // 随机页面顺序可能从短尾页建立候选；追加时与 FilterPage 保持相同块对齐，再交给 SortPending 填充。
        PadAppendTail(valid);
        pendingSlots_ = (valid + (FLOAT_BLOCK_ELEMENTS - 1)) / FLOAT_BLOCK_ELEMENTS * FLOAT_BLOCK_ELEMENTS;
        pendingValid_ = valid;
        PipeBarrier<PIPE_V>();
    }

    // 在指定页面区间执行种子建立和阈值过滤。
    __aicore__ inline void SelectPageRange(uint64_t row, uint32_t first, uint32_t end)
    {
        accumulatorCount_ = 0;
        pendingSlots_ = 0;
        pendingValid_ = 0;
        threshold_ = SortPadding();
        for (uint32_t page = first; page < end; ++page) {
            const uint32_t offset = page * data_->tile;
            const uint32_t valid = Min<uint64_t>(data_->tile, data_->axis - offset);
            // 首个短页或 UB 块不足 K 项时，先补足有效累积段，再开始阈值过滤。
            const bool seed = accumulatorCount_ < data_->k && pendingSlots_ == 0;
            if (seed) {
                SeedPage(row, offset, valid);
            } else {
                FilterPage(row, offset, valid);
            }
            if (pendingSlots_ >= data_->pendingLimit || accumulatorCount_ < data_->k || page + 1 == end) {
                CompactCandidates();
            }
        }
    }

    // 将一个分片结果归并到当前有序累积段。
    template <bool FixedDescriptors = false>
    __aicore__ inline void MergePartial(const GlobalTensor<float>& source, uint32_t count)
    {
        // 读入的局部候选与 Acc 都有序；最多两份 K，完整归并后截取前 K 项。
        auto input = accumulatorCount_ == 0 ? accumulator_ : ping_;
        Fence<HardEvent::V_MTE2>();
        DataCopyExtParams params{1, count * PAIR_BYTES, 0, 0, 0};
        DataCopyPad(input, source, params, DataCopyPadExtParams<float>{false, 0, 0, 0});
        Fence<HardEvent::MTE2_V>();
        if (accumulatorCount_ == 0) {
            accumulatorCount_ = count;
            return;
        }

        if constexpr (FixedDescriptors) {
            // 两份候选各至多 K<=4096；固定拆成前 2048 项与尾段，避免动态 LocalTensor 数组。
            const uint16_t accHead = static_cast<uint16_t>(Min<uint32_t>(SORT_STAGE4_ITEMS, accumulatorCount_));
            const uint16_t srcHead = static_cast<uint16_t>(Min<uint32_t>(SORT_STAGE4_ITEMS, count));
            const uint16_t accTail = static_cast<uint16_t>(accumulatorCount_ - accHead);
            const uint16_t srcTail = static_cast<uint16_t>(count - srcHead);
            uint16_t lengths[MERGE_WAYS] = {accHead, srcHead, accTail, srcTail};
            const uint16_t valid = static_cast<uint16_t>(MERGE_TWO_SOURCES_MASK |
                                                         (accTail ? MERGE_THIRD_SOURCE_MASK : 0) |
                                                         (srcTail ? MERGE_FOURTH_SOURCE_MASK : 0));
            MrgSortSrcList<float> sources(accumulator_, ping_, accumulator_[SORT_STAGE4_ITEMS * PAIR_COMPONENTS],
                                          ping_[SORT_STAGE4_ITEMS * PAIR_COMPONENTS]);
            LocalTensor<float> output = View<float>(mergeAfterPing_);
            MrgSort4Info info(lengths, false, valid, 1);
            MrgSort(output, sources, info);
            PipeBarrier<PIPE_V>();
            accumulatorCount_ = Min<uint32_t>(static_cast<uint32_t>(data_->k), accumulatorCount_ + count);
            DataCopy(accumulator_.ReinterpretCast<uint32_t>(), output.ReinterpretCast<uint32_t>(),
                     (accumulatorCount_ + (PAIR_ALIGNMENT - 1)) / PAIR_ALIGNMENT * FLOAT_BLOCK_ELEMENTS);
            PipeBarrier<PIPE_V>();
            return;
        }

        LocalTensor<float> parts[MERGE_WAYS] = {ping_, ping_, ping_, ping_};
        uint16_t lengths[MERGE_WAYS] = {0, 0, 0, 0};
        uint32_t routes = 0;
        // K=4096 时单条序列拆成两段，保证每路长度不超过硬件的 4095 限制。
        for (uint32_t offset = 0; offset < accumulatorCount_; offset += SORT_STAGE4_ITEMS) {
            parts[routes] = accumulator_[offset * PAIR_COMPONENTS];
            lengths[routes] = static_cast<uint16_t>(Min<uint32_t>(SORT_STAGE4_ITEMS, accumulatorCount_ - offset));
            ++routes;
        }
        for (uint32_t offset = 0; offset < count; offset += SORT_STAGE4_ITEMS) {
            parts[routes] = ping_[offset * PAIR_COMPONENTS];
            lengths[routes] = static_cast<uint16_t>(Min<uint32_t>(SORT_STAGE4_ITEMS, count - offset));
            ++routes;
        }
        LocalTensor<float> merged = View<float>(mergeAfterPing_);
        MrgSortSrcList<float> inputs(parts[0], parts[1], parts[MERGE_THIRD_SOURCE], parts[MERGE_FOURTH_SOURCE]);
        MrgSort4Info info(lengths, false, static_cast<uint16_t>((1U << routes) - 1), 1);
        MrgSort(merged, inputs, info);
        PipeBarrier<PIPE_V>();
        accumulatorCount_ = Min<uint32_t>(static_cast<uint32_t>(data_->k), accumulatorCount_ + count);
        const uint32_t copyWords = (accumulatorCount_ + (PAIR_ALIGNMENT - 1)) / PAIR_ALIGNMENT * FLOAT_BLOCK_ELEMENTS;
        DataCopy(accumulator_.ReinterpretCast<uint32_t>(), merged.ReinterpretCast<uint32_t>(), copyWords);
        PipeBarrier<PIPE_V>();
    }

    // 按字节偏移取得指定类型的 UB 局部张量视图。
    template <typename U>
    __aicore__ inline LocalTensor<U> View(uint32_t byteOffset)
    {
        return arena_.Get<U>()[byteOffset / sizeof(U)];
    }

    // 补齐候选追加位置的块对齐尾部，避免未初始化值进入排序。
    __aicore__ inline void PadAppendTail(uint32_t selected)
    {
        const uint32_t tail = selected % FLOAT_BLOCK_ELEMENTS;
        if (tail == 0) {
            return;
        }
        // 每段候选按 8 个 FP32/UINT32 槽位对齐。只改最后一个 32B 块中的无效位置。
        const uint32_t alignedStart = pendingSlots_ + selected - tail;
        uint64_t paddingMask[VECTOR_MASK_WORDS] = {MASK_BYTE_ALL_BITS & ~((1ULL << tail) - 1), 0};
        Duplicate(pendingKey_[alignedStart], SortPadding(), paddingMask, 1, 1, VECTOR_REPEAT_BLOCKS);
        Duplicate(pendingIndex_[alignedStart], UINT32_MAX, paddingMask, 1, 1, VECTOR_REPEAT_BLOCKS);
        PipeBarrier<PIPE_V>();
    }

    // 加载一页并生成阈值掩码，将入选键和索引追加到候选区。
    __aicore__ inline void FilterPage(uint64_t row, uint32_t start, uint32_t valid)
    {
        DataCopyExtParams params{1, static_cast<uint32_t>(valid * sizeof(T)), 0, 0, 0};
        DataCopyPad(raw_, x_[row * data_->axis + start], params, DataCopyPadExtParams<T>{false, 0, 0, 0});
        Fence<HardEvent::MTE2_V>();

        // CompareScalar 按完整 256B repeat 工作；扫描块为 64 的倍数，尾部 key 固定为 -Inf。
        if (valid < data_->tile) {
            Duplicate(key_, SortPadding(), data_->tile);
            PipeBarrier<PIPE_V>();
        }
        EncodeSortKeys(key_, raw_, valid, data_->largest);
        PipeBarrier<PIPE_V>();
        FilterSortKeys<true>(mask_, raw_.template ReinterpretCast<uint8_t>(), key_, threshold_, valid, data_->tile,
                             data_->largest);
        PipeBarrier<PIPE_V>();

        uint64_t selectedValues = 0;
        uint64_t selectedIndices = 0;
        const GatherMaskParams gatherParams(1, 1, VECTOR_REPEAT_BLOCKS, 0);
        GatherMask(pendingKey_[pendingSlots_], key_, mask_.ReinterpretCast<uint32_t>(), true, data_->tile, gatherParams,
                   selectedValues);
        GatherMask(pendingIndex_[pendingSlots_], baseIndex_.ReinterpretCast<uint32_t>(),
                   mask_.ReinterpretCast<uint32_t>(), true, data_->tile, gatherParams, selectedIndices);
        Fence<HardEvent::V_S>();
        const uint32_t selected = static_cast<uint32_t>(selectedIndices);
        if (selected == 0) {
            return;
        }
        Adds(pendingIndex_[pendingSlots_].ReinterpretCast<int32_t>(),
             pendingIndex_[pendingSlots_].ReinterpretCast<int32_t>(), static_cast<int32_t>(start), selected);
        PipeBarrier<PIPE_V>();
        PadAppendTail(selected);
        pendingSlots_ += (selected + (FLOAT_BLOCK_ELEMENTS - 1)) / FLOAT_BLOCK_ELEMENTS * FLOAT_BLOCK_ELEMENTS;
        pendingValid_ += selected;
    }

    // 排序待处理候选并与当前累积段合并，更新过滤阈值。
    __aicore__ inline LocalTensor<float> SortPending(uint32_t padded)
    {
        if (padded > pendingSlots_) {
            Duplicate(pendingKey_[pendingSlots_], SortPadding(), padded - pendingSlots_);
            Duplicate(pendingIndex_[pendingSlots_], UINT32_MAX, padded - pendingSlots_);
        }
        PipeBarrier<PIPE_V>();
        // Sort32 固定写 ping，避免在排序入口按层数奇偶选择整份 LocalTensor。
        // 最终结果允许落在ping或pong；CompactCandidates只选择归并输出的字节偏移。
        Sort32(ping_, pendingKey_, pendingIndex_, padded / SORT32_ELEMENTS);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> source = ping_;
        LocalTensor<float> destination = pong_;
        for (uint32_t span = SORT32_ELEMENTS; span < padded; span *= MERGE_WAYS) {
            const uint32_t groups = padded / (MERGE_WAYS * span);
            if (groups != 0) {
                const uint16_t length = static_cast<uint16_t>(span);
                uint16_t lengths[MERGE_WAYS] = {length, length, length, length};
                MrgSortSrcList<float> inputs(source, source[PAIR_COMPONENTS * span],
                                             source[MERGE_THIRD_SOURCE * PAIR_COMPONENTS * span],
                                             source[MERGE_FOURTH_SOURCE_COMPONENT_OFFSET * span]);
                MrgSort4Info info(lengths, false, MERGE_ALL_SOURCES_MASK, static_cast<uint8_t>(groups));
                MrgSort(destination, inputs, info);
            }
            const uint32_t tailStart = groups * MERGE_WAYS * span;
            const uint32_t tail = padded - tailStart;
            if (tail != 0 && tail <= span) {
                DataCopy(destination[PAIR_COMPONENTS * tailStart].ReinterpretCast<uint32_t>(),
                         source[PAIR_COMPONENTS * tailStart].ReinterpretCast<uint32_t>(), tail * PAIR_COMPONENTS);
            } else if (tail > span) {
                LocalTensor<float> parts[MERGE_WAYS] = {source, source, source, source};
                uint16_t lengths[MERGE_WAYS] = {0, 0, 0, 0};
                uint32_t count = 0;
                for (uint32_t offset = tailStart; offset < padded; offset += span) {
                    parts[count] = source[PAIR_COMPONENTS * offset];
                    lengths[count] = static_cast<uint16_t>(Min<uint32_t>(span, padded - offset));
                    ++count;
                }
                MrgSortSrcList<float> inputs(parts[0], parts[1], parts[MERGE_THIRD_SOURCE], parts[MERGE_FOURTH_SOURCE]);
                MrgSort4Info info(lengths, false, static_cast<uint16_t>((1U << count) - 1), 1);
                MrgSort(destination[PAIR_COMPONENTS * tailStart], inputs, info);
            }
            PipeBarrier<PIPE_V>();
            auto temporary = source;
            source = destination;
            destination = temporary;
        }
        return source;
    }

    // 压缩并归并新候选，只保留可能进入前 K 的部分。
    __aicore__ inline void CompactCandidates()
    {
        if (pendingSlots_ == 0) {
            return;
        }
        const uint32_t padded = (pendingSlots_ + (SORT32_ELEMENTS - 1)) / SORT32_ELEMENTS * SORT32_ELEMENTS;
        auto sorted = SortPending(padded);
        LocalTensor<float> result = sorted;
        if (accumulatorCount_ != 0) {
            // 每层PIPE_V屏障已完成排序。只复用middle和未保存最终结果的另一侧缓冲。
            // 先选择标量偏移，再创建视图，避免把LocalTensor条件选择带入排序循环。
            const uint32_t mergeOffset = sorted.GetPhyAddr() == ping_.GetPhyAddr() ? mergeAfterPing_ : mergeAfterPong_;
            auto merged = View<float>(mergeOffset);
            // 新候选仅前 K 项可能进入结果；固定描述符在硬件安全的段长边界拆分各路。
            const uint32_t selected = Min<uint32_t>(padded, data_->k);
            const uint16_t accHead = Min<uint32_t>(SORT_STAGE4_ITEMS, accumulatorCount_);
            const uint16_t srcHead = Min<uint32_t>(SORT_STAGE4_ITEMS, selected);
            const uint16_t accTail = accumulatorCount_ - accHead;
            const uint16_t srcTail = selected - srcHead;
            uint16_t lengths[MERGE_WAYS] = {accHead, srcHead, accTail, srcTail};
            const uint16_t valid = static_cast<uint16_t>(MERGE_TWO_SOURCES_MASK |
                                                         (accTail ? MERGE_THIRD_SOURCE_MASK : 0) |
                                                         (srcTail ? MERGE_FOURTH_SOURCE_MASK : 0));
            MrgSortSrcList<float> inputs(accumulator_, sorted, accumulator_[SORT_STAGE4_ITEMS * PAIR_COMPONENTS],
                                         sorted[SORT_STAGE4_ITEMS * PAIR_COMPONENTS]);
            MrgSort4Info info(lengths, false, valid, 1);
            MrgSort(merged, inputs, info);
            PipeBarrier<PIPE_V>();
            result = merged;
        }
        accumulatorCount_ = Min<uint32_t>(static_cast<uint32_t>(data_->k), accumulatorCount_ + pendingValid_);
        const uint32_t copyWords = (accumulatorCount_ + (PAIR_ALIGNMENT - 1)) / PAIR_ALIGNMENT * FLOAT_BLOCK_ELEMENTS;
        DataCopy(accumulator_.ReinterpretCast<uint32_t>(), result.ReinterpretCast<uint32_t>(), copyWords);
        Fence<HardEvent::V_S>();
        // 只有 K 个真实元素就绪后才提高阈值；对齐填充不计入有效候选数。
        if (accumulatorCount_ == data_->k) {
            threshold_ = accumulator_.GetValue((accumulatorCount_ - 1) * PAIR_COMPONENTS);
        }
        pendingSlots_ = 0;
        pendingValid_ = 0;
    }

    // 分块解码行结果并写回值和指定宽度的原轴索引。
    __aicore__ inline void EmitRow(uint64_t row)
    {
        for (uint32_t offset = 0; offset < data_->k; offset += data_->outputTile) {
            const uint32_t count = Min<uint64_t>(data_->outputTile, data_->k - offset);
            uint64_t selected = 0;
            const GatherMaskParams params(1, 1, VECTOR_REPEAT_BLOCKS, 0);
            GatherMask(key_, accumulator_[offset * PAIR_COMPONENTS], uint8_t(1), true, count * PAIR_COMPONENTS, params,
                       selected);
            // 输出索引复用候选区，保留 baseIndex_ 供下一行继续过滤。
            GatherMask(pendingIndex_, accumulator_[offset * PAIR_COMPONENTS].ReinterpretCast<uint32_t>(),
                       uint8_t(GATHER_INDEX_PATTERN), true, count * PAIR_COMPONENTS, params, selected);
            PipeBarrier<PIPE_V>();
            if (!data_->largest) {
                Adds(key_.template ReinterpretCast<int32_t>(), key_.template ReinterpretCast<int32_t>(), INT32_MIN,
                     count);
                PipeBarrier<PIPE_V>();
            }
            if constexpr (std::is_same<T, float>::value) {
                RestoreNaNPayload<float>(key_, key_, pendingIndex_,
                                         reinterpret_cast<const __gm__ float*>(x_.GetPhyAddr()), row * data_->axis,
                                         count, data_->largest);
                StoreToGm(values_[row * data_->k + offset], key_, count);
            } else {
                Cast(raw_, key_, RoundMode::CAST_RINT, count);
                PipeBarrier<PIPE_V>();
                RestoreNaNPayload<T>(raw_, key_, pendingIndex_, reinterpret_cast<const __gm__ T*>(x_.GetPhyAddr()),
                                     row * data_->axis, count, data_->largest);
                StoreToGm(values_[row * data_->k + offset], raw_, count);
            }
            if (wideIndices_) {
                // 输出阶段的候选索引与 ping 缓冲互不重叠；有序累积段仍需供后续输出块读取。
                auto wide = ping_.ReinterpretCast<int64_t>();
                Cast(wide, pendingIndex_.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, count);
                PipeBarrier<PIPE_V>();
                GlobalTensor<int64_t> destination;
                destination.SetGlobalBuffer((__gm__ int64_t*)indices_.GetPhyAddr());
                StoreToGm(destination[row * data_->k + offset], wide, count);
            } else {
                StoreToGm(indices_[row * data_->k + offset], pendingIndex_.ReinterpretCast<int32_t>(), count);
            }
        }
    }

    const SelectTopKBaseTilingData* data_ = nullptr;
    TBuf<TPosition::VECCALC> arena_;
    GlobalTensor<T> x_;
    GlobalTensor<T> values_;
    GlobalTensor<int32_t> indices_;
    bool wideIndices_ = false;
    LocalTensor<T> raw_;
    LocalTensor<float> key_;
    LocalTensor<int32_t> baseIndex_;
    LocalTensor<uint8_t> mask_;
    LocalTensor<float> pendingKey_;
    LocalTensor<uint32_t> pendingIndex_;
    LocalTensor<float> ping_;
    LocalTensor<float> pong_;
    LocalTensor<float> accumulator_;
    uint32_t mergeAfterPing_ = 0;
    uint32_t mergeAfterPong_ = 0;
    uint32_t accumulatorCount_ = 0;
    uint32_t pendingSlots_ = 0;
    uint32_t pendingValid_ = 0;
    float threshold_ = SortPadding();
};
} // namespace SelectTopKBase

#endif
