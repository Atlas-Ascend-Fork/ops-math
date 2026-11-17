/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SELECT_TOP_K_BASE_WIDE_SELECT_H
#define SELECT_TOP_K_BASE_WIDE_SELECT_H

#include "select_top_k_base_common.h"
#include "select_top_k_base_select_layout.h"
#include "select_top_k_base_index.h"

namespace SelectTopKBase {
// 少行、小 K 宽块过滤：先从种子建立安全阈值，再扫描大块，只排序实际入选项。
// 输入分布影响候选数量，但不影响结果；最坏情况下分批处理全部入选项。
template <typename T>
class TopKWideSelect {
public:
    // 绑定宽块过滤的输入、输出、分片工作区和 UB 布局。
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices, GM_ADDR workspace,
                                const SelectTopKBaseTilingData* data, TPipe* pipe, bool wideIndices = false)
    {
        data_ = data;
        wideIndices_ = wideIndices;
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        values_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(values));
        indices_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices));
        partials_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
        const WideSelectLayout layout(data_->tile, static_cast<uint32_t>(data_->k));
        pipe->InitBuffer(arena_, data_->ubBytes);
        key_ = arena_.Get<float>()[layout.key / VALUE32_BYTES];
        index_ = arena_.Get<uint32_t>()[layout.index / VALUE32_BYTES];
        value_ = arena_.Get<float>()[layout.value / VALUE32_BYTES];
        raw_ = value_.ReinterpretCast<T>();
        mask_ = arena_.Get<uint8_t>()[layout.mask];
        accumulator_ = arena_.Get<float>()[layout.accumulator / VALUE32_BYTES];
        finalCapacity_ = Min<uint32_t>(Tuning::FINAL_MERGE_CAPACITY,
                                       layout.accumulator / DOUBLE_PAIR_BYTES / SORT32_ELEMENTS * SORT32_ELEMENTS);
    }

    // 各核扫描分片、过滤候选并汇总每行最终结果。
    __aicore__ inline void Process()
    {
        const uint32_t shards = data_->shardCount;
        for (uint64_t task = GetBlockIdx(); task < data_->rows * shards; task += data_->cores) {
            const uint64_t row = task / shards;
            const uint32_t shard = task % shards;
            const uint32_t begin = data_->axis * shard / shards;
            const uint32_t end = data_->axis * (shard + 1) / shards;
            Seed(row, begin);
            for (uint32_t start = begin + data_->seedItems; start < end; start += data_->tile) {
                Filter(row, start, Min<uint32_t>(data_->tile, end - start));
            }
            if (shards == 1) {
                Emit(row);
                continue;
            }
            const uint32_t stride = static_cast<uint32_t>((data_->k + (PAIR_ALIGNMENT - 1)) / PAIR_ALIGNMENT *
                                                          PAIR_ALIGNMENT);
            // GM 槽位补齐部分必须是无效项，不能让 Acc 的历史尾部参与后续归并。
            Fence<HardEvent::V_S>();
            for (uint32_t i = data_->k; i < stride; ++i) {
                accumulator_.SetValue(i * PAIR_COMPONENTS, SortPadding());
            }
            Fence<HardEvent::S_MTE3>();
            Fence<HardEvent::V_MTE3>();
            DataCopyExtParams copy{1, stride * PAIR_BYTES, 0, 0, 0};
            DataCopyPad(partials_[(row * data_->rowWorkspacePairs + shard * stride) * PAIR_COMPONENTS], accumulator_,
                        copy);
            SyncAll();
            if (task < data_->rows) {
                MergePartials(task, shards, stride);
                Emit(task);
            }
        }
    }

private:
    // 读取指定输入片段并等待 DMA 完成。
    __aicore__ inline void Load(uint64_t row, uint32_t start, uint32_t count)
    {
        // 上一轮过滤/输出完成后才能复用 raw/value。
        Fence<HardEvent::V_MTE2>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPad(raw_, x_[row * data_->axis + start], copy, DataCopyPadExtParams<T>{false, 0, 0, 0});
    }

    // 生成排序键和原轴索引，填充无效尾部。
    __aicore__ inline void MakeKeys(uint32_t count, uint32_t padded)
    {
        if (count < padded) {
            Duplicate(key_, SortPadding(), padded);
            PipeBarrier<PIPE_V>();
        }
        Fence<HardEvent::MTE2_V>();
        EncodeSortKeys(key_, raw_, count, data_->largest);
        PipeBarrier<PIPE_V>();
    }

    // 归并两路候选，更新当前前 K 项。
    __aicore__ inline LocalTensor<float> MergeRuns(LocalTensor<float> input, LocalTensor<float> output, uint32_t count,
                                                   uint32_t span)
    {
        for (; span < count; span *= MERGE_WAYS) {
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

    // 将当前有序段复制到可持续保留的累积缓冲。
    __aicore__ inline void SaveAccumulator(const LocalTensor<float>& sorted)
    {
        const uint32_t words = static_cast<uint32_t>((data_->k + (PAIR_ALIGNMENT - 1)) / PAIR_ALIGNMENT *
                                                     FLOAT_BLOCK_ELEMENTS);
        DataCopy(accumulator_.ReinterpretCast<uint32_t>(), sorted.ReinterpretCast<uint32_t>(), words);
        Fence<HardEvent::V_S>();
        threshold_ = accumulator_.GetValue((data_->k - 1) * PAIR_COMPONENTS);
    }

    // 排序初始输入片段，建立完整有效的 K 项累积段。
    __aicore__ inline void Seed(uint64_t row, uint32_t start)
    {
        Load(row, start, data_->seedItems);
        CreateLinearIndex(index_.ReinterpretCast<int32_t>(), start, data_->seedItems);
        MakeKeys(data_->seedItems, data_->seedItems);
        // 原值在 Cast 后失效，排序输出可覆盖 value；原 key 在 Sort32 后也可复用。
        Sort32(value_, key_, index_, data_->seedItems / SORT32_ELEMENTS);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> sorted = MergeRuns(value_, key_, data_->seedItems, SORT32_ELEMENTS);
        SaveAccumulator(sorted);
    }

    // 将候选数补齐到 Sort32 粒度，写入无效编码。
    __aicore__ inline void PadCandidates(uint32_t selected)
    {
        const uint32_t aligned = selected / FLOAT_BLOCK_ELEMENTS * FLOAT_BLOCK_ELEMENTS;
        const uint32_t tail = selected % FLOAT_BLOCK_ELEMENTS;
        if (tail != 0) {
            uint64_t paddingMask[VECTOR_MASK_WORDS] = {MASK_BYTE_ALL_BITS & ~((1ULL << tail) - 1), 0};
            Duplicate(value_[aligned], SortPadding(), paddingMask, 1, 1, VECTOR_REPEAT_BLOCKS);
        }
        const uint32_t next = (selected + (FLOAT_BLOCK_ELEMENTS - 1)) / FLOAT_BLOCK_ELEMENTS * FLOAT_BLOCK_ELEMENTS;
        const uint32_t padded = (selected + (SORT32_ELEMENTS - 1)) / SORT32_ELEMENTS * SORT32_ELEMENTS;
        if (next < padded) {
            Duplicate(value_[next], SortPadding(), padded - next);
        }
        PipeBarrier<PIPE_V>();
    }

    // 排序新候选并与已有累积段合并。
    __aicore__ inline void Combine(const LocalTensor<float>& source, const LocalTensor<float>& output, uint32_t count)
    {
        uint16_t lengths[MERGE_WAYS] = {static_cast<uint16_t>(data_->k), static_cast<uint16_t>(count), 0, 0};
        MrgSortSrcList<float> parts(accumulator_, source, source, source);
        MrgSort4Info info(lengths, false, MERGE_TWO_SOURCES_MASK, 1);
        MrgSort(output, parts, info);
        PipeBarrier<PIPE_V>();
        SaveAccumulator(output);
    }

    // 根据当前阈值筛选宽输入块，并分批处理入选项。
    __aicore__ inline void Filter(uint64_t row, uint32_t start, uint32_t count)
    {
        Load(row, start, count);
        const uint32_t padded = (count + (FLOAT_VECTOR_ELEMENTS - 1)) / FLOAT_VECTOR_ELEMENTS * FLOAT_VECTOR_ELEMENTS;
        CreateLinearIndex(index_.ReinterpretCast<int32_t>(), start, padded);
        MakeKeys(count, padded);
        FilterSortKeys(mask_, raw_.template ReinterpretCast<uint8_t>(), key_, threshold_, count, padded,
                       data_->largest);
        PipeBarrier<PIPE_V>();
        uint64_t selectedValues = 0;
        uint64_t selectedIndices = 0;
        const GatherMaskParams params(1, 1, VECTOR_REPEAT_BLOCKS, 0);
        GatherMask(value_, key_, mask_.ReinterpretCast<uint32_t>(), true, padded, params, selectedValues);
        // 必须先完成 key 的读取，再将压缩后的索引写回 key 区。
        PipeBarrier<PIPE_V>();
        GatherMask(key_.ReinterpretCast<uint32_t>(), index_, mask_.ReinterpretCast<uint32_t>(), true, padded, params,
                   selectedIndices);
        Fence<HardEvent::V_S>();
        const uint32_t selected = static_cast<uint32_t>(selectedIndices);
        if (selected == 0) {
            return;
        }
        PadCandidates(selected);
        // 筛选后的 SoA 保留在 value/key；完整 index 已失效，复用为两个排序缓冲。
        LocalTensor<float> ping = index_.ReinterpretCast<float>();
        LocalTensor<float> pong = ping[data_->candidateSortTile * PAIR_COMPONENTS];
        for (uint32_t offset = 0; offset < selected; offset += data_->candidateSortTile) {
            const uint32_t valid = Min<uint32_t>(data_->candidateSortTile, selected - offset);
            const uint32_t sortCount = (valid + (SORT32_ELEMENTS - 1)) / SORT32_ELEMENTS * SORT32_ELEMENTS;
            Sort32(ping, value_[offset], key_[offset].ReinterpretCast<uint32_t>(), sortCount / SORT32_ELEMENTS);
            PipeBarrier<PIPE_V>();
            LocalTensor<float> sorted = MergeRuns(ping, pong, sortCount, SORT32_ELEMENTS);
            LocalTensor<float> output = sorted.GetPhyAddr() == ping.GetPhyAddr() ? pong : ping;
            Combine(sorted, output, Min<uint32_t>(static_cast<uint32_t>(data_->k), valid));
        }
    }

    // 归并各核写出的分片候选，生成行级结果。
    __aicore__ inline void MergePartials(uint64_t row, uint32_t shards, uint32_t stride)
    {
        // 此时三份扫描数组均失效，两份候选缓冲覆盖其前部；Acc 保持独立。
        LocalTensor<float> ping = arena_.Get<float>();
        LocalTensor<float> pong = ping[finalCapacity_ * PAIR_COMPONENTS];
        const uint32_t batchShards = finalCapacity_ / stride;
        bool haveAccumulator = false;
        for (uint32_t first = 0; first < shards; first += batchShards) {
            const uint32_t count = Min<uint32_t>(batchShards, shards - first) * stride;
            if (haveAccumulator) {
                Fence<HardEvent::V_MTE2>();
            }
            DataCopyExtParams copy{1, count * PAIR_BYTES, 0, 0, 0};
            DataCopyPad(ping, partials_[(row * data_->rowWorkspacePairs + first * stride) * PAIR_COMPONENTS], copy,
                        DataCopyPadExtParams<float>{false, 0, 0, 0});
            Fence<HardEvent::MTE2_V>();
            LocalTensor<float> sorted = MergeRuns(ping, pong, count, stride);
            if (haveAccumulator) {
                LocalTensor<float> output = sorted.GetPhyAddr() == ping.GetPhyAddr() ? pong : ping;
                Combine(sorted, output, static_cast<uint32_t>(data_->k));
            } else {
                SaveAccumulator(sorted);
                haveAccumulator = true;
            }
        }
    }

    // 恢复候选对应的原值位模式，输出 values 和索引。
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
        // 半精度输出位于原值缓冲，FP32 输出位于键缓冲；另一份空闲缓冲用于 INT64 索引。
        auto wide = value_.ReinterpretCast<int64_t>();
        if constexpr (!std::is_same<T, float>::value)
            wide = key_.ReinterpretCast<int64_t>();
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

    const SelectTopKBaseTilingData* data_ = nullptr;
    TBuf<TPosition::VECCALC> arena_;
    GlobalTensor<T> x_, values_;
    GlobalTensor<int32_t> indices_;
    bool wideIndices_ = false;
    GlobalTensor<float> partials_;
    LocalTensor<float> key_, value_, accumulator_;
    LocalTensor<uint32_t> index_;
    LocalTensor<uint8_t> mask_;
    LocalTensor<T> raw_;
    uint32_t finalCapacity_ = 0;
    float threshold_ = SortPadding();
};
} // namespace SelectTopKBase
#endif
