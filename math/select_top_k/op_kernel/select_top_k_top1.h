/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SELECT_TOP_K_TOP1_H_
#define SELECT_TOP_K_TOP1_H_
#include "select_top_k_tiling_data.h"
#include "select_top_k_base_common.h"

namespace SelectTopK {
using namespace SelectTopKBase;
// K=1 使用归约，每个分片保留一个原轴索引；
// 最终按该索引复制输入位模式，保留正负零与 NaN 载荷。
template <typename T, bool WideIndices = false>
class Top1 {
    using Bits = typename std::conditional<sizeof(T) == VALUE16_BYTES, uint16_t, uint32_t>::type;
    using Index = typename std::conditional<WideIndices, int64_t, int32_t>::type;

public:
    // 并行扫描各分片并归约最优项，必要时汇总分片结果后按原索引写回。
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR values, GM_ADDR indices, GM_ADDR workspace,
                               const SelectTopKBaseTilingData& data, TPipe* pipe)
    {
        GlobalTensor<T> input;
        GlobalTensor<Bits> inputBits, outputBits;
        GlobalTensor<Index> outputIndices;
        GlobalTensor<float> partial;
        input.SetGlobalBuffer((__gm__ T*)x);
        inputBits.SetGlobalBuffer((__gm__ Bits*)x);
        outputBits.SetGlobalBuffer((__gm__ Bits*)values);
        outputIndices.SetGlobalBuffer((__gm__ Index*)indices);
        partial.SetGlobalBuffer((__gm__ float*)workspace);
        TBuf<TPosition::VECCALC> arena;
        pipe->InitBuffer(arena, data.ubBytes);
        auto raw = arena.Get<T>();
        auto key = arena.Get<float>()[TOP1_TILE * sizeof(T) / VALUE32_BYTES];
        auto temporary = key[TOP1_TILE];
        auto result = temporary[TOP1_REDUCE_TEMP_ITEMS];
        const uint64_t span = DivUp(data.axis, data.shardCount);
        for (uint64_t task = GetBlockIdx(); task < data.rows * data.shardCount; task += data.cores) {
            const uint64_t row = task / data.shardCount;
            const uint32_t shard = task % data.shardCount;
            const uint64_t first = shard * span;
            const uint64_t end = Min<uint64_t>(data.axis, first + span);
            float best = 0;
            uint32_t bestIndex = UINT32_MAX;
            for (uint64_t start = first; start < end; start += TOP1_TILE) {
                const uint32_t count = Min<uint64_t>(TOP1_TILE, end - start);
                DataCopyPad(raw, input[row * data.axis + start],
                            DataCopyExtParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0},
                            DataCopyPadExtParams<T>{false, 0, 0, 0});
                Fence<HardEvent::MTE2_V>();
                if constexpr (std::is_same<T, float>::value) {
                    DataCopy(key.ReinterpretCast<uint32_t>(), raw.template ReinterpretCast<uint32_t>(),
                             static_cast<uint32_t>(DivUp(count, FLOAT_BLOCK_ELEMENTS) * FLOAT_BLOCK_ELEMENTS));
                } else {
                    Cast(key, raw, RoundMode::CAST_NONE, count);
                }
                PipeBarrier<PIPE_V>();
                Reduce(result, key, temporary, count, data.largest);
                const float value = result.GetValue(0);
                if (bestIndex == UINT32_MAX || Better(value, best, data.largest)) {
                    best = value;
                    bestIndex = start + result.ReinterpretCast<uint32_t>().GetValue(1);
                }
                Fence<HardEvent::S_V>();
            }
            if (data.shardCount == 1) {
                Store(row, bestIndex, inputBits, outputBits, outputIndices, raw, result, data.axis);
            } else {
                Fence<HardEvent::V_S>();
                result.SetValue(0, best);
                result.ReinterpretCast<uint32_t>().SetValue(1, bestIndex);
                Fence<HardEvent::S_V>();
                StoreToGm(partial[task * PAIR_COMPONENTS], result, PAIR_COMPONENTS);
            }
        }
        if (data.shardCount == 1)
            return;
        SyncAll();
        for (uint64_t row = GetBlockIdx(); row < data.rows; row += data.cores) {
            DataCopyPad(raw.template ReinterpretCast<float>(), partial[row * data.shardCount * PAIR_COMPONENTS],
                        DataCopyExtParams{1, data.shardCount * PAIR_BYTES, 0, 0, 0},
                        DataCopyPadExtParams<float>{false, 0, 0, 0});
            Fence<HardEvent::MTE2_V>();
            uint64_t selected = 0;
            GatherMask(key, raw.template ReinterpretCast<float>(), uint8_t(1), true, data.shardCount * PAIR_COMPONENTS,
                       GatherMaskParams(1, 1, VECTOR_REPEAT_BLOCKS, 0), selected);
            PipeBarrier<PIPE_V>();
            Reduce(result, key, temporary, data.shardCount, data.largest);
            const uint32_t shard = result.ReinterpretCast<uint32_t>().GetValue(1);
            const uint32_t index = raw.template ReinterpretCast<uint32_t>().GetValue(shard * PAIR_COMPONENTS + 1);
            Store(row, index, inputBits, outputBits, outputIndices, raw, result, data.axis);
        }
    }

private:
    // 按 largest 和 NaN 排序规则判断新值是否优于当前值。
    __aicore__ inline bool Better(float value, float best, bool largest)
    {
        if (value != value)
            return largest && best == best;
        if (best != best)
            return !largest;
        return largest ? value > best : value < best;
    }
    // 执行单页归约，并修正取最小值时含 NaN 页面的结果。
    __aicore__ inline void Reduce(const LocalTensor<float>& result, const LocalTensor<float>& key,
                                  const LocalTensor<float>& temporary, uint32_t count, bool largest)
    {
        if (largest)
            ReduceMax(result, key, temporary, count, true);
        else
            ReduceMin(result, key, temporary, count, true);
        Fence<HardEvent::V_S>();
        // 硬件最大、最小归约都会传播 NaN；取最小时，含 NaN 的页面需回退到标量比较。
        const float reduced = result.GetValue(0);
        if (!largest && reduced != reduced) {
            float best = reduced;
            uint32_t index = result.ReinterpretCast<uint32_t>().GetValue(1);
            for (uint32_t i = 0; i < count; ++i) {
                const float value = key.GetValue(i);
                if (value == value && (best != best || value < best)) {
                    best = value;
                    index = i;
                }
            }
            result.SetValue(0, best);
            result.ReinterpretCast<uint32_t>().SetValue(1, index);
        }
    }
    // 从原输入取回位模式；标准模式直接写INT64，其他路径保留INT32供既有收尾偏移。
    __aicore__ inline void Store(uint64_t row, uint32_t index, const GlobalTensor<Bits>& input,
                                 const GlobalTensor<Bits>& values, const GlobalTensor<Index>& indices,
                                 const LocalTensor<T>& raw, const LocalTensor<float>& result, uint64_t axis)
    {
        Fence<HardEvent::V_S>();
        raw.template ReinterpretCast<Bits>().SetValue(0, input.GetValue(row * axis + index));
        result.template ReinterpretCast<Index>().SetValue(0, static_cast<Index>(index));
        Fence<HardEvent::S_V>();
        StoreToGm(values[row], raw.template ReinterpretCast<Bits>(), 1);
        StoreToGm(indices[row], result.template ReinterpretCast<Index>(), 1);
    }
};
} // namespace SelectTopK
#endif
