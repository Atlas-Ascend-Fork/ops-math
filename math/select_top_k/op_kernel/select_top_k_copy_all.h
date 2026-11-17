/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SELECT_TOP_K_COPY_ALL_H_
#define SELECT_TOP_K_COPY_ALL_H_
#include "select_top_k_tiling_data.h"
#include "select_top_k_base_common.h"
#include "select_top_k_base_index.h"
namespace SelectTopK {
using namespace SelectTopKBase;
// K=N且无需排序时，所有输入都是有效结果；保留原顺序及原始位模式。
template <typename T>
class CopyAll {
    using Bits = typename std::conditional<sizeof(T) == VALUE16_BYTES, uint16_t, uint32_t>::type;

public:
    // 每个任务只处理一行内的连续片段，不跨行生成索引。
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR values, GM_ADDR indices, const SelectTopKBaseTilingData& data,
                               TPipe* pipe)
    {
        GlobalTensor<Bits> input, output;
        GlobalTensor<int64_t> outputIndices;
        input.SetGlobalBuffer((__gm__ Bits*)x);
        output.SetGlobalBuffer((__gm__ Bits*)values);
        outputIndices.SetGlobalBuffer((__gm__ int64_t*)indices);
        TBuf<TPosition::VECCALC> arena;
        pipe->InitBuffer(arena, data.ubBytes);
        auto raw = arena.Get<Bits>();
        auto index = arena.Get<int32_t>()[data.tile * sizeof(T) / INDEX32_BYTES];
        auto wide = arena.Get<int64_t>()[data.tile * (sizeof(T) + INDEX32_BYTES) / INDEX64_BYTES];
        for (uint64_t task = GetBlockIdx(); task < data.rows * data.tiles; task += data.cores) {
            const uint64_t row = task / data.tiles;
            const uint64_t start = task % data.tiles * data.tile;
            const uint32_t count = Min<uint64_t>(data.tile, data.axis - start);
            const uint64_t address = row * data.axis + start;
            DataCopyPad(raw, input[address], DataCopyExtParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0},
                        DataCopyPadExtParams<Bits>{false, 0, 0, 0});
            CreateLinearIndex(index, static_cast<int32_t>(start), count);
            Cast(wide, index, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            Fence<HardEvent::MTE2_V>();
            // 原值搬入和索引计算均已完成，两次写出共用一组事件；写完再复用整段UB。
            Fence<HardEvent::V_MTE3>();
            DataCopyPad(output[address], raw, DataCopyExtParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0});
            DataCopyPad(outputIndices[address], wide,
                        DataCopyExtParams{1, static_cast<uint32_t>(count * INDEX64_BYTES), 0, 0, 0});
            Fence<HardEvent::MTE3_S>();
        }
    }
};
} // namespace SelectTopK
#endif
