/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SELECT_TOP_K_PREPARE_INPUT_H_
#define SELECT_TOP_K_PREPARE_INPUT_H_
#include "select_top_k_finalize.h"
namespace SelectTopK {
// 将 [outer,N,inner] 重排为连续排序行，逐位保留原始值；此阶段计入 SelectTopK 的内核耗时。
template <typename T>
__aicore__ inline void PrepareInput(GM_ADDR source, GM_ADDR destination, const SelectTopKTilingData& data, TPipe* pipe)
{
    using Bits = typename std::conditional<sizeof(T) == VALUE16_BYTES, uint16_t, uint32_t>::type;
    const uint32_t inner = data.inputInner, block = data.inputBlock;
    const uint32_t capacity = block * inner;
    const uint64_t axis = data.base.axis, outer = data.base.rows / inner;
    TBuf<TPosition::VECCALC> arena;
    pipe->InitBuffer(arena, capacity * (DOUBLE_BUFFER_COUNT * sizeof(T) + INDEX32_BYTES));
    auto input = arena.Get<Bits>();
    auto output = input[capacity];
    auto offsets = arena.Get<uint32_t>()[DOUBLE_BUFFER_COUNT * capacity * sizeof(T) / INDEX32_BYTES];
    ArithProgression<int32_t>(offsets.template ReinterpretCast<int32_t>(), 0, static_cast<int32_t>(inner * sizeof(T)),
                              block);
    Fence<HardEvent::S_V>();
    SetMaskNorm();
    ResetMask();
    PipeBarrier<PIPE_V>();
    for (uint32_t generated = block; generated < capacity; generated *= INDEX_SEQUENCE_GROWTH) {
        const uint32_t copied = capacity - generated < generated ? capacity - generated : generated;
        Adds(offsets[generated].template ReinterpretCast<int32_t>(), offsets.template ReinterpretCast<int32_t>(),
             static_cast<int32_t>(generated / block * sizeof(T)), copied);
        PipeBarrier<PIPE_V>();
    }
    PipeBarrier<PIPE_V>();
    GlobalTensor<Bits> in, out;
    in.SetGlobalBuffer((__gm__ Bits*)source);
    out.SetGlobalBuffer((__gm__ Bits*)destination);
    const uint64_t blocks = (axis + block - 1) / block;
    for (uint64_t task = GetBlockIdx(); task < outer * blocks; task += data.base.cores) {
        const uint64_t batch = task / blocks, start = task % blocks * block;
        const uint32_t valid = axis - start < block ? axis - start : block;
        Fence<HardEvent::V_MTE2>();
        DataCopyPad(input, in[(batch * axis + start) * inner],
                    DataCopyExtParams{1, static_cast<uint32_t>(valid * inner * sizeof(T)), 0, 0, 0},
                    DataCopyPadExtParams<Bits>{false, 0, 0, 0});
        Fence<HardEvent::MTE2_V>();
        Gather(output, input, offsets, 0, capacity);
        Fence<HardEvent::V_MTE3>();
        DataCopyExtParams copy{static_cast<uint16_t>(inner), static_cast<uint32_t>(valid * sizeof(T)),
                               static_cast<uint32_t>(block * sizeof(T) / UB_BLOCK_BYTES -
                                                     (valid * sizeof(T) + (UB_BLOCK_BYTES - 1)) / UB_BLOCK_BYTES),
                               static_cast<uint32_t>((axis - valid) * sizeof(T)), 0};
        DataCopyPad(out[batch * inner * axis + start], output, copy);
        Fence<HardEvent::MTE3_V>();
    }
}
} // namespace SelectTopK
#endif
