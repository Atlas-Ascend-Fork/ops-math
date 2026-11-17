/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SELECT_TOP_K_REFORMAT_H_
#define SELECT_TOP_K_REFORMAT_H_
#include "select_top_k_finalize.h"
#include "select_top_k_output_reuse_layout.h"
namespace SelectTopK {
// 共用重排与同步流程；编译期选择常规或分阶段复用布局，仅用于全轴、零偏移的双输出模式。
template <typename T, typename I, bool Reuse = false>
__aicore__ inline void ReformatOutput(GM_ADDR sourceValues, GM_ADDR sourceIndices, GM_ADDR values, GM_ADDR indices,
                                      const SelectTopKTilingData& data, TPipe* pipe)
{
    const uint64_t inner = data.outputInner, outer = data.base.rows / inner, k = data.outputK;
    const uint32_t cores = data.base.cores, block = data.outputBlock;
    const uint32_t capacity = block * inner;
    // 二次幂行步长会使相邻 Gather 通道访问相同 UB bank；源行增加间隔并保持 DMA 块对齐。
    const OutputReuseLayout layout(inner, block, k, sizeof(T), sizeof(I), Reuse);
    const uint32_t pitch = layout.pitch;
    TBuf<TPosition::VECCALC> arena;
    pipe->InitBuffer(arena, layout.bytes);
    auto inputValues = arena.Get<T>()[layout.inputValuesOffset / sizeof(T)];
    auto outputValues = arena.Get<T>()[layout.outputValuesOffset / sizeof(T)];
    auto inputIndices = arena.Get<int32_t>()[layout.inputIndicesOffset / INDEX32_BYTES];
    auto outputIndices32 = arena.Get<int32_t>()[layout.outputIndices32Offset / INDEX32_BYTES];
    auto offsets = arena.Get<uint32_t>()[layout.offsetsOffset / INDEX32_BYTES];
    auto outputIndices = arena.Get<I>()[layout.outputIndicesOffset / sizeof(I)];
    // 偏移表周期同时整除行宽和一个块的 INT32 元素数；复制周期时推进整列，奇数行宽也保持目标块对齐。
    uint32_t period = inner;
    while (period % FLOAT_BLOCK_ELEMENTS != 0)
        period += inner;
    for (uint32_t i = 0; i < period; ++i) {
        offsets.SetValue(i, (i % inner * pitch + i / inner) * INDEX32_BYTES);
    }
    Fence<HardEvent::S_V>();
    SetMaskNorm();
    ResetMask();
    for (uint32_t generated = period; generated < capacity; generated *= INDEX_SEQUENCE_GROWTH) {
        const uint32_t copied = capacity - generated < generated ? capacity - generated : generated;
        Adds(offsets[generated].template ReinterpretCast<int32_t>(), offsets.template ReinterpretCast<int32_t>(),
             static_cast<int32_t>(generated / inner * INDEX32_BYTES), copied);
        PipeBarrier<PIPE_V>();
    }
    GlobalTensor<T> inValues, outValues;
    GlobalTensor<int32_t> inIndices;
    GlobalTensor<I> outIndices;
    inValues.SetGlobalBuffer((__gm__ T*)sourceValues);
    inIndices.SetGlobalBuffer((__gm__ int32_t*)sourceIndices);
    outValues.SetGlobalBuffer((__gm__ T*)values);
    outIndices.SetGlobalBuffer((__gm__ I*)indices);
    const uint64_t blocks = (k + block - 1) / block;
    for (uint64_t task = GetBlockIdx(); task < outer * blocks; task += cores) {
        const uint64_t batch = task / blocks;
        const uint64_t start = task % blocks * block;
        const uint32_t valid = k - start < block ? k - start : block;
        const uint32_t count = valid * inner;
        const uint64_t source = batch * inner * k + start;
        const uint64_t dest = (batch * k + start) * inner;
        Fence<HardEvent::V_MTE2>();
        DataCopyExtParams valueCopy{static_cast<uint16_t>(inner), static_cast<uint32_t>(valid * sizeof(T)),
                                    static_cast<uint32_t>((k - valid) * sizeof(T)),
                                    static_cast<uint32_t>(pitch * sizeof(T) / UB_BLOCK_BYTES -
                                                          (valid * sizeof(T) + (UB_BLOCK_BYTES - 1)) / UB_BLOCK_BYTES),
                                    0};
        DataCopyPad(inputValues, inValues[source], valueCopy, DataCopyPadExtParams<T>{false, 0, 0, 0});
        DataCopyExtParams indexCopy{
            static_cast<uint16_t>(inner), valid * INDEX32_BYTES, static_cast<uint32_t>((k - valid) * INDEX32_BYTES),
            pitch / FLOAT_BLOCK_ELEMENTS - (valid + (FLOAT_BLOCK_ELEMENTS - 1)) / FLOAT_BLOCK_ELEMENTS, 0};
        DataCopyPad(inputIndices, inIndices[source], indexCopy, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        Fence<HardEvent::MTE2_V>();
        Gather(outputIndices32, inputIndices, offsets, 0, count);
        PipeBarrier<PIPE_V>();
        if constexpr (sizeof(T) == VALUE16_BYTES) {
            ShiftRight(offsets, offsets, static_cast<uint32_t>(1), count);
            PipeBarrier<PIPE_V>();
            Gather(outputValues.template ReinterpretCast<uint16_t>(), inputValues.template ReinterpretCast<uint16_t>(),
                   offsets, 0, count);
            PipeBarrier<PIPE_V>();
            ShiftLeft(offsets, offsets, static_cast<uint32_t>(1), count);
        } else {
            Gather(outputValues.template ReinterpretCast<uint32_t>(), inputValues.template ReinterpretCast<uint32_t>(),
                   offsets, 0, count);
        }
        PipeBarrier<PIPE_V>();
        if constexpr (sizeof(I) == INDEX64_BYTES) {
            Cast(outputIndices, outputIndices32, RoundMode::CAST_NONE, count);
        } else {
            DataCopy(outputIndices.template ReinterpretCast<int32_t>(), outputIndices32,
                     (count + (FLOAT_BLOCK_ELEMENTS - 1)) / FLOAT_BLOCK_ELEMENTS * FLOAT_BLOCK_ELEMENTS);
        }
        Fence<HardEvent::V_MTE3>();
        DataCopyExtParams vout{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPad(outValues[dest], outputValues, vout);
        DataCopyExtParams iout{1, static_cast<uint32_t>(count * sizeof(I)), 0, 0, 0};
        DataCopyPad(outIndices[dest], outputIndices, iout);
        Fence<HardEvent::MTE3_V>();
    }
}

} // namespace SelectTopK
#endif
