/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SORT_WITH_INDEX_ARCH22_COMMON_H
#define SORT_WITH_INDEX_ARCH22_COMMON_H

#include "kernel_operator.h"
#include "sort_with_index_constants.h"

namespace SortWithIndexA2 {
using namespace AscendC;

template <typename U>
__aicore__ inline U AlignUp(U value, uint32_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

template <HardEvent event>
__aicore__ inline void EventSync()
{
    event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
    SetFlag<event>(id);
    WaitFlag<event>(id);
}

template <typename T>
__aicore__ inline void MakeSortKeys(LocalTensor<int16_t> bits, LocalTensor<int16_t> key, LocalTensor<int16_t> a,
                                    LocalTensor<int16_t> b, LocalTensor<float> fp, LocalTensor<uint8_t> cmp,
                                    bool descending, uint32_t count)
{
    // 通过整数位编码识别 IEEE 特殊值，避免直接对 NaN 做浮点比较。
    Duplicate(b, ABS_MASK, count);
    PipeBarrier<PIPE_V>();
    And(key, bits, b, count);
    PipeBarrier<PIPE_V>();
    Cast(fp, key, RoundMode::CAST_NONE, count);
    ShiftRight(a, bits, int16_t(SIGN_SHIFT), count);
    Duplicate(b, SIGN_MASK_SIGNED, count);
    PipeBarrier<PIPE_V>();
    Or(a, a, b, count);
    PipeBarrier<PIPE_V>();
    And(b, bits, a, count);
    Or(key, bits, a, count);
    PipeBarrier<PIPE_V>();
    Not(b, b, count);
    PipeBarrier<PIPE_V>();
    And(key, key, b, count);
    // 与原生 Sort 保持一致：正负零视为相等；FP16 NaN 保留符号和有效载荷顺序，BF16 NaN 先规范化。
    // VSEL 仅选择 half 位模式，不对原值做数值转换或运算。
    Duplicate(b, SIGN_MASK_SIGNED, count);
    Compares<float, uint8_t>(cmp, fp, 0.0f, CMPMODE::EQ, AlignUp(count, SIMD_ALIGN_ELEMENTS));
    PipeBarrier<PIPE_V>();
    Select(key.template ReinterpretCast<half>(), cmp, b.template ReinterpretCast<half>(),
           key.template ReinterpretCast<half>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    PipeBarrier<PIPE_V>();
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        Duplicate(b, int16_t(-1), count);
        Compares<float, uint8_t>(cmp, fp, float(BF16_INFINITY_BITS), CMPMODE::GT, AlignUp(count, SIMD_ALIGN_ELEMENTS));
        PipeBarrier<PIPE_V>();
        Select(key.template ReinterpretCast<half>(), cmp, b.template ReinterpretCast<half>(),
               key.template ReinterpretCast<half>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        PipeBarrier<PIPE_V>();
        // LOCAL 和基数排序会将该 UB 值缓冲直接用于输出。
        Duplicate(b, int16_t(ABS_MASK), count);
        PipeBarrier<PIPE_V>();
        Select(bits.template ReinterpretCast<half>(), cmp, b.template ReinterpretCast<half>(),
               bits.template ReinterpretCast<half>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        PipeBarrier<PIPE_V>();
    }
    if (descending) {
        Not(key, key, count);
        PipeBarrier<PIPE_V>();
    }
}

// 按位置重新读取 GM 原值的归并路径，在输出时将 BF16 NaN 规范化为 0x7fff，与原生 Sort 一致且不修改输入。
// magnitude 和 cmp 必须使用已空闲的临时区，不能覆盖输出值。
template <typename T>
__aicore__ inline void CanonicalizeSortValues(LocalTensor<uint16_t> values, LocalTensor<uint16_t> magnitude,
                                              LocalTensor<uint8_t> cmp, uint32_t count)
{
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        Duplicate(magnitude, uint16_t(ABS_MASK), count);
        PipeBarrier<PIPE_V>();
        And(magnitude, values, magnitude, count);
        PipeBarrier<PIPE_V>();
        Adds(magnitude.template ReinterpretCast<int16_t>(), magnitude.template ReinterpretCast<int16_t>(), int16_t(1),
             count);
        PipeBarrier<PIPE_V>();
        ShiftRight(magnitude, magnitude, uint16_t(1), count);
        PipeBarrier<PIPE_V>();
        // ceil(absBits/2) 按 half 解释为有限值；BF16 无穷大映射为 0x3fc0，所有 NaN 编码均严格大于它。
        Compares<half, uint8_t>(cmp, magnitude.template ReinterpretCast<half>(), half(1.9375f), CMPMODE::GT,
                                AlignUp(count, SIMD_ALIGN_ELEMENTS));
        PipeBarrier<PIPE_V>();
        Duplicate(magnitude, uint16_t(ABS_MASK), count);
        PipeBarrier<PIPE_V>();
        Select(values.template ReinterpretCast<half>(), cmp, magnitude.template ReinterpretCast<half>(),
               values.template ReinterpretCast<half>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void FastPositions(LocalTensor<float> positions, uint32_t count)
{
    // A2 CreateVecIndex 每次扩展 64 项；这里倍增生成精确 FP32 整数位置，将指令发射层数降为对数级。
    uint32_t seed = count < FP32_POSITION_SEED_ELEMENTS ? count : FP32_POSITION_SEED_ELEMENTS;
    for (uint32_t i = 0; i < seed; ++i)
        positions.SetValue(i, static_cast<float>(static_cast<int32_t>(i)));
    EventSync<HardEvent::S_V>();
    for (uint32_t filled = FP32_POSITION_SEED_ELEMENTS; filled < count; filled *= 2) {
        uint32_t size = count - filled < filled ? count - filled : filled;
        Adds(positions[filled], positions, static_cast<float>(static_cast<int32_t>(filled)), size);
        PipeBarrier<PIPE_V>();
    }
}

} // namespace SortWithIndexA2
#endif // SORT_WITH_INDEX_ARCH22_COMMON_H
