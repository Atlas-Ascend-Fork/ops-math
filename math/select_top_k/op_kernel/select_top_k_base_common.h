/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SELECT_TOP_K_BASE_COMMON_H
#define SELECT_TOP_K_BASE_COMMON_H
#include "kernel_operator.h"
#include "select_top_k_base_tiling_data.h"
#include <type_traits>

namespace SelectTopKBase {
using namespace AscendC;
// 返回两个同类型值中的较小值。
template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a < b ? a : b;
}
// 以除法和余数计算向上整除，避免相加溢出。
__aicore__ inline uint64_t DivUp(uint64_t n, uint64_t d) { return n / d + (n % d != 0); }
// 通过指定硬事件等待前序流水完成，保护缓冲读写顺序。
template <HardEvent event>
__aicore__ inline void Fence()
{
    SetFlag<event>(EVENT_ID0);
    WaitFlag<event>(EVENT_ID0);
}

// 按字节偏移取得 UB 子视图；元素数量只设置视图边界，不改变缓冲布局或初始化内容。
template <typename U>
__aicore__ inline LocalTensor<U> UbSubview(TBuf<TPosition::VECCALC>& arena, uint32_t byteOffset, uint32_t elements)
{
    auto view = arena.Get<U>()[byteOffset / sizeof(U)];
    view.SetSize(elements);
    return view;
}

// 将 UB 各行的前 count 项紧凑写入 GM；rowStride 以元素计，流水同步由调用模板管理。
template <typename U>
__aicore__ inline void WriteRowsToGm(const GlobalTensor<U>& destination, const LocalTensor<U>& source, uint32_t rows,
                                     uint64_t count, uint32_t rowStride)
{
    const uint32_t blockElements = UB_BLOCK_BYTES / sizeof(U);
    const uint32_t aligned = DivUp(count, blockElements) * blockElements;
    DataCopyExtParams copy{static_cast<uint16_t>(rows), static_cast<uint32_t>(count * sizeof(U)),
                           static_cast<uint32_t>((rowStride - aligned) * sizeof(U) / UB_BLOCK_BYTES), 0, 0};
    DataCopyPad(destination, source, copy);
}

// 二维 DMA 搬入有效行并补齐末尾一个块；整行清零、其余尾部和同步均由调用模板管理。
template <typename U>
__aicore__ inline void LoadPaddedRows(const LocalTensor<U>& destination, const GlobalTensor<U>& source, uint32_t rows,
                                      uint64_t count, uint32_t rowStride, U padding)
{
    const uint32_t blockElements = UB_BLOCK_BYTES / sizeof(U);
    const uint32_t aligned = DivUp(count, blockElements) * blockElements;
    DataCopyExtParams copy{static_cast<uint16_t>(rows), static_cast<uint32_t>(count * sizeof(U)), 0,
                           static_cast<uint32_t>((rowStride - aligned) * sizeof(U) / UB_BLOCK_BYTES), 0};
    DataCopyPadExtParams<U> pad{true, 0, static_cast<uint8_t>(aligned - count), padding};
    DataCopyPad(destination, source, copy, pad);
}

// 单段 UB -> GM 搬运：向量结果完成后启动 MTE3，等待写回后才能复用源缓冲。
template <typename U>
__aicore__ inline void StoreToGm(const GlobalTensor<U>& destination, const LocalTensor<U>& source, uint32_t count)
{
    Fence<HardEvent::V_MTE3>();
    DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(U)), 0, 0, 0};
    DataCopyPad(destination, source, params);
    Fence<HardEvent::MTE3_S>();
}

// Sort32 按浮点编码排序：有效 NaN 统一为正静默 NaN，无效填充使用严格更小的负 NaN 编码。
__aicore__ inline float SortPadding()
{
    union {
        uint32_t bits;
        float value;
    } pad{FP32_PADDING_BITS};
    return pad.value;
}
// 返回有效 NaN 的统一排序键编码。
__aicore__ inline float SortNaN()
{
    union {
        uint32_t bits;
        float value;
    } nan{FP32_QUIET_NAN_BITS};
    return nan.value;
}
// 通过 FP32 位模式判断是否为 NaN。
__aicore__ inline bool IsNaN(float value)
{
    union {
        float value;
        uint32_t bits;
    } data{value};
    return (data.bits & FP32_ABS_BITS) > FP32_POS_INF_BITS;
}

// 使用调用方提供的掩码区编码排序键；是否复用原值区和行尾屏蔽均由调用模板决定。
template <typename T>
__aicore__ inline void EncodeSortKeys(const LocalTensor<float>& key, const LocalTensor<T>& raw,
                                      const LocalTensor<uint8_t>& finite, uint32_t count, bool largest)
{
    if constexpr (std::is_same<T, float>::value) {
        Adds(key.template ReinterpretCast<int32_t>(), raw.template ReinterpretCast<int32_t>(), int32_t(0), count);
    } else {
        Cast(key, raw, RoundMode::CAST_NONE, count);
    }
    PipeBarrier<PIPE_V>();
    // A2 Compare 的计数模式会截断不足一个 repeat 的尾部，需向上对齐。
    // 各调用方的键缓冲后均有已分配存储，可安全读取尾块；Select 仅使用有效位。
    auto compareKey = key;
    compareKey.SetSize((count + (FLOAT_VECTOR_ELEMENTS - 1)) / FLOAT_VECTOR_ELEMENTS * FLOAT_VECTOR_ELEMENTS);
    Compare(finite, compareKey, compareKey, CMPMODE::EQ,
            (count + (FLOAT_VECTOR_ELEMENTS - 1)) / FLOAT_VECTOR_ELEMENTS * FLOAT_VECTOR_ELEMENTS);
    PipeBarrier<PIPE_V>();
    Select(key, finite, key, SortNaN(), SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
    PipeBarrier<PIPE_V>();
    if (!largest) {
        Adds(key.template ReinterpretCast<int32_t>(), key.template ReinterpretCast<int32_t>(), INT32_MIN, count);
        PipeBarrier<PIPE_V>();
    }
}

// 原值转换后不再使用时复用其缓冲存放掩码，保留既有调用方的 UB 生命周期。
template <typename T>
__aicore__ inline void EncodeSortKeys(const LocalTensor<float>& key, const LocalTensor<T>& raw, uint32_t count,
                                      bool largest)
{
    EncodeSortKeys(key, raw, raw.template ReinterpretCast<uint8_t>(), count, largest);
}

// EncodeSortKeys 后临时区仍保留有限值掩码，可直接复用；合并掩码后清除无效尾位。
template <bool ReuseFiniteMask = false>
__aicore__ inline void FilterSortKeys(const LocalTensor<uint8_t>& mask, const LocalTensor<uint8_t>& scratch,
                                      const LocalTensor<float>& key, float threshold, uint32_t valid, uint32_t padded,
                                      bool largest)
{
    if (IsNaN(threshold)) {
        if (largest) {
            Duplicate(mask.ReinterpretCast<uint16_t>(), uint16_t(0),
                      (padded + (MASK_WORD16_BITS - 1)) / MASK_WORD16_BITS);
        } else {
            Compare(mask, key, key, CMPMODE::EQ, padded);
        }
    } else {
        CompareScalar(mask, key, threshold, CMPMODE::GT, padded);
        if (largest) {
            // A2 的无序 NE 对 NaN 返回假，因此改为对 EQ 结果取反。
            if constexpr (!ReuseFiniteMask) {
                Compare(scratch, key, key, CMPMODE::EQ, padded);
                PipeBarrier<PIPE_V>();
            }
            Not(scratch.ReinterpretCast<uint16_t>(), scratch.ReinterpretCast<uint16_t>(),
                (padded + (MASK_WORD16_BITS - 1)) / MASK_WORD16_BITS);
            PipeBarrier<PIPE_V>();
            Or(mask.ReinterpretCast<uint16_t>(), mask.ReinterpretCast<uint16_t>(), scratch.ReinterpretCast<uint16_t>(),
               (padded + (MASK_WORD16_BITS - 1)) / MASK_WORD16_BITS);
        }
    }
    PipeBarrier<PIPE_V>();
    if (valid < padded) {
        Fence<HardEvent::V_S>();
        auto words = mask.ReinterpretCast<uint32_t>();
        uint32_t word = valid / MASK_WORD32_BITS;
        if (valid % MASK_WORD32_BITS != 0) {
            words.SetValue(word, words.GetValue(word) & ((1U << (valid % MASK_WORD32_BITS)) - 1));
            ++word;
        }
        for (; word < (padded + (MASK_WORD32_BITS - 1)) / MASK_WORD32_BITS; ++word) {
            words.SetValue(word, 0);
        }
        Fence<HardEvent::S_V>();
    }
}

// 按原轴索引恢复 NaN 原始位模式，保留输入的符号与载荷。
template <typename T>
__aicore__ inline void RestoreNaNPayload(const LocalTensor<T>& output, const LocalTensor<float>& key,
                                         const LocalTensor<uint32_t>& indices, const __gm__ T* input, uint64_t rowStart,
                                         uint32_t count, bool largest)
{
    using Bits = typename std::conditional<sizeof(T) == VALUE16_BYTES, uint16_t, uint32_t>::type;
    GlobalTensor<Bits> original;
    original.SetGlobalBuffer(const_cast<__gm__ Bits*>(reinterpret_cast<const __gm__ Bits*>(input)));
    auto bits = output.template ReinterpretCast<Bits>();
    Fence<HardEvent::V_S>();
    // NaN 位于连续前缀或后缀；普通有限值输出只做一次标量检查，无需逐元素读取 GM。
    for (uint32_t j = 0; j < count; ++j) {
        const uint32_t i = largest ? j : count - j - 1;
        if (!IsNaN(key.GetValue(i))) {
            break;
        }
        bits.SetValue(i, original.GetValue(rowStart + indices.GetValue(i)));
    }
    Fence<HardEvent::S_V>();
}

// 将固定长度的有序段四路合并为下一层；调用方提供互不重叠的源、目的缓冲。
__aicore__ inline void MergeTileStage(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t padded,
                                      uint32_t span)
{
    // 四路等长段连续存放时，硬件 repeat 会同时推进源、目的地址 4 * span 个 pair。
    // 调用方的块容量最多为8192项，本层 span 最大2048、repeat 最大64，均在硬件接口范围内。
    const uint32_t groupPairs = MERGE_WAYS * span;
    const uint32_t fullGroups = padded / groupPairs;
    if (fullGroups != 0) {
        const uint16_t length = static_cast<uint16_t>(span);
        uint16_t lengths[MERGE_WAYS] = {length, length, length, length};
        MrgSortSrcList<float> sources(src, src[span * PAIR_COMPONENTS], src[span * MERGE_WAYS],
                                      src[span * MERGE_FOURTH_SOURCE_COMPONENT_OFFSET]);
        MrgSort4Info info(lengths, false, MERGE_ALL_SOURCES_MASK, static_cast<uint8_t>(fullGroups));
        MrgSort(dst, sources, info);
    }

    const uint32_t tailBase = fullGroups * groupPairs;
    if (tailBase < padded) {
        const uint32_t tailPairs = padded - tailBase;
        if (tailPairs <= span) {
            // 单路尾段已经有序；长度为 32 个 pair 的倍数，UB 拷贝不需要补齐或 Gather。
            DataCopy(dst[tailBase * PAIR_COMPONENTS].ReinterpretCast<uint32_t>(),
                     src[tailBase * PAIR_COMPONENTS].ReinterpretCast<uint32_t>(), tailPairs * PAIR_COMPONENTS);
        } else {
            // 不满四路或末路不足 span 时只执行一次，禁止用 repeat 假定尾段等长。
            LocalTensor<float> inputs[MERGE_WAYS] = {src[tailBase * PAIR_COMPONENTS], src[tailBase * PAIR_COMPONENTS],
                                                     src[tailBase * PAIR_COMPONENTS], src[tailBase * PAIR_COMPONENTS]};
            uint16_t lengths[MERGE_WAYS] = {0, 0, 0, 0};
            uint32_t routes = 0;
            for (uint32_t i = 0; i < MERGE_WAYS && tailBase + i * span < padded; ++i) {
                const uint32_t start = tailBase + i * span;
                inputs[i] = src[start * PAIR_COMPONENTS];
                lengths[i] = static_cast<uint16_t>(Min<uint32_t>(span, padded - start));
                ++routes;
            }
            MrgSortSrcList<float> sources(inputs[0], inputs[1], inputs[MERGE_THIRD_SOURCE],
                                          inputs[MERGE_FOURTH_SOURCE]);
            MrgSort4Info info(lengths, false, static_cast<uint16_t>((1U << routes) - 1), 1);
            MrgSort(dst[tailBase * PAIR_COMPONENTS], sources, info);
        }
    }
    // 本层各组互不重叠；整层完成后才交换 ping/pong，供下一层读取。
    PipeBarrier<PIPE_V>();
}

// 归并当前有效窗口并返回各路消耗量；窗口和输出缓冲由调用方管理。
__aicore__ inline LocalTensor<float> MergeWindow(const LocalTensor<float> (&windows)[MERGE_WAYS],
                                                 const LocalTensor<float>& merged, const uint32_t active[MERGE_WAYS],
                                                 uint32_t q, const uint32_t h[MERGE_WAYS],
                                                 uint32_t consumed[MERGE_WAYS], uint32_t& m)
{
    for (uint32_t i = 0; i < MERGE_WAYS; ++i) {
        consumed[i] = 0;
    }
    // 只剩一路时直接使用该窗口；不发出不必要的归并指令。
    if (q == 1) {
        m = h[active[0]];
        consumed[active[0]] = m;
        return windows[active[0]];
    }
    uint16_t lengths[MERGE_WAYS] = {0, 0, 0, 0};
    LocalTensor<float> inputs[MERGE_WAYS] = {windows[active[0]], windows[active[0]], windows[active[0]],
                                             windows[active[0]]};
    for (uint32_t s = 0; s < q; ++s) {
        inputs[s] = windows[active[s]];
        lengths[s] = h[active[s]];
    }
    MrgSortSrcList<float> list(inputs[0], inputs[1], inputs[MERGE_THIRD_SOURCE], inputs[MERGE_FOURTH_SOURCE]);
    MrgSort4Info info(lengths, true, static_cast<uint16_t>((1U << q) - 1), 1);
    MrgSort(merged, list, info);
    Fence<HardEvent::V_S>();
    uint16_t raw[MERGE_WAYS] = {0, 0, 0, 0};
    GetMrgSortResult(raw[0], raw[1], raw[MERGE_THIRD_SOURCE], raw[MERGE_FOURTH_SOURCE]);
    m = 0;
    for (uint32_t s = 0; s < q; ++s) {
        consumed[active[s]] = raw[s];
        m += raw[s];
    }
    return merged;
}

} // namespace SelectTopKBase
#endif
