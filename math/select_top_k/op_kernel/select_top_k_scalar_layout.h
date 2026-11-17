/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SELECT_TOP_K_SCALAR_LAYOUT_H_
#define SELECT_TOP_K_SCALAR_LAYOUT_H_
#include "select_top_k_tiling_data.h"

namespace SelectTopK {
constexpr uint32_t FP16_EXPONENT_MASK = 0x7c00;     // 半精度指数全1、尾数全0，即正无穷的位模式。
constexpr uint32_t BF16_EXPONENT_MASK = 0x7f80;     // BF16正无穷的位模式。
constexpr uint32_t FP32_EXPONENT_MASK = 0x7f800000; // FP32正无穷的位模式。
constexpr uint32_t FP16_SIGN_MASK = 0x8000;         // 两种16位类型的符号位。
constexpr uint32_t FP32_SIGN_MASK = 0x80000000;     // FP32符号位。
constexpr uint32_t SCALAR_RANK_SHIFT = 32;          // 高32位保存整数秩，低32位保存原轴索引。
constexpr uint32_t SCALAR_RANK_HIGH_BIT = SCALAR_RANK_SHIFT - 1; // 无符号差值的最高位用于产生全位掩码。
constexpr uint32_t SCALAR_BYTE_BITS = 8;                         // 每个原始字节的位数。
constexpr uint32_t SCALAR_WORD_BITS = sizeof(uint64_t) * SCALAR_BYTE_BITS; // 成组标量搬运的位宽。

// 以无符号位运算映射递增秩，避免逐元素分支；正负零同秩，所有NaN排在正无穷之后。
template <uint32_t SignMask, uint32_t ExponentMask, uint32_t WordMask>
#if defined(__CCE_AICORE__)
__aicore__
#endif
    inline uint32_t
    ScalarRank(uint32_t bits)
{
    const uint32_t magnitude = bits & (SignMask - 1);
    const uint32_t signMask = 0U - (bits / SignMask);
    const uint32_t rank = (bits ^ ((signMask & WordMask) | SignMask)) & WordMask;
    // magnitude最多为0x7fffffff：减1的最高位仅在零时置位，指数阈值减它仅在NaN时置位。
    const uint32_t zeroMask = 0U - ((magnitude - 1U) >> SCALAR_RANK_HIGH_BIT);
    const uint32_t nanMask = 0U - ((ExponentMask - magnitude) >> SCALAR_RANK_HIGH_BIT);
    return (rank & ~zeroMask) | (SignMask & zeroMask) | nanMask;
}

// 将秩和原轴索引合为一个64位记录；始终递增比较，方向只影响高位的秩。
#if defined(__CCE_AICORE__)
__aicore__
#endif
    inline uint64_t
    PackScalarRecord(uint32_t rank, uint32_t index, bool largest)
{
    const uint32_t directionMask = 0U - static_cast<uint32_t>(largest);
    const uint32_t orderedRank = rank ^ directionMask;
    return (static_cast<uint64_t>(orderedRank) << SCALAR_RANK_SHIFT) | index;
}

// 向有序前缀插入一个记录；每次移位只读写一次64位UB数据，不分开搬运秩和索引。
template <typename Records>
#if defined(__CCE_AICORE__)
__aicore__
#endif
    inline void
    InsertScalarRecord(Records& records, uint32_t count, uint64_t value)
{
    while (count > 0) {
        const uint64_t previous = records.GetValue(count - 1);
        if (previous <= value)
            break;
        records.SetValue(count, previous);
        --count;
    }
    records.SetValue(count, value);
}

// 从打包原值中删除一个元素后计算指定输出字；数组末尾的零哨兵提供跨字拼接高位。
template <uint32_t ValueBits, uint32_t WordCount>
#if defined(__CCE_AICORE__)
__aicore__
#endif
    inline uint64_t
    ScalarCompactedWord(const uint64_t (&words)[WordCount], uint32_t word, uint32_t skipped)
{
    static_assert(ValueBits > 0 && ValueBits < SCALAR_WORD_BITS && SCALAR_WORD_BITS % ValueBits == 0);
    constexpr uint32_t itemsPerWord = SCALAR_WORD_BITS / ValueBits;
    const uint32_t start = word * itemsPerWord;
    if (skipped >= start + itemsPerWord)
        return words[word];
    const uint64_t shifted = (words[word] >> ValueBits) | (words[word + 1] << (SCALAR_WORD_BITS - ValueBits));
    const uint32_t retainedBits = skipped > start ? (skipped - start) * ValueBits : 0;
    const uint64_t prefixMask = (uint64_t{1} << retainedBits) - 1;
    return (words[word] & prefixMask) | (shifted & ~prefixMask);
}

// 原值、打包排序记录和两份输出互不重叠，每个DMA起始地址均满足块对齐。
struct ScalarRowLayout {
    uint32_t records;
    uint32_t values;
    uint32_t wideIndices;
    uint32_t bytes;

    // 固定16项容量，Host和Kernel使用相同账本，原始输入一直保留到输出位恢复。
#if defined(__CCE_AICORE__)
    __aicore__
#endif
        constexpr explicit ScalarRowLayout(uint32_t valueBytes)
        : records(SCALAR_ROW_ITEMS * valueBytes),
          values(records + SCALAR_ROW_ITEMS * SelectTopKBase::INDEX64_BYTES),
          wideIndices(values + SCALAR_ROW_ITEMS * valueBytes),
          bytes(wideIndices + SCALAR_ROW_ITEMS * SelectTopKBase::INDEX64_BYTES)
    {}
};
} // namespace SelectTopK
#endif
