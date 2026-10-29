/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file top_k_constant_var_simd.h
 * \brief topk sort constant var
 */
#ifndef TOP_K_CONSTANT_VAR_SIMD_H
#define TOP_K_CONSTANT_VAR_SIMD_H
#include <cstdint>
#include <utility>
namespace topkV2 {
// Host/Kernel contracts; algorithm-specific performance thresholds stay with their routes.
constexpr uint32_t TOPK_SMALL_K_CAPACITY = 32U;
constexpr uint32_t TOPK_BITONIC_SORT_POLICY = 1U;
constexpr uint32_t TOPK_DIRECT_MODE = 11U;
constexpr uint32_t TOPK_DIRECT_SMALL_RANK_MAX_K = 8U;

const uint32_t THREAD_DIM_NUM = 1024;
const int32_t ONE_TIMES_B64_NUM = 32;
const int32_t ONE_TIMES_B32_NUM = 64;
const int32_t ONE_TIMES_B16_NUM = 128;
const int32_t ONE_TIMES_B8_NUM = 256;
const uint16_t SORT_DATA_TIMES = 4;
const uint32_t CONST_TWO = 2;
const uint16_t TWO_BIT_SORT_VALUE_0 = 0;
const uint16_t TWO_BIT_SORT_VALUE_1 = 1;
const uint16_t TWO_BIT_SORT_VALUE_2 = 2;
const uint16_t TWO_BIT_SORT_VALUE_3 = 3;
const uint8_t TWO_BIT_SORT_OFFSET_STEP = 2;
const uint16_t RADIX_SORT_BIN_NUM = 256;
const uint32_t SHIFT_BIT_NUM = 8; // radix sort 每趟处理的位数（8 bit/pass -> 256 bins）
const uint32_t BITS_PER_BYTE = 8; // 每字节的位数，用于 bit 掩码字节数换算
const uint32_t B_64_CONV_NUM = 4;
const uint32_t XOR_OP_VALUE = 0x80000000;
const uint64_t XOR_OP_VALUE_B64 = 0x8000000000000000;
const uint16_t XOR_OP_VALUE_B16 = (uint16_t(1) << 15);
const uint8_t XOR_OP_VALUE_B8 = (uint8_t(1) << 7);
const uint16_t LOWEST_KEY_VALUE_B16 = uint16_t(-1);
const uint32_t LOWEST_KEY_VALUE_B32 = uint32_t(-1);
const int16_t STATE_BIT_SHF_VALUE = 30;
const int32_t AGGREGATE_READY_FLAG = 1;
const int32_t PREFIX_READY_FLAG = 2;
const uint32_t UB_CAN_HOLD_SIZE = 4096;
const uint32_t ZERO_VALUE_FLAG_B32 = 0;
const uint16_t ZERO_VALUE_FLAG_B16 = 0;
const uint32_t B64_BITE_SIZE = 8;
const uint32_t B32_BITE_SIZE = 4;
const uint32_t B16_BITE_SIZE = 2;
const uint32_t B8_BITE_SIZE = 1;
const uint32_t LAST_DIM_SINGLE_CORE_MOD = 1;
const uint32_t CLEAR_UB_VALUE = 0;
const uint32_t UB_AGLIN_VALUE = 32;
const uint32_t GATHER_AGLIN_VALUE = 256; // GatherMask/Compares API 要求的对齐字节数
const uint32_t DOUBLE_BUFFER_FACTOR = 2;
const uint32_t AGGREGATE_READY_MASK = 0x40000000;
const uint32_t PREFIX_READY_MASK = 0x80000000;
const uint32_t VALUE_MASK = 0x3fffffff;
const uint32_t HIST_MASK_OUT_LEN = 8;
const uint32_t NOT_INIT_COUNT_INDEX = 0;
const uint32_t AGG_READY_COUNT_INDEX = 8;
const uint32_t PREFIX_READY_COUNT_INDEX = 16;
const uint32_t SORT_AC_SHARE_BUFFER_FACTOR = 20;
const uint32_t SORT_AC_SHARE_BUFFER_FACTOR_2 = 512;
const uint32_t NOT_INIT_MODE = 0;
const uint32_t AGG_READY_MODE = 1;
const uint32_t PREFIX_READY_MODE = 2;
const int32_t XOR_OP_VALUE_FP = 0x80000000;
const int16_t XOR_OP_VALUE_HALF = 0x8000;
const int32_t SMALL_SORT_MAX_DATA_SIZE = 128;
const uint32_t CONCAT_AGLIN_VALUE = 16;
const uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t BITONIC_SMALL_TOPK_SIZE = TOPK_SMALL_K_CAPACITY; // Bitonic Small TopK 双调排序网络上限
} // namespace topkV2

namespace topkV2 {
// Shared constexpr network and Host cost model. Both the GM SIMT implementation
// and its Host selector consume this definition; keep them in one source.
constexpr uint32_t TOPK_DIRECT_THREADS = 512, TOPK_PAIR_THREADS = 1024, TOPK_SCALAR_THREADS = 256;
constexpr uint32_t TOPK_DIRECT_WARP_SIZE = 32, TOPK_DIRECT_INDEX_BITS = 5;
constexpr uint32_t TOPK_SCALAR_ROW_ALIGNMENT = 32, TOPK_DIRECT_BITONIC_POLICY = TOPK_BITONIC_SORT_POLICY;
constexpr uint32_t TOPK_DIRECT_PAIR_AXIS = 2, TOPK_DIRECT_SCALAR_MAX_AXIS = 8;
constexpr uint32_t TOPK_DIRECT_KEY_SORT_MIN_AXIS = 4, TOPK_DIRECT_KEY_SORT_MAX_AXIS = 16;
constexpr uint32_t TOPK_SCALAR_MIN_N = TOPK_DIRECT_PAIR_AXIS + 1U, TOPK_SCALAR_MAX_N = TOPK_DIRECT_WARP_SIZE;
template <uint32_t Width, uint32_t N = 0U>
inline constexpr uint32_t TOPK_SCALAR_ROW_THREADS = N ? TOPK_SCALAR_THREADS :
                                                        (Width == TOPK_DIRECT_PAIR_AXIS ? TOPK_PAIR_THREADS :
                                                                                          TOPK_DIRECT_THREADS);
static_assert((1U << TOPK_DIRECT_INDEX_BITS) == TOPK_DIRECT_WARP_SIZE);
// The host cost and device exchanges are derived from this same pruned sorting circuit.
template <uint32_t K, uint32_t Limit = TOPK_DIRECT_WARP_SIZE>
struct TopKScalarSelectNetwork {
    struct Exchange {
        uint32_t left, right;
        bool greater;
    };
    Exchange steps[TOPK_DIRECT_WARP_SIZE * TOPK_DIRECT_INDEX_BITS * (TOPK_DIRECT_INDEX_BITS + 1U) / 4U] = {};
    uint32_t count = 0;
    uint32_t order[K] = {};
};
template <uint32_t K, uint32_t Limit>
constexpr TopKScalarSelectNetwork<K, Limit> MakeTopKScalarSelectNetwork()
{
    TopKScalarSelectNetwork<K, Limit> result;
    int32_t slots[TOPK_DIRECT_WARP_SIZE] = {};
    for (uint32_t i = 0; i < TOPK_DIRECT_WARP_SIZE; ++i) {
        slots[i] = i < K ? static_cast<int32_t>(i) : -1;
    }
    for (uint32_t size = 2; size <= Limit; size *= 2) {
        for (uint32_t stride = size / 2; stride > 0; stride /= 2) {
            for (uint32_t thread = 0; thread < TOPK_DIRECT_WARP_SIZE / 2U; ++thread) {
                uint32_t a = 2 * thread - (thread & (stride - 1)), b = a + stride;
                bool descending = size != TOPK_DIRECT_WARP_SIZE && (thread & (size / 2)) != 0;
                if (slots[a] >= 0 && slots[b] >= 0) {
                    result.steps[result.count++] = {static_cast<uint32_t>(slots[a]), static_cast<uint32_t>(slots[b]),
                                                    descending};
                } else if ((slots[b] < 0) == descending) {
                    int32_t tmp = slots[a];
                    slots[a] = slots[b];
                    slots[b] = tmp;
                }
            }
        }
    }
    for (uint32_t i = 0; i < K; ++i) {
        result.order[i] = static_cast<uint32_t>(slots[i]);
    }
    return result;
}
template <uint32_t K, uint32_t Limit = TOPK_DIRECT_WARP_SIZE>
inline constexpr auto TOPK_SCALAR_SELECT_NETWORK = MakeTopKScalarSelectNetwork<K, Limit>();

struct TopKScalarCounts {
    uint16_t values[TOPK_SCALAR_MAX_N];
};
template <std::size_t... K>
constexpr auto TopKScalarComparisonCounts(std::index_sequence<K...>)
{
    return TopKScalarCounts{{TOPK_SCALAR_SELECT_NETWORK<K + 1U>.count...}};
}
inline constexpr auto TOPK_SCALAR_COMPARISONS = TopKScalarComparisonCounts(
    std::make_index_sequence<TOPK_SCALAR_MAX_N>{});
// Relative work units: n*rank models rank insertion, n*k candidate placement,
// and network.count key/index exchanges, weighted by 2, 1 and 8 respectively.
// Weights and the budget are empirical routing parameters for these launch widths.
constexpr uint32_t TOPK_SCALAR_RANK_WORK = 2U, TOPK_SCALAR_COMPARE_WORK = 8U;
constexpr uint32_t TOPK_SCALAR_MAX_WAVE_RATIO = TOPK_SCALAR_THREADS / (TOPK_DIRECT_THREADS / TOPK_DIRECT_WARP_SIZE);
// Compare scalar work with a concurrent warp-row wave using MAX_WAVE_RATIO.
// Revalidate routing when the algorithm, launch configuration or platform changes.
constexpr uint32_t TOPK_SCALAR_WAVE_BUDGET = 112U;
#ifdef __CCE_AICORE__
__aicore__
#endif
    inline constexpr uint32_t
    TopKScalarWork(uint32_t n, uint32_t k)
{
    uint32_t rank = k < n - k + 1U ? k : n - k + 1U;
    return TOPK_SCALAR_RANK_WORK * n * rank + n * k + TOPK_SCALAR_COMPARE_WORK * TOPK_SCALAR_COMPARISONS.values[k - 1U];
}
#ifdef __CCE_AICORE__
__aicore__
#endif
    inline constexpr bool
    IsTopKScalarSelectShape(int64_t n, int64_t k)
{
    return n >= TOPK_SCALAR_MIN_N && n <= TOPK_SCALAR_MAX_N && k > 0 && k < n &&
           TopKScalarWork(n, k) <= TOPK_SCALAR_MAX_WAVE_RATIO * TOPK_SCALAR_WAVE_BUDGET;
}
#ifdef __CCE_AICORE__
__aicore__
#endif
    inline constexpr bool
    CanUseTopKScalarSelect(bool narrowType, uint64_t rows, int64_t n, int64_t k, bool sorted, int64_t policy,
                           bool contiguousLastAxis, uint32_t platformCores)
{
    const bool compatible = narrowType && sorted && contiguousLastAxis && policy == TOPK_DIRECT_BITONIC_POLICY;
    if (!compatible || !IsTopKScalarSelectShape(n, k)) {
        return false;
    }
    if (!platformCores || !rows || rows > UINT32_MAX / uint64_t(n)) {
        return false;
    }
    uint64_t warpRows = uint64_t(platformCores) * (TOPK_DIRECT_THREADS / TOPK_DIRECT_WARP_SIZE);
    uint64_t warpWaves = (rows - 1U) / warpRows + 1U;
    // ceil(ceil(rows / warpRows) / ratio) equals ceil(rows / scalarRows).
    uint64_t scalarWaves = (warpWaves - 1U) / TOPK_SCALAR_MAX_WAVE_RATIO + 1U;
    return TopKScalarWork(n, k) * scalarWaves <= TOPK_SCALAR_WAVE_BUDGET * warpWaves;
}
} // namespace topkV2

#ifdef __CCE_AICORE__
#define TOPK_INPUT_SELECT_CONSTEXPR __aicore__ inline constexpr
#else
#define TOPK_INPUT_SELECT_CONSTEXPR inline constexpr
#endif
namespace topkV2 {
// Shared Host/Kernel allocation and metadata contract, not a standalone algorithm.
constexpr uint32_t INPUT_SELECT_TILING_FLAG = 0x49534C31U;
constexpr uint32_t INPUT_SELECT_GROUP_ROWS = 8U;
// Each row reserves 256 histogram words plus 8 words: 2 state words and 6 reserved/padding words.
constexpr uint32_t INPUT_SELECT_HIST_STRIDE = RADIX_SORT_BIN_NUM + 8U;
constexpr uint32_t INPUT_SELECT_METADATA_KEY_WORD = RADIX_SORT_BIN_NUM;
constexpr uint32_t INPUT_SELECT_METADATA_RANK_WORD = INPUT_SELECT_METADATA_KEY_WORD + 1U;
constexpr uint32_t INPUT_SELECT_METADATA_HIGH_SHIFT = 16U;
constexpr uint32_t INPUT_SELECT_METADATA_LOW_MASK = 0xffffU;
constexpr uint32_t INPUT_SELECT_METADATA_HALFWORDS_PER_WORD = 2U;
// B8 threshold selection writes [strictCount:key] directly to KEY. Its consumer
// does not use RANK or perform small-rank decode/repack.
// B16 histogram selection also writes [strictCount:key] directly to KEY, with
// the residual rank in RANK's low half.
// Only B16 small-rank selection first writes [equalQuota:key] to KEY. DecodeAllEq
// reads that quota before RepackSmallRank replaces it with strictCount; RANK is
// not consumed by this path. Collect uses strictCount as the equal-segment start.
// B32 stores the complete 32-bit key in KEY and the residual rank in RANK's low
// half. Packed-halfword helpers apply only to B8/B16 state.
// B64 uses separate selection state and does not consume this metadata layout.
TOPK_INPUT_SELECT_CONSTEXPR uint32_t TopKInputSelectMetadataCount(uint32_t packed)
{
    return packed >> INPUT_SELECT_METADATA_HIGH_SHIFT;
}
TOPK_INPUT_SELECT_CONSTEXPR uint32_t TopKInputSelectMetadataKey(uint32_t packed)
{
    return packed & INPUT_SELECT_METADATA_LOW_MASK;
}
TOPK_INPUT_SELECT_CONSTEXPR uint32_t TopKInputSelectMetadataPack(uint32_t key, uint32_t count)
{
    return (key & INPUT_SELECT_METADATA_LOW_MASK) | (count << INPUT_SELECT_METADATA_HIGH_SHIFT);
}
constexpr uint32_t INPUT_SELECT_LOAD_GUARD_BYTES = 256U;
constexpr uint32_t INPUT_SELECT_CANDIDATES = TOPK_SMALL_K_CAPACITY;
constexpr uint32_t INPUT_SELECT_STRIDE_ALIGN_ELEMENTS = 32U;
constexpr uint32_t INPUT_SELECT_VECTOR_ITEMS = 64U;
constexpr uint32_t INPUT_SELECT_SINGLE_MIN_AXIS = 1025U;
constexpr uint32_t INPUT_SELECT_SINGLE_MAX_AXIS = 8192U;
constexpr uint32_t INPUT_SELECT_B8_SINGLE_MIN_AXIS = 513U;
constexpr uint32_t INPUT_SELECT_B16_SINGLE_MIN_AXIS = 513U;
constexpr uint32_t INPUT_SELECT_B16_SINGLE_MAX_AXIS = 16384U;
constexpr uint32_t INPUT_SELECT_MAX_AXIS = INPUT_SELECT_SINGLE_MAX_AXIS;

// Shared by Host UB budgeting and Kernel input allocation/addressing; the unit is elements.
TOPK_INPUT_SELECT_CONSTEXPR uint32_t TopKInputSelectStride(uint32_t axisLength)
{
    return (axisLength + INPUT_SELECT_STRIDE_ALIGN_ELEMENTS - 1U) / INPUT_SELECT_STRIDE_ALIGN_ELEMENTS *
           INPUT_SELECT_STRIDE_ALIGN_ELEMENTS;
}

// Default Host performance interval only. Medium integer/BF16 routes may prefer
// shorter rows. The Kernel's actual n/k limits are checked by UseTopKUbRadixSelect;
// neither that capability nor the allocation budget is relaxed by a medium route.
TOPK_INPUT_SELECT_CONSTEXPR bool IsTopKInputSelectDefaultAxisPreferred(uint64_t n, uint32_t valueBytes)
{
    uint32_t minAxis = valueBytes == 1U ?
                           INPUT_SELECT_B8_SINGLE_MIN_AXIS :
                           (valueBytes == 2U ? INPUT_SELECT_B16_SINGLE_MIN_AXIS : INPUT_SELECT_SINGLE_MIN_AXIS);
    uint32_t maxAxis = valueBytes == 2U ? INPUT_SELECT_B16_SINGLE_MAX_AXIS : INPUT_SELECT_SINGLE_MAX_AXIS;
    return n >= minAxis && n <= maxAxis;
}

TOPK_INPUT_SELECT_CONSTEXPR uint64_t TopKInputSelectKeyBytesShared(uint32_t stride, uint32_t rows, uint32_t valueBytes)
{
    return uint64_t(stride) * rows * valueBytes + INPUT_SELECT_LOAD_GUARD_BYTES;
}
TOPK_INPUT_SELECT_CONSTEXPR uint64_t TopKInputSelectScratchBytes(uint32_t stride, uint32_t rows, uint32_t valueBytes)
{
    return TopKInputSelectKeyBytesShared(stride, rows, valueBytes) +
           INPUT_SELECT_GROUP_ROWS * INPUT_SELECT_HIST_STRIDE * sizeof(int32_t);
}
TOPK_INPUT_SELECT_CONSTEXPR uint64_t TopKInputSelectUbBytes(uint32_t stride, uint32_t rows, uint32_t valueBytes,
                                                            uint32_t indexBytes)
{
    uint64_t input = uint64_t(stride) * rows * valueBytes + INPUT_SELECT_LOAD_GUARD_BYTES;
    uint64_t value = uint64_t(INPUT_SELECT_CANDIDATES) * rows * valueBytes;
    uint64_t index = uint64_t(INPUT_SELECT_CANDIDATES) * rows * indexBytes + INPUT_SELECT_LOAD_GUARD_BYTES;
    uint64_t indexTmp = indexBytes == sizeof(int64_t) ?
                            uint64_t(INPUT_SELECT_CANDIDATES) * rows * sizeof(int32_t) + INPUT_SELECT_LOAD_GUARD_BYTES :
                            0U;
    return input + value + index + indexTmp + TopKInputSelectScratchBytes(stride, rows, valueBytes);
}
} // namespace topkV2
#undef TOPK_INPUT_SELECT_CONSTEXPR

namespace topkV2 {
// Private mode0 marker: the host has budgeted two input buffers.
constexpr uint32_t INPUT_SELECT_PREFETCH_TILING_FLAG = 0x49534C32U;
constexpr uint32_t INPUT_SELECT_PREFETCH_MAX_ROWS = 6U;
#ifdef __CCE_AICORE__
__aicore__
#endif
    inline constexpr uint64_t
    TopKInputSelectPrefetchUbBytes(uint32_t stride, uint32_t rows, uint32_t valueBytes, uint32_t indexBytes)
{
    return TopKInputSelectUbBytes(stride, rows, valueBytes, indexBytes) + uint64_t(stride) * rows * valueBytes +
           INPUT_SELECT_LOAD_GUARD_BYTES;
}
} // namespace topkV2
#endif
