/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_bincount_simt.h
 * \brief SIMT kernel for sparse_bincount operator.
 *        Phase1: clear output + SyncAll + Phase2: scatter accumulate via atomic_add.
 *        Deterministic path: fixed-order gather accumulation, bitwise reproducible.
 *        Precise path: two-float or exact-integer accumulation for reproducible results.
 */

#ifndef SPARSE_BINCOUNT_SIMT_H
#define SPARSE_BINCOUNT_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/common_functions.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "sparse_bincount_tiling_data.h"
#include "sparse_bincount_tiling_key.h"
#include "sparse_bincount_struct.h"

namespace NsSparseBincount {

using namespace AscendC;

// Thread count for index width paths: u32 more threads for latency hiding, u64 fewer due to register pressure
static constexpr uint32_t THREAD_NUM_U32 = 1024;
static constexpr uint32_t THREAD_NUM_U64 = 512;

// IEEE 754 float32 bit layout
static constexpr uint32_t FLOAT_SIGN_BIT_POS = 31U;
static constexpr uint32_t FLOAT_EXP_SHIFT = 23U;
static constexpr uint32_t FLOAT_EXP_ALL_ONES = 0xFFU;
static constexpr uint32_t FLOAT_MANTISSA_MASK = 0x007FFFFFU;
static constexpr uint32_t FLOAT_IMPLICIT_ONE_BIT = 0x00800000U;
static constexpr uint32_t FLOAT_NAN_BITS = 0x7FC00000U;
static constexpr uint32_t FLOAT_POS_INF_BITS = 0x7F800000U;
static constexpr uint32_t FLOAT_NEG_INF_BITS = 0xFF800000U;
static constexpr uint32_t FLOAT_SIGN_MASK = 0x80000000U;
static constexpr uint32_t FLOAT_ZERO_BITS = 0U;

// Float32 significand: 23 explicit bits, max exact integer has highest bit at 22
static constexpr int32_t FLOAT_SIGNIFICAND_BITS = 23;
static constexpr int32_t FLOAT_EXACT_INT_MAX_BIT = 22;
static constexpr int32_t FLOAT_EXP_BIAS = 22;
static constexpr int32_t FLOAT_EXP_INF_VALUE = 255;
static constexpr uint64_t FLOAT_SIGNIFICAND_OVERFLOW = 0x01000000ULL;
static constexpr uint64_t FLOAT_SIGNIFICAND_MASK24 = 0x00FFFFFFULL;

// Exact accumulator limb layout: 6 × uint64 = 384 bits, index shift = log2(64)
static constexpr uint32_t LIMB_BITS = 64U;
static constexpr uint32_t LIMB_INDEX_SHIFT = 6U;
static constexpr uint32_t LIMB_BIT_OFFSET_MASK = 63U;
static constexpr int32_t ACC_LIMB_COUNT = 6;
static constexpr int32_t ACC_TOTAL_BITS = ACC_LIMB_COUNT * 64;
static constexpr uint64_t ACC_ALL_ONES = ~0ULL;

// Carry/borrow sentinel values for multi-limb arithmetic
static constexpr uint64_t CARRY_ONE = 1ULL;
static constexpr uint64_t CARRY_NONE = 0ULL;

// U32 index overflow boundary (2^32)
static constexpr uint64_t U32_OVERFLOW_BOUND = 0x100000000ULL;

// Binary search bit-shift stages for SparseHighestBit64 (progressive halving)
static constexpr uint32_t BIT_SEARCH_HALF = 32U;
static constexpr uint32_t BIT_SEARCH_QUARTER = 16U;
static constexpr uint32_t BIT_SEARCH_BYTE = 8U;
static constexpr uint32_t BIT_SEARCH_NIBBLE = 4U;
static constexpr uint32_t BIT_SEARCH_PAIR = 2U;
static constexpr uint32_t BIT_SEARCH_SINGLE = 1U;
static constexpr int32_t BIT_SEARCH_HALF_WEIGHT = 32;
static constexpr int32_t BIT_SEARCH_QUARTER_WEIGHT = 16;
static constexpr int32_t BIT_SEARCH_BYTE_WEIGHT = 8;
static constexpr int32_t BIT_SEARCH_NIBBLE_WEIGHT = 4;
static constexpr int32_t BIT_SEARCH_PAIR_WEIGHT = 2;
static constexpr int32_t BIT_SEARCH_SINGLE_WEIGHT = 1;

// Limb base bit positions for SparseHighestAccBit
static constexpr int32_t LIMB5_BASE_BIT = 5 * LIMB_BITS;
static constexpr int32_t LIMB4_BASE_BIT = 4 * LIMB_BITS;
static constexpr int32_t LIMB3_BASE_BIT = 3 * LIMB_BITS;
static constexpr int32_t LIMB2_BASE_BIT = 2 * LIMB_BITS;
static constexpr int32_t LIMB1_BASE_BIT = 1 * LIMB_BITS;

// Core count and flag sentinels
static constexpr uint32_t MIN_EFFECTIVE_CORES = 1U;
static constexpr uint32_t CORE_ZERO = 0U;

// Adjacent limb offset in carry/borrow propagation
static constexpr uint32_t NEXT_LIMB_OFFSET = 1U;

// Accumulator sign bit shift (top bit of highest limb)
static constexpr uint32_t ACC_SIGN_BIT_SHIFT = LIMB_BITS - 1U;

// Pre-computed bit search thresholds (2^N)
static constexpr uint64_t BIT_THRESHOLD_2E32 = 1ULL << BIT_SEARCH_HALF;
static constexpr uint64_t BIT_THRESHOLD_2E16 = 1ULL << BIT_SEARCH_QUARTER;
static constexpr uint64_t BIT_THRESHOLD_2E8 = 1ULL << BIT_SEARCH_BYTE;
static constexpr uint64_t BIT_THRESHOLD_2E4 = 1ULL << BIT_SEARCH_NIBBLE;
static constexpr uint64_t BIT_THRESHOLD_2E2 = 1ULL << BIT_SEARCH_PAIR;
static constexpr uint64_t BIT_THRESHOLD_2E1 = 1ULL << BIT_SEARCH_SINGLE;

// Limb indices for cascade and structural access
static constexpr uint32_t LIMB_0 = 0U;
static constexpr uint32_t LIMB_1 = 1U;
static constexpr uint32_t LIMB_2 = 2U;
static constexpr uint32_t LIMB_3 = 3U;
static constexpr uint32_t LIMB_4 = 4U;
static constexpr uint32_t LIMB_5 = 5U;

// IEEE 754 zero-state sentinels
static constexpr uint32_t FLOAT_EXP_ZERO = 0U;
static constexpr uint32_t FLOAT_SIGN_POS = 0U;
static constexpr uint32_t FLOAT_MANTISSA_ZERO = 0U;

// Target not found sentinel
static constexpr int64_t TARGET_NOT_FOUND = -1;

// Range length adjustment for extra partitions
static constexpr int64_t PARTITION_EXTRA = 1;
static constexpr int64_t PARTITION_BASE = 0;

// Right-shift with carry bit offsets
static constexpr uint32_t SIGN_BIT_POSITIVE = 0U;
static constexpr uint32_t BIT_SHIFT_NONE = 0U;
static constexpr uint32_t BIT_ADJ_PREV = 1U;
static constexpr uint64_t BIT_MASK_ONE = 1ULL;

// Deterministic exact weighted histogram: integer limb atomics + compact per-output flags
static constexpr uint32_t HIST_TOTAL_WORDS = 16U;
static constexpr uint32_t HIST_POS_BASE = 0U;
static constexpr uint32_t HIST_NEG_BASE = 6U;
static constexpr uint32_t HIST_FLAGS_WORD = 12U;
static constexpr uint64_t HIST_FLAG_NAN = 1ULL;
static constexpr uint64_t HIST_FLAG_POS_INF = 2ULL;
static constexpr uint64_t HIST_FLAG_NEG_INF = 4ULL;
static constexpr uint64_t HIST_FLAG_HIT = 8ULL;

static constexpr int64_t SINGLE_PASS_MAX_OUTPUT = 2LL * 1024 * 1024;

// 1D scatter write, atomic_add 1.0f or weight depending on template flags
template <bool BINARY_OUTPUT, bool HAS_WEIGHTS>
__simt_callee__ inline void ScatterWrite1D(__gm__ float* output, int64_t binIdx, uint64_t idx, __gm__ float* weights)
{
    if constexpr (BINARY_OUTPUT) {
        asc_atomic_add(&output[binIdx], 1.0f);
    } else if constexpr (HAS_WEIGHTS) {
        asc_atomic_add(&output[binIdx], weights[idx]);
    } else {
        asc_atomic_add(&output[binIdx], 1.0f);
    }
}

// ND scatter write with batch bounds check, atomic_add to batch*sizeVal+binIdx
template <bool BINARY_OUTPUT, bool HAS_WEIGHTS>
__simt_callee__ inline void ScatterWriteND(__gm__ float* output, __gm__ int64_t* indices, uint64_t idx,
                                           int64_t indicesColNum, int64_t binIdx, int64_t sizeVal,
                                           int64_t denseShapeRows, __gm__ float* weights)
{
    int64_t batch = indices[idx * indicesColNum];
    if (batch >= 0 && batch < denseShapeRows) {
        int64_t offset = batch * sizeVal + binIdx;
        if constexpr (BINARY_OUTPUT) {
            asc_atomic_add(&output[offset], 1.0f);
        } else if constexpr (HAS_WEIGHTS) {
            asc_atomic_add(&output[offset], weights[idx]);
        } else {
            asc_atomic_add(&output[offset], 1.0f);
        }
    }
}

// Grid-stride clear output to zero
template <typename V, typename IDX_T, uint32_t THREAD_NUM>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void OpSparseBincountClearOutput(int64_t outputSize,
                                                                                             __gm__ float* output)
{
    for (IDX_T idx = static_cast<IDX_T>(blockIdx.x * blockDim.x + threadIdx.x); idx < static_cast<IDX_T>(outputSize);
         idx += static_cast<IDX_T>(blockDim.x * gridDim.x)) {
        output[idx] = 0.0f;
    }
}

// Grid-stride scatter accumulate via atomic_add, 1D or ND dispatch
template <typename V, typename IDX_T, uint32_t THREAD_NUM, bool IS_1D, bool BINARY_OUTPUT, bool HAS_WEIGHTS>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void OpSparseBincountScatterAccumulate(
    int64_t numValues, int64_t sizeVal, int64_t denseShapeRows, int64_t indicesColNum, __gm__ int64_t* indices,
    __gm__ V* values, __gm__ float* weights, __gm__ float* output)
{
    for (IDX_T idx = static_cast<IDX_T>(blockIdx.x * blockDim.x + threadIdx.x); idx < static_cast<IDX_T>(numValues);
         idx += static_cast<IDX_T>(blockDim.x * gridDim.x)) {
        int64_t binIdx = static_cast<int64_t>(values[idx]);
        if (binIdx >= 0 && binIdx < sizeVal) {
            if constexpr (IS_1D) {
                ScatterWrite1D<BINARY_OUTPUT, HAS_WEIGHTS>(output, binIdx, idx, weights);
            } else {
                ScatterWriteND<BINARY_OUTPUT, HAS_WEIGHTS>(output, indices, idx, indicesColNum, binIdx, sizeVal,
                                                           denseShapeRows, weights);
            }
        }
    }
}

// Clamp output to binary 0/1 for binary_output mode
template <typename V, typename IDX_T, uint32_t THREAD_NUM>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void OpSparseBincountClampBinary(int64_t outputSize,
                                                                                             __gm__ float* output)
{
    for (IDX_T idx = static_cast<IDX_T>(blockIdx.x * blockDim.x + threadIdx.x); idx < static_cast<IDX_T>(outputSize);
         idx += static_cast<IDX_T>(blockDim.x * gridDim.x)) {
        if (output[idx] > 0.0f) {
            output[idx] = 1.0f;
        }
    }
}

template <typename IDX_T, uint32_t THREAD_NUM>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void OpSparseBincountClearCounts(int64_t outputSize,
                                                                                             __gm__ uint32_t* counts)
{
    for (IDX_T idx = static_cast<IDX_T>(blockIdx.x * blockDim.x + threadIdx.x); idx < static_cast<IDX_T>(outputSize);
         idx += static_cast<IDX_T>(blockDim.x * gridDim.x)) {
        counts[idx] = 0U;
    }
}

template <typename V, typename IDX_T, uint32_t THREAD_NUM, bool IS_1D>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void OpSparseBincountCountAccumulate(
    int64_t numValues, int64_t sizeVal, int64_t denseShapeRows, int64_t indicesColNum, __gm__ int64_t* indices,
    __gm__ V* values, __gm__ uint32_t* counts)
{
    for (IDX_T idx = static_cast<IDX_T>(blockIdx.x * blockDim.x + threadIdx.x); idx < static_cast<IDX_T>(numValues);
         idx += static_cast<IDX_T>(blockDim.x * gridDim.x)) {
        const int64_t binIdx = static_cast<int64_t>(values[idx]);
        if (binIdx < 0 || binIdx >= sizeVal) {
            continue;
        }
        int64_t target;
        if constexpr (IS_1D) {
            target = binIdx;
        } else {
            const int64_t batch = static_cast<int64_t>(indices[idx * indicesColNum]);
            if (batch < 0 || batch >= denseShapeRows) {
                continue;
            }
            target = batch * sizeVal + binIdx;
        }
        (void)asc_atomic_add(counts + target, static_cast<uint32_t>(1));
    }
}

template <typename IDX_T, uint32_t THREAD_NUM, bool BINARY_OUTPUT>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void OpSparseBincountFinalizeCount(int64_t outputSize,
                                                                                               __gm__ uint32_t* counts,
                                                                                               __gm__ float* output)
{
    for (IDX_T idx = static_cast<IDX_T>(blockIdx.x * blockDim.x + threadIdx.x); idx < static_cast<IDX_T>(outputSize);
         idx += static_cast<IDX_T>(blockDim.x * gridDim.x)) {
        const uint32_t count = counts[idx];
        if constexpr (BINARY_OUTPUT) {
            output[idx] = (count != 0) ? 1.0F : 0.0F;
        } else {
            output[idx] = static_cast<float>(count);
        }
    }
}

// Deterministic gather: each thread owns disjoint outputs and accumulates in fixed value order for bitwise
// reproducibility
template <typename V, typename IDX_T, uint32_t THREAD_NUM, bool IS_1D, bool BINARY_OUTPUT, bool HAS_WEIGHTS>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void OpSparseBincountDetermineAccumulate(
    int64_t numValues, int64_t sizeVal, int64_t denseShapeRows, int64_t indicesColNum, __gm__ int64_t* indices,
    __gm__ V* values, __gm__ float* weights, __gm__ float* output, int64_t outputSize)
{
    for (IDX_T outIdx = static_cast<IDX_T>(blockIdx.x * blockDim.x + threadIdx.x);
         outIdx < static_cast<IDX_T>(outputSize); outIdx += static_cast<IDX_T>(blockDim.x * gridDim.x)) {
        float acc = 0.0f;
        for (int64_t j = 0; j < numValues; ++j) {
            int64_t target = TARGET_NOT_FOUND;
            int64_t binIdx = static_cast<int64_t>(values[j]);
            if (binIdx >= 0 && binIdx < sizeVal) {
                if constexpr (IS_1D) {
                    target = binIdx;
                } else {
                    int64_t batch = indices[j * indicesColNum];
                    if (batch >= 0 && batch < denseShapeRows) {
                        target = batch * sizeVal + binIdx;
                    }
                }
            }
            if (target == static_cast<int64_t>(outIdx)) {
                if constexpr (BINARY_OUTPUT) {
                    acc += 1.0f;
                } else if constexpr (HAS_WEIGHTS) {
                    acc += weights[j];
                } else {
                    acc += 1.0f;
                }
            }
        }
        if constexpr (BINARY_OUTPUT) {
            output[outIdx] = (acc > 0.0f) ? 1.0f : 0.0f;
        } else {
            output[outIdx] = acc;
        }
    }
}

// Add with carry: returns 1 if wrapped, 0 otherwise
__simt_callee__ inline uint64_t SparseAddWithCarry(uint64_t& target, uint64_t value)
{
    const uint64_t previous = target;
    target += value;
    return target < previous ? CARRY_ONE : CARRY_NONE;
}

// Subtract with borrow: returns 1 if borrowed, 0 otherwise
__simt_callee__ inline uint64_t SparseSubWithBorrow(uint64_t& target, uint64_t value)
{
    const uint64_t previous = target;
    target -= value;
    return previous < value ? CARRY_ONE : CARRY_NONE;
}

// Add value to the specified limb with carry propagation through higher limbs
__simt_callee__ inline void SparseAddLimb(SparseBincountExactAccumulator& acc, uint32_t wordIndex, uint64_t value)
{
    uint64_t carry = CARRY_NONE;
    if (wordIndex == LIMB_0) {
        carry = SparseAddWithCarry(acc.limb0, value);
        carry = carry == CARRY_NONE ? CARRY_NONE : SparseAddWithCarry(acc.limb1, carry);
        carry = carry == CARRY_NONE ? CARRY_NONE : SparseAddWithCarry(acc.limb2, carry);
        carry = carry == CARRY_NONE ? CARRY_NONE : SparseAddWithCarry(acc.limb3, carry);
        if (carry != CARRY_NONE) {
            carry = SparseAddWithCarry(acc.limb4, carry);
            acc.limb5 += carry;
        }
    } else if (wordIndex == LIMB_1) {
        carry = SparseAddWithCarry(acc.limb1, value);
        carry = carry == CARRY_NONE ? CARRY_NONE : SparseAddWithCarry(acc.limb2, carry);
        carry = carry == CARRY_NONE ? CARRY_NONE : SparseAddWithCarry(acc.limb3, carry);
        if (carry != CARRY_NONE) {
            carry = SparseAddWithCarry(acc.limb4, carry);
            acc.limb5 += carry;
        }
    } else if (wordIndex == LIMB_2) {
        carry = SparseAddWithCarry(acc.limb2, value);
        carry = carry == CARRY_NONE ? CARRY_NONE : SparseAddWithCarry(acc.limb3, carry);
        if (carry != CARRY_NONE) {
            carry = SparseAddWithCarry(acc.limb4, carry);
            acc.limb5 += carry;
        }
    } else if (wordIndex == LIMB_3) {
        carry = SparseAddWithCarry(acc.limb3, value);
        if (carry != CARRY_NONE) {
            carry = SparseAddWithCarry(acc.limb4, carry);
            acc.limb5 += carry;
        }
    } else {
        carry = SparseAddWithCarry(acc.limb4, value);
        acc.limb5 += carry;
    }
}

// Subtract value from the specified limb with borrow propagation through higher limbs
__simt_callee__ inline void SparseSubLimb(SparseBincountExactAccumulator& acc, uint32_t wordIndex, uint64_t value)
{
    uint64_t borrow = CARRY_NONE;
    if (wordIndex == LIMB_0) {
        borrow = SparseSubWithBorrow(acc.limb0, value);
        borrow = borrow == CARRY_NONE ? CARRY_NONE : SparseSubWithBorrow(acc.limb1, borrow);
        borrow = borrow == CARRY_NONE ? CARRY_NONE : SparseSubWithBorrow(acc.limb2, borrow);
        borrow = borrow == CARRY_NONE ? CARRY_NONE : SparseSubWithBorrow(acc.limb3, borrow);
        if (borrow != CARRY_NONE) {
            borrow = SparseSubWithBorrow(acc.limb4, borrow);
            acc.limb5 -= borrow;
        }
    } else if (wordIndex == LIMB_1) {
        borrow = SparseSubWithBorrow(acc.limb1, value);
        borrow = borrow == CARRY_NONE ? CARRY_NONE : SparseSubWithBorrow(acc.limb2, borrow);
        borrow = borrow == CARRY_NONE ? CARRY_NONE : SparseSubWithBorrow(acc.limb3, borrow);
        if (borrow != CARRY_NONE) {
            borrow = SparseSubWithBorrow(acc.limb4, borrow);
            acc.limb5 -= borrow;
        }
    } else if (wordIndex == LIMB_2) {
        borrow = SparseSubWithBorrow(acc.limb2, value);
        borrow = borrow == CARRY_NONE ? CARRY_NONE : SparseSubWithBorrow(acc.limb3, borrow);
        if (borrow != CARRY_NONE) {
            borrow = SparseSubWithBorrow(acc.limb4, borrow);
            acc.limb5 -= borrow;
        }
    } else if (wordIndex == LIMB_3) {
        borrow = SparseSubWithBorrow(acc.limb3, value);
        if (borrow != CARRY_NONE) {
            borrow = SparseSubWithBorrow(acc.limb4, borrow);
            acc.limb5 -= borrow;
        }
    } else {
        borrow = SparseSubWithBorrow(acc.limb4, value);
        acc.limb5 -= borrow;
    }
}

// Decompose a float32 into a 384-bit exact fixed-point accumulator with NaN/Inf tracking
__simt_callee__ inline void SparseAccumulateExactFloat(float value, SparseBincountExactAccumulator& acc)
{
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = value;
    const uint32_t sign = bits.u >> FLOAT_SIGN_BIT_POS;
    const uint32_t exponent = (bits.u >> FLOAT_EXP_SHIFT) & FLOAT_EXP_ALL_ONES;
    uint32_t mantissa = bits.u & FLOAT_MANTISSA_MASK;
    if (exponent == FLOAT_EXP_ALL_ONES) {
        if (mantissa != FLOAT_MANTISSA_ZERO) {
            acc.hasNan = true;
        } else if (sign == FLOAT_SIGN_POS) {
            acc.hasPositiveInf = true;
        } else {
            acc.hasNegativeInf = true;
        }
        return;
    }
    if (exponent != FLOAT_EXP_ZERO) {
        mantissa |= FLOAT_IMPLICIT_ONE_BIT;
    }
    const uint32_t shift = exponent == FLOAT_EXP_ZERO ? BIT_SHIFT_NONE : exponent - LIMB_1;
    const uint32_t wordIndex = shift >> LIMB_INDEX_SHIFT;
    const uint32_t bitOffset = shift & LIMB_BIT_OFFSET_MASK;
    const uint64_t lowPart = static_cast<uint64_t>(mantissa) << bitOffset;
    const uint64_t highPart = bitOffset == 0U ? 0ULL : static_cast<uint64_t>(mantissa) >> (LIMB_BITS - bitOffset);
    if (sign == FLOAT_SIGN_POS) {
        SparseAddLimb(acc, wordIndex, lowPart);
        if (highPart != CARRY_NONE) {
            SparseAddLimb(acc, wordIndex + NEXT_LIMB_OFFSET, highPart);
        }
    } else {
        SparseSubLimb(acc, wordIndex, lowPart);
        if (highPart != CARRY_NONE) {
            SparseSubLimb(acc, wordIndex + NEXT_LIMB_OFFSET, highPart);
        }
    }
}

// Read the limb at the specified word index (0-5), returns 0 for out of range
__simt_callee__ inline uint64_t SparseGetLimb(const SparseBincountExactAccumulator& acc, uint32_t wordIndex)
{
    if (wordIndex == LIMB_0) {
        return acc.limb0;
    }
    if (wordIndex == LIMB_1) {
        return acc.limb1;
    }
    if (wordIndex == LIMB_2) {
        return acc.limb2;
    }
    if (wordIndex == LIMB_3) {
        return acc.limb3;
    }
    if (wordIndex == LIMB_4) {
        return acc.limb4;
    }
    return wordIndex == LIMB_5 ? acc.limb5 : CARRY_NONE;
}

// Two's complement magnitude conversion: invert all limbs then add 1
__simt_callee__ inline void SparseMakeMagnitude(SparseBincountExactAccumulator& acc)
{
    acc.limb0 = ~acc.limb0;
    acc.limb1 = ~acc.limb1;
    acc.limb2 = ~acc.limb2;
    acc.limb3 = ~acc.limb3;
    acc.limb4 = ~acc.limb4;
    acc.limb5 = ~acc.limb5;
    SparseAddLimb(acc, CORE_ZERO, CARRY_ONE);
}

// Find the highest set bit position (0-63) in a uint64 value
__simt_callee__ inline int32_t SparseHighestBit64(uint64_t value)
{
    int32_t bit = 0;
    if (value >= BIT_THRESHOLD_2E32) {
        value >>= BIT_SEARCH_HALF;
        bit += BIT_SEARCH_HALF_WEIGHT;
    }
    if (value >= BIT_THRESHOLD_2E16) {
        value >>= BIT_SEARCH_QUARTER;
        bit += BIT_SEARCH_QUARTER_WEIGHT;
    }
    if (value >= BIT_THRESHOLD_2E8) {
        value >>= BIT_SEARCH_BYTE;
        bit += BIT_SEARCH_BYTE_WEIGHT;
    }
    if (value >= BIT_THRESHOLD_2E4) {
        value >>= BIT_SEARCH_NIBBLE;
        bit += BIT_SEARCH_NIBBLE_WEIGHT;
    }
    if (value >= BIT_THRESHOLD_2E2) {
        value >>= BIT_SEARCH_PAIR;
        bit += BIT_SEARCH_PAIR_WEIGHT;
    }
    if (value >= BIT_THRESHOLD_2E1) {
        ++bit;
    }
    return bit;
}

// Find the highest set bit (0-383) across all accumulator limbs, -1 if zero
__simt_callee__ inline int32_t SparseHighestAccBit(const SparseBincountExactAccumulator& acc)
{
    if (acc.limb5 != CARRY_NONE) {
        return LIMB5_BASE_BIT + SparseHighestBit64(acc.limb5);
    }
    if (acc.limb4 != CARRY_NONE) {
        return LIMB4_BASE_BIT + SparseHighestBit64(acc.limb4);
    }
    if (acc.limb3 != CARRY_NONE) {
        return LIMB3_BASE_BIT + SparseHighestBit64(acc.limb3);
    }
    if (acc.limb2 != CARRY_NONE) {
        return LIMB2_BASE_BIT + SparseHighestBit64(acc.limb2);
    }
    if (acc.limb1 != CARRY_NONE) {
        return LIMB1_BASE_BIT + SparseHighestBit64(acc.limb1);
    }
    if (acc.limb0 != CARRY_NONE) {
        return SparseHighestBit64(acc.limb0);
    }
    return -1;
}

// Extract a uint64 window from the accumulator starting at the given bit shift
__simt_callee__ inline uint64_t SparseShiftRight(const SparseBincountExactAccumulator& acc, uint32_t shift)
{
    const uint32_t wordIndex = shift >> LIMB_INDEX_SHIFT;
    const uint32_t bitOffset = shift & LIMB_BIT_OFFSET_MASK;
    uint64_t value = SparseGetLimb(acc, wordIndex) >> bitOffset;
    if (bitOffset != BIT_SHIFT_NONE) {
        value |= SparseGetLimb(acc, wordIndex + NEXT_LIMB_OFFSET) << (LIMB_BITS - bitOffset);
    }
    return value;
}

// Check if any non-zero bits exist below the given bit position (for round-to-nearest-even sticky bit)
__simt_callee__ inline bool SparseAnyBitsBelow(const SparseBincountExactAccumulator& acc, uint32_t bitCount)
{
    const uint32_t completeWords = bitCount >> LIMB_INDEX_SHIFT;
    for (uint32_t index = CORE_ZERO; index < completeWords; ++index) {
        if (SparseGetLimb(acc, index) != CARRY_NONE) {
            return true;
        }
    }
    const uint32_t remainingBits = bitCount & LIMB_BIT_OFFSET_MASK;
    if (remainingBits == BIT_SHIFT_NONE) {
        return false;
    }
    const uint64_t mask = (BIT_MASK_ONE << remainingBits) - CARRY_ONE;
    return (SparseGetLimb(acc, completeWords) & mask) != CARRY_NONE;
}

// Convert the exact accumulator back to float32 with round-to-nearest-even and NaN/Inf handling
__simt_callee__ inline float SparseFinalizeExactFloat(SparseBincountExactAccumulator acc)
{
    union {
        float f;
        uint32_t u;
    } result;
    if (acc.hasNan || (acc.hasPositiveInf && acc.hasNegativeInf)) {
        result.u = FLOAT_NAN_BITS;
        return result.f;
    }
    if (acc.hasPositiveInf) {
        result.u = FLOAT_POS_INF_BITS;
        return result.f;
    }
    if (acc.hasNegativeInf) {
        result.u = FLOAT_NEG_INF_BITS;
        return result.f;
    }
    const bool negative = (acc.limb5 >> ACC_SIGN_BIT_SHIFT) != CARRY_NONE;
    if (negative) {
        SparseMakeMagnitude(acc);
    }
    int32_t highestBit = SparseHighestAccBit(acc);
    if (highestBit < 0) {
        result.u = FLOAT_ZERO_BITS;
        return result.f;
    }
    if (highestBit <= FLOAT_EXACT_INT_MAX_BIT) {
        result.u = (negative ? FLOAT_SIGN_MASK : 0U) | static_cast<uint32_t>(acc.limb0);
        return result.f;
    }
    const uint32_t rightShift = static_cast<uint32_t>(highestBit - FLOAT_SIGNIFICAND_BITS);
    uint64_t significand = SparseShiftRight(acc, rightShift) & FLOAT_SIGNIFICAND_MASK24;
    if (rightShift != BIT_SHIFT_NONE) {
        const bool roundBit = ((SparseGetLimb(acc, (rightShift - BIT_ADJ_PREV) >> LIMB_INDEX_SHIFT) >>
                                ((rightShift - BIT_ADJ_PREV) & LIMB_BIT_OFFSET_MASK)) &
                               BIT_MASK_ONE) == CARRY_ONE;
        const bool sticky = SparseAnyBitsBelow(acc, rightShift - BIT_ADJ_PREV);
        if (roundBit && (sticky || ((significand & BIT_MASK_ONE) != CARRY_NONE))) {
            ++significand;
            if (significand == FLOAT_SIGNIFICAND_OVERFLOW) {
                significand >>= BIT_SEARCH_SINGLE;
                ++highestBit;
            }
        }
    }
    const int32_t exponent = highestBit - FLOAT_EXP_BIAS;
    if (exponent >= FLOAT_EXP_INF_VALUE) {
        result.u = negative ? FLOAT_NEG_INF_BITS : FLOAT_POS_INF_BITS;
        return result.f;
    }
    result.u = (negative ? FLOAT_SIGN_MASK : 0U) | (static_cast<uint32_t>(exponent) << FLOAT_EXP_SHIFT) |
               (static_cast<uint32_t>(significand) & FLOAT_MANTISSA_MASK);
    return result.f;
}

// Merge one word from a deterministic partial into the accumulator, propagating carry
__simt_callee__ inline void SparseMergeDetPartialWord(uint64_t& destination, uint64_t value, uint64_t carry,
                                                      uint64_t& outCarry)
{
    const uint64_t firstCarry = SparseAddWithCarry(destination, value);
    const uint64_t secondCarry = SparseAddWithCarry(destination, carry);
    outCarry = firstCarry | secondCarry;
}

__simt_callee__ inline void SparseAddHistogramWeight(__gm__ SparseBincountIntegerHistogram* slot,
                                                     __gm__ uint64_t* flags, float weight)
{
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = weight;
    const uint32_t sign = bits.u >> FLOAT_SIGN_BIT_POS;
    const uint32_t exponent = (bits.u >> FLOAT_EXP_SHIFT) & FLOAT_EXP_ALL_ONES;
    const uint32_t mantissa = bits.u & FLOAT_MANTISSA_MASK;
    if (exponent == FLOAT_EXP_ALL_ONES) {
        const uint64_t special = (mantissa != FLOAT_MANTISSA_ZERO) ?
                                     HIST_FLAG_NAN :
                                     (sign == FLOAT_SIGN_POS ? HIST_FLAG_POS_INF : HIST_FLAG_NEG_INF);
        (void)asc_atomic_or(flags, special | HIST_FLAG_HIT);
        return;
    }
    if (exponent == FLOAT_EXP_ZERO && mantissa == FLOAT_MANTISSA_ZERO) {
        return;
    }
    SparseBincountExactAccumulator value{};
    const float magnitude = (sign == FLOAT_SIGN_POS) ? weight : -weight;
    SparseAccumulateExactFloat(magnitude, value);
    const uint32_t base = (sign == FLOAT_SIGN_POS) ? HIST_POS_BASE : HIST_NEG_BASE;
    uint64_t carry = CARRY_NONE;
    bool hit = false;
    for (uint32_t word = 0U; word <= LIMB_5; ++word) {
        const uint64_t limb = SparseGetLimb(value, word);
        const uint64_t increment = limb + carry;
        carry = (increment < limb) ? CARRY_ONE : CARRY_NONE;
        if (increment != CARRY_NONE) {
            const uint64_t previous = asc_atomic_add(&slot->words[base + word], increment);
            carry += (previous > ~increment) ? CARRY_ONE : CARRY_NONE;
            hit = hit || (previous == CARRY_NONE);
        }
    }
    if (hit) {
        (void)asc_atomic_or(flags, HIST_FLAG_HIT);
    }
}

template <typename IDX_T, uint32_t THREAD_NUM>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void OpSparseBincountClearHistogramAll(
    int64_t outputSize, __gm__ SparseBincountIntegerHistogram* hist, __gm__ uint64_t* flags)
{
    for (IDX_T idx = static_cast<IDX_T>(blockIdx.x * blockDim.x + threadIdx.x); idx < static_cast<IDX_T>(outputSize);
         idx += static_cast<IDX_T>(blockDim.x * gridDim.x)) {
        for (uint32_t word = 0U; word < HIST_TOTAL_WORDS; ++word) {
            hist[idx].words[word] = CARRY_NONE;
        }
        flags[idx] = CARRY_NONE;
    }
    asc_threadfence();
}

template <typename IDX_T, uint32_t THREAD_NUM>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void OpSparseBincountClearHistogramFlags(
    int64_t outputSize, __gm__ uint64_t* flags)
{
    for (IDX_T idx = static_cast<IDX_T>(blockIdx.x * blockDim.x + threadIdx.x); idx < static_cast<IDX_T>(outputSize);
         idx += static_cast<IDX_T>(blockDim.x * gridDim.x)) {
        flags[idx] = CARRY_NONE;
    }
    asc_threadfence();
}

template <typename V, typename IDX_T, uint32_t THREAD_NUM, bool IS_1D>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void OpSparseBincountMarkHistogramHits(
    int64_t numValues, int64_t sizeVal, int64_t denseShapeRows, int64_t indicesColNum, __gm__ int64_t* indices,
    __gm__ V* values, __gm__ uint64_t* flags)
{
    for (IDX_T idx = static_cast<IDX_T>(blockIdx.x * blockDim.x + threadIdx.x); idx < static_cast<IDX_T>(numValues);
         idx += static_cast<IDX_T>(blockDim.x * gridDim.x)) {
        const int64_t binIdx = static_cast<int64_t>(values[idx]);
        if (binIdx < 0 || binIdx >= sizeVal) {
            continue;
        }
        int64_t target;
        if constexpr (IS_1D) {
            target = binIdx;
        } else {
            const int64_t batch = static_cast<int64_t>(indices[idx * indicesColNum]);
            if (batch < 0 || batch >= denseShapeRows) {
                continue;
            }
            target = batch * sizeVal + binIdx;
        }
        (void)asc_atomic_or(flags + target, HIST_FLAG_HIT);
    }
    asc_threadfence();
}

template <typename IDX_T, uint32_t THREAD_NUM>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void OpSparseBincountInitHistogramLimbs(
    int64_t outputSize, __gm__ SparseBincountIntegerHistogram* hist, __gm__ uint64_t* flags)
{
    for (IDX_T idx = static_cast<IDX_T>(blockIdx.x * blockDim.x + threadIdx.x); idx < static_cast<IDX_T>(outputSize);
         idx += static_cast<IDX_T>(blockDim.x * gridDim.x)) {
        if ((flags[idx] & HIST_FLAG_HIT) == 0U) {
            continue;
        }
        for (uint32_t word = 0U; word < HIST_FLAGS_WORD; ++word) {
            hist[idx].words[word] = CARRY_NONE;
        }
    }
    asc_threadfence();
}

template <typename V, typename IDX_T, uint32_t THREAD_NUM, bool IS_1D>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void OpSparseBincountScatterHistogram(
    int64_t numValues, int64_t sizeVal, int64_t denseShapeRows, int64_t indicesColNum, __gm__ int64_t* indices,
    __gm__ V* values, __gm__ float* weights, __gm__ SparseBincountIntegerHistogram* hist, __gm__ uint64_t* flags)
{
    asc_threadfence();
    for (IDX_T idx = static_cast<IDX_T>(blockIdx.x * blockDim.x + threadIdx.x); idx < static_cast<IDX_T>(numValues);
         idx += static_cast<IDX_T>(blockDim.x * gridDim.x)) {
        const int64_t binIdx = static_cast<int64_t>(values[idx]);
        if (binIdx < 0 || binIdx >= sizeVal) {
            continue;
        }
        int64_t target;
        if constexpr (IS_1D) {
            target = binIdx;
        } else {
            const int64_t batch = static_cast<int64_t>(indices[idx * indicesColNum]);
            if (batch < 0 || batch >= denseShapeRows) {
                continue;
            }
            target = batch * sizeVal + binIdx;
        }
        SparseAddHistogramWeight(hist + target, flags + target, weights[idx]);
    }
}

template <typename IDX_T, uint32_t THREAD_NUM>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void OpSparseBincountFinalizeHistogram(
    int64_t outputSize, __gm__ SparseBincountIntegerHistogram* hist, __gm__ uint64_t* flags, __gm__ float* output)
{
    for (IDX_T idx = static_cast<IDX_T>(blockIdx.x * blockDim.x + threadIdx.x); idx < static_cast<IDX_T>(outputSize);
         idx += static_cast<IDX_T>(blockDim.x * gridDim.x)) {
        const uint64_t flag = flags[idx];
        if ((flag & HIST_FLAG_HIT) == CARRY_NONE) {
            output[idx] = 0.0F;
            continue;
        }
        SparseBincountExactAccumulator positive{hist[idx].words[HIST_POS_BASE + LIMB_0],
                                                hist[idx].words[HIST_POS_BASE + LIMB_1],
                                                hist[idx].words[HIST_POS_BASE + LIMB_2],
                                                hist[idx].words[HIST_POS_BASE + LIMB_3],
                                                hist[idx].words[HIST_POS_BASE + LIMB_4],
                                                hist[idx].words[HIST_POS_BASE + LIMB_5],
                                                false,
                                                false,
                                                false};
        SparseBincountExactAccumulator negative{hist[idx].words[HIST_NEG_BASE + LIMB_0],
                                                hist[idx].words[HIST_NEG_BASE + LIMB_1],
                                                hist[idx].words[HIST_NEG_BASE + LIMB_2],
                                                hist[idx].words[HIST_NEG_BASE + LIMB_3],
                                                hist[idx].words[HIST_NEG_BASE + LIMB_4],
                                                hist[idx].words[HIST_NEG_BASE + LIMB_5],
                                                false,
                                                false,
                                                false};
        SparseMakeMagnitude(negative);
        uint64_t carry = SparseAddWithCarry(positive.limb0, negative.limb0);
        SparseMergeDetPartialWord(positive.limb1, negative.limb1, carry, carry);
        SparseMergeDetPartialWord(positive.limb2, negative.limb2, carry, carry);
        SparseMergeDetPartialWord(positive.limb3, negative.limb3, carry, carry);
        SparseMergeDetPartialWord(positive.limb4, negative.limb4, carry, carry);
        SparseMergeDetPartialWord(positive.limb5, negative.limb5, carry, carry);
        positive.hasNan = (flag & HIST_FLAG_NAN) != CARRY_NONE;
        positive.hasPositiveInf = (flag & HIST_FLAG_POS_INF) != CARRY_NONE;
        positive.hasNegativeInf = (flag & HIST_FLAG_NEG_INF) != CARRY_NONE;
        output[idx] = SparseFinalizeExactFloat(positive);
    }
}

template <typename V, typename IDX_T, uint32_t TN, bool IS_1D>
__aicore__ inline void RunSparseIntegerHistogram(uint32_t coreIdx, uint32_t coreNum, int64_t numValues,
                                                 int64_t outputSize, int64_t sizeVal, int64_t denseShapeRows,
                                                 int64_t indicesColNum, __gm__ int64_t* indicesGm, __gm__ V* valuesGm,
                                                 __gm__ float* weightsGm, __gm__ float* outputGm)
{
    (void)coreIdx;
    (void)coreNum;
    auto* hist = reinterpret_cast<__gm__ SparseBincountIntegerHistogram*>(GetSysWorkSpacePtr());
    auto* flags = reinterpret_cast<__gm__ uint64_t*>(hist + outputSize);
    if (outputSize <= SINGLE_PASS_MAX_OUTPUT) {
        asc_vf_call<OpSparseBincountClearHistogramAll<IDX_T, TN>>(dim3(TN), outputSize, hist, flags);
        SyncAll();
        asc_vf_call<OpSparseBincountScatterHistogram<V, IDX_T, TN, IS_1D>>(
            dim3(TN), numValues, sizeVal, denseShapeRows, indicesColNum, indicesGm, valuesGm, weightsGm, hist, flags);
        SyncAll();
        asc_vf_call<OpSparseBincountFinalizeHistogram<IDX_T, TN>>(dim3(TN), outputSize, hist, flags, outputGm);
        return;
    }
    asc_vf_call<OpSparseBincountClearHistogramFlags<IDX_T, TN>>(dim3(TN), outputSize, flags);
    SyncAll();
    asc_vf_call<OpSparseBincountMarkHistogramHits<V, IDX_T, TN, IS_1D>>(dim3(TN), numValues, sizeVal, denseShapeRows,
                                                                        indicesColNum, indicesGm, valuesGm, flags);
    SyncAll();
    asc_vf_call<OpSparseBincountInitHistogramLimbs<IDX_T, TN>>(dim3(TN), outputSize, hist, flags);
    SyncAll();
    asc_vf_call<OpSparseBincountScatterHistogram<V, IDX_T, TN, IS_1D>>(
        dim3(TN), numValues, sizeVal, denseShapeRows, indicesColNum, indicesGm, valuesGm, weightsGm, hist, flags);
    SyncAll();
    asc_vf_call<OpSparseBincountFinalizeHistogram<IDX_T, TN>>(dim3(TN), outputSize, hist, flags, outputGm);
}

// Kernel entry: reads size/dense_shape from GM, dispatches precise/deterministic/scatter paths by template flags
template <typename V, typename S, bool IS_PRECISE, bool IS_1D, bool BINARY_OUTPUT, bool HAS_WEIGHTS,
          bool IS_DETERMINISTIC>
__aicore__ inline void Process(GM_ADDR indices, GM_ADDR values, GM_ADDR denseShape, GM_ADDR size, GM_ADDR weights,
                               GM_ADDR output, const SparseBincountTilingData* tilingData)
{
    __gm__ S* sizeGm = (__gm__ S*)size;
    int64_t sizeVal = static_cast<int64_t>(sizeGm[0]);

    __gm__ int64_t* denseShapeGm = (__gm__ int64_t*)denseShape;
    int64_t denseShapeRows = 0;
    if constexpr (!IS_1D) {
        denseShapeRows = denseShapeGm[0];
    }

    int64_t outputSize = IS_1D ? sizeVal : denseShapeRows * sizeVal;

    __gm__ int64_t* indicesGm = (__gm__ int64_t*)indices;
    __gm__ V* valuesGm = (__gm__ V*)values;
    __gm__ float* weightsGm = (__gm__ float*)weights;
    __gm__ float* outputGm = (__gm__ float*)output;
    __gm__ uint32_t* outputCountGm = reinterpret_cast<__gm__ uint32_t*>(outputGm);

    if constexpr (IS_DETERMINISTIC && (BINARY_OUTPUT || !HAS_WEIGHTS)) {
        const uint64_t numValuesU64 = static_cast<uint64_t>(tilingData->numValues);
        const uint64_t outputSizeU64 = static_cast<uint64_t>(outputSize);
        if (numValuesU64 < U32_OVERFLOW_BOUND) {
            const uint64_t u32Step = static_cast<uint64_t>(THREAD_NUM_U32) *
                                     static_cast<uint64_t>(AscendC::GetBlockNum());
            const uint64_t u32SafeLimit = U32_OVERFLOW_BOUND - u32Step;
            if (numValuesU64 <= u32SafeLimit && outputSizeU64 <= u32SafeLimit) {
                asc_vf_call<OpSparseBincountClearCounts<uint32_t, THREAD_NUM_U32>>(dim3(THREAD_NUM_U32), outputSize,
                                                                                   outputCountGm);
                SyncAll();
                asc_vf_call<OpSparseBincountCountAccumulate<V, uint32_t, THREAD_NUM_U32, IS_1D>>(
                    dim3(THREAD_NUM_U32), tilingData->numValues, sizeVal, denseShapeRows, tilingData->indicesColNum,
                    indicesGm, valuesGm, outputCountGm);
                SyncAll();
                asc_vf_call<OpSparseBincountFinalizeCount<uint32_t, THREAD_NUM_U32, BINARY_OUTPUT>>(
                    dim3(THREAD_NUM_U32), outputSize, outputCountGm, outputGm);
            } else {
                asc_vf_call<OpSparseBincountClearCounts<uint64_t, THREAD_NUM_U64>>(dim3(THREAD_NUM_U64), outputSize,
                                                                                   outputCountGm);
                SyncAll();
                asc_vf_call<OpSparseBincountCountAccumulate<V, uint64_t, THREAD_NUM_U64, IS_1D>>(
                    dim3(THREAD_NUM_U64), tilingData->numValues, sizeVal, denseShapeRows, tilingData->indicesColNum,
                    indicesGm, valuesGm, outputCountGm);
                SyncAll();
                asc_vf_call<OpSparseBincountFinalizeCount<uint64_t, THREAD_NUM_U64, BINARY_OUTPUT>>(
                    dim3(THREAD_NUM_U64), outputSize, outputCountGm, outputGm);
            }
        } else {
            asc_vf_call<
                OpSparseBincountDetermineAccumulate<V, uint64_t, THREAD_NUM_U64, IS_1D, BINARY_OUTPUT, HAS_WEIGHTS>>(
                dim3(THREAD_NUM_U64), tilingData->numValues, sizeVal, denseShapeRows, tilingData->indicesColNum,
                indicesGm, valuesGm, weightsGm, outputGm, outputSize);
        }
        return;
    }

    if constexpr (IS_PRECISE) {
        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t coreNum = AscendC::GetBlockNum();
        const uint64_t numValuesU64 = static_cast<uint64_t>(tilingData->numValues);
        const uint64_t outputSizeU64 = static_cast<uint64_t>(outputSize);
        const uint64_t u32Step = static_cast<uint64_t>(THREAD_NUM_U32) * static_cast<uint64_t>(coreNum);
        const uint64_t u32SafeLimit = U32_OVERFLOW_BOUND - u32Step;
        if constexpr (IS_DETERMINISTIC) {
            if (numValuesU64 <= u32SafeLimit && outputSizeU64 <= u32SafeLimit) {
                RunSparseIntegerHistogram<V, uint32_t, THREAD_NUM_U32, IS_1D>(
                    coreIdx, coreNum, tilingData->numValues, outputSize, sizeVal, denseShapeRows,
                    tilingData->indicesColNum, indicesGm, valuesGm, weightsGm, outputGm);
            } else {
                RunSparseIntegerHistogram<V, uint64_t, THREAD_NUM_U64, IS_1D>(
                    coreIdx, coreNum, tilingData->numValues, outputSize, sizeVal, denseShapeRows,
                    tilingData->indicesColNum, indicesGm, valuesGm, weightsGm, outputGm);
            }
        } else {
            if (numValuesU64 <= u32SafeLimit && outputSizeU64 <= u32SafeLimit) {
                asc_vf_call<OpSparseBincountDetermineAccumulate<V, uint32_t, THREAD_NUM_U32, IS_1D, BINARY_OUTPUT,
                                                                HAS_WEIGHTS>>(
                    dim3(THREAD_NUM_U32), tilingData->numValues, sizeVal, denseShapeRows, tilingData->indicesColNum,
                    indicesGm, valuesGm, weightsGm, outputGm, outputSize);
            } else {
                asc_vf_call<OpSparseBincountDetermineAccumulate<V, uint64_t, THREAD_NUM_U64, IS_1D, BINARY_OUTPUT,
                                                                HAS_WEIGHTS>>(
                    dim3(THREAD_NUM_U64), tilingData->numValues, sizeVal, denseShapeRows, tilingData->indicesColNum,
                    indicesGm, valuesGm, weightsGm, outputGm, outputSize);
            }
        }
        return;
    }

    constexpr uint64_t U32_WRAP_BOUND = U32_OVERFLOW_BOUND;
    const uint64_t numValuesU64 = static_cast<uint64_t>(tilingData->numValues);
    const uint64_t outputSizeU64 = static_cast<uint64_t>(outputSize);
    const uint64_t u32Step = static_cast<uint64_t>(THREAD_NUM_U32) * static_cast<uint64_t>(AscendC::GetBlockNum());
    const uint64_t u32SafeLimit = U32_WRAP_BOUND - u32Step;

    if (numValuesU64 <= u32SafeLimit && outputSizeU64 <= u32SafeLimit) {
        constexpr uint32_t TN = THREAD_NUM_U32;
        if constexpr (IS_DETERMINISTIC) {
            asc_vf_call<OpSparseBincountDetermineAccumulate<V, uint32_t, TN, IS_1D, BINARY_OUTPUT, HAS_WEIGHTS>>(
                dim3(TN), tilingData->numValues, sizeVal, denseShapeRows, tilingData->indicesColNum, indicesGm,
                valuesGm, weightsGm, outputGm, outputSize);
            return;
        }
        asc_vf_call<OpSparseBincountClearOutput<V, uint32_t, TN>>(dim3(TN), outputSize, outputGm);
        SyncAll();
        asc_vf_call<OpSparseBincountScatterAccumulate<V, uint32_t, TN, IS_1D, BINARY_OUTPUT, HAS_WEIGHTS>>(
            dim3(TN), tilingData->numValues, sizeVal, denseShapeRows, tilingData->indicesColNum, indicesGm, valuesGm,
            weightsGm, outputGm);
        if constexpr (BINARY_OUTPUT) {
            SyncAll();
            asc_vf_call<OpSparseBincountClampBinary<V, uint32_t, TN>>(dim3(TN), outputSize, outputGm);
        }
    } else {
        constexpr uint32_t TN = THREAD_NUM_U64;
        if constexpr (IS_DETERMINISTIC) {
            asc_vf_call<OpSparseBincountDetermineAccumulate<V, uint64_t, TN, IS_1D, BINARY_OUTPUT, HAS_WEIGHTS>>(
                dim3(TN), tilingData->numValues, sizeVal, denseShapeRows, tilingData->indicesColNum, indicesGm,
                valuesGm, weightsGm, outputGm, outputSize);
            return;
        }
        asc_vf_call<OpSparseBincountClearOutput<V, uint64_t, TN>>(dim3(TN), outputSize, outputGm);
        SyncAll();
        asc_vf_call<OpSparseBincountScatterAccumulate<V, uint64_t, TN, IS_1D, BINARY_OUTPUT, HAS_WEIGHTS>>(
            dim3(TN), tilingData->numValues, sizeVal, denseShapeRows, tilingData->indicesColNum, indicesGm, valuesGm,
            weightsGm, outputGm);
        if constexpr (BINARY_OUTPUT) {
            SyncAll();
            asc_vf_call<OpSparseBincountClampBinary<V, uint64_t, TN>>(dim3(TN), outputSize, outputGm);
        }
    }
}

} // namespace NsSparseBincount

#endif // SPARSE_BINCOUNT_SIMT_H
