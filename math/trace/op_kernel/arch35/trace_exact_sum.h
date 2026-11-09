/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRACE_EXACT_SUM_H_
#define TRACE_EXACT_SUM_H_

#include "kernel_operator.h"
#include "simt_api/common_functions.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_sync_functions.h"
#include "trace_tiling_data.h"
#include <climits>

namespace NsTraceExact {
using namespace AscendC;

constexpr uint32_t WORD_COUNT = 11;
constexpr uint32_t THREAD_COUNT = 512;
constexpr uint32_t WARP_WIDTH = 32;
constexpr uint32_t PARTIAL_WORDS = WORD_COUNT + 1;
constexpr uint32_t WARP_COUNT = THREAD_COUNT / WARP_WIDTH;

// Every finite FP32 value is an integer multiple of 2^-149. Even int64 K
// cannot exceed the signed capacity of this 352-bit two-complement state.
struct ExactState {
    uint32_t words[WORD_COUNT];
    uint32_t flags;
};

__simt_callee__ __aicore__ inline void Clear(ExactState& state)
{
    for (uint32_t word = 0; word < WORD_COUNT; ++word) {
        state.words[word] = 0;
    }
    state.flags = 0;
}

__simt_callee__ __aicore__ inline void AddInput(ExactState& state, uint32_t bits)
{
    const uint32_t exponent = (bits >> 23) & 255u;
    const uint32_t fraction = bits & 0x7fffffu;
    if (exponent == 255u) {
        state.flags |= fraction != 0 ? 1u : ((bits >> 31) != 0 ? 4u : 2u);
        return;
    }
    const uint32_t mantissa = exponent == 0 ? fraction : (fraction | 0x800000u);
    if (mantissa == 0) {
        return;
    }
    const uint32_t shift = exponent == 0 ? 0 : exponent - 1;
    const uint32_t firstWord = shift >> 5;
    const uint32_t offset = shift & 31u;
    const uint32_t low = mantissa << offset;
    const uint32_t high = offset == 0 ? 0 : mantissa >> (32u - offset);
    const bool negative = (bits >> 31) != 0;
    uint64_t carry = 0;
    for (uint32_t word = firstWord; word < WORD_COUNT; ++word) {
        const uint32_t part = word == firstWord ? low : (word == firstWord + 1 ? high : 0);
        const uint32_t current = state.words[word];
        if (negative) {
            const uint64_t subtrahend = static_cast<uint64_t>(part) + carry;
            state.words[word] = current - static_cast<uint32_t>(subtrahend);
            carry = static_cast<uint64_t>(current) < subtrahend;
        } else {
            const uint64_t sum = static_cast<uint64_t>(current) + part + carry;
            state.words[word] = static_cast<uint32_t>(sum);
            carry = sum >> 32;
        }
        if (word >= firstWord + 1 && carry == 0) {
            break;
        }
    }
}

__simt_callee__ __aicore__ inline void WarpReduce(ExactState& state, uint32_t lane)
{
    uint64_t carry = 0;
    for (uint32_t word = 0; word < WORD_COUNT; ++word) {
        const uint32_t low = asc_reduce_add(state.words[word] & 0xffffu);
        const uint32_t high = asc_reduce_add(state.words[word] >> 16);
        const uint64_t total = static_cast<uint64_t>(low) + (static_cast<uint64_t>(high) << 16) + carry;
        state.words[word] = static_cast<uint32_t>(total);
        carry = total >> 32;
    }
    const uint32_t packed = ((state.flags & 1u) != 0 ? 1u : 0u) | ((state.flags & 2u) != 0 ? (1u << 6) : 0u) |
                            ((state.flags & 4u) != 0 ? (1u << 12) : 0u);
    const uint32_t counts = asc_reduce_add(packed);
    state.flags = ((counts & 63u) != 0 ? 1u : 0u) | ((counts & (63u << 6)) != 0 ? 2u : 0u) |
                  ((counts & (63u << 12)) != 0 ? 4u : 0u);
}

__simt_callee__ __aicore__ inline uint32_t FinalBits(ExactState& state)
{
    if ((state.flags & 1u) != 0 || (state.flags & 6u) == 6u) {
        return 0x7fc00000u;
    }
    if ((state.flags & 2u) != 0) {
        return 0x7f800000u;
    }
    if ((state.flags & 4u) != 0) {
        return 0xff800000u;
    }
    const uint32_t sign = state.words[WORD_COUNT - 1] & 0x80000000u;
    if (sign != 0) {
        uint64_t carry = 1;
        for (uint32_t word = 0; word < WORD_COUNT; ++word) {
            const uint64_t magnitude = static_cast<uint64_t>(~state.words[word]) + carry;
            state.words[word] = static_cast<uint32_t>(magnitude);
            carry = magnitude >> 32;
        }
    }
    uint32_t nonzeroWords = WORD_COUNT;
    while (nonzeroWords != 0 && state.words[nonzeroWords - 1] == 0) {
        --nonzeroWords;
    }
    if (nonzeroWords == 0) {
        return 0;
    }
    uint32_t top = state.words[nonzeroWords - 1];
    uint32_t topBit = 0;
    while ((top >>= 1) != 0) {
        ++topBit;
    }
    uint32_t highestBit = (nonzeroWords - 1) * 32 + topBit;
    if (highestBit < 23) {
        return sign | state.words[0];
    }
    const uint32_t shift = highestBit - 23;
    const uint32_t firstWord = shift >> 5;
    const uint64_t payload = static_cast<uint64_t>(state.words[firstWord]) |
                             (static_cast<uint64_t>(state.words[firstWord + 1]) << 32);
    uint32_t mantissa = static_cast<uint32_t>((payload >> (shift & 31u)) & 0xffffffu);
    if (shift != 0) {
        const uint32_t guardWord = (shift - 1) >> 5;
        const uint32_t guardOffset = (shift - 1) & 31u;
        const uint32_t guard = (state.words[guardWord] >> guardOffset) & 1u;
        bool sticky = (state.words[guardWord] & ((1u << guardOffset) - 1u)) != 0;
        for (uint32_t word = 0; word < guardWord; ++word) {
            sticky |= state.words[word] != 0;
        }
        if (guard != 0 && (sticky || (mantissa & 1u) != 0)) {
            ++mantissa;
        }
        if (mantissa == 0x1000000u) {
            mantissa >>= 1;
            ++highestBit;
        }
    }
    const uint32_t exponent = highestBit - 22;
    if (exponent >= 255) {
        return sign | 0x7f800000u;
    }
    return sign | (exponent << 23) | (mantissa & 0x7fffffu);
}

template <uint32_t NumThreads = THREAD_COUNT>
__simt_callee__ __aicore__ inline void ReduceComponent(ExactState& state, uint32_t lane, uint32_t warp,
                                                       __ubuf__ uint32_t* partials, __gm__ uint32_t* output)
{
    WarpReduce(state, lane);
    if constexpr (NumThreads == WARP_WIDTH) {
        if (lane == 0) {
            *output = FinalBits(state);
        }
        return;
    }
    if (lane == 0) {
        for (uint32_t word = 0; word < WORD_COUNT; ++word) {
            partials[warp * PARTIAL_WORDS + word] = state.words[word];
        }
        partials[warp * PARTIAL_WORDS + WORD_COUNT] = state.flags;
    }
    asc_syncthreads();
    if (warp == 0) {
        for (uint32_t word = 0; word < WORD_COUNT; ++word) {
            state.words[word] = lane < NumThreads / WARP_WIDTH ? partials[lane * PARTIAL_WORDS + word] : 0;
        }
        state.flags = lane < NumThreads / WARP_WIDTH ? partials[lane * PARTIAL_WORDS + WORD_COUNT] : 0;
        WarpReduce(state, lane);
        if (lane == 0) {
            *output = FinalBits(state);
        }
    }
    // All warps finish reading partials before the other complex component
    // reuses the same 768-byte UB scratch.
    asc_syncthreads();
}

template <typename IdxT, bool IsComplex, uint32_t NumThreads = THREAD_COUNT>
__simt_callee__ __aicore__ inline void AccumulateExact(int64_t diagonal, int64_t stride, __gm__ uint32_t* input,
                                                       __gm__ uint32_t* output, __ubuf__ uint32_t* partials)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid % WARP_WIDTH;
    const uint32_t warp = tid / WARP_WIDTH;
    ExactState real;
    Clear(real);
    if constexpr (IsComplex) {
        ExactState imag;
        Clear(imag);
        for (IdxT index = static_cast<IdxT>(tid); index < static_cast<IdxT>(diagonal);
             index += static_cast<IdxT>(NumThreads)) {
            const int64_t location = static_cast<int64_t>(index) * stride * 2;
            const uint32_t realBits = input[location];
            const uint32_t imagBits = input[location + 1];
            AddInput(real, realBits);
            AddInput(imag, imagBits);
        }
        ReduceComponent<NumThreads>(real, lane, warp, partials, output);
        ReduceComponent<NumThreads>(imag, lane, warp, partials, output + 1);
    } else {
        for (IdxT index = static_cast<IdxT>(tid); index < static_cast<IdxT>(diagonal);
             index += static_cast<IdxT>(NumThreads)) {
            AddInput(real, input[static_cast<int64_t>(index) * stride]);
        }
        ReduceComponent<NumThreads>(real, lane, warp, partials, output);
    }
}

} // namespace NsTraceExact
#endif
