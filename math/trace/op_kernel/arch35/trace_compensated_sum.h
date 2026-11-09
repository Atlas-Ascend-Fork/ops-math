/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRACE_COMPENSATED_SUM_H_
#define TRACE_COMPENSATED_SUM_H_

#include "trace_exact_sum.h"
#include "simt_api/device_functions.h"

namespace NsTraceCompensated {
using namespace AscendC;

constexpr uint32_t WARP_WIDTH = 32;
constexpr uint32_t THREAD_COUNT = 512;
constexpr uint32_t WARP_COUNT = THREAD_COUNT / WARP_WIDTH;
constexpr uint32_t HAS_HIGH = 8;
constexpr uint32_t HAS_LOW = 16;
constexpr int64_t TRIAL_LIMIT = int64_t(1) << 20;

__simt_callee__ __aicore__ inline float AddRN(float a, float b)
{
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
    return a + b;
}

__simt_callee__ __aicore__ inline float SubRN(float a, float b)
{
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
    return a - b;
}

__simt_callee__ __aicore__ inline uint32_t FloatBits(float value)
{
    union {
        float value;
        uint32_t bits;
    } word;
    word.value = value;
    return word.bits;
}

__simt_callee__ __aicore__ inline float BitsFloat(uint32_t bits)
{
    union {
        float value;
        uint32_t bits;
    } word;
    word.bits = bits;
    return word.value;
}

__simt_callee__ __aicore__ inline void OrderedFastTwoSum(float a, float b, float& sum, float& error)
{
    if ((FloatBits(a) & 0x7fffffffu) < (FloatBits(b) & 0x7fffffffu)) {
        const float temporary = a;
        a = b;
        b = temporary;
    }
    sum = AddRN(a, b);
    const float z = SubRN(sum, a);
    error = SubRN(b, z);
}

struct Channel {
    float high;
    float correction;
    uint32_t peak;
};

struct Component {
    Channel high;
    Channel low;
    uint32_t flags;
};

__simt_callee__ __aicore__ inline void ClearChannel(Channel& channel)
{
    channel.high = 0.0f;
    channel.correction = 0.0f;
    channel.peak = 0;
}

__simt_callee__ __aicore__ inline void ClearComponent(Component& component)
{
    ClearChannel(component.high);
    ClearChannel(component.low);
    component.flags = 0;
}

__simt_callee__ __aicore__ inline uint32_t ExponentCode(float value) { return (FloatBits(value) >> 23) & 255u; }

__simt_callee__ __aicore__ inline void AddCorrection(Channel& channel, float error)
{
    channel.correction = AddRN(channel.correction, error);
    const uint32_t code = ExponentCode(channel.correction);
    channel.peak = channel.peak > code ? channel.peak : code;
}

__simt_callee__ __aicore__ inline void AddChannel(Channel& channel, float value, bool first)
{
    if (first) {
        // The first actual nonzero term in this channel has no 0+x error.
        channel.high = value;
        channel.correction = 0.0f;
        return;
    }
    float error;
    OrderedFastTwoSum(channel.high, value, channel.high, error);
    AddCorrection(channel, error);
}

__simt_callee__ __aicore__ inline void AddOriginal(Component& component, uint32_t bits)
{
    const uint32_t exponent = (bits >> 23) & 255u;
    const uint32_t fraction = bits & 0x7fffffu;
    if (exponent == 255u) {
        component.flags |= fraction != 0 ? 1u : ((bits >> 31) != 0 ? 4u : 2u);
        return;
    }
    if ((bits & 0x7fffffffu) == 0) {
        return;
    }
    const bool high = exponent >= 89u;
    uint32_t mantissa = fraction;
    int32_t adjusted = static_cast<int32_t>(exponent) + (high ? -64 : 64);
    if (exponent == 0) {
        const uint32_t shift = static_cast<uint32_t>(__clz(static_cast<int>(fraction))) - 8u;
        mantissa = (fraction << shift) & 0x7fffffu;
        adjusted = 65 - static_cast<int32_t>(shift);
    }
    const float value = BitsFloat((bits & 0x80000000u) | (static_cast<uint32_t>(adjusted) << 23) | mantissa);
    const uint32_t flag = high ? HAS_HIGH : HAS_LOW;
    const bool first = (component.flags & flag) == 0;
    component.flags |= flag;
    if (high) {
        AddChannel(component.high, value, first);
    } else {
        AddChannel(component.low, value, first);
    }
}

__simt_callee__ __aicore__ inline uint32_t WarpFlags(uint32_t flags)
{
    const uint32_t packed = ((flags & 1u) != 0 ? 1u : 0u) | ((flags & 2u) != 0 ? (1u << 6) : 0u) |
                            ((flags & 4u) != 0 ? (1u << 12) : 0u) | ((flags & HAS_HIGH) != 0 ? (1u << 18) : 0u) |
                            ((flags & HAS_LOW) != 0 ? (1u << 24) : 0u);
    const uint32_t counts = asc_reduce_add(packed);
    return ((counts & 63u) != 0 ? 1u : 0u) | ((counts & (63u << 6)) != 0 ? 2u : 0u) |
           ((counts & (63u << 12)) != 0 ? 4u : 0u) | ((counts & (63u << 18)) != 0 ? HAS_HIGH : 0u) |
           ((counts & (63u << 24)) != 0 ? HAS_LOW : 0u);
}

__simt_callee__ __aicore__ inline void ReduceChannel(Channel& channel, uint32_t lane)
{
    const uint32_t initial = ExponentCode(channel.correction);
    channel.peak = channel.peak > initial ? channel.peak : initial;
    for (uint32_t offset = 16; offset != 0; offset >>= 1) {
        const float other = __shfl_down(channel.high, offset);
        if (lane < offset) {
            float error;
            OrderedFastTwoSum(channel.high, other, channel.high, error);
            AddCorrection(channel, error);
        }
    }
    channel.correction = asc_reduce_add(channel.correction);
    channel.peak = asc_reduce_max(channel.peak);
}

__simt_callee__ __aicore__ inline void ReduceComponent(Component& component, uint32_t lane)
{
    component.flags = WarpFlags(component.flags);
    if ((component.flags & 7u) != 0) {
        // Whole-warp category decision: finite channel values are irrelevant.
        return;
    }
    if ((component.flags & HAS_HIGH) != 0) {
        ReduceChannel(component.high, lane);
    }
    if ((component.flags & HAS_LOW) != 0) {
        ReduceChannel(component.low, lane);
    }
}

__simt_callee__ __aicore__ inline uint32_t Metadata(const Component& component)
{
    return component.high.peak | (component.low.peak << 8) | (component.flags << 16);
}

__simt_callee__ __aicore__ inline void StorePartial(const Component& component, __ubuf__ uint32_t* partials,
                                                    uint32_t offset)
{
    partials[offset] = FloatBits(component.high.high);
    partials[offset + 1] = FloatBits(component.high.correction);
    partials[offset + 2] = FloatBits(component.low.high);
    partials[offset + 3] = FloatBits(component.low.correction);
    partials[offset + 4] = Metadata(component);
}

__simt_callee__ __aicore__ inline void LoadPartial(Component& component, __ubuf__ uint32_t* partials, uint32_t offset,
                                                   bool active)
{
    ClearComponent(component);
    if (active) {
        component.high.high = BitsFloat(partials[offset]);
        component.high.correction = BitsFloat(partials[offset + 1]);
        component.low.high = BitsFloat(partials[offset + 2]);
        component.low.correction = BitsFloat(partials[offset + 3]);
        const uint32_t metadata = partials[offset + 4];
        component.high.peak = metadata & 255u;
        component.low.peak = (metadata >> 8) & 255u;
        component.flags = (metadata >> 16) & 31u;
    }
}

__simt_callee__ __aicore__ inline int32_t LeadingExponent(uint32_t bits)
{
    // Nonzero channel sums and EFT residuals remain on a normal grid.
    return static_cast<int32_t>((bits >> 23) & 255u) - 127;
}

__simt_callee__ __aicore__ inline uint32_t RescaleBits(uint32_t bits, int32_t scale)
{
    const uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude == 0) {
        return 0;
    }
    const uint32_t sign = bits & 0x80000000u;
    const int32_t exponent = static_cast<int32_t>((bits >> 23) & 255u) + scale;
    if (exponent >= 255) {
        return sign | 0x7f800000u;
    }
    if (exponent > 0) {
        return sign | (static_cast<uint32_t>(exponent) << 23) | (bits & 0x7fffffu);
    }
    const uint32_t drop = static_cast<uint32_t>(1 - exponent);
    if (drop > 24) {
        return 0;
    }
    const uint32_t mantissa = (bits & 0x7fffffu) | 0x800000u;
    uint32_t rounded = mantissa >> drop;
    const uint32_t remainder = mantissa & ((1u << drop) - 1u);
    const uint32_t halfway = 1u << (drop - 1u);
    if (remainder > halfway || (remainder == halfway && (rounded & 1u) != 0)) {
        ++rounded;
    }
    return rounded == 0 ? 0 : sign | rounded;
}

__simt_callee__ __aicore__ inline int32_t CeilLog2(uint32_t value)
{
    return value <= 1 ? 0 : 32 - __clz(static_cast<int>(value - 1));
}

__simt_callee__ __aicore__ inline void ChannelErrorPower(uint32_t peak, int32_t scale, int32_t grid, uint32_t diagonal,
                                                         int32_t& largest)
{
    if (peak == 0) {
        return;
    }
    const int32_t exponent = static_cast<int32_t>(peak) - 127;
    if (exponent > grid + 23) {
        const int32_t power = exponent - 23 + CeilLog2(diagonal + 527u) + scale;
        largest = largest > power ? largest : power;
    }
    if (exponent > grid + 17) {
        const int32_t power = exponent - 7 + scale;
        largest = largest > power ? largest : power;
    }
}

// The root lane passes only builtin scalars; no collective or GM write here.
__simt_callee__ __aicore__ static inline uint64_t RootFourTermFinalizer(uint32_t high, uint32_t highCorrection,
                                                                        uint32_t low, uint32_t lowCorrection,
                                                                        uint32_t metadata, uint32_t diagonal)
{
    const uint32_t flags = (metadata >> 16) & 31u;
    if ((flags & 1u) != 0 || (flags & 6u) == 6u) {
        return (uint64_t(1) << 32) | 0x7fc00000u;
    }
    if ((flags & 6u) != 0) {
        return (uint64_t(1) << 32) | ((flags & 2u) != 0 ? 0x7f800000u : 0xff800000u);
    }
    float qHigh;
    float deltaHigh;
    float qLow;
    float deltaLow;
    OrderedFastTwoSum(BitsFloat(high), BitsFloat(highCorrection), qHigh, deltaHigh);
    OrderedFastTwoSum(BitsFloat(low), BitsFloat(lowCorrection), qLow, deltaLow);
    const uint32_t highBits = FloatBits(qHigh);
    const uint32_t lowBits = FloatBits(qLow);
    const bool hasHigh = (highBits & 0x7fffffffu) != 0;
    const bool hasLow = (lowBits & 0x7fffffffu) != 0;
    int32_t largest = -512;
    ChannelErrorPower(metadata & 255u, 64, -125, diagonal, largest);
    ChannelErrorPower((metadata >> 8) & 255u, -64, -85, diagonal, largest);
    if (!hasHigh && !hasLow) {
        return largest == -512 ? uint64_t(1) << 32 : 0;
    }
    const int32_t highLeading = hasHigh ? LeadingExponent(highBits) + 64 : -512;
    const int32_t lowLeading = hasLow ? LeadingExponent(lowBits) - 64 : -512;
    const bool highDominant = highLeading >= lowLeading;
    const uint32_t selected = highDominant ? highBits : lowBits;
    const uint32_t delta = FloatBits(highDominant ? deltaHigh : deltaLow);
    const int32_t leading = highDominant ? highLeading : lowLeading;
    const int32_t scale = highDominant ? 64 : -64;
    if (hasHigh && hasLow) {
        const int32_t minorLeading = highDominant ? lowLeading : highLeading;
        if (leading - minorLeading < 54) {
            return 0;
        }
        const int32_t halfPower = LeadingExponent(selected) - ((selected & 0x7fffffu) == 0 ? 25 : 24);
        // Use exponent comparisons so a tiny midpoint threshold cannot FTZ.
        if ((delta & 0x7fffffffu) != 0 && LeadingExponent(delta) >= halfPower) {
            return 0;
        }
    }
    const uint32_t output = RescaleBits(selected, scale);
    if (largest == -512) {
        return (uint64_t(1) << 32) | output;
    }
    const int32_t boundPower = largest + 3;
    if (boundPower > leading - 27) {
        return output;
    }
    if ((leading == 127 && (output & 0x7fffffffu) >= 0x7f7fffffu) || (leading == 128 && (selected & 0x7fffffu) == 0)) {
        return output;
    }
    return (uint64_t(1) << 32) | output;
}

template <typename IdxT, bool IsComplex>
__simt_callee__ __aicore__ __attribute__((noinline)) void RootWarpExact(int64_t diagonal, int64_t stride,
                                                                        __gm__ uint32_t* input, __gm__ uint32_t* output)
{
    // Its32-thread constexpr path returns before every UB access/barrier.
    NsTraceExact::AccumulateExact<IdxT, IsComplex, WARP_WIDTH>(diagonal, stride, input, output,
                                                               static_cast<__ubuf__ uint32_t*>(nullptr));
}

__simt_callee__ __aicore__ inline uint32_t SingleBits(uint32_t bits)
{
    if ((bits & 0x7fffffffu) == 0) {
        return 0;
    }
    return ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x7fffffu) != 0) ? 0x7fc00000u : bits;
}

template <typename IdxT, bool IsComplex>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_COUNT) inline void DirectKernel(int64_t diagonal, int64_t stride,
                                                                                __gm__ uint32_t* input,
                                                                                __gm__ uint32_t* output,
                                                                                __ubuf__ uint32_t* partials)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid % WARP_WIDTH;
    const uint32_t warp = tid / WARP_WIDTH;
    if (diagonal <= 1) {
        if (tid == 0) {
            output[0] = diagonal == 0 ? 0 : SingleBits(input[0]);
            if constexpr (IsComplex) {
                output[1] = diagonal == 0 ? 0 : SingleBits(input[1]);
            }
        }
        return;
    }
    Component real;
    Component imag;
    ClearComponent(real);
    ClearComponent(imag);
    for (IdxT index = static_cast<IdxT>(tid); index < static_cast<IdxT>(diagonal);
         index += static_cast<IdxT>(THREAD_COUNT)) {
        const int64_t location = static_cast<int64_t>(index) * stride * (IsComplex ? 2 : 1);
        AddOriginal(real, input[location]);
        if constexpr (IsComplex) {
            AddOriginal(imag, input[location + 1]);
        }
    }
    if (static_cast<int64_t>(warp * WARP_WIDTH) < diagonal) {
        ReduceComponent(real, lane);
        if constexpr (IsComplex) {
            ReduceComponent(imag, lane);
        }
    }
    constexpr uint32_t WARP_WORDS = IsComplex ? 10 : 5;
    if (lane == 0) {
        StorePartial(real, partials, warp * WARP_WORDS);
        if constexpr (IsComplex) {
            StorePartial(imag, partials, warp * WARP_WORDS + 5);
        }
    }
    // Sole full-block publication barrier: all16 warp records are complete.
    asc_syncthreads();
    if (warp == 0) {
        LoadPartial(real, partials, lane * WARP_WORDS, lane < WARP_COUNT);
        ReduceComponent(real, lane);
        if constexpr (IsComplex) {
            LoadPartial(imag, partials, lane * WARP_WORDS + 5, lane < WARP_COUNT);
            ReduceComponent(imag, lane);
        }
        uint32_t miss = 0;
        if (lane == 0) {
            const uint64_t realResult = RootFourTermFinalizer(
                FloatBits(real.high.high), FloatBits(real.high.correction), FloatBits(real.low.high),
                FloatBits(real.low.correction), Metadata(real), static_cast<uint32_t>(diagonal));
            uint64_t imagResult = uint64_t(1) << 32;
            if constexpr (IsComplex) {
                imagResult = RootFourTermFinalizer(FloatBits(imag.high.high), FloatBits(imag.high.correction),
                                                   FloatBits(imag.low.high), FloatBits(imag.low.correction),
                                                   Metadata(imag), static_cast<uint32_t>(diagonal));
            }
            miss = ((realResult >> 32) & (imagResult >> 32)) == 0 ? 1 : 0;
            if (miss == 0) {
                output[0] = static_cast<uint32_t>(realResult);
                if constexpr (IsComplex) {
                    output[1] = static_cast<uint32_t>(imagResult);
                }
            }
        }
        if (diagonal >= 1024) {
            if (lane == 0) {
                // Direct partials use at most words0..159 of the192-word scratch.
                partials[NsTraceExact::WARP_COUNT * NsTraceExact::PARTIAL_WORDS - 1] = miss;
            }
        } else {
            miss = asc_shfl(miss, 0);
            if (miss != 0) {
                RootWarpExact<IdxT, IsComplex>(diagonal, stride, input, output);
            }
        }
    }
    if (diagonal >= 1024) {
        // Publish root's decision to every warp before any exact-path scratch use.
        asc_syncthreads();
        const uint32_t miss = partials[NsTraceExact::WARP_COUNT * NsTraceExact::PARTIAL_WORDS - 1];
        // Exact partials reuse all192 words; every thread must capture miss first.
        asc_syncthreads();
        if (miss != 0) {
            NsTraceExact::AccumulateExact<IdxT, IsComplex, THREAD_COUNT>(diagonal, stride, input, output, partials);
        }
    }
}

template <typename IdxT, bool IsComplex>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_COUNT) inline void ExactLargeKernel(int64_t diagonal, int64_t stride,
                                                                                    __gm__ uint32_t* input,
                                                                                    __gm__ uint32_t* output,
                                                                                    __ubuf__ uint32_t* partials)
{
    NsTraceExact::AccumulateExact<IdxT, IsComplex, THREAD_COUNT>(diagonal, stride, input, output, partials);
}

template <typename IdxT, bool IsComplex>
__aicore__ inline void Launch(int64_t diagonal, int64_t stride, __gm__ uint32_t* input, __gm__ uint32_t* output,
                              __ubuf__ uint32_t* partials)
{
    if (diagonal > TRIAL_LIMIT) {
        asc_vf_call<ExactLargeKernel<IdxT, IsComplex>>(dim3(THREAD_COUNT), diagonal, stride, input, output, partials);
    } else {
        asc_vf_call<DirectKernel<IdxT, IsComplex>>(dim3(THREAD_COUNT), diagonal, stride, input, output, partials);
    }
}

template <bool IsComplex>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR y, const TraceTilingData* tiling)
{
    LocalMemAllocator<AscendC::Hardware::UB> allocator;
    LocalTensor<uint32_t> scratch = allocator.Alloc<TPosition::VECCALC, uint32_t>(NsTraceExact::WARP_COUNT *
                                                                                  NsTraceExact::PARTIAL_WORDS);
    DataSyncBarrier<MemDsbT::UB>();
    __gm__ uint32_t* input = reinterpret_cast<__gm__ uint32_t*>(x);
    __gm__ uint32_t* output = reinterpret_cast<__gm__ uint32_t*>(y);
    __ubuf__ uint32_t* partials = reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr());
    const int64_t diagonal = tiling->diagSize;
    const int64_t stride = tiling->diagStride;
    const bool useInt32 = stride > 0 && diagonal <= static_cast<int64_t>(INT32_MAX / stride);
    if (useInt32) {
        Launch<int32_t, IsComplex>(diagonal, stride, input, output, partials);
    } else {
        Launch<int64_t, IsComplex>(diagonal, stride, input, output, partials);
    }
}

} // namespace NsTraceCompensated
#endif
