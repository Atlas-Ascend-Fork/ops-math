/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TOP_K_SMALL_K_UB_RADIX_SELECT_H
#define TOP_K_SMALL_K_UB_RADIX_SELECT_H
#include <cstdint>
#include <type_traits>
#include "top_k_small_size_bitonic_sort.h"
#include "top_k_constant_var_simd.h"
#include "radix_sort_topk_b8.h"
#include "radix_sort_topk_b64.h"

// Small-k UB radix selection: the caller owns GM/UB copying and synchronization.
// B8/B16/B32/B64 paths build keys, select a threshold, gather candidates and
// perform the final ordering. Values and source indices are returned in UB.
namespace topkV2 {

template <typename T>
using InputSelectKey = std::conditional_t<
    sizeof(T) == 1U, uint8_t,
    std::conditional_t<sizeof(T) == 2U, uint16_t, std::conditional_t<sizeof(T) == 4U, uint32_t, uint64_t>>>;

template <typename T, bool Bitonic>
__aicore__ inline bool UseTopKUbRadixSelect(uint32_t n, uint32_t k)
{
    constexpr uint32_t maxAxis = sizeof(T) == 2U ? INPUT_SELECT_B16_SINGLE_MAX_AXIS : INPUT_SELECT_MAX_AXIS;
    return Bitonic && n > 0U && n <= maxAxis && k >= 2U && k <= INPUT_SELECT_CANDIDATES && k <= n;
}

template <typename T>
__aicore__ inline uint32_t TopKUbRadixSelectKeyBytes(uint32_t inputStride, uint32_t rows)
{
    return static_cast<uint32_t>(TopKInputSelectKeyBytesShared(inputStride, rows, sizeof(InputSelectKey<T>)));
}

// Input and key allocations include one extra physical vector after the final row.
// Only [0,n) contributes to histograms; bitwise -0/+0 remain distinct at selection.
template <typename T, bool Largest>
__aicore__ inline void TopKUbRadixSelectBuildKeys(LocalTensor<T> input, LocalTensor<InputSelectKey<T>> keys, uint32_t n,
                                                  uint32_t rows, uint32_t inputStride)
{
    using K = InputSelectKey<T>;
    constexpr uint32_t WIDTH = INPUT_SELECT_LOAD_GUARD_BYTES / sizeof(K);
    constexpr K SIGN = K(1) << (sizeof(K) * 8U - 1U);
    constexpr K ALL = static_cast<K>(~K(0));
    constexpr K INF = std::is_same_v<T, half> ? K(0x7C00U) :
                                                (std::is_same_v<T, bfloat16_t> ? K(0x7F80U) : K(0x7F800000U));
    auto src = (__ubuf__ K*)input.GetPhyAddr();
    auto dst = (__ubuf__ K*)keys.GetPhyAddr();
    __VEC_SCOPE__
    {
        Reg::RegTensor<K> sign, all, absoluteMask;
        Reg::Duplicate(sign, SIGN);
        Reg::Duplicate(all, ALL);
        Reg::Duplicate(absoluteMask, static_cast<K>(SIGN - 1U));
        // Host admits only a complete platform-UB budget; row counts fit uint16_t.
        uint16_t rowCount = static_cast<uint16_t>(rows);
        uint16_t chunks = static_cast<uint16_t>((n + WIDTH - 1U) / WIDTH);
        for (uint16_t row = 0; row < rowCount; ++row) {
            uint32_t remaining = n;
            for (uint16_t chunk = 0; chunk < chunks; ++chunk) {
                uint32_t col = static_cast<uint32_t>(chunk) * WIDTH;
                Reg::MaskReg valid = Reg::UpdateMask<K>(remaining);
                Reg::RegTensor<K> raw, absolute, signBits, xorBits, key;
                Reg::LoadAlign(raw, src + row * inputStride + col);
                if constexpr (IsBitonicFloatType<T>) {
                    Reg::And(absolute, raw, absoluteMask, valid);
                    Reg::And(signBits, raw, sign, valid);
                    Reg::MaskReg negative, nan;
                    Reg::Compares<K, CMPMODE::NE>(negative, signBits, K(0), valid);
                    Reg::Select(xorBits, all, sign, negative);
                    Reg::Xor(key, raw, xorBits, valid);
                    Reg::Compares<K, CMPMODE::GT>(nan, absolute, INF, valid);
                    Reg::Select(key, all, key, nan);
                } else if constexpr (std::is_signed_v<T>) {
                    Reg::Xor(key, raw, sign, valid);
                } else {
                    key = raw;
                }
                if constexpr (Largest) {
                    Reg::Xor(key, key, all, valid);
                }
                Reg::StoreAlign(dst + row * inputStride + col, key, valid);
            }
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }
}

template <typename T>
__simd_callee__ inline void TopKUbRadixSelectLoadKeys32(Reg::RegTensor<uint32_t>& result,
                                                        __ubuf__ InputSelectKey<T>* src)
{
    static_assert(sizeof(T) == 4U, "Register key loads are used only by B32 selection");
    Reg::LoadAlign(result, src);
}

// Registers carry the number of matches across chunks and strict/equal scans.
// The destination mask is clipped BEFORE Scatter: at most k writes, even all-equal.
// The final histogram rank is the number of equal keys still required.
// It gives the strict output length, so both source-order streams can share loads.
template <typename PositionT>
__simd_callee__ inline void TopKUbRadixSelectCompact(__ubuf__ uint32_t* indices, Reg::RegTensor<PositionT>& source,
                                                     Reg::MaskReg& match, Reg::RegTensor<uint32_t>& count, uint32_t k)
{
    Reg::MaskReg full = Reg::CreateMask<PositionT>(), full32 = Reg::CreateMask<uint32_t>();
    Reg::RegTensor<PositionT> one, zero, flags, compact;
    Reg::RegTensor<uint32_t> total, zero32, lane, destination;
    Reg::Duplicate(one, PositionT(1));
    Reg::Duplicate(zero, PositionT(0));
    Reg::Select(flags, one, zero, match);
    if constexpr (sizeof(PositionT) == 2U) {
        Reg::Reduce<ReduceType::SUM, uint32_t, uint16_t>(total, flags, full);
    } else {
        Reg::Reduce<ReduceType::SUM, uint32_t>(total, flags, full);
    }
    Reg::Duplicate(zero32, 0U);
    Reg::Gather(total, total, zero32);
    Reg::GatherMask<PositionT, Reg::GatherMaskMode::NO_STORE_REG>(compact, source, match);
    Reg::Arange((Reg::RegTensor<int32_t>&)lane, 0);
    Reg::Add(destination, lane, count, full32);
    Reg::MaskReg inChunk, inQuota, write;
    Reg::Compare<uint32_t, CMPMODE::LT>(inChunk, lane, total, full32);
    Reg::Compares<uint32_t, CMPMODE::LT>(inQuota, destination, k, full32);
    Reg::And(write, inChunk, inQuota, full32);
    Reg::Select(destination, destination, zero32, write);
    if constexpr (sizeof(PositionT) == 2U) {
        Reg::RegTensor<uint32_t> expanded, unused;
        Reg::Interleave((Reg::RegTensor<uint16_t>&)expanded, (Reg::RegTensor<uint16_t>&)unused, compact, zero);
        Reg::Scatter(indices, expanded, destination, write);
    } else {
        Reg::Scatter(indices, compact, destination, write);
    }
    Reg::Add(count, count, total, full32);
}

template <typename T>
__simd_callee__ inline void TopKUbRadixSelectCollect(__ubuf__ InputSelectKey<T>* keys, __ubuf__ uint32_t* indices,
                                                     uint32_t n, uint32_t k, Reg::RegTensor<uint32_t>& thresholdReg)
{
    Reg::MaskReg full = Reg::CreateMask<uint32_t>();
    Reg::RegTensor<uint32_t> lane, count;
    Reg::Arange((Reg::RegTensor<int32_t>&)lane, 0);
    Reg::Duplicate(count, 0U);
    uint16_t scans = 2U;
    uint16_t chunks = static_cast<uint16_t>((n + INPUT_SELECT_VECTOR_ITEMS - 1U) / INPUT_SELECT_VECTOR_ITEMS);
    for (uint16_t equalPass = 0; equalPass < scans; ++equalPass) {
        uint32_t remaining = n;
        for (uint16_t chunk = 0; chunk < chunks; ++chunk) {
            uint32_t col = static_cast<uint32_t>(chunk) * INPUT_SELECT_VECTOR_ITEMS;
            Reg::MaskReg valid = Reg::UpdateMask<uint32_t>(remaining);
            Reg::RegTensor<uint32_t> value, sourceIndex;
            TopKUbRadixSelectLoadKeys32<T>(value, keys + col);
            Reg::MaskReg match;
            if (equalPass == 0U) {
                Reg::Compare<uint32_t, CMPMODE::LT>(match, value, thresholdReg, valid);
            } else {
                Reg::Compare<uint32_t, CMPMODE::EQ>(match, value, thresholdReg, valid);
            }
            Reg::And(match, match, valid, full);
            Reg::Adds(sourceIndex, lane, col, full);
            TopKUbRadixSelectCompact(indices, sourceIndex, match, count, k);
        }
    }
}

template <typename T>
__simd_callee__ inline void TopKUbRadixSelectCollectSharedU16(__ubuf__ InputSelectKey<T>* keys,
                                                              __ubuf__ uint32_t* indices, uint32_t n, uint32_t k,
                                                              Reg::RegTensor<uint32_t>& packedThreshold)
{
    Reg::MaskReg f32 = Reg::CreateMask<uint32_t>(), f16 = Reg::CreateMask<uint16_t>();
    Reg::RegTensor<uint32_t> strictCount, equalCount;
    Reg::Duplicate(strictCount, 0U);
    Reg::ShiftRights<uint32_t, int16_t>(equalCount, packedThreshold, int16_t(INPUT_SELECT_METADATA_HIGH_SHIFT), f32);
    Reg::RegTensor<uint16_t> threshold, zero, lane, source, key;
    Reg::Duplicate(zero, 0U);
    Reg::Arange((Reg::RegTensor<int16_t>&)lane, 0);
    Reg::RegTensor<uint16_t> unusedThreshold;
    Reg::DeInterleave(threshold, unusedThreshold, (Reg::RegTensor<uint16_t>&)packedThreshold, zero);
    Reg::Gather(threshold, threshold, zero);
    Reg::RegTensor<uint8_t> zero8;
    if constexpr (sizeof(T) == 1U) {
        Reg::Duplicate(zero8, uint8_t(0));
    }
    uint32_t remaining = n;
    uint16_t chunks = (n + ONE_TIMES_B16_NUM - 1U) / ONE_TIMES_B16_NUM;
    for (uint16_t chunk = 0; chunk < chunks; ++chunk) {
        uint16_t col = chunk * ONE_TIMES_B16_NUM;
        Reg::MaskReg valid = Reg::UpdateMask<uint16_t>(remaining), strict, equal;
        if constexpr (sizeof(T) == 1U) {
            Reg::RegTensor<uint8_t> raw;
            Reg::RegTensor<uint16_t> unused;
            Reg::LoadAlign(raw, keys + col);
            Reg::Interleave((Reg::RegTensor<uint8_t>&)key, (Reg::RegTensor<uint8_t>&)unused, raw, zero8);
        } else {
            Reg::LoadAlign(key, keys + col);
        }
        Reg::Adds(source, lane, col, f16);
        Reg::Compare<uint16_t, CMPMODE::LT>(strict, key, threshold, valid);
        Reg::And(strict, strict, valid, f16);
        Reg::Compare<uint16_t, CMPMODE::EQ>(equal, key, threshold, valid);
        Reg::And(equal, equal, valid, f16);
        TopKUbRadixSelectCompact(indices, source, strict, strictCount, k);
        TopKUbRadixSelectCompact(indices, source, equal, equalCount, k);
    }
}

// Candidate value bits by source index: float rows gather raw u32 patterns,
// 16-bit rows expand the gathered u16 lanes to one zero-extended u32 per item.
template <typename T>
__simd_callee__ inline void TopKUbRadixSelectGatherBits(Reg::RegTensor<uint32_t>& bits, __ubuf__ T* input,
                                                        Reg::RegTensor<uint32_t>& safeIndex, uint32_t k)
{
    uint32_t active = INPUT_SELECT_CANDIDATES;
    if constexpr (sizeof(T) == 4U) {
        Reg::MaskReg full = Reg::UpdateMask<uint32_t>(active);
        Reg::Gather(bits, (__ubuf__ uint32_t*)input, safeIndex, full);
    } else {
        Reg::MaskReg full = Reg::UpdateMask<uint32_t>(active);
        Reg::RegTensor<uint16_t> index16, raw;
        Reg::Cast<uint16_t, uint32_t, BITONIC_SMALL_CAST_U32_TO_U16>(index16, safeIndex, full);
        Reg::Pack(index16, (Reg::RegTensor<uint32_t>&)index16);
        uint32_t count16 = k;
        Reg::MaskReg mask16 = Reg::UpdateMask<uint16_t>(count16);
        Reg::Gather(raw, (__ubuf__ uint16_t*)input, index16, mask16);
        Reg::RegTensor<uint32_t> even, odd, unused;
        Reg::MaskReg full16 = Reg::CreateMask<uint16_t>();
        Reg::Cast<uint32_t, uint16_t, BITONIC_SMALL_CAST_U16_TO_U32_EVEN>(even, raw, full16);
        Reg::Cast<uint32_t, uint16_t, BITONIC_SMALL_CAST_U16_TO_U32_ODD>(odd, raw, full16);
        Reg::Interleave(bits, unused, even, odd);
    }
}

// Radix keys already order values monotonically with NaN collapsed, so the
// finalize key is one gather plus a signed-zero merge: radix ^ ~0 matches
// BitonicSmallRegBuildKey32's ordering and equality structure exactly (the
// smallest-direction NaN sentinel differs in value but never in rank or ties).
template <typename T, bool Largest>
__simd_callee__ inline void TopKUbRadixSelectFinishKey(Reg::RegTensor<uint32_t>& key, Reg::RegTensor<uint32_t>& bits,
                                                       Reg::RegTensor<uint32_t>& safeIndex,
                                                       __ubuf__ InputSelectKey<T>* keys, Reg::MaskReg& activeMask)
{
    static_assert(sizeof(T) == 4U, "Radix key reuse only covers 32-bit keys");
    constexpr uint32_t allBits = UINT32_MAX;
    constexpr uint32_t signBit = 1U << (sizeof(T) * 8U - 1U);
    Reg::Gather(key, (__ubuf__ uint32_t*)keys, safeIndex, activeMask);
    Reg::RegTensor<uint32_t> allBitsReg, absMaskReg, absBits, zeroKey;
    Reg::Duplicate(allBitsReg, allBits);
    Reg::Xor(key, key, allBitsReg, activeMask);
    if constexpr (IsBitonicFloatType<T>) {
        Reg::Duplicate(absMaskReg, signBit - 1U);
        Reg::And(absBits, bits, absMaskReg, activeMask);
        Reg::MaskReg zeroMask;
        Reg::Compares<uint32_t, CMPMODE::EQ>(zeroMask, absBits, 0U, activeMask);
        Reg::Duplicate(zeroKey, Largest ? signBit : (allBits ^ signBit));
        Reg::Select<uint32_t>(key, zeroKey, key, zeroMask);
    }
}

// Stage truncation keeps every stage whose XOR stride stays inside the
// next-pow2(k) candidate subnet. Wider strides only pair candidates against
// UINT32_MAX sentinels in direction 0, where the swap term is forced true and
// no exchange happens, so lanes [0, k) stay bitwise identical to the full
// 15-stage network (k = 32 keeps every stage).
template <bool CompareGroup, uint32_t Width, uint32_t Size = 2U, uint32_t Stride = Size / 2U>
__simd_callee__ inline void TopKUbRadixSelectNetworkW(Reg::RegTensor<uint32_t>& valueBits,
                                                      Reg::RegTensor<uint32_t>& key, Reg::RegTensor<uint32_t>& index,
                                                      Reg::RegTensor<uint32_t>& group, Reg::RegTensor<uint32_t>& lane,
                                                      Reg::MaskReg& activeMask)
{
    if constexpr (Stride < Width) {
        BitonicSmallReg32SwapStage<CompareGroup, Stride, Size>(valueBits, key, index, group, lane, activeMask);
    }
    if constexpr (Stride > 1U) {
        TopKUbRadixSelectNetworkW<CompareGroup, Width, Size, Stride / 2U>(valueBits, key, index, group, lane,
                                                                          activeMask);
    } else if constexpr (Size < BITONIC_SMALL_TOPK_SIZE) {
        TopKUbRadixSelectNetworkW<CompareGroup, Width, Size * 2U, Size>(valueBits, key, index, group, lane, activeMask);
    }
}

// Same stride filter for the sentinel-only all-equal stages.
template <uint32_t Width, uint32_t Size = 2U, uint32_t Stride = Size / 2U>
__simd_callee__ inline void TopKUbRadixSelectAllEqualNetworkW(Reg::RegTensor<uint32_t>& index,
                                                              Reg::RegTensor<uint32_t>& lane, Reg::MaskReg& activeMask)
{
    if constexpr (Stride < Width) {
        BitonicSmallRegAllEqualSwapStage<Stride, Size>(index, lane, activeMask);
    }
    if constexpr (Stride > 1U) {
        TopKUbRadixSelectAllEqualNetworkW<Width, Size, Stride / 2U>(index, lane, activeMask);
    } else if constexpr (Size < BITONIC_SMALL_TOPK_SIZE) {
        TopKUbRadixSelectAllEqualNetworkW<Width, Size * 2U, Size>(index, lane, activeMask);
    }
}

// All-equal candidates follow a fixed position-only permutation (R10): the
// network's equal-swap arrangement depends on positions, never values. Compute
// it once per group by running the sentinel-only network over arange.
template <uint32_t Width>
__simd_callee__ inline void TopKUbRadixSelectAllEqualSigma(Reg::RegTensor<uint32_t>& sigmaSafe, uint32_t k)
{
    uint32_t activeCount = INPUT_SELECT_CANDIDATES;
    Reg::MaskReg activeMask = Reg::UpdateMask<uint32_t>(activeCount);
    Reg::MaskReg validMask = Reg::UpdateMask<uint32_t>(k);
    Reg::RegTensor<uint32_t> sigma, invalid, zero, lane;
    Reg::Duplicate(invalid, UINT32_MAX);
    Reg::Duplicate(zero, 0U);
    Reg::Arange((Reg::RegTensor<int32_t>&)sigma, 0);
    Reg::Select<uint32_t>(sigma, sigma, invalid, validMask);
    Reg::Arange((Reg::RegTensor<int32_t>&)lane, 0);
    TopKUbRadixSelectAllEqualNetworkW<Width>(sigma, lane, activeMask);
    Reg::Select<uint32_t>(sigmaSafe, sigma, zero, validMask);
}

// Strict count 0 means every candidate shares one radix key: the final index
// order is the fixed all-equal permutation. Preserve each NaN payload by
// gathering values using the permuted indices, even when their keys agree.
template <typename T>
__simd_callee__ inline void TopKUbRadixSelectFinishAllEq(__ubuf__ T* input, __ubuf__ T* output,
                                                         __ubuf__ uint32_t* indices,
                                                         Reg::RegTensor<uint32_t>& sigmaSafe, uint32_t k)
{
    uint32_t active = INPUT_SELECT_CANDIDATES;
    Reg::MaskReg full = Reg::UpdateMask<uint32_t>(active);
    uint32_t validCount = k;
    Reg::MaskReg valid = Reg::UpdateMask<uint32_t>(validCount);
    Reg::RegTensor<uint32_t> index, safeIndex, zero, bits;
    Reg::Duplicate(zero, 0U);
    Reg::LoadAlign(index, indices);
    Reg::RegTensor<uint32_t> permuted;
    Reg::Gather(permuted, index, sigmaSafe);
    Reg::Select(safeIndex, permuted, zero, valid);
    TopKUbRadixSelectGatherBits<T>(bits, input, safeIndex, k);
    if constexpr (sizeof(T) == 4U) {
        Reg::StoreAlign((__ubuf__ uint32_t*)output, bits, valid);
    } else {
        BitonicSmallRegStoreB16Bits<T>(output, bits, k, valid);
    }
    Reg::StoreAlign(indices, permuted, valid);
}

__simd_callee__ inline void TopKUbRadixSelectStoreB8(__ubuf__ uint8_t* output, Reg::RegTensor<uint32_t>& bits,
                                                     uint32_t k)
{
    Reg::RegTensor<uint32_t> zero32;
    Reg::RegTensor<uint16_t> low16, unused16, zero16;
    Reg::RegTensor<uint8_t> low8, unused8;
    Reg::Duplicate(zero32, 0U);
    Reg::Duplicate(zero16, 0U);
    Reg::DeInterleave(low16, unused16, (Reg::RegTensor<uint16_t>&)bits, (Reg::RegTensor<uint16_t>&)zero32);
    Reg::DeInterleave(low8, unused8, (Reg::RegTensor<uint8_t>&)low16, (Reg::RegTensor<uint8_t>&)zero16);
    uint32_t remaining = k;
    Reg::MaskReg valid = Reg::UpdateMask<uint8_t>(remaining);
    Reg::StoreAlign(output, low8, valid);
}

// Two adjacent rows share one 64-lane instruction stream: row A occupies the
// low half, row B the high half. Every network stride stays inside a 32-lane
// half, so one pass of the truncated per-row network orders both rows at once.
// Rows arrive in Collect order; the standard comparator preserves the exact
// index arrangement. B16 rows gather and store raw bits independently.
// Byte keys gather through aligned words. The guard vector covers the last
// word even for odd-length rows, and invalid lanes use source index zero.
// Single rows and row pairs share index validity, the value network and typed
// stores. Adapters retain B32 pair scatter, independent raw B16 gathers, and
// B8 word gathers; no floating-point conversion is introduced.
// Format-specific finalizers keep their register instructions in the caller's VF.
template <typename T>
__simd_callee__ inline void TopKUbRadixSelectStoreValueBits(__ubuf__ T* output, Reg::RegTensor<uint32_t>& bits,
                                                            uint32_t k, Reg::MaskReg& valid)
{
    if constexpr (sizeof(T) == 4U) {
        Reg::StoreAlign((__ubuf__ uint32_t*)output, bits, valid);
    } else if constexpr (sizeof(T) == 2U) {
        BitonicSmallRegStoreB16Bits<T>(output, bits, k, valid);
    } else {
        TopKUbRadixSelectStoreB8((__ubuf__ uint8_t*)output, bits, k);
    }
}

template <typename T, bool Pair>
__simd_callee__ inline void TopKUbRadixSelectStoreContinuous(__ubuf__ T* output, __ubuf__ uint32_t* indices, uint32_t k,
                                                             uint32_t valueStride, uint32_t indexStride,
                                                             Reg::RegTensor<uint32_t>& bits,
                                                             Reg::RegTensor<uint32_t>& index,
                                                             Reg::RegTensor<uint32_t>& laneMod, Reg::MaskReg& full)
{
    static_assert(!Pair || sizeof(T) <= 2U, "Paired B32 values use their scatter path");
    uint32_t validCount = k;
    Reg::MaskReg valid32 = Reg::UpdateMask<uint32_t>(validCount);
    if constexpr (Pair) {
        Reg::RegTensor<uint32_t> upperLane, bitsB, indexB;
        Reg::Adds(upperLane, laneMod, INPUT_SELECT_CANDIDATES, full);
        Reg::Gather(bitsB, bits, upperLane);
        Reg::Gather(indexB, index, upperLane);
        if constexpr (sizeof(T) == 2U) {
            TopKUbRadixSelectStoreValueBits<T>(output, bits, k, valid32);
            TopKUbRadixSelectStoreValueBits<T>(output + valueStride, bitsB, k, valid32);
            Reg::StoreAlign(indices, index, valid32);
            Reg::StoreAlign(indices + indexStride, indexB, valid32);
        } else {
            TopKUbRadixSelectStoreValueBits<T>(output, bits, k, valid32);
            Reg::StoreAlign(indices, index, valid32);
            TopKUbRadixSelectStoreValueBits<T>(output + valueStride, bitsB, k, valid32);
            Reg::StoreAlign(indices + indexStride, indexB, valid32);
        }
    } else {
        TopKUbRadixSelectStoreValueBits<T>(output, bits, k, valid32);
        Reg::StoreAlign(indices, index, valid32);
    }
}

// Each format owns its complete candidate load/network/store stage. Registers
// shared by those steps stay local instead of crossing a loader interface.
template <typename T, bool Largest, uint32_t Width>
__simd_callee__ inline void TopKUbRadixSelectFinishPairB32(__ubuf__ T* input, __ubuf__ T* output,
                                                           __ubuf__ uint32_t* indices, __ubuf__ InputSelectKey<T>* keys,
                                                           uint32_t k, uint32_t inputStride, uint32_t valueStride,
                                                           uint32_t indexStride)
{
    uint32_t activeCount = 64U;
    Reg::MaskReg full = Reg::UpdateMask<uint32_t>(activeCount), valid, high;
    Reg::RegTensor<uint32_t> lane, laneMod, zero, invalid, index, bits, key, group;
    Reg::Arange((Reg::RegTensor<int32_t>&)lane, 0);
    Reg::RegTensor<uint32_t> mask31;
    Reg::Duplicate(mask31, INPUT_SELECT_CANDIDATES - 1U);
    Reg::And(laneMod, lane, mask31, full);
    Reg::Duplicate(zero, 0U);
    Reg::Duplicate(invalid, UINT32_MAX);
    Reg::Compares<uint32_t, CMPMODE::GE>(high, lane, INPUT_SELECT_CANDIDATES, full);
    Reg::Compares<uint32_t, CMPMODE::LT>(valid, laneMod, k, full);
    Reg::RegTensor<uint32_t> indexRawIdx;
    Reg::RegTensor<uint32_t> indexSafeIdx, safe;
    BitonicSmallRegPairGatherIndex(indexStride, laneMod, zero, full, valid, high, indexRawIdx, indexSafeIdx);
    BitonicSmallRegLoadPairIndex<false>(indices, indexSafeIdx, index, 0U, 0U, lane, full, valid);
    Reg::Select<uint32_t>(safe, index, zero, valid);
    Reg::RegTensor<uint32_t> rowOffset, srcOffset, srcIdx;
    Reg::Duplicate(rowOffset, inputStride);
    Reg::Select<uint32_t>(srcOffset, rowOffset, zero, high);
    Reg::Add(srcIdx, safe, srcOffset, full);
    Reg::Gather(bits, (__ubuf__ uint32_t*)input, srcIdx, full);
    TopKUbRadixSelectFinishKey<T, Largest>(key, bits, srcIdx, keys, full);
    Reg::Duplicate(group, 0U);
    TopKUbRadixSelectNetworkW<false, Width>(bits, key, index, group, lane, full);
    Reg::RegTensor<uint32_t> valueRawIdx, valueSafeIdx;
    BitonicSmallRegPairGatherIndex(valueStride, laneMod, zero, full, valid, high, valueRawIdx, valueSafeIdx);
    Reg::Scatter((__ubuf__ uint32_t*)output, bits, valueRawIdx, valid);
    Reg::Scatter(indices, index, indexRawIdx, valid);
}

template <typename T, bool Largest, uint32_t Width>
__simd_callee__ inline void TopKUbRadixSelectFinishPairB16(__ubuf__ T* input, __ubuf__ T* output,
                                                           __ubuf__ uint32_t* indices, uint32_t k, uint32_t inputStride,
                                                           uint32_t valueStride, uint32_t indexStride)
{
    uint32_t activeCount = 64U;
    Reg::MaskReg full = Reg::UpdateMask<uint32_t>(activeCount), valid, high;
    Reg::RegTensor<uint32_t> lane, laneMod, zero, invalid, index, bits, key, group;
    Reg::Arange((Reg::RegTensor<int32_t>&)lane, 0);
    Reg::RegTensor<uint32_t> mask31;
    Reg::Duplicate(mask31, INPUT_SELECT_CANDIDATES - 1U);
    Reg::And(laneMod, lane, mask31, full);
    Reg::Duplicate(zero, 0U);
    Reg::Duplicate(invalid, UINT32_MAX);
    Reg::Compares<uint32_t, CMPMODE::GE>(high, lane, INPUT_SELECT_CANDIDATES, full);
    Reg::Compares<uint32_t, CMPMODE::LT>(valid, laneMod, k, full);
    Reg::RegTensor<uint32_t> safe;
    uint32_t validCount = k;
    Reg::MaskReg valid32 = Reg::UpdateMask<uint32_t>(validCount);
    Reg::RegTensor<uint32_t> indexA, indexB, bitsA, bitsB;
    Reg::LoadAlign(indexA, indices);
    Reg::LoadAlign(indexB, indices + indexStride);
    Reg::Select(safe, indexA, zero, valid32);
    TopKUbRadixSelectGatherBits<T>(bitsA, input, safe, k);
    Reg::Select(safe, indexB, zero, valid32);
    TopKUbRadixSelectGatherBits<T>(bitsB, input + inputStride, safe, k);
    Reg::Select(bitsA, bitsA, zero, valid32);
    Reg::Select(bitsB, bitsB, zero, valid32);
    Reg::Select(indexA, indexA, invalid, valid32);
    Reg::Select(indexB, indexB, invalid, valid32);
    Reg::Gather(indexA, indexA, laneMod);
    Reg::Gather(indexB, indexB, laneMod);
    Reg::Gather(bitsA, bitsA, laneMod);
    Reg::Gather(bitsB, bitsB, laneMod);
    Reg::Select(bits, bitsB, bitsA, high);
    Reg::Select(index, indexB, indexA, high);
    BitonicSmallRegBuildKey32<T, Largest>(key, bits, full);
    Reg::Duplicate(group, 0U);
    TopKUbRadixSelectNetworkW<false, Width>(bits, key, index, group, lane, full);
    TopKUbRadixSelectStoreContinuous<T, true>(output, indices, k, valueStride, indexStride, bits, index, laneMod, full);
}

template <typename T, bool Largest, uint32_t Width, bool Pair>
__simd_callee__ inline void TopKUbRadixSelectFinishB8(__ubuf__ InputSelectKey<T>* keys, __ubuf__ T* output,
                                                      __ubuf__ uint32_t* indices, uint32_t k, uint32_t inputStride,
                                                      uint32_t valueStride, uint32_t indexStride)
{
    uint32_t activeCount = 64U;
    Reg::MaskReg full = Reg::UpdateMask<uint32_t>(activeCount), valid, high;
    Reg::RegTensor<uint32_t> lane, laneMod, zero, invalid, index, bits, key, group;
    Reg::Arange((Reg::RegTensor<int32_t>&)lane, 0);
    Reg::RegTensor<uint32_t> mask31;
    Reg::Duplicate(mask31, INPUT_SELECT_CANDIDATES - 1U);
    Reg::And(laneMod, lane, mask31, full);
    Reg::Duplicate(zero, 0U);
    Reg::Duplicate(invalid, UINT32_MAX);
    Reg::Compares<uint32_t, CMPMODE::GE>(high, lane, INPUT_SELECT_CANDIDATES, full);
    Reg::Compares<uint32_t, CMPMODE::LT>(valid, laneMod, k, full);
    if constexpr (!Pair) {
        Reg::MaskReg low;
        Reg::Not(low, high, full);
        Reg::And(valid, valid, low, full);
    }
    Reg::RegTensor<uint32_t> safe;
    Reg::RegTensor<uint32_t> rowOffset, position, bytePosition, wordPosition, shift, mask3, mask255, invert;
    Reg::Duplicate(rowOffset, Pair ? indexStride : 0U);
    Reg::Select(rowOffset, rowOffset, zero, high);
    Reg::Add(position, laneMod, rowOffset, full);
    Reg::Gather(index, indices, position, full);
    Reg::Select(safe, index, zero, valid);
    Reg::Select(index, index, invalid, valid);
    Reg::Duplicate(rowOffset, Pair ? inputStride : 0U);
    Reg::Select(rowOffset, rowOffset, zero, high);
    Reg::Add(bytePosition, safe, rowOffset, full);
    Reg::ShiftRights<uint32_t, int16_t>(wordPosition, bytePosition, int16_t(2), full);
    Reg::Duplicate(mask3, 3U);
    Reg::And(shift, bytePosition, mask3, full);
    Reg::ShiftLefts<uint32_t, int16_t>(shift, shift, int16_t(3), full);
    Reg::Gather(bits, (__ubuf__ uint32_t*)keys, wordPosition, full);
    Reg::ShiftRight(bits, bits, (Reg::RegTensor<int32_t>&)shift, full);
    Reg::Duplicate(mask255, 255U);
    Reg::And(bits, bits, mask255, full);
    // Invert the byte radix domain for the greater-is-better final network.
    Reg::Xor(key, bits, mask255, full);
    constexpr uint32_t rawXor = (std::is_signed_v<T> ? 128U : 0U) ^ (Largest ? 255U : 0U);
    Reg::Duplicate(invert, rawXor);
    Reg::Xor(bits, bits, invert, full);
    Reg::Duplicate(group, 0U);
    TopKUbRadixSelectNetworkW<false, Width>(bits, key, index, group, lane, full);
    TopKUbRadixSelectStoreContinuous<T, Pair>(output, indices, k, valueStride, indexStride, bits, index, laneMod, full);
}

template <typename T, bool Largest, uint32_t Width>
__simd_callee__ inline void TopKUbRadixSelectFinishSingle(__ubuf__ T* input, __ubuf__ T* output,
                                                          __ubuf__ uint32_t* indices, __ubuf__ InputSelectKey<T>* keys,
                                                          uint32_t k)
{
    uint32_t activeCount = INPUT_SELECT_CANDIDATES;
    Reg::MaskReg full = Reg::UpdateMask<uint32_t>(activeCount), valid, high;
    Reg::RegTensor<uint32_t> lane, laneMod, zero, invalid, index, bits, key, group;
    Reg::Arange((Reg::RegTensor<int32_t>&)lane, 0);
    Reg::RegTensor<uint32_t> mask31;
    Reg::Duplicate(mask31, INPUT_SELECT_CANDIDATES - 1U);
    Reg::And(laneMod, lane, mask31, full);
    Reg::Duplicate(zero, 0U);
    Reg::Duplicate(invalid, UINT32_MAX);
    Reg::Compares<uint32_t, CMPMODE::GE>(high, lane, INPUT_SELECT_CANDIDATES, full);
    Reg::Compares<uint32_t, CMPMODE::LT>(valid, laneMod, k, full);
    {
        Reg::MaskReg low;
        Reg::Not(low, high, full);
        Reg::And(valid, valid, low, full);
    }
    Reg::RegTensor<uint32_t> safe;
    Reg::LoadAlign(index, indices);
    Reg::Select(safe, index, zero, valid);
    TopKUbRadixSelectGatherBits<T>(bits, input, safe, k);
    Reg::Select(index, index, invalid, valid);
    if constexpr (sizeof(T) == 4U) {
        TopKUbRadixSelectFinishKey<T, Largest>(key, bits, safe, keys, full);
    } else {
        BitonicSmallRegBuildKey32<T, Largest>(key, bits, full);
    }
    Reg::Duplicate(group, 0U);
    TopKUbRadixSelectNetworkW<false, Width>(bits, key, index, group, lane, full);
    TopKUbRadixSelectStoreContinuous<T, false>(output, indices, k, 0U, 0U, bits, index, laneMod, full);
}

template <typename T, bool Largest, uint32_t Width, bool Pair = false>
__simd_callee__ inline void TopKUbRadixSelectFinish(__ubuf__ T* input, __ubuf__ T* output, __ubuf__ uint32_t* indices,
                                                    __ubuf__ InputSelectKey<T>* keys, uint32_t k,
                                                    uint32_t inputStride = 0U, uint32_t valueStride = 0U,
                                                    uint32_t indexStride = 0U)
{
    if constexpr (Pair && sizeof(T) == 4U) {
        TopKUbRadixSelectFinishPairB32<T, Largest, Width>(input, output, indices, keys, k, inputStride, valueStride,
                                                          indexStride);
    } else if constexpr (Pair && sizeof(T) == 2U) {
        TopKUbRadixSelectFinishPairB16<T, Largest, Width>(input, output, indices, k, inputStride, valueStride,
                                                          indexStride);
    } else if constexpr (sizeof(T) == 1U) {
        TopKUbRadixSelectFinishB8<T, Largest, Width, Pair>(keys, output, indices, k, inputStride, valueStride,
                                                           indexStride);
    } else {
        TopKUbRadixSelectFinishSingle<T, Largest, Width>(input, output, indices, keys, k);
    }
}

template <typename T>
__simd_callee__ inline void TopKUbRadixSelectCollectRow(__ubuf__ InputSelectKey<T>* keys, __ubuf__ uint32_t* indices,
                                                        __ubuf__ uint32_t* metadata, uint32_t n, uint32_t k)
{
    Reg::RegTensor<uint32_t> threshold;
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(threshold, metadata + INPUT_SELECT_METADATA_KEY_WORD);
    if constexpr (sizeof(T) <= 2U) {
        TopKUbRadixSelectCollectSharedU16<T>(keys, indices, n, k, threshold);
    } else {
        TopKUbRadixSelectCollect<T>(keys, indices, n, k, threshold);
    }
}

template <typename T>
__simd_callee__ inline void TopKUbRadixSelectCollectPair(__ubuf__ InputSelectKey<T>* keyPtr, __ubuf__ uint32_t* idx,
                                                         __ubuf__ uint32_t* thresholdPtr, uint32_t n, uint32_t k,
                                                         uint32_t r, uint32_t row, uint32_t inputStride,
                                                         uint32_t indexStride)
{
    TopKUbRadixSelectCollectRow<T>(keyPtr + r * inputStride, idx + r * indexStride,
                                   thresholdPtr + row * INPUT_SELECT_HIST_STRIDE, n, k);
    TopKUbRadixSelectCollectRow<T>(keyPtr + (r + 1U) * inputStride, idx + (r + 1U) * indexStride,
                                   thresholdPtr + (row + 1U) * INPUT_SELECT_HIST_STRIDE, n, k);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

// Width-specialized collect/finalize body of one row group. The bisheng
// backend rejects runtime-branched vector bodies of finalize size, so data
// dependent choices stay in scalar code and each branch owns a full
// __VEC_SCOPE__ (the row-pair pattern of reg_finalize). The all-equal branch
// recomputes the fixed permutation per pair inside its own scope.
template <typename T, bool Largest, uint32_t Width>
__simd_callee__ inline void TopKUbRadixSelectGroupRows(__ubuf__ InputSelectKey<T>* keyPtr, __ubuf__ T* src,
                                                       __ubuf__ T* dst, __ubuf__ uint32_t* idx,
                                                       __ubuf__ uint32_t* thresholdPtr, uint32_t n, uint32_t k,
                                                       uint32_t start, uint32_t count, uint8_t allEqBits,
                                                       uint32_t inputStride, uint32_t valueStride, uint32_t indexStride)
{
    uint16_t groupRows = static_cast<uint16_t>(count);
    uint16_t pairRows = groupRows - static_cast<uint16_t>(groupRows & 1U);
    for (uint16_t row = 0; row < pairRows; row += 2U) {
        uint32_t r = start + row;
        uint32_t allEqA = (allEqBits >> row) & 1U;
        uint32_t allEqB = (allEqBits >> (row + 1U)) & 1U;
        if constexpr (sizeof(T) > 1U) {
            if (allEqA != 0U && allEqB != 0U) {
                __VEC_SCOPE__
                {
                    Reg::RegTensor<uint32_t> sigmaSafe;
                    TopKUbRadixSelectAllEqualSigma<Width>(sigmaSafe, k);
                    TopKUbRadixSelectCollectPair<T>(keyPtr, idx, thresholdPtr, n, k, r, row, inputStride, indexStride);
                    TopKUbRadixSelectFinishAllEq<T>(src + r * inputStride, dst + r * valueStride, idx + r * indexStride,
                                                    sigmaSafe, k);
                    TopKUbRadixSelectFinishAllEq<T>(src + (r + 1U) * inputStride, dst + (r + 1U) * valueStride,
                                                    idx + (r + 1U) * indexStride, sigmaSafe, k);
                }
                continue;
            }
        }
        {
            __VEC_SCOPE__
            {
                TopKUbRadixSelectCollectPair<T>(keyPtr, idx, thresholdPtr, n, k, r, row, inputStride, indexStride);
                TopKUbRadixSelectFinish<T, Largest, Width, true>(src + r * inputStride, dst + r * valueStride,
                                                                 idx + r * indexStride, keyPtr + r * inputStride, k,
                                                                 inputStride, valueStride, indexStride);
            }
        }
    }
    if ((groupRows & 1U) != 0U) {
        uint16_t row = groupRows - 1U;
        uint32_t r = start + row;
        __VEC_SCOPE__
        {
            TopKUbRadixSelectCollectRow<T>(keyPtr + r * inputStride, idx + r * indexStride,
                                           thresholdPtr + row * INPUT_SELECT_HIST_STRIDE, n, k);
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            TopKUbRadixSelectFinish<T, Largest, Width>(src + r * inputStride, dst + r * valueStride,
                                                       idx + r * indexStride, keyPtr + r * inputStride, k);
        }
    }
}

template <typename T, bool Largest, uint32_t Width = 2U>
__aicore__ inline void TopKUbRadixSelectDispatchGroupRows(__ubuf__ InputSelectKey<T>* keys, __ubuf__ T* input,
                                                          __ubuf__ T* values, __ubuf__ uint32_t* indices,
                                                          __ubuf__ uint32_t* threshold, uint32_t n, uint32_t k,
                                                          uint32_t start, uint32_t rows, uint8_t allEqBits,
                                                          uint32_t inputStride, uint32_t valueStride,
                                                          uint32_t indexStride)
{
    if (k <= Width || Width == INPUT_SELECT_CANDIDATES) {
        TopKUbRadixSelectGroupRows<T, Largest, Width>(keys, input, values, indices, threshold, n, k, start, rows,
                                                      allEqBits, inputStride, valueStride, indexStride);
    } else if constexpr (Width < INPUT_SELECT_CANDIDATES) {
        TopKUbRadixSelectDispatchGroupRows<T, Largest, Width * 2U>(keys, input, values, indices, threshold, n, k, start,
                                                                   rows, allEqBits, inputStride, valueStride,
                                                                   indexStride);
    }
}

// Compare the two unsigned halves lexicographically; neither half is discarded.
// Reuse the existing invalid-lane/tie exchange rules with (low, high) as payload/key.
template <uint32_t Stride, uint32_t Size>
__simd_callee__ inline void TopKUbRadixSelectB64SwapStage(Reg::RegTensor<uint32_t>& lo, Reg::RegTensor<uint32_t>& hi,
                                                          Reg::RegTensor<uint32_t>& index,
                                                          Reg::RegTensor<uint32_t>& lane, Reg::MaskReg& active)
{
    Reg::RegTensor<uint32_t> stride, peerLane, peerLo, peerHi, peerIndex, lowLo, highLo, lowHi, highHi;
    Reg::RegTensor<uint32_t> lowIndex, highIndex, unusedGroup, unusedPeerGroup;
    Reg::Duplicate(stride, Stride);
    Reg::Xor(peerLane, lane, stride, active);
    Reg::Gather(peerLo, lo, peerLane);
    Reg::Gather(peerHi, hi, peerLane);
    Reg::Gather(peerIndex, index, peerLane);
    Reg::MaskReg lowLane, compare, highEqual, lowGreater;
    Reg::Compare<uint32_t, CMPMODE::LT>(lowLane, lane, peerLane, active);
    Reg::Select(lowLo, lo, peerLo, lowLane);
    Reg::Select(highLo, peerLo, lo, lowLane);
    Reg::Select(lowHi, hi, peerHi, lowLane);
    Reg::Select(highHi, peerHi, hi, lowLane);
    Reg::Select(lowIndex, index, peerIndex, lowLane);
    Reg::Select(highIndex, peerIndex, index, lowLane);
    Reg::Compare<uint32_t, CMPMODE::GT>(compare, lowHi, highHi, active);
    Reg::Compare<uint32_t, CMPMODE::EQ>(highEqual, lowHi, highHi, active);
    Reg::Compare<uint32_t, CMPMODE::GT>(lowGreater, lowLo, highLo, active);
    Reg::And(highEqual, highEqual, lowGreater, active);
    Reg::Or(compare, compare, highEqual, active);
    BitonicSmallReg32ApplySwapByMask<false, Size>(lo, hi, index, unusedGroup, peerLo, peerHi, peerIndex,
                                                  unusedPeerGroup, compare, lowIndex, highIndex, lane, active);
}

template <uint32_t Size = 2U, uint32_t Stride = Size / 2U>
__simd_callee__ inline void TopKUbRadixSelectB64Network(Reg::RegTensor<uint32_t>& lo, Reg::RegTensor<uint32_t>& hi,
                                                        Reg::RegTensor<uint32_t>& index, Reg::RegTensor<uint32_t>& lane,
                                                        Reg::MaskReg& active)
{
    TopKUbRadixSelectB64SwapStage<Stride, Size>(lo, hi, index, lane, active);
    if constexpr (Stride > 1U) {
        TopKUbRadixSelectB64Network<Size, Stride / 2U>(lo, hi, index, lane, active);
    } else if constexpr (Size < BITONIC_SMALL_TOPK_SIZE) {
        TopKUbRadixSelectB64Network<Size * 2U, Size>(lo, hi, index, lane, active);
    }
}

// One native B64 load serves both source-order streams. Compact on B32 lanes
// after packing the B64 predicate, retaining the complete 64-bit comparison.
__simd_callee__ inline void TopKUbRadixSelectCollectB64(__ubuf__ uint64_t* keys, __ubuf__ uint32_t* indices, uint32_t n,
                                                        uint32_t k, uint64_t threshold, uint32_t equalQuota)
{
    uint32_t activeCount = INPUT_SELECT_CANDIDATES;
    Reg::MaskReg full = Reg::UpdateMask<uint32_t>(activeCount);
    Reg::RegTensor<uint32_t> lane, sourceIndex, strictCount, equalCount;
    Reg::Arange((Reg::RegTensor<int32_t>&)lane, 0);
    Reg::Duplicate(strictCount, 0U);
    Reg::Duplicate(equalCount, k - equalQuota);
    Reg::MaskReg full64 = Reg::CreateMask<uint64_t>();
    Reg::RegTensor<uint64_t> thresholdReg;
    Reg::Duplicate(thresholdReg, threshold);
    uint32_t remaining = n;
    uint16_t chunks = static_cast<uint16_t>((n + INPUT_SELECT_CANDIDATES - 1U) / INPUT_SELECT_CANDIDATES);
    for (uint16_t chunk = 0; chunk < chunks; ++chunk) {
        uint32_t col = uint32_t(chunk) * INPUT_SELECT_CANDIDATES;
        Reg::MaskReg valid = Reg::UpdateMask<uint64_t>(remaining);
        Reg::RegTensor<uint64_t> key;
        Reg::LoadAlign(key, keys + col);
        Reg::MaskReg strict64, equal64, strict32, equal32;
        Reg::Compare<uint64_t, CMPMODE::LT>(strict64, key, thresholdReg, valid);
        Reg::Compare<uint64_t, CMPMODE::EQ>(equal64, key, thresholdReg, valid);
        Reg::And(strict64, strict64, valid, full64);
        Reg::And(equal64, equal64, valid, full64);
        Reg::Pack(strict32, strict64);
        Reg::Pack(equal32, equal64);
        Reg::And(strict32, strict32, full, full);
        Reg::And(equal32, equal32, full, full);
        Reg::Adds(sourceIndex, lane, col, full);
        TopKUbRadixSelectCompact(indices, sourceIndex, strict32, strictCount, k - equalQuota);
        TopKUbRadixSelectCompact(indices, sourceIndex, equal32, equalCount, k);
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

__simd_callee__ inline void TopKUbRadixSelectStoreB64(__ubuf__ uint64_t* input, __ubuf__ uint64_t* output,
                                                      __ubuf__ uint32_t* indices, Reg::RegTensor<uint32_t>& index,
                                                      uint32_t k)
{
    uint32_t validCount = k;
    Reg::MaskReg valid = Reg::UpdateMask<uint32_t>(validCount);
    Reg::RegTensor<uint32_t> zero, safe, word, nextWord, lo, hi, packed, unused;
    Reg::Duplicate(zero, 0U);
    Reg::Select(safe, index, zero, valid);
    Reg::Add(word, safe, safe, valid);
    Reg::Adds(nextWord, word, 1U, valid);
    Reg::Gather(lo, (__ubuf__ uint32_t*)input, word, valid);
    Reg::Gather(hi, (__ubuf__ uint32_t*)input, nextWord, valid);
    Reg::Interleave(packed, unused, lo, hi);
    uint32_t words = k * 2U;
    Reg::MaskReg store = Reg::UpdateMask<uint32_t>(words);
    Reg::StoreAlign((__ubuf__ uint32_t*)output, packed, store);
    Reg::StoreAlign(indices, index, valid);
}

template <typename T>
__aicore__ inline void TopKUbRadixSelectFinishB64Simd(__ubuf__ T* input, __ubuf__ uint64_t* keys, __ubuf__ T* output,
                                                      __ubuf__ uint32_t* indices, uint32_t n, uint32_t k,
                                                      uint64_t threshold, uint32_t equalQuota)
{
    static_assert(sizeof(T) == sizeof(uint64_t));
    __VEC_SCOPE__ { TopKUbRadixSelectCollectB64(keys, indices, n, k, threshold, equalQuota); }
    if (equalQuota == k) {
        __VEC_SCOPE__
        {
            Reg::RegTensor<uint32_t> sigma, index, permuted;
            TopKUbRadixSelectAllEqualSigma<INPUT_SELECT_CANDIDATES>(sigma, k);
            Reg::LoadAlign(index, indices);
            Reg::Gather(permuted, index, sigma);
            TopKUbRadixSelectStoreB64((__ubuf__ uint64_t*)input, (__ubuf__ uint64_t*)output, indices, permuted, k);
        }
    } else {
        __VEC_SCOPE__
        {
            uint32_t activeCount = INPUT_SELECT_CANDIDATES;
            Reg::MaskReg active = Reg::UpdateMask<uint32_t>(activeCount);
            uint32_t validCount = k;
            Reg::MaskReg valid = Reg::UpdateMask<uint32_t>(validCount);
            Reg::RegTensor<uint32_t> index, zero, invalid, safe, word, nextWord, lo, hi, all, lane;
            Reg::Duplicate(zero, 0U);
            Reg::Duplicate(invalid, UINT32_MAX);
            Reg::LoadAlign(index, indices);
            Reg::Select(safe, index, zero, valid);
            Reg::Select(index, index, invalid, valid);
            Reg::Add(word, safe, safe, active);
            Reg::Adds(nextWord, word, 1U, active);
            Reg::Gather(lo, (__ubuf__ uint32_t*)keys, word, active);
            Reg::Gather(hi, (__ubuf__ uint32_t*)keys, nextWord, active);
            Reg::Duplicate(all, UINT32_MAX);
            Reg::Xor(lo, lo, all, active);
            Reg::Xor(hi, hi, all, active);
            Reg::Arange((Reg::RegTensor<int32_t>&)lane, 0);
            TopKUbRadixSelectB64Network(lo, hi, index, lane, active);
            TopKUbRadixSelectStoreB64((__ubuf__ uint64_t*)input, (__ubuf__ uint64_t*)output, indices, index, k);
        }
    }
}

template <typename K>
__aicore__ inline void TopKUbRadixSelectAdvanceB64Threshold(__ubuf__ int32_t* h, K* desired, K* mask, uint32_t* rank,
                                                            uint32_t count, int32_t pass)
{
    for (uint32_t row = 0; row < count; ++row) {
        uint32_t low = 0U, high = RADIX_SORT_BIN_NUM - 1U;
        while (low < high) {
            uint32_t mid = (low + high) / 2U;
            if (static_cast<uint32_t>(h[row * INPUT_SELECT_HIST_STRIDE + mid]) < rank[row]) {
                low = mid + 1U;
            } else {
                high = mid;
            }
        }
        if (low > 0U) {
            rank[row] -= h[row * INPUT_SELECT_HIST_STRIDE + low - 1U];
        }
        desired[row] |= static_cast<K>(low) << (pass * SHIFT_BIT_NUM);
        mask[row] |= static_cast<K>(0xFFU) << (pass * SHIFT_BIT_NUM);
    }
}

template <typename T, bool Largest>
__aicore__ inline void RunTopKUbRadixSelectB64(LocalTensor<T> input, LocalTensor<T> values,
                                               LocalTensor<uint32_t> indices, LocalTensor<uint8_t> scratch, uint32_t n,
                                               uint32_t k, uint32_t rows, uint32_t allocatedRows, uint32_t inputStride,
                                               uint32_t valueStride, uint32_t indexStride)
{
    static_assert(std::is_integral_v<T> && sizeof(T) == sizeof(uint64_t));
    using K = InputSelectKey<T>;
    auto keys = scratch.template ReinterpretCast<K>();
    auto hist = scratch[TopKUbRadixSelectKeyBytes<T>(inputStride, allocatedRows)].template ReinterpretCast<int32_t>();
    auto h = (__ubuf__ int32_t*)hist.GetPhyAddr();
    for (uint32_t start = 0; start < rows; start += INPUT_SELECT_GROUP_ROWS) {
        uint32_t count = rows - start < INPUT_SELECT_GROUP_ROWS ? rows - start : INPUT_SELECT_GROUP_ROWS;
        TopKUbRadixSelectBuildKeys<T, Largest>(input[start * inputStride], keys[start * inputStride], n, count,
                                               inputStride);
        K desired[INPUT_SELECT_GROUP_ROWS] = {};
        K mask[INPUT_SELECT_GROUP_ROWS] = {};
        uint32_t rank[INPUT_SELECT_GROUP_ROWS];
        for (uint32_t row = 0; row < count; ++row) {
            rank[row] = k;
        }
        for (int32_t pass = sizeof(K) - 1; pass >= 0; --pass) {
            for (uint32_t row = 0; row < count; ++row) {
                RadixSortTopKB64<T, K, sizeof(K), Largest, int32_t> radix;
                radix.GetCumSum(keys[(start + row) * inputStride], hist[row * INPUT_SELECT_HIST_STRIDE], mask[row],
                                desired[row], static_cast<uint16_t>(pass), n);
            }
            event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(event);
            WaitFlag<HardEvent::V_S>(event);
            TopKUbRadixSelectAdvanceB64Threshold<K>(h, desired, mask, rank, count, pass);
        }
        for (uint32_t row = 0; row < count; ++row) {
            uint32_t r = start + row;
            TopKUbRadixSelectFinishB64Simd<T>((__ubuf__ T*)input[r * inputStride].GetPhyAddr(),
                                              (__ubuf__ uint64_t*)keys[r * inputStride].GetPhyAddr(),
                                              (__ubuf__ T*)values[r * valueStride].GetPhyAddr(),
                                              (__ubuf__ uint32_t*)indices[r * indexStride].GetPhyAddr(), n, k,
                                              desired[row], rank[row]);
        }
    }
}

} // namespace topkV2

namespace topkV2UbRadixSelect {
using namespace topkV2;

// Both native histograms use two 128-bin U16 halves and an exact U16 residual rank.
__simd_callee__ inline void TopKUbRadixSelectFindBucket(Reg::RegTensor<uint16_t>& h0, Reg::RegTensor<uint16_t>& h1,
                                                        Reg::RegTensor<uint16_t>& rank,
                                                        Reg::RegTensor<uint16_t>& zero16,
                                                        Reg::RegTensor<uint32_t>& zero32, Reg::MaskReg& full16,
                                                        Reg::MaskReg& full32, Reg::RegTensor<uint16_t>& bucket,
                                                        Reg::RegTensor<uint32_t>& bucket32)
{
    Reg::MaskReg below0, below1;
    Reg::Compare<uint16_t, CMPMODE::LT>(below0, h0, rank, full16);
    Reg::Compare<uint16_t, CMPMODE::LT>(below1, h1, rank, full16);
    Reg::RegTensor<uint16_t> one, flags0, flags1;
    Reg::Duplicate(one, 1U);
    Reg::Select(flags0, one, zero16, below0);
    Reg::Select(flags1, one, zero16, below1);
    Reg::RegTensor<uint32_t> count0, count1;
    Reg::Reduce<ReduceType::SUM, uint32_t, uint16_t>(count0, flags0, full16);
    Reg::Reduce<ReduceType::SUM, uint32_t, uint16_t>(count1, flags1, full16);
    Reg::Add(bucket32, count0, count1, full32);
    Reg::Gather(bucket32, bucket32, zero32);
    Reg::RegTensor<uint16_t> previous, mask127, cum0, cum1, preceding;
    Reg::Cast<uint16_t, uint32_t, BITONIC_SMALL_CAST_U32_TO_U16>(bucket, bucket32, full32);
    Reg::Pack(bucket, (Reg::RegTensor<uint32_t>&)bucket);
    Reg::Gather(bucket, bucket, zero16);
    Reg::Adds(previous, bucket, uint16_t(0xffffU), full16);
    Reg::Duplicate(mask127, 127U);
    Reg::And(previous, previous, mask127, full16);
    Reg::Gather(cum0, h0, previous);
    Reg::Gather(cum1, h1, previous);
    Reg::MaskReg upper, nonzero;
    Reg::Compares<uint16_t, CMPMODE::GT>(upper, bucket, uint16_t(128), full16);
    Reg::Select(preceding, cum1, cum0, upper);
    Reg::Compares<uint16_t, CMPMODE::NE>(nonzero, bucket, uint16_t(0), full16);
    Reg::Select(preceding, preceding, zero16, nonzero);
    Reg::Sub(rank, rank, preceding, full16);
}

// Preserve the alternating histogram chains; only their per-vector work is shared.
__simd_callee__ inline void TopKUbRadixSelectAccumulateB16(
    Reg::RegTensor<uint16_t>& inputVectorOne, Reg::MaskReg& histMask, Reg::RegTensor<uint16_t>& andMaskVector,
    Reg::RegTensor<uint16_t>& desiredReg, Reg::RegTensor<uint16_t>& firstRoundVector, uint16_t round, int16_t bitOffset,
    Reg::RegTensor<uint16_t>& zeroVector, Reg::MaskReg& predicateDefaultB16, Reg::RegTensor<uint16_t>& chistVectorZero,
    Reg::RegTensor<uint16_t>& chistVectorOne)
{
    // The first pass counts every valid element; later passes retain only the
    // selected high-bit prefix before accumulating the current radix byte.
    Reg::RegTensor<uint16_t> andValueZero;
    Reg::And(andValueZero, inputVectorOne, andMaskVector, histMask);
    Reg::MaskReg cmpValueMask;
    Reg::Compare<uint16_t, CMPMODE::EQ>(cmpValueMask, andValueZero, desiredReg, histMask);
    Reg::MaskReg newHistMask;
    Reg::And(newHistMask, cmpValueMask, histMask, predicateDefaultB16);
    Reg::MaskReg finalMask, firstRoundMask;
    Reg::Compares<uint16_t, CMPMODE::EQ>(firstRoundMask, firstRoundVector, round, histMask);
    Reg::Select(finalMask, histMask, newHistMask, firstRoundMask);
    Reg::RegTensor<uint16_t> shiftVecOne;
    Reg::ShiftRights<uint16_t, int16_t>(shiftVecOne, inputVectorOne, bitOffset, finalMask);
    Reg::RegTensor<uint8_t> shiftVecU8LowBit, shiftVecU8HighBit;
    Reg::DeInterleave(shiftVecU8LowBit, shiftVecU8HighBit, (Reg::RegTensor<uint8_t>&)shiftVecOne,
                      (Reg::RegTensor<uint8_t>&)zeroVector);
    Reg::MaskReg maskU8;
    Reg::Pack(maskU8, finalMask);
    Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(
        chistVectorZero, shiftVecU8LowBit, maskU8);
    Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(
        chistVectorOne, shiftVecU8LowBit, maskU8);
}

template <typename T, typename UNSIGNED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX>
struct RadixSortTopKB16InputSelect {
    __aicore__ inline RadixSortTopKB16InputSelect() {}
    __aicore__ inline void GetCumSum(LocalTensor<UNSIGNED_TYPE> inputX, LocalTensor<int32_t> blockCumSum,
                                     uint16_t round, uint32_t k, uint32_t numTileData);
};

// The high half is equal quota before small-rank decode, strict count afterwards.
__simd_callee__ inline void TopKUbRadixSelectStoreB16Metadata(__ubuf__ uint32_t* metadata,
                                                              Reg::RegTensor<uint16_t>& key,
                                                              Reg::RegTensor<uint16_t>& count, Reg::MaskReg& full32)
{
    Reg::RegTensor<uint32_t> packed, unused;
    Reg::Interleave((Reg::RegTensor<uint16_t>&)packed, (Reg::RegTensor<uint16_t>&)unused, key, count);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(metadata, packed, full32);
}

// Scalar pass configuration is prepared outside the VF. It owns no registers;
// the u16 block bounds retain the hardware-loop types used by the caller.
struct TopKUbRadixSelectB16HistogramPass {
    uint32_t numTileData;
    uint16_t repateTime;
    uint16_t pairCount;
    uint16_t tailBegin;
    uint16_t andDataMask;
    uint16_t firstRound;
    uint16_t round;
    int16_t bitOffset;
};

__simd_callee__ inline void TopKUbRadixSelectBuildB16Histogram(
    __ubuf__ uint16_t* inputXValuePtr, const TopKUbRadixSelectB16HistogramPass& pass,
    Reg::RegTensor<uint16_t>& desiredReg, Reg::RegTensor<uint16_t>& zeroVector, Reg::MaskReg& predicateDefaultB16,
    Reg::RegTensor<uint16_t>& chistVectorZero, Reg::RegTensor<uint16_t>& chistVectorOne)
{
    // Prefix and first-pass comparison vectors belong only to this histogram.
    Reg::RegTensor<uint16_t> andMaskVector, firstRoundVector;
    Reg::Duplicate(andMaskVector, pass.andDataMask, predicateDefaultB16);
    Reg::Duplicate(firstRoundVector, pass.firstRound, predicateDefaultB16);
    Reg::RegTensor<uint16_t> inputVectorOne;
    uint32_t inputElementNum = pass.numTileData;
    Reg::RegTensor<uint16_t> histZero1, histOne1;
    Reg::Duplicate(histZero1, 0, predicateDefaultB16);
    Reg::Duplicate(histOne1, 0, predicateDefaultB16);
    for (uint16_t group = 0; group < pass.pairCount; ++group) {
        {
            Reg::MaskReg histMask = Reg::UpdateMask<uint16_t>(inputElementNum);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(inputVectorOne, inputXValuePtr,
                                                                         ::topkV2::ONE_TIMES_B16_NUM);
            TopKUbRadixSelectAccumulateB16(inputVectorOne, histMask, andMaskVector, desiredReg, firstRoundVector,
                                           pass.round, pass.bitOffset, zeroVector, predicateDefaultB16, chistVectorZero,
                                           chistVectorOne);
        }
        {
            Reg::MaskReg histMask = Reg::UpdateMask<uint16_t>(inputElementNum);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(inputVectorOne, inputXValuePtr,
                                                                         ::topkV2::ONE_TIMES_B16_NUM);
            TopKUbRadixSelectAccumulateB16(inputVectorOne, histMask, andMaskVector, desiredReg, firstRoundVector,
                                           pass.round, pass.bitOffset, zeroVector, predicateDefaultB16, histZero1,
                                           histOne1);
        }
    }
    for (uint16_t i = pass.tailBegin; i < pass.repateTime; ++i) {
        Reg::MaskReg histMask = Reg::UpdateMask<uint16_t>(inputElementNum);
        Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(inputVectorOne, inputXValuePtr,
                                                                     ::topkV2::ONE_TIMES_B16_NUM);
        TopKUbRadixSelectAccumulateB16(inputVectorOne, histMask, andMaskVector, desiredReg, firstRoundVector,
                                       pass.round, pass.bitOffset, zeroVector, predicateDefaultB16, chistVectorZero,
                                       chistVectorOne);
    }
    Reg::Add(chistVectorZero, chistVectorZero, histZero1, predicateDefaultB16);
    Reg::Add(chistVectorOne, chistVectorOne, histOne1, predicateDefaultB16);
}

__simd_callee__ inline void TopKUbRadixSelectStoreB16Threshold(
    __ubuf__ int32_t* blockCumSumPtr, Reg::RegTensor<uint16_t>& chistVectorZero,
    Reg::RegTensor<uint16_t>& chistVectorOne, Reg::RegTensor<uint16_t>& rankReg, Reg::RegTensor<uint16_t>& desiredReg,
    Reg::RegTensor<uint16_t>& zeroVector, Reg::MaskReg& predicateDefaultB16, Reg::MaskReg& predicateDefaultB32,
    int16_t bitOffset, uint32_t k)
{
    auto state = (__ubuf__ uint16_t*)(blockCumSumPtr + ::topkV2::INPUT_SELECT_METADATA_KEY_WORD);
    Reg::RegTensor<uint32_t> bucket32, zero32;
    Reg::RegTensor<uint16_t> bucket, shifted, kReg, strictTotal;
    Reg::Duplicate(zero32, 0U);
    TopKUbRadixSelectFindBucket(chistVectorZero, chistVectorOne, rankReg, zeroVector, zero32, predicateDefaultB16,
                                predicateDefaultB32, bucket, bucket32);
    Reg::ShiftLefts<uint16_t, int16_t>(shifted, bucket, bitOffset, predicateDefaultB16);
    Reg::Or(desiredReg, desiredReg, shifted, predicateDefaultB16);
    Reg::Duplicate(kReg, uint16_t(k));
    Reg::Sub(strictTotal, kReg, rankReg, predicateDefaultB16);
    TopKUbRadixSelectStoreB16Metadata((__ubuf__ uint32_t*)(blockCumSumPtr + INPUT_SELECT_METADATA_KEY_WORD), desiredReg,
                                      strictTotal, predicateDefaultB32);
    Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_FIRST_ELEMENT_B16>(state + INPUT_SELECT_METADATA_HALFWORDS_PER_WORD,
                                                                      rankReg, predicateDefaultB16);
}

template <typename T, typename UNSIGNED_TYPE, int32_t NUM_PASS, bool IS_DESCEND, typename T_INDEX>
__aicore__ inline void RadixSortTopKB16InputSelect<T, UNSIGNED_TYPE, NUM_PASS, IS_DESCEND, T_INDEX>::GetCumSum(
    LocalTensor<UNSIGNED_TYPE> inputX, LocalTensor<int32_t> blockCumSum, uint16_t round, uint32_t k,
    uint32_t numTileData)
{
    __ubuf__ UNSIGNED_TYPE* inputXValuePtr = (__ubuf__ UNSIGNED_TYPE*)inputX.GetPhyAddr();
    __ubuf__ int32_t* blockCumSumPtr = (__ubuf__ int32_t*)blockCumSum.GetPhyAddr();
    UNSIGNED_TYPE andDataMask = round == NUM_PASS - 1 ? 0U : 0xFF00U;
    auto state = (__ubuf__ uint16_t*)(blockCumSumPtr + ::topkV2::INPUT_SELECT_METADATA_KEY_WORD);
    int16_t bitOffset = round * ::topkV2::SHIFT_BIT_NUM;
    uint16_t repateTime = (numTileData + ::topkV2::ONE_TIMES_B16_NUM - 1) / ::topkV2::ONE_TIMES_B16_NUM;
    const uint16_t pairCount = static_cast<uint16_t>(repateTime / 2U);
    const uint16_t tailBegin = static_cast<uint16_t>(pairCount * 2U);
    uint16_t roundValue = NUM_PASS - 1;
    const TopKUbRadixSelectB16HistogramPass pass{numTileData,           repateTime, pairCount, tailBegin,
                                                 uint16_t(andDataMask), roundValue, round,     bitOffset};
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> chistVectorZero, chistVectorOne;
        Reg::RegTensor<uint16_t> zeroVector;
        Reg::MaskReg predicateDefaultB32 = Reg::CreateMask<uint32_t>();
        Reg::MaskReg predicateDefaultB16 = Reg::CreateMask<uint16_t>();
        Reg::Duplicate(zeroVector, 0, predicateDefaultB16);
        Reg::Duplicate(chistVectorZero, 0, predicateDefaultB16);
        Reg::Duplicate(chistVectorOne, 0, predicateDefaultB16);
        Reg::RegTensor<uint16_t> desiredReg, rankReg;
        Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_BRC_B16>(desiredReg, state);
        Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_BRC_B16>(rankReg,
                                                              state + INPUT_SELECT_METADATA_HALFWORDS_PER_WORD);
        TopKUbRadixSelectBuildB16Histogram((__ubuf__ uint16_t*)inputXValuePtr, pass, desiredReg, zeroVector,
                                           predicateDefaultB16, chistVectorZero, chistVectorOne);
        TopKUbRadixSelectStoreB16Threshold(blockCumSumPtr, chistVectorZero, chistVectorOne, rankReg, desiredReg,
                                           zeroVector, predicateDefaultB16, predicateDefaultB32, bitOffset, k);
    }
}

// Input-select bounds n by 8192, so cumulative counts fit u16; keys remain full u32.
__simd_callee__ inline void TopKUbRadixSelectAccumulateB32(__ubuf__ uint32_t* source, Reg::MaskReg& valid,
                                                           Reg::RegTensor<uint32_t>& desired,
                                                           Reg::RegTensor<uint32_t>& prefixMask, int16_t bitOffset,
                                                           Reg::RegTensor<uint16_t>& h0, Reg::RegTensor<uint16_t>& h1)
{
    Reg::MaskReg f32 = Reg::CreateMask<uint32_t>(), selected, mask16, byteMask;
    Reg::RegTensor<uint32_t> raw, prefix, shifted, zero32;
    Reg::LoadAlign(raw, source);
    Reg::And(prefix, raw, prefixMask, valid);
    Reg::Compare<uint32_t, CMPMODE::EQ>(selected, prefix, desired, valid);
    Reg::And(selected, selected, valid, f32);
    Reg::ShiftRights<uint32_t, int16_t>(shifted, raw, bitOffset, selected);
    Reg::Duplicate(zero32, 0U);
    Reg::RegTensor<uint16_t> low16, unused16, zero16;
    Reg::Duplicate(zero16, 0U);
    Reg::DeInterleave(low16, unused16, (Reg::RegTensor<uint16_t>&)shifted, (Reg::RegTensor<uint16_t>&)zero32);
    Reg::Pack(mask16, selected);
    Reg::RegTensor<uint8_t> bytes, unusedBytes;
    Reg::DeInterleave(bytes, unusedBytes, (Reg::RegTensor<uint8_t>&)low16, (Reg::RegTensor<uint8_t>&)zero16);
    Reg::Pack(byteMask, mask16);
    Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(h0, bytes,
                                                                                                      byteMask);
    Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(h1, bytes,
                                                                                                      byteMask);
}

__simd_callee__ inline void TopKUbRadixSelectBuildB32Histogram(__ubuf__ uint32_t* source, uint32_t n, uint16_t chunks,
                                                               Reg::RegTensor<uint32_t>& desired,
                                                               Reg::RegTensor<uint32_t>& mask, int16_t bitOffset,
                                                               Reg::RegTensor<uint16_t>& h0,
                                                               Reg::RegTensor<uint16_t>& h1, Reg::MaskReg& f16)
{
    constexpr uint32_t items = INPUT_SELECT_LOAD_GUARD_BYTES / sizeof(uint32_t);
    uint32_t remaining = n;
    Reg::RegTensor<uint16_t> other0, other1;
    Reg::Duplicate(other0, 0U);
    Reg::Duplicate(other1, 0U);
    uint16_t pairs = chunks / 2U;
    for (uint16_t pair = 0; pair < pairs; ++pair) {
        Reg::MaskReg valid0 = Reg::UpdateMask<uint32_t>(remaining);
        TopKUbRadixSelectAccumulateB32(source + uint32_t(pair) * 2U * items, valid0, desired, mask, bitOffset, h0, h1);
        Reg::MaskReg valid1 = Reg::UpdateMask<uint32_t>(remaining);
        TopKUbRadixSelectAccumulateB32(source + (uint32_t(pair) * 2U + 1U) * items, valid1, desired, mask, bitOffset,
                                       other0, other1);
    }
    for (uint16_t tail = pairs * 2U; tail < chunks; ++tail) {
        Reg::MaskReg valid = Reg::UpdateMask<uint32_t>(remaining);
        TopKUbRadixSelectAccumulateB32(source + uint32_t(tail) * items, valid, desired, mask, bitOffset, h0, h1);
    }
    Reg::Add(h0, h0, other0, f16);
    Reg::Add(h1, h1, other1, f16);
}

__simd_vf__ inline void TopKUbRadixSelectB32ThresholdVF(__ubuf__ uint32_t* source, __ubuf__ uint32_t* state,
                                                        uint16_t pass, uint32_t n)
{
    constexpr uint32_t items = INPUT_SELECT_LOAD_GUARD_BYTES / sizeof(uint32_t);
    auto rankPtr = (__ubuf__ uint16_t*)(state + INPUT_SELECT_METADATA_RANK_WORD - INPUT_SELECT_METADATA_KEY_WORD);
    uint32_t prefixBits = pass == 3U ? 0U : UINT32_MAX << ((pass + 1U) * 8U);
    int16_t bitOffset = pass * 8U;
    uint16_t chunks = (n + items - 1U) / items;
    Reg::MaskReg f16 = Reg::CreateMask<uint16_t>(), f32 = Reg::CreateMask<uint32_t>();
    Reg::RegTensor<uint16_t> h0, h1, zero16, rank;
    Reg::RegTensor<uint32_t> desired, mask, zero32;
    Reg::Duplicate(h0, 0U);
    Reg::Duplicate(h1, 0U);
    Reg::Duplicate(zero16, 0U);
    Reg::Duplicate(zero32, 0U);
    Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_BRC_B16>(rank, rankPtr);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(desired, state);
    Reg::Duplicate(mask, prefixBits);
    TopKUbRadixSelectBuildB32Histogram(source, n, chunks, desired, mask, bitOffset, h0, h1, f16);
    Reg::RegTensor<uint16_t> bucket;
    Reg::RegTensor<uint32_t> bucket32;
    TopKUbRadixSelectFindBucket(h0, h1, rank, zero16, zero32, f16, f32, bucket, bucket32);

    Reg::RegTensor<uint32_t> result;
    Reg::ShiftLefts<uint32_t, int16_t>(result, bucket32, bitOffset, f32);
    Reg::Or(result, result, desired, f32);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(state, result, f32);
    Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_FIRST_ELEMENT_B16>(rankPtr, rank, f16);
}

__aicore__ inline void TopKUbRadixSelectNativeThresholdPass(LocalTensor<uint32_t> keys, LocalTensor<int32_t> histogram,
                                                            uint16_t pass, uint32_t n)
{
    asc_vf_call<TopKUbRadixSelectB32ThresholdVF>(
        (__ubuf__ uint32_t*)keys.GetPhyAddr(),
        (__ubuf__ uint32_t*)histogram.GetPhyAddr() + ::topkV2::INPUT_SELECT_METADATA_KEY_WORD, pass, n);
}

template <typename K>
__aicore__ inline void TopKUbRadixSelectInitNativeState(LocalTensor<int32_t> histogram, uint32_t rows, uint32_t k)
{
    static_assert(sizeof(K) == 2U || sizeof(K) == 4U);
    auto state = (__ubuf__ uint32_t*)histogram.GetPhyAddr();
    __VEC_SCOPE__
    {
        Reg::MaskReg full = Reg::CreateMask<uint32_t>();
        Reg::RegTensor<uint32_t> zero, rank;
        Reg::Duplicate(zero, 0U);
        Reg::Duplicate(rank, k);
        for (uint16_t row = 0; row < uint16_t(rows); ++row) {
            auto ptr = state + uint32_t(row) * INPUT_SELECT_HIST_STRIDE + ::topkV2::INPUT_SELECT_METADATA_KEY_WORD;
            Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(ptr, zero, full);
            Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                ptr + INPUT_SELECT_METADATA_RANK_WORD - INPUT_SELECT_METADATA_KEY_WORD, rank, full);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }
}

// B16 retains its native threshold representation and uses the common row finalizer.
__simd_callee__ inline void TopKUbRadixSelectFindB16Minimum(__ubuf__ uint16_t* source, uint32_t n, uint16_t chunks,
                                                            Reg::RegTensor<uint16_t>& threshold,
                                                            Reg::RegTensor<uint16_t>& invalid16,
                                                            Reg::RegTensor<uint16_t>& zero16, Reg::MaskReg& f16,
                                                            Reg::RegTensor<uint16_t>& best)
{
    uint32_t remain = n;
    for (uint16_t chunk = 0; chunk < chunks; ++chunk) {
        Reg::MaskReg valid = Reg::UpdateMask<uint16_t>(remain);
        Reg::RegTensor<uint16_t> value, selected, small;
        Reg::LoadAlign(value, source + uint32_t(chunk) * ::topkV2::ONE_TIMES_B16_NUM);
        Reg::MaskReg greater, equal, live;
        Reg::Compare<uint16_t, CMPMODE::GT>(greater, value, threshold, valid);
        Reg::Compare<uint16_t, CMPMODE::EQ>(equal, value, threshold, valid);
        Reg::Or(live, greater, equal, f16);
        Reg::Select(selected, value, invalid16, live);
        Reg::Reduce<ReduceType::MIN, uint16_t>(small, selected, f16);
        Reg::Gather(small, small, zero16);
        Reg::Min(best, best, small, f16);
    }
}

__simd_callee__ inline void TopKUbRadixSelectCountB16Matches(__ubuf__ uint16_t* source, uint32_t n, uint16_t chunks,
                                                             Reg::RegTensor<uint16_t>& best,
                                                             Reg::RegTensor<uint16_t>& ones16,
                                                             Reg::RegTensor<uint16_t>& zero16,
                                                             Reg::RegTensor<uint32_t>& zero32, Reg::MaskReg& f16,
                                                             Reg::MaskReg& f32, Reg::RegTensor<uint32_t>& matches)
{
    uint32_t remain = n;
    for (uint16_t chunk = 0; chunk < chunks; ++chunk) {
        Reg::MaskReg valid = Reg::UpdateMask<uint16_t>(remain), equal;
        Reg::RegTensor<uint16_t> value, flags;
        Reg::LoadAlign(value, source + uint32_t(chunk) * ::topkV2::ONE_TIMES_B16_NUM);
        Reg::Compare<uint16_t, CMPMODE::EQ>(equal, value, best, valid);
        Reg::Select(flags, ones16, zero16, equal);
        Reg::RegTensor<uint32_t> total;
        Reg::Reduce<ReduceType::SUM, uint32_t, uint16_t>(total, flags, f16);
        Reg::Gather(total, total, zero32);
        Reg::Add(matches, matches, total, f32);
    }
}

template <typename T>
__aicore__ inline void TopKUbRadixSelectRegisterThresholdB16(LocalTensor<InputSelectKey<T>> keys,
                                                             LocalTensor<int32_t> hist, uint32_t n, uint32_t k,
                                                             uint32_t rows, uint32_t stride)
{
    auto src = (__ubuf__ uint16_t*)keys.GetPhyAddr();
    auto out = (__ubuf__ uint32_t*)hist.GetPhyAddr();
    __VEC_SCOPE__
    {
        Reg::MaskReg f16 = Reg::CreateMask<uint16_t>(), f32 = Reg::CreateMask<uint32_t>();
        Reg::RegTensor<uint16_t> lane, zero16, ones16, invalid16;
        Reg::RegTensor<uint32_t> zero32;
        Reg::Arange((Reg::RegTensor<int16_t>&)lane, 0);
        Reg::Duplicate(zero16, 0U);
        Reg::Duplicate(ones16, 1U);
        Reg::Duplicate(invalid16, UINT16_MAX);
        Reg::Duplicate(zero32, 0U);
        for (uint16_t row = 0; row < uint16_t(rows); ++row) {
            Reg::RegTensor<uint16_t> threshold, tieCount;
            Reg::Duplicate(threshold, 0U);
            Reg::Duplicate(tieCount, 0U);
            for (uint16_t rank = 0; rank < uint16_t(k); ++rank) {
                Reg::RegTensor<uint16_t> best;
                Reg::Duplicate(best, UINT16_MAX);
                uint16_t chunks = (n + ::topkV2::ONE_TIMES_B16_NUM - 1U) / ::topkV2::ONE_TIMES_B16_NUM;
                TopKUbRadixSelectFindB16Minimum(src + uint32_t(row) * stride, n, chunks, threshold, invalid16, zero16,
                                                f16, best);
                Reg::MaskReg changed;
                Reg::Compare<uint16_t, CMPMODE::NE>(changed, best, threshold, f16);
                Reg::Select(tieCount, zero16, tieCount, changed);
                // Count all occurrences at the current key to decide whether the next order statistic stays on it.
                Reg::RegTensor<uint32_t> matches;
                Reg::Duplicate(matches, 0U);
                TopKUbRadixSelectCountB16Matches(src + uint32_t(row) * stride, n, chunks, best, ones16, zero16, zero32,
                                                 f16, f32, matches);
                Reg::Adds(tieCount, tieCount, uint16_t(1), f16);
                // Save the kth key and equal quota. Advance only after exhausting all equal occurrences.
                TopKUbRadixSelectStoreB16Metadata(
                    out + uint32_t(row) * INPUT_SELECT_HIST_STRIDE + INPUT_SELECT_METADATA_KEY_WORD, best, tieCount,
                    f32);
                Reg::RegTensor<uint16_t> total16;
                Reg::Cast<uint16_t, uint32_t, BITONIC_SMALL_CAST_U32_TO_U16>(total16, matches, f32);
                Reg::Pack(total16, (Reg::RegTensor<uint32_t>&)total16);
                Reg::Gather(total16, total16, zero16);
                Reg::MaskReg exhausted;
                Reg::Compare<uint16_t, CMPMODE::GE>(exhausted, tieCount, total16, f16);
                Reg::RegTensor<uint16_t> next;
                Reg::Adds(next, best, uint16_t(1), f16);
                Reg::Select(threshold, next, best, exhausted);
                Reg::Select(tieCount, zero16, tieCount, exhausted);
            }
        }
    }
}

// B16/B32 share full-row key construction and group scheduling. The typed
// branches keep the B16 small-rank metadata and native histogram instructions.
template <typename T, bool Largest>
__aicore__ inline void TopKUbRadixSelectThresholdNativeGroup(LocalTensor<InputSelectKey<T>> keys,
                                                             LocalTensor<int32_t> hist, uint32_t n, uint32_t k,
                                                             uint32_t count, uint32_t inputStride, bool smallRank)
{
    using K = InputSelectKey<T>;
    if constexpr (sizeof(T) == 2U) {
        if (smallRank) {
            TopKUbRadixSelectRegisterThresholdB16<T>(keys, hist, n, k, count, inputStride);
        }
    }
    if (!smallRank) {
        TopKUbRadixSelectInitNativeState<K>(hist, count, k);
        for (int32_t pass = sizeof(K) - 1; pass >= 0; --pass) {
            for (uint32_t row = 0; row < count; ++row) {
                if constexpr (sizeof(T) == 2U) {
                    RadixSortTopKB16InputSelect<T, K, sizeof(K), Largest, int32_t> radix;
                    radix.GetCumSum(keys[row * inputStride], hist[row * ::topkV2::INPUT_SELECT_HIST_STRIDE], pass, k,
                                    n);
                } else {
                    TopKUbRadixSelectNativeThresholdPass(
                        keys[row * inputStride], hist[row * ::topkV2::INPUT_SELECT_HIST_STRIDE], uint16_t(pass), n);
                }
            }
            __VEC_SCOPE__ { Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>(); }
        }
    }
}

__aicore__ inline uint8_t TopKUbRadixSelectDecodeAllEq(__ubuf__ uint32_t* state, uint32_t count, uint32_t k,
                                                       bool smallRank)
{
    uint8_t allEqBits = 0U;
    for (uint32_t row = 0; row < count; ++row) {
        const uint32_t base = row * INPUT_SELECT_HIST_STRIDE;
        uint32_t rank = smallRank ? TopKInputSelectMetadataCount(state[base + INPUT_SELECT_METADATA_KEY_WORD]) :
                                    state[base + INPUT_SELECT_METADATA_RANK_WORD] & INPUT_SELECT_METADATA_LOW_MASK;
        allEqBits |= uint8_t(rank == k) << row;
    }
    return allEqBits;
}

__aicore__ inline void TopKUbRadixSelectRepackSmallRank(LocalTensor<int32_t> hist, uint32_t count, uint32_t k)
{
    auto packedPtr = (__ubuf__ uint32_t*)hist.GetPhyAddr();
    __VEC_SCOPE__
    {
        Reg::MaskReg full = Reg::CreateMask<uint32_t>();
        Reg::RegTensor<uint32_t> word, high, low, mask, kReg, strict;
        Reg::Duplicate(mask, INPUT_SELECT_METADATA_LOW_MASK);
        Reg::Duplicate(kReg, k);
        for (uint16_t row = 0; row < uint16_t(count); ++row) {
            auto ptr = packedPtr + uint32_t(row) * ::topkV2::INPUT_SELECT_HIST_STRIDE +
                       ::topkV2::INPUT_SELECT_METADATA_KEY_WORD;
            Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(word, ptr);
            Reg::ShiftRights<uint32_t, int16_t>(high, word, int16_t(INPUT_SELECT_METADATA_HIGH_SHIFT), full);
            Reg::Sub(strict, kReg, high, full);
            Reg::ShiftLefts<uint32_t, int16_t>(high, strict, int16_t(INPUT_SELECT_METADATA_HIGH_SHIFT), full);
            Reg::And(low, word, mask, full);
            Reg::Or(word, low, high, full);
            Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(ptr, word, full);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }
}

template <typename T, bool Largest>
__aicore__ inline void RunB16B32(LocalTensor<T> input, LocalTensor<T> values, LocalTensor<uint32_t> indices,
                                 LocalTensor<uint8_t> scratch, uint32_t n, uint32_t k, uint32_t rows,
                                 uint32_t allocatedRows, uint32_t inputStride, uint32_t valueStride,
                                 uint32_t indexStride)
{
    static_assert(sizeof(T) == 2U || sizeof(T) == 4U);
    using K = ::topkV2::InputSelectKey<T>;
    auto keys = scratch.template ReinterpretCast<K>();
    auto hist = scratch[::topkV2::TopKUbRadixSelectKeyBytes<T>(inputStride, allocatedRows)]
                    .template ReinterpretCast<int32_t>();
    auto state = (__ubuf__ uint32_t*)hist.GetPhyAddr();
    ::topkV2::TopKUbRadixSelectBuildKeys<T, Largest>(input, keys, n, rows, inputStride);
    auto keyPtr = (__ubuf__ K*)keys.GetPhyAddr();
    auto src = (__ubuf__ T*)input.GetPhyAddr();
    auto dst = (__ubuf__ T*)values.GetPhyAddr();
    auto idx = (__ubuf__ uint32_t*)indices.GetPhyAddr();
    for (uint32_t start = 0; start < rows; start += ::topkV2::INPUT_SELECT_GROUP_ROWS) {
        uint32_t count = rows - start < ::topkV2::INPUT_SELECT_GROUP_ROWS ? rows - start :
                                                                            ::topkV2::INPUT_SELECT_GROUP_ROWS;
        const bool smallRank = sizeof(T) == 2U && k <= 2U;
        TopKUbRadixSelectThresholdNativeGroup<T, Largest>(keys[start * inputStride], hist, n, k, count, inputStride,
                                                          smallRank);
        event_t ready = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(ready);
        WaitFlag<HardEvent::V_S>(ready);
        // Read equal quota before its high half is replaced with strict count.
        uint8_t allEqBits = TopKUbRadixSelectDecodeAllEq(state, count, k, smallRank);
        if constexpr (sizeof(T) == 2U) {
            if (smallRank) {
                TopKUbRadixSelectRepackSmallRank(hist, count, k);
            }
        }
        ::topkV2::TopKUbRadixSelectDispatchGroupRows<T, Largest>(keyPtr, src, dst, idx, state, n, k, start, count,
                                                                 allEqBits, inputStride, valueStride, indexStride);
    }
}

// B8 keeps byte keys and the existing scratch budget. Its CDF, u16 source
// positions and paired SIMD network follow the optimized B16/B32 dataflow.
__simd_callee__ inline void TopKUbRadixSelectThresholdB8(__ubuf__ uint32_t* histogram, uint32_t k)
{
    Reg::MaskReg full = Reg::CreateMask<uint32_t>();
    Reg::RegTensor<uint32_t> zero, one, rank, buckets, accumulated;
    Reg::Duplicate(zero, 0U);
    Reg::Duplicate(one, 1U);
    Reg::Duplicate(rank, k);
    Reg::Duplicate(buckets, 0U);
    for (uint16_t part = 0; part < 4U; ++part) {
        Reg::RegTensor<uint32_t> cdf, flags, count;
        Reg::MaskReg below;
        Reg::LoadAlign(cdf, histogram + uint32_t(part) * 64U);
        Reg::Compare<uint32_t, CMPMODE::LT>(below, cdf, rank, full);
        Reg::Select(flags, one, zero, below);
        Reg::Reduce<ReduceType::SUM, uint32_t>(count, flags, full);
        Reg::Add(buckets, buckets, count, full);
    }
    Reg::Gather(buckets, buckets, zero);
    Reg::RegTensor<uint32_t> previous, safe, packed;
    Reg::Adds(previous, buckets, UINT32_MAX, full);
    Reg::MaskReg nonzero;
    Reg::Compares<uint32_t, CMPMODE::NE>(nonzero, buckets, 0U, full);
    Reg::Select(safe, previous, zero, nonzero);
    Reg::Gather(accumulated, histogram, safe, full);
    Reg::Select(accumulated, accumulated, zero, nonzero);
    Reg::ShiftLefts<uint32_t, int16_t>(packed, accumulated, int16_t(INPUT_SELECT_METADATA_HIGH_SHIFT), full);
    Reg::Or(packed, packed, buckets, full);
    Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
        histogram + ::topkV2::INPUT_SELECT_METADATA_KEY_WORD, packed, full);
}

template <typename T, bool Largest>
__aicore__ inline void RunB8(LocalTensor<T> input, LocalTensor<T> values, LocalTensor<uint32_t> indices,
                             LocalTensor<uint8_t> scratch, uint32_t n, uint32_t k, uint32_t rows,
                             uint32_t allocatedRows, uint32_t inputStride, uint32_t valueStride, uint32_t indexStride)
{
    static_assert(std::is_integral_v<T> && sizeof(T) == 1U);
    auto keys = scratch.template ReinterpretCast<uint8_t>();
    if constexpr (std::is_same_v<T, uint8_t> && !Largest) {
        // Unsigned ascending radix keys equal the input bytes: read them in
        // place and avoid an identity copy through an automatic vector scope.
        keys = input.template ReinterpretCast<uint8_t>();
    }
    auto hist = scratch[TopKUbRadixSelectKeyBytes<T>(inputStride, allocatedRows)].template ReinterpretCast<int32_t>();
    auto h = (__ubuf__ uint32_t*)hist.GetPhyAddr();
    for (uint32_t start = 0; start < rows; start += INPUT_SELECT_GROUP_ROWS) {
        uint32_t count = rows - start < INPUT_SELECT_GROUP_ROWS ? rows - start : INPUT_SELECT_GROUP_ROWS;
        if constexpr (!(std::is_same_v<T, uint8_t> && !Largest)) {
            TopKUbRadixSelectBuildKeys<T, Largest>(input[start * inputStride], keys[start * inputStride], n, count,
                                                   inputStride);
        }
        for (uint32_t row = 0; row < count; ++row) {
            RadixSortTopKB8<T, uint8_t, 1, Largest, int32_t> radix;
            radix.GetCumSum(keys[(start + row) * inputStride], hist[row * INPUT_SELECT_HIST_STRIDE], 0U, 0U, 0, n);
        }
        __VEC_SCOPE__
        {
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            for (uint16_t row = 0; row < uint16_t(count); ++row) {
                TopKUbRadixSelectThresholdB8(h + uint32_t(row) * INPUT_SELECT_HIST_STRIDE, k);
            }
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        }
        auto keyPtr = (__ubuf__ uint8_t*)keys.GetPhyAddr();
        auto out = (__ubuf__ uint8_t*)values.GetPhyAddr();
        auto idx = (__ubuf__ uint32_t*)indices.GetPhyAddr();
        TopKUbRadixSelectDispatchGroupRows<T, Largest>(keyPtr, (__ubuf__ T*)input.GetPhyAddr(), (__ubuf__ T*)out, idx,
                                                       h, n, k, start, count, 0U, inputStride, valueStride,
                                                       indexStride);
    }
}

} // namespace topkV2UbRadixSelect
namespace topkV2 {
template <typename T, bool Largest>
__aicore__ inline void RunTopKSmallKUbRadixSelect(LocalTensor<T> input, LocalTensor<T> values,
                                                  LocalTensor<uint32_t> indices, LocalTensor<uint8_t> scratch,
                                                  uint32_t n, uint32_t k, uint32_t rows, uint32_t allocatedRows,
                                                  uint32_t inputStride, uint32_t valueStride, uint32_t indexStride)
{
    if constexpr (sizeof(T) == 1U) {
        topkV2UbRadixSelect::RunB8<T, Largest>(input, values, indices, scratch, n, k, rows, allocatedRows, inputStride,
                                               valueStride, indexStride);
    } else if constexpr (sizeof(T) == 2U || sizeof(T) == 4U) {
        topkV2UbRadixSelect::RunB16B32<T, Largest>(input, values, indices, scratch, n, k, rows, allocatedRows,
                                                   inputStride, valueStride, indexStride);
    } else {
        RunTopKUbRadixSelectB64<T, Largest>(input, values, indices, scratch, n, k, rows, allocatedRows, inputStride,
                                            valueStride, indexStride);
    }
}
} // namespace topkV2
#endif
