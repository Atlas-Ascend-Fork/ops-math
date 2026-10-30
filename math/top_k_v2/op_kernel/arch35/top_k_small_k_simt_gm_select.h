/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TOP_K_SMALL_K_SIMT_GM_SELECT_H
#define TOP_K_SMALL_K_SIMT_GM_SELECT_H
#include <climits>
#include <cstdint>
#include <type_traits>
#include "top_k_small_bitonic_packed_simt.h"
#include "top_k_constant_var_simd.h"
namespace topkV2 {
// Small-k (1 <= k <= 32) SIMT selection reads GM rows and writes final values/indices.
// Host dtype/layout/row admission limits the axis length to at most 8192.
// Scalar, warp and threshold specializations support contiguous and strided rows;
// this is separate from the UB-input radix selection used by SingleBlock.
constexpr uint32_t TOPK_DIRECT_CONTIGUOUS_CHUNKS = 9, TOPK_DIRECT_SMALL_RANK_CHUNKS = 16;
constexpr uint32_t TOPK_DIRECT_SMALL_RANK_MIN_K = 2;
constexpr uint32_t TOPK_DIRECT_DEFAULT_RANK_MAX_K = 4, TOPK_DIRECT_FP32_RANK_CHUNKS = 2;
static_assert(TOPK_DIRECT_WARP_SIZE == BITONIC_SMALL_TOPK_SIZE);
// Common launch arguments are built once by RunTopKSmallKSimtGmSelect.
struct TopKSimtGmLaunch {
    GM_ADDR x;
    GM_ADDR values;
    GM_ADDR indices;
    uint64_t begin, end, inner;
    uint32_t n, k;
    bool bitonic;
};
template <auto Function, typename T, typename I, typename CountT = uint64_t, typename... Args>
__aicore__ inline void CallTopKSimtGm(const TopKSimtGmLaunch& call, dim3 threads, Args... args)
{
    asc_vf_call<Function>(threads, (__gm__ T*)call.x, (__gm__ T*)call.values, (__gm__ I*)call.indices,
                          static_cast<CountT>(call.begin), static_cast<CountT>(call.end), call.inner, args...);
}
template <typename T>
using TopKSimtGmKey = std::conditional_t<sizeof(T) == sizeof(uint64_t), uint64_t, uint32_t>;
template <typename T>
using TopKSimtGmReg = std::conditional_t<IsBitonicFloatType<T>, float, BitonicSmallRegType<T>>;
template <typename T>
using TopKSimtGmRaw = std::conditional_t<
    sizeof(T) == sizeof(uint8_t), uint8_t,
    std::conditional_t<sizeof(T) == sizeof(uint16_t), uint16_t,
                       std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t, uint64_t>>>;
// Selection keeps signed zeros distinct; all NaNs share the largest radix key.
template <typename T>
__simt_callee__ inline TopKSimtGmKey<T> BuildTopKSimtGmSelectionKey(TopKSimtGmRaw<T> raw)
{
    using Key = TopKSimtGmKey<T>;
    if constexpr (IsBitonicFloatType<T>) {
        constexpr Key SIGN = Key(1) << (sizeof(T) * CHAR_BIT - 1U);
        constexpr Key ALL = sizeof(T) == sizeof(uint16_t) ? Key(0xffffU) : Key(UINT32_MAX);
        constexpr Key INF = std::is_same_v<T, half> ? Key(0x7c00U) :
                                                      (std::is_same_v<T, bfloat16_t> ? Key(0x7f80U) : Key(0x7f800000U));
        Key bits = static_cast<Key>(raw);
        if ((bits & (SIGN - 1U)) > INF) {
            return ALL;
        }
        return bits ^ ((bits & SIGN) != 0U ? ALL : SIGN);
    } else {
        Key key = static_cast<Key>(raw);
        if constexpr (std::is_signed_v<T>) {
            key ^= Key(1) << (sizeof(T) * CHAR_BIT - 1U);
        }
        return key;
    }
}
template <typename T>
__simt_callee__ inline TopKSimtGmKey<T> TopKSimtGmLoadKey(__gm__ const T* input)
{
    return BuildTopKSimtGmSelectionKey<T>(*reinterpret_cast<__gm__ const TopKSimtGmRaw<T>*>(input));
}
template <typename CountT>
struct TopKSimtGmRowRange {
    CountT begin, end;
};
template <uint32_t Alignment = 1U, typename CountT>
__aicore__ inline TopKSimtGmRowRange<CountT> PartitionTopKSimtGmRows(CountT rows)
{
    static_assert(Alignment != 0U && (Alignment & (Alignment - 1U)) == 0U);
    CountT perCore = (rows + GetBlockNum() - 1U) / GetBlockNum();
    perCore = (perCore + Alignment - 1U) & ~CountT(Alignment - 1U);
    CountT begin = GetBlockIdx() * perCore;
    CountT end = begin + perCore < rows ? begin + perCore : rows;
    return {begin, end};
}
template <typename T, typename I>
__simt_callee__ inline void StoreTopKSimtGm(__gm__ const T* input, __gm__ T* output, __gm__ I* indices, uint64_t src,
                                            uint64_t dst, uint32_t sourceIndex, TopKSimtGmKey<T> key)
{
    if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t>) {
        output[dst] = static_cast<T>(key);
    } else {
        reinterpret_cast<__gm__ TopKSimtGmRaw<T>*>(output)[dst] = reinterpret_cast<__gm__ const TopKSimtGmRaw<T>*>(
            input)[src];
    }
    indices[dst] = static_cast<I>(sourceIndex);
}
struct TopKSimtGmRowAddress {
    uint64_t src, dst, stride;
};
template <bool Contiguous = false>
__simt_callee__ inline TopKSimtGmRowAddress GetTopKSimtGmRowAddress(uint64_t row, uint64_t inner, uint32_t n,
                                                                    uint32_t k)
{
    uint64_t stride = Contiguous ? 1U : inner, outer = Contiguous ? row : row / inner,
             offset = Contiguous ? 0U : row - outer * inner;
    return {outer * n * stride + offset, outer * k * stride + offset, stride};
}
template <typename Key, bool Maximum>
__simt_callee__ inline Key ReduceTopKSimtGmKey(Key value)
{
    if constexpr (sizeof(Key) == sizeof(uint32_t)) {
        return Maximum ? asc_reduce_max(value) : asc_reduce_min(value);
    } else {
#pragma unroll
        for (uint32_t shift = TOPK_DIRECT_WARP_SIZE / 2U; shift > 0U; shift /= 2U) {
            Key peer = asc_shfl_xor(value, shift, TOPK_DIRECT_WARP_SIZE);
            value = (Maximum ? peer > value : peer < value) ? peer : value;
        }
        return value;
    }
}
template <typename T>
__simt_callee__ inline TopKSimtGmKey<T> TopKSimtGmNumericalKey(TopKSimtGmKey<T> key)
{
    if constexpr (IsBitonicFloatType<T>) {
        constexpr auto sign = TopKSimtGmKey<T>(1) << (sizeof(T) * CHAR_BIT - 1U);
        if (key == sign - 1U) {
            key = sign;
        }
    }
    return key;
}
template <typename Key>
struct TopKSimtGmKeyIndex {
    Key key;
    uint32_t index;
    friend __simt_callee__ inline TopKSimtGmKeyIndex asc_shfl_xor(TopKSimtGmKeyIndex item, int32_t mask, int32_t width)
    {
        return {asc_shfl_xor(item.key, mask, width), asc_shfl_xor(item.index, mask, width)};
    }
};
template <typename Key>
__simt_callee__ inline bool operator>(TopKSimtGmKeyIndex<Key> a, TopKSimtGmKeyIndex<Key> b)
{
    return a.key > b.key || (a.key == b.key && a.index > b.index);
}
template <typename Key>
__simt_callee__ inline bool operator<(TopKSimtGmKeyIndex<Key> a, TopKSimtGmKeyIndex<Key> b)
{
    return b > a;
}
template <typename T>
using TopKSimtGmCandidate = std::conditional_t<(sizeof(T) <= sizeof(uint16_t)), uint32_t,
                                               TopKSimtGmKeyIndex<TopKSimtGmKey<T>>>;
template <typename T>
__simt_callee__ inline TopKSimtGmKeyIndex<TopKSimtGmKey<T>> TopKSimtGmCandidateInfo(TopKSimtGmCandidate<T> item)
{
    if constexpr (sizeof(T) <= sizeof(uint16_t)) {
        return {item >> TOPK_DIRECT_INDEX_BITS, item == UINT32_MAX ? UINT32_MAX : item & (TOPK_DIRECT_WARP_SIZE - 1U)};
    } else {
        return item;
    }
}
template <typename T>
__simt_callee__ inline TopKSimtGmCandidate<T> TopKSimtGmMakeCandidate(TopKSimtGmKey<T> key, uint32_t index)
{
    if constexpr (sizeof(T) <= sizeof(uint16_t)) {
        return index == UINT32_MAX ? UINT32_MAX : (key << TOPK_DIRECT_INDEX_BITS) | index;
    } else {
        return {key, index};
    }
}
// Exchange plans preserve CUDA's tie order; source indices remain payload.
template <typename T, uint32_t Count, uint32_t Limit, bool Largest, bool AllValid = false, bool Stable = false,
          uint32_t Step = 0U>
__simt_callee__ inline void TopKSimtGmSortCandidates(TopKSimtGmCandidate<T> (&items)[Count])
{
    if constexpr (Step < TOPK_SCALAR_SELECT_NETWORK<Count, Limit>.count) {
        constexpr auto e = TOPK_SCALAR_SELECT_NETWORK<Count, Limit>.steps[Step];
        auto a = items[e.left], b = items[e.right];
        auto ai = TopKSimtGmCandidateInfo<T>(a), bi = TopKSimtGmCandidateInfo<T>(b);
        bool better = Largest ? ai.key > bi.key : ai.key < bi.key;
        better = better || (Stable && ai.key == bi.key && ai.index < bi.index);
        bool comp = AllValid ? better : ((better && ai.index != UINT32_MAX) || bi.index == UINT32_MAX);
        if (comp == e.greater) {
            items[e.left] = b;
            items[e.right] = a;
        }
        TopKSimtGmSortCandidates<T, Count, Limit, Largest, AllValid, Stable, Step + 1U>(items);
    }
}
// Preserve the exchange circuit while avoiding a recursive template chain for static shapes.
template <typename T, uint32_t Count, uint32_t Limit, bool Largest, bool AllValid = false, bool Stable = false>
__simt_callee__ inline void TopKSimtGmSortCandidatesFixedLoop(TopKSimtGmCandidate<T> (&items)[Count])
{
    constexpr uint32_t count = TOPK_SCALAR_SELECT_NETWORK<Count, Limit>.count;
#pragma unroll
    for (uint32_t step = 0; step < count; ++step) {
        const auto e = TOPK_SCALAR_SELECT_NETWORK<Count, Limit>.steps[step];
        auto a = items[e.left], b = items[e.right];
        auto ai = TopKSimtGmCandidateInfo<T>(a), bi = TopKSimtGmCandidateInfo<T>(b);
        bool better = Largest ? ai.key > bi.key : ai.key < bi.key;
        better = better || (Stable && ai.key == bi.key && ai.index < bi.index);
        bool comp = AllValid ? better : ((better && ai.index != UINT32_MAX) || bi.index == UINT32_MAX);
        if (comp == e.greater) {
            items[e.left] = b;
            items[e.right] = a;
        }
    }
}
template <typename T, uint32_t COLS, uint32_t RANKS>
__simt_callee__ inline void LoadTopKSimtGmScalarThresholds(__gm__ const T* input, const TopKSimtGmRowAddress& address,
                                                           uint32_t n, uint32_t ranks, bool rankLargest,
                                                           TopKSimtGmKey<T> (&keys)[COLS],
                                                           TopKSimtGmKey<T> (&threshold)[RANKS])
{
    using Key = TopKSimtGmKey<T>;
#pragma unroll
    for (uint32_t p = 0; p < RANKS; ++p) {
        threshold[p] = rankLargest ? Key(0) : ~Key(0);
    }
#pragma unroll
    for (uint32_t j = 0; j < COLS; ++j) {
        keys[j] = j < n ? TopKSimtGmLoadKey(input + address.src + j * address.stride) : Key(0);
        Key carry = keys[j];
#pragma unroll
        for (uint32_t p = 0; p < RANKS; ++p) {
            if (j < n && p < ranks) {
                Key old = threshold[p];
                bool better = rankLargest ? carry > old : carry < old;
                threshold[p] = better ? carry : old;
                carry = better ? old : carry;
            }
        }
    }
}

template <typename T, bool Largest, uint32_t COLS, uint32_t OUT>
__simt_callee__ inline void CollectTopKSimtGmScalarCandidates(const TopKSimtGmKey<T> (&keys)[COLS],
                                                              TopKSimtGmKey<T> boundary, uint32_t n, uint32_t k,
                                                              bool bitonic, TopKSimtGmCandidate<T> (&items)[OUT])
{
    uint32_t strictCount = 0U, strictPrefix = 0U, equalPrefix = 0U;
#pragma unroll
    for (uint32_t j = 0; j < COLS; ++j) {
        strictCount += j < n && (Largest ? keys[j] > boundary : keys[j] < boundary) ? 1U : 0U;
    }
#pragma unroll
    for (uint32_t j = 0; j < COLS; ++j) {
        bool strict = j < n && (Largest ? keys[j] > boundary : keys[j] < boundary),
             equal = j < n && keys[j] == boundary;
        uint32_t rank = strict ? strictPrefix : (equal ? strictCount + equalPrefix : COLS);
        auto item = TopKSimtGmMakeCandidate<T>(bitonic ? TopKSimtGmNumericalKey<T>(keys[j]) : keys[j], j);
#pragma unroll
        for (uint32_t p = 0; p < OUT; ++p) {
            if (p < k && rank == p) {
                items[p] = item;
            }
        }
        strictPrefix += strict ? 1U : 0U;
        equalPrefix += equal ? 1U : 0U;
    }
}

template <typename T, typename I, uint32_t OUT>
__simt_callee__ inline void StoreTopKSimtGmScalarCandidates(__gm__ const T* input, __gm__ T* output, __gm__ I* indices,
                                                            const TopKSimtGmRowAddress& address, uint32_t k,
                                                            bool bitonic, const TopKSimtGmCandidate<T> (&items)[OUT])
{
#pragma unroll
    for (uint32_t p = 0; p < OUT; ++p) {
        if (p < k) {
            auto item = items[bitonic ? TOPK_SCALAR_SELECT_NETWORK<OUT, TOPK_DIRECT_WARP_SIZE>.order[p] : p];
            uint32_t source = TopKSimtGmCandidateInfo<T>(item).index;
            uint64_t dst = address.dst + p * address.stride;
            StoreTopKSimtGm<T, I>(input, output, indices, address.src + source * address.stride, dst, source,
                                  TopKSimtGmCandidateInfo<T>(item).key);
        }
    }
}

// Independent scalar rows share rank selection and final sorting across static/dynamic N/K.
template <typename T, typename I, bool Largest, uint32_t Width, bool Contiguous, bool Full = false, uint32_t N = 0U,
          uint32_t K = 0U, typename CountT = uint64_t>
__simt_vf__ __aicore__ LAUNCH_BOUND((TOPK_SCALAR_ROW_THREADS<Width, N>)) void TopKSimtGmScalarRows(
    __gm__ T* input, __gm__ T* output, __gm__ I* indices, CountT begin, CountT end, uint64_t inner, uint32_t runtimeN,
    uint32_t runtimeK, bool runtimeBitonic)
{
    static_assert((N == 0U) == (K == 0U), "Static N and K must be specified together");
    static_assert(N == 0U || (K <= N && K <= TOPK_DIRECT_WARP_SIZE), "Invalid static small-k shape");
    constexpr bool hasStaticShape = N != 0U;
    constexpr uint32_t inputCapacity = hasStaticShape ? N : Width;
    constexpr uint32_t candidateCapacity = hasStaticShape ? K : Width;
    constexpr uint32_t staticRanks = K < N - K + 1U ? K : N - K + 1U;
    constexpr uint32_t shapeRanks = hasStaticShape ? staticRanks : Width;
    constexpr uint32_t selectionRankCount = Full ? 1U : shapeRanks;
    constexpr uint32_t THREADS = TOPK_SCALAR_ROW_THREADS<Width, N>;
    using Key = TopKSimtGmKey<T>;
    const uint32_t n = hasStaticShape ? N : runtimeN, k = hasStaticShape ? K : runtimeK;
    const uint32_t ranks = Full ? 1U : (k < n - k + 1U ? k : n - k + 1U);
    const bool rankLargest = k <= n - k + 1U ? Largest : !Largest;
    // Static N/K is a Bitonic-only entry contract, not just a fixed buffer shape.
    const bool bitonic = hasStaticShape || runtimeBitonic;
    for (CountT row = begin + static_cast<uint32_t>(threadIdx.x); row < end; row += THREADS) {
        auto address = GetTopKSimtGmRowAddress<Contiguous>(row, inner, n, k);
        Key keys[inputCapacity], threshold[selectionRankCount];
        TopKSimtGmCandidate<T> items[candidateCapacity];
        LoadTopKSimtGmScalarThresholds<T, inputCapacity, selectionRankCount>(input, address, n, ranks, rankLargest,
                                                                             keys, threshold);
#pragma unroll
        for (uint32_t p = 0; p < candidateCapacity; ++p) {
            items[p] = TopKSimtGmMakeCandidate<T>(0, UINT32_MAX);
        }
        if (Full && !bitonic) {
#pragma unroll
            for (uint32_t p = 0; p < candidateCapacity; ++p) {
                items[p] = TopKSimtGmMakeCandidate<T>(TopKSimtGmNumericalKey<T>(keys[p]), p < n ? p : UINT32_MAX);
            }
        } else {
            CollectTopKSimtGmScalarCandidates<T, Largest, inputCapacity, candidateCapacity>(keys, threshold[ranks - 1U],
                                                                                            n, k, bitonic, items);
        }
        if (bitonic) {
            if constexpr (hasStaticShape && K > 2U) {
                TopKSimtGmSortCandidatesFixedLoop<T, candidateCapacity, TOPK_DIRECT_WARP_SIZE, Largest, true>(items);
            } else {
                TopKSimtGmSortCandidates<T, candidateCapacity, TOPK_DIRECT_WARP_SIZE, Largest, hasStaticShape>(items);
            }
        } else {
            TopKSimtGmSortCandidates<T, candidateCapacity, candidateCapacity, Largest, false, !Full>(items);
        }

        StoreTopKSimtGmScalarCandidates<T, I, candidateCapacity>(input, output, indices, address, k, bitonic, items);
    }
}
template <typename T, uint32_t N, uint32_t K, bool Largest = true, typename I = int32_t, typename CountT = uint32_t>
__aicore__ inline void RunTopKSimtGmScalarSelect(GM_ADDR x, GM_ADDR values, GM_ADDR indices, CountT rows)
{
    InitSocState();
    auto range = PartitionTopKSimtGmRows<TOPK_SCALAR_ROW_ALIGNMENT>(rows);
    TopKSimtGmLaunch call{x, values, indices, range.begin, range.end, 1U, N, K, true};
    CallTopKSimtGm<TopKSimtGmScalarRows<T, I, Largest, N, true, false, N, K, CountT>, T, I, CountT>(
        call, dim3(TOPK_SCALAR_THREADS), N, K, true);
}
template <typename T, typename I, bool Largest, uint32_t N, uint32_t Low = 1U, uint32_t High = N - 1U>
__aicore__ inline void DispatchTopKSimtGmScalarK(GM_ADDR x, GM_ADDR values, GM_ADDR indices, uint32_t rows, uint32_t k)
{
    if constexpr (Low == High) {
        if constexpr (IsTopKScalarSelectShape(N, Low)) {
            RunTopKSimtGmScalarSelect<T, N, Low, Largest, I>(x, values, indices, rows);
        }
    } else if (k <= (Low + High) / 2U) {
        DispatchTopKSimtGmScalarK<T, I, Largest, N, Low, (Low + High) / 2U>(x, values, indices, rows, k);
    } else {
        DispatchTopKSimtGmScalarK<T, I, Largest, N, (Low + High) / 2U + 1U, High>(x, values, indices, rows, k);
    }
}
template <typename T, typename I, bool Largest, uint32_t Low = TOPK_SCALAR_MIN_N, uint32_t High = TOPK_SCALAR_MAX_N>
__aicore__ inline void DispatchTopKSimtGmScalarSelect(GM_ADDR x, GM_ADDR values, GM_ADDR indices, uint32_t rows,
                                                      uint32_t n, uint32_t k)
{
    if constexpr (Low == High) {
        DispatchTopKSimtGmScalarK<T, I, Largest, Low>(x, values, indices, rows, k);
    } else if (n <= (Low + High) / 2U) {
        DispatchTopKSimtGmScalarSelect<T, I, Largest, Low, (Low + High) / 2U>(x, values, indices, rows, n, k);
    } else {
        DispatchTopKSimtGmScalarSelect<T, I, Largest, (Low + High) / 2U + 1U, High>(x, values, indices, rows, n, k);
    }
}
template <typename T, typename I, bool Largest>
__aicore__ inline bool TryRunTopKSimtGmScalarSelect(GM_ADDR x, GM_ADDR values, GM_ADDR indices,
                                                    const TopKV2TilingDataSimd& tiling)
{
    if constexpr (sizeof(T) <= sizeof(uint16_t)) {
        if (CanUseTopKScalarSelect(true, tiling.unsortedDimNum, tiling.lastAxisNum, tiling.topKRealValue,
                                   tiling.isSort != 0, tiling.sortPolicy, tiling.oneCoreRowNum == 1U,
                                   tiling.platformCoreNum)) {
            DispatchTopKSimtGmScalarSelect<T, I, Largest>(
                x, values, indices, static_cast<uint32_t>(tiling.unsortedDimNum),
                static_cast<uint32_t>(tiling.lastAxisNum), static_cast<uint32_t>(tiling.topKRealValue));
            return true;
        }
    }
    return false;
}
template <typename T, uint32_t Width, uint32_t Extra, bool Largest>
__simt_callee__ inline void TopKSimtGmNarrowBitonicFinalize(__gm__ T* input, uint32_t& index, uint64_t inner)
{
    auto key = index == UINT32_MAX ? TopKSimtGmKey<T>(0) :
                                     TopKSimtGmNumericalKey<T>(TopKSimtGmLoadKey(input + index * inner));
    BitonicSmallPackedBitonicNetwork<TopKSimtGmKey<T>, uint32_t, true, Width, Extra>(key, index, Largest);
}
template <typename T, typename I, bool CompareValue>
__simt_callee__ inline void TopKSimtGmTruncatedBitonicNetwork(T& value, I& index, bool isLargest, uint32_t width)
{
    for (uint32_t size = 2U; size <= BITONIC_SMALL_TOPK_SIZE; size *= 2U) {
        for (uint32_t stride = size / 2U; stride > 0U; stride /= 2U) {
            if (stride < width) {
                BitonicSmallSwapStage<T, I, CompareValue>(value, index, stride, size, isLargest);
            }
        }
    }
}
template <typename T, uint32_t Chunks>
__simt_callee__ inline TopKSimtGmKey<T> TopKSimtGmThresholdKey(const TopKSimtGmKey<T>* keys, uint32_t chunk,
                                                               uint32_t lane, uint32_t n, __gm__ const T* input,
                                                               uint64_t inner)
{
    if constexpr (Chunks > 0) {
        return keys[chunk];
    } else {
        uint32_t col = chunk * TOPK_DIRECT_WARP_SIZE + lane;
        return col < n ? TopKSimtGmLoadKey(input + col * inner) : ~TopKSimtGmKey<T>(0);
    }
}
template <typename Key, uint32_t Chunks, bool Largest>
__simt_callee__ inline Key TopKSimtGmSmallRankThreshold(const Key (&keys)[Chunks], uint32_t n, uint32_t k,
                                                        Key threshold)
{
    uint32_t lane = threadIdx.x, selected = 0U;
    while (selected < k) {
        uint32_t same = 0U;
        Key next = Largest ? Key(0) : ~Key(0);
#pragma unroll
        for (uint32_t c = 0; c < Chunks; ++c) {
            Key key = keys[c];
            bool valid = c * TOPK_DIRECT_WARP_SIZE + lane < n;
            same += valid && key == threshold ? 1U : 0U;
            if (valid && (Largest ? key < threshold : key > threshold)) {
                next = Largest ? (key > next ? key : next) : (key < next ? key : next);
            }
        }
        selected += asc_reduce_add(same);
        if (selected < k) {
            threshold = ReduceTopKSimtGmKey<Key, Largest>(next);
        }
    }
    return threshold;
}
template <typename T, uint32_t Chunks>
__simt_callee__ inline void LoadTopKSimtGmThresholdKeys(__gm__ const T* rowInput, uint64_t inner, uint32_t n,
                                                        uint32_t lane, uint32_t chunkCount, bool active,
                                                        TopKSimtGmKey<T> (&keys)[Chunks == 0U ? 1U : Chunks],
                                                        TopKSimtGmKey<T>& low, TopKSimtGmKey<T>& high)
{
    using Key = TopKSimtGmKey<T>;
#pragma unroll
    for (uint32_t c = 0; c < chunkCount; ++c) {
        uint32_t col = c * TOPK_DIRECT_WARP_SIZE + lane;
        Key key = active && col < n ? TopKSimtGmLoadKey(rowInput + col * inner) : ~Key(0);
        if constexpr (Chunks > 0) {
            keys[c] = key;
        }
        if (active && col < n) {
            low = key < low ? key : low;
            high = key > high ? key : high;
        }
    }
    low = ReduceTopKSimtGmKey<Key, false>(low);
    high = ReduceTopKSimtGmKey<Key, true>(high);
}

template <typename T, bool Largest, uint32_t Chunks>
__simt_callee__ inline TopKSimtGmKey<T> SearchTopKSimtGmThresholdBoundary(
    const TopKSimtGmKey<T> (&keys)[Chunks == 0U ? 1U : Chunks], __gm__ const T* rowInput, uint64_t inner, uint32_t n,
    uint32_t k, uint32_t lane, uint32_t chunkCount, bool active, uint32_t kth, TopKSimtGmKey<T> low,
    TopKSimtGmKey<T> high)
{
    using Key = TopKSimtGmKey<T>;
    uint32_t previousCount = UINT32_MAX;
    while (low < high) {
        Key mid = low + (high - low) / 2;
        uint32_t count = 0;
#pragma unroll
        for (uint32_t c = 0; c < chunkCount; ++c) {
            Key key = TopKSimtGmThresholdKey<T, Chunks>(keys, c, lane, active ? n : 0, rowInput, inner);
            count += c * TOPK_DIRECT_WARP_SIZE + lane < n && key <= mid ? 1U : 0U;
        }
        count = asc_reduce_add(count);
        bool exact = count == (Largest ? n - k : k), lower = exact ? !Largest : count > kth;
        Key boundary = lower ? mid : mid + 1U;
        if (exact || count == previousCount) {
            boundary = lower ? Key(0) : ~Key(0);
#pragma unroll
            for (uint32_t c = 0; c < chunkCount; ++c) {
                Key key = TopKSimtGmThresholdKey<T, Chunks>(keys, c, lane, active ? n : 0, rowInput, inner);
                if (c * TOPK_DIRECT_WARP_SIZE + lane < n) {
                    if (lower && key <= mid) {
                        boundary = key > boundary ? key : boundary;
                    }
                    if (!lower && key > mid) {
                        boundary = key < boundary ? key : boundary;
                    }
                }
            }
            boundary = lower ? ReduceTopKSimtGmKey<Key, true>(boundary) : ReduceTopKSimtGmKey<Key, false>(boundary);
        }
        if (exact) {
            low = boundary;
            break;
        }
        if (lower) {
            high = boundary;
        } else {
            low = boundary;
        }
        previousCount = count;
    }
    return low;
}

template <typename T, bool Largest, uint32_t Chunks>
__simt_callee__ inline uint32_t CollectTopKSimtGmThresholdCandidates(
    const TopKSimtGmKey<T> (&keys)[Chunks == 0U ? 1U : Chunks], __gm__ const T* rowInput, uint64_t inner, uint32_t n,
    uint32_t k, uint32_t lane, uint32_t chunkCount, bool active, TopKSimtGmKey<T> low)
{
    using Key = TopKSimtGmKey<T>;
    uint32_t selectedIndex = UINT32_MAX, gathered = 0;
    for (uint32_t pass = 0; pass < 2; ++pass) {
#pragma unroll
        for (uint32_t c = 0; c < chunkCount && gathered < k; ++c) {
            uint32_t col = c * TOPK_DIRECT_WARP_SIZE + lane;
            Key key = TopKSimtGmThresholdKey<T, Chunks>(keys, c, lane, active ? n : 0, rowInput, inner);
            bool chosen = active && col < n && (pass == 0 ? (Largest ? key > low : key < low) : key == low);
            uint32_t mask = asc_ballot(chosen), count = __popc(mask);
            if (lane >= gathered && lane < gathered + count && lane < k) {
                selectedIndex = c * TOPK_DIRECT_WARP_SIZE + BitonicSmallNthSetBit(mask, lane - gathered);
            }
            gathered += count;
        }
    }
    return selectedIndex;
}

template <typename T, typename I, bool Largest, bool Contiguous>
__simt_callee__ inline void FinalizeTopKSimtGmThresholdSelection(__gm__ T* input, __gm__ T* output, __gm__ I* indices,
                                                                 const TopKSimtGmRowAddress& address, uint64_t inner,
                                                                 uint32_t lane, uint32_t k, bool active,
                                                                 uint32_t finalizeWidth, uint32_t selectedIndex)
{
    using RegT = TopKSimtGmReg<T>;
    const uint64_t src = address.src;
    bool valid = active && lane < k;
    RegT selectedValue = RegT(0);
    if constexpr (sizeof(T) <= sizeof(uint16_t)) {
        TopKSimtGmNarrowBitonicFinalize<T, TOPK_DIRECT_WARP_SIZE, 0, Largest>(input + src, selectedIndex, inner);
    } else {
        selectedValue = valid ? static_cast<RegT>(input[src + selectedIndex * inner]) : RegT(0);
        TopKSimtGmTruncatedBitonicNetwork<RegT, uint32_t, true>(selectedValue, selectedIndex, Largest, finalizeWidth);
    }
    if (valid) {
        uint64_t dst = address.dst + lane * address.stride;
        if constexpr (Contiguous && sizeof(T) >= sizeof(uint32_t)) {
            output[dst] = static_cast<T>(selectedValue);
        } else {
            output[dst] = input[src + selectedIndex * inner];
        }
        indices[dst] = static_cast<I>(selectedIndex);
    }
}

template <typename T, typename I, bool Largest, uint32_t Chunks, bool UseSmallRank = false, bool Contiguous = false>
__simt_vf__ __aicore__ LAUNCH_BOUND(TOPK_DIRECT_THREADS) void TopKThresholdSimtGm(__gm__ T* input, __gm__ T* output,
                                                                                  __gm__ I* indices, uint64_t begin,
                                                                                  uint64_t end, uint64_t runtimeInner,
                                                                                  uint32_t n, uint32_t k)
{
    const uint64_t inner = Contiguous ? 1U : runtimeInner;
    using Key = TopKSimtGmKey<T>;
    uint32_t lane = threadIdx.x;
    constexpr uint32_t CACHE_SIZE = Chunks == 0 ? 1 : Chunks;
    uint32_t chunkCount = Chunks == 0 ? (n + TOPK_DIRECT_WARP_SIZE - 1U) / TOPK_DIRECT_WARP_SIZE : Chunks;
    uint32_t finalizeWidth = 2U;
    while (finalizeWidth < k && finalizeWidth < BITONIC_SMALL_TOPK_SIZE) {
        finalizeWidth <<= 1U;
    }
    for (uint64_t tile = begin; tile < end; tile += TOPK_DIRECT_THREADS / TOPK_DIRECT_WARP_SIZE) {
        uint64_t row = tile + threadIdx.y;
        bool active = row < end;
        auto address = GetTopKSimtGmRowAddress<Contiguous>(row, inner, n, k);
        uint64_t src = address.src;
        Key keys[CACHE_SIZE];
        Key low = ~Key(0), high = 0;
        LoadTopKSimtGmThresholdKeys<T, Chunks>(input + src, inner, n, lane, chunkCount, active, keys, low, high);
        uint32_t kth = Largest ? n - k : k - 1;
        if (k == n) {
            low = Largest ? low : high;
            high = low;
        }
        constexpr bool FP32_RANK = Chunks == TOPK_DIRECT_FP32_RANK_CHUNKS && std::is_same_v<T, float> && !UseSmallRank;
        if constexpr (FP32_RANK || (UseSmallRank && Chunks > 0U && Chunks <= TOPK_DIRECT_SMALL_RANK_CHUNKS)) {
            if (k >= (FP32_RANK ? 1U : TOPK_DIRECT_SMALL_RANK_MIN_K) && k <= TOPK_DIRECT_SMALL_RANK_MAX_K &&
                (FP32_RANK || inner == 1U) && active && low < high) {
                low = TopKSimtGmSmallRankThreshold<Key, Chunks, Largest>(keys, n, k, Largest ? high : low);
                high = low;
            }
        }
        low = SearchTopKSimtGmThresholdBoundary<T, Largest, Chunks>(keys, input + src, inner, n, k, lane, chunkCount,
                                                                    active, kth, low, high);
        uint32_t selectedIndex = CollectTopKSimtGmThresholdCandidates<T, Largest, Chunks>(
            keys, input + src, inner, n, k, lane, chunkCount, active, low);
        FinalizeTopKSimtGmThresholdSelection<T, I, Largest, Contiguous>(input, output, indices, address, inner, lane, k,
                                                                        active, finalizeWidth, selectedIndex);
    }
}
template <typename T, typename I, bool Largest, uint32_t Chunks>
__aicore__ inline void LaunchTopKThresholdSimtGm(const TopKSimtGmLaunch& call)
{
    constexpr bool CONTIGUOUS = Chunks > 0 && Chunks <= TOPK_DIRECT_CONTIGUOUS_CHUNKS;
    constexpr uint32_t RANK_LIMIT = ((Chunks == TOPK_DIRECT_FP32_RANK_CHUNKS && std::is_same_v<T, float>) ||
                                     (sizeof(TopKSimtGmKey<T>) == sizeof(uint64_t) &&
                                      Chunks >= TOPK_DIRECT_FP32_RANK_CHUNKS && CONTIGUOUS)) ?
                                        TOPK_DIRECT_SMALL_RANK_MAX_K :
                                        TOPK_DIRECT_DEFAULT_RANK_MAX_K;
    if constexpr (Chunks > 0 && Chunks <= TOPK_DIRECT_SMALL_RANK_CHUNKS) {
        if (call.inner == 1U) {
            if (call.k >= TOPK_DIRECT_SMALL_RANK_MIN_K && call.k <= RANK_LIMIT) {
                CallTopKSimtGm<TopKThresholdSimtGm<T, I, Largest, Chunks, true, CONTIGUOUS>, T, I>(
                    call, dim3(TOPK_DIRECT_WARP_SIZE, TOPK_DIRECT_THREADS / TOPK_DIRECT_WARP_SIZE), call.n, call.k);
                return;
            }
            if constexpr (CONTIGUOUS) {
                CallTopKSimtGm<TopKThresholdSimtGm<T, I, Largest, Chunks, false, true>, T, I>(
                    call, dim3(TOPK_DIRECT_WARP_SIZE, TOPK_DIRECT_THREADS / TOPK_DIRECT_WARP_SIZE), call.n, call.k);
                return;
            }
        }
    }
    CallTopKSimtGm<TopKThresholdSimtGm<T, I, Largest, Chunks>, T, I>(
        call, dim3(TOPK_DIRECT_WARP_SIZE, TOPK_DIRECT_THREADS / TOPK_DIRECT_WARP_SIZE), call.n, call.k);
}

// Warp rows cache raw inputs; partial and full selection share group/finalize/writeback.
template <typename T, typename I, bool Largest, uint32_t Width, bool Contiguous, bool Full>
__simt_vf__ __aicore__ LAUNCH_BOUND(TOPK_DIRECT_THREADS) void TopKSimtGmWarpRows(__gm__ T* input, __gm__ T* output,
                                                                                 __gm__ I* indices, uint64_t begin,
                                                                                 uint64_t end, uint64_t inner,
                                                                                 uint32_t n, uint32_t k, bool bitonic)
{
    using Raw = TopKSimtGmRaw<T>;
    using RawReg = std::conditional_t<(sizeof(T) <= sizeof(uint32_t)), uint32_t, uint64_t>;
    using Key = TopKSimtGmKey<T>;
    constexpr uint32_t EXTRA = BitonicSmallConstLog2(TOPK_DIRECT_WARP_SIZE) - BitonicSmallConstLog2(Width);
    uint32_t local = threadIdx.x % Width, base = threadIdx.x - local;
    uint32_t slot = threadIdx.y * (TOPK_DIRECT_WARP_SIZE / Width) + threadIdx.x / Width;
    for (uint64_t tile = begin; tile < end; tile += TOPK_DIRECT_THREADS / Width) {
        uint64_t row = tile + slot;
        auto address = GetTopKSimtGmRowAddress<Contiguous>(row, inner, n, k);
        bool sourceValid = row < end && local < n, valid = row < end && local < k;
        RawReg raw = sourceValid ? reinterpret_cast<__gm__ Raw*>(input)[address.src + local * address.stride] :
                                   RawReg(0);
        Key original = BuildTopKSimtGmSelectionKey<T>(static_cast<Raw>(raw));
        uint32_t index = sourceValid ? local : UINT32_MAX;
        if constexpr (!Full) {
            TopKSimtGmKeyIndex<Key> packed{original, Largest ? TOPK_DIRECT_WARP_SIZE - 1U - local : local};
            BitonicSmallPackedBitonicNetwork<TopKSimtGmKeyIndex<Key>, uint32_t, true, Width, 0>(packed, index, Largest);
        }
        index = valid ? index : UINT32_MAX;
        if (bitonic) {
            Key key = asc_shfl(original, base + (valid ? index : 0U), TOPK_DIRECT_WARP_SIZE);
            Key boundary = valid ? key : (Largest ? ~Key(0) : Key(0));
#pragma unroll
            for (uint32_t shift = Width / 2U; shift > 0U; shift /= 2U) {
                Key peer = asc_shfl_xor(boundary, shift, TOPK_DIRECT_WARP_SIZE);
                boundary = (Largest ? peer < boundary : peer > boundary) ? peer : boundary;
            }
            uint32_t group = valid ? index + (key == boundary ? Width : 0U) : UINT32_MAX;
            BitonicSmallPackedBitonicNetwork<uint32_t, uint32_t, true, Width, 0>(group, index, false);
        }
        if (Full || bitonic) {
            Key key = TopKSimtGmNumericalKey<T>(asc_shfl(original, base + (valid ? index : 0U), TOPK_DIRECT_WARP_SIZE));
            if (bitonic) {
                BitonicSmallPackedBitonicNetwork<Key, uint32_t, true, Width, EXTRA>(key, index, Largest);
            } else {
                BitonicSmallPackedBitonicNetwork<Key, uint32_t, true, Width, 0>(key, index, Largest);
            }
        }
        RawReg selected = asc_shfl(raw, base + (valid ? index : 0U), TOPK_DIRECT_WARP_SIZE);
        if (valid) {
            uint64_t dst = address.dst + local * address.stride;
            reinterpret_cast<__gm__ Raw*>(output)[dst] = static_cast<Raw>(selected);
            indices[dst] = static_cast<I>(index);
        }
    }
}
enum class TopKSimtGmWidthKernel { SCALAR, SCALAR_FULL, PACKED_FULL, KEY_SORT };
template <typename T, typename I, bool Largest, TopKSimtGmWidthKernel Kernel, uint32_t Width, bool Contiguous>
__aicore__ inline void LaunchTopKSimtGmWidth(const TopKSimtGmLaunch& call)
{
    if constexpr (Kernel == TopKSimtGmWidthKernel::SCALAR || Kernel == TopKSimtGmWidthKernel::SCALAR_FULL) {
        CallTopKSimtGm<
            TopKSimtGmScalarRows<T, I, Largest, Width, Contiguous, Kernel == TopKSimtGmWidthKernel::SCALAR_FULL>, T, I>(
            call, dim3(TOPK_SCALAR_ROW_THREADS<Width>), call.n, call.k, call.bitonic);
    } else {
        CallTopKSimtGm<
            TopKSimtGmWarpRows<T, I, Largest, Width, Contiguous, Kernel == TopKSimtGmWidthKernel::PACKED_FULL>, T, I>(
            call, dim3(TOPK_DIRECT_WARP_SIZE, TOPK_DIRECT_THREADS / TOPK_DIRECT_WARP_SIZE), call.n, call.k,
            call.bitonic);
    }
}
template <typename T, typename I, bool Largest, TopKSimtGmWidthKernel Kernel, uint32_t MaxWidth,
          uint32_t Width = TOPK_DIRECT_KEY_SORT_MIN_AXIS>
__aicore__ inline void DispatchTopKSimtGmWidth(const TopKSimtGmLaunch& call)
{
    static_assert(Width >= TOPK_DIRECT_PAIR_AXIS && Width <= MaxWidth && MaxWidth <= TOPK_DIRECT_WARP_SIZE &&
                  (Width & (Width - 1U)) == 0U && (MaxWidth & (MaxWidth - 1U)) == 0U);
    if constexpr (Width < MaxWidth) {
        if (call.n > Width) {
            DispatchTopKSimtGmWidth<T, I, Largest, Kernel, MaxWidth, Width * 2U>(call);
            return;
        }
    }
    if (call.inner == 1U && (Kernel != TopKSimtGmWidthKernel::PACKED_FULL || !std::is_same_v<T, half>)) {
        LaunchTopKSimtGmWidth<T, I, Largest, Kernel, Width, true>(call);
    } else {
        LaunchTopKSimtGmWidth<T, I, Largest, Kernel, Width, false>(call);
    }
}

// Each lane reserves Chunks key slots (32*Chunks columns per row). Tiers 9/17
// cover one chunk beyond 256/512 columns without allocating 16/24 slots.
// Contiguous/small-rank dispatch also depends on these tiers; merging them
// requires dtype-specific checks around 256/288 and 512/544 columns.
inline constexpr uint32_t TOPK_DIRECT_THRESHOLD_CHUNKS[] = {1U, 2U, 4U, 8U, 9U, 16U, 17U, 24U, 32U};
template <typename T, typename I, bool Largest, uint32_t Bucket = 0U>
__aicore__ inline void DispatchTopKThresholdSimtGm(const TopKSimtGmLaunch& call)
{
    constexpr uint32_t CHUNKS = TOPK_DIRECT_THRESHOLD_CHUNKS[Bucket],
                       BUCKETS = sizeof(TOPK_DIRECT_THRESHOLD_CHUNKS) / sizeof(TOPK_DIRECT_THRESHOLD_CHUNKS[0]);
    if (call.n <= CHUNKS * TOPK_DIRECT_WARP_SIZE) {
        LaunchTopKThresholdSimtGm<T, I, Largest, CHUNKS>(call);
    } else if constexpr (Bucket + 1U < BUCKETS) {
        DispatchTopKThresholdSimtGm<T, I, Largest, Bucket + 1U>(call);
    } else {
        LaunchTopKThresholdSimtGm<T, I, Largest, 0>(call);
    }
}
template <typename T, typename I, bool Largest>
__aicore__ inline void RunTopKSmallKSimtGmSelect(GM_ADDR x, GM_ADDR values, GM_ADDR indices,
                                                 const TopKV2TilingDataSimd& tiling)
{
    if (TryRunTopKSimtGmScalarSelect<T, I, Largest>(x, values, indices, tiling)) {
        return;
    }
    auto range = PartitionTopKSimtGmRows(tiling.unsortedDimNum);
    TopKSimtGmLaunch call = {x,
                             values,
                             indices,
                             range.begin,
                             range.end,
                             tiling.oneCoreRowNum,
                             static_cast<uint32_t>(tiling.lastAxisNum),
                             static_cast<uint32_t>(tiling.topKRealValue),
                             tiling.sortPolicy == TOPK_DIRECT_BITONIC_POLICY && tiling.isSort != 0};
    uint32_t n = call.n, k = call.k;
    uint64_t inner = call.inner;

    constexpr uint32_t FULL_SCALAR_WIDTH = sizeof(T) <= sizeof(uint16_t) ? TOPK_DIRECT_SCALAR_MAX_AXIS :
                                                                           TOPK_DIRECT_KEY_SORT_MIN_AXIS;
    if (k == n && n > TOPK_DIRECT_PAIR_AXIS && n <= TOPK_DIRECT_WARP_SIZE) {
        if (inner == 1 && n <= FULL_SCALAR_WIDTH) {
            DispatchTopKSimtGmWidth<T, I, Largest, TopKSimtGmWidthKernel::SCALAR_FULL, FULL_SCALAR_WIDTH>(call);
        } else {
            DispatchTopKSimtGmWidth<T, I, Largest, TopKSimtGmWidthKernel::PACKED_FULL, TOPK_DIRECT_WARP_SIZE>(call);
        }
        return;
    }
    if constexpr (IsBitonicFloatType<T> || sizeof(T) == sizeof(uint64_t)) {
        if (n >= TOPK_DIRECT_KEY_SORT_MIN_AXIS && n <= TOPK_DIRECT_KEY_SORT_MAX_AXIS) {
            DispatchTopKSimtGmWidth<T, I, Largest, TopKSimtGmWidthKernel::KEY_SORT, TOPK_DIRECT_KEY_SORT_MAX_AXIS>(
                call);
            return;
        }
    }
    if (n <= TOPK_DIRECT_SCALAR_MAX_AXIS) {
        DispatchTopKSimtGmWidth<T, I, Largest, TopKSimtGmWidthKernel::SCALAR, TOPK_DIRECT_SCALAR_MAX_AXIS,
                                TOPK_DIRECT_PAIR_AXIS>(call);
    } else {
        DispatchTopKThresholdSimtGm<T, I, Largest>(call);
    }
}
} // namespace topkV2
#endif
