/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DENSE_BINCOUNT_REGBASE_H
#define DENSE_BINCOUNT_REGBASE_H

#include "kernel_operator.h"
#include "simt_api/common_functions.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "dense_bincount_tiling_data.h"

namespace NsDenseBincount {
using namespace AscendC;

constexpr uint32_t UB_BLOCK_BYTES = 32U;
constexpr uint32_t PRIVATE_HIST_FULL = 1U;
constexpr uint32_t PRIVATE_HIST_ROW = 2U;
constexpr uint32_t PRIVATE_HIST_TILED = 3U;
constexpr uint32_t DETERMINISTIC_PARTIAL = 1U;
constexpr uint32_t DETERMINISTIC_HISTOGRAM = 2U;
constexpr uint64_t DETERMINISTIC_PARTIAL_BYTES = 64U;
constexpr uint64_t DETERMINISTIC_HISTOGRAM_SLOT_BYTES = 128U;

template <typename T, bool IS_1D>
__simt_callee__ inline bool MapInputToOutput(T value, int64_t index, int64_t inputRows, int64_t inputCols, int64_t size,
                                             int64_t& outputOffset)
{
    int64_t bin = static_cast<int64_t>(value);
    int64_t row = 0;
    if constexpr (!IS_1D) {
        row = index / inputCols;
        if (bin < 0) {
            const int64_t quotient = bin / size;
            const int64_t remainder = bin % size;
            const int64_t adjustment = remainder == 0 ? 0 : 1;
            if (quotient < adjustment - row) {
                return false;
            }
            row += quotient - adjustment;
            bin = remainder == 0 ? 0 : remainder + size;
        }
    }
    if (bin < 0 || bin >= size || row < 0 || row >= inputRows) {
        return false;
    }
    outputOffset = IS_1D ? bin : row * size + bin;
    return true;
}

struct ExactFloatAccumulator {
    uint64_t limb0;
    uint64_t limb1;
    uint64_t limb2;
    uint64_t limb3;
    uint64_t limb4;
    uint64_t limb5;
    bool hasNan;
    bool hasPositiveInf;
    bool hasNegativeInf;
};

__simt_callee__ inline uint64_t AddWithCarry(uint64_t& target, uint64_t value)
{
    const uint64_t previous = target;
    target += value;
    return target < previous ? 1ULL : 0ULL;
}

__simt_callee__ inline uint64_t SubWithBorrow(uint64_t& target, uint64_t value)
{
    const uint64_t previous = target;
    target -= value;
    return previous < value ? 1ULL : 0ULL;
}

__simt_callee__ inline void AddAccumulatorWord(ExactFloatAccumulator& accumulator, uint32_t wordIndex, uint64_t value)
{
    uint64_t carry = 0ULL;
    if (wordIndex == 0U) {
        carry = AddWithCarry(accumulator.limb0, value);
        carry = carry == 0ULL ? 0ULL : AddWithCarry(accumulator.limb1, carry);
        carry = carry == 0ULL ? 0ULL : AddWithCarry(accumulator.limb2, carry);
        carry = carry == 0ULL ? 0ULL : AddWithCarry(accumulator.limb3, carry);
        if (carry != 0ULL) {
            carry = AddWithCarry(accumulator.limb4, carry);
            accumulator.limb5 += carry;
        }
    } else if (wordIndex == 1U) {
        carry = AddWithCarry(accumulator.limb1, value);
        carry = carry == 0ULL ? 0ULL : AddWithCarry(accumulator.limb2, carry);
        carry = carry == 0ULL ? 0ULL : AddWithCarry(accumulator.limb3, carry);
        if (carry != 0ULL) {
            carry = AddWithCarry(accumulator.limb4, carry);
            accumulator.limb5 += carry;
        }
    } else if (wordIndex == 2U) {
        carry = AddWithCarry(accumulator.limb2, value);
        carry = carry == 0ULL ? 0ULL : AddWithCarry(accumulator.limb3, carry);
        if (carry != 0ULL) {
            carry = AddWithCarry(accumulator.limb4, carry);
            accumulator.limb5 += carry;
        }
    } else if (wordIndex == 3U) {
        carry = AddWithCarry(accumulator.limb3, value);
        if (carry != 0ULL) {
            carry = AddWithCarry(accumulator.limb4, carry);
            accumulator.limb5 += carry;
        }
    } else {
        carry = AddWithCarry(accumulator.limb4, value);
        accumulator.limb5 += carry;
    }
}

__simt_callee__ inline void SubAccumulatorWord(ExactFloatAccumulator& accumulator, uint32_t wordIndex, uint64_t value)
{
    uint64_t borrow = 0ULL;
    if (wordIndex == 0U) {
        borrow = SubWithBorrow(accumulator.limb0, value);
        borrow = borrow == 0ULL ? 0ULL : SubWithBorrow(accumulator.limb1, borrow);
        borrow = borrow == 0ULL ? 0ULL : SubWithBorrow(accumulator.limb2, borrow);
        borrow = borrow == 0ULL ? 0ULL : SubWithBorrow(accumulator.limb3, borrow);
        if (borrow != 0ULL) {
            borrow = SubWithBorrow(accumulator.limb4, borrow);
            accumulator.limb5 -= borrow;
        }
    } else if (wordIndex == 1U) {
        borrow = SubWithBorrow(accumulator.limb1, value);
        borrow = borrow == 0ULL ? 0ULL : SubWithBorrow(accumulator.limb2, borrow);
        borrow = borrow == 0ULL ? 0ULL : SubWithBorrow(accumulator.limb3, borrow);
        if (borrow != 0ULL) {
            borrow = SubWithBorrow(accumulator.limb4, borrow);
            accumulator.limb5 -= borrow;
        }
    } else if (wordIndex == 2U) {
        borrow = SubWithBorrow(accumulator.limb2, value);
        borrow = borrow == 0ULL ? 0ULL : SubWithBorrow(accumulator.limb3, borrow);
        if (borrow != 0ULL) {
            borrow = SubWithBorrow(accumulator.limb4, borrow);
            accumulator.limb5 -= borrow;
        }
    } else if (wordIndex == 3U) {
        borrow = SubWithBorrow(accumulator.limb3, value);
        if (borrow != 0ULL) {
            borrow = SubWithBorrow(accumulator.limb4, borrow);
            accumulator.limb5 -= borrow;
        }
    } else {
        borrow = SubWithBorrow(accumulator.limb4, value);
        accumulator.limb5 -= borrow;
    }
}

__simt_callee__ inline void AccumulateExactFloat(float value, ExactFloatAccumulator& accumulator)
{
    union FloatBits {
        float f;
        uint32_t u;
    } bits;
    bits.f = value;
    const uint32_t sign = bits.u >> 31U;
    const uint32_t exponent = (bits.u >> 23U) & 0xFFU;
    uint32_t mantissa = bits.u & 0x007FFFFFU;
    if (exponent == 0xFFU) {
        if (mantissa != 0U) {
            accumulator.hasNan = true;
        } else if (sign == 0U) {
            accumulator.hasPositiveInf = true;
        } else {
            accumulator.hasNegativeInf = true;
        }
        return;
    }
    if (exponent == 0U) {
        if (mantissa == 0U) {
            return;
        }
    } else {
        mantissa |= 0x00800000U;
    }
    const uint32_t shift = exponent == 0U ? 0U : exponent - 1U;
    const uint32_t wordIndex = shift >> 6U;
    const uint32_t bitOffset = shift & 63U;
    const uint64_t lowPart = static_cast<uint64_t>(mantissa) << bitOffset;
    const uint64_t highPart = bitOffset == 0U ? 0ULL : static_cast<uint64_t>(mantissa) >> (64U - bitOffset);
    if (sign == 0U) {
        AddAccumulatorWord(accumulator, wordIndex, lowPart);
        if (highPart != 0ULL) {
            AddAccumulatorWord(accumulator, wordIndex + 1U, highPart);
        }
    } else {
        SubAccumulatorWord(accumulator, wordIndex, lowPart);
        if (highPart != 0ULL) {
            SubAccumulatorWord(accumulator, wordIndex + 1U, highPart);
        }
    }
}

__simt_callee__ inline uint64_t GetAccumulatorWord(const ExactFloatAccumulator& accumulator, uint32_t wordIndex)
{
    if (wordIndex == 0U) {
        return accumulator.limb0;
    }
    if (wordIndex == 1U) {
        return accumulator.limb1;
    }
    if (wordIndex == 2U) {
        return accumulator.limb2;
    }
    if (wordIndex == 3U) {
        return accumulator.limb3;
    }
    if (wordIndex == 4U) {
        return accumulator.limb4;
    }
    return wordIndex == 5U ? accumulator.limb5 : 0ULL;
}

__simt_callee__ inline void MakeAccumulatorMagnitude(ExactFloatAccumulator& accumulator)
{
    accumulator.limb0 = ~accumulator.limb0;
    accumulator.limb1 = ~accumulator.limb1;
    accumulator.limb2 = ~accumulator.limb2;
    accumulator.limb3 = ~accumulator.limb3;
    accumulator.limb4 = ~accumulator.limb4;
    accumulator.limb5 = ~accumulator.limb5;
    AddAccumulatorWord(accumulator, 0U, 1ULL);
}

__simt_callee__ inline int32_t HighestBitInWord(uint64_t value)
{
    int32_t bit = 0;
    if (value >= (1ULL << 32U)) {
        value >>= 32U;
        bit += 32;
    }
    if (value >= (1ULL << 16U)) {
        value >>= 16U;
        bit += 16;
    }
    if (value >= (1ULL << 8U)) {
        value >>= 8U;
        bit += 8;
    }
    if (value >= (1ULL << 4U)) {
        value >>= 4U;
        bit += 4;
    }
    if (value >= (1ULL << 2U)) {
        value >>= 2U;
        bit += 2;
    }
    if (value >= (1ULL << 1U)) {
        ++bit;
    }
    return bit;
}

__simt_callee__ inline int32_t HighestAccumulatorBit(const ExactFloatAccumulator& accumulator)
{
    if (accumulator.limb5 != 0ULL) {
        return 320 + HighestBitInWord(accumulator.limb5);
    }
    if (accumulator.limb4 != 0ULL) {
        return 256 + HighestBitInWord(accumulator.limb4);
    }
    if (accumulator.limb3 != 0ULL) {
        return 192 + HighestBitInWord(accumulator.limb3);
    }
    if (accumulator.limb2 != 0ULL) {
        return 128 + HighestBitInWord(accumulator.limb2);
    }
    if (accumulator.limb1 != 0ULL) {
        return 64 + HighestBitInWord(accumulator.limb1);
    }
    return accumulator.limb0 == 0ULL ? -1 : HighestBitInWord(accumulator.limb0);
}

__simt_callee__ inline uint64_t ShiftAccumulatorRight(const ExactFloatAccumulator& accumulator, uint32_t shift)
{
    const uint32_t wordIndex = shift >> 6U;
    const uint32_t bitOffset = shift & 63U;
    uint64_t value = GetAccumulatorWord(accumulator, wordIndex) >> bitOffset;
    if (bitOffset != 0U) {
        value |= GetAccumulatorWord(accumulator, wordIndex + 1U) << (64U - bitOffset);
    }
    return value;
}

__simt_callee__ inline bool AnyAccumulatorBitsBelow(const ExactFloatAccumulator& accumulator, uint32_t bitCount)
{
    const uint32_t completeWords = bitCount >> 6U;
    for (uint32_t index = 0U; index < completeWords; ++index) {
        if (GetAccumulatorWord(accumulator, index) != 0ULL) {
            return true;
        }
    }
    const uint32_t remainingBits = bitCount & 63U;
    if (remainingBits == 0U) {
        return false;
    }
    return (GetAccumulatorWord(accumulator, completeWords) & ((1ULL << remainingBits) - 1ULL)) != 0ULL;
}

__simt_callee__ inline float FinalizeExactFloat(ExactFloatAccumulator accumulator)
{
    union FloatBits {
        float f;
        uint32_t u;
    } result;
    if (accumulator.hasNan || (accumulator.hasPositiveInf && accumulator.hasNegativeInf)) {
        result.u = 0x7FC00000U;
        return result.f;
    }
    if (accumulator.hasPositiveInf) {
        result.u = 0x7F800000U;
        return result.f;
    }
    if (accumulator.hasNegativeInf) {
        result.u = 0xFF800000U;
        return result.f;
    }
    const bool negative = (accumulator.limb5 >> 63U) != 0ULL;
    if (negative) {
        MakeAccumulatorMagnitude(accumulator);
    }
    int32_t highestBit = HighestAccumulatorBit(accumulator);
    if (highestBit < 0) {
        result.u = 0U;
        return result.f;
    }
    if (highestBit <= 22) {
        result.u = (negative ? 0x80000000U : 0U) | static_cast<uint32_t>(accumulator.limb0);
        return result.f;
    }
    const uint32_t rightShift = static_cast<uint32_t>(highestBit - 23);
    uint64_t significand = ShiftAccumulatorRight(accumulator, rightShift) & 0x00FFFFFFULL;
    if (rightShift != 0U) {
        const uint32_t roundIndex = rightShift - 1U;
        const bool roundBit = ((GetAccumulatorWord(accumulator, roundIndex >> 6U) >> (roundIndex & 63U)) & 1ULL) !=
                              0ULL;
        const bool sticky = AnyAccumulatorBitsBelow(accumulator, rightShift - 1U);
        if (roundBit && (sticky || ((significand & 1ULL) != 0ULL))) {
            ++significand;
            if (significand == 0x01000000ULL) {
                significand >>= 1U;
                ++highestBit;
            }
        }
    }
    const int32_t exponent = highestBit - 22;
    if (exponent >= 255) {
        result.u = negative ? 0xFF800000U : 0x7F800000U;
        return result.f;
    }
    result.u = (negative ? 0x80000000U : 0U) | (static_cast<uint32_t>(exponent) << 23U) |
               (static_cast<uint32_t>(significand) & 0x007FFFFFU);
    return result.f;
}

struct DeterministicPartial {
    uint64_t words[8];
};

static_assert(sizeof(DeterministicPartial) == DETERMINISTIC_PARTIAL_BYTES,
              "host and kernel disagree on the deterministic partial size");

struct IntegerHistogram {
    uint64_t words[16];
};

static_assert(sizeof(IntegerHistogram) == DETERMINISTIC_HISTOGRAM_SLOT_BYTES,
              "host and kernel disagree on the deterministic histogram size");

__simt_callee__ inline uint64_t MergeAccumulatorWord(uint64_t& destination, uint64_t value, uint64_t carry)
{
    const uint64_t firstCarry = AddWithCarry(destination, value);
    const uint64_t secondCarry = AddWithCarry(destination, carry);
    return firstCarry | secondCarry;
}

__simt_callee__ inline void MergeDeterministicPartial(ExactFloatAccumulator& sum, __gm__ DeterministicPartial* partial)
{
    uint64_t carry = AddWithCarry(sum.limb0, partial->words[0]);
    carry = MergeAccumulatorWord(sum.limb1, partial->words[1], carry);
    carry = MergeAccumulatorWord(sum.limb2, partial->words[2], carry);
    carry = MergeAccumulatorWord(sum.limb3, partial->words[3], carry);
    carry = MergeAccumulatorWord(sum.limb4, partial->words[4], carry);
    (void)MergeAccumulatorWord(sum.limb5, partial->words[5], carry);
    const uint64_t flags = partial->words[6];
    sum.hasNan = sum.hasNan || ((flags & 1U) != 0U);
    sum.hasPositiveInf = sum.hasPositiveInf || ((flags & 2U) != 0U);
    sum.hasNegativeInf = sum.hasNegativeInf || ((flags & 4U) != 0U);
}

template <typename T, uint32_t THREAD_NUM, bool IS_1D>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void BuildDeterministicPartials(
    uint32_t coreIdx, uint32_t coreNum, __gm__ T* input, __gm__ float* weights, int64_t numValues, int64_t inputRows,
    int64_t inputCols, int64_t size, int64_t outputElements, uint32_t partitions, __gm__ DeterministicPartial* partials)
{
    const int64_t totalPartitions = static_cast<int64_t>(coreNum) * partitions;
    for (int64_t task = threadIdx.x; task < outputElements * partitions; task += blockDim.x) {
        const int64_t outputIndex = task / partitions;
        const int64_t localPartition = task % partitions;
        const int64_t partition = static_cast<int64_t>(coreIdx) * partitions + localPartition;
        const int64_t baseLength = numValues / totalPartitions;
        const int64_t extra = numValues % totalPartitions;
        const int64_t begin = partition * baseLength + (partition < extra ? partition : extra);
        const int64_t end = begin + baseLength + (partition < extra ? 1 : 0);
        ExactFloatAccumulator sum{};
        for (int64_t index = begin; index < end; ++index) {
            int64_t mappedOutput = 0;
            if (MapInputToOutput<T, IS_1D>(input[index], index, inputRows, inputCols, size, mappedOutput) &&
                mappedOutput == outputIndex) {
                AccumulateExactFloat(weights[index], sum);
            }
        }
        const int64_t slot = (static_cast<int64_t>(coreIdx) * outputElements + outputIndex) * partitions +
                             localPartition;
        for (uint32_t word = 0U; word < 6U; ++word) {
            (void)asc_atomic_exch(&partials[slot].words[word], GetAccumulatorWord(sum, word));
        }
        const uint64_t flags = static_cast<uint64_t>(sum.hasNan) | (static_cast<uint64_t>(sum.hasPositiveInf) << 1U) |
                               (static_cast<uint64_t>(sum.hasNegativeInf) << 2U);
        (void)asc_atomic_exch(&partials[slot].words[6], flags);
    }
}

template <uint32_t THREAD_NUM>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void FinalizeDeterministicPartials(
    uint32_t coreIdx, uint32_t coreNum, int64_t outputElements, uint32_t partitions,
    __gm__ DeterministicPartial* partials, __gm__ float* output)
{
    const int64_t first = static_cast<int64_t>(threadIdx.x) * coreNum + coreIdx;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * coreNum;
    for (int64_t outputIndex = first; outputIndex < outputElements; outputIndex += stride) {
        ExactFloatAccumulator sum{};
        for (int64_t core = 0; core < static_cast<int64_t>(coreNum); ++core) {
            for (uint32_t partition = 0U; partition < partitions; ++partition) {
                const int64_t slot = (core * outputElements + outputIndex) * partitions + partition;
                MergeDeterministicPartial(sum, partials + slot);
            }
        }
        (void)asc_atomic_exch(output + outputIndex, FinalizeExactFloat(sum));
    }
}

__simt_callee__ inline void AddIntegerHistogram(__gm__ IntegerHistogram* slot, float weight)
{
    union FloatBits {
        float value;
        uint32_t bits;
    } magnitude;
    magnitude.value = weight;
    const bool negative = (magnitude.bits >> 31U) != 0U;
    magnitude.bits &= 0x7FFFFFFFU;
    ExactFloatAccumulator value{};
    AccumulateExactFloat(magnitude.value, value);
    const uint32_t base = negative ? 6U : 0U;
    uint64_t carry = 0U;
    bool firstNonzero = false;
    for (uint32_t word = 0U; word < 6U; ++word) {
        const uint64_t limb = GetAccumulatorWord(value, word);
        const uint64_t increment = limb + carry;
        carry = static_cast<uint64_t>(increment < limb);
        if (increment != 0U) {
            const uint64_t previous = asc_atomic_add(&slot->words[base + word], increment);
            carry += static_cast<uint64_t>(previous > ~increment);
            firstNonzero = firstNonzero || previous == 0U;
        }
    }
    const uint64_t flags = static_cast<uint64_t>(value.hasNan) |
                           (static_cast<uint64_t>(value.hasPositiveInf) << (negative ? 2U : 1U));
    if (firstNonzero || flags != 0U) {
        (void)asc_atomic_or(&slot->words[12], flags | 8U);
    }
}

__aicore__ inline void InitializeIntegerHistogram(uint32_t coreIdx, uint32_t coreNum, int64_t outputElements,
                                                  uint32_t bufferBytes, __gm__ IntegerHistogram* histogram, TPipe* pipe)
{
    const uint32_t tileWords = bufferBytes / sizeof(uint32_t);
    constexpr int64_t wordsPerSlot = DETERMINISTIC_HISTOGRAM_SLOT_BYTES / sizeof(uint32_t);
    const int64_t slotsPerCore = outputElements / coreNum + static_cast<int64_t>(outputElements % coreNum != 0U);
    const int64_t begin = static_cast<int64_t>(coreIdx) * slotsPerCore * wordsPerSlot;
    const int64_t lastSlot = (static_cast<int64_t>(coreIdx) + 1) * slotsPerCore;
    const int64_t end = (lastSlot < outputElements ? lastSlot : outputElements) * wordsPerSlot;
    TQue<TPosition::VECOUT, 1> queue;
    pipe->InitBuffer(queue, 1, bufferBytes);
    LocalTensor<uint32_t> zeros = queue.AllocTensor<uint32_t>();
    Duplicate(zeros, uint32_t{0}, tileWords);
    queue.EnQue(zeros);
    zeros = queue.DeQue<uint32_t>();
    GlobalTensor<uint32_t> target;
    target.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(histogram), outputElements * wordsPerSlot);
    for (int64_t offset = begin; offset < end; offset += tileWords) {
        const uint32_t count = static_cast<uint32_t>(end - offset < tileWords ? end - offset : tileWords);
        DataCopy(target[offset], zeros, count);
    }
    queue.FreeTensor(zeros);
}

template <typename T, uint32_t THREAD_NUM, bool IS_1D>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ScatterIntegerHistogram(
    uint32_t coreIdx, uint32_t coreNum, __gm__ T* input, __gm__ float* weights, int64_t numValues, int64_t inputRows,
    int64_t inputCols, int64_t size, __gm__ IntegerHistogram* histogram)
{
    const int64_t first = static_cast<int64_t>(coreIdx) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(coreNum) * blockDim.x;
    for (int64_t index = first; index < numValues; index += stride) {
        int64_t outputOffset = 0;
        if (MapInputToOutput<T, IS_1D>(input[index], index, inputRows, inputCols, size, outputOffset)) {
            AddIntegerHistogram(histogram + outputOffset, weights[index]);
        }
    }
}

template <uint32_t THREAD_NUM>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void FinalizeIntegerHistogram(uint32_t coreIdx, uint32_t coreNum,
                                                                                     int64_t outputElements,
                                                                                     __gm__ IntegerHistogram* histogram,
                                                                                     __gm__ float* output)
{
    const int64_t first = static_cast<int64_t>(coreIdx) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(coreNum) * blockDim.x;
    for (int64_t index = first; index < outputElements; index += stride) {
        const uint64_t flags = histogram[index].words[12];
        if ((flags & 8U) == 0U) {
            continue;
        }
        ExactFloatAccumulator sum{histogram[index].words[0],
                                  histogram[index].words[1],
                                  histogram[index].words[2],
                                  histogram[index].words[3],
                                  histogram[index].words[4],
                                  histogram[index].words[5],
                                  false,
                                  false,
                                  false};
        ExactFloatAccumulator negative{histogram[index].words[6],
                                       histogram[index].words[7],
                                       histogram[index].words[8],
                                       histogram[index].words[9],
                                       histogram[index].words[10],
                                       histogram[index].words[11],
                                       false,
                                       false,
                                       false};
        MakeAccumulatorMagnitude(negative);
        uint64_t carry = AddWithCarry(sum.limb0, negative.limb0);
        carry = MergeAccumulatorWord(sum.limb1, negative.limb1, carry);
        carry = MergeAccumulatorWord(sum.limb2, negative.limb2, carry);
        carry = MergeAccumulatorWord(sum.limb3, negative.limb3, carry);
        carry = MergeAccumulatorWord(sum.limb4, negative.limb4, carry);
        (void)MergeAccumulatorWord(sum.limb5, negative.limb5, carry);
        sum.hasNan = (flags & 1U) != 0U;
        sum.hasPositiveInf = (flags & 2U) != 0U;
        sum.hasNegativeInf = (flags & 4U) != 0U;
        (void)asc_atomic_exch(output + index, FinalizeExactFloat(sum));
    }
}

template <uint32_t THREAD_NUM>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void InitializeOutput(uint32_t coreIdx, uint32_t coreNum,
                                                                             __gm__ float* output, int64_t outputSize)
{
    const int64_t first = static_cast<int64_t>(coreIdx) * static_cast<int64_t>(blockDim.x) +
                          static_cast<int64_t>(threadIdx.x);
    const int64_t stride = static_cast<int64_t>(coreNum) * static_cast<int64_t>(blockDim.x);
    for (int64_t index = first; index < outputSize; index += stride) {
        output[index] = 0.0F;
    }
}

template <bool PRIVATE, bool BINARY_OUTPUT, bool HAS_WEIGHTS>
__simt_callee__ inline void ScatterOne(__gm__ float* output, __ubuf__ float* histogram, int64_t outputOffset,
                                       int64_t inputOffset, __gm__ float* weights)
{
    if constexpr (PRIVATE) {
        if constexpr (BINARY_OUTPUT) {
            if (histogram[outputOffset] != 1.0F) {
                (void)asc_atomic_exch(histogram + outputOffset, 1.0F);
            }
        } else if constexpr (HAS_WEIGHTS) {
            (void)asc_atomic_add(histogram + outputOffset, weights[inputOffset]);
        } else {
            (void)asc_atomic_add(histogram + outputOffset, 1.0F);
        }
    } else {
        if constexpr (BINARY_OUTPUT) {
            if (output[outputOffset] != 1.0F) {
                (void)asc_atomic_exch(output + outputOffset, 1.0F);
            }
        } else if constexpr (HAS_WEIGHTS) {
            (void)asc_atomic_add(output + outputOffset, weights[inputOffset]);
        } else {
            (void)asc_atomic_add(output + outputOffset, 1.0F);
        }
    }
}

template <typename T, uint32_t THREAD_NUM, bool IS_1D, bool BINARY_OUTPUT, bool HAS_WEIGHTS, bool PRIVATE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ScatterValues(
    uint32_t coreIdx, uint32_t coreNum, __gm__ T* input, __gm__ float* weights, __gm__ float* output,
    __ubuf__ float* histogram, int64_t numValues, int64_t inputRows, int64_t inputCols, int64_t size)
{
    const int64_t first = static_cast<int64_t>(coreIdx) * static_cast<int64_t>(blockDim.x) +
                          static_cast<int64_t>(threadIdx.x);
    const int64_t stride = static_cast<int64_t>(coreNum) * static_cast<int64_t>(blockDim.x);
    for (int64_t index = first; index < numValues; index += stride) {
        int64_t bin = static_cast<int64_t>(input[index]);
        int64_t row = 0;
        if constexpr (!IS_1D) {
            row = index / inputCols;
            if (bin < 0) {
                const int64_t quotient = bin / size;
                const int64_t remainder = bin % size;
                const int64_t adjustment = remainder == 0 ? 0 : 1;
                if (quotient < adjustment - row) {
                    continue;
                }
                row += quotient - adjustment;
                bin = remainder == 0 ? 0 : remainder + size;
            }
        }
        if (bin < 0 || bin >= size || row < 0 || row >= inputRows) {
            continue;
        }
        const int64_t outputOffset = IS_1D ? bin : row * size + bin;
        ScatterOne<PRIVATE, BINARY_OUTPUT, HAS_WEIGHTS>(output, histogram, outputOffset, index, weights);
    }
}

template <typename T, uint32_t THREAD_NUM, bool IS_1D>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void AccumulateOrderedWeighted(
    uint32_t coreIdx, uint32_t coreNum, __gm__ T* input, __gm__ float* weights, __gm__ float* output, int64_t numValues,
    int64_t inputRows, int64_t inputCols, int64_t size)
{
    const int64_t outputSize = IS_1D ? size : inputRows * size;
    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t warp = threadIdx.x / warpSize;
    const uint32_t warpsPerCore = blockDim.x / warpSize;
    const int64_t first = static_cast<int64_t>(coreIdx) * warpsPerCore + warp;
    const int64_t stride = static_cast<int64_t>(coreNum) * warpsPerCore;
    for (int64_t outputIndex = first; outputIndex < outputSize; outputIndex += stride) {
        float result = 0.0F;
        for (int64_t index = lane; index < numValues; index += warpSize) {
            int64_t bin = static_cast<int64_t>(input[index]);
            int64_t row = 0;
            if constexpr (!IS_1D) {
                row = index / inputCols;
                if (bin < 0) {
                    const int64_t quotient = bin / size;
                    const int64_t remainder = bin % size;
                    const int64_t adjustment = remainder == 0 ? 0 : 1;
                    if (quotient < adjustment - row) {
                        continue;
                    }
                    row += quotient - adjustment;
                    bin = remainder == 0 ? 0 : remainder + size;
                }
            }
            if (bin < 0 || bin >= size || row < 0 || row >= inputRows) {
                continue;
            }
            const int64_t mappedOutput = IS_1D ? bin : row * size + bin;
            if (mappedOutput == outputIndex) {
                result += weights[index];
            }
        }
        for (uint32_t offset = warpSize / 2U; offset > 0U; offset >>= 1U) {
            result += asc_shfl_down(result, offset);
        }
        if (lane == 0U) {
            (void)asc_atomic_exch(output + outputIndex, result);
        }
    }
}

template <typename T, uint32_t THREAD_NUM, bool BINARY_OUTPUT, bool HAS_WEIGHTS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ScatterValuesToTile(uint32_t coreIdx, uint32_t coreNum,
                                                                                __gm__ T* input, __gm__ float* weights,
                                                                                __ubuf__ float* histogram,
                                                                                int64_t numValues, int64_t size,
                                                                                int64_t tileStart, int64_t tileLength)
{
    const int64_t first = static_cast<int64_t>(coreIdx) * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(coreNum) * static_cast<int64_t>(blockDim.x);
    const int64_t tileEnd = tileStart + tileLength;
    for (int64_t index = first; index < numValues; index += stride) {
        const int64_t bin = static_cast<int64_t>(input[index]);
        if (bin < tileStart || bin >= tileEnd || bin < 0 || bin >= size) {
            continue;
        }
        ScatterOne<true, BINARY_OUTPUT, HAS_WEIGHTS>((__gm__ float*)nullptr, histogram, bin - tileStart, index,
                                                     weights);
    }
}

template <typename T, uint32_t THREAD_NUM>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ScatterBinaryRow(uint32_t coreIdx, uint32_t coresPerRow,
                                                                             __gm__ T* input, __gm__ float* output,
                                                                             __ubuf__ float* histogram,
                                                                             int64_t inputRows, int64_t inputCols,
                                                                             int64_t size)
{
    const uint32_t row = coreIdx / coresPerRow;
    const uint32_t coreInRow = coreIdx - row * coresPerRow;
    const int64_t rowBegin = static_cast<int64_t>(row) * inputCols;
    const int64_t firstCol = static_cast<int64_t>(coreInRow) * static_cast<int64_t>(blockDim.x) +
                             static_cast<int64_t>(threadIdx.x);
    const int64_t stride = static_cast<int64_t>(coresPerRow) * static_cast<int64_t>(blockDim.x);
    for (int64_t col = firstCol; col < inputCols; col += stride) {
        const int64_t inputOffset = rowBegin + col;
        int64_t bin = static_cast<int64_t>(input[inputOffset]);
        if (bin >= 0 && bin < size) {
            (void)asc_atomic_exch(histogram + bin, 1.0F);
            continue;
        }
        if (bin >= 0) {
            continue;
        }
        int64_t targetRow = static_cast<int64_t>(row);
        const int64_t quotient = bin / size;
        const int64_t remainder = bin % size;
        const int64_t adjustment = remainder == 0 ? 0 : 1;
        if (quotient < adjustment - targetRow) {
            continue;
        }
        targetRow += quotient - adjustment;
        bin = remainder == 0 ? 0 : remainder + size;
        if (targetRow >= 0 && targetRow < inputRows) {
            (void)asc_atomic_exch(output + targetRow * size + bin, 1.0F);
        }
    }
}

template <typename T, typename S, bool IS_1D, bool BINARY_OUTPUT, bool HAS_WEIGHTS, bool ORDERED, bool DETERMINISTIC>
class DenseBincountRegbase {
public:
    __aicore__ inline DenseBincountRegbase(const DenseBincountTilingData& tiling, TPipe* pipe)
        : tiling_(tiling), pipe_(pipe)
    {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR size, GM_ADDR weights, GM_ADDR output, GM_ADDR workspace)
    {
        input_ = reinterpret_cast<__gm__ T*>(input);
        size_ = reinterpret_cast<__gm__ S*>(size);
        weights_ = reinterpret_cast<__gm__ float*>(weights);
        output_ = reinterpret_cast<__gm__ float*>(output);
        workspace_ = workspace;
    }

    template <uint32_t THREAD_NUM>
    __aicore__ inline void ProcessPrivateHistogram(uint32_t coreIdx, uint32_t coreNum, int64_t size)
    {
        const uint32_t histogramBytes = tiling_.privateHistElems * static_cast<uint32_t>(sizeof(float));
        const uint32_t bufferBytes = ((histogramBytes + UB_BLOCK_BYTES - 1U) / UB_BLOCK_BYTES) * UB_BLOCK_BYTES;
        TQue<TPosition::VECOUT, 1> histogramQueue;
        pipe_->InitBuffer(histogramQueue, 1, bufferBytes);
        LocalTensor<float> histogram = histogramQueue.AllocTensor<float>();
        Duplicate(histogram, 0.0F, tiling_.privateHistElems);
        histogramQueue.EnQue(histogram);
        histogram = histogramQueue.DeQue<float>();
        asc_vf_call<ScatterValues<T, THREAD_NUM, IS_1D, BINARY_OUTPUT, HAS_WEIGHTS, true>>(
            dim3(THREAD_NUM), coreIdx, coreNum, input_, weights_, output_,
            reinterpret_cast<__ubuf__ float*>(histogram.GetPhyAddr()), tiling_.numValues, tiling_.inputRows,
            tiling_.inputCols, size);
        event_t histogramReady = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(histogramReady);
        WaitFlag<HardEvent::V_MTE3>(histogramReady);
        if constexpr (BINARY_OUTPUT) {
            SetAtomicMax<float>();
        } else {
            SetAtomicAdd<float>();
        }
        GlobalTensor<float> outputGlobal;
        outputGlobal.SetGlobalBuffer(output_);
        const DataCopyExtParams copyParams{static_cast<uint16_t>(1), histogramBytes, 0U, 0U, 0U};
        DataCopyPad(outputGlobal, histogram, copyParams);
        SetAtomicNone();
        histogramQueue.FreeTensor(histogram);
    }

    template <uint32_t THREAD_NUM>
    __aicore__ inline void ProcessBinaryRowHistogram(uint32_t coreIdx, uint32_t coreNum, int64_t size)
    {
        if (tiling_.inputRows <= 0 || coreNum < static_cast<uint32_t>(tiling_.inputRows)) {
            return;
        }
        const uint32_t coresPerRow = coreNum / static_cast<uint32_t>(tiling_.inputRows);
        const uint32_t activeCoreNum = coresPerRow * static_cast<uint32_t>(tiling_.inputRows);
        if (coreIdx >= activeCoreNum) {
            return;
        }
        const uint32_t histogramBytes = static_cast<uint32_t>(size) * static_cast<uint32_t>(sizeof(float));
        const uint32_t bufferBytes = ((histogramBytes + UB_BLOCK_BYTES - 1U) / UB_BLOCK_BYTES) * UB_BLOCK_BYTES;
        TQue<TPosition::VECOUT, 1> histogramQueue;
        pipe_->InitBuffer(histogramQueue, 1, bufferBytes);
        LocalTensor<float> histogram = histogramQueue.AllocTensor<float>();
        Duplicate(histogram, 0.0F, static_cast<uint32_t>(size));
        histogramQueue.EnQue(histogram);
        histogram = histogramQueue.DeQue<float>();
        asc_vf_call<ScatterBinaryRow<T, THREAD_NUM>>(dim3(THREAD_NUM), coreIdx, coresPerRow, input_, output_,
                                                     reinterpret_cast<__ubuf__ float*>(histogram.GetPhyAddr()),
                                                     tiling_.inputRows, tiling_.inputCols, size);
        event_t histogramReady = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(histogramReady);
        WaitFlag<HardEvent::V_MTE3>(histogramReady);
        SetAtomicMax<float>();
        GlobalTensor<float> outputGlobal;
        const uint32_t row = coreIdx / coresPerRow;
        outputGlobal.SetGlobalBuffer(output_ + static_cast<int64_t>(row) * size);
        const DataCopyExtParams copyParams{static_cast<uint16_t>(1), histogramBytes, 0U, 0U, 0U};
        DataCopyPad(outputGlobal, histogram, copyParams);
        SetAtomicNone();
        histogramQueue.FreeTensor(histogram);
    }

    template <uint32_t THREAD_NUM>
    __aicore__ inline void ProcessTiledHistogram(uint32_t coreIdx, uint32_t coreNum, int64_t size)
    {
        const uint32_t capacity = tiling_.privateHistElems;
        const uint32_t bufferBytes = capacity * static_cast<uint32_t>(sizeof(float));
        TQue<TPosition::VECOUT, 1> histogramQueue;
        pipe_->InitBuffer(histogramQueue, 1, bufferBytes);
        GlobalTensor<float> outputGlobal;
        for (int64_t tileStart = 0; tileStart < size; tileStart += capacity) {
            const uint32_t tileLength = static_cast<uint32_t>(
                size - tileStart < static_cast<int64_t>(capacity) ? size - tileStart : capacity);
            LocalTensor<float> histogram = histogramQueue.AllocTensor<float>();
            Duplicate(histogram, 0.0F, tileLength);
            histogramQueue.EnQue(histogram);
            histogram = histogramQueue.DeQue<float>();
            asc_vf_call<ScatterValuesToTile<T, THREAD_NUM, BINARY_OUTPUT, HAS_WEIGHTS>>(
                dim3(THREAD_NUM), coreIdx, coreNum, input_, weights_,
                reinterpret_cast<__ubuf__ float*>(histogram.GetPhyAddr()), tiling_.numValues, size, tileStart,
                tileLength);
            event_t histogramReady = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(histogramReady);
            WaitFlag<HardEvent::V_MTE3>(histogramReady);
            if constexpr (BINARY_OUTPUT) {
                SetAtomicMax<float>();
            } else {
                SetAtomicAdd<float>();
            }
            outputGlobal.SetGlobalBuffer(output_ + tileStart);
            const DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                               static_cast<uint32_t>(tileLength * sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(outputGlobal, histogram, copyParams);
            SetAtomicNone();
            histogramQueue.FreeTensor(histogram);
        }
    }

    __aicore__ inline void Process()
    {
        const int64_t size = static_cast<int64_t>(size_[0]);
        if (size <= 0) {
            return;
        }
        const int64_t outputSize = IS_1D ? size : tiling_.inputRows * size;
        constexpr uint32_t THREAD_NUM = sizeof(T) == sizeof(int64_t) ? 512U : 1024U;
        const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());
        const uint32_t launchCoreNum = static_cast<uint32_t>(GetBlockNum());
        const uint32_t coreNum = tiling_.usedCoreNum < launchCoreNum ? tiling_.usedCoreNum : launchCoreNum;
        const bool participates = coreIdx < coreNum;
        if constexpr (ORDERED) {
            static_assert(HAS_WEIGHTS && !BINARY_OUTPUT, "ordered key is only valid for weighted accumulation");
            if (participates) {
                asc_vf_call<AccumulateOrderedWeighted<T, THREAD_NUM, IS_1D>>(
                    dim3(THREAD_NUM), coreIdx, coreNum, input_, weights_, output_, tiling_.numValues, tiling_.inputRows,
                    tiling_.inputCols, size);
            }
            return;
        }
        if (participates) {
            asc_vf_call<InitializeOutput<THREAD_NUM>>(dim3(THREAD_NUM), coreIdx, coreNum, output_, outputSize);
        }
        SyncAll();
        if constexpr (DETERMINISTIC) {
            static_assert(HAS_WEIGHTS && !BINARY_OUTPUT, "deterministic key is only valid for weighted accumulation");
            if (tiling_.deterministicMode == DETERMINISTIC_PARTIAL) {
                auto partials = reinterpret_cast<__gm__ DeterministicPartial*>(workspace_);
                if (participates) {
                    asc_vf_call<BuildDeterministicPartials<T, THREAD_NUM, IS_1D>>(
                        dim3(THREAD_NUM), coreIdx, coreNum, input_, weights_, tiling_.numValues, tiling_.inputRows,
                        tiling_.inputCols, size, outputSize, tiling_.deterministicPartitions, partials);
                }
                SyncAll();
                if (participates) {
                    asc_vf_call<FinalizeDeterministicPartials<THREAD_NUM>>(dim3(THREAD_NUM), coreIdx, coreNum,
                                                                           outputSize, tiling_.deterministicPartitions,
                                                                           partials, output_);
                }
            } else if (tiling_.deterministicMode == DETERMINISTIC_HISTOGRAM) {
                auto histogram = reinterpret_cast<__gm__ IntegerHistogram*>(workspace_);
                if (participates) {
                    InitializeIntegerHistogram(coreIdx, coreNum, outputSize, tiling_.deterministicBufferBytes,
                                               histogram, pipe_);
                }
                SyncAll();
                if (participates) {
                    asc_vf_call<ScatterIntegerHistogram<T, THREAD_NUM, IS_1D>>(
                        dim3(THREAD_NUM), coreIdx, coreNum, input_, weights_, tiling_.numValues, tiling_.inputRows,
                        tiling_.inputCols, size, histogram);
                }
                SyncAll();
                if (participates) {
                    asc_vf_call<FinalizeIntegerHistogram<THREAD_NUM>>(dim3(THREAD_NUM), coreIdx, coreNum, outputSize,
                                                                      histogram, output_);
                }
            }
            return;
        }
        if (!participates) {
            return;
        }
        if (tiling_.privateHistMode == PRIVATE_HIST_ROW) {
            ProcessBinaryRowHistogram<THREAD_NUM>(coreIdx, coreNum, size);
        } else if (tiling_.privateHistMode == PRIVATE_HIST_TILED) {
            ProcessTiledHistogram<THREAD_NUM>(coreIdx, coreNum, size);
        } else if (tiling_.privateHistElems > 0U) {
            ProcessPrivateHistogram<THREAD_NUM>(coreIdx, coreNum, size);
        } else {
            asc_vf_call<ScatterValues<T, THREAD_NUM, IS_1D, BINARY_OUTPUT, HAS_WEIGHTS, false>>(
                dim3(THREAD_NUM), coreIdx, coreNum, input_, weights_, output_, (__ubuf__ float*)nullptr,
                tiling_.numValues, tiling_.inputRows, tiling_.inputCols, size);
        }
    }

private:
    const DenseBincountTilingData& tiling_;
    TPipe* pipe_ = nullptr;
    __gm__ T* input_ = nullptr;
    __gm__ S* size_ = nullptr;
    __gm__ float* weights_ = nullptr;
    __gm__ float* output_ = nullptr;
    GM_ADDR workspace_ = nullptr;
};
} // namespace NsDenseBincount

#endif
