/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SELECT_TOP_K_SCALAR_ROW_H_
#define SELECT_TOP_K_SCALAR_ROW_H_
#include "select_top_k_scalar_layout.h"
#include "select_top_k_base_common.h"

namespace SelectTopK {
using namespace SelectTopKBase;
// 最多16项的标准单行按需选择或排序，直接按原始位模式写出值和INT64索引。
template <typename T>
class ScalarRow {
    using Bits = typename std::conditional<sizeof(T) == VALUE16_BYTES, uint16_t, uint32_t>::type;

public:
    // 一次读入后，无序且只舍弃一项时线性选择，其余用打包排序；有效项以外不读写GM。
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR values, GM_ADDR indices, const SelectTopKBaseTilingData& data)
    {
        constexpr ScalarRowLayout layout(sizeof(T));
        LocalTensor<uint8_t> arena(TPosition::VECCALC, 0, layout.bytes);
        auto raw = arena.template ReinterpretCast<Bits>();
        auto records = arena.template ReinterpretCast<uint64_t>()[layout.records / INDEX64_BYTES];
        auto result = arena.template ReinterpretCast<Bits>()[layout.values / sizeof(T)];
        auto wide = arena.template ReinterpretCast<int64_t>()[layout.wideIndices / INDEX64_BYTES];
        GlobalTensor<Bits> input, valueOutput;
        GlobalTensor<int64_t> indexOutput;
        input.SetGlobalBuffer((__gm__ Bits*)x);
        valueOutput.SetGlobalBuffer((__gm__ Bits*)values);
        indexOutput.SetGlobalBuffer((__gm__ int64_t*)indices);
        DataCopyPad(raw, input, DataCopyExtParams{1, static_cast<uint32_t>(data.axis * sizeof(T)), 0, 0, 0},
                    DataCopyPadExtParams<Bits>{false, 0, 0, 0});
        if (!data.sorted && data.k + 1 == data.axis) {
            EmitWithoutWorst(raw, result, wide, records.template ReinterpretCast<float>(), data);
        } else {
            Fence<HardEvent::MTE2_S>();
            for (uint32_t item = 0; item < data.axis; ++item) {
                const uint64_t record = PackScalarRecord(Rank(raw.GetValue(item)), item, data.largest);
                InsertScalarRecord(records, item, record);
            }
            for (uint32_t item = 0; item < data.k; ++item) {
                const uint32_t index = static_cast<uint32_t>(records.GetValue(item));
                result.SetValue(item, raw.GetValue(index));
                wide.SetValue(item, static_cast<int64_t>(index));
            }
        }
        Fence<HardEvent::S_MTE3>();
        DataCopyPad(valueOutput, result, DataCopyExtParams{1, static_cast<uint32_t>(data.k * sizeof(T)), 0, 0, 0});
        DataCopyPad(indexOutput, wide, DataCopyExtParams{1, static_cast<uint32_t>(data.k * INDEX64_BYTES), 0, 0, 0});
#if defined(__ASCENDC_ENABLE_SUPER_KERNEL__)
        Fence<HardEvent::MTE3_S>();
#endif
    }

private:
    // FP16转FP32精确且非零子正规数变为正规数；一次向量归约定位待舍弃项，NaN回退整数秩。
    __aicore__ inline uint32_t ReduceOmittedHalf(const LocalTensor<Bits>& raw, const LocalTensor<float>& temporary,
                                                 const SelectTopKBaseTilingData& data)
    {
        static_assert(SCALAR_ROW_ITEMS * VALUE32_BYTES + UB_BLOCK_BYTES <= SCALAR_ROW_ITEMS * INDEX64_BYTES,
                      "Scalar reduction exceeds the record buffer");
        auto reduced = temporary[SCALAR_ROW_ITEMS];
        Fence<HardEvent::MTE2_V>();
        Cast(temporary, raw.template ReinterpretCast<half>(), RoundMode::CAST_NONE, static_cast<uint32_t>(data.axis));
        PipeBarrier<PIPE_V>();
        if (data.largest) {
            ReduceRepeat<ReduceType::MIN, float, float>(reduced, temporary, static_cast<int32_t>(data.axis), 1, 1, 1,
                                                        VECTOR_REPEAT_BLOCKS);
        } else {
            ReduceRepeat<ReduceType::MAX, float, float>(reduced, temporary, static_cast<int32_t>(data.axis), 1, 1, 1,
                                                        VECTOR_REPEAT_BLOCKS);
        }
        Fence<HardEvent::V_S>();
        ResetMask();
        const float value = reduced.GetValue(0);
        return value == value ? reduced.template ReinterpretCast<uint32_t>().GetValue(1) : UINT32_MAX;
    }

    // 无序K=N-1只舍弃最差项；FP16先归约，其余情况比较精确秩，输出始终拼接原始位。
    __aicore__ inline void EmitWithoutWorst(const LocalTensor<Bits>& raw, const LocalTensor<Bits>& result,
                                            const LocalTensor<int64_t>& wide, const LocalTensor<float>& temporary,
                                            const SelectTopKBaseTilingData& data)
    {
        uint32_t skipped = UINT32_MAX;
        if constexpr (std::is_same<T, half>::value) {
            skipped = ReduceOmittedHalf(raw, temporary, data);
        } else {
            // BF16/FP32可能含FP32子正规数，保留整数秩比较，不依赖浮点归约的处理方式。
            Fence<HardEvent::MTE2_S>();
        }
        constexpr uint32_t valueBits = sizeof(Bits) * SCALAR_BYTE_BITS;
        constexpr uint32_t itemsPerWord = sizeof(uint64_t) / sizeof(Bits);
        constexpr uint32_t wordCount = SCALAR_ROW_ITEMS / itemsPerWord;
        uint64_t words[wordCount + 1];
        auto packedInput = raw.template ReinterpretCast<uint64_t>();
#pragma unroll
        for (uint32_t word = 0; word < wordCount; ++word)
            words[word] = word * itemsPerWord < data.axis ? packedInput.GetValue(word) : 0;
        words[wordCount] = 0;
        if (skipped == UINT32_MAX) {
            const uint32_t directionMask = 0U - static_cast<uint32_t>(data.largest != 0);
            skipped = 0;
            uint32_t worstRank = Rank(static_cast<Bits>(words[0])) ^ directionMask;
#pragma unroll
            for (uint32_t item = 1; item < SCALAR_ROW_ITEMS; ++item) {
                if (item < data.axis) {
                    const Bits bits = static_cast<Bits>(words[item / itemsPerWord] >>
                                                        ((item % itemsPerWord) * valueBits));
                    const uint32_t rank = Rank(bits) ^ directionMask;
                    if (rank >= worstRank) {
                        worstRank = rank;
                        skipped = item;
                    }
                }
            }
        }
        auto packedOutput = result.template ReinterpretCast<uint64_t>();
#pragma unroll
        for (uint32_t word = 0; word < wordCount; ++word)
            packedOutput.SetValue(word, ScalarCompactedWord<valueBits>(words, word, skipped));
        for (uint32_t item = 0; item < data.k; ++item) {
            const uint32_t index = item + static_cast<uint32_t>(item >= skipped);
            wide.SetValue(item, static_cast<int64_t>(index));
        }
    }

    // 按实际dtype选择指数和符号掩码；NaN载荷只参与原值恢复，不参与数值排序。
    __aicore__ inline uint32_t Rank(Bits bits)
    {
        if constexpr (std::is_same<T, float>::value)
            return ScalarRank<FP32_SIGN_MASK, FP32_EXPONENT_MASK, UINT32_MAX>(bits);
        else if constexpr (std::is_same<T, half>::value)
            return ScalarRank<FP16_SIGN_MASK, FP16_EXPONENT_MASK, UINT16_MAX>(bits);
        else
            return ScalarRank<FP16_SIGN_MASK, BF16_EXPONENT_MASK, UINT16_MAX>(bits);
    }
};
} // namespace SelectTopK
#endif
