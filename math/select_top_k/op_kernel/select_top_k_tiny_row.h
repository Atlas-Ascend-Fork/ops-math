/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SELECT_TOP_K_TINY_ROW_H_
#define SELECT_TOP_K_TINY_ROW_H_
#include "select_top_k_tiling_data.h"
#include "select_top_k_base_common.h"
namespace SelectTopK {
using namespace SelectTopKBase;
// 标准单行按编译容量选择排序层次；中短FP32直接解码排序值，其他类型Gather原始位值。
template <typename T, uint32_t MaximumItems = TINY_ROW_ITEMS>
class TinyRow {
    using Bits = typename std::conditional<sizeof(T) == VALUE16_BYTES, uint16_t, uint32_t>::type;

public:
    // Host只为连续标准单行选择本入口，索引不需要偏移或额外输出恢复。
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR values, GM_ADDR indices, const SelectTopKBaseTilingData& data)
    {
        GlobalTensor<Bits> input, valueOutput;
        GlobalTensor<int64_t> indexOutput;
        input.SetGlobalBuffer((__gm__ Bits*)x);
        valueOutput.SetGlobalBuffer((__gm__ Bits*)values);
        indexOutput.SetGlobalBuffer((__gm__ int64_t*)indices);
        // 本入口独占UB且没有队列；按Host已核验的布局建立视图，无需注册动态缓冲。
        LocalTensor<uint8_t> arena(TPosition::VECCALC, 0, data.ubBytes);
        constexpr uint32_t slots = MaximumItems;
        constexpr uint32_t keySlots = (slots + FLOAT_VECTOR_ELEMENTS - 1) / FLOAT_VECTOR_ELEMENTS *
                                      FLOAT_VECTOR_ELEMENTS;
        auto raw = arena.ReinterpretCast<T>();
        auto key = arena.ReinterpretCast<float>()[slots * sizeof(T) / VALUE32_BYTES];
        auto index = key[keySlots].template ReinterpretCast<int32_t>();
        auto ping = index[slots].template ReinterpretCast<float>();
        auto pong = ping[slots * PAIR_COMPONENTS];
        const uint32_t padded = MaximumItems == SORT32_ELEMENTS ?
                                    SORT32_ELEMENTS :
                                    static_cast<uint32_t>(DivUp(data.axis, SORT32_ELEMENTS) * SORT32_ELEMENTS);
        auto inputBuffer = raw.template ReinterpretCast<Bits>();
        if constexpr (DIRECT_FLOAT_OUTPUT)
            inputBuffer = key.template ReinterpretCast<Bits>();
        DataCopyPad(inputBuffer, input, DataCopyExtParams{1, static_cast<uint32_t>(data.axis * sizeof(T)), 0, 0, 0},
                    DataCopyPadExtParams<Bits>{false, 0, 0, 0});
        CreateRowIndex(index);
        // 完整覆盖键区时无需预填充；先建立索引，避免索引的S/V事件等待无关Duplicate。
        if constexpr (!DIRECT_FLOAT_OUTPUT) {
            if (data.axis != keySlots) {
                Duplicate(key, SortPadding(), keySlots);
                PipeBarrier<PIPE_V>();
            }
        }
        Fence<HardEvent::MTE2_V>();
        if constexpr (DIRECT_FLOAT_OUTPUT)
            InitializeFloatTail(key, data.axis);
        if constexpr (MaximumItems == SORT32_ELEMENTS)
            EncodeSingleRun(key, raw, pong, data.axis, data.largest);
        else
            Encode(key, raw, pong, data.axis, data.largest);
        Sort32(ping, key, index.template ReinterpretCast<uint32_t>(), padded / SORT32_ELEMENTS);
        PipeBarrier<PIPE_V>();
        auto sorted = ping;
        if constexpr (MaximumItems > TINY_ROW_ITEMS) {
            sorted = MergeRowRuns(ping, pong, padded);
        } else if constexpr (MaximumItems > SORT32_ELEMENTS) {
            if (padded > SORT32_ELEMENTS) {
                uint16_t lengths[MERGE_WAYS] = {SORT32_ELEMENTS, SORT32_ELEMENTS, SORT32_ELEMENTS, SORT32_ELEMENTS};
                MrgSortSrcList<float> parts(ping, ping[SORT32_ELEMENTS * PAIR_COMPONENTS],
                                            ping[SORT32_ELEMENTS * MERGE_THIRD_SOURCE * PAIR_COMPONENTS],
                                            ping[SORT32_ELEMENTS * MERGE_FOURTH_SOURCE * PAIR_COMPONENTS]);
                MrgSort4Info info(lengths, false, static_cast<uint16_t>((1U << (padded / SORT32_ELEMENTS)) - 1), 1);
                MrgSort(pong, parts, info);
                PipeBarrier<PIPE_V>();
                sorted = pong;
            }
        }
        if constexpr (DIRECT_FLOAT_OUTPUT) {
            EmitFloatPairs(input, valueOutput, indexOutput, sorted, raw.template ReinterpretCast<float>(), index, pong,
                           data);
            return;
        }
        uint64_t selected = 0;
        GatherMask(index.template ReinterpretCast<uint32_t>(), sorted.template ReinterpretCast<uint32_t>(),
                   uint8_t(GATHER_INDEX_PATTERN), true, data.k * PAIR_COMPONENTS,
                   GatherMaskParams(1, 1, VECTOR_REPEAT_BLOCKS, 0), selected);
        PipeBarrier<PIPE_V>();
        auto offsets = ping.template ReinterpretCast<uint32_t>();
        auto result = key.template ReinterpretCast<Bits>();
        // GatherMask已消费有序pair；独立pong存放宽索引，避免覆盖ping中的Gather字节偏移。
        auto wide = pong.template ReinterpretCast<int64_t>();
        if constexpr (MaximumItems == SORT32_ELEMENTS) {
            uint64_t outputMask[VECTOR_MASK_WORDS] = {(uint64_t{1} << data.k) - 1, 0};
            Muls(offsets.template ReinterpretCast<int32_t>(), index, static_cast<int32_t>(sizeof(T)), outputMask, 1,
                 UnaryRepeatParams{1, 1, VECTOR_REPEAT_BLOCKS, VECTOR_REPEAT_BLOCKS});
            Cast(wide, index, RoundMode::CAST_NONE, outputMask, 1,
                 UnaryRepeatParams{1, 1, VECTOR_REPEAT_BLOCKS, VECTOR_REPEAT_BLOCKS / WIDENING_RATIO});
            PipeBarrier<PIPE_V>();
            Gather(result, raw.template ReinterpretCast<Bits>(), offsets, 0, outputMask, 1, VECTOR_REPEAT_BLOCKS);
            ResetMask();
        } else {
            Muls(offsets.template ReinterpretCast<int32_t>(), index, static_cast<int32_t>(sizeof(T)), data.k);
            Cast(wide, index, RoundMode::CAST_NONE, data.k);
            PipeBarrier<PIPE_V>();
            Gather(result, raw.template ReinterpretCast<Bits>(), offsets, 0, data.k);
        }
        // V_MTE3已等待全部向量结果，无需在同一依赖前额外插入PIPE_V。
        Fence<HardEvent::V_MTE3>();
        DataCopyPad(valueOutput, result, DataCopyExtParams{1, static_cast<uint32_t>(data.k * sizeof(T)), 0, 0, 0});
        DataCopyPad(indexOutput, wide, DataCopyExtParams{1, static_cast<uint32_t>(data.k * INDEX64_BYTES), 0, 0, 0});
        // 普通入口由RunTinyRow等待全部流水；超级核在框架上下文释放前显式等待写出。
#if defined(__ASCENDC_ENABLE_SUPER_KERNEL__)
        Fence<HardEvent::MTE3_S>();
#endif
    }

private:
    static constexpr bool DIRECT_FLOAT_OUTPUT = std::is_same<T, float>::value && MaximumItems == SINGLE_ROW_MAX_AXIS;
    static constexpr uint8_t GATHER_VALUE_PATTERN = 1; // 从交错pair中提取偶数位的FP32值。
    static constexpr uint32_t VECTOR_MASK_WORDS = 2;   // 位模式掩码由高、低两个64位字组成。
    static constexpr uint32_t WIDENING_RATIO = 2;      // 16→32和32→64转换的位宽比例。
    static constexpr uint32_t PACKED_INDEX_ITEMS = sizeof(uint64_t) / sizeof(int32_t);
    static constexpr uint32_t PARALLEL_INDEX_GROWTH = 4; // 每层独立生成三段，共扩展为四倍前缀。
    static_assert(MaximumItems >= INDEX_SEQUENCE_SEED * INDEX_SEQUENCE_GROWTH && MaximumItems <= SINGLE_ROW_MAX_AXIS &&
                  (MaximumItems & (MaximumItems - 1)) == 0);

    // 输入直接落入键区后，只填最后一个向量内的无效槽位，不覆盖任何有效原值。
    __aicore__ inline void InitializeFloatTail(const LocalTensor<float>& key, uint32_t count)
    {
        const uint32_t tail = count % FLOAT_VECTOR_ELEMENTS;
        if (tail != 0) {
            uint64_t mask[VECTOR_MASK_WORDS] = {UINT64_MAX << tail, 0};
            Duplicate(key[count - tail], SortPadding(), mask, 1, 1, VECTOR_REPEAT_BLOCKS);
            PipeBarrier<PIPE_V>();
            ResetMask();
        }
    }

    // FP32无需往返类型转换；仅NaN载荷从原GM按索引恢复，有限值和符号零按原位写回。
    __aicore__ inline void EmitFloatPairs(const GlobalTensor<Bits>& input, const GlobalTensor<Bits>& valueOutput,
                                          const GlobalTensor<int64_t>& indexOutput, const LocalTensor<float>& pairs,
                                          const LocalTensor<float>& result, const LocalTensor<int32_t>& index,
                                          const LocalTensor<float>& wideBuffer, const SelectTopKBaseTilingData& data)
    {
        uint64_t selected = 0;
        GatherMask(result, pairs, GATHER_VALUE_PATTERN, true, data.k * PAIR_COMPONENTS,
                   GatherMaskParams(1, 1, VECTOR_REPEAT_BLOCKS, 0), selected);
        GatherMask(index.template ReinterpretCast<uint32_t>(), pairs.template ReinterpretCast<uint32_t>(),
                   uint8_t(GATHER_INDEX_PATTERN), true, data.k * PAIR_COMPONENTS,
                   GatherMaskParams(1, 1, VECTOR_REPEAT_BLOCKS, 0), selected);
        PipeBarrier<PIPE_V>();
        if (!data.largest)
            Adds(result.template ReinterpretCast<int32_t>(), result.template ReinterpretCast<int32_t>(), INT32_MIN,
                 data.k);
        auto wide = wideBuffer.template ReinterpretCast<int64_t>();
        Cast(wide, index, RoundMode::CAST_NONE, data.k);
        Fence<HardEvent::V_S>();
        auto resultBits = result.template ReinterpretCast<uint32_t>();
        for (uint32_t item = 0; item < data.k; ++item) {
            const uint32_t position = data.largest ? item : data.k - 1 - item;
            if ((resultBits.GetValue(position) & FP32_ABS_BITS) <= FP32_POS_INF_BITS)
                break;
            resultBits.SetValue(position, input.GetValue(static_cast<uint32_t>(index.GetValue(position))));
        }
        // V_S已等待两份向量输出，S_MTE3再等待NaN补写；两份DMA都在这条依赖之后。
        Fence<HardEvent::S_MTE3>();
        DataCopyPad(valueOutput, resultBits,
                    DataCopyExtParams{1, static_cast<uint32_t>(data.k * VALUE32_BYTES), 0, 0, 0});
        DataCopyPad(indexOutput, wide, DataCopyExtParams{1, static_cast<uint32_t>(data.k * INDEX64_BYTES), 0, 0, 0});
#if defined(__ASCENDC_ENABLE_SUPER_KERNEL__)
        Fence<HardEvent::MTE3_S>();
#endif
    }

    // 按固定容量展开索引倍增：小于一条向量时用位掩码，其余使用完整repeat。
    template <uint32_t Generated>
    __aicore__ inline void ExtendRowIndex(const LocalTensor<int32_t>& index)
    {
        uint64_t mask[VECTOR_MASK_WORDS] = {UINT64_MAX, 0};
        if constexpr (Generated < FLOAT_VECTOR_ELEMENTS)
            mask[0] = (uint64_t{1} << Generated) - 1;
        constexpr uint8_t repeats = Generated < FLOAT_VECTOR_ELEMENTS ? 1 : Generated / FLOAT_VECTOR_ELEMENTS;
        Adds(index[Generated], index, static_cast<int32_t>(Generated), mask, repeats,
             UnaryRepeatParams{1, 1, VECTOR_REPEAT_BLOCKS, VECTOR_REPEAT_BLOCKS});
        PipeBarrier<PIPE_V>();
        if constexpr (Generated * INDEX_SEQUENCE_GROWTH < MaximumItems)
            ExtendRowIndex<Generated * INDEX_SEQUENCE_GROWTH>(index);
    }

    // 2048项入口从同一已就绪前缀独立生成三段，层末同步一次，将索引生成依赖从八层降为四层。
    template <uint32_t Generated>
    __aicore__ inline void ExtendWideRowIndex(const LocalTensor<int32_t>& index)
    {
        uint64_t mask = UINT64_MAX;
        if constexpr (Generated < FLOAT_VECTOR_ELEMENTS)
            mask = (uint64_t{1} << Generated) - 1;
        constexpr uint64_t maskCount = Generated < FLOAT_VECTOR_ELEMENTS ? Generated : FLOAT_VECTOR_ELEMENTS;
        constexpr uint8_t repeats = Generated < FLOAT_VECTOR_ELEMENTS ? 1 : Generated / FLOAT_VECTOR_ELEMENTS;
        SetVectorMask<int32_t>(0, mask);
#pragma unroll
        for (uint32_t part = 1; part < PARALLEL_INDEX_GROWTH; ++part) {
            Adds<int32_t, false>(index[part * Generated], index, static_cast<int32_t>(part * Generated), maskCount,
                                 repeats, UnaryRepeatParams{1, 1, VECTOR_REPEAT_BLOCKS, VECTOR_REPEAT_BLOCKS});
        }
        PipeBarrier<PIPE_V>();
        if constexpr (Generated * PARALLEL_INDEX_GROWTH < MaximumItems)
            ExtendWideRowIndex<Generated * PARALLEL_INDEX_GROWTH>(index);
    }

    // 新UB尚无向量写者；成对写首个索引块，等待标量写入后扩展到编译期容量。
    __aicore__ inline void CreateRowIndex(const LocalTensor<int32_t>& index)
    {
        auto packed = index.template ReinterpretCast<uint64_t>();
        for (uint32_t pair = 0; pair < INDEX_SEQUENCE_SEED / PACKED_INDEX_ITEMS; ++pair) {
            const uint32_t first = pair * PACKED_INDEX_ITEMS;
            packed.SetValue(pair, uint64_t{first} | (uint64_t{first + 1} << MASK_WORD32_BITS));
        }
        Fence<HardEvent::S_V>();
        SetMaskNorm();
        if constexpr (MaximumItems == SINGLE_ROW_MAX_AXIS)
            ExtendWideRowIndex<INDEX_SEQUENCE_SEED>(index);
        else
            ExtendRowIndex<INDEX_SEQUENCE_SEED>(index);
        ResetMask();
    }

    // 单Sort32只处理有效N项；用一次repeat的位掩码避免反复切换计数模式，填充键保持不变。
    __aicore__ inline void EncodeSingleRun(const LocalTensor<float>& key, const LocalTensor<T>& raw,
                                           const LocalTensor<float>& scratch, uint32_t count, bool largest)
    {
        uint64_t validMask[VECTOR_MASK_WORDS] = {(uint64_t{1} << count) - 1, 0};
        if constexpr (std::is_same<T, float>::value)
            Adds(key.template ReinterpretCast<int32_t>(), raw.template ReinterpretCast<int32_t>(), int32_t(0),
                 validMask, 1, UnaryRepeatParams{1, 1, VECTOR_REPEAT_BLOCKS, VECTOR_REPEAT_BLOCKS});
        else
            Cast(key, raw, RoundMode::CAST_NONE, validMask, 1,
                 UnaryRepeatParams{1, 1, VECTOR_REPEAT_BLOCKS, VECTOR_REPEAT_BLOCKS / WIDENING_RATIO});
        PipeBarrier<PIPE_V>();
        auto nanMask = scratch.template ReinterpretCast<uint8_t>();
        Compare(nanMask, key, key, CMPMODE::EQ, FLOAT_VECTOR_ELEMENTS);
        PipeBarrier<PIPE_V>();
        Select(key, nanMask, key, SortNaN(), SELMODE::VSEL_TENSOR_SCALAR_MODE, validMask, 1,
               BinaryRepeatParams{1, 1, 1, VECTOR_REPEAT_BLOCKS, VECTOR_REPEAT_BLOCKS, VECTOR_REPEAT_BLOCKS});
        PipeBarrier<PIPE_V>();
        if (!largest) {
            Adds(key.template ReinterpretCast<int32_t>(), key.template ReinterpretCast<int32_t>(), INT32_MIN, validMask,
                 1, UnaryRepeatParams{1, 1, VECTOR_REPEAT_BLOCKS, VECTOR_REPEAT_BLOCKS});
            PipeBarrier<PIPE_V>();
        }
        // 后续Sort32使用完整一组，不能继承有效N项的部分掩码。
        ResetMask();
    }

    // 固定段长的一层四路归并；满组批量执行，尾组仅使用真实长度。
    template <uint32_t Span>
    __aicore__ inline void MergeRowStage(LocalTensor<float> input, LocalTensor<float> output, uint32_t count)
    {
        constexpr uint32_t groupItems = Span * MERGE_WAYS;
        const uint32_t groups = count / groupItems;
        if (groups != 0) {
            uint16_t lengths[MERGE_WAYS] = {Span, Span, Span, Span};
            MrgSortSrcList<float> sources(input, input[Span * PAIR_COMPONENTS],
                                          input[Span * MERGE_THIRD_SOURCE * PAIR_COMPONENTS],
                                          input[Span * MERGE_FOURTH_SOURCE * PAIR_COMPONENTS]);
            MrgSort4Info info(lengths, false, MERGE_ALL_SOURCES_MASK, static_cast<uint8_t>(groups));
            MrgSort(output, sources, info);
        }
        const uint32_t start = groups * groupItems;
        const uint32_t tail = count - start;
        if (tail != 0 && tail <= Span) {
            DataCopy(output[start * PAIR_COMPONENTS].template ReinterpretCast<uint32_t>(),
                     input[start * PAIR_COMPONENTS].template ReinterpretCast<uint32_t>(), tail * PAIR_COMPONENTS);
        } else if (tail > Span) {
            LocalTensor<float> sources[MERGE_WAYS] = {input, input, input, input};
            uint16_t lengths[MERGE_WAYS] = {};
            uint32_t routes = 0;
            for (uint32_t part = 0; part < MERGE_WAYS; ++part) {
                const uint32_t offset = start + part * Span;
                if (offset < count) {
                    sources[part] = input[offset * PAIR_COMPONENTS];
                    lengths[part] = static_cast<uint16_t>(Min<uint32_t>(Span, count - offset));
                    ++routes;
                }
            }
            MrgSortSrcList<float> lists(sources[0], sources[1], sources[MERGE_THIRD_SOURCE],
                                        sources[MERGE_FOURTH_SOURCE]);
            MrgSort4Info info(lengths, false, static_cast<uint16_t>((1U << routes) - 1), 1);
            MrgSort(output[start * PAIR_COMPONENTS], lists, info);
        }
        PipeBarrier<PIPE_V>();
    }
    // 至多2048项只需32、128、512三层段长，编译时固定层次以减少循环和地址计算。
    __aicore__ inline LocalTensor<float> MergeRowRuns(LocalTensor<float> ping, LocalTensor<float> pong, uint32_t count)
    {
        constexpr uint32_t secondSpan = SORT32_ELEMENTS * MERGE_WAYS;
        constexpr uint32_t thirdSpan = secondSpan * MERGE_WAYS;
        static_assert(MaximumItems <= thirdSpan * MERGE_WAYS);
        MergeRowStage<SORT32_ELEMENTS>(ping, pong, count);
        if (count <= secondSpan)
            return pong;
        MergeRowStage<secondSpan>(pong, ping, count);
        if (count <= thirdSpan)
            return ping;
        MergeRowStage<thirdSpan>(ping, pong, count);
        return pong;
    }
    // 独立保存NaN掩码，键编码不覆盖稍后Gather需要的原始位数组。
    __aicore__ inline void Encode(const LocalTensor<float>& key, const LocalTensor<T>& raw,
                                  const LocalTensor<float>& scratch, uint32_t count, bool largest)
    {
        if constexpr (!DIRECT_FLOAT_OUTPUT) {
            if constexpr (std::is_same<T, float>::value)
                Adds(key.template ReinterpretCast<int32_t>(), raw.template ReinterpretCast<int32_t>(), int32_t(0),
                     count);
            else
                Cast(key, raw, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
        }
        auto mask = scratch.template ReinterpretCast<uint8_t>();
        const uint32_t compared = DivUp(count, FLOAT_VECTOR_ELEMENTS) * FLOAT_VECTOR_ELEMENTS;
        Compare(mask, key, key, CMPMODE::EQ, compared);
        PipeBarrier<PIPE_V>();
        Select(key, mask, key, SortNaN(), SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
        PipeBarrier<PIPE_V>();
        if (!largest) {
            Adds(key.template ReinterpretCast<int32_t>(), key.template ReinterpretCast<int32_t>(), INT32_MIN, count);
            PipeBarrier<PIPE_V>();
        }
    }
};
} // namespace SelectTopK
#endif
