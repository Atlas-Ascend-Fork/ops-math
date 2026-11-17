/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SORT_WITH_INDEX_MERGE_STREAM_NATIVE_ARCH22_H
#define SORT_WITH_INDEX_MERGE_STREAM_NATIVE_ARCH22_H
#include "merge_sort_with_index_base.h"

namespace SortWithIndexA2 {
// 每核独立处理整行，四个有序段通过固定大小的 UB 窗口流式归并，无需 CoRank 或核间同步。
template <typename T>
class MergeSortWithIndexStreamNative : public MergeSortWithIndexBase<T> {
public:
    __aicore__ inline void Process()
    {
        for (uint64_t row = GetBlockIdx(); row < this->rows_; row += GetBlockNum()) {
            if constexpr (AscendC::IsSameType<T, half>::value) {
                CacheFp16RankScores(row);
                for (uint32_t part = 0; part < STREAM_MERGE_RUNS; ++part) {
                    NativeLocalSort(row, part);
                }
                this->template NativeStreamMerge<true>(row);
            } else if (CacheNativeScores(row)) {
                for (uint32_t part = 0; part < STREAM_MERGE_RUNS; ++part) {
                    NativeLocalSort(row, part);
                }
                NativeStreamMerge(row);
            } else {
                // FP16 已在编译期分支处理，此处仅处理 BF16 含 NaN 的情况。
                CacheNanRankScores();
                for (uint32_t part = 0; part < STREAM_MERGE_RUNS; ++part) {
                    NativeLocalSort(row, part);
                }
                this->template NativeStreamMerge<true>(row);
            }
        }
    }

private:
    // arena 前 16*chunk 字节用于局部排序，后 16*chunk 字节缓存 FP32 编码。
    // small 按字节划分：mask[0,m)、flags[m,2m)、scratch[2m,16352)、result[16352,16384)，m 为掩码字节数。
    __aicore__ inline bool CacheNativeScores(uint64_t row)
    {
        const uint32_t aligned = AlignUp(this->length_, SIMD_ALIGN_ELEMENTS);
        const uint32_t packedWords = aligned / 16;
        const uint32_t maskBytes = AlignUp(aligned / 8, UB_BLOCK_BYTES);
        auto raw = this->template At<uint16_t>();
        auto scores = this->template At<float>(this->total_ * sizeof(float));
        auto mask = this->small_.template Get<uint8_t>();
        auto flags = this->small_.template Get<half>()[maskBytes / sizeof(half)];
        auto scratch = this->small_.template Get<half>()[2 * maskBytes / sizeof(half)];
        auto result = this->small_.template Get<half>()[(STREAM_MERGE_PAYLOAD_ELEMENTS * INDEX_BYTES - UB_BLOCK_BYTES) /
                                                        sizeof(half)];
        DataCopyPad(raw, this->input_[row * this->length_], {1, this->length_ * VALUE_BYTES, 0, 0, 0},
                    {false, 0, 0, 0});
        EventSync<HardEvent::MTE2_V>();
        Duplicate(scores.template ReinterpretCast<uint32_t>(), uint32_t(0xff800000), this->total_);
        PipeBarrier<PIPE_V>();
        Cast(scores, raw.template ReinterpretCast<T>(), RoundMode::CAST_NONE, this->length_);
        PipeBarrier<PIPE_V>();
        Compare(mask, scores, scores, CMPMODE::EQ, aligned);
        PipeBarrier<PIPE_V>();
        Not(mask.template ReinterpretCast<int16_t>(), mask.template ReinterpretCast<int16_t>(), packedWords);
        PipeBarrier<PIPE_V>();
        Cast(flags, mask.template ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, packedWords);
        PipeBarrier<PIPE_V>();
        Abs(flags, flags, packedWords);
        PipeBarrier<PIPE_V>();
        ReduceMax(result, flags, scratch, packedWords, false);
        EventSync<HardEvent::V_S>();
        const bool noNan = result.template ReinterpretCast<uint16_t>().GetValue(0) == 0;
        EventSync<HardEvent::S_V>();
        return noNan;
    }

    // FP16 原始位模式映射为带符号的精确排序编码，保留 NaN 符号、有效载荷及正负零，无需单独扫描 NaN 或组合位置。
    // 无效记录使用独立的 FP32 负无穷哨兵。
    __aicore__ inline void CacheFp16RankScores(uint64_t row)
    {
        auto raw = this->template At<uint16_t>();
        auto scores = this->template At<float>(this->total_ * sizeof(float));
        DataCopyPad(raw, this->input_[row * this->length_], {1, this->length_ * VALUE_BYTES, 0, 0, 0},
                    {false, 0, 0, 0});
        EventSync<HardEvent::MTE2_V>();
        Duplicate(scores.template ReinterpretCast<uint32_t>(), uint32_t(0xff800000), this->total_);
        PipeBarrier<PIPE_V>();
        CacheNanRankScores();
    }

    // 从 arena 起始处仍保留的原始数据构造带符号排序编码，保留正负零；BF16 的所有 NaN 统一映射为 +32767。
    __aicore__ inline void CacheNanRankScores()
    {
        const uint32_t aligned = AlignUp(this->length_, SIMD_ALIGN_ELEMENTS);
        const uint32_t maskBytes = AlignUp(aligned / 8, UB_BLOCK_BYTES);
        auto raw = this->template At<uint16_t>();
        auto magnitude = this->template At<uint16_t>(aligned * VALUE_BYTES);
        auto scores = this->template At<float>(this->total_ * sizeof(float));
        auto negativeScores = this->template At<float>();
        auto negativeMask = this->small_.template Get<uint8_t>();
        auto nanMask = this->small_.template Get<uint8_t>()[maskBytes];
        // 按有符号 INT16 解释原始位模式，检测符号时不对 NaN 做浮点运算。
        Cast(scores, raw.template ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, this->length_);
        PipeBarrier<PIPE_V>();
        Compares<float, uint8_t>(negativeMask, scores, 0.0f, CMPMODE::LT, aligned);
        Duplicate(magnitude, uint16_t(ABS_MASK), this->length_);
        PipeBarrier<PIPE_V>();
        And(magnitude, raw, magnitude, this->length_);
        PipeBarrier<PIPE_V>();
        Cast(scores, magnitude.template ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, this->length_);
        PipeBarrier<PIPE_V>();
        if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
            Compares<float, uint8_t>(nanMask, scores, float(BF16_INFINITY_BITS), CMPMODE::GT, aligned);
        }
        // 原始数据和绝对值编码不再使用，低地址区域可复用为负排序编码区。
        Muls(negativeScores, scores, -1.0f, this->length_);
        PipeBarrier<PIPE_V>();
        Select(scores, negativeMask, negativeScores, scores, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->length_);
        PipeBarrier<PIPE_V>();
        if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
            Duplicate(negativeScores, float(ABS_MASK), this->length_);
            PipeBarrier<PIPE_V>();
            Select(scores, nanMask, negativeScores, scores, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->length_);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void DecodeNanRankValues(LocalTensor<uint16_t> values, LocalTensor<float> scores, uint32_t count)
    {
        constexpr uint32_t batch = STREAM_MERGE_PAYLOAD_ELEMENTS / 2;
        auto signs = this->template At<uint16_t>(batch * VALUE_BYTES);
        auto signMask = this->template At<uint16_t>(2 * batch * VALUE_BYTES);
        uint64_t selected = 0;
        // Abs 前提取 FP32 符号位，确保还原时保留负零。
        GatherMask(signs, scores.template ReinterpretCast<uint16_t>(), uint8_t(2), true, 2 * count, {1, 1, 0, 0},
                   selected);
        Duplicate(signMask, uint16_t(SIGN_MASK), count);
        PipeBarrier<PIPE_V>();
        And(signs, signs, signMask, count);
        Abs(scores, scores, count);
        PipeBarrier<PIPE_V>();
        Cast(values.template ReinterpretCast<int16_t>(), scores, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();
        Or(values, values, signs, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void NativeLocalSort(uint64_t row, uint32_t part)
    {
        const uint32_t off = part * this->chunk_;
        const uint32_t count = off < this->length_ ?
                                   (this->length_ - off < this->chunk_ ? this->length_ - off : this->chunk_) :
                                   0;
        auto scores = this->template At<float>(sizeof(float) * this->chunk_);
        auto tokens = this->template At<int32_t>();
        auto ping = this->template At<float>(MERGE_PROPOSAL_BYTES * this->chunk_);
        auto pong = this->template At<float>();
        DataCopy(scores, this->template At<float>(this->total_ * sizeof(float))[off], this->chunk_);
        PipeBarrier<PIPE_V>();
        if (!this->descending_) {
            Adds(scores.template ReinterpretCast<int32_t>(), scores.template ReinterpretCast<int32_t>(),
                 int32_t(-2147483647 - 1), count);
            PipeBarrier<PIPE_V>();
        }
        Duplicate(tokens, int32_t(0), this->chunk_);
        EventSync<HardEvent::V_MTE2>();
        DataCopyPad(tokens, this->inputIndex_[row * this->length_ + off], {1, count * INDEX_BYTES, 0, 0, 0},
                    {false, 0, 0, 0});
        EventSync<HardEvent::MTE2_V>();
        Sort32(ping, scores, tokens.template ReinterpretCast<uint32_t>(), this->chunk_ / MERGE_SORT_RUN_ELEMENTS);
        PipeBarrier<PIPE_V>();
        uint32_t runs = this->chunk_ / MERGE_SORT_RUN_ELEMENTS;
        uint32_t runLength = MERGE_SORT_RUN_ELEMENTS;
        bool inPing = true;
        while (runs > 1) {
            MergeStage(inPing ? pong : ping, inPing ? ping : pong, this->chunk_, runLength, runs);
            PipeBarrier<PIPE_ALL>();
            inPing = !inPing;
            runLength *= MERGE_LIST_COUNT;
            runs = (runs + MERGE_LIST_COUNT - 1) / MERGE_LIST_COUNT;
        }
        EventSync<HardEvent::V_MTE3>();
        DataCopy(this->pingGm_[off * MERGE_PROPOSAL_ELEMENTS], inPing ? ping : pong,
                 this->chunk_ * MERGE_PROPOSAL_ELEMENTS);
        PipeBarrier<PIPE_ALL>();
    }

    template <bool rankScores = false>
    __aicore__ inline void NativeStreamMerge(uint64_t row)
    {
        uint32_t consumed[STREAM_MERGE_RUNS] = {0, 0, 0, 0};
        uint32_t written = 0;
        const uint32_t window = this->chunk_ / 2 < STREAM_MERGE_WINDOW ? this->chunk_ / 2 : STREAM_MERGE_WINDOW;
        const uint32_t outputCapacity = window * STREAM_MERGE_RUNS;
        auto output = this->template At<float>(outputCapacity * MERGE_PROPOSAL_BYTES);
        constexpr uint32_t extractBatch = STREAM_MERGE_PAYLOAD_ELEMENTS / 2;
        auto scores = this->small_.template Get<float>();
        auto positions = this->small_.template Get<uint32_t>()[extractBatch];
        while (written < this->total_) {
            LocalTensor<float> inputs[MERGE_LIST_COUNT];
            uint16_t counts[MERGE_LIST_COUNT] = {0, 0, 0, 0};
            uint32_t ids[MERGE_LIST_COUNT], active = 0;
            // 前一次值搬出仍读取 arena 起始区域，补充窗口前须等待搬出完成。
            EventSync<HardEvent::MTE3_MTE2>();
            EventSync<HardEvent::V_MTE2>();
            for (uint32_t part = 0; part < STREAM_MERGE_RUNS; ++part) {
                uint32_t remaining = this->chunk_ - consumed[part];
                if (remaining) {
                    inputs[active] = this->template At<float>(active * window * MERGE_PROPOSAL_BYTES);
                    counts[active] = remaining < window ? remaining : window;
                    DataCopyPad(inputs[active],
                                this->pingGm_[(part * this->chunk_ + consumed[part]) * MERGE_PROPOSAL_ELEMENTS],
                                {1, uint32_t(counts[active]) * MERGE_PROPOSAL_BYTES, 0, 0, 0}, {false, 0, 0, 0});
                    ids[active++] = part;
                }
            }
            EventSync<HardEvent::MTE2_V>();
            EventSync<HardEvent::MTE3_V>();
            uint32_t merged[MERGE_LIST_COUNT] = {0, 0, 0, 0};
            if (active == 1) {
                merged[0] = counts[0];
                DataCopy(output, inputs[0],
                         AlignUp(uint32_t(counts[0]), UB_BLOCK_BYTES / MERGE_PROPOSAL_BYTES) * MERGE_PROPOSAL_ELEMENTS);
            } else {
                for (uint32_t list = active; list < MERGE_LIST_COUNT; ++list) {
                    inputs[list] = inputs[0];
                }
                MrgSortSrcList<float> lists(inputs[0], inputs[1], inputs[2], inputs[3]);
                MrgSort<float, true>(output, lists, counts, merged, (1U << active) - 1, 1);
            }
            uint32_t count = 0;
            for (uint32_t list = 0; list < active; ++list) {
                consumed[ids[list]] += merged[list];
                count += merged[list];
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t begin = 0; begin < count; begin += extractBatch) {
                uint32_t part = count - begin < extractBatch ? count - begin : extractBatch;
                EventSync<HardEvent::MTE3_V>();
                Extract(scores, positions, output[begin * MERGE_PROPOSAL_ELEMENTS],
                        AlignUp(part, MERGE_SORT_RUN_ELEMENTS) / MERGE_SORT_RUN_ELEMENTS);
                PipeBarrier<PIPE_V>();
                const uint32_t destination = written + begin;
                const uint32_t valid = destination < this->length_ ?
                                           (this->length_ - destination < part ? this->length_ - destination : part) :
                                           0;
                if (valid) {
                    if (!this->descending_) {
                        Adds(scores.template ReinterpretCast<int32_t>(), scores.template ReinterpretCast<int32_t>(),
                             int32_t(-2147483647 - 1), valid);
                        PipeBarrier<PIPE_V>();
                    }
                    auto values = this->template At<T>();
                    if constexpr (rankScores) {
                        DecodeNanRankValues(values.template ReinterpretCast<uint16_t>(), scores, valid);
                    } else {
                        Cast(values, scores, RoundMode::CAST_RINT, valid);
                    }
                    EventSync<HardEvent::V_MTE3>();
                    DataCopyPad(this->output_[row * this->length_ + destination],
                                values.template ReinterpretCast<uint16_t>(), {1, valid * VALUE_BYTES, 0, 0, 0});
                    DataCopyPad(this->outputIndex_[row * this->length_ + destination],
                                positions.template ReinterpretCast<int32_t>(), {1, valid * INDEX_BYTES, 0, 0, 0});
                }
            }
            written += count;
        }
        PipeBarrier<PIPE_ALL>();
    }
};
} // namespace SortWithIndexA2
#endif // SORT_WITH_INDEX_MERGE_STREAM_NATIVE_ARCH22_H
