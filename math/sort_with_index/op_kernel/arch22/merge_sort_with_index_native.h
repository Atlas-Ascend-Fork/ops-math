/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MERGE_SORT_WITH_INDEX_NATIVE_H
#define MERGE_SORT_WITH_INDEX_NATIVE_H
#include "merge_sort_with_index_base.h"
namespace SortWithIndexA2 {
// GM 归并各轮保留原生排序记录，避免逐窗口重建记录并重新读取全部值和索引。
template <class T, bool RankPayload = false>
class NativeMergeSortWithIndex : public MergeSortWithIndexBase<T> {
public:
    template <bool UseWorkspace = true>
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR outputIndex, GM_ADDR workspace,
                                const SortWithIndexTilingDataA2& data)
    {
        this->length_ = data.length;
        this->rows_ = data.rows;
        this->chunk_ = data.alignedLength;
        this->descending_ = data.descending & 1U;
        this->fanout_ = data.localFanout;
        contiguousRows_ = data.mergeGroupSpan != 0;
        span_ = data.mergeGroupSpan;
        rowsPerCore_ = data.rowsPerCore;
        lane_ = GetBlockIdx() % this->fanout_;
        groups_ = GetBlockNum() / this->fanout_;
        runs_ = data.tileCount;
        total_ = contiguousRows_ ? this->length_ : runs_ * this->chunk_;
        const uint32_t rankSpan = AlignUp((this->length_ + this->fanout_ - 1) / this->fanout_, 128);
        ownedBegin_ = lane_ * rankSpan;
        if (ownedBegin_ > this->length_)
            ownedBegin_ = this->length_;
        ownedEnd_ = ownedBegin_ + rankSpan;
        if (ownedEnd_ > this->length_)
            ownedEnd_ = this->length_;
        uint32_t bytes = NATIVE_MERGE_LOCAL_BYTES_PER_ELEMENT * this->chunk_;
        if (bytes < NATIVE_MERGE_ARENA_BYTES)
            bytes = NATIVE_MERGE_ARENA_BYTES;
        this->pipe_.InitBuffer(this->arena_, bytes);
        this->pipe_.InitBuffer(this->small_, NATIVE_MERGE_SMALL_BYTES);
        this->pipe_.InitBuffer(positionSeed_, NATIVE_MERGE_POSITION_SEED_BYTES);
        auto seed = positionSeed_.template Get<float>();
        for (uint32_t i = 0; i < 64; ++i)
            seed.SetValue(i, float(int32_t(i)));
        EventSync<HardEvent::S_V>();

        this->input_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(x));
        this->inputIndex_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index));
        this->output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(y));
        this->outputIndex_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(outputIndex));
        if constexpr (UseWorkspace) {
            workspace += uint64_t(GetBlockIdx() / this->fanout_) * AlignUp(total_, 16U) * 16;
            for (uint32_t s = 0; s < 2; ++s) {
                positions_[s].SetGlobalBuffer(
                    reinterpret_cast<__gm__ float*>(workspace + uint64_t(s) * AlignUp(total_, 16U) * 8));
                compactRecords_[s].SetGlobalBuffer(
                    reinterpret_cast<__gm__ uint32_t*>(workspace + uint64_t(s) * AlignUp(total_, 16U) * 8));
            }
        }
    }
    __aicore__ inline void Process()
    {
        if (!contiguousRows_) {
            for (uint64_t row = GetBlockIdx(); row < this->rows_; row += GetBlockNum()) {
                nativeValues_ = UseFiniteValues(row);
                halfRecords_ = nativeValues_ && AscendC::IsSameType<T, half>::value;
                ownedBegin_ = 0;
                ownedEnd_ = this->length_;
                for (uint32_t p = 0; p < runs_; ++p)
                    SortRun(row, p);
                uint32_t source = 0;
                mergeEnd_ = total_;
                for (uint32_t width = 1; width < runs_; width *= 4) {
                    bool final = width * 4 >= runs_;
                    for (uint32_t start = 0; start < runs_; start += width * 4) {
                        uint32_t lists = (runs_ - start + width - 1) / width;
                        if (lists > 4)
                            lists = 4;
                        Merge(row, source, start * this->chunk_, width * this->chunk_, lists, final);
                    }
                    source = 1 - source;
                }
                WriteValues(row, source);
            }
            return;
        }
        const uint32_t group = GetBlockIdx() / this->fanout_;
        for (uint32_t wave = 0; wave < rowsPerCore_; ++wave) {
            const uint64_t row = uint64_t(group) * rowsPerCore_ + wave;
            const bool valid = row < this->rows_;
            nativeValues_ = valid && UseFiniteValues(row);
            rankValues_ = RankPayload && valid && !nativeValues_;
            carryPayload_ = nativeValues_ || rankValues_;
            halfRecords_ = nativeValues_ && AscendC::IsSameType<T, half>::value;
            if (valid) {
                const uint32_t first = lane_ * span_;
                uint32_t end = first + span_;
                if (end > this->length_)
                    end = this->length_;
                for (uint32_t off = first; off < end; off += this->chunk_) {
                    overrideOff_ = off;
                    overrideCount_ = end - off < this->chunk_ ? end - off : this->chunk_;
                    SortRun(row, off / this->chunk_);
                }
                if (this->fanout_ > 1 && first < end) {
                    mergeEnd_ = end;
                    const uint32_t maximumRuns = (span_ + this->chunk_ - 1) / this->chunk_;
                    const uint32_t localRuns = (end - first + this->chunk_ - 1) / this->chunk_;
                    uint32_t source = 0;
                    for (uint32_t width = 1; width < maximumRuns; width *= 4) {
                        for (uint32_t start = 0; start < localRuns; start += width * 4) {
                            uint32_t lists = (localRuns - start + width - 1) / width;
                            if (lists > 4)
                                lists = 4;
                            Merge(row, source, first + start * this->chunk_, width * this->chunk_, lists, false);
                        }
                        source = 1 - source;
                    }
                }
            }
            if (this->fanout_ > 1)
                SyncAll();
            uint32_t source = 0;
            if (this->fanout_ > 1) {
                const uint32_t maximumRuns = (span_ + this->chunk_ - 1) / this->chunk_;
                for (uint32_t width = 1; width < maximumRuns; width *= 4)
                    source = 1 - source;
            }
            mergeEnd_ = total_;
            const uint32_t run = this->fanout_ > 1 ? span_ : this->chunk_;
            const uint32_t count = this->fanout_ > 1 ? this->fanout_ : runs_;
            if (valid && lane_ == 0) {
                // 即使只有一个有序段，也需通过最终 Merge 解出记录中的值和索引。
                if (count == 1 && carryPayload_) {
                    Merge(row, source, 0, run, 1, true);
                    source = 1 - source;
                }
                for (uint32_t width = 1; width < count; width *= 4) {
                    const bool final = width * 4 >= count;
                    for (uint32_t start = 0; start < count; start += width * 4) {
                        uint32_t lists = (count - start + width - 1) / width;
                        if (lists > 4)
                            lists = 4;
                        Merge(row, source, start * run, width * run, lists, final);
                    }
                    source = 1 - source;
                }
            } else {
                for (uint32_t width = 1; width < count; width *= 4)
                    source = 1 - source;
            }
            if (this->fanout_ > 1)
                SyncAll();
            if (valid)
                WriteValues(row, source);
            if (this->fanout_ > 1 && wave + 1 < rowsPerCore_)
                SyncAll();
        }
    }

protected:
    __aicore__ inline void SeedPositions(LocalTensor<float> positions, uint32_t count)
    {
        const uint32_t seed = count < 64 ? count : 64;
        DataCopy(positions, positionSeed_.template Get<float>(), seed);
        PipeBarrier<PIPE_V>();
        for (uint32_t filled = 64; filled < count; filled *= 2) {
            const uint32_t size = count - filled < filled ? count - filled : filled;
            Adds(positions[filled], positions, float(int32_t(filled)), size);
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline bool UseFiniteValues(uint64_t row)
    {
        constexpr uint32_t scanElements = STREAM_NATIVE_AXIS_LIMIT;
        const uint32_t arenaBytes = this->chunk_ * NATIVE_MERGE_LOCAL_BYTES_PER_ELEMENT > NATIVE_MERGE_ARENA_BYTES ?
                                        this->chunk_ * NATIVE_MERGE_LOCAL_BYTES_PER_ELEMENT :
                                        NATIVE_MERGE_ARENA_BYTES;
        cachedScores_ = false;
        if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
            if (contiguousRows_ && this->fanout_ == 1 && this->length_ <= scanElements) {
                return CacheNativeScores(row, arenaBytes);
            }
        }
        cachedInput_ = contiguousRows_ && this->fanout_ == 1 && this->length_ <= scanElements;
        cachedInputOffset_ = arenaBytes - AlignUp(this->length_, DMA_VALUE_ELEMENTS) * VALUE_BYTES;
        // 短行输入暂存于缓冲区高地址部分，全部 SortRun 消费后该区域才可供归并复用。
        auto raw = this->template At<uint16_t>(cachedInput_ ? cachedInputOffset_ : 0);
        auto codes = this->template At<uint16_t>(2 * scanElements);
        auto scratch = this->template At<half>(4 * scanElements);
        auto result = this->small_.template Get<half>();
        constexpr uint16_t infinityCode = AscendC::IsSameType<T, half>::value ? FP16_INFINITY_BITS : BF16_INFINITY_BITS;
        for (uint32_t off = 0; off < this->length_; off += scanElements) {
            const uint32_t count = this->length_ - off < scanElements ? this->length_ - off : scanElements;
            DataCopyPad(raw, this->input_[row * this->length_ + off], {1, count * VALUE_BYTES, 0, 0, 0},
                        {false, 0, 0, 0});
            EventSync<HardEvent::MTE2_V>();
            Duplicate(codes, uint16_t(ABS_MASK), count);
            PipeBarrier<PIPE_V>();
            And(codes, raw, codes, count);
            PipeBarrier<PIPE_V>();
            // ceil(absBits/2) 位于 [0,0x4000]，按 half 解释为有序有限值。
            // 无穷大的编码为偶数，因此最小 NaN 编码也严格排在其后，无需对 NaN 做浮点运算。
            Adds(codes.template ReinterpretCast<int16_t>(), codes.template ReinterpretCast<int16_t>(), int16_t(1),
                 count);
            PipeBarrier<PIPE_V>();
            ShiftRight(codes, codes, uint16_t(1), count);
            PipeBarrier<PIPE_V>();
            ReduceMax(result, codes.template ReinterpretCast<half>(), scratch, count, false);
            EventSync<HardEvent::V_S>();
            const bool hasNan = result.template ReinterpretCast<uint16_t>().GetValue(0) > infinityCode / 2;
            EventSync<HardEvent::S_V>();
            if (hasNan)
                return false;
        }
        return true;
    }
    __aicore__ inline bool CacheNativeScores(uint64_t row, uint32_t arenaBytes)
    {
        const uint32_t aligned = AlignUp(this->length_, SIMD_ALIGN_ELEMENTS);
        cachedInput_ = false;
        cachedScores_ = true;
        cachedInputOffset_ = arenaBytes - aligned * sizeof(float);
        auto raw = this->template At<uint16_t>();
        auto scores = this->template At<float>(cachedInputOffset_);
        auto mask = this->template At<uint8_t>(STREAM_NATIVE_AXIS_LIMIT * VALUE_BYTES);
        auto flags = this->template At<half>(STREAM_NATIVE_AXIS_LIMIT * INDEX_BYTES);
        auto scratch = this->template At<half>(STREAM_NATIVE_AXIS_LIMIT * INDEX_BYTES + STREAM_NATIVE_AXIS_LIMIT / 8);
        DataCopyPad(raw, this->input_[row * this->length_], {1, this->length_ * VALUE_BYTES, 0, 0, 0},
                    {false, 0, 0, 0});
        EventSync<HardEvent::MTE2_V>();
        Duplicate(scores.template ReinterpretCast<uint32_t>(), uint32_t(0xff800000), aligned);
        PipeBarrier<PIPE_V>();
        Cast(scores, raw.template ReinterpretCast<T>(), RoundMode::CAST_NONE, this->length_);
        PipeBarrier<PIPE_V>();
        // 对自身相等比较的结果取反，再归约压缩掩码，检测包括尾段在内的 NaN。
        // A2 有序 NE 比较对 NaN 返回假，因此不能直接用 NE 检测。
        Compare(mask, scores, scores, CMPMODE::EQ, aligned);
        PipeBarrier<PIPE_V>();
        Not(mask.template ReinterpretCast<int16_t>(), mask.template ReinterpretCast<int16_t>(), aligned / 16);
        PipeBarrier<PIPE_V>();
        Cast(flags, mask.template ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, aligned / 16);
        PipeBarrier<PIPE_V>();
        Abs(flags, flags, aligned / 16);
        PipeBarrier<PIPE_V>();
        auto result = this->small_.template Get<half>();
        ReduceMax(result, flags, scratch, aligned / 16, false);
        EventSync<HardEvent::V_S>();
        const bool finite = result.template ReinterpretCast<uint16_t>().GetValue(0) == 0;
        EventSync<HardEvent::S_V>();
        return finite;
    }
    __aicore__ inline void HalfMergeStage(LocalTensor<half> dst, LocalTensor<half> src, uint32_t total,
                                          uint32_t runLength, uint32_t runCount)
    {
        uint32_t fullGroups = runCount / MERGE_LIST_COUNT;
        uint32_t tailRuns = runCount % MERGE_LIST_COUNT;
        if (tailRuns == 0 && total != runCount * runLength) {
            --fullGroups;
            tailRuns = MERGE_LIST_COUNT;
        }
        uint32_t mergedCounts[MERGE_LIST_COUNT];
        if (fullGroups > 0) {
            uint16_t counts[MERGE_LIST_COUNT] = {uint16_t(runLength), uint16_t(runLength), uint16_t(runLength),
                                                 uint16_t(runLength)};
            uint32_t stride = GetSortOffset<half>(runLength);
            MrgSortSrcList<half> lists(src, src[stride], src[stride * 2], src[stride * 3]);
            MrgSort<half, false>(dst, lists, counts, mergedCounts, (1U << MERGE_LIST_COUNT) - 1, fullGroups);
        }
        uint32_t tailStart = fullGroups * MERGE_LIST_COUNT * runLength;
        if (tailStart == total) {
            return;
        }
        if (tailRuns == 1) {
            DataCopy(dst[GetSortOffset<half>(tailStart)], src[GetSortOffset<half>(tailStart)],
                     GetSortOffset<half>(total - tailStart));
            return;
        }
        uint16_t counts[MERGE_LIST_COUNT] = {0, 0, 0, 0};
        uint32_t offsets[MERGE_LIST_COUNT] = {tailStart, tailStart, tailStart, tailStart};
        for (uint32_t list = 0; list < tailRuns; ++list) {
            offsets[list] = tailStart + list * runLength;
            uint32_t remaining = total - offsets[list];
            counts[list] = remaining < runLength ? remaining : runLength;
        }
        MrgSortSrcList<half> lists(src[GetSortOffset<half>(offsets[0])], src[GetSortOffset<half>(offsets[1])],
                                   src[GetSortOffset<half>(offsets[2])], src[GetSortOffset<half>(offsets[3])]);
        MrgSort<half, false>(dst[GetSortOffset<half>(tailStart)], lists, counts, mergedCounts, (1U << tailRuns) - 1, 1);
    }

    __aicore__ inline LocalTensor<float> HalfLocalSort(uint64_t row, uint32_t part)
    {
        uint32_t n = this->chunk_, off = contiguousRows_ ? overrideOff_ : part * n;
        uint32_t count = contiguousRows_ ?
                             overrideCount_ :
                             (off < this->length_ ? (this->length_ - off < n ? this->length_ - off : n) : 0);
        const uint32_t sortCount = contiguousRows_ ? AlignUp(count, 32U) : n;
        auto positions = this->template At<int32_t>(2 * n);
        if (carryPayload_) {
            Duplicate(positions, int32_t(0), sortCount);
            EventSync<HardEvent::V_MTE2>();
            DataCopyPad(positions, this->inputIndex_[row * this->length_ + off], {1, count * INDEX_BYTES, 0, 0, 0},
                        {false, 0, 0, 0});
            EventSync<HardEvent::MTE2_V>();
        } else {
            SeedPositions(positions.template ReinterpretCast<float>(), sortCount);
            Cast(positions, positions.template ReinterpretCast<float>(), RoundMode::CAST_RINT, sortCount);
            PipeBarrier<PIPE_V>();
            Adds(positions, positions, int32_t(off), sortCount);
            PipeBarrier<PIPE_V>();
        }
        auto values = this->template At<half>();
        Duplicate(values.template ReinterpretCast<uint16_t>(), uint16_t(0xfc00), sortCount);
        PipeBarrier<PIPE_V>();
        if (count) {
            if (cachedInput_) {
                DataCopy(values.template ReinterpretCast<uint16_t>(),
                         this->template At<uint16_t>(cachedInputOffset_)[off], AlignUp(count, DMA_VALUE_ELEMENTS));
                PipeBarrier<PIPE_V>();
            } else {
                EventSync<HardEvent::V_MTE2>();
                DataCopyPad(values.template ReinterpretCast<uint16_t>(), this->input_[row * this->length_ + off],
                            {1, count * VALUE_BYTES, 0, 0, 0}, {false, 0, 0, 0});
                EventSync<HardEvent::MTE2_V>();
            }
        }
        if (count < sortCount) {
            EventSync<HardEvent::MTE2_S>();
            uint32_t aligned = AlignUp(count, 16);
            for (uint32_t i = count; i < aligned; ++i)
                values.template ReinterpretCast<uint16_t>().SetValue(i, uint16_t(0xfc00));
            EventSync<HardEvent::S_V>();
            if (aligned < sortCount)
                Duplicate(values.template ReinterpretCast<uint16_t>()[aligned], uint16_t(0xfc00), sortCount - aligned);
            PipeBarrier<PIPE_V>();
        }
        if (!this->descending_) {
            Adds(values.template ReinterpretCast<int16_t>(), values.template ReinterpretCast<int16_t>(),
                 int16_t(-32768), count);
            PipeBarrier<PIPE_V>();
        }
        auto ping = this->template At<half>(8 * n), pong = this->template At<half>();
        for (uint32_t block = 0; block < sortCount / 32; block += 255) {
            const uint32_t repeats = sortCount / 32 - block < 255 ? sortCount / 32 - block : 255;
            Sort32(ping[GetSortOffset<half>(block * 32)], values[block * 32],
                   positions.template ReinterpretCast<uint32_t>()[block * 32], repeats);
        }
        PipeBarrier<PIPE_V>();
        uint32_t runs = sortCount / 32, len = 32;
        bool inPing = true;
        while (runs > 1) {
            HalfMergeStage(inPing ? pong : ping, inPing ? ping : pong, sortCount, len, runs);
            PipeBarrier<PIPE_V>();
            inPing = !inPing;
            len *= 4;
            runs = (runs + 3) / 4;
        }
        return (inPing ? ping : pong).template ReinterpretCast<float>();
    }
    __aicore__ inline void NativeScores(uint32_t count)
    {
        auto score = this->template At<float>(this->SCORE_OFFSET * this->chunk_);
        auto temp = this->template At<float>(this->TEMP_OFFSET * this->chunk_);
        auto mask = this->small_.template Get<uint8_t>();
        Duplicate(score, MERGE_PADDING_SCORE, this->chunk_);
        PipeBarrier<PIPE_V>();
        Cast(score, this->template At<int16_t>(this->KEY_OFFSET * this->chunk_), RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Compares<float, uint8_t>(mask, score, 0.0f, CMPMODE::LT, AlignUp(count, 128));
        Adds(temp, score, 65536.0f, count);
        PipeBarrier<PIPE_V>();
        Select(score, mask, temp, score, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        PipeBarrier<PIPE_V>();
        Muls(score, score, -1.0f, count);
        PipeBarrier<PIPE_V>();
    }
    // 将原始 16 位编码映射为可精确还原的 FP32 排序编码；行内存在 NaN 时所有段统一使用该表示。
    // BF16 NaN 规范化为 0x7fff，FP16 保留符号和有效载荷位。
    __aicore__ inline void MakeRankScores(uint32_t count, uint32_t sortCount)
    {
        auto raw = this->template At<uint16_t>();
        auto magnitude = this->template At<uint16_t>(this->KEY_OFFSET * this->chunk_);
        auto scores = this->template At<float>(this->SCORE_OFFSET * this->chunk_);
        auto negative = this->template At<float>(this->TEMP_OFFSET * this->chunk_);
        auto negativeMask = this->small_.template Get<uint8_t>();
        // TEMP 占用 [8*A,12*A) 字节，后续区域在 Sort32 前可复用为第二份掩码，无需扩大 UB。
        auto nanMask = this->template At<uint8_t>(12 * this->chunk_);
        Duplicate(scores, MERGE_PADDING_SCORE, sortCount);
        PipeBarrier<PIPE_V>();
        Cast(scores, raw.template ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Compares<float, uint8_t>(negativeMask, scores, 0.0f, CMPMODE::LT, AlignUp(count, SIMD_ALIGN_ELEMENTS));
        Duplicate(magnitude, uint16_t(ABS_MASK), count);
        PipeBarrier<PIPE_V>();
        And(magnitude, raw, magnitude, count);
        PipeBarrier<PIPE_V>();
        Cast(scores, magnitude.template ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
            Compares<float, uint8_t>(nanMask, scores, float(BF16_INFINITY_BITS), CMPMODE::GT,
                                     AlignUp(count, SIMD_ALIGN_ELEMENTS));
        }
        Muls(negative, scores, -1.0f, count);
        PipeBarrier<PIPE_V>();
        Select(scores, negativeMask, negative, scores, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        PipeBarrier<PIPE_V>();
        if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
            Duplicate(negative, float(ABS_MASK), count);
            PipeBarrier<PIPE_V>();
            Select(scores, nanMask, negative, scores, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
            PipeBarrier<PIPE_V>();
        }
        if (!this->descending_) {
            Adds(scores.template ReinterpretCast<int32_t>(), scores.template ReinterpretCast<int32_t>(),
                 int32_t(-2147483647 - 1), count);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void DecodeRankValues(LocalTensor<uint16_t> values, LocalTensor<float> scores,
                                            LocalTensor<uint16_t> signs, uint32_t count)
    {
        uint64_t selected = 0;
        // 取绝对值前保留负零的符号位。
        GatherMask(signs, scores.template ReinterpretCast<uint16_t>(), uint8_t(2), true, 2 * count, {1, 1, 0, 0},
                   selected);
        PipeBarrier<PIPE_V>();
        ShiftRight(signs, signs, uint16_t(15), count);
        PipeBarrier<PIPE_V>();
        ShiftLeft(signs, signs, uint16_t(15), count);
        Abs(scores, scores, count);
        PipeBarrier<PIPE_V>();
        Cast(values.template ReinterpretCast<int16_t>(), scores, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();
        Or(values, values, signs, count);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline LocalTensor<float> NativeLocalSort(uint64_t row, uint32_t part, bool localPositions = false,
                                                         bool publish = true)
    {
        uint32_t off = contiguousRows_ ? overrideOff_ : part * this->chunk_;
        uint32_t count = row < this->rows_ && off < this->length_ ?
                             (this->length_ - off < this->chunk_ ? this->length_ - off : this->chunk_) :
                             0;
        if (contiguousRows_)
            count = overrideCount_;
        const uint32_t sortCount = contiguousRows_ ? AlignUp(count, 32U) : this->chunk_;
        if (count && cachedScores_ && nativeValues_) {
            auto score = this->template At<float>(this->SCORE_OFFSET * this->chunk_);
            DataCopy(score, this->template At<float>(cachedInputOffset_)[off], sortCount);
            PipeBarrier<PIPE_V>();
            if (!this->descending_) {
                Adds(score.template ReinterpretCast<int32_t>(), score.template ReinterpretCast<int32_t>(),
                     int32_t(-2147483647 - 1), count);
                PipeBarrier<PIPE_V>();
            }
        } else if (count) {
            if (cachedInput_) {
                DataCopy(this->template At<uint16_t>(), this->template At<uint16_t>(cachedInputOffset_)[off],
                         AlignUp(count, DMA_VALUE_ELEMENTS));
                PipeBarrier<PIPE_V>();
            } else {
                DataCopyPad(this->template At<uint16_t>(), this->input_[row * this->length_ + off],
                            {1, count * VALUE_BYTES, 0, 0, 0}, {false, 0, 0, 0});
                EventSync<HardEvent::MTE2_V>();
            }
            if (rankValues_) {
                MakeRankScores(count, sortCount);
            } else if (nativeValues_) {
                auto score = this->template At<float>(this->SCORE_OFFSET * this->chunk_);
                Duplicate(score.template ReinterpretCast<uint32_t>(), uint32_t(0xff800000), sortCount);
                PipeBarrier<PIPE_V>();
                Cast(score, this->template At<T>(), RoundMode::CAST_NONE, count);
                PipeBarrier<PIPE_V>();
                if (!this->descending_) {
                    Adds(score.template ReinterpretCast<int32_t>(), score.template ReinterpretCast<int32_t>(),
                         int32_t(-2147483647 - 1), count);
                    PipeBarrier<PIPE_V>();
                }
            } else {
                this->MakeKeys(count);
                NativeScores(count);
            }
        } else {
            Duplicate(this->template At<float>(this->SCORE_OFFSET * this->chunk_), MERGE_PADDING_SCORE, sortCount);
            PipeBarrier<PIPE_V>();
        }
        // 排序编码生成后，原始输入和键所在区域可复用。
        auto positions = this->template At<int32_t>();
        if (carryPayload_) {
            Duplicate(positions, int32_t(0), sortCount);
            EventSync<HardEvent::V_MTE2>();
            DataCopyPad(positions, this->inputIndex_[row * this->length_ + off], {1, count * INDEX_BYTES, 0, 0, 0},
                        {false, 0, 0, 0});
            EventSync<HardEvent::MTE2_V>();
        } else {
            SeedPositions(positions.template ReinterpretCast<float>(), sortCount);
            PipeBarrier<PIPE_V>();
            Adds(positions.template ReinterpretCast<float>(), positions.template ReinterpretCast<float>(),
                 localPositions ? 0.0f : float(int32_t(off)), sortCount);
            PipeBarrier<PIPE_V>();
            Cast(positions, positions.template ReinterpretCast<float>(), RoundMode::CAST_RINT, sortCount);
            PipeBarrier<PIPE_V>();
        }
        auto ping = this->template At<float>(8 * this->chunk_);
        // Sort32 已消费编码和位置，其 8*N 字节区域可复用为 MrgSort 的另一端；缓冲区共需 16*N 字节。
        auto pong = this->template At<float>();
        for (uint32_t block = 0; block < sortCount / 32; block += 255) {
            const uint32_t remain = sortCount / 32 - block;
            Sort32(ping[GetSortOffset<float>(block * 32)],
                   this->template At<float>(this->SCORE_OFFSET * this->chunk_)[block * 32],
                   positions.template ReinterpretCast<uint32_t>()[block * 32], remain < 255 ? remain : 255);
        }
        PipeBarrier<PIPE_V>();
        uint32_t runs = sortCount / MERGE_SORT_RUN_ELEMENTS, runLength = MERGE_SORT_RUN_ELEMENTS;
        bool inPing = true;
        while (runs > 1) {
            MergeStage(inPing ? pong : ping, inPing ? ping : pong, sortCount, runLength, runs);
            PipeBarrier<PIPE_ALL>();
            inPing = !inPing;
            runLength *= MERGE_LIST_COUNT;
            runs = (runs + MERGE_LIST_COUNT - 1) / MERGE_LIST_COUNT;
        }
        if (publish) {
            EventSync<HardEvent::V_MTE3>();
            DataCopy(this->pingGm_[off * MERGE_PROPOSAL_ELEMENTS], inPing ? ping : pong,
                     this->chunk_ * MERGE_PROPOSAL_ELEMENTS);
        }
        PipeBarrier<PIPE_ALL>();
        return inPing ? ping : pong;
    }

    __aicore__ inline void SortRun(uint64_t row, uint32_t part)
    {
        auto sorted = halfRecords_ ? HalfLocalSort(row, part) : NativeLocalSort(row, part, false, false);
        EventSync<HardEvent::V_MTE3>();
        if (contiguousRows_)
            DataCopyPad(positions_[0][2 * overrideOff_], sorted, {1, overrideCount_ * 8, 0, 0, 0});
        else
            DataCopy(positions_[0][2 * part * this->chunk_], sorted, 2 * this->chunk_);
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Merge(uint64_t row, uint32_t source, uint32_t start, uint32_t run, uint32_t lists,
                                 bool final)
    {
        const uint32_t window = this->fanout_ == 1 ? 2432 : 2688;
        const uint32_t scratch = this->fanout_ == 1 ? window * 8 * 8 : 0;
        uint32_t consumed[4] = {0, 0, 0, 0}, written = 0;
        const uint32_t end = contiguousRows_ ? mergeEnd_ : total_;
        uint32_t groupCount = end - start < run * lists ? end - start : run * lists;
        while (written < groupCount) {
            uint16_t counts[4] = {0, 0, 0, 0};
            uint32_t ids[4], active = 0;
            for (uint32_t p = 0; p < lists; ++p) {
                uint32_t length = end - start - p * run < run ? end - start - p * run : run;
                uint32_t remain = length - consumed[p];
                if (remain) {
                    counts[active] = remain < window ? remain : window;
                    ids[active] = p;
                    DataCopyPad(this->template At<float>()[active * window * 2],
                                positions_[source][2 * (start + p * run + consumed[p])],
                                {1, uint32_t(counts[active]) * 8, 0, 0, 0}, {false, 0, 0, 0});

                    ++active;
                }
            }
            EventSync<HardEvent::MTE2_V>();
            uint32_t used[4] = {0, 0, 0, 0}, other[4] = {0, 0, 0, 0};
            auto v = this->template At<float>();
            auto vo = this->template At<float>(4 * window * 8);
            if (active > 1) {
                MrgSortSrcList<float> vl(v, v[window * 2], v[window * 4], v[window * 6]);

                if (halfRecords_) {
                    auto h = v.template ReinterpretCast<half>();
                    auto o = vo.template ReinterpretCast<half>();
                    MrgSortSrcList<half> hl(h, h[window * 4], h[window * 8], h[window * 12]);
                    MrgSort<half, true>(o, hl, counts, used, (1U << active) - 1, 1);
                } else
                    MrgSort<float, true>(vo, vl, counts, used, (1U << active) - 1, 1);

            } else {
                used[0] = counts[0];
                DataCopy(vo, v, AlignUp(uint32_t(counts[0]), 4) * 2);
            }
            PipeBarrier<PIPE_V>();
            uint32_t count = 0;
            for (uint32_t l = 0; l < active; ++l) {
                count += used[l];
                consumed[ids[l]] += used[l];
            }
            uint32_t off = start + written;
            EventSync<HardEvent::V_MTE3>();
            if (!(contiguousRows_ && final && (nativeValues_ || rankValues_) && (this->fanout_ == 1 || carryPayload_)))
                DataCopyPad(positions_[1 - source][2 * off], vo, {1, count * 8, 0, 0, 0});
            if (final && (nativeValues_ || rankValues_)) {
                auto token = this->template At<uint32_t>(scratch +
                                                         (halfRecords_ ? 8192 : (this->fanout_ == 1 ? 12288 : 8192)));
                const uint32_t emit = halfRecords_ ? 4096 : (this->fanout_ == 1 ? 3072 : 2048);
                for (uint32_t at = 0; at < count && off + at < this->length_; at += emit) {
                    uint32_t n = count - at < emit ? count - at : emit;
                    if (this->length_ - off - at < n)
                        n = this->length_ - off - at;
                    uint32_t valueOffset = scratch;
                    if (halfRecords_) {
                        auto value = this->template At<half>(scratch);
                        // 两路输出均限制为 N 项；SDK 的 half Extract 会扩大索引写入范围，可能覆盖至 2*N 项。
                        uint64_t selected = 0;
                        GatherMask(value, vo[2 * at].template ReinterpretCast<half>(), uint8_t(3), true, 4 * n,
                                   {1, 1, 0, 0}, selected);
                        PipeBarrier<PIPE_V>();
                        GatherMask(token, vo[2 * at].template ReinterpretCast<uint32_t>(), uint8_t(2), true, 2 * n,
                                   {1, 1, 0, 0}, selected);
                        PipeBarrier<PIPE_V>();
                        if (!this->descending_)
                            Adds(value.template ReinterpretCast<int16_t>(), value.template ReinterpretCast<int16_t>(),
                                 int16_t(-32768), n);
                    } else {
                        auto value = this->template At<float>(scratch);
                        Extract(value, token, vo[2 * at], AlignUp(n, 32) / 32);
                        PipeBarrier<PIPE_V>();
                        if (!this->descending_) {
                            Adds(value.template ReinterpretCast<int32_t>(), value.template ReinterpretCast<int32_t>(),
                                 int32_t(-2147483647 - 1), n);
                            PipeBarrier<PIPE_V>();
                        }
                        valueOffset = scratch + (this->fanout_ == 1 ? 24576 : 16384);
                        if (rankValues_) {
                            // MrgSort 消费输入窗口后，将其中一块空闲区复用为 16 位符号临时区，避开 vo 及 Extract
                            // 的编码和索引区。
                            auto signs = this->template At<uint16_t>(this->fanout_ == 1 ? 0 : 24576);
                            DecodeRankValues(this->template At<uint16_t>(valueOffset), value, signs, n);
                        } else {
                            Cast(this->template At<T>(valueOffset), value, RoundMode::CAST_RINT, n);
                            PipeBarrier<PIPE_V>();
                        }
                    }
                    EventSync<HardEvent::V_MTE3>();
                    DataCopyPad(this->output_[row * this->length_ + off + at], this->template At<uint16_t>(valueOffset),
                                {1, n * 2, 0, 0, 0});
                    if (carryPayload_) {
                        DataCopyPad(this->outputIndex_[row * this->length_ + off + at],
                                    token.template ReinterpretCast<int32_t>(), {1, n * INDEX_BYTES, 0, 0, 0});
                    } else if (contiguousRows_ && this->fanout_ == 1) {
                        DataCopyPad(compactRecords_[1 - source][off + at], token, {1, n * 4, 0, 0, 0});
                    }
                    EventSync<HardEvent::MTE3_V>();
                }
            }
            PipeBarrier<PIPE_ALL>();
            written += count;
        }
    }
    // 长行切片需同时检查上下界；N <= 65536 时保留双切片的单次比较路径。
    __aicore__ inline void SliceMask(LocalTensor<uint32_t> positions, LocalTensor<uint8_t> masks, uint32_t base,
                                     uint32_t size, uint32_t count)
    {
        auto upperMask = this->small_.template Get<uint8_t>()[512];
        union {
            uint32_t i;
            float f;
        } lower, upper;
        lower.i = base;
        upper.i = base + size;
        const uint32_t aligned = AlignUp(count, 128);
        Compares<float, uint8_t>(masks, positions.template ReinterpretCast<float>(), lower.f, CMPMODE::GE, aligned);
        Compares<float, uint8_t>(upperMask, positions.template ReinterpretCast<float>(), upper.f, CMPMODE::LT, aligned);
        PipeBarrier<PIPE_V>();
        And(masks.template ReinterpretCast<uint16_t>(), masks.template ReinterpretCast<uint16_t>(),
            upperMask.template ReinterpretCast<uint16_t>(), aligned / 16);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void GatherLongValues(uint64_t row, uint32_t source, uint32_t batch, LocalTensor<int32_t> data,
                                            LocalTensor<float> records, LocalTensor<uint32_t> positions,
                                            LocalTensor<int32_t> result)
    {
        constexpr uint32_t VALUE_SLICE_ELEMENTS = 65536;
        auto offsets = positions.template ReinterpretCast<uint32_t>();
        auto masks = this->small_.template Get<uint8_t>();
        auto partial = records.template ReinterpretCast<half>();
        for (uint32_t base = 0; base < this->length_; base += VALUE_SLICE_ELEMENTS) {
            const uint32_t size = this->length_ - base < VALUE_SLICE_ELEMENTS ? this->length_ - base :
                                                                                VALUE_SLICE_ELEMENTS;
            DataCopyPad(data.template ReinterpretCast<uint16_t>(), this->input_[row * this->length_ + base],
                        {1, size * 2, 0, 0, 0}, {false, 0, 0, 0});
            EventSync<HardEvent::MTE2_V>();
            for (uint32_t off = ownedBegin_; off < ownedEnd_; off += batch) {
                const uint32_t n = ownedEnd_ - off < batch ? ownedEnd_ - off : batch;
                DataCopyPad(records, positions_[source][2 * off], {1, n * 8, 0, 0, 0}, {false, 0, 0, 0});
                if (base)
                    DataCopyPad(result.template ReinterpretCast<uint16_t>(), this->output_[row * this->length_ + off],
                                {1, n * 2, 0, 0, 0}, {false, 0, 0, 0});
                EventSync<HardEvent::MTE2_V>();
                if (!base)
                    Duplicate(result.template ReinterpretCast<uint16_t>(), uint16_t(0), n);
                uint64_t selected = 0;
                GatherMask(positions, records.template ReinterpretCast<uint32_t>(), uint8_t(2), true, 2 * n,
                           {1, 1, 0, 0}, selected);
                PipeBarrier<PIPE_V>();
                SliceMask(positions, masks, base, size, n);
                // 掩码屏蔽不属于当前切片的数据；取模保证实际 UB 访问始终位于已分配源区域内。
                ShiftLeft(offsets, positions, uint32_t(16), n);
                PipeBarrier<PIPE_V>();
                ShiftRight(offsets, offsets, uint32_t(15), n);
                PipeBarrier<PIPE_V>();
                Gather(partial, data.template ReinterpretCast<half>(), offsets, 0, n);
                PipeBarrier<PIPE_V>();
                Select(result.template ReinterpretCast<half>(), masks, partial, result.template ReinterpretCast<half>(),
                       SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
                PipeBarrier<PIPE_V>();
                CanonicalizeSortValues<T>(result.template ReinterpretCast<uint16_t>(),
                                          records.template ReinterpretCast<uint16_t>(), masks, n);
                EventSync<HardEvent::V_MTE3>();
                DataCopyPad(this->output_[row * this->length_ + off], result.template ReinterpretCast<uint16_t>(),
                            {1, n * 2, 0, 0, 0});
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    // 最终归并已输出值和紧凑 INT32 位置，读取调用方索引时无需再预留排序记录区。
    __aicore__ inline void WriteCompactPayload(uint64_t row, uint32_t source)
    {
        const uint32_t sourceBytes = AlignUp(this->length_, DMA_VALUE_ELEMENTS) * INDEX_BYTES;
        uint32_t arenaBytes = NATIVE_MERGE_LOCAL_BYTES_PER_ELEMENT * this->chunk_;
        if (arenaBytes < NATIVE_MERGE_ARENA_BYTES)
            arenaBytes = NATIVE_MERGE_ARENA_BYTES;
        constexpr uint32_t maxVectorElements = UINT8_MAX * FP32_REPEAT_ELEMENTS;
        const uint32_t capacity = (arenaBytes - sourceBytes) / (2 * INDEX_BYTES);
        const uint32_t batch = (capacity < maxVectorElements ? capacity : maxVectorElements) / SIMD_ALIGN_ELEMENTS *
                               SIMD_ALIGN_ELEMENTS;
        auto data = this->template At<int32_t>();
        auto offsets = this->template At<uint32_t>(sourceBytes);
        auto result = this->template At<int32_t>(sourceBytes + batch * INDEX_BYTES);
        DataCopyPad(data, this->inputIndex_[row * this->length_], {1, this->length_ * INDEX_BYTES, 0, 0, 0},
                    {false, 0, 0, 0});
        EventSync<HardEvent::MTE2_V>();
        for (uint32_t off = 0; off < this->length_; off += batch) {
            const uint32_t count = this->length_ - off < batch ? this->length_ - off : batch;
            EventSync<HardEvent::V_MTE2>();
            DataCopyPad(offsets, compactRecords_[source][off], {1, count * INDEX_BYTES, 0, 0, 0}, {false, 0, 0, 0});
            EventSync<HardEvent::MTE2_V>();
            ShiftLeft(offsets, offsets, uint32_t(2), count);
            PipeBarrier<PIPE_V>();
            Gather(result, data, offsets, 0, count);
            EventSync<HardEvent::V_MTE3>();
            DataCopyPad(this->outputIndex_[row * this->length_ + off], result, {1, count * INDEX_BYTES, 0, 0, 0});
            EventSync<HardEvent::MTE3_V>();
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void WriteValues(uint64_t row, uint32_t source)
    {
        if (carryPayload_)
            return;
        uint32_t arenaBytes = NATIVE_MERGE_LOCAL_BYTES_PER_ELEMENT * this->chunk_;
        if (arenaBytes < NATIVE_MERGE_ARENA_BYTES)
            arenaBytes = NATIVE_MERGE_ARENA_BYTES;
        const uint64_t sourceBytes = uint64_t(AlignUp(this->length_, DMA_VALUE_ELEMENTS)) * INDEX_BYTES;
        constexpr uint32_t minimumGatherScratch = 2 * STREAM_MERGE_PAYLOAD_ELEMENTS * INDEX_BYTES;
        // 整行索引和较大的 Gather 批次均能放入 UB 时，直接使用最终归并输出的位置，不受局部排序段长限制。
        if (nativeValues_ && contiguousRows_ && this->fanout_ == 1 &&
            sourceBytes + minimumGatherScratch <= arenaBytes) {
            WriteCompactPayload(row, source);
            return;
        }
        uint32_t sourceLength = AlignUp(this->length_, 16);
        uint32_t capacity = 4 * sourceLength < arenaBytes ? (arenaBytes - 4 * sourceLength) / 16 / 128 * 128 : 0;
        bool single = capacity >= 1024;
        const uint32_t C = single ? (capacity < 3328 ? capacity : 3328) : 3328;
        const uint32_t S = single ? sourceLength : 32768;
        const uint32_t recordOffset = 4 * S, positionOffset = recordOffset + 8 * C;
        const uint32_t resultOffset = positionOffset + 4 * C;
        auto data = this->template At<int32_t>();
        auto records = this->template At<float>(recordOffset);
        auto positions = this->template At<uint32_t>(positionOffset);
        auto offsets = positions.template ReinterpretCast<int32_t>();
        auto result = this->template At<int32_t>(resultOffset);

        auto part = records.template ReinterpretCast<int32_t>();
        auto masks = this->small_.template Get<uint8_t>();
        if (!nativeValues_ && this->length_ > 65536) {
            GatherLongValues(row, source, C, data, records, positions, result);
        } else if (!nativeValues_) {
            DataCopyPad(data.template ReinterpretCast<uint16_t>(), this->input_[row * this->length_],
                        {1, this->length_ * 2, 0, 0, 0}, {false, 0, 0, 0});
            EventSync<HardEvent::MTE2_V>();
            for (uint32_t off = ownedBegin_; off < ownedEnd_; off += C) {
                uint32_t n = ownedEnd_ - off < C ? ownedEnd_ - off : C;
                DataCopyPad(records, positions_[source][2 * off], {1, n * 8, 0, 0, 0}, {false, 0, 0, 0});
                EventSync<HardEvent::MTE2_V>();
                uint64_t selected = 0;
                GatherMask(positions, records.template ReinterpretCast<uint32_t>(), uint8_t(2), true, 2 * n,
                           {1, 1, 0, 0}, selected);
                PipeBarrier<PIPE_V>();
                Muls(offsets, positions.template ReinterpretCast<int32_t>(), int32_t(2), n);
                PipeBarrier<PIPE_V>();
                Gather(result.template ReinterpretCast<uint16_t>(), data.template ReinterpretCast<uint16_t>(),
                       offsets.template ReinterpretCast<uint32_t>(), 0, n);
                PipeBarrier<PIPE_V>();
                CanonicalizeSortValues<T>(result.template ReinterpretCast<uint16_t>(),
                                          records.template ReinterpretCast<uint16_t>(), masks, n);
                EventSync<HardEvent::V_MTE3>();
                DataCopyPad(this->output_[row * this->length_ + off], result.template ReinterpretCast<uint16_t>(),
                            {1, n * 2, 0, 0, 0});
                PipeBarrier<PIPE_ALL>();
            }
        }
        // 源切片在处理各输出 tile 时保持驻留，避免逐 tile 重复搬入整块源数据。
        for (uint32_t base = 0; base < this->length_; base += S) {
            uint32_t size = this->length_ - base < S ? this->length_ - base : S;
            DataCopyPad(data, this->inputIndex_[row * this->length_ + base], {1, size * 4, 0, 0, 0}, {false, 0, 0, 0});
            EventSync<HardEvent::MTE2_V>();
            for (uint32_t off = ownedBegin_; off < ownedEnd_; off += C) {
                uint32_t n = ownedEnd_ - off < C ? ownedEnd_ - off : C;
                if (contiguousRows_ && nativeValues_ && this->fanout_ == 1)
                    DataCopyPad(positions, compactRecords_[source][off], {1, n * 4, 0, 0, 0}, {false, 0, 0, 0});
                else
                    DataCopyPad(records, positions_[source][2 * off], {1, n * 8, 0, 0, 0}, {false, 0, 0, 0});
                if (base)
                    DataCopyPad(result, this->outputIndex_[row * this->length_ + off], {1, n * 4, 0, 0, 0},
                                {false, 0, 0, 0});
                EventSync<HardEvent::MTE2_V>();
                if (!base)
                    Duplicate(result, int32_t(0), n);
                uint64_t selected = 0;
                if (!(contiguousRows_ && nativeValues_ && this->fanout_ == 1))
                    GatherMask(positions, records.template ReinterpretCast<uint32_t>(), uint8_t(2), true, 2 * n,
                               {1, 1, 0, 0}, selected);
                PipeBarrier<PIPE_V>();
                if (this->length_ <= S) {
                    Muls(offsets, positions.template ReinterpretCast<int32_t>(), int32_t(4), n);
                    PipeBarrier<PIPE_V>();
                    Gather(result, data, offsets.template ReinterpretCast<uint32_t>(), 0, n);
                } else {
                    // 非负 INT32 位置按位解释为 float 后保持大小关系，可直接用于范围比较。
                    if (this->length_ > 2 * S) {
                        SliceMask(positions, masks, base, size, n);
                    } else {
                        union {
                            uint32_t i;
                            float f;
                        } boundary;
                        boundary.i = S;
                        Compares<float, uint8_t>(masks, positions.template ReinterpretCast<float>(), boundary.f,
                                                 base == 0 ? CMPMODE::LT : CMPMODE::GE, AlignUp(n, 128));
                    }
                    // 切片大小 S=32768，局部字节偏移为 (position % S)*4。
                    // 非当前切片的读取仍限定在源缓冲内，由 Select 屏蔽其结果。
                    // 原地转换位置为字节偏移前，须确保 Compare 已读取完位置。
                    PipeBarrier<PIPE_V>();
                    ShiftLeft(offsets.template ReinterpretCast<uint32_t>(), positions, uint32_t(17), n);
                    PipeBarrier<PIPE_V>();
                    ShiftRight(offsets.template ReinterpretCast<uint32_t>(),
                               offsets.template ReinterpretCast<uint32_t>(), uint32_t(15), n);
                    PipeBarrier<PIPE_V>();
                    Gather(part, data, offsets.template ReinterpretCast<uint32_t>(), 0, n);
                    PipeBarrier<PIPE_V>();
                    Select(result.template ReinterpretCast<float>(), masks, part.template ReinterpretCast<float>(),
                           result.template ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
                }
                EventSync<HardEvent::V_MTE3>();
                DataCopyPad(this->outputIndex_[row * this->length_ + off], result, {1, n * 4, 0, 0, 0});
                PipeBarrier<PIPE_ALL>();
            }
        }
    }
    TBuf<TPosition::VECCALC> positionSeed_;
    GlobalTensor<float> positions_[2];
    GlobalTensor<uint32_t> compactRecords_[2];
    uint32_t runs_, total_, span_ = 0, rowsPerCore_ = 1, lane_ = 0, groups_ = 1;
    uint32_t ownedBegin_ = 0, ownedEnd_ = 0, mergeEnd_ = 0, overrideOff_ = 0, overrideCount_ = 0;
    uint32_t cachedInputOffset_ = 0;
    // 原生排序编码和有限值路径直接携带原始 INT32 索引；SingleRun 的 NaN 路径仍按位置取回索引。
    bool carryPayload_ = false;
    bool rankValues_ = false;
    bool cachedInput_ = false;
    bool cachedScores_ = false;
    bool contiguousRows_ = false;
    bool halfRecords_ = false, nativeValues_ = false;
};
} // namespace SortWithIndexA2
#endif
