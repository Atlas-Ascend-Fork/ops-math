/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RADIX_SORT_WITH_INDEX_MULTI_BLOCK_H
#define RADIX_SORT_WITH_INDEX_MULTI_BLOCK_H

#include "radix_sort_with_index_base.h"
#include "radix_sort_with_index_pipeline.h"

namespace SortWithIndexA2 {
// 从低位到高位每轮处理 2 bit，结合各 tile 直方图进行全局稳定基数排序。
template <typename T>
class RadixSortWithIndexMultiBlock : public RadixSortWithIndexBase<T, MULTI_BLOCK_MODE> {
    using Base = RadixSortWithIndexBase<T, MULTI_BLOCK_MODE>;
    using Base::alignedLength_;
    using Base::coresPerRow_;
    using Base::digitScratchBuf_;
    using Base::EventSync;
    using Base::histogramGm_;
    using Base::histogramRowsGm_;
    using Base::histogramRowStride_;
    using Base::indexOutputBuf_;
    using Base::inputGm_;
    using Base::inputIndexGm_;
    using Base::keyOutputBuf_;
    using Base::lastAxisNum_;
    using Base::MakeKeys;
    using Base::maskBuf_;
    using Base::outputGm_;
    using Base::outputIndexGm_;
    using Base::prefixRecordsBuf_;
    using Base::rowNum_;
    using Base::scratchIndexGm_;
    using Base::scratchValueGm_;
    using Base::tileNum_;
    using Base::valueKeyBuf_;
    using Base::valueOutputBuf_;

public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR outputIndex, GM_ADDR workspace,
                                const SortWithIndexTilingDataA2& data)
    {
        Base::Init(x, index, y, outputIndex, workspace, data);
        pipeline_.Init(this->pipe_, alignedLength_);
    }

    __aicore__ inline void Process()
    {
        // 各核处理连续 tile，按核号扫描摘要即可保持 tile 原顺序；余数部分优先分给前面的核。
        const uint32_t tilesPerCore = tileNum_ / coresPerRow_;
        const uint32_t remainder = tileNum_ % coresPerRow_;
        const uint32_t core = GetBlockIdx() % coresPerRow_;
        tileBegin_ = core * tilesPerCore + (core < remainder ? core : remainder);
        tileEnd_ = tileBegin_ + tilesPerCore + (core < remainder ? 1 : 0);
        const uint32_t groups = GetBlockNum() / coresPerRow_;
        if (coresPerRow_ > 1 && groups > 1) {
            // SyncAll 同步本次启动的所有核；尾部无有效行的组也须参与相同次数的同步。
            const uint32_t group = GetBlockIdx() / coresPerRow_;
            for (uint64_t begin = 0; begin < rowNum_; begin += groups) {
                const uint64_t row = begin + group;
                ProcessRow(row, true, row < rowNum_);
            }
            return;
        }
        // 独立行无需同步；只有一个协作组时，可在处理完有效基数位后直接停止。
        for (uint64_t row = GetBlockIdx() / coresPerRow_; row < rowNum_; row += groups) {
            ProcessRow(row);
        }
    }

private:
    __aicore__ inline void MakeDigits(uint32_t pass, uint32_t count)
    {
        auto key = valueKeyBuf_.template Get<uint16_t>()[alignedLength_];
        auto digit = valueOutputBuf_.template Get<uint16_t>();
        auto three = keyOutputBuf_.template Get<uint16_t>();
        ShiftRight(digit, key, uint16_t(pass * RADIX_BITS), count);
        Duplicate(three, uint16_t(RADIX_MASK), count);
        PipeBarrier<PIPE_V>();
        And(digit, digit, three, count);
        PipeBarrier<PIPE_V>();
        Cast(digitScratchBuf_.template Get<float>(), digit.template ReinterpretCast<int16_t>(), RoundMode::CAST_NONE,
             count);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void BucketMask(uint32_t bin, uint32_t count)
    {
        Compares<float, uint8_t>(maskBuf_.template Get<uint8_t>(), digitScratchBuf_.template Get<float>(),
                                 static_cast<float>(static_cast<int32_t>(bin)), CMPMODE::EQ,
                                 AlignUp(count, SIMD_ALIGN_ELEMENTS));
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline uint32_t Count(uint32_t count)
    {
        uint64_t found = 0;
        GatherMask(indexOutputBuf_.template Get<float>(), digitScratchBuf_.template Get<float>(),
                   maskBuf_.template Get<uint32_t>(), true, count, {1, 1, 0, 0}, found);
        this->template EventSync<HardEvent::V_S>();
        return static_cast<uint32_t>(found);
    }
    __aicore__ inline void CompactPayloads(LocalTensor<uint16_t> input, LocalTensor<uint16_t> output, uint32_t count)
    {
        uint64_t valueCount = 0, indexCount = 0;
        GatherMask(output, input, maskBuf_.template Get<uint16_t>(), true, count, {1, 1, 0, 0}, valueCount);
        GatherMask(pipeline_.OutputIndices(output), pipeline_.InputIndices(input), maskBuf_.template Get<uint32_t>(),
                   true, count, {1, 1, 0, 0}, indexCount);
    }
    __aicore__ inline void PrefetchTile(const GlobalTensor<uint16_t>& source, const GlobalTensor<int32_t>& sourceIndex,
                                        uint32_t tile, bool needIndex)
    {
        uint32_t offset = tile * alignedLength_;
        uint32_t count = lastAxisNum_ - offset < alignedLength_ ? lastAxisNum_ - offset : alignedLength_;
        pipeline_.Prefetch(source, sourceIndex, offset, count, needIndex);
    }
    template <bool RetainInput, bool LocalHistograms>
    __aicore__ inline void BuildHistograms(const GlobalTensor<uint16_t>& source,
                                           const GlobalTensor<int32_t>& sourceIndex, uint32_t pass)
    {
        uint32_t coreCounts[RADIX_BINS] = {0};
        int32_t coreMinimum = INT32_MAX, coreMaximum = INT32_MIN;
        if (tileBegin_ < tileEnd_)
            PrefetchTile(source, sourceIndex, tileBegin_, RetainInput);
        for (uint32_t tile = tileBegin_; tile < tileEnd_; ++tile) {
            uint32_t off = tile * alignedLength_,
                     count = lastAxisNum_ - off < alignedLength_ ? lastAxisNum_ - off : alignedLength_;
            auto input = pipeline_.AcquireInput();
            if (tile + 1 < tileEnd_)
                PrefetchTile(source, sourceIndex, tile + 1, false);
            MakeKeys(input.template ReinterpretCast<int16_t>(), count);
            int32_t minimum = 0, maximum = 0;
            if (pass == 0) {
                Cast(digitScratchBuf_.template Get<float>(), valueKeyBuf_.template Get<int16_t>()[alignedLength_],
                     RoundMode::CAST_NONE, count);
                PipeBarrier<PIPE_V>();
                ReduceMin(indexOutputBuf_.template Get<float>(), digitScratchBuf_.template Get<float>(),
                          valueOutputBuf_.template Get<float>(), count, false);
                ReduceMax(indexOutputBuf_.template Get<float>()[REDUCE_MAX_OFFSET],
                          digitScratchBuf_.template Get<float>(), keyOutputBuf_.template Get<float>(), count, false);
                this->template EventSync<HardEvent::V_S>();
                minimum = indexOutputBuf_.template Get<float>().GetValue(0);
                maximum = indexOutputBuf_.template Get<float>().GetValue(REDUCE_MAX_OFFSET);
                this->template EventSync<HardEvent::S_V>();
                coreMinimum = coreMinimum < minimum ? coreMinimum : minimum;
                coreMaximum = coreMaximum > maximum ? coreMaximum : maximum;
            }
            MakeDigits(pass, count);
            auto record = keyOutputBuf_.template Get<int32_t>();
            uint32_t zero = count;
            for (uint32_t bin = 1; bin < RADIX_BINS; ++bin) {
                BucketMask(bin, count);
                uint32_t found = Count(count);
                zero -= found;
                record.SetValue(bin, found);
                coreCounts[bin] += found;
            }
            record.SetValue(0, zero);
            coreCounts[0] += zero;
            if (RetainInput) {
                // 摘要交换期间保留排序键、当前位和队列槽，Scatter 直接复用，避免二次扫描。
                retainedInput_ = input;
                for (uint32_t bin = 0; bin < RADIX_BINS; ++bin)
                    retainedCounts_[bin] = coreCounts[bin];
            } else if (LocalHistograms) {
                auto records = prefixRecordsBuf_.template Get<int32_t>();
                for (uint32_t bin = 0; bin < RADIX_BINS; ++bin)
                    records.SetValue(tile * HISTOGRAM_RECORD_ELEMENTS + bin, record.GetValue(bin));
                this->template EventSync<HardEvent::S_V>();
            } else {
                this->template EventSync<HardEvent::S_MTE3>();
                DataCopy(histogramGm_[tile * HISTOGRAM_RECORD_ELEMENTS], record, HISTOGRAM_RECORD_ELEMENTS);
                this->template EventSync<HardEvent::MTE3_V>();
            }
            if (!RetainInput)
                pipeline_.ReleaseInput(input);
        }
        if (LocalHistograms) {
            localMinimum_ = coreMinimum;
            localMaximum_ = coreMaximum;
            for (uint32_t bin = 0; bin < RADIX_BINS; ++bin)
                localTotals_[bin] = coreCounts[bin];
            return;
        }
        // 摘要由标量 SetValue 写入；tile 的 MTE3_V 等待仅保护向量缓冲复用，还需同步保护标量写入。
        this->template EventSync<HardEvent::MTE3_S>();
        // 直方图同步前每核写一份摘要；tile 记录仅含桶计数，最小和最大键存入核摘要。
        auto summary = keyOutputBuf_.template Get<int32_t>();
        for (uint32_t bin = 0; bin < RADIX_BINS; ++bin)
            summary.SetValue(bin, coreCounts[bin]);
        summary.SetValue(HISTOGRAM_MIN_OFFSET, coreMinimum);
        summary.SetValue(HISTOGRAM_MAX_OFFSET, coreMaximum);
        this->template EventSync<HardEvent::S_MTE3>();
        DataCopy(histogramGm_[(tileNum_ + GetBlockIdx() % coresPerRow_) * HISTOGRAM_RECORD_ELEMENTS], summary,
                 HISTOGRAM_RECORD_ELEMENTS);
        PipeBarrier<PIPE_ALL>();
    }
    template <bool LocalHistograms>
    __aicore__ inline void ReadCoreOffsets(uint32_t pass, uint32_t* running, int32_t& minimum, int32_t& maximum)
    {
        if (LocalHistograms) {
            minimum = localMinimum_;
            maximum = localMaximum_;
            uint32_t bucketBegin = 0;
            for (uint32_t bin = 0; bin < RADIX_BINS; ++bin) {
                running[bin] = bucketBegin;
                bucketBegin += localTotals_[bin];
            }
            return;
        }
        auto records = prefixRecordsBuf_.template Get<int32_t>();
        uint32_t totals[RADIX_BINS] = {0};
        const uint32_t core = GetBlockIdx() % coresPerRow_;
        // 仅汇总核摘要：本核桶起点 = 前面所有桶的总数 + 前面各核在同桶中的计数。
        for (uint32_t begin = 0; begin < coresPerRow_; begin += PREFIX_BATCH_TILES) {
            uint32_t remaining = coresPerRow_ - begin;
            uint32_t size = remaining < PREFIX_BATCH_TILES ? remaining : PREFIX_BATCH_TILES;
            this->template EventSync<HardEvent::S_MTE2>();
            DataCopy(records, histogramGm_[(tileNum_ + begin) * HISTOGRAM_RECORD_ELEMENTS],
                     size * HISTOGRAM_RECORD_ELEMENTS);
            this->template EventSync<HardEvent::MTE2_S>();
            for (uint32_t t = 0; t < size; ++t) {
                for (uint32_t b = 0; b < RADIX_BINS; ++b) {
                    uint32_t count = records.GetValue(t * HISTOGRAM_RECORD_ELEMENTS + b);
                    totals[b] += count;
                    if (begin + t < core)
                        running[b] += count;
                }
                if (pass == 0) {
                    int32_t lo = records.GetValue(t * HISTOGRAM_RECORD_ELEMENTS + HISTOGRAM_MIN_OFFSET),
                            hi = records.GetValue(t * HISTOGRAM_RECORD_ELEMENTS + HISTOGRAM_MAX_OFFSET);
                    minimum = minimum < lo ? minimum : lo;
                    maximum = maximum > hi ? maximum : hi;
                }
            }
        }
        uint32_t bucketBegin = 0;
        for (uint32_t b = 0; b < RADIX_BINS; ++b) {
            running[b] += bucketBegin;
            bucketBegin += totals[b];
        }
    }
    template <bool RetainInput, bool LocalHistograms>
    __aicore__ inline void ScatterBuckets(uint64_t row, const GlobalTensor<uint16_t>& source,
                                          const GlobalTensor<int32_t>& sourceIndex, uint32_t pass, bool writeOutput,
                                          uint32_t* running)
    {
        auto records = prefixRecordsBuf_.template Get<int32_t>();
        auto target = writeOutput ? outputGm_[row * lastAxisNum_] : scratchValueGm_;
        auto targetIndex = writeOutput ? outputIndexGm_[row * lastAxisNum_] : scratchIndexGm_;
        if (!RetainInput && tileBegin_ < tileEnd_)
            PrefetchTile(source, sourceIndex, tileBegin_, true);
        for (uint32_t begin = tileBegin_; begin < tileEnd_; begin += PREFIX_BATCH_TILES) {
            uint32_t size = tileEnd_ - begin < PREFIX_BATCH_TILES ? tileEnd_ - begin : PREFIX_BATCH_TILES;
            if (!RetainInput && !LocalHistograms) {
                this->template EventSync<HardEvent::S_MTE2>();
                DataCopy(records, histogramGm_[begin * HISTOGRAM_RECORD_ELEMENTS], size * HISTOGRAM_RECORD_ELEMENTS);
                this->template EventSync<HardEvent::MTE2_S>();
            }
            for (uint32_t t = 0; t < size; ++t) {
                uint32_t counts[RADIX_BINS];
                for (uint32_t b = 0; b < RADIX_BINS; ++b)
                    counts[b] = RetainInput ? retainedCounts_[b] : records.GetValue(t * HISTOGRAM_RECORD_ELEMENTS + b);
                uint32_t tile = begin + t;
                uint32_t off = tile * alignedLength_,
                         count = lastAxisNum_ - off < alignedLength_ ? lastAxisNum_ - off : alignedLength_;
                auto input = RetainInput ? retainedInput_ : pipeline_.AcquireInput();
                if (tile + 1 < tileEnd_)
                    PrefetchTile(source, sourceIndex, tile + 1, true);
                if (!RetainInput) {
                    MakeKeys(input.template ReinterpretCast<int16_t>(), count);
                    MakeDigits(pass, count);
                }
                for (uint32_t bin = 0; bin < RADIX_BINS; ++bin) {
                    uint32_t found = counts[bin];
                    if (found == 0)
                        continue;
                    BucketMask(bin, count);
                    auto output = pipeline_.AllocateOutput();
                    CompactPayloads(input, output, count);
                    pipeline_.WriteOutput(output, target, targetIndex, running[bin], found);
                }
                pipeline_.ReleaseInput(input);
                for (uint32_t b = 0; b < RADIX_BINS; ++b)
                    running[b] += counts[b];
            }
        }
        // 本轮 SyncAll 前完成全部数据写出，下一轮其他核才可安全读取。
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline uint32_t RadixPasses(int32_t minimum, int32_t maximum)
    {
        uint32_t different = uint16_t(minimum) ^ uint16_t(maximum);
        uint32_t passes = 1;
        while (different >= RADIX_BINS) {
            different >>= RADIX_BITS;
            ++passes;
        }
        return passes;
    }
    __aicore__ inline uint32_t ReadWavePasses(uint64_t firstRow)
    {
        // 所有活跃组已写出第零轮摘要，各核据此得到相同循环上限，无需新增同步或 GM 记录；跳过无有效行的尾组。
        const uint32_t groups = GetBlockNum() / coresPerRow_;
        auto records = prefixRecordsBuf_.template Get<int32_t>();
        uint32_t wavePasses = 1;
        for (uint32_t group = 0; group < groups && firstRow + group < rowNum_; ++group) {
            this->template EventSync<HardEvent::S_MTE2>();
            DataCopy(records,
                     histogramRowsGm_[uint64_t(group) * histogramRowStride_ +
                                      uint64_t(tileNum_) * HISTOGRAM_RECORD_ELEMENTS],
                     coresPerRow_ * HISTOGRAM_RECORD_ELEMENTS);
            this->template EventSync<HardEvent::MTE2_S>();
            int32_t minimum = INT32_MAX, maximum = INT32_MIN;
            for (uint32_t core = 0; core < coresPerRow_; ++core) {
                const int32_t lo = records.GetValue(core * HISTOGRAM_RECORD_ELEMENTS + HISTOGRAM_MIN_OFFSET);
                const int32_t hi = records.GetValue(core * HISTOGRAM_RECORD_ELEMENTS + HISTOGRAM_MAX_OFFSET);
                minimum = minimum < lo ? minimum : lo;
                maximum = maximum > hi ? maximum : hi;
            }
            const uint32_t passes = RadixPasses(minimum, maximum);
            wavePasses = wavePasses > passes ? wavePasses : passes;
        }
        return wavePasses;
    }
    __aicore__ inline void ProcessRow(uint64_t row, bool grouped = false, bool activeRow = true)
    {
        // 每行只选择一次处理路径，避免通用多 tile 循环逐 tile 判断复用条件。
        if (tileEnd_ == tileBegin_ + 1) {
            if (UseLocalHistograms())
                ProcessRowImpl<true, true>(row, grouped, activeRow);
            else
                ProcessRowImpl<true, false>(row, grouped, activeRow);
        } else if (UseLocalHistograms()) {
            ProcessRowImpl<false, true>(row, grouped, activeRow);
        } else {
            ProcessRowImpl<false, false>(row, grouped, activeRow);
        }
    }
    template <bool RetainInput, bool LocalHistograms>
    __aicore__ inline void ProcessRowImpl(uint64_t row, bool grouped, bool activeRow)
    {
        uint32_t passes = MAX_RADIX_PASSES;
        uint32_t wavePasses = MAX_RADIX_PASSES;
        // 不同行的键范围和排序轮数可能不同；已完成的组仅参与同步，不再读写数据。
        for (uint32_t pass = 0; pass < (grouped ? wavePasses : passes); ++pass) {
            const bool activePass = activeRow && pass < passes;
            bool writeOutput = (pass % 2) == ((passes - 1) % 2);
            auto source = inputGm_;
            auto sourceIndex = inputIndexGm_;
            if (activePass) {
                source = pass == 0 ? inputGm_[row * lastAxisNum_] :
                                     (writeOutput ? scratchValueGm_ : outputGm_[row * lastAxisNum_]);
                sourceIndex = pass == 0 ? inputIndexGm_[row * lastAxisNum_] :
                                          (writeOutput ? scratchIndexGm_ : outputIndexGm_[row * lastAxisNum_]);
                BuildHistograms<RetainInput, LocalHistograms>(source, sourceIndex, pass);
            }
            if (coresPerRow_ > 1)
                SyncAll();
            if (grouped && pass == 0) {
                wavePasses = ReadWavePasses(row - GetBlockIdx() / coresPerRow_);
            }
            if (activePass) {
                uint32_t running[RADIX_BINS] = {0};
                int32_t minimum = INT32_MAX, maximum = INT32_MIN;
                ReadCoreOffsets<LocalHistograms>(pass, running, minimum, maximum);
                if (pass == 0) {
                    passes = RadixPasses(minimum, maximum);
                    writeOutput = (pass % 2) == ((passes - 1) % 2);
                }
                ScatterBuckets<RetainInput, LocalHistograms>(row, source, sourceIndex, pass, writeOutput, running);
            }
            if (coresPerRow_ > 1)
                SyncAll();
        }
    }
    __aicore__ inline bool UseLocalHistograms() const
    {
        // 单核且本行 tile 记录能全部放入前缀缓冲时，将直方图保留在 UB。
        return coresPerRow_ == 1 && tileNum_ <= PREFIX_BATCH_TILES;
    }
    LocalTensor<uint16_t> retainedInput_;
    uint32_t retainedCounts_[RADIX_BINS];
    uint32_t localTotals_[RADIX_BINS];
    int32_t localMinimum_, localMaximum_;
    uint32_t tileBegin_, tileEnd_;
    RadixSortWithIndexPipeline pipeline_;
};
} // namespace SortWithIndexA2

#endif // RADIX_SORT_WITH_INDEX_MULTI_BLOCK_H
