/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file exp_segsum_grad_short_batches.h
 * \brief ExpSegsumGrad kernel for processing short sequences in batches on Ascend 950.
 */

#ifndef EXP_SEGSUM_GRAD_SHORT_BATCHES_ARCH35_H
#define EXP_SEGSUM_GRAD_SHORT_BATCHES_ARCH35_H

#include <type_traits>
#include "kernel_operator.h"
#include "exp_segsum_grad_tiling_data.h"

namespace ExpSegsumGradArch35 {
using namespace AscendC;

template <typename T>
class ExpSegsumGradShortBatches {
public:
    __aicore__ inline void Init(GM_ADDR gradOutput, GM_ADDR gradSelf, GM_ADDR gradInput,
                                const ExpSegsumGradTilingDataArch35* tilingData)
    {
        const uint32_t block = GetBlockIdx();
        if (block >= tilingData->needCoreNum) {
            return;
        }
        batchStart_ = tilingData->batchStart[block];
        batchEnd_ = tilingData->batchEnd[block];
        length_ = tilingData->tailDimLength;
        batchesPerTile_ = tilingData->slideSize;
        matrixElems_ = length_ * length_;
        const uint32_t blockElems = BLOCK_BYTES / sizeof(T);
        rowStride_ = (length_ + blockElems - 1) / blockElems * blockElems;
        gradInput_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradInput));
        const uint32_t rows = batchesPerTile_ * length_;
        pipe_.InitBuffer(result_, 1, rows * sizeof(T));
        if (length_ == 1) {
            return;
        }
        gradOutput_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradOutput));
        gradSelf_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradSelf));
        const uint32_t inputBytes = batchesPerTile_ * matrixElems_ * sizeof(T);
        pipe_.InitBuffer(gradOutputQueue_, 1, inputBytes);
        pipe_.InitBuffer(gradSelfQueue_, 1, inputBytes);
        pipe_.InitBuffer(prefix_, rows * rowStride_ * sizeof(float));
        if constexpr (!std::is_same<T, float>::value) {
            pipe_.InitBuffer(reduced_, rows * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        for (int64_t batch = batchStart_; batch < batchEnd_; batch += batchesPerTile_) {
            const int64_t remaining = batchEnd_ - batch;
            const uint32_t batches = remaining < batchesPerTile_ ? remaining : batchesPerTile_;
            const uint32_t rows = batches * length_;
            LocalTensor<T> output = result_.template AllocTensor<T>();
            if (length_ == 1) {
                using Bits = std::conditional_t<std::is_same<T, float>::value, float, uint16_t>;
                Duplicate(output.template ReinterpretCast<Bits>(), static_cast<Bits>(0), rows);
            } else {
                ComputeTile(batch, batches, rows, output);
            }
            result_.EnQue(output);
            output = result_.template DeQue<T>();
            const DataCopyExtParams copy{1, rows * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            DataCopyPad(gradInput_[batch * length_], output, copy);
            result_.FreeTensor(output);
        }
    }

private:
    static constexpr uint32_t BLOCK_BYTES = 32;
    static constexpr uint32_t BATCH_LANES = 64;
    static constexpr Reg::CastTrait WIDEN_B16 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                 Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    static __simd_vf__ inline void BuildPrefix(__ubuf__ T* gradOutput, __ubuf__ T* gradSelf, __ubuf__ float* prefix,
                                               uint32_t batches, uint32_t length, uint32_t rowStride, uint16_t groups)
    {
        const uint32_t matrixElems = length * length;
        const uint32_t prefixElems = length * rowStride;
        Reg::RegTensor<uint32_t> lane, inputBase, prefixBase;
        Reg::MaskReg full = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane), static_cast<int32_t>(0));
        Reg::Muls(inputBase, lane, matrixElems, full);
        Reg::Muls(prefixBase, lane, prefixElems, full);
        for (uint16_t group = 0; group < groups; ++group) {
            const uint32_t batchOffset = static_cast<uint32_t>(group) * BATCH_LANES;
            const uint32_t remaining = batches - batchOffset;
            const uint32_t active = remaining < BATCH_LANES ? remaining : BATCH_LANES;
            uint32_t activeF32 = active;
            uint32_t activeB16 = active;
            Reg::MaskReg valid = Reg::UpdateMask<float>(activeF32);
            Reg::MaskReg validB16 = Reg::UpdateMask<half>(activeB16);
            __ubuf__ T* gradOutputGroup = gradOutput + batchOffset * matrixElems;
            __ubuf__ T* gradSelfGroup = gradSelf + batchOffset * matrixElems;
            __ubuf__ float* prefixGroup = prefix + batchOffset * prefixElems;
            // A lane owns one complete matrix. Visit each used column from bottom to top.
            for (uint16_t column = 0; column + 1 < length; ++column) {
                Reg::RegTensor<float> carry;
                Reg::Duplicate(carry, 0.0f, full);
                for (uint16_t reverseRow = length; reverseRow > column + 1; --reverseRow) {
                    const uint32_t row = reverseRow - 1;
                    Reg::RegTensor<uint32_t> offsets, prefixOffsets;
                    Reg::RegTensor<float> grad, self, product;
                    Reg::Adds(offsets, inputBase, row * length + column, full);
                    if constexpr (std::is_same<T, float>::value) {
                        Reg::Gather(grad, gradOutputGroup, offsets, valid);
                        Reg::Gather(self, gradSelfGroup, offsets, valid);
                    } else {
                        Reg::RegTensor<uint16_t> offsetsB16;
                        Reg::RegTensor<T> gathered, expanded;
                        Reg::Pack(offsetsB16, offsets);
                        Reg::Gather(gathered, gradOutputGroup, offsetsB16, validB16);
                        Reg::UnPack<uint32_t, uint16_t, Reg::HighLowPart::LOWEST>(
                            reinterpret_cast<Reg::RegTensor<uint32_t>&>(expanded),
                            reinterpret_cast<Reg::RegTensor<uint16_t>&>(gathered));
                        Reg::Cast<float, T, WIDEN_B16>(grad, expanded, valid);
                        Reg::Gather(gathered, gradSelfGroup, offsetsB16, validB16);
                        Reg::UnPack<uint32_t, uint16_t, Reg::HighLowPart::LOWEST>(
                            reinterpret_cast<Reg::RegTensor<uint32_t>&>(expanded),
                            reinterpret_cast<Reg::RegTensor<uint16_t>&>(gathered));
                        Reg::Cast<float, T, WIDEN_B16>(self, expanded, valid);
                    }
                    Reg::Mul(product, grad, self, valid);
                    Reg::Add(carry, product, carry, valid);
                    Reg::Adds(prefixOffsets, prefixBase, row * rowStride + column, full);
                    Reg::Scatter(prefixGroup, carry, prefixOffsets, valid);
                }
            }
        }
    }

    __aicore__ inline void ComputeTile(int64_t batch, uint32_t batches, uint32_t rows, LocalTensor<T>& output)
    {
        const int64_t offset = batch * matrixElems_;
        const DataCopyExtParams copy{1, batches * matrixElems_ * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        const DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        LocalTensor<T> gradOutput = gradOutputQueue_.template AllocTensor<T>();
        LocalTensor<T> gradSelf = gradSelfQueue_.template AllocTensor<T>();
        DataCopyPad(gradOutput, gradOutput_[offset], copy, pad);
        DataCopyPad(gradSelf, gradSelf_[offset], copy, pad);
        gradOutputQueue_.EnQue(gradOutput);
        gradSelfQueue_.EnQue(gradSelf);
        gradOutput = gradOutputQueue_.template DeQue<T>();
        gradSelf = gradSelfQueue_.template DeQue<T>();

        LocalTensor<float> prefix = prefix_.Get<float>();
        Duplicate(prefix, 0.0f, rows * rowStride_);
        PipeBarrier<PIPE_V>();
        const uint16_t groups = (batches + BATCH_LANES - 1) / BATCH_LANES;
        asc_vf_call<BuildPrefix>(reinterpret_cast<__ubuf__ T*>(gradOutput.GetPhyAddr()),
                                 reinterpret_cast<__ubuf__ T*>(gradSelf.GetPhyAddr()),
                                 reinterpret_cast<__ubuf__ float*>(prefix.GetPhyAddr()), batches, length_, rowStride_,
                                 groups);
        PipeBarrier<PIPE_V>();
        gradOutputQueue_.FreeTensor(gradOutput);
        gradSelfQueue_.FreeTensor(gradSelf);

        uint32_t shape[] = {rows, rowStride_};
        if constexpr (std::is_same<T, float>::value) {
            ReduceSum<float, Pattern::Reduce::AR, false>(output, prefix, shape, true);
        } else {
            LocalTensor<float> reduced = reduced_.Get<float>();
            ReduceSum<float, Pattern::Reduce::AR, false>(reduced, prefix, shape, true);
            PipeBarrier<PIPE_V>();
            Cast(output, reduced, RoundMode::CAST_RINT, rows);
        }
        PipeBarrier<PIPE_V>();
    }

    TPipe pipe_;
    GlobalTensor<T> gradOutput_, gradSelf_, gradInput_;
    TQue<QuePosition::VECIN, 1> gradOutputQueue_, gradSelfQueue_;
    TQue<QuePosition::VECOUT, 1> result_;
    TBuf<QuePosition::VECCALC> prefix_, reduced_;
    int64_t batchStart_ = 0;
    int64_t batchEnd_ = 0;
    uint32_t length_ = 0;
    uint32_t batchesPerTile_ = 0;
    uint32_t matrixElems_ = 0;
    uint32_t rowStride_ = 0;
};

} // namespace ExpSegsumGradArch35

#endif // EXP_SEGSUM_GRAD_SHORT_BATCHES_ARCH35_H
