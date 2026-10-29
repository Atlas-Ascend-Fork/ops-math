/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPOSE_TRANSDATA_5HD_021_ARCH31_H
#define TRANSPOSE_TRANSDATA_5HD_021_ARCH31_H

#include "transpose_base_arch31.h"

namespace Transpose {
constexpr uint32_t KIRIN_021_TRANS_ELEMS = 16;
constexpr uint32_t KIRIN_021_BLOCK_BYTES = 32;
constexpr uint32_t KIRIN_021_BUFFER_COUNT = 2;

template <typename T>
class KirinKernelTransDataTo5HD021 {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const KirinTranspose021VConvOpTilingData* tiling, TPipe* pipe)
    {
        tiling_ = tiling;
        blockIdx_ = GetBlockIdx();
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        if (tiling_->useHSplit) {
            const uint64_t fullCores = tiling_->usedCoreNum - 1;
            hStart_ = blockIdx_ < fullCores ? blockIdx_ * tiling_->hPerCore : fullCores * tiling_->hPerCore;
            hCount_ = blockIdx_ < fullCores ? tiling_->hPerCore : tiling_->hTailCore;
            nCount_ = tiling_->nLen;
        } else {
            const uint64_t fullCores = tiling_->usedCoreNum - 1;
            nStart_ = blockIdx_ < fullCores ? blockIdx_ * tiling_->nPerCore : fullCores * tiling_->nPerCore;
            nCount_ = blockIdx_ < fullCores ? tiling_->nPerCore : tiling_->nTailCore;
        }
        const uint32_t bufferCount = tiling_->useTwoDimensionalSplit ? 1 : KIRIN_021_BUFFER_COUNT;
        pipe->InitBuffer(inputQueue_, bufferCount, static_cast<uint32_t>(tiling_->availableUbSize));
        pipe->InitBuffer(outputQueue_, bufferCount, static_cast<uint32_t>(tiling_->availableUbSize));
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= tiling_->usedCoreNum) {
            return;
        }
        for (uint64_t n = 0; n < nCount_; ++n) {
            if (tiling_->useHSplit) {
                ProcessHSplit(n);
            } else {
                ProcessBatch(nStart_ + n);
            }
        }
    }

private:
    static constexpr uint32_t blockElems = KIRIN_021_BLOCK_BYTES / sizeof(T);
    static constexpr uint32_t dstStrideFactor = sizeof(T) == 4 ? 2 : 1;

    __aicore__ inline void Compute8Bit(uint32_t rows, uint32_t columns, uint32_t rowGroups)
    {
        LocalTensor<T> src = inputQueue_.DeQue<T>();
        LocalTensor<T> dst = outputQueue_.AllocTensor<T>();
        const uint16_t columnBlocks = static_cast<uint16_t>(columns / blockElems);
        const uint8_t evenRepeats = static_cast<uint8_t>((rowGroups + 1) / 2);
        const uint8_t oddRepeats = static_cast<uint8_t>(rowGroups / 2);
        TransDataTo5HDParams even{false, false, evenRepeats,
                                  static_cast<uint16_t>(evenRepeats <= 1 ? 0 : dstStrideFactor),
                                  static_cast<uint16_t>(evenRepeats <= 1 ? 0 : columns)};
        TransDataTo5HDParams odd{true, false, oddRepeats, static_cast<uint16_t>(oddRepeats <= 1 ? 0 : dstStrideFactor),
                                 static_cast<uint16_t>(oddRepeats <= 1 ? 0 : columns)};
        for (uint16_t columnBlock = 0; columnBlock < columnBlocks; ++columnBlock) {
            uint64_t evenSrc[KIRIN_021_TRANS_ELEMS];
            uint64_t oddSrc[KIRIN_021_TRANS_ELEMS];
            for (uint16_t row = 0; row < KIRIN_021_TRANS_ELEMS; ++row) {
                evenSrc[row] = src[row * columns + columnBlock * blockElems].GetPhyAddr();
                if (oddRepeats > 0) {
                    oddSrc[row] = src[(row + KIRIN_021_TRANS_ELEMS) * columns + columnBlock * blockElems].GetPhyAddr();
                }
            }
            for (uint16_t half = 0; half < 2; ++half) {
                uint64_t dstList[KIRIN_021_TRANS_ELEMS];
                const uint16_t dstRow = columnBlock * blockElems + half * KIRIN_021_TRANS_ELEMS;
                for (uint16_t index = 0; index < KIRIN_021_TRANS_ELEMS; ++index) {
                    dstList[index] = dst[(dstRow + index) * rows].GetPhyAddr();
                }
                even.srcHighHalf = half != 0;
                TransDataTo5HD<T>(dstList, evenSrc, even);
                if (oddRepeats > 0) {
                    PipeBarrier<PIPE_V>();
                    odd.srcHighHalf = half != 0;
                    TransDataTo5HD<T>(dstList, oddSrc, odd);
                }
                if (half == 0) {
                    PipeBarrier<PIPE_V>();
                }
            }
        }
        outputQueue_.EnQue<T>(dst);
        inputQueue_.FreeTensor(src);
    }

    __aicore__ inline void ComputeRConv(uint32_t rows, uint32_t columns, uint32_t rowGroups, uint32_t columnGroups)
    {
        if constexpr (sizeof(T) == 1) {
            Compute8Bit(rows, columns, rowGroups);
            return;
        }
        LocalTensor<T> src = inputQueue_.DeQue<T>();
        LocalTensor<T> dst = outputQueue_.AllocTensor<T>();
        TransDataTo5HDParams params{
            false, false, static_cast<uint8_t>(rows / KIRIN_021_TRANS_ELEMS),
            static_cast<uint16_t>(rows == KIRIN_021_TRANS_ELEMS ? 0 : dstStrideFactor),
            static_cast<uint16_t>(rows == KIRIN_021_TRANS_ELEMS ? 0 : columnGroups * KIRIN_021_TRANS_ELEMS)};
        for (uint16_t column = 0; column < columnGroups; ++column) {
            uint64_t srcList[KIRIN_021_TRANS_ELEMS];
            uint64_t dstList[KIRIN_021_TRANS_ELEMS];
            for (uint16_t row = 0; row < KIRIN_021_TRANS_ELEMS; ++row) {
                srcList[row] = src[row * columns + column * blockElems].GetPhyAddr();
            }
            if constexpr (sizeof(T) == 4) {
                for (uint16_t row = 0; row < KIRIN_021_TRANS_ELEMS / 2; ++row) {
                    dstList[2 * row] = dst[(row + column * blockElems) * rows].GetPhyAddr();
                    dstList[2 * row + 1] = dst[(row + column * blockElems) * rows + blockElems].GetPhyAddr();
                }
            } else {
                for (uint16_t row = 0; row < KIRIN_021_TRANS_ELEMS; ++row) {
                    dstList[row] = dst[(row + column * KIRIN_021_TRANS_ELEMS) * rows].GetPhyAddr();
                }
            }
            TransDataTo5HD<T>(dstList, srcList, params);
        }
        outputQueue_.EnQue<T>(dst);
        inputQueue_.FreeTensor(src);
    }

    __aicore__ inline void ComputeCConv(uint32_t rows, uint32_t columns, uint32_t rowGroups)
    {
        if constexpr (sizeof(T) == 1) {
            Compute8Bit(rows, columns, rowGroups);
            return;
        }
        LocalTensor<T> src = inputQueue_.DeQue<T>();
        LocalTensor<T> dst = outputQueue_.AllocTensor<T>();
        const uint8_t repeats = static_cast<uint8_t>(columns / blockElems);
        TransDataTo5HDParams params{false, false, repeats,
                                    static_cast<uint16_t>(repeats <= 1 ? 0 : rowGroups * KIRIN_021_TRANS_ELEMS),
                                    static_cast<uint16_t>(repeats <= 1 ? 0 : 1)};
        for (uint16_t group = 0; group < rowGroups; ++group) {
            uint64_t srcList[KIRIN_021_TRANS_ELEMS];
            uint64_t dstList[KIRIN_021_TRANS_ELEMS];
            for (uint16_t row = 0; row < KIRIN_021_TRANS_ELEMS; ++row) {
                srcList[row] = src[row * columns + group * KIRIN_021_TRANS_ELEMS * columns].GetPhyAddr();
            }
            if constexpr (sizeof(T) == 4) {
                for (uint16_t row = 0; row < KIRIN_021_TRANS_ELEMS / 2; ++row) {
                    dstList[2 * row] = dst[row * rows + group * KIRIN_021_TRANS_ELEMS].GetPhyAddr();
                    dstList[2 * row + 1] = dst[row * rows + group * KIRIN_021_TRANS_ELEMS + blockElems].GetPhyAddr();
                }
            } else {
                for (uint16_t row = 0; row < KIRIN_021_TRANS_ELEMS; ++row) {
                    dstList[row] = dst[row * rows + group * KIRIN_021_TRANS_ELEMS].GetPhyAddr();
                }
            }
            TransDataTo5HD<T>(dstList, srcList, params);
        }
        outputQueue_.EnQue<T>(dst);
        inputQueue_.FreeTensor(src);
    }

    __aicore__ inline void Compute(uint32_t rows, uint32_t columns, uint32_t rowGroups, uint32_t columnGroups)
    {
        if (tiling_->useRConv) {
            ComputeRConv(rows, columns, rowGroups, columnGroups);
        } else {
            ComputeCConv(rows, columns, rowGroups);
        }
    }

    __aicore__ inline void CopyIn(uint64_t n, uint32_t loop, uint32_t rows, uint32_t columns)
    {
        LocalTensor<T> local = inputQueue_.AllocTensor<T>();
        const uint64_t batchOffset = n * tiling_->hLen * tiling_->wLen;
        const uint64_t tileOffset = tiling_->useRConv ?
                                        static_cast<uint64_t>(loop) * tiling_->rUbSplitPara.ubAlignFactor *
                                            KIRIN_021_TRANS_ELEMS * tiling_->wLen :
                                        static_cast<uint64_t>(loop) * tiling_->cUbSplitPara.ubAlignFactor *
                                            KIRIN_021_TRANS_ELEMS;
        const uint64_t rowOffset = tiling_->useRConv ? static_cast<uint64_t>(loop) *
                                                           tiling_->rUbSplitPara.ubAlignFactor * KIRIN_021_TRANS_ELEMS :
                                                       0;
        const uint32_t validRows = static_cast<uint32_t>(
            tiling_->useRConv ? (tiling_->hLen - rowOffset < rows ? tiling_->hLen - rowOffset : rows) : tiling_->hLen);
        DataCopyExtParams params;
        const uint32_t alignment = sizeof(T) == 1 ? blockElems : KIRIN_021_TRANS_ELEMS;
        if (tiling_->wLen % alignment == 0 && (tiling_->useRConv || columns == tiling_->wLen)) {
            params = {1, static_cast<uint32_t>(validRows * columns * sizeof(T)), 0, 0, 0};
        } else {
            const uint64_t columnOffset = tiling_->useRConv ?
                                              0 :
                                              static_cast<uint64_t>(loop) * tiling_->cUbSplitPara.ubAlignFactor *
                                                  KIRIN_021_TRANS_ELEMS;
            const uint32_t validColumns = static_cast<uint32_t>(
                tiling_->wLen - columnOffset < columns ? tiling_->wLen - columnOffset : columns);
            params = {static_cast<uint16_t>(validRows), static_cast<uint32_t>(validColumns * sizeof(T)),
                      static_cast<uint32_t>((tiling_->wLen - validColumns) * sizeof(T)),
                      static_cast<uint32_t>((columns - validColumns) * sizeof(T) / KIRIN_021_BLOCK_BYTES), 0};
        }
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(local, input_[batchOffset + tileOffset], params, pad);
        inputQueue_.EnQue(local);
    }

    __aicore__ inline void CopyOut(uint64_t n, uint32_t loop, uint32_t rows, uint32_t columns)
    {
        LocalTensor<T> local = outputQueue_.DeQue<T>();
        const uint64_t batchOffset = n * tiling_->wLen * tiling_->hLen;
        const uint64_t tileOffset = tiling_->useRConv ?
                                        static_cast<uint64_t>(loop) * tiling_->rUbSplitPara.ubAlignFactor *
                                            KIRIN_021_TRANS_ELEMS :
                                        static_cast<uint64_t>(loop) * tiling_->cUbSplitPara.ubAlignFactor *
                                            KIRIN_021_TRANS_ELEMS * tiling_->hLen;
        const uint64_t rowOffset = tiling_->useRConv ? tileOffset : 0;
        const uint64_t columnOffset = tiling_->useRConv ?
                                          0 :
                                          static_cast<uint64_t>(loop) * tiling_->cUbSplitPara.ubAlignFactor *
                                              KIRIN_021_TRANS_ELEMS;
        const uint32_t validRows = static_cast<uint32_t>(tiling_->hLen - rowOffset < rows ? tiling_->hLen - rowOffset :
                                                                                            rows);
        const uint32_t validColumns = static_cast<uint32_t>(
            tiling_->wLen - columnOffset < columns ? tiling_->wLen - columnOffset : columns);
        DataCopyExtParams params;
        const uint32_t alignment = sizeof(T) == 1 ? blockElems : KIRIN_021_TRANS_ELEMS;
        if (tiling_->hLen % alignment == 0 && tiling_->wLen % alignment == 0 &&
            (!tiling_->useRConv || rows == tiling_->hLen)) {
            params = {1, static_cast<uint32_t>(rows * columns * sizeof(T)), 0, 0, 0};
        } else {
            params = {static_cast<uint16_t>(validColumns), static_cast<uint32_t>(validRows * sizeof(T)),
                      static_cast<uint32_t>((rows - validRows) * sizeof(T) / KIRIN_021_BLOCK_BYTES),
                      static_cast<uint32_t>(tiling_->useRConv ? (tiling_->hLen - validRows) * sizeof(T) : 0), 0};
        }
        DataCopyPad(output_[batchOffset + tileOffset], local, params);
        outputQueue_.FreeTensor(local);
    }

    __aicore__ inline void ProcessBatch(uint64_t n)
    {
        for (uint32_t loop = 0; loop < tiling_->ubLoopCount; ++loop) {
            const bool tail = loop + 1 == tiling_->ubLoopCount;
            const uint32_t rowGroups = static_cast<uint32_t>(tail ? tiling_->rUbSplitPara.ubTailAlignFactor :
                                                                    tiling_->rUbSplitPara.ubAlignFactor);
            const uint32_t rows = static_cast<uint32_t>(tail ? tiling_->rUbSplitPara.ubTailFactor :
                                                               tiling_->rUbSplitPara.ubFactor);
            const uint32_t columnGroups = static_cast<uint32_t>(tail ? tiling_->cUbSplitPara.ubTailAlignFactor :
                                                                       tiling_->cUbSplitPara.ubAlignFactor);
            const uint32_t columns = static_cast<uint32_t>(tail ? tiling_->cUbSplitPara.ubTailFactor :
                                                                  tiling_->cUbSplitPara.ubFactor);
            CopyIn(n, loop, rows, columns);
            Compute(rows, columns, rowGroups, columnGroups);
            CopyOut(n, loop, rows, columns);
        }
    }

    __aicore__ inline void ProcessHSplit(uint64_t n)
    {
        if (tiling_->useTwoDimensionalSplit) {
            ProcessTwoDimensional(n);
            return;
        }
        const uint32_t rows = static_cast<uint32_t>(hCount_);
        const uint32_t rowGroups = (rows + KIRIN_021_TRANS_ELEMS - 1) / KIRIN_021_TRANS_ELEMS;
        for (uint32_t loop = 0; loop < tiling_->ubLoopCount; ++loop) {
            const bool tail = loop + 1 == tiling_->ubLoopCount;
            const uint32_t columnGroups = static_cast<uint32_t>(tail ? tiling_->cUbSplitPara.ubTailAlignFactor :
                                                                       tiling_->cUbSplitPara.ubAlignFactor);
            const uint32_t columns = static_cast<uint32_t>(tail ? tiling_->cUbSplitPara.ubTailFactor :
                                                                  tiling_->cUbSplitPara.ubFactor);
            LocalTensor<T> inputLocal = inputQueue_.AllocTensor<T>();
            const uint64_t columnOffset = static_cast<uint64_t>(loop) * tiling_->cUbSplitPara.ubFactor;
            const uint32_t validRows = static_cast<uint32_t>(
                tiling_->hLen - hStart_ < hCount_ ? tiling_->hLen - hStart_ : hCount_);
            const uint32_t validColumns = static_cast<uint32_t>(
                tiling_->wLen - columnOffset < columns ? tiling_->wLen - columnOffset : columns);
            DataCopyExtParams inputParams{
                static_cast<uint16_t>(validRows), static_cast<uint32_t>(validColumns * sizeof(T)),
                static_cast<uint32_t>((tiling_->wLen - validColumns) * sizeof(T)),
                static_cast<uint32_t>((columns - validColumns) * sizeof(T) / KIRIN_021_BLOCK_BYTES), 0};
            DataCopyPadExtParams<T> pad{false, 0, 0, 0};
            DataCopyPad(inputLocal, input_[n * tiling_->hLen * tiling_->wLen + hStart_ * tiling_->wLen + columnOffset],
                        inputParams, pad);
            inputQueue_.EnQue(inputLocal);
            Compute(rows, columns, rowGroups, columnGroups);

            LocalTensor<T> outputLocal = outputQueue_.DeQue<T>();
            DataCopyExtParams outputParams{
                static_cast<uint16_t>(validColumns), static_cast<uint32_t>(validRows * sizeof(T)),
                static_cast<uint32_t>((rows - validRows) * sizeof(T) / KIRIN_021_BLOCK_BYTES),
                static_cast<uint32_t>((tiling_->hLen - validRows) * sizeof(T)), 0};
            const uint64_t outputOffset = n * tiling_->wLen * tiling_->hLen + columnOffset * tiling_->hLen + hStart_;
            DataCopyPad(output_[outputOffset], outputLocal, outputParams);
            outputQueue_.FreeTensor(outputLocal);
        }
    }

    __aicore__ inline void ProcessTwoDimensional(uint64_t n)
    {
        const uint64_t batchInputOffset = n * tiling_->hLen * tiling_->wLen;
        const uint64_t batchOutputOffset = n * tiling_->wLen * tiling_->hLen;
        for (uint32_t rLoop = 0; rLoop < tiling_->rUbSplitPara.ubCount; ++rLoop) {
            const bool rTail = rLoop + 1 == tiling_->rUbSplitPara.ubCount;
            const uint32_t rowGroups = static_cast<uint32_t>(rTail ? tiling_->rUbSplitPara.ubTailAlignFactor :
                                                                     tiling_->rUbSplitPara.ubAlignFactor);
            const uint32_t rows = static_cast<uint32_t>(rTail ? tiling_->rUbSplitPara.ubTailFactor :
                                                                tiling_->rUbSplitPara.ubFactor);
            const uint64_t rowOffset = static_cast<uint64_t>(rLoop) * tiling_->rUbSplitPara.ubFactor;
            const uint64_t globalRowOffset = hStart_ + rowOffset;
            const uint32_t validRows = static_cast<uint32_t>(
                tiling_->hLen - globalRowOffset < rows ? tiling_->hLen - globalRowOffset : rows);

            for (uint32_t cLoop = 0; cLoop < tiling_->cUbSplitPara.ubCount; ++cLoop) {
                const bool cTail = cLoop + 1 == tiling_->cUbSplitPara.ubCount;
                const uint32_t columnGroups = static_cast<uint32_t>(cTail ? tiling_->cUbSplitPara.ubTailAlignFactor :
                                                                            tiling_->cUbSplitPara.ubAlignFactor);
                const uint32_t columns = static_cast<uint32_t>(cTail ? tiling_->cUbSplitPara.ubTailFactor :
                                                                       tiling_->cUbSplitPara.ubFactor);
                const uint64_t columnOffset = static_cast<uint64_t>(cLoop) * tiling_->cUbSplitPara.ubFactor;
                const uint32_t validColumns = static_cast<uint32_t>(
                    tiling_->wLen - columnOffset < columns ? tiling_->wLen - columnOffset : columns);

                LocalTensor<T> inputLocal = inputQueue_.AllocTensor<T>();
                DataCopyExtParams inputParams{
                    static_cast<uint16_t>(validRows), static_cast<uint32_t>(validColumns * sizeof(T)),
                    static_cast<uint32_t>((tiling_->wLen - validColumns) * sizeof(T)),
                    static_cast<uint32_t>((columns - validColumns) * sizeof(T) / KIRIN_021_BLOCK_BYTES), 0};
                DataCopyPadExtParams<T> pad{false, 0, 0, 0};
                const uint64_t inputOffset = batchInputOffset + globalRowOffset * tiling_->wLen + columnOffset;
                DataCopyPad(inputLocal, input_[inputOffset], inputParams, pad);
                inputQueue_.EnQue(inputLocal);
                ComputeRConv(rows, columns, rowGroups, columnGroups);

                LocalTensor<T> outputLocal = outputQueue_.DeQue<T>();
                DataCopyExtParams outputParams{
                    static_cast<uint16_t>(validColumns), static_cast<uint32_t>(validRows * sizeof(T)),
                    static_cast<uint32_t>((rows - validRows) * sizeof(T) / KIRIN_021_BLOCK_BYTES),
                    static_cast<uint32_t>((tiling_->hLen - validRows) * sizeof(T)), 0};
                const uint64_t outputOffset = batchOutputOffset + columnOffset * tiling_->hLen + globalRowOffset;
                DataCopyPad(output_[outputOffset], outputLocal, outputParams);
                outputQueue_.FreeTensor(outputLocal);
            }
        }
    }

    const KirinTranspose021VConvOpTilingData* tiling_ = nullptr;
    TQue<TPosition::VECIN, 1> inputQueue_;
    TQue<TPosition::VECOUT, 1> outputQueue_;
    GlobalTensor<T> input_;
    GlobalTensor<T> output_;
    uint64_t blockIdx_ = 0;
    uint64_t nStart_ = 0;
    uint64_t nCount_ = 0;
    uint64_t hStart_ = 0;
    uint64_t hCount_ = 0;
};
} // namespace Transpose

#endif // TRANSPOSE_TRANSDATA_5HD_021_ARCH31_H
