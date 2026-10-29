/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONCAT_D_ARCH31_H_
#define CONCAT_D_ARCH31_H_

#include "kernel_operator.h"
#include "interface/kernel_operator_list_tensor_intf.h"
#include "concat_d_tiling_data.h"

namespace ConcatDArch31 {
using namespace AscendC;
using concat_d_arch31::ConcatDTilingDataPod;

class KernelConcatD {
public:
    __aicore__ inline KernelConcatD() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ConcatDTilingDataPod* tiling)
    {
        y_ = y;
        tiling_ = tiling;
        blockIdx_ = static_cast<uint32_t>(GetBlockIdx());
        blockNum_ = static_cast<uint32_t>(GetBlockNum());
        inputList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));

        if ((tiling_->flags & concat_d_arch31::FLAG_COMPACT_SCATTER) != 0U) {
            pipe_.InitBuffer(scatterSrcQueue_, 2U, tiling_->tileBytes);
            pipe_.InitBuffer(scatterOutQueue_, 2U, tiling_->tileBytes);
            pipe_.InitBuffer(scatterIndexBuffer_, tiling_->tensorNum * concat_d_arch31::COMPACT_SCATTER_TABLE_BYTES);
        } else {
            pipe_.InitBuffer(copyQueue_, 2U, tiling_->tileBytes);
        }
    }

    __aicore__ inline void Process()
    {
        if (blockNum_ == 0U || tiling_->tensorNum == 0U || tiling_->tensorNum > concat_d_arch31::MAX_INPUTS ||
            tiling_->outerCount == 0U || tiling_->outputRowBytes == 0U) {
            return;
        }
        if ((tiling_->flags & concat_d_arch31::FLAG_COMPACT_SCATTER) != 0U) {
            if (blockNum_ == 1U && blockIdx_ == 0U) {
                ProcessCompactScatter();
            }
            return;
        }

        const uint32_t begin = tiling_->tensorNum * blockIdx_ / blockNum_;
        const uint32_t end = tiling_->tensorNum * (blockIdx_ + 1U) / blockNum_;
        uint64_t outputColumn = PrefixRowBytes(begin);
        for (uint32_t tensorIdx = begin; tensorIdx < end; ++tensorIdx) {
            const uint64_t rowBytes = tiling_->inputRowBytes[tensorIdx];
            __gm__ uint8_t* src = SourceAddress(tensorIdx);
            if (tiling_->outerCount == 1U || tiling_->tensorNum == 1U) {
                CopyContiguous(src, reinterpret_cast<__gm__ uint8_t*>(y_) + outputColumn * tiling_->outerCount,
                               rowBytes * tiling_->outerCount);
            } else {
                CopyTensorRows(src, outputColumn, rowBytes);
            }
            outputColumn += rowBytes;
        }
    }

private:
    __aicore__ inline void ProcessCompactScatter()
    {
        if (tiling_->tensorNum < 2U || tiling_->outputRowBytes % sizeof(uint16_t) != 0U ||
            tiling_->outputRowBytes > tiling_->tileBytes) {
            return;
        }
        const uint32_t outputWidth = static_cast<uint32_t>(tiling_->outputRowBytes / sizeof(uint16_t));
        LocalTensor<uint32_t> scatterTables = scatterIndexBuffer_.Get<uint32_t>();
        uint32_t tablePrefix = 0U;
        for (uint32_t tensorIdx = 0U; tensorIdx < tiling_->tensorNum; ++tensorIdx) {
            if (tiling_->inputRowBytes[tensorIdx] == 0U || tiling_->inputRowBytes[tensorIdx] % sizeof(uint16_t) != 0U) {
                return;
            }
            const uint32_t inputWidth = static_cast<uint32_t>(tiling_->inputRowBytes[tensorIdx] / sizeof(uint16_t));
            if (CanUseCachedScatterTable(inputWidth, outputWidth, tablePrefix)) {
                GenerateScatterOffsets(scatterTables[tensorIdx * concat_d_arch31::COMPACT_SCATTER_TABLE_ELEMS],
                                       inputWidth, outputWidth, tablePrefix);
            }
            tablePrefix += inputWidth;
        }

        const uint32_t rowsPerBatch = tiling_->tileBytes / (outputWidth * sizeof(uint16_t));
        GlobalTensor<uint16_t> outputGm;
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(y_), tiling_->outerCount * outputWidth);
        for (uint64_t outerBase = 0U; outerBase < tiling_->outerCount; outerBase += rowsPerBatch) {
            uint32_t batchRows = rowsPerBatch;
            if (tiling_->outerCount - outerBase < rowsPerBatch) {
                batchRows = static_cast<uint32_t>(tiling_->outerCount - outerBase);
            }

            LocalTensor<uint16_t> output = scatterOutQueue_.AllocTensor<uint16_t>();
            uint32_t outputPrefix = 0U;
            for (uint32_t tensorIdx = 0U; tensorIdx < tiling_->tensorNum; ++tensorIdx) {
                const uint32_t inputWidth = static_cast<uint32_t>(tiling_->inputRowBytes[tensorIdx] / sizeof(uint16_t));
                __gm__ uint8_t* input = SourceAddress(tensorIdx);
                ScatterInput(input + outerBase * tiling_->inputRowBytes[tensorIdx], batchRows * inputWidth,
                             scatterTables[tensorIdx * concat_d_arch31::COMPACT_SCATTER_TABLE_ELEMS], output,
                             inputWidth, outputWidth, outputPrefix);
                outputPrefix += inputWidth;
            }
            scatterOutQueue_.EnQue(output);
            output = scatterOutQueue_.DeQue<uint16_t>();
            CopyScatterOut(outputGm[outerBase * outputWidth], output, batchRows * outputWidth);
            scatterOutQueue_.FreeTensor(output);
        }
    }

    __aicore__ inline bool CanUseCachedScatterTable(uint32_t inputWidth, uint32_t outputWidth, uint32_t outputPrefix)
    {
        if (inputWidth == 0U || concat_d_arch31::COMPACT_SCATTER_TABLE_ELEMS % inputWidth != 0U) {
            return false;
        }
        const uint32_t lane = concat_d_arch31::COMPACT_SCATTER_TABLE_ELEMS - 1U;
        const uint32_t maxElements = (lane / inputWidth) * outputWidth + outputPrefix + lane % inputWidth;
        return maxElements <= UINT16_MAX / sizeof(uint16_t);
    }

    __aicore__ inline void GenerateScatterOffsets(LocalTensor<uint32_t> offsets, uint32_t inputWidth,
                                                  uint32_t outputWidth, uint32_t outputPrefix)
    {
        auto dstAddr = reinterpret_cast<__ubuf__ int32_t*>(offsets.GetPhyAddr());
        for (uint32_t base = 0U; base < concat_d_arch31::COMPACT_SCATTER_TABLE_ELEMS; base += 64U) {
            __VEC_SCOPE__
            {
                MicroAPI::RegTensor<int32_t> position;
                MicroAPI::RegTensor<int32_t> divisor;
                MicroAPI::RegTensor<int32_t> row;
                MicroAPI::RegTensor<int32_t> rowStart;
                MicroAPI::RegTensor<int32_t> column;
                MicroAPI::RegTensor<int32_t> index;
                MicroAPI::MaskReg mask = MicroAPI::CreateMask<uint32_t>();
                MicroAPI::Arange<int32_t>(position, static_cast<int32_t>(base));
                MicroAPI::Duplicate<int32_t>(divisor, static_cast<int32_t>(inputWidth));
                MicroAPI::Div<int32_t>(row, position, divisor, mask);
                MicroAPI::Muls<int32_t>(rowStart, row, static_cast<int32_t>(inputWidth), mask);
                MicroAPI::Sub<int32_t>(column, position, rowStart, mask);
                MicroAPI::Muls<int32_t>(index, row, static_cast<int32_t>(outputWidth), mask);
                MicroAPI::Add<int32_t>(index, index, column, mask);
                MicroAPI::Adds<int32_t>(index, index, static_cast<int32_t>(outputPrefix), mask);
                MicroAPI::Muls<int32_t>(index, index, 2, mask);
                MicroAPI::DataCopy<int32_t, MicroAPI::StoreDist::DIST_NORM>(dstAddr + base, index, mask);
            }
        }
    }

    __aicore__ inline void CopyScatterIn(LocalTensor<uint16_t> dst, __gm__ uint8_t* src, uint32_t elements)
    {
        GlobalTensor<uint16_t> srcGm;
        srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(src), elements);
        const DataCopyExtParams copy{1U, static_cast<uint32_t>(elements * sizeof(uint16_t)), 0U, 0U, 0U};
        const DataCopyPadExtParams<uint16_t> noPad{false, 0U, 0U, 0U};
        DataCopyPad(dst, srcGm, copy, noPad);
    }

    __aicore__ inline void CopyScatterOut(GlobalTensor<uint16_t> dst, LocalTensor<uint16_t> src, uint32_t elements)
    {
        const DataCopyExtParams copy{1U, static_cast<uint32_t>(elements * sizeof(uint16_t)), 0U, 0U, 0U};
        DataCopyPad(dst, src, copy);
    }

    __aicore__ inline void ScatterInput(__gm__ uint8_t* srcGm, uint32_t elements, LocalTensor<uint32_t> offsets,
                                        LocalTensor<uint16_t> output, uint32_t inputWidth, uint32_t outputWidth,
                                        uint32_t outputPrefix)
    {
        LocalTensor<uint16_t> src = scatterSrcQueue_.AllocTensor<uint16_t>();
        CopyScatterIn(src, srcGm, elements);
        scatterSrcQueue_.EnQue(src);
        src = scatterSrcQueue_.DeQue<uint16_t>();

        if (CanUseCachedScatterTable(inputWidth, outputWidth, outputPrefix)) {
            ScatterWithCachedTable(src, elements, offsets, output, inputWidth, outputWidth);
        } else {
            ScatterWithDynamicIndices(src, elements, output, inputWidth, outputWidth, outputPrefix);
        }
        scatterSrcQueue_.FreeTensor(src);
    }

    __aicore__ inline void ScatterWithCachedTable(LocalTensor<uint16_t> src, uint32_t elements,
                                                  LocalTensor<uint32_t> offsets, LocalTensor<uint16_t> output,
                                                  uint32_t inputWidth, uint32_t outputWidth)
    {
        uint32_t handled = 0U;
        while (elements - handled >= concat_d_arch31::COMPACT_SCATTER_VECTOR_ELEMS) {
            uint32_t repeats = (elements - handled) / concat_d_arch31::COMPACT_SCATTER_VECTOR_ELEMS;
            if (repeats > concat_d_arch31::COMPACT_SCATTER_REPEATS) {
                repeats = concat_d_arch31::COMPACT_SCATTER_REPEATS;
            }
            const uint32_t rowsHandled = handled / inputWidth;
            const uint32_t dstBaseBytes = rowsHandled * outputWidth * sizeof(uint16_t);
            Scatter<uint16_t>(output, src[handled], offsets, dstBaseBytes,
                              static_cast<uint64_t>(concat_d_arch31::COMPACT_SCATTER_VECTOR_ELEMS),
                              static_cast<uint8_t>(repeats), 8U);
            handled += repeats * concat_d_arch31::COMPACT_SCATTER_VECTOR_ELEMS;
        }
        if (handled < elements) {
            const uint32_t rowsHandled = handled / inputWidth;
            const uint32_t dstBaseBytes = rowsHandled * outputWidth * sizeof(uint16_t);
            Scatter<uint16_t>(output, src[handled], offsets, dstBaseBytes, static_cast<uint64_t>(elements - handled),
                              1U, 8U);
        }
    }

    __aicore__ inline void ScatterWithDynamicIndices(LocalTensor<uint16_t> src, uint32_t elements,
                                                     LocalTensor<uint16_t> output, uint32_t inputWidth,
                                                     uint32_t outputWidth, uint32_t outputPrefix)
    {
        for (uint32_t handled = 0U; handled < elements; handled += concat_d_arch31::COMPACT_SCATTER_VECTOR_ELEMS) {
            uint32_t active = elements - handled;
            if (active > concat_d_arch31::COMPACT_SCATTER_VECTOR_ELEMS) {
                active = concat_d_arch31::COMPACT_SCATTER_VECTOR_ELEMS;
            }
            auto srcAddr = reinterpret_cast<__ubuf__ uint16_t*>(src.GetPhyAddr()) + handled;
            auto dstAddr = reinterpret_cast<__ubuf__ uint16_t*>(output.GetPhyAddr());
            __VEC_SCOPE__
            {
                MicroAPI::RegTensor<uint16_t> srcReg;
                MicroAPI::RegTensor<int16_t> position;
                MicroAPI::RegTensor<int16_t> divisor;
                MicroAPI::RegTensor<int16_t> row;
                MicroAPI::RegTensor<int16_t> rowStart;
                MicroAPI::RegTensor<int16_t> column;
                MicroAPI::RegTensor<int16_t> index;
                uint32_t maskCount = active;
                MicroAPI::MaskReg mask = MicroAPI::UpdateMask<uint16_t>(maskCount);
                MicroAPI::DataCopy<uint16_t, MicroAPI::LoadDist::DIST_NORM>(srcReg, srcAddr);
                MicroAPI::Arange<int16_t>(position, static_cast<int16_t>(handled));
                MicroAPI::Duplicate<int16_t>(divisor, static_cast<int16_t>(inputWidth));
                MicroAPI::Div<int16_t>(row, position, divisor, mask);
                MicroAPI::Muls<int16_t>(rowStart, row, static_cast<int16_t>(inputWidth), mask);
                MicroAPI::Sub<int16_t>(column, position, rowStart, mask);
                MicroAPI::Muls<int16_t>(index, row, static_cast<int16_t>(outputWidth), mask);
                MicroAPI::Add<int16_t>(index, index, column, mask);
                MicroAPI::Adds<int16_t>(index, index, static_cast<int16_t>(outputPrefix), mask);
                MicroAPI::DataCopyScatter<uint16_t, uint16_t>(
                    dstAddr, srcReg, reinterpret_cast<MicroAPI::RegTensor<uint16_t>&>(index), mask);
            }
        }
    }

    __aicore__ inline uint64_t PrefixRowBytes(uint32_t tensorIdx)
    {
        uint64_t prefix = 0U;
        for (uint32_t i = 0U; i < tensorIdx; ++i) {
            prefix += tiling_->inputRowBytes[i];
        }
        return prefix;
    }

    __aicore__ inline __gm__ uint8_t* SourceAddress(uint32_t tensorIdx)
    {
        return inputList_.GetDataPtr<uint8_t>(tensorIdx);
    }

    __aicore__ inline uint32_t AlignUp32(uint32_t value) { return (value + 31U) & ~31U; }

    __aicore__ inline bool IsAlignedTransfer(__gm__ uint8_t* src, __gm__ uint8_t* dst, uint32_t bytes)
    {
        return ((reinterpret_cast<uint64_t>(src) | reinterpret_cast<uint64_t>(dst) | bytes) & 31U) == 0U;
    }

    __aicore__ inline void CopyContiguous(__gm__ uint8_t* src, __gm__ uint8_t* dst, uint64_t totalBytes)
    {
        for (uint64_t offset = 0U; offset < totalBytes; offset += tiling_->tileBytes) {
            uint32_t activeBytes = tiling_->tileBytes;
            if (totalBytes - offset < tiling_->tileBytes) {
                activeBytes = static_cast<uint32_t>(totalBytes - offset);
            }
            if (IsAlignedTransfer(src + offset, dst + offset, activeBytes)) {
                CopyAlignedBytes(src + offset, dst + offset, activeBytes);
            } else {
                CopyBytes(src + offset, dst + offset, activeBytes);
            }
        }
    }

    __aicore__ inline void CopyTensorRows(__gm__ uint8_t* src, uint64_t outputColumn, uint64_t rowBytes)
    {
        if (rowBytes <= UINT32_MAX - 31U && tiling_->outputRowBytes - rowBytes <= UINT32_MAX) {
            const uint32_t rowBytes32 = static_cast<uint32_t>(rowBytes);
            const uint32_t ubRowBytes = AlignUp32(rowBytes32);
            if (ubRowBytes <= tiling_->tileBytes) {
                const uint32_t rowsPerBatch = tiling_->tileBytes / ubRowBytes;
                for (uint64_t outerBase = 0U; outerBase < tiling_->outerCount; outerBase += rowsPerBatch) {
                    uint32_t batchRows = rowsPerBatch;
                    if (tiling_->outerCount - outerBase < rowsPerBatch) {
                        batchRows = static_cast<uint32_t>(tiling_->outerCount - outerBase);
                    }
                    CopyRowBatch(
                        src + outerBase * rowBytes,
                        reinterpret_cast<__gm__ uint8_t*>(y_) + outerBase * tiling_->outputRowBytes + outputColumn,
                        batchRows, rowBytes32, static_cast<uint32_t>(tiling_->outputRowBytes - rowBytes));
                }
                return;
            }
        }

        for (uint64_t outerIdx = 0U; outerIdx < tiling_->outerCount; ++outerIdx) {
            __gm__ uint8_t* rowSrc = src + outerIdx * rowBytes;
            __gm__ uint8_t* rowDst = reinterpret_cast<__gm__ uint8_t*>(y_) + outerIdx * tiling_->outputRowBytes +
                                     outputColumn;
            CopyContiguous(rowSrc, rowDst, rowBytes);
        }
    }

    __aicore__ inline void CopyRowBatch(__gm__ uint8_t* src, __gm__ uint8_t* dst, uint32_t batchRows, uint32_t rowBytes,
                                        uint32_t dstStrideBytes)
    {
        GlobalTensor<uint16_t> srcGm;
        GlobalTensor<uint16_t> dstGm;
        const uint64_t srcSpan = static_cast<uint64_t>(batchRows) * rowBytes;
        const uint64_t dstSpan = static_cast<uint64_t>(batchRows - 1U) *
                                     (static_cast<uint64_t>(rowBytes) + dstStrideBytes) +
                                 rowBytes;
        srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(src), (srcSpan + 1U) / sizeof(uint16_t));
        dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(dst), (dstSpan + 1U) / sizeof(uint16_t));

        LocalTensor<uint16_t> local = copyQueue_.AllocTensor<uint16_t>();
        const DataCopyExtParams gather{static_cast<uint16_t>(batchRows), rowBytes, 0U, 0U, 0U};
        const DataCopyPadExtParams<uint16_t> noPad{false, 0U, 0U, 0U};
        DataCopyPad(local, srcGm, gather, noPad);
        copyQueue_.EnQue(local);
        local = copyQueue_.DeQue<uint16_t>();
        const DataCopyExtParams scatter{static_cast<uint16_t>(batchRows), rowBytes, 0U, dstStrideBytes, 0U};
        DataCopyPad(dstGm, local, scatter);
        copyQueue_.FreeTensor(local);
    }

    __aicore__ inline void CopyAlignedBytes(__gm__ uint8_t* src, __gm__ uint8_t* dst, uint32_t activeBytes)
    {
        GlobalTensor<uint16_t> srcGm;
        GlobalTensor<uint16_t> dstGm;
        const uint32_t activeElements = activeBytes / sizeof(uint16_t);
        srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(src), activeElements);
        dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(dst), activeElements);
        LocalTensor<uint16_t> local = copyQueue_.AllocTensor<uint16_t>();
        DataCopy(local, srcGm, activeElements);
        copyQueue_.EnQue(local);
        local = copyQueue_.DeQue<uint16_t>();
        DataCopy(dstGm, local, activeElements);
        copyQueue_.FreeTensor(local);
    }

    __aicore__ inline void CopyBytes(__gm__ uint8_t* src, __gm__ uint8_t* dst, uint32_t activeBytes)
    {
        GlobalTensor<uint16_t> srcGm;
        GlobalTensor<uint16_t> dstGm;
        srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(src), (activeBytes + 1U) / sizeof(uint16_t));
        dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(dst), (activeBytes + 1U) / sizeof(uint16_t));
        LocalTensor<uint16_t> local = copyQueue_.AllocTensor<uint16_t>();
        const DataCopyExtParams params{1U, activeBytes, 0U, 0U, 0U};
        const DataCopyPadExtParams<uint16_t> noPad{false, 0U, 0U, 0U};
        DataCopyPad(local, srcGm, params, noPad);
        copyQueue_.EnQue(local);
        local = copyQueue_.DeQue<uint16_t>();
        DataCopyPad(dstGm, local, params);
        copyQueue_.FreeTensor(local);
    }

    TPipe pipe_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 2> copyQueue_;
    TQue<QuePosition::VECIN, 2> scatterSrcQueue_;
    TQue<QuePosition::VECOUT, 2> scatterOutQueue_;
    TBuf<TPosition::VECCALC> scatterIndexBuffer_;
    ListTensorDesc inputList_;
    GM_ADDR y_ = nullptr;
    const ConcatDTilingDataPod* tiling_ = nullptr;
    uint32_t blockIdx_ = 0U;
    uint32_t blockNum_ = 0U;
};
} // namespace ConcatDArch31

#endif // CONCAT_D_ARCH31_H_
