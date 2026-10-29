/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPOSE_STRIDED_DATACOPY_ARCH31_H
#define TRANSPOSE_STRIDED_DATACOPY_ARCH31_H

#include "transpose_base_arch31.h"

namespace Transpose {
template <typename T>
class TransposeStridedDataCopy {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const KirinTransposeStridedDataCopyOpTilingData* tiling,
                                TPipe* pipe)
    {
        tiling_ = tiling;
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        const uint32_t bufBytes = static_cast<uint32_t>(tiling_->mTile * tiling_->wBytes);
        pipe->InitBuffer(inQueue_, 2, bufBytes);
    }

    __aicore__ inline void Process()
    {
        W_ = tiling_->wLen;
        W_bytes_ = tiling_->wBytes;
        M_ = tiling_->mLen;
        N_ = tiling_->nLen;
        B_ = tiling_->bLen;
        mTile_ = tiling_->mTile;
        batchStride_ = tiling_->batchStride;
        srcStrideBytes_ = static_cast<uint32_t>((N_ - 1UL) * W_bytes_);
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        mTilesPerColumn_ = (M_ + mTile_ - 1UL) / mTile_;
        tilesPerBatch_ = N_ * mTilesPerColumn_;
        totalTiles_ = B_ * tilesPerBatch_;

        if (totalTiles_ == 0) {
            return;
        }

        for (uint64_t tileIdx = 0; tileIdx < totalTiles_; ++tileIdx) {
            uint64_t srcOffset = 0;
            uint64_t dstOffset = 0;
            uint16_t curMTile = 0;
            DecodeTile(tileIdx, srcOffset, dstOffset, curMTile);

            LocalTensor<T> inLocal = inQueue_.AllocTensor<T>();
            DataCopyExtParams copyInParams{curMTile, static_cast<uint32_t>(W_bytes_), srcStrideBytes_, 0, 0};
            DataCopyPad(inLocal, input_[srcOffset], copyInParams, padParams);
            inQueue_.EnQue(inLocal);

            LocalTensor<T> outLocal = inQueue_.DeQue<T>();
            DataCopyExtParams copyOutParams{curMTile, static_cast<uint32_t>(W_bytes_), 0, 0, 0};
            DataCopyPad(output_[dstOffset], outLocal, copyOutParams);
            inQueue_.FreeTensor(outLocal);
        }
    }

private:
    __aicore__ inline void DecodeTile(uint64_t tileIdx, uint64_t& srcOffset, uint64_t& dstOffset,
                                      uint16_t& curMTile) const
    {
        const uint64_t b = tileIdx / tilesPerBatch_;
        const uint64_t rem = tileIdx % tilesPerBatch_;
        const uint64_t n = rem / mTilesPerColumn_;
        const uint64_t mSlot = rem % mTilesPerColumn_;
        const uint64_t mOffset = mSlot * mTile_;
        const uint64_t curMTile64 = (M_ - mOffset) < mTile_ ? (M_ - mOffset) : mTile_;
        const uint64_t batchOffset = b * batchStride_;
        srcOffset = batchOffset + n * W_ + mOffset * N_ * W_;
        dstOffset = batchOffset + n * M_ * W_ + mOffset * W_;
        curMTile = static_cast<uint16_t>(curMTile64);
    }

    const KirinTransposeStridedDataCopyOpTilingData* tiling_ = nullptr;
    GlobalTensor<T> input_;
    GlobalTensor<T> output_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> inQueue_;

    uint64_t W_ = 0;
    uint64_t W_bytes_ = 0;
    uint64_t M_ = 0;
    uint64_t N_ = 0;
    uint64_t B_ = 0;
    uint64_t mTile_ = 0;
    uint64_t batchStride_ = 0;
    uint32_t srcStrideBytes_ = 0;
    uint64_t mTilesPerColumn_ = 0;
    uint64_t tilesPerBatch_ = 0;
    uint64_t totalTiles_ = 0;
};
} // namespace Transpose

#endif // TRANSPOSE_STRIDED_DATACOPY_ARCH31_H
