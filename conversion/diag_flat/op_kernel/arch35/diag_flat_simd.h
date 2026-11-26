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
 * \file diag_flat_simd.h
 * \brief DiagFlat SIMD outer shell + SIMT kernel (arch35, DAV_3510)
 *
 * Execution flow (per tile):
 *   大宽度(outWidth >= DIAG_FLAT_SPARSE_MIN_WIDTH, 对角稀疏):
 *     AllocTensor → Duplicate(0) → 标量 scatter 对角线元素 → EnQue → DeQue → DataCopyPad
 *   其余(小宽度，对角线稠密):
 *     AllocTensor → asc_vf_call (SIMT 全量填充) → EnQue → DeQue → DataCopyPad
 *
 * 稀疏路径把「每元素 UintDiv + 非对角 GM 读」替换为「向量 Duplicate + 每行一次标量 GM 读」，
 * 主体变成纯 N×N 向量写。
 */

#ifndef __DIAG_FLAT_ARCH35_SIMD_H__
#define __DIAG_FLAT_ARCH35_SIMD_H__

#include <type_traits>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "diag_flat_tiling_data.h" // kept for backward compat; fields now in DiagV2Arch35TilingData

using namespace AscendC;

constexpr int32_t DIAG_FLAT_BUFFER_NUM = 2;
constexpr uint32_t DIAG_FLAT_THREAD_NUM = 1024;
// outWidth 小于该值时对角线较稠密，标量 scatter 不划算，保留 SIMT 全量填充
constexpr int64_t DIAG_FLAT_SPARSE_MIN_WIDTH = 64;

// ================================================================
// SIMT kernel: fill UB buffer (diagonal ← xGm, non-diagonal ← 0)
// Every thread always writes — no divergent UB store.
// ================================================================

template <typename T>
__simt_vf__ __aicore__
LAUNCH_BOUND(DIAG_FLAT_THREAD_NUM) void SimtDiagFlat(__gm__ T* xGm,      // input (1D contiguous)
                                                     int64_t tileOffset, // global output element offset of this tile
                                                     int64_t tileElems,  // number of output elements in this tile
                                                     int64_t outWidth,   // W = N + |k|
                                                     uint64_t outWidthMagic, // fast div magic  for outWidth
                                                     uint64_t outWidthShift, // fast div shift  for outWidth
                                                     int64_t diagonal,       // k (signed)
                                                     int64_t numInput,       // N = total input elements
                                                     __ubuf__ T* yUb)        // UB output buffer (packed) — LAST
{
    for (int64_t pos = threadIdx.x; pos < tileElems; pos += blockDim.x) {
        int64_t globalPos = tileOffset + pos;

        uint64_t row = Simt::UintDiv(static_cast<uint64_t>(globalPos), outWidthMagic, outWidthShift);
        uint64_t col = static_cast<uint64_t>(globalPos) - row * static_cast<uint64_t>(outWidth);

        int64_t inputIdx = (diagonal >= 0) ? static_cast<int64_t>(row) : static_cast<int64_t>(col);
        bool onDiag = (static_cast<int64_t>(col) - static_cast<int64_t>(row) == diagonal);
        bool inRange = (inputIdx >= 0) && (inputIdx < numInput);
        bool doRead = onDiag && inRange;

        T gmVal = static_cast<T>(0);
        if (numInput > 0) {
            gmVal = xGm[doRead ? inputIdx : 0];
        }
        yUb[pos] = doRead ? gmVal : static_cast<T>(0);
    }
}

// ================================================================
// SIMD outer shell: pipe/buffer management + fill + DMA output
// ================================================================

template <typename T>
class DiagFlatSimd {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const DiagFlatArch35TilingData* tilingData)
    {
        td_ = tilingData;
        xGm_.SetGlobalBuffer((__gm__ T*)x);
        yGm_.SetGlobalBuffer((__gm__ T*)y);

        uint64_t blockIdx = GetBlockIdx();
        tileOffset_ = blockIdx * td_->outPerCore;
        remaining_ = min(td_->outPerCore, td_->outTotal - tileOffset_);

        uint32_t bufBytes = static_cast<uint32_t>(
            ((td_->tileLength + DIAG_FLAT_THREAD_NUM - 1) / DIAG_FLAT_THREAD_NUM) * DIAG_FLAT_THREAD_NUM * sizeof(T));
        pipe_.InitBuffer(outQueue_, DIAG_FLAT_BUFFER_NUM, bufBytes);

        GetUintDivMagicAndShift(outWidthMagic_, outWidthShift_, static_cast<uint64_t>(td_->outWidth));
    }

    __aicore__ inline void Process()
    {
        if (remaining_ <= 0)
            return;

        int64_t end = tileOffset_ + remaining_;
        int64_t curTileLen;
        for (int64_t curStart = tileOffset_; curStart < end; curStart += td_->tileLength) {
            curTileLen = min(td_->tileLength, end - curStart);
            auto outBuf = outQueue_.AllocTensor<T>();

            bool sparse = (td_->outWidth >= DIAG_FLAT_SPARSE_MIN_WIDTH);
            if (sparse) {
                // 向量整块写 0，再只把对角线元素 scatter 进 UB。
                // 8 字节 dtype(fp64/int64/complex64) 按 32 bit 位宽写 0，避免 64 bit Duplicate 被模拟拖慢
                using ZeroT = typename std::conditional<sizeof(T) <= 4, T, uint32_t>::type;
                constexpr int32_t ZERO_FACTOR = (sizeof(T) <= 4) ? 1 :
                                                                   static_cast<int32_t>(sizeof(T) / sizeof(uint32_t));
                LocalTensor<ZeroT> zeroBuf = outBuf.template ReinterpretCast<ZeroT>();
                Duplicate<ZeroT>(zeroBuf, static_cast<ZeroT>(0), static_cast<int32_t>(curTileLen) * ZERO_FACTOR);
                PipeBarrier<PIPE_ALL>();
                FillDiagSparse(outBuf, curStart, curTileLen);
                PipeBarrier<PIPE_ALL>();
            } else {
                auto yUb = reinterpret_cast<__ubuf__ T*>(outBuf.GetPhyAddr());
                asc_vf_call<SimtDiagFlat<T>>(dim3(DIAG_FLAT_THREAD_NUM), (__gm__ T*)xGm_.GetPhyAddr(), curStart,
                                             curTileLen, td_->outWidth, outWidthMagic_, outWidthShift_, td_->diagonal,
                                             td_->numInput, yUb);
            }

            outQueue_.EnQue(outBuf);
            LocalTensor<T> readyBuf = outQueue_.DeQue<T>();
            DataCopyPad(yGm_[curStart], readyBuf, {1, static_cast<uint32_t>(curTileLen * sizeof(T)), 0, 0, 0});
            outQueue_.FreeTensor(outBuf);
        }
    }

private:
    // 仅处理落在 [curStart, curStart+curTileLen) 内的对角线元素；每行最多一个。
    __aicore__ inline void FillDiagSparse(const LocalTensor<T>& yUb, int64_t curStart, int64_t curTileLen)
    {
        const int64_t width = td_->outWidth;
        const int64_t diagonal = td_->diagonal;
        const int64_t numInput = td_->numInput;
        if (width <= 0 || curTileLen <= 0) {
            return;
        }
        const int64_t tileEnd = curStart + curTileLen;
        const int64_t rowStart = curStart / width;
        const int64_t rowEnd = (tileEnd - 1) / width;
        for (int64_t row = rowStart; row <= rowEnd; ++row) {
            const int64_t col = row + diagonal;
            if (col < 0 || col >= width) {
                continue;
            }
            const int64_t globalPos = row * width + col;
            if (globalPos < curStart || globalPos >= tileEnd) {
                continue;
            }
            const int64_t inputIdx = (diagonal >= 0) ? row : col;
            if (inputIdx < 0 || inputIdx >= numInput) {
                continue;
            }
            yUb.SetValue(static_cast<uint32_t>(globalPos - curStart), xGm_.GetValue(static_cast<uint64_t>(inputIdx)));
        }
    }

    const DiagFlatArch35TilingData* td_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    TPipe pipe_;
    TQue<QuePosition::VECOUT, DIAG_FLAT_BUFFER_NUM> outQueue_;
    int64_t tileOffset_;
    int64_t remaining_;
    uint64_t outWidthMagic_;
    uint64_t outWidthShift_;
};

#endif // __DIAG_FLAT_ARCH35_SIMD_H__
