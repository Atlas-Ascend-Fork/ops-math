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
 * \file transpose_n_last_small.h
 * \brief TK_N_LAST_SMALL(10014): 三/四维尾轴不变统一搬运（专为小 shape）
 *
 * 静态 Tensor + Mutex 双缓冲。本模板专为小 shape，shape 相关中间量一律用 int32/uint32，
 * 节省计算开销。保持"类定义 + 类外函数体定义"结构，便于对比差异。
 */
#ifndef TRANSPOSE_N_LAST_SMALL_H
#define TRANSPOSE_N_LAST_SMALL_H

#include "transpose_base.h"

namespace Transpose {
class TransposeNLastSmall {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TransposeNLastSmallTilingData* data);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopySingleBlock(GM_ADDR x, GM_ADDR y, const TransposeNLastSmallTilingData* data,
                                           uint32_t core, uint32_t srcLoopStride);
    __aicore__ inline void InitTiles(uint32_t core);
    __aicore__ inline void ProcessTiles(uint32_t bufferOffset);
    template <AscendC::MutexID mutexId>
    __aicore__ inline void CopyTile(uint32_t aTile, uint32_t bTile, uint32_t cTile, uint32_t srcBase, uint32_t dstBase,
                                    AscendC::LocalTensor<uint8_t>& local, AscendC::GlobalTensor<uint8_t>& input,
                                    AscendC::GlobalTensor<uint8_t>& output);

    GM_ADDR x_;
    GM_ADDR y_;
    const TransposeNLastSmallTilingData* data_;
    uint32_t rowBytes_;
    uint32_t srcTileStride_;
    uint32_t dstTileStride_;
    uint32_t aTiles_;
    uint32_t srcOuterStride_;
    uint32_t dstOuterStride_;
    uint32_t outAStride_;
    uint32_t outBStride_;
    uint32_t tileA_;
    uint32_t tailA_;
    uint32_t tileB_;
    uint32_t tailB_;
    uint32_t bTiles_;
    uint32_t cTiles_;
    uint32_t tileBytes_;
    uint32_t start_;
    uint32_t count_;
};

__aicore__ inline void TransposeNLastSmall::Init(GM_ADDR x, GM_ADDR y, const TransposeNLastSmallTilingData* data)
{
    x_ = x;
    y_ = y;
    data_ = data;
}

__aicore__ inline void TransposeNLastSmall::Process()
{
    const uint32_t core = AscendC::GetBlockIdx();
    if (core >= static_cast<uint32_t>(GetBlockNum())) {
        return;
    }
    const uint32_t srcLoopStride = data_->srcLoopStrideBytes;
    if (srcLoopStride != 0) {
        CopySingleBlock(x_, y_, data_, core, srcLoopStride);
    } else {
        InitTiles(core);
        ProcessTiles(data_->bufferOffset);
    }
}

__aicore__ inline void TransposeNLastSmall::CopySingleBlock(GM_ADDR x, GM_ADDR y,
                                                            const TransposeNLastSmallTilingData* data, uint32_t core,
                                                            uint32_t srcLoopStride)
{
    using namespace AscendC;
    const bool tail = core + 1 == static_cast<uint32_t>(GetBlockNum());
    const uint32_t a = tail ? data->tailA : data->tileA;
    const uint32_t bytes = tail ? data->tailSrcBytes : data->srcTileStride;
    GlobalTensor<uint8_t> input;
    GlobalTensor<uint8_t> output;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x) + core * data->srcTileStride);
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y) + core * data->dstTileStride);
    LocalTensor<uint8_t> local(TPosition::VECIN, 0, data->srcTileStride);
    DataCopyExtParams copyIn{1, bytes, 0, 0, 0};
    DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
    Mutex::Lock<PIPE_MTE2>(0);
#if defined(__DAV_920R1__) || defined(__DAV_920R2__)
    DataCopyPad(local, input, copyIn, pad, true);
#else
    DataCopyPad(local, input, copyIn, pad);
#endif
    Mutex::Unlock<PIPE_MTE2>(0);
    Mutex::Lock<PIPE_MTE3>(0);
    LoopModeParams loop{a, 1, srcLoopStride, data->rowBytes, 0, 0};
    SetLoopModePara(loop, DataCopyMVType::UB_TO_OUT);
    DataCopyExtParams copyOut{data->tileB, data->rowBytes, 0, data->outDstStrideBytes, 0};
    DataCopyPad(output, local, copyOut);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    Mutex::Unlock<PIPE_MTE3>(0);
}

__aicore__ inline void TransposeNLastSmall::InitTiles(uint32_t core)
{
    rowBytes_ = data_->rowBytes;
    srcTileStride_ = data_->srcTileStride;
    dstTileStride_ = data_->dstTileStride;
    outBStride_ = data_->outDstStrideBytes + rowBytes_;
    aTiles_ = data_->aTiles;
    srcOuterStride_ = data_->srcOuterStride;
    dstOuterStride_ = srcOuterStride_ != 0 ? data_->dstOuterStride : 0;
    outAStride_ = srcOuterStride_ != 0 ? data_->dstAStride : rowBytes_;
    tileA_ = data_->tileA;
    tailA_ = data_->tailA;
    tileB_ = data_->tileB;
    tailB_ = data_->tailB;
    bTiles_ = data_->bTiles;
    cTiles_ = data_->cTiles;
    tileBytes_ = data_->tileBytes;
    const uint32_t blocks = data_->blocksPerCore;
    const uint32_t extra = data_->extraBlocks;
    start_ = core * blocks + (core < extra ? core : extra);
    count_ = blocks + (core < extra ? 1 : 0);
}

__aicore__ inline void TransposeNLastSmall::ProcessTiles(uint32_t bufferOffset)
{
    AscendC::GlobalTensor<uint8_t> input;
    AscendC::GlobalTensor<uint8_t> output;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x_));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y_));
    AscendC::LocalTensor<uint8_t> local[2]{
        AscendC::LocalTensor<uint8_t>(AscendC::TPosition::VECIN, 0, bufferOffset),
        AscendC::LocalTensor<uint8_t>(AscendC::TPosition::VECIN, bufferOffset, bufferOffset)};
    const uint32_t rowTile = start_ / cTiles_;
    uint32_t c = start_ % cTiles_;
    uint32_t a = rowTile / bTiles_;
    uint32_t b = rowTile % bTiles_;
    const uint32_t outer = srcOuterStride_ != 0 ? a / aTiles_ : 0;
    a -= outer * aTiles_;
    uint32_t srcBase = outer * srcOuterStride_;
    uint32_t dstBase = outer * dstOuterStride_;
    for (uint32_t index = 0; index < count_; ++index) {
        if ((index & 1) == 0) {
            CopyTile<0>(a, b, c, srcBase, dstBase, local[0], input, output);
        } else {
            CopyTile<1>(a, b, c, srcBase, dstBase, local[1], input, output);
        }
        if (++c == cTiles_) {
            c = 0;
            if (++b == bTiles_) {
                b = 0;
                if (++a == aTiles_) {
                    a = 0;
                    srcBase += srcOuterStride_;
                    dstBase += dstOuterStride_;
                }
            }
        }
    }
}

template <AscendC::MutexID mutexId>
__aicore__ inline void TransposeNLastSmall::CopyTile(uint32_t aTile, uint32_t bTile, uint32_t cTile, uint32_t srcBase,
                                                     uint32_t dstBase, AscendC::LocalTensor<uint8_t>& local,
                                                     AscendC::GlobalTensor<uint8_t>& input,
                                                     AscendC::GlobalTensor<uint8_t>& output)
{
    using namespace AscendC;
    const uint32_t dstA = aTile * dstTileStride_;
    const uint32_t bStart = bTile * tileB_;
    const uint32_t column = cTile * tileBytes_;
    const uint32_t aCount = aTile + 1 == aTiles_ ? tailA_ : tileA_;
    const uint32_t bCount = bTile + 1 == bTiles_ ? tailB_ : tileB_;
    const uint32_t bytes = tileBytes_ < rowBytes_ - column ? tileBytes_ : rowBytes_ - column;
    const uint32_t alignedBytes = (bytes + 31) / 32 * 32;
    const uint32_t rows = aCount * bCount;
    const uint32_t src = srcBase + aTile * srcTileStride_ + bStart * rowBytes_ + column;
    const uint32_t dst = dstBase + bStart * outBStride_ + dstA + column;
    DataCopyExtParams copyIn{static_cast<uint16_t>(bytes == alignedBytes ? 1 : rows),
                             bytes == alignedBytes ? rows * bytes : bytes, 0, 0, 0};
    DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
    Mutex::Lock<PIPE_MTE2>(mutexId);
#if defined(__DAV_920R1__) || defined(__DAV_920R2__)
    DataCopyPad(local, input[src], copyIn, pad, true);
#else
    DataCopyPad(local, input[src], copyIn, pad);
#endif
    Mutex::Unlock<PIPE_MTE2>(mutexId);
    Mutex::Lock<PIPE_MTE3>(mutexId);
    LoopModeParams loop{bCount, aCount, alignedBytes, outBStride_, bCount * alignedBytes, outAStride_};
    SetLoopModePara(loop, DataCopyMVType::UB_TO_OUT);
    DataCopyExtParams copyOut{1, bytes, 0, 0, 0};
    DataCopyPad(output[dst], local, copyOut);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    Mutex::Unlock<PIPE_MTE3>(mutexId);
}

} // namespace Transpose

#endif // TRANSPOSE_N_LAST_SMALL_H
