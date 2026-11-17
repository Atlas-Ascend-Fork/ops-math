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
 * \file transpose_2d_transdata.h
 * \brief tilingKey 10011 TRANSPOSE_2D_TRANSDATA: TransDataTo5HD 全 full-tile 转置
 *
 * 适用场景：2D perm=[1,0]，dtype 为 16bit/32bit，且 m/n 均 16 对齐。
 * 核心机制：按 tile 搬入 UB 后使用 TransDataTo5HD 硬件指令完成 16x16 块转置，
 * 再按列搬出到输出 GM，MTE2/V/MTE3 通过 Mutex credit 组成三流水。
 */
#ifndef TRANSPOSE_2D_TRANSDATA_H
#define TRANSPOSE_2D_TRANSDATA_H

#include "transpose_base.h"

namespace Transpose {
using namespace AscendC;

static constexpr uint32_t TD_TRANSELEM = 16;

template <typename T>
class Transpose2DTransData {
public:
    __aicore__ inline Transpose2DTransData(){};
    __aicore__ inline ~Transpose2DTransData()
    {
        ReleaseMutexID(mutexIn0_);
        ReleaseMutexID(mutexOut0_);
        ReleaseMutexID(mutexOut1_);
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, __tiling_data_ptr__ Transpose2DTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessTile(uint32_t inRow, uint32_t inCol, uint32_t rows, uint32_t cols,
                                       LocalTensor<T>& tin, LocalTensor<T>& tout, MutexID mIn, MutexID mOut);
    __aicore__ inline void ComputeTranspose(LocalTensor<T>& src, LocalTensor<T>& dst, uint32_t rows, uint32_t cols);

    // tile 几何由 tiling 下发: tileM 跟随 m (m<=cap 时取 m, 保证整行搬出不被切碎),
    // tileN 由 UB 预算决定。避免 m 很小时 tile 退化成 4KB 小块。
    int32_t tileM_ = 0;
    int32_t tileN_ = 0;
    uint32_t tmcPad_ = 0;
    uint32_t tmrPad_ = 0;

    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;
    int32_t blockIdx_;
    int32_t m_;
    int32_t n_;
    int32_t coreNum_;
    __tiling_data_ptr__ Transpose2DTilingData* tiling_;

    // 静态Tensor: LocalMemAllocator分配, 不通过TPipe (in 单缓冲, out 双缓冲)
    LocalTensor<T> inPing;
    LocalTensor<T> outPing;
    LocalTensor<T> outPong;

    // Mutex 按 buffer 生命周期拆分: mIn 门控 inbuf 复用 (MTE2写/V读交替),
    // mOut[b] 门控 outbuf[b] 复用 (V写/MTE3读交替), 各自独立归还, 消除跨段过度同步
    MutexID mutexIn0_{0};
    MutexID mutexOut0_{0};
    MutexID mutexOut1_{0};

    // UB bank 冲突规避 (DAV_3510: 16bank x 512行, 8group x 2bank, 每组 2 读口+2 写口):
    // 行步长额外 +32B (PAD_EL=32/S 个元素) 拉开 16 个并发 sL/dL 地址的 bank 分布。
    static constexpr uint32_t PAD_EL = BLOCK_SIZE_BYTE / sizeof(T);

    // Scalar hoisting: full-tile 预计算参数 (标量成员, 非数组间接读)
    DataCopyExtParams fullCopyIn_;
    DataCopyExtParams fullCopyOut_;
    uint8_t fullCB_ = 0; // full tile 的 TransDataTo5HD repeatTimes
};

template <typename T>
__aicore__ inline void Transpose2DTransData<T>::Init(GM_ADDR x, GM_ADDR y,
                                                     __tiling_data_ptr__ Transpose2DTilingData* tilingData)
{
    AscendC::InitSocState();
    blockIdx_ = GetBlockIdx();
    tiling_ = tilingData;
    m_ = tiling_->m;
    n_ = tiling_->n;
    coreNum_ = tiling_->coreNum;
    tileM_ = tiling_->tileM;
    tileN_ = tiling_->tileN;
    // 窄列且列宽为奇数个 32B block 时无需源侧 padding 即可避免 bank 冲突,
    // 此时 UB 内输入 tile 连续, 搬入可退化为单次连续 DataCopyPad。
    constexpr uint32_t BLOCK_ELEMS = static_cast<uint32_t>(BLOCK_SIZE_BYTE) / sizeof(T);
    uint32_t srcPad = PAD_EL;
    if (tileN_ == n_ && ((static_cast<uint32_t>(tileN_) / BLOCK_ELEMS) % 2U == 1U)) {
        srcPad = 0;
    }
    tmcPad_ = static_cast<uint32_t>(tileN_) + srcPad;
    tmrPad_ = static_cast<uint32_t>(tileM_) + PAD_EL;
    inputGM_.SetGlobalBuffer((__gm__ T*)x);
    outputGM_.SetGlobalBuffer((__gm__ T*)y);

    mutexIn0_ = AllocMutexID();
    mutexOut0_ = AllocMutexID();
    mutexOut1_ = AllocMutexID();

    // 静态Tensor: 用LocalMemAllocator分配UB, 行步长含 padding (bank 冲突规避)
    // 池深 in 1+out 2。tile 尺寸由 tiling 自适应给出, 预算约 248KB。
    // MTE3(慢侧)保留双缓冲; in 单缓冲下 MTE2(t+1) 仅等 V(t) 读完 (mIn 在 V 读后即归还),
    // 周期 = max(MTE2+V, MTE3)。
    const uint32_t inTileElems = static_cast<uint32_t>(tileM_) * tmcPad_;
    const uint32_t outTileElems = static_cast<uint32_t>(tileN_) * tmrPad_;
    LocalMemAllocator<Hardware::UB> ubAlloc;
    inPing = ubAlloc.Alloc<T>(inTileElems);
    outPing = ubAlloc.Alloc<T>(outTileElems);
    outPong = ubAlloc.Alloc<T>(outTileElems);

    // Scalar hoisting: full-tile 常量参数预计算 (标量, Init 一次)
    constexpr uint32_t S = sizeof(T);
    if (tmcPad_ == static_cast<uint32_t>(tileN_)) {
        fullCopyIn_.blockCount = 1;
        fullCopyIn_.blockLen = static_cast<uint32_t>(tileM_) * static_cast<uint32_t>(tileN_) * S;
        fullCopyIn_.srcStride = 0;
        fullCopyIn_.dstStride = 0;
    } else {
        fullCopyIn_.blockCount = static_cast<uint16_t>(tileM_);
        fullCopyIn_.blockLen = static_cast<uint32_t>(tileN_) * S;
        fullCopyIn_.srcStride = static_cast<int64_t>(n_ - tileN_) * static_cast<int64_t>(S);
        fullCopyIn_.dstStride = static_cast<int64_t>(PAD_EL * S) / static_cast<int64_t>(BLOCK_SIZE_BYTE);
    }

    fullCopyOut_.blockCount = static_cast<uint16_t>(tileN_);
    fullCopyOut_.blockLen = static_cast<uint32_t>(tileM_) * S;
    fullCopyOut_.srcStride = static_cast<int64_t>(PAD_EL * S) / static_cast<int64_t>(BLOCK_SIZE_BYTE);
    fullCopyOut_.dstStride = static_cast<int64_t>(m_ - tileM_) * static_cast<int64_t>(S);

    fullCB_ = static_cast<uint8_t>(tileN_ / TD_TRANSELEM);
}

template <typename T>
__aicore__ inline void Transpose2DTransData<T>::ComputeTranspose(LocalTensor<T>& src, LocalTensor<T>& dst,
                                                                 uint32_t rows, uint32_t cols)
{
    if constexpr (sizeof(T) <= sizeof(int32_t)) {
        constexpr uint32_t S = sizeof(T);
        constexpr uint32_t BS = static_cast<uint32_t>(BLOCK_SIZE_BYTE) / S;
        constexpr uint32_t halfBlocks = TD_TRANSELEM / BS;
        uint32_t rB = rows / TD_TRANSELEM;
        uint32_t cB = (cols == static_cast<uint32_t>(tileN_)) ? fullCB_ : static_cast<uint8_t>(cols / TD_TRANSELEM);

        // Scalar hoisting: 增量地址计算 — 基址一次算好, 内循环只加常量步长
        // (消除 16 次乘法 + 16 次 GetPhyAddr 调用)
        const uint64_t srcPhysBase = reinterpret_cast<uint64_t>(src.GetPhyAddr());
        const uint64_t dstPhysBase = reinterpret_cast<uint64_t>(dst.GetPhyAddr());
        const uint64_t srcStep = static_cast<uint64_t>(tmcPad_) * S;
        const uint64_t dstStep = static_cast<uint64_t>(tmrPad_) * S;

        // 列方向只有一个 16x16 block (窄列 n=16): 把 repeat 用在行块方向,
        // 一次指令覆盖最多 255 个行块, 减少 TransDataTo5HD 的发射与地址构造开销。
        if (cB == 1) {
            const uint16_t dstRepUnit = static_cast<uint16_t>((TD_TRANSELEM * S) / BLOCK_SIZE_BYTE);
            const uint16_t srcRepUnit = static_cast<uint16_t>((TD_TRANSELEM * tmcPad_ * S) / BLOCK_SIZE_BYTE);
            uint32_t a = 0;
            while (a < rB) {
                uint32_t chunk = (rB - a > 255U) ? 255U : (rB - a);
                for (uint32_t p = 0; p < halfBlocks; ++p) {
                    uint64_t sL[16];
                    uint64_t dL[16];
                    uint64_t sa = srcPhysBase + (static_cast<uint64_t>(a) * TD_TRANSELEM * tmcPad_) * S +
                                  static_cast<uint64_t>(p * BS) * S;
                    uint64_t da = dstPhysBase + (static_cast<uint64_t>(a) * TD_TRANSELEM) * S +
                                  static_cast<uint64_t>(p * BS) * tmrPad_ * S;
                    for (int n = 0; n < 16; ++n) {
                        sL[n] = sa;
                        sa += srcStep;
                    }
                    if constexpr (halfBlocks == 1) {
                        for (int n = 0; n < 16; ++n) {
                            dL[n] = da;
                            da += dstStep;
                        }
                    } else {
                        constexpr uint64_t bsStep = static_cast<uint64_t>(BS) * S;
                        for (int n = 0; n < 16; n += 2) {
                            dL[n] = da;
                            dL[n + 1] = da + bsStep;
                            da += dstStep;
                        }
                    }
                    PipeBarrier<PIPE_V>();
                    const uint16_t dstRS = (chunk == 1) ? 0 : dstRepUnit;
                    const uint16_t srcRS = (chunk == 1) ? 0 : srcRepUnit;
                    TransDataTo5HDParams pr(false, false, static_cast<uint8_t>(chunk), dstRS, srcRS);
                    TransDataTo5HD<T>(dL, sL, pr);
                }
                a += chunk;
            }
            return;
        }

        for (uint32_t a = 0; a < rB; ++a) {
            uint64_t srcAddr = srcPhysBase + (static_cast<uint64_t>(a * TD_TRANSELEM) * tmcPad_) * S;
            uint64_t dstColBase = dstPhysBase + (static_cast<uint64_t>(a * TD_TRANSELEM)) * S;
            for (uint32_t p = 0; p < halfBlocks; ++p) {
                uint64_t sL[16];
                uint64_t dL[16];

                // src: 基址 + p*BS*S, 16 个地址按 srcStep 递增
                uint64_t sa = srcAddr + p * BS * S;
                for (int n = 0; n < 16; ++n) {
                    sL[n] = sa;
                    sa += srcStep;
                }

                // dst: 基址 + p*BS*tmrPad_*S, 按模式递增
                if constexpr (halfBlocks == 1) {
                    uint64_t da = dstColBase + static_cast<uint64_t>(p * BS) * tmrPad_ * S;
                    for (int n = 0; n < 16; ++n) {
                        dL[n] = da;
                        da += dstStep;
                    }
                } else {
                    uint64_t da = dstColBase + static_cast<uint64_t>(p * BS) * tmrPad_ * S;
                    constexpr uint64_t bsStep = static_cast<uint64_t>(BS) * S;
                    for (int n = 0; n < 16; n += 2) {
                        dL[n] = da;
                        dL[n + 1] = da + bsStep;
                        da += dstStep;
                    }
                }

                PipeBarrier<PIPE_V>();
                // repeatTimes==1 时没有下一个 repeat，repStride 必须为 0；
                // 否则指令不写出（tail tile 恰好 16 列时输出全零）。
                const uint16_t dstRepStride = (cB == 1) ? 0 : static_cast<uint16_t>(tmrPad_ * TD_TRANSELEM / BS);
                const uint16_t srcRepStride = (cB == 1) ? 0 : static_cast<uint16_t>(TD_TRANSELEM / BS);
                TransDataTo5HDParams pr(false, false, static_cast<uint8_t>(cB), dstRepStride, srcRepStride);
                TransDataTo5HD<T>(dL, sL, pr);
            }
        }
    }
}

template <typename T>
__aicore__ inline void Transpose2DTransData<T>::ProcessTile(uint32_t inRow, uint32_t inCol, uint32_t rows,
                                                            uint32_t cols, LocalTensor<T>& tin, LocalTensor<T>& tout,
                                                            MutexID mIn, MutexID mOut)
{
    constexpr uint32_t S = sizeof(T);
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    // Scalar hoisting: full tile 用预计算参数, 尾 tile 现算
    DataCopyExtParams copyInParams;
    DataCopyExtParams copyOutParams;
    const bool isFull = (rows == static_cast<uint32_t>(tileM_)) && (cols == static_cast<uint32_t>(tileN_));
    if (isFull) {
        copyInParams = fullCopyIn_;
        copyOutParams = fullCopyOut_;
    } else {
        if (tmcPad_ == static_cast<uint32_t>(tileN_) && cols == static_cast<uint32_t>(tileN_)) {
            copyInParams.blockCount = 1;
            copyInParams.blockLen = rows * cols * S;
            copyInParams.srcStride = 0;
            copyInParams.dstStride = 0;
        } else {
            copyInParams.blockCount = static_cast<uint16_t>(rows);
            copyInParams.blockLen = cols * S;
            copyInParams.srcStride = static_cast<int64_t>((n_ - cols) * S);
            copyInParams.dstStride = static_cast<int64_t>((tmcPad_ - cols) * S) / static_cast<int64_t>(BLOCK_SIZE_BYTE);
        }
        copyOutParams.blockCount = static_cast<uint16_t>(cols);
        copyOutParams.blockLen = rows * S;
        copyOutParams.srcStride = static_cast<int64_t>((tmrPad_ - rows) * S) / static_cast<int64_t>(BLOCK_SIZE_BYTE);
        copyOutParams.dstStride = static_cast<int64_t>((m_ - rows) * S);
    }

    // MTE2 搬入: mIn 门控 inbuf 复用 — 上一轮读者是 V(i-2), 其计算完成即归还
    Mutex::Lock<PIPE_MTE2>(mIn);
    DataCopyPad(tin, inputGM_[(uint64_t)inRow * n_ + inCol], copyInParams, padParams);
    Mutex::Unlock<PIPE_MTE2>(mIn);

    // V 转置: 等 mIn (本轮搬入完成) + mOut (outbuf 空闲, 上一轮 MTE3(i-2) 搬出完已归还)
    Mutex::Lock<PIPE_V>(mIn);
    Mutex::Lock<PIPE_V>(mOut);
    ComputeTranspose(tin, tout, rows, cols);
    Mutex::Unlock<PIPE_V>(mIn);  // inbuf 读完即归还 → MTE2(i+2) 可写, 不再等 MTE3
    Mutex::Unlock<PIPE_V>(mOut); // outbuf 写完 → MTE3(i) 可搬出

    // V 转置 → MTE3 搬出, mOut 门控数据就绪
    Mutex::Lock<PIPE_MTE3>(mOut);
    DataCopyPad(outputGM_[(uint64_t)inCol * m_ + inRow], tout, copyOutParams);
    Mutex::Unlock<PIPE_MTE3>(mOut); // 搬出完 → outbuf 归还, V(i+2) 可写
}

template <typename T>
__aicore__ inline void Transpose2DTransData<T>::Process()
{
    uint32_t bid = static_cast<uint32_t>(blockIdx_);
    uint32_t bnum = static_cast<uint32_t>(coreNum_);

    uint32_t tileM = static_cast<uint32_t>(tileM_);
    uint32_t tileN = static_cast<uint32_t>(tileN_);
    uint32_t rowTiles = (static_cast<uint32_t>(m_) + tileM - 1) / tileM;
    uint32_t colTiles = (static_cast<uint32_t>(n_) + tileN - 1) / tileN;
    uint32_t total = rowTiles * colTiles;

    // Scalar hoisting: 增量推进 (rowIdx, colIdx) 替代每 tile 的 t/colTiles 整数除法
    // t 步进 bnum: 分解 bnum = stepQ*colTiles + stepR, 每 tile 仅一次条件减法
    uint32_t stepQ = bnum / colTiles;
    uint32_t stepR = bnum % colTiles;
    uint32_t rowIdx = bid / colTiles;
    uint32_t colIdx = bid % colTiles;

    uint32_t i = 0;
    for (uint32_t t = bid; t < total; t += bnum, ++i) {
        uint32_t inRow = rowIdx * tileM;
        uint32_t inCol = colIdx * tileN;
        uint32_t rows = (inRow + tileM <= static_cast<uint32_t>(m_)) ? tileM : (m_ - inRow);
        uint32_t cols = (inCol + tileN <= static_cast<uint32_t>(n_)) ? tileN : (n_ - inCol);
        // Ping/Pong 交替 (out 池): in 单缓冲下 MTE2(t+1) 与 MTE3(t) 并行,
        // 仅被 V(t) 读完门控
        uint32_t b = i & 1;
        MutexID curIn = mutexIn0_;
        MutexID curOut = b ? mutexOut1_ : mutexOut0_;
        LocalTensor<T>& tin = inPing;
        LocalTensor<T>& tout = b ? outPong : outPing;
        ProcessTile(inRow, inCol, rows, cols, tin, tout, curIn, curOut);
        colIdx += stepR;
        rowIdx += stepQ;
        if (colIdx >= colTiles) {
            colIdx -= colTiles;
            ++rowIdx;
        }
    }
}

} // namespace Transpose

#endif // TRANSPOSE_2D_TRANSDATA_H
