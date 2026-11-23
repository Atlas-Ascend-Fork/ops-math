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
 * \file transpose_small_shape_2d.h
 * \brief TilingKey=10012 SMALL_SHAPE_2D 策略实现（2D perm=[1,0] 的 UB 中转 4x4 SIMT）
 *
 *   - x0/x1/y0/y1 双缓冲 + Mutex；
 *   - MTE2 二维搬入（每行后 32B pad），MTE3 二维搬出；
 *   - asc_vf_call 做 4x4 直接 gather（1/2/4B 用 char4/short4/int4 向量写，8B 用 4 次标量写）；
 *   - zDim 按实际行块数动态收缩，减少空闲线程。
 *
 * 4x4 块内部映射固定为“竖向读 4、横向写 4”；块间先横向铺满当前 4 行，再进入下一组 4 行：
 *   x=g、y=bc、z=br。
 *
 * 适用：reduced 2D [M, N]、perm=[1,0]、dsize ∈ {1,2,4,8}，且 64KB <= 总字节数 < 小 shape 阈值。
 */
#ifndef TRANSPOSE_SMALL_SHAPE_2D_H
#define TRANSPOSE_SMALL_SHAPE_2D_H

#include "transpose_base.h"
#include "simt_api/asc_simt.h"

namespace Transpose {
using namespace AscendC;

namespace SmallShape2D {
constexpr uint32_t MAX_THREAD = 2048; ///< UB 模式 VF 的 launch bound
constexpr uint32_t BUFFER_NUM = 2;    ///< 双缓冲
} // namespace SmallShape2D

/**
 * @brief 一个 chunk 的 4x4 直接 gather 转置（4 线程/子块，无 shfl）
 *
 * @tparam T 元素类型
 *
 * dim3: x=g(0..3)，y=bc(0..SUB_AXIS-1)，z=br(0..zDim-1)。
 */
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SmallShape2D::MAX_THREAD) inline void TransposeSmallShape2DVF(
    __ubuf__ T* xBase, __ubuf__ T* yBase, uint32_t tileCnt, uint32_t inPitch, uint32_t outPitch, uint32_t tileBase,
    uint32_t cols, uint32_t actRows)
{
    constexpr uint32_t TILE = 128 / sizeof(T);
    constexpr uint32_t SUB_AXIS = TILE / 4;

    const uint32_t g = static_cast<uint32_t>(threadIdx.x);
    const uint32_t bc = static_cast<uint32_t>(threadIdx.y);
    const uint32_t br0 = static_cast<uint32_t>(threadIdx.z);
    const uint32_t brDim = static_cast<uint32_t>(blockDim.z);
    if (brDim == 0u) {
        return;
    }

    const uint32_t brLimit = (actRows + 3u) / 4u; // 只处理有效输入行对应的 4x4 子块
    if (br0 >= brLimit) {
        return;
    }
    const uint32_t colBase = 4u * bc + g;
    const uint32_t colLimit = cols - tileBase * TILE; // = 本 chunk 的列数
    if (colBase >= colLimit) {
        return; // 水平尾块：该线程整列无效
    }

    for (uint32_t br = br0; br < brLimit; br += brDim) {
        __ubuf__ T* src = xBase + 4u * br * inPitch + colBase;
        __ubuf__ T* dst = yBase + colBase * outPitch + 4u * br;
        for (uint32_t tile = 0; tile < tileCnt && (colBase + tile * TILE) < colLimit; ++tile) {
            T v0 = src[0];
            T v1 = src[inPitch];
            T v2 = src[2u * inPitch];
            T v3 = src[3u * inPitch];

            if constexpr (sizeof(T) == 1) {
                *(__ubuf__ char4*)dst = make_char4(v0, v1, v2, v3);
            } else if constexpr (sizeof(T) == 2) {
                *(__ubuf__ short4*)dst = make_short4(v0, v1, v2, v3);
            } else if constexpr (sizeof(T) == 4) {
                *(__ubuf__ int4*)dst = make_int4(v0, v1, v2, v3);
            } else {
                // 8B：bisheng 对 longlong4 的 UB store 存在前端崩溃，改用 4 次标量写
                dst[0] = v0;
                dst[1] = v1;
                dst[2] = v2;
                dst[3] = v3;
            }
            src += TILE;
            dst += TILE * outPitch;
        }
    }
}

/**
 * @brief UB 模式：MTE2 二维搬入 + 3D 4x4 SIMT + MTE3 二维搬出，双缓冲流水
 */
template <typename T>
class TransposeSmallShape2D : public TransposeBase<T> {
public:
    __aicore__ inline TransposeSmallShape2D(){};
    __aicore__ inline ~TransposeSmallShape2D()
    {
        if (mutexReady_) {
            ReleaseMutexID(mutexX0_);
            ReleaseMutexID(mutexX1_);
            ReleaseMutexID(mutexY0_);
            ReleaseMutexID(mutexY1_);
        }
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, __tiling_data_ptr__ TransposeSmallShape2DTilingData* tilingData);
    __aicore__ inline void Process();

private:
    static constexpr uint32_t DSIZE = sizeof(T);
    static constexpr uint32_t TILE = 128 / DSIZE;
    static constexpr uint32_t PAD = 32 / DSIZE;
    static constexpr uint32_t OUT_PITCH = TILE + PAD;
    static constexpr uint32_t SUB_AXIS = TILE / 4;

    struct WorkItem {
        uint32_t stripe;
        uint32_t rowStart;
        uint32_t actRows;
        uint32_t colStart;
        uint32_t chunkCols;
    };

    static __aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b) { return (a < b) ? a : b; }

    __aicore__ inline WorkItem GetWork(uint32_t w) const;
    __aicore__ inline void CopyIn(const WorkItem& wi, uint32_t stage);
    __aicore__ inline void Compute(const WorkItem& wi, uint32_t stage);
    __aicore__ inline void CopyOut(const WorkItem& wi, uint32_t stage);

    uint32_t rows_{0};
    uint32_t cols_{0};
    uint32_t workBegin_{0};
    uint32_t workEnd_{0};
    uint32_t chunksPerStripe_{0};
    uint32_t chunkCols_{0};
    uint32_t totalWork_{0};
    uint32_t inPitch_{0};
    uint32_t bufBytes_{0};
    int32_t blockIdx_{0};
    bool mutexReady_{false};
    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;
    LocalTensor<T> xLocal_[SmallShape2D::BUFFER_NUM];
    LocalTensor<T> yLocal_[SmallShape2D::BUFFER_NUM];
    MutexID mutexX0_{0};
    MutexID mutexX1_{0};
    MutexID mutexY0_{0};
    MutexID mutexY1_{0};
};

template <typename T>
__aicore__ inline void TransposeSmallShape2D<T>::Init(GM_ADDR x, GM_ADDR y,
                                                      __tiling_data_ptr__ TransposeSmallShape2DTilingData* tilingData)
{
    blockIdx_ = static_cast<int32_t>(GetBlockIdx());
    rows_ = tilingData->rows;
    cols_ = tilingData->cols;
    chunkCols_ = tilingData->chunkCols;
    totalWork_ = tilingData->totalWork;
    inPitch_ = tilingData->inPitch;
    bufBytes_ = tilingData->bufBytes;
    if (rows_ == 0u || cols_ == 0u || chunkCols_ == 0u || totalWork_ == 0u || inPitch_ == 0u || bufBytes_ == 0u) {
        return;
    }

    const uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
    if (blockNum == 0u) {
        return;
    }
    // 均衡分配 work item：blockIdx 0..blockNum-1 各拿 0~1 个差异
    workBegin_ = static_cast<uint32_t>(blockIdx_) * totalWork_ / blockNum;
    workEnd_ = (static_cast<uint32_t>(blockIdx_) + 1u) * totalWork_ / blockNum;
    chunksPerStripe_ = (cols_ + chunkCols_ - 1u) / chunkCols_;

    inputGM_.SetGlobalBuffer((__gm__ T*)x);
    outputGM_.SetGlobalBuffer((__gm__ T*)y);

    mutexX0_ = AllocMutexID();
    mutexX1_ = AllocMutexID();
    mutexY0_ = AllocMutexID();
    mutexY1_ = AllocMutexID();
    mutexReady_ = true;

    // 静态 Tensor：x0 | x1 | y0 | y1，每块 bufBytes 字节（Host 已按 UB 预算均分）
    LocalMemAllocator<Hardware::UB> ubAlloc;
    const uint32_t bufElems = bufBytes_ / DSIZE;
    xLocal_[0] = ubAlloc.Alloc<T>(bufElems);
    xLocal_[1] = ubAlloc.Alloc<T>(bufElems);
    yLocal_[0] = ubAlloc.Alloc<T>(bufElems);
    yLocal_[1] = ubAlloc.Alloc<T>(bufElems);
}

template <typename T>
__aicore__ inline void TransposeSmallShape2D<T>::Process()
{
    if (rows_ == 0u || cols_ == 0u || chunkCols_ == 0u || bufBytes_ == 0u || workBegin_ >= workEnd_) {
        return;
    }

    // 单层 work-item 循环 + 双缓冲
    CopyIn(GetWork(workBegin_), workBegin_ & 1u);
    for (uint32_t w = workBegin_; w < workEnd_; ++w) {
        const uint32_t stage = w & 1u;
        if (w + 1u < workEnd_) {
            CopyIn(GetWork(w + 1u), (stage + 1u) & 1u);
        }
        Compute(GetWork(w), stage);
        CopyOut(GetWork(w), stage);
    }
}

template <typename T>
__aicore__ inline typename TransposeSmallShape2D<T>::WorkItem TransposeSmallShape2D<T>::GetWork(uint32_t w) const
{
    WorkItem wi;
    const uint32_t stripeIdx = w / chunksPerStripe_;
    const uint32_t chunkIdx = w - stripeIdx * chunksPerStripe_;
    wi.stripe = stripeIdx;
    wi.rowStart = stripeIdx * TILE;
    wi.actRows = MinU32(TILE, rows_ - wi.rowStart);
    wi.colStart = chunkIdx * chunkCols_;
    wi.chunkCols = MinU32(chunkCols_, cols_ - wi.colStart);
    return wi;
}

template <typename T>
__aicore__ inline void TransposeSmallShape2D<T>::CopyIn(const WorkItem& wi, uint32_t stage)
{
    const uint32_t elemOff = wi.rowStart * cols_ + wi.colStart;
    const uint32_t blockLen = wi.chunkCols * DSIZE;
    const uint32_t alignedLen = (blockLen + 31u) & ~31u;
    // 保持 UB 行间距固定为 inPitch；尾 chunk 用 dstStride 补齐差额
    const uint32_t dstStride = (inPitch_ * DSIZE - alignedLen) / 32u;
    const MutexID mutexId = (stage == 0u) ? mutexX0_ : mutexX1_;

    Mutex::Lock<PIPE_MTE2>(mutexId);
    DataCopyExtParams params;
    params.blockCount = static_cast<uint16_t>(wi.actRows);
    params.blockLen = blockLen;
    params.srcStride = static_cast<int64_t>((cols_ - wi.chunkCols) * DSIZE);
    params.dstStride = static_cast<int64_t>(dstStride);
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad<T>(xLocal_[stage], inputGM_[elemOff], params, padParams);
    Mutex::Unlock<PIPE_MTE2>(mutexId);
}

template <typename T>
__aicore__ inline void TransposeSmallShape2D<T>::Compute(const WorkItem& wi, uint32_t stage)
{
    const uint32_t chunkTiles = (wi.chunkCols + TILE - 1u) / TILE;
    const uint32_t tileBase = wi.colStart / TILE; // colStart 必为 TILE 的整数倍
    uint32_t zDim = SmallShape2D::MAX_THREAD / (4u * SUB_AXIS);
    if (zDim > SUB_AXIS) {
        zDim = SUB_AXIS;
    }
    // 动态线程：行块数不足时收缩 z 维，避免空转
    const uint32_t brLimit = (wi.actRows + 3u) / 4u;
    if (zDim > brLimit) {
        zDim = brLimit;
    }
    if (zDim == 0u) {
        zDim = 1u;
    }

    const MutexID inMutexId = (stage == 0u) ? mutexX0_ : mutexX1_;
    const MutexID outMutexId = (stage == 0u) ? mutexY0_ : mutexY1_;

    Mutex::Lock<PIPE_V>(inMutexId);
    Mutex::Lock<PIPE_V>(outMutexId);
    __ubuf__ T* xBase = (__ubuf__ T*)xLocal_[stage].GetPhyAddr();
    __ubuf__ T* yBase = (__ubuf__ T*)yLocal_[stage].GetPhyAddr();
    // 横向：y 维是列组（SUB_AXIS），z 维是 br 组（zDim）
    asc_vf_call<TransposeSmallShape2DVF<T>>(dim3(4u, SUB_AXIS, zDim), xBase, yBase, chunkTiles, inPitch_, OUT_PITCH,
                                            tileBase, cols_, wi.actRows);
    Mutex::Unlock<PIPE_V>(inMutexId);
    Mutex::Unlock<PIPE_V>(outMutexId);
}

template <typename T>
__aicore__ inline void TransposeSmallShape2D<T>::CopyOut(const WorkItem& wi, uint32_t stage)
{
    const uint32_t blockLen = wi.actRows * DSIZE;
    const uint32_t alignedLen = (blockLen + 31u) & ~31u;
    const uint32_t srcStride = (OUT_PITCH * DSIZE - alignedLen) / 32u;
    const uint32_t elemOff = wi.colStart * rows_ + wi.rowStart;
    const MutexID mutexId = (stage == 0u) ? mutexY0_ : mutexY1_;

    Mutex::Lock<PIPE_MTE3>(mutexId);
    DataCopyExtParams params;
    params.blockCount = static_cast<uint16_t>(wi.chunkCols);
    params.blockLen = blockLen;
    params.srcStride = static_cast<int64_t>(srcStride);
    params.dstStride = static_cast<int64_t>((rows_ - wi.actRows) * DSIZE);
    DataCopyPad<T>(outputGM_[elemOff], yLocal_[stage], params);
    Mutex::Unlock<PIPE_MTE3>(mutexId);
}

} // namespace Transpose

#endif // TRANSPOSE_SMALL_SHAPE_2D_H
