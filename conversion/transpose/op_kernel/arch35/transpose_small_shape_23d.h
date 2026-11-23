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
 * \file transpose_small_shape_23d.h
 * \brief tilingKey 10011 TK_SMALL_SHAPE_23D: 二/三维纯 SIMT, GM→GM 直读直写
 *
 * 合并原 2D/3D 两套 SIMT 小 shape 模板，统一消费 TransposeSmallShape23DTilingData：
 *   - 分核由 Host 按原 SMALL_SHAPE 1024B 对齐公式算好（blkFactor/blkTailFactor），
 *     实际核数不下发，kernel 侧通过 GetBlockNum() 获取（blockDim 精确等于 Host 分核结果）；
 *   - 每核线程数按"本核元素数向上取 2 的幂"动态设置（保留动态线程数），上限 THREAD_DIM；
 *   - 本模板专为小 shape，shape 相关中间量一律用 int32/uint32，节省计算开销；
 *   - TilingData 通过 GET_TILING_DATA_PTR_WITH_STRUCT 以 GM 指针直接访问，避免栈拷贝。
 */
#ifndef TRANSPOSE_SMALL_SHAPE_23D_H
#define TRANSPOSE_SMALL_SHAPE_23D_H

#include "transpose_base.h"
#include "transpose_small_shape.h"

namespace Transpose {
using namespace AscendC;

template <typename T>
class TransposeSmallShape23D : public TransposeBase<T> {
public:
    using TilingData = TransposeSmallShape23DTilingData; ///< 二/三维 Kernel 的 TilingData 具体类型
    __aicore__ inline TransposeSmallShape23D(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, __tiling_data_ptr__ TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessDimTwo();
    __aicore__ inline void ProcessDimThree();
    __aicore__ inline uint32_t CalcThreadDim(uint32_t blkProcessNum) const;

    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;
    int32_t blockIdx_;
    __tiling_data_ptr__ TilingData* tiling_;
};

template <typename T>
__aicore__ inline void TransposeSmallShape23D<T>::Init(GM_ADDR x, GM_ADDR y, __tiling_data_ptr__ TilingData* tilingData)
{
    blockIdx_ = static_cast<int32_t>(GetBlockIdx());
    tiling_ = tilingData;
    inputGM_.SetGlobalBuffer((__gm__ T*)x);
    outputGM_.SetGlobalBuffer((__gm__ T*)y);
}

template <typename T>
__aicore__ inline void TransposeSmallShape23D<T>::Process()
{
    if (tiling_->dim == DIM_TWO) {
        ProcessDimTwo();
    } else {
        ProcessDimThree();
    }
}

/**
 * @brief 动态线程数：本核元素数向上取 2 的幂，下限 AN_EIGHTH，上限 THREAD_DIM。
 */
template <typename T>
__aicore__ inline uint32_t TransposeSmallShape23D<T>::CalcThreadDim(uint32_t blkProcessNum) const
{
    uint32_t threadDim = static_cast<uint32_t>(AN_EIGHTH_THREAD_DIM);
    const uint32_t cap = static_cast<uint32_t>(THREAD_DIM);
    while (threadDim < blkProcessNum && threadDim < cap) {
        threadDim <<= 1;
    }
    return threadDim;
}

template <typename T>
__aicore__ inline void TransposeSmallShape23D<T>::ProcessDimTwo()
{
    // 分核由 Host 算好，实际使用核数由 blockDim 下发，此处用 GetBlockNum() 获取
    const uint32_t coreNum = static_cast<uint32_t>(GetBlockNum());
    if (static_cast<uint32_t>(blockIdx_) >= coreNum) {
        return;
    }
    uint32_t blkProcessNum = tiling_->blkFactor;
    uint32_t blkStartOffset = static_cast<uint32_t>(blockIdx_) * tiling_->blkFactor;
    if (static_cast<uint32_t>(blockIdx_) == coreNum - 1 && tiling_->blkTailFactor != 0) {
        blkProcessNum = tiling_->blkTailFactor;
    }

    uint32_t outputShape0 = static_cast<uint32_t>(tiling_->inputShape[0]);
    uint32_t outputShape1 = static_cast<uint32_t>(tiling_->inputShape[1]);
    // 2 维转置输出 shape = (in[1], in[0])，此处沿用原 SimtComputeDimTwo 语义：
    // 输出线性索引按 (outputShape0, outputShape1) 分解，故取 inputShape 原序即可。
    // 2 维只需一次除法：除以 outputShape0 的商即维度1索引，末维无需再算魔术数。
    uint32_t magic0, shift0;
    GetUintDivMagicAndShift(magic0, shift0, outputShape0);

    const uint32_t threadDim = CalcThreadDim(blkProcessNum);
    asc_vf_call<SimtComputeDimTwo<T>>(dim3(threadDim), (__gm__ T*)(inputGM_.GetPhyAddr()),
                                      (__gm__ volatile T*)(outputGM_.GetPhyAddr()), blkProcessNum, blkStartOffset,
                                      outputShape0, outputShape1, magic0, shift0);
}

template <typename T>
__aicore__ inline void TransposeSmallShape23D<T>::ProcessDimThree()
{
    const uint32_t coreNum = static_cast<uint32_t>(GetBlockNum());
    if (static_cast<uint32_t>(blockIdx_) >= coreNum) {
        return;
    }
    const uint32_t dim = static_cast<uint32_t>(tiling_->dim);
    uint32_t blkProcessNum = tiling_->blkFactor;
    uint32_t blkStartOffset = static_cast<uint32_t>(blockIdx_) * tiling_->blkFactor;
    if (static_cast<uint32_t>(blockIdx_) == coreNum - 1 && tiling_->blkTailFactor != 0) {
        blkProcessNum = tiling_->blkTailFactor;
    }

    // 由 inputShape + perm 推导输出 shape 与源步长（小 shape，统一 int32/uint32）
    uint32_t outputShape[3] = {0, 0, 0};
    uint32_t dstStride[3] = {1, 1, 1};
    uint32_t dstStrideTmp[3] = {1, 1, 1};

    for (uint32_t i = 1; i < dim; i++) {
        dstStrideTmp[i] = static_cast<uint32_t>(tiling_->inputShape[dim - i]) * dstStrideTmp[i - 1];
    }
    for (uint32_t i = 0; i < dim; i++) {
        const uint32_t axis = static_cast<uint32_t>(tiling_->perm[dim - 1 - i]);
        outputShape[i] = static_cast<uint32_t>(tiling_->inputShape[axis]);
        dstStride[i] = dstStrideTmp[dim - 1 - axis];
    }

    // 末维不需要除法（余数即索引），只为主维准备魔术数/位移。
    uint32_t shift[3];
    uint32_t m[3];
    for (uint32_t i = 0; i + 1 < dim; i++) {
        GetUintDivMagicAndShift(m[i], shift[i], outputShape[i]);
    }

    const uint32_t threadDim = CalcThreadDim(blkProcessNum);
    asc_vf_call<SimtComputeDimThree<T>>(dim3(threadDim), (__gm__ T*)(inputGM_.GetPhyAddr()),
                                        (__gm__ volatile T*)(outputGM_.GetPhyAddr()), blkProcessNum, blkStartOffset,
                                        outputShape[0], outputShape[1], outputShape[2], dstStride[0], dstStride[1],
                                        dstStride[2], m[0], m[1], shift[0], shift[1]);
}

} // namespace Transpose
#endif // TRANSPOSE_SMALL_SHAPE_23D_H
