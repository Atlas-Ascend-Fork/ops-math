/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file transpose_small_shape.h
 * \brief TilingKey=10001 SMALL_SHAPE 策略实现（四~八维精简 TilingData）
 *
 * 适用场景：总数据量（字节）< SMALL_SHAPE_BYTES_THRESHOLD，简化后维数 4-8D。
 * 二/三维 SIMT 小 shape 由 transpose_small_shape_23d.h（10011）处理。
 *
 * 分核沿用 refactor-tp 原有逻辑，由 Host 一次算好（blkFactor/blkTailFactor），
 * kernel 侧不再自行分核；实际核数由 blockDim 下发，kernel 用 GetBlockNum() 获取。
 *
 * 核心机制：SIMT（Single Instruction Multiple Thread）模式
 *   - 利用 DAV_3510 的 SIMT 硬件能力，每个线程独立计算输出位置
 *   - 直接 GM→GM 读写，无需 UB 中转，减少一次数据搬运
 *   - 使用 asc_vf_call 启动 SIMT 函数，2048线程并行
 *
 * 地址计算原理（混合基分解）：
 *   - 线程按 threadIdx.x 遍历输出元素
 *   - 使用 Simt::UintDiv 快速除法进行混合基分解：将线性输出索引 yIdx 分解到各维度
 *   - inputIndex[d] = yIdx 在维度 d 上的余数（即 yIdx % outputShape[d]）
 *   - xIdx = Σ(inputIndex[d] * dstStride[d])，其中 dstStride[d] 为输入张量在维度 d 上的步长
 *   - 最终 outputGM[coreOffset + idx] = inputGM[xIdx]
 *
 * 快速除法优化：
 *   - GetUintDivMagicAndShift 预计算魔术数 m[i] 和位移 shift[i]
 *   - Simt::UintDiv(yIdx, m, shift) 替代昂贵的硬件除法指令
 */
#ifndef TRANSPOSE_SMALL_SHAPE_H
#define TRANSPOSE_SMALL_SHAPE_H

#include "transpose_base.h"
#include "simt_api/asc_simt.h"

/* SIMT 线程数配置：FPGA 环境使用较小线程数，实际 NPU 使用 2048 线程 */
#ifdef __DAV_FPGA__
constexpr int32_t THREAD_DIM = 512; // FPGA 环境：512 线程
constexpr int32_t HALF_THREAD_DIM = 512;
constexpr int32_t QUARTER_THREAD_DIM = 512;
constexpr int32_t AN_EIGHTH_THREAD_DIM = 256;
#else
constexpr int32_t THREAD_DIM = 2048;          // NPU 环境：2048 线程（DAV_3510 SIMT 最大线程数）
constexpr int32_t HALF_THREAD_DIM = 1024;     // 1024 线程
constexpr int32_t QUARTER_THREAD_DIM = 512;   // 512 线程
constexpr int32_t AN_EIGHTH_THREAD_DIM = 256; // 256 线程
#endif

namespace Transpose {
using namespace AscendC;

/**
 * @brief SMALL_SHAPE 策略类（精简 tilingdata 版），使用 SIMT 模式实现小数据量转置
 *
 * 直接 GM→GM 读写，无需 UB 中转。每个线程独立计算输出位置。
 * 注意：此类不需要 TQueBind 缓冲队列，也不需要 TPipe。
 *
 * @tparam T        数据元素类型
 */
template <typename T>
class TransposeSmallShape : public TransposeBase<T> {
public:
    using TilingData = TransposeSmallShape48DTilingData; ///< 四~八维 Kernel 的 TilingData 具体类型
    __aicore__ inline TransposeSmallShape(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, __tiling_data_ptr__ TilingData* tilingData);
    __aicore__ inline void Process();

private:
    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;
    int32_t blockIdx_;
    __tiling_data_ptr__ TilingData* tilingData_;
};

template <typename T>
__aicore__ inline void TransposeSmallShape<T>::Init(GM_ADDR x, GM_ADDR y, __tiling_data_ptr__ TilingData* tilingData)
{
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;
    inputGM_.SetGlobalBuffer((__gm__ T*)x);
    outputGM_.SetGlobalBuffer((__gm__ T*)y);
}

/**
 * @brief 2维 SIMT 计算函数
 *
 * 对2维转置进行逐元素地址计算。每个线程按 threadIdx.x 遍历输出元素，
 * 使用 UintDiv 快速除法将线性输出索引分解到2个维度，计算对应的输入地址。
 *
 * 混合基分解算法（以2维为例）：
 *   yIdx = coreOffset + idx  （线性输出索引）
 *   q = yIdx / outputShape0                   （一次除法，商复用于下一位）
 *   inputIndex0 = yIdx - q * outputShape0     （第0维的索引，即余数）
 *   inputIndex1 = q                           （第1维的索引，末维天然小于 outputShape1）
 *   xIdx = inputIndex0 * outputShape1 + inputIndex1  （线性输入索引，2维特例）
 *
 * @param inputGM       输入 GM 地址
 * @param outputGM      输出 GM 地址（volatile 修饰避免编译器优化写顺序）
 * @param coreFactor    当前核需处理的元素数
 * @param coreOffset    当前核在全局输出中的起始偏移
 * @param outputShape0  输出 shape[0]（维度0大小）
 * @param outputShape1  输出 shape[1]（维度1大小）
 * @param m0            UintDiv 魔术数（用于快速除法）
 * @param s0            UintDiv 位移数（用于快速除法）
 *
 * 除法只需要一次：yIdx / outputShape0 的商即为维度1的索引（末维索引天然小于
 * outputShape1），余数为维度0的索引；末维无需再做一次 UintDiv。
 */
template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void SimtComputeDimTwo(__gm__ T* inputGM, __gm__ volatile T* outputGM, uint32_t coreFactor, uint32_t coreOffset,
                           uint32_t outputShape0, uint32_t outputShape1, uint32_t m0, uint32_t s0)
{
    for (uint32_t idx = static_cast<uint32_t>(threadIdx.x); idx < coreFactor;
         idx += static_cast<uint32_t>(blockDim.x)) {
        uint32_t yIdx = coreOffset + idx;
        const uint32_t quotient = Simt::UintDiv(yIdx, m0, s0);
        uint32_t inputIndex0 = yIdx - quotient * outputShape0;
        uint32_t inputIndex1 = quotient;
        uint32_t xIdx = inputIndex0 * outputShape1 + inputIndex1;
        outputGM[coreOffset + idx] = inputGM[xIdx];
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void SimtComputeDimThree(__gm__ T* inputGM, __gm__ volatile T* outputGM, uint32_t coreFactor, uint32_t coreOffset,
                             uint32_t outputShape0, uint32_t outputShape1, uint32_t outputShape2, uint32_t dstStride0,
                             uint32_t dstStride1, uint32_t dstStride2, uint32_t m0, uint32_t m1, uint32_t s0,
                             uint32_t s1)
{
    for (uint32_t idx = static_cast<uint32_t>(threadIdx.x); idx < coreFactor;
         idx += static_cast<uint32_t>(blockDim.x)) {
        uint32_t yIdx = coreOffset + idx;
        const uint32_t quotient0 = Simt::UintDiv(yIdx, m0, s0);
        uint32_t inputIndex0 = yIdx - quotient0 * outputShape0;
        const uint32_t quotient1 = Simt::UintDiv(quotient0, m1, s1);
        uint32_t inputIndex1 = quotient0 - quotient1 * outputShape1;
        uint32_t inputIndex2 = quotient1; // 末维索引天然小于 outputShape2，无需再除
        uint32_t xIdx = inputIndex0 * dstStride0 + inputIndex1 * dstStride1 + inputIndex2 * dstStride2;
        outputGM[coreOffset + idx] = inputGM[xIdx];
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void SimtComputeDimFour(__gm__ T* inputGM, __gm__ volatile T* outputGM, uint32_t coreFactor, uint32_t coreOffset,
                            uint32_t outputShape0, uint32_t outputShape1, uint32_t outputShape2, uint32_t outputShape3,
                            uint32_t dstStride0, uint32_t dstStride1, uint32_t dstStride2, uint32_t dstStride3,
                            uint32_t m0, uint32_t m1, uint32_t m2, uint32_t s0, uint32_t s1, uint32_t s2)
{
    for (uint32_t idx = static_cast<uint32_t>(threadIdx.x); idx < coreFactor;
         idx += static_cast<uint32_t>(blockDim.x)) {
        uint32_t yIdx = coreOffset + idx;
        const uint32_t quotient0 = Simt::UintDiv(yIdx, m0, s0);
        uint32_t inputIndex0 = yIdx - quotient0 * outputShape0;
        const uint32_t quotient1 = Simt::UintDiv(quotient0, m1, s1);
        uint32_t inputIndex1 = quotient0 - quotient1 * outputShape1;
        const uint32_t quotient2 = Simt::UintDiv(quotient1, m2, s2);
        uint32_t inputIndex2 = quotient1 - quotient2 * outputShape2;
        uint32_t inputIndex3 = quotient2; // 末维索引天然小于 outputShape3，无需再除
        uint32_t xIdx = inputIndex0 * dstStride0 + inputIndex1 * dstStride1 + inputIndex2 * dstStride2 +
                        inputIndex3 * dstStride3;
        outputGM[coreOffset + idx] = inputGM[xIdx];
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void SimtComputeDimFive(__gm__ T* inputGM, __gm__ volatile T* outputGM, uint32_t coreFactor, uint32_t coreOffset,
                            uint32_t outputShape0, uint32_t outputShape1, uint32_t outputShape2, uint32_t outputShape3,
                            uint32_t outputShape4, uint32_t dstStride0, uint32_t dstStride1, uint32_t dstStride2,
                            uint32_t dstStride3, uint32_t dstStride4, uint32_t m0, uint32_t m1, uint32_t m2,
                            uint32_t m3, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3)
{
    for (uint32_t idx = static_cast<uint32_t>(threadIdx.x); idx < coreFactor;
         idx += static_cast<uint32_t>(blockDim.x)) {
        uint32_t yIdx = coreOffset + idx;
        const uint32_t quotient0 = Simt::UintDiv(yIdx, m0, s0);
        uint32_t inputIndex0 = yIdx - quotient0 * outputShape0;
        const uint32_t quotient1 = Simt::UintDiv(quotient0, m1, s1);
        uint32_t inputIndex1 = quotient0 - quotient1 * outputShape1;
        const uint32_t quotient2 = Simt::UintDiv(quotient1, m2, s2);
        uint32_t inputIndex2 = quotient1 - quotient2 * outputShape2;
        const uint32_t quotient3 = Simt::UintDiv(quotient2, m3, s3);
        uint32_t inputIndex3 = quotient2 - quotient3 * outputShape3;
        uint32_t inputIndex4 = quotient3; // 末维索引天然小于 outputShape4，无需再除
        uint32_t xIdx = inputIndex0 * dstStride0 + inputIndex1 * dstStride1 + inputIndex2 * dstStride2 +
                        inputIndex3 * dstStride3 + inputIndex4 * dstStride4;
        outputGM[coreOffset + idx] = inputGM[xIdx];
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void SimtComputeDimSix(__gm__ T* inputGM, __gm__ volatile T* outputGM, uint32_t coreFactor, uint32_t coreOffset,
                           uint32_t outputShape0, uint32_t outputShape1, uint32_t outputShape2, uint32_t outputShape3,
                           uint32_t outputShape4, uint32_t outputShape5, uint32_t dstStride0, uint32_t dstStride1,
                           uint32_t dstStride2, uint32_t dstStride3, uint32_t dstStride4, uint32_t dstStride5,
                           uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3, uint32_t m4, uint32_t s0, uint32_t s1,
                           uint32_t s2, uint32_t s3, uint32_t s4)
{
    for (uint32_t idx = static_cast<uint32_t>(threadIdx.x); idx < coreFactor;
         idx += static_cast<uint32_t>(blockDim.x)) {
        uint32_t yIdx = coreOffset + idx;
        const uint32_t quotient0 = Simt::UintDiv(yIdx, m0, s0);
        uint32_t inputIndex0 = yIdx - quotient0 * outputShape0;
        const uint32_t quotient1 = Simt::UintDiv(quotient0, m1, s1);
        uint32_t inputIndex1 = quotient0 - quotient1 * outputShape1;
        const uint32_t quotient2 = Simt::UintDiv(quotient1, m2, s2);
        uint32_t inputIndex2 = quotient1 - quotient2 * outputShape2;
        const uint32_t quotient3 = Simt::UintDiv(quotient2, m3, s3);
        uint32_t inputIndex3 = quotient2 - quotient3 * outputShape3;
        const uint32_t quotient4 = Simt::UintDiv(quotient3, m4, s4);
        uint32_t inputIndex4 = quotient3 - quotient4 * outputShape4;
        uint32_t inputIndex5 = quotient4; // 末维索引天然小于 outputShape5，无需再除
        uint32_t xIdx = inputIndex0 * dstStride0 + inputIndex1 * dstStride1 + inputIndex2 * dstStride2 +
                        inputIndex3 * dstStride3 + inputIndex4 * dstStride4 + inputIndex5 * dstStride5;
        outputGM[coreOffset + idx] = inputGM[xIdx];
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void SimtComputeDimSeven(__gm__ T* inputGM, __gm__ volatile T* outputGM, uint32_t coreFactor, uint32_t coreOffset,
                             uint32_t outputShape0, uint32_t outputShape1, uint32_t outputShape2, uint32_t outputShape3,
                             uint32_t outputShape4, uint32_t outputShape5, uint32_t outputShape6, uint32_t dstStride0,
                             uint32_t dstStride1, uint32_t dstStride2, uint32_t dstStride3, uint32_t dstStride4,
                             uint32_t dstStride5, uint32_t dstStride6, uint32_t m0, uint32_t m1, uint32_t m2,
                             uint32_t m3, uint32_t m4, uint32_t m5, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3,
                             uint32_t s4, uint32_t s5)
{
    for (uint32_t idx = static_cast<uint32_t>(threadIdx.x); idx < coreFactor;
         idx += static_cast<uint32_t>(blockDim.x)) {
        uint32_t yIdx = coreOffset + idx;
        const uint32_t quotient0 = Simt::UintDiv(yIdx, m0, s0);
        uint32_t inputIndex0 = yIdx - quotient0 * outputShape0;
        const uint32_t quotient1 = Simt::UintDiv(quotient0, m1, s1);
        uint32_t inputIndex1 = quotient0 - quotient1 * outputShape1;
        const uint32_t quotient2 = Simt::UintDiv(quotient1, m2, s2);
        uint32_t inputIndex2 = quotient1 - quotient2 * outputShape2;
        const uint32_t quotient3 = Simt::UintDiv(quotient2, m3, s3);
        uint32_t inputIndex3 = quotient2 - quotient3 * outputShape3;
        const uint32_t quotient4 = Simt::UintDiv(quotient3, m4, s4);
        uint32_t inputIndex4 = quotient3 - quotient4 * outputShape4;
        const uint32_t quotient5 = Simt::UintDiv(quotient4, m5, s5);
        uint32_t inputIndex5 = quotient4 - quotient5 * outputShape5;
        uint32_t inputIndex6 = quotient5; // 末维索引天然小于 outputShape6，无需再除
        uint32_t xIdx = inputIndex0 * dstStride0 + inputIndex1 * dstStride1 + inputIndex2 * dstStride2 +
                        inputIndex3 * dstStride3 + inputIndex4 * dstStride4 + inputIndex5 * dstStride5 +
                        inputIndex6 * dstStride6;
        outputGM[coreOffset + idx] = inputGM[xIdx];
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
    void SimtComputeDimEight(__gm__ T* inputGM, __gm__ volatile T* outputGM, uint32_t coreFactor, uint32_t coreOffset,
                             uint32_t outputShape0, uint32_t outputShape1, uint32_t outputShape2, uint32_t outputShape3,
                             uint32_t outputShape4, uint32_t outputShape5, uint32_t outputShape6, uint32_t outputShape7,
                             uint32_t dstStride0, uint32_t dstStride1, uint32_t dstStride2, uint32_t dstStride3,
                             uint32_t dstStride4, uint32_t dstStride5, uint32_t dstStride6, uint32_t dstStride7,
                             uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3, uint32_t m4, uint32_t m5, uint32_t m6,
                             uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3, uint32_t s4, uint32_t s5, uint32_t s6)
{
    for (uint32_t idx = static_cast<uint32_t>(threadIdx.x); idx < coreFactor;
         idx += static_cast<uint32_t>(blockDim.x)) {
        uint32_t yIdx = coreOffset + idx;
        const uint32_t quotient0 = Simt::UintDiv(yIdx, m0, s0);
        uint32_t inputIndex0 = yIdx - quotient0 * outputShape0;
        const uint32_t quotient1 = Simt::UintDiv(quotient0, m1, s1);
        uint32_t inputIndex1 = quotient0 - quotient1 * outputShape1;
        const uint32_t quotient2 = Simt::UintDiv(quotient1, m2, s2);
        uint32_t inputIndex2 = quotient1 - quotient2 * outputShape2;
        const uint32_t quotient3 = Simt::UintDiv(quotient2, m3, s3);
        uint32_t inputIndex3 = quotient2 - quotient3 * outputShape3;
        const uint32_t quotient4 = Simt::UintDiv(quotient3, m4, s4);
        uint32_t inputIndex4 = quotient3 - quotient4 * outputShape4;
        const uint32_t quotient5 = Simt::UintDiv(quotient4, m5, s5);
        uint32_t inputIndex5 = quotient4 - quotient5 * outputShape5;
        const uint32_t quotient6 = Simt::UintDiv(quotient5, m6, s6);
        uint32_t inputIndex6 = quotient5 - quotient6 * outputShape6;
        uint32_t inputIndex7 = quotient6; // 末维索引天然小于 outputShape7，无需再除
        uint32_t xIdx = inputIndex0 * dstStride0 + inputIndex1 * dstStride1 + inputIndex2 * dstStride2 +
                        inputIndex3 * dstStride3 + inputIndex4 * dstStride4 + inputIndex5 * dstStride5 +
                        inputIndex6 * dstStride6 + inputIndex7 * dstStride7;
        outputGM[coreOffset + idx] = inputGM[xIdx];
    }
}

template <typename T>
__aicore__ inline void TransposeSmallShape<T>::Process()
{
    // 分核完全由 Host 算好（1024B 对齐），实际核数由 blockDim 下发，此处用 GetBlockNum() 获取。
    const uint32_t dim = static_cast<uint32_t>(tilingData_->dim);
    const uint32_t coreNum = static_cast<uint32_t>(GetBlockNum());
    if (static_cast<uint32_t>(blockIdx_) >= coreNum) {
        return;
    }
    uint32_t blkProcessNum = static_cast<uint32_t>(tilingData_->blkFactor);
    uint32_t blkStartOffset = static_cast<uint32_t>(blockIdx_) * static_cast<uint32_t>(tilingData_->blkFactor);
    if (static_cast<uint32_t>(blockIdx_) == coreNum - 1 && tilingData_->blkTailFactor != 0) {
        blkProcessNum = static_cast<uint32_t>(tilingData_->blkTailFactor);
    }

    // 由 inputShape + perm 推导输出 shape 与源步长
    uint32_t outputShape[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t dstStride[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    uint32_t dstStrideTmp[8] = {1, 1, 1, 1, 1, 1, 1, 1};

    for (uint32_t i = 1; i < dim; i++) {
        dstStrideTmp[i] = static_cast<uint32_t>(tilingData_->inputShape[dim - i]) * dstStrideTmp[i - 1];
    }
    for (uint32_t i = 0; i < dim; i++) {
        outputShape[i] = static_cast<uint32_t>(tilingData_->inputShape[tilingData_->perm[dim - 1 - i]]);
        dstStride[i] = dstStrideTmp[dim - 1 - static_cast<uint32_t>(tilingData_->perm[dim - 1 - i])];
    }

    // 末维不需要除法（余数即索引），只为主维准备魔术数/位移，省去一次 kernel 侧预计算。
    uint32_t shift[8];
    uint32_t m[8];
    for (uint32_t i = 0; i + 1 < dim; i++) {
        GetUintDivMagicAndShift(m[i], shift[i], outputShape[i]);
    }

    if (dim == DIM_THREE) {
        asc_vf_call<SimtComputeDimThree<T>>(dim3(THREAD_DIM), (__gm__ T*)(inputGM_.GetPhyAddr()),
                                            (__gm__ volatile T*)(outputGM_.GetPhyAddr()), blkProcessNum, blkStartOffset,
                                            outputShape[0], outputShape[1], outputShape[2], dstStride[0], dstStride[1],
                                            dstStride[2], m[0], m[1], shift[0], shift[1]);
    } else if (dim == DIM_FOUR) {
        asc_vf_call<SimtComputeDimFour<T>>(
            dim3(THREAD_DIM), (__gm__ T*)(inputGM_.GetPhyAddr()), (__gm__ volatile T*)(outputGM_.GetPhyAddr()),
            blkProcessNum, blkStartOffset, outputShape[0], outputShape[1], outputShape[2], outputShape[3], dstStride[0],
            dstStride[1], dstStride[2], dstStride[3], m[0], m[1], m[2], shift[0], shift[1], shift[2]);
    } else if (dim == DIM_FIVE) {
        asc_vf_call<SimtComputeDimFive<T>>(
            dim3(THREAD_DIM), (__gm__ T*)(inputGM_.GetPhyAddr()), (__gm__ volatile T*)(outputGM_.GetPhyAddr()),
            blkProcessNum, blkStartOffset, outputShape[0], outputShape[1], outputShape[2], outputShape[3],
            outputShape[4], dstStride[0], dstStride[1], dstStride[2], dstStride[3], dstStride[4], m[0], m[1], m[2],
            m[3], shift[0], shift[1], shift[2], shift[3]);
    } else if (dim == DIM_SIX) {
        asc_vf_call<SimtComputeDimSix<T>>(
            dim3(THREAD_DIM), (__gm__ T*)(inputGM_.GetPhyAddr()), (__gm__ volatile T*)(outputGM_.GetPhyAddr()),
            blkProcessNum, blkStartOffset, outputShape[0], outputShape[1], outputShape[2], outputShape[3],
            outputShape[4], outputShape[5], dstStride[0], dstStride[1], dstStride[2], dstStride[3], dstStride[4],
            dstStride[5], m[0], m[1], m[2], m[3], m[4], shift[0], shift[1], shift[2], shift[3], shift[4]);
    } else if (dim == DIM_SEVEN) {
        asc_vf_call<SimtComputeDimSeven<T>>(
            dim3(THREAD_DIM), (__gm__ T*)(inputGM_.GetPhyAddr()), (__gm__ volatile T*)(outputGM_.GetPhyAddr()),
            blkProcessNum, blkStartOffset, outputShape[0], outputShape[1], outputShape[2], outputShape[3],
            outputShape[4], outputShape[5], outputShape[6], dstStride[0], dstStride[1], dstStride[2], dstStride[3],
            dstStride[4], dstStride[5], dstStride[6], m[0], m[1], m[2], m[3], m[4], m[5], shift[0], shift[1], shift[2],
            shift[3], shift[4], shift[5]);
    } else if (dim == DIM_EIGHT) {
        asc_vf_call<SimtComputeDimEight<T>>(
            dim3(THREAD_DIM), (__gm__ T*)(inputGM_.GetPhyAddr()), (__gm__ volatile T*)(outputGM_.GetPhyAddr()),
            blkProcessNum, blkStartOffset, outputShape[0], outputShape[1], outputShape[2], outputShape[3],
            outputShape[4], outputShape[5], outputShape[6], outputShape[7], dstStride[0], dstStride[1], dstStride[2],
            dstStride[3], dstStride[4], dstStride[5], dstStride[6], dstStride[7], m[0], m[1], m[2], m[3], m[4], m[5],
            m[6], shift[0], shift[1], shift[2], shift[3], shift[4], shift[5], shift[6]);
    }
}

} // namespace Transpose

#endif // TRANSPOSE_SMALL_SHAPE_H
