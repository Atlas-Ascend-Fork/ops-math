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
 * \file transpose_tensor_move.h
 * \brief TilingKey=10000 TENSOR_MOVE 策略实现
 *
 * 适用场景：经过 RemoveAxisV2 + MergeAxisV2 后 reduced dim == 1，即输入融合后仅剩单一维度，
 * 等价于纯数据搬运，无需任何转置操作。
 *
 * 核心设计：
 *   - 普通/关联入口统一使用紧凑 TransposeTensorMoveTilingData
 *   - 使用静态 Tensor 双缓冲，沿用逐块搬运循环
 *   - 每次搬入 inUbFactor 个元素到 UB，再原样写回 GM
 *   - 使用 Mutex 管理搬入、搬出与缓冲复用
 *
 * 数据流：GM → DataCopyPad(UB) → DataCopyPad(GM)，纯顺序搬运无转置
 *
 * 注：保持"类定义 + 类外函数体定义"结构，便于与重构前版本对比差异。
 */
#ifndef TRANSPOSE_TENSOR_MOVE_H
#define TRANSPOSE_TENSOR_MOVE_H

#include "transpose_base.h"

namespace Transpose {
using namespace AscendC;

/**
 * @brief TENSOR_MOVE 策略类，实现纯数据搬运
 *
 * @tparam T 数据元素类型（int8_t/int16_t/int32_t/int64_t 或复合类型）
 */
template <typename T>
class TransposeTensorMove {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, __tiling_data_ptr__ TransposeTensorMoveTilingData* data);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyBlock(LocalTensor<T>& local, int64_t offset, uint32_t copyNum, MutexID mutexId);

    GlobalTensor<T> inputGM_;          ///< 输入 GM 张量
    GlobalTensor<T> outputGM_;         ///< 输出 GM 张量
    LocalTensor<T> local_[BUFFER_NUM]; ///< 静态双缓冲
    uint32_t inUbFactor_ = 0;          ///< 单次 UB 搬运块长（元素数）
    uint32_t tailElements_ = 0;        ///< 末核尾块元素数（非末核为 0）
    uint64_t fullBlocks_ = 0;          ///< 当前核需搬运的完整主块数
    int64_t srcOffset_ = 0;            ///< 当前核在全局数据中的起始偏移（元素数）
};

template <typename T>
__aicore__ inline void TransposeTensorMove<T>::Init(GM_ADDR x, GM_ADDR y,
                                                    __tiling_data_ptr__ TransposeTensorMoveTilingData* data)
{
    const uint64_t blockIdx = GetBlockIdx();
    const uint64_t extra = data->extraBlockCores;
    const uint64_t mainBlocksPerCore = data->mainBlocksPerCore;
    const uint64_t startBlock = blockIdx * mainBlocksPerCore + (blockIdx < extra ? blockIdx : extra);
    inUbFactor_ = data->mainBlockElements;
    srcOffset_ = startBlock * inUbFactor_;
    fullBlocks_ = mainBlocksPerCore + (blockIdx < extra ? 1 : 0);
    tailElements_ = blockIdx + 1 == static_cast<uint64_t>(GetBlockNum()) ? data->tailElements : 0;
    inputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
    local_[0] = LocalTensor<T>(TPosition::VECIN, 0, inUbFactor_);
    // 第二个静态缓冲额外错开 BANK_OFFSET_BYTES，避免同 bank 冲突。
    local_[1] = LocalTensor<T>(TPosition::VECIN,
                               inUbFactor_ * sizeof(T) + TransposeTensorMoveTilingData::BANK_OFFSET_BYTES, inUbFactor_);
}

template <typename T>
__aicore__ inline void TransposeTensorMove<T>::Process()
{
    const uint64_t loopNum = fullBlocks_ + (tailElements_ != 0);
    for (uint64_t loopIdx = 0; loopIdx < loopNum; ++loopIdx) {
        const uint32_t copyNum = loopIdx < fullBlocks_ ? inUbFactor_ : tailElements_;
        const uint32_t bufferIndex = loopIdx % BUFFER_NUM;
        CopyBlock(local_[bufferIndex], srcOffset_ + loopIdx * inUbFactor_, copyNum, static_cast<MutexID>(bufferIndex));
    }
}

template <typename T>
__aicore__ inline void TransposeTensorMove<T>::CopyBlock(LocalTensor<T>& local, int64_t offset, uint32_t copyNum,
                                                         MutexID mutexId)
{
    DataCopyExtParams params{1, static_cast<uint32_t>(copyNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    Mutex::Lock<PIPE_MTE2>(mutexId);
#if defined(__DAV_920R1__) || defined(__DAV_920R2__)
    DataCopyPad(local, inputGM_[offset], params, pad, true);
#else
    DataCopyPad(local, inputGM_[offset], params, pad);
#endif
    Mutex::Unlock<PIPE_MTE2>(mutexId);
    Mutex::Lock<PIPE_MTE3>(mutexId);
    DataCopyPad(outputGM_[offset], local, params);
    Mutex::Unlock<PIPE_MTE3>(mutexId);
}

} // namespace Transpose

#endif // TRANSPOSE_TENSOR_MOVE_H
