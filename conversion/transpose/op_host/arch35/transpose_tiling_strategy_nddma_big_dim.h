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
 * \file transpose_tiling_strategy_nddma_big_dim.h
 * \brief NDDMA_BIG_DIM(10005) 策略类
 *
 * 适用场景：dim>5，将多维 shape 压缩为 5 维 NDDMA 格式进行搬运，仅切输出侧一根轴。
 * TilingData 使用紧凑 TransposeNddmaBigDimTilingData，仅下发 kernel 实际消费的字段。
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_NDDMA_BIG_DIM_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_NDDMA_BIG_DIM_H

#include "transpose_tiling_strategy.h"

namespace optiling {

/**
 * @brief NDDMA_BIG_DIM(10005) 策略
 */
class TransposeNddmaBigDimStrategy : public TransposeNddmaGroupStrategy {
public:
    using TransposeNddmaGroupStrategy::TransposeNddmaGroupStrategy;

    bool IsSupport() const override;

protected:
    TilingStatus DoExecute() override;

    uint32_t GetUsedCoreNum() const override { return static_cast<uint32_t>(realCoreNum_); }
    uint64_t GetTilingKey() const override { return static_cast<uint64_t>(TK_NDDMA_BIG_DIM); }

    bool SetTilingData() override;
    void LogTilingData() override;

private:
    void CalcUBSplitInfo(); ///< bigDimBufferNum_ + DoSplitUBBigDim + fallback 单缓冲
    void CalcBlockSplitInfoForBigDim();
    void SetRealCoreNumAndBlkFactor(int64_t coreNum);
    void DoSplitUBBigDim();
    void FlushBaseNumForBigDim();

    // UB 切分结果
    int64_t ubElement_ = 1;     ///< UB 元素预算
    int64_t outCutIndex_ = 0;   ///< 输出切分轴索引
    int64_t outUbFactor_ = 0;   ///< 输出主块切分因子
    int64_t outTailFactor_ = 0; ///< 输出尾块元素数

    // NDDMA_BIG_DIM 私有中间状态（仅本模板使用）
    int64_t realCoreNum_ = 0;                    ///< 实际使用核数
    int64_t blkFactor_ = 0;                      ///< 分核：每核基础 block 数
    int64_t blkTailFactor_ = 0;                  ///< 分核：尾核额外 block 数
    int64_t bigDimBufferNum_ = 1;                ///< NDDMA_BIG_DIM 缓冲数（默认单缓冲，DAV_9201/9202=2）
    int64_t totalNddmaNum_ = 1;                  ///< NDDMA 循环总次数
    int64_t baseInShape_[TP_MAX_AXIS_NUM] = {0}; ///< 基础输入 shape（地址基数）
    int64_t baseNddmaShape_[TP_NDDMA_MAX_DIM_NUM] = {0}; ///< NDDMA 5 维基础 shape
    int64_t nddmaIdx_[TP_NDDMA_MAX_DIM_NUM] = {-1};      ///< NDDMA 5 维索引（首个 -1，其余 0）
};

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_NDDMA_BIG_DIM_H
