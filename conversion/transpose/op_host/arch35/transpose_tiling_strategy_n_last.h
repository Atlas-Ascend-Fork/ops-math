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
 * \file transpose_tiling_strategy_n_last.h
 * \brief N_LAST(10004) 策略类
 *
 * 适用场景：尾轴不转置且尾轴元素数不少于 32，按连续行搬移；
 * 非 SIMT 平台要求数据量不小于 4MB。
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_N_LAST_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_N_LAST_H

#include "transpose_tiling_strategy.h"

namespace optiling {

/**
 * @brief N_LAST(10004) 策略
 */
class TransposeNLastStrategy : public TransposeTilingStrategy {
public:
    using TransposeTilingStrategy::TransposeTilingStrategy;

    bool IsSupport() const override;

protected:
    TilingStatus DoExecute() override;

    uint32_t GetUsedCoreNum() const override { return static_cast<uint32_t>(realCoreNum_); }
    uint64_t GetTilingKey() const override { return static_cast<uint64_t>(TK_N_LAST); }

    bool SetTilingData() override;
    void LogTilingData() override;

private:
    void CalcUBSplitInfo();    ///< 双缓冲，ubElement = ubSize/2/eleBytes
    void CalcBlockSplitInfo(); ///< N_LAST 分核
    void CalcBlockSplitInfoForNLastTranspose();
    void SetRealCoreNumAndBlkFactor(int64_t coreNum);
    bool UbOutOfBoundCheckNLast(int64_t currentSplitIndex, int64_t currentSplitValue);
    void FindSplitFactorByRateNLast(int64_t currentSplitIndex, int64_t currentInShapeDim, int64_t remainingTotalElment);
    void FindSplitFactorByMultiplesNLast(int64_t currentSplitIndex, int64_t currentInShapeDim,
                                         int64_t remainingTotalElment, int64_t coreNumMultiples);
    void CheckInUbFactorValid(int64_t& currentSplitIndex, int64_t& currentInShapeDim, int64_t& remainingTotalElment,
                              int64_t& coreNumMultiples, int64_t* solvedTotalElment);

    // UB 切分结果
    int64_t ubElement_ = 1;    ///< UB 元素预算
    int64_t inUbElement_ = 1;  ///< 输入侧剩余预算
    int64_t inCutIndex_ = 0;   ///< 输入切分轴索引
    int64_t inUbFactor_ = 0;   ///< 输入主块切分因子
    int64_t inTailFactor_ = 0; ///< 输入尾块元素数

    // N_LAST 私有中间状态（仅本模板使用）
    int64_t realCoreNum_ = 0;   ///< 实际使用核数
    int64_t blkFactor_ = 0;     ///< 分核：每核基础 block 数
    int64_t blkTailFactor_ = 0; ///< 分核：尾核额外 block 数
};

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_N_LAST_H
