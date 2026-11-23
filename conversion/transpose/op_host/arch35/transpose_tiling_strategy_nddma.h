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
 * \file transpose_tiling_strategy_nddma.h
 * \brief NDDMA 策略类（NDDMA_CUT_1(10002)/NDDMA_CUT_2(10003) 共用）
 *
 * 适用场景：dim<=5，使用 NDDMA 5 维搬运；按切分轴数量选择单轴切分或双轴切分。
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_NDDMA_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_NDDMA_H

#include "transpose_tiling_strategy.h"

namespace optiling {

/**
 * @brief NDDMA 策略类（NDDMA_CUT_1(10002)/NDDMA_CUT_2(10003) 共用）
 */
class TransposeNddmaStrategy : public TransposeNddmaGroupStrategy {
private:
    /**
     * @brief UB 切分信息
     *
     * 记录 Tiling 计算过程中的 UB 切分参数：输入/输出侧预算、切分轴与切分因子等。
     */
    struct SplitInfo {
        int64_t ubElement = 1;     ///< UB 元素预算
        int64_t inUbElement = 1;   ///< 输入侧剩余预算
        int64_t outUbElement = 1;  ///< 输出侧剩余预算
        int64_t inUbActual = 1;    ///< 输入侧已吞入轴容量乘积
        int64_t inCutIndex = 0;    ///< 输入切分轴索引
        int64_t outCutIndex = 0;   ///< 输出切分轴索引
        int64_t inUbFactor = 0;    ///< 输入主块切分因子
        int64_t outUbFactor = 0;   ///< 输出主块切分因子
        int64_t inTailFactor = 0;  ///< 输入尾块元素数
        int64_t outTailFactor = 0; ///< 输出尾块元素数
    };

    // NDDMA_CUT_2 的 4 种数据区间边界
    struct Interval {
        int64_t start = 0; ///< 区间起始全局 block 索引
        int64_t end = 0;   ///< 区间结束全局 block 索引
    };

public:
    using TransposeNddmaGroupStrategy::TransposeNddmaGroupStrategy;

    bool IsSupport() const override;

protected:
    TilingStatus DoExecute() override;

    uint32_t GetUsedCoreNum() const override { return static_cast<uint32_t>(realCoreNum_); }
    uint64_t GetTilingKey() const override { return tilingKey_; }

    bool SetTilingData() override;
    void LogTilingData() override;

private:
    void CalcUBSplitInfo();    ///< UB 预算 + DoSplitUB 判定单/双切
    void CalcBlockSplitInfo(); ///< 按内部 tilingKey_ 走 10002/10003 分核
    void NDDMADimExpand();     ///< 5 维扩展
    void GetInUbShapeInfo();   ///< 按内部 tilingKey_ 走 10002/10003 UB shape
    void GetIntervalInfo();    ///< 仅 NDDMA_CUT_2 填写区间边界
    void SetRealCoreNumAndBlkFactor(int64_t coreNum);

    // NDDMA 私有实现
    void DoSplitUB();
    int64_t DoSplitUBInput(bool tryAbsorb = false);
    void DoSplitUBOutputScan(int64_t remainingTotalElment);
    int64_t FindOutIndex(int64_t index);
    bool UbOutOfBoundCheck(int64_t currentSplitIndex, int64_t currentSplitValue, bool calcIn);
    void FindSplitFactorByMultiplesLast(int64_t currentSplitIndex, int64_t currentInShapeDim,
                                        int64_t remainingTotalElment, int64_t coreNumMultiples);
    void CalcBlockSplitInfoForCutOnce();
    void CalcBlockSplitInfoForCutTwice();
    void CalcInUbShapeInfoForCutOnce();
    void CalcInUbShapeInfoForCutTwice();
    void GetIntervalInfoForCutTwice();

    // NDDMA 私有中间状态（仅本模板使用）
    SplitInfo splitInfo_{};     ///< UB 切分信息
    int64_t realCoreNum_ = 0;   ///< 实际使用核数
    int64_t blkFactor_ = 0;     ///< 分核：每核基础 block 数
    int64_t blkTailFactor_ = 0; ///< 分核：尾核额外 block 数
    uint64_t tilingKey_ = 0;    ///< 当前策略 tilingKey（NDDMA_CUT_1/NDDMA_CUT_2）
    int64_t expandedPerm_[TP_NDDMA_MAX_DIM_NUM] = {0, 1, 2, 3, 4};        ///< 5 维扩展 perm
    int64_t expandedInputShape_[TP_NDDMA_MAX_DIM_NUM] = {1, 1, 1, 1, 1};  ///< 5 维扩展输入 shape
    int64_t expandedOutputShape_[TP_NDDMA_MAX_DIM_NUM] = {1, 1, 1, 1, 1}; ///< 5 维扩展输出 shape
    int64_t inUbMainSrcShape_[TP_NDDMA_MAX_DIM_NUM] = {0};                ///< 主块 UB 内输入 5 维 shape
    int64_t inUbMainDstShape_[TP_NDDMA_MAX_DIM_NUM] = {0};                ///< 主块 UB 内输出 5 维 shape
    int64_t inUbInputTailSrcShape_[TP_NDDMA_MAX_DIM_NUM] = {0};           ///< 输入尾块 UB 内输入 shape
    int64_t inUbInputTailDstShape_[TP_NDDMA_MAX_DIM_NUM] = {0};           ///< 输入尾块 UB 内输出 shape
    int64_t inUbOutputTailSrcShape_[TP_NDDMA_MAX_DIM_NUM] = {0};          ///< 输出尾块 UB 内输入 shape
    int64_t inUbOutputTailDstShape_[TP_NDDMA_MAX_DIM_NUM] = {0};          ///< 输出尾块 UB 内输出 shape
    int64_t inUbTailSrcShape_[TP_NDDMA_MAX_DIM_NUM] = {0};                ///< 全尾块 UB 内输入 shape
    int64_t inUbTailDstShape_[TP_NDDMA_MAX_DIM_NUM] = {0};                ///< 全尾块 UB 内输出 shape
    Interval offsetRangeMain_{};                                          ///< NDDMA_CUT_2 区间边界
    Interval offsetRangeInputTail_{};                                     ///< NDDMA_CUT_2 输入尾块区间
    Interval offsetRangeOutputTail_{};                                    ///< NDDMA_CUT_2 输出尾块区间
    Interval offsetRangeTail_{};                                          ///< NDDMA_CUT_2 全尾块区间
};

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_NDDMA_H
