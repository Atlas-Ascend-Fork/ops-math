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
 * \file transpose_tiling_strategy_small_shape.h
 * \brief SMALL_SHAPE(10001) 策略类
 *
 * 适用场景：平台支持 SIMT，且输入数据量小于 SIMT 小 shape 阈值。
 * 数据在 GM 之间直读直写，不经过 UB。
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_SMALL_SHAPE_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_SMALL_SHAPE_H

#include "transpose_tiling_strategy.h"

namespace optiling {

/**
 * @brief SMALL_SHAPE(10001) 策略
 *
 * 准入：平台支持 SIMT 且 totalVolumeActual*eleLenInBytes < 本模板小 shape 阈值。
 * 尾轴不转置场景由注册在前的 N_LAST 优先处理；本模板注册在 NDDMA/NDDMA_BIG_DIM 之前，
 * 剩余模板无需再判体积阈值。
 */
class TransposeSmallShapeStrategy : public TransposeTilingStrategy {
public:
    using TransposeTilingStrategy::TransposeTilingStrategy;

    // 准入检查，纯判定、无副作用
    bool IsSupport() const override;

protected:
    TilingStatus DoExecute() override;

    uint32_t GetUsedCoreNum() const override { return static_cast<uint32_t>(realCoreNum_); }
    uint64_t GetTilingKey() const override { return tilingKey_; }

    bool SetTilingData() override;
    void LogTilingData() override;

private:
    void CalcBlockSplitInfo();
    void NDDMADimExpand();                ///< 5 维扩展（仅 dim<=5 调用）
    void CalcInUbShapeInfoForNoNeedCut(); ///< 无需切分的 UB 内 shape

    // UB 切分结果
    int64_t outCutIndex_ = 0;   ///< 输出切分轴索引
    int64_t outUbFactor_ = 0;   ///< 输出主块切分因子
    int64_t outTailFactor_ = 0; ///< 输出尾块元素数

    // SMALL_SHAPE 私有中间状态（仅本模板使用）
    int64_t realCoreNum_ = 0;                                             ///< 实际使用核数
    int64_t blkFactor_ = 0;                                               ///< 分核：每核基础 block 数
    int64_t blkTailFactor_ = 0;                                           ///< 分核：尾核额外 block 数
    uint64_t tilingKey_ = 0;                                              ///< 当前策略 tilingKey
    int64_t expandedPerm_[TP_NDDMA_MAX_DIM_NUM] = {0, 1, 2, 3, 4};        ///< 5 维扩展 perm
    int64_t expandedInputShape_[TP_NDDMA_MAX_DIM_NUM] = {1, 1, 1, 1, 1};  ///< 5 维扩展输入 shape
    int64_t expandedOutputShape_[TP_NDDMA_MAX_DIM_NUM] = {1, 1, 1, 1, 1}; ///< 5 维扩展输出 shape
    // SMALL_SHAPE 的 UB 内 shape（无需切分场景）
    int64_t inUbMainSrcShape_[TP_NDDMA_MAX_DIM_NUM] = {0}; ///< 主块 UB 内输入 5 维 shape
    int64_t inUbMainDstShape_[TP_NDDMA_MAX_DIM_NUM] = {0}; ///< 主块 UB 内输出 5 维 shape
    int64_t inUbTailSrcShape_[TP_NDDMA_MAX_DIM_NUM] = {0}; ///< 全尾块 UB 内输入 5 维 shape
    int64_t inUbTailDstShape_[TP_NDDMA_MAX_DIM_NUM] = {0}; ///< 全尾块 UB 内输出 5 维 shape
};

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_SMALL_SHAPE_H
