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
 * \file transpose_tiling_strategy_tensor_move.h
 * \brief TENSOR_MOVE(10000) 策略类
 *
 * 适用场景：退化为 1 维的纯数据搬运（无需处理转置轴）。
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_TENSOR_MOVE_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_TENSOR_MOVE_H

#include "transpose_tiling_strategy.h"

namespace optiling {

/**
 * @brief TENSOR_MOVE(10000) 策略
 *
 * 1 维纯搬运：数据线性排布、无需处理转置。双缓冲流水线：
 * ubElement = ubSize/2/eleBytes（一半 CopyIn、一半 CopyOut）。
 */
class TransposeTensorMoveStrategy : public TransposeTilingStrategy {
public:
    using TransposeTilingStrategy::TransposeTilingStrategy;

    // 准入检查，纯判定、无副作用
    bool IsSupport() const override;

protected:
    // 执行 tiling（收尾 SetTilingData/SetContext/LogTilingData 由通用基类 Execute 完成）
    TilingStatus DoExecute() override;

    uint32_t GetUsedCoreNum() const override { return static_cast<uint32_t>(realCoreNum_); }
    uint64_t GetTilingKey() const override { return tilingKey_; }

    bool SetTilingData() override;
    void LogTilingData() override;

private:
    void CalcUBSplitInfo();    ///< ubElement = ubSize/2/eleBytes（双缓冲）
    void CalcBlockSplitInfo(); ///< 一维均分分核
    void NDDMADimExpand();     ///< 5 维扩展（仅 dim<=5 调用）

    // UB 切分结果
    int64_t ubElement_ = 1;  ///< UB 元素预算
    int64_t inUbFactor_ = 0; ///< 输入主块切分因子

    // TENSOR_MOVE 私有中间状态（仅本模板使用）
    int64_t realCoreNum_ = 0;                                             ///< 实际使用核数
    int64_t blkFactor_ = 0;                                               ///< 分核：每核基础 block 数
    int64_t blkTailFactor_ = 0;                                           ///< 分核：尾核额外 block 数
    uint64_t tilingKey_ = 0;                                              ///< 当前策略 tilingKey
    int64_t expandedPerm_[TP_NDDMA_MAX_DIM_NUM] = {0, 1, 2, 3, 4};        ///< 5 维扩展 perm
    int64_t expandedInputShape_[TP_NDDMA_MAX_DIM_NUM] = {1, 1, 1, 1, 1};  ///< 5 维扩展输入 shape
    int64_t expandedOutputShape_[TP_NDDMA_MAX_DIM_NUM] = {1, 1, 1, 1, 1}; ///< 5 维扩展输出 shape
};

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_TENSOR_MOVE_H
