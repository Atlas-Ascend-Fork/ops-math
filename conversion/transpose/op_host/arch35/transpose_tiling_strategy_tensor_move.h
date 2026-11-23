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
 * 适用场景：dim==1 一维纯搬运（Permute 退化为 Copy）。
 *
 * IsSupport()：shapeInfo_.dim == 1。
 * DoExecute()：动态主块划分，主块按 cache line 对齐，完整块在参与核间均分，
 *   末核额外处理唯一尾块，普通/关联入口统一使用紧凑 TilingData（同布局）。
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_TENSOR_MOVE_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_TENSOR_MOVE_H

#include "transpose_tiling_strategy.h"

namespace optiling {

/**
 * @brief TENSOR_MOVE(10000) 策略
 *
 * 1 维纯搬运：数据线性排布、无需处理转置。统一紧凑 TilingData（普通/关联同布局）。
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
    uint64_t GetTilingKey() const override { return static_cast<uint64_t>(TK_TENSOR_MOVE); }

    bool SetTilingData() override;
    void LogTilingData() override;

private:
    // TENSOR_MOVE 私有中间状态（仅本策略使用），与 TilingData 字段一一对应
    uint64_t mainBlocksPerCore_ = 0; ///< 每核基础完整主块数
    uint32_t mainBlockElements_ = 0; ///< 单个主块元素数
    uint32_t extraBlockCores_ = 0;   ///< 额外多分一个主块的核数
    uint32_t tailElements_ = 0;      ///< 末核唯一尾块元素数
    int64_t realCoreNum_ = 0;        ///< 实际使用核数（对应 TilingData.coreNum）
};

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_TENSOR_MOVE_H
