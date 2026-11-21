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
 * \file transpose_tiling_strategy_transto5hd_10_perf.h
 * \brief TRANSTO5HD_10_PERF(10017) 策略类
 *
 * 适用场景：2D perm=[1,0] 的 TransDataTo5HD 全 tile 转置（DAV_3510/DAV_5102）。
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_TRANSTO5HD_10_PERF_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_TRANSTO5HD_10_PERF_H

#include "transpose_tiling_strategy.h"

namespace optiling {

/**
 * @brief TRANSTO5HD_10_PERF(10017) 策略
 *
 * TransDataTo5HD 全 tile 转置：m x n 的二维转置，按 tileM x tileN 的 tile 搬入 UB
 * 后用 16x16 块转置指令处理，再按列搬出。tile 几何由本策略按 shape/UB 预算自适应确定。
 */
class TransposeTransTo5HD10PerfStrategy : public TransposeTilingStrategy {
public:
    using TransposeTilingStrategy::TransposeTilingStrategy;

    // 准入检查：2D perm=[1,0] 特化的全部切分前条件（纯判定、无副作用）
    bool IsSupport() const override;

protected:
    // 执行切分（计算 tile 几何）；UB 预算不足一个 tile 时 FALLBACK
    TilingStatus DoExecute() override;

    // 写 TransposeTransTo5HD10PerfTilingData（容量不足 → FAILURE）
    bool SetTilingData() override;
    // 打印 TilingData
    void LogTilingData() override;

    uint32_t GetUsedCoreNum() const override { return static_cast<uint32_t>(coreNum_); }
    uint64_t GetTilingKey() const override { return static_cast<uint64_t>(TK_TRANSTO5HD_10_PERF); }

private:
    /**
     * @brief 计算 tile 几何并存入本策略中间态（不写 TilingData）
     *
     * UB 预算无法容纳至少一个 16x16 tile 时返回 FALLBACK（回落主流程）。
     *
     * @return TilingStatus::SUCCESS 切分完成；TilingStatus::FALLBACK UB 预算不足
     */
    TilingStatus CalcTileGeometry();

    // 切分结果（DoExecute 产出，SetTilingData/LogTilingData 消费）
    int64_t m_ = 0;     ///< 输入行数
    int64_t n_ = 0;     ///< 输入列数
    int64_t tileM_ = 0; ///< 行方向 tile 大小
    int64_t tileN_ = 0; ///< 列方向 tile 大小
};

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_TRANSTO5HD_10_PERF_H
