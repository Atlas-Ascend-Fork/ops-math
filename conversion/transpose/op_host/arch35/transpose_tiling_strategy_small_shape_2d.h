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
 * \file transpose_tiling_strategy_small_shape_2d.h
 * \brief SMALL_SHAPE_2D(10012) 策略类
 *
 * 适用场景：reduced 维度为 2、reducedPerm == [1,0] 的二维转置，dsize ∈ {1,2,4,8}，
 * 且 64KB <= 总字节数 < SIMT 小 shape 阈值。
 *
 * 实现路线：UB 双缓冲 + MTE2/MTE3 + 3D 4x4 SIMT。Host 预留 32KB DCACHE 并下发
 * chunkCols/totalWork/inPitch/bufBytes；与 SMALL_SHAPE（纯直读）相互独立。
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_SMALL_SHAPE_2D_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_SMALL_SHAPE_2D_H

#include "transpose_tiling_strategy.h"

namespace optiling {

/**
 * @brief SMALL_SHAPE_2D(10012) 策略
 *
 * 准入：SIMT 平台 && reduced dim==2 && reducedPerm==[1,0] && dsize ∈ {1,2,4,8}
 * && cols>4 && 64KB <= 数据量 < 小 shape 阈值。注册在 SMALL_SHAPE 之前，保证中等
 * small shape 走 UB 4x4 模板；UB 预算不足时 FALLBACK 给 SMALL_SHAPE（纯直读）兜底。
 */
class TransposeSmallShape2DStrategy : public TransposeTilingStrategy {
public:
    using TransposeTilingStrategy::TransposeTilingStrategy;

    // 准入检查，纯判定、无副作用
    bool IsSupport() const override;

protected:
    TilingStatus DoExecute() override;

    uint32_t GetUsedCoreNum() const override { return static_cast<uint32_t>(blockNum_); }
    uint64_t GetTilingKey() const override { return static_cast<uint64_t>(TK_SMALL_SHAPE_2D); }

    bool SetTilingData() override;
    void LogTilingData() override;

    /**
     * @brief 模板形状/平台准入
     */
    bool IsShapeSupported() const;

private:
    /**
     * @brief UB 切分总入口：预算 → chunk 列数 → 分核
     *
     * 128B tile 为 stripe；N 方向按 UB 预算切 chunkCols，必要时缩小 chunkCols
     * 以产生更多 work item。UB 预算不足一个 tile 时返回 FALLBACK。
     */
    TilingStatus CalcUbSplit();
    TilingStatus CalcUbBudget();  ///< 计算 tile / 单 buffer 字节数（UB 预算）
    TilingStatus CalcChunkCols(); ///< 计算 chunkCols / inPitch（含并行度修正与 UB 收缩）
    int64_t AdjustChunkColsForParallelism(int64_t chunkCols) const; ///< 按目标 chunk 数修正列宽
    TilingStatus ShrinkChunkColsToUb(int64_t chunkCols);            ///< 收缩 chunkCols 到单 buffer 预算内
    void CalcWorkDistribution();                                    ///< 计算 totalWork / blockNum

    int64_t rows_ = 0;         ///< 输入行数 M（= 输出列数）
    int64_t cols_ = 0;         ///< 输入列数 N（= 输出行数）
    int64_t tile_ = 0;         ///< 128B tile 的元素数
    int64_t totalStripes_ = 0; ///< M 方向 stripe 数
    int64_t blockNum_ = 0;     ///< 实际使用核数（= blockDim）
    int64_t chunkCols_ = 0;    ///< N 方向单 chunk 列数
    int64_t totalWork_ = 0;    ///< 总 work item 数
    int64_t inPitch_ = 0;      ///< UB 输入行间距（元素）
    int64_t bufBytes_ = 0;     ///< 单 buffer 字节数
};

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_SMALL_SHAPE_2D_H
