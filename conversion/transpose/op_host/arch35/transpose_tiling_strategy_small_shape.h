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
 * \brief SMALL_SHAPE(10001)/SMALL_SHAPE_23D(10011) 策略类
 *
 * 适用场景：平台支持 SIMT，且输入数据量小于 SIMT 小 shape 阈值。
 * 数据在 GM 之间直读直写，不经过 UB；2D/3D 统一走 24B 精简 TilingData，4-8D 走 52B。
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_SMALL_SHAPE_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_SMALL_SHAPE_H

#include "transpose_tiling_strategy.h"

namespace optiling {

/**
 * @brief SMALL_SHAPE(10001)/SMALL_SHAPE_23D(10011) 策略
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

    // 分核完全由 Host 算好，blockDim 精确等于 realCoreNum_（不再空跑）
    uint32_t GetUsedCoreNum() const override { return static_cast<uint32_t>(realCoreNum_); }
    uint64_t GetTilingKey() const override { return tilingKey_; }

    bool SetTilingData() override;
    void LogTilingData() override;

private:
    /**
     * @brief 按 TilingDataT 写入精简码流（23D/48D 共用同一套代码）
     * @tparam TilingDataT TransposeSmallShapeTilingData<DIM> 的具体实例
     */
    template <typename TilingDataT>
    bool SetTilingDataImpl();
    void CalcBlockSplitInfo(); ///< 1024B 对齐 Host 分核

    // SMALL_SHAPE 私有中间状态（仅本模板使用）
    int64_t realCoreNum_ = 0;   ///< 实际使用核数（= blockDim；不下发，kernel 用 GetBlockNum() 获取）
    int64_t blkFactor_ = 0;     ///< 分核：每核基础元素数（1024B 对齐）
    int64_t blkTailFactor_ = 0; ///< 分核：末核元素数
    uint64_t tilingKey_ = 0;    ///< 当前策略 tilingKey
    // 精简码流字段（2-8D 共用；按 dim 选择 <3> 或 <TP_MAX_AXIS_NUM> 实例）
    uint8_t dim_ = 0;                          ///< 实际简化后维数（2-8）
    int32_t inputShape_[TP_MAX_AXIS_NUM] = {}; ///< 简化后输入 shape
    int8_t perm_[TP_MAX_AXIS_NUM] = {};        ///< 简化后 perm，未用槽位 -1
};

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_SMALL_SHAPE_H
