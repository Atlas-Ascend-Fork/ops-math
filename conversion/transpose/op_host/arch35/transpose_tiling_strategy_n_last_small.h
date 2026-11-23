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
 * \file transpose_tiling_strategy_n_last_small.h
 * \brief TK_N_LAST_SMALL(10014) 三/四维尾轴不变小搬运策略类
 *
 * 适用场景：3/4 维、尾轴不转置、尾维元素数 >= 32（连续行搬移）且总字节 <= int32 max（约 2GiB）。
 *   - IsSupport()：dim==3||dim==4 && !isLastAxisTranspose && reducedInShape[dim-1] >= 32
 *     && totalVolumeActual <= (int32 max - 1) / eleLenInBytes；
 *   - DoExecute()：tilingKey=10014，写 72B TransposeNLastSmallTilingData。
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_N_LAST_SMALL_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_N_LAST_SMALL_H

#include <array>
#include "transpose_tiling_strategy.h"

namespace optiling {

/**
 * @brief TK_N_LAST_SMALL(10014) 策略
 */
class TransposeNLastSmallStrategy : public TransposeTilingStrategy {
public:
    using TransposeTilingStrategy::TransposeTilingStrategy;

    bool IsSupport() const override;

protected:
    TilingStatus DoExecute() override;

    uint32_t GetUsedCoreNum() const override { return usedCoreNum_; }
    uint64_t GetTilingKey() const override { return static_cast<uint64_t>(TK_N_LAST_SMALL); }

    bool SetTilingData() override;
    void LogTilingData() override;

private:
    /**
     * @brief 10014 切分中间量（DoExecute 各阶段共享）
     */
    struct NLastSmallSplit {
        bool fourDim = false;
        bool singleBlock = false;
        int64_t firstAxis = 0;
        int64_t outer = 1;
        int64_t a = 0;
        int64_t b = 0;
        int64_t rowBytes = 0;
        int64_t bytes = 0;
        int64_t targetCores = 0;
        int64_t mainA = 0;
        int64_t singleCores = 0;
        int64_t aTiles = 0;
        int64_t totalTiles = 0;
        int64_t cores = 0;
        int64_t tailA = 0;
        std::array<int64_t, 4> dstStrides{}; ///< 各输入轴的源跨步（元素）
    };

    void CollectNLastSmallShape(NLastSmallSplit& split) const;          ///< 阶段1：shape/行字节/步长
    void DecideNLastSmallBlock(NLastSmallSplit& split) const;           ///< 阶段2：核数/主块/单块判定
    void CalcNLastSmallTileConfig(NLastSmallSplit& split);              ///< 阶段3：tile/尾块/核分配
    void CalcNLastSmallMultiBlockTile(const NLastSmallSplit& split);    ///< 非单块模式的 UB 切分
    void StoreNLastSmallTilingDataFields(const NLastSmallSplit& split); ///< 阶段4：落到类成员

    // 10014 72B 结构字段（逐字段回填，避免整结构赋值）
    uint32_t rowBytes_ = 0;           ///< 单行字节数
    uint32_t srcTileStride_ = 0;      ///< 单 tile 源跨步字节
    uint32_t dstTileStride_ = 0;      ///< 单 tile 目标跨步字节
    uint32_t outDstStrideBytes_ = 0;  ///< 输出目标跨块步长字节
    uint32_t srcLoopStrideBytes_ = 0; ///< 单块输出 UB 循环步长字节
    uint32_t tailSrcBytes_ = 0;       ///< 尾块源字节数
    uint16_t tileA_ = 0;              ///< tile A 维长度
    uint16_t tailA_ = 0;              ///< A 维尾块长度
    uint16_t usedCoreNum_ = 0;        ///< 实际使用核数
    uint16_t tileB_ = 0;              ///< tile B 维长度
    uint32_t bTiles_ = 0;             ///< B 维 tile 数
    uint32_t cTiles_ = 0;             ///< C 维 tile 数
    uint32_t tileBytes_ = 0;          ///< 单 tile 字节数
    uint32_t bufferOffset_ = 0;       ///< 第二块 UB 缓冲偏移
    uint32_t blocksPerCore_ = 0;      ///< 每核基础 block 数
    uint16_t extraBlocks_ = 0;        ///< 尾核额外 block 数
    uint16_t tailB_ = 0;              ///< B 维尾块长度
    uint32_t aTiles_ = 0;             ///< A 维 tile 数
    uint32_t srcOuterStride_ = 0;     ///< 四维源外层跨步字节
    uint32_t dstOuterStride_ = 0;     ///< 四维目标外层跨步字节
    uint32_t dstAStride_ = 0;         ///< 四维目标 A 维跨步字节
};

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_N_LAST_SMALL_H
