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
 * \file transpose_tiling_strategy_small_shape_2d.cpp
 * \brief SMALL_SHAPE_2D(10012) 策略类实现
 *
 * 准入：reduced dim==2、reducedPerm==[1,0]、dsize ∈ {1,2,4,8}、cols>4，
 * 且 64KB <= 数据量 < 小 shape 阈值（DAV_5102 为 1MB，其余 4MB）。
 * stripe × chunk 切分 + 32KB DCACHE 预留。
 */

#include <algorithm>
#include "transpose_tiling_strategy_small_shape_2d.h"

namespace optiling {
namespace {
// 与 SMALL_SHAPE 一致的阈值口径
static constexpr int64_t SMALL_SHAPE_2D_BYTES_THRES_HOLD = 4000000; ///< 默认小 shape 阈值（4MB）
static constexpr int64_t SMALL_SHAPE_2D_MIN_BYTES = 64 * 1024;      ///< 低于 64KB 交给 SMALL_SHAPE
static constexpr int64_t SMALL_SHAPE_2D_MIN_COLS = 4;               ///< cols<=4 交给 SMALL_SHAPE
static constexpr int64_t SMALL_SHAPE_2D_DCACHE_SIZE = 32 * 1024;    ///< SIMT DCACHE 预留
static constexpr int64_t SMALL_SHAPE_2D_BLOCK_COUNT_MAX = 4095;     ///< DataCopyPad blockCount 上限
static constexpr int64_t SMALL_SHAPE_2D_TILE_BYTES = 128;           ///< 单 tile 基础字节数（128B）
static constexpr int64_t SMALL_SHAPE_2D_PAD_BYTES = 32;             ///< 输出行尾额外 pad 字节
static constexpr int64_t SMALL_SHAPE_2D_ROW_BYTES = SMALL_SHAPE_2D_TILE_BYTES + SMALL_SHAPE_2D_PAD_BYTES;
static constexpr int64_t SMALL_SHAPE_2D_UB_BUFFER_NUM = 4; ///< UB 缓冲数（x0/x1/y0/y1）
static constexpr int64_t SMALL_SHAPE_2D_MIN_CORE_NUM = 1;  ///< 至少使用 1 核

inline bool SmallShape2DIsSupportedDsize(int64_t dsize) { return dsize == 1 || dsize == 2 || dsize == 4 || dsize == 8; }
} // namespace

/**
 * @brief 准入检查：形状/平台准入
 */
bool TransposeSmallShape2DStrategy::IsSupport() const { return IsShapeSupported(); }

/**
 * @brief 形状/平台准入：SIMT 平台 && 2D perm=[1,0] && 1/2/4/8B && [64KB, 小 shape 阈值)
 */
bool TransposeSmallShape2DStrategy::IsShapeSupported() const
{
    if (!isSupportSimt_) {
        return false;
    }

    if (npuArch_ == Ops::Base::DAV_5102) {
        return false;
    }

    if (!IsPermValue({1, 0})) {
        return false;
    }
    // 4bit 类型的 eleLenInBytes 是 SDK 编码值，不落在 {1,2,4,8}，天然被排除
    if (!SmallShape2DIsSupportedDsize(shapeInfo_.eleLenInBytes)) {
        return false;
    }
    const int64_t rows = shapeInfo_.reducedInShape[0];
    const int64_t cols = shapeInfo_.reducedInShape[1];
    if (rows <= 0 || cols <= SMALL_SHAPE_2D_MIN_COLS) {
        return false;
    }
    const int64_t volumeBytes = shapeInfo_.totalVolumeActual * shapeInfo_.eleLenInBytes;

    return volumeBytes >= SMALL_SHAPE_2D_MIN_BYTES && volumeBytes < SMALL_SHAPE_2D_BYTES_THRES_HOLD;
}

TilingStatus TransposeSmallShape2DStrategy::DoExecute()
{
    rows_ = shapeInfo_.reducedInShape[0];
    cols_ = shapeInfo_.reducedInShape[1];
    if (CalcUbSplit() != TilingStatus::SUCCESS) {
        return TilingStatus::FALLBACK; // UB 预算不足，交给 SMALL_SHAPE（纯直读）兜底
    }
    // SIMD/SIMT 混合：为 SIMT 预留 32KB DCACHE（平台 ubSize 已扣除 VF stack）
    const int64_t localMem = ubSize_ - SMALL_SHAPE_2D_DCACHE_SIZE;
    if (localMem > 0 && context_->SetLocalMemorySize(static_cast<uint32_t>(localMem)) != ge::GRAPH_SUCCESS) {
        OP_LOGD(context_->GetNodeName(), "SmallShape2D SetLocalMemorySize not supported by context, skip.");
    }
    return TilingStatus::SUCCESS;
}

/**
 * @brief UB：stripe × chunk 切分
 *
 * tile = 128B / dsize；stripe = tile 个输入行；N 方向按 UB 预算切 chunkCols。
 * M 很小时缩小 chunkCols 以产生更多 work item。UB 预算不足一个 tile 时返回 FALLBACK。
 */
TilingStatus TransposeSmallShape2DStrategy::CalcUbSplit()
{
    if (CalcUbBudget() != TilingStatus::SUCCESS) {
        return TilingStatus::FALLBACK;
    }
    if (CalcChunkCols() != TilingStatus::SUCCESS) {
        return TilingStatus::FALLBACK;
    }
    CalcWorkDistribution();
    return (blockNum_ > 0) ? TilingStatus::SUCCESS : TilingStatus::FALLBACK;
}

/**
 * @brief 计算 tile 元素数与单 buffer 字节数（UB 扣除 DCACHE 后四等分）
 */
TilingStatus TransposeSmallShape2DStrategy::CalcUbBudget()
{
    tile_ = SMALL_SHAPE_2D_TILE_BYTES / shapeInfo_.eleLenInBytes;
    totalStripes_ = Ops::Base::CeilDiv(rows_, tile_);

    // UB 均分 4 份（x0/x1/y0/y1），Host 只预留 DCACHE
    const int64_t kernelUb = ubSize_ - SMALL_SHAPE_2D_DCACHE_SIZE;
    if (kernelUb <= 0) {
        return TilingStatus::FALLBACK;
    }
    bufBytes_ = Ops::Base::FloorAlign(kernelUb / SMALL_SHAPE_2D_UB_BUFFER_NUM, ubBlockSize_);
    return (bufBytes_ > 0) ? TilingStatus::SUCCESS : TilingStatus::FALLBACK;
}

/**
 * @brief 计算 chunkCols 与 inPitch：初值取 buffer 容量上限，再做并行度修正与 UB 收缩
 */
TilingStatus TransposeSmallShape2DStrategy::CalcChunkCols()
{
    // 单 buffer 能放下的最大 chunk（chunkCols 为 tile 的整数倍；N<tile 时取 N）
    const int64_t perTileBytes = SMALL_SHAPE_2D_ROW_BYTES * tile_;
    const int64_t chunkTiles = bufBytes_ / perTileBytes;
    if (chunkTiles == 0) {
        return TilingStatus::FALLBACK;
    }
    const int64_t maxByCount = (SMALL_SHAPE_2D_BLOCK_COUNT_MAX / tile_) * tile_;
    int64_t chunkCols = std::min(std::min(chunkTiles * tile_, maxByCount), cols_);
    if (chunkCols <= 0) {
        return TilingStatus::FALLBACK;
    }
    chunkCols = AdjustChunkColsForParallelism(chunkCols);
    return ShrinkChunkColsToUb(chunkCols);
}

/**
 * @brief 并行度修正：按目标 chunk 数切，避免 stripe 数变化导致核数悬崖
 */
int64_t TransposeSmallShape2DStrategy::AdjustChunkColsForParallelism(int64_t chunkCols) const
{
    const int64_t coreNum = std::max<int64_t>(coreNum_, SMALL_SHAPE_2D_MIN_CORE_NUM);
    const int64_t maxChunks = Ops::Base::CeilDiv(cols_, tile_);
    const int64_t targetChunks = std::min(maxChunks, Ops::Base::CeilDiv(coreNum, totalStripes_));
    if (targetChunks <= 1) {
        return chunkCols;
    }
    if (cols_ < tile_) {
        return cols_;
    }
    int64_t byParallel = Ops::Base::CeilDiv(cols_, targetChunks);
    byParallel = (byParallel / tile_) * tile_;
    if (byParallel < tile_) {
        byParallel = tile_;
    }
    return std::min(byParallel, chunkCols);
}

/**
 * @brief 收缩 chunkCols：确保输入/输出 buffer 各自不超过单 buffer 预算
 */
TilingStatus TransposeSmallShape2DStrategy::ShrinkChunkColsToUb(int64_t chunkCols)
{
    const int64_t eleLen = shapeInfo_.eleLenInBytes;
    const int64_t outPitch = tile_ + SMALL_SHAPE_2D_PAD_BYTES / eleLen;
    for (;;) {
        const int64_t inPitch = (CeilAlignUBBlkBytes(chunkCols * eleLen) + SMALL_SHAPE_2D_PAD_BYTES) / eleLen;
        const int64_t inBytes = tile_ * inPitch * eleLen;
        const int64_t outBytes = chunkCols * outPitch * eleLen;
        if (inBytes <= bufBytes_ && outBytes <= bufBytes_) {
            inPitch_ = inPitch;
            chunkCols_ = chunkCols;
            return TilingStatus::SUCCESS;
        }
        if (chunkCols <= tile_) {
            return TilingStatus::FALLBACK;
        }
        chunkCols -= tile_;
    }
}

/**
 * @brief 按 chunkCols 计算总 work item 数与实际使用核数
 */
void TransposeSmallShape2DStrategy::CalcWorkDistribution()
{
    const int64_t coreNum = std::max<int64_t>(coreNum_, SMALL_SHAPE_2D_MIN_CORE_NUM);
    const int64_t chunksPerStripe = Ops::Base::CeilDiv(cols_, chunkCols_);
    totalWork_ = totalStripes_ * chunksPerStripe;
    blockNum_ = std::min(coreNum, totalWork_);
}

/**
 * @brief 写 TransposeSmallShape2DTilingData（容量不足 → FAILURE）
 */
bool TransposeSmallShape2DStrategy::SetTilingData()
{
    TransposeSmallShape2DTilingData* tilingData = context_->GetTilingData<TransposeSmallShape2DTilingData>();
    if (tilingData == nullptr) {
        OP_LOGE(context_->GetNodeName(), "get tiling data failed, capacity is not enough!");
        return false;
    }
    *tilingData = TransposeSmallShape2DTilingData{};
    tilingData->rows = static_cast<uint32_t>(rows_);
    tilingData->cols = static_cast<uint32_t>(cols_);
    tilingData->chunkCols = static_cast<uint32_t>(chunkCols_);
    tilingData->totalWork = static_cast<uint32_t>(totalWork_);
    tilingData->inPitch = static_cast<uint32_t>(inPitch_);
    tilingData->bufBytes = static_cast<uint32_t>(bufBytes_);
    return true;
}

void TransposeSmallShape2DStrategy::LogTilingData()
{
    OP_LOGI(context_->GetNodeName(),
            "[SmallShape2D] rows:%ld, cols:%ld, blockNum:%ld, tile:%ld, totalStripes:%ld, chunkCols:%ld, "
            "totalWork:%ld, inPitch:%ld, bufBytes:%ld, Tiling4Transpose ends.",
            rows_, cols_, blockNum_, tile_, totalStripes_, chunkCols_, totalWork_, inPitch_, bufBytes_);
}

} // namespace optiling
