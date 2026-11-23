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
 * \file transpose_tiling_strategy_n_last_small.cpp
 * \brief TK_N_LAST_SMALL(10014) 策略类实现
 *
 * 普通入口与关联入口统一：3/4 维、尾轴不变、尾维>=32、总字节<= int32 max（约 2GiB）即命中，
 * 写 72B TransposeNLastSmallTilingData（不再做 5 维扩展）。
 * 与 N_LAST(10004) 完全独立，10004 的分核/因子回溯/退轴逻辑不在此处。
 */

#include "transpose_tiling_strategy_n_last_small.h"

#include <algorithm>
#include <limits>

namespace optiling {
namespace {
static constexpr int64_t
    N_LAST_SMALL_MAX_BYTES = std::numeric_limits<int32_t>::max(); ///< 10014 支持的最大总字节（<= int32 max，约 2GiB）
static constexpr int64_t N_LAST_SMALL_MIN_CORES = 16;             ///< 10014 最少使用核数
static constexpr int64_t N_LAST_SMALL_BYTES_PER_CORE = 8 * 1024; ///< 10014 每核目标字节数
static constexpr int64_t N_LAST_SMALL_DIM_THREE = 3;
static constexpr int64_t N_LAST_SMALL_DIM_FOUR = 4;
static constexpr int64_t N_LAST_SMALL_MOVEALIGN_LAST_MIN_ELE = 32; ///< 尾轴最小元素数
static constexpr int64_t N_LAST_SMALL_BUFFER_NUM = 2;              ///< 双缓冲
static constexpr int64_t N_LAST_SMALL_BYTES_MARGIN = 1;            ///< 总字节上界预留，避免边界溢出
static constexpr int64_t N_LAST_SMALL_PARALLELISM_FACTOR = 2; ///< 每核至少 2 个 block 才认为并行度足够
static constexpr int64_t N_LAST_SMALL_SINGLE_TILE_COUNT = 1;  ///< 单块模式的 tile 数
} // namespace

/**
 * @brief 准入检查：3/4 维、尾轴不转置、尾轴元素数 >= 32、总字节 <= int32 max（约 2GiB）
 */
bool TransposeNLastSmallStrategy::IsSupport() const
{
    return (shapeInfo_.dim == N_LAST_SMALL_DIM_THREE || shapeInfo_.dim == N_LAST_SMALL_DIM_FOUR) &&
           !shapeInfo_.isLastAxisTranspose &&
           shapeInfo_.reducedInShape[shapeInfo_.dim - 1] >= N_LAST_SMALL_MOVEALIGN_LAST_MIN_ELE &&
           shapeInfo_.totalVolumeActual <=
               (N_LAST_SMALL_MAX_BYTES - N_LAST_SMALL_BYTES_MARGIN) / shapeInfo_.eleLenInBytes;
}

namespace {
int64_t GetNLastSmallCoreNum(int64_t bytes, int64_t availableCores)
{
    return std::min(availableCores,
                    std::max(N_LAST_SMALL_MIN_CORES, Ops::Base::CeilDiv(bytes, N_LAST_SMALL_BYTES_PER_CORE)));
}
} // namespace

/**
 * @brief 阶段1：提取三/四维简化 shape、行字节数与各轴源步长
 */
void TransposeNLastSmallStrategy::CollectNLastSmallShape(NLastSmallSplit& split) const
{
    split.fourDim = shapeInfo_.dim == N_LAST_SMALL_DIM_FOUR;
    split.firstAxis = shapeInfo_.dim - N_LAST_SMALL_DIM_THREE;
    split.outer = split.fourDim ? shapeInfo_.reducedInShape[0] : 1;
    split.a = shapeInfo_.reducedInShape[split.firstAxis];
    split.b = shapeInfo_.reducedInShape[split.firstAxis + 1];
    split.rowBytes = shapeInfo_.reducedInShape[shapeInfo_.dim - 1] * shapeInfo_.eleLenInBytes;

    int64_t stride = shapeInfo_.eleLenInBytes;
    for (int64_t i = shapeInfo_.dim - 1; i >= 0; --i) {
        const int64_t axis = shapeInfo_.reducedPerm[i];
        split.dstStrides[axis] = stride;
        stride *= shapeInfo_.reducedInShape[axis];
    }
}

/**
 * @brief 阶段2：目标核数、A 维主块以及是否走单块模式
 */
void TransposeNLastSmallStrategy::DecideNLastSmallBlock(NLastSmallSplit& split) const
{
    split.bytes = shapeInfo_.totalVolumeActual * shapeInfo_.eleLenInBytes;
    split.targetCores = GetNLastSmallCoreNum(split.bytes, coreNum_);
    split.mainA = Ops::Base::CeilDiv(split.a, split.targetCores);
    split.singleCores = Ops::Base::CeilDiv(split.a, split.mainA);
    const bool enoughParallelism = split.singleCores * N_LAST_SMALL_PARALLELISM_FACTOR >= split.targetCores ||
                                   split.bytes <= split.singleCores * N_LAST_SMALL_BYTES_PER_CORE;
    split.singleBlock = !split.fourDim && enoughParallelism && split.rowBytes % ubBlockSize_ == 0 &&
                        split.mainA * split.b * split.rowBytes <= ubSize_ &&
                        split.mainA * split.b * split.rowBytes <= TransposeNLastSmallTilingData::MAX_BLOCK_BYTES &&
                        split.b <= TransposeNLastSmallTilingData::MAX_BLOCK_COUNT &&
                        split.mainA <= std::numeric_limits<uint16_t>::max() &&
                        split.singleCores <= std::numeric_limits<uint16_t>::max();
}

/**
 * @brief 阶段3：确定 tileA/tileB/tileBytes/tile 数与核分配
 */
void TransposeNLastSmallStrategy::CalcNLastSmallTileConfig(NLastSmallSplit& split)
{
    // 默认单 tile 配置；非单块时再按 UB 预算重新切 tileA/tileB/tileBytes。
    tileA_ = static_cast<uint16_t>(split.mainA);
    tileB_ = static_cast<uint16_t>(split.b);
    tileBytes_ = static_cast<uint32_t>(split.rowBytes);
    bufferOffset_ = 0;
    bTiles_ = N_LAST_SMALL_SINGLE_TILE_COUNT;
    cTiles_ = N_LAST_SMALL_SINGLE_TILE_COUNT;
    if (!split.singleBlock) {
        CalcNLastSmallMultiBlockTile(split);
    }

    split.aTiles = Ops::Base::CeilDiv(split.a, static_cast<int64_t>(tileA_));
    split.totalTiles = split.outer * split.aTiles * static_cast<int64_t>(bTiles_) * static_cast<int64_t>(cTiles_);
    split.cores = std::min(coreNum_, split.totalTiles);
    split.tailA = split.a - (split.aTiles - 1) * static_cast<int64_t>(tileA_);
}

/**
 * @brief 非单块模式：按 UB 预算切 tileA/tileB/tileBytes
 */
void TransposeNLastSmallStrategy::CalcNLastSmallMultiBlockTile(const NLastSmallSplit& split)
{
    bufferOffset_ = static_cast<uint32_t>(
        Ops::Base::FloorAlign(std::min(ubSize_ / N_LAST_SMALL_BUFFER_NUM,
                                       static_cast<int64_t>(TransposeNLastSmallTilingData::MAX_BLOCK_BYTES)),
                              ubBlockSize_));
    const int64_t tileBudget = std::min(
        static_cast<int64_t>(bufferOffset_),
        std::max(std::min(Ops::Base::CeilAlign(split.rowBytes, ubBlockSize_), N_LAST_SMALL_BYTES_PER_CORE),
                 Ops::Base::CeilAlign(Ops::Base::CeilDiv(split.bytes, split.targetCores), ubBlockSize_)));
    tileBytes_ = static_cast<uint32_t>(std::min<int64_t>(split.rowBytes, tileBudget));
    const int64_t rowStride = Ops::Base::CeilAlign(static_cast<int64_t>(tileBytes_), ubBlockSize_);
    const int64_t maxRows = std::min(tileBudget / rowStride,
                                     static_cast<int64_t>(TransposeNLastSmallTilingData::MAX_BLOCK_COUNT));
    tileB_ = static_cast<uint16_t>(std::min<int64_t>(split.b, maxRows));
    tileA_ = static_cast<uint16_t>(std::min<int64_t>(split.a, maxRows / static_cast<int64_t>(tileB_)));
    bTiles_ = static_cast<uint32_t>(Ops::Base::CeilDiv(split.b, static_cast<int64_t>(tileB_)));
    cTiles_ = static_cast<uint32_t>(Ops::Base::CeilDiv(split.rowBytes, static_cast<int64_t>(tileBytes_)));
}

/**
 * @brief 阶段4：把中间量落到类成员（后续 SetTilingData 回填）
 */
void TransposeNLastSmallStrategy::StoreNLastSmallTilingDataFields(const NLastSmallSplit& split)
{
    // 总字节数不超过 int32 max（约 2GiB），A 方向的完整源/目标步长均可用无符号 32 位表示。
    rowBytes_ = static_cast<uint32_t>(split.rowBytes);
    srcTileStride_ = static_cast<uint32_t>(static_cast<int64_t>(tileA_) * split.b * split.rowBytes);
    dstTileStride_ = static_cast<uint32_t>(static_cast<int64_t>(tileA_) * split.dstStrides[split.firstAxis]);
    outDstStrideBytes_ = static_cast<uint32_t>(split.dstStrides[split.firstAxis + 1] - split.rowBytes);
    srcLoopStrideBytes_ = static_cast<uint32_t>(split.singleBlock ? split.b * split.rowBytes : 0);
    tailSrcBytes_ = static_cast<uint32_t>(split.tailA * split.b * split.rowBytes);
    tailA_ = static_cast<uint16_t>(split.tailA);
    usedCoreNum_ = static_cast<uint16_t>(split.cores);
    blocksPerCore_ = static_cast<uint32_t>(split.totalTiles / split.cores);
    extraBlocks_ = static_cast<uint16_t>(split.totalTiles % split.cores);
    tailB_ = static_cast<uint16_t>(split.b - (static_cast<int64_t>(bTiles_) - 1) * static_cast<int64_t>(tileB_));
    aTiles_ = static_cast<uint32_t>(split.aTiles);
    srcOuterStride_ = static_cast<uint32_t>(split.fourDim ? split.a * split.b * split.rowBytes : 0);
    dstOuterStride_ = static_cast<uint32_t>(split.fourDim ? split.dstStrides[0] : 0);
    dstAStride_ = static_cast<uint32_t>(split.fourDim ? split.dstStrides[split.firstAxis] : 0);
}

/**
 * @brief 普通/关联入口 10014：三/四维尾轴不变的分核、tile/尾块、单块/分块与寻址步长
 */
TilingStatus TransposeNLastSmallStrategy::DoExecute()
{
    NLastSmallSplit split;
    CollectNLastSmallShape(split);
    DecideNLastSmallBlock(split);
    CalcNLastSmallTileConfig(split);
    StoreNLastSmallTilingDataFields(split);
    return TilingStatus::SUCCESS;
}

/**
 * @brief 填写 10014 消费的注册 TilingData 字段（72B TransposeNLastSmallTilingData）
 */
bool TransposeNLastSmallStrategy::SetTilingData()
{
    OP_LOGD(context_->GetNodeName(), "Entering SetTilingData.");
    auto* data = context_->GetTilingData<TransposeNLastSmallTilingData>();
    if (data == nullptr) {
        OP_LOGE(context_->GetNodeName(), "get tiling data failed, capacity is not enough!");
        return false;
    }
    data->rowBytes = rowBytes_;
    data->srcTileStride = srcTileStride_;
    data->dstTileStride = dstTileStride_;
    data->outDstStrideBytes = outDstStrideBytes_;
    data->srcLoopStrideBytes = srcLoopStrideBytes_;
    data->tailSrcBytes = tailSrcBytes_;
    data->tileA = tileA_;
    data->tailA = tailA_;
    data->tileB = tileB_;
    data->bTiles = bTiles_;
    data->cTiles = cTiles_;
    data->tileBytes = tileBytes_;
    data->bufferOffset = bufferOffset_;
    data->blocksPerCore = blocksPerCore_;
    data->extraBlocks = extraBlocks_;
    data->tailB = tailB_;
    data->aTiles = aTiles_;
    data->srcOuterStride = srcOuterStride_;
    data->dstOuterStride = dstOuterStride_;
    data->dstAStride = dstAStride_;
    return true;
}

void TransposeNLastSmallStrategy::LogTilingData()
{
    OP_LOGI(context_->GetNodeName(), "Entering LogTilingData.");
    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        OP_LOGI(context_->GetNodeName(), "reducedInShape[%ld] is:%ld, reducedOutShape[%ld]:%ld, reducedPerm[%ld]:%ld.",
                i, shapeInfo_.reducedInShape[i], i, shapeInfo_.reducedOutShape[i], i, shapeInfo_.reducedPerm[i]);
    }
    OP_LOGI(context_->GetNodeName(),
            "NLastSmall tilingData is rowBytes:%u, srcTileStride:%u, tileA:%u, tileB:%u, coreNum:%u, "
            "aTiles:%u, bTiles:%u, cTiles:%u, bufferOffset:%u, Tiling4Transpose ends.",
            rowBytes_, srcTileStride_, static_cast<unsigned>(tileA_), static_cast<unsigned>(tileB_),
            static_cast<unsigned>(usedCoreNum_), aTiles_, bTiles_, cTiles_, bufferOffset_);
}

} // namespace optiling
