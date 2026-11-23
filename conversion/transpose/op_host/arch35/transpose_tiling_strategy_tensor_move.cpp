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
 * \file transpose_tiling_strategy_tensor_move.cpp
 * \brief TENSOR_MOVE(10000) 策略类实现
 *
 * 准入条件、分核与 TilingData 回填均使用紧凑 TransposeTensorMoveTilingData，
 * 普通入口与关联算子入口共用同一布局，不再调用通用 NDDMA 5 维扩展。
 */

#include "transpose_tiling_strategy_tensor_move.h"

#include <algorithm>

namespace optiling {
namespace {
static constexpr int64_t TENSOR_MOVE_ALIGN_BYTES = TransposeTensorMoveTilingData::CACHE_LINE_BYTES;
static constexpr uint64_t TENSOR_MOVE_MIN_TILE_BYTES = 16 * 1024;
static constexpr uint64_t TENSOR_MOVE_PREFERRED_TILE_BYTES = 32 * 1024;
static constexpr uint64_t TENSOR_MOVE_MAX_OVERLAP_ROUNDS = 8;
static constexpr uint64_t TENSOR_MOVE_BUFFER_NUM = 2; ///< 双缓冲
} // namespace

bool TransposeTensorMoveStrategy::IsSupport() const { return shapeInfo_.dim == 1; }

TilingStatus TransposeTensorMoveStrategy::DoExecute()
{
    OP_LOGD(context_->GetNodeName(), "Entering DoExecute of TENSOR_MOVE.");
    // 动态主块划分（普通/关联统一）：主块按 cache line 对齐，完整块在参与核间均分，
    // 末核额外处理唯一尾块；UB 双缓冲错位后按对齐粒度取可用上限。
    const int64_t maxTileBytes = Ops::Base::FloorAlign(
        (ubSize_ - TransposeTensorMoveTilingData::BANK_OFFSET_BYTES) / static_cast<int64_t>(TENSOR_MOVE_BUFFER_NUM),
        TENSOR_MOVE_ALIGN_BYTES);
    if (maxTileBytes < TENSOR_MOVE_ALIGN_BYTES) {
        OP_LOGE(context_->GetNodeName(), "Insufficient UB for TensorMove.");
        return TilingStatus::FAILURE;
    }
    const uint64_t elementBytes = static_cast<uint64_t>(shapeInfo_.eleLenInBytes);
    const uint64_t totalElements = static_cast<uint64_t>(shapeInfo_.totalVolumeActual);
    const uint64_t maxTileElements = static_cast<uint64_t>(maxTileBytes) / elementBytes;
    const uint64_t minTileElements = std::min(TENSOR_MOVE_MIN_TILE_BYTES / elementBytes, maxTileElements);
    const uint64_t alignElements = static_cast<uint64_t>(TENSOR_MOVE_ALIGN_BYTES) / elementBytes;
    const uint64_t perCoreElements = Ops::Base::CeilDiv(totalElements, static_cast<uint64_t>(coreNum_));
    const uint64_t requiredRounds = Ops::Base::CeilDiv(perCoreElements, maxTileElements);
    const uint64_t overlapRounds = std::min(
        TENSOR_MOVE_MAX_OVERLAP_ROUNDS,
        Ops::Base::CeilDiv(perCoreElements, TENSOR_MOVE_PREFERRED_TILE_BYTES / elementBytes));
    const uint64_t rounds = std::max<uint64_t>(1, std::max(requiredRounds, overlapRounds));
    // 按目标轮数分摊标准块，避免少量核多搬一个大块。
    const uint64_t alignedTileElements = Ops::Base::CeilAlign(Ops::Base::CeilDiv(perCoreElements, rounds),
                                                              alignElements);
    const uint64_t tileElements = std::clamp(alignedTileElements, minTileElements, maxTileElements);
    const uint64_t fullBlocks = totalElements / tileElements;
    const uint64_t tail = totalElements % tileElements;
    const uint64_t copyBlocks = fullBlocks + (tail != 0);
    const uint64_t usedCores = std::min<uint64_t>(static_cast<uint64_t>(coreNum_), std::max<uint64_t>(1, copyBlocks));
    mainBlocksPerCore_ = fullBlocks / usedCores;
    mainBlockElements_ = static_cast<uint32_t>(tileElements);
    realCoreNum_ = static_cast<int64_t>(usedCores);
    extraBlockCores_ = static_cast<uint32_t>(fullBlocks % usedCores);
    tailElements_ = static_cast<uint32_t>(tail);
    return TilingStatus::SUCCESS;
}

/**
 * @brief 回填 TENSOR_MOVE 消费的 TilingData
 *
 * 普通/关联入口统一写紧凑 TransposeTensorMoveTilingData。
 */
bool TransposeTensorMoveStrategy::SetTilingData()
{
    OP_LOGD(context_->GetNodeName(), "Entering SetTilingData.");
    auto* data = context_->GetTilingData<TransposeTensorMoveTilingData>();
    if (data == nullptr) {
        OP_LOGE(context_->GetNodeName(), "get tiling data failed, capacity is not enough!");
        return false;
    }
    *data = TransposeTensorMoveTilingData{};
    data->mainBlocksPerCore = mainBlocksPerCore_;
    data->mainBlockElements = mainBlockElements_;
    data->extraBlockCores = extraBlockCores_;
    data->tailElements = tailElements_;
    return true;
}

void TransposeTensorMoveStrategy::LogTilingData()
{
    OP_LOGI(context_->GetNodeName(), "Entering LogTilingData.");
    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        OP_LOGI(context_->GetNodeName(), "reducedInShape[%ld] is:%ld, reducedOutShape[%ld]:%ld, reducedPerm[%ld]:%ld.",
                i, shapeInfo_.reducedInShape[i], i, shapeInfo_.reducedOutShape[i], i, shapeInfo_.reducedPerm[i]);
    }
    OP_LOGI(context_->GetNodeName(),
            "tilingData is mainBlocksPerCore:%lu, mainBlockElements:%u, coreNum:%u, extraBlockCores:%u, "
            "tailElements:%u, Tiling4Transpose ends. ",
            mainBlocksPerCore_, mainBlockElements_, static_cast<uint32_t>(realCoreNum_), extraBlockCores_,
            tailElements_);
}

} // namespace optiling
