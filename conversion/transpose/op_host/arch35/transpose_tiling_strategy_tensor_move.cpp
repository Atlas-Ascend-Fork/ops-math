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
 * 实现 1 维纯搬运路径的 UB 切分、分核、5 维扩展与 TilingData 回填。
 */

#include "transpose_tiling_strategy_tensor_move.h"

namespace optiling {
namespace {
static constexpr uint64_t TENSOR_MOVE_BUFFER_NUM = 2; ///< 双缓冲
} // namespace

bool TransposeTensorMoveStrategy::IsSupport() const { return shapeInfo_.dim == 1; }

TilingStatus TransposeTensorMoveStrategy::DoExecute()
{
    tilingKey_ = TK_TENSOR_MOVE;
    CalcUBSplitInfo();
    CalcBlockSplitInfo();
    NDDMADimExpand();
    return TilingStatus::SUCCESS;
}

void TransposeTensorMoveStrategy::CalcUBSplitInfo()
{
    ubElement_ = ubSize_ / TENSOR_MOVE_BUFFER_NUM / shapeInfo_.eleLenInBytes;
}

void TransposeTensorMoveStrategy::CalcBlockSplitInfo()
{
    OP_LOGD(context_->GetNodeName(), "Entering CalcBlockSplitInfoForTensorMove.");
    if (shapeInfo_.totalVolumeActual < coreNum_) {
        realCoreNum_ = 1;
        blkFactor_ = shapeInfo_.totalVolumeActual;
        blkTailFactor_ = 0;
        inUbFactor_ = ubElement_;
    } else {
        realCoreNum_ = coreNum_;
        blkFactor_ = shapeInfo_.totalVolumeActual / coreNum_;
        blkTailFactor_ = shapeInfo_.totalVolumeActual % coreNum_;
        inUbFactor_ = ubElement_;
    }
}

/**
 * @brief 5 维扩展（TENSOR_MOVE dim==1，低维左侧补 1）
 */
void TransposeTensorMoveStrategy::NDDMADimExpand()
{
    int64_t offset = (shapeInfo_.dim < TP_NDDMA_MAX_DIM_NUM) ? (TP_NDDMA_MAX_DIM_NUM - shapeInfo_.dim) : 0;
    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        expandedPerm_[i + offset] = shapeInfo_.reducedPerm[i] + offset;
        expandedInputShape_[i + offset] = shapeInfo_.reducedInShape[i];
        expandedOutputShape_[i + offset] = shapeInfo_.reducedOutShape[i];
    }
}

/**
 * @brief 填写 TENSOR_MOVE 消费的注册 TilingData 字段
 *
 * kernel（transpose_tensor_move.h）消费：ubSize / realCoreNum / blkFactor /
 * blkTailFactor / inUbFactor + 通用 shape/perm（permSize）。
 */
bool TransposeTensorMoveStrategy::SetTilingData()
{
    OP_LOGD(context_->GetNodeName(), "Entering SetTilingData.");
    TransposeOpTilingData* tilingData = GetTransposeOpTilingDataBuffer(context_);
    if (tilingData == nullptr) {
        return false;
    }
    *tilingData = TransposeOpTilingData{};
    tilingData->permSize = shapeInfo_.dim;
    tilingData->realCoreNum = realCoreNum_;
    tilingData->blkFactor = blkFactor_;
    tilingData->blkTailFactor = blkTailFactor_;
    tilingData->ubSize = ubSize_;
    tilingData->inUbFactor = inUbFactor_;
    tilingData->bigDimBufferNum = 1;
    tilingData->totalNddmaNum = 1;
    tilingData->nddmaIdx[0] = -1;
    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        tilingData->inputShape[i] = shapeInfo_.reducedInShape[i];
        tilingData->outputShape[i] = shapeInfo_.reducedOutShape[i];
        tilingData->perm[i] = shapeInfo_.reducedPerm[i];
    }
    std::copy(expandedPerm_, expandedPerm_ + TP_NDDMA_MAX_DIM_NUM, tilingData->expandedPerm);
    std::copy(expandedInputShape_, expandedInputShape_ + TP_NDDMA_MAX_DIM_NUM, tilingData->expandedInputShape);
    std::copy(expandedOutputShape_, expandedOutputShape_ + TP_NDDMA_MAX_DIM_NUM, tilingData->expandedOutputShape);
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
            "tilingData is permSize:%ld, realCoreNum:%ld, blkFactor:%ld, blkTailFactor:%ld, ubSize:%ld, "
            "inUbFactor:%ld, Tiling4Transpose ends. ",
            shapeInfo_.dim, realCoreNum_, blkFactor_, blkTailFactor_, ubSize_, inUbFactor_);
}

} // namespace optiling
