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
 * \file transpose_tiling_strategy_small_shape.cpp
 * \brief SMALL_SHAPE(10001) 策略类实现
 *
 * 实现 SIMT 小 shape 路径的阈值推导、128B 对齐分核与 TilingData 回填。
 */

#include "transpose_tiling_strategy_small_shape.h"

namespace optiling {
namespace {
// SIMT 小 shape 阈值默认值（4MB）
static constexpr int64_t TRANSPOSE_SMALL_SHAPE_BYTES_THRES_HOLD = 4000000;
static constexpr int64_t SMALL_SHAPE_SPLIT_BYTES_ALIGN_SIZE = 128;        ///< 128 字节对齐
static constexpr int64_t SMALL_SHAPE_BYTES_THRES_HOLD_DAV_5102 = 1000000; ///< DAV_5102 阈值：1MB
} // namespace

/**
 * @brief 准入检查：SIMT 平台 && 体积 < SIMT 小 shape 阈值（纯判定、无副作用）
 *
 * 阈值推导：
 *   - !isRelatedTranspose && arch==DAV_5102 → 1MB；
 *   - 其他 → 4MB。
 */
bool TransposeSmallShapeStrategy::IsSupport() const
{
    if (!isSupportSimt_) {
        return false;
    }
    const int64_t volumeBytes = shapeInfo_.totalVolumeActual * shapeInfo_.eleLenInBytes;
    int64_t threshold = TRANSPOSE_SMALL_SHAPE_BYTES_THRES_HOLD;
    if (!isRelatedTranspose_ && npuArch_ == Ops::Base::DAV_5102) {
        threshold = SMALL_SHAPE_BYTES_THRES_HOLD_DAV_5102;
    }
    return volumeBytes < threshold;
}

TilingStatus TransposeSmallShapeStrategy::DoExecute()
{
    tilingKey_ = TK_SMALL_SHAPE;
    CalcBlockSplitInfo();
    if (shapeInfo_.dim <= TP_NDDMA_MAX_DIM_NUM) {
        NDDMADimExpand();
        CalcInUbShapeInfoForNoNeedCut();
    }
    return TilingStatus::SUCCESS;
}

void TransposeSmallShapeStrategy::CalcBlockSplitInfo()
{
    OP_LOGD(context_->GetNodeName(), "Entering CalcBlockSplitInfoForSmallShape.");
    int64_t totalElements = shapeInfo_.totalVolumeActual;
    int64_t alignElements = SMALL_SHAPE_SPLIT_BYTES_ALIGN_SIZE / shapeInfo_.eleLenInBytes;
    if (totalElements < alignElements || totalElements < coreNum_) {
        realCoreNum_ = 1;
        blkFactor_ = totalElements;
        blkTailFactor_ = 0;
        return;
    }
    // 先向上均分再对齐，确保可用核数能够覆盖全部元素。
    int64_t blkFactor = Ops::Base::CeilDiv(totalElements, coreNum_);
    int64_t ceilAlignFactor = Ops::Base::CeilAlign(blkFactor, alignElements);
    realCoreNum_ = Ops::Base::CeilDiv(totalElements, ceilAlignFactor);
    blkFactor_ = ceilAlignFactor;
    blkTailFactor_ = ceilAlignFactor == 0 ? 0 : totalElements % ceilAlignFactor;
}

void TransposeSmallShapeStrategy::NDDMADimExpand()
{
    int64_t offset = (shapeInfo_.dim < TP_NDDMA_MAX_DIM_NUM) ? (TP_NDDMA_MAX_DIM_NUM - shapeInfo_.dim) : 0;
    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        expandedPerm_[i + offset] = shapeInfo_.reducedPerm[i] + offset;
        expandedInputShape_[i + offset] = shapeInfo_.reducedInShape[i];
        expandedOutputShape_[i + offset] = shapeInfo_.reducedOutShape[i];
    }
}

/**
 * @brief 计算"无需切分"场景的 UB 内 shape（SMALL_SHAPE 直读直写路径）
 *
 * 只有一个切分轴（outCutIndex）需要以 outUbFactor/outTailFactor 切分，
 * 其余进入 block 的轴要么取完整输出 shape、要么置 1（外层循环负责）。
 */
void TransposeSmallShapeStrategy::CalcInUbShapeInfoForNoNeedCut()
{
    int64_t outCutIndexExpand = outCutIndex_ + TP_NDDMA_MAX_DIM_NUM - shapeInfo_.dim;
    for (int64_t i = 0; i < TP_NDDMA_MAX_DIM_NUM; i++) {
        if (i > outCutIndexExpand) {
            inUbMainDstShape_[i] = expandedOutputShape_[i];
            inUbTailDstShape_[i] = expandedOutputShape_[i];
        } else if (i == outCutIndexExpand) {
            inUbMainDstShape_[i] = outUbFactor_;
            inUbTailDstShape_[i] = outTailFactor_;
        } else {
            inUbMainDstShape_[i] = 1;
            inUbTailDstShape_[i] = 1;
        }
    }
    for (int64_t i = 0; i < TP_NDDMA_MAX_DIM_NUM; i++) {
        inUbMainSrcShape_[expandedPerm_[i]] = inUbMainDstShape_[i];
        inUbTailSrcShape_[expandedPerm_[i]] = inUbTailDstShape_[i];
    }
}

/**
 * @brief 填写 SMALL_SHAPE 消费的注册 TilingData 字段
 *
 * kernel（transpose_small_shape.h）消费：permSize / realCoreNum / blkFactor /
 * blkTailFactor + inputShape/outputShape/perm。
 */
bool TransposeSmallShapeStrategy::SetTilingData()
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
    std::copy(inUbMainSrcShape_, inUbMainSrcShape_ + TP_NDDMA_MAX_DIM_NUM, tilingData->inUbMainSrcShape);
    std::copy(inUbMainDstShape_, inUbMainDstShape_ + TP_NDDMA_MAX_DIM_NUM, tilingData->inUbMainDstShape);
    std::copy(inUbTailSrcShape_, inUbTailSrcShape_ + TP_NDDMA_MAX_DIM_NUM, tilingData->inUbTailSrcShape);
    std::copy(inUbTailDstShape_, inUbTailDstShape_ + TP_NDDMA_MAX_DIM_NUM, tilingData->inUbTailDstShape);
    return true;
}

void TransposeSmallShapeStrategy::LogTilingData()
{
    OP_LOGI(context_->GetNodeName(), "Entering LogTilingData.");
    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        OP_LOGI(context_->GetNodeName(), "reducedInShape[%ld] is:%ld, reducedOutShape[%ld]:%ld, reducedPerm[%ld]:%ld.",
                i, shapeInfo_.reducedInShape[i], i, shapeInfo_.reducedOutShape[i], i, shapeInfo_.reducedPerm[i]);
    }
    OP_LOGI(context_->GetNodeName(),
            "tilingData is permSize:%ld, realCoreNum:%ld, blkFactor:%ld, blkTailFactor:%ld, Tiling4Transpose ends. ",
            shapeInfo_.dim, realCoreNum_, blkFactor_, blkTailFactor_);
}

} // namespace optiling
