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
 * \brief SMALL_SHAPE(10001)/SMALL_SHAPE_23D(10011) 策略类实现
 *
 * 2D/3D 统一走 10011 的 Host 分核精简 TilingData，4-8D 走 10001 的 Host 分核精简 TilingData。
 * 分核按 1024B 对齐公式（Host 侧一次算好），kernel 侧不再自行分核；
 * blockDim 精确等于 Host 的 realCoreNum，避免空跑；realCoreNum 不再写入精简码流，
 * kernel 侧通过 GetBlockNum() 获取实际使用核数。
 */

#include "transpose_tiling_strategy_small_shape.h"

namespace optiling {
namespace {
// SIMT 小 shape 阈值默认值（4MB）
static constexpr int64_t TRANSPOSE_SMALL_SHAPE_BYTES_THRES_HOLD = 4000000;
static constexpr int64_t SMALL_SHAPE_SPLIT_BYTES_ALIGN_SIZE = 1024;       ///< 1024 字节对齐
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

/**
 * @brief 小 shape 准入后统一填充：一次算好分核、简化 shape/perm 与 tilingKey。
 *
 * 2D/3D → 10011（24B 的 <3> 实例）；4-8D → 10001（<TP_MAX_AXIS_NUM> 实例，perm 未用槽位 -1）。
 * 普通/关联入口统一走本实现。
 */
TilingStatus TransposeSmallShapeStrategy::DoExecute()
{
    OP_LOGD(context_->GetNodeName(), "Entering DoExecute.");
    const int64_t dim = shapeInfo_.dim;
    // Host 侧统一按 SMALL_SHAPE 1024B 对齐公式分核
    CalcBlockSplitInfo();
    dim_ = static_cast<uint8_t>(dim);
    for (int64_t i = 0; i < TP_MAX_AXIS_NUM; i++) {
        inputShape_[i] = 0;
        perm_[i] = -1;
    }
    for (int64_t i = 0; i < dim; i++) {
        inputShape_[i] = static_cast<int32_t>(shapeInfo_.reducedInShape[i]);
        perm_[i] = static_cast<int8_t>(shapeInfo_.reducedPerm[i]);
    }
    // 2/3D 走 24B 的 <3> 实例，4-8D 走 <TP_MAX_AXIS_NUM> 实例
    tilingKey_ = (dim <= 3) ? TK_SMALL_SHAPE_23D : TK_SMALL_SHAPE;
    return TilingStatus::SUCCESS;
}

/**
 * @brief 小 shape 分核（1024B 对齐）
 *
 * 先按核数向上均分，再按 1024B 对齐上取整得到每核主块元素数，最后用主块元素数反推
 * 实际核数，保证 realCoreNum * blkFactor >= totalElements：
 *   blkFactor      = CeilDiv(totalElements, coreNum)
 *   ceilAlignFactor= CeilAlign(blkFactor, 1024 / eleLen)
 *   realCoreNum    = CeilDiv(totalElements, ceilAlignFactor)
 *   blkFactor_     = ceilAlignFactor
 *   blkTailFactor_ = totalElements % ceilAlignFactor
 * 该公式与原 Host 逻辑逐字一致，kernel 只消费 blkFactor/blkTailFactor；
 * 实际核数通过 blockDim 下发，kernel 侧用 GetBlockNum() 获取。
 */
void TransposeSmallShapeStrategy::CalcBlockSplitInfo()
{
    OP_LOGD(context_->GetNodeName(), "Entering CalcBlockSplitInfoForSmallShape.");
    const int64_t alignElements = SMALL_SHAPE_SPLIT_BYTES_ALIGN_SIZE / shapeInfo_.eleLenInBytes;
    const int64_t totalElements = shapeInfo_.totalVolumeActual;
    if (totalElements < alignElements || totalElements < coreNum_) {
        realCoreNum_ = 1;
        blkFactor_ = totalElements;
        blkTailFactor_ = 0;
        return;
    }
    const int64_t blkFactor = Ops::Base::CeilDiv(totalElements, coreNum_);
    const int64_t ceilAlignFactor = Ops::Base::CeilAlign(blkFactor, alignElements);
    realCoreNum_ = Ops::Base::CeilDiv(totalElements, ceilAlignFactor);
    blkFactor_ = ceilAlignFactor;
    blkTailFactor_ = totalElements % ceilAlignFactor;
}

/**
 * @brief 按 TilingDataT 写入精简码流（23D/48D 共用同一套代码）
 *
 * TilingDataT::AXIS_NUM 为实例容量（<3> 或 <TP_MAX_AXIS_NUM>），写入全部槽位。
 */
template <typename TilingDataT>
bool TransposeSmallShapeStrategy::SetTilingDataImpl()
{
    auto* data = context_->GetTilingData<TilingDataT>();
    if (data == nullptr) {
        OP_LOGE(context_->GetNodeName(), "get tiling data failed, capacity is not enough!");
        return false;
    }
    data->blkFactor = static_cast<uint32_t>(blkFactor_);
    data->blkTailFactor = static_cast<uint32_t>(blkTailFactor_);
    data->dim = dim_;
    for (int64_t i = 0; i < TilingDataT::AXIS_NUM; i++) {
        data->inputShape[i] = inputShape_[i];
        data->perm[i] = perm_[i];
    }
    return true;
}

/**
 * @brief 按 tilingKey_ 选择 23D/48D 的 TilingData 类型，复用同一套写入逻辑
 */
bool TransposeSmallShapeStrategy::SetTilingData()
{
    OP_LOGD(context_->GetNodeName(), "Entering SetTilingData.");
    if (tilingKey_ == TK_SMALL_SHAPE_23D) {
        return SetTilingDataImpl<TransposeSmallShape23DTilingData>();
    }
    return SetTilingDataImpl<TransposeSmallShape48DTilingData>();
}

void TransposeSmallShapeStrategy::LogTilingData()
{
    OP_LOGI(context_->GetNodeName(), "Entering LogTilingData.");
    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        OP_LOGI(context_->GetNodeName(), "reducedInShape[%ld] is:%ld, reducedOutShape[%ld]:%ld, reducedPerm[%ld]:%ld.",
                i, shapeInfo_.reducedInShape[i], i, shapeInfo_.reducedOutShape[i], i, shapeInfo_.reducedPerm[i]);
    }
    OP_LOGI(context_->GetNodeName(),
            "Small tilingData is blkFactor:%ld, blkTailFactor:%ld, dim:%u, Tiling4Transpose ends.", blkFactor_,
            blkTailFactor_, static_cast<unsigned>(dim_));
}

} // namespace optiling
