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
 * \file transpose_tiling_strategy.cpp
 * \brief Transpose Tiling 策略基类实现
 *
 * 实现算子侧策略基类：
 *   - TransposeTilingStrategy：通用上下文注入（3 参构造），不含模板专属逻辑；
 *   - TransposeNddmaGroupStrategy：NDDMA 模板族专属缓冲数推导（仅 NDDMA/NDDMA_BIG_DIM）；
 *   - GetTransposeOpTilingDataBuffer：10000-10005 模板共用的注册 TilingData 取址工具。
 * 收尾（SetTilingData/SetContext/LogTilingData）由通用基类 Execute() 模板方法统一完成。
 */

#include "transpose_tiling_strategy.h"

namespace optiling {

namespace {
static constexpr int64_t NDDMA_GROUP_BUFFER_NUM_DOUBLE_BUF = 2; ///< 双缓冲
static constexpr int64_t NDDMA_GROUP_BUFFER_NUM_SINGLE_BUF = 1; ///< 单缓冲
} // namespace

TransposeTilingStrategy::TransposeTilingStrategy(gert::TilingContext* context, const TransposePlatformInfo& platform,
                                                 const TransposeOpParam& opParam)
    : Base(context, platform, opParam),
      shapeInfo_(opParam_.shapeInfo),
      coreNum_(platform_.coreNum),
      ubSize_(platform_.ubSize),
      cacheLineSize_(platform_.cacheLineSize),
      ubBlockSize_(platform_.ubBlockSize),
      npuArch_(platform_.npuArch),
      isSupportSimt_(platform_.isSupportSimt),
      isRelatedTranspose_(opParam_.isRelatedTranspose)
{}

/**
 * @brief 判断 perm 取值：维数必须等于 perm 长度，且各轴取值逐一相等
 */
bool TransposeTilingStrategy::IsPermValue(std::initializer_list<int64_t> perm) const
{
    if (shapeInfo_.dim != static_cast<int64_t>(perm.size())) {
        return false;
    }
    int64_t axis = 0;
    for (int64_t value : perm) {
        if (shapeInfo_.reducedPerm[axis] != value) {
            return false;
        }
        ++axis;
    }
    return true;
}

TransposeNddmaGroupStrategy::TransposeNddmaGroupStrategy(gert::TilingContext* context,
                                                         const TransposePlatformInfo& platform,
                                                         const TransposeOpParam& opParam)
    : TransposeTilingStrategy(context, platform, opParam)
{}

/**
 * @brief NDDMA 缓冲数推导（仅按平台架构）：DAV_9201/DAV_9202 双缓冲，其他单缓冲
 */
int64_t TransposeNddmaGroupStrategy::GetNddmaBufferNum() const
{
    if (npuArch_ == Ops::Base::DAV_9201 || npuArch_ == Ops::Base::DAV_9202) {
        return NDDMA_GROUP_BUFFER_NUM_DOUBLE_BUF;
    }
    return NDDMA_GROUP_BUFFER_NUM_SINGLE_BUF;
}

/**
 * @brief 取注册 TilingData（10000-10005 模板共用），容量不足时记录错误并返回 nullptr
 */
TransposeOpTilingData* GetTransposeOpTilingDataBuffer(gert::TilingContext* context)
{
    TransposeOpTilingData* tilingData = context->GetTilingData<TransposeOpTilingData>();
    if (tilingData == nullptr) {
        OP_LOGE(context->GetNodeName(), "get tiling data failed, capacity is not enough!");
        return nullptr;
    }
    return tilingData;
}

} // namespace optiling
