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
 * \file pad_v2_tiling_arch35.cpp
 * \brief PadV2 tiling - 复用 PadV3 实现
 */
#include "pad_v2_tiling_arch35.h"
#include "conversion/pad_v3/op_host/arch35/pad_v3_tiling_arch35.h"
#include "log/log.h"

namespace optiling {

static ge::graphStatus Tiling4PadV2(gert::TilingContext* context)
{
    // paddings 为 ValueDepend 输入，数据可能不下发；在进入共享 Tiling 前拦截空数据，
    // 避免下游解引用（CODE.02/1；共享实现归属 pad_v3 责任田，此处只防护 PadV2 通路）
    auto paddingsTensor = context->GetInputTensor(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, paddingsTensor);
    OP_CHECK_IF(paddingsTensor->GetShapeSize() > 0 && paddingsTensor->GetAddr() == nullptr,
                OP_LOGE(context, "paddings const data cannot be nullptr"), return ge::GRAPH_FAILED);

    // PadV2 复用 PadV3 的 Tiling 类，PadV3 侧 constant_values 校验被 isPadV3_ 门控，
    // 这里补齐 PadV2 的入口契约校验（README 约束#2 / proto attention / 基线 TBE 行为）：
    // 1) constant_values 与 x 同 dtype（基线 TBE 显式拒绝不匹配）
    auto xDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    auto cvDesc = context->GetInputDesc(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, cvDesc);
    OP_CHECK_IF(xDesc->GetDataType() != cvDesc->GetDataType(),
                OP_LOGE(context, "The dtype of constant_values must be the same as the dtype of input."),
                return ge::GRAPH_FAILED);
    // 2) constant_values 必须为标量（0 维或元素数为 1；空 tensor 会被 kernel 读元素 0 越界）
    auto cvTensor = context->GetInputTensor(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, cvTensor);
    OP_CHECK_IF(
        cvTensor->GetShapeSize() != 1,
        OP_LOGE(context, "constant_values must be a scalar tensor, got shape size %ld", cvTensor->GetShapeSize()),
        return ge::GRAPH_FAILED);

    // 直接使用 PadV3 的 Tiling 类
    PadACTiling tilingObject(context);
    return tilingObject.DoTiling();
}

static ge::graphStatus TilingPrepare4PadV2(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4PadV2 entering.");
    return ge::GRAPH_SUCCESS;
}

// 注册 PadV2 的 tiling 接口
IMPL_OP_OPTILING(PadV2).Tiling(Tiling4PadV2).TilingParse<PadV3CompileInfo>(TilingPrepare4PadV2);

} // namespace optiling
