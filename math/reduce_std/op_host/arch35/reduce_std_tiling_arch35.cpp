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
 * \file reduce_std_tiling_arch35.cpp
 * \brief
 */

#include "math/reduce_var/op_host/arch35/reduce_var_tiling.h"
#include "log/log.h"
#include "math/reduce_std/op_kernel/arch35/reduce_std_tiling_key.h"

namespace optiling {
constexpr size_t INPUT_INDEX_X = 0;
constexpr size_t INDEX_ATTR_UNBIASED = 1;

static ge::graphStatus Tiling4ReduceStd(gert::TilingContext* context)
{
    auto compileInfo = reinterpret_cast<const ReduceVarCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    auto xDesc = context->GetInputDesc(INPUT_INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const ge::DataType xDtype = xDesc->GetDataType();
    OP_CHECK_IF(
        (xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_FLOAT),
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x", Ops::Base::ToString(xDtype), "DT_FLOAT16 or DT_FLOAT"),
        return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* attrUnbiased = attrs->GetAttrPointer<bool>(INDEX_ATTR_UNBIASED);
    const int64_t correction = (attrUnbiased == nullptr || *attrUnbiased) ? 1 : 0;

    Ops::Base::ReduceTilingKey key;
    auto* tilingData = context->GetTilingData<ReduceVarTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    *tilingData = ReduceVarTilingData{};
    ReduceVarTiling tiling(context, compileInfo, tilingData, correction, true);
    OP_CHECK_IF((tiling.RunTiling(key) != ge::GRAPH_SUCCESS),
                OP_LOGE(context->GetNodeName(), "RunTiling Failed for ReduceStd"), return ge::GRAPH_FAILED);
    uint64_t tilingKey;
    GEN_REDUCE_STD_TILING_KEY(tilingKey, key);
    OP_LOGI(context->GetNodeName(),
            "correction:%ld patternID:%u, loopARCount:%u, loopInnerARCount:%u, "
            "Tiling Key is:%lu",
            correction, key.patternID, key.loopARCount, key.loopInnerARCount, tilingKey);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4ReduceStd(gert::TilingParseContext* context)
{
    OP_CHECK_IF((context == nullptr), OP_LOGE("ReduceStd", "context is null"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ReduceStd).Tiling(Tiling4ReduceStd).TilingParse<ReduceVarCompileInfo>(TilingPrepare4ReduceStd);

} // namespace optiling
