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
 * \file atan2_tiling.cpp
 * \brief Adapt the Atan2 buffer-aware schedule to the graph tiling context.
 */

#include "atan2_tiling_schedule.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/atan2_tiling_key.h"

namespace optiling {

struct Atan2CompileInfo {};

static ge::graphStatus Atan2TilingFunc(gert::TilingContext* context)
{
    auto* input = context->GetInputShape(0);
    auto* descriptor = context->GetInputDesc(0);
    auto* platformInfo = context->GetPlatformInfo();
    auto* data = context->GetTilingData<Atan2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, input);
    OP_CHECK_NULL_WITH_CONTEXT(context, descriptor);
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    OP_CHECK_NULL_WITH_CONTEXT(context, data);

    uint32_t elementBytes = 0;
    switch (descriptor->GetDataType()) {
        case ge::DT_FLOAT:
            elementBytes = sizeof(float);
            break;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            elementBytes = sizeof(uint16_t);
            break;
        default:
            OP_LOGE(context, "Atan2 requires FLOAT, FLOAT16 or BF16 input");
            return ge::GRAPH_FAILED;
    }

    // One platform object supplies all resources, including library workspace.
    platform_ascendc::PlatformAscendC platform(platformInfo);
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const uint32_t availableCores = platform.GetCoreNumAiv();
    if (availableCores == 0 || ubBytes == 0) {
        OP_LOGE(context, "Atan2 requires nonzero AIV cores and UB capacity");
        return ge::GRAPH_FAILED;
    }

    *data = Atan2TilingData{};
    const int64_t elements = input->GetStorageShape().GetShapeSize();
    uint32_t activeCores = 1;
    if (elements > 0) {
        if (!atan2::BuildSchedule(static_cast<uint64_t>(elements), elementBytes, ubBytes, availableCores, *data,
                                  activeCores)) {
            OP_LOGE(context, "Atan2 shape or UB capacity cannot fit the kernel schedule");
            return ge::GRAPH_FAILED;
        }
        auto* workspace = context->GetWorkspaceSizes(1);
        OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
        workspace[0] = static_cast<size_t>(platform.GetLibApiWorkSpaceSize()) + 512U;
    }
    context->SetBlockDim(activeCores);
    context->SetTilingKey(GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAtan2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Atan2).Tiling(Atan2TilingFunc).TilingParse<Atan2CompileInfo>(TilingParseForAtan2);
} // namespace optiling
