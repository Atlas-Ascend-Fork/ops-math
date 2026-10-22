/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <limits>
#include "log/log.h"
#include "fused_exponential_utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/fused_exponential_tiling_data.h"

namespace optiling {
// Scheduling policy, in elements; 64 elements aligns both 2- and 4-byte types.
constexpr uint64_t ELEMENTS_PER_CORE_TARGET = 1024;
constexpr uint64_t ELEMENT_ALIGNMENT = 64;
constexpr uint64_t MAX_TILE_ELEMENTS = 4096;
constexpr uint64_t RESERVED_UB_BYTES = 8192;
// Two T queues plus four FP32 buffers require at most 24 bytes/element.
// Budget 32 bytes/element conservatively; this is not the hardware UB block size.
constexpr uint64_t UB_BYTES_PER_ELEMENT = 32;
constexpr uint64_t MIN_UB_BYTES = 2 * RESERVED_UB_BYTES;
constexpr uint64_t DEFAULT_TILING_KEY = 0; // dtype is specialized by DTYPE_U.
// frexp of any positive finite DOUBLE produces power in [-1073, 1024].
// The API passes exponent = -power for reciprocal scaling, hence [-1024, 1073].
constexpr int64_t MIN_SCALE_EXPONENT = -1024;
constexpr int64_t MAX_SCALE_EXPONENT = 1073;
static ge::graphStatus Tiling(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("FusedExponential", "Tiling context is null"), return ge::GRAPH_FAILED);
    auto shape = context->GetInputShape(0);
    auto desc = context->GetInputDesc(0);
    auto attrs = context->GetAttrs();
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, desc);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    const uint32_t maxCores = platform.GetCoreNumAiv();
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    OP_CHECK_IF(maxCores == 0 || ubBytes < MIN_UB_BYTES, OP_LOGE(context, "Insufficient AIV cores or UB capacity"),
                return ge::GRAPH_FAILED);
    auto data = context->GetTilingData<FusedExponentialTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, data);
    const auto scale = attrs->GetFloat(0);
    const auto exponent = attrs->GetInt(1);
    const auto zero = attrs->GetBool(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale);
    OP_CHECK_NULL_WITH_CONTEXT(context, exponent);
    OP_CHECK_NULL_WITH_CONTEXT(context, zero);
    OP_CHECK_IF(*exponent < MIN_SCALE_EXPONENT || *exponent > MAX_SCALE_EXPONENT,
                OP_LOGE(context, "Scale exponent is outside the DOUBLE decomposition range"), return ge::GRAPH_FAILED);
    const auto dtype = desc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16,
                OP_LOGE(context, "Unsupported input dtype"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !fused_exponential::GetElementCount(shape->GetStorageShape(), dtype == ge::DT_FLOAT ? 4 : 2, data->length),
        OP_LOGE(context, "Invalid shape or tensor byte size overflow"), return ge::GRAPH_FAILED);
    const uint64_t chunks = fused_exponential::CeilDiv(data->length, ELEMENTS_PER_CORE_TARGET);
    const uint32_t cores = static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(chunks, maxCores)));
    data->coreLength = fused_exponential::CeilDiv(data->length, cores * ELEMENT_ALIGNMENT) * ELEMENT_ALIGNMENT;
    data->tileLength = static_cast<uint32_t>(
        std::min(MAX_TILE_ELEMENTS,
                 (ubBytes - RESERVED_UB_BYTES) / UB_BYTES_PER_ELEMENT / ELEMENT_ALIGNMENT * ELEMENT_ALIGNMENT));
    data->epsilon = dtype == ge::DT_FLOAT16 ? 0x1p-11f : (dtype == ge::DT_BF16 ? 0x1p-8f : 0x1p-24f);
    data->scale = *scale;
    data->exponent = static_cast<int32_t>(*exponent);
    data->zero = *zero ? 1 : 0;
    context->SetBlockDim(cores);
    context->SetTilingKey(DEFAULT_TILING_KEY);
    auto workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
struct FusedExponentialCompileInfo {};
static ge::graphStatus TilingParse(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(FusedExponential).Tiling(Tiling).TilingParse<FusedExponentialCompileInfo>(TilingParse);
} // namespace optiling
