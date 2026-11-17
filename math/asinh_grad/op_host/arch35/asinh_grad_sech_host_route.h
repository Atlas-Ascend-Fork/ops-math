/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 * You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Private target routing and UB budget for the reviewed certified-sech scheme.
#ifndef ASINH_GRAD_SECH_HOST_ROUTE_H
#define ASINH_GRAD_SECH_HOST_ROUTE_H

#include <string>
#include "platform/platform_infos_def.h"

namespace NsAsinhGradSechHostRoute {

// May increase only to compensate a proven final compiler UB allocation delta.
static constexpr uint64_t EXTRA_STATIC_RESERVE = 0;
static constexpr uint64_t OWNED_TABLE_AND_FLAGS = 864;

static inline uint64_t Select(gert::TilingContext* context, ge::DataType dtype, int64_t totalNum,
                              uint64_t originalSelector, uint64_t usableUB, AsinhGradTilingData* tiling)
{
    if (originalSelector != 1U || dtype != ge::DT_FLOAT || totalNum <= 1024) {
        return originalSelector;
    }
    auto* platform = context->GetPlatformInfo();
    std::string fullSoc;
    if (platform == nullptr || !platform->GetPlatformRes("version", "SoC_version", fullSoc) ||
        fullSoc != "Ascend950PR_9579") {
        return originalSelector;
    }
    if (usableUB < EXTRA_STATIC_RESERVE + OWNED_TABLE_AND_FLAGS) {
        return originalSelector;
    }
    const uint64_t available = usableUB - EXTRA_STATIC_RESERVE - OWNED_TABLE_AND_FLAGS;
    uint64_t factor = (available / (3U * 2U * sizeof(float)) / 64U) * 64U;
    if (factor > 10496U) {
        factor = 10496U;
    }
    if (factor < 64U) {
        return originalSelector;
    }
    tiling->ubFactor = static_cast<int64_t>(factor);
    return 2U;
}

} // namespace NsAsinhGradSechHostRoute
#endif
