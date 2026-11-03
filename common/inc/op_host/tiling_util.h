/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tiling_util.h
 * \brief
 */

#pragma once

#include "register/op_impl_registry.h"
#include "platform/platform_ascendc.h"
#include "platform/soc_spec.h"
#include "op_common/op_host/util/op_const_def.h"

#include <set>

namespace Ops {
namespace Math {
namespace OpTiling {

static bool IsSimtSupportedNpuArch(NpuArch npuArch)
{
    const static std::set<NpuArch> simtSupportedNpuArchs = {Ops::Base::DAV_3510, Ops::Base::DAV_5102,
                                                            Ops::Base::DAV_9201, Ops::Base::DAV_9202};
    return simtSupportedNpuArchs.find(npuArch) != simtSupportedNpuArchs.end();
}

static inline bool IsSimtSupported(const gert::TilingParseContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto npuArch = ascendcPlatform.GetCurNpuArch();
    return IsSimtSupportedNpuArch(npuArch);
}

static inline bool IsSimtSupported(const gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto npuArch = ascendcPlatform.GetCurNpuArch();
    return IsSimtSupportedNpuArch(npuArch);
}
} // namespace OpTiling
} // namespace Math
} // namespace Ops
