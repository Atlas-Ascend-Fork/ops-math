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
 * \file hypot_tiling_arch22.h
 * \brief hypot arch22 tiling: broadcast fusion, scene decision, UB/multi-core split
 */

#ifndef HYPOT_TILING_ARCH22_H
#define HYPOT_TILING_ARCH22_H

#include "../../op_kernel/hypot_tiling_data.h"

namespace optiling {
struct HypotCompileInfoArch22 {
    int32_t totalCoreNum = 0;
    int64_t ubSize = 0;
};
} // namespace optiling
#endif // HYPOT_TILING_ARCH22_H
