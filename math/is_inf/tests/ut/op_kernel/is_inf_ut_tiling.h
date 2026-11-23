/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IS_INF_UT_TILING_H
#define IS_INF_UT_TILING_H

#include "../../../op_host/is_inf_tiling_def.h"

// Kernel-side mirror of the serialized IsInfTilingData fields.
struct IsInfTilingData {
    uint64_t totalDataCount;
    uint32_t usableUbSize;
    uint32_t needCoreNum;
    uint64_t perCoreDataCount;
    uint64_t tailDataCoreNum;
    uint64_t lastCoreDataCount;
};

using TestUtDefaultTilingStruct = IsInfTilingData;

#endif // IS_INF_UT_TILING_H
