/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <cstddef>
#include <cstdint>

struct alignas(8) SinhTilingData {
    uint64_t totalLength; // Valid element count N; scalar 1, empty 0.
    uint64_t blockLength; // Per-core capacity L, in elements.
    uint64_t tileLength;  // Per-tile UB capacity T, in elements.
};

static_assert(sizeof(SinhTilingData) == 24);
static_assert(alignof(SinhTilingData) == 8);
static_assert(offsetof(SinhTilingData, totalLength) == 0);
static_assert(offsetof(SinhTilingData, blockLength) == 8);
static_assert(offsetof(SinhTilingData, tileLength) == 16);
