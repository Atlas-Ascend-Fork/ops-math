/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SELECT_TOP_K_TILING_DATA_H_
#define SELECT_TOP_K_TILING_DATA_H_
#include "select_top_k_config.h"
#include "select_top_k_sort_schedule.h"
#include "select_top_k_base_tiling_data.h"

namespace SelectTopK {
enum Flags : uint32_t {
    HAS_BEGIN = 1,
    HAS_END = 2,
    BROADCAST_BEGIN = 4,
    BROADCAST_END = 8,
    RANGE = 16,
    TEMP_VALUES = 32,
    TEMP_INDICES = 64,
    WRITE_VALUES = 128,
    WRITE_INDICES = 256,
    FINALIZE = 512,
    DIRECT_INDICES64 = 1024,
    REUSE_LAYOUT = 2048
};
}

struct SelectTopKTilingData {
    SelectTopKBaseTilingData base;
    uint64_t outputK = 0;
    int64_t indicesOffset = 0;
    int64_t indexFill = -1;
    uint64_t valuesWorkspaceOffset = 0;
    uint64_t indicesWorkspaceOffset = 0;
    uint64_t workspaceBytes = 0;
    float valueFill = 0;
    uint32_t flags = 0;
    uint32_t outputInner = 1; // 将 [outer, inner, K] 恢复为 [outer, K, inner]。
    uint32_t outputBlock = 0;
    uint32_t inputInner = 1;
    uint32_t inputBlock = 0;
    uint64_t inputWorkspaceOffset = 0;
    SelectTopK::SortSchedule schedule;
};
static_assert(sizeof(SelectTopKTilingData) == SelectTopK::TILING_DATA_BYTES, "SelectTopK tiling ABI changed");
#endif
