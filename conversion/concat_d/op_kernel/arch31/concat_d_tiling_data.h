/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONCAT_D_ARCH31_TILING_DATA_H_
#define CONCAT_D_ARCH31_TILING_DATA_H_

#include <cstdint>

namespace concat_d_arch31 {
constexpr uint32_t TILING_VERSION = 1U;
constexpr uint32_t MAX_INPUTS = 32U;
constexpr uint32_t FLAG_COMPACT_SCATTER = 1U << 1U;
constexpr uint32_t COMPACT_SCATTER_VECTOR_ELEMS = 128U;
constexpr uint32_t COMPACT_SCATTER_REPEATS = 4U;
constexpr uint32_t COMPACT_SCATTER_TABLE_ELEMS = 512U;
constexpr uint32_t COMPACT_SCATTER_TABLE_BYTES = COMPACT_SCATTER_TABLE_ELEMS * sizeof(uint32_t);
constexpr uint32_t COMPACT_SCATTER_UB_BYTES = 112U * 1024U;
} // namespace concat_d_arch31

#define CONCAT_D_ARCH31_TILING_SCALAR_FIELDS(FIELD) \
    FIELD(uint32_t, version)                        \
    FIELD(uint32_t, flags)                          \
    FIELD(uint32_t, tensorNum)                      \
    FIELD(uint32_t, tileBytes)                      \
    FIELD(uint64_t, outerCount)                     \
    FIELD(uint64_t, outputRowBytes)

namespace concat_d_arch31 {
#define CONCAT_D_ARCH31_POD_FIELD(type, name) type name;
struct ConcatDTilingDataPod {
    CONCAT_D_ARCH31_TILING_SCALAR_FIELDS(CONCAT_D_ARCH31_POD_FIELD)
    uint64_t inputRowBytes[MAX_INPUTS];
};
#undef CONCAT_D_ARCH31_POD_FIELD

static_assert(sizeof(ConcatDTilingDataPod) == 288U, "ConcatD Arch31 tiling ABI changed");
} // namespace concat_d_arch31

#endif // CONCAT_D_ARCH31_TILING_DATA_H_
