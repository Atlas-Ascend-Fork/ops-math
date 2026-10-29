/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_BASE_ARCH31_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_BASE_ARCH31_H

#include <array>
#include <cstdint>

#include "transpose_tiling_arch31.h"

namespace optiling {
struct KirinTransposeShapeInfo {
    uint32_t rank = 0;
    uint32_t elementBytes = 0;
    ge::DataType inputDtype = ge::DT_UNDEFINED;
    uint64_t totalElements = 0;
    std::array<uint64_t, TRANSPOSE_MAX_AXIS_NUM> inputShape{};
    std::array<uint64_t, TRANSPOSE_MAX_AXIS_NUM> outputShape{};
    std::array<int64_t, TRANSPOSE_MAX_AXIS_NUM> perm{};
};

struct KirinReducedShapeInfo {
    uint32_t rank = 0;
    std::array<uint64_t, TRANSPOSE_MAX_AXIS_NUM> outShape{};
    std::array<uint64_t, TRANSPOSE_MAX_AXIS_NUM> srcStride{};
    std::array<uint64_t, TRANSPOSE_MAX_AXIS_NUM> outStride{};
};

bool BuildKirinReducedShape(const KirinTransposeShapeInfo& input, KirinReducedShapeInfo& output);
} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_BASE_ARCH31_H
