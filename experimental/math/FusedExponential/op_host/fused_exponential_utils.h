/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef FUSED_EXPONENTIAL_UTILS_H
#define FUSED_EXPONENTIAL_UTILS_H

#include <cstdint>
#include <cstddef>
#include <limits>

namespace fused_exponential {
constexpr size_t MAX_RANK = 8;
// Byte counts must also fit signed framework shape/offset arithmetic.
template <typename Shape>
bool GetElementCount(const Shape& shape, uint64_t elementBytes, uint64_t& count)
{
    if (shape.GetDimNum() > MAX_RANK || elementBytes == 0) {
        return false;
    }
    bool empty = false;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) < 0) {
            return false;
        }
        empty = empty || shape.GetDim(i) == 0;
    }
    count = empty ? 0 : 1;
    if (empty) {
        return true;
    }
    const uint64_t limit = static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / elementBytes;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const auto dim = static_cast<uint64_t>(shape.GetDim(i));
        if (count > limit / dim) {
            return false;
        }
        count *= dim;
    }
    return true;
}
inline uint64_t CeilDiv(uint64_t value, uint64_t divisor) { return value / divisor + (value % divisor != 0); }
} // namespace fused_exponential
#endif
