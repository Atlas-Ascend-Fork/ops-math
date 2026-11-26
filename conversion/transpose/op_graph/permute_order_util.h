/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_MATH_CONVERSION_TRANSPOSE_PERMUTE_ORDER_UTIL_H_
#define OPS_MATH_CONVERSION_TRANSPOSE_PERMUTE_ORDER_UTIL_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ops {
inline bool NormalizePermuteOrder(const std::vector<int64_t>& order, size_t rank, std::vector<int64_t>& perm)
{
    if (order.size() > rank) {
        return false;
    }
    std::vector<bool> used(rank, false);
    perm.clear();
    perm.reserve(rank);
    const int64_t rankValue = static_cast<int64_t>(rank);
    for (const int64_t value : order) {
        const int64_t normalized = value < 0 ? value + rankValue : value;
        if (normalized < 0 || normalized >= rankValue || used[static_cast<size_t>(normalized)]) {
            return false;
        }
        used[static_cast<size_t>(normalized)] = true;
        perm.push_back(normalized);
    }
    for (size_t i = 0; i < rank; ++i) {
        if (!used[i]) {
            perm.push_back(static_cast<int64_t>(i));
        }
    }
    return true;
}
} // namespace ops

#endif // OPS_MATH_CONVERSION_TRANSPOSE_PERMUTE_ORDER_UTIL_H_
