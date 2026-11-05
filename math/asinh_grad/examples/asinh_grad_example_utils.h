/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASINH_GRAD_EXAMPLE_UTILS_H
#define ASINH_GRAD_EXAMPLE_UTILS_H

#include <cstdint>
#include <cstring>

namespace AsinhGradExample {

// The examples generate zero and normal finite FP16 values. Round to nearest
// even; values below the normal range are flushed to zero, overflow to infinity.
inline uint16_t FloatToHalfBits(float value)
{
    uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16U) & 0x8000U);
    int32_t exponent = static_cast<int32_t>((bits >> 23U) & 0xFFU) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFFU;
    if (exponent <= 0) {
        return sign;
    }
    if (exponent >= 0x1F) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    mantissa += 0x00000FFFU + ((mantissa >> 13U) & 1U);
    if (mantissa >= 0x800000U) {
        mantissa = 0U;
        exponent += 1;
        if (exponent >= 0x1F) {
            return static_cast<uint16_t>(sign | 0x7C00U);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10U) | (mantissa >> 13U));
}

} // namespace AsinhGradExample

#endif
