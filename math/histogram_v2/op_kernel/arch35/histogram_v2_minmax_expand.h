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
 * \file histogram_v2_minmax_expand.h
 * \brief min==max range expansion aligned with CPU bins_edges_correction (#163506):
 *        for float, expand to the farther of "adjacent representable value" and "+/- 1";
 *        for integer types, raw +/- 1 is already exact.
 */
#ifndef HISTOGRAM_V2_MINMAX_EXPAND_H
#define HISTOGRAM_V2_MINMAX_EXPAND_H

namespace HistogramV2Common {

// Next representable float toward -inf, IEEE-754 sign-magnitude bit step.
// +inf saturates to FLT_MAX (mirrors nextafterf(inf, -inf)); NaN never reaches here
// because the caller only expands when minValue == maxValue (false for NaN).
__aicore__ inline float FloatNextAfterDown(float v)
{
    union {
        float f;
        uint32_t u;
    } cvt;
    cvt.f = v;
    if (cvt.u == 0x7F800000u) { // +inf -> FLT_MAX
        cvt.u = 0x7F7FFFFFu;
    } else if (v > 0) { // positive: one bit down in magnitude
        cvt.u -= 1;
    } else if (v < 0) { // negative: one bit up in magnitude (-inf stays -inf via wrap guard below)
        cvt.u = (cvt.u == 0xFF800000u) ? 0xFF800000u : cvt.u + 1;
    } else { // +/-0 -> smallest negative denormal
        cvt.u = 0x80000001u;
    }
    return cvt.f;
}

// Next representable float toward +inf.
__aicore__ inline float FloatNextAfterUp(float v)
{
    union {
        float f;
        uint32_t u;
    } cvt;
    cvt.f = v;
    if (cvt.u == 0xFF800000u) { // -inf -> -FLT_MAX
        cvt.u = 0xFF7FFFFFu;
    } else if (v > 0) { // positive: one bit up in magnitude (+inf stays +inf via wrap guard below)
        cvt.u = (cvt.u == 0x7F800000u) ? 0x7F800000u : cvt.u + 1;
    } else if (v < 0) { // negative: one bit down in magnitude
        cvt.u -= 1;
    } else { // +/-0 -> smallest positive denormal
        cvt.u = 0x00000001u;
    }
    return cvt.f;
}

template <typename T>
struct MinMaxExpander {
    // Integer compute types: +/- 1 is exact, keep the raw expansion.
    static __aicore__ inline void Expand(T& minValue, T& maxValue)
    {
        minValue = minValue - 1;
        maxValue = maxValue + 1;
    }
};

template <>
struct MinMaxExpander<float> {
    // float: take the farther of the adjacent representable value and +/- 1.
    // m +/- 1 in fp32 is either exact (|m| < 2^24) or rounds back toward m (|m| >= 2^24),
    // in which case the nextafter candidate always wins the comparison, so the rounding
    // error of "m +/- 1" never affects the result -- pointwise equivalent to the CPU
    // double version min(nextafter(x, lowest), x - 1) / max(nextafter(x, highest), x + 1).
    //
    // Boundary note: at |m| == FLT_MAX the farther candidate becomes +/-inf (there is no
    // farther finite float). This is kept intentionally: left < right still holds, the
    // value >= min && value <= max comparisons filter NaN, and an inf endpoint makes
    // minMaxLength infinite so all (finite, == m) samples deterministically land in bin 0.
    // Clamping the candidate back to +/-FLT_MAX would collapse the range to zero width
    // (division by zero in the bin-index math), which is strictly worse. CPU puts such
    // samples in the middle bin via double-precision edges; matching that would require
    // double bin math for a pathological all-FLT_MAX input, which is not worth it.
    static __aicore__ inline void Expand(float& minValue, float& maxValue)
    {
        const float m = minValue;
        const float down = FloatNextAfterDown(m);
        const float up = FloatNextAfterUp(m);
        const float minusOne = m - 1.0f;
        const float plusOne = m + 1.0f;
        minValue = down < minusOne ? down : minusOne;
        maxValue = up > plusOne ? up : plusOne;
    }
};

// Expand [minValue, maxValue] when they are equal so that the bin-index
// division never sees a zero range. Keeps left < right unconditionally.
template <typename T>
__aicore__ inline void ExpandMinMaxIfEqual(T& minValue, T& maxValue)
{
    if (minValue == maxValue) {
        MinMaxExpander<T>::Expand(minValue, maxValue);
    }
}

} // namespace HistogramV2Common

#endif // HISTOGRAM_V2_MINMAX_EXPAND_H
