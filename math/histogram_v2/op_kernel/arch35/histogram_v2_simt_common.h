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
 * \file histogram_v2_simt_common.h
 * \brief Common pieces of the unified arch35 HistogramV2 SIMT kernel: THREAD_NUM,
 *        input-dtype to compute-type mapping, and the single copy of the bin-index
 *        semantics (range filter + bin formula + last-bin clamp).
 */
#ifndef HISTOGRAM_V2_SIMT_COMMON_H
#define HISTOGRAM_V2_SIMT_COMMON_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "histogram_v2_minmax_expand.h"
#include "histogram_v2_scalar_cast.h"
#include "histogram_v2_tiling_data.h"

#ifndef HISTOGRAM_V2_THREAD_NUM
#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM = 128;
#else
constexpr uint32_t THREAD_NUM = 512;
#endif
#define HISTOGRAM_V2_THREAD_NUM
#endif

namespace HistogramV2SIMT {
using namespace AscendC;

// Input dtype -> bin-index compute type. fp32/fp16/bf16 compute in float;
// int32/int64 compute in int64 (aligned with CUDA acc_type); int8/uint8/int16 in int32.
template <typename X>
struct ComputeTypeOf {
    using Type = float;
};
template <>
struct ComputeTypeOf<int32_t> {
    using Type = int64_t;
};
template <>
struct ComputeTypeOf<int64_t> {
    using Type = int64_t;
};
template <>
struct ComputeTypeOf<int8_t> {
    using Type = int32_t;
};
template <>
struct ComputeTypeOf<uint8_t> {
    using Type = int32_t;
};
template <>
struct ComputeTypeOf<int16_t> {
    using Type = int32_t;
};

// Minimal compile-time conditional (kernel-side type_traits is not assumed).
template <bool COND, typename T, typename F>
struct ConditionalType {
    using Type = T;
};
template <typename T, typename F>
struct ConditionalType<false, T, F> {
    using Type = F;
};
} // namespace HistogramV2SIMT

namespace HistogramV2Common {

// Single copy of the bin-index semantics, shared by all load modes:
// drop values outside [minValue, maxValue] (NaN is filtered by the same pair of
// comparisons), bin = trunc((value - minValue) * bins / minMaxLength) with the
// arithmetic performed strictly in subtract -> multiply -> divide -> truncate order,
// and the value == maxValue edge clamped into the last bin.
// The caller must have expanded min/max via ExpandMinMaxIfEqual and precomputed
// minMaxLength = maxValue - minValue.
template <typename COMPUTE_TYPE>
__simt_callee__ __aicore__ inline bool ComputeBinIndex(COMPUTE_TYPE value, COMPUTE_TYPE minValue, COMPUTE_TYPE maxValue,
                                                       COMPUTE_TYPE minMaxLength, int64_t bins, int32_t& indexBin)
{
    // 正向判定与 CPU 对齐（HistogramKernel.cpp：范围外与 NaN 元素均丢弃；
    // NaN 对 >=/<= 恒 false -> !(...) 为 true -> 丢弃；反向写法会把 NaN 漏进 bin0）
    if (!(value >= minValue && value <= maxValue)) {
        return false;
    }
    // 下标只需表示 [0, bins)，用 int32 保存：中间乘除仍按 int64（bins 为 int64）计算，
    // 结果不变；64 位引用出参会让 SIMT 编译器把下标落到线程本地内存，实测慢 30%+
    indexBin = static_cast<int32_t>((value - minValue) * bins / minMaxLength);
    // 防御性钳位：value==maxValue 时 indexBin==bins 收进最后一 bin；
    // NaN 输入的 bin 计算结果依硬件转换行为而定（950 上浮点->整数归 0），钳位兜底防越界
    if (indexBin >= bins) {
        indexBin = static_cast<int32_t>(bins) - 1;
    } else if (indexBin < 0) {
        indexBin = 0;
    }
    return true;
}

} // namespace HistogramV2Common

#endif // HISTOGRAM_V2_SIMT_COMMON_H
