/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RRELU_WITH_NOISE_TILING_H
#define RRELU_WITH_NOISE_TILING_H

#include <cstdint>
#include <cstring>

#include "../../../../random_common/op_kernel/arch35/random_unified_tiling_data_arch35.h"
#include "kernel_tiling/kernel_tiling.h"

#if defined(ASCENDC_CPU_DEBUG)
// 本仓 CPU debug 环境(cann-9.2.0)的 cpudebug stub(__shfl* 两参)与
// aarch64-linux/asc 的 warp 实现(三参, 带 width)签名不匹配, 导致
// random_kernel_base.h -> asc_simt.h 编译失败(仓内 random SIMT 算子的
// op_kernel UT 均受影响)。此处补齐缺失的三参重载; 这些 warp 接口在
// rrelu_with_noise kernel 路径中不会被实际调用, 仅保证可编译。
#if !defined(RRELU_UT_SHFL_COMPAT)
#define RRELU_UT_SHFL_COMPAT
inline int32_t __shfl(int32_t var, int32_t srcLane, int32_t) { return var; }
inline uint32_t __shfl(uint32_t var, int32_t srcLane, int32_t) { return var; }
inline float __shfl(float var, int32_t srcLane, int32_t) { return var; }
inline int64_t __shfl(int64_t var, int32_t srcLane, int32_t) { return var; }
inline uint64_t __shfl(uint64_t var, int32_t srcLane, int32_t) { return var; }
inline int32_t __shfl_up(int32_t var, uint32_t, int32_t) { return var; }
inline uint32_t __shfl_up(uint32_t var, uint32_t, int32_t) { return var; }
inline float __shfl_up(float var, uint32_t, int32_t) { return var; }
inline int64_t __shfl_up(int64_t var, uint32_t, int32_t) { return var; }
inline uint64_t __shfl_up(uint64_t var, uint32_t, int32_t) { return var; }
inline int32_t __shfl_down(int32_t var, uint32_t, int32_t) { return var; }
inline uint32_t __shfl_down(uint32_t var, uint32_t, int32_t) { return var; }
inline float __shfl_down(float var, uint32_t, int32_t) { return var; }
inline int64_t __shfl_down(int64_t var, uint32_t, int32_t) { return var; }
inline uint64_t __shfl_down(uint64_t var, uint32_t, int32_t) { return var; }
inline int32_t __shfl_xor(int32_t var, int32_t, int32_t) { return var; }
inline uint32_t __shfl_xor(uint32_t var, int32_t, int32_t) { return var; }
inline float __shfl_xor(float var, int32_t, int32_t) { return var; }
inline int64_t __shfl_xor(int64_t var, int32_t, int32_t) { return var; }
inline uint64_t __shfl_xor(uint64_t var, int32_t, int32_t) { return var; }
#endif // RRELU_UT_SHFL_COMPAT
#endif // defined(ASCENDC_CPU_DEBUG)

static inline unsigned long long __mul_i32toi64(unsigned int lhs, unsigned int rhs)
{
    return static_cast<unsigned long long>(lhs) * rhs;
}

#define __aicore__
#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, RandomUnifiedSimtTilingDataStruct* constData)
{
    const __gm__ uint32_t* src = reinterpret_cast<const __gm__ uint32_t*>(tiling);
    uint32_t* dst = reinterpret_cast<uint32_t*>(constData);
    for (size_t i = 0; i < sizeof(RandomUnifiedSimtTilingDataStruct) / sizeof(uint32_t); ++i) {
        *(dst + i) = *(src + i);
    }
}
#else
inline void InitTilingData(uint8_t* tiling, RandomUnifiedSimtTilingDataStruct* constData)
{
    std::memcpy(constData, tiling, sizeof(RandomUnifiedSimtTilingDataStruct));
}
#endif // __NPU_TILING__

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)              \
    __ubuf__ tilingStruct* tilingDataPointer = reinterpret_cast<__ubuf__ tilingStruct*>( \
        reinterpret_cast<__ubuf__ uint8_t*>(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                             \
    InitTilingData(tilingArg, &tilingData)

#define GET_TILING_DATA(tilingData, tilingArg)    \
    RandomUnifiedSimtTilingDataStruct tilingData; \
    InitTilingData(tilingArg, &tilingData)

#endif // RRELU_WITH_NOISE_TILING_H
