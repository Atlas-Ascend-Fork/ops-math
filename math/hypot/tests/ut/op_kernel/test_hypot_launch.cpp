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
 * \file test_hypot_launch.cpp
 * \brief hypot kernel UT launcher TU (tikicpulib CPU mock), kept ge-free because kernel_type.h
 *        macros clash with ge::DataType. The entry reads DTYPE_SELF/DTYPE_OTHER/DTYPE_OUT aliases
 *        (float in this TU); other dtype combinations run through hypot_ut<T1, T2, T3, SCENE>.
 */

#include <cstdint>
#include <cstddef>
#include "tikicpulib.h"

// scene ids (plain macros; duplicated from op_kernel/hypot_tiling_data.h to keep this TU ge-free)
#define HYPOT_UT_SCENE_ONE_DIM 0U
#define HYPOT_UT_SCENE_MULTI_DIM_BRC 1U
#define HYPOT_UT_SCENE_MULTI_DIM_STRUCT 2U
#define HYPOT_UT_SCENE_MULTI_DIM_STREAM 3U
#define HYPOT_UT_SCENE_MULTI_DIM_VECIDX 4U

// kernel template entry + implementation
#include "../../../op_kernel/hypot.cpp"

namespace NsHypot {
// mirror of the device entry with explicit dtype template arguments (the device entry resolves
// them from the per-combination DTYPE_SELF/DTYPE_OTHER/DTYPE_OUT aliases)
template <typename T1, typename T2, typename T3, int BROADCAST_SCENE>
__global__ __aicore__ void hypot_ut(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(HypotTilingData);
    GET_TILING_DATA_WITH_STRUCT(HypotTilingData, tilingData, tiling);
    HypotKernelImpl<T1, T2, T3, BROADCAST_SCENE>(x1, x2, y, &tilingData);
}
} // namespace NsHypot

using NsHypot::hypot_ut;

// launch the FP32 kernel of the requested scene through the real entry; false for an unknown scene
bool LaunchHypotKernelFp32(uint32_t scene, int64_t tilingKey, size_t blockNum, uint8_t* x1, uint8_t* x2, uint8_t* y,
                           uint8_t* workspace, uint8_t* tiling)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    if (scene == HYPOT_UT_SCENE_ONE_DIM) {
        ICPU_RUN_KF((hypot<HYPOT_SCENE_ONE_DIM>), blockNum, x1, x2, y, workspace, tiling);
    } else if (scene == HYPOT_UT_SCENE_MULTI_DIM_VECIDX) {
        ICPU_RUN_KF((hypot<HYPOT_SCENE_MULTI_DIM_VECIDX>), blockNum, x1, x2, y, workspace, tiling);
    } else if (scene == HYPOT_UT_SCENE_MULTI_DIM_STRUCT) {
        ICPU_RUN_KF((hypot<HYPOT_SCENE_MULTI_DIM_STRUCT>), blockNum, x1, x2, y, workspace, tiling);
    } else if (scene == HYPOT_UT_SCENE_MULTI_DIM_BRC) {
        ICPU_RUN_KF((hypot<HYPOT_SCENE_MULTI_DIM_BRC>), blockNum, x1, x2, y, workspace, tiling);
    } else {
        return false;
    }
    return true;
}

// launch the FP16 kernel of the requested scene (dtype arguments resolved explicitly)
bool LaunchHypotKernelFp16(uint32_t scene, int64_t tilingKey, size_t blockNum, uint8_t* x1, uint8_t* x2, uint8_t* y,
                           uint8_t* workspace, uint8_t* tiling)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    if (scene == HYPOT_UT_SCENE_ONE_DIM) {
        ICPU_RUN_KF((hypot_ut<half, half, half, HYPOT_SCENE_ONE_DIM>), blockNum, x1, x2, y, workspace, tiling);
    } else if (scene == HYPOT_UT_SCENE_MULTI_DIM_VECIDX) {
        ICPU_RUN_KF((hypot_ut<half, half, half, HYPOT_SCENE_MULTI_DIM_VECIDX>), blockNum, x1, x2, y, workspace, tiling);
    } else if (scene == HYPOT_UT_SCENE_MULTI_DIM_STRUCT) {
        ICPU_RUN_KF((hypot_ut<half, half, half, HYPOT_SCENE_MULTI_DIM_STRUCT>), blockNum, x1, x2, y, workspace, tiling);
    } else if (scene == HYPOT_UT_SCENE_MULTI_DIM_STREAM) {
        ICPU_RUN_KF((hypot_ut<half, half, half, HYPOT_SCENE_MULTI_DIM_STREAM>), blockNum, x1, x2, y, workspace, tiling);
    } else if (scene == HYPOT_UT_SCENE_MULTI_DIM_BRC) {
        ICPU_RUN_KF((hypot_ut<half, half, half, HYPOT_SCENE_MULTI_DIM_BRC>), blockNum, x1, x2, y, workspace, tiling);
    } else {
        return false;
    }
    return true;
}

// launch the BF16 kernel of the requested scene (dtype arguments resolved explicitly)
bool LaunchHypotKernelBf16(uint32_t scene, int64_t tilingKey, size_t blockNum, uint8_t* x1, uint8_t* x2, uint8_t* y,
                           uint8_t* workspace, uint8_t* tiling)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    if (scene == HYPOT_UT_SCENE_ONE_DIM) {
        ICPU_RUN_KF((hypot_ut<bfloat16_t, bfloat16_t, bfloat16_t, HYPOT_SCENE_ONE_DIM>), blockNum, x1, x2, y, workspace,
                    tiling);
    } else if (scene == HYPOT_UT_SCENE_MULTI_DIM_VECIDX) {
        ICPU_RUN_KF((hypot_ut<bfloat16_t, bfloat16_t, bfloat16_t, HYPOT_SCENE_MULTI_DIM_VECIDX>), blockNum, x1, x2, y,
                    workspace, tiling);
    } else if (scene == HYPOT_UT_SCENE_MULTI_DIM_STRUCT) {
        ICPU_RUN_KF((hypot_ut<bfloat16_t, bfloat16_t, bfloat16_t, HYPOT_SCENE_MULTI_DIM_STRUCT>), blockNum, x1, x2, y,
                    workspace, tiling);
    } else if (scene == HYPOT_UT_SCENE_MULTI_DIM_STREAM) {
        ICPU_RUN_KF((hypot_ut<bfloat16_t, bfloat16_t, bfloat16_t, HYPOT_SCENE_MULTI_DIM_STREAM>), blockNum, x1, x2, y,
                    workspace, tiling);
    } else if (scene == HYPOT_UT_SCENE_MULTI_DIM_BRC) {
        ICPU_RUN_KF((hypot_ut<bfloat16_t, bfloat16_t, bfloat16_t, HYPOT_SCENE_MULTI_DIM_BRC>), blockNum, x1, x2, y,
                    workspace, tiling);
    } else {
        return false;
    }
    return true;
}

// launch a mixed-dtype kernel (FP16 x BF16 -> FP32): ONE_DIM / MULTI_DIM_BRC / MULTI_DIM_VECIDX,
// matching the host routing contract for mixed launches
bool LaunchHypotKernelMixF16Bf16(uint32_t scene, int64_t tilingKey, size_t blockNum, uint8_t* x1, uint8_t* x2,
                                 uint8_t* y, uint8_t* workspace, uint8_t* tiling)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    if (scene == HYPOT_UT_SCENE_ONE_DIM) {
        ICPU_RUN_KF((hypot_ut<half, bfloat16_t, float, HYPOT_SCENE_ONE_DIM>), blockNum, x1, x2, y, workspace, tiling);
    } else if (scene == HYPOT_UT_SCENE_MULTI_DIM_VECIDX) {
        ICPU_RUN_KF((hypot_ut<half, bfloat16_t, float, HYPOT_SCENE_MULTI_DIM_VECIDX>), blockNum, x1, x2, y, workspace,
                    tiling);
    } else if (scene == HYPOT_UT_SCENE_MULTI_DIM_BRC) {
        ICPU_RUN_KF((hypot_ut<half, bfloat16_t, float, HYPOT_SCENE_MULTI_DIM_BRC>), blockNum, x1, x2, y, workspace,
                    tiling);
    } else {
        return false;
    }
    return true;
}

// launch a mixed-dtype kernel (FP32 x BF16 -> FP32): ONE_DIM / MULTI_DIM_BRC / MULTI_DIM_VECIDX
bool LaunchHypotKernelMixF32Bf16(uint32_t scene, int64_t tilingKey, size_t blockNum, uint8_t* x1, uint8_t* x2,
                                 uint8_t* y, uint8_t* workspace, uint8_t* tiling)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    if (scene == HYPOT_UT_SCENE_ONE_DIM) {
        ICPU_RUN_KF((hypot_ut<float, bfloat16_t, float, HYPOT_SCENE_ONE_DIM>), blockNum, x1, x2, y, workspace, tiling);
    } else if (scene == HYPOT_UT_SCENE_MULTI_DIM_VECIDX) {
        ICPU_RUN_KF((hypot_ut<float, bfloat16_t, float, HYPOT_SCENE_MULTI_DIM_VECIDX>), blockNum, x1, x2, y, workspace,
                    tiling);
    } else if (scene == HYPOT_UT_SCENE_MULTI_DIM_BRC) {
        ICPU_RUN_KF((hypot_ut<float, bfloat16_t, float, HYPOT_SCENE_MULTI_DIM_BRC>), blockNum, x1, x2, y, workspace,
                    tiling);
    } else {
        return false;
    }
    return true;
}
