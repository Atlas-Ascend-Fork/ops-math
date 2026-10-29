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
 * \file test_reduce_std_with_mean_v2.cpp
 * \brief Kernel unit tests for ReduceStdWithMeanV2 operator
 *
 * Coverage targets:
 *   - fp16: small 1D, 2D, 3D shapes
 *   - invert=true / false
 *   - correction=0 / 1
 *   - Different reduce lengths
 *   - Multi-tile reduce (ubLength < reduceLen): exercises the Chan
 *     parallel-merge path of chunked Welford, which ATK shapes (small
 *     reduce dims) never reach.
 *   - Dual outputs: std (out) and mean (meanOut)
 *
 * TilingKey mapping (REDUCE_STD_WITH_MEAN_V2_*):
 *   REDUCE_STD_WITH_MEAN_V2_FP16 = 0, REDUCE_STD_WITH_MEAN_V2_FP32 = 1, REDUCE_STD_WITH_MEAN_V2_BF16 = 2
 */

#include "../../../op_kernel/reduce_std_with_mean_v2.cpp"
#include "../../../op_kernel/reduce_std_with_mean_v2_two_stage_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "tikicpulib.h"

namespace {

template <typename T1, typename T2>
T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

template <typename T>
T CastInput(float value)
{
    return static_cast<T>(value);
}

template <typename T>
float ToFloat(T value)
{
    return static_cast<float>(value);
}

// Golden reference for chunked Welford: mean = sum/N, then
// std = sqrt(sum((x-mean)^2) / max(0, N-correction)) [+ eps/invert].
// Returns (std, mean) per non-reduce row.
std::pair<std::vector<float>, std::vector<float>> WelfordGolden(const std::vector<float>& self, int64_t nonReduce,
                                                                int64_t reduceLen, int64_t correction, float eps,
                                                                bool invert)
{
    std::vector<float> stdOut(static_cast<size_t>(nonReduce), 0.0f);
    std::vector<float> meanOut(static_cast<size_t>(nonReduce), 0.0f);
    for (int64_t m = 0; m < nonReduce; ++m) {
        double sum = 0.0;
        for (int64_t i = 0; i < reduceLen; ++i) {
            sum += self[m * reduceLen + i];
        }
        float mean = static_cast<float>(sum / static_cast<double>(reduceLen));
        double sumSq = 0.0;
        for (int64_t i = 0; i < reduceLen; ++i) {
            double diff = self[m * reduceLen + i] - mean;
            sumSq += diff * diff;
        }
        float denom = static_cast<float>(reduceLen - correction);
        if (denom < 0.0f) {
            denom = 0.0f;
        }
        float var = (denom > 0.0f) ? static_cast<float>(sumSq / denom) : 0.0f;
        if (invert) {
            float tmp = std::sqrt(var + eps);
            stdOut[m] = (tmp > 0.0f) ? (1.0f / tmp) : 0.0f;
        } else {
            stdOut[m] = std::sqrt(var);
        }
        meanOut[m] = mean;
    }
    return {stdOut, meanOut};
}

// For fp32/bf16 kernel invocations the ut build's DTYPE_SELF is fixed to
// half, so the extern "C" kernel entry would be instantiated with the wrong
// dtype. Invoke the kernel class directly instead (C++ mangled names do not
// collide across template instantiations).
template <typename T, uint32_t schMode>
void RunWelfordKernelDirect(GM_ADDR self, GM_ADDR output, GM_ADDR meanOut, GM_ADDR workspace, GM_ADDR tiling)
{
    NsReduceStdWithMeanV2::ReduceStdWithMeanV2<T, schMode> op;
    op.Init(self, output, meanOut, workspace, reinterpret_cast<const ReduceStdWithMeanV2TilingData*>(tiling));
    op.Process();
}

template <typename T, uint32_t schMode>
void RunReduceStdWithMeanV2Case(const std::vector<float>& selfData, int64_t nonReduce, int64_t reduceLen,
                                int64_t ubLength, int64_t correction, float eps, bool invert, float atol, float rtol,
                                bool vectorize = false, int64_t ubM = 0)
{
    const size_t inputCount = static_cast<size_t>(nonReduce * reduceLen);
    const size_t outputCount = static_cast<size_t>(nonReduce);
    const size_t inputByteSize = inputCount * sizeof(T);
    const size_t outputByteSize = outputCount * sizeof(T);

    std::vector<T> typedSelf(inputCount);
    for (size_t i = 0; i < inputCount; ++i) {
        typedSelf[i] = CastInput<T>(selfData[i]);
    }

    uint8_t* selfGM = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(inputByteSize, 32)));
    uint8_t* outputGM = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(outputByteSize, 32)));
    uint8_t* meanOutGM = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(outputByteSize, 32)));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(32));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(ReduceStdWithMeanV2TilingData)));
    ASSERT_NE(selfGM, nullptr);
    ASSERT_NE(outputGM, nullptr);
    ASSERT_NE(meanOutGM, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    std::memcpy(selfGM, typedSelf.data(), inputByteSize);
    std::memset(outputGM, 0, outputByteSize);
    std::memset(meanOutGM, 0, outputByteSize);

    // Populate TilingData
    auto* tilingData = reinterpret_cast<ReduceStdWithMeanV2TilingData*>(tiling);
    tilingData->totalNonReduce = nonReduce;
    tilingData->reduceLength = reduceLen;
    tilingData->blockFactor = nonReduce;
    tilingData->ubLength = ubLength;
    tilingData->correction = correction;
    tilingData->eps = eps;
    tilingData->invert = invert;
    tilingData->vectorize = vectorize;
    tilingData->ubM = ubM;

    uint32_t blockDim = 1;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    if (vectorize && !std::is_same_v<T, half>) {
        // fp32/bf16 in the half-typed ut build: call the kernel class directly
        // (the extern "C" entry is instantiated with DTYPE_SELF=half only).
        auto func = RunWelfordKernelDirect<T, schMode>;
        ICPU_RUN_KF(func, blockDim, selfGM, outputGM, meanOutGM, workspace, tiling);
    } else {
        auto func = reduce_std_with_mean_v2<schMode>;
        ICPU_RUN_KF(func, blockDim, selfGM, outputGM, meanOutGM, workspace, tiling);
    }

    const auto* output = reinterpret_cast<const T*>(outputGM);
    const auto* meanOut = reinterpret_cast<const T*>(meanOutGM);
    const auto golden = WelfordGolden(selfData, nonReduce, reduceLen, correction, eps, invert);
    for (size_t i = 0; i < outputCount; ++i) {
        const float actualStd = ToFloat(output[i]);
        const float expectStd = golden.first[i];
        EXPECT_NEAR(actualStd, expectStd, atol + rtol * std::abs(expectStd))
            << "std mismatch at index " << i << ": actual=" << actualStd << ", expect=" << expectStd;
        const float actualMean = ToFloat(meanOut[i]);
        const float expectMean = golden.second[i];
        EXPECT_NEAR(actualMean, expectMean, atol + rtol * std::abs(expectMean))
            << "mean mismatch at index " << i << ": actual=" << actualMean << ", expect=" << expectMean;
    }

    AscendC::GmFree(selfGM);
    AscendC::GmFree(outputGM);
    AscendC::GmFree(meanOutGM);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// Helper to generate float test data
std::vector<float> MakeRandomData(size_t count, float low, float high)
{
    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i) {
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        data[i] = low + r * (high - low);
    }
    return data;
}

class ReduceStdWithMeanV2KernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ReduceStdWithMeanV2KernelTest SetUp" << std::endl;
        srand(42);
    }
    static void TearDownTestCase() { std::cout << "ReduceStdWithMeanV2KernelTest TearDown" << std::endl; }
};

} // namespace

// ==========================================================================
// fp16 (DTYPE_SELF=half, schMode=REDUCE_STD_WITH_MEAN_V2_FP16=0)
// ==========================================================================

TEST_F(ReduceStdWithMeanV2KernelTest, fp16_1d_small_std)
{
    const int64_t nonReduce = 1;
    const int64_t reduceLen = 8;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -5.0f, 5.0f);
    RunReduceStdWithMeanV2Case<half, 0>(selfData, nonReduce, reduceLen, reduceLen, 0, 0.0f, false, 2e-2f, 2e-2f);
}

TEST_F(ReduceStdWithMeanV2KernelTest, fp16_2d_std)
{
    const int64_t nonReduce = 4;
    const int64_t reduceLen = 32;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -10.0f, 10.0f);
    RunReduceStdWithMeanV2Case<half, 0>(selfData, nonReduce, reduceLen, reduceLen, 0, 0.0f, false, 2e-2f, 2e-2f);
}

TEST_F(ReduceStdWithMeanV2KernelTest, fp16_3d_std)
{
    const int64_t nonReduce = 6; // 2 * 3
    const int64_t reduceLen = 16;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -5.0f, 5.0f);
    RunReduceStdWithMeanV2Case<half, 0>(selfData, nonReduce, reduceLen, reduceLen, 0, 0.0f, false, 2e-2f, 2e-2f);
}

TEST_F(ReduceStdWithMeanV2KernelTest, fp16_invert_true)
{
    const int64_t nonReduce = 4;
    const int64_t reduceLen = 32;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -5.0f, 5.0f);
    RunReduceStdWithMeanV2Case<half, 0>(selfData, nonReduce, reduceLen, reduceLen, 0, 0.001f, true, 3e-2f, 3e-2f);
}

TEST_F(ReduceStdWithMeanV2KernelTest, fp16_correction_1)
{
    const int64_t nonReduce = 4;
    const int64_t reduceLen = 32;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -10.0f, 10.0f);
    RunReduceStdWithMeanV2Case<half, 0>(selfData, nonReduce, reduceLen, reduceLen, 1, 0.0f, false, 3e-2f, 3e-2f);
}

TEST_F(ReduceStdWithMeanV2KernelTest, fp16_invert_true_correction_1)
{
    const int64_t nonReduce = 4;
    const int64_t reduceLen = 32;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -5.0f, 5.0f);
    RunReduceStdWithMeanV2Case<half, 0>(selfData, nonReduce, reduceLen, reduceLen, 1, 0.001f, true, 3e-2f, 3e-2f);
}

// Multi-tile reduce: ubLength=64, reduceLen=512 -> 8 tiles.
// Exercises the Chan parallel-merge path (delta/count/m2 accumulation).
TEST_F(ReduceStdWithMeanV2KernelTest, fp16_multi_tile_merge)
{
    const int64_t nonReduce = 2;
    const int64_t reduceLen = 512;
    const int64_t ubLength = 64;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -10.0f, 10.0f);
    RunReduceStdWithMeanV2Case<half, 0>(selfData, nonReduce, reduceLen, ubLength, 0, 0.0f, false, 3e-2f, 3e-2f);
}

TEST_F(ReduceStdWithMeanV2KernelTest, fp16_multi_tile_invert)
{
    const int64_t nonReduce = 3;
    const int64_t reduceLen = 256;
    const int64_t ubLength = 32;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -5.0f, 5.0f);
    RunReduceStdWithMeanV2Case<half, 0>(selfData, nonReduce, reduceLen, ubLength, 1, 0.001f, true, 4e-2f, 4e-2f);
}

// ==========================================================================
// fp32 (DTYPE_SELF=float, schMode=REDUCE_STD_WITH_MEAN_V2_FP32=1)
// ==========================================================================
#ifdef REDUCE_STD_KERNEL_TEST_FP32

TEST_F(ReduceStdWithMeanV2KernelTest, fp32_2d_std)
{
    const int64_t nonReduce = 4;
    const int64_t reduceLen = 32;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -10.0f, 10.0f);
    RunReduceStdWithMeanV2Case<float, 1>(selfData, nonReduce, reduceLen, reduceLen, 0, 0.0f, false, 1e-5f, 1e-5f);
}

TEST_F(ReduceStdWithMeanV2KernelTest, fp32_invert_true)
{
    const int64_t nonReduce = 4;
    const int64_t reduceLen = 64;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -5.0f, 5.0f);
    RunReduceStdWithMeanV2Case<float, 1>(selfData, nonReduce, reduceLen, reduceLen, 0, 0.001f, true, 1e-5f, 1e-5f);
}

TEST_F(ReduceStdWithMeanV2KernelTest, fp32_correction_1)
{
    const int64_t nonReduce = 4;
    const int64_t reduceLen = 32;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -10.0f, 10.0f);
    RunReduceStdWithMeanV2Case<float, 1>(selfData, nonReduce, reduceLen, reduceLen, 1, 0.0f, false, 1e-5f, 1e-5f);
}

TEST_F(ReduceStdWithMeanV2KernelTest, fp32_multi_tile_merge)
{
    const int64_t nonReduce = 2;
    const int64_t reduceLen = 256;
    const int64_t ubLength = 64;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -5.0f, 5.0f);
    RunReduceStdWithMeanV2Case<float, 1>(selfData, nonReduce, reduceLen, ubLength, 0, 0.0f, false, 1e-5f, 1e-5f);
}

#endif // REDUCE_STD_KERNEL_TEST_FP32

// ==========================================================================
// bf16 (DTYPE_SELF=bfloat, schMode=REDUCE_STD_WITH_MEAN_V2_BF16=2)
// ==========================================================================
#ifdef REDUCE_STD_KERNEL_TEST_BF16

TEST_F(ReduceStdWithMeanV2KernelTest, bf16_2d_std)
{
    const int64_t nonReduce = 4;
    const int64_t reduceLen = 32;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -10.0f, 10.0f);
    RunReduceStdWithMeanV2Case<bfloat16_t, 2>(selfData, nonReduce, reduceLen, reduceLen, 0, 0.0f, false, 3e-2f, 3e-2f);
}

TEST_F(ReduceStdWithMeanV2KernelTest, bf16_invert_true)
{
    const int64_t nonReduce = 4;
    const int64_t reduceLen = 32;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -5.0f, 5.0f);
    RunReduceStdWithMeanV2Case<bfloat16_t, 2>(selfData, nonReduce, reduceLen, reduceLen, 0, 0.001f, true, 4e-2f, 4e-2f);
}

TEST_F(ReduceStdWithMeanV2KernelTest, bf16_multi_tile_merge)
{
    const int64_t nonReduce = 2;
    const int64_t reduceLen = 256;
    const int64_t ubLength = 64;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -5.0f, 5.0f);
    RunReduceStdWithMeanV2Case<bfloat16_t, 2>(selfData, nonReduce, reduceLen, ubLength, 0, 0.0f, false, 3e-2f, 3e-2f);
}

#endif // REDUCE_STD_KERNEL_TEST_BF16

// ==========================================================================
// M-vectorized path (ProcessVectorized): BlockReduceSum-based batch reduce.
// Enabled by tiling when reduceLen <= 64 (fp32 compute domain, one vcgadd
// repeat per row), single tile (reduceLen <= ubLength), 32B-aligned rows.
// fp16 cases run through the extern "C" kernel entry; fp32/bf16 cases invoke
// the kernel class directly (the ut build's DTYPE_SELF is fixed to half, so
// the entry would be instantiated with the wrong dtype).
// ==========================================================================

TEST_F(ReduceStdWithMeanV2KernelTest, fp16_vectorized_multi_block)
{
    // 2 batches of ubM=8 rows; single tile per row
    const int64_t nonReduce = 16;
    const int64_t reduceLen = 32;
    const int64_t ubM = 8;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -10.0f, 10.0f);
    RunReduceStdWithMeanV2Case<half, 0>(selfData, nonReduce, reduceLen, reduceLen, 0, 0.0f, false, 3e-2f, 3e-2f, true,
                                        ubM);
}

TEST_F(ReduceStdWithMeanV2KernelTest, fp16_vectorized_tail_block)
{
    // tail batch curM < ubM (3 batches: 8 + 8 + 4); invert + correction combo
    const int64_t nonReduce = 20;
    const int64_t reduceLen = 16;
    const int64_t ubM = 8;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -5.0f, 5.0f);
    RunReduceStdWithMeanV2Case<half, 0>(selfData, nonReduce, reduceLen, reduceLen, 1, 0.001f, true, 4e-2f, 4e-2f, true,
                                        ubM);
}

TEST_F(ReduceStdWithMeanV2KernelTest, fp16_vectorized_invert)
{
    // invert + correction combo, single batch, R = 64 (mask = 64, b = 8)
    const int64_t nonReduce = 8;
    const int64_t reduceLen = 64;
    const int64_t ubM = 8;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -5.0f, 5.0f);
    RunReduceStdWithMeanV2Case<half, 0>(selfData, nonReduce, reduceLen, reduceLen, 1, 0.001f, true, 4e-2f, 4e-2f, true,
                                        ubM);
}

TEST_F(ReduceStdWithMeanV2KernelTest, fp32_vectorized_multi_block)
{
    // 2 batches of ubM=8 rows; fp32 via direct kernel-class invocation
    const int64_t nonReduce = 16;
    const int64_t reduceLen = 32;
    const int64_t ubM = 8;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -10.0f, 10.0f);
    RunReduceStdWithMeanV2Case<float, 1>(selfData, nonReduce, reduceLen, reduceLen, 0, 0.0f, false, 1e-5f, 1e-5f, true,
                                         ubM);
}

TEST_F(ReduceStdWithMeanV2KernelTest, fp32_vectorized_tail_invert)
{
    // tail batch (8 + 8 + 4); R = 64; invert + correction
    const int64_t nonReduce = 20;
    const int64_t reduceLen = 64;
    const int64_t ubM = 8;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -5.0f, 5.0f);
    RunReduceStdWithMeanV2Case<float, 1>(selfData, nonReduce, reduceLen, reduceLen, 1, 0.001f, true, 1e-5f, 1e-5f, true,
                                         ubM);
}

TEST_F(ReduceStdWithMeanV2KernelTest, bf16_vectorized_multi_block)
{
    // 2 batches of ubM=8 rows; bf16 via direct kernel-class invocation
    const int64_t nonReduce = 16;
    const int64_t reduceLen = 32;
    const int64_t ubM = 8;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -10.0f, 10.0f);
    RunReduceStdWithMeanV2Case<bfloat16_t, 2>(selfData, nonReduce, reduceLen, reduceLen, 0, 0.0f, false, 3e-2f, 3e-2f,
                                              true, ubM);
}

TEST_F(ReduceStdWithMeanV2KernelTest, bf16_vectorized_tail_invert)
{
    // tail batch (8 + 8 + 4); R = 16; invert + correction
    const int64_t nonReduce = 20;
    const int64_t reduceLen = 16;
    const int64_t ubM = 8;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -5.0f, 5.0f);
    RunReduceStdWithMeanV2Case<bfloat16_t, 2>(selfData, nonReduce, reduceLen, reduceLen, 1, 0.001f, true, 4e-2f, 4e-2f,
                                              true, ubM);
}

// ==========================================================================
// P1 segmented path (PARALLEL_PLAN.md P1): R > 64 (multiple of 64) with the
// segmented pass1 (S = R/64 segments, mask = 64) + per-row ReduceSum pass2.
// R <= 64 still exercises the same unified pass1/pass2 structure (S = 1).
// ==========================================================================

TEST_F(ReduceStdWithMeanV2KernelTest, fp16_vectorized_segmented_128)
{
    // R=128 (S=2); 3 batches of ubM=8; invert + correction combo
    const int64_t nonReduce = 24;
    const int64_t reduceLen = 128;
    const int64_t ubM = 8;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -10.0f, 10.0f);
    RunReduceStdWithMeanV2Case<half, 0>(selfData, nonReduce, reduceLen, reduceLen, 1, 0.001f, true, 4e-2f, 4e-2f, true,
                                        ubM);
}

TEST_F(ReduceStdWithMeanV2KernelTest, fp16_vectorized_segmented_256_tail)
{
    // R=256 (S=4); tail batch (8 + 8 + 4); correction=1
    const int64_t nonReduce = 20;
    const int64_t reduceLen = 256;
    const int64_t ubM = 8;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -5.0f, 5.0f);
    RunReduceStdWithMeanV2Case<half, 0>(selfData, nonReduce, reduceLen, reduceLen, 1, 0.0f, false, 4e-2f, 4e-2f, true,
                                        ubM);
}

TEST_F(ReduceStdWithMeanV2KernelTest, fp32_vectorized_segmented_256)
{
    // R=256 (S=4); 2 batches of ubM=8; fp32 via direct kernel-class invocation
    const int64_t nonReduce = 16;
    const int64_t reduceLen = 256;
    const int64_t ubM = 8;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -10.0f, 10.0f);
    RunReduceStdWithMeanV2Case<float, 1>(selfData, nonReduce, reduceLen, reduceLen, 0, 0.0f, false, 1e-5f, 1e-5f, true,
                                         ubM);
}

TEST_F(ReduceStdWithMeanV2KernelTest, fp32_vectorized_segmented_512_tail)
{
    // R=512 (S=8); tail batch (8 + 8 + 4); invert + correction
    const int64_t nonReduce = 20;
    const int64_t reduceLen = 512;
    const int64_t ubM = 8;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -5.0f, 5.0f);
    RunReduceStdWithMeanV2Case<float, 1>(selfData, nonReduce, reduceLen, reduceLen, 1, 0.001f, true, 1e-5f, 1e-5f, true,
                                         ubM);
}

TEST_F(ReduceStdWithMeanV2KernelTest, bf16_vectorized_segmented_128)
{
    // R=128 (S=2); 2 batches of ubM=8; bf16 via direct kernel-class invocation
    const int64_t nonReduce = 16;
    const int64_t reduceLen = 128;
    const int64_t ubM = 8;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -5.0f, 5.0f);
    RunReduceStdWithMeanV2Case<bfloat16_t, 2>(selfData, nonReduce, reduceLen, reduceLen, 0, 0.0f, false, 3e-2f, 3e-2f,
                                              true, ubM);
}

TEST_F(ReduceStdWithMeanV2KernelTest, bf16_vectorized_segmented_192)
{
    // R=192 (S=3, odd segment count); 2 batches of ubM=16; invert + correction
    // combo. S=3 exercises the non-power-of-2 segment path: pass1 repeat =
    // curM*S = 48 <= 255, pass2 per-row ReduceSum over 8*S = 24 partial sums.
    const int64_t nonReduce = 32;
    const int64_t reduceLen = 192;
    const int64_t ubM = 16;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -5.0f, 5.0f);
    RunReduceStdWithMeanV2Case<bfloat16_t, 2>(selfData, nonReduce, reduceLen, reduceLen, 1, 0.001f, true, 4e-2f, 4e-2f,
                                              true, ubM);
}

TEST_F(ReduceStdWithMeanV2KernelTest, bf16_vectorized_segmented_192_single_tail)
{
    // R=192 (S=3); M=17 = ubM*2 + 1 -> last batch curM=1 (single-row tail),
    // the minimal-curM boundary for pass1 repeat = curM*S = 3.
    const int64_t nonReduce = 17;
    const int64_t reduceLen = 192;
    const int64_t ubM = 8;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -10.0f, 10.0f);
    RunReduceStdWithMeanV2Case<bfloat16_t, 2>(selfData, nonReduce, reduceLen, reduceLen, 0, 0.0f, false, 3e-2f, 3e-2f,
                                              true, ubM);
}

// ==========================================================================
// two-stage cross-core reduction (PARALLEL_PLAN.md two-stage): stage 1
// (ReduceStdWithMeanV2Partial, blockDim=K, one core per R segment)
// writes per-(row, segment) (n, mean, m2) triples to a float32 partial
// tensor [M, K, 3]; stage 2 (ReduceStdWithMeanV2Merge) Chan-merges the
// K triples per row. The two-stage result must equal the single-kernel
// golden (Chan merge is exact for any partition, fp32 summation order
// differences ~1e-7). Both kernel classes are invoked directly (the ut
// build's DTYPE_SELF is fixed to half, so the extern "C" entries would be
// instantiated with the wrong dtype).
// ==========================================================================

template <typename T, uint32_t schMode>
void RunWelfordPartialDirect(GM_ADDR self, GM_ADDR partial, GM_ADDR workspace, GM_ADDR tiling)
{
    NsReduceStdWithMeanV2::ReduceStdWithMeanV2Partial<T, schMode> op;
    op.Init(self, partial, workspace, reinterpret_cast<const ReduceStdWithMeanV2TilingData*>(tiling));
    op.Process();
}

template <typename T, uint32_t schMode>
void RunWelfordMergeDirect(GM_ADDR partial, GM_ADDR output, GM_ADDR meanOut, GM_ADDR workspace, GM_ADDR tiling)
{
    NsReduceStdWithMeanV2::ReduceStdWithMeanV2Merge<T, schMode> op;
    op.Init(partial, output, meanOut, workspace, reinterpret_cast<const ReduceStdWithMeanV2TilingData*>(tiling));
    op.Process();
}

template <typename T, uint32_t schMode>
void RunTwoStageWelfordCase(const std::vector<float>& selfData, int64_t nonReduce, int64_t reduceLen, int64_t ubLength,
                            int64_t partialK, int64_t correction, float eps, bool invert, float atol, float rtol,
                            uint32_t mergeBlockDim = 1)
{
    const size_t inputCount = static_cast<size_t>(nonReduce * reduceLen);
    const size_t outputCount = static_cast<size_t>(nonReduce);
    const size_t inputByteSize = inputCount * sizeof(T);
    const size_t outputByteSize = outputCount * sizeof(float); // Merge emits fp32
    const size_t partialCount = static_cast<size_t>(nonReduce * partialK * 3);

    std::vector<T> typedSelf(inputCount);
    for (size_t i = 0; i < inputCount; ++i) {
        typedSelf[i] = CastInput<T>(selfData[i]);
    }

    uint8_t* selfGM = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(inputByteSize, 32)));
    uint8_t* partialGM = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(partialCount * sizeof(float), 32)));
    uint8_t* outputGM = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(outputByteSize, 32)));
    uint8_t* meanOutGM = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(outputByteSize, 32)));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(32));
    uint8_t* tilingP = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(ReduceStdWithMeanV2TilingData)));
    uint8_t* tilingM = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(ReduceStdWithMeanV2TilingData)));
    ASSERT_NE(selfGM, nullptr);
    ASSERT_NE(partialGM, nullptr);
    ASSERT_NE(outputGM, nullptr);
    ASSERT_NE(meanOutGM, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tilingP, nullptr);
    ASSERT_NE(tilingM, nullptr);

    std::memcpy(selfGM, typedSelf.data(), inputByteSize);
    std::memset(partialGM, 0, partialCount * sizeof(float));
    std::memset(outputGM, 0, outputByteSize);
    std::memset(meanOutGM, 0, outputByteSize);

    // stage 1 tiling (Partial): blockDim = partialK
    auto* tilingDataP = reinterpret_cast<ReduceStdWithMeanV2TilingData*>(tilingP);
    tilingDataP->totalNonReduce = nonReduce;
    tilingDataP->reduceLength = reduceLen;
    tilingDataP->blockFactor = nonReduce;
    tilingDataP->ubLength = ubLength;
    tilingDataP->correction = correction;
    tilingDataP->eps = eps;
    tilingDataP->invert = invert;
    tilingDataP->vectorize = false;
    tilingDataP->ubM = 0;
    tilingDataP->stage = 1;
    tilingDataP->partialK = partialK;
    tilingDataP->partialSeg = (reduceLen + partialK - 1) / partialK;

    auto funcP = RunWelfordPartialDirect<T, schMode>;
    ICPU_RUN_KF(funcP, static_cast<uint32_t>(partialK), selfGM, partialGM, workspace, tilingP);

    // spot-check the partials: n must equal the segment length of each core
    const int64_t rSeg = (reduceLen + partialK - 1) / partialK;
    const auto* partials = reinterpret_cast<const float*>(partialGM);
    for (int64_t m = 0; m < nonReduce; ++m) {
        for (int64_t k = 0; k < partialK; ++k) {
            const int64_t segStart = k * rSeg;
            const int64_t segLen = (segStart + rSeg <= reduceLen) ? rSeg : (reduceLen - segStart);
            EXPECT_NEAR(partials[m * partialK * 3 + k * 3], static_cast<float>(segLen), 1e-5f)
                << "partial n mismatch at (m,k)=(" << m << "," << k << ")";
        }
    }

    // stage 2 tiling (Merge): blockDim = mergeBlockDim
    auto* tilingDataM = reinterpret_cast<ReduceStdWithMeanV2TilingData*>(tilingM);
    tilingDataM->totalNonReduce = nonReduce;
    tilingDataM->reduceLength = reduceLen;
    tilingDataM->blockFactor = nonReduce;
    tilingDataM->ubLength = 0;
    tilingDataM->correction = correction;
    tilingDataM->eps = eps;
    tilingDataM->invert = invert;
    tilingDataM->vectorize = false;
    tilingDataM->ubM = 0;
    tilingDataM->stage = 2;
    tilingDataM->partialK = partialK;
    tilingDataM->partialSeg = 0;

    auto funcM = RunWelfordMergeDirect<T, schMode>;
    ICPU_RUN_KF(funcM, mergeBlockDim, partialGM, outputGM, meanOutGM, workspace, tilingM);

    const auto* output = reinterpret_cast<const float*>(outputGM);
    const auto* meanOut = reinterpret_cast<const float*>(meanOutGM);
    const auto golden = WelfordGolden(selfData, nonReduce, reduceLen, correction, eps, invert);
    for (size_t i = 0; i < outputCount; ++i) {
        const float actualStd = output[i];
        const float expectStd = golden.first[i];
        EXPECT_NEAR(actualStd, expectStd, atol + rtol * std::abs(expectStd))
            << "std mismatch at index " << i << ": actual=" << actualStd << ", expect=" << expectStd;
        const float actualMean = meanOut[i];
        const float expectMean = golden.second[i];
        EXPECT_NEAR(actualMean, expectMean, atol + rtol * std::abs(expectMean))
            << "mean mismatch at index " << i << ": actual=" << actualMean << ", expect=" << expectMean;
    }

    AscendC::GmFree(selfGM);
    AscendC::GmFree(partialGM);
    AscendC::GmFree(outputGM);
    AscendC::GmFree(meanOutGM);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tilingP);
    AscendC::GmFree(tilingM);
}

TEST_F(ReduceStdWithMeanV2KernelTest, fp32_p3_twostage_4096x2)
{
    // M=4, R=4096, K=2 -> segs 2048/2048; invert + correction combo
    const int64_t nonReduce = 4;
    const int64_t reduceLen = 4096;
    const int64_t ubLength = 4096; // whole segment in one tile
    const int64_t partialK = 2;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -10.0f, 10.0f);
    RunTwoStageWelfordCase<float, 1>(selfData, nonReduce, reduceLen, ubLength, partialK, 1, 0.001f, true, 1e-5f, 1e-5f);
}

TEST_F(ReduceStdWithMeanV2KernelTest, fp32_p3_twostage_6144x3)
{
    // M=16, R=6144, K=3 -> segs 2048/2048/2048 (exact); multi-tile rows
    const int64_t nonReduce = 16;
    const int64_t reduceLen = 6144;
    const int64_t ubLength = 2048;
    const int64_t partialK = 3;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -5.0f, 5.0f);
    RunTwoStageWelfordCase<float, 1>(selfData, nonReduce, reduceLen, ubLength, partialK, 0, 0.0f, false, 1e-5f, 1e-5f);
}

TEST_F(ReduceStdWithMeanV2KernelTest, fp32_p3_twostage_merge_row_split)
{
    // Merge with blockDim=2: rows split across 2 merge cores; K=4
    const int64_t nonReduce = 8;
    const int64_t reduceLen = 8192;
    const int64_t ubLength = 4096;
    const int64_t partialK = 4;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -10.0f, 10.0f);
    RunTwoStageWelfordCase<float, 1>(selfData, nonReduce, reduceLen, ubLength, partialK, 1, 0.001f, true, 1e-5f, 1e-5f,
                                     2);
}

TEST_F(ReduceStdWithMeanV2KernelTest, fp16_two_stage_2304x2)
{
    // fp16: M=6, R=2304, K=2 -> segs 1152/1152; R*2 % 32 == 0
    const int64_t nonReduce = 6;
    const int64_t reduceLen = 2304;
    const int64_t ubLength = 1152;
    const int64_t partialK = 2;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -5.0f, 5.0f);
    RunTwoStageWelfordCase<half, 0>(selfData, nonReduce, reduceLen, ubLength, partialK, 1, 0.001f, true, 4e-2f, 4e-2f);
}

TEST_F(ReduceStdWithMeanV2KernelTest, bf16_two_stage_5000x3_short_tail)
{
    // bf16: M=5, R=5000, K=3 -> segs 1667/1667/1666 (last segment short and
    // not a 64 multiple); R*2 % 32 == 0 (10000B rows)
    const int64_t nonReduce = 5;
    const int64_t reduceLen = 5000;
    const int64_t ubLength = 1667;
    const int64_t partialK = 3;
    auto selfData = MakeRandomData(nonReduce * reduceLen, -10.0f, 10.0f);
    RunTwoStageWelfordCase<bfloat16_t, 2>(selfData, nonReduce, reduceLen, ubLength, partialK, 0, 0.0f, false, 4e-2f,
                                          4e-2f);
}
