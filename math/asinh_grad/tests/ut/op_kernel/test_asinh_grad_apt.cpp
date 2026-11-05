/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#endif

#include "../../../op_kernel/arch35/asinh_grad_tiling_data.h"

using namespace std;

extern "C" __global__ __aicore__ void asinh_grad(GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling);

class AsinhGradKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "AsinhGradKernelTest SetUp" << endl; }
    static void TearDownTestCase() { cout << "AsinhGradKernelTest TearDown" << endl; }
};

TEST_F(AsinhGradKernelTest, test_fp32_basic)
{
    constexpr size_t numElements = 256;
    size_t dataSize = numElements * sizeof(float);
    size_t tilingSize = sizeof(AsinhGradTilingData);
    uint32_t blockDim = 1;

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    // Keep the kernel test deterministic.  GmAlloc does not guarantee initialized
    // storage, and the extreme-value-safe implementation exercises Exp/Div on
    // every input element.
    std::memset(y, 0, dataSize);
    std::memset(dy, 0, dataSize);
    std::memset(z, 0, dataSize);
    std::memset(workspace, 0, 1024 * 1024);
    std::memset(tiling, 0, tilingSize);

    const float maxValue = std::numeric_limits<float>::max();
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float yValues[] = {0.0f,
                             -0.0f,
                             2.0f,
                             -2.0f,
                             88.0f,
                             90.0f,
                             inf,
                             -inf,
                             nan,
                             2.0f,
                             inf,
                             1e-5f,
                             -2e-5f,
                             3e-5f,
                             -1e-4f,
                             1.25e-4f,
                             -3.66e-4f,
                             88.7228f,
                             88.723f,
                             89.0f,
                             -89.0f,
                             89.4159f,
                             89.4161f,
                             89.0f,
                             177.4456f,
                             177.4458f,
                             maxValue,
                             89.0f,
                             90.0f,
                             -90.0f,
                             0.00586495036f,
                             0.0624999962747097f,
                             0.0625f,
                             -0.0625000074505806f,
                             -0.6603320240974426f,
                             -1.1568982601165771f,
                             1.9999998807907104f,
                             -2.000000238418579f,
                             15.999999046325684f,
                             16.0f,
                             -16.000001907348633f,
                             89.0f,
                             1.0f,
                             1.0f};
    const float dyValues[] = {maxValue,
                              -maxValue,
                              maxValue,
                              -maxValue,
                              maxValue,
                              1.0f,
                              maxValue,
                              -maxValue,
                              1.0f,
                              inf,
                              inf,
                              maxValue,
                              -maxValue,
                              maxValue,
                              -maxValue,
                              maxValue,
                              -maxValue,
                              1e38f,
                              1e38f,
                              1e38f,
                              -1e38f,
                              1e38f,
                              1e38f,
                              inf,
                              maxValue,
                              maxValue,
                              maxValue,
                              -0.0f,
                              inf,
                              -inf,
                              -8.96823646e-21f,
                              maxValue,
                              -maxValue,
                              maxValue,
                              -maxValue,
                              -maxValue,
                              maxValue,
                              -maxValue,
                              maxValue,
                              -maxValue,
                              maxValue,
                              1.0f,
                              std::numeric_limits<float>::min(),
                              std::numeric_limits<float>::denorm_min()};
    constexpr size_t valueCount = sizeof(yValues) / sizeof(yValues[0]);
    for (size_t i = 0; i < numElements; ++i) {
        reinterpret_cast<float*>(y)[i] = yValues[i % valueCount];
        reinterpret_cast<float*>(dy)[i] = dyValues[i % valueCount];
    }

    AsinhGradTilingData* tilingData = reinterpret_cast<AsinhGradTilingData*>(tiling);
    tilingData->totalNum = numElements;
    tilingData->blockFactor = numElements;
    tilingData->ubFactor = numElements;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(asinh_grad, blockDim, y, dy, z, workspace, tiling);

    for (size_t i = 0; i < numElements; ++i) {
        const double coshValue = std::cosh(static_cast<double>(yValues[i % valueCount]));
        const float denominator = static_cast<float>(coshValue);
        const float expected = std::isfinite(denominator) && std::abs(yValues[i % valueCount]) > 0.000244140625f ?
                                   static_cast<float>(static_cast<double>(dyValues[i % valueCount]) / coshValue) :
                                   dyValues[i % valueCount] / denominator;
        const float actual = reinterpret_cast<float*>(z)[i];
        if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(actual)) << i;
        } else if (std::isinf(expected) || expected == 0.0f) {
            EXPECT_EQ(actual, expected) << i;
            EXPECT_EQ(std::signbit(actual), std::signbit(expected)) << i;
        } else if (std::abs(yValues[i % valueCount]) <= 0.000244140625f) {
            EXPECT_EQ(actual, expected) << i;
        } else if (std::abs(yValues[i % valueCount]) <= 2.0f) {
            EXPECT_LE(std::abs((actual - expected) / expected), std::numeric_limits<float>::epsilon()) << i;
        } else {
            EXPECT_LE(std::abs((actual - expected) / expected), 1e-6f) << i;
        }
    }

    AscendC::GmFree(y);
    AscendC::GmFree(dy);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// Moving scalar dispatch outside the VF must preserve uniform tiles as well as
// the mixed extreme-value tile above, including subnormal quotients and tails.
TEST_F(AsinhGradKernelTest, test_fp32_uniform_ranges)
{
    constexpr size_t count = 129;
    constexpr size_t bytes = count * sizeof(float);
    auto* y = static_cast<float*>(AscendC::GmAlloc(bytes));
    auto* dy = static_cast<float*>(AscendC::GmAlloc(bytes));
    auto* z = static_cast<float*>(AscendC::GmAlloc(bytes));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(64));
    auto* tiling = static_cast<AsinhGradTilingData*>(AscendC::GmAlloc(sizeof(AsinhGradTilingData)));
    tiling->totalNum = count;
    tiling->blockFactor = count;
    tiling->ubFactor = 192;
    const float inf = std::numeric_limits<float>::infinity();
    const float cases[][2] = {{0.0f, std::numeric_limits<float>::denorm_min()},
                              {1.0f, 1.0f},
                              {1.0f, std::numeric_limits<float>::min()},
                              {50.0f, 10.0f},
                              {50.0f, 1.0f},
                              {89.0f, 1.0f},
                              {90.0f, 1.0f},
                              {90.0f, inf}};
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    for (const auto& values : cases) {
        for (size_t i = 0; i < count; ++i) {
            y[i] = i % 2 == 0 ? values[0] : -values[0];
            dy[i] = i % 2 == 0 ? values[1] : -values[1];
            z[i] = 7.0f;
        }
        ICPU_RUN_KF(asinh_grad, 1, reinterpret_cast<uint8_t*>(y), reinterpret_cast<uint8_t*>(dy),
                    reinterpret_cast<uint8_t*>(z), workspace, reinterpret_cast<uint8_t*>(tiling));
        for (size_t i = 0; i < count; ++i) {
            double denominator = std::cosh(static_cast<double>(y[i]));
            float expected = static_cast<float>(denominator) == inf ?
                                 dy[i] / inf :
                                 static_cast<float>(static_cast<double>(dy[i]) / denominator);
            if (std::isnan(expected)) {
                EXPECT_TRUE(std::isnan(z[i])) << values[0] << ", " << i;
            } else if (expected == 0.0f) {
                EXPECT_EQ(z[i], expected) << values[0] << ", " << i;
                EXPECT_EQ(std::signbit(z[i]), std::signbit(expected)) << i;
            } else {
                double tolerance = std::max(static_cast<double>(std::numeric_limits<float>::denorm_min()),
                                            std::abs(static_cast<double>(expected)) * 1e-6);
                EXPECT_NEAR(z[i], expected, tolerance) << values[0] << ", " << i;
            }
        }
    }
    AscendC::GmFree(y);
    AscendC::GmFree(dy);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(AsinhGradKernelTest, test_fp32_large)
{
    constexpr size_t numElements = 2048;
    size_t dataSize = numElements * sizeof(float);
    size_t tilingSize = sizeof(AsinhGradTilingData);
    uint32_t blockDim = 1;

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    std::memset(y, 0, dataSize);
    std::memset(dy, 0, dataSize);
    std::memset(z, 0, dataSize);
    std::memset(workspace, 0, 1024 * 1024);
    std::memset(tiling, 0, tilingSize);

    AsinhGradTilingData* tilingData = reinterpret_cast<AsinhGradTilingData*>(tiling);
    tilingData->totalNum = numElements;
    tilingData->blockFactor = numElements;
    tilingData->ubFactor = 1024;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(asinh_grad, blockDim, y, dy, z, workspace, tiling);

    AscendC::GmFree(y);
    AscendC::GmFree(dy);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(AsinhGradKernelTest, test_fp32_empty)
{
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(AsinhGradTilingData));
    *reinterpret_cast<float*>(value) = 7.0f;
    std::memset(tiling, 0, sizeof(AsinhGradTilingData));

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(asinh_grad, 1, value, value, value, value, tiling);

    EXPECT_EQ(*reinterpret_cast<float*>(value), 7.0f);
    AscendC::GmFree(value);
    AscendC::GmFree(tiling);
}
