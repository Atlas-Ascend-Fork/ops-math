/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_kernel/log_add_exp_tiling_data.h"

extern "C" __global__ __aicore__ void log_add_exp_fp16_ut(GM_ADDR x, GM_ADDR y, GM_ADDR out, GM_ADDR tiling);
extern "C" __global__ __aicore__ void log_add_exp_bf16_ut(GM_ADDR x, GM_ADDR y, GM_ADDR out, GM_ADDR tiling);

namespace {
struct GmDeleter {
    void operator()(uint8_t* ptr) const { AscendC::GmFree(ptr); }
};
using GmBuffer = std::unique_ptr<uint8_t, GmDeleter>;

uint16_t Encode(float value, bool bf16)
{
    if (bf16) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return static_cast<uint16_t>(bits >> 16);
    }
    half value16 = static_cast<half>(value);
    uint16_t bits;
    std::memcpy(&bits, &value16, sizeof(bits));
    return bits;
}

float Decode(uint16_t bits, bool bf16)
{
    if (bf16) {
        uint32_t bits32 = static_cast<uint32_t>(bits) << 16;
        float value;
        std::memcpy(&value, &bits32, sizeof(value));
        return value;
    }
    half value16;
    std::memcpy(&value16, &bits, sizeof(bits));
    return static_cast<float>(value16);
}

float Golden(float x, float y)
{
    if (std::isnan(x) || std::isnan(y)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (x == y && std::isinf(x)) {
        return x;
    }
    return std::fmax(x, y) + std::log1p(std::exp(-std::fabs(x - y)));
}
} // namespace

class LogAddExpSpecialValuesTest : public testing::TestWithParam<bool> {
protected:
    void Run(int64_t count, int64_t ubFactor)
    {
        bool bf16 = GetParam();
        float inf = std::numeric_limits<float>::infinity();
        float nan = std::numeric_limits<float>::quiet_NaN();
        const std::array<std::array<float, 2>, 24> pairs = {
            {{inf, inf},    {-inf, -inf},  {inf, -inf},    {-inf, inf},  {inf, 1.0f},       {1.0f, inf},
             {-inf, 1.0f},  {1.0f, -inf},  {nan, inf},     {inf, nan},   {nan, -inf},       {-inf, nan},
             {nan, nan},    {nan, 1.0f},   {1.0f, nan},    {nan, 0.0f},  {0.0f, 0.0f},      {1.0f, 2.0f},
             {-2.0f, 3.0f}, {3.0f, -2.0f}, {-1.0f, -1.0f}, {1.0f, 1.0f}, {-100.0f, -90.0f}, {100.0f, 90.0f}}};
        size_t bytes = ((count * sizeof(uint16_t) + 31) / 32) * 32;
        GmBuffer x(static_cast<uint8_t*>(AscendC::GmAlloc(bytes)));
        GmBuffer y(static_cast<uint8_t*>(AscendC::GmAlloc(bytes)));
        GmBuffer out(static_cast<uint8_t*>(AscendC::GmAlloc(bytes)));
        GmBuffer tiling(static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(LogAddExpTilingData))));
        ASSERT_NE(x, nullptr);
        ASSERT_NE(y, nullptr);
        ASSERT_NE(out, nullptr);
        ASSERT_NE(tiling, nullptr);
        auto* x16 = reinterpret_cast<uint16_t*>(x.get());
        auto* y16 = reinterpret_cast<uint16_t*>(y.get());
        for (int64_t i = 0; i < count; ++i) {
            x16[i] = Encode(pairs[i % pairs.size()][0], bf16);
            y16[i] = Encode(pairs[i % pairs.size()][1], bf16);
        }
        std::memset(out.get(), 0x5A, bytes);
        LogAddExpTilingData tilingData{};
        tilingData.totalLength = count;
        tilingData.blockFactor = count;
        tilingData.ubFactor = ubFactor;
        tilingData.dimNum = 1;
        std::memcpy(tiling.get(), &tilingData, sizeof(tilingData));
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        if (bf16) {
            ICPU_RUN_KF(log_add_exp_bf16_ut, 1, x.get(), y.get(), out.get(), tiling.get());
        } else {
            ICPU_RUN_KF(log_add_exp_fp16_ut, 1, x.get(), y.get(), out.get(), tiling.get());
        }
        auto* out16 = reinterpret_cast<uint16_t*>(out.get());
        for (int64_t i = 0; i < count; ++i) {
            SCOPED_TRACE(i);
            float expected = Golden(Decode(x16[i], bf16), Decode(y16[i], bf16));
            float actual = Decode(out16[i], bf16);
            if (std::isnan(expected)) {
                EXPECT_TRUE(std::isnan(actual));
            } else if (std::isinf(expected)) {
                EXPECT_EQ(actual, expected);
            } else {
                ASSERT_TRUE(std::isfinite(actual));
                EXPECT_NEAR(actual, expected, (bf16 ? 0.02f : 0.005f) * (1.0f + std::fabs(expected)));
            }
        }
    }
};

TEST_P(LogAddExpSpecialValuesTest, ShortTile16) { Run(48, 16); }
TEST_P(LogAddExpSpecialValuesTest, ShortTile32AndTail) { Run(65, 32); }
TEST_P(LogAddExpSpecialValuesTest, ShortTile48AndTail) { Run(65, 48); }
TEST_P(LogAddExpSpecialValuesTest, CompareBoundaryAndTail) { Run(65, 80); }
TEST_P(LogAddExpSpecialValuesTest, MultipleTilesAndTail) { Run(257, 80); }
TEST_P(LogAddExpSpecialValuesTest, DefaultUbTileAndTail) { Run(4913, 4912); }

INSTANTIATE_TEST_SUITE_P(LowPrecision, LogAddExpSpecialValuesTest, testing::Values(false, true));
