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
 * \file test_hypot.cpp
 * \brief hypot kernel UT (tikicpulib CPU mock): FP32 numerically-stable main chain
 *        (Abs->Max/Min->Div->Mul->Adds->Sqrt->Mul) and order-sensitive special-value fix chain
 *        (NaN -> +inf -> +0), OneDim / MultiDimBroadcast scenes
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "tiling_case_executor.h"
#include "../../../op_host/arch22/hypot_tiling_arch22.h"

using namespace NsHypot;

// kernel launch lives in test_hypot_launch.cpp (separate TU: kernel_type.h DT_FLOAT macros
// clash with the ge::DataType enum pulled in by the faker headers)
bool LaunchHypotKernelFp32(uint32_t scene, int64_t tilingKey, size_t blockNum, uint8_t* x1, uint8_t* x2, uint8_t* y,
                           uint8_t* workspace, uint8_t* tiling);
bool LaunchHypotKernelFp16(uint32_t scene, int64_t tilingKey, size_t blockNum, uint8_t* x1, uint8_t* x2, uint8_t* y,
                           uint8_t* workspace, uint8_t* tiling);
bool LaunchHypotKernelBf16(uint32_t scene, int64_t tilingKey, size_t blockNum, uint8_t* x1, uint8_t* x2, uint8_t* y,
                           uint8_t* workspace, uint8_t* tiling);
bool LaunchHypotKernelMixF16Bf16(uint32_t scene, int64_t tilingKey, size_t blockNum, uint8_t* x1, uint8_t* x2,
                                 uint8_t* y, uint8_t* workspace, uint8_t* tiling);
bool LaunchHypotKernelMixF32Bf16(uint32_t scene, int64_t tilingKey, size_t blockNum, uint8_t* x1, uint8_t* x2,
                                 uint8_t* y, uint8_t* workspace, uint8_t* tiling);

namespace {
constexpr int32_t kCoreNum = 40;
constexpr int64_t kUbSize = 196608;

// spec.yaml numerical_tolerance.float32
constexpr double kFp32Rtol = 0.0009765625;     // 2^-10
constexpr double kFp32Atol = 1.52587890625e-5; // 2^-16
// spec.yaml numerical_tolerance.float16 / bfloat16 (rtol == atol per dtype)
constexpr double kFp16Tol = 0.001953125; // 2^-9
constexpr double kBf16Tol = 0.015625;    // 2^-6

inline size_t Align32(size_t size) { return (size + 31U) / 32U * 32U; }

inline float BitsToFloat(uint32_t u)
{
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline uint32_t FloatToBits(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

const float kNan = BitsToFloat(0x7FC00000U);
const float kInf = BitsToFloat(0x7F800000U);
const float kNegInf = BitsToFloat(0xFF800000U);
const float kNegZero = BitsToFloat(0x80000000U);

using TD_ = gert::TilingContextPara::TensorDescription;
} // namespace

// ---------------- FP16/BF16 dtype-branch helpers ----------------

// float -> fp16 bits (round-to-nearest-even), used to build kernel inputs
inline uint16_t FloatToHalfBits(float f)
{
    uint32_t u = FloatToBits(f);
    uint32_t sign = (u >> 16) & 0x8000U;
    int32_t exp = static_cast<int32_t>((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = u & 0x7FFFFFU;
    if ((u & 0x7FFFFFFFU) > 0x7F800000U) {
        return static_cast<uint16_t>(sign | 0x7E00U); // NaN -> canonical quiet NaN
    }
    if (exp <= 0) { // subnormal / zero
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant |= 0x800000U;
        uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t half = mant >> shift;
        uint32_t rem = mant & ((1U << shift) - 1U);
        if (rem > (1U << (shift - 1)) || (rem == (1U << (shift - 1)) && (half & 1U) != 0U)) {
            half++;
        }
        return static_cast<uint16_t>(sign | half);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U); // overflow -> inf
    }
    uint32_t half = (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
    uint32_t rem = mant & 0x1FFFU;
    if (rem > 0x1000U || (rem == 0x1000U && (half & 1U) != 0U)) {
        half++;
    }
    return static_cast<uint16_t>(sign | half);
}

inline float HalfBitsToFloat(uint16_t h)
{
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000U) << 16;
    uint32_t exp = (h >> 10) & 0x1FU;
    uint32_t mant = h & 0x3FFU;
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) {
            u = sign;
        } else {
            int32_t e = -1;
            while ((mant & 0x400U) == 0U) {
                mant <<= 1;
                e--;
            }
            mant &= 0x3FFU;
            u = sign | (static_cast<uint32_t>(e + 15 + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31U) {
        u = sign | 0x7F800000U | (mant << 13);
    } else {
        u = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    return BitsToFloat(u);
}

// float -> bf16 bits, round-to-nearest-even (kernel CAST_RINT contract)
inline uint16_t FloatToBf16BitsRint(float f)
{
    uint32_t u = FloatToBits(f);
    if ((u & 0x7FFFFFFFU) > 0x7F800000U) {
        return 0x7FC0U; // NaN -> canonical quiet NaN
    }
    uint32_t bias = 0x7FFFU + ((u >> 16) & 1U);
    return static_cast<uint16_t>((u + bias) >> 16);
}

// float -> bf16 bits, truncation (not used by the kernel)
inline uint16_t FloatToBf16BitsTrunc(float f) { return static_cast<uint16_t>(FloatToBits(f) >> 16); }

inline float Bf16BitsToFloat(uint16_t b) { return BitsToFloat(static_cast<uint32_t>(b) << 16); }

inline std::vector<uint8_t> ToBytes(const std::vector<uint16_t>& bits)
{
    std::vector<uint8_t> bytes(bits.size() * sizeof(uint16_t));
    std::memcpy(bytes.data(), bits.data(), bytes.size());
    return bytes;
}

inline std::vector<uint16_t> FromBytes(const std::vector<uint8_t>& bytes)
{
    std::vector<uint16_t> bits(bytes.size() / sizeof(uint16_t));
    std::memcpy(bits.data(), bytes.data(), bytes.size() / sizeof(uint16_t) * sizeof(uint16_t));
    return bits;
}

using LaunchFn = bool (*)(uint32_t, int64_t, size_t, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*);

// dtype-generic runner: real host tiling + tikicpulib CPU mock execution (raw byte buffers)
static bool RunHypotKernelRaw(const TD_& x1Desc, const std::vector<uint8_t>& x1Bytes, const TD_& x2Desc,
                              const std::vector<uint8_t>& x2Bytes, const TD_& yDesc, size_t yBytes, uint32_t scene,
                              LaunchFn launch, std::vector<uint8_t>& yOut)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot", {x1Desc, x2Desc}, {yDesc}, &compileInfo);
    TilingInfo info;
    if (!ExecuteTiling(para, info)) {
        return false;
    }
    // pin the host-side TilingKey (scene-only, dtype rides the per-combination binary) against
    // the template the launcher will instantiate, so a host reroute can never silently desync
    // from the mock kernel binary
    const int64_t hostScene = info.tilingKey & 0xFF;
    if (hostScene != static_cast<int64_t>(scene)) {
        ADD_FAILURE() << "host TilingKey scene " << hostScene << " != requested launch scene " << scene;
        return false;
    }

    yOut.assign(yBytes, 0);

    uint8_t* x1 = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(x1Bytes.size())));
    uint8_t* x2 = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(x2Bytes.size())));
    uint8_t* y = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(yBytes)));
    size_t wsSize = info.workspaceSizes.empty() ? 0 : static_cast<size_t>(info.workspaceSizes[0]);
    uint8_t* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(wsSize > 0 ? wsSize : 32)));
    uint8_t* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(info.tilingDataSize)));
    std::memcpy(x1, x1Bytes.data(), x1Bytes.size());
    std::memcpy(x2, x2Bytes.data(), x2Bytes.size());
    std::memset(y, 0, yBytes);
    std::memcpy(tiling, info.tilingData.get(), info.tilingDataSize);

    bool ok = launch(scene, info.tilingKey, info.blockNum, x1, x2, y, workspace, tiling);
    if (ok) {
        std::memcpy(yOut.data(), y, yBytes);
    }
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    return ok;
}

static bool RunHypotKernelFp16(const TD_& x1Desc, const std::vector<uint16_t>& x1Bits, const TD_& x2Desc,
                               const std::vector<uint16_t>& x2Bits, const TD_& yDesc, size_t yNumel, uint32_t scene,
                               std::vector<uint16_t>& yBits)
{
    std::vector<uint8_t> yBytes;
    bool ok = RunHypotKernelRaw(x1Desc, ToBytes(x1Bits), x2Desc, ToBytes(x2Bits), yDesc, yNumel * sizeof(uint16_t),
                                scene, LaunchHypotKernelFp16, yBytes);
    if (ok) {
        yBits = FromBytes(yBytes);
    }
    return ok;
}

static bool RunHypotKernelBf16(const TD_& x1Desc, const std::vector<uint16_t>& x1Bits, const TD_& x2Desc,
                               const std::vector<uint16_t>& x2Bits, const TD_& yDesc, size_t yNumel, uint32_t scene,
                               std::vector<uint16_t>& yBits)
{
    std::vector<uint8_t> yBytes;
    bool ok = RunHypotKernelRaw(x1Desc, ToBytes(x1Bits), x2Desc, ToBytes(x2Bits), yDesc, yNumel * sizeof(uint16_t),
                                scene, LaunchHypotKernelBf16, yBytes);
    if (ok) {
        yBits = FromBytes(yBytes);
    }
    return ok;
}

// golden compare for 2B dtypes: golden source is the exact (already rounded) input values
static void ExpectNearGolden16(const std::vector<uint16_t>& y, const std::vector<float>& x1,
                               const std::vector<float>& x2, double tol, bool isBf16, const std::string& tag)
{
    ASSERT_EQ(y.size(), x1.size());
    ASSERT_EQ(y.size(), x2.size());
    for (size_t i = 0; i < y.size(); i++) {
        float yVal = isBf16 ? Bf16BitsToFloat(y[i]) : HalfBitsToFloat(y[i]);
        double ref = std::hypot(static_cast<double>(x1[i]), static_cast<double>(x2[i]));
        double diff = std::fabs(static_cast<double>(yVal) - ref);
        double limit = tol + tol * std::fabs(ref);
        EXPECT_LE(diff, limit) << tag << " idx=" << i << " x1=" << x1[i] << " x2=" << x2[i] << " y=" << yVal
                               << " ref=" << ref;
    }
}

// NaN check that works for both fp16 (exp 0x1F) and bf16 (exp 0xFF) bit patterns
inline bool IsNan16(uint16_t bits, bool isBf16)
{
    uint16_t absBits = bits & 0x7FFFU;
    return isBf16 ? (absBits > 0x7F80U) : (absBits > 0x7C00U);
}

// tikicpulib mock Cast saturates inf and flushes NaN, so UT asserts membership in the mock/NPU
// plausible outcome set; the FP32 special-chain test pins the fix-chain control flow bitwise.
inline bool IsInfOutcome16(uint16_t bits, bool isBf16)
{
    uint16_t absBits = bits & 0x7FFFU;
    return absBits == (isBf16 ? 0x7F80U : 0x7C00U) || absBits == (isBf16 ? 0x7F7FU : 0x7BFFU);
}
inline bool IsNanOutcome16(uint16_t bits, bool isBf16)
{
    return IsNan16(bits, isBf16) || IsInfOutcome16(bits, isBf16) || bits == 0x0000U;
}

class HypotKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HypotKernelTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "HypotKernelTest TearDown" << std::endl; }
};

// run the kernel through real host tiling (ExecuteTiling) + tikicpulib CPU mock execution
static bool RunHypotKernel(const TD_& x1Desc, const std::vector<float>& x1Data, const TD_& x2Desc,
                           const std::vector<float>& x2Data, const TD_& yDesc, size_t yNumel, uint32_t scene,
                           std::vector<float>& yOut)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot", {x1Desc, x2Desc}, {yDesc}, &compileInfo);
    TilingInfo info;
    if (!ExecuteTiling(para, info)) {
        return false;
    }
    // pin the host-side TilingKey (scene-only, dtype rides the per-combination binary) against
    // the template the launcher will instantiate, so a host reroute can never silently desync
    // from the mock kernel binary
    const int64_t hostScene = info.tilingKey & 0xFF;
    if (hostScene != static_cast<int64_t>(scene)) {
        ADD_FAILURE() << "host TilingKey scene " << hostScene << " != requested launch scene " << scene;
        return false;
    }

    yOut.assign(yNumel, 0.0f);

    uint8_t* x1 = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(x1Data.size() * sizeof(float))));
    uint8_t* x2 = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(x2Data.size() * sizeof(float))));
    uint8_t* y = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(yNumel * sizeof(float))));
    size_t wsSize = info.workspaceSizes.empty() ? 0 : static_cast<size_t>(info.workspaceSizes[0]);
    uint8_t* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(wsSize > 0 ? wsSize : 32)));
    uint8_t* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(info.tilingDataSize)));
    std::memcpy(x1, x1Data.data(), x1Data.size() * sizeof(float));
    std::memcpy(x2, x2Data.data(), x2Data.size() * sizeof(float));
    std::memset(y, 0, yNumel * sizeof(float));
    std::memcpy(tiling, info.tilingData.get(), info.tilingDataSize);

    bool ok = LaunchHypotKernelFp32(scene, info.tilingKey, info.blockNum, x1, x2, y, workspace, tiling);
    if (ok) {
        std::memcpy(yOut.data(), y, yNumel * sizeof(float));
    }
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    return ok;
}

static void ExpectNearGolden(const std::vector<float>& y, const std::vector<float>& x1, const std::vector<float>& x2,
                             const std::string& tag)
{
    ASSERT_EQ(y.size(), x1.size());
    ASSERT_EQ(y.size(), x2.size());
    for (size_t i = 0; i < y.size(); i++) {
        double ref = std::hypot(static_cast<double>(x1[i]), static_cast<double>(x2[i]));
        double diff = std::fabs(static_cast<double>(y[i]) - ref);
        double limit = kFp32Atol + kFp32Rtol * std::fabs(ref);
        EXPECT_LE(diff, limit) << tag << " idx=" << i << " x1=" << x1[i] << " x2=" << x2[i] << " y=" << y[i]
                               << " ref=" << ref;
    }
}

// main chain accuracy: multi-tile (ubFormer=256 -> 4 tiles) + multi-core (4 cores) FP32 path
TEST_F(HypotKernelTest, onedim_main_chain_values)
{
    const size_t n = 1024;
    std::vector<float> x1(n), x2(n);
    for (size_t i = 0; i < n; i++) {
        x1[i] = 0.5f + static_cast<float>(i % 97) * 0.25f;
        x2[i] = 1.0f + static_cast<float>(i % 89) * 0.125f;
        if ((i & 1) != 0) {
            x1[i] = -x1[i]; // sign must not affect the result (Abs first)
        }
    }
    std::vector<float> y;
    ASSERT_TRUE(RunHypotKernel({{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND}, x1,
                               {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND}, x2,
                               {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND}, n, HYPOT_SCENE_ONE_DIM, y));
    ExpectNearGolden(y, x1, x2, "onedim_main_chain_values");
}

// numerical stability: (1e30, 1e30) must stay finite (~1.414e30), no premature square overflow
TEST_F(HypotKernelTest, onedim_stable_large_1e30)
{
    const size_t n = 256;
    std::vector<float> x1(n, 1.0e30f), x2(n, 1.0e30f);
    std::vector<float> y;
    ASSERT_TRUE(RunHypotKernel({{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND}, x1,
                               {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND}, x2,
                               {{{256}, {256}}, ge::DT_FLOAT, ge::FORMAT_ND}, n, HYPOT_SCENE_ONE_DIM, y));
    for (size_t i = 0; i < n; i++) {
        EXPECT_TRUE(std::isfinite(y[i])) << "idx=" << i << " y=" << y[i];
    }
    ExpectNearGolden(y, x1, x2, "onedim_stable_large_1e30");
}

// magnitude gap: (1e4, 1e-4) -> ~1e4, small component must not be lost
TEST_F(HypotKernelTest, onedim_magnitude_gap)
{
    const size_t n = 64;
    std::vector<float> x1(n, 1.0e4f), x2(n, 1.0e-4f);
    std::vector<float> y;
    ASSERT_TRUE(RunHypotKernel({{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND}, x1,
                               {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND}, x2,
                               {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND}, n, HYPOT_SCENE_ONE_DIM, y));
    ExpectNearGolden(y, x1, x2, "onedim_magnitude_gap");
}

// subnormal inputs: square must not underflow away (r = mn/mx keeps ratio in [0,1])
TEST_F(HypotKernelTest, onedim_subnormal)
{
    const float sub = BitsToFloat(0x00400000U); // ~5.877e-39, subnormal in fp32
    const size_t n = 64;
    std::vector<float> x1(n, sub), x2(n, sub);
    std::vector<float> y;
    ASSERT_TRUE(RunHypotKernel({{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND}, x1,
                               {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND}, x2,
                               {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND}, n, HYPOT_SCENE_ONE_DIM, y));
    for (size_t i = 0; i < n; i++) {
        double ref = std::hypot(static_cast<double>(sub), static_cast<double>(sub));
        EXPECT_GT(y[i], 0.0f) << "subnormal underflow at idx=" << i;
        double rel = std::fabs(static_cast<double>(y[i]) - ref) / std::fabs(ref);
        EXPECT_LE(rel, 1e-3) << "idx=" << i << " y=" << y[i] << " ref=" << ref;
    }
}

// special-value fix chain order: NaN fix first, then inf (inf wins over NaN), then +0 (both zero)
TEST_F(HypotKernelTest, onedim_special_value_fix_chain)
{
    const size_t n = 64;
    std::vector<float> x1(n, 3.0f), x2(n, 4.0f); // tail elements: plain (3,4) -> 5
    x1[0] = kNan;
    x2[0] = 1.5f; // -> NaN
    x1[1] = 1.5f;
    x2[1] = kNan; // -> NaN
    x1[2] = kNan;
    x2[2] = kNan; // -> NaN
    x1[3] = kInf;
    x2[3] = kNan; // -> +inf (inf priority over NaN)
    x1[4] = kNan;
    x2[4] = kNegInf; // -> +inf
    x1[5] = kNegInf;
    x2[5] = 0.0f; // -> +inf (output always non-negative)
    x1[6] = kInf;
    x2[6] = 2.0f; // -> +inf
    x1[7] = kNegInf;
    x2[7] = kNegInf; // -> +inf
    x1[8] = 0.0f;
    x2[8] = kNegZero; // -> +0 (bitwise)
    x1[9] = kNegZero;
    x2[9] = kNegZero; // -> +0 (bitwise)
    x1[10] = 0.0f;
    x2[10] = 5.0f; // -> 5 (one side zero degenerates to abs of the other)
    x1[11] = 5.0f;
    x2[11] = 0.0f; // -> 5
    std::vector<float> y;
    ASSERT_TRUE(RunHypotKernel({{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND}, x1,
                               {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND}, x2,
                               {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND}, n, HYPOT_SCENE_ONE_DIM, y));

    for (int i = 0; i <= 2; i++) {
        EXPECT_TRUE(std::isnan(y[i])) << "NaN propagation broken at idx=" << i;
    }
    for (int i = 3; i <= 7; i++) {
        EXPECT_EQ(FloatToBits(y[i]), 0x7F800000U) << "inf priority/non-negative broken at idx=" << i;
    }
    EXPECT_EQ(FloatToBits(y[8]), 0x00000000U) << "double-zero must be +0";
    EXPECT_EQ(FloatToBits(y[9]), 0x00000000U) << "double-zero must be +0";
    EXPECT_NEAR(y[10], 5.0, 1e-6);
    EXPECT_NEAR(y[11], 5.0, 1e-6);
    for (size_t i = 12; i < n; i++) {
        EXPECT_NEAR(y[i], 5.0, 5.0 * kFp32Rtol + kFp32Atol) << "idx=" << i;
    }
}

// scalar input: scalarFlag bit0 -> kernel Duplicate expansion path (no DataCopyPad for x1)
TEST_F(HypotKernelTest, onedim_scalar_input_duplicate)
{
    const size_t n = 64;
    std::vector<float> x1(1, 3.0f);
    std::vector<float> x2(n, 4.0f);
    std::vector<float> y;
    ASSERT_TRUE(RunHypotKernel({{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}, x1,
                               {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND}, x2,
                               {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND}, n, HYPOT_SCENE_ONE_DIM, y));
    for (size_t i = 0; i < n; i++) {
        EXPECT_NEAR(y[i], 5.0, 5.0 * kFp32Rtol + kFp32Atol) << "idx=" << i;
    }
}

// VECIDX scene: [3,1] x [1,4] -> [3,4] cross broadcast;
// row 2 carries 1e30 to check the stable chain on the Gather-expansion path
TEST_F(HypotKernelTest, vecidx_brc_3x1_1x4)
{
    std::vector<float> x1 = {3.0f, 5.0f, 1.0e30f};
    std::vector<float> x2 = {4.0f, 12.0f, 0.0f, -1.0f};
    std::vector<float> y;
    ASSERT_TRUE(RunHypotKernel({{{3, 1}, {3, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}, x1,
                               {{{1, 4}, {1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, x2,
                               {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, 12, HYPOT_SCENE_MULTI_DIM_VECIDX, y));
    ASSERT_EQ(y.size(), 12U);
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 4; j++) {
            double ref = std::hypot(static_cast<double>(x1[i]), static_cast<double>(x2[j]));
            double diff = std::fabs(static_cast<double>(y[i * 4 + j]) - ref);
            double limit = kFp32Atol + kFp32Rtol * std::fabs(ref);
            EXPECT_LE(diff, limit) << "row=" << i << " col=" << j << " y=" << y[i * 4 + j] << " ref=" << ref;
        }
    }
    EXPECT_TRUE(std::isfinite(y[8])) << "1e30 broadcast row must stay finite";
}

// VECIDX scene: [2,3] x [3] -> [2,3]; x1 contiguous, x2 row repeat
TEST_F(HypotKernelTest, vecidx_brc_leftpad_2x3_3)
{
    std::vector<float> x1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> x2 = {3.0f, 4.0f, 0.0f};
    std::vector<float> y;
    ASSERT_TRUE(RunHypotKernel({{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}, x1,
                               {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND}, x2,
                               {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}, 6, HYPOT_SCENE_MULTI_DIM_VECIDX, y));
    ASSERT_EQ(y.size(), 6U);
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            double ref = std::hypot(static_cast<double>(x1[i * 3 + j]), static_cast<double>(x2[j]));
            double diff = std::fabs(static_cast<double>(y[i * 3 + j]) - ref);
            double limit = kFp32Atol + kFp32Rtol * std::fabs(ref);
            EXPECT_LE(diff, limit) << "row=" << i << " col=" << j << " y=" << y[i * 3 + j] << " ref=" << ref;
        }
    }
}

// ---------------- FP16/BF16 dtype branches + regression cases ----------------

// VECIDX scene fp16 coverage: [4,1,8] x [4,6,1] -> [4,6,8] cross broadcast through the
// vectorized-index Gather expansion
TEST_F(HypotKernelTest, vecidx_fp16_cross_4x1x8_4x6x1)
{
    std::vector<uint16_t> x1(32), x2(24);
    std::vector<float> x1f(32), x2f(24);
    for (size_t i = 0; i < 32; i++) {
        x1[i] = FloatToHalfBits(1.0f + static_cast<float>(i % 9) * 0.5f);
        x1f[i] = HalfBitsToFloat(x1[i]);
    }
    for (size_t j = 0; j < 24; j++) {
        x2[j] = FloatToHalfBits(2.0f + static_cast<float>(j % 7) * 0.25f);
        x2f[j] = HalfBitsToFloat(x2[j]);
    }
    std::vector<uint16_t> y;
    ASSERT_TRUE(RunHypotKernelFp16({{{4, 1, 8}, {4, 1, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, x1,
                                   {{{4, 6, 1}, {4, 6, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}, x2,
                                   {{{4, 6, 8}, {4, 6, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, 192,
                                   HYPOT_SCENE_MULTI_DIM_VECIDX, y));
    ASSERT_EQ(y.size(), 192U);
    for (size_t r = 0; r < 4; r++) {
        for (size_t a = 0; a < 6; a++) {
            for (size_t i = 0; i < 8; i++) {
                float yVal = HalfBitsToFloat(y[(r * 6 + a) * 8 + i]);
                double ref = std::hypot(static_cast<double>(x1f[r * 8 + i]), static_cast<double>(x2f[r * 6 + a]));
                double limit = kFp16Tol + kFp16Tol * std::fabs(ref);
                EXPECT_LE(std::fabs(static_cast<double>(yVal) - ref), limit) << "r=" << r << " a=" << a << " i=" << i;
            }
        }
    }
}

// FP16 OneDim main chain: Cast(CAST_NONE) -> FP32 stable chain -> Cast back, fp16 tolerance 2^-9
TEST_F(HypotKernelTest, fp16_onedim_main_chain)
{
    const size_t n = 1024;
    std::vector<uint16_t> x1(n), x2(n);
    std::vector<float> x1f(n), x2f(n);
    for (size_t i = 0; i < n; i++) {
        float v1 = 0.5f + static_cast<float>(i % 97) * 0.25f;
        float v2 = 1.0f + static_cast<float>(i % 89) * 0.125f;
        if ((i & 1) != 0) {
            v1 = -v1;
        }
        x1[i] = FloatToHalfBits(v1);
        x2[i] = FloatToHalfBits(v2);
        x1f[i] = HalfBitsToFloat(x1[i]);
        x2f[i] = HalfBitsToFloat(x2[i]);
    }
    std::vector<uint16_t> y;
    ASSERT_TRUE(RunHypotKernelFp16({{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, x1,
                                   {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, x2,
                                   {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, n, HYPOT_SCENE_ONE_DIM, y));
    ExpectNearGolden16(y, x1f, x2f, kFp16Tol, false, "fp16_onedim_main_chain");
}

// FP16 special-value fix chain (NaN trim e2e contract at kernel level): NaN propagates through
// the trimmed chain, inf keeps priority over NaN, double-zero is bitwise +0. Cast-back of
// inf/NaN is a mock fidelity gap (see IsInfOutcome16/IsNanOutcome16); NPU truth is ST-covered.
TEST_F(HypotKernelTest, fp16_onedim_special_values)
{
    const uint16_t hNan = 0x7E00U, hInf = 0x7C00U, hNegInf = 0xFC00U, hNegZero = 0x8000U;
    const uint16_t h15 = FloatToHalfBits(1.5f), h2 = FloatToHalfBits(2.0f);
    const uint16_t h3 = FloatToHalfBits(3.0f), h4 = FloatToHalfBits(4.0f), h5 = FloatToHalfBits(5.0f);
    const uint16_t hZero = 0x0000U;
    const uint16_t hBig = FloatToHalfBits(60000.0f); // hypot(60000,60000) overflows fp16 -> inf
    const size_t n = 64;
    std::vector<uint16_t> x1(n, h3), x2(n, h4);
    x1[0] = hNan;
    x2[0] = h15;
    x1[1] = h15;
    x2[1] = hNan;
    x1[2] = hNan;
    x2[2] = hNan;
    x1[3] = hInf;
    x2[3] = hNan;
    x1[4] = hNan;
    x2[4] = hNegInf;
    x1[5] = hNegInf;
    x2[5] = hZero;
    x1[6] = hInf;
    x2[6] = h2;
    x1[7] = hNegInf;
    x2[7] = hNegInf;
    x1[8] = hZero;
    x2[8] = hNegZero;
    x1[9] = hNegZero;
    x2[9] = hNegZero;
    x1[10] = hZero;
    x2[10] = h5;
    x1[11] = h5;
    x2[11] = hZero;
    x1[12] = hBig; // finite overflow -> inf (spec: only when the math result exceeds the dtype range)
    x2[12] = hBig;
    std::vector<uint16_t> y;
    ASSERT_TRUE(RunHypotKernelFp16({{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, x1,
                                   {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, x2,
                                   {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, n, HYPOT_SCENE_ONE_DIM, y));
    for (int i = 0; i <= 2; i++) {
        EXPECT_TRUE(IsNanOutcome16(y[i], false))
            << "fp16 NaN case broken at idx=" << i << " bits=0x" << std::hex << y[i];
    }
    for (int i = 3; i <= 7; i++) {
        EXPECT_TRUE(IsInfOutcome16(y[i], false))
            << "fp16 inf case broken at idx=" << i << " bits=0x" << std::hex << y[i];
    }
    EXPECT_EQ(y[8], 0x0000U) << "fp16 double-zero must be +0";
    EXPECT_EQ(y[9], 0x0000U) << "fp16 double-zero must be +0";
    EXPECT_EQ(y[10], h5);
    EXPECT_EQ(y[11], h5);
    EXPECT_TRUE(IsInfOutcome16(y[12], false)) << "fp16 overflow must yield inf, bits=0x" << std::hex << y[12];
    for (size_t i = 13; i < n; i++) {
        EXPECT_EQ(y[i], h5) << "idx=" << i; // hypot(3,4)=5 is exact in fp16
    }
}

// BF16 OneDim round mode: v*sqrt(2) is not bf16-representable, so CAST_RINT (RNE) decides the bits
// (discriminates a truncation regression). Remaining elements use the FP32 chain with 2^-6 tolerance.
TEST_F(HypotKernelTest, bf16_onedim_cast_rint_round_mode)
{
    const size_t n = 256;
    std::vector<uint16_t> x1(n), x2(n);
    std::vector<float> x1f(n), x2f(n);
    // discriminating prefix: v*sqrt(2) mantissa fractions straddle the bf16 rounding boundary
    const float kDisc[8] = {1.5f, 2.5f, 3.5f, 0.75f, 5.5f, 1.25f, 0.375f, 6.0f};
    for (size_t i = 0; i < 8; i++) {
        x1[i] = FloatToBf16BitsRint(kDisc[i]);
        x2[i] = x1[i];
        x1f[i] = Bf16BitsToFloat(x1[i]);
        x2f[i] = x1f[i];
    }
    for (size_t i = 8; i < n; i++) {
        float v1 = 0.3f + static_cast<float>(i % 50) * 0.17f;
        float v2 = 0.7f + static_cast<float>(i % 43) * 0.11f;
        x1[i] = FloatToBf16BitsRint(v1);
        x2[i] = FloatToBf16BitsRint(v2);
        x1f[i] = Bf16BitsToFloat(x1[i]);
        x2f[i] = Bf16BitsToFloat(x2[i]);
    }
    std::vector<uint16_t> y;
    ASSERT_TRUE(RunHypotKernelBf16({{{256}, {256}}, ge::DT_BF16, ge::FORMAT_ND}, x1,
                                   {{{256}, {256}}, ge::DT_BF16, ge::FORMAT_ND}, x2,
                                   {{{256}, {256}}, ge::DT_BF16, ge::FORMAT_ND}, n, HYPOT_SCENE_ONE_DIM, y));
    size_t discriminating = 0;
    for (size_t i = 0; i < 8; i++) {
        float f32Golden = static_cast<float>(std::hypot(static_cast<double>(x1f[i]), static_cast<double>(x2f[i])));
        uint16_t expectRint = FloatToBf16BitsRint(f32Golden);
        if (expectRint != FloatToBf16BitsTrunc(f32Golden)) {
            discriminating++;
        }
        EXPECT_EQ(y[i], expectRint) << "CAST_RINT round mode broken at idx=" << i << " v=" << x1f[i];
    }
    EXPECT_GT(discriminating, 0U) << "test lost its CAST_RINT-vs-truncation discrimination power";
    ExpectNearGolden16(y, x1f, x2f, kBf16Tol, true, "bf16_onedim_cast_rint_round_mode");
}

// BF16 special-value fix chain: NaN propagation, inf priority, bitwise +0
TEST_F(HypotKernelTest, bf16_onedim_special_values)
{
    const uint16_t bNan = 0x7FC0U, bInf = 0x7F80U, bNegInf = 0xFF80U, bNegZero = 0x8000U;
    const uint16_t b15 = FloatToBf16BitsRint(1.5f), b2 = FloatToBf16BitsRint(2.0f);
    const uint16_t b3 = FloatToBf16BitsRint(3.0f), b4 = FloatToBf16BitsRint(4.0f), b5 = FloatToBf16BitsRint(5.0f);
    const uint16_t bZero = 0x0000U;
    const uint16_t bBig = FloatToBf16BitsRint(3.0e38f); // hypot(3e38,3e38) overflows bf16 -> inf
    const size_t n = 64;
    std::vector<uint16_t> x1(n, b3), x2(n, b4);
    x1[0] = bNan;
    x2[0] = b15;
    x1[1] = b15;
    x2[1] = bNan;
    x1[2] = bNan;
    x2[2] = bNan;
    x1[3] = bInf;
    x2[3] = bNan;
    x1[4] = bNan;
    x2[4] = bNegInf;
    x1[5] = bNegInf;
    x2[5] = bZero;
    x1[6] = bInf;
    x2[6] = b2;
    x1[7] = bNegInf;
    x2[7] = bNegInf;
    x1[8] = bZero;
    x2[8] = bNegZero;
    x1[9] = bNegZero;
    x2[9] = bNegZero;
    x1[10] = bZero;
    x2[10] = b5;
    x1[11] = b5;
    x2[11] = bZero;
    x1[12] = bBig; // finite overflow -> inf (spec: only when the math result exceeds the dtype range)
    x2[12] = bBig;
    std::vector<uint16_t> y;
    ASSERT_TRUE(RunHypotKernelBf16({{{64}, {64}}, ge::DT_BF16, ge::FORMAT_ND}, x1,
                                   {{{64}, {64}}, ge::DT_BF16, ge::FORMAT_ND}, x2,
                                   {{{64}, {64}}, ge::DT_BF16, ge::FORMAT_ND}, n, HYPOT_SCENE_ONE_DIM, y));
    for (int i = 0; i <= 2; i++) {
        EXPECT_TRUE(IsNanOutcome16(y[i], true))
            << "bf16 NaN case broken at idx=" << i << " bits=0x" << std::hex << y[i];
    }
    for (int i = 3; i <= 7; i++) {
        EXPECT_TRUE(IsInfOutcome16(y[i], true))
            << "bf16 inf case broken at idx=" << i << " bits=0x" << std::hex << y[i];
    }
    EXPECT_EQ(y[8], 0x0000U) << "bf16 double-zero must be +0";
    EXPECT_EQ(y[9], 0x0000U) << "bf16 double-zero must be +0";
    EXPECT_EQ(y[10], b5);
    EXPECT_EQ(y[11], b5);
    EXPECT_TRUE(IsInfOutcome16(y[12], true)) << "bf16 overflow must yield inf, bits=0x" << std::hex << y[12];
    for (size_t i = 13; i < n; i++) {
        EXPECT_EQ(y[i], b5) << "idx=" << i;
    }
}

// BF16 scalar input (scalarFlag bit1): ReadScalarAsFloat bit-splice path (bisheng has no scalar bf16
// cast) + Duplicate expansion; hypot(3,4)=5 is exact in bf16
TEST_F(HypotKernelTest, bf16_onedim_scalar_x2)
{
    const size_t n = 64;
    std::vector<uint16_t> x1(n, FloatToBf16BitsRint(3.0f));
    std::vector<uint16_t> x2(1, FloatToBf16BitsRint(4.0f));
    std::vector<uint16_t> y;
    ASSERT_TRUE(RunHypotKernelBf16({{{64}, {64}}, ge::DT_BF16, ge::FORMAT_ND}, x1,
                                   {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND}, x2,
                                   {{{64}, {64}}, ge::DT_BF16, ge::FORMAT_ND}, n, HYPOT_SCENE_ONE_DIM, y));
    const uint16_t b5 = FloatToBf16BitsRint(5.0f);
    for (size_t i = 0; i < n; i++) {
        EXPECT_EQ(y[i], b5) << "idx=" << i;
    }
}

// FP16 scalar input (scalarFlag bit0) through the OneDim Duplicate expansion path
TEST_F(HypotKernelTest, fp16_onedim_scalar_x1)
{
    const size_t n = 64;
    std::vector<uint16_t> x1(1, FloatToHalfBits(3.0f));
    std::vector<uint16_t> x2(n, FloatToHalfBits(4.0f));
    std::vector<uint16_t> y;
    ASSERT_TRUE(RunHypotKernelFp16({{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND}, x1,
                                   {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, x2,
                                   {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, n, HYPOT_SCENE_ONE_DIM, y));
    const uint16_t h5 = FloatToHalfBits(5.0f);
    for (size_t i = 0; i < n; i++) {
        EXPECT_EQ(y[i], h5) << "idx=" << i;
    }
}

// FP16 MultiDim NONE per-row cast regression: raw rows are pitched at ceil(innerLen,16)=16 while
// FP32 rows use pitch 8; shape [342,3] keeps the case on MULTI_DIM_BRC with innerLen=3.
TEST_F(HypotKernelTest, fp16_multidim_none_row_perrow_cast)
{
    const size_t rows = 342;
    const float x2Vals[3] = {4.0f, 3.0f, 2.5f};
    std::vector<uint16_t> x1(rows * 3), x2(3);
    std::vector<float> x1f(rows * 3), x2f(3);
    for (size_t i = 0; i < rows * 3; i++) {
        x1[i] = FloatToHalfBits(0.5f + static_cast<float>(i % 11) * 0.5f);
        x1f[i] = HalfBitsToFloat(x1[i]);
    }
    for (size_t j = 0; j < 3; j++) {
        x2[j] = FloatToHalfBits(x2Vals[j]);
        x2f[j] = HalfBitsToFloat(x2[j]);
    }
    std::vector<uint16_t> y;
    ASSERT_TRUE(RunHypotKernelFp16(
        {{{342, 3}, {342, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND}, x1, {{{1, 3}, {1, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        x2, {{{342, 3}, {342, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND}, 1026, HYPOT_SCENE_MULTI_DIM_BRC, y));
    ASSERT_EQ(y.size(), 1026U);
    for (size_t i = 0; i < 342; i++) {
        for (size_t j = 0; j < 3; j++) {
            float yVal = HalfBitsToFloat(y[i * 3 + j]);
            double ref = std::hypot(static_cast<double>(x1f[i * 3 + j]), static_cast<double>(x2f[j]));
            double limit = kFp16Tol + kFp16Tol * std::fabs(ref);
            EXPECT_LE(std::fabs(static_cast<double>(yVal) - ref), limit)
                << "row=" << i << " col=" << j << " y=" << yVal << " ref=" << ref;
        }
    }
}

// BF16 MultiDim: [257,1]x[1,4] -> x1 BRC_COL splat, x2 BRC_ROW with B=257; cast-back runs per row.
TEST_F(HypotKernelTest, bf16_multidim_col_row)
{
    const size_t rows = 257;
    const float x2Vals[4] = {4.0f, 12.0f, 0.75f, -1.0f};
    std::vector<uint16_t> x1(rows), x2(4);
    std::vector<float> x1f(rows), x2f(4);
    for (size_t i = 0; i < rows; i++) {
        float v = 3.0f + static_cast<float>(i % 5);
        x1[i] = FloatToBf16BitsRint(v);
        x1f[i] = Bf16BitsToFloat(x1[i]);
    }
    for (size_t j = 0; j < 4; j++) {
        x2[j] = FloatToBf16BitsRint(x2Vals[j]);
        x2f[j] = Bf16BitsToFloat(x2[j]);
    }
    std::vector<uint16_t> y;
    ASSERT_TRUE(RunHypotKernelBf16(
        {{{257, 1}, {257, 1}}, ge::DT_BF16, ge::FORMAT_ND}, x1, {{{1, 4}, {1, 4}}, ge::DT_BF16, ge::FORMAT_ND}, x2,
        {{{257, 4}, {257, 4}}, ge::DT_BF16, ge::FORMAT_ND}, rows * 4, HYPOT_SCENE_MULTI_DIM_BRC, y));
    ASSERT_EQ(y.size(), rows * 4);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < 4; j++) {
            float yVal = Bf16BitsToFloat(y[i * 4 + j]);
            double ref = std::hypot(static_cast<double>(x1f[i]), static_cast<double>(x2f[j]));
            double limit = kBf16Tol + kBf16Tol * std::fabs(ref);
            EXPECT_LE(std::fabs(static_cast<double>(yVal) - ref), limit)
                << "row=" << i << " col=" << j << " y=" << yVal << " ref=" << ref;
        }
    }
}

// multi-block vcopy regression: BRC_ROW expansion with I=16 -> CopyChunks moves 16 elements (two 32B blocks)
// in ONE repeat; CopyRepeatParams{0,0,0,0} semantics would leave blocks 2..n overlapping (only the first
// 8 elements correct). Every element of both rows is asserted.
TEST_F(HypotKernelTest, fp32_multidim_vcopy_multiblock)
{
    std::vector<float> x1(1040), x2(16);
    for (size_t i = 0; i < 1040; i++) {
        x1[i] = 0.5f + static_cast<float>(i % 13) * 0.5f;
    }
    for (size_t j = 0; j < 16; j++) {
        x2[j] = 1.0f + static_cast<float>(j) * 0.25f;
    }
    std::vector<float> y;
    ASSERT_TRUE(RunHypotKernel(
        {{{65, 16}, {65, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}, x1, {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND}, x2,
        {{{65, 16}, {65, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}, 1040, HYPOT_SCENE_MULTI_DIM_BRC, y));
    ASSERT_EQ(y.size(), 1040U);
    for (size_t i = 0; i < 65; i++) {
        for (size_t j = 0; j < 16; j++) {
            double ref = std::hypot(static_cast<double>(x1[i * 16 + j]), static_cast<double>(x2[j]));
            double limit = kFp32Atol + kFp32Rtol * std::fabs(ref);
            EXPECT_LE(std::fabs(static_cast<double>(y[i * 16 + j]) - ref), limit)
                << "row=" << i << " col=" << j << " y=" << y[i * 16 + j] << " ref=" << ref
                << " (vcopy multi-block coverage)";
        }
    }
}

// multi-block vcopy extension: I=72 -> CopyChunks loops over 64-element chunks (64 + 8)
TEST_F(HypotKernelTest, fp32_multidim_vcopy_chunk_loop)
{
    std::vector<float> x1(1080), x2(72);
    for (size_t i = 0; i < 1080; i++) {
        x1[i] = 0.25f + static_cast<float>(i % 17) * 0.5f;
    }
    for (size_t j = 0; j < 72; j++) {
        x2[j] = 0.5f + static_cast<float>(j % 23) * 0.25f;
    }
    std::vector<float> y;
    ASSERT_TRUE(RunHypotKernel(
        {{{15, 72}, {15, 72}}, ge::DT_FLOAT, ge::FORMAT_ND}, x1, {{{72}, {72}}, ge::DT_FLOAT, ge::FORMAT_ND}, x2,
        {{{15, 72}, {15, 72}}, ge::DT_FLOAT, ge::FORMAT_ND}, 1080, HYPOT_SCENE_MULTI_DIM_BRC, y));
    ASSERT_EQ(y.size(), 1080U);
    for (size_t i = 0; i < 15; i++) {
        for (size_t j = 0; j < 72; j++) {
            double ref = std::hypot(static_cast<double>(x1[i * 72 + j]), static_cast<double>(x2[j]));
            double limit = kFp32Atol + kFp32Rtol * std::fabs(ref);
            EXPECT_LE(std::fabs(static_cast<double>(y[i * 72 + j]) - ref), limit) << "row=" << i << " col=" << j;
        }
    }
}

// GM out-of-bounds regression: ubSplitAxis=1, multi-core split -> flatIdx uses the dp*ubOuter
// divisor; [8,5000]x[5000] = 80 tiles over 40 cores.
TEST_F(HypotKernelTest, fp32_multidim_ubouter_division_chain)
{
    const size_t rows = 8, cols = 5000;
    std::vector<float> x1(rows * cols), x2(cols);
    for (size_t i = 0; i < rows * cols; i++) {
        x1[i] = 0.5f + static_cast<float>(i % 97) * 0.25f;
        if ((i & 1) != 0) {
            x1[i] = -x1[i];
        }
    }
    for (size_t j = 0; j < cols; j++) {
        x2[j] = 1.0f + static_cast<float>(j % 89) * 0.125f;
    }
    std::vector<float> y;
    ASSERT_TRUE(RunHypotKernel(
        {{{8, 5000}, {8, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND}, x1, {{{5000}, {5000}}, ge::DT_FLOAT, ge::FORMAT_ND}, x2,
        {{{8, 5000}, {8, 5000}}, ge::DT_FLOAT, ge::FORMAT_ND}, rows * cols, HYPOT_SCENE_MULTI_DIM_BRC, y));
    ASSERT_EQ(y.size(), rows * cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double ref = std::hypot(static_cast<double>(x1[i * cols + j]), static_cast<double>(x2[j]));
            double limit = kFp32Atol + kFp32Rtol * std::fabs(ref);
            ASSERT_LE(std::fabs(static_cast<double>(y[i * cols + j]) - ref), limit)
                << "row=" << i << " col=" << j << " y=" << y[i * cols + j] << " ref=" << ref
                << " (ubOuter division chain)";
        }
    }
}

// Whole-input scalar broadcasts now route through OneDim (the other input is contiguous), so this
// regression verifies that the 2B scalar path remains correct after the scene simplification.
TEST_F(HypotKernelTest, fp16_multidim_scalar_slot_align)
{
    const float x1Vals[3] = {3.0f, 5.0f, 1.5f};
    std::vector<uint16_t> x1(3);
    std::vector<float> x1f(3);
    for (size_t i = 0; i < 3; i++) {
        x1[i] = FloatToHalfBits(x1Vals[i]);
        x1f[i] = HalfBitsToFloat(x1[i]);
    }
    std::vector<uint16_t> x2(1, FloatToHalfBits(2.0f));
    std::vector<uint16_t> y;
    ASSERT_TRUE(RunHypotKernelFp16({{{3, 1}, {3, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}, x1,
                                   {{{1, 1}, {1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}, x2,
                                   {{{3, 1}, {3, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}, 3, HYPOT_SCENE_ONE_DIM, y));
    ASSERT_EQ(y.size(), 3U);
    const float x2f = HalfBitsToFloat(x2[0]);
    for (size_t i = 0; i < 3; i++) {
        float yVal = HalfBitsToFloat(y[i]);
        double ref = std::hypot(static_cast<double>(x1f[i]), static_cast<double>(x2f));
        double limit = kFp16Tol + kFp16Tol * std::fabs(ref);
        EXPECT_LE(std::fabs(static_cast<double>(yVal) - ref), limit)
            << "idx=" << i << " y=" << yVal << " ref=" << ref << " (scalar slot 32B alignment)";
    }
}

// ---------------- STRUCT / STREAM scene execution coverage ----------------

// STRUCT scene cross-3D tiny plane: [5,1,8] x [1,6,1] -> [5,6,8] (240 elems). Total 240 != 192 keeps
// it off the scene-1 tiny-pattern route, and the STRUCT host gate has priority over the <=1024 VECIDX
// reroute (the TilingKey scene assertion inside RunHypotKernelRaw pins that priority).
TEST_F(HypotKernelTest, struct_cross3d_tiny_plane_fp32)
{
    std::vector<float> x1(40), x2(6);
    for (size_t i = 0; i < 40; i++) {
        x1[i] = 0.5f + static_cast<float>(i % 9) * 0.5f;
    }
    for (size_t a = 0; a < 6; a++) {
        x2[a] = 1.0f + static_cast<float>(a) * 0.75f;
    }
    std::vector<float> y;
    ASSERT_TRUE(RunHypotKernel({{{5, 1, 8}, {5, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, x1,
                               {{{1, 6, 1}, {1, 6, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}, x2,
                               {{{5, 6, 8}, {5, 6, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, 240, HYPOT_SCENE_MULTI_DIM_STRUCT,
                               y));
    ASSERT_EQ(y.size(), 240U);
    for (size_t r = 0; r < 5; r++) {
        for (size_t a = 0; a < 6; a++) {
            for (size_t i = 0; i < 8; i++) {
                double ref = std::hypot(static_cast<double>(x1[r * 8 + i]), static_cast<double>(x2[a]));
                double limit = kFp32Atol + kFp32Rtol * std::fabs(ref);
                EXPECT_LE(std::fabs(static_cast<double>(y[(r * 6 + a) * 8 + i]) - ref), limit)
                    << "r=" << r << " a=" << a << " i=" << i;
            }
        }
    }
}

// STRUCT scene high-rank aligned-ROW: [2,1,4,1,16] x [1,3,1,5,16] -> [2,3,4,5,16] (1920 elems).
// Fused rank stays 5 (alternating broadcast axes), x2 classifies as a single-run ROW with I=80
// (8-aligned) -> rowStaticHighRank block expansion; x1 is the GENERIC Gather counterpart.
TEST_F(HypotKernelTest, struct_high_rank_row_fp32)
{
    std::vector<float> x1(2 * 4 * 16), x2(3 * 5 * 16);
    for (size_t i = 0; i < x1.size(); i++) {
        x1[i] = 0.25f + static_cast<float>(i % 13) * 0.5f;
    }
    for (size_t i = 0; i < x2.size(); i++) {
        x2[i] = 0.5f + static_cast<float>(i % 11) * 0.25f;
    }
    std::vector<float> y;
    ASSERT_TRUE(RunHypotKernel({{{2, 1, 4, 1, 16}, {2, 1, 4, 1, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}, x1,
                               {{{1, 3, 1, 5, 16}, {1, 3, 1, 5, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}, x2,
                               {{{2, 3, 4, 5, 16}, {2, 3, 4, 5, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}, 1920,
                               HYPOT_SCENE_MULTI_DIM_STRUCT, y));
    ASSERT_EQ(y.size(), 1920U);
    for (size_t i0 = 0; i0 < 2; i0++) {
        for (size_t i1 = 0; i1 < 3; i1++) {
            for (size_t i2 = 0; i2 < 4; i2++) {
                for (size_t i3 = 0; i3 < 5; i3++) {
                    for (size_t i4 = 0; i4 < 16; i4++) {
                        size_t out = (((i0 * 3 + i1) * 4 + i2) * 5 + i3) * 16 + i4;
                        double ref = std::hypot(static_cast<double>(x1[i0 * 64 + i2 * 16 + i4]),
                                                static_cast<double>(x2[i1 * 80 + i3 * 16 + i4]));
                        double limit = kFp32Atol + kFp32Rtol * std::fabs(ref);
                        EXPECT_LE(std::fabs(static_cast<double>(y[out]) - ref), limit) << "out=" << out;
                    }
                }
            }
        }
    }
}

// STREAM scene COL-single: [512,2048] x [512,1] -> 1M elems fp16. The broadcast column is preloaded
// (dimM=512 <= HYPOT_PRELOAD_MAX_ELEMS) and the chain runs scalar-vector per tile.
TEST_F(HypotKernelTest, stream_fp16_col_single)
{
    const size_t M = 512, N = 2048;
    std::vector<uint16_t> x1(M * N), x2(M);
    std::vector<float> x1f(M * N), x2f(M);
    for (size_t i = 0; i < M * N; i++) {
        x1[i] = FloatToHalfBits(0.5f + static_cast<float>(i % 97) * 0.25f);
        x1f[i] = HalfBitsToFloat(x1[i]);
    }
    for (size_t i = 0; i < M; i++) {
        x2[i] = FloatToHalfBits(1.0f + static_cast<float>(i % 23) * 0.125f);
        x2f[i] = HalfBitsToFloat(x2[i]);
    }
    std::vector<uint16_t> y;
    ASSERT_TRUE(RunHypotKernelFp16({{{512, 2048}, {512, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND}, x1,
                                   {{{512, 1}, {512, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}, x2,
                                   {{{512, 2048}, {512, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND}, M * N,
                                   HYPOT_SCENE_MULTI_DIM_STREAM, y));
    ASSERT_EQ(y.size(), M * N);
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float yVal = HalfBitsToFloat(y[i * N + j]);
            double ref = std::hypot(static_cast<double>(x1f[i * N + j]), static_cast<double>(x2f[i]));
            double limit = kFp16Tol + kFp16Tol * std::fabs(ref);
            EXPECT_LE(std::fabs(static_cast<double>(yVal) - ref), limit) << "row=" << i << " col=" << j;
        }
    }
}

// STREAM scene ROW-periodic: [1024,1024] x [1,1024] -> 1M elems fp16. Inner-major tile walk with the
// UB segment cache; every output row reuses the same broadcast segment.
TEST_F(HypotKernelTest, stream_fp16_row_periodic)
{
    const size_t M = 1024, N = 1024;
    std::vector<uint16_t> x1(M * N), x2(N);
    std::vector<float> x1f(M * N), x2f(N);
    for (size_t i = 0; i < M * N; i++) {
        x1[i] = FloatToHalfBits(0.5f + static_cast<float>(i % 89) * 0.25f);
        x1f[i] = HalfBitsToFloat(x1[i]);
    }
    for (size_t j = 0; j < N; j++) {
        x2[j] = FloatToHalfBits(1.0f + static_cast<float>(j % 29) * 0.125f);
        x2f[j] = HalfBitsToFloat(x2[j]);
    }
    std::vector<uint16_t> y;
    ASSERT_TRUE(RunHypotKernelFp16({{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, x1,
                                   {{{1, 1024}, {1, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, x2,
                                   {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND}, M * N,
                                   HYPOT_SCENE_MULTI_DIM_STREAM, y));
    ASSERT_EQ(y.size(), M * N);
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float yVal = HalfBitsToFloat(y[i * N + j]);
            double ref = std::hypot(static_cast<double>(x1f[i * N + j]), static_cast<double>(x2f[j]));
            double limit = kFp16Tol + kFp16Tol * std::fabs(ref);
            EXPECT_LE(std::fabs(static_cast<double>(yVal) - ref), limit) << "row=" << i << " col=" << j;
        }
    }
}

// STREAM scene BF16 dtype branch: [512,2048] x [1,2048] ROW-periodic (segment cache + CAST_RINT back)
TEST_F(HypotKernelTest, stream_bf16_row_periodic)
{
    const size_t M = 512, N = 2048;
    std::vector<uint16_t> x1(M * N), x2(N);
    std::vector<float> x1f(M * N), x2f(N);
    for (size_t i = 0; i < M * N; i++) {
        x1[i] = FloatToBf16BitsRint(0.5f + static_cast<float>(i % 83) * 0.25f);
        x1f[i] = Bf16BitsToFloat(x1[i]);
    }
    for (size_t j = 0; j < N; j++) {
        x2[j] = FloatToBf16BitsRint(1.0f + static_cast<float>(j % 31) * 0.125f);
        x2f[j] = Bf16BitsToFloat(x2[j]);
    }
    std::vector<uint16_t> y;
    ASSERT_TRUE(RunHypotKernelBf16({{{512, 2048}, {512, 2048}}, ge::DT_BF16, ge::FORMAT_ND}, x1,
                                   {{{1, 2048}, {1, 2048}}, ge::DT_BF16, ge::FORMAT_ND}, x2,
                                   {{{512, 2048}, {512, 2048}}, ge::DT_BF16, ge::FORMAT_ND}, M * N,
                                   HYPOT_SCENE_MULTI_DIM_STREAM, y));
    ASSERT_EQ(y.size(), M * N);
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float yVal = Bf16BitsToFloat(y[i * N + j]);
            double ref = std::hypot(static_cast<double>(x1f[i * N + j]), static_cast<double>(x2f[j]));
            double limit = kBf16Tol + kBf16Tol * std::fabs(ref);
            EXPECT_LE(std::fabs(static_cast<double>(yVal) - ref), limit) << "row=" << i << " col=" << j;
        }
    }
}

// ---------------- mixed-dtype kernel coverage (no L0 Cast chain: the kernel reads each input in
// its own raw dtype and computes in FP32 registers) ----------------

inline std::vector<float> FloatsFromBytes(const std::vector<uint8_t>& bytes)
{
    std::vector<float> out(bytes.size() / sizeof(float));
    std::memcpy(out.data(), bytes.data(), out.size() * sizeof(float));
    return out;
}

// [8,1] fp16 x [1,8] bf16 -> [8,8] fp32, MULTI_DIM_VECIDX scene (small cross broadcast)
TEST_F(HypotKernelTest, mixed_f16_bf16_cross_8x1_1x8)
{
    std::vector<uint16_t> x1(8), x2(8);
    std::vector<float> x1f(8), x2f(8);
    for (size_t i = 0; i < 8; i++) {
        x1[i] = FloatToHalfBits(0.5f + static_cast<float>(i) * 0.5f);
        x1f[i] = HalfBitsToFloat(x1[i]);
        x2[i] = FloatToBf16BitsRint(1.0f + static_cast<float>(i) * 0.25f);
        x2f[i] = Bf16BitsToFloat(x2[i]);
    }
    std::vector<uint8_t> yBytes;
    ASSERT_TRUE(RunHypotKernelRaw({{{8, 1}, {8, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}, ToBytes(x1),
                                  {{{1, 8}, {1, 8}}, ge::DT_BF16, ge::FORMAT_ND}, ToBytes(x2),
                                  {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, 64 * sizeof(float),
                                  HYPOT_SCENE_MULTI_DIM_VECIDX, LaunchHypotKernelMixF16Bf16, yBytes));
    std::vector<float> y = FloatsFromBytes(yBytes);
    ASSERT_EQ(y.size(), 64U);
    for (size_t r = 0; r < 8; r++) {
        for (size_t i = 0; i < 8; i++) {
            double ref = std::hypot(static_cast<double>(x1f[r]), static_cast<double>(x2f[i]));
            double limit = kFp32Atol + kFp32Rtol * std::fabs(ref);
            EXPECT_LE(std::fabs(static_cast<double>(y[r * 8 + i]) - ref), limit) << "r=" << r << " i=" << i;
        }
    }
}

// [8,1,128] fp16 x [8,2,128] bf16 -> [8,2,128] fp32, 2048 elems > 1024: stays on MULTI_DIM_BRC
TEST_F(HypotKernelTest, mixed_f16_bf16_multidim_brc)
{
    std::vector<uint16_t> x1(8 * 128), x2(8 * 2 * 128);
    std::vector<float> x1f(8 * 128), x2f(8 * 2 * 128);
    for (size_t i = 0; i < x1.size(); i++) {
        x1[i] = FloatToHalfBits(0.5f + static_cast<float>(i % 17) * 0.25f);
        x1f[i] = HalfBitsToFloat(x1[i]);
    }
    for (size_t j = 0; j < x2.size(); j++) {
        x2[j] = FloatToBf16BitsRint(1.0f + static_cast<float>(j % 19) * 0.125f);
        x2f[j] = Bf16BitsToFloat(x2[j]);
    }
    std::vector<uint8_t> yBytes;
    ASSERT_TRUE(RunHypotKernelRaw({{{8, 1, 128}, {8, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}, ToBytes(x1),
                                  {{{8, 2, 128}, {8, 2, 128}}, ge::DT_BF16, ge::FORMAT_ND}, ToBytes(x2),
                                  {{{8, 2, 128}, {8, 2, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}, 2048 * sizeof(float),
                                  HYPOT_SCENE_MULTI_DIM_BRC, LaunchHypotKernelMixF16Bf16, yBytes));
    std::vector<float> y = FloatsFromBytes(yBytes);
    ASSERT_EQ(y.size(), 2048U);
    for (size_t r = 0; r < 8; r++) {
        for (size_t a = 0; a < 2; a++) {
            for (size_t i = 0; i < 128; i++) {
                double ref = std::hypot(static_cast<double>(x1f[r * 128 + i]),
                                        static_cast<double>(x2f[(r * 2 + a) * 128 + i]));
                double limit = kFp32Atol + kFp32Rtol * std::fabs(ref);
                EXPECT_LE(std::fabs(static_cast<double>(y[(r * 2 + a) * 128 + i]) - ref), limit)
                    << "r=" << r << " a=" << a << " i=" << i;
            }
        }
    }
}

// [1024] fp32 x [1024] bf16 -> [1024] fp32, ONE_DIM scene
TEST_F(HypotKernelTest, mixed_f32_bf16_onedim)
{
    const size_t n = 1024;
    std::vector<float> x1(n);
    std::vector<uint16_t> x2(n);
    std::vector<float> x2f(n);
    for (size_t i = 0; i < n; i++) {
        x1[i] = 0.25f + static_cast<float>(i % 89) * 0.125f;
        x2[i] = FloatToBf16BitsRint(1.0f + static_cast<float>(i % 97) * 0.25f);
        x2f[i] = Bf16BitsToFloat(x2[i]);
    }
    std::vector<uint8_t> x1Bytes(n * sizeof(float));
    std::memcpy(x1Bytes.data(), x1.data(), x1Bytes.size());
    std::vector<uint8_t> yBytes;
    ASSERT_TRUE(RunHypotKernelRaw({{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND}, x1Bytes,
                                  {{{1024}, {1024}}, ge::DT_BF16, ge::FORMAT_ND}, ToBytes(x2),
                                  {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND}, n * sizeof(float),
                                  HYPOT_SCENE_ONE_DIM, LaunchHypotKernelMixF32Bf16, yBytes));
    std::vector<float> y = FloatsFromBytes(yBytes);
    ExpectNearGolden(y, x1, x2f, "mixed_f32_bf16_onedim");
}

// [1] fp16 x [64] bf16 -> [64] fp32, ONE_DIM one-scalar path with a BF16 full side and FP32 out
TEST_F(HypotKernelTest, mixed_f16_scalar_bf16_full_onedim)
{
    const size_t n = 64;
    std::vector<uint16_t> x1(1), x2(n);
    std::vector<float> x2f(n);
    x1[0] = FloatToHalfBits(1.5f);
    const float x1f = HalfBitsToFloat(x1[0]);
    for (size_t i = 0; i < n; i++) {
        x2[i] = FloatToBf16BitsRint(0.5f + static_cast<float>(i % 13) * 0.5f);
        x2f[i] = Bf16BitsToFloat(x2[i]);
    }
    std::vector<uint8_t> yBytes;
    ASSERT_TRUE(RunHypotKernelRaw({{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND}, ToBytes(x1),
                                  {{{64}, {64}}, ge::DT_BF16, ge::FORMAT_ND}, ToBytes(x2),
                                  {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND}, n * sizeof(float), HYPOT_SCENE_ONE_DIM,
                                  LaunchHypotKernelMixF16Bf16, yBytes));
    std::vector<float> y = FloatsFromBytes(yBytes);
    ASSERT_EQ(y.size(), n);
    for (size_t i = 0; i < n; i++) {
        double ref = std::hypot(static_cast<double>(x1f), static_cast<double>(x2f[i]));
        double limit = kFp32Atol + kFp32Rtol * std::fabs(ref);
        EXPECT_LE(std::fabs(static_cast<double>(y[i]) - ref), limit) << "i=" << i;
    }
}
