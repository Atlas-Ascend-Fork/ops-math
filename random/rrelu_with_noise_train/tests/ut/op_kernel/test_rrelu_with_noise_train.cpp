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
 * \file test_rrelu_with_noise_train.cpp
 * \brief rrelu_with_noise_train op_kernel UT(仅 training 语义, fp32 变体)
 * kernel 为单 tiling key + DTYPE_X 编译期 dtype 变体, UT 以 -DDTYPE_X=float
 * 编译 fp32 变体覆盖基础功能; fp16/bf16 由 ST 用例集覆盖。
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../../random_common/op_kernel/arch35/random_unified_tiling_data_arch35.h"

extern "C" __global__ __aicore__ void rrelu_with_noise_train(GM_ADDR x, GM_ADDR seed, GM_ADDR offset, GM_ADDR y,
                                                             GM_ADDR noise, GM_ADDR workspace, GM_ADDR tiling);

namespace {
constexpr uint32_t kNumBlocks = 1;
constexpr uint64_t kTilingKey = 100; // RRELU_WITH_NOISE_DEFAULT_TILING_KEY
constexpr int64_t kElementCount = 256;

// 与 tiling 侧 RandomTilingArch35::CalcExecutionPoliciesForBlocks 对齐常量一致
constexpr int64_t kSimtThreadGroupSize = 256; // SIMT_THREAD_GROUP_SIZE
constexpr int64_t kAicClusterCount = 78;      // AIC_CLUSTER_COUNT
constexpr int64_t kMaxThreadsPerAic = 2048;   // MAX_THREADS_PER_AIC
constexpr int64_t kUnrollFactor = 4;
constexpr int64_t kCounterIncr = 4; // MAX_PRNG_COUNTER_INCR

inline size_t Align32(size_t size) { return (size + 31U) / 32U * 32U; }

int64_t CalcGrid(int64_t numel)
{
    int64_t grid = (numel + kSimtThreadGroupSize - 1) / kSimtThreadGroupSize;
    if (grid < 1) {
        grid = 1;
    }
    int64_t gridClamp = kAicClusterCount * (kMaxThreadsPerAic / kSimtThreadGroupSize);
    return (grid > gridClamp) ? gridClamp : grid;
}

int64_t CalcCounterOffset(int64_t numel)
{
    int64_t grid = CalcGrid(numel);
    int64_t totalThreads = grid * kSimtThreadGroupSize;
    int64_t threadsPerRound = totalThreads * kUnrollFactor;
    int64_t roundsNeeded = (numel + threadsPerRound - 1) / threadsPerRound;
    return roundsNeeded * kCounterIncr;
}

// 镜像 CalcExecutionPoliciesForBlocks(unroll=4): 多 splitBlock 时先按总 numel 推进全局
// 流位置, 再逐 block 累加 counterOffset
void FillSplitBlocks(RandomUnifiedSimtTilingDataStruct* td, int64_t numel, int64_t offset, int64_t splitCount)
{
    int64_t accumulatedOffset = offset;
    if (splitCount > 1) {
        accumulatedOffset += CalcCounterOffset(numel);
    }
    td->splitBlockCount = splitCount;
    int64_t blockNumel = numel / splitCount;
    for (int64_t i = 0; i < splitCount; i++) {
        int64_t curNumel = (i == splitCount - 1) ? (numel - blockNumel * (splitCount - 1)) : blockNumel;
        td->splitBlocks[i].numel = curNumel;
        td->splitBlocks[i].gmOffset = blockNumel * i;
        td->splitBlocks[i].kernelOffset = accumulatedOffset;
        td->splitBlocks[i].grid = CalcGrid(curNumel);
        td->splitBlocks[i].totalThreads = td->splitBlocks[i].grid * kSimtThreadGroupSize;
        accumulatedOffset += CalcCounterOffset(curNumel);
    }
}

// 通用执行: 填 tiling -> CPU 仿真跑 kernel -> 拷回 y/noise
void RunRrelu(const std::vector<float>& x, float lower, float upper, int64_t seed, int64_t offset,
              std::vector<float>& y, std::vector<float>& noise, int64_t usedCoreNum = 1, int64_t splitCount = 1)
{
    const int64_t numel = static_cast<int64_t>(x.size());
    auto* xGm = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(numel * sizeof(float))));
    auto* seedGm = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* offsetGm = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* yGm = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(numel * sizeof(float))));
    auto* noiseGm = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(numel * sizeof(float))));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * 1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUnifiedSimtTilingDataStruct))));

    std::memcpy(xGm, x.data(), numel * sizeof(float));
    // seed/offset 张量化: kernel 运行时从 GM 读取(各 1 元素 INT64)
    *reinterpret_cast<int64_t*>(seedGm) = seed;
    *reinterpret_cast<int64_t*>(offsetGm) = offset;
    // 哨兵初始化: 正常路径全部元素都会被覆写, 用于 usedCoreNum=0 守卫用例验证"未写入"
    float* yPtr = reinterpret_cast<float*>(yGm);
    float* noisePtr = reinterpret_cast<float*>(noiseGm);
    const float ySentinel = 7.0f;
    const float noiseSentinel = 9.0f;
    for (int64_t i = 0; i < numel; i++) {
        yPtr[i] = ySentinel;
        noisePtr[i] = noiseSentinel;
    }
    std::memset(tiling, 0, sizeof(RandomUnifiedSimtTilingDataStruct));

    auto* td = reinterpret_cast<RandomUnifiedSimtTilingDataStruct*>(tiling);
    td->usedCoreNum = usedCoreNum;
    td->outputSize = numel;
    td->ubSize = 229376;
    td->fromFp32 = lower;
    td->toFp32 = upper;
    // 张量化后 tiling 侧 seed/offset 恒为 0(前缀从 0 累积), 运行时加 GM 读到的 realOffset
    FillSplitBlocks(td, numel, 0, splitCount);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKey);
    ICPU_RUN_KF(rrelu_with_noise_train, kNumBlocks, xGm, seedGm, offsetGm, yGm, noiseGm, workspace, tiling);

    y.assign(yPtr, yPtr + numel);
    noise.assign(noisePtr, noisePtr + numel);

    AscendC::GmFree(xGm);
    AscendC::GmFree(seedGm);
    AscendC::GmFree(offsetGm);
    AscendC::GmFree(yGm);
    AscendC::GmFree(noiseGm);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// 语义校验(training): y = x<=0 ? x*noise : x; noise = x<=0 ? 保留随机值 : 1.0
// x==0 走随机分支, NaN 走正分支
void CheckTrainingSemantics(const std::vector<float>& x, const std::vector<float>& y, const std::vector<float>& noise,
                            float lower, float upper, float relTol)
{
    for (size_t i = 0; i < x.size(); i++) {
        float xf = x[i];
        float yf = y[i];
        float nf = noise[i];
        if (std::isnan(xf)) {
            EXPECT_TRUE(std::isnan(yf)) << "i=" << i << " y should be NaN";
            EXPECT_FLOAT_EQ(nf, 1.0f) << "i=" << i << " noise should be 1.0 at NaN";
        } else if (xf > 0.0f) {
            EXPECT_FLOAT_EQ(yf, xf) << "i=" << i;
            EXPECT_FLOAT_EQ(nf, 1.0f) << "i=" << i;
        } else {
            EXPECT_GE(nf, lower) << "i=" << i << " noise=" << nf << " below lower";
            // 无边界翻转: 浮点舍入可使 noise 触到上界
            EXPECT_LE(nf, upper) << "i=" << i << " noise=" << nf << " above upper";
            float expected = xf * nf;
            float tol = std::max(1e-3f, std::abs(expected) * relTol);
            EXPECT_NEAR(yf, expected, tol) << "i=" << i << " x=" << xf << " noise=" << nf;
        }
    }
}

std::vector<float> MakeMixedInput(int64_t numel)
{
    std::vector<float> x(static_cast<size_t>(numel));
    for (int64_t i = 0; i < numel; i++) {
        switch (i % 8) {
            case 0:
                x[static_cast<size_t>(i)] = 1.5f + static_cast<float>(i);
                break;
            case 1:
                x[static_cast<size_t>(i)] = -2.5f - static_cast<float>(i) * 0.25f;
                break;
            case 2:
                x[static_cast<size_t>(i)] = 0.0f;
                break;
            case 3:
                x[static_cast<size_t>(i)] = -0.001f * static_cast<float>(i + 1);
                break;
            case 4:
                x[static_cast<size_t>(i)] = 100.0f;
                break;
            case 5:
                x[static_cast<size_t>(i)] = -77.5f;
                break;
            case 6:
                x[static_cast<size_t>(i)] = 0.5f;
                break;
            default:
                x[static_cast<size_t>(i)] = -1.0f;
                break;
        }
    }
    return x;
}
} // namespace

class RreluWithNoiseKernelTest : public testing::Test {};

// fp32 默认区间(lower=0.125, upper=1/3), 混合正/负/零输入
TEST_F(RreluWithNoiseKernelTest, smoke_float32_default_range)
{
    std::vector<float> x = MakeMixedInput(kElementCount);
    std::vector<float> y;
    std::vector<float> noise;
    RunRrelu(x, 0.125f, 1.0f / 3.0f, 42, 0, y, noise);
    ASSERT_EQ(y.size(), x.size());
    ASSERT_EQ(noise.size(), x.size());
    CheckTrainingSemantics(x, y, noise, 0.125f, 1.0f / 3.0f, 1e-5f);
}

// fp32 自定义区间, 覆盖负数斜率(lower=-0.5, upper=2.0)
TEST_F(RreluWithNoiseKernelTest, smoke_float32_custom_range)
{
    std::vector<float> x = MakeMixedInput(kElementCount);
    std::vector<float> y;
    std::vector<float> noise;
    RunRrelu(x, -0.5f, 2.0f, 123, 0, y, noise);
    CheckTrainingSemantics(x, y, noise, -0.5f, 2.0f, 1e-5f);
}

// 全正输入: y 与 x 逐位一致, noise 全部被改写为 1.0
TEST_F(RreluWithNoiseKernelTest, all_positive_input)
{
    std::vector<float> x(static_cast<size_t>(kElementCount));
    for (int64_t i = 0; i < kElementCount; i++) {
        x[static_cast<size_t>(i)] = 0.25f + static_cast<float>(i);
    }
    std::vector<float> y;
    std::vector<float> noise;
    RunRrelu(x, 0.125f, 1.0f / 3.0f, 42, 0, y, noise);
    for (int64_t i = 0; i < kElementCount; i++) {
        EXPECT_FLOAT_EQ(y[static_cast<size_t>(i)], x[static_cast<size_t>(i)]) << "i=" << i;
        EXPECT_FLOAT_EQ(noise[static_cast<size_t>(i)], 1.0f) << "i=" << i;
    }
}

// 确定性: 相同 seed/offset 两次运行的 y 与 noise 完全一致
TEST_F(RreluWithNoiseKernelTest, determinism_same_seed)
{
    std::vector<float> x = MakeMixedInput(kElementCount);
    std::vector<float> y1;
    std::vector<float> noise1;
    std::vector<float> y2;
    std::vector<float> noise2;
    RunRrelu(x, 0.125f, 1.0f / 3.0f, 777, 0, y1, noise1);
    RunRrelu(x, 0.125f, 1.0f / 3.0f, 777, 0, y2, noise2);
    EXPECT_EQ(0, std::memcmp(y1.data(), y2.data(), y1.size() * sizeof(float)));
    EXPECT_EQ(0, std::memcmp(noise1.data(), noise2.data(), noise1.size() * sizeof(float)));
}

// 不同 seed 产生不同随机流
TEST_F(RreluWithNoiseKernelTest, different_seed_changes_noise)
{
    std::vector<float> x = MakeMixedInput(kElementCount);
    std::vector<float> y1;
    std::vector<float> noise1;
    std::vector<float> y2;
    std::vector<float> noise2;
    RunRrelu(x, 0.125f, 1.0f / 3.0f, 42, 0, y1, noise1);
    RunRrelu(x, 0.125f, 1.0f / 3.0f, 43, 0, y2, noise2);
    size_t noiseDiffs = 0;
    for (size_t i = 0; i < noise1.size(); i++) {
        if (std::memcmp(&noise1[i], &noise2[i], sizeof(float)) != 0) {
            noiseDiffs++;
        }
    }
    EXPECT_GT(noiseDiffs, 0U) << "different seeds produced identical noise";
}

// offset 推进随机流: offset=4(4 的倍数)与 offset=0 的随机流不同
TEST_F(RreluWithNoiseKernelTest, offset_shifts_stream)
{
    std::vector<float> x = MakeMixedInput(kElementCount);
    std::vector<float> y1;
    std::vector<float> noise1;
    std::vector<float> y2;
    std::vector<float> noise2;
    RunRrelu(x, 0.125f, 1.0f / 3.0f, 42, 0, y1, noise1);
    RunRrelu(x, 0.125f, 1.0f / 3.0f, 42, 4, y2, noise2);
    size_t noiseDiffs = 0;
    for (size_t i = 0; i < noise1.size(); i++) {
        if (std::memcmp(&noise1[i], &noise2[i], sizeof(float)) != 0) {
            noiseDiffs++;
        }
    }
    EXPECT_GT(noiseDiffs, 0U) << "offset=4 did not shift the random stream";
}

// 边界语义: x==0 走随机分支(noise 不被改写为 1.0), NaN 走正分支(y=NaN, noise=1.0)
TEST_F(RreluWithNoiseKernelTest, zero_and_nan_semantics)
{
    std::vector<float> x = MakeMixedInput(kElementCount);
    const size_t zeroIdx = 100; // MakeMixedInput 中 i%8==2 的位置已是 0, 这里显式固定校验点
    x[zeroIdx] = 0.0f;
    x[5] = std::numeric_limits<float>::quiet_NaN();
    x[130] = std::numeric_limits<float>::quiet_NaN();

    std::vector<float> y;
    std::vector<float> noise;
    RunRrelu(x, 0.125f, 1.0f / 3.0f, 42, 0, y, noise);
    CheckTrainingSemantics(x, y, noise, 0.125f, 1.0f / 3.0f, 1e-5f);

    // 显式钉住关键语义
    EXPECT_FLOAT_EQ(y[zeroIdx], 0.0f);
    EXPECT_LE(noise[zeroIdx], 1.0f / 3.0f) << "noise at x==0 must stay random (<= upper)";
    EXPECT_LT(noise[zeroIdx], 1.0f) << "noise at x==0 must not be overwritten to 1.0";
}

// 多 splitBlock: 按 CalcExecutionPoliciesForBlocks 逻辑切 2 块, 全量语义校验
TEST_F(RreluWithNoiseKernelTest, multiple_split_blocks)
{
    std::vector<float> x = MakeMixedInput(kElementCount);
    std::vector<float> y;
    std::vector<float> noise;
    RunRrelu(x, 0.125f, 1.0f / 3.0f, 42, 0, y, noise, 1, 2);
    ASSERT_EQ(y.size(), x.size());
    CheckTrainingSemantics(x, y, noise, 0.125f, 1.0f / 3.0f, 1e-5f);
}

// usedCoreNum=0: 所有 block 提前返回, y/noise 保持哨兵值未被写入
TEST_F(RreluWithNoiseKernelTest, used_core_num_guard)
{
    std::vector<float> x = MakeMixedInput(kElementCount);
    std::vector<float> y;
    std::vector<float> noise;
    RunRrelu(x, 0.125f, 1.0f / 3.0f, 42, 0, y, noise, 0, 1);
    for (size_t i = 0; i < x.size(); i++) {
        EXPECT_FLOAT_EQ(y[i], 7.0f) << "i=" << i << " y should keep sentinel";
        EXPECT_FLOAT_EQ(noise[i], 9.0f) << "i=" << i << " noise should keep sentinel";
    }
}

// 负 offset(4 倍数)基址回归: kernel 经真实 PhiloxAlgParsInit 计算基址(uint64 域
// (offset+3)/4 floor, 负 offset 在加法处提升翻转), -4/-8/-12 -> 2^62-1/-2/-3,
// 与 golden.py 修正后镜像(_philox_counter_base)同构; 锚点为 golden numpy 复刻
// 的逐位流值(x=-1.0f*1024, seed=1, lower=0.125, upper=0.333333, tt=1024)
TEST_F(RreluWithNoiseKernelTest, negative_offset_stream_matches_golden)
{
    struct Anchor {
        int64_t offset;
        uint32_t bits[8];
    };
    const Anchor anchors[] = {
        {-4, {0x3E44FEAEu, 0x3EA4A328u, 0x3E9E49EAu, 0x3E889244u, 0x3E8C09F5u, 0x3E833168u, 0x3E4F31D7u, 0x3E1C1A1Eu}},
        {-8, {0x3E1C4248u, 0x3E9AB6FBu, 0x3E58A970u, 0x3E60D337u, 0x3E721F7Au, 0x3E0B93D9u, 0x3E981516u, 0x3E9AD7D6u}},
        {-12, {0x3E285A30u, 0x3E84D9BDu, 0x3E8E9ED8u, 0x3E0026CEu, 0x3E2093BAu, 0x3E9E78DBu, 0x3EA3F8BAu, 0x3E8C0432u}},
    };
    std::vector<float> x(1024, -1.0f);
    for (const auto& a : anchors) {
        std::vector<float> y;
        std::vector<float> noise;
        RunRrelu(x, 0.125f, 0.333333f, 1, a.offset, y, noise);
        for (int i = 0; i < 8; i++) {
            uint32_t bits = 0;
            std::memcpy(&bits, &noise[i], sizeof(bits));
            EXPECT_EQ(bits, a.bits[i]) << "offset=" << a.offset << " i=" << i;
        }
    }
}

// 负 offset 量化分组实证: kernel 真实行为按 4 段量化分组(纯观察断言, 不依赖代码解读)
// 预期: 流(0)==流(-1)==流(-2)==流(-3); 流(-4..-7)同组; 流(-8..-11)同组; 组间互异
TEST_F(RreluWithNoiseKernelTest, negative_offset_quantization_groups)
{
    const int64_t offsets[] = {0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12};
    const size_t kNum = sizeof(offsets) / sizeof(offsets[0]);
    std::vector<float> x(1024, -1.0f);
    std::vector<std::vector<float>> streams(kNum);
    for (size_t i = 0; i < kNum; i++) {
        std::vector<float> y;
        std::vector<float> noise;
        RunRrelu(x, 0.125f, 0.333333f, 1, offsets[i], y, noise);
        streams[i] = noise;
    }
    auto idx = [&](int64_t off) {
        for (size_t i = 0; i < kNum; i++) {
            if (offsets[i] == off) {
                return i;
            }
        }
        return kNum;
    };
    auto same = [&](int64_t a, int64_t b) { return streams[idx(a)] == streams[idx(b)]; };
    EXPECT_TRUE(same(0, -1)) << "offset=-1 应与 0 同流";
    EXPECT_TRUE(same(0, -2)) << "offset=-2 应与 0 同流";
    EXPECT_TRUE(same(0, -3)) << "offset=-3 应与 0 同流";
    EXPECT_TRUE(same(-4, -5));
    EXPECT_TRUE(same(-4, -6));
    EXPECT_TRUE(same(-4, -7));
    EXPECT_TRUE(same(-8, -9));
    EXPECT_TRUE(same(-8, -10));
    EXPECT_TRUE(same(-8, -11));
    EXPECT_FALSE(same(0, -4)) << "0 与 -4 应不同流";
    EXPECT_FALSE(same(-4, -8)) << "-4 与 -8 应不同流";
    EXPECT_FALSE(same(-8, -12)) << "-8 与 -12 应不同流";
}
