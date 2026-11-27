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
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../../random_common/op_kernel/arch35/random_unified_tiling_data_arch35.h"

__global__ __aicore__ void stateless_bernoulli(GM_ADDR shape, GM_ADDR prob, GM_ADDR seed, GM_ADDR offset, GM_ADDR y,
                                               GM_ADDR workspace, GM_ADDR tiling);

namespace {
constexpr uint32_t kNumBlocks = 1;
constexpr uint64_t kTilingKey = 100;
constexpr uint64_t kTilingKeyDense = 101;
constexpr uint64_t kTilingKeyBcast = 102;
constexpr int64_t kElementCount = 256;

inline size_t Align32(size_t size) { return (size + 31U) / 32U * 32U; }
} // namespace

class StatelessBernoulliKernelTest : public testing::Test {};

TEST_F(StatelessBernoulliKernelTest, smoke_float)
{
    auto* shape = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(2 * sizeof(int32_t))));
    auto* prob = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(float))));
    auto* seed = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* offset = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * sizeof(float))));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * 1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUnifiedSimtTilingDataStruct))));

    std::memset(y, 0, kElementCount * sizeof(float));
    std::memset(tiling, 0, sizeof(RandomUnifiedSimtTilingDataStruct));
    reinterpret_cast<int32_t*>(shape)[0] = 16;
    reinterpret_cast<int32_t*>(shape)[1] = 16;
    *reinterpret_cast<float*>(prob) = 0.5f;
    *reinterpret_cast<int64_t*>(seed) = 42;
    *reinterpret_cast<int64_t*>(offset) = 0;

    auto* tilingData = reinterpret_cast<RandomUnifiedSimtTilingDataStruct*>(tiling);
    tilingData->usedCoreNum = kNumBlocks;
    tilingData->outputSize = kElementCount;
    // tiling 端恒填 0 占位（新行为），真实 seed=42 由 kernel 从 GM 读取，值级期望不变
    tilingData->seed = 0;
    tilingData->offset = 0;
    tilingData->extraInt64Param1 = 1; // 标量语义
    // scalar 路径经 splitBlocks 取执行参数（对齐 tiling CalcExecutionPoliciesForBlocks 公式）
    tilingData->splitBlockCount = 1;
    tilingData->splitBlocks[0].numel = kElementCount;
    tilingData->splitBlocks[0].gmOffset = 0;
    tilingData->splitBlocks[0].grid = 1;           // ceil(256/256)
    tilingData->splitBlocks[0].totalThreads = 256; // 256 * min(ceil(256/256), 78*8)
    tilingData->splitBlocks[0].kernelOffset = 0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKey);
    ICPU_RUN_KF(stateless_bernoulli, kNumBlocks, shape, prob, seed, offset, y, workspace, tiling);

    auto* yData = reinterpret_cast<float*>(y);
    int64_t ones = 0;
    for (int64_t i = 0; i < kElementCount; ++i) {
        EXPECT_TRUE(yData[i] == 0.0f || yData[i] == 1.0f);
        ones += static_cast<int64_t>(yData[i]);
    }
    // 值级断言：oracle 期望（seed=42, p=0.5, n=256, T=256, 交织映射 + 严格小于）
    EXPECT_EQ(ones, 130);

    AscendC::GmFree(shape);
    AscendC::GmFree(prob);
    AscendC::GmFree(seed);
    AscendC::GmFree(offset);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// tiling 端 seed/offset 恒填 0 占位后，非零 offset 必须经 GM 读取才能生效：
// GM offset=8（counter 起点 = ⌈8/4⌉ = 2 组），golden oracle 期望 ones=121。
// 该用例通过证明：若 kernel 仍消费 tilingData 静态值（offset=0），结果将是 130 并失败。
TEST_F(StatelessBernoulliKernelTest, scalar_reads_offset_from_gm)
{
    auto* shape = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(2 * sizeof(int32_t))));
    auto* prob = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(float))));
    auto* seed = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* offset = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * sizeof(float))));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * 1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUnifiedSimtTilingDataStruct))));

    std::memset(y, 0, kElementCount * sizeof(float));
    std::memset(tiling, 0, sizeof(RandomUnifiedSimtTilingDataStruct));
    reinterpret_cast<int32_t*>(shape)[0] = 16;
    reinterpret_cast<int32_t*>(shape)[1] = 16;
    *reinterpret_cast<float*>(prob) = 0.5f;
    *reinterpret_cast<int64_t*>(seed) = 42;
    *reinterpret_cast<int64_t*>(offset) = 8; // 非零 offset：仅 GM 承载，tiling 端为 0 占位

    auto* tilingData = reinterpret_cast<RandomUnifiedSimtTilingDataStruct*>(tiling);
    tilingData->usedCoreNum = kNumBlocks;
    tilingData->outputSize = kElementCount;
    tilingData->seed = 0; // tiling 恒 0 占位（与 torchair 静态图下 tiling 读不到值的行为一致）
    tilingData->offset = 0;
    tilingData->extraInt64Param1 = 1; // 标量语义
    tilingData->splitBlockCount = 1;
    tilingData->splitBlocks[0].numel = kElementCount;
    tilingData->splitBlocks[0].gmOffset = 0;
    tilingData->splitBlocks[0].grid = 1;
    tilingData->splitBlocks[0].totalThreads = 256;
    tilingData->splitBlocks[0].kernelOffset = 0; // 纯相对增量

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKey);
    ICPU_RUN_KF(stateless_bernoulli, kNumBlocks, shape, prob, seed, offset, y, workspace, tiling);

    auto* yData = reinterpret_cast<float*>(y);
    int64_t ones = 0;
    for (int64_t i = 0; i < kElementCount; ++i) {
        EXPECT_TRUE(yData[i] == 0.0f || yData[i] == 1.0f);
        ones += static_cast<int64_t>(yData[i]);
    }
    // 值级断言：oracle 期望（seed=42, offset=8, p=0.5, n=256, T=256, 交织映射 + 严格小于）
    EXPECT_EQ(ones, 121);

    AscendC::GmFree(shape);
    AscendC::GmFree(prob);
    AscendC::GmFree(seed);
    AscendC::GmFree(offset);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// 张量语义 dense 路径（tilingKey=101, 连续映射 + u <= p）：GM offset=8 经 kernel 动态读取生效。
// golden oracle 期望 ones=123（若 kernel 仍消费 tilingData 静态 offset=0 则得到 137 并失败）。
TEST_F(StatelessBernoulliKernelTest, dense_reads_offset_from_gm)
{
    auto* shape = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(2 * sizeof(int32_t))));
    auto* prob = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * sizeof(float))));
    auto* seed = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* offset = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * sizeof(float))));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * 1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUnifiedSimtTilingDataStruct))));

    std::memset(y, 0, kElementCount * sizeof(float));
    std::memset(tiling, 0, sizeof(RandomUnifiedSimtTilingDataStruct));
    reinterpret_cast<int32_t*>(shape)[0] = 16;
    reinterpret_cast<int32_t*>(shape)[1] = 16;
    for (int64_t i = 0; i < kElementCount; ++i) {
        reinterpret_cast<float*>(prob)[i] = 0.5f; // prob 与输出同 shape（dense）
    }
    *reinterpret_cast<int64_t*>(seed) = 42;
    *reinterpret_cast<int64_t*>(offset) = 8; // 非零 offset：仅 GM 承载，tiling 端为 0 占位

    auto* tilingData = reinterpret_cast<RandomUnifiedSimtTilingDataStruct*>(tiling);
    tilingData->usedCoreNum = kNumBlocks;
    tilingData->outputSize = kElementCount;
    tilingData->seed = 0; // tiling 恒 0 占位（与 torchair 静态图下 tiling 读不到值的行为一致）
    tilingData->offset = 0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKeyDense);
    ICPU_RUN_KF(stateless_bernoulli, kNumBlocks, shape, prob, seed, offset, y, workspace, tiling);

    auto* yData = reinterpret_cast<float*>(y);
    int64_t ones = 0;
    for (int64_t i = 0; i < kElementCount; ++i) {
        EXPECT_TRUE(yData[i] == 0.0f || yData[i] == 1.0f);
        ones += static_cast<int64_t>(yData[i]);
    }
    // 值级断言：oracle 期望（seed=42, offset=8, p=0.5, n=256, 连续映射 + u <= p）
    EXPECT_EQ(ones, 123);

    AscendC::GmFree(shape);
    AscendC::GmFree(prob);
    AscendC::GmFree(seed);
    AscendC::GmFree(offset);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// 张量语义广播路径（tilingKey=102, prob=[16,1] 广播到 [16,16]）：GM offset=8 经 kernel 动态读取生效。
// golden oracle 期望 ones=123（与 dense 同值: 广播维 stride=0 反复读同一概率值, 流结构一致）。
TEST_F(StatelessBernoulliKernelTest, bcast_reads_offset_from_gm)
{
    auto* shape = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(2 * sizeof(int32_t))));
    auto* prob = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * sizeof(float)))); // prob=[16,1]
    auto* seed = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* offset = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * sizeof(float))));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(16 * 1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUnifiedSimtTilingDataStruct))));

    std::memset(y, 0, kElementCount * sizeof(float));
    std::memset(tiling, 0, sizeof(RandomUnifiedSimtTilingDataStruct));
    reinterpret_cast<int32_t*>(shape)[0] = 16;
    reinterpret_cast<int32_t*>(shape)[1] = 16;
    for (int64_t i = 0; i < 16; ++i) {
        reinterpret_cast<float*>(prob)[i] = 0.5f; // prob=[16,1]，前导维广播（stride=0）
    }
    *reinterpret_cast<int64_t*>(seed) = 42;
    *reinterpret_cast<int64_t*>(offset) = 8; // 非零 offset：仅 GM 承载，tiling 端为 0 占位

    auto* tilingData = reinterpret_cast<RandomUnifiedSimtTilingDataStruct*>(tiling);
    tilingData->usedCoreNum = kNumBlocks;
    tilingData->outputSize = kElementCount;
    tilingData->seed = 0; // tiling 恒 0 占位（与 torchair 静态图下 tiling 读不到值的行为一致）
    tilingData->offset = 0;
    // 广播参数表（对齐 tiling ComputeProbBroadcastTable 输出）：
    // probDims=2, probSizes=[16,16], probStrides=[0,1]（前导维广播 stride=0, 末维连续 stride=1）
    tilingData->probDims = 2;
    tilingData->probSizes[0] = 16;
    tilingData->probSizes[1] = 16;
    tilingData->probStrides[0] = 0;
    tilingData->probStrides[1] = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKeyBcast);
    ICPU_RUN_KF(stateless_bernoulli, kNumBlocks, shape, prob, seed, offset, y, workspace, tiling);

    auto* yData = reinterpret_cast<float*>(y);
    int64_t ones = 0;
    for (int64_t i = 0; i < kElementCount; ++i) {
        EXPECT_TRUE(yData[i] == 0.0f || yData[i] == 1.0f);
        ones += static_cast<int64_t>(yData[i]);
    }
    // 值级断言：oracle 期望（seed=42, offset=8, p=0.5, n=256, 连续映射 + u <= p, prob 广播）
    EXPECT_EQ(ones, 123);

    AscendC::GmFree(shape);
    AscendC::GmFree(prob);
    AscendC::GmFree(seed);
    AscendC::GmFree(offset);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
