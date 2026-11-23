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
 * \file test_transpose_tiling_strategy_small_shape_2d.cpp
 * \brief SMALL_SHAPE_2D(10012) 策略 UT：准入边界（纯策略直调）+ UB 分核 + 端到端命中 + 回落
 *
 * SMALL_SHAPE_2D 承接 2D perm=[1,0] 且 64KB <= 数据量 < 小 shape 阈值的场景；
 * 与 SMALL_SHAPE（<64KB 纯直读）相互独立。
 */

#include "test_transpose_tiling_strategy_common.h"
#include "../../../../op_host/arch35/transpose_tiling_strategy_small_shape_2d.h"

using namespace transpose_strategy_ut;

namespace {
gert::TilingContextPara Make2DPara(optiling::TransposeCompilerInfo& compileInfo, std::initializer_list<int64_t> in,
                                   std::initializer_list<int64_t> out, ge::DataType dtype)
{
    static int64_t permValue[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({in, in}, dtype, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &permValue);
    gert::TilingContextPara::TensorDescription y({out, out}, dtype, ge::FORMAT_ND);
    return gert::TilingContextPara("Transpose", {x, perm}, {y}, &compileInfo);
}
} // namespace

/* ============================== 准入检查 ============================== */

// [1850,128] fp32 = 947KB：>=64KB 命中
TEST(TransposeSmallShape2DStrategy, is_support_true_for_64kb_plus)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = Make2DPara(compileInfo, {1850, 128}, {128, 1850}, ge::DT_FLOAT);
    ShapeInfo si = MakeShape({1850, 128}, {1, 0}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeSmallShape2DStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    EXPECT_TRUE(strategy.IsSupport());
}

// [64,48] fp32 = 12KB：<64KB 不命中，交给 SMALL_SHAPE
TEST(TransposeSmallShape2DStrategy, is_support_false_below_64kb)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = Make2DPara(compileInfo, {64, 48}, {48, 64}, ge::DT_FLOAT);
    ShapeInfo si = MakeShape({64, 48}, {1, 0}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeSmallShape2DStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    EXPECT_FALSE(strategy.IsSupport());
}

// [4096,4] fp32 = 64KB 且 cols=4：cols<=4 不命中，交给 SMALL_SHAPE
TEST(TransposeSmallShape2DStrategy, is_support_false_when_cols_le_4)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = Make2DPara(compileInfo, {4096, 4}, {4, 4096}, ge::DT_FLOAT);
    ShapeInfo si = MakeShape({4096, 4}, {1, 0}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeSmallShape2DStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    EXPECT_FALSE(strategy.IsSupport());
}

// 超过 4MB 小 shape 阈值：不命中，交给 GATHER/NDDMA 等大 shape 模板
TEST(TransposeSmallShape2DStrategy, is_support_false_above_small_threshold)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = Make2DPara(compileInfo, {2000, 2000}, {2000, 2000}, ge::DT_FLOAT);
    ShapeInfo si = MakeShape({2000, 2000}, {1, 0}, 4); // 16MB
    auto bundle = BuildTilingContext(para);
    optiling::TransposeSmallShape2DStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    EXPECT_FALSE(strategy.IsSupport());
}

// perm 不是 [1,0]：不命中
TEST(TransposeSmallShape2DStrategy, is_support_false_when_perm_not_10)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = Make2DPara(compileInfo, {1850, 128}, {1850, 128}, ge::DT_FLOAT);
    ShapeInfo si = MakeShape({1850, 128}, {0, 1}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeSmallShape2DStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    EXPECT_FALSE(strategy.IsSupport());
}

/* ============================== 分核 / 端到端 ============================== */

// [1000,20] fp32 = 80KB：UB 模式 -> tile=32, totalStripes=32, chunkCols=20,
// totalWork=32, blockNum=32, inPitch=32, bufBytes=55296（coreNum=48, ubSize=253952）
TEST(TransposeSmallShape2DStrategy, e2e_hits_this_strategy)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = Make2DPara(compileInfo, {1000, 20}, {20, 1000}, ge::DT_FLOAT);
    TilingInfo e2e;
    ASSERT_TRUE(ExecuteTiling(para, e2e));
    EXPECT_EQ(e2e.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE_2D));
    EXPECT_EQ(e2e.blockNum, 32U); // totalWork(32) < coreNum(48)
    const auto* data = GetTilingData<TransposeSmallShape2DTilingData>(e2e);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->rows, 1000U);
    EXPECT_EQ(data->cols, 20U);
    EXPECT_EQ(data->chunkCols, 20U);
    EXPECT_EQ(data->totalWork, 32U);
    EXPECT_EQ(data->inPitch, 32U);
    EXPECT_EQ(data->bufBytes, 55296U);
    ASSERT_EQ(e2e.workspaceSizes.size(), 1U);
    EXPECT_EQ(e2e.workspaceSizes[0], 0U);
}

// [20,1000] fp32 = 80KB：H==20<=32，由 SMALL_SHAPE_2D 承接；
// 为提升并行度 N 方向切成 chunkCols=32 -> totalStripes=1, totalWork=32
TEST(TransposeSmallShape2DStrategy, e2e_with_n_tail)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = Make2DPara(compileInfo, {20, 1000}, {1000, 20}, ge::DT_FLOAT);
    TilingInfo e2e;
    ASSERT_TRUE(ExecuteTiling(para, e2e));
    EXPECT_EQ(e2e.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE_2D));
    const auto* data = GetTilingData<TransposeSmallShape2DTilingData>(e2e);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->chunkCols, 32U);
    EXPECT_EQ(data->totalWork, 32U);
    EXPECT_EQ(data->inPitch, 40U);
    EXPECT_EQ(data->bufBytes, 55296U);
}

// UB 预算不足（ubSize=8KB < 32KB DCACHE）：DoExecute 返回 FALLBACK，交给 SMALL_SHAPE 兜底
TEST(TransposeSmallShape2DStrategy, fallback_when_ub_budget_too_small)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = Make2DPara(compileInfo, {1850, 128}, {128, 1850}, ge::DT_FLOAT);
    ShapeInfo si = MakeShape({1850, 128}, {1, 0}, 4);
    auto bundle = BuildTilingContext(para);
    auto platform = MakePlatformInfo();
    platform.ubSize = 8192;
    optiling::TransposeSmallShape2DStrategy strategy(bundle.context, platform, MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    EXPECT_EQ(strategy.Execute(), TilingStatus::FALLBACK);
}

// TilingData 容量不足：SetTilingData 返回 FAILURE，整体 tiling 失败
TEST(TransposeSmallShape2DStrategy, direct_tiling_data_insufficient)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = Make2DPara(compileInfo, {1850, 128}, {128, 1850}, ge::DT_FLOAT);
    para.tilingDataSize_ = 8;
    ShapeInfo si = MakeShape({1850, 128}, {1, 0}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeSmallShape2DStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    EXPECT_EQ(strategy.Execute(), TilingStatus::FAILURE);
}

// Ascend950PR(56 AIV)：829x256 fp16 为 13 stripe，目标 chunk 数=4 -> 52 核
TEST(TransposeSmallShape2DStrategy, split_829x256_fp16_uses_target_chunks_56cores)
{
    auto compileInfo = MakeCompilerInfo(56, 253952);
    auto para = Make2DPara(compileInfo, {829, 256}, {256, 829}, ge::DT_FLOAT16);
    ShapeInfo si = MakeShape({829, 256}, {1, 0}, 2);
    auto bundle = BuildTilingContext(para);
    auto platform = MakePlatformInfo();
    platform.coreNum = 56;
    platform.ubSize = 253952;
    optiling::TransposeSmallShape2DStrategy strategy(bundle.context, platform, MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    const auto* data = GetTilingData<TransposeSmallShape2DTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->chunkCols, 64U);
    EXPECT_EQ(data->totalWork, 52U);
    EXPECT_EQ(bundle.context->GetBlockDim(), 52U);
}

// Ascend950PR(56 AIV)：836x256 fp16 为 14 stripe，旧门限会掉到 14 核；
// 新目标 chunk 数=4 -> 56 核，消除 13/14 stripe 的性能悬崖。
TEST(TransposeSmallShape2DStrategy, split_836x256_fp16_uses_all_56cores)
{
    auto compileInfo = MakeCompilerInfo(56, 253952);
    auto para = Make2DPara(compileInfo, {836, 256}, {256, 836}, ge::DT_FLOAT16);
    ShapeInfo si = MakeShape({836, 256}, {1, 0}, 2);
    auto bundle = BuildTilingContext(para);
    auto platform = MakePlatformInfo();
    platform.coreNum = 56;
    platform.ubSize = 253952;
    optiling::TransposeSmallShape2DStrategy strategy(bundle.context, platform, MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    const auto* data = GetTilingData<TransposeSmallShape2DTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->chunkCols, 64U);
    EXPECT_EQ(data->totalWork, 56U);
    EXPECT_EQ(bundle.context->GetBlockDim(), 56U);
}

/* ============================================================================
 * SMALL_SHAPE_2D(10012)：64KB 边界 + 4 buffer + 32KB DCACHE UB 占用
 * ========================================================================== */
TEST(TransposeSmallShape2DStrategy, small_shape_2d_boundary_and_ub_budget)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeDummyPara(compileInfo);
    auto bundle = BuildTilingContext(para);

    ShapeInfo hit = MakeShape({2048, 8}, {1, 0}, 4); // 65536B
    optiling::TransposeSmallShape2DStrategy hitStrategy(bundle.context, MakePlatformInfo(), MakeOpParam(hit));
    EXPECT_TRUE(hitStrategy.IsSupport());

    ShapeInfo miss = MakeShape({2047, 8}, {1, 0}, 4); // 65504B < 64KB
    optiling::TransposeSmallShape2DStrategy missStrategy(bundle.context, MakePlatformInfo(), MakeOpParam(miss));
    EXPECT_FALSE(missStrategy.IsSupport());

    ShapeInfo threshold = MakeShape({1024, 1024}, {1, 0}, 4); // 4MB，等于阈值不命中
    optiling::TransposeSmallShape2DStrategy thresholdStrategy(bundle.context, MakePlatformInfo(),
                                                              MakeOpParam(threshold));
    EXPECT_FALSE(thresholdStrategy.IsSupport());

    auto runBundle = BuildTilingContext(para);
    ShapeInfo shape = MakeShape({1000, 20}, {1, 0}, 4);
    optiling::TransposeSmallShape2DStrategy strategy(runBundle.context, MakePlatformInfo(), MakeOpParam(shape));
    ASSERT_TRUE(strategy.IsSupport());
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    const auto* data = GetTilingData<TransposeSmallShape2DTilingData>(runBundle.context);
    ASSERT_NE(data, nullptr);
    EXPECT_GT(data->bufBytes, 0U);
    EXPECT_EQ(data->rows, 1000U);
    EXPECT_EQ(data->cols, 20U);

    // kernel：x0 | x1 | y0 | y1 各 bufBytes，另有 32KB SIMT DCACHE 预留。
    constexpr int64_t kDcacheBytes = 32 * 1024;
    EXPECT_LE(static_cast<int64_t>(data->bufBytes) * 4 + kDcacheBytes, kUbSize);
    const int64_t tile = 128 / 4;
    EXPECT_LE(static_cast<int64_t>(tile) * data->inPitch * 4, static_cast<int64_t>(data->bufBytes));
    const int64_t outPitch = tile + 32 / 4;
    EXPECT_LE(static_cast<int64_t>(data->chunkCols) * outPitch * 4, static_cast<int64_t>(data->bufBytes));
}
