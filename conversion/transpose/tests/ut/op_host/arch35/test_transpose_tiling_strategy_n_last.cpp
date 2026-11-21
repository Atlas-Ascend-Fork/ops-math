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
 * \file test_transpose_tiling_strategy_n_last.cpp
 * \brief N_LAST(10004) 策略 UT：直调策略（逐字段断言）+ 端到端命中
 */

#include "test_transpose_tiling_strategy_common.h"
#include "../../../../op_host/arch35/transpose_tiling_strategy_n_last.h"

using namespace transpose_strategy_ut;

namespace {
gert::TilingContextPara MakePara(optiling::TransposeCompilerInfo& compileInfo)
{
    static int64_t permValue[3] = {1, 0, 2};
    gert::TilingContextPara::TensorDescription x({{100, 200, 64}, {100, 200, 64}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &permValue);
    gert::TilingContextPara::TensorDescription out({{200, 100, 64}, {200, 100, 64}}, ge::DT_FLOAT, ge::FORMAT_ND);
    return gert::TilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
}

// NDDMA_CUT_2 with inTailFactor=0 and outTailFactor!=0, covers offsetRangeOutputTail

/* ==================== 分核回溯分支补充用例 ==================== */

/* ==================== 公开 Execute 直调（e2e 无法构造的 UB 预算/小体积） ==================== */
optiling::TransposePlatformInfo MakeNLastPlatform(int64_t ubSize)
{
    optiling::TransposePlatformInfo platform = MakePlatformInfo();
    platform.ubSize = ubSize;
    return platform;
}
} // namespace

// 体积小于平台阈值时 N_LAST 不命中：
//   - 非 SIMT 平台由 NDDMA/NDDMA_BIG_DIM 承接；
//   - SIMT 平台由后续 SMALL_SHAPE 承接（N_LAST 注册在前，但体积门槛使其让位）。
TEST(TransposeNLastStrategy, is_support_false_below_threshold)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    // 384KB < 4MB，尾轴 48 不转置且 >= 32
    ShapeInfo si = MakeShape({32, 64, 48}, {1, 0, 2}, 4);
    auto bundle = BuildTilingContext(para);

    optiling::TransposePlatformInfo nonSimtPlatform = MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5162),
                                                                       false);
    optiling::TransposeNLastStrategy nonSimtStrategy(bundle.context, nonSimtPlatform, MakeOpParam(si));
    EXPECT_FALSE(nonSimtStrategy.IsSupport());

    // SIMT 平台同样受体积门槛约束，小 shape 交由后续 SMALL_SHAPE 承接
    optiling::TransposeNLastStrategy simtStrategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    EXPECT_FALSE(simtStrategy.IsSupport());
}

// DAV_5102 的 N_LAST 体积阈值为 400KB（与旧版 EntryTilingTemplate 一致）
TEST(TransposeNLastStrategy, is_support_dav_5102_volume_threshold_boundary)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    auto bundle = BuildTilingContext(para);
    constexpr int64_t kThreshold = 400000;
    auto platform = MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5102), true);

    // 尾轴不转置、尾轴 >= 32；体积恰好等于 400KB → 命中
    ShapeInfo hit = MakeShape({1, kThreshold / 4}, {0, 1}, 4);
    optiling::TransposeNLastStrategy hitStrategy(bundle.context, platform, MakeOpParam(hit));
    EXPECT_TRUE(hitStrategy.IsSupport());

    // 低于 400KB → 不命中
    ShapeInfo miss = MakeShape({1, kThreshold / 4 - 1}, {0, 1}, 4);
    optiling::TransposeNLastStrategy missStrategy(bundle.context, platform, MakeOpParam(miss));
    EXPECT_FALSE(missStrategy.IsSupport());
}

// 非 SIMT 平台体积门控边界：恰好等于 4MB 命中，低于 1 个元素（4 字节）不命中
TEST(TransposeNLastStrategy, is_support_non_simt_volume_threshold_boundary)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    auto bundle = BuildTilingContext(para);
    constexpr int64_t kThreshold = 4000000; // 非 SIMT 平台写死的 4MB 阈值
    auto platform = MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5162), false);

    // 尾轴不转置（perm={0,1}）、尾轴元素数 >= 32，体积恰好等于阈值 → 命中
    ShapeInfo hit = MakeShape({1, kThreshold / 4}, {0, 1}, 4);
    optiling::TransposeNLastStrategy hitStrategy(bundle.context, platform, MakeOpParam(hit));
    EXPECT_TRUE(hitStrategy.IsSupport());

    ShapeInfo miss = MakeShape({1, kThreshold / 4 - 1}, {0, 1}, 4);
    optiling::TransposeNLastStrategy missStrategy(bundle.context, platform, MakeOpParam(miss));
    EXPECT_FALSE(missStrategy.IsSupport());
}

TEST(TransposeNLastStrategy, e2e_hits_this_strategy)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    TilingInfo e2e;
    ASSERT_TRUE(ExecuteTiling(para, e2e));
    EXPECT_EQ(e2e.tilingKey, TK_N_LAST);
    const auto* data = GetTilingData<TransposeOpTilingData>(e2e);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->permSize, 3);
    EXPECT_GT(data->realCoreNum, 0);
    EXPECT_GT(data->inUbFactor, 0);
}

TEST(TransposeNLastStrategy, transpose_tiling_n_last_4d_no_last_transpose)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    // perm = {1, 0, 2, 3}: 只交换前两轴, 尾轴保持不转置。MergeAxis 后尾轴 (30*64=1920)
    // 仍不参与转置且 >= 32; 总量 4.6MB 超过 SMALL 门槛 → 走 N_LAST。
    int64_t perm_value[4] = {1, 0, 2, 3};
    gert::TilingContextPara::TensorDescription x({{30, 20, 30, 64}, {30, 20, 30, 64}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{20, 30, 30, 64}, {20, 30, 30, 64}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_N_LAST));
}

// 4D NDDMA NDDMA_CUT_1/NDDMA_CUT_2 path (large shape, dim <= 5, last axis transposed, not n_last)

TEST(TransposeNLastStrategy, transpose_tiling_n_last_large_3d)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    // perm {2, 0, 1}: last axis perm[2]=1 != 2 -> isLastAxisTranspose=true
    // For N_LAST we need !isLastAxisTranspose, so use perm {0, 2, 1}
    // perm {0, 2, 1}: perm[2]=1 != 2 -> still last axis transpose
    // perm {1, 0, 2}: perm[2]=2 == 2 -> not last axis transpose!
    int64_t perm_value[3] = {1, 0, 2};
    gert::TilingContextPara::TensorDescription x({{100, 200, 64}, {100, 200, 64}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{200, 100, 64}, {200, 100, 64}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_N_LAST));
}

// Shape with axis=1 (reduced after RemoveAxis), 3D with one 1-dim axis

TEST(TransposeNLastStrategy, transpose_tiling_n_last_large_4d)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[4] = {1, 0, 2, 3};
    gert::TilingContextPara::TensorDescription x({{200, 100, 64, 128}, {200, 100, 64, 128}}, ge::DT_FLOAT,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{100, 200, 64, 128}, {100, 200, 64, 128}}, ge::DT_FLOAT,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// NDDMA_CUT_2 path with both tail factors non-zero, covers GetIntervalInfoForCutTwice tail ranges

TEST(TransposeNLastStrategy, transpose_tiling_n_last_multiples_split)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {1, 0, 2};
    gert::TilingContextPara::TensorDescription x({{500, 400, 64}, {500, 400, 64}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{400, 500, 64}, {400, 500, 64}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_N_LAST));
}

// NDDMA_BIG_DIM with small outer dimensions to trigger CalcBlockSplitInfoForBigDim low-core branch

TEST(TransposeNLastStrategy, transpose_tiling_n_last_small_corenum)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 4;
    compileInfo.ubSize = 196608;

    // Small coreNum makes coreNumTmp more likely < coreNum → RateNLast
    int64_t perm_value[3] = {1, 0, 2};
    gert::TilingContextPara::TensorDescription x({{10, 5, 128}, {10, 5, 128}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{5, 10, 128}, {5, 10, 128}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// 尾轴放不下 UB 且核利用率不足 → Rate 回溯并命中因子
TEST(TransposeNLastStrategy, direct_rate_branch_unreachable_by_e2e)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    ShapeInfo si = MakeShape({1, 1, 100000}, {0, 1, 2}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeNLastStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_N_LAST));
    const auto* data = GetTilingData<TransposeOpTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->permSize, 3);
    EXPECT_GT(data->inUbFactor, 0);
    EXPECT_GT(data->realCoreNum, 0);
}

// 尾轴可整体吸收、核利用率不足 → Rate 回溯分支
TEST(TransposeNLastStrategy, direct_rate_absorb_branch_unreachable_by_e2e)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    ShapeInfo si = MakeShape({2, 3, 32}, {1, 0, 2}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeNLastStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_N_LAST));
    const auto* data = GetTilingData<TransposeOpTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->permSize, 3);
    EXPECT_GT(data->inUbFactor, 0);
}

// inUbFactor 无效 → 退轴重切
TEST(TransposeNLastStrategy, direct_revert_axis_unreachable_by_e2e)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    ShapeInfo si = MakeShape({1, 64, 33}, {1, 0, 2}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeNLastStrategy strategy(bundle.context, MakeNLastPlatform(288), MakeOpParam(si));
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_N_LAST));
    const auto* data = GetTilingData<TransposeOpTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->permSize, 3);
    EXPECT_EQ(data->ubSize, 288);
    EXPECT_GT(data->inUbFactor, 0);
}

// 块数不足以铺满核 → 每核 1 个 block
TEST(TransposeNLastStrategy, direct_block_less_than_corenum_unreachable_by_e2e)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    ShapeInfo si = MakeShape({1, 1, 32}, {0, 1, 2}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeNLastStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetBlockDim(), 32U);
    const auto* data = GetTilingData<TransposeOpTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->realCoreNum, 32);
    EXPECT_EQ(data->blkFactor, 1);
    EXPECT_EQ(data->blkTailFactor, 0);
}

// TilingData 容量不足：码流不写入，但 tilingKey/blockDim 仍按策略结果写回
TEST(TransposeNLastStrategy, direct_tiling_data_insufficient)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    para.tilingDataSize_ = 8;
    ShapeInfo si = MakeShape({30, 20, 30, 64}, {1, 0, 2, 3}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeNLastStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    // 码流容量不足：SetTilingData 返回 FAILURE，整体 tiling 失败
    EXPECT_EQ(strategy.Execute(), TilingStatus::FAILURE);
}
