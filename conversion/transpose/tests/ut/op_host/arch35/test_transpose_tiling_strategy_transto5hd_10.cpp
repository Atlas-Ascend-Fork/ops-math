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
 * \file test_transpose_tiling_strategy_transto5hd_10.cpp
 * \brief TRANSTO5HD_10(10007) 策略 UT：直调策略 + TilingData 逐字段断言
 *
 * 说明：这两个模板仅在 DAV_5102 生效；UT 端到端执行环境 SoC 为 ascend950（DAV_3510），
 * 注册表端到端无法命中 5HD 路径，故此处只做策略直调（直调 context 显式构造 DAV_5102）。
 */

#include "test_transpose_tiling_strategy_common.h"
#include "../../../../op_host/arch35/transpose_tiling_strategy_transto5hd_10.h"

using namespace transpose_strategy_ut;

namespace {
gert::TilingContextPara MakeTransTo5HD10Para(optiling::TransposeCompilerInfo& compileInfo)
{
    static int64_t permValue[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{16, 8}, {16, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &permValue);
    gert::TilingContextPara::TensorDescription out({{8, 16}, {8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    return gert::TilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
}
} // namespace

TEST(TransposeTransTo5HD10Strategy, related_mode_support_false)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeTransTo5HD10Para(compileInfo);
    ShapeInfo si = MakeShape({16, 8}, {1, 0}, 2);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeTransTo5HD10Strategy strategy(
        bundle.context, MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5102)), MakeOpParam(si, true));
    EXPECT_FALSE(strategy.IsSupport());
}

TEST(TransposeTransTo5HD10Strategy, direct_reject_perm)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeTransTo5HD10Para(compileInfo);
    ShapeInfo si = MakeShape({16, 8}, {0, 1}, 2);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeTransTo5HD10Strategy strategy(
        bundle.context, MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5102)), MakeOpParam(si));
    EXPECT_FALSE(strategy.IsSupport()); // perm 非 [1,0]
}

TEST(TransposeTransTo5HD10Strategy, direct_reject_dim)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeTransTo5HD10Para(compileInfo);
    ShapeInfo si = MakeShape({16, 8, 4}, {1, 0, 2}, 2);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeTransTo5HD10Strategy strategy(
        bundle.context, MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5102)), MakeOpParam(si));
    EXPECT_FALSE(strategy.IsSupport()); // dim != 2
}

TEST(TransposeTransTo5HD10Strategy, direct_reject_dtype)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeTransTo5HD10Para(compileInfo);
    ShapeInfo si = MakeShape({16, 8}, {1, 0}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeTransTo5HD10Strategy strategy(
        bundle.context, MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5102)), MakeOpParam(si));
    EXPECT_FALSE(strategy.IsSupport()); // 非 16bit
}

TEST(TransposeTransTo5HD10Strategy, direct_reject_small_first_dim)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeTransTo5HD10Para(compileInfo);
    ShapeInfo si = MakeShape({5, 8}, {1, 0}, 2);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeTransTo5HD10Strategy strategy(
        bundle.context, MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5102)), MakeOpParam(si));
    EXPECT_FALSE(strategy.IsSupport()); // shape[0] <= 5
}

// workspace 取不到（GetWorkspaceSizes(1)==nullptr）：收尾阶段写回失败 → FAILURE（硬错误）
TEST(TransposeTransTo5HD10Strategy, direct_workspace_null_failure)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeTransTo5HD10Para(compileInfo);
    ShapeInfo si = MakeShape({16, 8}, {1, 0}, 2);
    auto bundle = BuildTilingContext(para, false);
    optiling::TransposeTransTo5HD10Strategy strategy(
        bundle.context, MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5102)), MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    EXPECT_EQ(strategy.Execute(), TilingStatus::FAILURE);
}

// DAV_5102 直调：成功路径（e2e SoC 为 DAV_3510，无法命中 10007）
TEST(TransposeTransTo5HD10Strategy, direct_success_unreachable_by_e2e)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeTransTo5HD10Para(compileInfo);
    ShapeInfo si = MakeShape({16, 8}, {1, 0}, 2);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeTransTo5HD10Strategy strategy(
        bundle.context, MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5102)), MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    EXPECT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_TRANSTO5HD_10));
    EXPECT_EQ(bundle.context->GetBlockDim(), 1);
    const auto* data = GetTilingData<TransposeTransTo5HD10TilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->AvailableUbSize, 63488);
    EXPECT_EQ(data->RLen, 16);
    EXPECT_EQ(data->CLen, 8);
}

/* ============================================================================
 * TRANSTO5HD_10(10007)：DAV_5102 16bit 2D，超 uint32 行数 + UB 占用
 * ========================================================================== */
TEST(TransposeTransTo5HD10Strategy, transto5hd_10_large_shape_and_ub_budget)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeDummyPara(compileInfo);
    const int64_t arch = static_cast<int64_t>(Ops::Base::DAV_5102);
    const int64_t eleLen = 2;

    // 边界：shape[0]>5 才命中（16bit、perm=[1,0]、DAV_5102）
    auto boundaryBundle = BuildTilingContext(para);
    ShapeInfo r5 = MakeShape({5, 8}, {1, 0}, eleLen);
    optiling::TransposeTransTo5HD10Strategy r5Strategy(boundaryBundle.context, MakePlatformInfo(arch), MakeOpParam(r5));
    EXPECT_FALSE(r5Strategy.IsSupport());
    ShapeInfo r6 = MakeShape({6, 8}, {1, 0}, eleLen);
    optiling::TransposeTransTo5HD10Strategy r6Strategy(boundaryBundle.context, MakePlatformInfo(arch), MakeOpParam(r6));
    EXPECT_TRUE(r6Strategy.IsSupport());

    // 超大：RLen > uint32，CLen 保持可整块入 UB。
    auto bundle = BuildTilingContext(para);
    ShapeInfo si = MakeShape({kHugeDim, 64}, {1, 0}, eleLen);
    optiling::TransposeTransTo5HD10Strategy strategy(bundle.context, MakePlatformInfo(arch), MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    const auto* data = GetTilingData<TransposeTransTo5HD10TilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->RLen, kHugeDim);
    EXPECT_EQ(data->CLen, 64);
    EXPECT_GT(data->RLen, static_cast<int64_t>(kUint32Max));
    EXPECT_GT(data->AvailableUbSize, 0);

    // kernel：输入/输出各 2 个 buffer，每个 AvailableUbSize。
    EXPECT_LE(static_cast<int64_t>(data->AvailableUbSize) * 4, kUbSize);
}
