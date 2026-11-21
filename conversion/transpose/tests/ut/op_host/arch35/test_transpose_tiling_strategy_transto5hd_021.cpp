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
 * \file test_transpose_tiling_strategy_transto5hd_021.cpp
 * \brief TRANSTO5HD_021(10008) 策略 UT：直调策略（DAV_5102 context）+ 准入/回落分支
 */

#include "test_transpose_tiling_strategy_common.h"
#include "../../../../op_host/arch35/transpose_tiling_strategy_transto5hd_021.h"

using namespace transpose_strategy_ut;

namespace {
gert::TilingContextPara Make021Para(optiling::TransposeCompilerInfo& compileInfo)
{
    static int64_t permValue[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{128, 16, 128}, {128, 16, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &permValue);
    gert::TilingContextPara::TensorDescription out({{128, 128, 16}, {128, 128, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    return gert::TilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
}
} // namespace

TEST(TransposeTransTo5HD021Strategy, related_mode_support_false)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = Make021Para(compileInfo);
    ShapeInfo si = MakeShape({128, 16, 128}, {0, 2, 1}, 2);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeTransTo5HD021Strategy strategy(
        bundle.context, MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5102)), MakeOpParam(si, true));
    EXPECT_FALSE(strategy.IsSupport());
}

TEST(TransposeTransTo5HD021Strategy, transpose_transto5hd_021_float32_rconv)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{2, 16, 8}, {2, 16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{2, 8, 16}, {2, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    // 021 准入（dim==3 且 perm=[0,2,1]、dtype/HW 与体积门槛）不满足 → 不命中，
    // 交由后续策略兜底（只校验准入，避免依赖运行时 npuArch 的 tilingKey）
    ShapeInfo si = MakeShape({2, 16, 8}, {0, 2, 1}, 4);
    auto bundle = BuildTilingContext(tilingContextPara);
    optiling::TransposeTransTo5HD021Strategy strategy(
        bundle.context, MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5102)), MakeOpParam(si));
    EXPECT_FALSE(strategy.IsSupport());
}

TEST(TransposeTransTo5HD021Strategy, transpose_021vconv_float32_rconv_aligned_w)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{1, 32, 24}, {1, 32, 24}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{1, 24, 32}, {1, 24, 32}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    // 021 准入（dim==3 且 perm=[0,2,1]、dtype/HW 与体积门槛）不满足 → 不命中，
    // 交由后续策略兜底（只校验准入，避免依赖运行时 npuArch 的 tilingKey）
    ShapeInfo si = MakeShape({32, 24}, {1, 0}, 4);
    auto bundle = BuildTilingContext(tilingContextPara);
    optiling::TransposeTransTo5HD021Strategy strategy(
        bundle.context, MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5102)), MakeOpParam(si));
    EXPECT_FALSE(strategy.IsSupport());
}

TEST(TransposeTransTo5HD021Strategy, transpose_021vconv_float32_cconv)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{1, 8, 32}, {1, 8, 32}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{1, 32, 8}, {1, 32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    // 021 准入（dim==3 且 perm=[0,2,1]、dtype/HW 与体积门槛）不满足 → 不命中，
    // 交由后续策略兜底（只校验准入，避免依赖运行时 npuArch 的 tilingKey）
    ShapeInfo si = MakeShape({8, 32}, {1, 0}, 4);
    auto bundle = BuildTilingContext(tilingContextPara);
    optiling::TransposeTransTo5HD021Strategy strategy(
        bundle.context, MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5102)), MakeOpParam(si));
    EXPECT_FALSE(strategy.IsSupport());
}

TEST(TransposeTransTo5HD021Strategy, transpose_021vconv_int32_rconv)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{2, 16, 16}, {2, 16, 16}}, ge::DT_INT32, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{2, 16, 16}, {2, 16, 16}}, ge::DT_INT32, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    // 021 准入（dim==3 且 perm=[0,2,1]、dtype/HW 与体积门槛）不满足 → 不命中，
    // 交由后续策略兜底（只校验准入，避免依赖运行时 npuArch 的 tilingKey）
    ShapeInfo si = MakeShape({2, 16, 16}, {0, 2, 1}, 4);
    auto bundle = BuildTilingContext(tilingContextPara);
    optiling::TransposeTransTo5HD021Strategy strategy(
        bundle.context, MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5102)), MakeOpParam(si));
    EXPECT_FALSE(strategy.IsSupport());
}

TEST(TransposeTransTo5HD021Strategy, transpose_021vconv_int8_rconv)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{2, 32, 32}, {2, 32, 32}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{2, 32, 32}, {2, 32, 32}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    // 021 准入（dim==3 且 perm=[0,2,1]、dtype/HW 与体积门槛）不满足 → 不命中，
    // 交由后续策略兜底（只校验准入，避免依赖运行时 npuArch 的 tilingKey）
    ShapeInfo si = MakeShape({2, 32, 32}, {0, 2, 1}, 1);
    auto bundle = BuildTilingContext(tilingContextPara);
    optiling::TransposeTransTo5HD021Strategy strategy(
        bundle.context, MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5102)), MakeOpParam(si));
    EXPECT_FALSE(strategy.IsSupport());
}

TEST(TransposeTransTo5HD021Strategy, transpose_021vconv_uint8_rconv)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{1, 64, 32}, {1, 64, 32}}, ge::DT_UINT8, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{1, 32, 64}, {1, 32, 64}}, ge::DT_UINT8, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    // 021 准入（dim==3 且 perm=[0,2,1]、dtype/HW 与体积门槛）不满足 → 不命中，
    // 交由后续策略兜底（只校验准入，避免依赖运行时 npuArch 的 tilingKey）
    ShapeInfo si = MakeShape({64, 32}, {1, 0}, 1);
    auto bundle = BuildTilingContext(tilingContextPara);
    optiling::TransposeTransTo5HD021Strategy strategy(
        bundle.context, MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5102)), MakeOpParam(si));
    EXPECT_FALSE(strategy.IsSupport());
}

// 5D NDDMA path, dim = 5 (<= TP_NDDMA_MAX_DIM_NUM), last axis transposed, large shape

/* ==================== 准入/失败分支补充用例 ==================== */

TEST(TransposeTransTo5HD021Strategy, direct_reject_perm)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = Make021Para(compileInfo);
    ShapeInfo si = MakeShape({2, 16, 16}, {0, 1, 2}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeTransTo5HD021Strategy strategy(
        bundle.context, MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5102)), MakeOpParam(si));
    EXPECT_FALSE(strategy.IsSupport()); // perm 非 [0,2,1]
}

TEST(TransposeTransTo5HD021Strategy, direct_reject_dim)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = Make021Para(compileInfo);
    ShapeInfo si = MakeShape({16, 16}, {0, 2, 1}, 4); // perm 合法但 dim!=3
    auto bundle = BuildTilingContext(para);
    optiling::TransposeTransTo5HD021Strategy strategy(
        bundle.context, MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5102)), MakeOpParam(si));
    EXPECT_FALSE(strategy.IsSupport()); // dim != 3
}

TEST(TransposeTransTo5HD021Strategy, direct_reject_dtype)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = Make021Para(compileInfo);
    ShapeInfo si = MakeShape({1, 64, 64}, {0, 2, 1}, 8);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeTransTo5HD021Strategy strategy(
        bundle.context, MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5102)), MakeOpParam(si));
    EXPECT_FALSE(strategy.IsSupport()); // 非 8/16/32bit
}

TEST(TransposeTransTo5HD021Strategy, direct_reject_small_hw)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = Make021Para(compileInfo);
    ShapeInfo si = MakeShape({1, 8, 64}, {0, 2, 1}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeTransTo5HD021Strategy strategy(
        bundle.context, MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5102)), MakeOpParam(si));
    EXPECT_FALSE(strategy.IsSupport()); // H <= 8
}

TEST(TransposeTransTo5HD021Strategy, direct_reject_hw_product)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = Make021Para(compileInfo);
    ShapeInfo si = MakeShape({1, 9, 9}, {0, 2, 1}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeTransTo5HD021Strategy strategy(
        bundle.context, MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5102)), MakeOpParam(si));
    EXPECT_FALSE(strategy.IsSupport()); // H*W < 448
}

TEST(TransposeTransTo5HD021Strategy, direct_reject_align_efficiency)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = Make021Para(compileInfo);
    ShapeInfo si = MakeShape({1, 33, 33}, {0, 2, 1}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeTransTo5HD021Strategy strategy(
        bundle.context, MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5102)), MakeOpParam(si));
    EXPECT_FALSE(strategy.IsSupport()); // HW 填充效率 <= 50%
}

TEST(TransposeTransTo5HD021Strategy, direct_reject_small_volume)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = Make021Para(compileInfo);
    ShapeInfo si = MakeShape({1, 33, 32}, {0, 2, 1}, 1);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeTransTo5HD021Strategy strategy(
        bundle.context, MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5102)), MakeOpParam(si));
    EXPECT_FALSE(strategy.IsSupport()); // 总数据量 < 70KB
}

TEST(TransposeTransTo5HD021Strategy, transpose_021vconv_int8_cconv_fallback)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{1, 16, 32}, {1, 16, 32}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{1, 32, 16}, {1, 32, 16}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    // 体积 512B < 021 准入的 70KB 门槛 → 021 不命中，由 SMALL_SHAPE(10001) 兜底。
    // 首轴 size=1 被 RemoveAxis 消除，简化后 dim=2（16,32）/ perm=[1,0]
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE));
    const auto* tilingData = GetTilingData<TransposeOpTilingData>(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->permSize, 2);
}

// 1D tensor (perm=[0]), expect TENSOR_MOVE tiling key 10000

TEST(TransposeTransTo5HD021Strategy, transpose_tiling_fp16_3d_perm_021)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{100, 200, 64}, {100, 200, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{100, 64, 200}, {100, 64, 200}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

TEST(TransposeTransTo5HD021Strategy, transpose_021vconv_int8_cconv_hlen16_wlen128)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{128, 16, 128}, {128, 16, 128}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{128, 128, 16}, {128, 128, 16}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    // 10008 仅在 DAV_5102 命中：直调策略并固定 DAV_5102，tilingKey 确定为 10008
    ShapeInfo si = MakeShape({128, 16, 128}, {0, 2, 1}, 1);
    auto bundle = BuildTilingContext(tilingContextPara);
    optiling::TransposeTransTo5HD021Strategy strategy(
        bundle.context, MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5102)), MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_TRANSTO5HD_021));
    const auto* tilingData = GetTilingData<TransposeTransTo5HD021TilingData>(bundle.context);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->HLen, 16);
    EXPECT_EQ(tilingData->WLen, 128);
    EXPECT_EQ(tilingData->UseRConv, false);
    EXPECT_EQ(tilingData->HAlignBlockElem, 32);
    EXPECT_EQ(tilingData->WAlignBlockElem, 128);
    EXPECT_EQ(bundle.context->GetWorkspaceNum(), 1U);
}

TEST(TransposeTransTo5HD021Strategy, transpose_021vconv_int8_cconv_hlen16_wlen64)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{1600, 16, 64}, {1600, 16, 64}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{1600, 64, 16}, {1600, 64, 16}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    // 10008 仅在 DAV_5102 命中：直调策略并固定 DAV_5102，tilingKey 确定为 10008
    ShapeInfo si = MakeShape({1600, 16, 64}, {0, 2, 1}, 1);
    auto bundle = BuildTilingContext(tilingContextPara);
    optiling::TransposeTransTo5HD021Strategy strategy(
        bundle.context, MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5102)), MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_TRANSTO5HD_021));
    const auto* tilingData = GetTilingData<TransposeTransTo5HD021TilingData>(bundle.context);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->HLen, 16);
    EXPECT_EQ(tilingData->WLen, 64);
    EXPECT_EQ(tilingData->UseRConv, false);
    EXPECT_EQ(tilingData->HAlignBlockElem, 32);
    EXPECT_EQ(tilingData->WAlignBlockElem, 64);
    EXPECT_EQ(bundle.context->GetWorkspaceNum(), 1U);
}

// workspace 取不到（GetWorkspaceSizes(1)==nullptr）：收尾阶段写回失败 → FAILURE（硬错误）
TEST(TransposeTransTo5HD021Strategy, direct_workspace_null_failure)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = Make021Para(compileInfo);
    ShapeInfo si = MakeShape({1, 300, 300}, {0, 2, 1}, 1);
    auto bundle = BuildTilingContext(para, false);
    optiling::TransposeTransTo5HD021Strategy strategy(
        bundle.context, MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5102)), MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    EXPECT_EQ(strategy.Execute(), TilingStatus::FAILURE);
}

// DAV_5102 直调：成功路径（e2e SoC 为 DAV_3510，无法命中 10008）
TEST(TransposeTransTo5HD021Strategy, direct_success_unreachable_by_e2e)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = Make021Para(compileInfo);
    ShapeInfo si = MakeShape({1, 300, 300}, {0, 2, 1}, 1);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeTransTo5HD021Strategy strategy(
        bundle.context, MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5102)), MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    EXPECT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_TRANSTO5HD_021));
    EXPECT_GT(bundle.context->GetBlockDim(), 0U);
    const auto* data = GetTilingData<TransposeTransTo5HD021TilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    ASSERT_NE(data, nullptr);
    EXPECT_GT(data->UsedCoreNum, 0);
    EXPECT_GT(data->AvailableUbSize, 0);
    EXPECT_EQ(data->HLen, 300);
    EXPECT_EQ(data->WLen, 300);
}
