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
 * \file test_transpose_tiling_strategy_gather.cpp
 * \brief GATHER(10006) 策略 UT：直调策略（逐字段断言）+ 端到端命中
 */

#include <gmock/gmock.h>
#include "test_transpose_tiling_strategy_common.h"
#include "../../../../op_host/arch35/transpose_tiling_strategy_gather.h"
#include "../../../../op_host/arch35/transpose_tiling_strategy_nddma.h"

using namespace transpose_strategy_ut;

namespace {
gert::TilingContextPara MakePara(optiling::TransposeCompilerInfo& compileInfo)
{
    static int64_t permValue[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{8192, 62}, {8192, 62}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &permValue);
    gert::TilingContextPara::TensorDescription out({{62, 8192}, {62, 8192}}, ge::DT_FLOAT, ge::FORMAT_ND);
    return gert::TilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
}
} // namespace

TEST(TransposeGatherStrategy, related_mode_support_false)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    ShapeInfo si = MakeShape({8192, 62}, {1, 0}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeGatherStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si, true));
    EXPECT_FALSE(strategy.IsSupport());
}

TEST(TransposeGatherStrategy, e2e_hits_this_strategy)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    TilingInfo e2e;
    ASSERT_TRUE(ExecuteTiling(para, e2e));
    EXPECT_EQ(e2e.tilingKey, TK_GATHER);
    const auto* data = GetTilingData<TransposeGatherTilingData>(e2e);
    ASSERT_NE(data, nullptr);
    EXPECT_GT(data->usedCoreCnt, 0U);
    EXPECT_GT(data->dataTensorSize, 0U);
}

TEST(TransposeGatherStrategy, transpose_tiling_01)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[4] = {3, 0, 1, 2};
    gert::TilingContextPara::TensorDescription x({{36, 203, 26, 31}, {36, 203, 26, 31}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{31, 36, 203, 26}, {31, 36, 203, 26}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10006;
    std::vector<size_t> expectWorkspaces = {0};
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);
    const auto* tilingData = GetTilingData<Transpose::TransposeGatherTilingData>(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->tilingKey, 10006);
    EXPECT_EQ(tilingData->dataTensorSize, 43680);
    EXPECT_EQ(tilingData->indexTensorSize, 21840);
    EXPECT_EQ(tilingData->usedCoreCnt, 40);
    EXPECT_EQ(tilingData->blkAxesCnt, 1);
    EXPECT_EQ(tilingData->blkInUbCutPos, 0);
    EXPECT_EQ(tilingData->blkOutUbCutPos, -1);
    EXPECT_EQ(tilingData->ubAxesCnt, 2);
    EXPECT_EQ(tilingData->inUbInCutPos, 0);
    EXPECT_EQ(tilingData->inUbOutCutPos, 0);
    EXPECT_EQ(tilingData->outUbInCutPos, 1);
    EXPECT_EQ(tilingData->outUbOutCutPos, 1);
    EXPECT_EQ(tilingData->blkFactor, 27);
    EXPECT_EQ(tilingData->blkTailFactor, 27);
    EXPECT_EQ(tilingData->inUbCutAxisSize, 190008);
    EXPECT_EQ(tilingData->outUbCutAxisSize, 190008);
    EXPECT_EQ(tilingData->inUbCutAxisFactor, 176);
    EXPECT_EQ(tilingData->outUbCutAxisFactor, 176);
    EXPECT_EQ(tilingData->axis0InSrcStride, 0);
    EXPECT_EQ(tilingData->axis1InSrcStride, 0);
    EXPECT_EQ(tilingData->axis2InSrcStride, 0);
    EXPECT_EQ(tilingData->axis0OutDstStride, 190008);
    EXPECT_EQ(tilingData->axis1OutDstStride, 0);
    EXPECT_EQ(tilingData->axis2OutDstStride, 0);
    EXPECT_THAT(tilingData->blkAxes, ::testing::ElementsAreArray(std::vector<int64_t>{1080, 1, 1, 1, 1, 1, 1, 1}));
    EXPECT_THAT(tilingData->blkAxesInAOffset,
                ::testing::ElementsAreArray(std::vector<int64_t>{5456, 1, 1, 1, 1, 1, 1, 1}));
    EXPECT_THAT(tilingData->blkAxesOutAOffset,
                ::testing::ElementsAreArray(std::vector<int64_t>{176, 1, 1, 1, 1, 1, 1, 1}));
    EXPECT_THAT(tilingData->inUbAxes, ::testing::ElementsAreArray(std::vector<int64_t>{176, 31, 1, 1, 1, 1}));
    EXPECT_THAT(tilingData->outUbAxes, ::testing::ElementsAreArray(std::vector<int64_t>{31, 176, 1, 1, 1, 1}));
    EXPECT_THAT(tilingData->ubPerm, ::testing::ElementsAreArray(std::vector<int64_t>{1, 0, 15, 15, 15, 15}));
}

// Gather 因 MTE gate / bank 冲突被拒（DoExecute → FALLBACK），由 NDDMA 兜底。
// NDDMA 单/双切由 UB 预算（随 npuArch 的缓冲数变化）决定，故直调策略并固定 DAV_3510
// （单缓冲），tilingKey 确定为 NDDMA_CUT_1(10002)。
TEST(TransposeGatherStrategy, direct_gather_reject_then_nddma_cut_once)
{
    auto compileInfo = MakeCompilerInfo();
    int64_t perm_value[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{1000000, 3}, {1000000, 3}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{3, 1000000}, {3, 1000000}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    ShapeInfo si = MakeShape({1000000, 3}, {1, 0}, 4);

    auto gatherBundle = BuildTilingContext(tilingContextPara);
    optiling::TransposeGatherStrategy gather(gatherBundle.context, MakePlatformInfo(), MakeOpParam(si));
    ASSERT_TRUE(gather.IsSupport());
    EXPECT_EQ(gather.Execute(), TilingStatus::FALLBACK);

    auto nddmaBundle = BuildTilingContext(tilingContextPara);
    optiling::TransposeNddmaStrategy nddma(nddmaBundle.context, MakePlatformInfo(), MakeOpParam(si));
    ASSERT_TRUE(nddma.IsSupport());
    ASSERT_EQ(nddma.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(nddmaBundle.context->GetTilingKey(), static_cast<uint64_t>(TK_NDDMA_CUT_1));
    const auto* data = GetTilingData<TransposeOpTilingData>(nddmaBundle.context);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->permSize, 2);
}

// Last-axis-transpose 3D whose gather candidate is rejected by the MTE gate / bank conflict
// checks (the split totalSizeInUb or index step does not meet the hardware constraints).
// perm [2, 1, 0] has no adjacent ascending pair, so the shape stays 3D after MergeAxis.
// The tiling must still succeed through the fallback NDDMA/CUT path.

// 同上（3D perm [2,1,0]）：Gather 被拒后由 NDDMA 兜底，DAV_3510 单缓冲下 tilingKey 确定为
// NDDMA_CUT_2(10003)。
TEST(TransposeGatherStrategy, direct_gather_reject_then_nddma_cut_twice)
{
    auto compileInfo = MakeCompilerInfo();
    int64_t perm_value[3] = {2, 1, 0};
    gert::TilingContextPara::TensorDescription x({{500, 4096, 7}, {500, 4096, 7}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{7, 4096, 500}, {7, 4096, 500}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    ShapeInfo si = MakeShape({500, 4096, 7}, {2, 1, 0}, 4);

    auto gatherBundle = BuildTilingContext(tilingContextPara);
    optiling::TransposeGatherStrategy gather(gatherBundle.context, MakePlatformInfo(), MakeOpParam(si));
    ASSERT_TRUE(gather.IsSupport());
    EXPECT_EQ(gather.Execute(), TilingStatus::FALLBACK);

    auto nddmaBundle = BuildTilingContext(tilingContextPara);
    optiling::TransposeNddmaStrategy nddma(nddmaBundle.context, MakePlatformInfo(), MakeOpParam(si));
    ASSERT_TRUE(nddma.IsSupport());
    ASSERT_EQ(nddma.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(nddmaBundle.context->GetTilingKey(), static_cast<uint64_t>(TK_NDDMA_CUT_2));
    const auto* data = GetTilingData<TransposeOpTilingData>(nddmaBundle.context);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->permSize, 3);
}

// Second absorption-success data point in the slack window: [6000, 6000, 8] perm [1, 0, 2].
// C*B = 8*6000 = 48000 lands in (ubElement/2, ubElement], so after absorbing axis 2 and
// axis 1 the input factor floor(63488/48000) = 1 and the 32B-aligned validation burst
// 8 * 1 * 6000 = 48000 fits the budget. First strategy cuts axis 1 (NDDMA_CUT_2), absorption
// retry keeps NDDMA_CUT_1 cutting axis 0 with inUbFactor=1.
