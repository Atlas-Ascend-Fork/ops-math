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

#include <utility>

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

// 2D [8192,62] fp32 的全量调度现由注册在前的 SMALL_SHAPE_2D(10012) 承接；
// 本用例直调 Gather，保留 Gather 路径自身的端到端校验。
TEST(TransposeGatherStrategy, direct_hits_this_strategy)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    ShapeInfo si = MakeShape({8192, 62}, {1, 0}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeGatherStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_GATHER));
    EXPECT_EQ(bundle.context->GetBlockDim(), 41U);
    const auto* data = GetTilingData<TransposeGatherTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->dataTensorSize, 50784);
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
    EXPECT_EQ(tilingInfo.blockNum, 40U);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);
    const auto* tilingData = GetTilingData<Transpose::TransposeGatherTilingData>(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->dataTensorSize, 43680);
    EXPECT_EQ(tilingData->indexTensorSize, 21840);
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

// [1000000,3] fp32：尾轴仅 3 列，Gather 因 MTE gate/块数约束被拒（DoExecute → FALLBACK）。
TEST(TransposeGatherStrategy, direct_gather_reject_mte_gate)
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
}

// [500,4096,7] perm [2,1,0]：3D 尾轴转置，perm 无相邻升序对（MergeAxis 后仍为 3D）；
// Gather 因 bank conflict/MTE 约束被拒（FALLBACK）。
TEST(TransposeGatherStrategy, direct_gather_reject_bank_conflict)
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
}

/* ============================================================================
 * GATHER(10006)：4B/8B 的 4×data + index UB 占用 + 超 uint32 shape
 *
 * 1B/2B 的 data buffer 更小（ub/6、ub/5），而 Gather 索引需预留 1/4 UB 且
 * MTE 效率门槛为 16KB，当前 shape 下会被硬件约束拒绝并回落 NDDMA；这里只对
 * 可命中的 4B（索引同宽）与 8B（索引 32bit）分支校验 UB 公式。
 * ========================================================================== */
TEST(TransposeGatherStrategy, gather_ub_budget_4b_8b)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeDummyPara(compileInfo);

    std::vector<std::pair<int64_t, ShapeInfo>> cases;
    cases.push_back({4, MakeShape({8192, 62}, {1, 0}, 4)});
    cases.push_back({8, MakeShape({36, 203, 26, 31}, {3, 0, 1, 2}, 8)});
    for (const auto& item : cases) {
        const int64_t eleLen = item.first;
        auto bundle = BuildTilingContext(para);
        optiling::TransposeGatherStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(item.second));
        ASSERT_TRUE(strategy.IsSupport()) << "eleLen=" << eleLen;
        ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS) << "eleLen=" << eleLen;
        const auto* data = GetTilingData<TransposeGatherTilingData>(bundle.context);
        ASSERT_NE(data, nullptr) << "eleLen=" << eleLen;
        EXPECT_GT(data->dataTensorSize, 0U) << "eleLen=" << eleLen;
        EXPECT_GT(data->indexTensorSize, 0U) << "eleLen=" << eleLen;
        // kernel：in0 | in1 | out0 | out1 | index。
        const uint64_t ubUsage = static_cast<uint64_t>(data->dataTensorSize) * 4U + data->indexTensorSize;
        EXPECT_LE(ubUsage, static_cast<uint64_t>(kUbSize)) << "eleLen=" << eleLen;
    }
}

TEST(TransposeGatherStrategy, gather_large_shape_and_boundary)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeDummyPara(compileInfo);

    auto bundle = BuildTilingContext(para);
    ShapeInfo si = MakeShape({kHugeDim, 62}, {1, 0}, 4);
    optiling::TransposeGatherStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    const auto* data = GetTilingData<TransposeGatherTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->inUbCutAxisSize, kHugeDim);
    EXPECT_GT(data->inUbCutAxisSize, static_cast<int64_t>(kUint32Max));
    EXPECT_EQ(data->outUbCutAxisSize, kHugeDim);
    EXPECT_LE(static_cast<uint64_t>(data->dataTensorSize) * 4U + data->indexTensorSize, static_cast<uint64_t>(kUbSize));

    // 边界：related / 尾轴不转置 / 4bit 均不命中。
    auto missBundle = BuildTilingContext(para);
    ShapeInfo notLast = MakeShape({128, 64}, {0, 1}, 4);
    optiling::TransposeGatherStrategy notLastStrategy(missBundle.context, MakePlatformInfo(), MakeOpParam(notLast));
    EXPECT_FALSE(notLastStrategy.IsSupport());

    ShapeInfo b4 = MakeShape({128, 64}, {1, 0}, ge::kDataTypeSizeBitOffset + 4);
    optiling::TransposeGatherStrategy b4Strategy(missBundle.context, MakePlatformInfo(), MakeOpParam(b4));
    EXPECT_FALSE(b4Strategy.IsSupport());

    ShapeInfo related = MakeShape({128, 64}, {1, 0}, 4);
    optiling::TransposeGatherStrategy relatedStrategy(missBundle.context, MakePlatformInfo(),
                                                      MakeOpParam(related, true));
    EXPECT_FALSE(relatedStrategy.IsSupport());
}
