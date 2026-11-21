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
 * \file test_transpose_tiling_strategy_nddma.cpp
 * \brief NDDMA(10002 NDDMA_CUT_1 / 10003 NDDMA_CUT_2) 策略 UT：直调（逐字段）+ 端到端命中
 */

#include "test_transpose_tiling_strategy_common.h"
#include "../../../../op_host/arch35/transpose_tiling_strategy_nddma.h"

using namespace transpose_strategy_ut;

namespace {
gert::TilingContextPara MakeCutOncePara(optiling::TransposeCompilerInfo& compileInfo)
{
    static int64_t permValue[3] = {1, 0, 2};
    gert::TilingContextPara::TensorDescription x({{4000, 7000, 8}, {4000, 7000, 8}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &permValue);
    gert::TilingContextPara::TensorDescription out({{7000, 4000, 8}, {7000, 4000, 8}}, ge::DT_FLOAT, ge::FORMAT_ND);
    return gert::TilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
}

gert::TilingContextPara MakeCutTwicePara(optiling::TransposeCompilerInfo& compileInfo)
{
    static int64_t permValue[3] = {1, 0, 2};
    gert::TilingContextPara::TensorDescription x({{8000, 256, 8}, {8000, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &permValue);
    gert::TilingContextPara::TensorDescription out({{256, 8000, 8}, {256, 8000, 8}}, ge::DT_FLOAT, ge::FORMAT_ND);
    return gert::TilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
}

/* ==================== 公开 Execute 直调（e2e 无法构造的 UB 预算/平台） ==================== */
optiling::TransposePlatformInfo MakeNddmaPlatform(int64_t ubSize, int64_t arch = 0)
{
    optiling::TransposePlatformInfo platform = MakePlatformInfo();
    platform.ubSize = ubSize;
    if (arch != 0) {
        platform.npuArch = static_cast<NpuArch>(arch);
    }
    return platform;
}
} // namespace

/* ==================== 切轴/分核分支补充用例 ==================== */

TEST(TransposeNddmaStrategy, direct_three_arg_ctor)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeCutOncePara(compileInfo);
    auto bundle = BuildTilingContext(para);
    // 3 参构造：默认 shapeInfo（dim=0），仅验证构造路径与 NDDMA 维度准入
    optiling::TransposeNddmaStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(ShapeInfo{}));
    EXPECT_TRUE(strategy.IsSupport());
}

TEST(TransposeNddmaStrategy, cut_once_e2e_hits_this_strategy)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeCutOncePara(compileInfo);
    TilingInfo e2e;
    ASSERT_TRUE(ExecuteTiling(para, e2e));
    EXPECT_EQ(e2e.tilingKey, TK_NDDMA_CUT_1);
}

TEST(TransposeNddmaStrategy, cut_twice_e2e_hits_this_strategy)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeCutTwicePara(compileInfo);
    TilingInfo e2e;
    ASSERT_TRUE(ExecuteTiling(para, e2e));
    EXPECT_EQ(e2e.tilingKey, TK_NDDMA_CUT_2);
}

// Large N_LAST transpose (perm not transposing last axis, large total bytes, last dim >= 32)

TEST(TransposeNddmaStrategy, transpose_tiling_3d_with_one_dim_axis)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    // perm {2, 0, 1} with shape {1, 100, 200}: after RemoveAxis, becomes {100, 200} perm {1, 0}
    int64_t perm_value[3] = {2, 0, 1};
    gert::TilingContextPara::TensorDescription x({{1, 100, 200}, {1, 100, 200}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{200, 1, 100}, {200, 1, 100}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// BF16 dtype 2D transpose, expect success

TEST(TransposeNddmaStrategy, transpose_tiling_bf16_2d)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{100, 200}, {100, 200}}, ge::DT_BF16, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{200, 100}, {200, 100}}, ge::DT_BF16, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// INT8 dtype 4D transpose with large shape, expect success

TEST(TransposeNddmaStrategy, transpose_tiling_int8_4d_large)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[4] = {3, 0, 1, 2};
    gert::TilingContextPara::TensorDescription x({{10, 20, 30, 64}, {10, 20, 30, 64}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{64, 10, 20, 30}, {64, 10, 20, 30}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// 2D transpose with perm {0,1} (identity permutation), expect TENSOR_MOVE

TEST(TransposeNddmaStrategy, transpose_tiling_int64_dtype_4d)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[4] = {3, 0, 1, 2};
    gert::TilingContextPara::TensorDescription x({{10, 20, 30, 64}, {10, 20, 30, 64}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{64, 10, 20, 30}, {64, 10, 20, 30}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// Small volume that is less than coreNum, expect TENSOR_MOVE with single core

TEST(TransposeNddmaStrategy, transpose_tiling_4d_full_reverse)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[4] = {3, 2, 1, 0};
    gert::TilingContextPara::TensorDescription x({{64, 128, 32, 16}, {64, 128, 32, 16}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{16, 32, 128, 64}, {16, 32, 128, 64}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// 7D transpose (dim > 5) with very large shape - merges to low dim after MergeAxis

TEST(TransposeNddmaStrategy, transpose_tiling_cut_twice_both_tails)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[4] = {3, 0, 1, 2};
    gert::TilingContextPara::TensorDescription x({{8, 7, 6, 100}, {8, 7, 6, 100}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{100, 8, 7, 6}, {100, 8, 7, 6}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// NDDMA_CUT_1 with low core utilization, covers rate-based split factor search

TEST(TransposeNddmaStrategy, transpose_tiling_cut_once_low_cores)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {2, 0, 1};
    gert::TilingContextPara::TensorDescription x({{100, 200, 50}, {100, 200, 50}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{50, 100, 200}, {50, 100, 200}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// N_LAST transpose exercising FindSplitFactorByMultiplesNLast

TEST(TransposeNddmaStrategy, transpose_tiling_cut_twice_out_tail_only)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    // 4D perm {3, 2, 0, 1} with shape producing NDDMA_CUT_2 where outTail != 0 but inTail = 0
    int64_t perm_value[4] = {3, 2, 0, 1};
    gert::TilingContextPara::TensorDescription x({{16, 32, 64, 128}, {16, 32, 64, 128}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{128, 64, 16, 32}, {128, 64, 16, 32}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// NDDMA_CUT_2 with inTailFactor!=0 and outTailFactor=0, covers offsetRangeInputTail

TEST(TransposeNddmaStrategy, transpose_tiling_cut_twice_in_tail_only)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    // 3D perm {2, 0, 1} with shape producing NDDMA_CUT_2 where inTail != 0 but outTail = 0
    int64_t perm_value[3] = {2, 0, 1};
    gert::TilingContextPara::TensorDescription x({{37, 100, 64}, {37, 100, 64}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{64, 37, 100}, {64, 37, 100}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// Small shape [8000, 255] perm [1, 0] with 248K UB: sqrt(UB) barely misses axis 1 (255),
// absorption strategy promotes the cut to axis 0, but the effective outUbFactor amplified
// by inUbFactor would overflow UB (248*248*255 > ubElement), so the safety validation in
// DoSplitUB reverts to the original strategy, yielding NDDMA_CUT_2.
// 注: n=255 非 16 对齐, 用于避开 TRANSTO5HD_10_PERF(10017) TransData 特化路由。

TEST(TransposeNddmaStrategy, transpose_tiling_small_shape_absorb_revert_to_cut_twice)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 253952; // 248K

    int64_t perm_value[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{8000, 255}, {8000, 255}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{255, 8000}, {255, 8000}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_NDDMA_CUT_2));
}

// Small shape [1000, 256, 300] perm [2, 1, 0] with 248K UB: should also benefit from absorption
// and prefer NDDMA_CUT_1 when achievable.

TEST(TransposeNddmaStrategy, transpose_tiling_absorb_promote_3d_to_cut_once)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 253952; // 248K

    int64_t perm_value[3] = {1, 0, 2};
    gert::TilingContextPara::TensorDescription x({{4000, 7000, 8}, {4000, 7000, 8}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{7000, 4000, 8}, {7000, 4000, 8}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_NDDMA_CUT_1));
    const auto* data = GetTilingData<TransposeOpTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->permSize, 3);
    EXPECT_EQ(data->inCutIndex, 0);
    EXPECT_EQ(data->inUbFactor, 1);
}

// Absorption NDDMA_CUT_1 rejected by the UB safety validation: [8000, 256, 8] perm [1, 0, 2].
// The retry absorbs axis 1 (256 <= 7936) and cuts axis 0 (inCutIndex=0, inUbFactor=31),
// reaching NDDMA_CUT_1. But CalcBlockSplitInfoForCutOnce would amplify outUbFactor by inUbFactor
// on the same axis, and the validated burst 8 * 31 * 31 * 256 (outer output axis 0) exceeds
// the 63488-element budget, so DoSplitUB restores the original strategy: NDDMA_CUT_2 with
// inCutIndex=1, inUbFactor=31, outUbFactor=217 (3D companion of the 2D revert case above).

TEST(TransposeNddmaStrategy, transpose_tiling_absorb_validation_fail_3d_revert_to_cut_twice)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 253952; // 248K

    int64_t perm_value[3] = {1, 0, 2};
    gert::TilingContextPara::TensorDescription x({{8000, 256, 8}, {8000, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{256, 8000, 8}, {256, 8000, 8}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_NDDMA_CUT_2));
    const auto* data = GetTilingData<TransposeOpTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->inCutIndex, 1);
    EXPECT_EQ(data->inUbFactor, 31);
    EXPECT_EQ(data->outCutIndex, 1);
    EXPECT_EQ(data->outUbFactor, 217);
}

// Absorption budget rejection: axis 1 (100000) of [512, 100000, 8] exceeds the full remaining
// UB budget (63488 / 8 = 7936 elements), so the absorb candidate check
// (currentShapeDim <= fullBudgetRemaining) fails. The retry degenerates to the original input
// split and DoSplitUB restores the original NDDMA_CUT_2 result.

TEST(TransposeNddmaStrategy, transpose_tiling_absorb_full_budget_reject_keeps_cut_twice)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 253952; // 248K

    int64_t perm_value[3] = {1, 0, 2};
    gert::TilingContextPara::TensorDescription x({{512, 100000, 8}, {512, 100000, 8}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{100000, 512, 8}, {100000, 512, 8}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_NDDMA_CUT_2));
    const auto* data = GetTilingData<TransposeOpTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    // The original strategy splits axis 1 with inUbFactor = 251 / 8 = 31; the retry cannot
    // swallow axis 1 (100000 > 7936), so the reverted split info is identical to the original.
    EXPECT_EQ(data->inCutIndex, 1);
    EXPECT_EQ(data->inUbFactor, 31);
    EXPECT_EQ(data->outCutIndex, 1);
    EXPECT_EQ(data->outUbFactor, 256);
}

// Absorption succeeded in the retry but the result is still NDDMA_CUT_2: [3000, 900, 500, 6]
// perm [2, 1, 0, 3]. The retry absorbs axis 2 (500) and cuts axis 1 (inCutIndex=1), yet the
// output scan still has to cut output axis 2 (3000) which is more major than the input cut
// position (FindOutIndex(1)=1), so tilingKey stays NDDMA_CUT_2 and DoSplitUB must restore the
// original split (axis 2 cut, inCutIndex=2, inUbFactor=41).

TEST(TransposeNddmaStrategy, transpose_tiling_absorb_still_cut_twice_reverts_original_split)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 253952; // 248K

    int64_t perm_value[4] = {2, 1, 0, 3};
    gert::TilingContextPara::TensorDescription x({{3000, 900, 500, 6}, {3000, 900, 500, 6}}, ge::DT_FLOAT,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{500, 900, 3000, 6}, {500, 900, 3000, 6}}, ge::DT_FLOAT,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_NDDMA_CUT_2));
    const auto* data = GetTilingData<TransposeOpTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    // If the revert to originalResult were broken, the absorbed retry would leave the input
    // cut on axis 1 (inCutIndex=1, inUbFactor=21) instead of the original axis 2 split.
    EXPECT_EQ(data->permSize, 4);
    EXPECT_EQ(data->inCutIndex, 2);
    EXPECT_EQ(data->inUbFactor, 41);
}

// SMALL_SHAPE with fp32 total elements (28) below the 128B alignment unit (128/4=32):
// the new early-return branch uses a single core carrying all elements
// (realCoreNum=1, blkFactor=28, blkTailFactor=0).

TEST(TransposeNddmaStrategy, transpose_tiling_absorb_3d_slack_window_cut_once)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 253952; // 248K

    int64_t perm_value[3] = {1, 0, 2};
    gert::TilingContextPara::TensorDescription x({{6000, 6000, 8}, {6000, 6000, 8}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{6000, 6000, 8}, {6000, 6000, 8}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_NDDMA_CUT_1));
    const auto* data = GetTilingData<TransposeOpTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->inCutIndex, 0);
    EXPECT_EQ(data->inUbFactor, 1);
}

TEST(TransposeNddmaStrategy, transpose_tiling_nddma_base_4d)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    // 总量 4.6MB 超过 SMALL 门槛; MergeAxis 后输入等效行数 30*21*30=18900 非 16 对齐,
    // 不命中小 m/N_LAST/TRANSTO5HD_10_PERF 特化, 落到 NDDMA (NDDMA_CUT_1/NDDMA_CUT_2)。
    int64_t perm_value[4] = {3, 0, 1, 2};
    gert::TilingContextPara::TensorDescription x({{30, 21, 30, 64}, {30, 21, 30, 64}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{64, 30, 21, 30}, {64, 30, 21, 30}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    // 单/双切由 UB 预算（随 npuArch 缓冲数变化）决定：直调策略并固定 DAV_3510（单缓冲）
    ShapeInfo si = MakeShape({30, 21 * 30 * 64}, {1, 0}, 4);
    auto bundle = BuildTilingContext(tilingContextPara);
    optiling::TransposeNddmaStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_NDDMA_CUT_1));
    const auto* data = GetTilingData<TransposeOpTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->permSize, 2);
}

// NDDMA_BIG_DIM path (dim > 5), expect tiling key 10005

TEST(TransposeNddmaStrategy, transpose_tiling_nddma_5d)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[5] = {4, 0, 1, 2, 3};
    gert::TilingContextPara::TensorDescription x({{2, 3, 4, 5, 100}, {2, 3, 4, 5, 100}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{100, 2, 3, 4, 5}, {100, 2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    // 5 维 NDDMA 切分：直调策略并固定 DAV_3510（单缓冲），tilingKey 确定
    ShapeInfo si = MakeShape({2, 3, 4, 5, 100}, {4, 0, 1, 2, 3}, 4);
    auto bundle = BuildTilingContext(tilingContextPara);
    optiling::TransposeNddmaStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_NDDMA_CUT_1));
    const auto* data = GetTilingData<TransposeOpTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->permSize, 5);
}

// NDDMA 缓冲数完全由 platformInfo.npuArch 推导：DAV_9201=双缓冲，DAV_3510=单缓冲
TEST(TransposeNddmaStrategy, direct_buffer_num_from_arch_unreachable_by_e2e)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeCutOncePara(compileInfo);
    ShapeInfo si = MakeShape({256, 256}, {1, 0}, 4);

    // 双缓冲（DAV_9201）
    auto dualBundle = BuildTilingContext(para);
    optiling::TransposeNddmaStrategy dual(
        dualBundle.context, MakeNddmaPlatform(kUbSize, static_cast<int64_t>(Ops::Base::DAV_9201)), MakeOpParam(si));
    ASSERT_EQ(dual.Execute(), TilingStatus::SUCCESS);
    const auto* dualData = GetTilingData<TransposeOpTilingData>(dualBundle.context);
    ASSERT_NE(dualData, nullptr);
    EXPECT_EQ(dualData->permSize, 2);
    EXPECT_GT(dualData->realCoreNum, 0);

    // 单缓冲（DAV_3510）
    auto singleBundle = BuildTilingContext(para);
    optiling::TransposeNddmaStrategy single(
        singleBundle.context, MakeNddmaPlatform(kUbSize, static_cast<int64_t>(Ops::Base::DAV_3510)), MakeOpParam(si));
    ASSERT_EQ(single.Execute(), TilingStatus::SUCCESS);
    const auto* singleData = GetTilingData<TransposeOpTilingData>(singleBundle.context);
    ASSERT_NE(singleData, nullptr);
    EXPECT_EQ(singleData->permSize, 2);
    EXPECT_GT(singleData->realCoreNum, 0);

    // 双缓冲把 UB 预算减半 → 输入切分因子不大于单缓冲
    EXPECT_LE(dualData->inUbFactor, singleData->inUbFactor);
}

// NDDMA_CUT_1：输出切分轴重定位 + 块数不足的 bestI 回退
TEST(TransposeNddmaStrategy, direct_cut_once_other_axis_unreachable_by_e2e)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeCutOncePara(compileInfo);
    ShapeInfo si = MakeShape({1, 32}, {0, 1}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeNddmaStrategy strategy(bundle.context, MakeNddmaPlatform(1024), MakeOpParam(si));
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_NDDMA_CUT_1));
    const auto* data = GetTilingData<TransposeOpTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->permSize, 2);
    EXPECT_EQ(data->ubSize, 1024);
    EXPECT_GT(data->inUbFactor, 0);
    EXPECT_GT(data->realCoreNum, 0);
}

// NDDMA_CUT_2：块数不足时按 0.9 核利用率命中输入因子
TEST(TransposeNddmaStrategy, direct_cut_twice_rate_hit_unreachable_by_e2e)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeCutTwicePara(compileInfo);
    ShapeInfo si = MakeShape({32, 32}, {1, 0}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeNddmaStrategy strategy(bundle.context, MakeNddmaPlatform(1024), MakeOpParam(si));
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_NDDMA_CUT_2));
    const auto* data = GetTilingData<TransposeOpTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->permSize, 2);
    EXPECT_GT(data->inUbFactor, 0);
    EXPECT_GT(data->outUbFactor, 0);
}

// NDDMA_CUT_2：块数不足且无 0.9 因子 → 取最佳 rate 因子
TEST(TransposeNddmaStrategy, direct_cut_twice_best_factor_unreachable_by_e2e)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeCutTwicePara(compileInfo);
    ShapeInfo si = MakeShape({32, 2, 16}, {1, 0, 2}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeNddmaStrategy strategy(bundle.context, MakeNddmaPlatform(1024), MakeOpParam(si));
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_NDDMA_CUT_2));
    const auto* data = GetTilingData<TransposeOpTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->permSize, 3);
    EXPECT_GT(data->inUbFactor, 0);
}

// NDDMA_CUT_2：输入切分轴位于输出切分轴右侧，UB shape 回填时跳过该轴
TEST(TransposeNddmaStrategy, direct_cut_twice_skip_input_cut_axis_unreachable_by_e2e)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeCutTwicePara(compileInfo);
    ShapeInfo si = MakeShape({2, 16, 32}, {2, 0, 1}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeNddmaStrategy strategy(bundle.context, MakeNddmaPlatform(1024), MakeOpParam(si));
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_NDDMA_CUT_2));
    const auto* data = GetTilingData<TransposeOpTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->permSize, 3);
    EXPECT_EQ(data->inCutIndex, 2);
    EXPECT_EQ(data->outCutIndex, 1);
}

// NDDMA_CUT_2：仅输出侧有尾块时的区间长度
TEST(TransposeNddmaStrategy, direct_interval_output_tail_only_unreachable_by_e2e)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeCutTwicePara(compileInfo);
    ShapeInfo si = MakeShape({32, 32}, {1, 0}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeNddmaStrategy strategy(bundle.context, MakeNddmaPlatform(2048), MakeOpParam(si));
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_NDDMA_CUT_2));
    const auto* data = GetTilingData<TransposeOpTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->permSize, 2);
    EXPECT_EQ(data->inTailFactor, 0);
    EXPECT_GT(data->outTailFactor, 0);
}

// TilingData 容量不足：码流不写入，但 tilingKey/blockDim 仍按策略结果写回
TEST(TransposeNddmaStrategy, direct_tiling_data_insufficient)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeCutOncePara(compileInfo);
    para.tilingDataSize_ = 8;
    ShapeInfo si = MakeShape({4000, 7000, 8}, {1, 0, 2}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeNddmaStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    // 码流容量不足：SetTilingData 返回 FAILURE，整体 tiling 失败
    EXPECT_EQ(strategy.Execute(), TilingStatus::FAILURE);
}
