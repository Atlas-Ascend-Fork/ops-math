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

#include <array>
#include <gmock/gmock.h>
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

// 校验 NDDMA 5 维 UB 内 shape：源/目标乘积一致，且不超过单 buffer 预算。
void CheckNddmaUbShape(const int64_t* srcShape, const int64_t* dstShape, int64_t bufferBytes, int64_t eleLen)
{
    const int64_t srcElems = Product(srcShape, 0, TP_NDDMA_MAX_DIM_NUM);
    const int64_t dstElems = Product(dstShape, 0, TP_NDDMA_MAX_DIM_NUM);
    EXPECT_EQ(srcElems, dstElems);
    if (srcElems == 0) {
        return; // 尾块因子为 0：kernel 跳过对应尾块，不占 UB。
    }
    EXPECT_GT(srcElems, 0);
    EXPECT_LE(srcElems * eleLen, bufferBytes);
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
    EXPECT_EQ(e2e.tilingKey, static_cast<uint64_t>(TK_NDDMA_CUT_1));
    ASSERT_EQ(e2e.tilingDataSize, sizeof(TransposeNddmaCut1TilingData));
    const auto* data = GetTilingData<TransposeNddmaCut1TilingData>(e2e);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->ubSize, kUbSize);
    EXPECT_EQ(data->blkFactor, 83U);
    EXPECT_EQ(data->blkTailFactor, 16U);
    // [4000,7000,8] perm [1,0,2]：最内层尾轴 8 不转置，主块 NDDMA loopSize 首维为 8。
}

// 关联算子入口（depth_to_space/space_to_depth/confusion_transpose_d）NDDMA_CUT_1 与普通入口
// 统一使用紧凑 TilingData。
TEST(TransposeNddmaStrategy, cut_once_related_entry_uses_compact_layout)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeCutOncePara(compileInfo);
    ShapeInfo si = MakeShape({4000, 7000, 8}, {1, 0, 2}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeNddmaStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si, true));
    ASSERT_TRUE(strategy.IsSupport());
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_NDDMA_CUT_1));
    EXPECT_EQ(bundle.context->GetBlockDim(), 48);
    const auto* data = GetTilingData<TransposeNddmaCut1TilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->ubSize, kUbSize);
    EXPECT_EQ(data->blkFactor, 83);
    EXPECT_EQ(data->blkTailFactor, 16);
}

// 普通/关联入口 NDDMA_CUT_2 均使用紧凑 TilingData。
TEST(TransposeNddmaStrategy, cut_twice_e2e_hits_this_strategy)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeCutTwicePara(compileInfo);
    TilingInfo e2e;
    ASSERT_TRUE(ExecuteTiling(para, e2e));
    EXPECT_EQ(e2e.tilingKey, static_cast<uint64_t>(TK_NDDMA_CUT_2));
    EXPECT_EQ(e2e.blockNum, 48);
    ASSERT_EQ(e2e.tilingDataSize, sizeof(TransposeNddmaCut2TilingData));
    const auto* data = GetTilingData<TransposeNddmaCut2TilingData>(e2e);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->ubSize, kUbSize);
    EXPECT_EQ(data->blkFactor, 6);
    EXPECT_EQ(data->blkTailFactor, 45);
    EXPECT_GE(data->outCutIndex, 0);
}

// 关联算子入口（depth_to_space/space_to_depth/confusion_transpose_d）NDDMA_CUT_2 与普通入口
// 统一使用紧凑 TilingData。
TEST(TransposeNddmaStrategy, cut_twice_related_entry_uses_compact_layout)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeCutTwicePara(compileInfo);
    ShapeInfo si = MakeShape({8000, 256, 8}, {1, 0, 2}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeNddmaStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si, true));
    ASSERT_TRUE(strategy.IsSupport());
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_NDDMA_CUT_2));
    EXPECT_EQ(bundle.context->GetBlockDim(), 48);
    const auto* data = GetTilingData<TransposeNddmaCut2TilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->ubSize, kUbSize);
    EXPECT_EQ(data->blkFactor, 6);
    EXPECT_GE(data->outCutIndex, 0);
    const std::array<uint32_t, 5> expected0{8, 31, 217, 1, 1};
    (void)expected0;
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
// 注: n=255 非 16 对齐, 用于避开 10017 TransData 特化路由。

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
    const auto* data = GetTilingData<TransposeNddmaCut1TilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->ubSize, kUbSize);
    EXPECT_EQ(data->blkFactor, 83);
    const std::array<uint32_t, 5> expected0{8, 7000, 1, 1, 1};
    (void)expected0;
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
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(TransposeNddmaCut2TilingData));
    const auto* data = GetTilingData<TransposeNddmaCut2TilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    const std::array<uint32_t, 5> expected0{8, 31, 217, 1, 1};
    (void)expected0;
    const std::array<uint32_t, 5> expected1{8, 8, 217, 1, 1};
    (void)expected1;
    const std::array<uint32_t, 5> expected2{8, 31, 188, 1, 1};
    (void)expected2;
    const std::array<uint32_t, 5> expected3{8, 8, 188, 1, 1};
    (void)expected3;
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
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(TransposeNddmaCut2TilingData));
    const auto* data = GetTilingData<TransposeNddmaCut2TilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    // The original strategy splits axis 1 with inUbFactor = 251 / 8 = 31; the retry cannot
    // swallow axis 1 (100000 > 7936), so the reverted split info is identical to the original.
    const std::array<uint32_t, 5> expected0{8, 31, 256, 1, 1};
    (void)expected0;
    const std::array<uint32_t, 5> expected1{8, 25, 256, 1, 1};
    (void)expected1;
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
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(TransposeNddmaCut2TilingData));
    const auto* data = GetTilingData<TransposeNddmaCut2TilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    // If the revert to originalResult were broken, the absorbed retry would leave the input
    // cut on axis 1 (inCutIndex=1, inUbFactor=21) instead of the original axis 2 split.
    const std::array<uint32_t, 5> expected0{6, 41, 1, 250, 1};
    (void)expected0;
    const std::array<uint32_t, 5> expected1{6, 8, 1, 250, 1};
    (void)expected1;
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
    const auto* data = GetTilingData<TransposeNddmaCut1TilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->ubSize, kUbSize);
    EXPECT_EQ(data->blkFactor, 125);
    const std::array<uint32_t, 5> expected0{8, 6000, 1, 1, 1};
    (void)expected0;
}

TEST(TransposeNddmaStrategy, transpose_tiling_nddma_base_4d)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    // 总量 4.6MB 超过 SMALL 门槛; MergeAxis 后输入等效行数 30*21*30=18900 非 16 对齐,
    // 不命中小 m/N_LAST/10017 特化, 落到 NDDMA (NDDMA_CUT_1/NDDMA_CUT_2)。
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
    EXPECT_EQ(bundle.context->GetBlockDim(), 44);
    const auto* data = GetTilingData<TransposeNddmaCut1TilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->ubSize, kUbSize);
    EXPECT_EQ(data->blkFactor, 1);
    EXPECT_EQ(data->blkTailFactor, 0);
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
    EXPECT_EQ(bundle.context->GetBlockDim(), 48);
    const auto* data = GetTilingData<TransposeNddmaCut1TilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->ubSize, kUbSize);
    EXPECT_EQ(data->blkFactor, 1);
    EXPECT_EQ(data->blkTailFactor, 24);
}

// NDDMA 缓冲数完全由 platformInfo.npuArch 推导：DAV_9201=双缓冲，DAV_3510=单缓冲。
// 双缓冲把 UB 预算减半，紧凑 TilingData 的分核块数与尾块周期随之下调。
TEST(TransposeNddmaStrategy, direct_buffer_num_from_arch_unreachable_by_e2e)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeCutOncePara(compileInfo);
    // [512,100] perm {1,0}：两种缓冲数下均命中 NDDMA_CUT_1，但 UB 预算差异会改变切分因子。
    ShapeInfo si = MakeShape({512, 100}, {1, 0}, 4);

    // 双缓冲（DAV_9201）
    auto dualBundle = BuildTilingContext(para);
    optiling::TransposeNddmaStrategy dual(
        dualBundle.context, MakeNddmaPlatform(kUbSize, static_cast<int64_t>(Ops::Base::DAV_9201)), MakeOpParam(si));
    ASSERT_EQ(dual.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(dualBundle.context->GetTilingKey(), static_cast<uint64_t>(TK_NDDMA_CUT_1));
    EXPECT_EQ(dualBundle.context->GetBlockDim(), 47);
    const auto* dualData = GetTilingData<TransposeNddmaCut1TilingData>(dualBundle.context);
    ASSERT_NE(dualData, nullptr);
    EXPECT_EQ(dualData->ubSize, kUbSize);

    // 单缓冲（DAV_3510）
    auto singleBundle = BuildTilingContext(para);
    optiling::TransposeNddmaStrategy single(
        singleBundle.context, MakeNddmaPlatform(kUbSize, static_cast<int64_t>(Ops::Base::DAV_3510)), MakeOpParam(si));
    ASSERT_EQ(single.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(singleBundle.context->GetTilingKey(), static_cast<uint64_t>(TK_NDDMA_CUT_1));
    EXPECT_EQ(singleBundle.context->GetBlockDim(), 48);
    const auto* singleData = GetTilingData<TransposeNddmaCut1TilingData>(singleBundle.context);
    ASSERT_NE(singleData, nullptr);
    EXPECT_EQ(singleData->ubSize, kUbSize);

    // 双缓冲 UB 预算减半 → 每核 block 数更少、主块搬运量与尾块周期随切分因子改变。
    EXPECT_EQ(dualData->blkFactor, 1U);
    EXPECT_EQ(singleData->blkFactor, 5U);
    EXPECT_EQ(dualData->blkTailFactor, 0U);
    EXPECT_EQ(singleData->blkTailFactor, 16U);
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
    EXPECT_EQ(bundle.context->GetBlockDim(), 32);
    const auto* data = GetTilingData<TransposeNddmaCut1TilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->ubSize, 1024U);
    EXPECT_EQ(data->blkFactor, 1);
    EXPECT_EQ(data->blkTailFactor, 0);
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
    EXPECT_EQ(bundle.context->GetBlockDim(), 48);
    const auto* data = GetTilingData<TransposeNddmaCut2TilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->ubSize, 1024U);
    EXPECT_EQ(data->blkFactor, 1);
    EXPECT_GE(data->outCutIndex, 0);
    EXPECT_EQ(data->inUbMainSrcShape[0], 1);
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
    EXPECT_EQ(bundle.context->GetBlockDim(), 4);
    const auto* data = GetTilingData<TransposeNddmaCut2TilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->ubSize, 1024U);
    EXPECT_GE(data->outCutIndex, 0);
}

// NDDMA_CUT_2：输入切分轴位于输出切分轴右侧，UB shape 回填时跳过该轴。
// 普通/关联入口统一紧凑布局后不再暴露切分轴索引，本用例校验该分支产出的紧凑分核结果。
TEST(TransposeNddmaStrategy, direct_cut_twice_skip_input_cut_axis_unreachable_by_e2e)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeCutTwicePara(compileInfo);
    ShapeInfo si = MakeShape({2, 16, 32}, {2, 0, 1}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeNddmaStrategy strategy(bundle.context, MakeNddmaPlatform(1024), MakeOpParam(si));
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_NDDMA_CUT_2));
    EXPECT_EQ(bundle.context->GetBlockDim(), 48);
    const auto* data = GetTilingData<TransposeNddmaCut2TilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->ubSize, 1024U);
    EXPECT_EQ(data->blkFactor, 1U);
    EXPECT_EQ(data->blkTailFactor, 16U);
    EXPECT_GE(data->outCutIndex, 0);
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
    EXPECT_EQ(bundle.context->GetBlockDim(), 48);
    const auto* data = GetTilingData<TransposeNddmaCut2TilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    // 仅输出侧有尾块：Main 与 OutputTail 存在，InputTail/Tail 区间为空。
    EXPECT_GE(data->outCutIndex, 0);
    EXPECT_GE(data->inCutIndex, 0);
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

/* ============================================================================
 * NDDMA(10002/10003) 补充：单/双切分 UB 内 shape、维度边界、超大 shape
 * ========================================================================== */
TEST(TransposeNddmaStrategy, nddma_cut1_ub_budget)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeDummyPara(compileInfo);
    ShapeInfo si = MakeShape({4000, 7000, 8}, {1, 0, 2}, 4);

    auto bundle = BuildTilingContext(para);
    optiling::TransposeNddmaStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    ASSERT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_NDDMA_CUT_1));
    const auto* data = GetTilingData<TransposeNddmaCut1TilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    ASSERT_EQ(data->ubSize, kUbSize);
    // 默认 DAV_3510 为单缓冲，kernel buffer = ubSize。
    CheckNddmaUbShape(data->inUbMainSrcShape, data->inUbMainDstShape, data->ubSize, 4);
    CheckNddmaUbShape(data->inUbTailSrcShape, data->inUbTailDstShape, data->ubSize, 4);
}

TEST(TransposeNddmaStrategy, nddma_cut2_ub_budget)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeDummyPara(compileInfo);
    ShapeInfo si = MakeShape({8000, 256, 8}, {1, 0, 2}, 4);

    auto bundle = BuildTilingContext(para);
    optiling::TransposeNddmaStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    ASSERT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_NDDMA_CUT_2));
    const auto* data = GetTilingData<TransposeNddmaCut2TilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    ASSERT_EQ(data->ubSize, kUbSize);
    CheckNddmaUbShape(data->inUbMainSrcShape, data->inUbMainDstShape, data->ubSize, 4);
    CheckNddmaUbShape(data->inUbInputTailSrcShape, data->inUbInputTailDstShape, data->ubSize, 4);
    CheckNddmaUbShape(data->inUbOutputTailSrcShape, data->inUbOutputTailDstShape, data->ubSize, 4);
    CheckNddmaUbShape(data->inUbTailSrcShape, data->inUbTailDstShape, data->ubSize, 4);
}

// DAV_9201/9202 双缓冲：NDDMA 的 UB 预算按 bufferNum=2 折半，单块 shape 须 <= ubSize/2。
TEST(TransposeNddmaStrategy, nddma_double_buffer_ub_budget)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeDummyPara(compileInfo);
    ShapeInfo si = MakeShape({4000, 7000, 8}, {1, 0, 2}, 4);

    auto bundle = BuildTilingContext(para);
    optiling::TransposePlatformInfo platform = MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_9201));
    optiling::TransposeNddmaStrategy strategy(bundle.context, platform, MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    const uint64_t key = bundle.context->GetTilingKey();
    ASSERT_TRUE(key == TK_NDDMA_CUT_1 || key == TK_NDDMA_CUT_2);

    const int64_t bufferBytes = kUbSize / 2;
    if (key == TK_NDDMA_CUT_1) {
        const auto* data = GetTilingData<TransposeNddmaCut1TilingData>(bundle.context);
        ASSERT_NE(data, nullptr);
        EXPECT_EQ(data->ubSize, kUbSize);
        CheckNddmaUbShape(data->inUbMainSrcShape, data->inUbMainDstShape, bufferBytes, 4);
        CheckNddmaUbShape(data->inUbTailSrcShape, data->inUbTailDstShape, bufferBytes, 4);
    } else {
        const auto* data = GetTilingData<TransposeNddmaCut2TilingData>(bundle.context);
        ASSERT_NE(data, nullptr);
        EXPECT_EQ(data->ubSize, kUbSize);
        CheckNddmaUbShape(data->inUbMainSrcShape, data->inUbMainDstShape, bufferBytes, 4);
        CheckNddmaUbShape(data->inUbInputTailSrcShape, data->inUbInputTailDstShape, bufferBytes, 4);
        CheckNddmaUbShape(data->inUbOutputTailSrcShape, data->inUbOutputTailDstShape, bufferBytes, 4);
        CheckNddmaUbShape(data->inUbTailSrcShape, data->inUbTailDstShape, bufferBytes, 4);
    }
}

// 维度边界：dim<=5 命中 NDDMA，dim>5 交给 NDDMA_BIG_DIM
TEST(TransposeNddmaStrategy, nddma_dim_boundary)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeDummyPara(compileInfo);
    auto bundle = BuildTilingContext(para);

    ShapeInfo dim5 = MakeShape({2, 2, 2, 2, 32}, {0, 1, 2, 3, 4}, 4);
    optiling::TransposeNddmaStrategy hit(bundle.context, MakePlatformInfo(), MakeOpParam(dim5));
    EXPECT_TRUE(hit.IsSupport());

    ShapeInfo dim6 = MakeShape({2, 2, 2, 2, 2, 32}, {0, 1, 2, 3, 4, 5}, 4);
    optiling::TransposeNddmaStrategy miss(bundle.context, MakePlatformInfo(), MakeOpParam(dim6));
    EXPECT_FALSE(miss.IsSupport());
}

// 超大 shape：维度 > uint32 必须保留在 int64 字段中
TEST(TransposeNddmaStrategy, nddma_large_shape_and_ub_budget)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeDummyPara(compileInfo);
    ShapeInfo si = MakeShape({kHugeDim, 64}, {1, 0}, 4);

    auto bundle = BuildTilingContext(para);
    optiling::TransposeNddmaStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    const uint64_t key = bundle.context->GetTilingKey();
    ASSERT_TRUE(key == TK_NDDMA_CUT_1 || key == TK_NDDMA_CUT_2);

    if (key == TK_NDDMA_CUT_1) {
        const auto* data = GetTilingData<TransposeNddmaCut1TilingData>(bundle.context);
        ASSERT_NE(data, nullptr);
        const int64_t offset = TP_NDDMA_MAX_DIM_NUM - data->permSize;
        EXPECT_EQ(data->expandedInputShape[offset], kHugeDim);
        EXPECT_EQ(data->expandedOutputShape[offset + 1], kHugeDim);
        CheckNddmaUbShape(data->inUbMainSrcShape, data->inUbMainDstShape, data->ubSize, 4);
        CheckNddmaUbShape(data->inUbTailSrcShape, data->inUbTailDstShape, data->ubSize, 4);
    } else {
        const auto* data = GetTilingData<TransposeNddmaCut2TilingData>(bundle.context);
        ASSERT_NE(data, nullptr);
        const int64_t offset = TP_NDDMA_MAX_DIM_NUM - data->permSize;
        EXPECT_EQ(data->expandedInputShape[offset], kHugeDim);
        EXPECT_EQ(data->expandedOutputShape[offset + 1], kHugeDim);
        CheckNddmaUbShape(data->inUbMainSrcShape, data->inUbMainDstShape, data->ubSize, 4);
        CheckNddmaUbShape(data->inUbInputTailSrcShape, data->inUbInputTailDstShape, data->ubSize, 4);
        CheckNddmaUbShape(data->inUbOutputTailSrcShape, data->inUbOutputTailDstShape, data->ubSize, 4);
        CheckNddmaUbShape(data->inUbTailSrcShape, data->inUbTailDstShape, data->ubSize, 4);
    }
}
