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
 * \file test_transpose_tiling_strategy_small_shape.cpp
 * \brief SMALL_SHAPE(10001, SIMT) 策略 UT：直调策略（逐字段断言）+ 端到端命中
 */

#include "test_transpose_tiling_strategy_common.h"
#include "../../../../op_host/arch35/transpose_tiling_strategy_small_shape.h"

using namespace transpose_strategy_ut;

namespace {
gert::TilingContextPara MakePara(optiling::TransposeCompilerInfo& compileInfo)
{
    static int64_t permValue[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{4, 7}, {4, 7}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &permValue);
    gert::TilingContextPara::TensorDescription out({{7, 4}, {7, 4}}, ge::DT_FLOAT, ge::FORMAT_ND);
    return gert::TilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
}

// ============================================================================
// UTs for PR changes:
//   1. DoSplitUBInput absorption strategy (promote NDDMA_CUT_2 to NDDMA_CUT_1)
//   2. UB out-of-bound safety validation with fallback to the original strategy
//   3. CalcBlockSplitInfoForSmallShape 128B-aligned core split (CeilAlign/FloorAlign)
//   4. TransposeGatherTiling MTE gate / bank conflict rejection split
// ============================================================================
inline void ExpectSmallShapeCoverage(const TilingInfo& tilingInfo, const TransposeOpTilingData* opTiling,
                                     int64_t totalElements, int64_t coreBudget, int64_t elementBytes)
{
    ASSERT_NE(opTiling, nullptr);
    ASSERT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE));
    const int64_t realCoreNum = opTiling->realCoreNum;
    const int64_t blkFactor = opTiling->blkFactor;
    const int64_t blkTailFactor = opTiling->blkTailFactor;
    ASSERT_GT(realCoreNum, 0);
    EXPECT_LE(realCoreNum, coreBudget);
    EXPECT_EQ(tilingInfo.blockNum, static_cast<uint64_t>(realCoreNum));
    ASSERT_GT(blkFactor, 0);
    ASSERT_GE(blkTailFactor, 0);
    ASSERT_LT(blkTailFactor, blkFactor);
    if (realCoreNum > 1) {
        EXPECT_EQ(blkFactor * elementBytes % 128, 0);
    }
    const int64_t lastBlockElements = blkTailFactor == 0 ? blkFactor : blkTailFactor;
    EXPECT_EQ((realCoreNum - 1) * blkFactor + lastBlockElements, totalElements);
}
} // namespace

// SIMT 不受支持的平台：SMALL_SHAPE 不命中（由 NDDMA/NDDMA_BIG_DIM 兜底）
TEST(TransposeSmallShapeStrategy, is_support_false_when_simt_unsupported)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    ShapeInfo si = MakeShape({4, 7}, {1, 0}, 4);

    auto bundle = BuildTilingContext(para);
    auto platform = MakePlatformInfo();
    platform.isSupportSimt = false;
    optiling::TransposeSmallShapeStrategy strategy(bundle.context, platform, MakeOpParam(si));
    EXPECT_FALSE(strategy.IsSupport());
}

// Gather rejected, NDDMA fallback: [1000000, 3] perm [1, 0] (12MB, above the small-shape
// threshold). The gather candidate block holds only axis1(3) x a small cut of axis0, whose
// totalSizeInUb stays below the MTE gate (16*1024), so CalcUbAxesInfo rejects gather and the
// main NDDMA flow serves the transpose with NDDMA_CUT_1 on axis 0.

/* ==================== 阈值分支补充用例 ==================== */

// DAV_5102 平台：小 shape 阈值降为 1MB（尾轴转置场景）
TEST(TransposeSmallShapeStrategy, direct_threshold_dav5102)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    ShapeInfo si = MakeShape({480, 512}, {1, 0}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeSmallShapeStrategy strategy(
        bundle.context, MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5102)), MakeOpParam(si));
    // 960KB 数据量：不满足 4MB 默认阈值，但满足 DAV_5102 的 1MB 阈值
    EXPECT_TRUE(strategy.IsSupport());
}

TEST(TransposeSmallShapeStrategy, e2e_hits_this_strategy)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    TilingInfo e2e;
    ASSERT_TRUE(ExecuteTiling(para, e2e));
    EXPECT_EQ(e2e.tilingKey, TK_SMALL_SHAPE);
}

TEST(TransposeSmallShapeStrategy, transpose_tiling_small_shape_2d)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{10, 20}, {10, 20}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{20, 10}, {20, 10}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE));
}

// 4D N_LAST transpose (last axis not transposed, large shape, last axis >= 32), expect key 10004

TEST(TransposeSmallShapeStrategy, transpose_tiling_small_3d_shape_absorb)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 253952; // 248K

    int64_t perm_value[3] = {1, 0, 2};
    gert::TilingContextPara::TensorDescription x({{1000, 256, 300}, {1000, 256, 300}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{256, 1000, 300}, {256, 1000, 300}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// Absorption success: [4000, 7000, 8] perm [1, 0, 2] with 248K UB. The sqrt budget (251)
// cannot hold axis 1 (7000) after swallowing axis 2, so the original strategy cuts axis 1
// (inCutIndex=1, inUbFactor=31) and yields NDDMA_CUT_2. The absorption retry swallows axis 1
// with the full remaining budget (63488/8=7936 >= 7000) and cuts axis 0 instead with a single
// element per block (inCutIndex=0, inUbFactor=1). The amplified effective outUbFactor check
// passes (8 * 1 * 7000 = 56000 <= 63488), so the NDDMA_CUT_1 result is kept.

TEST(TransposeSmallShapeStrategy, transpose_tiling_small_shape_fp32_below_align_single_core)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 253952; // 248K

    int64_t perm_value[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{4, 7}, {4, 7}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{7, 4}, {7, 4}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE));
    const auto* data = GetTilingData<TransposeOpTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->realCoreNum, 1);
    EXPECT_EQ(data->blkFactor, 28);
    EXPECT_EQ(data->blkTailFactor, 0);
    ASSERT_NO_FATAL_FAILURE(ExpectSmallShapeCoverage(tilingInfo, data, 4 * 7, compileInfo.coreNum, 4));
}

// SMALL_SHAPE with int8: 20*6=120 elements < alignment unit (128/1=128) -> single core,
// blkFactor = 120 (the whole tensor in one 128B-undersized chunk).

TEST(TransposeSmallShapeStrategy, transpose_tiling_small_shape_int8_below_align_single_core)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 253952; // 248K

    int64_t perm_value[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{20, 6}, {20, 6}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{6, 20}, {6, 20}}, ge::DT_INT8, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE));
    const auto* data = GetTilingData<TransposeOpTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->realCoreNum, 1);
    EXPECT_EQ(data->blkFactor, 120);
    EXPECT_EQ(data->blkTailFactor, 0);
}

// SMALL_SHAPE with fp16 [5, 8] (shape[0] <= 5 keeps the TRANSTO5HD_10 path off): 40 elements is
// below the fp16 alignment unit (128/2=64), so a single core carries all 40 elements.

TEST(TransposeSmallShapeStrategy, transpose_tiling_small_shape_fp16_below_align_single_core)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 253952; // 248K

    int64_t perm_value[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{5, 8}, {5, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{8, 5}, {8, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE));
    const auto* data = GetTilingData<TransposeOpTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->realCoreNum, 1);
    EXPECT_EQ(data->blkFactor, 40);
    EXPECT_EQ(data->blkTailFactor, 0);
}

// SMALL_SHAPE floor-align branch: [256, 150, 4] fp32 -> 153600 elements, blkFactor=3200 is
// already 32-element (128B) aligned. total - floorAlign*(coreNum-1) == floorAlign, so all 48
// cores are used with blkFactor=3200 and an empty tail.

TEST(TransposeSmallShapeStrategy, transpose_tiling_small_shape_floor_align_full_cores)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 253952; // 248K

    int64_t perm_value[3] = {1, 0, 2};
    gert::TilingContextPara::TensorDescription x({{256, 150, 4}, {256, 150, 4}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{150, 256, 4}, {150, 256, 4}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE));
    const auto* data = GetTilingData<TransposeOpTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->realCoreNum, 48);
    EXPECT_EQ(data->blkFactor, 3200);
    EXPECT_EQ(data->blkTailFactor, 0);
    ASSERT_NO_FATAL_FAILURE(ExpectSmallShapeCoverage(tilingInfo, data, 256 * 150 * 4, compileInfo.coreNum, 4));
}

TEST(TransposeSmallShapeStrategy, transpose_tiling_small_shape_core_budget_with_tail)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 253952;

    int64_t permValue[] = {1, 0, 2};
    gert::TilingContextPara::TensorDescription x({{2, 9601, 8}, {2, 9601, 8}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &permValue);
    gert::TilingContextPara::TensorDescription y({{9601, 2, 8}, {9601, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara contextPara("Transpose", {x, perm}, {y}, &compileInfo, compileInfo.coreNum,
                                        compileInfo.ubSize);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(contextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE));
    const auto* data = GetTilingData<TransposeOpTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    ASSERT_NO_FATAL_FAILURE(ExpectSmallShapeCoverage(tilingInfo, data, 2 * 9601 * 8, compileInfo.coreNum, 4));
    EXPECT_EQ(tilingInfo.blockNum, 48);
    EXPECT_EQ(data->realCoreNum, 48);
    EXPECT_EQ(data->blkFactor, 3232);
    EXPECT_EQ(data->blkTailFactor, 1712);
}

TEST(TransposeSmallShapeStrategy, transpose_tiling_small_shape_core_budget_dtypes_and_limits)
{
    struct SplitCase {
        ge::DataType dtype;
        int64_t elementBytes;
        int64_t coreBudget;
        int64_t realCoreNum;
        int64_t blkFactor;
        int64_t blkTailFactor;
        int64_t middleDim = 9601;
        int64_t lastDim = 8;
    };
    const SplitCase cases[] = {
        {ge::DT_FLOAT16, 2, 48, 48, 3264, 208}, {ge::DT_INT8, 1, 48, 47, 3328, 528},
        {ge::DT_FLOAT, 4, 8, 8, 19232, 18992},  {ge::DT_FLOAT, 4, 1, 1, 153632, 153616},
        {ge::DT_FLOAT, 4, 48, 1, 40, 0, 5, 4},
    };
    for (const auto& testCase : cases) {
        SCOPED_TRACE(::testing::Message() << "dtype=" << testCase.dtype << ", coreBudget=" << testCase.coreBudget);
        optiling::TransposeCompilerInfo compileInfo;
        compileInfo.coreNum = testCase.coreBudget;
        compileInfo.ubSize = 253952;
        int64_t permValue[] = {1, 0, 2};
        gert::TilingContextPara::TensorDescription x(
            {{2, testCase.middleDim, testCase.lastDim}, {2, testCase.middleDim, testCase.lastDim}}, testCase.dtype,
            ge::FORMAT_ND);
        gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &permValue);
        gert::TilingContextPara::TensorDescription y(
            {{testCase.middleDim, 2, testCase.lastDim}, {testCase.middleDim, 2, testCase.lastDim}}, testCase.dtype,
            ge::FORMAT_ND);
        gert::TilingContextPara contextPara("Transpose", {x, perm}, {y}, &compileInfo, compileInfo.coreNum,
                                            compileInfo.ubSize);
        TilingInfo tilingInfo;
        ASSERT_TRUE(ExecuteTiling(contextPara, tilingInfo));
        EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE));
        const auto* data = GetTilingData<TransposeOpTilingData>(tilingInfo);
        ASSERT_NE(data, nullptr);
        ASSERT_NO_FATAL_FAILURE(ExpectSmallShapeCoverage(tilingInfo, data, 2 * testCase.middleDim * testCase.lastDim,
                                                         testCase.coreBudget, testCase.elementBytes));
        EXPECT_EQ(tilingInfo.blockNum, testCase.realCoreNum);
        EXPECT_EQ(data->blkFactor, testCase.blkFactor);
        EXPECT_EQ(data->blkTailFactor, testCase.blkTailFactor);
    }
}

// SMALL_SHAPE ceil-align branch: [200, 200, 4] fp32 -> 160000 elements, raw blkFactor=3333.
// floorAlign=3328 cannot cover the tail (160000 - 3328*47 = 3584 > 3328), so the ceil-aligned
// factor CeilAlign(3333, 32)=3360 is used: realCoreNum = CeilDiv(160000, 3360) = 48,
// tail = 160000 % 3360 = 2080.

TEST(TransposeSmallShapeStrategy, transpose_tiling_small_shape_ceil_align_reduced_cores)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 253952; // 248K

    int64_t perm_value[3] = {1, 0, 2};
    gert::TilingContextPara::TensorDescription x({{200, 200, 4}, {200, 200, 4}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{200, 200, 4}, {200, 200, 4}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE));
    const auto* data = GetTilingData<TransposeOpTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->realCoreNum, 48);
    EXPECT_EQ(data->blkFactor, 3360);
    EXPECT_EQ(data->blkTailFactor, 2080);
}

// SMALL_SHAPE ceil-align with fp16 alignment unit 64: [16, 60, 4] fp16 -> 3840 elements,
// blkFactor=80 -> floorAlign=64 leaves a 832-element tail beyond one floor chunk, so the
// ceil-aligned 128 is used: realCoreNum = CeilDiv(3840, 128) = 30 cores, empty tail.

TEST(TransposeSmallShapeStrategy, transpose_tiling_small_shape_fp16_ceil_align)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 253952; // 248K

    int64_t perm_value[3] = {1, 0, 2};
    gert::TilingContextPara::TensorDescription x({{16, 60, 4}, {16, 60, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{60, 16, 4}, {60, 16, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE));
    const auto* data = GetTilingData<TransposeOpTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->realCoreNum, 30);
    EXPECT_EQ(data->blkFactor, 128);
    EXPECT_EQ(data->blkTailFactor, 0);
}

// SMALL_SHAPE with fp64 [3, 5]: 15 elements is below the fp64 alignment unit (128/8=16),
// so the whole tensor is carried by one core (realCoreNum=1, blkFactor=15).

TEST(TransposeSmallShapeStrategy, transpose_tiling_small_shape_fp64_below_align_single_core)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 253952; // 248K

    int64_t perm_value[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{3, 5}, {3, 5}}, ge::DT_DOUBLE, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{5, 3}, {5, 3}}, ge::DT_DOUBLE, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE));
    const auto* data = GetTilingData<TransposeOpTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->realCoreNum, 1);
    EXPECT_EQ(data->blkFactor, 15);
    EXPECT_EQ(data->blkTailFactor, 0);
}

// TilingData 容量不足：码流不写入，但 tilingKey/blockDim 仍按策略结果写回
TEST(TransposeSmallShapeStrategy, direct_tiling_data_insufficient)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    para.tilingDataSize_ = 8;
    ShapeInfo si = MakeShape({32, 32}, {1, 0}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeSmallShapeStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    // 码流容量不足：SetTilingData 返回 FAILURE，整体 tiling 失败
    EXPECT_EQ(strategy.Execute(), TilingStatus::FAILURE);
}
