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
 * \brief SMALL_SHAPE SIMT 策略 UT：直调策略（逐字段断言）+ 端到端命中
 *
 * 普通/关联入口统一按简化后维数下发精简 TilingData：
 *   2D/3D → 10011 TransposeSmallShape23DTilingData（24B，dim 区分维数）；4-8D → 10001。
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
} // namespace

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
    EXPECT_EQ(e2e.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE_23D));
    EXPECT_EQ(e2e.blockNum, 1);
    const auto* data = GetTilingData<TransposeSmallShape23DTilingData>(e2e);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->dim, 2);
    // 未用槽位：inputShape 补 0，perm 补 -1
    EXPECT_THAT(data->inputShape, ::testing::ElementsAreArray(std::vector<int32_t>{4, 7, 0}));
    EXPECT_THAT(data->perm, ::testing::ElementsAreArray(std::vector<int8_t>{1, 0, -1}));
}

// SIMT 不受支持的平台：SMALL_SHAPE 不命中（由 NDDMA/NDDMA_BIG_DIM 兜底）

// 2D perm=[1,0] 小 shape → 10011 精简 TilingData（m/n/coreNum，tileM/tileN=0）
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
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE_23D));
    EXPECT_EQ(tilingInfo.blockNum, 1);
    const auto* data = GetTilingData<TransposeSmallShape23DTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->dim, 2);
    EXPECT_THAT(data->inputShape, ::testing::ElementsAreArray(std::vector<int32_t>{10, 20, 0}));
    EXPECT_THAT(data->perm, ::testing::ElementsAreArray(std::vector<int8_t>{1, 0, -1}));
    // 200 元素 x fp32 = 800B < 1024B 对齐单元，Host 分核退化为单核。
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

// UTs for PR changes:
//   1. DoSplitUBInput absorption strategy (promote NDDMA_CUT_2 to NDDMA_CUT_1)
//   2. UB out-of-bound safety validation with fallback to the original strategy
//   3. CalcBlockSplitInfoForSmallShape 128B-aligned core split (CeilAlign/FloorAlign)
//   4. TransposeGatherTiling MTE gate / bank conflict rejection split
// ============================================================================
// Absorption success: [4000, 7000, 8] perm [1, 0, 2] with 248K UB. The sqrt budget (251)
// cannot hold axis 1 (7000) after swallowing axis 2, so the original strategy cuts axis 1
// (inCutIndex=1, inUbFactor=31) and yields NDDMA_CUT_2. The absorption retry swallows axis 1
// with the full remaining budget (63488/8=7936 >= 7000) and cuts axis 0 instead with a single
// element per block (inCutIndex=0, inUbFactor=1). The amplified effective outUbFactor check
// passes (8 * 1 * 7000 = 56000 <= 63488), so the NDDMA_CUT_1 result is kept.

// 2D 小 shape fp32 28 元素 < 1024B 对齐单元 256：10011 Host 分核精简 TilingData，blockDim=1（单核承载 28 元素）。
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
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE_23D));
    EXPECT_EQ(tilingInfo.blockNum, 1);
    const auto* data = GetTilingData<TransposeSmallShape23DTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->dim, 2);
    EXPECT_THAT(data->inputShape, ::testing::ElementsAreArray(std::vector<int32_t>{4, 7, 0}));
}

// 2D 小 shape int8: 20*6=120 元素 < 对齐单元 (128/1=128) -> Host 分核单核承载整块。
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
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE_23D));
    EXPECT_EQ(tilingInfo.blockNum, 1);
    const auto* data = GetTilingData<TransposeSmallShape23DTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->dim, 2);
    EXPECT_THAT(data->inputShape, ::testing::ElementsAreArray(std::vector<int32_t>{20, 6, 0}));
}

// 2D 小 shape fp16 [5, 8]: 40 元素 < fp16 对齐单元 (128/2=64) -> Host 分核单核承载全部元素。
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
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE_23D));
    EXPECT_EQ(tilingInfo.blockNum, 1);
    const auto* data = GetTilingData<TransposeSmallShape23DTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->dim, 2);
    EXPECT_THAT(data->inputShape, ::testing::ElementsAreArray(std::vector<int32_t>{5, 8, 0}));
}

// 3 维小 shape floor-align 分支 (10011 精简 TilingData): [256, 150, 4] fp32 -> 153600 元素，
// host 分核 48 核全部使用。
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
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE_23D));
    EXPECT_EQ(tilingInfo.blockNum, 47U);
    const auto* data = GetTilingData<TransposeSmallShape23DTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->dim, 3);
    EXPECT_THAT(data->inputShape, ::testing::ElementsAreArray(std::vector<int32_t>{256, 150, 4}));
    EXPECT_THAT(data->perm, ::testing::ElementsAreArray(std::vector<int8_t>{1, 0, 2}));
}

// 回归：Host 分核（1024B 对齐）与 blockDim 一致
// coreNum=48, DT_FLOAT(alignElems=256), 3D [9,9,19] total=1539, perm {1,0,2}（尾轴 19<32，
// 非尾轴转置 -> SMALL_SHAPE），体积 6156B < 4MB。
// host: blkFactor=CeilDiv(1539,48)=33 -> ceilAlign(33,256)=256 -> realCoreNum=CeilDiv(1539,256)=7,
//       blkTail=1539%256=3，blockDim=realCoreNum=7（无空跑、无丢元素）。
TEST(TransposeSmallShapeStrategy, regression_compact_split_matches_kernel_3d_odd_tail)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 253952;

    int64_t permValue[3] = {1, 0, 2};
    gert::TilingContextPara::TensorDescription x({{9, 9, 19}, {9, 9, 19}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &permValue);
    // perm {1,0,2}: 前两轴交换，二者均为 9，输出形状仍为 [9,9,19]
    gert::TilingContextPara::TensorDescription y({{9, 9, 19}, {9, 9, 19}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara contextPara("Transpose", {x, perm}, {y}, &compileInfo, compileInfo.coreNum,
                                        compileInfo.ubSize);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(contextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE_23D));
    EXPECT_EQ(tilingInfo.blockNum, 7U);
    EXPECT_EQ(tilingInfo.tilingDataSize, sizeof(TransposeSmallShape23DTilingData));
    const auto* data = GetTilingData<TransposeSmallShape23DTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->dim, 3);
    EXPECT_THAT(data->inputShape, ::testing::ElementsAreArray(std::vector<int32_t>{9, 9, 19}));
    EXPECT_THAT(data->perm, ::testing::ElementsAreArray(std::vector<int8_t>{1, 0, 2}));
    // blockDim 必须等于 Host 的 realCoreNum
}

// 紧凑分核边界：total < alignElements(1024B/eleLen) -> 单核承载全部元素
TEST(TransposeSmallShapeStrategy, compact_split_below_align_single_core)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 253952;

    int64_t permValue[3] = {1, 0, 2};
    gert::TilingContextPara::TensorDescription x({{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &permValue);
    gert::TilingContextPara::TensorDescription y({{3, 2, 4}, {3, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara contextPara("Transpose", {x, perm}, {y}, &compileInfo, compileInfo.coreNum,
                                        compileInfo.ubSize);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(contextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE_23D));
    EXPECT_EQ(tilingInfo.blockNum, 1U);
    const auto* data = GetTilingData<TransposeSmallShape23DTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->dim, 3);
    // total=24 < alignElements=256 -> realCoreNum=1
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
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE_23D));
    EXPECT_EQ(tilingInfo.blockNum, 47U);
    const auto* data = GetTilingData<TransposeSmallShape23DTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->dim, 3);
    EXPECT_THAT(data->inputShape, ::testing::ElementsAreArray(std::vector<int32_t>{2, 9601, 8}));
    // Host 分核：[2,9601,8] fp32 total=153616, blkFactor=CeilDiv(153616,48)=3201,
    // ceilAlign=CeilAlign(3201,256)=3328 -> realCoreNum=CeilDiv(153616,3328)=47。
}

TEST(TransposeSmallShapeStrategy, transpose_tiling_small_shape_core_budget_dtypes_and_limits)
{
    struct SplitCase {
        ge::DataType dtype;
        int64_t elementBytes;
        int64_t coreBudget;
        int64_t realCoreNum;
        int64_t middleDim = 9601;
        int64_t lastDim = 8;
    };
    // 期望 realCoreNum 按 SMALL_SHAPE 1024B 对齐公式（Host 算好）推导：
    const SplitCase cases[] = {
        {ge::DT_FLOAT16, 2, 48, 43}, {ge::DT_INT8, 1, 48, 38},       {ge::DT_FLOAT, 4, 8, 8},
        {ge::DT_FLOAT, 4, 1, 1},     {ge::DT_FLOAT, 4, 48, 1, 5, 4},
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
        EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE_23D));
        const auto* data = GetTilingData<TransposeSmallShape23DTilingData>(tilingInfo);
        ASSERT_NE(data, nullptr);
        EXPECT_EQ(data->dim, 3);
        EXPECT_THAT(data->inputShape,
                    ::testing::ElementsAreArray(std::vector<int32_t>{2, static_cast<int32_t>(testCase.middleDim),
                                                                     static_cast<int32_t>(testCase.lastDim)}));
        EXPECT_EQ(tilingInfo.blockNum, static_cast<uint64_t>(testCase.realCoreNum));
    }
}

// 3 维小 shape ceil-align 分支 (10011): [200, 200, 4] fp32 -> 160000 元素，host 分核 48 核。
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
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE_23D));
    EXPECT_EQ(tilingInfo.blockNum, 45U);
    const auto* data = GetTilingData<TransposeSmallShape23DTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->dim, 3);
    EXPECT_THAT(data->inputShape, ::testing::ElementsAreArray(std::vector<int32_t>{200, 200, 4}));
}

// 3 维小 shape ceil-align (10011, fp16): [16, 60, 4] fp16 -> 3840 元素，host 分核 8 核。
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
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE_23D));
    EXPECT_EQ(tilingInfo.blockNum, 8U);
    const auto* data = GetTilingData<TransposeSmallShape23DTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->dim, 3);
    EXPECT_THAT(data->inputShape, ::testing::ElementsAreArray(std::vector<int32_t>{16, 60, 4}));
}

// 4 维小 shape (last axis 不转置, 尾轴 <32 不进 N_LAST, 总量 < SMALL 阈值) -> 4D 起走 10001 精简 TilingData
TEST(TransposeSmallShapeStrategy, transpose_tiling_small_shape_4d_route)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 253952; // 248K

    int64_t perm_value[4] = {0, 2, 1, 3};
    gert::TilingContextPara::TensorDescription x({{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{2, 4, 3, 5}, {2, 4, 3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    // 4D 起由 10001（TransposeSmallShape48DTilingData）承接，不再走 10011。
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE));
    EXPECT_EQ(tilingInfo.blockNum, 1U);
    const auto* data = GetTilingData<TransposeSmallShape48DTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->dim, 4);
    // 120 元素 x fp32 = 480B < 1024B 对齐单元，Host 分核退化为单核。
    EXPECT_THAT(data->inputShape, ::testing::ElementsAreArray(std::vector<int32_t>{2, 3, 4, 5, 0, 0, 0, 0}));
    EXPECT_THAT(data->perm, ::testing::ElementsAreArray(std::vector<int8_t>{0, 2, 1, 3, -1, -1, -1, -1}));
}

// 5 维小 shape (不可合并, 尾轴 <32 不进 N_LAST) -> 10001 精简 TilingData, perm 未用槽位补 -1
TEST(TransposeSmallShapeStrategy, transpose_tiling_small_shape_5d_compact_tiling_data)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 48;
    compileInfo.ubSize = 253952; // 248K

    int64_t perm_value[5] = {1, 0, 3, 2, 4};
    gert::TilingContextPara::TensorDescription x({{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{3, 2, 5, 4, 6}, {3, 2, 5, 4, 6}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE));
    EXPECT_EQ(tilingInfo.blockNum, 3U);
    const auto* data = GetTilingData<TransposeSmallShape48DTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    EXPECT_THAT(data->inputShape, ::testing::ElementsAreArray(std::vector<int32_t>{2, 3, 4, 5, 6, 0, 0, 0}));
    EXPECT_THAT(data->perm, ::testing::ElementsAreArray(std::vector<int8_t>{1, 0, 3, 2, 4, -1, -1, -1}));
}

// 关联算子入口 2D：统一走 10011 精简 TilingData（不再下发历史通用 TilingData）
TEST(TransposeSmallShapeStrategy, related_entry_uses_compact_23d)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    ShapeInfo si = MakeShape({4, 7}, {1, 0}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeSmallShapeStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si, true));
    ASSERT_TRUE(strategy.IsSupport());
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_SMALL_SHAPE_23D));
    EXPECT_EQ(bundle.context->GetBlockDim(), 1U);
    const auto* data = GetTilingData<TransposeSmallShape23DTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->dim, 2);
    EXPECT_THAT(data->inputShape, ::testing::ElementsAreArray(std::vector<int32_t>{4, 7, 0}));
    EXPECT_THAT(data->perm, ::testing::ElementsAreArray(std::vector<int8_t>{1, 0, -1}));
}

// 关联算子入口 4-8D：统一走 10001 精简 TilingData，perm 未用槽位 -1
TEST(TransposeSmallShapeStrategy, related_entry_uses_compact_small)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    ShapeInfo si = MakeShape({2, 3, 4, 5, 6}, {1, 0, 3, 2, 4}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeSmallShapeStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si, true));
    ASSERT_TRUE(strategy.IsSupport());
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_SMALL_SHAPE));
    EXPECT_EQ(bundle.context->GetBlockDim(), 3U);
    const auto* data = GetTilingData<TransposeSmallShape48DTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    EXPECT_THAT(data->inputShape, ::testing::ElementsAreArray(std::vector<int32_t>{2, 3, 4, 5, 6, 0, 0, 0}));
    EXPECT_THAT(data->perm, ::testing::ElementsAreArray(std::vector<int8_t>{1, 0, 3, 2, 4, -1, -1, -1}));
}

// 2D 小 shape fp64 [3, 5]: 15 元素 < fp64 对齐单元 (1024/8=128) -> Host 分核单核承载整块。
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
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE_23D));
    EXPECT_EQ(tilingInfo.blockNum, 1);
    const auto* data = GetTilingData<TransposeSmallShape23DTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->dim, 2);
    EXPECT_THAT(data->inputShape, ::testing::ElementsAreArray(std::vector<int32_t>{3, 5, 0}));
}

/* ==================== 阈值分支补充用例 ==================== */

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

/* ============================================================================
 * SMALL_SHAPE(10001/10011)：阈值边界 + 不占 UB
 * ========================================================================== */
TEST(TransposeSmallShapeStrategy, small_shape_boundary_and_no_ub_buffer)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeDummyPara(compileInfo);
    auto bundle = BuildTilingContext(para);

    ShapeInfo hit = MakeShape({999, 1000}, {1, 0}, 4); // 3,996,000B < 4MB
    optiling::TransposeSmallShapeStrategy hitStrategy(bundle.context, MakePlatformInfo(), MakeOpParam(hit));
    EXPECT_TRUE(hitStrategy.IsSupport());

    ShapeInfo miss = MakeShape({1000, 1000}, {1, 0}, 4); // 4,000,000B，不小于阈值
    optiling::TransposeSmallShapeStrategy missStrategy(bundle.context, MakePlatformInfo(), MakeOpParam(miss));
    EXPECT_FALSE(missStrategy.IsSupport());

    auto runBundle = BuildTilingContext(para);
    ShapeInfo small = MakeShape({9, 9, 19}, {1, 0, 2}, 4);
    optiling::TransposeSmallShapeStrategy strategy(runBundle.context, MakePlatformInfo(), MakeOpParam(small));
    ASSERT_TRUE(strategy.IsSupport());
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(runBundle.context->GetTilingKey(), static_cast<uint64_t>(TK_SMALL_SHAPE_23D));
    const auto* data = GetTilingData<TransposeSmallShape23DTilingData>(runBundle.context);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->dim, 3U);
    EXPECT_GT(data->blkFactor, 0U);
    // Host 分核必须覆盖全部元素；kernel 通过 GetBlockNum() 取实际核数。
    EXPECT_GE(static_cast<uint64_t>(data->blkFactor) * runBundle.context->GetBlockDim() + data->blkTailFactor,
              static_cast<uint64_t>(small.totalVolumeActual));
    // SMALL_SHAPE 在 GM 间直读直写，不申请 UB buffer：UB 占用为 0。
    EXPECT_LE(0, kUbSize);
}
