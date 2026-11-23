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
 * \file test_transpose_tiling_strategy_nddma_big_dim.cpp
 * \brief NDDMA_BIG_DIM(10005) 策略 UT：直调策略（逐字段断言）+ 端到端命中
 */

#include "test_transpose_tiling_strategy_common.h"
#include "../../../../op_host/arch35/transpose_tiling_strategy_nddma_big_dim.h"

using namespace transpose_strategy_ut;

namespace {
gert::TilingContextPara MakePara(optiling::TransposeCompilerInfo& compileInfo)
{
    static int64_t permValue[6] = {0, 3, 1, 4, 2, 5};
    gert::TilingContextPara::TensorDescription x({{100, 200, 10, 20, 30, 8}, {100, 200, 10, 20, 30, 8}}, ge::DT_FLOAT,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, &permValue);
    gert::TilingContextPara::TensorDescription out({{100, 20, 200, 30, 10, 8}, {100, 20, 200, 30, 10, 8}}, ge::DT_FLOAT,
                                                   ge::FORMAT_ND);
    return gert::TilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
}

// N_LAST with fewer cores available, exercises Rate-based split

/* ==================== 切轴/分核分支补充用例 ==================== */
optiling::TransposePlatformInfo MakeBigDimPlatform(int64_t ubSize)
{
    optiling::TransposePlatformInfo platform = MakePlatformInfo();
    platform.ubSize = ubSize;
    return platform;
}
} // namespace

TEST(TransposeNddmaBigDimStrategy, e2e_hits_this_strategy)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    TilingInfo e2e;
    ASSERT_TRUE(ExecuteTiling(para, e2e));
    EXPECT_EQ(e2e.tilingKey, TK_NDDMA_BIG_DIM);
    EXPECT_EQ(e2e.blockNum, 48);
    const auto* data = GetTilingData<TransposeNddmaBigDimTilingData>(e2e);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->permSize, 6);
    EXPECT_EQ(data->totalNddmaNum, 60000);
}

// Perm with int32 dtype (DT_INT32), expect tiling success

TEST(TransposeNddmaBigDimStrategy, transpose_tiling_7d_dim_over_5)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[7] = {6, 0, 1, 2, 3, 4, 5};
    gert::TilingContextPara::TensorDescription x({{2, 2, 2, 2, 2, 2, 200}, {2, 2, 2, 2, 2, 2, 200}}, ge::DT_FLOAT,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{7}, {7}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{200, 2, 2, 2, 2, 2, 2}, {200, 2, 2, 2, 2, 2, 2}}, ge::DT_FLOAT,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// NDDMA_BIG_DIM path: 6D where perm has no consecutive pairs (dim stays 6 after MergeAxis).
// perm={0,3,1,4,2,5} keeps all 6 axes. isLastAxisTranspose=false (perm[5]=5).
// reducedInShape[5]=8 < 32 skips N_LAST. dim=6 > 5 → NDDMA_BIG_DIM.

TEST(TransposeNddmaBigDimStrategy, transpose_tiling_big_dim_6d_no_consecutive_perm)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[6] = {0, 3, 1, 4, 2, 5};
    gert::TilingContextPara::TensorDescription x({{100, 200, 10, 20, 30, 8}, {100, 200, 10, 20, 30, 8}}, ge::DT_FLOAT,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{100, 20, 200, 30, 10, 8}, {100, 20, 200, 30, 10, 8}}, ge::DT_FLOAT,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_NDDMA_BIG_DIM));
}

// 2D perm [1, 0], fp16, m/n 均 16 对齐, 512KB (DAV_3510 门槛): 走 10017 TransData 特化模板

TEST(TransposeNddmaBigDimStrategy, transpose_tiling_big_dim_6d_gather_fallback)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[6] = {3, 0, 4, 1, 5, 2};
    gert::TilingContextPara::TensorDescription x({{50, 60, 10, 100, 80, 30}, {50, 60, 10, 100, 80, 30}}, ge::DT_FLOAT,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{100, 50, 80, 60, 30, 10}, {100, 50, 80, 60, 30, 10}}, ge::DT_FLOAT,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// Error path: x shape with zero value axis, expect failure

TEST(TransposeNddmaBigDimStrategy, transpose_tiling_big_dim_small_outer)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    // 6D with very large inner axis, small outer axes. perm no consecutive pairs.
    int64_t perm_value[6] = {0, 3, 1, 4, 2, 5};
    gert::TilingContextPara::TensorDescription x({{2, 2, 2, 2, 2, 250000}, {2, 2, 2, 2, 2, 250000}}, ge::DT_FLOAT,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{2, 2, 2, 2, 2, 250000}, {2, 2, 2, 2, 2, 250000}}, ge::DT_FLOAT,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

TEST(TransposeNddmaBigDimStrategy, transpose_tiling_big_dim_6d)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    // 6D transpose with perm = {5, 0, 1, 2, 3, 4}, total > threshold
    int64_t perm_value[6] = {5, 0, 1, 2, 3, 4};
    gert::TilingContextPara::TensorDescription x({{2, 3, 4, 5, 6, 100}, {2, 3, 4, 5, 6, 100}}, ge::DT_FLOAT,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{100, 2, 3, 4, 5, 6}, {100, 2, 3, 4, 5, 6}}, ge::DT_FLOAT,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    // 5 维以上压缩由 NDDMA_BIG_DIM 处理：直调策略并固定 DAV_3510，tilingKey 确定为 NDDMA_BIG_DIM(10005)
    ShapeInfo si = MakeShape({2, 3, 4, 5, 6, 100}, {5, 0, 1, 2, 3, 4}, 4);
    auto bundle = BuildTilingContext(tilingContextPara);
    optiling::TransposeNddmaBigDimStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_NDDMA_BIG_DIM));
    EXPECT_EQ(bundle.context->GetBlockDim(), 48);
    const auto* data = GetTilingData<TransposeNddmaBigDimTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->permSize, 6);
}

/* ==================== 公开 Execute 直调（e2e 无法构造的 UB 预算/小体积） ==================== */

// DAV_9201 双缓冲且输出切分轴为最后一个原始轴 → 策略内 fallback 单缓冲
TEST(TransposeNddmaBigDimStrategy, direct_double_buffer_fallback_unreachable_by_e2e)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    ShapeInfo si = MakeShape({1, 1, 1, 1, 1, 100000}, {0, 1, 2, 3, 4, 5}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposePlatformInfo platform = MakeBigDimPlatform(2048);
    platform.npuArch = Ops::Base::DAV_9201;
    optiling::TransposeNddmaBigDimStrategy strategy(bundle.context, platform, MakeOpParam(si));
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_NDDMA_BIG_DIM));
    EXPECT_EQ(bundle.context->GetBlockDim(), 48);
    const auto* data = GetTilingData<TransposeNddmaBigDimTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->permSize, 6);
    EXPECT_EQ(data->bufferNum, 1); // 双缓冲 fallback 单缓冲
}

// ubElement == outShape 且 5 维预算尚未耗尽：ternary 取 ubElement
TEST(TransposeNddmaBigDimStrategy, direct_ub_equal_shape_in_budget_unreachable_by_e2e)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    ShapeInfo si = MakeShape({1, 1, 1, 1, 1, 8}, {0, 1, 2, 3, 4, 5}, 1);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeNddmaBigDimStrategy strategy(bundle.context, MakeBigDimPlatform(16), MakeOpParam(si));
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_NDDMA_BIG_DIM));
    EXPECT_EQ(bundle.context->GetBlockDim(), 8);
    const auto* data = GetTilingData<TransposeNddmaBigDimTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->permSize, 6);
    EXPECT_EQ(data->ubBytes, 16);
}

// 5 维预算耗尽 + 块数不足：dimSize==0 分支与 bestI 回退/每核 1 block
TEST(TransposeNddmaBigDimStrategy, direct_dims_zero_and_low_core_unreachable_by_e2e)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    ShapeInfo si = MakeShape({8, 24, 8, 8, 8, 8}, {0, 1, 2, 3, 4, 5}, 1);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeNddmaBigDimStrategy strategy(bundle.context, MakeBigDimPlatform(100000), MakeOpParam(si));
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_NDDMA_BIG_DIM));
    EXPECT_EQ(bundle.context->GetBlockDim(), 48);
    const auto* data = GetTilingData<TransposeNddmaBigDimTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->permSize, 6);
    EXPECT_EQ(data->outCutIndex, 1);
}

// TilingData 容量不足：码流不写入，但 tilingKey/blockDim 仍按策略结果写回
TEST(TransposeNddmaBigDimStrategy, direct_tiling_data_insufficient)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    para.tilingDataSize_ = 8;
    ShapeInfo si = MakeShape({2, 3, 4, 5, 6, 100}, {5, 0, 1, 2, 3, 4}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeNddmaBigDimStrategy strategy(bundle.context, MakeBigDimPlatform(253952), MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    // 码流容量不足：SetTilingData 返回 FAILURE，整体 tiling 失败
    EXPECT_EQ(strategy.Execute(), TilingStatus::FAILURE);
}

// ubElement == outShape（等值）时 ternary 取 ubElement
TEST(TransposeNddmaBigDimStrategy, direct_ub_equal_shape_unreachable_by_e2e)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    ShapeInfo si = MakeShape({1, 1, 1, 1, 1, 8}, {0, 1, 2, 3, 4, 5}, 1);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeNddmaBigDimStrategy strategy(bundle.context, MakeBigDimPlatform(8), MakeOpParam(si));
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_NDDMA_BIG_DIM));
    EXPECT_EQ(bundle.context->GetBlockDim(), 8);
    const auto* data = GetTilingData<TransposeNddmaBigDimTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->permSize, 6);
    EXPECT_EQ(data->ubBytes, 8);
}

/* ============================================================================
 * NDDMA_BIG_DIM(10005)：超 uint32 压缩 + buffer/单块 UB 占用
 * ========================================================================== */
TEST(TransposeNddmaBigDimStrategy, nddma_big_dim_large_shape_and_ub_budget)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeDummyPara(compileInfo);
    const std::vector<int64_t> in = {1000, 1000, 1000, 100, 10, 10};
    const std::vector<int64_t> perm = {5, 0, 1, 2, 3, 4};
    ShapeInfo si = MakeShape(in, perm, 4);

    auto bundle = BuildTilingContext(para);
    optiling::TransposeNddmaBigDimStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_NDDMA_BIG_DIM));
    const auto* data = GetTilingData<TransposeNddmaBigDimTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);

    EXPECT_EQ(data->permSize, 6);
    EXPECT_EQ(data->inputShape[0], 1000);
    EXPECT_EQ(data->inputShape[2], 1000);
    ASSERT_GE(data->outCutIndex, 0);
    ASSERT_LT(data->outCutIndex, data->permSize);
    // totalNddmaNum 只统计 outCutIndex 及其右侧轴（切分轴取 outUbFactor）。
    const int64_t expectedNddmaNum = static_cast<int64_t>(data->outUbFactor) *
                                     Product(data->outputShape, data->outCutIndex + 1, data->permSize);
    EXPECT_EQ(data->totalNddmaNum, expectedNddmaNum);
    // 超大 shape 必须完整保留在 int64 字段中（首轴基数 = 右侧所有输入轴乘积）。
    EXPECT_GT(data->baseInShape[0], static_cast<int64_t>(kUint32Max));
    ASSERT_GT(data->bufferNum, 0U);
    ASSERT_GT(data->ubBytes, 0U);
    // kernel：bufferNum 个 ubBytes 的静态 Tensor。
    EXPECT_LE(static_cast<uint64_t>(data->ubBytes) * data->bufferNum, static_cast<uint64_t>(kUbSize));

    // 单块 UB 数据量 = outUbFactor × 输出切分轴右侧所有轴乘积。
    const int64_t innerElems = Product(data->outputShape, data->outCutIndex + 1, data->permSize);
    const int64_t blockElems = static_cast<int64_t>(data->outUbFactor) * innerElems;
    EXPECT_GT(blockElems, 0);
    EXPECT_LE(blockElems * 4, static_cast<int64_t>(data->ubBytes));
}

// 维度边界：dim>5 命中 NDDMA_BIG_DIM，dim<=5 交给 NDDMA
TEST(TransposeNddmaBigDimStrategy, nddma_big_dim_dim_boundary)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeDummyPara(compileInfo);
    auto bundle = BuildTilingContext(para);

    ShapeInfo dim6 = MakeShape({2, 2, 2, 2, 2, 32}, {0, 1, 2, 3, 4, 5}, 4);
    optiling::TransposeNddmaBigDimStrategy hit(bundle.context, MakePlatformInfo(), MakeOpParam(dim6));
    EXPECT_TRUE(hit.IsSupport());

    ShapeInfo dim5 = MakeShape({2, 2, 2, 2, 32}, {0, 1, 2, 3, 4}, 4);
    optiling::TransposeNddmaBigDimStrategy miss(bundle.context, MakePlatformInfo(), MakeOpParam(dim5));
    EXPECT_FALSE(miss.IsSupport());
}
