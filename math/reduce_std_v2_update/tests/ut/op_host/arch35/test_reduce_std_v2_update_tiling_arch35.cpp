/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_kernel/arch35/reduce_std_v2_update_tiling_data.h"

using namespace std;
using namespace ge;

struct ReduceStdV2UpdateCompileInfo {};

class ReduceStdV2UpdateTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ReduceStdV2UpdateTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ReduceStdV2UpdateTiling TearDown" << std::endl; }
};

TEST_F(ReduceStdV2UpdateTiling, tiling_fp32_2d)
{
    ReduceStdV2UpdateCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ReduceStdV2Update",
        {
            {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(1))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 4U, std::vector<size_t>{16777216});
}

TEST_F(ReduceStdV2UpdateTiling, tiling_fp16_all_reduce)
{
    ReduceStdV2UpdateCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ReduceStdV2Update",
        {
            {{{128, 64}, {128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{128, 64}, {128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 1}, {1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(0))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 4U, std::vector<size_t>{16777216});
}

TEST_F(ReduceStdV2UpdateTiling, tiling_bf16_negative_dim)
{
    ReduceStdV2UpdateCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ReduceStdV2Update",
        {
            {{{8, 128, 128}, {8, 128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8, 128, 128}, {8, 128, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{8, 128, 1}, {8, 128, 1}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({-1})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(1))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 4U, std::vector<size_t>{16777216});
}

TEST_F(ReduceStdV2UpdateTiling, tiling_empty_r_correction_0)
{
    ReduceStdV2UpdateCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ReduceStdV2Update",
        {
            {{{4, 0}, {4, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 0}, {4, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(0))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 2U, std::vector<size_t>{16777216});
}

TEST_F(ReduceStdV2UpdateTiling, tiling_empty_r_correction_1)
{
    ReduceStdV2UpdateCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ReduceStdV2Update",
        {
            {{{4, 0}, {4, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 0}, {4, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{4}, {4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(1))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 2U, std::vector<size_t>{16777216});
}

TEST_F(ReduceStdV2UpdateTiling, tiling_unsupported_dtype_rejected)
{
    ReduceStdV2UpdateCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ReduceStdV2Update",
        {{{{4, 8}, {4, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(1))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0U, std::vector<size_t>{});
}

TEST_F(ReduceStdV2UpdateTiling, tiling_mean_unsupported_dtype_rejected)
{
    ReduceStdV2UpdateCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ReduceStdV2Update",
        {{{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{4, 8}, {4, 8}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(1))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0U, std::vector<size_t>{});
}

TEST_F(ReduceStdV2UpdateTiling, tiling_unsupported_format_rejected)
{
    ReduceStdV2UpdateCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ReduceStdV2Update",
        {{{{1, 2, 3, 4}, {1, 2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
         {{{1, 2, 3, 4}, {1, 2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 2, 3}, {1, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({3})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(1))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0U, std::vector<size_t>{});
}

TEST_F(ReduceStdV2UpdateTiling, tiling_mean_unsupported_format_rejected)
{
    ReduceStdV2UpdateCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ReduceStdV2Update",
        {{{{1, 2, 3, 4}, {1, 2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 2, 3, 4}, {1, 2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {{{{1, 2, 3}, {1, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({3})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(1))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0U, std::vector<size_t>{});
}

TEST_F(ReduceStdV2UpdateTiling, tiling_output_unsupported_format_rejected)
{
    ReduceStdV2UpdateCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ReduceStdV2Update",
        {{{{1, 2, 3, 4}, {1, 2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 2, 3, 4}, {1, 2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 2, 3}, {1, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({3})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(1))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0U, std::vector<size_t>{});
}

TEST_F(ReduceStdV2UpdateTiling, tiling_dtype_mismatch_rejected)
{
    ReduceStdV2UpdateCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ReduceStdV2Update",
        {{{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{4, 8}, {4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(1))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0U, std::vector<size_t>{});
}

TEST_F(ReduceStdV2UpdateTiling, tiling_shape_mismatch_rejected)
{
    ReduceStdV2UpdateCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ReduceStdV2Update",
        {{{{2, 6}, {2, 6}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(1))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0U, std::vector<size_t>{});
}

TEST_F(ReduceStdV2UpdateTiling, tiling_dim_out_of_range_rejected)
{
    ReduceStdV2UpdateCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ReduceStdV2Update",
        {{{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(1))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0U, std::vector<size_t>{});
}

TEST_F(ReduceStdV2UpdateTiling, tiling_correction_beyond_one_accepted_with_correct_cof)
{
    ReduceStdV2UpdateCompileInfo compileInfo;
    // correction 取值域对齐基线/torch.var：任意非负整数可接受，分母为 N-correction
    gert::TilingContextPara correctionTwo(
        "ReduceStdV2Update",
        {{{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(2))},
        &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(correctionTwo, tilingInfo));
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(ReduceStdV2UpdateTilingData));
    const auto* tilingData = reinterpret_cast<const ReduceStdV2UpdateTilingData*>(tilingInfo.tilingData.get());
    // N=8, correction=2 → cof = 1/6
    EXPECT_NEAR(tilingData->cof, 1.0F / 6.0F, 1.0e-9F);

    // correction > N：分母 N-correction<0 → host 预计算 cof=NaN（kernel 仅乘法，无除零路径）
    gert::TilingContextPara correctionBeyondN(
        "ReduceStdV2Update",
        {{{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(9))},
        &compileInfo);
    TilingInfo beyondInfo;
    ASSERT_TRUE(ExecuteTiling(correctionBeyondN, beyondInfo));
    ASSERT_GE(beyondInfo.tilingDataSize, sizeof(ReduceStdV2UpdateTilingData));
    const auto* beyondTd = reinterpret_cast<const ReduceStdV2UpdateTilingData*>(beyondInfo.tilingData.get());
    EXPECT_TRUE(std::isnan(beyondTd->cof)) << "correction>N must yield NaN cof (N=8, correction=9)";

    // 负 correction 仍拒绝
    gert::TilingContextPara correctionNeg(
        "ReduceStdV2Update",
        {{{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(-1))},
        &compileInfo);
    ExecuteTestCase(correctionNeg, ge::GRAPH_FAILED, 0U, std::vector<size_t>{});
}

TEST_F(ReduceStdV2UpdateTiling, tiling_cache_tree_span_exceeds_16kb_rejected)
{
    ReduceStdV2UpdateCompileInfo compileInfo;
    // fp16 [100,256,120,65536] dim=[1,3]：laneN=240、rLoopCntTotal=209920 → 树 18 层 × 240 = 4320 > 4096
    gert::TilingContextPara tilingContextPara(
        "ReduceStdV2Update",
        {{{{100, 256, 120, 65536}, {100, 256, 120, 65536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{100, 256, 120, 65536}, {100, 256, 120, 65536}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{100, 120}, {100, 120}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 3})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(1))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0U, std::vector<size_t>{});
}

TEST_F(ReduceStdV2UpdateTiling, tiling_gm_stride_exceeds_uint32_rejected)
{
    ReduceStdV2UpdateCompileInfo compileInfo;
    // fp32 [65, 2^30+1] dim=[1]：atmp=2 → aUbFactor=2 → ubAxes[1]=aSplit 的 blockCount=2，
    // 被消费的块间步长 (2^30+1)×4B=4294967300 超出 uint32 上限 → 拒绝
    gert::TilingContextPara consumedStride(
        "ReduceStdV2Update",
        {{{{65, 1073741825}, {65, 1073741825}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{65, 1073741825}, {65, 1073741825}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{65}, {65}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(1))},
        &compileInfo);
    ExecuteTestCase(consumedStride, ge::GRAPH_FAILED, 0U, std::vector<size_t>{});

    // 静态检查报告原始触发例：fp16 [1,2,2,2^30+1,2] dim=[1,3] → ubAxes[1]=axis2（整轴 A(2)，blockCount=2），
    // 被消费步长 (2^30+1)×2×2B=2^32+4 超限 → 拒绝
    gert::TilingContextPara reportTrigger(
        "ReduceStdV2Update",
        {{{{1, 2, 2, 1073741825, 2}, {1, 2, 2, 1073741825, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1, 2, 2, 1073741825, 2}, {1, 2, 2, 1073741825, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{1, 2, 2}, {1, 2, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 3})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(1))},
        &compileInfo);
    ExecuteTestCase(reportTrigger, ge::GRAPH_FAILED, 0U, std::vector<size_t>{});
}

TEST_F(ReduceStdV2UpdateTiling, tiling_gm_stride_not_consumed_axis_accepted)
{
    ReduceStdV2UpdateCompileInfo compileInfo;
    // fp32 [2, 2^30+1] dim=[1]：axis0 跨度超 uint32，但 atmp=1 → aUbFactor=1 → ubAxes[1] 的
    // blockCount 恒为 1，srcStride 不被消费 → 须接受（收窄到"被消费步长"的语义锚点）
    gert::TilingContextPara tilingContextPara(
        "ReduceStdV2Update",
        {{{{2, 1073741825}, {2, 1073741825}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 1073741825}, {2, 1073741825}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(1))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo))
        << "stride of non-consumed axis (blockCount==1) must be accepted";
}

TEST_F(ReduceStdV2UpdateTiling, tiling_group_cache_tree_span_exceeds_16kb_rejected)
{
    ReduceStdV2UpdateCompileInfo compileInfo;
    // fp16 [3,65536,2,44000000,48] dim=[1,3]（tail-A group 模板）：laneN=192、rGroupCnt=32、
    // 每组 rCount=ceil(rLoopCntTotal/32)≈8.5e8 → 30 层 × 192 = 5760 > 4096 → 拒绝。
    // group Phase1 每组独立建缓存树（DoOneAChunkGroup/Phase1OutputToWorkspace），同样受 16KB 约束。
    // 注：ubAxes[1]=axis2（A(2) 整轴）步长 44000000×48×2B≈4.2GB ≤ uint32，不触发 CODE.18 拦截，
    // 确保本用例命中 group 缓存树校验而非步长校验。
    gert::TilingContextPara groupSpan(
        "ReduceStdV2Update",
        {{{{3, 65536, 2, 44000000, 48}, {3, 65536, 2, 44000000, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{3, 65536, 2, 44000000, 48}, {3, 65536, 2, 44000000, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{3, 2, 48}, {3, 2, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 3})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(1))},
        &compileInfo);
    ExecuteTestCase(groupSpan, ge::GRAPH_FAILED, 0U, std::vector<size_t>{});

    // 正常 group 用例（每组 rCount=20 → 5 层 × 192 = 960 ≤ 4096）须通过
    gert::TilingContextPara groupOk(
        "ReduceStdV2Update",
        {{{{3, 64, 2, 1000, 48}, {3, 64, 2, 1000, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{3, 64, 2, 1000, 48}, {3, 64, 2, 1000, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{3, 2, 48}, {3, 2, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 3})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(1))},
        &compileInfo);
    TilingInfo okInfo;
    ASSERT_TRUE(ExecuteTiling(groupOk, okInfo)) << "group case within cache span must be accepted";
}

TEST_F(ReduceStdV2UpdateTiling, tiling_bound_2p31_elem_padded_axis_not_rejected)
{
    ReduceStdV2UpdateCompileInfo compileInfo;
    // L0 bound 用例 031：bf16 (2^31,) dim=[-1] → pattern [A(1), R(2^31)]，补 1 轴跨度=2^32B 超 uint32
    // 但该轴 ubSize=1（blockCount=1，srcStride 不消费），须接受（修复前被 GM 跨度校验误拒）
    gert::TilingContextPara boundRank1(
        "ReduceStdV2Update",
        {{{{2147483648}, {2147483648}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2147483648}, {2147483648}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({-1})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(1))},
        &compileInfo);
    TilingInfo info1;
    ASSERT_TRUE(ExecuteTiling(boundRank1, info1)) << "bound 2^31-elem rank-1 must be accepted";

    // L0 bound 用例 033：fp32 (1,1,1,1,2^31) dim=[-2]（归约 size-1 轴）→ pattern [A(1), R(1), A(2^31)]，
    // 补 1 轴跨度=2^32B 须跳过；2^31 轴自身 stride=1 正常
    gert::TilingContextPara boundRank5(
        "ReduceStdV2Update",
        {{{{1, 1, 1, 1, 2147483648}, {1, 1, 1, 1, 2147483648}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 1, 1, 1, 2147483648}, {1, 1, 1, 1, 2147483648}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 1, 1, 2147483648}, {1, 1, 1, 2147483648}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({-2})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(1))},
        &compileInfo);
    TilingInfo info5;
    ASSERT_TRUE(ExecuteTiling(boundRank5, info5)) << "bound 2^31 rank-5 padded-axis must be accepted";

    // L0 bound 用例 035：fp32 (1,1,1,1,1,1,1,2^31) dim=[2,-5,-8,-3,-4] → 归约全部 size-1 轴，
    // 非归约 2^31 轴为 A → pattern [A(1), R(1), A(2^31)]，同样须接受
    gert::TilingContextPara boundRank8(
        "ReduceStdV2Update",
        {{{{1, 1, 1, 1, 1, 1, 1, 2147483648}, {1, 1, 1, 1, 1, 1, 1, 2147483648}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 1, 1, 1, 1, 1, 1, 2147483648}, {1, 1, 1, 1, 1, 1, 1, 2147483648}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 1, 2147483648}, {1, 1, 2147483648}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("dim",
                                         Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, -5, -8, -3, -4})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(0))},
        &compileInfo);
    TilingInfo info8;
    ASSERT_TRUE(ExecuteTiling(boundRank8, info8)) << "bound 2^31 rank-8 padded-axis must be accepted";
}

TEST_F(ReduceStdV2UpdateTiling, tiling_whole_axis_rub_factor_clamped_to_ub_budget)
{
    ReduceStdV2UpdateCompileInfo compileInfo;
    // fp16 [8192,150] dim=[1]（静态检查 SAFE.01 复算用例）：整轴一次装入时 CeilAlign(150,16)=160
    // 会超出 UB 预算 rIMax=159，须回退 floor 对齐为 144（SAFE.01 修复前 rUbFactorAlign 可为 160）
    gert::TilingContextPara tilingContextPara(
        "ReduceStdV2Update",
        {{{{8192, 150}, {8192, 150}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{8192, 150}, {8192, 150}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(1))},
        &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(ReduceStdV2UpdateTilingData));
    const auto* tilingData = reinterpret_cast<const ReduceStdV2UpdateTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->rUbFactorAlign, 144);
    EXPECT_EQ(tilingData->rUbFactor, 144);
}

TEST_F(ReduceStdV2UpdateTiling, tiling_rank_9_rejected)
{
    ReduceStdV2UpdateCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ReduceStdV2Update",
        {{{{1, 1, 1, 1, 1, 1, 1, 1, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 1, 1, 1, 1, 1, 1, 1, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({8})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(1))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0U, std::vector<size_t>{});
}

TEST_F(ReduceStdV2UpdateTiling, tiling_empty_a_uses_zero_operation_path)
{
    ReduceStdV2UpdateCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ReduceStdV2Update",
        {{{{0, 4}, {0, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{0, 4}, {0, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(0))},
        &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(ReduceStdV2UpdateTilingData));
    const auto* tilingData = reinterpret_cast<const ReduceStdV2UpdateTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->usedCoreNum, 0);
    EXPECT_EQ(tilingInfo.blockNum, 1U);
}

TEST_F(ReduceStdV2UpdateTiling, tiling_empty_r_uses_nan_and_fits_single_buffer_budget)
{
    constexpr uint64_t ubSize = 248U * 1024U;
    ReduceStdV2UpdateCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ReduceStdV2Update",
        {{{{32768, 0}, {32768, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{32768, 0}, {32768, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{32768}, {32768}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(0))},
        &compileInfo, 1U, ubSize);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(ReduceStdV2UpdateTilingData));
    const auto* tilingData = reinterpret_cast<const ReduceStdV2UpdateTilingData*>(tilingInfo.tilingData.get());
    EXPECT_TRUE(std::isnan(tilingData->cof));
    EXPECT_LE(tilingData->postReduceUbSize + tilingData->tmpBufUbSize, static_cast<int64_t>(ubSize));
}

TEST_F(ReduceStdV2UpdateTiling, tiling_normal_non_positive_denominator_uses_nan_cof)
{
    ReduceStdV2UpdateCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "ReduceStdV2Update",
        {{{{4, 1}, {4, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{4, 1}, {4, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
         gert::TilingContextPara::OpAttr("if_std", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(1))},
        &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(ReduceStdV2UpdateTilingData));
    const auto* tilingData = reinterpret_cast<const ReduceStdV2UpdateTilingData*>(tilingInfo.tilingData.get());
    EXPECT_TRUE(std::isnan(tilingData->cof));
}
