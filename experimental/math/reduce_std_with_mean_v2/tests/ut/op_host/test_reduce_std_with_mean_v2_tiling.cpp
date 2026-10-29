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
 * \file test_reduce_std_with_mean_v2_tiling.cpp
 * \brief Unit tests for ReduceStdWithMeanV2 host-side tiling function
 *
 * Coverage targets:
 *   - Normal tiling: fp16/fp32/bf16, various shapes
 *   - Explicit empty dim: reduce all dimensions, including the Partial path
 *   - Edge cases: small reduce dim, large reduce dim, multi-dim input
 *   - Error cases: unsupported dtype
 *
 * TilingKey mapping (REDUCE_STD_WITH_MEAN_V2_* in reduce_std_with_mean_v2_tiling_key.h):
 *   TilingKey 0: REDUCE_STD_WITH_MEAN_V2_FP16 (fp16)
 *   TilingKey 1: REDUCE_STD_WITH_MEAN_V2_FP32 (fp32)
 *   TilingKey 2: REDUCE_STD_WITH_MEAN_V2_BF16 (bf16)
 *
 * Note: single input (self) + two outputs (out, meanOut).
 * Baseline cases omit attrs: tiling falls back to reduceLength = last dim.
 * Empty-dim regressions pass explicit attrs and inspect the reduction domain.
 */

#include <gtest/gtest.h>
#include <vector>

#include "reduce_std_with_mean_v2_tiling.h"
#include "../../../op_kernel/reduce_std_with_mean_v2_tiling_data.h"
#include "../../../op_kernel/reduce_std_with_mean_v2_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

// ==========================================================================
// Test Fixture
// ==========================================================================

class ReduceStdWithMeanV2TilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "ReduceStdWithMeanV2TilingTest SetUp" << endl; }

    static void TearDownTestCase() { cout << "ReduceStdWithMeanV2TilingTest TearDown" << endl; }
};

// ==========================================================================
// Explicit dim attributes: empty means reduce all, not just the last dim
// ==========================================================================

TEST_F(ReduceStdWithMeanV2TilingTest, dim_empty_reduce_all)
{
    ReduceStdWithMeanV2CompileInfo compileInfo;
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16}) {
        for (bool keepdim : {false, true}) {
            SCOPED_TRACE(testing::Message() << "dtype=" << dtype << ", keepdim=" << keepdim);
            const gert::StorageShape outputShape = keepdim ? gert::StorageShape{{1, 1, 1}, {1, 1, 1}} :
                                                             gert::StorageShape{{}, {}};
            gert::TilingContextPara ctx("ReduceStdWithMeanV2", {{{{2, 3, 4}, {2, 3, 4}}, dtype, ge::FORMAT_ND}},
                                        {{outputShape, dtype, ge::FORMAT_ND}, {outputShape, dtype, ge::FORMAT_ND}},
                                        {{"dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({})},
                                         {"unbiased", Ops::Math::AnyValue::CreateFrom<bool>(false)},
                                         {"keepdim", Ops::Math::AnyValue::CreateFrom<bool>(keepdim)},
                                         {"invert", Ops::Math::AnyValue::CreateFrom<bool>(false)},
                                         {"eps", Ops::Math::AnyValue::CreateFrom<float>(0.0f)},
                                         {"correction", Ops::Math::AnyValue::CreateFrom<int64_t>(0)}},
                                        &compileInfo);
            TilingInfo tilingInfo;
            ASSERT_TRUE(ExecuteTiling(ctx, tilingInfo));
            ASSERT_GE(tilingInfo.tilingDataSize, sizeof(ReduceStdWithMeanV2TilingData));
            auto* tilingData = reinterpret_cast<const ReduceStdWithMeanV2TilingData*>(tilingInfo.tilingData.get());
            EXPECT_EQ(tilingData->reduceLength, 24);
            EXPECT_EQ(tilingData->totalNonReduce, 1);
            EXPECT_EQ(tilingInfo.blockNum, 1);
        }
    }
}

TEST_F(ReduceStdWithMeanV2TilingTest, explicit_last_dim)
{
    ReduceStdWithMeanV2CompileInfo compileInfo;
    gert::TilingContextPara ctx("ReduceStdWithMeanV2", {{{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                {{{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                {{"dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({-1})},
                                 {"unbiased", Ops::Math::AnyValue::CreateFrom<bool>(false)},
                                 {"keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)},
                                 {"invert", Ops::Math::AnyValue::CreateFrom<bool>(false)},
                                 {"eps", Ops::Math::AnyValue::CreateFrom<float>(0.0f)},
                                 {"correction", Ops::Math::AnyValue::CreateFrom<int64_t>(0)}},
                                &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(ctx, tilingInfo));
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(ReduceStdWithMeanV2TilingData));
    auto* tilingData = reinterpret_cast<const ReduceStdWithMeanV2TilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->reduceLength, 4);
    EXPECT_EQ(tilingData->totalNonReduce, 2);
}

TEST_F(ReduceStdWithMeanV2TilingTest, partial_dim_empty_reduce_all)
{
    ReduceStdWithMeanV2PartialCompileInfo compileInfo;
    gert::TilingContextPara ctx("ReduceStdWithMeanV2Partial",
                                {{{{2, 3, 2048}, {2, 3, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                {{{{1, 6, 3}, {1, 6, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                {{"dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({})},
                                 {"partialK", Ops::Math::AnyValue::CreateFrom<int64_t>(6)},
                                 {"keepdim", Ops::Math::AnyValue::CreateFrom<bool>(false)},
                                 {"invert", Ops::Math::AnyValue::CreateFrom<bool>(false)},
                                 {"eps", Ops::Math::AnyValue::CreateFrom<float>(0.0f)},
                                 {"correction", Ops::Math::AnyValue::CreateFrom<int64_t>(0)}},
                                &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(ctx, tilingInfo));
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(ReduceStdWithMeanV2TilingData));
    auto* tilingData = reinterpret_cast<const ReduceStdWithMeanV2TilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->reduceLength, 12288);
    EXPECT_EQ(tilingData->totalNonReduce, 1);
    EXPECT_EQ(tilingData->partialK, 6);
    EXPECT_EQ(tilingData->partialSeg, 2048);
    EXPECT_EQ(tilingInfo.blockNum, 6);
}

// ==========================================================================
// Normal Cases — fp32 (TilingKey=1, REDUCE_STD_WITH_MEAN_V2_FP32)
// ==========================================================================

/// Basic: fp32, 2D [64, 128]
TEST_F(ReduceStdWithMeanV2TilingTest, fp32_2d)
{
    ReduceStdWithMeanV2CompileInfo compileInfo;
    gert::TilingContextPara ctx("ReduceStdWithMeanV2",
                                {
                                    {{{64, 128}, {128, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                },
                                {
                                    {{{64, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    {{{64, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                },
                                &compileInfo);
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, 1, std::vector<size_t>{0});
}

/// fp32, 3D [8, 16, 256]
TEST_F(ReduceStdWithMeanV2TilingTest, fp32_3d)
{
    ReduceStdWithMeanV2CompileInfo compileInfo;
    gert::TilingContextPara ctx("ReduceStdWithMeanV2",
                                {
                                    {{{8, 16, 256}, {4096, 256, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                },
                                {
                                    {{{8, 16, 1}, {16, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    {{{8, 16, 1}, {16, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                },
                                &compileInfo);
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, 1, std::vector<size_t>{0});
}

/// fp32, single-element reduce dim [32, 1], reduce=1
TEST_F(ReduceStdWithMeanV2TilingTest, fp32_reduce_dim_1)
{
    ReduceStdWithMeanV2CompileInfo compileInfo;
    gert::TilingContextPara ctx("ReduceStdWithMeanV2",
                                {
                                    {{{32, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                },
                                {
                                    {{{32, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    {{{32, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                },
                                &compileInfo);
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, 1, std::vector<size_t>{0});
}

/// fp32, non-reduce=1 [1, 256]
TEST_F(ReduceStdWithMeanV2TilingTest, fp32_nonreduce_1)
{
    ReduceStdWithMeanV2CompileInfo compileInfo;
    gert::TilingContextPara ctx("ReduceStdWithMeanV2",
                                {
                                    {{{1, 256}, {256, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                },
                                {
                                    {{{1, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                    {{{1, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                },
                                &compileInfo);
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, 1, std::vector<size_t>{0});
}

// ==========================================================================
// Normal Cases — fp16 (TilingKey=0, REDUCE_STD_WITH_MEAN_V2_FP16)
// ==========================================================================

/// Basic: fp16, 2D [64, 128], reduce last dim (128)
TEST_F(ReduceStdWithMeanV2TilingTest, fp16_2d_default)
{
    ReduceStdWithMeanV2CompileInfo compileInfo;
    gert::TilingContextPara ctx("ReduceStdWithMeanV2",
                                {
                                    {{{64, 128}, {128, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                },
                                {
                                    {{{64, 1}, {1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                    {{{64, 1}, {1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                },
                                &compileInfo);
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, 0, std::vector<size_t>{0});
}

/// fp16, large 2D [1024, 2048], reduce last dim
TEST_F(ReduceStdWithMeanV2TilingTest, fp16_large_2d)
{
    ReduceStdWithMeanV2CompileInfo compileInfo;
    gert::TilingContextPara ctx("ReduceStdWithMeanV2",
                                {
                                    {{{1024, 2048}, {2048, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                },
                                {
                                    {{{1024, 1}, {1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                    {{{1024, 1}, {1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                },
                                &compileInfo);
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, 0, std::vector<size_t>{0});
}

/// fp16, multi-dim 3D [4, 8, 64], reduce last dim (64), non-reduce=32
TEST_F(ReduceStdWithMeanV2TilingTest, fp16_3d_multi_dim)
{
    ReduceStdWithMeanV2CompileInfo compileInfo;
    gert::TilingContextPara ctx("ReduceStdWithMeanV2",
                                {
                                    {{{4, 8, 64}, {512, 64, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                },
                                {
                                    {{{4, 8, 1}, {8, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                    {{{4, 8, 1}, {8, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                },
                                &compileInfo);
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, 0, std::vector<size_t>{0});
}

/// fp16, 1D input [256] — rank-1, non-reduce=1, reduce=256
TEST_F(ReduceStdWithMeanV2TilingTest, fp16_1d)
{
    ReduceStdWithMeanV2CompileInfo compileInfo;
    gert::TilingContextPara ctx("ReduceStdWithMeanV2",
                                {
                                    {{{256}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                },
                                {
                                    {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                    {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                },
                                &compileInfo);
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, 0, std::vector<size_t>{0});
}

// ==========================================================================
// Normal Cases — bf16 (TilingKey=2, REDUCE_STD_WITH_MEAN_V2_BF16)
// ==========================================================================

/// Basic: bf16, 2D [64, 128]
TEST_F(ReduceStdWithMeanV2TilingTest, bf16_2d)
{
    ReduceStdWithMeanV2CompileInfo compileInfo;
    gert::TilingContextPara ctx("ReduceStdWithMeanV2",
                                {
                                    {{{64, 128}, {128, 1}}, ge::DT_BF16, ge::FORMAT_ND},
                                },
                                {
                                    {{{64, 1}, {1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
                                    {{{64, 1}, {1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
                                },
                                &compileInfo);
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, 2, std::vector<size_t>{0});
}

// ==========================================================================
// Error Cases
// ==========================================================================

/// Unsupported dtype (INT32) — should return GRAPH_FAILED
TEST_F(ReduceStdWithMeanV2TilingTest, unsupported_dtype_int32)
{
    ReduceStdWithMeanV2CompileInfo compileInfo;
    gert::TilingContextPara ctx("ReduceStdWithMeanV2",
                                {
                                    {{{64, 128}, {128, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                },
                                {
                                    {{{64, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                    {{{64, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                },
                                &compileInfo);
    ExecuteTestCase(ctx, ge::GRAPH_FAILED);
}

/// Unsupported dtype (INT8)
TEST_F(ReduceStdWithMeanV2TilingTest, unsupported_dtype_int8)
{
    ReduceStdWithMeanV2CompileInfo compileInfo;
    gert::TilingContextPara ctx("ReduceStdWithMeanV2",
                                {
                                    {{{64, 128}, {128, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                },
                                {
                                    {{{64, 1}, {1, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                    {{{64, 1}, {1, 1}}, ge::DT_INT8, ge::FORMAT_ND},
                                },
                                &compileInfo);
    ExecuteTestCase(ctx, ge::GRAPH_FAILED);
}

// ==========================================================================
// Boundary Cases
// ==========================================================================

/// Empty tensor: totalNum=0 → should succeed with blockDim=1
TEST_F(ReduceStdWithMeanV2TilingTest, empty_tensor)
{
    ReduceStdWithMeanV2CompileInfo compileInfo;
    gert::TilingContextPara ctx("ReduceStdWithMeanV2",
                                {
                                    {{{0, 128}, {128, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                },
                                {
                                    {{{0, 1}, {1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                    {{{0, 1}, {1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                },
                                &compileInfo);
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, 0, std::vector<size_t>{0});
}

/// All-zero shape element (totalNum=0 case)
TEST_F(ReduceStdWithMeanV2TilingTest, zero_dim_element)
{
    ReduceStdWithMeanV2CompileInfo compileInfo;
    gert::TilingContextPara ctx("ReduceStdWithMeanV2",
                                {
                                    {{{4, 0}, {0, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                },
                                {
                                    {{{4, 1}, {1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                    {{{4, 1}, {1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                },
                                &compileInfo);
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, 0, std::vector<size_t>{0});
}
