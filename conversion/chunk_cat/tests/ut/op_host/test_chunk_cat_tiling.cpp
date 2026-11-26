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
 * \file test_chunk_cat_tiling.cpp
 * \brief
 */

#include <iostream>
#include <cstring>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/chunk_cat_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace gert;
using namespace optiling;

namespace {
void CheckOutputTiling(const gert::TilingContextPara& context, int64_t expectedRows, int64_t expectedCols)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    EXPECT_EQ(info.tilingKey, 0);
    EXPECT_EQ(info.workspaceSizes, std::vector<int64_t>{16777216});
    ASSERT_NE(info.tilingData.get(), nullptr);
    ASSERT_GE(info.tilingDataSize, sizeof(ChunkCatTilingData));
    ChunkCatTilingData data{};
    std::memcpy(&data, info.tilingData.get(), sizeof(data));
    EXPECT_EQ(data.outputRow, expectedRows);
    EXPECT_EQ(data.outputCol, expectedCols);
    EXPECT_GT(data.inUbSize, 0);
    EXPECT_GT(data.outUbSize, 0);
    EXPECT_GT(info.blockNum, 0U);
    EXPECT_GT(data.blockRowNum, 0);
    EXPECT_GT(data.blockColNum, 0);
    EXPECT_GT(data.ubRowFactor, 0);
    EXPECT_GT(data.ubColFactor, 0);
    if (expectedCols == 0) {
        // Empty tiling must encode one idle block, with zero column work and
        // nonzero divisors. Check the serialized data, not just graphStatus.
        EXPECT_EQ(info.blockNum, 1U);
        EXPECT_EQ(data.blockRowNum, 1);
        EXPECT_EQ(data.blockColNum, 1);
        EXPECT_EQ(data.ubRowFactor, 1);
        EXPECT_EQ(data.ubColFactor, 1);
        EXPECT_EQ(data.blockRowFactor, expectedRows);
        EXPECT_EQ(data.tailBlockRowFactor, expectedRows);
        EXPECT_EQ(data.blockColFactor, 0);
        EXPECT_EQ(data.tailBlockColFactor, 0);
    }
}
} // namespace

class ChunkCatTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ChunkCat Tiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ChunkCat Tiling TearDown" << std::endl; }
};

TEST_F(ChunkCatTilingTest, chunk_cat_float16_021_success_case)
{
    optiling::ChunkCatCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ChunkCat",
                                              {
                                                  {{{4, 8}, {4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{4, 16}, {4, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{4, 12}, {4, 12}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4, 36}, {4, 36}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"dim", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
                                                  {"num_chunks", Ops::Math::AnyValue::CreateFrom<int64_t>(4)},
                                              },
                                              {3}, {1}, &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Empty input (GetShapeSize==0) yields outputCol_==0; tiling must short-circuit without
// integer div-by-zero in DoUbSplit/DoBlockSplit (CWE-369 / case-20261030-101335-b9b04ce9b2).
TEST_F(ChunkCatTilingTest, chunk_cat_empty_tensor_output_col_zero_success)
{
    optiling::ChunkCatCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ChunkCat",
                                              {
                                                  {{{0, 3}, {0, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 0}, {2, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"dim", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
                                                  {"num_chunks", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                              },
                                              {1}, {1}, &compileInfo);
    CheckOutputTiling(tilingContextPara, 2, 0);
}

TEST_F(ChunkCatTilingTest, chunk_cat_empty_tensor_shape_zero_success)
{
    optiling::ChunkCatCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ChunkCat",
                                              {
                                                  {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 0}, {1, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"dim", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
                                                  {"num_chunks", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              {1}, {1}, &compileInfo);
    CheckOutputTiling(tilingContextPara, 1, 0);
}

// Mixed non-empty + empty: empty contributes 0 cols, outputCol_>0, normal split must succeed
// (kernel zero-col skip is covered separately; host must not reject this graph shape).
TEST_F(ChunkCatTilingTest, chunk_cat_mixed_empty_tensor_tiling_success)
{
    optiling::ChunkCatCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ChunkCat",
                                              {
                                                  {{{4, 4}, {4, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{0}, {0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 8}, {2, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"dim", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
                                                  {"num_chunks", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                              },
                                              {2}, {1}, &compileInfo);
    CheckOutputTiling(tilingContextPara, 2, 8);
}

TEST_F(ChunkCatTilingTest, chunk_cat_empty_tensor_fp16_to_fp32_success)
{
    optiling::ChunkCatCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("ChunkCat",
                                              {
                                                  {{{4, 0}, {4, 0}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 0}, {2, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"dim", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
                                                  {"num_chunks", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                              },
                                              {1}, {1}, &compileInfo);
    CheckOutputTiling(tilingContextPara, 2, 0);
}
