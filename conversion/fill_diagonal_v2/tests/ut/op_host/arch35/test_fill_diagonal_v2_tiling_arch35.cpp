/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../../op_host/fill_diagonal_v2_tiling.h"

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace gert;
using namespace optiling;

class FillDiagonalV2TilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "FillDiagonalV2 TilingArch35 SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "FillDiagonalV2 TilingArch35 TearDown" << std::endl; }
};

TEST_F(FillDiagonalV2TilingArch35, test_fill_diagonal_v2_float)
{
    optiling::FillDiagonalV2CompileInfo compileInfo = {48, 196608, 16777216};

    gert::TilingContextPara tilingContextPara(
        "FillDiagonalV2",
        {{{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("wrap", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "256 17 256 196608 32 32 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FillDiagonalV2TilingArch35, test_fill_diagonal_v2_float_cube)
{
    optiling::FillDiagonalV2CompileInfo compileInfo = {48, 196608, 16777216};

    gert::TilingContextPara tilingContextPara(
        "FillDiagonalV2",
        {{{{16, 16, 16}, {16, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{16, 16, 16}, {16, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{16, 16, 16}, {16, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("wrap", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "4096 273 4096 196608 85 101 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FillDiagonalV2TilingArch35, test_fill_diagonal_v2_float16)
{
    optiling::FillDiagonalV2CompileInfo compileInfo = {48, 196608, 16777216};

    gert::TilingContextPara tilingContextPara(
        "FillDiagonalV2",
        {{{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("wrap", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "256 17 256 196608 64 64 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FillDiagonalV2TilingArch35, test_fill_diagonal_v2_small_shape_less_than_cache_line)
{
    optiling::FillDiagonalV2CompileInfo compileInfo = {48, 196608, 16777216};

    gert::TilingContextPara tilingContextPara(
        "FillDiagonalV2",
        {{{{3, 3, 3, 3}, {3, 3, 3, 3}}, ge::DT_INT16, ge::FORMAT_ND}, {{{1}, {1}}, ge::DT_INT16, ge::FORMAT_ND}},
        {
            {{{3, 3, 3, 3}, {3, 3, 3, 3}}, ge::DT_INT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("wrap", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0U, "81 40 81 196608 81 81 ", {16777216});
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.blockNum, 1U);
}

TEST_F(FillDiagonalV2TilingArch35, test_fill_diagonal_v2_small_shape_two_cache_lines)
{
    optiling::FillDiagonalV2CompileInfo compileInfo = {48, 196608, 16777216};

    gert::TilingContextPara tilingContextPara(
        "FillDiagonalV2", {{{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}},
        {
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("wrap", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0U, "64 9 64 196608 32 32 ", {16777216});
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.blockNum, 2U);
}

TEST_F(FillDiagonalV2TilingArch35, test_fill_diagonal_v2_rank1_abnormal)
{
    optiling::FillDiagonalV2CompileInfo compileInfo = {48, 196608, 16777216};

    gert::TilingContextPara tilingContextPara(
        "FillDiagonalV2", {{{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("wrap", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FillDiagonalV2TilingArch35, test_fill_diagonal_v2_dims_not_equal_abnormal)
{
    optiling::FillDiagonalV2CompileInfo compileInfo = {48, 196608, 16777216};

    gert::TilingContextPara tilingContextPara(
        "FillDiagonalV2",
        {{{{2, 2, 3}, {2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{2, 2, 3}, {2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {
            {{{2, 2, 3}, {2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("wrap", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FillDiagonalV2TilingArch35, test_fill_diagonal_v2_invalid_dtype_uint16_rejected)
{
    optiling::FillDiagonalV2CompileInfo compileInfo = {48, 196608, 16777216};

    gert::TilingContextPara tilingContextPara(
        "FillDiagonalV2",
        {{{{16, 16}, {16, 16}}, ge::DT_UINT16, ge::FORMAT_ND}, {{{16, 16}, {16, 16}}, ge::DT_UINT16, ge::FORMAT_ND}},
        {
            {{{16, 16}, {16, 16}}, ge::DT_UINT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("wrap", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FillDiagonalV2TilingArch35, test_fill_diagonal_v2_dtype_mismatch_rejected)
{
    optiling::FillDiagonalV2CompileInfo compileInfo = {48, 196608, 16777216};

    gert::TilingContextPara tilingContextPara(
        "FillDiagonalV2",
        {{{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("wrap", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
