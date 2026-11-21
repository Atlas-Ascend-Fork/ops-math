/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "math/cholesky/op_host/cholesky_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "log/log.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"

using namespace ge;
using namespace std;

class CholeskyTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CholeskyTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "CholeskyTiling TearDown" << std::endl; }
};

TEST_F(CholeskyTiling, cholesky_test_tiling_case0)
{
    optiling::CholeskyCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "Cholesky",
        {
            {{{3, 6, 6}, {3, 6, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{3, 6, 6}, {3, 6, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("upper", Ops::Math::AnyValue::CreateFrom<bool>(true))}, &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "6 3 4294967302 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CholeskyTiling, cholesky_test_tiling_accepts_ncl_input)
{
    optiling::CholeskyCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "Cholesky",
        {
            {{{3, 6, 6}, {3, 6, 6}}, ge::DT_FLOAT, ge::FORMAT_NCL},
        },
        {
            {{{3, 6, 6}, {3, 6, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("upper", Ops::Math::AnyValue::CreateFrom<bool>(true))}, &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "6 3 4294967302 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CholeskyTiling, cholesky_test_tiling_single_large_matrix)
{
    optiling::CholeskyCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "Cholesky",
        {
            {{{8192, 8192}, {8192, 8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8192, 8192}, {8192, 8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("upper", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "8192 1 137438953728 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.blockNum, 32U);
}

TEST_F(CholeskyTiling, cholesky_test_tiling_rejects_rank_nine)
{
    optiling::CholeskyCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "Cholesky",
        {
            {{{1, 1, 1, 1, 1, 1, 1, 2, 2}, {1, 1, 1, 1, 1, 1, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 1, 1, 1, 1, 1, 2, 2}, {1, 1, 1, 1, 1, 1, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("upper", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(CholeskyTiling, cholesky_test_tiling_rejects_non_square_matrix)
{
    optiling::CholeskyCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "Cholesky",
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("upper", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(CholeskyTiling, cholesky_test_tiling_rejects_oversized_matrix)
{
    optiling::CholeskyCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "Cholesky",
        {
            {{{8193, 8193}, {8193, 8193}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8193, 8193}, {8193, 8193}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("upper", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(CholeskyTiling, cholesky_test_tiling_rejects_unsupported_dtype)
{
    optiling::CholeskyCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "Cholesky",
        {
            {{{2, 2}, {2, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 2}, {2, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("upper", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}
// 空且非方阵：方阵契约检查不豁免空张量，tiling 必须拒绝
TEST_F(CholeskyTiling, cholesky_test_tiling_rejects_empty_non_square)
{
    optiling::CholeskyCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "Cholesky",
        {
            {{{0, 2, 3}, {0, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{0, 2, 3}, {0, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("upper", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

namespace {
// 批量维零值/负值用例共用的入参构造：origin 与 storage shape 同形，upper 由入参给定
gert::TilingContextPara MakeCholeskyPara(void* compileInfo, const std::initializer_list<int64_t>& shape, bool upper)
{
    const gert::StorageShape storageShape(shape, shape);
    return gert::TilingContextPara(
        "Cholesky", {gert::TilingContextPara::TensorDescription(storageShape, ge::DT_FLOAT, ge::FORMAT_ND)},
        {gert::TilingContextPara::TensorDescription(storageShape, ge::DT_FLOAT, ge::FORMAT_ND)},
        {gert::TilingContextPara::OpAttr("upper", Ops::Math::AnyValue::CreateFrom<bool>(upper))}, compileInfo);
}
} // namespace

// 合法空 batch（首维为 0）：matrixNumCount==0 不得产出 SetBlockDim(0)，运行时要求非零 blockDim。
// 与 aclnn 第一段 IsEmpty() 早退契约一致：kernel 批量循环体一次都不进入，不下发任何计算
TEST_F(CholeskyTiling, cholesky_test_tiling_empty_batch_leading_dim)
{
    optiling::CholeskyCompileInfo compileInfo = {48};
    auto tilingContextPara = MakeCholeskyPara(&compileInfo, {0, 4, 4}, false);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "4 0 4294967300 ", {16777216});
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.blockNum, 1U);
}

// 合法空 batch 且最后一维为 96：arch22 无协同分支，同样不得产出 SetBlockDim(0)
TEST_F(CholeskyTiling, cholesky_test_tiling_empty_batch_at_coop_threshold)
{
    optiling::CholeskyCompileInfo compileInfo = {48};
    auto tilingContextPara = MakeCholeskyPara(&compileInfo, {0, 96, 96}, false);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "96 0 4294967392 ", {16777216});
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.blockNum, 1U);
}

// 合法空 batch（内层批量维为 0）：与首维为 0 同型
TEST_F(CholeskyTiling, cholesky_test_tiling_empty_batch_inner_dim)
{
    optiling::CholeskyCompileInfo compileInfo = {48};
    auto tilingContextPara = MakeCholeskyPara(&compileInfo, {2, 0, 4, 4}, false);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "4 0 4294967300 ", {16777216});
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.blockNum, 1U);
}

// 合法空 batch 且最后一维为 200：arch22 无协同分支，min(coreNum, 0)==0，修复后为 1
TEST_F(CholeskyTiling, cholesky_test_tiling_empty_batch_large_matrix)
{
    optiling::CholeskyCompileInfo compileInfo = {48};
    auto tilingContextPara = MakeCholeskyPara(&compileInfo, {0, 200, 200}, false);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "200 0 4294967496 ", {16777216});
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.blockNum, 1U);
}

// 最后一维为 0 的空张量（文档已声明的合法输入）：matrixNumCount==1，行为不得因本次修复改变
TEST_F(CholeskyTiling, cholesky_test_tiling_empty_last_dim_single_matrix)
{
    optiling::CholeskyCompileInfo compileInfo = {48};
    auto tilingContextPara = MakeCholeskyPara(&compileInfo, {0, 0}, false);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "0 1 4294967296 ", {16777216});
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.blockNum, 1U);
}

// 最后一维为 0 的空 batch（文档已声明的合法输入）：matrixNumCount==3，行为不得因本次修复改变
TEST_F(CholeskyTiling, cholesky_test_tiling_empty_last_dim_batch)
{
    optiling::CholeskyCompileInfo compileInfo = {48};
    auto tilingContextPara = MakeCholeskyPara(&compileInfo, {3, 0, 0}, false);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "0 3 4294967296 ", {16777216});
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.blockNum, 3U);
}

// 批量维为 -1：static_cast<uint64_t> 会回绕成 18446744073709551615，必须显式拒绝
TEST_F(CholeskyTiling, cholesky_test_tiling_rejects_negative_batch_dim)
{
    optiling::CholeskyCompileInfo compileInfo = {48};
    auto tilingContextPara = MakeCholeskyPara(&compileInfo, {-1, 4, 4}, false);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// 内层批量维为 -1：与首维为 -1 同型，累乘前回绕成 18446744073709551614
TEST_F(CholeskyTiling, cholesky_test_tiling_rejects_negative_inner_batch_dim)
{
    optiling::CholeskyCompileInfo compileInfo = {48};
    auto tilingContextPara = MakeCholeskyPara(&compileInfo, {2, -1, 4, 4}, false);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// 批量维为 GE 未知维标记 -2：与 -1 同一判定，必须显式拒绝
TEST_F(CholeskyTiling, cholesky_test_tiling_rejects_unknown_batch_dim)
{
    optiling::CholeskyCompileInfo compileInfo = {48};
    auto tilingContextPara = MakeCholeskyPara(&compileInfo, {-2, 4, 4}, false);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}
