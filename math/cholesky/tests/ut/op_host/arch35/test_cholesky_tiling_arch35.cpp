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
#include <limits>
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
    // 16MB 系统保留区 + 用户区：arch35 协同面板路径 needCoreNum=coreNumPlatForm=48
    // ⇒ 投票 48*4=192B（32B 对齐后 192B）+ 面板缩放 48*64*4=12288B = 12480B
    std::vector<size_t> expectWorkspaces = {16777216 + 12480};
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
    // 16MB 系统保留区 + 用户区：arch35 协同面板路径 needCoreNum=coreNumPlatForm=48
    // ⇒ 投票 48*4=192B（32B 对齐后 192B）+ 面板缩放 48*64*4=12288B = 12480B
    std::vector<size_t> expectWorkspaces = {16777216 + 12480};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CholeskyTiling, cholesky_test_tiling_single_large_matrix)
{
    // 单矩阵 M=8192 命中 Cube 键（M>=2048 且 M%64==0 ⇒ cube=true）：key = upper | cube<<1 = 2；
    // blockDim = AIC 数；workspace 追加 -Uᵀ/Uᵀ 两块 (2*64*M float)。
    optiling::CholeskyCompileInfo compileInfo = {48, 24};
    gert::TilingContextPara tilingContextPara(
        "Cholesky",
        {
            {{{8192, 8192}, {8192, 8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{8192, 8192}, {8192, 8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("upper", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo);
    uint64_t expectTilingKey = 2;
    string expectTilingData = "8192 1 137438953728 ";
    // 16MB 系统保留区 + 用户区：投票 48*4=192B + 面板缩放 48*64*4=12288B
    // + UT 两块 2*64*8192*4=4194304B，32B 对齐 = 4206784B
    std::vector<size_t> expectWorkspaces = {16777216 + 4206784};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.blockNum, 24U);
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

// CODE.05：RawTilingData 为空（框架失败模式）时必须在首次解引用前返回 GRAPH_FAILED。
// tilingDataSize 取 UINT64_MAX 使 TilingData::CreateCap 因加法溢出返回空指针
TEST_F(CholeskyTiling, cholesky_test_tiling_rejects_null_raw_tiling_data)
{
    optiling::CholeskyCompileInfo compileInfo = {48};
    gert::TilingContextPara tilingContextPara(
        "Cholesky",
        {
            {{{2, 2}, {2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 2}, {2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("upper", Ops::Math::AnyValue::CreateFrom<bool>(false))}, &compileInfo, 64,
        262144, std::numeric_limits<uint64_t>::max());
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

TEST_F(CholeskyTiling, cholesky_test_tiling_cube_workspace_uses_launched_vector_cores)
{
    // MIX_AIC_1_2 launches 64 AIV tasks for blockDim=32, regardless of the platform AIV count.
    optiling::CholeskyCompileInfo compileInfo = {48, 32};
    auto tilingContextPara = MakeCholeskyPara(&compileInfo, {2048, 2048}, true);
    // 64*4 vote bytes + 64*64*4 scale bytes + 2*64*2048*4 transpose bytes.
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 3, "2048 1 34359738624 ", {16777216 + 1065216});
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.blockNum, 32U);
}

TEST_F(CholeskyTiling, cholesky_test_tiling_without_cube_cores_uses_vector_path)
{
    optiling::CholeskyCompileInfo compileInfo = {48, 0};
    for (const bool upper : {false, true}) {
        auto tilingContextPara = MakeCholeskyPara(&compileInfo, {2048, 2048}, upper);
        ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, upper ? 1 : 0, "2048 1 34359738624 ", {16777216 + 12480});
        TilingInfo tilingInfo;
        ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
        EXPECT_EQ(tilingInfo.blockNum, 48U);
    }
}

// 合法空 batch（首维为 0）：matrixNumCount==0 不得产出 SetBlockDim(0)，运行时要求非零 blockDim。
// 与 aclnn 第一段 IsEmpty() 早退契约一致：kernel 批量循环体一次都不进入，不下发任何计算
TEST_F(CholeskyTiling, cholesky_test_tiling_empty_batch_leading_dim)
{
    optiling::CholeskyCompileInfo compileInfo = {48};
    auto tilingContextPara = MakeCholeskyPara(&compileInfo, {0, 4, 4}, false);
    // 空 batch 仍按全核下发（needCoreNum=48）：投票 192B + 缩放 12288B
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "4 0 4294967300 ", {16777216 + 12480});
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.blockNum, 48U);
}

// 合法空 batch 且最后一维恰为协同阈值：96 > 96 为假，走非协同分支，同样不得产出 SetBlockDim(0)
TEST_F(CholeskyTiling, cholesky_test_tiling_empty_batch_at_coop_threshold)
{
    optiling::CholeskyCompileInfo compileInfo = {48};
    auto tilingContextPara = MakeCholeskyPara(&compileInfo, {0, 96, 96}, false);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "96 0 4294967392 ", {16777216 + 12480});
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.blockNum, 48U);
}

// 合法空 batch（内层批量维为 0）：与首维为 0 同型
TEST_F(CholeskyTiling, cholesky_test_tiling_empty_batch_inner_dim)
{
    optiling::CholeskyCompileInfo compileInfo = {48};
    auto tilingContextPara = MakeCholeskyPara(&compileInfo, {2, 0, 4, 4}, false);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "4 0 4294967300 ", {16777216 + 12480});
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.blockNum, 48U);
}

// 合法空 batch 且最后一维超过协同阈值：走协同分支，blockDim 保持全核（修复前后行为一致）
TEST_F(CholeskyTiling, cholesky_test_tiling_empty_batch_cooperative_keeps_full_cores)
{
    optiling::CholeskyCompileInfo compileInfo = {48};
    auto tilingContextPara = MakeCholeskyPara(&compileInfo, {0, 200, 200}, false);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "200 0 4294967496 ", {16777216 + 12480});
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.blockNum, 48U);
}

// 最后一维为 0 的空张量（文档已声明的合法输入）：matrixNumCount==1，行为不得因本次修复改变
TEST_F(CholeskyTiling, cholesky_test_tiling_empty_last_dim_single_matrix)
{
    optiling::CholeskyCompileInfo compileInfo = {48};
    auto tilingContextPara = MakeCholeskyPara(&compileInfo, {0, 0}, false);
    // matrixNumCount==1 且 M<=4 ⇒ 单核：投票 1*4=4B（对齐 32B）+ 缩放 1*64*4=256B = 288B
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "0 1 4294967296 ", {16777216 + 288});
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.blockNum, 1U);
}

// 最后一维为 0 的空 batch（文档已声明的合法输入）：matrixNumCount==3，行为不得因本次修复改变
TEST_F(CholeskyTiling, cholesky_test_tiling_empty_last_dim_batch)
{
    optiling::CholeskyCompileInfo compileInfo = {48};
    auto tilingContextPara = MakeCholeskyPara(&compileInfo, {3, 0, 0}, false);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, "0 3 4294967296 ", {16777216 + 12480});
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.blockNum, 48U);
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
