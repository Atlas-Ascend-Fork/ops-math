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
 * \file test_stateless_sample_multinomial_tiling.cpp
 * \brief StatelessSampleMultinomial tiling UT
 */

#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "err_mgr.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/stateless_sample_multinomial_tiling.h"

namespace {
// 校验根因层正式首报:第一条为算子侧正式宏上报(EZ0012/EZ0010),
// 第二条为共享 RandomTilingArch35::DoTiling 外层对 UniqueProcess 失败的既有 EZ9999 重复日志。
void ExpectFormalFirstError(const char* code, const std::map<std::string, std::string>& expectArgs)
{
    auto reports = error_message::GetErrMgrRawErrorMessages();
    ASSERT_EQ(reports.size(), 2U);
    ASSERT_NE(reports[0].error_id.get(), nullptr);
    EXPECT_STREQ(reports[0].error_id.get(), code);
    std::map<std::string, std::string> args;
    for (size_t i = 0; i < reports[0].args_key.size(); ++i) {
        ASSERT_NE(reports[0].args_key[i].get(), nullptr);
        ASSERT_NE(reports[0].args_value[i].get(), nullptr);
        args.emplace(reports[0].args_key[i].get(), reports[0].args_value[i].get());
    }
    EXPECT_EQ(args, expectArgs);
    ASSERT_NE(reports[1].error_id.get(), nullptr);
    EXPECT_STREQ(reports[1].error_id.get(), "EZ9999");
}
} // namespace

class StatelessSampleMultinomialTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "StatelessSampleMultinomialTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "StatelessSampleMultinomialTilingTest TearDown" << std::endl; }
};

TEST_F(StatelessSampleMultinomialTilingTest, one_dim_float)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    int64_t seedValue = 12345;
    int64_t offsetValue = 0;

    gert::TilingContextPara tilingContextPara("StatelessSampleMultinomial",
                                              {
                                                  {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedValue},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetValue},
                                              },
                                              {
                                                  {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"num_samples", Ops::Math::AnyValue::CreateFrom<int64_t>(8)},
                                              },
                                              &compileInfo);

    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// x 为 3D 时应失败并正式上报 EZ0012(x rank 必须为 1D/2D),首错为根因层正式码
TEST_F(StatelessSampleMultinomialTilingTest, three_dim_x_reports_formal_shapedim)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    int64_t seedValue = 12345;
    int64_t offsetValue = 0;

    gert::TilingContextPara tilingContextPara("StatelessSampleMultinomial",
                                              {
                                                  {{{3, 4, 5}, {3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{3, 4, 5}, {3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedValue},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetValue},
                                              },
                                              {
                                                  {{{8}, {8}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"num_samples", Ops::Math::AnyValue::CreateFrom<int64_t>(8)},
                                              },
                                              &compileInfo);

    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    error_message::GetErrMgrRawErrorMessages();
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
    ExpectFormalFirstError("EZ0012", {{"param_name", "input tensor x"},
                                      {"op_name", "StatelessSampleMultinomial"},
                                      {"incorrect_dim", "3D"},
                                      {"reason", "x must be 1D or 2D"}});
}

// norm_probs 与 x shape 不一致时应失败并正式上报 EZ0010(双 tensor shape 必须一致)
TEST_F(StatelessSampleMultinomialTilingTest, norm_probs_shape_mismatch_reports_formal_shapes)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    int64_t seedValue = 7;
    int64_t offsetValue = 4;

    gert::TilingContextPara tilingContextPara("StatelessSampleMultinomial",
                                              {
                                                  {{{3, 10}, {3, 10}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{3, 11}, {3, 11}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedValue},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetValue},
                                              },
                                              {
                                                  {{{3, 5}, {3, 5}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"num_samples", Ops::Math::AnyValue::CreateFrom<int64_t>(5)},
                                              },
                                              &compileInfo);

    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    error_message::GetErrMgrRawErrorMessages();
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
    ExpectFormalFirstError("EZ0010", {{"param_names", "x and norm_probs"},
                                      {"op_name", "StatelessSampleMultinomial"},
                                      {"incorrect_shapes", "[3, 10] and [3, 11]"},
                                      {"reason", "the shapes of x and norm_probs must be the same"}});
}

TEST_F(StatelessSampleMultinomialTilingTest, two_dim_float16)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    int64_t seedValue = 7;
    int64_t offsetValue = 4;

    gert::TilingContextPara tilingContextPara("StatelessSampleMultinomial",
                                              {
                                                  {{{3, 10}, {3, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{3, 10}, {3, 10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedValue},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetValue},
                                              },
                                              {
                                                  {{{3, 5}, {3, 5}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"num_samples", Ops::Math::AnyValue::CreateFrom<int64_t>(5)},
                                              },
                                              &compileInfo);

    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(StatelessSampleMultinomialTilingTest, two_dim_bf16)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    int64_t seedValue = 99;
    int64_t offsetValue = 8;

    gert::TilingContextPara tilingContextPara("StatelessSampleMultinomial",
                                              {
                                                  {{{2, 64}, {2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{2, 64}, {2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedValue},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetValue},
                                              },
                                              {
                                                  {{{2, 16}, {2, 16}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"num_samples", Ops::Math::AnyValue::CreateFrom<int64_t>(16)},
                                              },
                                              &compileInfo);

    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(StatelessSampleMultinomialTilingTest, offset_multiple_of_four)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    int64_t seedValue = 1;
    int64_t offsetValue = 4;

    gert::TilingContextPara tilingContextPara("StatelessSampleMultinomial",
                                              {
                                                  {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedValue},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetValue},
                                              },
                                              {
                                                  {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"num_samples", Ops::Math::AnyValue::CreateFrom<int64_t>(4)},
                                              },
                                              &compileInfo);

    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}
