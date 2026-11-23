/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include "err_mgr.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "math/cumulative_logsumexp/op_kernel/arch35/cumulative_logsumexp_tiling_key.h"

class CumulativeLogsumexpTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CumulativeLogsumexpTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "CumulativeLogsumexpTiling TearDown" << std::endl; }
};

TEST_F(CumulativeLogsumexpTiling, fp32_rank3_axis1_inclusive_forward)
{
    int32_t axisValue = 1;
    int64_t compileInfo = 0;
    gert::TilingContextPara tilingContextPara(
        "CumulativeLogsumexp",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &axisValue},
        },
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("exclusive", Ops::Math::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("reverse", Ops::Math::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo, 8);

    uint64_t expectTilingKey = GET_TPL_TILING_KEY(CUMULATIVE_LOGSUMEXP_TPL_SCH_MODE_FLOAT);
    std::string expectTilingData = "24 2 3 4 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.blockNum, 3);
}

TEST_F(CumulativeLogsumexpTiling, fp16_negative_axis_exclusive_reverse)
{
    int64_t axisValue = -1;
    int64_t compileInfo = 0;
    gert::TilingContextPara tilingContextPara(
        "CumulativeLogsumexp",
        {
            {{{3, 5, 7}, {3, 5, 7}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &axisValue},
        },
        {
            {{{3, 5, 7}, {3, 5, 7}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("exclusive", Ops::Math::AnyValue::CreateFrom<bool>(true)),
            gert::TilingContextPara::OpAttr("reverse", Ops::Math::AnyValue::CreateFrom<bool>(true)),
        },
        &compileInfo, 64);

    uint64_t expectTilingKey = GET_TPL_TILING_KEY(CUMULATIVE_LOGSUMEXP_TPL_SCH_MODE_FLOAT16);
    std::string expectTilingData = "105 15 7 1 1 1 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(CumulativeLogsumexpTiling, int16_axis_scalar_supported)
{
    int16_t axisValue = 0;
    int64_t compileInfo = 0;
    gert::TilingContextPara tilingContextPara(
        "CumulativeLogsumexp",
        {
            {{{8, 16}, {8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT16, ge::FORMAT_ND, true, &axisValue},
        },
        {
            {{{8, 16}, {8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("exclusive", Ops::Math::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("reverse", Ops::Math::AnyValue::CreateFrom<bool>(true)),
        },
        &compileInfo);

    uint64_t expectTilingKey = GET_TPL_TILING_KEY(CUMULATIVE_LOGSUMEXP_TPL_SCH_MODE_FLOAT16);
    std::string expectTilingData = "128 1 8 16 0 1 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

static void CheckRootRaw(const std::string& code, const std::map<std::string, std::string>& expected)
{
    auto messages = error_message::GetErrMgrRawErrorMessages();
    ASSERT_EQ(messages.size(), 1U);
    ASSERT_NE(messages[0].error_id, nullptr);
    EXPECT_EQ(std::string(messages[0].error_id.get()), code);
    ASSERT_EQ(messages[0].args_key.size(), expected.size() + 1U);
    ASSERT_EQ(messages[0].args_value.size(), messages[0].args_key.size());
    std::map<std::string, std::string> args;
    for (size_t i = 0; i < messages[0].args_key.size(); ++i) {
        ASSERT_NE(messages[0].args_key[i], nullptr);
        ASSERT_NE(messages[0].args_value[i], nullptr);
        EXPECT_TRUE(args.emplace(messages[0].args_key[i].get(), messages[0].args_value[i].get()).second);
    }
    EXPECT_FALSE(args.at("op_name").empty());
    for (const auto& p : expected) {
        EXPECT_EQ(args.at(p.first), p.second);
    }
    EXPECT_TRUE(error_message::GetErrMgrRawErrorMessages().empty());
}

TEST_F(CumulativeLogsumexpTiling, unsupported_x_dtype_failed)
{
    int32_t axisValue = 0;
    int64_t compileInfo = 0;
    gert::TilingContextPara tilingContextPara(
        "CumulativeLogsumexp",
        {
            {{{2, 4}, {2, 4}}, ge::DT_DOUBLE, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &axisValue},
        },
        {
            {{{2, 4}, {2, 4}}, ge::DT_DOUBLE, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("exclusive", Ops::Math::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("reverse", Ops::Math::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo);

    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    (void)error_message::GetErrMgrRawErrorMessages();
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, std::vector<size_t>{});
    CheckRootRaw(
        "EZ0020",
        {{"param_name", "x"}, {"incorrect_dtype", "DT_DOUBLE"}, {"reason", "x dtype must be float32 or float16."}});
}

TEST_F(CumulativeLogsumexpTiling, axis_out_of_range_failed)
{
    int64_t axisValue = 2;
    int64_t compileInfo = 0;
    gert::TilingContextPara tilingContextPara(
        "CumulativeLogsumexp",
        {
            {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &axisValue},
        },
        {
            {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("exclusive", Ops::Math::AnyValue::CreateFrom<bool>(false)),
            gert::TilingContextPara::OpAttr("reverse", Ops::Math::AnyValue::CreateFrom<bool>(false)),
        },
        &compileInfo);

    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    (void)error_message::GetErrMgrRawErrorMessages();
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, std::vector<size_t>{});
    CheckRootRaw("EZ0026",
                 {{"param_name", "axis"}, {"incorrect_value", "2"}, {"reason", "axis must be in range [-2, 1]."}});
}
