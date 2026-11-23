/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_tensor_equal_tiling_arch35.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "err_mgr.h"
#include "../../../../op_host/arch35/tensor_equal_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class TensorEqualTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TensorEqualTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TensorEqualTilingTest TearDown" << std::endl; }
};

TEST_F(TensorEqualTilingTest, TensorEqualReportsUbFailure)
{
    optiling::TensorEqualCompileInfo compileInfo = {64, 0};
    gert::TilingContextPara tilingContextPara("TensorEqual",
                                              {
                                                  {{{12, 12}, {12, 12}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{12, 12}, {12, 12}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1}, {1}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    error_message::GetErrMgrRawErrorMessages();
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
    auto reports = error_message::GetErrMgrRawErrorMessages();
    ASSERT_EQ(reports.size(), 1U);
    ASSERT_EQ(reports[0].args_key.size(), 2U);
    ASSERT_EQ(reports[0].args_value.size(), 2U);
    ASSERT_NE(reports[0].error_id.get(), nullptr);
    ASSERT_NE(reports[0].args_value[1].get(), nullptr);
    EXPECT_EQ(std::string(reports[0].error_id.get()), "EZ1008");
    EXPECT_STREQ(reports[0].args_key[0].get(), "opName");
    EXPECT_STREQ(reports[0].args_key[1].get(), "reason");
    EXPECT_STREQ(reports[0].args_value[0].get(), "TensorEqual");
    EXPECT_NE(std::string(reports[0].args_value[1].get()).find("actual=0"), std::string::npos);
    EXPECT_NE(std::string(reports[0].args_value[1].get()).find("expected>0"), std::string::npos);
    EXPECT_EQ(std::string(reports[0].error_id.get()), "EZ1008");
    EXPECT_EQ(std::string(reports[0].error_id.get()).find("EZ9999"), std::string::npos);
}

TEST_F(TensorEqualTilingTest, tensor_equal_test_01)
{
    optiling::TensorEqualCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara("TensorEqual",
                                              {
                                                  {{{12, 12}, {12, 12}}, ge::DT_INT64, ge::FORMAT_ND},
                                                  {{{12, 12}, {12, 12}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1}, {1}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 121;
    string expectTilingData = "144 8 1 253952 144 1 1 144 144 121 ";
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
