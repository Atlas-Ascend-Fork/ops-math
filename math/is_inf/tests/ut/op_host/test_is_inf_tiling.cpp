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
 * \file test_is_inf_tiling.cpp
 * \brief dynamic tiling test of is_inf
 */

#include <iostream>

#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/is_inf_tiling_def.h"

class IsInfTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "IsInfTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "IsInfTiling TearDown" << std::endl; }
};

struct IsInfCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    bool isAscend310P = false;
};

TEST_F(IsInfTiling, is_inf_test_tiling_case0)
{
    IsInfCompileInfo compileInfo = {48, 196608, false};
    gert::TilingContextPara tilingContextPara("IsInf",
                                              {
                                                  {{{3, 6, 5}, {3, 6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 6, 5}, {3, 6, 5}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 103;
    string expectTilingData = "90 46179488366593 512 1 1 1 512 90 10752 1 ";
    std::vector<size_t> expectWorkspaces = {0};
    // ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(IsInfTiling, is_inf_tiling_float32_zero_workspace)
{
    optiling::IsInfCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("IsInf",
                                              {
                                                  {{{3, 6, 5}, {3, 6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 6, 5}, {3, 6, 5}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    // IsInf 的 kernel 不使用 workspace，tiling 申报 0，kernel 需支持零 workspace 执行。
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);
}

TEST_F(IsInfTiling, is_inf_tiling_float16_zero_workspace)
{
    optiling::IsInfCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("IsInf",
                                              {
                                                  {{{3, 6, 5}, {3, 6, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 6, 5}, {3, 6, 5}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);
}

TEST_F(IsInfTiling, is_inf_tiling_bfloat16_zero_workspace)
{
    optiling::IsInfCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("IsInf",
                                              {
                                                  {{{3, 6, 5}, {3, 6, 5}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 6, 5}, {3, 6, 5}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);
}
