/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <cstring>
#include <gtest/gtest.h>
#include "../../../op_host/lin_space_tiling.h"
#include "../../../op_host/lin_space_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class LinSpaceTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "LinSpaceTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "LinSpaceTiling TearDown" << std::endl; }
};

TEST_F(LinSpaceTiling, lin_space_tiling_001)
{
    int32_t start = 0;
    int32_t stop = 8;
    int32_t num = 10;
    optiling::LinSpaceCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara("LinSpace",
                                              {{{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &start},
                                               {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &stop},
                                               {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &num}},
                                              {
                                                  {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 1002;
    string expectTilingData = "64 262144 10 1 21760 10 1 10 10 10 1 10 10 0 5 1 5 1 5 4683743612465315840 1063489081 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    // ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(LinSpaceTiling, lin_space_tiling_int64_big_num)
{
    int32_t start = 0;
    int32_t stop = 1;
    int64_t num = 16777219;
    optiling::LinSpaceCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara("LinSpace",
                                              {{{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &start},
                                               {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &stop},
                                               {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &num}},
                                              {
                                                  {{{16777219}, {16777219}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    TilingInfo tinfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tinfo));
    int64_t tilingNum = 0;
    // ascend950 regbase tiling 序列化 168 字节，ascend910b legacy tiling 序列化 128 字节，
    // sizeof(生成类) 含基类开销不可作判据，按序列化大小区分平台路径
    if (tinfo.tilingDataSize == 168) {
        EXPECT_EQ(tinfo.tilingKey, static_cast<int64_t>(optiling::DOUBLE_CAST_TILING_KEY));
        EXPECT_EQ(tinfo.blockNum, static_cast<size_t>(48));
        std::vector<int64_t> expectWorkspaces = {0};
        EXPECT_EQ(tinfo.workspaceSizes, expectWorkspaces);
    } else {
        EXPECT_EQ(tinfo.tilingDataSize, 128);
        EXPECT_EQ(tinfo.tilingKey, static_cast<int64_t>(602));
        EXPECT_EQ(tinfo.blockNum, static_cast<size_t>(48));
        std::vector<int64_t> expectWorkspaces = {32};
        EXPECT_EQ(tinfo.workspaceSizes, expectWorkspaces);
    }
    memcpy(&tilingNum, tinfo.tilingData.get() + 16, sizeof(int64_t));
    EXPECT_EQ(tilingNum, static_cast<int64_t>(16777219));
}

TEST_F(LinSpaceTiling, lin_space_tiling_int64_num)
{
    int32_t start = 0;
    int32_t stop = 1;
    int64_t num = 17;
    optiling::LinSpaceCompileInfo compileInfo = {48, 196608};
    gert::TilingContextPara tilingContextPara("LinSpace",
                                              {{{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &start},
                                               {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &stop},
                                               {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &num}},
                                              {
                                                  {{{17}, {17}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    TilingInfo tinfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tinfo));
    int64_t tilingNum = 0;
    if (tinfo.tilingDataSize == 168) {
        EXPECT_EQ(tinfo.tilingKey, static_cast<int64_t>(optiling::DOUBLE_CAST_TILING_KEY));
        EXPECT_EQ(tinfo.blockNum, static_cast<size_t>(1));
        std::vector<int64_t> expectWorkspaces = {0};
        EXPECT_EQ(tinfo.workspaceSizes, expectWorkspaces);
    } else {
        EXPECT_EQ(tinfo.tilingDataSize, 128);
        EXPECT_EQ(tinfo.tilingKey, static_cast<int64_t>(601));
        EXPECT_EQ(tinfo.blockNum, static_cast<size_t>(3));
        std::vector<int64_t> expectWorkspaces = {32};
        EXPECT_EQ(tinfo.workspaceSizes, expectWorkspaces);
    }
    memcpy(&tilingNum, tinfo.tilingData.get() + 16, sizeof(int64_t));
    EXPECT_EQ(tilingNum, static_cast<int64_t>(17));
}
