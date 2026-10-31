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
#include <gtest/gtest.h>
#include "../../../op_host/lin_space_tiling.h"
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
    // num=16777219（INT64）场景：legacy tiling 直接读取 int64_t 数值，
    // 计数必须保留 16777219，不得经 float 转换丢失精度（见 issue #3347）
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
    // tiling 计数保留 16777219：numPerCore 向上取整对齐后仍与 InferShape 输出维度一致
    TilingInfo tinfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tinfo));
    // tiling data 字段布局由 LinSpaceTilingData 定义（3 个 float 后接 int64 字段，4 字节对齐填充），
    // num 位于第 3 个 int64（索引 2），必须与输入 16777219 完全一致
    ASSERT_EQ(tinfo.tilingDataSize, static_cast<size_t>(128));
    const int64_t* p = reinterpret_cast<const int64_t*>(tinfo.tilingData.get());
    EXPECT_EQ(p[2], static_cast<int64_t>(16777219));
    EXPECT_EQ(tinfo.tilingKey, static_cast<int64_t>(602));
    EXPECT_EQ(tinfo.blockNum, static_cast<size_t>(48));
    std::vector<int64_t> expectWorkspaces = {32};
    EXPECT_EQ(tinfo.workspaceSizes, expectWorkspaces);
}

TEST_F(LinSpaceTiling, lin_space_tiling_int64_num)
{
    // num=17（INT64）小计数对照：tiling 计数必须保留 17，输出维度与 InferShape 一致
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
    ASSERT_EQ(tinfo.tilingDataSize, static_cast<size_t>(128));
    const int64_t* p = reinterpret_cast<const int64_t*>(tinfo.tilingData.get());
    EXPECT_EQ(p[2], static_cast<int64_t>(17));
    EXPECT_EQ(tinfo.tilingKey, static_cast<int64_t>(601));
    EXPECT_EQ(tinfo.blockNum, static_cast<size_t>(3));
    std::vector<int64_t> expectWorkspaces = {32};
    EXPECT_EQ(tinfo.workspaceSizes, expectWorkspaces);
}
