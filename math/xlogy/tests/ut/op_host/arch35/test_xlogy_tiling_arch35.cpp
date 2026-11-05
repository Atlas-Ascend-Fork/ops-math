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
 * \file test_xlogy_tiling_arch35.cpp
 * \brief Xlogy tiling ut：覆盖输入dtype一致性校验与rank上界校验，以及正常shape的tiling执行
 */

#include "../../../../op_host/arch35/xlogy_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class XlogyTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "XlogyTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "XlogyTilingTest TearDown" << std::endl; }
};

// 正常用例：fp16 同dtype、4维，tiling执行成功
TEST_F(XlogyTilingTest, test_tiling_fp16_success)
{
    optiling::XlogyCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Xlogy",
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo);
    TilingInfo tilingInfo;
    auto ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
}

// 正常用例：fp32 广播shape（x1为{4,2}，x2为{2}），tiling执行成功
TEST_F(XlogyTilingTest, test_tiling_fp32_broadcast_success)
{
    optiling::XlogyCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Xlogy",
                                              {
                                                  {{{4, 2}, {4, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4, 2}, {4, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo);
    TilingInfo tilingInfo;
    auto ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
}

// 正常用例：bf16 8维，tiling执行成功
TEST_F(XlogyTilingTest, test_tiling_bf16_rank8_success)
{
    optiling::XlogyCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Xlogy",
        {
            {{{2, 3, 4, 5, 6, 7, 8, 2}, {2, 3, 4, 5, 6, 7, 8, 2}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 3, 4, 5, 6, 7, 8, 1}, {2, 3, 4, 5, 6, 7, 8, 1}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{2, 3, 4, 5, 6, 7, 8, 2}, {2, 3, 4, 5, 6, 7, 8, 2}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {}, &compileInfo);
    TilingInfo tilingInfo;
    auto ret = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ret);
}

// 输入dtype不一致：x1为FP16、x2为FP32，tiling失败
TEST_F(XlogyTilingTest, test_tiling_dtype_mismatch_failed)
{
    optiling::XlogyCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Xlogy",
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 输入dtype不一致：x1为BF16、x2为FP16，tiling失败
TEST_F(XlogyTilingTest, test_tiling_dtype_mismatch_bf16_failed)
{
    optiling::XlogyCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Xlogy",
                                              {
                                                  {{{8, 4}, {8, 4}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{8, 4}, {8, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 4}, {8, 4}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// rank越界：归一化后仍为9维（各维均非1，不会被PadAndSqueeze压缩），超过最大rank 8，tiling失败
TEST_F(XlogyTilingTest, test_tiling_rank_exceed_failed)
{
    optiling::XlogyCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Xlogy",
        {
            {{{2, 3, 4, 5, 6, 7, 8, 2, 2}, {2, 3, 4, 5, 6, 7, 8, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3, 4, 5, 6, 7, 8, 2, 2}, {2, 3, 4, 5, 6, 7, 8, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 3, 4, 5, 6, 7, 8, 2, 2}, {2, 3, 4, 5, 6, 7, 8, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// rank越界：广播归一化后rank超过8（x1为9维、x2为低维广播），tiling失败
TEST_F(XlogyTilingTest, test_tiling_rank_exceed_broadcast_failed)
{
    optiling::XlogyCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Xlogy",
        {
            {{{2, 3, 4, 5, 6, 7, 8, 2, 2}, {2, 3, 4, 5, 6, 7, 8, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 3, 4, 5, 6, 7, 8, 2, 2}, {2, 3, 4, 5, 6, 7, 8, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 不支持dtype：x1/x2为INT32，tiling失败
TEST_F(XlogyTilingTest, test_tiling_unsupported_dtype_failed)
{
    optiling::XlogyCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara("Xlogy",
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
