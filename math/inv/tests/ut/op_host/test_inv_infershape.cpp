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
 * \file test_inv_infershape.cpp
 * \brief Inv 算子 InferShape 推导 UT：单输入单输出逐元素，输出 shape 与输入一致
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class InvInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "InvInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "InvInfershape TearDown" << std::endl; }
};

TEST_F(InvInfershape, inv_infershape_multi_dim_test)
{
    gert::InfershapeContextPara infershapeContextPara("Inv",
                                                      {
                                                          {{{4, 3, 4}, {4, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(InvInfershape, inv_infershape_one_dim_test)
{
    gert::InfershapeContextPara infershapeContextPara("Inv",
                                                      {
                                                          {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {16},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(InvInfershape, inv_infershape_int32_test)
{
    gert::InfershapeContextPara infershapeContextPara("Inv",
                                                      {
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(InvInfershape, inv_infershape_bfloat16_test)
{
    gert::InfershapeContextPara infershapeContextPara("Inv",
                                                      {
                                                          {{{5, 8}, {5, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5, 8},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(InvInfershape, inv_infershape_scalar_test)
{
    gert::InfershapeContextPara infershapeContextPara("Inv",
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(InvInfershape, inv_infershape_eight_dim_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Inv",
        {
            {{{2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 2, 2, 2, 2, 2, 2, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ====================================================================
// 拒绝负例 —— 推理门与 Tiling 门对称（FR-06/FR-07）：rank>8 / 越界 dtype 显式拒绝
// ====================================================================

// rank=9（刚越界）-> kMaxRank 校验返回 GRAPH_FAILED
TEST_F(InvInfershape, inv_infershape_failed_rank9)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Inv",
        {
            {{{2, 2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// rank=12（明显越界）-> 同一校验拒绝
TEST_F(InvInfershape, inv_infershape_failed_rank12)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Inv",
        {
            {{{2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1}, {2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// 说明：InferDataType 回调注册于 op_graph/inv_graph_infer.cpp（上游 7906a4184 的
// def-driven dtype 分离结构），ophost UT 目标不链接 op_graph 源，故本文件不覆盖
// InferDataType 门；其 dtype 域校验由验证工作流 host-execution 节点经交付包
// （libinv_host_callbacks.so）直接回调探针验证。
