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
 * \file test_hypot_infershape.cpp
 * \brief hypot infershape UT: numpy broadcast shape derivation (spec.yaml shape_rule / boundary_conditions)
 */

#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class HypotInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HypotInferShapeTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "HypotInferShapeTest TearDown" << std::endl; }
};

// same shape, no broadcast: [2,3] x [2,3] -> [2,3]
TEST_F(HypotInferShapeTest, infershape_same_shape)
{
    gert::InfershapeContextPara para("Hypot",
                                     {
                                         {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// broadcast with size-1 axes: [3,1] x [1,4] -> [3,4]
TEST_F(HypotInferShapeTest, infershape_broadcast_3x1_1x4)
{
    gert::InfershapeContextPara para("Hypot",
                                     {
                                         {{{3, 1}, {3, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{1, 4}, {1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 4}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 0-dim scalar + tensor: [] x [2,3] -> [2,3]
TEST_F(HypotInferShapeTest, infershape_scalar_tensor)
{
    gert::InfershapeContextPara para("Hypot",
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// different ranks, left-pad 1: [2,3] x [3] -> [2,3]
TEST_F(HypotInferShapeTest, infershape_left_pad)
{
    gert::InfershapeContextPara para("Hypot",
                                     {
                                         {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// rank-8 upper bound broadcast
TEST_F(HypotInferShapeTest, infershape_rank8)
{
    gert::InfershapeContextPara para(
        "Hypot",
        {
            {{{1, 1, 1, 1, 1, 1, 1, 2}, {1, 1, 1, 1, 1, 1, 1, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 1, 1, 1, 1, 1, 1, 1}, {2, 1, 1, 1, 1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 1, 1, 1, 1, 1, 1, 2}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// empty tensor (0 dim after broadcast): [0,3] x [3] -> [0,3]
TEST_F(HypotInferShapeTest, infershape_empty_tensor)
{
    gert::InfershapeContextPara para("Hypot",
                                     {
                                         {{{0, 3}, {0, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{0, 3}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// illegal broadcast: [2] x [3] -> fail
TEST_F(HypotInferShapeTest, infershape_broadcast_failed)
{
    gert::InfershapeContextPara para("Hypot",
                                     {
                                         {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}
