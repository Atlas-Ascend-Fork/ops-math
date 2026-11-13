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
 * \file test_stateless_bernoulli_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

using namespace std;

class StatelessBernoulliInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "StatelessBernoulli SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "StatelessBernoulli TearDown" << std::endl; }
};

TEST_F(StatelessBernoulliInfershape, stateless_bernoulli_infershape_test1)
{
    vector<int64_t> shapeValue = {1};
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessBernoulli",
        {
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {
                {{1}, {1}},
                ge::DT_FLOAT,
                ge::FORMAT_ND,
            },
            {
                {{1}, {1}},
                ge::DT_INT64,
                ge::FORMAT_ND,
            },
            {
                {{1}, {1}},
                ge::DT_INT64,
                ge::FORMAT_ND,
            },
        },
        {
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StatelessBernoulliInfershape, stateless_bernoulli_infershape_test2)
{
    vector<int32_t> shapeValue = {1};
    gert::InfershapeContextPara infershapeContextPara(
        "StatelessBernoulli",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, shapeValue.data()},
            {
                {{1}, {1}},
                ge::DT_FLOAT,
                ge::FORMAT_ND,
            },
            {
                {{1}, {1}},
                ge::DT_INT64,
                ge::FORMAT_ND,
            },
            {
                {{1}, {1}},
                ge::DT_INT64,
                ge::FORMAT_ND,
            },
        },
        {
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(StatelessBernoulliInfershape, rank_zero_shape_ignores_placeholder)
{
    for (int64_t placeholder : {0L, 1L, 7L}) {
        gert::InfershapeContextPara params("StatelessBernoulli",
                                           {{{{}, {}}, ge::DT_INT64, ge::FORMAT_ND, true, &placeholder},
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND}},
                                           {{{{9}, {9}}, ge::DT_UINT8, ge::FORMAT_ND}});
        ExecuteTestCase(params, ge::GRAPH_SUCCESS, std::vector<std::vector<int64_t>>{{}});
    }
}

TEST_F(StatelessBernoulliInfershape, rank_zero_int32_shape_ignores_placeholder)
{
    for (int32_t placeholder : {0, 1, 7}) {
        gert::InfershapeContextPara params("StatelessBernoulli",
                                           {{{{}, {}}, ge::DT_INT32, ge::FORMAT_ND, true, &placeholder},
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND}},
                                           {{{{9}, {9}}, ge::DT_UINT8, ge::FORMAT_ND}});
        ExecuteTestCase(params, ge::GRAPH_SUCCESS, std::vector<std::vector<int64_t>>{{}});
    }
}

TEST_F(StatelessBernoulliInfershape, empty_shape_list_produces_scalar)
{
    gert::InfershapeContextPara params("StatelessBernoulli",
                                       {{{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND}},
                                       {{{{9}, {9}}, ge::DT_UINT8, ge::FORMAT_ND}});
    ExecuteTestCase(params, ge::GRAPH_SUCCESS, std::vector<std::vector<int64_t>>{{}});
}

TEST_F(StatelessBernoulliInfershape, rank_zero_shape_requires_all_inputs)
{
    int64_t placeholder = 0;
    for (size_t index : {1UL, 2UL, 3UL}) {
        gert::InfershapeContextPara params("StatelessBernoulli",
                                           {{{{}, {}}, ge::DT_INT64, ge::FORMAT_ND, true, &placeholder},
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND}},
                                           {{{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND}});
        params.nullInputIndices_ = {index};
        ExecuteTestCase(params, ge::GRAPH_FAILED, {});
    }
}

TEST_F(StatelessBernoulliInfershape, rank_zero_shape_rejects_float_dtype)
{
    float placeholder = 0;
    gert::InfershapeContextPara params("StatelessBernoulli",
                                       {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND, true, &placeholder},
                                        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND}},
                                       {{{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND}});
    ExecuteTestCase(params, ge::GRAPH_FAILED, {});
}

TEST_F(StatelessBernoulliInfershape, promoted_scalar_shape_uses_origin_rank)
{
    int32_t placeholder32 = 7;
    int64_t placeholder64 = 7;
    for (auto dtype : {ge::DT_INT32, ge::DT_INT64}) {
        void* data = dtype == ge::DT_INT32 ? static_cast<void*>(&placeholder32) : static_cast<void*>(&placeholder64);
        gert::InfershapeContextPara params("StatelessBernoulli",
                                           {{{{}, {1}}, dtype, ge::FORMAT_ND, true, data},
                                            {{{}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
                                            {{{}, {1}}, ge::DT_INT64, ge::FORMAT_ND}},
                                           {{{{9}, {9}}, ge::DT_UINT8, ge::FORMAT_ND}});
        ExecuteTestCase(params, ge::GRAPH_SUCCESS, std::vector<std::vector<int64_t>>{{}});
    }
}

TEST_F(StatelessBernoulliInfershape, one_element_shape_list_is_not_a_scalar_placeholder)
{
    int64_t dimension = 7;
    gert::InfershapeContextPara params("StatelessBernoulli",
                                       {{{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &dimension},
                                        {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
                                        {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND}},
                                       {{{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND}});
    ExecuteTestCase(params, ge::GRAPH_SUCCESS, std::vector<std::vector<int64_t>>{{7}});
}
