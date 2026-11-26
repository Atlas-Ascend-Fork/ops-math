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
 * \file test_permute_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

using namespace ge;

class PermuteInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PermuteInferShapeTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "PermuteInferShapeTest TearDown" << std::endl; }
};

static gert::InfershapeContextPara MakePermuteCase(const gert::StorageShape& xShape, const std::vector<int64_t>& order)
{
    gert::InfershapeContextPara::TensorDescription x(xShape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(xShape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::OpAttr orderAttr("order",
                                                  Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(order));
    return gert::InfershapeContextPara("Permute", {x}, {y}, {orderAttr});
}

TEST_F(PermuteInferShapeTest, permute_full_order)
{
    auto para = MakePermuteCase({{2, 3, 4, 5}, {2, 3, 4, 5}}, {0, 2, 3, 1});
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 4, 5, 3}});
}

TEST_F(PermuteInferShapeTest, permute_partial_order)
{
    auto para = MakePermuteCase({{2, 3, 4, 5}, {2, 3, 4, 5}}, {1, 0});
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{3, 2, 4, 5}});
}

TEST_F(PermuteInferShapeTest, permute_negative_order)
{
    auto para = MakePermuteCase({{2, 3, 4, 5}, {2, 3, 4, 5}}, {-1, 0, 1, 2});
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{5, 2, 3, 4}});
}

TEST_F(PermuteInferShapeTest, permute_empty_order_is_identity)
{
    auto para = MakePermuteCase({{2, 3, 4, 5}, {2, 3, 4, 5}}, {});
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 3, 4, 5}});
}

TEST_F(PermuteInferShapeTest, permute_unknown_rank)
{
    auto para = MakePermuteCase({{-2}, {-2}}, {0, 1, 2});
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{-1, -1, -1}});
}

TEST_F(PermuteInferShapeTest, permute_unknown_rank_empty_order)
{
    auto para = MakePermuteCase({{-2}, {-2}}, {});
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{-2}});
}

TEST_F(PermuteInferShapeTest, permute_unknown_rank_partial_order)
{
    auto para = MakePermuteCase({{-2}, {-2}}, {1, 0});
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{-1, -1}});
}

TEST_F(PermuteInferShapeTest, permute_unknown_rank_duplicate_order_failed)
{
    auto para = MakePermuteCase({{-2}, {-2}}, {0, 0});
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

TEST_F(PermuteInferShapeTest, permute_unknown_rank_order_out_of_range_failed)
{
    auto para = MakePermuteCase({{-2}, {-2}}, {0, 2});
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

TEST_F(PermuteInferShapeTest, permute_duplicate_order_failed)
{
    auto para = MakePermuteCase({{2, 3, 4, 5}, {2, 3, 4, 5}}, {0, 0, 2, 3});
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

TEST_F(PermuteInferShapeTest, permute_order_out_of_range_failed)
{
    auto para = MakePermuteCase({{2, 3, 4, 5}, {2, 3, 4, 5}}, {0, 1, 2, 4});
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

TEST_F(PermuteInferShapeTest, permute_order_longer_than_rank_failed)
{
    auto para = MakePermuteCase({{2, 3, 4, 5}, {2, 3, 4, 5}}, {0, 1, 2, 3, 4});
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}
