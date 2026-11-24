/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class SelectTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SelectTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SelectTest TearDown" << std::endl; }
};

TEST_F(SelectTest, select_infer_shape_0)
{
    gert::StorageShape shape = {{2, 100}, {2, 100}};
    gert::StorageShape yShape = {{2, 100}, {2, 100}};
    gert::InfershapeContextPara::TensorDescription condition(shape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription x1(shape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription x2(shape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(yShape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Select", {condition, x1, x2}, {y});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 100},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SelectTest, select_infer_shape_dynamic_dim)
{
    // Dynamic shape: x1=[-1,2], x2=[2,2], output should be [2,2].
    gert::StorageShape x1Shape = {{-1, 2}, {-1, 2}};
    gert::StorageShape x2Shape = {{2, 2}, {2, 2}};
    gert::StorageShape conditionShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape;
    gert::InfershapeContextPara::TensorDescription condition(conditionShape, ge::DT_BOOL, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription x1(x1Shape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription x2(x2Shape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(yShape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Select", {condition, x1, x2}, {y});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SelectTest, select_infer_shape_unknown_rank_x1)
{
    // Unknown rank: x1=[-2], x2=[2,2], output should take x2 shape [2,2].
    gert::StorageShape x1Shape = {{ge::UNKNOWN_DIM_NUM}, {ge::UNKNOWN_DIM_NUM}};
    gert::StorageShape x2Shape = {{2, 2}, {2, 2}};
    gert::StorageShape conditionShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape;
    gert::InfershapeContextPara::TensorDescription condition(conditionShape, ge::DT_BOOL, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription x1(x1Shape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription x2(x2Shape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(yShape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Select", {condition, x1, x2}, {y});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SelectTest, select_infer_shape_unknown_rank_both)
{
    // Both unknown rank: x1=[-2], x2=[-2], output keeps [-2].
    gert::StorageShape x1Shape = {{ge::UNKNOWN_DIM_NUM}, {ge::UNKNOWN_DIM_NUM}};
    gert::StorageShape x2Shape = {{ge::UNKNOWN_DIM_NUM}, {ge::UNKNOWN_DIM_NUM}};
    gert::StorageShape conditionShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape;
    gert::InfershapeContextPara::TensorDescription condition(conditionShape, ge::DT_BOOL, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription x1(x1Shape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription x2(x2Shape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(yShape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Select", {condition, x1, x2}, {y});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {ge::UNKNOWN_DIM_NUM},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SelectTest, select_infer_shape_unknown_rank_x2)
{
    // x2 unknown rank, x1 known: x1=[2,2], x2=[-2], output keeps x1 [2,2].
    gert::StorageShape x1Shape = {{2, 2}, {2, 2}};
    gert::StorageShape x2Shape = {{ge::UNKNOWN_DIM_NUM}, {ge::UNKNOWN_DIM_NUM}};
    gert::StorageShape conditionShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape;
    gert::InfershapeContextPara::TensorDescription condition(conditionShape, ge::DT_BOOL, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription x1(x1Shape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription x2(x2Shape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(yShape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Select", {condition, x1, x2}, {y});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SelectTest, select_infer_shape_rank_mismatch)
{
    // Rank mismatch: x1=[-1,-1,3], x2=[-1,3], no fixing, output keeps x1 [-1,-1,3].
    gert::StorageShape x1Shape = {{-1, -1, 3}, {-1, -1, 3}};
    gert::StorageShape x2Shape = {{-1, 3}, {-1, 3}};
    gert::StorageShape conditionShape = {{3}, {3}};
    gert::StorageShape yShape;
    gert::InfershapeContextPara::TensorDescription condition(conditionShape, ge::DT_BOOL, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription x1(x1Shape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription x2(x2Shape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(yShape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Select", {condition, x1, x2}, {y});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SelectTest, select_infer_shape_x2_dynamic_dim)
{
    // x2 dynamic: x1=[2,5], x2=[2,-1], y keeps x1 [2,5]; helper reads x2 only.
    gert::StorageShape x1Shape = {{2, 5}, {2, 5}};
    gert::StorageShape x2Shape = {{2, -1}, {2, -1}};
    gert::StorageShape conditionShape = {{2, 5}, {2, 5}};
    gert::StorageShape yShape;
    gert::InfershapeContextPara::TensorDescription condition(conditionShape, ge::DT_BOOL, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription x1(x1Shape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription x2(x2Shape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(yShape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Select", {condition, x1, x2}, {y});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SelectTest, select_infer_shape_known_dim_conflict)
{
    // Known dim conflict: x1=[2,2], x2=[3,2], y keeps x1 [2,2] and infershape still succeeds.
    gert::StorageShape x1Shape = {{2, 2}, {2, 2}};
    gert::StorageShape x2Shape = {{3, 2}, {3, 2}};
    gert::StorageShape conditionShape = {{2, 2}, {2, 2}};
    gert::StorageShape yShape;
    gert::InfershapeContextPara::TensorDescription condition(conditionShape, ge::DT_BOOL, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription x1(x1Shape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription x2(x2Shape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y(yShape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Select", {condition, x1, x2}, {y});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
