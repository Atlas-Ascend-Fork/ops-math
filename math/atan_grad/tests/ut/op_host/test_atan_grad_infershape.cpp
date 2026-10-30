/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

class AtanGradInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AtanGradInfershapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AtanGradInfershapeTest TearDown" << std::endl; }
};

TEST_F(AtanGradInfershapeTest, atan_grad_infershape_2d)
{
    gert::StorageShape shape = {{32, 32}, {32, 32}};

    gert::InfershapeContextPara infershapeContextPara(
        "AtanGrad", {{shape, ge::DT_FLOAT, ge::FORMAT_ND}, {shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{shape, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{32, 32}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AtanGradInfershapeTest, atan_grad_infershape_1d)
{
    gert::StorageShape shape = {{128}, {128}};

    gert::InfershapeContextPara infershapeContextPara(
        "AtanGrad", {{shape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{shape, ge::DT_FLOAT16, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AtanGradInfershapeTest, atan_grad_infershape_3d)
{
    gert::StorageShape shape = {{4, 8, 16}, {4, 8, 16}};

    gert::InfershapeContextPara infershapeContextPara(
        "AtanGrad", {{shape, ge::DT_BF16, ge::FORMAT_ND}, {shape, ge::DT_BF16, ge::FORMAT_ND}},
        {{shape, ge::DT_BF16, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 8, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AtanGradInfershapeTest, atan_grad_infershape_4d)
{
    gert::StorageShape shape = {{2, 3, 4, 5}, {2, 3, 4, 5}};

    gert::InfershapeContextPara infershapeContextPara(
        "AtanGrad", {{shape, ge::DT_FLOAT, ge::FORMAT_ND}, {shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{shape, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// README 约束：y/dy shape 必须完全相同（不支持广播），shape 不一致在推理层拦截
TEST_F(AtanGradInfershapeTest, atan_grad_infershape_shape_mismatch_intercept)
{
    gert::StorageShape yShape = {{32, 32}, {32, 32}};
    gert::StorageShape dyShape = {{64, 64}, {64, 64}};

    gert::InfershapeContextPara infershapeContextPara(
        "AtanGrad", {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}, {dyShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{yShape, ge::DT_FLOAT, ge::FORMAT_ND}});

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// 元素数相同但维度排列不同（(4,8) vs (8,4)）也视为 shape 不一致
TEST_F(AtanGradInfershapeTest, atan_grad_infershape_dim_order_mismatch_intercept)
{
    gert::StorageShape yShape = {{4, 8}, {4, 8}};
    gert::StorageShape dyShape = {{8, 4}, {8, 4}};

    gert::InfershapeContextPara infershapeContextPara(
        "AtanGrad", {{yShape, ge::DT_FLOAT16, ge::FORMAT_ND}, {dyShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{yShape, ge::DT_FLOAT16, ge::FORMAT_ND}});

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// 维度数（rank）不一致拦截：0维标量 y 搭配 1维 dy
TEST_F(AtanGradInfershapeTest, atan_grad_infershape_rank_mismatch_intercept)
{
    gert::StorageShape scalarShape = {{}, {}};
    gert::StorageShape dyShape = {{1}, {1}};

    gert::InfershapeContextPara infershapeContextPara(
        "AtanGrad", {{scalarShape, ge::DT_FLOAT, ge::FORMAT_ND}, {dyShape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{scalarShape, ge::DT_FLOAT, ge::FORMAT_ND}});

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// README 约束：支持空 Tensor（元素个数为0）
TEST_F(AtanGradInfershapeTest, atan_grad_infershape_empty_tensor)
{
    gert::StorageShape shape = {{0}, {0}};

    gert::InfershapeContextPara infershapeContextPara(
        "AtanGrad", {{shape, ge::DT_FLOAT, ge::FORMAT_ND}, {shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{shape, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// README 约束：支持 0 维标量，此时 dy 也必须为 0 维
TEST_F(AtanGradInfershapeTest, atan_grad_infershape_scalar)
{
    gert::StorageShape shape = {{}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "AtanGrad", {{shape, ge::DT_FLOAT, ge::FORMAT_ND}, {shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{shape, ge::DT_FLOAT, ge::FORMAT_ND}});

    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// y shape=None（输入 shape 指针为空）在推理层拦截
TEST_F(AtanGradInfershapeTest, atan_grad_infershape_null_input_intercept)
{
    gert::StorageShape shape = {{32}, {32}};

    gert::InfershapeContextPara infershapeContextPara(
        "AtanGrad", {{shape, ge::DT_FLOAT, ge::FORMAT_ND}, {shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{shape, ge::DT_FLOAT, ge::FORMAT_ND}});
    infershapeContextPara.nullInputIndices_ = {0};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// dy shape=None（必选输入 dy 的 shape 指针为空）在推理层拦截
TEST_F(AtanGradInfershapeTest, atan_grad_infershape_null_dy_intercept)
{
    gert::StorageShape shape = {{32}, {32}};

    gert::InfershapeContextPara infershapeContextPara(
        "AtanGrad", {{shape, ge::DT_FLOAT, ge::FORMAT_ND}, {shape, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{shape, ge::DT_FLOAT, ge::FORMAT_ND}});
    infershapeContextPara.nullInputIndices_ = {1};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}
