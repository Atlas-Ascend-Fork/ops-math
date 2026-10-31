/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class BatchToSpaceInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BatchToSpaceInfershapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "BatchToSpaceInfershapeTest TearDown" << std::endl; }
};

TEST_F(BatchToSpaceInfershapeTest, batch_to_space_basic_4d)
{
    int64_t cropsValues[4] = {0, 0, 0, 0};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpace",
        {
            {{{8, 28, 28, 128}, {8, 28, 28, 128}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
            {{{2, 2}, {2, 2}}, ge::DT_INT64, ge::FORMAT_ND, true, cropsValues},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            gert::InfershapeContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 56, 56, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BatchToSpaceInfershapeTest, batch_to_space_nonconst_crops_valid_shape)
{
    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpace",
        {
            {{{8, 28, 28, 128}, {8, 28, 28, 128}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
            {{{2, 2}, {2, 2}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            gert::InfershapeContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{2, -1, -1, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BatchToSpaceInfershapeTest, batch_to_space_nonconst_crops_wrong_element_count)
{
    // 非 const crops 不在推理阶段校验元素数（对齐 batch_to_space_nd）：
    // 动态图下非 const crops 的形状是未知描述符（-1/-2），GetShapeSize() 是无意义值，
    // 据此拒绝会阻断合法动态图；错误元素数交由运行时 tiling 以真实形状拒绝。
    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpace",
        {
            {{{8, 28, 28, 128}, {8, 28, 28, 128}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
            {{{4, 2}, {4, 2}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            gert::InfershapeContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{2, -1, -1, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BatchToSpaceInfershapeTest, batch_to_space_nonconst_crops_unknown_rank_desc)
{
    // 动态图未知秩描述符（-2，TTK geir dynamic 模式实测形态）：不得被
    // "crops must have 4 elements, got -2" 误拒；输出 H/W 未知，C 原样透传。
    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpace",
        {
            {{{8, 28, 28, 128}, {8, 28, 28, 128}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
            {{{-2}, {-2}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            gert::InfershapeContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{2, -1, -1, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BatchToSpaceInfershapeTest, batch_to_space_nonconst_crops_unknown_dims_desc)
{
    // 动态图已知秩、未知维描述符（-1,-1，TTK geir dynamic 模式实测形态，
    // GetShapeSize() 值为 1）：同上不得误拒。
    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpace",
        {
            {{{8, 28, 28, 128}, {8, 28, 28, 128}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
            {{{-1, -1}, {-1, -1}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            gert::InfershapeContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{2, -1, -1, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BatchToSpaceInfershapeTest, batch_to_space_const_crops_wrong_element_count)
{
    int64_t cropsValues[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpace",
        {
            {{{8, 28, 28, 128}, {8, 28, 28, 128}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
            {{{4, 2}, {4, 2}}, ge::DT_INT64, ge::FORMAT_ND, true, cropsValues},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            gert::InfershapeContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
        });

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(BatchToSpaceInfershapeTest, batch_to_space_zero_batch)
{
    int64_t cropsValues[4] = {0, 0, 0, 0};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpace",
        {
            {{{0, 28, 28, 128}, {0, 28, 28, 128}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
            {{{2, 2}, {2, 2}}, ge::DT_INT64, ge::FORMAT_ND, true, cropsValues},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            gert::InfershapeContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{0, 56, 56, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BatchToSpaceInfershapeTest, batch_to_space_zero_c)
{
    int64_t cropsValues[4] = {0, 0, 0, 0};

    gert::InfershapeContextPara infershapeContextPara(
        "BatchToSpace",
        {
            {{{8, 28, 28, 0}, {8, 28, 28, 0}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
            {{{2, 2}, {2, 2}}, ge::DT_INT64, ge::FORMAT_ND, true, cropsValues},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            gert::InfershapeContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 56, 56, 0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
