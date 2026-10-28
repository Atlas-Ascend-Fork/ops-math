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

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
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
