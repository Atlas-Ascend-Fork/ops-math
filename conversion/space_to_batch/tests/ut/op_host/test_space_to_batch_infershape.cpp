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

class SpaceToBatchInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SpaceToBatchInfershapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SpaceToBatchInfershapeTest TearDown" << std::endl; }
};

TEST_F(SpaceToBatchInfershapeTest, KnownDimDivisible)
{
    // in [2, 4, 4, 3], bs=2, pad=[[1,1],[1,1]]
    // padded = [6, 6], out_spatial = [3, 3], batch = 2*2*2 = 8
    std::vector<int32_t> paddingsValues = {1, 1, 1, 1};

    gert::InfershapeContextPara para(
        "SpaceToBatch",
        {
            {{{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            gert::InfershapeContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{8, 3, 3, 3}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SpaceToBatchInfershapeTest, KnownDimNotDivisible)
{
    // in [1, 4, 4, 3], bs=2, pad=[[1,0],[0,0]]
    // H_padded = 5, not divisible by 2 → should fail
    std::vector<int32_t> paddingsValues = {1, 0, 0, 0};

    gert::InfershapeContextPara para(
        "SpaceToBatch",
        {
            {{{1, 4, 4, 3}, {1, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            gert::InfershapeContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
        });

    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(SpaceToBatchInfershapeTest, PaddingsFlat4Supported)
{
    // paddings declared as {4}, flat order [pad_top, pad_bottom, pad_left, pad_right]
    // in [1, 4, 4, 3], bs=2, pad=[1,1,0,0] → H_padded=6, W_padded=4 → out [4, 3, 2, 3]
    std::vector<int32_t> paddingsValues = {1, 1, 0, 0};

    gert::InfershapeContextPara para(
        "SpaceToBatch",
        {
            {{{1, 4, 4, 3}, {1, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            gert::InfershapeContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 3, 2, 3}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SpaceToBatchInfershapeTest, Paddings1x4Supported)
{
    // paddings declared as {1,4}, flat 4 elements
    // in [1, 4, 4, 3], bs=2, pad=0 → out [4, 2, 2, 3]
    std::vector<int32_t> paddingsValues = {0, 0, 0, 0};

    gert::InfershapeContextPara para(
        "SpaceToBatch",
        {
            {{{1, 4, 4, 3}, {1, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{1, 4}, {1, 4}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            gert::InfershapeContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 2, 2, 3}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SpaceToBatchInfershapeTest, ZeroNDimSupported)
{
    // in [0, 4, 4, 3], bs=2, pad=0 → out [0, 2, 2, 3]
    std::vector<int32_t> paddingsValues = {0, 0, 0, 0};

    gert::InfershapeContextPara para(
        "SpaceToBatch",
        {
            {{{0, 4, 4, 3}, {0, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            gert::InfershapeContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{0, 2, 2, 3}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SpaceToBatchInfershapeTest, ZeroCDimSupported)
{
    // in [1, 4, 4, 0], bs=2, pad=0 → out [4, 2, 2, 0]
    std::vector<int32_t> paddingsValues = {0, 0, 0, 0};

    gert::InfershapeContextPara para(
        "SpaceToBatch",
        {
            {{{1, 4, 4, 0}, {1, 4, 4, 0}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            gert::InfershapeContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 2, 2, 0}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SpaceToBatchInfershapeTest, ZeroHDimWithPaddingSupported)
{
    // in [1, 3, 0, 2], bs=2, pad=[[4,1],[1,1]] → H_padded=8, W_padded=2 → out [4, 4, 1, 2]
    std::vector<int32_t> paddingsValues = {4, 1, 1, 1};

    gert::InfershapeContextPara para(
        "SpaceToBatch",
        {
            {{{1, 3, 0, 2}, {1, 3, 0, 2}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            gert::InfershapeContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 4, 1, 2}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SpaceToBatchInfershapeTest, UnknownSpatialDimKept)
{
    // in [1, -1, -1, 3], bs=2, pad=[[0,0],[0,0]]
    // batch = 1*2*2 = 4, H/W stay unknown, C unchanged
    std::vector<int32_t> paddingsValues = {0, 0, 0, 0};

    gert::InfershapeContextPara para(
        "SpaceToBatch",
        {
            {{{1, -1, -1, 3}, {1, -1, -1, 3}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            gert::InfershapeContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {{4, -1, -1, 3}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}
