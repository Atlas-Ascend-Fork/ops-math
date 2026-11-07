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

class SpaceToBatchNDInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SpaceToBatchNDInfershapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SpaceToBatchNDInfershapeTest TearDown" << std::endl; }
};

TEST_F(SpaceToBatchNDInfershapeTest, basic_4d_with_padding)
{
    // in [1, 4, 4, 3], bs=[2,2], pad=[[1,1],[1,1]]
    // padded = [6, 6], out_spatial = [3, 3], batch = 1*4 = 4
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> paddingsValues = {1, 1, 1, 1};

    gert::InfershapeContextPara para("SpaceToBatchND",
                                     {
                                         {{{1, 4, 4, 3}, {1, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                         {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });

    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 3, 3, 3}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SpaceToBatchNDInfershapeTest, basic_5d_with_inner_dims)
{
    // in [1, 4, 4, 2, 3], bs=[2,2], pad=[[0,0],[1,1]]
    // batch = 1*4 = 4, out_spatial = [2, 3], inner = [2, 3]
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> paddingsValues = {0, 0, 1, 1};

    gert::InfershapeContextPara para("SpaceToBatchND",
                                     {
                                         {{{1, 4, 4, 2, 3}, {1, 4, 4, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                         {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });

    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 2, 3, 2, 3}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SpaceToBatchNDInfershapeTest, basic_4d_1d_bs)
{
    // in [8, 996, 1, 512], bs=[2], pad=[[0,0]]
    // batch = 8*2 = 16, out_spatial = 996/2 = 498, trailing = [1, 512]
    std::vector<int32_t> blockShapeValues = {2};
    std::vector<int32_t> paddingsValues = {0, 0};

    gert::InfershapeContextPara para("SpaceToBatchND",
                                     {
                                         {{{8, 996, 1, 512}, {8, 996, 1, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                         {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });

    std::vector<std::vector<int64_t>> expectOutputShape = {{16, 498, 1, 512}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SpaceToBatchNDInfershapeTest, block_shape_zero_should_fail)
{
    std::vector<int32_t> blockShapeValues = {0};
    std::vector<int32_t> paddingsValues = {0, 0};

    gert::InfershapeContextPara para("SpaceToBatchND",
                                     {
                                         {{{2, 4, 3}, {2, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                         {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });

    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(SpaceToBatchNDInfershapeTest, negative_paddings_should_fail)
{
    std::vector<int32_t> blockShapeValues = {1};
    std::vector<int32_t> paddingsValues = {-2, 0};

    gert::InfershapeContextPara para("SpaceToBatchND",
                                     {
                                         {{{2, 1}, {2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                         {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });

    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(SpaceToBatchNDInfershapeTest, negative_paddings_large_should_fail)
{
    std::vector<int32_t> blockShapeValues = {1};
    std::vector<int32_t> paddingsValues = {-6, 0};

    gert::InfershapeContextPara para("SpaceToBatchND",
                                     {
                                         {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                         {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });

    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(SpaceToBatchNDInfershapeTest, non_divisible_spatial_dim_should_fail)
{
    std::vector<int32_t> blockShapeValues = {2};
    std::vector<int32_t> paddingsValues = {0, 0};

    gert::InfershapeContextPara para("SpaceToBatchND",
                                     {
                                         {{{1, 5, 4}, {1, 5, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                         {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });

    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(SpaceToBatchNDInfershapeTest, non_divisible_with_padding_should_fail)
{
    std::vector<int32_t> blockShapeValues = {2};
    std::vector<int32_t> paddingsValues = {1, 0};

    gert::InfershapeContextPara para("SpaceToBatchND",
                                     {
                                         {{{2, 6, 3}, {2, 6, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                         {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });

    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(SpaceToBatchNDInfershapeTest, batch_mul_overflow_should_fail)
{
    // block_shape = [2^62, 4], batchMul = 2^62 * 4 overflows int64
    std::vector<int64_t> blockShapeValues = {4611686018427387904LL, 4LL};
    std::vector<int64_t> paddingsValues = {0LL, 0LL, 0LL, 0LL};

    gert::InfershapeContextPara para("SpaceToBatchND",
                                     {
                                         {{{1, 4, 4}, {1, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeValues.data()},
                                         {{{2, 2}, {2, 2}}, ge::DT_INT64, ge::FORMAT_ND, true, paddingsValues.data()},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });

    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(SpaceToBatchNDInfershapeTest, first_dim_overflow_should_fail)
{
    // x dim0 = 2^62, block_shape = [4], firstDim * batchMul overflows int64
    std::vector<int64_t> blockShapeValues = {4LL};
    std::vector<int64_t> paddingsValues = {0LL, 0LL};

    gert::InfershapeContextPara para(
        "SpaceToBatchND",
        {
            {{{4611686018427387904LL, 1}, {4611686018427387904LL, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{1, 2}, {1, 2}}, ge::DT_INT64, ge::FORMAT_ND, true, paddingsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });

    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_F(SpaceToBatchNDInfershapeTest, padded_dim_overflow_should_fail)
{
    // spatial dim = INT64_MAX - 1, paddings = [1, 1], paddedDim overflows int64
    std::vector<int64_t> blockShapeValues = {1LL};
    std::vector<int64_t> paddingsValues = {1LL, 1LL};

    gert::InfershapeContextPara para(
        "SpaceToBatchND",
        {
            {{{1, 9223372036854775806LL, 4}, {1, 9223372036854775806LL, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{1, 2}, {1, 2}}, ge::DT_INT64, ge::FORMAT_ND, true, paddingsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });

    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
