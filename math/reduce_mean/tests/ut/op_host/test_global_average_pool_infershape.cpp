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
 * \file test_global_average_pool_infershape.cpp
 * \brief GlobalAveragePool infershape UT
 */
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class GlobalAveragePoolInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GlobalAveragePoolInferShapeTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "GlobalAveragePoolInferShapeTest TearDown" << std::endl; }
};

// 3D ND: [N, C, D] -> [N, C, 1]
TEST_F(GlobalAveragePoolInferShapeTest, GlobalAveragePool_3d_nd)
{
    gert::InfershapeContextPara infershapeContextPara("GlobalAveragePool",
                                                      {{{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                      {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {{2, 3, 1}});
}

// 4D ND: [N, C, H, W] -> [N, C, 1, 1]
TEST_F(GlobalAveragePoolInferShapeTest, GlobalAveragePool_4d_nd)
{
    gert::InfershapeContextPara infershapeContextPara("GlobalAveragePool",
                                                      {{{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                      {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {{2, 3, 1, 1}});
}

// 4D NCHW: [N, C, H, W] -> [N, C, 1, 1]
TEST_F(GlobalAveragePoolInferShapeTest, GlobalAveragePool_4d_nchw)
{
    gert::InfershapeContextPara infershapeContextPara("GlobalAveragePool",
                                                      {{{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
                                                      {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {{2, 3, 1, 1}});
}

// 5D ND: [N, C, D1, D2, D3] -> [N, C, 1, 1, 1]
TEST_F(GlobalAveragePoolInferShapeTest, GlobalAveragePool_5d_nd)
{
    gert::InfershapeContextPara infershapeContextPara(
        "GlobalAveragePool", {{{{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {{2, 3, 1, 1, 1}});
}

// 5D NCDHW: [N, C, D, H, W] -> [N, C, 1, 1, 1]
TEST_F(GlobalAveragePoolInferShapeTest, GlobalAveragePool_5d_ncdhw)
{
    gert::InfershapeContextPara infershapeContextPara(
        "GlobalAveragePool", {{{{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}}, ge::DT_FLOAT, ge::FORMAT_NCDHW}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NCDHW}});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {{2, 3, 1, 1, 1}});
}

// 2D 输入不支持
TEST_F(GlobalAveragePoolInferShapeTest, GlobalAveragePool_2d_fail)
{
    gert::InfershapeContextPara infershapeContextPara("GlobalAveragePool",
                                                      {{{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                                      {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// 4D NDHWC 格式不支持
TEST_F(GlobalAveragePoolInferShapeTest, GlobalAveragePool_4d_ndhwc_fail)
{
    gert::InfershapeContextPara infershapeContextPara("GlobalAveragePool",
                                                      {{{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_NDHWC}},
                                                      {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NDHWC}});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// 6D 输入不支持
TEST_F(GlobalAveragePoolInferShapeTest, GlobalAveragePool_6d_fail)
{
    gert::InfershapeContextPara infershapeContextPara(
        "GlobalAveragePool", {{{{2, 3, 4, 5, 6, 7}, {2, 3, 4, 5, 6, 7}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}
