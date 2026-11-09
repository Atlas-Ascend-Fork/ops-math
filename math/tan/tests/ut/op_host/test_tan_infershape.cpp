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
 * \file test_tan_infershape.cpp
 * \brief Tan InferShape UT
 *
 * Covers paths in op_host/tan_infershape.cpp (InferShapeTan):
 * 输出 shape 与输入 shape 逐维一致(静态/动态维/未知秩/空张量/多维)。
 * 空指针分支(OP_CHECK_NULL_WITH_CONTEXT)在 faker 域不可达——输入/输出 shape
 * 槽由 InfershapeContextFaker 构建器保证非空。
 */

#include <gtest/gtest.h>
#include "infershape_case_executor.h"
#include "infershape_context_faker.h"

class TanInferShapeTest : public testing::Test {};

// 静态 4D shape 逐维拷贝
TEST_F(TanInferShapeTest, StaticShape4d)
{
    gert::InfershapeContextPara contextPara("Tan",
                                            {
                                                {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            });
    std::vector<std::vector<int64_t>> expectedOutputShapes = {{1, 64, 2, 64}};
    ExecuteTestCase(contextPara, ge::GRAPH_SUCCESS, expectedOutputShapes);
}

// 1D shape
TEST_F(TanInferShapeTest, StaticShape1d)
{
    gert::InfershapeContextPara contextPara("Tan",
                                            {
                                                {{{7}, {7}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            });
    std::vector<std::vector<int64_t>> expectedOutputShapes = {{7}};
    ExecuteTestCase(contextPara, ge::GRAPH_SUCCESS, expectedOutputShapes);
}

// FP16 5D shape
TEST_F(TanInferShapeTest, StaticShape5dFp16)
{
    gert::InfershapeContextPara contextPara("Tan",
                                            {
                                                {{{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                            });
    std::vector<std::vector<int64_t>> expectedOutputShapes = {{1, 2, 3, 4, 5}};
    ExecuteTestCase(contextPara, ge::GRAPH_SUCCESS, expectedOutputShapes);
}

// 动态维(-1)原样传播
TEST_F(TanInferShapeTest, DynamicDimension)
{
    gert::InfershapeContextPara contextPara("Tan",
                                            {
                                                {{{-1, 4}, {-1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            });
    std::vector<std::vector<int64_t>> expectedOutputShapes = {{-1, 4}};
    ExecuteTestCase(contextPara, ge::GRAPH_SUCCESS, expectedOutputShapes);
}

// 未知秩(-2)原样传播
TEST_F(TanInferShapeTest, UnknownRank)
{
    gert::InfershapeContextPara contextPara("Tan",
                                            {
                                                {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                            });
    std::vector<std::vector<int64_t>> expectedOutputShapes = {{-2}};
    ExecuteTestCase(contextPara, ge::GRAPH_SUCCESS, expectedOutputShapes);
}

// 空张量 shape({0,3,4})逐维拷贝
TEST_F(TanInferShapeTest, EmptyTensor)
{
    gert::InfershapeContextPara contextPara("Tan",
                                            {
                                                {{{0, 3, 4}, {0, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            });
    std::vector<std::vector<int64_t>> expectedOutputShapes = {{0, 3, 4}};
    ExecuteTestCase(contextPara, ge::GRAPH_SUCCESS, expectedOutputShapes);
}
