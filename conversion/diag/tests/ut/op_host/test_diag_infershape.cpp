/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_diag_infershape.cpp
 * \brief Diag InferShape UT：覆盖 1D/2D 双份展开、unknown-rank 透传、非法输入分支。
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class diag : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "diag Proto Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "diag Proto Test TearDown" << std::endl; }
};

// 1D 输入：x(6,) → y(6,6)
TEST_F(diag, diag_infershape_1d)
{
    gert::InfershapeContextPara infershapeContextPara("Diag",
                                                      {
                                                          {{{6}, {6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{6, 6}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 2D 输入：x(3,4) → y(3,4,3,4)
TEST_F(diag, diag_infershape_2d)
{
    gert::InfershapeContextPara infershapeContextPara("Diag",
                                                      {
                                                          {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 4, 3, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// unknown rank：y 透传 unknown rank
TEST_F(diag, diag_infershape_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara("Diag",
                                                      {
                                                          {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
