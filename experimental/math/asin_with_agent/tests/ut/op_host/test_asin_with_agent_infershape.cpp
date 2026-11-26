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
#include <vector>
#include "infershape_case_executor.h"

class AsinWithAgentInfershape : public testing::Test {};

TEST_F(AsinWithAgentInfershape, fp32_multi_dim)
{
    gert::InfershapeContextPara para("AsinWithAgent",
                                     {
                                         {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AsinWithAgentInfershape, fp16_scalar)
{
    gert::InfershapeContextPara para("AsinWithAgent",
                                     {
                                         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AsinWithAgentInfershape, fp32_dynamic_dim)
{
    gert::InfershapeContextPara para("AsinWithAgent",
                                     {
                                         {{{-1, 4}, {-1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 4}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AsinWithAgentInfershape, fp32_unknown_rank)
{
    gert::InfershapeContextPara para("AsinWithAgent",
                                     {
                                         {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AsinWithAgentInfershape, int32_empty_tensor)
{
    gert::InfershapeContextPara para("AsinWithAgent",
                                     {
                                         {{{2, 0, 4}, {2, 0, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 0, 4}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}
