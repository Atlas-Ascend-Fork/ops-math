/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "infershape_case_executor.h"
#include "infershape_context_faker.h"

class SinkhornInfershape : public testing::Test {};

TEST_F(SinkhornInfershape, concrete_2d)
{
    gert::InfershapeContextPara context("Sinkhorn", {{{{8, 16}, {8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{8, 16}});
}

TEST_F(SinkhornInfershape, empty_2d)
{
    gert::InfershapeContextPara context("Sinkhorn", {{{{0, 16}, {0, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}});
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{0, 16}});
}

TEST_F(SinkhornInfershape, dynamic_2d)
{
    gert::InfershapeContextPara context("Sinkhorn", {{{{-1, 16}, {-1, 16}}, ge::DT_BF16, ge::FORMAT_ND}},
                                        {{{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}});
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{-1, 16}});
}

TEST_F(SinkhornInfershape, unknown_rank)
{
    gert::InfershapeContextPara context("Sinkhorn", {{{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{-2}});
}

TEST_F(SinkhornInfershape, invalid_rank)
{
    gert::InfershapeContextPara context("Sinkhorn", {{{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    ExecuteTestCase(context, ge::GRAPH_FAILED, {});
}

TEST_F(SinkhornInfershape, invalid_negative_dimension)
{
    gert::InfershapeContextPara context("Sinkhorn", {{{{-3, 16}, {-3, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    ExecuteTestCase(context, ge::GRAPH_FAILED, {});
}
