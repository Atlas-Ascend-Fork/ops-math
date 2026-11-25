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

class AddV2InferShapeTest : public testing::Test {};

TEST_F(AddV2InferShapeTest, StaticShapePreservesFirstInputShape)
{
    gert::InfershapeContextPara context("AddV2",
                                        {
                                            {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{3, 4}});
}

TEST_F(AddV2InferShapeTest, HigherRankShapePreservesFirstInputShape)
{
    gert::InfershapeContextPara context("AddV2",
                                        {
                                            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{2, 3, 4}});
}

TEST_F(AddV2InferShapeTest, DynamicShapePreservesFirstInputShape)
{
    gert::InfershapeContextPara context("AddV2",
                                        {
                                            {{{-1, 4}, {-1, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                            {{{-1, 4}, {-1, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{-1, 4}});
}
