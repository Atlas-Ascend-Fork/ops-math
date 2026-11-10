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
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

namespace {
void CheckLerp(const std::initializer_list<int64_t>& start, const std::initializer_list<int64_t>& end,
               const std::initializer_list<int64_t>& weight, ge::DataType dtype, const std::vector<int64_t>& expected)
{
    gert::InfershapeContextPara para("Lerp",
                                     {{{start, start}, dtype, ge::FORMAT_ND},
                                      {{end, end}, dtype, ge::FORMAT_ND},
                                      {{weight, weight}, dtype, ge::FORMAT_ND}},
                                     {{{{}, {}}, dtype, ge::FORMAT_ND}});
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {expected});
}
} // namespace

TEST(ExperimentalLerpInferShape, EqualShapesFloat) { CheckLerp({2, 3}, {2, 3}, {2, 3}, ge::DT_FLOAT, {2, 3}); }

TEST(ExperimentalLerpInferShape, EndAndWeightBroadcastToStart)
{
    CheckLerp({2, 3, 4}, {1, 3, 1}, {4}, ge::DT_FLOAT16, {2, 3, 4});
}

TEST(ExperimentalLerpInferShape, ScalarBFloat16) { CheckLerp({}, {}, {}, ge::DT_BF16, {}); }

TEST(ExperimentalLerpInferShape, EmptyTensor) { CheckLerp({2, 0, 4}, {1}, {1}, ge::DT_FLOAT, {2, 0, 4}); }

TEST(ExperimentalLerpInferShape, DynamicDimension) { CheckLerp({-1, 4}, {-1, 4}, {1}, ge::DT_FLOAT, {-1, 4}); }
