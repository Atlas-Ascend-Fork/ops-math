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
void CheckPow2(const std::initializer_list<int64_t>& x1, const std::initializer_list<int64_t>& x2,
               const std::vector<int64_t>& expected, ge::graphStatus status = ge::GRAPH_SUCCESS,
               const std::vector<size_t>& nullInputs = {})
{
    gert::InfershapeContextPara para(
        "Pow2", {{{x1, x1}, ge::DT_FLOAT, ge::FORMAT_ND}, {{x2, x2}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}}, std::vector<uint32_t>{}, std::vector<uint32_t>{}, nullInputs);
    ExecuteTestCase(para, status,
                    status == ge::GRAPH_SUCCESS ? std::vector<std::vector<int64_t>>{expected} :
                                                  std::vector<std::vector<int64_t>>{});
}
} // namespace

TEST(ExperimentalPow2InferShape, EqualShapes) { CheckPow2({2, 3}, {2, 3}, {2, 3}); }

TEST(ExperimentalPow2InferShape, RightAlignedDifferentRanks) { CheckPow2({2, 2, 2}, {2, 2}, {2, 2, 2}); }

TEST(ExperimentalPow2InferShape, BothInputsBroadcast) { CheckPow2({2, 1, 4}, {3, 1}, {2, 3, 4}); }

TEST(ExperimentalPow2InferShape, ScalarBase) { CheckPow2({}, {2, 3}, {2, 3}); }

TEST(ExperimentalPow2InferShape, ScalarExponent) { CheckPow2({2, 3}, {}, {2, 3}); }

TEST(ExperimentalPow2InferShape, BothScalars) { CheckPow2({}, {}, {}); }

TEST(ExperimentalPow2InferShape, EqualEmptyShapes) { CheckPow2({0, 3}, {0, 3}, {0, 3}); }

TEST(ExperimentalPow2InferShape, IncompatibleDimensionsFail) { CheckPow2({2, 3}, {2, 2}, {}, ge::GRAPH_FAILED); }

TEST(ExperimentalPow2InferShape, IncompatibleRanksFail) { CheckPow2({2, 2, 2}, {2, 3}, {}, ge::GRAPH_FAILED); }

TEST(ExperimentalPow2InferShape, MissingBaseFails) { CheckPow2({2, 3}, {2, 3}, {}, ge::GRAPH_FAILED, {0}); }

TEST(ExperimentalPow2InferShape, MissingExponentFails) { CheckPow2({2, 3}, {2, 3}, {}, ge::GRAPH_FAILED, {1}); }
