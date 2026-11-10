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
void CheckCdistGrad(const std::initializer_list<int64_t>& grad, const std::initializer_list<int64_t>& x1,
                    const std::initializer_list<int64_t>& x2, ge::DataType dtype, const std::vector<int64_t>& expected,
                    const std::vector<size_t>& nullInputs = {})
{
    gert::InfershapeContextPara para("CdistGrad",
                                     {{{grad, grad}, dtype, ge::FORMAT_ND},
                                      {{x1, x1}, dtype, ge::FORMAT_ND},
                                      {{x2, x2}, dtype, ge::FORMAT_ND},
                                      {{grad, grad}, dtype, ge::FORMAT_ND}},
                                     {{{{}, {}}, dtype, ge::FORMAT_ND}}, std::vector<uint32_t>{},
                                     std::vector<uint32_t>{}, nullInputs);
    ExecuteTestCase(
        para, nullInputs.empty() ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED,
        nullInputs.empty() ? std::vector<std::vector<int64_t>>{expected} : std::vector<std::vector<int64_t>>{});
}
} // namespace

// The experimental implementation returns x1's shape, not the pairwise-distance shape.
TEST(ExperimentalCdistGradInferShape, UnbatchedFloat) { CheckCdistGrad({4, 7}, {4, 5}, {7, 5}, ge::DT_FLOAT, {4, 5}); }

TEST(ExperimentalCdistGradInferShape, BatchedFloat16)
{
    CheckCdistGrad({2, 3, 7}, {2, 3, 5}, {2, 7, 5}, ge::DT_FLOAT16, {2, 3, 5});
}

TEST(ExperimentalCdistGradInferShape, MultipleBatchDimensions)
{
    CheckCdistGrad({2, 3, 4, 7}, {2, 3, 4, 5}, {2, 3, 7, 5}, ge::DT_FLOAT, {2, 3, 4, 5});
}

TEST(ExperimentalCdistGradInferShape, EmptyPoints)
{
    CheckCdistGrad({2, 0, 7}, {2, 0, 5}, {2, 7, 5}, ge::DT_FLOAT, {2, 0, 5});
}

TEST(ExperimentalCdistGradInferShape, DynamicBatchDimension)
{
    CheckCdistGrad({-1, 3, 7}, {-1, 3, 5}, {-1, 7, 5}, ge::DT_FLOAT, {-1, 3, 5});
}

TEST(ExperimentalCdistGradInferShape, MissingX1Fails) { CheckCdistGrad({4, 7}, {4, 5}, {7, 5}, ge::DT_FLOAT, {}, {1}); }
