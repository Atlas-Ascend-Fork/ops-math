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

class ExperimentalLessInferShapeTest : public testing::TestWithParam<std::vector<int64_t>> {};

TEST_P(ExperimentalLessInferShapeTest, PreservesInputShape)
{
    const auto& shape = GetParam();
    gert::StorageShape inputShape;
    for (const auto dim : shape) {
        inputShape.MutableOriginShape().AppendDim(dim);
        inputShape.MutableStorageShape().AppendDim(dim);
    }
    gert::InfershapeContextPara context("Less",
                                        {
                                            {inputShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                            {inputShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                        },
                                        {
                                            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
                                        });
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {shape});
}

INSTANTIATE_TEST_SUITE_P(ShapeCoverage, ExperimentalLessInferShapeTest,
                         testing::Values(std::vector<int64_t>{}, std::vector<int64_t>{2, 3, 4},
                                         std::vector<int64_t>{-1, 4}, std::vector<int64_t>{-2},
                                         std::vector<int64_t>{2, 0, 4}, std::vector<int64_t>{1, 2, 1, 2, 1, 2, 1, 2}));
