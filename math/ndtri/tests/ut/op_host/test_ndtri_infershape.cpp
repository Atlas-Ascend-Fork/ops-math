/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class NdtriInferShape : public testing::TestWithParam<std::vector<int64_t>> {};

TEST_P(NdtriInferShape, OutputFollowsInput)
{
    const auto& shape = GetParam();
    gert::Shape inputShape;
    for (const auto dim : shape) {
        inputShape.AppendDim(dim);
    }
    gert::StorageShape inputStorageShape;
    inputStorageShape.MutableOriginShape() = inputShape;
    inputStorageShape.MutableStorageShape() = inputShape;
    const std::vector<gert::InfershapeContextPara::TensorDescription> inputs = {
        {inputStorageShape, ge::DT_FLOAT, ge::FORMAT_ND}};
    const std::vector<gert::InfershapeContextPara::TensorDescription> outputs = {
        {gert::StorageShape{}, ge::DT_FLOAT, ge::FORMAT_ND}};
    gert::InfershapeContextPara context("Ndtri", inputs, outputs);
    const std::vector<std::vector<int64_t>> expectedOutputShapes = {shape};
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, expectedOutputShapes);
}

INSTANTIATE_TEST_SUITE_P(ShapeContracts, NdtriInferShape,
                         testing::Values(std::vector<int64_t>{4, 3, 4}, std::vector<int64_t>{},
                                         std::vector<int64_t>{2, 0, 3}, std::vector<int64_t>{2, -1, 3},
                                         std::vector<int64_t>{-2}));
