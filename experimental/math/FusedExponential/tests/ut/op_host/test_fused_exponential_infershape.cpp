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
#include "infershape_case_executor.h"
#include "infershape_context_faker.h"
#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace {
class FusedExponentialInferShape : public testing::TestWithParam<std::vector<int64_t>> {};
TEST_P(FusedExponentialInferShape, PreservesShapeForEachSupportedDtype)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}) {
        gert::StorageShape input;
        for (auto dim : GetParam()) {
            input.MutableShape().AppendDim(dim);
            input.MutableStorageShape().AppendDim(dim);
        }
        gert::StorageShape output = {};
        gert::InfershapeContextPara context("FusedExponential", {{input, dtype, ge::FORMAT_ND}},
                                            {{output, dtype, ge::FORMAT_ND}});
        ExecuteTestCase(context, ge::GRAPH_SUCCESS, {GetParam()});
    }
}
INSTANTIATE_TEST_SUITE_P(Shapes, FusedExponentialInferShape,
                         testing::Values(std::vector<int64_t>{}, std::vector<int64_t>{0}, std::vector<int64_t>{2, 0, 3},
                                         std::vector<int64_t>{65}, std::vector<int64_t>{3, 5},
                                         std::vector<int64_t>{1, 2, 1, 2, 1, 2, 1, 2}));

TEST(FusedExponentialInferDtype, PreservesEachSupportedDtype)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}) {
        gert::OpInferDataTypeContextBuilder builder;
        auto holder = builder.OpType("FusedExponential")
                          .OpName("FusedExponential")
                          .IONum(1, 1)
                          .InputTensorDesc(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                          .OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();
        auto* context = holder.GetContext();
        ASSERT_NE(context, nullptr);
        auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
        ASSERT_NE(registry, nullptr);
        auto* implementation = registry->GetOpImpl("FusedExponential");
        ASSERT_NE(implementation, nullptr);
        ASSERT_NE(implementation->infer_datatype, nullptr);
        EXPECT_EQ(implementation->infer_datatype(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(context->GetOutputDataType(0), dtype);
    }
}

TEST(FusedExponentialInference, RejectsNullContexts)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto* implementation = registry->GetOpImpl("FusedExponential");
    ASSERT_NE(implementation, nullptr);
    ASSERT_NE(implementation->infer_shape, nullptr);
    ASSERT_NE(implementation->infer_datatype, nullptr);
    EXPECT_EQ(implementation->infer_shape(nullptr), ge::GRAPH_FAILED);
    EXPECT_EQ(implementation->infer_datatype(nullptr), ge::GRAPH_FAILED);
}
} // namespace
