/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_split_d_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

using namespace ge;
class SplitDInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SplitDInferShapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SplitDInferShapeTest TearDown" << std::endl; }
};

// x {12, 8}, split_dim=1, num_split=2. Expect each output {12, 4}.
TEST_F(SplitDInferShapeTest, split_d_infershape_success)
{
    gert::InfershapeContextPara infershapeContextPara("SplitD",
                                                      {
                                                          {{{12, 8}, {12, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"split_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                                          {"num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{12, 4}, {12, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Negative split_dim (-1 resolves to last dim). x {12, 8}, split_dim=-1, num_split=4. Expect {12, 2}.
TEST_F(SplitDInferShapeTest, split_d_infershape_negative_split_dim)
{
    gert::InfershapeContextPara infershapeContextPara("SplitD",
                                                      {
                                                          {{{12, 8}, {12, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"split_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
                                                          {"num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(4)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{12, 2}, {12, 2}, {12, 2}, {12, 2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// split_dim not divisible by num_split: x {12, 8}, split_dim=1, num_split=3. Expect failure.
TEST_F(SplitDInferShapeTest, split_d_infershape_not_divisible)
{
    gert::InfershapeContextPara infershapeContextPara("SplitD",
                                                      {
                                                          {{{12, 8}, {12, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"split_dim", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                                          {"num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// InferDataType: x is bfloat16, all dynamic outputs (num_split=2) must be bfloat16
TEST_F(SplitDInferShapeTest, split_d_infer_datatype_bf16_multi_output)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("SplitD");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("SplitD").OpName("SplitD");
    builder.IONum(1, 2);
    builder.InputTensorDesc(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND);
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = opImpl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_BF16);
}

// InferDataType: x is float32, all dynamic outputs (num_split=3) must be float32
TEST_F(SplitDInferShapeTest, split_d_infer_datatype_float_multi_output)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("SplitD");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("SplitD").OpName("SplitD");
    builder.IONum(1, 3);
    builder.InputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(2, ge::FORMAT_ND, ge::FORMAT_ND);
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = opImpl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT);
}
