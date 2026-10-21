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
 * \file test_phony_concat_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

class PhonyConcatTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PhonyConcatTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "PhonyConcatTest TearDown" << std::endl; }
};

// 4 inputs {4,2,32,32} fp16, concat_dim={2,1}, N={2,2}: dim2*2=64, dim1*2=4. Expect output {4,4,64,32}.
TEST_F(PhonyConcatTest, phony_concat_infer_shape_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PhonyConcat",
        {
            {{{4, 2, 32, 32}, {4, 2, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 2, 32, 32}, {4, 2, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 2, 32, 32}, {4, 2, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 2, 32, 32}, {4, 2, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 1})},
            {"N", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2})},
        },
        {4}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 4, 64, 32}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// negative concat_dim: concat_dim={2,-3} resolves -3 to dim 1. Expect output {4,4,64,32}.
TEST_F(PhonyConcatTest, phony_concat_infer_shape_negative_concat_dim)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PhonyConcat",
        {
            {{{4, 2, 32, 32}, {4, 2, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 2, 32, 32}, {4, 2, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 2, 32, 32}, {4, 2, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 2, 32, 32}, {4, 2, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, -3})},
            {"N", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2})},
        },
        {4}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 4, 64, 32}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// concat_dim size(2) != N size(3). Expect failure.
TEST_F(PhonyConcatTest, phony_concat_infer_shape_size_mismatch)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PhonyConcat",
        {
            {{{4, 2, 32, 32}, {4, 2, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 2, 32, 32}, {4, 2, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 2, 32, 32}, {4, 2, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 2, 32, 32}, {4, 2, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, -3})},
            {"N", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, -1})},
        },
        {4}, {1});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// product of N(2*4=8) != input num(4). Expect failure.
TEST_F(PhonyConcatTest, phony_concat_infer_shape_n_product_mismatch)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PhonyConcat",
        {
            {{{4, 2, 32, 32}, {4, 2, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 2, 32, 32}, {4, 2, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 2, 32, 32}, {4, 2, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 2, 32, 32}, {4, 2, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, -3})},
            {"N", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 4})},
        },
        {4}, {1});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// concat_dim[1]=4 is out of range [-4, 4). Expect failure.
TEST_F(PhonyConcatTest, phony_concat_infer_shape_concat_dim_out_of_range)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PhonyConcat",
        {
            {{{4, 2, 32, 32}, {4, 2, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 2, 32, 32}, {4, 2, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 2, 32, 32}, {4, 2, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 2, 32, 32}, {4, 2, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 4})},
            {"N", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2})},
        },
        {4}, {1});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// product of N(-2*-2=4) equals input num, but N[0]=-2 < 1. Expect failure.
TEST_F(PhonyConcatTest, phony_concat_infer_shape_n_less_than_one)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PhonyConcat",
        {
            {{{4, 2, 32, 32}, {4, 2, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 2, 32, 32}, {4, 2, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 2, 32, 32}, {4, 2, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 2, 32, 32}, {4, 2, 32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 1})},
            {"N", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({-2, -2})},
        },
        {4}, {1});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// input shape is unknown rank {-2}: output keeps unknown rank.
TEST_F(PhonyConcatTest, phony_concat_infer_shape_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PhonyConcat",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
            {"N", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2})},
        },
        {2}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// dynamic dim: input {4,-1,32}, concat_dim={1}, N={2}: dim1 is -1 and kept as -1. Expect output {4,-1,32}.
TEST_F(PhonyConcatTest, phony_concat_infer_shape_dynamic_dim)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PhonyConcat",
        {
            {{{4, -1, 32}, {4, -1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, -1, 32}, {4, -1, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"concat_dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})},
            {"N", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2})},
        },
        {2}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, -1, 32}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// InferDataType copies the first input dtype to the output when all 4 dynamic inputs share the same dtype.
TEST_F(PhonyConcatTest, phony_concat_infer_data_type_success)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry, nullptr);
    auto op_impl = space_registry->GetOpImpl("PhonyConcat");
    ASSERT_NE(op_impl, nullptr);
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("PhonyConcat").OpName("PhonyConcat");
    builder.IONum(4, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto context_holder = builder.Build();
    auto* context = context_holder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = op_impl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

// InferDataType fails when dynamic input dtypes are inconsistent (RT1.0 verifier semantics):
// input[1] is DT_FLOAT while the others are DT_FLOAT16.
TEST_F(PhonyConcatTest, phony_concat_infer_data_type_mismatch)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry, nullptr);
    auto op_impl = space_registry->GetOpImpl("PhonyConcat");
    ASSERT_NE(op_impl, nullptr);
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("PhonyConcat").OpName("PhonyConcat");
    builder.IONum(4, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto context_holder = builder.Build();
    auto* context = context_holder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = op_impl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
