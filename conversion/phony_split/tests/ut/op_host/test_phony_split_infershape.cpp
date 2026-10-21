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
 * \file test_phony_split_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

class PhonySplitTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PhonySplitTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "PhonySplitTest TearDown" << std::endl; }
};

// input {4,4,64,32} fp16, split_dim={2,1}, num_split={2,2}, 4 outputs:
// dim2 64/2=32, dim1 4/2=2. Expect all outputs {4,2,32,32}.
TEST_F(PhonySplitTest, phony_split_infer_shape_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PhonySplit",
        {
            {{{4, 4, 64, 32}, {4, 4, 64, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"split_dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 1})},
            {"num_split", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2})},
        },
        {1}, {4});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 2, 32, 32}, {4, 2, 32, 32}, {4, 2, 32, 32}, {4, 2, 32, 32}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// negative split_dim: split_dim={2,-3} resolves -3 to dim 1. Expect all outputs {4,2,32,32}.
TEST_F(PhonySplitTest, phony_split_infer_shape_negative_split_dim)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PhonySplit",
        {
            {{{4, 4, 64, 32}, {4, 4, 64, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"split_dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, -3})},
            {"num_split", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2})},
        },
        {1}, {4});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 2, 32, 32}, {4, 2, 32, 32}, {4, 2, 32, 32}, {4, 2, 32, 32}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// split_dim size(4) != num_split size(2). Expect failure.
TEST_F(PhonySplitTest, phony_split_infer_shape_size_mismatch)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PhonySplit",
        {
            {{{4, 4, 64, 32}, {4, 4, 64, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"split_dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 1, 0, 0})},
            {"num_split", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2})},
        },
        {1}, {4});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// split_dim[1]=-6 out of range [-4, 4). Expect failure.
TEST_F(PhonySplitTest, phony_split_infer_shape_split_dim_out_of_range)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PhonySplit",
        {
            {{{4, 4, 64, 32}, {4, 4, 64, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"split_dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, -6})},
            {"num_split", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2})},
        },
        {1}, {4});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// product of num_split(2*2=4) != output num(2). Expect failure.
TEST_F(PhonySplitTest, phony_split_infer_shape_num_split_product_mismatch)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PhonySplit",
        {
            {{{4, 4, 64, 32}, {4, 4, 64, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"split_dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 1})},
            {"num_split", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2})},
        },
        {1}, {2});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// product of num_split(-2*-2=4) equals output num, but num_split[0]=-2 < 1. Expect failure.
TEST_F(PhonySplitTest, phony_split_infer_shape_num_split_less_than_one)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PhonySplit",
        {
            {{{4, 4, 64, 32}, {4, 4, 64, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"split_dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 1})},
            {"num_split", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({-2, -2})},
        },
        {1}, {4});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// dim0=4 not divisible by num_split=3 (from source UT). Expect failure.
TEST_F(PhonySplitTest, phony_split_infer_shape_not_divisible)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PhonySplit",
        {
            {{{4, 4, 64, 32}, {4, 4, 64, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"split_dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
            {"num_split", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({3})},
        },
        {1}, {3});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// input shape is unknown rank {-2}: all outputs keep unknown rank.
TEST_F(PhonySplitTest, phony_split_infer_shape_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PhonySplit",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"split_dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
            {"num_split", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2})},
        },
        {1}, {2});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}, {-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// dynamic dim: input {4,-1,64}, split_dim={1}, num_split={2}: dim1 is -1 and kept as -1.
TEST_F(PhonySplitTest, phony_split_infer_shape_dynamic_dim)
{
    gert::InfershapeContextPara infershapeContextPara(
        "PhonySplit",
        {
            {{{4, -1, 64}, {4, -1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"split_dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})},
            {"num_split", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2})},
        },
        {1}, {2});
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, -1, 64}, {4, -1, 64}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// InferDataType copies the input dtype to all dynamic outputs.
TEST_F(PhonySplitTest, phony_split_infer_data_type_success)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry, nullptr);
    auto op_impl = space_registry->GetOpImpl("PhonySplit");
    ASSERT_NE(op_impl, nullptr);
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("PhonySplit").OpName("PhonySplit");
    builder.IONum(1, 2);
    builder.InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND);
    auto context_holder = builder.Build();
    auto* context = context_holder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = op_impl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT16);
}
