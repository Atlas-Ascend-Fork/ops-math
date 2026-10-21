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
 * \file test_trans_shape_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

class TransShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TransShapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TransShapeTest TearDown" << std::endl; }
};

// input {2,3,4} fp16, outShape={6,4}: output shape equals outShape.
TEST_F(TransShapeTest, trans_shape_infer_shape_success)
{
    gert::InfershapeContextPara infershapeContextPara(
        "TransShape",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"outShape", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({6, 4})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{6, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// input {2,3}, outShape={3,2}: shape changed to outShape regardless of input dims.
TEST_F(TransShapeTest, trans_shape_infer_shape_reorder)
{
    gert::InfershapeContextPara infershapeContextPara(
        "TransShape",
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"outShape", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({3, 2})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// empty outShape (default): output is a scalar (dim_num=0).
TEST_F(TransShapeTest, trans_shape_infer_shape_empty_out_shape)
{
    gert::InfershapeContextPara infershapeContextPara(
        "TransShape",
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"outShape", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// unknown rank input {-2} with outShape={2,3}: output still equals outShape.
TEST_F(TransShapeTest, trans_shape_infer_shape_unknown_rank_input)
{
    gert::InfershapeContextPara infershapeContextPara(
        "TransShape",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"outShape", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 3})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// InferDataType copies the input dtype to the output.
TEST_F(TransShapeTest, trans_shape_infer_data_type_success)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry, nullptr);
    auto op_impl = space_registry->GetOpImpl("TransShape");
    ASSERT_NE(op_impl, nullptr);
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("TransShape").OpName("TransShape");
    builder.IONum(1, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto context_holder = builder.Build();
    auto* context = context_holder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = op_impl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}
