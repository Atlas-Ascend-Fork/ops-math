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
 * \file test_merge_infershape.cpp
 * \brief Unit test for Merge InferShape and InferDataType.
 */
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/tensor.h"
#include "infershape_context_faker.h"
#include "op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

class MergeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Merge SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "Merge TearDown" << std::endl; }

    void CheckDataTypes(const std::vector<ge::DataType>& input_types, ge::graphStatus expected_status)
    {
        auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
        ASSERT_NE(registry, nullptr);
        auto impl = registry->GetOpImpl("Merge");
        ASSERT_NE(impl, nullptr);
        ASSERT_NE(impl->infer_datatype, nullptr);
        gert::OpInferDataTypeContextBuilder builder;
        builder.OpType("Merge").OpName("Merge");
        builder.IOInstanceNum({static_cast<uint32_t>(input_types.size())}, {1, 1});
        for (size_t i = 0; i < input_types.size(); ++i) {
            builder.InputTensorDesc(i, input_types[i], ge::FORMAT_ND, ge::FORMAT_ND);
        }
        builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND);
        auto holder = builder.Build();
        auto* context = holder.GetContext();
        ASSERT_NE(context, nullptr);
        ASSERT_EQ(context->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
        ASSERT_EQ(context->SetOutputDataType(1, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
        ASSERT_EQ(impl->infer_datatype(context), expected_status);
        if (expected_status == ge::GRAPH_SUCCESS) {
            ASSERT_FALSE(input_types.empty());
            EXPECT_EQ(context->GetOutputDataType(0), input_types[0]);
            EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT32);
        } else {
            EXPECT_EQ(context->GetOutputDataType(0), ge::DT_UNDEFINED);
            EXPECT_EQ(context->GetOutputDataType(1), ge::DT_UNDEFINED);
        }
    }

    void CheckShapes(const gert::StorageShape& x0_shape, const gert::StorageShape& x1_shape,
                     const gert::Shape& expected)
    {
        auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
        ASSERT_NE(registry, nullptr);
        auto impl = registry->GetOpImpl("Merge");
        ASSERT_NE(impl, nullptr);
        ASSERT_NE(impl->infer_shape, nullptr);
        const gert::StorageFormat format(ge::FORMAT_ND, ge::FORMAT_ND, {});
        gert::Tensor x0(x0_shape, format, ge::DT_FLOAT16);
        gert::Tensor x1(x1_shape, format, ge::DT_FLOAT16);
        auto holder = gert::InferShapeContextFaker()
                          .SetOpType("Merge")
                          .IrInstanceNum({2}, {1, 1})
                          .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                          .InputTensors({&x0, &x1})
                          .Build();
        auto* context = holder.GetContext();
        ASSERT_NE(context, nullptr);
        for (size_t i = 0; i < 2U; ++i) {
            auto* shape = context->GetOutputShape(i);
            ASSERT_NE(shape, nullptr);
            shape->SetDimNum(1);
            shape->SetDim(0, 777);
        }
        ASSERT_EQ(impl->infer_shape(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(*context->GetOutputShape(0), expected);
        EXPECT_EQ(context->GetOutputShape(1)->GetDimNum(), 0U);
    }
};

TEST_F(MergeTest, merge_registered_success)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Merge"), nullptr);
}

TEST_F(MergeTest, merge_infer_shape_known)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Merge"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Merge")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x0 = {{32}, {32}};
    gert::StorageShape x1 = {{32}, {32}};

    const gert::StorageFormat format(ge::FORMAT_ND, ge::FORMAT_ND, {});
    gert::Tensor x0_tensor(x0, format, ge::DT_FLOAT16);
    gert::Tensor x1_tensor(x1, format, ge::DT_FLOAT16);
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Merge")
                      .IrInstanceNum({2}, {1, 1})
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensors({&x0_tensor, &x1_tensor})
                      .Build();

    ASSERT_NE(holder.GetContext(), nullptr);
    ASSERT_NE(infer_shape_func, nullptr);
    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    auto* out_y = holder.GetContext()->GetOutputShape(0);
    EXPECT_EQ(out_y->GetDimNum(), 1U);
    EXPECT_EQ(out_y->GetDim(0), 32);
    auto* out_v = holder.GetContext()->GetOutputShape(1);
    EXPECT_EQ(out_v->GetDimNum(), 0U);
}

TEST_F(MergeTest, merge_infer_shape_unknown_dim)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Merge"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Merge")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x0 = {{-1}, {-1}};
    gert::StorageShape x1 = {{64}, {64}};

    const gert::StorageFormat format(ge::FORMAT_ND, ge::FORMAT_ND, {});
    gert::Tensor x0_tensor(x0, format, ge::DT_FLOAT16);
    gert::Tensor x1_tensor(x1, format, ge::DT_FLOAT16);
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Merge")
                      .IrInstanceNum({2}, {1, 1})
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensors({&x0_tensor, &x1_tensor})
                      .Build();

    ASSERT_NE(holder.GetContext(), nullptr);
    ASSERT_NE(infer_shape_func, nullptr);
    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    auto* out_y = holder.GetContext()->GetOutputShape(0);
    EXPECT_EQ(out_y->GetDimNum(), 1U);
    EXPECT_EQ(out_y->GetDim(0), -1);
}

TEST_F(MergeTest, merge_infer_shape_rank_mismatch_unknown_rank)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Merge"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Merge")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x0 = {{32}, {32}};
    gert::StorageShape x1 = {{4, 4}, {4, 4}};

    const gert::StorageFormat format(ge::FORMAT_ND, ge::FORMAT_ND, {});
    gert::Tensor x0_tensor(x0, format, ge::DT_FLOAT16);
    gert::Tensor x1_tensor(x1, format, ge::DT_FLOAT16);
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Merge")
                      .IrInstanceNum({2}, {1, 1})
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensors({&x0_tensor, &x1_tensor})
                      .Build();

    ASSERT_NE(holder.GetContext(), nullptr);
    ASSERT_NE(infer_shape_func, nullptr);
    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    auto* out_y = holder.GetContext()->GetOutputShape(0);
    EXPECT_EQ(out_y->GetDimNum(), 1U);
    EXPECT_EQ(out_y->GetDim(0), -2);
}

TEST_F(MergeTest, merge_infer_datatype)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Merge"), nullptr);
    auto op_impl = space_registry->GetOpImpl("Merge");
    ASSERT_NE(op_impl, nullptr);
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Merge").OpName("Merge");
    builder.IOInstanceNum({2}, {1, 1});
    builder.InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND);
    auto context_holder = builder.Build();
    auto* context = context_holder.GetContext();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(op_impl->infer_datatype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT32);
}

TEST_F(MergeTest, merge_infer_shape_scalar) { CheckShapes({{}, {}}, {{}, {}}, {}); }

TEST_F(MergeTest, merge_infer_shape_unknown_rank) { CheckShapes({{-2}, {-2}}, {{2, 3}, {2, 3}}, {-2}); }

TEST_F(MergeTest, merge_infer_shape_different_dims) { CheckShapes({{2, 3}, {2, 3}}, {{4, 3}, {4, 3}}, {-1, 3}); }

TEST_F(MergeTest, merge_infer_datatype_mixed_input_types_failed)
{
    CheckDataTypes({ge::DT_FLOAT16, ge::DT_FLOAT}, ge::GRAPH_FAILED);
}

TEST_F(MergeTest, merge_infer_datatype_later_input_type_mismatch_failed)
{
    CheckDataTypes({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32}, ge::GRAPH_FAILED);
}

TEST_F(MergeTest, merge_infer_datatype_three_matching_inputs)
{
    CheckDataTypes({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}, ge::GRAPH_SUCCESS);
}

TEST_F(MergeTest, merge_infer_datatype_single_input) { CheckDataTypes({ge::DT_FLOAT}, ge::GRAPH_SUCCESS); }
