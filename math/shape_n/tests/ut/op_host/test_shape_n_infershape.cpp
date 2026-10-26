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
#include "register/op_impl_registry.h"
#include "infershape_context_faker.h"
#include "op_infer_datatype_context_builder.h"
#include "exe_graph/runtime/kernel_run_context.h"
#include "base/registry/op_impl_space_registry_v2.h"

class SHAPE_SHAPEN_UT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SHAPE_SHAPEN_UT SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SHAPE_SHAPEN_UT TearDown" << std::endl; }
};

TEST_F(SHAPE_SHAPEN_UT, TensorShapeRT2)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("ShapeN"), nullptr);
    auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
                                .GetSpaceRegistry()
                                ->GetOpImpl("ShapeN")
                                ->infer_shape;
    gert::StorageShape input_shape_0 = {{1, 3, 4, 5}, {1, 3, 4, 5}};
    gert::StorageShape input_shape_1 = {{1, 3, 4}, {1, 3, 4}};
    gert::StorageShape output_shape_0 = {{}, {}};
    gert::StorageShape output_shape_1 = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("ShapeN")
                      .NodeIoNum(2, 2)
                      .InputTensors({(gert::Tensor*)&input_shape_0, (gert::Tensor*)&input_shape_1})
                      .OutputShapes({&output_shape_0, &output_shape_1})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(*(holder.GetContext()->GetOutputShape(0)), gert::Shape({4}));
    EXPECT_EQ(*(holder.GetContext()->GetOutputShape(1)), gert::Shape({3}));
}

TEST_F(SHAPE_SHAPEN_UT, VectorShapeNRT2)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("ShapeN"), nullptr);
    auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
                                .GetSpaceRegistry()
                                ->GetOpImpl("ShapeN")
                                ->infer_shape;
    gert::StorageShape input_shape_0 = {{5}, {5}};
    gert::StorageShape input_shape_1 = {{4}, {4}};
    gert::StorageShape output_shape_0 = {{}, {}};
    gert::StorageShape output_shape_1 = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("ShapeN")
                      .NodeIoNum(2, 2)
                      .InputTensors({(gert::Tensor*)&input_shape_0, (gert::Tensor*)&input_shape_1})
                      .OutputShapes({&output_shape_0, &output_shape_1})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(*(holder.GetContext()->GetOutputShape(0)), gert::Shape({1}));
    EXPECT_EQ(*(holder.GetContext()->GetOutputShape(1)), gert::Shape({1}));
}

TEST_F(SHAPE_SHAPEN_UT, ScalarShapeN)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("ShapeN"), nullptr);
    auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
                                .GetSpaceRegistry()
                                ->GetOpImpl("ShapeN")
                                ->infer_shape;
    gert::StorageShape input_shape_0 = {{}, {}};
    gert::StorageShape input_shape_1 = {{}, {}};
    gert::StorageShape output_shape_0 = {{}, {}};
    gert::StorageShape output_shape_1 = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("ShapeN")
                      .NodeIoNum(2, 2)
                      .InputTensors({(gert::Tensor*)&input_shape_0, (gert::Tensor*)&input_shape_1})
                      .OutputShapes({&output_shape_0, &output_shape_1})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(*(holder.GetContext()->GetOutputShape(0)), gert::Shape({0}));
    EXPECT_EQ(*(holder.GetContext()->GetOutputShape(1)), gert::Shape({0}));
}

TEST_F(SHAPE_SHAPEN_UT, UnknownRankShapeN)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("ShapeN"), nullptr);
    auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
                                .GetSpaceRegistry()
                                ->GetOpImpl("ShapeN")
                                ->infer_shape;
    gert::StorageShape input_shape_0 = {{-2}, {-2}};
    gert::StorageShape input_shape_1 = {{1, 3, 4}, {1, 3, 4}};
    gert::StorageShape output_shape_0 = {{}, {}};
    gert::StorageShape output_shape_1 = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("ShapeN")
                      .NodeIoNum(2, 2)
                      .InputTensors({(gert::Tensor*)&input_shape_0, (gert::Tensor*)&input_shape_1})
                      .OutputShapes({&output_shape_0, &output_shape_1})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(*(holder.GetContext()->GetOutputShape(0)), gert::Shape({-1}));
    EXPECT_EQ(*(holder.GetContext()->GetOutputShape(1)), gert::Shape({3}));
}

TEST_F(SHAPE_SHAPEN_UT, EmptyInput)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("ShapeN"), nullptr);
    auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
                                .GetSpaceRegistry()
                                ->GetOpImpl("ShapeN")
                                ->infer_shape;
    gert::StorageShape input_shape_0 = {{1, 3, 4, 5}, {1, 3, 4, 5}};
    gert::StorageShape input_shape_1 = {{1, 3, 4}, {1, 3, 4}};
    gert::StorageShape output_shape_0 = {{}, {}};
    gert::StorageShape output_shape_1 = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("ShapeN")
                      .NodeIoNum(2, 2)
                      .InputTensors({(gert::Tensor*)&input_shape_0, (gert::Tensor*)&input_shape_1})
                      .OutputShapes({&output_shape_0, &output_shape_1})
                      .Build();
    // 将两个输入的值指针置空，模拟空输入（与 infershape_case_executor 的 nullInputIndices 处理一致）
    auto* kernel_ctx = reinterpret_cast<KernelRunContext*>(holder.GetContext());
    ASSERT_NE(kernel_ctx, nullptr);
    ASSERT_LT(1U, kernel_ctx->input_size);
    kernel_ctx->values[0]->data.pointer = nullptr;
    kernel_ctx->values[1]->data.pointer = nullptr;

    EXPECT_NE(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
}

TEST_F(SHAPE_SHAPEN_UT, InferDataTypeDefault)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("ShapeN"), nullptr);
    auto op_impl = space_registry->GetOpImpl("ShapeN");
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("ShapeN").OpName("ShapeN");
    builder.IONum(2, 2);
    builder.InputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND);
    auto context_holder = builder.Build();
    auto* context = context_holder.GetContext();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(op_impl->infer_datatype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT32);
}

TEST_F(SHAPE_SHAPEN_UT, InferDataTypeInt64)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("ShapeN"), nullptr);
    auto op_impl = space_registry->GetOpImpl("ShapeN");
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("ShapeN").OpName("ShapeN");
    builder.IONum(2, 2);
    builder.InputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.AppendAttr(static_cast<int64_t>(ge::DT_INT64));
    auto context_holder = builder.Build();
    auto* context = context_holder.GetContext();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(op_impl->infer_datatype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT64);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT64);
}

TEST_F(SHAPE_SHAPEN_UT, InferDataTypeInvalidValue)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("ShapeN"), nullptr);
    auto op_impl = space_registry->GetOpImpl("ShapeN");
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("ShapeN").OpName("ShapeN");
    builder.IONum(2, 2);
    builder.InputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.AppendAttr(static_cast<int64_t>(ge::DT_FLOAT));
    auto context_holder = builder.Build();
    auto* context = context_holder.GetContext();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(op_impl->infer_datatype(context), ge::GRAPH_FAILED);
}
