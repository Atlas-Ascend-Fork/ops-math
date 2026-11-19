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

namespace ops {
class ShapeUT : public testing::Test {};

TEST_F(ShapeUT, TensorShape)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("Shape"), nullptr);
    auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
                                .GetSpaceRegistry()
                                ->GetOpImpl("Shape")
                                ->infer_shape;
    gert::StorageShape input_shape = {{1, 3, 4, 5}, {1, 3, 4, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Shape")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(*(holder.GetContext()->GetOutputShape(0)), gert::Shape({4}));
}

TEST_F(ShapeUT, VectorShape)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("Shape"), nullptr);
    auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
                                .GetSpaceRegistry()
                                ->GetOpImpl("Shape")
                                ->infer_shape;
    gert::StorageShape input_shape = {{5}, {5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Shape")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(*(holder.GetContext()->GetOutputShape(0)), gert::Shape({1}));
}

TEST_F(ShapeUT, ScalarShape)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("Shape"), nullptr);
    auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
                                .GetSpaceRegistry()
                                ->GetOpImpl("Shape")
                                ->infer_shape;
    gert::StorageShape input_shape = {{}, {}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Shape")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(*(holder.GetContext()->GetOutputShape(0)), gert::Shape({0}));
}

TEST_F(ShapeUT, UnknownRank)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("Shape"), nullptr);
    auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
                                .GetSpaceRegistry()
                                ->GetOpImpl("Shape")
                                ->infer_shape;
    gert::StorageShape input_shape = {{-2}, {-2}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Shape")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(*(holder.GetContext()->GetOutputShape(0)), gert::Shape({-1}));
}

TEST_F(ShapeUT, EmptyInput)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("Shape"), nullptr);
    auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
                                .GetSpaceRegistry()
                                ->GetOpImpl("Shape")
                                ->infer_shape;
    gert::StorageShape input_shape = {{1, 3, 4, 5}, {1, 3, 4, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Shape")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Build();
    // 将输入 0 的值指针置空，模拟空输入（与 infershape_case_executor 的 nullInputIndices 处理一致）
    auto* kernel_ctx = reinterpret_cast<KernelRunContext*>(holder.GetContext());
    ASSERT_NE(kernel_ctx, nullptr);
    ASSERT_LT(0U, kernel_ctx->input_size);
    kernel_ctx->values[0]->data.pointer = nullptr;

    EXPECT_NE(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
}

TEST_F(ShapeUT, InferDataTypeDefault)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Shape"), nullptr);
    auto op_impl = space_registry->GetOpImpl("Shape");
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Shape").OpName("Shape");
    builder.IONum(1, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto context_holder = builder.Build();
    auto* context = context_holder.GetContext();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(op_impl->infer_datatype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
}

TEST_F(ShapeUT, InferDataTypeInt64)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Shape"), nullptr);
    auto op_impl = space_registry->GetOpImpl("Shape");
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Shape").OpName("Shape");
    builder.IONum(1, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.AppendAttr(static_cast<int64_t>(ge::DT_INT64));
    auto context_holder = builder.Build();
    auto* context = context_holder.GetContext();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(op_impl->infer_datatype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT64);
}

TEST_F(ShapeUT, InferDataTypeInvalidValue)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Shape"), nullptr);
    auto op_impl = space_registry->GetOpImpl("Shape");
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Shape").OpName("Shape");
    builder.IONum(1, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.AppendAttr(static_cast<int64_t>(ge::DT_FLOAT));
    auto context_holder = builder.Build();
    auto* context = context_holder.GetContext();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(op_impl->infer_datatype(context), ge::GRAPH_FAILED);
}
} // namespace ops
