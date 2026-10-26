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
#include "base/registry/op_impl_space_registry_v2.h"

class ForRTTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "For SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "For TearDown" << std::endl; }
};

// For has 3 fixed inputs (start, limit, delta) + 1 dynamic input = 4 inputs, 1 dynamic output.
// InferShape returns GRAPH_SUCCESS: the output shape is determined at runtime
// by the "body" subgraph, so no concrete shape can be derived at the InferShape stage.
TEST_F(ForRTTest, infer_shape_success)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("For"), nullptr);
    auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
                                .GetSpaceRegistry()
                                ->GetOpImpl("For")
                                ->infer_shape;
    gert::StorageShape scalar_shape = {{}, {}};
    gert::StorageShape input_shape = {{1, 3, 4, 5}, {1, 3, 4, 5}};
    gert::StorageShape output_shape = {{}, {}};

    std::vector<gert::Tensor*> input_tensors = {
        (gert::Tensor*)&scalar_shape,
        (gert::Tensor*)&scalar_shape,
        (gert::Tensor*)&scalar_shape,
        (gert::Tensor*)&input_shape,
    };

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("For")
                      .NodeIoNum(4, 1)
                      .InputTensors(input_tensors)
                      .OutputShapes({&output_shape})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
}

TEST_F(ForRTTest, infer_data_type_success)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry, nullptr);
    auto op_impl = space_registry->GetOpImpl("For");
    ASSERT_NE(op_impl, nullptr);
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("For").OpName("For");
    // 3 fixed inputs + 1 dynamic input = 4 inputs, 1 dynamic output
    builder.IONum(4, 1);
    builder.InputTensorDesc(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto context_holder = builder.Build();
    auto* context = context_holder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = op_impl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// 边界场景：For 为 DYNAMIC_INPUT + DYNAMIC_OUTPUT，多动态输入(2) + 多动态输出(2)
// 总输入 = 3 固定 + 2 动态 = 5，总输出 = 2
// InferShape 仍返回 GRAPH_SUCCESS（输出 shape 由 body 子图运行时决定）
TEST_F(ForRTTest, infer_shape_dynamic_io_success)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("For"), nullptr);
    auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
                                .GetSpaceRegistry()
                                ->GetOpImpl("For")
                                ->infer_shape;
    gert::StorageShape scalar_shape = {{}, {}};
    gert::StorageShape input0_shape = {{1, 3, 4, 5}, {1, 3, 4, 5}};
    gert::StorageShape input1_shape = {{2, 6}, {2, 6}};
    gert::StorageShape output0_shape = {{}, {}};
    gert::StorageShape output1_shape = {{}, {}};

    std::vector<gert::Tensor*> input_tensors = {
        (gert::Tensor*)&scalar_shape, (gert::Tensor*)&scalar_shape, (gert::Tensor*)&scalar_shape,
        (gert::Tensor*)&input0_shape, (gert::Tensor*)&input1_shape,
    };
    std::vector<gert::StorageShape*> output_shapes = {
        &output0_shape,
        &output1_shape,
    };

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("For")
                      .NodeIoNum(5, 2)
                      .InputTensors(input_tensors)
                      .OutputShapes(output_shapes)
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
}

// 边界场景：start (input[0]) 非标量，InferShape 返回 GRAPH_FAILED
TEST_F(ForRTTest, infer_shape_non_scalar_start_fail)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("For"), nullptr);
    auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
                                .GetSpaceRegistry()
                                ->GetOpImpl("For")
                                ->infer_shape;
    gert::StorageShape scalar_shape = {{}, {}};
    gert::StorageShape non_scalar_shape = {{1, 2}, {1, 2}};
    gert::StorageShape input_shape = {{2, 3}, {2, 3}};
    gert::StorageShape output_shape = {{}, {}};

    std::vector<gert::Tensor*> input_tensors = {
        (gert::Tensor*)&non_scalar_shape,
        (gert::Tensor*)&scalar_shape,
        (gert::Tensor*)&scalar_shape,
        (gert::Tensor*)&input_shape,
    };

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("For")
                      .NodeIoNum(4, 1)
                      .InputTensors(input_tensors)
                      .OutputShapes({&output_shape})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

// 边界场景：limit (input[1]) dtype 不是 DT_INT32，InferDataType 返回 GRAPH_FAILED
TEST_F(ForRTTest, infer_data_type_wrong_dtype_limit_fail)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry, nullptr);
    auto op_impl = space_registry->GetOpImpl("For");
    ASSERT_NE(op_impl, nullptr);
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("For").OpName("For");
    builder.IONum(4, 1);
    builder.InputTensorDesc(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto context_holder = builder.Build();
    auto* context = context_holder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = op_impl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// 边界场景：动态输入为标量（shape 为空），验证 InferShape 仍返回 GRAPH_SUCCESS
TEST_F(ForRTTest, infer_shape_scalar_input_success)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("For"), nullptr);
    auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
                                .GetSpaceRegistry()
                                ->GetOpImpl("For")
                                ->infer_shape;
    gert::StorageShape scalar_shape = {{}, {}};
    gert::StorageShape output_shape = {{}, {}};

    std::vector<gert::Tensor*> input_tensors = {
        (gert::Tensor*)&scalar_shape,
        (gert::Tensor*)&scalar_shape,
        (gert::Tensor*)&scalar_shape,
        (gert::Tensor*)&scalar_shape,
    };

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("For")
                      .NodeIoNum(4, 1)
                      .InputTensors(input_tensors)
                      .OutputShapes({&output_shape})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
}
