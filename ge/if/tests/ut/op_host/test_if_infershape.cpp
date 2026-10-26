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

class IfRTTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "If SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "If TearDown" << std::endl; }
};

// InferShape for If returns GRAPH_SUCCESS unconditionally.
// The output shape is determined at runtime by the subgraph selected through `cond`.
TEST_F(IfRTTest, infer_shape_success)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("If"), nullptr);
    auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
                                .GetSpaceRegistry()
                                ->GetOpImpl("If")
                                ->infer_shape;
    gert::StorageShape cond_shape = {{}, {}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("If")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&cond_shape})
                      .OutputShapes({&output_shape})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
}

TEST_F(IfRTTest, infer_data_type_success)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry, nullptr);
    auto op_impl = space_registry->GetOpImpl("If");
    ASSERT_NE(op_impl, nullptr);
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("If").OpName("If");
    builder.IONum(1, 1);
    builder.InputTensorDesc(0, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto context_holder = builder.Build();
    auto* context = context_holder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = op_impl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// 边界场景：If 为 DYNAMIC_INPUT + DYNAMIC_OUTPUT，cond + 2 动态输入 + 2 动态输出
// InferShape 仍返回 GRAPH_SUCCESS（输出 shape 由子图运行时决定，RT1.0 无校验）
TEST_F(IfRTTest, infer_shape_dynamic_io_success)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("If"), nullptr);
    auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
                                .GetSpaceRegistry()
                                ->GetOpImpl("If")
                                ->infer_shape;
    gert::StorageShape cond_shape = {{}, {}};
    gert::StorageShape input0_shape = {{1, 3, 4, 5}, {1, 3, 4, 5}};
    gert::StorageShape input1_shape = {{2, 6}, {2, 6}};
    gert::StorageShape output0_shape = {{}, {}};
    gert::StorageShape output1_shape = {{}, {}};

    std::vector<gert::Tensor*> input_tensors = {
        (gert::Tensor*)&cond_shape,
        (gert::Tensor*)&input0_shape,
        (gert::Tensor*)&input1_shape,
    };
    std::vector<gert::StorageShape*> output_shapes = {
        &output0_shape,
        &output1_shape,
    };

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("If")
                      .NodeIoNum(3, 2)
                      .InputTensors(input_tensors)
                      .OutputShapes(output_shapes)
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
}
