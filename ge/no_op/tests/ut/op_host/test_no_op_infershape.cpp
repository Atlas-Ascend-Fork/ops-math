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

class NoOpRTTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "NoOp SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "NoOp TearDown" << std::endl; }
};

// NoOp has 0 inputs and 0 outputs. It does nothing — only serves as a
// placeholder for control edges. InferShape returns GRAPH_SUCCESS.
TEST_F(NoOpRTTest, infer_shape_success)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("NoOp"), nullptr);
    auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
                                .GetSpaceRegistry()
                                ->GetOpImpl("NoOp")
                                ->infer_shape;

    auto holder = gert::InferShapeContextFaker().SetOpType("NoOp").NodeIoNum(0, 0).Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
}

TEST_F(NoOpRTTest, infer_data_type_success)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry, nullptr);
    auto op_impl = space_registry->GetOpImpl("NoOp");
    ASSERT_NE(op_impl, nullptr);
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    // NoOp has 0 inputs/outputs, but OpInferDataTypeContextBuilder does not support
    // IONum(0, 0) (Build() returns nullptr). Use IONum(1, 1) with a dummy descriptor
    // to obtain a non-null context — NoOp's InferDataType ignores the context entirely.
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("NoOp").OpName("NoOp");
    builder.IONum(1, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto context_holder = builder.Build();
    auto* context = context_holder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = op_impl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
