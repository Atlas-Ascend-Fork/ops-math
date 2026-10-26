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
#include <iostream>
#include "register/op_impl_registry.h"
#include "infershape_context_faker.h"
#include "op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

class SwitchByIndexTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SwitchByIndex SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SwitchByIndex TearDown" << std::endl; }
};

TEST_F(SwitchByIndexTest, switch_by_index_registered_success)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("SwitchByIndex"), nullptr);
}

TEST_F(SwitchByIndexTest, switch_by_index_infer_shape_success)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("SwitchByIndex"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("SwitchByIndex")->infer_shape;
    gert::StorageShape x1_shape = {{1}, {1}};
    gert::StorageShape x2_shape = {{10}, {10}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SwitchByIndex")
                      .NodeIoNum(2, 0)
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensors({(gert::Tensor*)&x1_shape, (gert::Tensor*)&x2_shape})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
}

TEST_F(SwitchByIndexTest, switch_by_index_infer_datatype_success)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("SwitchByIndex"), nullptr);
    auto op_impl = space_registry->GetOpImpl("SwitchByIndex");
    ASSERT_NE(op_impl, nullptr);
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    // OpInferDataTypeContextBuilder requires at least one output to build a context; SwitchByIndex has no
    // output and its InferDataType does not touch outputs, so a dummy output desc is used here.
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("SwitchByIndex").OpName("SwitchByIndex");
    builder.IONum(2, 1);
    builder.InputTensorDesc(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto context_holder = builder.Build();
    auto* context = context_holder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = op_impl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
