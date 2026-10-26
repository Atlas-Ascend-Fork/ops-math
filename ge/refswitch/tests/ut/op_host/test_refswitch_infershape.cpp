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
 * \file test_refswitch_infershape.cpp
 * \brief Unit test for RefSwitch InferShape and InferDataType.
 */
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/tensor.h"
#include "infershape_context_faker.h"
#include "op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

class RefSwitchTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RefSwitch SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "RefSwitch TearDown" << std::endl; }
};

TEST_F(RefSwitchTest, refswitch_registered_success)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("RefSwitch"), nullptr);
}

TEST_F(RefSwitchTest, refswitch_infer_shape_success)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("RefSwitch"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("RefSwitch")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape data_shape = {{32, 16}, {32, 16}};
    gert::StorageShape pred_shape = {{}, {}};

    const gert::StorageFormat format(ge::FORMAT_ND, ge::FORMAT_ND, {});
    gert::Tensor data_tensor(data_shape, format, ge::DT_FLOAT16);
    gert::Tensor pred_tensor(pred_shape, format, ge::DT_BOOL);
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("RefSwitch")
                      .NodeIoNum(2, 2)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensors({&data_tensor, &pred_tensor})
                      .Build();

    ASSERT_NE(holder.GetContext(), nullptr);
    ASSERT_NE(infer_shape_func, nullptr);
    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2U);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 32);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 16);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(1)->GetDimNum(), 2U);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(1)->GetDim(0), 32);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(1)->GetDim(1), 16);
}

TEST_F(RefSwitchTest, refswitch_infer_shape_pred_not_scalar_failed)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("RefSwitch"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("RefSwitch")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape data_shape = {{32}, {32}};
    gert::StorageShape pred_shape = {{2}, {2}};

    const gert::StorageFormat format(ge::FORMAT_ND, ge::FORMAT_ND, {});
    gert::Tensor data_tensor(data_shape, format, ge::DT_FLOAT16);
    gert::Tensor pred_tensor(pred_shape, format, ge::DT_BOOL);
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("RefSwitch")
                      .NodeIoNum(2, 2)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensors({&data_tensor, &pred_tensor})
                      .Build();

    ASSERT_NE(holder.GetContext(), nullptr);
    ASSERT_NE(infer_shape_func, nullptr);
    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(RefSwitchTest, refswitch_infer_datatype_success)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("RefSwitch"), nullptr);
    auto op_impl = space_registry->GetOpImpl("RefSwitch");
    ASSERT_NE(op_impl, nullptr);
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("RefSwitch").OpName("RefSwitch");
    builder.IONum(2, 2);
    builder.InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND);
    auto context_holder = builder.Build();
    auto* context = context_holder.GetContext();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(op_impl->infer_datatype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT16);
}
