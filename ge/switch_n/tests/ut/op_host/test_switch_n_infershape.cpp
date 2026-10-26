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
 * \file test_switch_n_infershape.cpp
 * \brief Unit test for SwitchN InferShape and InferDataType.
 */
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "register/op_impl_registry.h"
#include "infershape_context_faker.h"
#include "op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

class SwitchNTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SwitchN SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SwitchN TearDown" << std::endl; }
};

TEST_F(SwitchNTest, switch_n_registered_success)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("SwitchN"), nullptr);
}

TEST_F(SwitchNTest, switch_n_infer_shape_success)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("SwitchN"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("SwitchN")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape data_shape = {{4, 5}, {4, 5}};
    gert::StorageShape pred_value_shape = {{}, {}};
    const gert::StorageFormat format(ge::FORMAT_ND, ge::FORMAT_ND, {});
    gert::Tensor data_tensor(data_shape, format, ge::DT_FLOAT);
    gert::Tensor pred_tensor(pred_value_shape, format, ge::DT_INT64);

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SwitchN")
                      .IrInstanceNum({1, 1}, {2})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputTensors({&data_tensor, &pred_tensor})
                      .Build();

    ASSERT_NE(holder.GetContext(), nullptr);
    ASSERT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(*holder.GetContext()->GetOutputShape(0), gert::Shape({4, 5}));
    EXPECT_EQ(*holder.GetContext()->GetOutputShape(1), gert::Shape({4, 5}));
}

TEST_F(SwitchNTest, switch_n_infer_datatype_success)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("SwitchN"), nullptr);
    auto op_impl = space_registry->GetOpImpl("SwitchN");
    ASSERT_NE(op_impl, nullptr);
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("SwitchN").OpName("SwitchN");
    builder.IONum(2, 2);
    builder.InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND);
    auto context_holder = builder.Build();
    auto* context = context_holder.GetContext();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(op_impl->infer_datatype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT16);
}

class SwitchNShapeTest : public SwitchNTest, public testing::WithParamInterface<std::vector<int64_t>> {};

TEST_P(SwitchNShapeTest, forwards_shape_to_all_outputs)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto op_impl = registry->GetOpImpl("SwitchN");
    ASSERT_NE(op_impl, nullptr);
    ASSERT_NE(op_impl->infer_shape, nullptr);

    const auto& dims = GetParam();
    gert::StorageShape data_shape;
    data_shape.MutableOriginShape().SetDimNum(dims.size());
    data_shape.MutableStorageShape().SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        data_shape.MutableOriginShape().SetDim(i, dims[i]);
        data_shape.MutableStorageShape().SetDim(i, dims[i]);
    }
    gert::StorageShape pred_shape = {{}, {}};
    const gert::StorageFormat format(ge::FORMAT_ND, ge::FORMAT_ND, {});
    gert::Tensor data_tensor(data_shape, format, ge::DT_FLOAT);
    gert::Tensor pred_tensor(pred_shape, format, ge::DT_INT64);
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SwitchN")
                      .IrInstanceNum({1, 1}, {3})
                      .InputTensors({&data_tensor, &pred_tensor})
                      .Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(op_impl->infer_shape(context), ge::GRAPH_SUCCESS);
    for (size_t i = 0; i < 3U; ++i) {
        const auto* output_shape = context->GetOutputShape(i);
        ASSERT_NE(output_shape, nullptr);
        EXPECT_EQ(*output_shape, data_shape.GetOriginShape());
    }
}

INSTANTIATE_TEST_SUITE_P(StaticDynamicAndScalar, SwitchNShapeTest,
                         testing::Values(std::vector<int64_t>{4, 5}, std::vector<int64_t>{-1, 3},
                                         std::vector<int64_t>{-2}, std::vector<int64_t>{}));
