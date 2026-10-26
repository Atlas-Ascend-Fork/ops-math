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

class GatherShapesUT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GatherShapesUT Proto Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "GatherShapesUT Proto Test TearDown" << std::endl; }
};

TEST_F(GatherShapesUT, gathershapes_infershape_test)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("GatherShapes"), nullptr);
    auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
                                .GetSpaceRegistry()
                                ->GetOpImpl("GatherShapes")
                                ->infer_shape;
    gert::StorageShape x_shape_0 = {{1, 2, 3}, {1, 2, 3}};
    gert::StorageShape x_shape_1 = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape output_shape = {{}, {}};
    std::vector<std::vector<int64_t>> axes = {{0, 0}, {0, 1}, {1, 1}, {1, 2}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("GatherShapes")
                      .NodeIoNum(2, 1)
                      .InputTensors({(gert::Tensor*)&x_shape_0, (gert::Tensor*)&x_shape_1})
                      .OutputShapes({&output_shape})
                      .Attr("axes", axes)
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 1U);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 4);
}

TEST_F(GatherShapesUT, gathershapes_infershape_empty_axes_test)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("GatherShapes"), nullptr);
    auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
                                .GetSpaceRegistry()
                                ->GetOpImpl("GatherShapes")
                                ->infer_shape;
    gert::StorageShape x_shape_0 = {{1, 2, 3}, {1, 2, 3}};
    gert::StorageShape output_shape = {{}, {}};
    std::vector<std::vector<int64_t>> axes = {};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("GatherShapes")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&x_shape_0})
                      .OutputShapes({&output_shape})
                      .Attr("axes", axes)
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(GatherShapesUT, gathershapes_infershape_invalid_input_index_test)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("GatherShapes"), nullptr);
    auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
                                .GetSpaceRegistry()
                                ->GetOpImpl("GatherShapes")
                                ->infer_shape;
    gert::StorageShape x_shape_0 = {{1, 2, 3}, {1, 2, 3}};
    gert::StorageShape x_shape_1 = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape output_shape = {{}, {}};
    std::vector<std::vector<int64_t>> axes = {{3, 0}, {0, 1}, {1, 1}, {1, 2}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("GatherShapes")
                      .NodeIoNum(2, 1)
                      .InputTensors({(gert::Tensor*)&x_shape_0, (gert::Tensor*)&x_shape_1})
                      .OutputShapes({&output_shape})
                      .Attr("axes", axes)
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(GatherShapesUT, gathershapes_infershape_invalid_dim_index_test)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("GatherShapes"), nullptr);
    auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
                                .GetSpaceRegistry()
                                ->GetOpImpl("GatherShapes")
                                ->infer_shape;
    gert::StorageShape x_shape_0 = {{1, 2, 3}, {1, 2, 3}};
    gert::StorageShape output_shape = {{}, {}};
    std::vector<std::vector<int64_t>> axes = {{0, 5}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("GatherShapes")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&x_shape_0})
                      .OutputShapes({&output_shape})
                      .Attr("axes", axes)
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(GatherShapesUT, gathershapes_infershape_invalid_pair_size_test)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("GatherShapes"), nullptr);
    auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
                                .GetSpaceRegistry()
                                ->GetOpImpl("GatherShapes")
                                ->infer_shape;
    gert::StorageShape x_shape_0 = {{1, 2, 3}, {1, 2, 3}};
    gert::StorageShape output_shape = {{}, {}};
    std::vector<std::vector<int64_t>> axes = {{0}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("GatherShapes")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&x_shape_0})
                      .OutputShapes({&output_shape})
                      .Attr("axes", axes)
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(GatherShapesUT, gathershapes_infershape_unknown_rank_skip_check_test)
{
    ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("GatherShapes"), nullptr);
    auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance()
                                .GetSpaceRegistry()
                                ->GetOpImpl("GatherShapes")
                                ->infer_shape;
    gert::StorageShape x_shape_0 = {{-2}, {-2}};
    gert::StorageShape output_shape = {{}, {}};
    // 与 V1 行为对齐：存在未知秩输入时跳过 axes 越界检查（此处 axes[0][0]=3 越界，但输入为未知秩）
    std::vector<std::vector<int64_t>> axes = {{3, 0}, {0, 1}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("GatherShapes")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&x_shape_0})
                      .OutputShapes({&output_shape})
                      .Attr("axes", axes)
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 1U);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 2);
}

TEST_F(GatherShapesUT, InferDataTypeDefault)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("GatherShapes"), nullptr);
    auto op_impl = space_registry->GetOpImpl("GatherShapes");
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("GatherShapes").OpName("GatherShapes");
    builder.IONum(1, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.AppendAttr(std::vector<std::vector<int64_t>>{{0, 0}});
    auto context_holder = builder.Build();
    auto* context = context_holder.GetContext();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(op_impl->infer_datatype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
}

TEST_F(GatherShapesUT, InferDataTypeInt64)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("GatherShapes"), nullptr);
    auto op_impl = space_registry->GetOpImpl("GatherShapes");
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("GatherShapes").OpName("GatherShapes");
    builder.IONum(1, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.AppendAttr(std::vector<std::vector<int64_t>>{{0, 0}});
    builder.AppendAttr(static_cast<int64_t>(ge::DT_INT64));
    auto context_holder = builder.Build();
    auto* context = context_holder.GetContext();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(op_impl->infer_datatype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT64);
}

TEST_F(GatherShapesUT, InferDataTypeInvalidValue)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("GatherShapes"), nullptr);
    auto op_impl = space_registry->GetOpImpl("GatherShapes");
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("GatherShapes").OpName("GatherShapes");
    builder.IONum(1, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.AppendAttr(std::vector<std::vector<int64_t>>{{0, 0}});
    builder.AppendAttr(static_cast<int64_t>(ge::DT_FLOAT));
    auto context_holder = builder.Build();
    auto* context = context_holder.GetContext();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(op_impl->infer_datatype(context), ge::GRAPH_FAILED);
}
