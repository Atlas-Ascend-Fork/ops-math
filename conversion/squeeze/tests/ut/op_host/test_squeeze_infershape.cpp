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

class SqueezeUT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Squeeze SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "Squeeze TearDown" << std::endl; }
};

TEST_F(SqueezeUT, Squeeze_registered_success)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Squeeze"), nullptr);
}

TEST_F(SqueezeUT, Squeeze_negative_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Squeeze")->infer_shape;
    gert::StorageShape input_shape = {{1, 3, 4, 5}, {1, 3, 4, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Squeeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{-4})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 5);
}

TEST_F(SqueezeUT, Squeeze_one_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Squeeze")->infer_shape;
    gert::StorageShape input_shape = {{3, 1, 4, 5}, {3, 1, 4, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Squeeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{1})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 5);
}

TEST_F(SqueezeUT, Squeeze_two_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Squeeze")->infer_shape;
    gert::StorageShape input_shape = {{1, 3, 4, 5, 1}, {1, 3, 4, 5, 1}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Squeeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{0, 4})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 5);
}

TEST_F(SqueezeUT, Squeeze_three_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Squeeze")->infer_shape;
    gert::StorageShape input_shape = {{3, 4, 1, 5, 1, 1}, {3, 4, 1, 5, 1, 1}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Squeeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{2, 4, 5})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 5);
}

TEST_F(SqueezeUT, Squeeze_unsorted_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Squeeze")->infer_shape;
    gert::StorageShape input_shape = {{3, 4, 1, 5, 1, 1}, {3, 4, 1, 5, 1, 1}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Squeeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{5, 4, 2})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 5);
}

TEST_F(SqueezeUT, Squeeze_empty_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Squeeze")->infer_shape;
    gert::StorageShape input_shape = {{3, 4, 1, 5, 1, 1}, {3, 4, 1, 5, 1, 1}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Squeeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 5);
}

// empty axis and no 1-dims in x: all dims are kept as-is
TEST_F(SqueezeUT, Squeeze_no_1dims_passthrough)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Squeeze")->infer_shape;
    gert::StorageShape input_shape = {{2, 3}, {2, 3}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Squeeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 3);
}

// empty axis and all dims of x are 1: the output is a scalar
TEST_F(SqueezeUT, Squeeze_all_1dims_to_scalar)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Squeeze")->infer_shape;
    gert::StorageShape input_shape = {{1, 1, 1}, {1, 1, 1}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Squeeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 0);
}

// no attr is set: the empty-list default of the axis attr squeezes all 1-dims
TEST_F(SqueezeUT, Squeeze_no_attr_default)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Squeeze")->infer_shape;
    gert::StorageShape input_shape = {{2, 1, 3}, {2, 1, 3}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Squeeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 3);
}

// empty axis with a 0 dim and a 1 dim mixed in x: the 0 dim is kept, the 1 dim is squeezed
TEST_F(SqueezeUT, Squeeze_zero_dim_mixed)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Squeeze")->infer_shape;
    gert::StorageShape input_shape = {{2, 0, 1}, {2, 0, 1}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Squeeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 0);
}

// x is not data dependent, no other input exists
TEST_F(SqueezeUT, Squeeze_data_dependency)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Squeeze"), nullptr);
    EXPECT_FALSE(space_registry->GetOpImpl("Squeeze")->IsInputDataDependency(0));
}

TEST_F(SqueezeUT, Squeeze_out_of_range_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Squeeze")->infer_shape;
    gert::StorageShape input_shape = {{3, 4, 5, 1}, {3, 4, 5, 1}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Squeeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{5})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(SqueezeUT, Squeeze_repetive_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Squeeze")->infer_shape;
    gert::StorageShape input_shape = {{3, 4, 5, 1, 1}, {3, 4, 5, 1, 1}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Squeeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{3, 3, 4})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 5);
}

TEST_F(SqueezeUT, Squeeze_not_dim1_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Squeeze")->infer_shape;
    gert::StorageShape input_shape = {{3, 4, 5, 1, 1}, {3, 4, 5, 1, 1}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Squeeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{3, 1})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(SqueezeUT, Squeeze_unknown_rank)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Squeeze")->infer_shape;
    gert::StorageShape input_shape = {{-2}, {-2}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Squeeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), -2);
}

TEST_F(SqueezeUT, Squeeze_empty_axis_with_unknown_dim)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Squeeze")->infer_shape;
    gert::StorageShape input_shape = {{-1, 1, -1, 1, 5}, {-1, 1, -1, 1, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Squeeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), -2);
}

TEST_F(SqueezeUT, Squeeze_one_axis_with_unknown_dim)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Squeeze")->infer_shape;
    gert::StorageShape input_shape = {{-1, 1, 5}, {-1, 1, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Squeeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{1})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), -1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 5);
}

TEST_F(SqueezeUT, Squeeze_axis_point_to_unknown_dim)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Squeeze")->infer_shape;
    gert::StorageShape input_shape = {{-1, 1, 5}, {-1, 1, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Squeeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{0})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 5);
}

TEST_F(SqueezeUT, Squeeze_negative_axis_point_to_unknown_dim)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Squeeze")->infer_shape;
    gert::StorageShape input_shape = {{-1, 1, 5}, {-1, 1, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Squeeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{-3})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 5);
}

TEST_F(SqueezeUT, Squeeze_infer_datatype)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Squeeze"), nullptr);
    auto op_impl = space_registry->GetOpImpl("Squeeze");
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Squeeze").OpName("Squeeze");
    builder.IONum(1, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto context_holder = builder.Build();
    auto* context = context_holder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = op_impl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    // output dtype is same as input
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}
