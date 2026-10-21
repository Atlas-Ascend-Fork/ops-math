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
#include <iostream>

class UnsqueezeUT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Unsqueeze SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "Unsqueeze TearDown" << std::endl; }
};

TEST_F(UnsqueezeUT, Unsqueeze_registered_success)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Unsqueeze"), nullptr);
}

TEST_F(UnsqueezeUT, Unsqueeze_negative_axes)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Unsqueeze")->infer_shape;

    gert::StorageShape input_shape = {{1, 3, 2, 5}, {1, 3, 2, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Unsqueeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axes", std::vector<int64_t>{-5})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 5);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(3), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(4), 5);
}

TEST_F(UnsqueezeUT, Unsqueeze_one_axes)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Unsqueeze")->infer_shape;

    gert::StorageShape input_shape = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Unsqueeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axes", std::vector<int64_t>{1})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(3), 5);
}

TEST_F(UnsqueezeUT, Unsqueeze_two_axes)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Unsqueeze")->infer_shape;

    gert::StorageShape input_shape = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Unsqueeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axes", std::vector<int64_t>{0, 4})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 5);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(3), 5);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(4), 1);
}

TEST_F(UnsqueezeUT, Unsqueeze_three_axes)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Unsqueeze")->infer_shape;

    gert::StorageShape input_shape = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Unsqueeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axes", std::vector<int64_t>{2, 4, 5})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 6);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(3), 5);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(4), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(5), 1);
}

TEST_F(UnsqueezeUT, Unsqueeze_unsorted_axes)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Unsqueeze")->infer_shape;

    gert::StorageShape input_shape = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Unsqueeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axes", std::vector<int64_t>{5, 4, 2})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 6);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(3), 5);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(4), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(5), 1);
}

TEST_F(UnsqueezeUT, Unsqueeze_empty_axes)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Unsqueeze")->infer_shape;

    gert::StorageShape input_shape = {{3, 1, 4, 5}, {3, 1, 4, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Unsqueeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axes", std::vector<int64_t>{})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(3), 5);
}

// no attr is set: the empty-list default of the axes attr inserts no dim, the output keeps the
// input shape
TEST_F(UnsqueezeUT, Unsqueeze_no_attr_default)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Unsqueeze")->infer_shape;
    gert::StorageShape input_shape = {{2, 3}, {2, 3}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Unsqueeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 3);
}

// x is not data dependent, no other input exists
TEST_F(UnsqueezeUT, Unsqueeze_data_dependency)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Unsqueeze"), nullptr);
    EXPECT_FALSE(space_registry->GetOpImpl("Unsqueeze")->IsInputDataDependency(0));
}

TEST_F(UnsqueezeUT, Unsqueeze_out_of_range_axes)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Unsqueeze")->infer_shape;

    gert::StorageShape input_shape = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Unsqueeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axes", std::vector<int64_t>{4})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

// out_dim_size is 4 and the valid axis range is [-4, 4), a more negative axis is out of range
TEST_F(UnsqueezeUT, Unsqueeze_negative_out_of_range_axes)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Unsqueeze")->infer_shape;

    gert::StorageShape input_shape = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Unsqueeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axes", std::vector<int64_t>{-5})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(UnsqueezeUT, Unsqueeze_repetive_axes)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Unsqueeze")->infer_shape;

    gert::StorageShape input_shape = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Unsqueeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axes", std::vector<int64_t>{1, 1})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(UnsqueezeUT, Unsqueeze_larger_than_kMaxDimNum_axes)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Unsqueeze")->infer_shape;
    gert::StorageShape input_shape = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape output_shape = {{}, {}};

    // 3 + 23 = 26 > kMaxDimNum(25), expect FAILED; axes are pairwise distinct so the failure can
    // only come from the dim-num limit
    std::vector<int64_t> large_axes;
    for (int64_t idx = 0; idx < 23; ++idx) {
        large_axes.push_back(idx);
    }
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Unsqueeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axes", large_axes)
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(UnsqueezeUT, Unsqueeze_unknown_rank)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Unsqueeze")->infer_shape;
    gert::StorageShape input_shape = {{-2}, {-2}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Unsqueeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axes", std::vector<int64_t>{0})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), -2);
}

TEST_F(UnsqueezeUT, Unsqueeze_one_axes_with_unknown_dim)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Unsqueeze")->infer_shape;
    gert::StorageShape input_shape = {{-1, 5}, {-1, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Unsqueeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axes", std::vector<int64_t>{1})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), -1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 5);
}

// a scalar x is broadcast to a 1-elem shape, the common way of feeding a scalar into
// element-wise ops
TEST_F(UnsqueezeUT, Unsqueeze_scalar_x)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Unsqueeze")->infer_shape;

    gert::StorageShape input_shape = {{}, {}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Unsqueeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axes", std::vector<int64_t>{0})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 1);
}

// a negative axis with an unknown dim in x: real axis is 0 after normalization, the unknown
// dim is copied as-is
TEST_F(UnsqueezeUT, Unsqueeze_negative_axes_with_unknown_dim)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("Unsqueeze")->infer_shape;

    gert::StorageShape input_shape = {{-1, 5}, {-1, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Unsqueeze")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axes", std::vector<int64_t>{-3})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), -1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 5);
}

TEST_F(UnsqueezeUT, Unsqueeze_infer_datatype)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Unsqueeze"), nullptr);
    auto op_impl = space_registry->GetOpImpl("Unsqueeze");
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Unsqueeze").OpName("Unsqueeze");
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
