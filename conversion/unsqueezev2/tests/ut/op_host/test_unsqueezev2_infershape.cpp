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

class UnsqueezeV2UT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "UnsqueezeV2 SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "UnsqueezeV2 TearDown" << std::endl; }
};

// x is not data dependent, the axis attr is read on compile stage
TEST_F(UnsqueezeV2UT, UnsqueezeV2_data_dependency)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("UnsqueezeV2"), nullptr);
    EXPECT_FALSE(space_registry->GetOpImpl("UnsqueezeV2")->IsInputDataDependency(0));
}

// negative axis is normalized by the y dim num: x={1,3,2,5} with axis {-5} gives y dim num 5,
// -5 points to dim 0
TEST_F(UnsqueezeV2UT, UnsqueezeV2_negative_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("UnsqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{1, 3, 2, 5}, {1, 3, 2, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("UnsqueezeV2")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{-5})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 5);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(3), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(4), 5);
}

TEST_F(UnsqueezeV2UT, UnsqueezeV2_one_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("UnsqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("UnsqueezeV2")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{1})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(3), 5);
}

TEST_F(UnsqueezeV2UT, UnsqueezeV2_two_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("UnsqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("UnsqueezeV2")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{0, 4})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 5);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(3), 5);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(4), 1);
}

TEST_F(UnsqueezeV2UT, UnsqueezeV2_three_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("UnsqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("UnsqueezeV2")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{2, 4, 5})
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

TEST_F(UnsqueezeV2UT, UnsqueezeV2_unsorted_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("UnsqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("UnsqueezeV2")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{5, 4, 2})
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

// repeated axes on the same position are rejected, same as the RT1.0 sort-unique check
TEST_F(UnsqueezeV2UT, UnsqueezeV2_repeated_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("UnsqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("UnsqueezeV2")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{1, 1})
                      .Build();

    EXPECT_NE(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
}

// the repeated check is on normalized positions (RT1.0 normalizes negatives before sort-unique):
// {2,-3} on a 3-dim x both point to dim 2 of the 5-dim output (3 x dims + 2 axes)
TEST_F(UnsqueezeV2UT, UnsqueezeV2_mixed_sign_repeated_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("UnsqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("UnsqueezeV2")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{2, -3})
                      .Build();

    EXPECT_NE(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
}

// readable empty axis: no dim is inserted, the output is the same as the input (1-dims are kept,
// unlike the squeeze semantics)
TEST_F(UnsqueezeV2UT, UnsqueezeV2_empty_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("UnsqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{3, 1, 4, 5}, {3, 1, 4, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("UnsqueezeV2")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(3), 5);
}

// the axis attr is not set (GetListInt returns nullptr): same as an explicit empty list, no dim is
// inserted (the RT1.0 GetAttr failure is ignored)
TEST_F(UnsqueezeV2UT, UnsqueezeV2_unset_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("UnsqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{3, 1, 4, 5}, {3, 1, 4, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("UnsqueezeV2")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(3), 5);
}

// axis out of range: y dim num is 4 (3+1), axis 4 is not in [-4, 4)
TEST_F(UnsqueezeV2UT, UnsqueezeV2_out_of_range_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("UnsqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("UnsqueezeV2")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{4})
                      .Build();

    EXPECT_NE(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
}

// negative axis out of range: -5 + 4 = -1 < 0
TEST_F(UnsqueezeV2UT, UnsqueezeV2_negative_axis_out_of_range)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("UnsqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("UnsqueezeV2")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{-5})
                      .Build();

    EXPECT_NE(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
}

// x dim num + axis num exceeds kMaxDimNum: 3 + 23 = 26 > 25; the 23 axes are distinct, so the
// failure can only come from the dim num limit (not from repeated or out-of-range axes)
TEST_F(UnsqueezeV2UT, UnsqueezeV2_larger_than_kMaxDimNum_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("UnsqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape output_shape = {{}, {}};
    std::vector<int64_t> axis;
    for (int64_t idx = 0; idx < 23; ++idx) {
        axis.push_back(idx);
    }

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("UnsqueezeV2")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", axis)
                      .Build();

    EXPECT_NE(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
}

// x dim num + axis num equals kMaxDimNum exactly: 22 + 3 = 25, the boundary is inclusive (26 fails,
// see larger_than_kMaxDimNum_axis); the axes hit the first, middle and last valid positions
TEST_F(UnsqueezeV2UT, UnsqueezeV2_equal_kMaxDimNum_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("UnsqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
                                      {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("UnsqueezeV2")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{0, 11, 24})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 25);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(11), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(12), 12);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(23), 23);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(24), 1);
}

// unknown rank(-2) passes through regardless of the axis attr
TEST_F(UnsqueezeV2UT, UnsqueezeV2_unknown_rank_with_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("UnsqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{-2}, {-2}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("UnsqueezeV2")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{1, 2, 3, 4, 5})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), -2);
}

// unknown rank(-2) passes through with an empty axis, same branch as unknown_rank_with_axis
TEST_F(UnsqueezeV2UT, UnsqueezeV2_unknown_rank_with_empty_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("UnsqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{-2}, {-2}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("UnsqueezeV2")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), -2);
}

// unknown dims(-1) of x are kept as-is in the output
TEST_F(UnsqueezeV2UT, UnsqueezeV2_unknown_dims)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("UnsqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{3, 4, -1}, {3, 4, -1}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("UnsqueezeV2")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{0})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(3), -1);
}

// scalar x: insert one dim at 0 gives y={1}
TEST_F(UnsqueezeV2UT, UnsqueezeV2_scalar_x)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("UnsqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("UnsqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{}, {}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("UnsqueezeV2")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{0})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 1);
}

TEST_F(UnsqueezeV2UT, UnsqueezeV2_registered_success)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("UnsqueezeV2"), nullptr);
}

TEST_F(UnsqueezeV2UT, UnsqueezeV2_infer_datatype)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("UnsqueezeV2"), nullptr);
    auto op_impl = space_registry->GetOpImpl("UnsqueezeV2");
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("UnsqueezeV2").OpName("UnsqueezeV2");
    builder.IONum(1, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto context_holder = builder.Build();
    auto* context = context_holder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = op_impl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    // output dtype is same as input x
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}
