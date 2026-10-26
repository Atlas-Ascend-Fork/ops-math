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

class SqueezeV2UT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SqueezeV2 SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SqueezeV2 TearDown" << std::endl; }
};

// x is not data dependent, the axis attr is read on compile stage
TEST_F(SqueezeV2UT, SqueezeV2_data_dependency)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("SqueezeV2"), nullptr);
    EXPECT_FALSE(space_registry->GetOpImpl("SqueezeV2")->IsInputDataDependency(0));
}

// negative axis is normalized by the x dim num: -4 points to dim 0
TEST_F(SqueezeV2UT, SqueezeV2_negative_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("SqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{1, 3, 2, 5}, {1, 3, 2, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SqueezeV2")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{-4})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 5);
}

TEST_F(SqueezeV2UT, SqueezeV2_one_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("SqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{3, 1, 4, 5}, {3, 1, 4, 5}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SqueezeV2")
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

TEST_F(SqueezeV2UT, SqueezeV2_three_dim)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("SqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{3, 1, 4}, {3, 1, 4}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SqueezeV2")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{1})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 4);
}

TEST_F(SqueezeV2UT, SqueezeV2_two_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("SqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{1, 3, 4, 5, 1}, {1, 3, 4, 5, 1}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SqueezeV2")
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

TEST_F(SqueezeV2UT, SqueezeV2_three_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("SqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{3, 4, 1, 5, 1, 1}, {3, 4, 1, 5, 1, 1}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SqueezeV2")
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

TEST_F(SqueezeV2UT, SqueezeV2_unsorted_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("SqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{3, 4, 1, 5, 1, 1}, {3, 4, 1, 5, 1, 1}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SqueezeV2")
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

// repeated axis values are deduped: {3, 3, 4} acts as {3, 4}, the deduped num 2 is less than dim num 5
TEST_F(SqueezeV2UT, SqueezeV2_repeated_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("SqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{3, 4, 5, 1, 1}, {3, 4, 5, 1, 1}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SqueezeV2")
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

// readable empty axis: squeeze all dims whose value is 1
TEST_F(SqueezeV2UT, SqueezeV2_empty_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("SqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{3, 4, 1, 5, 1, 1}, {3, 4, 1, 5, 1, 1}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SqueezeV2")
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

// the axis attr is not set (GetListInt returns nullptr): same as an explicit empty list, squeeze all
// dims whose value is 1 (the RT1.0 GetAttr failure is ignored)
TEST_F(SqueezeV2UT, SqueezeV2_unset_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("SqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{3, 1, 4, 5, 1}, {3, 1, 4, 5, 1}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SqueezeV2")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 5);
}

// equivalent to the RT1.0 rule: when the deduped axis num is not less than the x dim num, the output
// falls back to the input shape
TEST_F(SqueezeV2UT, SqueezeV2_axis_num_not_less_than_dim_num)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("SqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{1, 1}, {1, 1}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SqueezeV2")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{0, 1})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 1);
}

// the count rule is the raw-value dedup of RT1.0: {0,-2} has 2 distinct raw values (not less than
// dim num 2), the output falls back to the input shape; a normalized-position count (1 < 2) would
// wrongly squeeze dim 0
TEST_F(SqueezeV2UT, SqueezeV2_mixed_sign_axis_raw_dedup_count)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("SqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{1, 3}, {1, 3}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SqueezeV2")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{0, -2})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 3);
}

// out-of-range axis does not fail: the output falls back to the input shape (RT1.0 semantics)
TEST_F(SqueezeV2UT, SqueezeV2_out_of_range_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("SqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{3, 4, 5, 1}, {3, 4, 5, 1}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SqueezeV2")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{4})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 5);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(3), 1);
}

// the dim pointed by axis is neither 1 nor -1: infershape fails
TEST_F(SqueezeV2UT, SqueezeV2_error_dim)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("SqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{3, 4, 5, 1, 1}, {3, 4, 5, 1, 1}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SqueezeV2")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{3, 1})
                      .Build();

    EXPECT_NE(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
}

// unknown rank(-2) passes through regardless of the axis attr
TEST_F(SqueezeV2UT, SqueezeV2_unknown_rank_with_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("SqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{-2}, {-2}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SqueezeV2")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{4})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), -2);
}

// unknown rank(-2) passes through with an empty axis, same branch as unknown_rank_with_axis
TEST_F(SqueezeV2UT, SqueezeV2_unknown_rank_with_empty_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("SqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{-2}, {-2}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SqueezeV2")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), -2);
}

// x has unknown dims and axis is empty: cannot determine which unknown dim is 1, output is unknown rank
TEST_F(SqueezeV2UT, SqueezeV2_unknown_dims_with_empty_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("SqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{3, 4, 1, 5, -1, 1}, {3, 4, 1, 5, -1, 1}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SqueezeV2")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), -2);
}

// unknown dims(-1) pointed by axis are treated as 1 and squeezed; complementary to the dynamic(-1)
// scenario of the graph example, where the unknown dim is not pointed by axis and kept as -1
TEST_F(SqueezeV2UT, SqueezeV2_unknown_dims)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("SqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{3, 4, -1, 5, -1, 1}, {3, 4, 1, 5, 1, 1}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SqueezeV2")
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

// the unknown dim(-1) not pointed by axis is kept as-is: same combination as the dynamic(-1)
// scenario of the graph example
TEST_F(SqueezeV2UT, SqueezeV2_unknown_dim_not_pointed_by_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("SqueezeV2"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV2")->infer_shape;
    gert::StorageShape input_shape = {{-1, 1, 3}, {-1, 1, 3}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SqueezeV2")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Attr("axis", std::vector<int64_t>{1})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), -1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 3);
}

TEST_F(SqueezeV2UT, SqueezeV2_registered_success)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("SqueezeV2"), nullptr);
}

TEST_F(SqueezeV2UT, SqueezeV2_infer_datatype)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("SqueezeV2"), nullptr);
    auto op_impl = space_registry->GetOpImpl("SqueezeV2");
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("SqueezeV2").OpName("SqueezeV2");
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
