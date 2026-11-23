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
 * \file test_flatten_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

class FlattenTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "FlattenTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "FlattenTest TearDown" << std::endl; }
};

// input {1,2,3}, axis=1: split at 1. Expect output {1,6}.
TEST_F(FlattenTest, flatten_infer_shape_success)
{
    gert::InfershapeContextPara infershapeContextPara("Flatten",
                                                      {
                                                          {{{1, 2, 3}, {1, 2, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 6}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// input {2,3,4,5}, axis=2: split at 2. Expect output {6,20}.
TEST_F(FlattenTest, flatten_infer_shape_mid_axis)
{
    gert::InfershapeContextPara infershapeContextPara("Flatten",
                                                      {
                                                          {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{6, 20}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// input {2,3,4}, axis=0: the first group is empty. Expect output {1,24}.
TEST_F(FlattenTest, flatten_infer_shape_axis_zero)
{
    gert::InfershapeContextPara infershapeContextPara("Flatten",
                                                      {
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 24}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// input {2,3,4}, axis=3 (== rank): the second group is empty. Expect output {24,1}.
TEST_F(FlattenTest, flatten_infer_shape_axis_equal_rank)
{
    gert::InfershapeContextPara infershapeContextPara("Flatten",
                                                      {
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{24, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// negative axis: input {2,3,4}, axis=-1 (resolves to 2). Expect output {6,4}.
TEST_F(FlattenTest, flatten_infer_shape_negative_axis)
{
    gert::InfershapeContextPara infershapeContextPara("Flatten",
                                                      {
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{6, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// dynamic dim in the second group: input {2,-1,4}, axis=1. Expect output {2,-1}.
TEST_F(FlattenTest, flatten_infer_shape_unknown_dim_second_group)
{
    gert::InfershapeContextPara infershapeContextPara("Flatten",
                                                      {
                                                          {{{2, -1, 4}, {2, -1, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// dynamic dim in the first group: input {-1,3,4}, axis=1. Expect output {-1,12}.
TEST_F(FlattenTest, flatten_infer_shape_unknown_dim_first_group)
{
    gert::InfershapeContextPara infershapeContextPara("Flatten",
                                                      {
                                                          {{{-1, 3, 4}, {-1, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 12}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// rank-1 input with unknown dim keeps rank 1 (RT1.0 semantics): {-1}, axis=1. Expect output {-1}.
TEST_F(FlattenTest, flatten_infer_shape_rank_one_unknown_dim)
{
    gert::InfershapeContextPara infershapeContextPara("Flatten",
                                                      {
                                                          {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// unknown rank input {-2} keeps unknown rank. Expect output {-2}.
TEST_F(FlattenTest, flatten_infer_shape_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara("Flatten",
                                                      {
                                                          {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// axis=4 out of range [-3,3] for input {2,3,4}. Expect failure.
TEST_F(FlattenTest, flatten_infer_shape_axis_out_of_range)
{
    gert::InfershapeContextPara infershapeContextPara("Flatten",
                                                      {
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(4)},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// axis=-4 out of range [-3,3] for input {2,3,4}. Expect failure.
TEST_F(FlattenTest, flatten_infer_shape_negative_axis_out_of_range)
{
    gert::InfershapeContextPara infershapeContextPara("Flatten",
                                                      {
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-4)},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// Migrated from canndev Flatten UT: input {4,3,4}, axis=1. Expect output {4,12}.
TEST_F(FlattenTest, flatten_infershape_1)
{
    gert::InfershapeContextPara infershapeContextPara("Flatten",
                                                      {
                                                          {{{4, 3, 4}, {4, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 12}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Migrated from canndev Flatten UT: input {-1,-1,-1}, axis=1. Expect output {-1,-1}.
TEST_F(FlattenTest, flatten_infershape_2)
{
    gert::InfershapeContextPara infershapeContextPara("Flatten",
                                                      {
                                                          {{{-1, -1, -1}, {-1, -1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Migrated from canndev Flatten UT: unknown-rank input {-2}. Expect output {-2}.
TEST_F(FlattenTest, flatten_infershape_3)
{
    gert::InfershapeContextPara infershapeContextPara("Flatten",
                                                      {
                                                          {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Migrated from canndev Flatten UT: input {4,3,4,5}, axis=2. Expect output {12,20}.
TEST_F(FlattenTest, flatten_infershape_5)
{
    gert::InfershapeContextPara infershapeContextPara("Flatten",
                                                      {
                                                          {{{4, 3, 4, 5}, {4, 3, 4, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{12, 20}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Migrated from canndev Flatten UT: input {4,3,4,5}, axis=-2. Expect output {12,20}.
TEST_F(FlattenTest, flatten_infershape_static_shape)
{
    gert::InfershapeContextPara infershapeContextPara("Flatten",
                                                      {
                                                          {{{4, 3, 4, 5}, {4, 3, 4, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-2)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{12, 20}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Migrated from canndev Flatten UT: input {4,3,4,5}, axis=0. Expect output {1,240}.
TEST_F(FlattenTest, flatten_infershape_static_shape1)
{
    gert::InfershapeContextPara infershapeContextPara("Flatten",
                                                      {
                                                          {{{4, 3, 4, 5}, {4, 3, 4, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 240}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Migrated from canndev Flatten UT: input {4,3,4,5}, axis=4 (== rank). Expect output {240,1}.
TEST_F(FlattenTest, flatten_infershape_static_shape3)
{
    gert::InfershapeContextPara infershapeContextPara("Flatten",
                                                      {
                                                          {{{4, 3, 4, 5}, {4, 3, 4, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(4)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{240, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Migrated from canndev Flatten UT: input {4,3,4,5}, axis=-4 (== -rank). Expect output {1,240}.
TEST_F(FlattenTest, flatten_infershape_static_shape5)
{
    gert::InfershapeContextPara infershapeContextPara("Flatten",
                                                      {
                                                          {{{4, 3, 4, 5}, {4, 3, 4, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-4)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 240}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Migrated from canndev Flatten UT: input {4,3,4,5}, axis=-5 out of range. Expect failure.
TEST_F(FlattenTest, flatten_infershape_static_shape6)
{
    gert::InfershapeContextPara infershapeContextPara("Flatten",
                                                      {
                                                          {{{4, 3, 4, 5}, {4, 3, 4, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-5)},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// Migrated from canndev Flatten UT: input {4,3,4,5}, axis=6 out of range. Expect failure.
TEST_F(FlattenTest, flatten_infershape_static_shape7)
{
    gert::InfershapeContextPara infershapeContextPara("Flatten",
                                                      {
                                                          {{{4, 3, 4, 5}, {4, 3, 4, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(6)},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// Migrated from canndev Flatten UT: input {5,-1,-1}, axis=1. Expect output {5,-1}.
TEST_F(FlattenTest, flatten_infershape_default_axis)
{
    gert::InfershapeContextPara infershapeContextPara("Flatten",
                                                      {
                                                          {{{5, -1, -1}, {5, -1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{5, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Migrated from canndev Flatten UT: input {5,-1,-1}, axis=2. Expect output {-1,-1}.
TEST_F(FlattenTest, flatten_infershape_x_range_empty)
{
    gert::InfershapeContextPara infershapeContextPara("Flatten",
                                                      {
                                                          {{{5, -1, -1}, {5, -1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Migrated from canndev Flatten UT: 1-D unknown input {-1}, axis=1. Expect output {-1}.
TEST_F(FlattenTest, flatten_infershape_x_1d)
{
    gert::InfershapeContextPara infershapeContextPara("Flatten",
                                                      {
                                                          {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// InferDataType copies the input dtype to the output.
TEST_F(FlattenTest, flatten_infer_data_type_success)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry, nullptr);
    auto op_impl = space_registry->GetOpImpl("Flatten");
    ASSERT_NE(op_impl, nullptr);
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Flatten").OpName("Flatten");
    builder.IONum(1, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto context_holder = builder.Build();
    auto* context = context_holder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = op_impl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}
