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
 * \file test_flatten_v2_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

class FlattenV2Test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "FlattenV2Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "FlattenV2Test TearDown" << std::endl; }
};

// input {1,2,3}, axis=1, end_axis=-1 (default): flatten dims [1,2] into one. Expect output {1,6}.
TEST_F(FlattenV2Test, flatten_v2_infer_shape_success)
{
    gert::InfershapeContextPara infershapeContextPara("FlattenV2",
                                                      {
                                                          {{{1, 2, 3}, {1, 2, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                                          {"end_axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 6}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// input {2,3,4,5}, axis=1, end_axis=2: flatten middle dims. Expect output {2,12,5}.
TEST_F(FlattenV2Test, flatten_v2_infer_shape_mid_range)
{
    gert::InfershapeContextPara infershapeContextPara("FlattenV2",
                                                      {
                                                          {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                                          {"end_axis", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 12, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// input {2,3,4}, axis=0, end_axis=-1: flatten all dims. Expect output {24}.
TEST_F(FlattenV2Test, flatten_v2_infer_shape_flatten_all)
{
    gert::InfershapeContextPara infershapeContextPara("FlattenV2",
                                                      {
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
                                                          {"end_axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{24}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// negative axis: input {2,3,4}, axis=-2 (resolves to 1), end_axis=-1 (resolves to 2). Expect output {2,12}.
TEST_F(FlattenV2Test, flatten_v2_infer_shape_negative_axis)
{
    gert::InfershapeContextPara infershapeContextPara("FlattenV2",
                                                      {
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-2)},
                                                          {"end_axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 12}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// dynamic dim: input {2,-1,4}, axis=1, end_axis=2: flattened range contains -1. Expect output {2,-1}.
TEST_F(FlattenV2Test, flatten_v2_infer_shape_dynamic_dim)
{
    gert::InfershapeContextPara infershapeContextPara("FlattenV2",
                                                      {
                                                          {{{2, -1, 4}, {2, -1, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                                          {"end_axis", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// unknown rank input {-2}: rejected per RT1.0 verifier semantics. Expect failure.
TEST_F(FlattenV2Test, flatten_v2_infer_shape_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara("FlattenV2",
                                                      {
                                                          {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                                          {"end_axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// axis=3 out of range [0,3) for input {2,3,4}. Expect failure.
TEST_F(FlattenV2Test, flatten_v2_infer_shape_axis_out_of_range)
{
    gert::InfershapeContextPara infershapeContextPara("FlattenV2",
                                                      {
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(3)},
                                                          {"end_axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// axis=2 greater than end_axis=1. Expect failure.
TEST_F(FlattenV2Test, flatten_v2_infer_shape_axis_greater_than_end_axis)
{
    gert::InfershapeContextPara infershapeContextPara("FlattenV2",
                                                      {
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                                          {"end_axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// InferDataType copies the input dtype to the output.
TEST_F(FlattenV2Test, flatten_v2_infer_data_type_success)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry, nullptr);
    auto op_impl = space_registry->GetOpImpl("FlattenV2");
    ASSERT_NE(op_impl, nullptr);
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("FlattenV2").OpName("FlattenV2");
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
