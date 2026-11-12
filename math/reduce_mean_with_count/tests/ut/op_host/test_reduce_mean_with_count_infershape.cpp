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
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

class ReduceMeanWithCountTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ReduceMeanWithCountTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ReduceMeanWithCountTest TearDown" << std::endl; }
};

// Test: reduce along axes [1, 2] with keep_dims=true
TEST_F(ReduceMeanWithCountTest, ReduceMeanWithCount_infer_keep_dims_true)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceMeanWithCount",
        {
            // 3 inputs: x, count, count_sum (no axes input tensor)
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("axes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})),
         gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 1, 1, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: reduce along axes [1, -2] with keep_dims=false (negative axis)
TEST_F(ReduceMeanWithCountTest, ReduceMeanWithCount_infer_keep_dims_false)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceMeanWithCount",
        {
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 5, 16, 16}, {3, 5, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("axes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, -2})),
         gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Test: reduce along all axes with keep_dims=true
TEST_F(ReduceMeanWithCountTest, ReduceMeanWithCount_infer_all_axes)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceMeanWithCount",
        {
            {{{2, 100, 4}, {2, 100, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 100, 4}, {2, 100, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 100, 4}, {2, 100, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("axes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 1, 2})),
         gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 1, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

static void RunInferDataTypeCase(const ge::DataType inputDtype, const ge::DataType expectOutputDtype)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("ReduceMeanWithCount");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("ReduceMeanWithCount").OpName("ReduceMeanWithCount");
    builder.IONum(3, 1);
    builder.InputTensorDesc(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(2, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = opImpl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), expectOutputDtype);
}

// Test: y.dtype -> x.dtype (float32)
TEST_F(ReduceMeanWithCountTest, ReduceMeanWithCount_infer_datatype_float)
{
    RunInferDataTypeCase(ge::DT_FLOAT, ge::DT_FLOAT);
}

// Test: y.dtype -> x.dtype (float16)
TEST_F(ReduceMeanWithCountTest, ReduceMeanWithCount_infer_datatype_float16)
{
    RunInferDataTypeCase(ge::DT_FLOAT16, ge::DT_FLOAT16);
}

// Test: y.dtype -> x.dtype (bf16)
TEST_F(ReduceMeanWithCountTest, ReduceMeanWithCount_infer_datatype_bf16)
{
    RunInferDataTypeCase(ge::DT_BF16, ge::DT_BF16);
}
