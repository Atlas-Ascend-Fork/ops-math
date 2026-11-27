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
#include <string>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

class TopKV2DInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TopKV2DInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TopKV2DInfershape TearDown" << std::endl; }
};

// ===== InferDataType：values 跟随输入 x，indices 固定 DT_INT32（与 V1 TopKInferCommon 对齐） =====

TEST_F(TopKV2DInfershape, inferdatatype_values_follow_x_indices_int32)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    const auto* opImpl = registry->GetOpImpl("TopKV2D");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    for (const ge::DataType xDtype : {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32}) {
        gert::OpInferDataTypeContextBuilder builder;
        builder.OpType("TopKV2D").OpName("TopKV2D");
        builder.IONum(3, 2);
        builder.InputTensorDesc(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.InputTensorDesc(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.InputTensorDesc(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND);
        auto holder = builder.Build();
        auto* context = holder.GetContext();
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(opImpl->infer_datatype(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(context->GetOutputDataType(0), xDtype);
        EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT32);
    }
}

TEST_F(TopKV2DInfershape, inferdatatype_null_context_fail)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    const auto* opImpl = registry->GetOpImpl("TopKV2D");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    EXPECT_EQ(opImpl->infer_datatype(nullptr), ge::GRAPH_FAILED);
}

// ===== InferShape 回归：k 取常量输入值，dim 缺省为最后一维 =====

TEST_F(TopKV2DInfershape, infershape_const_k_last_dim)
{
    int32_t kValue = 2;
    gert::InfershapeContextPara infershapeContextPara("TopKV2D",
                                                      {
                                                          {{{4, 6}, {4, 6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &kValue},
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {{4, 2}, {4, 2}});
}
