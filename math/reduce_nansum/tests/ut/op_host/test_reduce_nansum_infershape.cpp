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
 * \file test_reduce_nansum_infershape.cpp
 * \brief ReduceNansum InferDataType UT.
 */
#include <iostream>
#include <gtest/gtest.h>
#include "op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

class ReduceNansumInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ReduceNansumInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ReduceNansumInfershape TearDown" << std::endl; }
};

static void RunInferDataTypeCase(const ge::DataType inputDtype, const ge::DataType expectOutputDtype)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("ReduceNansum");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("ReduceNansum").OpName("ReduceNansum");
    builder.IONum(2, 1);
    builder.InputTensorDesc(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = opImpl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), expectOutputDtype);
}

// y.dtype -> x.dtype（float32）
TEST_F(ReduceNansumInfershape, reduce_nansum_infer_datatype_float) { RunInferDataTypeCase(ge::DT_FLOAT, ge::DT_FLOAT); }

// y.dtype -> x.dtype（float16）
TEST_F(ReduceNansumInfershape, reduce_nansum_infer_datatype_float16)
{
    RunInferDataTypeCase(ge::DT_FLOAT16, ge::DT_FLOAT16);
}

// y.dtype -> x.dtype（bf16）
TEST_F(ReduceNansumInfershape, reduce_nansum_infer_datatype_bf16) { RunInferDataTypeCase(ge::DT_BF16, ge::DT_BF16); }
