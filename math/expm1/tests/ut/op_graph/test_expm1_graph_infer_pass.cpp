/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeExpm1(gert::InferDataTypeContext* context);
} // namespace ops

class Expm1InferDataTypeTest : public testing::TestWithParam<ge::DataType> {};

TEST_P(Expm1InferDataTypeTest, expm1_inferdatatype_test)
{
    auto inferDataTypeFunc = ops::InferDataTypeExpm1;

    ge::DataType inputDtype = GetParam();
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Expm1").OpName("expm1_dtype").IONum(1, 1);
    builder.InputTensorDesc(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    ASSERT_NE(holder.GetContext(), nullptr);
    ASSERT_EQ(holder.GetContext()->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    EXPECT_EQ(inferDataTypeFunc(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputDataType(0), inputDtype);
}

INSTANTIATE_TEST_SUITE_P(Expm1InferDataTypeSuite, Expm1InferDataTypeTest,
                         testing::Values(ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT));
