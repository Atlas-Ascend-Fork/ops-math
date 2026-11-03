/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN AS IS BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>
#include "base/context_builder/op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeCast(gert::InferDataTypeContext* context);
} // namespace ops

class CastInferDataTypeTest : public testing::TestWithParam<ge::DataType> {};

TEST_P(CastInferDataTypeTest, cast_inferdatatype_test)
{
    auto inferDataTypeFunc = ops::InferDataTypeCast;

    ge::DataType dstDtype = GetParam();
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Cast").OpName("cast_dtype").IONum(1, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.AppendAttr(static_cast<int64_t>(dstDtype));
    auto holder = builder.Build();
    ASSERT_NE(holder.GetContext(), nullptr);
    ASSERT_EQ(holder.GetContext()->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    EXPECT_EQ(inferDataTypeFunc(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputDataType(0), dstDtype);
}

INSTANTIATE_TEST_SUITE_P(CastInferDataTypeSuite, CastInferDataTypeTest,
                         testing::Values(ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_INT64,
                                         ge::DT_INT16, ge::DT_INT8, ge::DT_UINT8, ge::DT_BOOL, ge::DT_DOUBLE));

// 非法 dst_type 属性值必须在推导处直接报错
TEST(CastInferDataTypeInvalidTest, cast_inferdatatype_invalid_dst_type_test)
{
    auto inferDataTypeFunc = ops::InferDataTypeCast;

    const std::vector<int64_t> invalidDstTypes = {static_cast<int64_t>(-1), static_cast<int64_t>(ge::DT_UNDEFINED),
                                                  static_cast<int64_t>(ge::DT_MAX), 999, 4294967301LL};
    for (const int64_t invalidDstType : invalidDstTypes) {
        gert::OpInferDataTypeContextBuilder builder;
        builder.OpType("Cast").OpName("cast_invalid").IONum(1, 1);
        builder.InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.AppendAttr(invalidDstType);
        auto holder = builder.Build();
        ASSERT_NE(holder.GetContext(), nullptr);
        ASSERT_EQ(holder.GetContext()->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
        EXPECT_EQ(inferDataTypeFunc(holder.GetContext()), ge::GRAPH_FAILED);
    }
}
