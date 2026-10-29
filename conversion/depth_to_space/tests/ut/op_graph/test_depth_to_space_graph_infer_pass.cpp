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
#include <iostream>

#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "op_graph/graph_infer_comm_funcs.h"

class DepthToSpaceGraphInferTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DepthToSpaceGraphInferTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "DepthToSpaceGraphInferTest TearDown" << std::endl; }
};

TEST_F(DepthToSpaceGraphInferTest, depth_to_space_infer_datatype)
{
    const ge::DataType xDtypes[] = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_BF16};
    for (const auto xDtype : xDtypes) {
        gert::OpInferDataTypeContextBuilder builder;
        builder.OpType("DepthToSpace").OpName("DepthToSpace");
        builder.IONum(1, 1);
        builder.InputTensorDesc(0, xDtype, ge::FORMAT_NHWC, ge::FORMAT_NHWC);
        builder.OutputTensorDesc(0, ge::FORMAT_NHWC, ge::FORMAT_NHWC);
        auto contextHolder = builder.Build();

        auto* context = contextHolder.GetContext();
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(ops::InferDataTypeOutputSameAsInput(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(context->GetOutputDataType(0), xDtype);
    }
}
