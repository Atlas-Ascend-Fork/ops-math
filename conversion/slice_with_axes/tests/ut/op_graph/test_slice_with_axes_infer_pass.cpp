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
 * \file test_slice_with_axes_infer_pass.cpp
 * \brief SliceWithAxes InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeSliceWithAxes(gert::InferDataTypeContext* context);
}

namespace {
// x / offsets / size 分别使用不同 dtype，确保推导实现取的是输入 0（x）而非其它索引
void RunSliceWithAxesInferDataType(ge::DataType xDtype)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("SliceWithAxes");
    builder.OpName("SliceWithAxes");
    builder.IONum(3, 1);
    builder.InputTensorDesc(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeSliceWithAxes(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), xDtype); // out0 = x dtype
}
} // namespace

TEST(SliceWithAxesGraphInfer, InferDataTypeFloat) { RunSliceWithAxesInferDataType(ge::DT_FLOAT); }

TEST(SliceWithAxesGraphInfer, InferDataTypeInt32) { RunSliceWithAxesInferDataType(ge::DT_INT32); }

TEST(SliceWithAxesGraphInfer, InferDataTypeBool) { RunSliceWithAxesInferDataType(ge::DT_BOOL); }
