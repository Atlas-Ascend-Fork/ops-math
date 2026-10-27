/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>

#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "gtest/gtest.h"

namespace ops {
ge::graphStatus InferDataTypeForMinimumGrad(gert::InferDataTypeContext* context);
} // namespace ops

namespace {
constexpr size_t kInputNum = 3U;
constexpr size_t kOutputNum = 2U;

// 构建 (grads, x1, x2) -> (y1, y2) 的 InferDataType 上下文并执行推导
// 返回推导结果，输出 dtype 通过 outputs 带回
static ge::graphStatus CheckInferDataType(const std::vector<ge::DataType>& inputDtypes,
                                          std::vector<ge::DataType>& outputs)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("MinimumGrad").OpName("MinimumGrad");
    builder.IONum(kInputNum, kOutputNum);
    for (size_t i = 0; i < kInputNum; ++i) {
        builder.InputTensorDesc(static_cast<int32_t>(i), inputDtypes[i], ge::FORMAT_ND, ge::FORMAT_ND);
    }
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND);
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ret = ops::InferDataTypeForMinimumGrad(context);
    outputs = {context->GetOutputDataType(0), context->GetOutputDataType(1)};
    return ret;
}
} // namespace

// 白名单 dtype 全支持：输出与 grads 一致
TEST(MinimumGradGraphInfer, infer_datatype_whitelist)
{
    const std::vector<ge::DataType> whitelist = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT32};
    for (const auto dtype : whitelist) {
        std::vector<ge::DataType> outputs;
        EXPECT_EQ(CheckInferDataType({dtype, dtype, dtype}, outputs), ge::GRAPH_SUCCESS);
        EXPECT_EQ(outputs[0], dtype);
        EXPECT_EQ(outputs[1], dtype);
    }
}

// 白名单外 dtype：拒绝
TEST(MinimumGradGraphInfer, infer_datatype_unsupported_fail)
{
    std::vector<ge::DataType> outputs;
    EXPECT_EQ(CheckInferDataType({ge::DT_INT8, ge::DT_INT8, ge::DT_INT8}, outputs), ge::GRAPH_FAILED);
}

// x1 与 grads dtype 不一致：拒绝（无类型提升）
TEST(MinimumGradGraphInfer, infer_datatype_x1_mismatch_fail)
{
    std::vector<ge::DataType> outputs;
    EXPECT_EQ(CheckInferDataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16}, outputs), ge::GRAPH_FAILED);
}

// x2 与 grads dtype 不一致：拒绝
TEST(MinimumGradGraphInfer, infer_datatype_x2_mismatch_fail)
{
    std::vector<ge::DataType> outputs;
    EXPECT_EQ(CheckInferDataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16}, outputs), ge::GRAPH_FAILED);
}
