/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>

#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "gtest/gtest.h"

#include "../../../op_graph/atan_grad_graph_infer_internal.h"

namespace {
struct DtypeCase {
    ge::DataType y;
    ge::DataType dy;
    ge::graphStatus expected;
};
} // namespace

TEST(AtanGradGraphInferTest, test_infer_datatype_intercept_mixed_dtype)
{
    const std::vector<DtypeCase> cases = {
        {ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_SUCCESS},
        {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::GRAPH_SUCCESS},
        {ge::DT_BF16, ge::DT_BF16, ge::GRAPH_SUCCESS},
        // README 约束：y/dy/z dtype 必须完全一致，混合 dtype 必须在推理层拦截
        {ge::DT_FLOAT16, ge::DT_FLOAT, ge::GRAPH_FAILED},
        {ge::DT_FLOAT, ge::DT_FLOAT16, ge::GRAPH_FAILED},
        {ge::DT_BF16, ge::DT_FLOAT, ge::GRAPH_FAILED},
        {ge::DT_FLOAT16, ge::DT_BF16, ge::GRAPH_FAILED},
        {ge::DT_FLOAT, ge::DT_INT32, ge::GRAPH_FAILED},
        // 接口层白名单拦截：dtype 一致但不在支持列表（bool/complex128/int32 等）
        {ge::DT_BOOL, ge::DT_BOOL, ge::GRAPH_FAILED},
        {ge::DT_COMPLEX128, ge::DT_COMPLEX128, ge::GRAPH_FAILED},
        {ge::DT_DOUBLE, ge::DT_DOUBLE, ge::GRAPH_FAILED},
        {ge::DT_INT32, ge::DT_INT32, ge::GRAPH_FAILED},
        // dy 单独为不支持的 dtype
        {ge::DT_FLOAT, ge::DT_BOOL, ge::GRAPH_FAILED},
        // 必选输入缺失（GetInputDataType 返回 DT_UNDEFINED）
        {ge::DT_UNDEFINED, ge::DT_FLOAT, ge::GRAPH_FAILED},
        {ge::DT_FLOAT, ge::DT_UNDEFINED, ge::GRAPH_FAILED},
        {ge::DT_UNDEFINED, ge::DT_UNDEFINED, ge::GRAPH_FAILED},
    };

    for (const auto& dtypeCase : cases) {
        gert::OpInferDataTypeContextBuilder builder;
        builder.OpType("AtanGrad").OpName("AtanGrad");
        builder.IONum(2, 1);
        builder.InputTensorDesc(0, dtypeCase.y, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.InputTensorDesc(1, dtypeCase.dy, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
        auto contextHolder = builder.Build();
        auto* context = contextHolder.GetContext();
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(ops::atan_grad_graph_infer_internal::InferDataType(context), dtypeCase.expected);
        if (dtypeCase.expected == ge::GRAPH_SUCCESS) {
            EXPECT_EQ(context->GetOutputDataType(0), dtypeCase.y);
        }
    }
}
