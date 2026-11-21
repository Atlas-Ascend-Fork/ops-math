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
 * \file test_prod_force_se_a_graph_infer_pass.cpp
 * \brief ProdForceSeA InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "experiment/metadef/common/util/error_manager/error_manager.h"

namespace ops {
ge::graphStatus InferDataTypeProdForceSeA(gert::InferDataTypeContext* context);
} // namespace ops

namespace {
constexpr const char* OP_TYPE = "ProdForceSeA";

void RunInferDataType(ge::DataType inDtype, ge::graphStatus expectStatus, const char* expectErrorId)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType(OP_TYPE).OpName(OP_TYPE);
    builder.IONum(1, 1);
    builder.InputTensorDesc(0, inDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);

    auto& errors = ErrorManager::GetInstance();
    ASSERT_EQ(errors.Init(), 0);
    (void)errors.GetRawErrorMessages();

    EXPECT_EQ(ops::InferDataTypeProdForceSeA(context), expectStatus);
    if (expectErrorId != nullptr) {
        const auto reports = errors.GetRawErrorMessages();
        ASSERT_EQ(reports.size(), 1U);
        EXPECT_EQ(reports[0].error_id, expectErrorId);
    } else {
        EXPECT_EQ(context->GetOutputDataType(0), inDtype);
    }
}
} // namespace

TEST(ProdForceSeAGraphInfer, InferDataTypeFloatAccepted) { RunInferDataType(ge::DT_FLOAT, ge::GRAPH_SUCCESS, nullptr); }

TEST(ProdForceSeAGraphInfer, InferDataTypeFloat16Rejected)
{
    RunInferDataType(ge::DT_FLOAT16, ge::GRAPH_FAILED, "EZ0019");
}
