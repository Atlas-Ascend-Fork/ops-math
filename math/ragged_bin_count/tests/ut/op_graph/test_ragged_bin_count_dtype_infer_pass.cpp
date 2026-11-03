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
#include "graph/operator_factory.h"
#include "gtest/gtest.h"

#include "../../../op_graph/ragged_bin_count_graph_infer_internal.h"
#include "../../../op_graph/ragged_bin_count_proto.h"

namespace {
struct DtypeCase {
    ge::DataType splits;
    ge::DataType values;
    ge::DataType size;
    ge::DataType weights;
    ge::graphStatus expected;
};

void CheckRegisteredSplitVerifier(const std::vector<int64_t>& splitShape, ge::DataType valueType, bool binaryOutput,
                                  ge::graphStatus expected)
{
    SCOPED_TRACE(::testing::Message() << "splits=" << ::testing::PrintToString(splitShape)
                                      << ", value dtype=" << valueType << ", binary=" << binaryOutput);
    auto op = ge::OperatorFactory::CreateOperator("ragged_bin_count_verifier", "RaggedBinCount");
    ASSERT_FALSE(op.IsEmpty());
    const std::vector<std::vector<int64_t>> inputShapes = {splitShape, {2613}, {1}, {3, 871}};
    const ge::DataType inputTypes[] = {ge::DT_INT64, valueType, valueType, ge::DT_FLOAT};
    const char* const inputNames[] = {"splits", "values", "size", "weights"};
    for (size_t index = 0; index < inputShapes.size(); ++index) {
        ge::TensorDesc desc(ge::Shape(inputShapes[index]), ge::FORMAT_ND, inputTypes[index]);
        desc.SetOriginShape(ge::Shape(inputShapes[index]));
        desc.SetOriginFormat(ge::FORMAT_ND);
        ASSERT_EQ(op.UpdateInputDesc(inputNames[index], desc), ge::GRAPH_SUCCESS);
    }
    ge::TensorDesc output(ge::Shape({-1, -1}), ge::FORMAT_ND, ge::DT_FLOAT);
    output.SetOriginShape(ge::Shape({-1, -1}));
    output.SetOriginFormat(ge::FORMAT_ND);
    ASSERT_EQ(op.UpdateOutputDesc("output", output), ge::GRAPH_SUCCESS);
    op.SetAttr("binary_output", binaryOutput);

    // Let unknown dimensions reach the registered callback instead of GE's static common verifier.
    EXPECT_EQ(op.VerifyAllAttr(/* disable_common_verifier = */ true), expected);
    EXPECT_EQ(op.VerifyAllAttr(/* disable_common_verifier = */ true), expected);
    if (splitShape[0] >= 0) {
        EXPECT_EQ(op.VerifyAllAttr(), expected);
    }
    for (size_t index = 0; index < inputShapes.size(); ++index) {
        const auto desc = op.GetInputDescByName(inputNames[index]);
        EXPECT_EQ(desc.GetShape().GetDims(), inputShapes[index]);
        EXPECT_EQ(desc.GetOriginShape().GetDims(), inputShapes[index]);
        EXPECT_EQ(desc.GetDataType(), inputTypes[index]);
        EXPECT_EQ(desc.GetFormat(), ge::FORMAT_ND);
        EXPECT_EQ(desc.GetOriginFormat(), ge::FORMAT_ND);
    }
    const auto actualOutput = op.GetOutputDescByName("output");
    EXPECT_EQ(actualOutput.GetShape().GetDims(), std::vector<int64_t>({-1, -1}));
    EXPECT_EQ(actualOutput.GetOriginShape().GetDims(), std::vector<int64_t>({-1, -1}));
    EXPECT_EQ(actualOutput.GetDataType(), ge::DT_FLOAT);
    EXPECT_EQ(actualOutput.GetFormat(), ge::FORMAT_ND);
    EXPECT_EQ(actualOutput.GetOriginFormat(), ge::FORMAT_ND);
    bool actualBinaryOutput = !binaryOutput;
    ASSERT_EQ(op.GetAttr("binary_output", actualBinaryOutput), ge::GRAPH_SUCCESS);
    EXPECT_EQ(actualBinaryOutput, binaryOutput);
}
} // namespace

TEST(RaggedBinCountGraphInferTest, test_infer_datatype_accepts_only_two_native_combinations)
{
    const std::vector<DtypeCase> cases = {
        {ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, ge::GRAPH_SUCCESS},
        {ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_FLOAT, ge::GRAPH_SUCCESS},
        {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, ge::GRAPH_FAILED},
        {ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_FAILED},
        {ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_FLOAT, ge::GRAPH_FAILED},
        {ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_DOUBLE, ge::GRAPH_FAILED},
    };

    for (const auto& dtypeCase : cases) {
        gert::OpInferDataTypeContextBuilder builder;
        builder.OpType("RaggedBinCount").OpName("RaggedBinCount");
        builder.IONum(4, 1);
        builder.InputTensorDesc(0, dtypeCase.splits, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.InputTensorDesc(1, dtypeCase.values, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.InputTensorDesc(2, dtypeCase.size, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.InputTensorDesc(3, dtypeCase.weights, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
        auto contextHolder = builder.Build();
        auto* context = contextHolder.GetContext();
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(ops::ragged_bin_count_graph_infer_internal::InferDataType(context), dtypeCase.expected);
        if (dtypeCase.expected == ge::GRAPH_SUCCESS) {
            EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
        }
    }
}

TEST(RaggedBinCountGraphInferTest, test_registered_verifier_rejects_zero_and_one_split)
{
    for (const auto valueType : {ge::DT_INT32, ge::DT_INT64}) {
        for (const bool binaryOutput : {false, true}) {
            CheckRegisteredSplitVerifier({0}, valueType, binaryOutput, ge::GRAPH_FAILED);
            CheckRegisteredSplitVerifier({1}, valueType, binaryOutput, ge::GRAPH_FAILED);
        }
    }
}

TEST(RaggedBinCountGraphInferTest, test_registered_verifier_preserves_valid_and_unknown_shapes)
{
    for (const auto valueType : {ge::DT_INT32, ge::DT_INT64}) {
        for (const bool binaryOutput : {false, true}) {
            for (const auto length : {2, 4, -1, -2}) {
                CheckRegisteredSplitVerifier({length}, valueType, binaryOutput, ge::GRAPH_SUCCESS);
            }
        }
    }
}
