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
#include "graph/operator_reg.h"
#include "graph/gnode.h"

// Host UT does not initialise the installed OPP. Register the two external IRs
// needed by this parser, with the signatures supplied by the CANN headers.
namespace ge {
REG_OP(Reshape)
    .INPUT(x, TensorType::ALL())
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axis, Int, 0)
    .ATTR(num_axes, Int, -1)
    .OP_END_FACTORY_REG(Reshape)

        REG_OP(RaggedBincount)
    .INPUT(splits, TensorType({DT_INT64}))
    .INPUT(values, TensorType({DT_INT32, DT_INT64}))
    .INPUT(size, TensorType({DT_INT32, DT_INT64}))
    .INPUT(weights, TensorType({DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(output, TensorType({DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE}))
    .ATTR(binary_output, Bool, false)
    .OP_END_FACTORY_REG(RaggedBincount)
} // namespace ge

#include "../../../framework/ragged_bin_count_tf_plugin.cpp"

namespace {
ge::Operator MakeSource(ge::DataType indexType, ge::DataType weightsType, bool binary)
{
    auto source = ge::op::RaggedBincount("tf_destination");
    source.SetAttr("Tidx", indexType);
    source.SetAttr("T", weightsType);
    source.SetAttr("binary_output", binary);
    return source;
}

ge::Operator MakeDestination(ge::DataType indexType, ge::DataType weightsType,
                             const std::vector<int64_t>& valuesShape = {6},
                             const std::vector<int64_t>& weightsShape = {2, 3})
{
    auto op = ge::op::RaggedBinCount("tf_destination");
    op.UpdateInputDesc("splits", ge::TensorDesc(ge::Shape({3}), ge::FORMAT_ND, ge::DT_INT64));
    op.UpdateInputDesc("values", ge::TensorDesc(ge::Shape(valuesShape), ge::FORMAT_ND, indexType));
    op.UpdateInputDesc("size", ge::TensorDesc(ge::Shape(), ge::FORMAT_ND, indexType));
    op.UpdateInputDesc("weights", ge::TensorDesc(ge::Shape(weightsShape), ge::FORMAT_ND, weightsType));
    op.UpdateOutputDesc("output", ge::TensorDesc(ge::Shape({2, 4}), ge::FORMAT_ND, weightsType));
    return op;
}

ge::graphStatus FindNodeByName(const ge::Graph& graph, const char* expectedName, ge::GNode& result)
{
    for (const auto& node : graph.GetAllNodes()) {
        ge::AscendString name;
        if (node.GetName(name) == ge::GRAPH_SUCCESS && std::string(name.GetString()) == expectedName) {
            result = node;
            return ge::GRAPH_SUCCESS;
        }
    }
    return ge::GRAPH_FAILED;
}

ge::TensorDesc GetInputDesc(const ge::GNode& node, int32_t index)
{
    ge::TensorDesc desc;
    EXPECT_EQ(node.GetInputDesc(index, desc), ge::GRAPH_SUCCESS);
    return desc;
}

ge::TensorDesc GetOutputDesc(const ge::GNode& node, int32_t index)
{
    ge::TensorDesc desc;
    EXPECT_EQ(node.GetOutputDesc(index, desc), ge::GRAPH_SUCCESS);
    return desc;
}

void CheckSizeDataEdge(const ge::Graph& graph, bool reshapeExpected)
{
    bool checked = false;
    for (const auto& node : graph.GetAllNodes()) {
        ge::AscendString name;
        ASSERT_EQ(node.GetName(name), ge::GRAPH_SUCCESS);
        if (std::string(name.GetString()) != "tf_destination") {
            continue;
        }
        auto input = node.GetInDataNodesAndPortIndexs(2).first;
        ASSERT_NE(input, nullptr);
        ge::AscendString type;
        ASSERT_EQ(input->GetType(type), ge::GRAPH_SUCCESS);
        EXPECT_STREQ(type.GetString(), reshapeExpected ? "Reshape" : "Data");
        if (reshapeExpected) {
            input = input->GetInDataNodesAndPortIndexs(0).first;
            ASSERT_NE(input, nullptr);
        }
        int64_t index = -1;
        ASSERT_EQ(input->GetAttr("index", index), ge::GRAPH_SUCCESS);
        EXPECT_EQ(index, 2);
        checked = true;
    }
    EXPECT_TRUE(checked);
}
} // namespace

TEST(RaggedBinCountTfPluginTest, ScalarSizeUsesRealReshapeAndPreservesDataDependency)
{
    for (const auto indexType : {ge::DT_INT32, ge::DT_INT64}) {
        for (const bool binary : {false, true}) {
            auto source = MakeSource(indexType, ge::DT_FLOAT, binary);
            auto destination = MakeDestination(indexType, ge::DT_FLOAT);
            ASSERT_EQ(domi::ParseParamsRaggedBinCount(source, destination), domi::SUCCESS);
            ge::AscendString originalType;
            ASSERT_EQ(destination.GetAttr("original_type", originalType), ge::GRAPH_SUCCESS);
            EXPECT_STREQ(originalType.GetString(), "RaggedBincount");
            ge::Graph graph("native_tf_ragged");
            ASSERT_EQ(domi::ParseOpToGraphRaggedBinCount(destination, graph), domi::SUCCESS);
            ge::GNode result;
            ASSERT_EQ(FindNodeByName(graph, "tf_destination", result), ge::GRAPH_SUCCESS);
            ge::AscendString type;
            ASSERT_EQ(result.GetType(type), ge::GRAPH_SUCCESS);
            EXPECT_STREQ(type.GetString(), "RaggedBinCount");
            EXPECT_EQ(GetInputDesc(result, 2).GetShape().GetDims(), std::vector<int64_t>({1}));
            EXPECT_EQ(GetInputDesc(result, 2).GetOriginShape().GetDims(), std::vector<int64_t>({1}));
            EXPECT_EQ(GetInputDesc(result, 2).GetDataType(), indexType);
            EXPECT_EQ(GetInputDesc(result, 1).GetShape().GetDims(), std::vector<int64_t>({6}));
            EXPECT_EQ(GetInputDesc(result, 3).GetShape().GetDims(), std::vector<int64_t>({2, 3}));
            bool actualBinary = !binary;
            ASSERT_EQ(result.GetAttr("binary_output", actualBinary), ge::GRAPH_SUCCESS);
            EXPECT_EQ(actualBinary, binary);
            ge::GNode reshape;
            ASSERT_EQ(FindNodeByName(graph, "size_to_vector", reshape), ge::GRAPH_SUCCESS);
            EXPECT_EQ(GetInputDesc(reshape, 0).GetShape().GetDimNum(), 0U);
            EXPECT_EQ(GetInputDesc(reshape, 0).GetDataType(), indexType);
            ge::Tensor targetShape;
            ASSERT_EQ(reshape.GetInputConstData(1, targetShape), ge::GRAPH_SUCCESS);
            ASSERT_EQ(targetShape.GetSize(), sizeof(int64_t));
            EXPECT_EQ(*reinterpret_cast<const int64_t*>(targetShape.GetData()), 1);
            CheckSizeDataEdge(graph, true);
        }
    }
}

TEST(RaggedBinCountTfPluginTest, IntegerAndDoubleWeightsKeepOriginalTargetAndScalarSize)
{
    for (const auto weightsType : {ge::DT_INT32, ge::DT_INT64, ge::DT_DOUBLE}) {
        for (const auto indexType : {ge::DT_INT32, ge::DT_INT64}) {
            auto source = MakeSource(indexType, weightsType, true);
            auto destination = MakeDestination(indexType, weightsType);
            ASSERT_EQ(domi::ParseParamsRaggedBinCount(source, destination), domi::SUCCESS);
            ge::Graph graph("legacy_tf_ragged");
            ASSERT_EQ(domi::ParseOpToGraphRaggedBinCount(destination, graph), domi::SUCCESS);
            ge::GNode result;
            ASSERT_EQ(FindNodeByName(graph, "tf_destination", result), ge::GRAPH_SUCCESS);
            ge::AscendString type;
            ASSERT_EQ(result.GetType(type), ge::GRAPH_SUCCESS);
            EXPECT_STREQ(type.GetString(), "RaggedBincount");
            EXPECT_EQ(GetInputDesc(result, 2).GetShape().GetDimNum(), 0U);
            EXPECT_EQ(GetInputDesc(result, 2).GetDataType(), indexType);
            EXPECT_EQ(GetInputDesc(result, 3).GetDataType(), weightsType);
            EXPECT_EQ(GetOutputDesc(result, 0).GetDataType(), weightsType);
            ge::GNode reshape;
            EXPECT_EQ(FindNodeByName(graph, "size_to_vector", reshape), ge::GRAPH_FAILED);
            CheckSizeDataEdge(graph, false);
        }
    }
}

TEST(RaggedBinCountTfPluginTest, Issue6ShapesArePreservedWithoutAllocatingTensorPayloads)
{
    auto source = MakeSource(ge::DT_INT64, ge::DT_FLOAT, false);
    auto destination = MakeDestination(ge::DT_INT64, ge::DT_FLOAT, {91054373}, {2923, 31151});
    ASSERT_EQ(domi::ParseParamsRaggedBinCount(source, destination), domi::SUCCESS);
    ge::Graph graph("issue6_metadata");
    ASSERT_EQ(domi::ParseOpToGraphRaggedBinCount(destination, graph), domi::SUCCESS);
    ge::GNode result;
    ASSERT_EQ(FindNodeByName(graph, "tf_destination", result), ge::GRAPH_SUCCESS);
    EXPECT_EQ(GetInputDesc(result, 1).GetShape().GetDims(), std::vector<int64_t>({91054373}));
    EXPECT_EQ(GetInputDesc(result, 3).GetShape().GetDims(), std::vector<int64_t>({2923, 31151}));
}

TEST(RaggedBinCountTfPluginTest, MissingOrUnsupportedTypeAttributesAreRejected)
{
    auto destination = MakeDestination(ge::DT_INT32, ge::DT_FLOAT);
    auto missing = ge::op::RaggedBincount("missing_types");
    EXPECT_EQ(domi::ParseParamsRaggedBinCount(missing, destination), domi::FAILED);
    auto badIndex = MakeSource(ge::DT_FLOAT, ge::DT_FLOAT, false);
    EXPECT_EQ(domi::ParseParamsRaggedBinCount(badIndex, destination), domi::FAILED);
    auto badWeights = MakeSource(ge::DT_INT32, ge::DT_FLOAT16, false);
    EXPECT_EQ(domi::ParseParamsRaggedBinCount(badWeights, destination), domi::FAILED);
}
