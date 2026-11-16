/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ge/es_graph_builder.h"
#include "ge/compliant_node_builder.h"
#include "graph/attr_value.h"
#include "graph/tensor.h"
#include "ge/fusion/pass/fusion_pass_reg.h"

#include "../../../op_graph/fusion_pass/transpose_empty_perm_fusion_pass.h"

using namespace ge;
using namespace ge::fusion;
using namespace ops;

namespace ge {
namespace fusion {
// GE keeps the fusion pass registry internal. Declaring the two queried members here allows the UT
// to fetch the pass factory that REG_FUSION_PASS installed and exercise its creation lambda.
class PassRegistry {
public:
    static PassRegistry& GetInstance();
    std::vector<FusionPassRegistrationData> GetFusionPassRegDataByStage(CustomPassStage stage) const;
};
} // namespace fusion
} // namespace ge

namespace {
const std::string kTypeConst = "Const";
const std::string kTypeTranspose = "Transpose";

// Build a Transpose graph: x -> Transpose(x, perm=Const(value)) -> y.
// permValues may be empty, in which case the Const is an empty tensor of shape permShape.
std::shared_ptr<Graph> BuildTransposeGraph(const std::vector<int64_t>& xDims, const std::vector<int32_t>& permValues,
                                           const std::vector<int64_t>& permShape)
{
    auto builder = es::EsGraphBuilder("transpose_empty_perm_ut");
    auto* graph = builder.GetCGraphBuilder()->GetGraph();

    auto x = builder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, xDims);
    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT);
    xDesc.SetShape(Shape(xDims));
    xDesc.SetFormat(FORMAT_ND);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);

    // Const perm node carrying an explicit (possibly empty) tensor value attr.
    Tensor permTensor;
    TensorDesc permDesc(Shape(permShape), FORMAT_ND, DT_INT32);
    permTensor.SetTensorDesc(permDesc);
    if (!permValues.empty()) {
        permTensor.SetData(reinterpret_cast<const uint8_t*>(permValues.data()), permValues.size() * sizeof(int32_t));
    }
    AttrValue valueAttr;
    valueAttr.SetAttrValue(permTensor);

    GNode constNode = es::CompliantNodeBuilder(graph)
                          .OpType("Const")
                          .Name("perm_const")
                          .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .IrDefAttrs({{"value", es::CompliantNodeBuilder::kEsAttrOptional, "VT_TENSOR", valueAttr}})
                          .InstanceOutputShape("y", permShape)
                          .InstanceOutputDataType("y", DT_INT32)
                          .InstanceOutputFormat("y", FORMAT_ND)
                          .Build();

    GNode transposeNode = es::CompliantNodeBuilder(graph)
                              .OpType("Transpose")
                              .Name("transpose")
                              .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                            {"perm", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                              .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                              .Build();
    (void)es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), transposeNode, 0);
    (void)es::AddEdgeAndUpdatePeerDesc(*graph, constNode, 0, transposeNode, 1);
    // ES does not propagate descs; fill them explicitly so the pass can read the data rank.
    (void)transposeNode.UpdateInputDesc(0, xDesc);
    TensorDesc permOutDesc;
    (void)constNode.GetOutputDesc(0, permOutDesc);
    (void)transposeNode.UpdateInputDesc(1, permOutDesc);

    auto y = builder.GetCGraphBuilder()->GetTensorHolderFromNode(transposeNode, 0);
    return builder.BuildAndReset(std::vector<es::EsTensorHolder>{es::EsTensorHolder(y)});
}

// Build two Transpose nodes sharing one perm Const (must be left untouched by the pass).
std::shared_ptr<Graph> BuildSharedPermGraph(const std::vector<int64_t>& xDims)
{
    auto builder = es::EsGraphBuilder("transpose_shared_perm_ut");
    auto* graph = builder.GetCGraphBuilder()->GetGraph();

    auto x0 = builder.CreateInput(0, "x0", DT_FLOAT, FORMAT_ND, xDims);
    auto x1 = builder.CreateInput(1, "x1", DT_FLOAT, FORMAT_ND, xDims);
    TensorDesc xDesc;
    x0.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT);
    xDesc.SetShape(Shape(xDims));
    xDesc.SetFormat(FORMAT_ND);
    x0.GetProducer()->UpdateOutputDesc(0, xDesc);
    x1.GetProducer()->UpdateOutputDesc(0, xDesc);

    Tensor permTensor;
    TensorDesc permDesc(Shape(std::vector<int64_t>{0}), FORMAT_ND, DT_INT32);
    permTensor.SetTensorDesc(permDesc);
    AttrValue valueAttr;
    valueAttr.SetAttrValue(permTensor);
    GNode constNode = es::CompliantNodeBuilder(graph)
                          .OpType("Const")
                          .Name("shared_perm_const")
                          .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .IrDefAttrs({{"value", es::CompliantNodeBuilder::kEsAttrOptional, "VT_TENSOR", valueAttr}})
                          .InstanceOutputShape("y", std::vector<int64_t>{0})
                          .InstanceOutputDataType("y", DT_INT32)
                          .InstanceOutputFormat("y", FORMAT_ND)
                          .Build();
    TensorDesc permOutDesc;
    (void)constNode.GetOutputDesc(0, permOutDesc);

    std::vector<es::EsTensorHolder> outs;
    const es::EsTensorHolder xHolders[2] = {x0, x1};
    for (int32_t i = 0; i < 2; ++i) {
        const std::string transposeName = "transpose_" + std::to_string(i);
        GNode transposeNode = es::CompliantNodeBuilder(graph)
                                  .OpType("Transpose")
                                  .Name(transposeName.c_str())
                                  .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                                {"perm", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                                  .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                                  .Build();
        (void)es::AddEdgeAndUpdatePeerDesc(*graph, *xHolders[i].GetProducer(), xHolders[i].GetProducerOutIndex(),
                                           transposeNode, 0);
        (void)es::AddEdgeAndUpdatePeerDesc(*graph, constNode, 0, transposeNode, 1);
        (void)transposeNode.UpdateInputDesc(0, xDesc);
        (void)transposeNode.UpdateInputDesc(1, permOutDesc);
        outs.emplace_back(es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(transposeNode, 0)));
    }
    return builder.BuildAndReset(outs);
}

// Read the perm Const's value tensor from the graph. Returns false if not found.
bool ReadPermConstValue(const std::shared_ptr<Graph>& graph, Tensor& outTensor)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        if (node.GetType(type) != GRAPH_SUCCESS) {
            continue;
        }
        if (std::string(type.GetString()) != kTypeConst) {
            continue;
        }
        if (node.GetAttr("value", outTensor) != GRAPH_SUCCESS) {
            continue;
        }
        return true;
    }
    return false;
}
} // namespace

class TransposeEmptyPermFusionPassTest : public testing::Test {
protected:
    static int32_t CountNodesOfType(const std::shared_ptr<Graph>& graph, const std::string& target)
    {
        int32_t count = 0;
        for (auto node : graph->GetAllNodes()) {
            AscendString type;
            node.GetType(type);
            if (type == target.c_str()) {
                ++count;
            }
        }
        return count;
    }
};

TEST_F(TransposeEmptyPermFusionPassTest, instantiationTest)
{
    TransposeEmptyPermFusionPass pass;
    SUCCEED();
}

TEST_F(TransposeEmptyPermFusionPassTest, emptyPermFilledWithReverse)
{
    const std::vector<int64_t> xDims = {2, 3, 4, 5};
    auto graphPtr = BuildTransposeGraph(xDims, {}, {0});
    ASSERT_NE(graphPtr, nullptr);

    TransposeEmptyPermFusionPass pass;
    CustomPassContext passContext;
    EXPECT_EQ(pass.Run(graphPtr, passContext), SUCCESS);

    Tensor permTensor;
    ASSERT_TRUE(ReadPermConstValue(graphPtr, permTensor));
    const auto dims = permTensor.GetTensorDesc().GetShape().GetDims();
    ASSERT_EQ(dims.size(), 1U);
    EXPECT_EQ(dims[0], static_cast<int64_t>(xDims.size()));

    const int32_t* values = reinterpret_cast<const int32_t*>(permTensor.GetData());
    ASSERT_NE(values, nullptr);
    const std::vector<int32_t> expected = {3, 2, 1, 0};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(values[i], expected[i]) << "index " << i;
    }
    // Topology unchanged.
    EXPECT_EQ(CountNodesOfType(graphPtr, kTypeTranspose), 1);
    EXPECT_EQ(CountNodesOfType(graphPtr, kTypeConst), 1);

    // The perm input desc shape must no longer be {0}; tiling rejects an empty perm shape.
    for (auto node : graphPtr->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (std::string(type.GetString()) != kTypeTranspose) {
            continue;
        }
        TensorDesc permInDesc;
        ASSERT_EQ(node.GetInputDesc(1, permInDesc), GRAPH_SUCCESS);
        EXPECT_EQ(permInDesc.GetShape().GetDims(), std::vector<int64_t>({4}));
    }
}

TEST_F(TransposeEmptyPermFusionPassTest, nonEmptyPermKeptUnchanged)
{
    const std::vector<int64_t> xDims = {2, 3, 4, 5};
    const std::vector<int32_t> perm = {0, 2, 3, 1};
    auto graphPtr = BuildTransposeGraph(xDims, perm, {4});
    ASSERT_NE(graphPtr, nullptr);

    TransposeEmptyPermFusionPass pass;
    CustomPassContext passContext;
    EXPECT_EQ(pass.Run(graphPtr, passContext), GRAPH_NOT_CHANGED);

    Tensor permTensor;
    ASSERT_TRUE(ReadPermConstValue(graphPtr, permTensor));
    const int32_t* values = reinterpret_cast<const int32_t*>(permTensor.GetData());
    ASSERT_NE(values, nullptr);
    for (size_t i = 0; i < perm.size(); ++i) {
        EXPECT_EQ(values[i], perm[i]);
    }
}

TEST_F(TransposeEmptyPermFusionPassTest, sharedEmptyPermConstReplacedPerConsumer)
{
    const std::vector<int64_t> xDims = {2, 3, 4, 5};
    auto graphPtr = BuildSharedPermGraph(xDims);
    ASSERT_NE(graphPtr, nullptr);

    TransposeEmptyPermFusionPass pass;
    CustomPassContext passContext;
    // The shared Const is replaced (never mutated in place): each Transpose gets its own
    // reverse-perm Const node, so the two consumers no longer fight over a single value.
    EXPECT_EQ(pass.Run(graphPtr, passContext), SUCCESS);

    int32_t checked = 0;
    for (auto node : graphPtr->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (std::string(type.GetString()) != kTypeTranspose) {
            continue;
        }
        TensorDesc permInDesc;
        ASSERT_EQ(node.GetInputDesc(1, permInDesc), GRAPH_SUCCESS);
        EXPECT_EQ(permInDesc.GetShape().GetDims(), std::vector<int64_t>({4}));
        auto producer = node.GetInDataNodesAndPortIndexs(1).first;
        ASSERT_NE(producer, nullptr);
        Tensor permTensor;
        ASSERT_EQ(producer->GetAttr("value", permTensor), GRAPH_SUCCESS);
        const auto dims = permTensor.GetTensorDesc().GetShape().GetDims();
        ASSERT_EQ(dims.size(), 1U);
        EXPECT_EQ(dims[0], static_cast<int64_t>(xDims.size()));
        const int32_t* values = reinterpret_cast<const int32_t*>(permTensor.GetData());
        ASSERT_NE(values, nullptr);
        EXPECT_EQ(values[0], 3);
        EXPECT_EQ(values[1], 2);
        EXPECT_EQ(values[2], 1);
        EXPECT_EQ(values[3], 0);
        ++checked;
    }
    EXPECT_EQ(checked, 2);
}

namespace {

// Build a value tensor for the perm Const. An empty `values` produces an empty (shape-size 0) tensor.
Tensor MakePermValueTensor(ge::DataType dtype, const std::vector<int64_t>& shape, const std::vector<int64_t>& values)
{
    Tensor permTensor;
    TensorDesc permDesc(Shape(shape), FORMAT_ND, dtype);
    permTensor.SetTensorDesc(permDesc);
    if (values.empty()) {
        return permTensor;
    }
    if (dtype == DT_INT64) {
        std::vector<int64_t> typed(values.begin(), values.end());
        (void)permTensor.SetData(reinterpret_cast<const uint8_t*>(typed.data()), typed.size() * sizeof(int64_t));
    } else if (dtype == DT_INT32) {
        std::vector<int32_t> typed(values.begin(), values.end());
        (void)permTensor.SetData(reinterpret_cast<const uint8_t*>(typed.data()), typed.size() * sizeof(int32_t));
    } else if (dtype == DT_FLOAT) {
        std::vector<float> typed(values.begin(), values.end());
        (void)permTensor.SetData(reinterpret_cast<const uint8_t*>(typed.data()), typed.size() * sizeof(float));
    }
    return permTensor;
}

// Build a graph with a Transpose whose perm input is produced by an arbitrary node kind.
// - producerType: "Const", "Constant" or "Data" (Data means the perm comes from a graph input).
// - withValueAttr: whether the producer carries a `value` tensor attribute.
// - connectPerm: whether the perm producer is wired to the Transpose perm input.
// - transposeDefinesPermInput: whether the Transpose IR declares the perm input at all.
std::shared_ptr<Graph> BuildGraphWithPermProducer(const std::vector<int64_t>& xDims, const std::string& producerType,
                                                  ge::DataType permDtype, const std::vector<int64_t>& permShape,
                                                  const std::vector<int64_t>& permValues, bool withValueAttr,
                                                  bool connectPerm, bool transposeDefinesPermInput = true,
                                                  bool controlEdgeFromPermProducer = false,
                                                  bool setDataInputDesc = true)
{
    auto builder = es::EsGraphBuilder("transpose_empty_perm_branch_ut");
    auto* graph = builder.GetCGraphBuilder()->GetGraph();

    auto x = builder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, xDims);
    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT);
    xDesc.SetShape(Shape(xDims));
    xDesc.SetFormat(FORMAT_ND);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);

    GNode producerNode;
    if (connectPerm) {
        if (producerType == "Data") {
            auto permData = builder.CreateInput(1, "perm", permDtype, FORMAT_ND, permShape);
            producerNode = *permData.GetProducer();
        } else {
            Tensor permTensor = MakePermValueTensor(permDtype, permShape, permValues);
            AttrValue valueAttr;
            valueAttr.SetAttrValue(permTensor);
            es::CompliantNodeBuilder producerBuilder(graph);
            producerBuilder.OpType(producerType.c_str())
                .Name("perm_producer")
                .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                .InstanceOutputShape("y", permShape)
                .InstanceOutputDataType("y", permDtype)
                .InstanceOutputFormat("y", FORMAT_ND);
            if (withValueAttr) {
                producerBuilder.IrDefAttrs(
                    {{"value", es::CompliantNodeBuilder::kEsAttrOptional, "VT_TENSOR", valueAttr}});
            }
            producerNode = producerBuilder.Build();
        }
    }

    std::vector<es::CompliantNodeBuilder::IrInputDef> transposeInputs = {
        {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}};
    if (transposeDefinesPermInput) {
        transposeInputs.push_back({"perm", es::CompliantNodeBuilder::kEsIrInputRequired, ""});
    }
    GNode transposeNode = es::CompliantNodeBuilder(graph)
                              .OpType("Transpose")
                              .Name("transpose")
                              .IrDefInputs(transposeInputs)
                              .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                              .Build();
    (void)es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), transposeNode, 0);
    if (setDataInputDesc) {
        (void)transposeNode.UpdateInputDesc(0, xDesc);
    }
    if (connectPerm) {
        (void)es::AddEdgeAndUpdatePeerDesc(*graph, producerNode, 0, transposeNode, 1);
        TensorDesc permOutDesc;
        (void)producerNode.GetOutputDesc(0, permOutDesc);
        (void)transposeNode.UpdateInputDesc(1, permOutDesc);
        if (controlEdgeFromPermProducer) {
            // Keep a control dependency alive so removing the old perm Const fails.
            (void)graph->AddControlEdge(producerNode, transposeNode);
        }
    }

    auto y = builder.GetCGraphBuilder()->GetTensorHolderFromNode(transposeNode, 0);
    return builder.BuildAndReset(std::vector<es::EsTensorHolder>{es::EsTensorHolder(y)});
}

} // namespace

// Null graph pointer must be reported as FAILED without touching the context.
TEST_F(TransposeEmptyPermFusionPassTest, nullGraphReturnsFailed)
{
    TransposeEmptyPermFusionPass pass;
    CustomPassContext passContext;
    GraphPtr graphPtr = nullptr;
    EXPECT_EQ(pass.Run(graphPtr, passContext), FAILED);
}

// A pass name already set by the framework must be preserved (no default overwrite).
TEST_F(TransposeEmptyPermFusionPassTest, presetPassNameKept)
{
    auto graphPtr = BuildTransposeGraph({2, 3, 4, 5}, {}, {0});
    ASSERT_NE(graphPtr, nullptr);

    TransposeEmptyPermFusionPass pass;
    CustomPassContext passContext;
    passContext.SetPassName("PresetPass");
    EXPECT_EQ(pass.Run(graphPtr, passContext), SUCCESS);
    EXPECT_STREQ(passContext.GetPassName().GetString(), "PresetPass");
}

// Transpose without a perm input (fewer than 2 inputs) is skipped.
TEST_F(TransposeEmptyPermFusionPassTest, transposeWithTooFewInputsSkipped)
{
    auto graphPtr = BuildGraphWithPermProducer({2, 3, 4, 5}, kTypeConst, DT_INT32, {0}, {}, true, false, false);
    ASSERT_NE(graphPtr, nullptr);

    TransposeEmptyPermFusionPass pass;
    CustomPassContext passContext;
    EXPECT_EQ(pass.Run(graphPtr, passContext), GRAPH_NOT_CHANGED);
}

// Scalar data input (rank 0) cannot provide a reverse perm, node is skipped.
TEST_F(TransposeEmptyPermFusionPassTest, scalarDataInputSkipped)
{
    auto graphPtr = BuildGraphWithPermProducer({}, kTypeConst, DT_INT32, {0}, {}, true, true);
    ASSERT_NE(graphPtr, nullptr);

    TransposeEmptyPermFusionPass pass;
    CustomPassContext passContext;
    EXPECT_EQ(pass.Run(graphPtr, passContext), GRAPH_NOT_CHANGED);
}

// Perm input without a producer node is skipped.
TEST_F(TransposeEmptyPermFusionPassTest, permWithoutProducerSkipped)
{
    auto graphPtr = BuildGraphWithPermProducer({2, 3, 4, 5}, kTypeConst, DT_INT32, {0}, {}, true, false);
    ASSERT_NE(graphPtr, nullptr);

    TransposeEmptyPermFusionPass pass;
    CustomPassContext passContext;
    EXPECT_EQ(pass.Run(graphPtr, passContext), GRAPH_NOT_CHANGED);
}

// Perm produced by a non-Const node is left untouched.
TEST_F(TransposeEmptyPermFusionPassTest, permProducerNotConstSkipped)
{
    auto graphPtr = BuildGraphWithPermProducer({2, 3, 4, 5}, "Data", DT_INT32, {0}, {}, true, true);
    ASSERT_NE(graphPtr, nullptr);

    TransposeEmptyPermFusionPass pass;
    CustomPassContext passContext;
    EXPECT_EQ(pass.Run(graphPtr, passContext), GRAPH_NOT_CHANGED);
}

// Const producer without a `value` attribute is skipped.
TEST_F(TransposeEmptyPermFusionPassTest, constWithoutValueAttrSkipped)
{
    auto graphPtr = BuildGraphWithPermProducer({2, 3, 4, 5}, kTypeConst, DT_INT32, {0}, {}, false, true);
    ASSERT_NE(graphPtr, nullptr);

    TransposeEmptyPermFusionPass pass;
    CustomPassContext passContext;
    EXPECT_EQ(pass.Run(graphPtr, passContext), GRAPH_NOT_CHANGED);
}

// `Constant` producer (accepted alias of Const) with an int64 empty perm is materialized.
TEST_F(TransposeEmptyPermFusionPassTest, constantTypeInt64EmptyPermFilled)
{
    const std::vector<int64_t> xDims = {2, 3, 4, 5};
    auto graphPtr = BuildGraphWithPermProducer(xDims, "Constant", DT_INT64, {0}, {}, true, true);
    ASSERT_NE(graphPtr, nullptr);

    TransposeEmptyPermFusionPass pass;
    CustomPassContext passContext;
    EXPECT_EQ(pass.Run(graphPtr, passContext), SUCCESS);

    Tensor permTensor;
    ASSERT_TRUE(ReadPermConstValue(graphPtr, permTensor));
    EXPECT_EQ(permTensor.GetTensorDesc().GetDataType(), DT_INT64);
    const auto dims = permTensor.GetTensorDesc().GetShape().GetDims();
    ASSERT_EQ(dims.size(), 1U);
    EXPECT_EQ(dims[0], static_cast<int64_t>(xDims.size()));
    const int64_t* values = reinterpret_cast<const int64_t*>(permTensor.GetData());
    ASSERT_NE(values, nullptr);
    const std::vector<int64_t> expected = {3, 2, 1, 0};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(values[i], expected[i]) << "index " << i;
    }
}

// Empty perm Const with an unsupported dtype (float) is skipped and the graph is unchanged.
TEST_F(TransposeEmptyPermFusionPassTest, unsupportedPermDtypeSkipped)
{
    auto graphPtr = BuildGraphWithPermProducer({2, 3, 4, 5}, kTypeConst, DT_FLOAT, {0}, {}, true, true);
    ASSERT_NE(graphPtr, nullptr);

    TransposeEmptyPermFusionPass pass;
    CustomPassContext passContext;
    EXPECT_EQ(pass.Run(graphPtr, passContext), GRAPH_NOT_CHANGED);
}

// Registered factory can create the pass (covers the REG_FUSION_PASS creation lambda).
TEST_F(TransposeEmptyPermFusionPassTest, registeredFactoryCreatesPass)
{
    auto registrations = ge::fusion::PassRegistry::GetInstance().GetFusionPassRegDataByStage(
        CustomPassStage::kAfterInferShape);
    bool found = false;
    for (auto& registration : registrations) {
        AscendString passName = registration.GetPassName();
        if (passName.GetString() == nullptr) {
            continue;
        }
        if (std::string(passName.GetString()) != "TransposeEmptyPermFusionPass") {
            continue;
        }
        found = true;
        EXPECT_EQ(registration.GetStage(), CustomPassStage::kAfterInferShape);
        auto createPassFn = registration.GetCreatePassFn();
        ASSERT_NE(createPassFn, nullptr);
        std::unique_ptr<FusionBasePass> pass(createPassFn());
        ASSERT_NE(pass, nullptr);
    }
    EXPECT_TRUE(found);
}

// A control edge attached to the old perm Const prevents its removal; the pass still fills the perm.
TEST_F(TransposeEmptyPermFusionPassTest, controlEdgeKeepsOldConst)
{
    const std::vector<int64_t> xDims = {2, 3, 4, 5};
    auto graphPtr = BuildGraphWithPermProducer(xDims, kTypeConst, DT_INT32, {0}, {}, true, true, true, true);
    ASSERT_NE(graphPtr, nullptr);

    TransposeEmptyPermFusionPass pass;
    CustomPassContext passContext;
    EXPECT_EQ(pass.Run(graphPtr, passContext), SUCCESS);

    // A reverse perm Const was materialized for the Transpose.
    bool foundReverseConst = false;
    for (auto node : graphPtr->GetAllNodes()) {
        AscendString name;
        (void)node.GetName(name);
        if (name.GetString() != nullptr && std::string(name.GetString()) == "perm_producer_reverse_perm") {
            foundReverseConst = true;
            Tensor permTensor;
            ASSERT_EQ(node.GetAttr("value", permTensor), GRAPH_SUCCESS);
            const auto dims = permTensor.GetTensorDesc().GetShape().GetDims();
            ASSERT_EQ(dims.size(), 1U);
            EXPECT_EQ(dims[0], static_cast<int64_t>(xDims.size()));
        }
    }
    EXPECT_TRUE(foundReverseConst);
}

// Data input desc is absent: the pass must skip the node instead of deriving a bogus rank.
TEST_F(TransposeEmptyPermFusionPassTest, missingDataInputDescSkipped)
{
    auto graphPtr = BuildGraphWithPermProducer({2, 3, 4, 5}, kTypeConst, DT_INT32, {0}, {}, true, true, true, false,
                                               false);
    ASSERT_NE(graphPtr, nullptr);

    TransposeEmptyPermFusionPass pass;
    CustomPassContext passContext;
    EXPECT_EQ(pass.Run(graphPtr, passContext), GRAPH_NOT_CHANGED);
}

namespace {

// Transpose whose IR only declares the data input, while the perm edge is attached to the
// undeclared port 1: updating the perm desc on that port must fail and abort the replacement.
std::shared_ptr<Graph> BuildGraphWithPermOnUndeclaredPort(const std::vector<int64_t>& xDims)
{
    auto builder = es::EsGraphBuilder("transpose_undeclared_perm_port_ut");
    auto* graph = builder.GetCGraphBuilder()->GetGraph();

    auto x = builder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, xDims);
    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT);
    xDesc.SetShape(Shape(xDims));
    xDesc.SetFormat(FORMAT_ND);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);

    Tensor permTensor;
    TensorDesc permDesc(Shape(std::vector<int64_t>{0}), FORMAT_ND, DT_INT32);
    permTensor.SetTensorDesc(permDesc);
    AttrValue valueAttr;
    valueAttr.SetAttrValue(permTensor);
    GNode constNode = es::CompliantNodeBuilder(graph)
                          .OpType("Const")
                          .Name("undeclared_perm_const")
                          .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .IrDefAttrs({{"value", es::CompliantNodeBuilder::kEsAttrOptional, "VT_TENSOR", valueAttr}})
                          .InstanceOutputShape("y", std::vector<int64_t>{0})
                          .InstanceOutputDataType("y", DT_INT32)
                          .InstanceOutputFormat("y", FORMAT_ND)
                          .Build();

    GNode transposeNode = es::CompliantNodeBuilder(graph)
                              .OpType("Transpose")
                              .Name("transpose")
                              .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                              .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                              .Build();
    (void)es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), transposeNode, 0);
    (void)transposeNode.UpdateInputDesc(0, xDesc);
    (void)graph->AddDataEdge(constNode, 0, transposeNode, 1);
    TensorDesc permOutDesc;
    (void)constNode.GetOutputDesc(0, permOutDesc);
    (void)transposeNode.UpdateInputDesc(1, permOutDesc);

    auto y = builder.GetCGraphBuilder()->GetTensorHolderFromNode(transposeNode, 0);
    return builder.BuildAndReset(std::vector<es::EsTensorHolder>{es::EsTensorHolder(y)});
}

} // namespace

// Perm desc update on an undeclared port fails, so the pass reports no change.
TEST_F(TransposeEmptyPermFusionPassTest, undeclaredPermPortUpdateFails)
{
    const std::vector<int64_t> xDims = {2, 3, 4, 5};
    auto graphPtr = BuildGraphWithPermOnUndeclaredPort(xDims);
    ASSERT_NE(graphPtr, nullptr);

    TransposeEmptyPermFusionPass pass;
    CustomPassContext passContext;
    EXPECT_EQ(pass.Run(graphPtr, passContext), GRAPH_NOT_CHANGED);
}
