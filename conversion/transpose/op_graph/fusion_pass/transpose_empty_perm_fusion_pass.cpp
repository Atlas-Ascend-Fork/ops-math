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
 * \file transpose_empty_perm_fusion_pass.cpp
 * \brief Fill the ONNX default reverse perm for Transpose whose perm Const is empty.
 *
 * Background
 * ----------
 * ONNX Transpose's `perm` attribute is optional: when it is not provided the operator reverses the
 * input axes, i.e. perm = [n-1, n-2, ..., 0] with n = input rank. The ONNX plugin parses the node
 * before any shape is known, so it cannot compute the reverse perm itself; it keeps an empty perm
 * Const instead. InferShape falls back to the reverse permutation so that shape inference does not
 * fail, but the perm *tensor* is still empty, which downstream tiling/kernel cannot consume.
 *
 * This pass runs at kAfterInferShape (rank is now known), finds every Transpose whose perm input is
 * an empty Const and rewrites that Const's `value` to the reverse permutation. Only the Const value
 * is modified - the graph topology is untouched.
 *
 * Route: graph_base_pass / FusionBasePass (whole-graph traversal + local attribute rewrite), matching
 * the "only change attributes / need whole-graph traversal" branch of the pass selection matrix.
 * In-place Const value rewrite follows the same idiom as ops-nn
 * matmul/mat_mul_v3/.../matmul_reshape_transpose_fusion_pass.cpp (ChangeConstNode).
 */

#include <cstdint>
#include <string>
#include <vector>

#include "graph/attr_value.h"
#include "graph/tensor.h"
#include "log/log.h"
#include "ge/compliant_node_builder.h"
// REG_FUSION_PASS lives in pattern_fusion_pass.h; include it to obtain the registration macro
// (same as other FusionBasePass implementations in this repo, e.g. ConcatToConcatDFusionPass).
#include "ge/fusion/pass/pattern_fusion_pass.h"

#include "transpose_empty_perm_fusion_pass.h"

using namespace ge;
using namespace ge::fusion;

namespace ops {
namespace {

static const std::string PASS_NAME = "TransposeEmptyPermFusionPass";
static const std::string TYPE_TRANSPOSE = "Transpose";
static const std::string TYPE_CONST = "Const";
static const std::string TYPE_CONSTANT = "Constant";
static const char* const ATTR_VALUE = "value";

static const int32_t DATA_INPUT_IDX = 0;
static const int32_t PERM_INPUT_IDX = 1;
// Transpose always takes exactly one data input and one perm input.
static const size_t TRANSPOSE_MIN_INPUT_NUM = 2;

std::string GetNodeNameStr(const GNode& node)
{
    AscendString name;
    if (node.GetName(name) != GRAPH_SUCCESS || name.GetString() == nullptr) {
        return "<unknown>"; // LCOV_EXCL_LINE: live graph nodes always carry a name
    }
    return name.GetString();
}

std::string GetNodeTypeStr(const GNode& node)
{
    AscendString type;
    if (node.GetType(type) != GRAPH_SUCCESS || type.GetString() == nullptr) {
        return ""; // LCOV_EXCL_LINE: live graph nodes always carry a type
    }
    return type.GetString();
}

// Build a 1-D tensor holding `rank` elements: the reverse permutation [rank-1, ..., 1, 0].
// dtype is kept identical to the original perm Const so the op information library still matches.
bool BuildReversePermTensor(int64_t rank, DataType dtype, Tensor& outTensor)
{
    TensorDesc desc(Shape(std::vector<int64_t>{rank}), FORMAT_ND, dtype);
    outTensor.SetTensorDesc(desc);
    if (dtype == DT_INT32) {
        std::vector<int32_t> values(static_cast<size_t>(rank));
        for (int64_t i = 0; i < rank; ++i) {
            values[static_cast<size_t>(i)] = static_cast<int32_t>(rank - 1 - i);
        }
        return outTensor.SetData(reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(int32_t)) ==
               GRAPH_SUCCESS;
    }
    if (dtype == DT_INT64) {
        std::vector<int64_t> values(static_cast<size_t>(rank));
        for (int64_t i = 0; i < rank; ++i) {
            values[static_cast<size_t>(i)] = rank - 1 - i;
        }
        return outTensor.SetData(reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(int64_t)) ==
               GRAPH_SUCCESS;
    }
    OP_LOGW(PASS_NAME.c_str(), "Unsupported perm dtype %d for empty perm, skip node.", static_cast<int32_t>(dtype));
    return false;
}

// Desc for the rewritten perm tensor (1-D of length `rank`). Used to keep shape continuity after
// InferShape: the value rewrite alone leaves the perm desc at its original empty shape {0}, which
// later tiling rejects.
TensorDesc MakePermDesc(int64_t rank, DataType dtype)
{
    const Shape permShape(std::vector<int64_t>{rank});
    TensorDesc desc(permShape, FORMAT_ND, dtype);
    desc.SetShape(permShape);
    desc.SetOriginShape(permShape);
    return desc;
}

// Update the perm tensor desc on both the Const producer and the Transpose's perm input, so the
// inferred/compiled shapes no longer report an empty perm.
bool UpdatePermDescs(const GNode& transposeNode, GNode& constNode, int64_t rank, DataType dtype,
                     const std::string& nodeName)
{
    const TensorDesc newPermDesc = MakePermDesc(rank, dtype);
    if (constNode.UpdateOutputDesc(0, newPermDesc) != GRAPH_SUCCESS) {
        // LCOV_EXCL_START: defensive path, graph desc update cannot fail for a live node
        OP_LOGE(PASS_NAME.c_str(), "Failed to update perm Const output desc for Transpose %s.", nodeName.c_str());
        return false;
        // LCOV_EXCL_STOP
    }
    GNode transposeCopy = transposeNode;
    if (transposeCopy.UpdateInputDesc(PERM_INPUT_IDX, newPermDesc) != GRAPH_SUCCESS) {
        // LCOV_EXCL_START: defensive path, graph desc update cannot fail for a live node
        OP_LOGE(PASS_NAME.c_str(), "Failed to update perm input desc for Transpose %s.", nodeName.c_str());
        return false;
        // LCOV_EXCL_STOP
    }
    return true;
}

// Locate the empty perm Const feeding node's perm input. Returns the Const node on success.
// An empty perm is a Const whose `value` tensor has shape size 0 (the plugin emits shape {0}).
GNodePtr FindEmptyPermConst(const GNode& node, const std::string& nodeName, DataType& dtype)
{
    auto permInput = node.GetInDataNodesAndPortIndexs(PERM_INPUT_IDX);
    GNodePtr constNode = permInput.first;
    if (constNode == nullptr) {
        OP_LOGD(PASS_NAME.c_str(), "Transpose %s perm input has no producer, skip.", nodeName.c_str());
        return nullptr;
    }
    const std::string constType = GetNodeTypeStr(*constNode);
    if (constType != TYPE_CONST && constType != TYPE_CONSTANT) {
        OP_LOGD(PASS_NAME.c_str(), "Transpose %s perm producer type [%s] is not a Const, skip.", nodeName.c_str(),
                constType.c_str());
        return nullptr;
    }

    Tensor permTensor;
    if (constNode->GetAttr(AscendString(ATTR_VALUE), permTensor) != GRAPH_SUCCESS) {
        OP_LOGD(PASS_NAME.c_str(), "Failed to get value attr of Transpose %s perm Const, skip.", nodeName.c_str());
        return nullptr;
    }
    if (permTensor.GetTensorDesc().GetShape().GetShapeSize() != 0) {
        // perm was actually provided; nothing to do.
        return nullptr;
    }
    dtype = permTensor.GetTensorDesc().GetDataType();
    return constNode;
}

// Replace this Transpose's empty perm Const with a dedicated reverse-perm Const node and rewire the
// perm input to it. Uniform semantics: the perm Const is always replaced by a fresh node (never
// mutated in place), so a Const shared by several consumers is safe - each Transpose that needs a
// different reverse perm gets its own Const.
bool ReplaceEmptyPermConstWithReverse(GraphPtr& graph, const GNode& node, const GNode& srcConstNode, int64_t rank,
                                      DataType dtype, const std::string& nodeName)
{
    Tensor reverseTensor;
    if (!BuildReversePermTensor(rank, dtype, reverseTensor)) {
        return false;
    }
    AttrValue valueAttr;
    if (valueAttr.SetAttrValue(reverseTensor) != GRAPH_SUCCESS) {
        // LCOV_EXCL_START: defensive path, AttrValue::SetAttrValue cannot reject a valid tensor
        OP_LOGE(PASS_NAME.c_str(), "Failed to build value attr for the cloned perm Const of Transpose %s.",
                nodeName.c_str());
        return false;
        // LCOV_EXCL_STOP
    }
    const std::string newConstName = GetNodeNameStr(srcConstNode) + "_reverse_perm";
    GNode newConstNode = es::CompliantNodeBuilder(graph.get())
                             .OpType(TYPE_CONST.c_str())
                             .Name(newConstName.c_str())
                             .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                             .IrDefAttrs(
                                 {{ATTR_VALUE, es::CompliantNodeBuilder::kEsAttrOptional, "VT_TENSOR", valueAttr}})
                             .InstanceOutputShape("y", std::vector<int64_t>{rank})
                             .InstanceOutputDataType("y", dtype)
                             .InstanceOutputFormat("y", FORMAT_ND)
                             .Build();

    GNode transposeCopy = node;
    GNode srcConstCopy = srcConstNode;
    if (graph->RemoveEdge(srcConstCopy, 0, transposeCopy, PERM_INPUT_IDX) != GRAPH_SUCCESS) {
        // LCOV_EXCL_START: defensive path, the edge was just resolved from a data node
        OP_LOGE(PASS_NAME.c_str(), "Failed to remove the shared perm edge of Transpose %s.", nodeName.c_str());
        return false;
        // LCOV_EXCL_STOP
    }
    if (graph->AddDataEdge(newConstNode, 0, transposeCopy, PERM_INPUT_IDX) != GRAPH_SUCCESS) {
        // LCOV_EXCL_START: defensive path, the target port was just freed
        OP_LOGE(PASS_NAME.c_str(), "Failed to add the cloned perm edge for Transpose %s.", nodeName.c_str());
        return false;
        // LCOV_EXCL_STOP
    }
    // Keep the perm shape consistent after InferShape (tiling rejects an empty perm shape).
    if (!UpdatePermDescs(node, newConstNode, rank, dtype, nodeName)) {
        return false; // LCOV_EXCL_LINE
    }
    return true;
}

// Rewrite one Transpose node's empty perm so it holds the reverse permutation. Returns true if changed.
// - perm Const has a single consumer: overwrite its value in place (keeps the graph topology).
// - perm Const is shared: clone a dedicated reverse-perm Const and rewire only this Transpose to it.
bool TryFillEmptyPerm(const GNode& node, GraphPtr& graph)
{
    const std::string nodeName = GetNodeNameStr(node);

    if (node.GetInputsSize() < TRANSPOSE_MIN_INPUT_NUM) {
        OP_LOGD(PASS_NAME.c_str(), "Transpose %s has %zu input(s), skip.", nodeName.c_str(), node.GetInputsSize());
        return false;
    }

    TensorDesc xDesc;
    if (node.GetInputDesc(DATA_INPUT_IDX, xDesc) != GRAPH_SUCCESS) {
        // LCOV_EXCL_START: defensive path, a live Transpose always has a data input desc
        OP_LOGD(PASS_NAME.c_str(), "Failed to get input desc of Transpose %s, skip.", nodeName.c_str());
        return false;
        // LCOV_EXCL_STOP
    }
    const int64_t rank = static_cast<int64_t>(xDesc.GetShape().GetDimNum());
    if (rank < 1) {
        OP_LOGD(PASS_NAME.c_str(), "Transpose %s data input rank [%ld] is unknown or scalar, skip.", nodeName.c_str(),
                rank);
        return false;
    }

    DataType dtype = DT_UNDEFINED;
    GNodePtr constNode = FindEmptyPermConst(node, nodeName, dtype);
    if (constNode == nullptr) {
        return false;
    }

    // Always treat the empty perm Const as a replaced node (no shared / in-place distinction).
    if (!ReplaceEmptyPermConstWithReverse(graph, node, *constNode, rank, dtype, nodeName)) {
        return false;
    }

    // The replaced empty perm Const is now dead unless other consumers still reference it.
    if (constNode->GetOutDataNodesAndPortIndexs(0).empty()) {
        GNode oldConstCopy = *constNode;
        if (graph->RemoveNode(oldConstCopy) != GRAPH_SUCCESS) {
            // LCOV_EXCL_START: defensive path, the node has no data consumer at this point
            OP_LOGW(PASS_NAME.c_str(), "Failed to remove the replaced empty perm Const of Transpose %s.",
                    nodeName.c_str());
            // LCOV_EXCL_STOP
        }
    }

    OP_LOGI(PASS_NAME.c_str(), "Replaced empty perm of Transpose %s with a reverse perm Const (rank=%ld, dtype=%d).",
            nodeName.c_str(), rank, static_cast<int32_t>(dtype));
    return true;
}
} // namespace

Status TransposeEmptyPermFusionPass::Run(GraphPtr& graph, CustomPassContext& passContext)
{
    // In real compilation the framework sets the pass name before Run; keep it valid for direct UT
    // invocation with a default-constructed context.
    if (passContext.GetPassName().GetLength() == 0) {
        passContext.SetPassName(PASS_NAME.c_str());
    }
    OP_LOGI(PASS_NAME.c_str(), "TransposeEmptyPermFusionPass fusion start.");

    if (graph == nullptr) {
        OP_LOGE(PASS_NAME.c_str(), "Input graph is null.");
        return FAILED;
    }

    int32_t filledCount = 0;
    for (auto node : graph->GetAllNodes()) {
        if (GetNodeTypeStr(node) != TYPE_TRANSPOSE) {
            continue;
        }
        if (TryFillEmptyPerm(node, graph)) {
            ++filledCount;
        }
    }

    OP_LOGI(PASS_NAME.c_str(), "TransposeEmptyPermFusionPass fusion end, %d node(s) filled.", filledCount);
    return (filledCount > 0) ? SUCCESS : GRAPH_NOT_CHANGED;
}

REG_FUSION_PASS(TransposeEmptyPermFusionPass).Stage(CustomPassStage::kAfterInferShape);

} // namespace ops
