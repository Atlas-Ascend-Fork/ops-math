/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "platform/platform_info.h"
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "es_math_ops.h"
#include "../../../op_graph/fusion_pass/topk_fusion_pass.h"
#include "register/register_custom_pass.h"

using namespace std;
using namespace ge;
using namespace fe;
using namespace ge::fusion;
using namespace ops;

namespace {
const std::string kPassName = "TopKFusionPass";
constexpr int32_t kInputXIdx = 0;
constexpr int32_t kInputKIdx = 1;
constexpr int32_t kOutputValuesIdx = 0;
constexpr int32_t kOutputIndicesIdx = 1;
const char* const kRegbaseArchDav3510 = "3510";
const char* const kRegbaseArchDav5102 = "5102";
const char* const kNonRegbaseArchDav2201 = "2201";

// UT stub for rtGetSocSpec: the UT executable does not link the runtime lib, while the pass gates
// on NpuArch via this API. The stub feeds the arch from the ASCEND_NPU_ARCH env variable (same
// convention as the concat UT and third_party/opbase runtime_stub.cpp). Weak definition: yields
// to any real one.
extern "C" {
typedef int32_t rtError_t;
static const int32_t RT_ERROR_NONE = 0;
rtError_t rtGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen);
}

__attribute__((weak)) rtError_t rtGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen)
{
    const char* arch = std::getenv("ASCEND_NPU_ARCH");
    if (arch == nullptr || std::strcmp(label, "version") != 0 || std::strcmp(key, "NpuArch") != 0) {
        return static_cast<rtError_t>(1);
    }
    (void)std::snprintf(val, maxLen, "%s", arch);
    return RT_ERROR_NONE;
}

static void SetNpuArch(const char* arch)
{
    if (arch != nullptr) {
        (void)setenv("ASCEND_NPU_ARCH", arch, 1);
    } else {
        (void)unsetenv("ASCEND_NPU_ARCH");
    }
}
} // namespace

struct TopkOutHolder {
    es::EsTensorHolder values;
    es::EsTensorHolder indices;
};

class TopKFusionPassTest : public testing::Test {
protected:
    // Default platform: Ascend950 (regbase arch 3510), where the pass is enabled.
    static void SetUpTestCase() { SetPlatformInfo("Ascend950"); }

    void SetUp() override { SetPlatformInfo("Ascend950"); }

    // Set the platform info and the NpuArch env stub together (the arch follows the
    // soc_arch_variables.cmake mapping of the repo). An explicit arch override
    // (e.g. "5102" for MC62CM12A) can be passed via npuArch.
    static void SetPlatformInfo(const std::string& soc, const char* npuArch = nullptr)
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = soc;
        optiCompilationInfo.soc_version = soc;
        PlatformInfoManager::Instance().platform_info_map_[soc] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
        if (npuArch != nullptr) {
            SetNpuArch(npuArch);
        } else {
            SetNpuArch(SocToArch(soc));
        }
    }

    // soc -> npu arch, aligned with cmake/soc_arch_variables.cmake.
    static const char* SocToArch(const std::string& soc)
    {
        if (soc == "Ascend950") {
            return kRegbaseArchDav3510;
        }
        if (soc == "MC62CM12A" || soc.rfind("Ascend630", 0) == 0) {
            return kRegbaseArchDav5102;
        }
        if (soc == "Ascend031") {
            return "3004";
        }
        if (soc.rfind("Ascend310B", 0) == 0) {
            return "3002";
        }
        if (soc.rfind("Ascend310P", 0) == 0) {
            return "2002";
        }
        if (soc.rfind("Ascend610", 0) == 0) {
            return "3102";
        }
        if (soc == "Ascend035") {
            return "3505";
        }
        return kNonRegbaseArchDav2201; // Ascend910 / 910B* / 910_93
    }

    // Build a TopK node with a const k input.
    static TopkOutHolder BuildTopKNodeWithConstK(es::EsGraphBuilder& graphBuilder, const es::EsTensorHolder& x,
                                                 int32_t kValue, bool sorted = true, int64_t dim = -1,
                                                 bool largest = true)
    {
        auto kConst = graphBuilder.CreateConst(std::vector<int32_t>{kValue}, std::vector<int64_t>{1});
        auto output = es::TopK(x, kConst, sorted, largest, dim);
        return {output.values, output.indices};
    }

    // Build a TopK node with a dynamic k input (a graph input, not a const).
    static TopkOutHolder BuildTopKNodeWithDynamicK(es::EsGraphBuilder& graphBuilder, const es::EsTensorHolder& x,
                                                   const es::EsTensorHolder& k, bool sorted = true, int64_t dim = -1,
                                                   bool largest = true)
    {
        auto output = es::TopK(x, k, sorted, largest, dim);
        return {output.values, output.indices};
    }

    // Update the input/output descs of the TopK node so that shape checks in the
    // pass can read valid information. The values/indices output shape keeps the
    // x shape with the dim-th dim replaced by k.
    static void UpdateTopKNodeDesc(const TopkOutHolder& output, const TensorDesc& xDesc, const TensorDesc& kDesc,
                                   int64_t kValue, int64_t dim = -1)
    {
        GNode topkNode = *output.values.GetProducer();
        topkNode.UpdateInputDesc(kInputXIdx, xDesc);
        topkNode.UpdateInputDesc(kInputKIdx, kDesc);

        TensorDesc valuesDesc = xDesc;
        auto dims = xDesc.GetShape().GetDims();
        const int64_t dimSize = static_cast<int64_t>(dims.size());
        int64_t dimAim = dim;
        if (dimAim < 0) {
            dimAim += dimSize;
        }
        if (!dims.empty() && dimAim >= 0 && dimAim < dimSize) {
            dims[static_cast<size_t>(dimAim)] = kValue;
        }
        valuesDesc.SetShape(Shape(dims));
        valuesDesc.SetOriginShape(Shape(dims));
        topkNode.UpdateOutputDesc(kOutputValuesIdx, valuesDesc);

        TensorDesc indicesDesc = valuesDesc;
        indicesDesc.SetDataType(DT_INT32);
        topkNode.UpdateOutputDesc(kOutputIndicesIdx, indicesDesc);
    }

    // Count nodes of the given type in the graph.
    static int32_t CountNodeType(const GraphPtr& graph, const std::string& type)
    {
        int32_t count = 0;
        for (auto node : graph->GetAllNodes()) {
            AscendString nodeType;
            node.GetType(nodeType);
            if (nodeType == type.c_str()) {
                count++;
            }
        }
        return count;
    }

    // Collect the int64 const values of the perm input (input 1) of every Transpose node.
    // The regbase operator information library only registers the two-input Transpose
    // (x + perm const input), so each inserted Transpose must carry an int64 const perm.
    static std::vector<std::vector<int64_t>> CollectTransposePermValues(const GraphPtr& graph)
    {
        std::vector<std::vector<int64_t>> perms;
        for (auto node : graph->GetAllNodes()) {
            AscendString nodeType;
            node.GetType(nodeType);
            if (nodeType != "Transpose") {
                continue;
            }
            Tensor permTensor;
            if (node.GetInputConstData(1, permTensor) != GRAPH_SUCCESS) {
                perms.emplace_back(std::vector<int64_t>{});
                continue;
            }
            const int64_t* permData = reinterpret_cast<const int64_t*>(permTensor.GetData());
            if (permData == nullptr) {
                perms.emplace_back(std::vector<int64_t>{});
                continue;
            }
            const size_t permSize = permTensor.GetSize() / sizeof(int64_t);
            perms.emplace_back(std::vector<int64_t>(permData, permData + permSize));
        }
        return perms;
    }

    // Collect the ListInt perm attr of every TransposeD node (the legacy-platform
    // transposition op carries the perm as an attribute, not as an input).
    static std::vector<std::vector<int64_t>> CollectTransposeDPermAttrs(const GraphPtr& graph)
    {
        std::vector<std::vector<int64_t>> perms;
        for (auto node : graph->GetAllNodes()) {
            AscendString nodeType;
            node.GetType(nodeType);
            if (nodeType != "TransposeD") {
                continue;
            }
            std::vector<int64_t> permAttr;
            if (node.GetAttr("perm", permAttr) == SUCCESS) {
                perms.emplace_back(permAttr);
            } else {
                perms.emplace_back(std::vector<int64_t>{});
            }
        }
        return perms;
    }

    // Read the uint16 fp16 bit-pattern data of the assist_seq input (input 1) of the
    // single TopKD node in the graph. Returns an empty vector when unavailable.
    static std::vector<uint16_t> CollectTopKDAssistSeqData(const GraphPtr& graph)
    {
        for (auto node : graph->GetAllNodes()) {
            AscendString nodeType;
            node.GetType(nodeType);
            if (nodeType != "TopKD") {
                continue;
            }
            Tensor assistTensor;
            if (node.GetInputConstData(1, assistTensor) != GRAPH_SUCCESS) {
                return std::vector<uint16_t>{};
            }
            const uint16_t* assistData = reinterpret_cast<const uint16_t*>(assistTensor.GetData());
            if (assistData == nullptr) {
                return std::vector<uint16_t>{};
            }
            const size_t assistSize = assistTensor.GetSize() / sizeof(uint16_t);
            return std::vector<uint16_t>(assistData, assistData + assistSize);
        }
        return std::vector<uint16_t>{};
    }

    // Collect the given input desc dtypes of all nodes of the given type.
    static std::vector<DataType> CollectNodeTypeInputDtypes(const GraphPtr& graph, const std::string& type,
                                                            int32_t inputIdx)
    {
        std::vector<DataType> dtypes;
        for (auto node : graph->GetAllNodes()) {
            AscendString nodeType;
            node.GetType(nodeType);
            if (nodeType == type.c_str()) {
                TensorDesc desc;
                if (node.GetInputDesc(inputIdx, desc) == SUCCESS) {
                    dtypes.emplace_back(desc.GetDataType());
                }
            }
        }
        return dtypes;
    }

    // Collect the output shapes of all nodes of the given type in the graph.
    static std::vector<std::vector<int64_t>> CollectNodeTypeOutputShapes(const GraphPtr& graph, const std::string& type,
                                                                         int32_t outputIdx)
    {
        std::vector<std::vector<int64_t>> shapes;
        for (auto node : graph->GetAllNodes()) {
            AscendString nodeType;
            node.GetType(nodeType);
            if (nodeType == type.c_str()) {
                TensorDesc desc;
                if (node.GetOutputDesc(outputIdx, desc) == SUCCESS) {
                    shapes.emplace_back(desc.GetShape().GetDims());
                }
            }
        }
        return shapes;
    }

    // Count how many shapes in the list equal the expected one.
    static int32_t CountShape(const std::vector<std::vector<int64_t>>& shapes, const std::vector<int64_t>& expected)
    {
        int32_t count = 0;
        for (const auto& shape : shapes) {
            if (shape == expected) {
                count++;
            }
        }
        return count;
    }

    // Build a TopKV2 source node with an explicit dim attr (frontend-style, e.g. from
    // ONNX/PyTorch adapters). Mirrors the pass's pattern-2 construction but allows a
    // non-default dim for dim!=last test scenarios. sortPolicy >= 0 additionally
    // declares the optional regbase sort_policy attr on the source.
    static TopkOutHolder BuildTopKV2SourceNode(es::EsGraphBuilder& graphBuilder, const es::EsTensorHolder& x,
                                               const es::EsTensorHolder& k, int64_t dim, int64_t sortPolicy = -1)
    {
        auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
        // CompliantNodeBuilder is move-only: build the attr list first, then drive
        // the builder in place (chaining into a named variable would copy-construct).
        std::vector<es::CompliantNodeBuilder::IrAttrDef> attrs = {
            {"sorted", es::CompliantNodeBuilder::kEsAttrRequired, "Bool", es::CreateFrom(true)},
            {"dim", es::CompliantNodeBuilder::kEsAttrRequired, "Int", es::CreateFrom(dim)},
            {"largest", es::CompliantNodeBuilder::kEsAttrRequired, "Bool", es::CreateFrom(true)},
        };
        if (sortPolicy >= 0) {
            attrs.push_back(
                {"sort_policy", es::CompliantNodeBuilder::kEsAttrRequired, "Int", es::CreateFrom(sortPolicy)});
        }
        auto topkV2Node = es::CompliantNodeBuilder(graph)
                              .OpType("TopKV2")
                              .Name("topk_v2_source")
                              .IrDefInputs({
                                  {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"k", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                              })
                              .IrDefOutputs({
                                  {"values", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
                                  {"indices", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
                              })
                              .IrDefAttrs(attrs)
                              .Build();
        if (es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), topkV2Node, kInputXIdx) !=
            GRAPH_SUCCESS) {
            return {es::EsTensorHolder(), es::EsTensorHolder()};
        }
        if (es::AddEdgeAndUpdatePeerDesc(*graph, *k.GetProducer(), k.GetProducerOutIndex(), topkV2Node, kInputKIdx) !=
            GRAPH_SUCCESS) {
            return {es::EsTensorHolder(), es::EsTensorHolder()};
        }
        return {es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(topkV2Node, 0)),
                es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(topkV2Node, 1))};
    }

    // Collect the given int64 attr from all nodes of the given type.
    static std::vector<int64_t> CollectNodeTypeAttr(const GraphPtr& graph, const std::string& type,
                                                    const std::string& attrName)
    {
        std::vector<int64_t> values;
        for (auto node : graph->GetAllNodes()) {
            AscendString nodeType;
            node.GetType(nodeType);
            if (nodeType == type.c_str()) {
                int64_t attrValue = -1;
                if (node.GetAttr(attrName.c_str(), attrValue) == SUCCESS) {
                    values.emplace_back(attrValue);
                }
            }
        }
        return values;
    }

    // Collect the output dtypes of all nodes of the given type.
    static std::vector<DataType> CollectNodeTypeOutputDtypes(const GraphPtr& graph, const std::string& type,
                                                             int32_t outputIdx)
    {
        std::vector<DataType> dtypes;
        for (auto node : graph->GetAllNodes()) {
            AscendString nodeType;
            node.GetType(nodeType);
            if (nodeType == type.c_str()) {
                TensorDesc desc;
                if (node.GetOutputDesc(outputIdx, desc) == SUCCESS) {
                    dtypes.emplace_back(desc.GetDataType());
                }
            }
        }
        return dtypes;
    }

    // Count the occurrences of the given dtype in a dtype list.
    static int32_t CountDtype(const std::vector<DataType>& dtypes, DataType expected)
    {
        int32_t count = 0;
        for (const auto& dtype : dtypes) {
            if (dtype == expected) {
                count++;
            }
        }
        return count;
    }

    // Run the pass on the graph and return the status.
    static Status RunPass(GraphPtr& graph)
    {
        // Use dynamic allocation to avoid CustomPassContextImpl incomplete type destructor issue.
        CustomPassContext* passContextPtr = new CustomPassContext();
        TopKFusionPass pass;
        Status status = pass.Run(graph, *passContextPtr);
        delete passContextPtr;
        return status;
    }
};

// Test 1: pattern creation test
TEST_F(TopKFusionPassTest, patternTest)
{
    TopKFusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_EQ(patterns.size(), 2); // one for TopK, one for TopKV2
}

// Test 2: fusion success - const k on the last dim (dim = -1), float16, 2D.
// Both regbase archs fuse to TopKV2 (k kept as an input, no assist_seq): TopKV2 is the
// only TopK-family op in the regbase operator information library (no TopKV2D/TopKD kernel).
TEST_F(TopKFusionPassTest, fusionSuccessConstKDimLastFp16)
{
    std::vector<int64_t> dimsX{32, 100};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto output = BuildTopKNodeWithConstK(graphBuilder, x, 5);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 5);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 1);     // const k path fuses to TopKV2
    EXPECT_EQ(CountNodeType(graph, "TopK"), 0);       // source node is replaced
    EXPECT_EQ(CountNodeType(graph, "Transpose"), 0);  // dim is last, no transpose
    EXPECT_EQ(CountNodeType(graph, "TransposeD"), 0); // regbase has no TransposeD kernel

    // The values output shape keeps the x shape with the last dim replaced by k.
    auto valuesShapes = CollectNodeTypeOutputShapes(graph, "TopKV2", kOutputValuesIdx);
    EXPECT_EQ(CountShape(valuesShapes, std::vector<int64_t>({32, 5})), 1);

    // Only the source k const remains: no assist_seq const is created (TopKV2 takes none).
    EXPECT_EQ(CountNodeType(graph, "Const"), 1);
}

// Test 3: fusion success - const k with dim != -1, float16, 3D.
// The transposes are two-input Transpose ops (x + int64 const perm), the only
// transposition op in the regbase operator information library.
TEST_F(TopKFusionPassTest, fusionSuccessConstKDimNotLastFp16)
{
    std::vector<int64_t> dimsX{4, 64, 32};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto output = BuildTopKNodeWithConstK(graphBuilder, x, 3, true, 1); // dim = 1

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 3, 1); // dim = 1

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 1);
    EXPECT_EQ(CountNodeType(graph, "TopK"), 0);
    // 1 input transpose + 2 output transposes
    EXPECT_EQ(CountNodeType(graph, "Transpose"), 3);
    EXPECT_EQ(CountNodeType(graph, "TransposeD"), 0); // no attribute-based TransposeD on regbase

    // Every Transpose carries an int64 const perm input with the expected swap perm [0,2,1].
    auto perms = CollectTransposePermValues(graph);
    ASSERT_EQ(perms.size(), 3);
    for (const auto& perm : perms) {
        EXPECT_EQ(perm, std::vector<int64_t>({0, 2, 1}));
    }
    // Const nodes: 1 source k (int32) + 3 perm consts (int64), no assist_seq const.
    EXPECT_EQ(CountNodeType(graph, "Const"), 4);

    // Shape chain check for dimsX={4,64,32}, dim=1, k=3, perm=[0,2,1]:
    //   input Transpose output : {4,32,64} (dim 1 and 2 swapped)
    //   TopKV2 values output   : {4,32,3}  (last dim replaced by k)
    //   output Transpose       : {4,3,32}  (swapped back, equals the source TopK output)
    auto transposeShapes = CollectNodeTypeOutputShapes(graph, "Transpose", kOutputValuesIdx);
    EXPECT_EQ(CountShape(transposeShapes, std::vector<int64_t>({4, 32, 64})), 1);
    EXPECT_EQ(CountShape(transposeShapes, std::vector<int64_t>({4, 3, 32})), 2);
    auto fusedShapes = CollectNodeTypeOutputShapes(graph, "TopKV2", kOutputValuesIdx);
    EXPECT_EQ(CountShape(fusedShapes, std::vector<int64_t>({4, 32, 3})), 1);
}

// Test 4: fusion success - dynamic k on the last dim, float16, 2D
TEST_F(TopKFusionPassTest, fusionSuccessDynamicKDimLastFp16)
{
    std::vector<int64_t> dimsX{16, 128};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto k = graphBuilder.CreateInput(kInputKIdx, "k", DT_INT32, FORMAT_ND, std::vector<int64_t>{1});
    auto output = BuildTopKNodeWithDynamicK(graphBuilder, x, k);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    k.GetProducer()->GetOutputDesc(0, kDesc);
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    k.GetProducer()->UpdateOutputDesc(0, kDesc);
    UpdateTopKNodeDesc(output, xDesc, kDesc, 10);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 1); // dynamic k path fuses to TopKV2
    EXPECT_EQ(CountNodeType(graph, "TopK"), 0);
    EXPECT_EQ(CountNodeType(graph, "Transpose"), 0);
}

// Test 5: fusion success - dynamic k with dim != -1, float16, 4D
TEST_F(TopKFusionPassTest, fusionSuccessDynamicKDimNotLastFp16)
{
    std::vector<int64_t> dimsX{2, 8, 16, 24};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto k = graphBuilder.CreateInput(kInputKIdx, "k", DT_INT32, FORMAT_ND, std::vector<int64_t>{1});
    auto output = BuildTopKNodeWithDynamicK(graphBuilder, x, k, true, 0); // dim = 0

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    k.GetProducer()->GetOutputDesc(0, kDesc);
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    k.GetProducer()->UpdateOutputDesc(0, kDesc);
    UpdateTopKNodeDesc(output, xDesc, kDesc, 4, 0); // dim = 0

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 1);
    EXPECT_EQ(CountNodeType(graph, "TopK"), 0);
    EXPECT_EQ(CountNodeType(graph, "Transpose"), 3);

    // Dynamic-k TopKV2 output descs are set manually (no full-graph infershape is run):
    // for dimsX={2,8,16,24}, dim=0, k=4 the source TopK output shape is {4,8,16,24} and
    // the fused TopKV2 output is the swapped shape with the last dim = k: {24,8,16,4}.
    auto fusedShapes = CollectNodeTypeOutputShapes(graph, "TopKV2", kOutputValuesIdx);
    EXPECT_EQ(CountShape(fusedShapes, std::vector<int64_t>({24, 8, 16, 4})), 1);
}

// Test 6: fusion success - float32 dtype, const k, dim = -1
TEST_F(TopKFusionPassTest, fusionSuccessConstKFloat32)
{
    std::vector<int64_t> dimsX{8, 512};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT, FORMAT_ND, dimsX);
    auto output = BuildTopKNodeWithConstK(graphBuilder, x, 16);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 16);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 1);
}

// Test 7: fusion success - bfloat16 dtype, const k, dim = -1
TEST_F(TopKFusionPassTest, fusionSuccessConstKBf16)
{
    std::vector<int64_t> dimsX{4, 256};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_BF16, FORMAT_ND, dimsX);
    auto output = BuildTopKNodeWithConstK(graphBuilder, x, 8);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_BF16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 8);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 1);
}

// Test 8: unsupported x dtype (double - outside the 11-dtype regbase whitelist;
// int32 is a supported dtype since the 3510 list was aligned to the legacy
// BigGuard set) - no fusion
TEST_F(TopKFusionPassTest, unsupportedDtypeFail)
{
    std::vector<int64_t> dimsX{16, 64};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_DOUBLE, FORMAT_ND, dimsX);
    auto output = BuildTopKNodeWithConstK(graphBuilder, x, 5);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_DOUBLE);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 5);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopK"), 1);    // source node is kept
    EXPECT_EQ(CountNodeType(graph, "TopKV2D"), 0); // no fusion happened
    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 0);
}

// Test 9: sorted = false - no fusion on the legacy archs (on regbase the legacy
// BigGuard branch allows sorted=false, covered by mc62SortedFalseFused and
// ascend950SortedFalseFused)
TEST_F(TopKFusionPassTest, sortedFalseFail)
{
    SetPlatformInfo("Ascend910B1");
    std::vector<int64_t> dimsX{16, 64};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto output = BuildTopKNodeWithConstK(graphBuilder, x, 5, false); // sorted = false

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 5);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopK"), 1);
    EXPECT_EQ(CountNodeType(graph, "TopKV2D"), 0);
    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 0);
}

// Test 10: const k with unsupported dtype (float) - no fusion
TEST_F(TopKFusionPassTest, kInvalidDtypeFail)
{
    std::vector<int64_t> dimsX{16, 64};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto kConst = graphBuilder.CreateConst(std::vector<float>{5.0f}, std::vector<int64_t>{1}); // float k, invalid
    auto out = es::TopK(x, kConst, true, true, -1);
    TopkOutHolder output = {out.values, out.indices};

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_FLOAT);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 5);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopK"), 1);
    EXPECT_EQ(CountNodeType(graph, "TopKV2D"), 0);
    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 0);
}

// Test 11: dim out of range - no fusion
TEST_F(TopKFusionPassTest, dimOutOfRangeFail)
{
    std::vector<int64_t> dimsX{16, 64};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto output = BuildTopKNodeWithConstK(graphBuilder, x, 5, true, 5); // dim = 5, out of range

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 5);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopK"), 1);
    EXPECT_EQ(CountNodeType(graph, "TopKV2D"), 0);
}

// Test 12: fusion on Ascend950 platform - const k, dim = -1
TEST_F(TopKFusionPassTest, fusionSuccessOn950)
{
    SetPlatformInfo("Ascend950");
    std::vector<int64_t> dimsX{32, 100};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto output = BuildTopKNodeWithConstK(graphBuilder, x, 5);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 5);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    // Ascend950 (DAV_3510) is part of the legacy IsRegBase() set: const k also fuses to
    // TopKV2 (the legacy BigGuard behaviour - the regbase op_info has no TopKV2D kernel).
    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 1);
    EXPECT_EQ(CountNodeType(graph, "TopKV2D"), 0);
}

// Test 13: negative dim is normalized (-1 means the last dim)
TEST_F(TopKFusionPassTest, negativeDimNormalizedSuccess)
{
    std::vector<int64_t> dimsX{4, 32, 48};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto output = BuildTopKNodeWithConstK(graphBuilder, x, 6, true, -3); // dim = -3 -> 0

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 6, -3); // dim = -3

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 1);
    EXPECT_EQ(CountNodeType(graph, "Transpose"), 3); // dim 0 is not the last dim

    // Shape chain check for dimsX={4,32,48}, dim=-3->0, k=6, perm=[2,1,0]:
    //   input Transpose output : {48,32,4}
    //   TopKV2 values output   : {48,32,6}
    //   output Transpose       : {6,32,48} (equals the source TopK output)
    auto transposeShapes = CollectNodeTypeOutputShapes(graph, "Transpose", kOutputValuesIdx);
    EXPECT_EQ(CountShape(transposeShapes, std::vector<int64_t>({48, 32, 4})), 1);
    EXPECT_EQ(CountShape(transposeShapes, std::vector<int64_t>({6, 32, 48})), 2);
    auto fusedShapes = CollectNodeTypeOutputShapes(graph, "TopKV2", kOutputValuesIdx);
    EXPECT_EQ(CountShape(fusedShapes, std::vector<int64_t>({48, 32, 6})), 1);
}

// Test 14: compatibility processing (D1 scenario) - the pass inherits the legacy
// built-in TopKFusionPass slot at the kCompatibleInherited stage. This test
// validates that the version guards (Patterns idle guard + MeetRequirements
// runtime check) let the pass work normally on the current GE runtime on both
// regbase archs: DAV_3510 (Ascend950) and DAV_5102 (MC62CM12A) both take over
// with the legacy BigGuard behaviour (TopKV2 target, k kept as an input).
TEST_F(TopKFusionPassTest, compatibleInheritedStageMultiPlatformTest)
{
    // Ascend950 (DAV_3510): const k on the last dim fuses to TopKV2 (legacy BigGuard
    // behaviour - the regbase op_info registers TopKV2 only, no TopKV2D kernel).
    SetPlatformInfo("Ascend950");
    {
        std::vector<int64_t> dimsX{8, 64};
        Shape shapeX(dimsX);

        auto graphBuilder = es::EsGraphBuilder("test_stage_950");
        auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
        auto output = BuildTopKNodeWithConstK(graphBuilder, x, 4);

        TensorDesc xDesc;
        x.GetProducer()->GetOutputDesc(0, xDesc);
        xDesc.SetDataType(DT_FLOAT16);
        xDesc.SetShape(shapeX);
        x.GetProducer()->UpdateOutputDesc(0, xDesc);
        TensorDesc kDesc;
        kDesc.SetDataType(DT_INT32);
        kDesc.SetShape(Shape(std::vector<int64_t>{1}));
        UpdateTopKNodeDesc(output, xDesc, kDesc, 4);

        std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
        Status status = RunPass(graph);
        EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

        EXPECT_EQ(CountNodeType(graph, "TopKV2"), 1);
        EXPECT_EQ(CountNodeType(graph, "TopKV2D"), 0);
        EXPECT_EQ(CountNodeType(graph, "TopK"), 0);
    }

    // MC62CM12A (DAV_5102): the pass takes over with the same legacy BigGuard behaviour -
    // const k also fuses to TopKV2 (k kept as an input, no assist_seq).
    SetPlatformInfo("MC62CM12A");
    {
        std::vector<int64_t> dimsX{16, 96};
        Shape shapeX(dimsX);

        auto graphBuilder = es::EsGraphBuilder("test_stage_5102");
        auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
        auto output = BuildTopKNodeWithConstK(graphBuilder, x, 7);

        TensorDesc xDesc;
        x.GetProducer()->GetOutputDesc(0, xDesc);
        xDesc.SetDataType(DT_FLOAT16);
        xDesc.SetShape(shapeX);
        x.GetProducer()->UpdateOutputDesc(0, xDesc);
        TensorDesc kDesc;
        kDesc.SetDataType(DT_INT32);
        kDesc.SetShape(Shape(std::vector<int64_t>{1}));
        UpdateTopKNodeDesc(output, xDesc, kDesc, 7);

        std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
        Status status = RunPass(graph);
        EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

        EXPECT_EQ(CountNodeType(graph, "TopKV2"), 1);  // const k fuses to TopKV2 on DAV_5102
        EXPECT_EQ(CountNodeType(graph, "TopKV2D"), 0); // TopKV2D has no regbase AICore kernel
        EXPECT_EQ(CountNodeType(graph, "TopK"), 0);

        // The values output shape keeps the x shape with the last dim replaced by k.
        auto valuesShapes = CollectNodeTypeOutputShapes(graph, "TopKV2", kOutputValuesIdx);
        EXPECT_EQ(CountShape(valuesShapes, std::vector<int64_t>({16, 7})), 1);
    }
}

// Test 15: platform gating - a platform without any TopK-family op_info entry
// (Ascend035, NpuArch 3505) must stay silent, keeping the source TopK node for the
// AICPU fallback.
TEST_F(TopKFusionPassTest, nonRegbasePlatformNotFused)
{
    SetPlatformInfo("Ascend035");
    std::vector<int64_t> dimsX{32, 100};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test_non_regbase");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto output = BuildTopKNodeWithConstK(graphBuilder, x, 5);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 5);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopK"), 1);    // source node is kept
    EXPECT_EQ(CountNodeType(graph, "TopKV2D"), 0); // no fusion happened
    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 0);
}

// Test 16: platform gating - when the NpuArch query fails (no arch info available), the
// pass must stay silent instead of fusing on an unknown platform.
TEST_F(TopKFusionPassTest, npuArchQueryFailNotFused)
{
    SetPlatformInfo("Ascend950", nullptr);
    SetNpuArch(nullptr); // unset ASCEND_NPU_ARCH: the rtGetSocSpec stub returns failure
    std::vector<int64_t> dimsX{16, 64};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test_arch_query_fail");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto output = BuildTopKNodeWithConstK(graphBuilder, x, 5);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 5);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopK"), 1);
    EXPECT_EQ(CountNodeType(graph, "TopKV2D"), 0);
}

// Test 17: DAV_5102 const k with dim != last - fuses to TopKV2 wrapped by Transpose x3,
// the two-input Transpose (x + perm const) registered in the regbase op_info.
TEST_F(TopKFusionPassTest, mc62ConstKDimNotLastFused)
{
    SetPlatformInfo("MC62CM12A");
    std::vector<int64_t> dimsX{4, 64, 32};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test_mc62_dim_not_last");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto output = BuildTopKNodeWithConstK(graphBuilder, x, 3, true, 1); // dim = 1

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 3, 1); // dim = 1

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 1);
    EXPECT_EQ(CountNodeType(graph, "TopKV2D"), 0);
    EXPECT_EQ(CountNodeType(graph, "Transpose"), 3);  // 1 input + 2 output transposes
    EXPECT_EQ(CountNodeType(graph, "TransposeD"), 0); // no TransposeD on regbase

    // Every Transpose carries an int64 const perm input with the expected swap perm [0,2,1].
    auto perms = CollectTransposePermValues(graph);
    ASSERT_EQ(perms.size(), 3);
    for (const auto& perm : perms) {
        EXPECT_EQ(perm, std::vector<int64_t>({0, 2, 1}));
    }

    // Shape chain check for dimsX={4,64,32}, dim=1, k=3, perm=[0,2,1]:
    //   input Transpose output : {4,32,64}
    //   TopKV2 values output    : {4,32,3}
    //   output Transpose       : {4,3,32} x2 (equals the source TopK output)
    auto transposeShapes = CollectNodeTypeOutputShapes(graph, "Transpose", kOutputValuesIdx);
    EXPECT_EQ(CountShape(transposeShapes, std::vector<int64_t>({4, 32, 64})), 1);
    EXPECT_EQ(CountShape(transposeShapes, std::vector<int64_t>({4, 3, 32})), 2);
    auto fusedShapes = CollectNodeTypeOutputShapes(graph, "TopKV2", kOutputValuesIdx);
    EXPECT_EQ(CountShape(fusedShapes, std::vector<int64_t>({4, 32, 3})), 1);
}

// Test 18: DAV_5102 extended dtype whitelist - int32 x fuses (rejected on DAV_3510, where
// only fp16/fp32/bf16 are accepted), aligned with the legacy BigGuard 11-dtype list.
TEST_F(TopKFusionPassTest, mc62ExtendedDtypeInt32Fused)
{
    SetPlatformInfo("MC62CM12A");
    std::vector<int64_t> dimsX{8, 48};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test_mc62_int32");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_INT32, FORMAT_ND, dimsX);
    auto output = BuildTopKNodeWithConstK(graphBuilder, x, 5);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_INT32);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 5);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 1); // int32 x fuses on DAV_5102
    EXPECT_EQ(CountNodeType(graph, "TopK"), 0);
}

// Test 18b: DAV_3510 (Ascend950) serves the same full 11-dtype whitelist as the
// legacy BigGuard branch (kTopkSupportDtype applies to 3510 and 5102 alike via
// IsRegBase(); the ascend950 op_info registers the same 11 x dtypes for TopKV2 as
// mc62). The earlier 3-dtype narrowing was a shipped decision dropped in favour
// of legacy alignment - extended dtypes must fuse on 950 as well.
TEST_F(TopKFusionPassTest, ascend950ExtendedDtypeInt32Fused)
{
    SetPlatformInfo("Ascend950");
    std::vector<int64_t> dimsX{8, 48};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test_950_int32");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_INT32, FORMAT_ND, dimsX);
    auto output = BuildTopKNodeWithConstK(graphBuilder, x, 5);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_INT32);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 5);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 1); // int32 x fuses on DAV_3510 too
    EXPECT_EQ(CountNodeType(graph, "TopK"), 0);
}

// Test 19: DAV_5102 allows sorted=false (legacy BigGuard behaviour); DAV_3510 keeps
// rejecting it (covered by sortedFalseFail on the default platform).
TEST_F(TopKFusionPassTest, mc62SortedFalseFused)
{
    SetPlatformInfo("MC62CM12A");
    std::vector<int64_t> dimsX{16, 64};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test_mc62_sorted_false");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto output = BuildTopKNodeWithConstK(graphBuilder, x, 5, false); // sorted = false

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 5);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 1); // sorted=false fuses on DAV_5102
    EXPECT_EQ(CountNodeType(graph, "TopK"), 0);
}

// Test 19b: Ascend950 (DAV_3510) allows sorted=false as well - the legacy BigGuard
// branch (= IsRegBase, both regbase archs) applies the relaxed sorted gate uniformly;
// the earlier isDav5102-only narrowing was dropped in favour of legacy alignment.
TEST_F(TopKFusionPassTest, ascend950SortedFalseFused)
{
    SetPlatformInfo("Ascend950");
    std::vector<int64_t> dimsX{16, 64};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test_950_sorted_false");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto output = BuildTopKNodeWithConstK(graphBuilder, x, 5, false); // sorted = false

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 5);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 1); // sorted=false fuses on DAV_3510 too
    EXPECT_EQ(CountNodeType(graph, "TopK"), 0);
}

// Test 20: the regbase archs keep an int64 const k as-is into TopKV2 (legacy BigGuard
// behaviour - IsRegBase covers 3510 and 5102 alike, and the ascend950 op_info registers
// k as int32/int64, same as mc62; the earlier isDav5102-only narrowing was dropped).
TEST_F(TopKFusionPassTest, mc62ConstKInt64Fused)
{
    SetPlatformInfo("MC62CM12A");
    std::vector<int64_t> dimsX{32, 100};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test_mc62_int64_k");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto kConst = graphBuilder.CreateConst(std::vector<int64_t>{6}, std::vector<int64_t>{1}); // int64 const k
    auto out = es::TopK(x, kConst, true, true, -1);
    TopkOutHolder output = {out.values, out.indices};

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT64);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 6);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 1); // int64 const k fuses on DAV_5102
    EXPECT_EQ(CountNodeType(graph, "TopK"), 0);

    // The values output shape keeps the x shape with the last dim replaced by k.
    auto valuesShapes = CollectNodeTypeOutputShapes(graph, "TopKV2", kOutputValuesIdx);
    EXPECT_EQ(CountShape(valuesShapes, std::vector<int64_t>({32, 6})), 1);
    // The int64 const k is kept as-is (not converted).
    auto constDtypes = CollectNodeTypeOutputDtypes(graph, "Const", 0);
    EXPECT_EQ(CountDtype(constDtypes, DT_INT64), 1);
}

// Test 21: DAV_5102 indices_dtype fix - a frontend TopKV2 source (const int64 k, dim != last)
// with an int64 indices output gets fused, and the rebuilt TopKV2 carries the
// indices_dtype=DT_INT64 attr (legacy BigGuard behaviour). The re-matched rebuilt node is
// skipped idempotently (no infinite loop).
TEST_F(TopKFusionPassTest, mc62TopKV2Int64IndicesAttrFixed)
{
    SetPlatformInfo("MC62CM12A");
    std::vector<int64_t> dimsX{4, 64, 32};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test_mc62_indices_dtype");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto kConst = graphBuilder.CreateConst(std::vector<int64_t>{3}, std::vector<int64_t>{1}); // int64 const k
    auto output = BuildTopKV2SourceNode(graphBuilder, x, kConst, 1);                          // dim = 1

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT64);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    GNode sourceNode = *output.values.GetProducer();
    sourceNode.UpdateInputDesc(kInputXIdx, xDesc);
    sourceNode.UpdateInputDesc(kInputKIdx, kDesc);

    // values output: x shape with dim 1 replaced by k; indices output: int64 (ONNX-style).
    TensorDesc valuesDesc = xDesc;
    valuesDesc.SetShape(Shape(std::vector<int64_t>({4, 3, 32})));
    valuesDesc.SetOriginShape(Shape(std::vector<int64_t>({4, 3, 32})));
    sourceNode.UpdateOutputDesc(kOutputValuesIdx, valuesDesc);
    TensorDesc indicesDesc = valuesDesc;
    indicesDesc.SetDataType(DT_INT64);
    sourceNode.UpdateOutputDesc(kOutputIndicesIdx, indicesDesc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 1); // rebuilt with the fixed attr
    EXPECT_EQ(CountNodeType(graph, "TopKV2D"), 0);
    EXPECT_EQ(CountNodeType(graph, "Transpose"), 3); // dim = 1, transposes inserted

    // The rebuilt TopKV2 carries indices_dtype = DT_INT64 (9).
    auto indicesDtypeAttrs = CollectNodeTypeAttr(graph, "TopKV2", "indices_dtype");
    ASSERT_EQ(indicesDtypeAttrs.size(), 1);
    EXPECT_EQ(indicesDtypeAttrs[0], static_cast<int64_t>(DT_INT64));

    // The final indices output keeps the int64 dtype (restored from the source node).
    // Transpose output-0 dtypes: {fp16 (input side), fp16 (values output side), int64 (indices output side)}.
    auto transposeDtypes = CollectNodeTypeOutputDtypes(graph, "Transpose", kOutputValuesIdx);
    EXPECT_EQ(CountDtype(transposeDtypes, DT_INT64), 1);
    EXPECT_EQ(CountDtype(transposeDtypes, DT_FLOAT16), 2);
}

// Test 22: regbase op-library regression - the replacement graph must only contain
// ops registered in the regbase operator information library:
//   - the transposition is the two-input Transpose (x + int64 const perm), never the
//     attribute-based TransposeD (no regbase AICore kernel - the real-device failure
//     "No operator information library that supports operator ... (operator type:
//     TransposeD) can be found");
//   - the fused op is TopKV2 (x + k), never TopKV2D/TopKD (no regbase kernel either);
//   - the perm input desc of every Transpose is an int64 tensor of the x rank.
TEST_F(TopKFusionPassTest, regbaseOpLibraryCompatibleReplacement)
{
    SetPlatformInfo("Ascend950");
    std::vector<int64_t> dimsX{2, 6, 8};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test_regbase_compat");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto output = BuildTopKNodeWithConstK(graphBuilder, x, 4, true, 2); // dim = 2 is the last dim

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 4, 2);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    // Only regbase-registered ops are generated.
    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 1);
    EXPECT_EQ(CountNodeType(graph, "TopKV2D"), 0);
    EXPECT_EQ(CountNodeType(graph, "TopKD"), 0);
    EXPECT_EQ(CountNodeType(graph, "TransposeD"), 0);

    // The dim-2 (last-dim) source needs no transposes, and only the k const remains.
    EXPECT_EQ(CountNodeType(graph, "Transpose"), 0);
    EXPECT_EQ(CountNodeType(graph, "Const"), 1);

    // Re-run the same scenario with dim = 0 (non-last): three two-input Transposes are
    // inserted and every perm input is an int64 const of the x rank with the swap perm.
    SetPlatformInfo("Ascend950");
    auto graphBuilder2 = es::EsGraphBuilder("test_regbase_compat_not_last");
    auto x2 = graphBuilder2.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto output2 = BuildTopKNodeWithConstK(graphBuilder2, x2, 4, true, 0); // dim = 0

    TensorDesc xDesc2;
    x2.GetProducer()->GetOutputDesc(0, xDesc2);
    xDesc2.SetDataType(DT_FLOAT16);
    xDesc2.SetShape(shapeX);
    x2.GetProducer()->UpdateOutputDesc(0, xDesc2);
    UpdateTopKNodeDesc(output2, xDesc2, kDesc, 4, 0);

    std::shared_ptr<Graph> graph2 = graphBuilder2.BuildAndReset({output2.values, output2.indices});
    Status status2 = RunPass(graph2);
    EXPECT_TRUE(status2 == SUCCESS || status2 == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph2, "TopKV2"), 1);
    EXPECT_EQ(CountNodeType(graph2, "TransposeD"), 0);
    EXPECT_EQ(CountNodeType(graph2, "Transpose"), 3);
    auto perms = CollectTransposePermValues(graph2);
    ASSERT_EQ(perms.size(), 3);
    for (const auto& perm : perms) {
        // rank-3 perm swapping dim 0 with the last dim: [2, 1, 0].
        EXPECT_EQ(perm, std::vector<int64_t>({2, 1, 0}));
    }
}

// Test 22b: the rebuilt regbase TopKV2 carries the source sort_policy attr.
// Dropping it would fall back to the default sorting algorithm and change the
// equal-value tie-break order (sort_policy = 1 selects the bitonic order that
// the kernel st goldens simulate for small-k sorted cases).
TEST_F(TopKFusionPassTest, regbaseTopKV2CarriesSortPolicyAttr)
{
    SetPlatformInfo("Ascend950");
    std::vector<int64_t> dimsX{4, 64, 32};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test_950_sort_policy");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto kConst = graphBuilder.CreateConst(std::vector<int32_t>{3}, std::vector<int64_t>{1});
    auto output = BuildTopKV2SourceNode(graphBuilder, x, kConst, 1, 1); // dim = 1, sort_policy = 1

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    GNode sourceNode = *output.values.GetProducer();
    sourceNode.UpdateInputDesc(kInputXIdx, xDesc);
    sourceNode.UpdateInputDesc(kInputKIdx, kDesc);
    TensorDesc valuesDesc = xDesc;
    valuesDesc.SetShape(Shape(std::vector<int64_t>({4, 3, 32}))); // dim1 replaced by k=3
    valuesDesc.SetOriginShape(Shape(std::vector<int64_t>({4, 3, 32})));
    sourceNode.UpdateOutputDesc(kOutputValuesIdx, valuesDesc);
    TensorDesc indicesDesc = valuesDesc;
    indicesDesc.SetDataType(DT_INT32);
    sourceNode.UpdateOutputDesc(kOutputIndicesIdx, indicesDesc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    // Rebuilt with the transposed last-dim layout (regbase rule).
    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 1);
    EXPECT_EQ(CountNodeType(graph, "Transpose"), 3);

    // The sort_policy attr survives the rebuild, and dim is forced to -1.
    auto policies = CollectNodeTypeAttr(graph, "TopKV2", "sort_policy");
    ASSERT_EQ(policies.size(), 1);
    EXPECT_EQ(policies[0], 1);
    auto dims = CollectNodeTypeAttr(graph, "TopKV2", "dim");
    ASSERT_EQ(dims.size(), 1);
    EXPECT_EQ(dims[0], -1);
}

// ============================================================================
// Legacy tbe platform tests (DAV_2201/3002/2002/3004/3102): the old op_info
// registers TopKD (const k, k becomes an attr + assist_seq input), TopKV2
// (dynamic k, no indices_dtype attr) and the attribute-based TransposeD.
// ============================================================================

// Test 23: Ascend910B const k on the last dim fuses to TopKD (legacy
// ConstToAttrWithNode behaviour): k becomes the required int attr, the assist_seq
// is a 2048-value fp16 const input, and the source k const stays in the graph as
// an isolated node (it leaves the subgraph boundary).
TEST_F(TopKFusionPassTest, legacyConstKFusesToTopKD)
{
    SetPlatformInfo("Ascend910B1");
    std::vector<int64_t> dimsX{32, 100};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test_910b_const_k");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto output = BuildTopKNodeWithConstK(graphBuilder, x, 5);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 5);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopKD"), 1);
    EXPECT_EQ(CountNodeType(graph, "TopK"), 0);
    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 0);
    EXPECT_EQ(CountNodeType(graph, "TransposeD"), 0); // last dim, no transpose

    // The k attr carries the const value (attr_k is required int in the TopKD IR).
    auto kAttrs = CollectNodeTypeAttr(graph, "TopKD", "k");
    ASSERT_EQ(kAttrs.size(), 1);
    EXPECT_EQ(kAttrs[0], 5);

    // The assist_seq input is the regular 2048-value fp16 flavour.
    auto assistData = CollectTopKDAssistSeqData(graph);
    ASSERT_EQ(assistData.size(), 2048U);
    EXPECT_EQ(assistData[0], 0x0000U);    // fp16(0)
    EXPECT_EQ(assistData[1], 0x3C00U);    // fp16(1)
    EXPECT_EQ(assistData[2047], 0x67FFU); // fp16(2047)

    // Const nodes: 1 source k (isolated); the assist_seq is a legacy "Constant"
    // weight node (SetWeights + SetType(..., "Constant") in the legacy pass - the
    // tbe Pre-compile of TopKD requires the Constant channel, a plain "Const"
    // fails the pre-compile with an empty python stack on Ascend910_93).
    EXPECT_EQ(CountNodeType(graph, "Const"), 1);
    EXPECT_EQ(CountNodeType(graph, "Constant"), 1);

    // The values output shape keeps the x shape with the last dim replaced by k.
    auto valuesShapes = CollectNodeTypeOutputShapes(graph, "TopKD", kOutputValuesIdx);
    EXPECT_EQ(CountShape(valuesShapes, std::vector<int64_t>({32, 5})), 1);
    // The indices output is int32 (legacy TopKD IR).
    auto indicesDtypes = CollectNodeTypeOutputDtypes(graph, "TopKD", kOutputIndicesIdx);
    EXPECT_EQ(CountDtype(indicesDtypes, DT_INT32), 1);
}

// Test 24: Ascend910B const k with dim != last - TopKD wrapped by TransposeD x3
// (perm as a ListInt attr, the legacy InsertSingleNode behaviour), with the
// full manually-computed shape chain.
TEST_F(TopKFusionPassTest, legacyConstKDimNotLastTransposeD)
{
    SetPlatformInfo("Ascend910B1");
    std::vector<int64_t> dimsX{4, 64, 32};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test_910b_const_k_not_last");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto output = BuildTopKNodeWithConstK(graphBuilder, x, 3, true, 1); // dim = 1

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 3, 1); // dim = 1

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopKD"), 1);
    EXPECT_EQ(CountNodeType(graph, "TopK"), 0);
    EXPECT_EQ(CountNodeType(graph, "TransposeD"), 3); // 1 input + 2 output transposes
    EXPECT_EQ(CountNodeType(graph, "Transpose"), 0);  // no two-input Transpose on legacy

    // Every TransposeD carries the swap perm [0,2,1] as a ListInt attr.
    auto perms = CollectTransposeDPermAttrs(graph);
    ASSERT_EQ(perms.size(), 3);
    for (const auto& perm : perms) {
        EXPECT_EQ(perm, std::vector<int64_t>({0, 2, 1}));
    }

    // Shape chain check for dimsX={4,64,32}, dim=1, k=3, perm=[0,2,1]:
    //   input TransposeD output  : {4,32,64}
    //   TopKD values output      : {4,32,3}
    //   output TransposeD        : {4,3,32} x2 (equals the source TopK output)
    auto transposeShapes = CollectNodeTypeOutputShapes(graph, "TransposeD", kOutputValuesIdx);
    EXPECT_EQ(CountShape(transposeShapes, std::vector<int64_t>({4, 32, 64})), 1);
    EXPECT_EQ(CountShape(transposeShapes, std::vector<int64_t>({4, 3, 32})), 2);
    auto fusedShapes = CollectNodeTypeOutputShapes(graph, "TopKD", kOutputValuesIdx);
    EXPECT_EQ(CountShape(fusedShapes, std::vector<int64_t>({4, 32, 3})), 1);

    // The dim attr is forced to -1 on the non-last-dim path.
    auto dimAttrs = CollectNodeTypeAttr(graph, "TopKD", "dim");
    ASSERT_EQ(dimAttrs.size(), 1);
    EXPECT_EQ(dimAttrs[0], -1);
}

// Test 25: Ascend910B dynamic k fuses to TopKV2 (k kept as an input) and the
// legacy TopKV2 IR has no indices_dtype attr (attr.list = sorted,dim,largest).
TEST_F(TopKFusionPassTest, legacyDynamicKFusesToTopKV2)
{
    SetPlatformInfo("Ascend910B1");
    std::vector<int64_t> dimsX{16, 128};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test_910b_dynamic_k");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto k = graphBuilder.CreateInput(kInputKIdx, "k", DT_INT32, FORMAT_ND, std::vector<int64_t>{1});
    auto output = BuildTopKNodeWithDynamicK(graphBuilder, x, k);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    k.GetProducer()->GetOutputDesc(0, kDesc);
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    k.GetProducer()->UpdateOutputDesc(0, kDesc);
    UpdateTopKNodeDesc(output, xDesc, kDesc, 10);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 1);
    EXPECT_EQ(CountNodeType(graph, "TopK"), 0);
    EXPECT_EQ(CountNodeType(graph, "TopKD"), 0);

    // No indices_dtype attr on the legacy TopKV2.
    auto indicesDtypeAttrs = CollectNodeTypeAttr(graph, "TopKV2", "indices_dtype");
    EXPECT_EQ(indicesDtypeAttrs.size(), 0);

    // The indices output desc is int32 (the legacy IR fixed output1 dtype).
    auto indicesDtypes = CollectNodeTypeOutputDtypes(graph, "TopKV2", kOutputIndicesIdx);
    EXPECT_EQ(CountDtype(indicesDtypes, DT_INT32), 1);
}

// Test 26: Ascend910B dynamic k with dim != last - TopKV2 wrapped by TransposeD x3;
// the dynamic-k output shape takes the source output's dimAim-th dim as the last dim.
TEST_F(TopKFusionPassTest, legacyDynamicKDimNotLastTransposeD)
{
    SetPlatformInfo("Ascend910B1");
    std::vector<int64_t> dimsX{2, 8, 16, 24};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test_910b_dynamic_k_not_last");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto k = graphBuilder.CreateInput(kInputKIdx, "k", DT_INT32, FORMAT_ND, std::vector<int64_t>{1});
    auto output = BuildTopKNodeWithDynamicK(graphBuilder, x, k, true, 0); // dim = 0

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    k.GetProducer()->GetOutputDesc(0, kDesc);
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    k.GetProducer()->UpdateOutputDesc(0, kDesc);
    UpdateTopKNodeDesc(output, xDesc, kDesc, 4, 0); // dim = 0

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 1);
    EXPECT_EQ(CountNodeType(graph, "TopK"), 0);
    EXPECT_EQ(CountNodeType(graph, "TransposeD"), 3);

    auto perms = CollectTransposeDPermAttrs(graph);
    ASSERT_EQ(perms.size(), 3);
    for (const auto& perm : perms) {
        EXPECT_EQ(perm, std::vector<int64_t>({3, 1, 2, 0})); // swap dim 0 with dim 3
    }

    // Dynamic-k TopKV2 output: swapped shape with the last dim = k (source output dim0).
    auto fusedShapes = CollectNodeTypeOutputShapes(graph, "TopKV2", kOutputValuesIdx);
    EXPECT_EQ(CountShape(fusedShapes, std::vector<int64_t>({24, 8, 16, 4})), 1);
}

// Test 27: Ascend910B rejects sorted=false; Ascend310P rejects bfloat16 (bf16 is a
// DAV_2201-only legacy capability, aligned with the legacy IsSupportBf16).
TEST_F(TopKFusionPassTest, legacySortedFalseAndBf16Gates)
{
    // 910B sorted=false -> no fusion.
    SetPlatformInfo("Ascend910B1");
    {
        auto graphBuilder = es::EsGraphBuilder("test_910b_sorted_false");
        std::vector<int64_t> dimsX{16, 64};
        auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
        auto output = BuildTopKNodeWithConstK(graphBuilder, x, 5, false); // sorted = false

        TensorDesc xDesc;
        x.GetProducer()->GetOutputDesc(0, xDesc);
        xDesc.SetDataType(DT_FLOAT16);
        xDesc.SetShape(Shape(dimsX));
        x.GetProducer()->UpdateOutputDesc(0, xDesc);
        TensorDesc kDesc;
        kDesc.SetDataType(DT_INT32);
        kDesc.SetShape(Shape(std::vector<int64_t>{1}));
        UpdateTopKNodeDesc(output, xDesc, kDesc, 5);

        std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
        Status status = RunPass(graph);
        EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
        EXPECT_EQ(CountNodeType(graph, "TopK"), 1);
        EXPECT_EQ(CountNodeType(graph, "TopKD"), 0);
    }

    // 310P bfloat16 -> no fusion (TopKD/TopKV2 on 310P only take fp16/fp32).
    SetPlatformInfo("Ascend310P1");
    {
        auto graphBuilder = es::EsGraphBuilder("test_310p_bf16");
        std::vector<int64_t> dimsX{8, 64};
        auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_BF16, FORMAT_ND, dimsX);
        auto output = BuildTopKNodeWithConstK(graphBuilder, x, 4);

        TensorDesc xDesc;
        x.GetProducer()->GetOutputDesc(0, xDesc);
        xDesc.SetDataType(DT_BF16);
        xDesc.SetShape(Shape(dimsX));
        x.GetProducer()->UpdateOutputDesc(0, xDesc);
        TensorDesc kDesc;
        kDesc.SetDataType(DT_INT32);
        kDesc.SetShape(Shape(std::vector<int64_t>{1}));
        UpdateTopKNodeDesc(output, xDesc, kDesc, 4);

        std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
        Status status = RunPass(graph);
        EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
        EXPECT_EQ(CountNodeType(graph, "TopK"), 1);
        EXPECT_EQ(CountNodeType(graph, "TopKD"), 0);
    }

    // 910B bfloat16 -> fuses (TopKD/TopKV2 on 910B take fp16/fp32/bf16).
    SetPlatformInfo("Ascend910B1");
    {
        auto graphBuilder = es::EsGraphBuilder("test_910b_bf16");
        std::vector<int64_t> dimsX{8, 64};
        auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_BF16, FORMAT_ND, dimsX);
        auto output = BuildTopKNodeWithConstK(graphBuilder, x, 4);

        TensorDesc xDesc;
        x.GetProducer()->GetOutputDesc(0, xDesc);
        xDesc.SetDataType(DT_BF16);
        xDesc.SetShape(Shape(dimsX));
        x.GetProducer()->UpdateOutputDesc(0, xDesc);
        TensorDesc kDesc;
        kDesc.SetDataType(DT_INT32);
        kDesc.SetShape(Shape(std::vector<int64_t>{1}));
        UpdateTopKNodeDesc(output, xDesc, kDesc, 4);

        std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
        Status status = RunPass(graph);
        EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
        EXPECT_EQ(CountNodeType(graph, "TopKD"), 1);
    }
}

// Test 28: Ascend910B keeps an int64 const k as the TopKD k attr (the legacy pass
// converts the int64 k desc to int32 and reads the low 32 bits; the value is
// identical for all realistic k).
TEST_F(TopKFusionPassTest, legacyConstKInt64FusedToTopKD)
{
    SetPlatformInfo("Ascend910B1");
    std::vector<int64_t> dimsX{32, 100};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test_910b_int64_k");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto kConst = graphBuilder.CreateConst(std::vector<int64_t>{6}, std::vector<int64_t>{1}); // int64 const k
    auto out = es::TopK(x, kConst, true, true, -1);
    TopkOutHolder output = {out.values, out.indices};

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT64);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 6);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopKD"), 1);
    EXPECT_EQ(CountNodeType(graph, "TopK"), 0);

    // The k attr carries the parsed value of the int64 const.
    auto kAttrs = CollectNodeTypeAttr(graph, "TopKD", "k");
    ASSERT_EQ(kAttrs.size(), 1);
    EXPECT_EQ(kAttrs[0], 6);

    // The indices output is normalized to int32 (legacy int64-k desc conversion).
    auto indicesDtypes = CollectNodeTypeOutputDtypes(graph, "TopKD", kOutputIndicesIdx);
    EXPECT_EQ(CountDtype(indicesDtypes, DT_INT32), 1);
}

// Test 29: Ascend910B dynamic int64 k stays silent: the legacy pass faked the
// op_info match by rewriting the in-place k desc to int32, which cannot be carried
// over to the replacement-graph framework (the source k producer keeps its int64
// desc), so fusing would risk an op selection failure - the AICPU TopK fallback
// is the safe equivalent.
TEST_F(TopKFusionPassTest, legacyDynamicKInt64NotFused)
{
    SetPlatformInfo("Ascend910B1");
    std::vector<int64_t> dimsX{16, 128};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test_910b_int64_dynamic_k");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto k = graphBuilder.CreateInput(kInputKIdx, "k", DT_INT64, FORMAT_ND, std::vector<int64_t>{1});
    auto output = BuildTopKNodeWithDynamicK(graphBuilder, x, k);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    k.GetProducer()->GetOutputDesc(0, kDesc);
    kDesc.SetDataType(DT_INT64);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    k.GetProducer()->UpdateOutputDesc(0, kDesc);
    UpdateTopKNodeDesc(output, xDesc, kDesc, 10);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopK"), 1); // stays silent, AICPU fallback
    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 0);
    EXPECT_EQ(CountNodeType(graph, "TopKD"), 0);
}

// Test 30: Ascend310B const k with an unknown-shape x uses the tik assist_seq
// flavour (4096 fp16 values RNE-rounded + 4096 fp16 rounding gaps), the legacy
// AssitHelp(n, out, false) behaviour.
TEST_F(TopKFusionPassTest, tikAssistSeqFlavourOn310B)
{
    SetPlatformInfo("Ascend310B1");
    std::vector<int64_t> dimsX{16, -1}; // unknown shape
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test_310b_tik_assist");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto output = BuildTopKNodeWithConstK(graphBuilder, x, 5);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 5);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopKD"), 1);

    // The tik assist flavour: 4096 + 4096 fp16 values.
    auto assistData = CollectTopKDAssistSeqData(graph);
    ASSERT_EQ(assistData.size(), 8192U);
    // First half: fp16(RNE(i)).
    EXPECT_EQ(assistData[0], 0x0000U);    // fp16(0)
    EXPECT_EQ(assistData[1], 0x3C00U);    // fp16(1)
    EXPECT_EQ(assistData[2047], 0x67FFU); // fp16(2047)
    EXPECT_EQ(assistData[2048], 0x6800U); // fp16(2048) = 2^11
    EXPECT_EQ(assistData[2049], 0x6800U); // RNE(2049) = 2048 (tie to even mantissa)
    EXPECT_EQ(assistData[2051], 0x6802U); // RNE(2051) = 2052
    EXPECT_EQ(assistData[4095], 0x6C00U); // RNE(4095) = 4096 = 2^12 (tie to even mantissa)
    // Second half: fp16(i - RNE(i)) rounding gaps.
    EXPECT_EQ(assistData[4096 + 0], 0x0000U);    // gap(0) = 0
    EXPECT_EQ(assistData[4096 + 2049], 0x3C00U); // gap(2049) = 1 -> fp16(1)
    EXPECT_EQ(assistData[4096 + 2051], 0xBC00U); // gap(2051) = -1 -> fp16(-1)
    EXPECT_EQ(assistData[4096 + 4095], 0xBC00U); // gap(4095) = 4095 - 4096 = -1 -> fp16(-1)

    // A static shape on 310B keeps the regular 2048 flavour (tik needs unknown shape).
    SetPlatformInfo("Ascend310B1");
    auto graphBuilder2 = es::EsGraphBuilder("test_310b_static_assist");
    std::vector<int64_t> dimsX2{16, 100};
    auto x2 = graphBuilder2.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX2);
    auto output2 = BuildTopKNodeWithConstK(graphBuilder2, x2, 5);
    TensorDesc xDesc2;
    x2.GetProducer()->GetOutputDesc(0, xDesc2);
    xDesc2.SetDataType(DT_FLOAT16);
    xDesc2.SetShape(Shape(dimsX2));
    x2.GetProducer()->UpdateOutputDesc(0, xDesc2);
    UpdateTopKNodeDesc(output2, xDesc2, kDesc, 5);
    std::shared_ptr<Graph> graph2 = graphBuilder2.BuildAndReset({output2.values, output2.indices});
    Status status2 = RunPass(graph2);
    EXPECT_TRUE(status2 == SUCCESS || status2 == GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodeType(graph2, "TopKD"), 1);
    auto assistData2 = CollectTopKDAssistSeqData(graph2);
    ASSERT_EQ(assistData2.size(), 2048U);
}

// Test 31: Ascend031 has no TopKV2 entry in its op_info: the dynamic-k path stays
// silent, while the const-k TopKD path still fuses (fp16 only).
TEST_F(TopKFusionPassTest, ascend031DynamicKNotFusedConstKFused)
{
    SetPlatformInfo("Ascend031");
    {
        auto graphBuilder = es::EsGraphBuilder("test_031_dynamic_k");
        std::vector<int64_t> dimsX{16, 64};
        auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
        auto k = graphBuilder.CreateInput(kInputKIdx, "k", DT_INT32, FORMAT_ND, std::vector<int64_t>{1});
        auto output = BuildTopKNodeWithDynamicK(graphBuilder, x, k);

        TensorDesc xDesc;
        x.GetProducer()->GetOutputDesc(0, xDesc);
        xDesc.SetDataType(DT_FLOAT16);
        xDesc.SetShape(Shape(dimsX));
        x.GetProducer()->UpdateOutputDesc(0, xDesc);
        TensorDesc kDesc;
        k.GetProducer()->GetOutputDesc(0, kDesc);
        kDesc.SetDataType(DT_INT32);
        kDesc.SetShape(Shape(std::vector<int64_t>{1}));
        k.GetProducer()->UpdateOutputDesc(0, kDesc);
        UpdateTopKNodeDesc(output, xDesc, kDesc, 5);

        std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
        Status status = RunPass(graph);
        EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
        EXPECT_EQ(CountNodeType(graph, "TopK"), 1); // no TopKV2 entry on 031
        EXPECT_EQ(CountNodeType(graph, "TopKV2"), 0);
        EXPECT_EQ(CountNodeType(graph, "TopKD"), 0);
    }

    SetPlatformInfo("Ascend031");
    {
        auto graphBuilder = es::EsGraphBuilder("test_031_const_k");
        std::vector<int64_t> dimsX{16, 64};
        auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
        auto output = BuildTopKNodeWithConstK(graphBuilder, x, 5);

        TensorDesc xDesc;
        x.GetProducer()->GetOutputDesc(0, xDesc);
        xDesc.SetDataType(DT_FLOAT16);
        xDesc.SetShape(Shape(dimsX));
        x.GetProducer()->UpdateOutputDesc(0, xDesc);
        TensorDesc kDesc;
        kDesc.SetDataType(DT_INT32);
        kDesc.SetShape(Shape(std::vector<int64_t>{1}));
        UpdateTopKNodeDesc(output, xDesc, kDesc, 5);

        std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
        Status status = RunPass(graph);
        EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
        EXPECT_EQ(CountNodeType(graph, "TopKD"), 1); // TopKD is registered on 031
        EXPECT_EQ(CountNodeType(graph, "TopK"), 0);
    }
}

// Test 32: a native TopKV2 source with a const k on a legacy platform is rebuilt
// into TopKD even on the last dim (the legacy ConstToAttrWithNode behaviour; the
// idempotency guard only covers the regbase TopKV2 self-replacement).
TEST_F(TopKFusionPassTest, legacyNativeTopKV2ConstKFusedToTopKD)
{
    SetPlatformInfo("Ascend910B1");
    std::vector<int64_t> dimsX{16, 96};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test_910b_native_topkv2");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto kConst = graphBuilder.CreateConst(std::vector<int32_t>{7}, std::vector<int64_t>{1});
    auto output = BuildTopKV2SourceNode(graphBuilder, x, kConst, -1); // dim = -1 (last)

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    GNode sourceNode = *output.values.GetProducer();
    sourceNode.UpdateInputDesc(kInputXIdx, xDesc);
    sourceNode.UpdateInputDesc(kInputKIdx, kDesc);
    TensorDesc valuesDesc = xDesc;
    valuesDesc.SetShape(Shape(std::vector<int64_t>({16, 7})));
    valuesDesc.SetOriginShape(Shape(std::vector<int64_t>({16, 7})));
    sourceNode.UpdateOutputDesc(kOutputValuesIdx, valuesDesc);
    TensorDesc indicesDesc = valuesDesc;
    indicesDesc.SetDataType(DT_INT32);
    sourceNode.UpdateOutputDesc(kOutputIndicesIdx, indicesDesc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopKD"), 1); // rebuilt into TopKD, not skipped
    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 0);
    auto kAttrs = CollectNodeTypeAttr(graph, "TopKD", "k");
    ASSERT_EQ(kAttrs.size(), 1);
    EXPECT_EQ(kAttrs[0], 7);
}

// Test 33: a native TopKV2 source with a dynamic k and dim != last on a legacy
// platform is rebuilt with dim = -1 and wrapped by TransposeD x3 (the legacy pass
// inserts the transposes for non-last-dim sources regardless of k constness).
TEST_F(TopKFusionPassTest, legacyNativeTopKV2DynamicKDimNotLast)
{
    SetPlatformInfo("Ascend910B1");
    std::vector<int64_t> dimsX{4, 64, 32};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test_910b_native_topkv2_not_last");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto k = graphBuilder.CreateInput(kInputKIdx, "k", DT_INT32, FORMAT_ND, std::vector<int64_t>{1});
    auto output = BuildTopKV2SourceNode(graphBuilder, x, k, 1); // dim = 1

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    k.GetProducer()->GetOutputDesc(0, kDesc);
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    k.GetProducer()->UpdateOutputDesc(0, kDesc);
    GNode sourceNode = *output.values.GetProducer();
    sourceNode.UpdateInputDesc(kInputXIdx, xDesc);
    sourceNode.UpdateInputDesc(kInputKIdx, kDesc);
    TensorDesc valuesDesc = xDesc;
    valuesDesc.SetShape(Shape(std::vector<int64_t>({4, 3, 32}))); // dim1 replaced by k=3
    valuesDesc.SetOriginShape(Shape(std::vector<int64_t>({4, 3, 32})));
    sourceNode.UpdateOutputDesc(kOutputValuesIdx, valuesDesc);
    TensorDesc indicesDesc = valuesDesc;
    indicesDesc.SetDataType(DT_INT32);
    sourceNode.UpdateOutputDesc(kOutputIndicesIdx, indicesDesc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 1); // rebuilt, not skipped
    EXPECT_EQ(CountNodeType(graph, "TopKD"), 0);
    EXPECT_EQ(CountNodeType(graph, "TransposeD"), 3);
    // The rebuilt TopKV2 carries dim = -1.
    auto dimAttrs = CollectNodeTypeAttr(graph, "TopKV2", "dim");
    ASSERT_EQ(dimAttrs.size(), 1);
    EXPECT_EQ(dimAttrs[0], -1);
}

// Test 34: Ascend310P and Ascend610Lite const k fuse to TopKD (fp16/fp32 dtype
// coverage of the remaining legacy archs).
TEST_F(TopKFusionPassTest, legacy310PAnd610LiteConstKFused)
{
    SetPlatformInfo("Ascend310P1");
    {
        auto graphBuilder = es::EsGraphBuilder("test_310p_const_k");
        std::vector<int64_t> dimsX{8, 64};
        auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT, FORMAT_ND, dimsX); // f32 ok on 310P
        auto output = BuildTopKNodeWithConstK(graphBuilder, x, 4);

        TensorDesc xDesc;
        x.GetProducer()->GetOutputDesc(0, xDesc);
        xDesc.SetDataType(DT_FLOAT);
        xDesc.SetShape(Shape(dimsX));
        x.GetProducer()->UpdateOutputDesc(0, xDesc);
        TensorDesc kDesc;
        kDesc.SetDataType(DT_INT32);
        kDesc.SetShape(Shape(std::vector<int64_t>{1}));
        UpdateTopKNodeDesc(output, xDesc, kDesc, 4);

        std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
        Status status = RunPass(graph);
        EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
        EXPECT_EQ(CountNodeType(graph, "TopKD"), 1);
        EXPECT_EQ(CountNodeType(graph, "TopK"), 0);
    }

    SetPlatformInfo("Ascend610Lite");
    {
        auto graphBuilder = es::EsGraphBuilder("test_610lite_const_k");
        std::vector<int64_t> dimsX{8, 64};
        auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
        auto output = BuildTopKNodeWithConstK(graphBuilder, x, 4);

        TensorDesc xDesc;
        x.GetProducer()->GetOutputDesc(0, xDesc);
        xDesc.SetDataType(DT_FLOAT16);
        xDesc.SetShape(Shape(dimsX));
        x.GetProducer()->UpdateOutputDesc(0, xDesc);
        TensorDesc kDesc;
        kDesc.SetDataType(DT_INT32);
        kDesc.SetShape(Shape(std::vector<int64_t>{1}));
        UpdateTopKNodeDesc(output, xDesc, kDesc, 4);

        std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
        Status status = RunPass(graph);
        EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
        EXPECT_EQ(CountNodeType(graph, "TopKD"), 1);
        EXPECT_EQ(CountNodeType(graph, "TopK"), 0);
    }
}

// Test 35: a 0-dim scalar const k (the ONNX TopK K form, empty dims + valid
// dtype) fuses normally on regbase: the scalar desc must be used as-is and
// must NOT trigger the producer fallback - the legacy shape-empty condition
// mistook the scalar for an invalid desc and read the port-0 (x) producer
// (hard-coded), polluting the replacement graph's k Data desc with the x
// dtype (DT_FLOAT) and failing the ATC dispatch of the fused node
// (topk_v2_TopK_*, "The data type DT_FLOAT of the input k is not supported").
TEST_F(TopKFusionPassTest, scalarConstKDimLastFp32Fused)
{
    SetPlatformInfo("Ascend950");
    std::vector<int64_t> dimsX{2, 4};

    auto graphBuilder = es::EsGraphBuilder("test_scalar_const_k");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT, FORMAT_ND, dimsX);
    // 0-dim scalar const k (empty dims), exactly the ONNX TopK K form
    auto kConst = graphBuilder.CreateConst(std::vector<int32_t>{4}, std::vector<int64_t>{});
    auto output = es::TopK(x, kConst, true, true, -1);
    TopkOutHolder outHolder{output.values, output.indices};

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT);
    xDesc.SetShape(Shape(dimsX));
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{})); // scalar: empty dims
    UpdateTopKNodeDesc(outHolder, xDesc, kDesc, 4);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({outHolder.values, outHolder.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 1); // scalar k is a valid desc, fuses
    EXPECT_EQ(CountNodeType(graph, "TopK"), 0);

    // The fused TopKV2's input descs must carry the producer dtypes (x float,
    // k int32), not the CompliantNodeBuilder default (DT_FLOAT) nor the
    // un-written state - a DT_FLOAT k is exactly what fails the ATC dispatch
    // ("The data type DT_FLOAT of the input k is not supported").
    for (auto node : graph->GetAllNodes()) {
        AscendString nodeType;
        node.GetType(nodeType);
        if (nodeType != "TopKV2") {
            continue;
        }
        TensorDesc fusedXDesc;
        TensorDesc fusedKDesc;
        ASSERT_EQ(node.GetInputDesc(kInputXIdx, fusedXDesc), SUCCESS);
        ASSERT_EQ(node.GetInputDesc(kInputKIdx, fusedKDesc), SUCCESS);
        EXPECT_EQ(fusedXDesc.GetDataType(), DT_FLOAT); // x stays float
        EXPECT_EQ(fusedKDesc.GetDataType(), DT_INT32); // k backfilled to int32
    }
}

// Test 36: an unset k input desc (a default-constructed TensorDesc reads
// DT_FLOAT with an empty shape - what ATC-converted graphs leave behind) with
// an int32 const producer resolves the desc from the producer and still
// fuses. Before the producer-fallback fix the default-constructed desc went
// to the dtype gate as-is and was rejected outright ("K only supports
// int32/int64"), and the legacy shape-empty fallback read the port-0 (x)
// producer, polluting the k desc with DT_FLOAT and failing the ATC dispatch
// of the fused node (topk_v2_TopK_*, "The data type DT_FLOAT of the input k
// is not supported").
TEST_F(TopKFusionPassTest, kDescUnsetResolvedFromProducer)
{
    SetPlatformInfo("Ascend950");
    std::vector<int64_t> dimsX{2, 4};

    auto graphBuilder = es::EsGraphBuilder("test_kdesc_unset");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT, FORMAT_ND, dimsX);
    auto kConst = graphBuilder.CreateConst(std::vector<int32_t>{4}, std::vector<int64_t>{1});
    auto output = es::TopK(x, kConst, true, true, -1);
    TopkOutHolder outHolder{output.values, output.indices};

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT);
    xDesc.SetShape(Shape(dimsX));
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    // k node desc intentionally left undefined: only the producer (Const) carries it.
    TensorDesc kDescUndefined;
    UpdateTopKNodeDesc(outHolder, xDesc, kDescUndefined, 4);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({outHolder.values, outHolder.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 1); // desc resolved from the producer, fuses
    EXPECT_EQ(CountNodeType(graph, "TopK"), 0);
}

// Test 37: unknown-rank x ([-2]) with dim = -1 fuses (legacy L617 exemption:
// the dim range check is skipped for unknown-rank shapes; dimAim normalizes to
// the single dim = last dim, so no transposes are inserted).
TEST_F(TopKFusionPassTest, unknownRankDimLastFused)
{
    SetPlatformInfo("Ascend910B1");
    std::vector<int64_t> dimsX{-2}; // unknown rank marker
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test_unknown_rank");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto output = BuildTopKNodeWithConstK(graphBuilder, x, 3);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 3);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodeType(graph, "TopKD"), 1); // unknown rank + dim=-1 still fuses
    EXPECT_EQ(CountNodeType(graph, "TransposeD"), 0);
}

// Test 38: unknown-rank x with an out-of-range explicit dim (5): the legacy pass
// exempts the primary dim check (unknown rank) but aborts at the second dim
// check inside the transpose leg (L844, PARAM_INVALID) - the rebuilt pass keeps
// the source node untouched instead of crashing on the swap indices.
TEST_F(TopKFusionPassTest, unknownRankDimOutOfRangeAborts)
{
    SetPlatformInfo("Ascend910B1");
    std::vector<int64_t> dimsX{-2};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test_unknown_rank_oob");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto output = BuildTopKNodeWithConstK(graphBuilder, x, 3, true, 5); // dim attr = 5, out of rank-1

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 3, 5); // dim = 5, out of rank-1

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    // The second dim check aborts the replacement (nullptr -> framework error),
    // the graph must stay unchanged (no partial replacement, no crash).
    EXPECT_EQ(CountNodeType(graph, "TopK"), 1);
    EXPECT_EQ(CountNodeType(graph, "TransposeD"), 0);
    EXPECT_EQ(CountNodeType(graph, "TopKD"), 0);
}

// Test 39: output-side transposes are only inserted for consumed outputs
// (legacy IsInsertTransposeForOutput): a non-last-dim graph whose indices
// output is dangling gets only the values-side transpose (input + values = 2
// TransposeD), not 3.
TEST_F(TopKFusionPassTest, danglingIndicesOutputSkipsIndicesTranspose)
{
    SetPlatformInfo("Ascend910B1");
    std::vector<int64_t> dimsX{4, 6, 8}; // rank 3, dim = 0 -> non-last
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test_dangling_indices");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto output = BuildTopKNodeWithConstK(graphBuilder, x, 2, true, 0);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 2, 0);

    // Only values is a graph output: the indices output is dangling.
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopKD"), 1);
    // input-side 1 + values-side 1 (indices dangling -> no indices transpose)
    EXPECT_EQ(CountNodeType(graph, "TransposeD"), 2);
}

// Test 40: regbase output-side transposes are UNCONDITIONAL (legacy BigGuard
// InsertTransposeForTopkV2Node inserts both, no IsInsertTransposeForOutput
// check there): even a dangling indices output gets 3 Transposes on DAV_5102.
TEST_F(TopKFusionPassTest, regbaseDanglingIndicesStillInsertsAllTransposes)
{
    SetPlatformInfo("MC62CM12A");
    std::vector<int64_t> dimsX{4, 6, 8}; // rank 3, dim = 0 -> non-last
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test_regbase_dangling_indices");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto output = BuildTopKNodeWithConstK(graphBuilder, x, 2, true, 0);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 2, 0);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values}); // indices dangling
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 1);
    EXPECT_EQ(CountNodeType(graph, "Transpose"), 3); // unconditional on regbase
}

// Test 22c: a regbase TopKV2 source WITHOUT the optional sort_policy attr (e.g.
// an older frontend graph) is rebuilt WITHOUT the attr too - strict canndev
// parity: the legacy in-place SetType preserves the source attribute set, so an
// attr-less source keeps an attr-less node and the kernel runs on the IR
// default (0). Writing a defaulted value here would silently switch the kernel
// algorithm for attr-less sources.
TEST_F(TopKFusionPassTest, regbaseTopKV2WithoutSortPolicyAttrKeepsAttrAbsent)
{
    SetPlatformInfo("Ascend950");
    std::vector<int64_t> dimsX{4, 64, 32};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test_950_sort_policy_default");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto kConst = graphBuilder.CreateConst(std::vector<int32_t>{3}, std::vector<int64_t>{1});
    auto output = BuildTopKV2SourceNode(graphBuilder, x, kConst, 1); // no sort_policy attr on the source

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    GNode sourceNode = *output.values.GetProducer();
    sourceNode.UpdateInputDesc(kInputXIdx, xDesc);
    sourceNode.UpdateInputDesc(kInputKIdx, kDesc);
    TensorDesc valuesDesc = xDesc;
    valuesDesc.SetShape(Shape(std::vector<int64_t>({4, 3, 32})));
    valuesDesc.SetOriginShape(Shape(std::vector<int64_t>({4, 3, 32})));
    sourceNode.UpdateOutputDesc(kOutputValuesIdx, valuesDesc);
    TensorDesc indicesDesc = valuesDesc;
    indicesDesc.SetDataType(DT_INT32);
    sourceNode.UpdateOutputDesc(kOutputIndicesIdx, indicesDesc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopKV2"), 1);
    // Strict canndev parity: no source attr -> no rebuilt attr (IR default 0).
    auto policies = CollectNodeTypeAttr(graph, "TopKV2", "sort_policy");
    EXPECT_EQ(policies.size(), 0);
}

// Test 40: (review fix) non-last-dim dangling indices keeps the fused node's
// INTERNAL desc: without its output transpose the indices output stays the
// transposed last-dim layout [8,6,2] (set in 5.2b) and must NOT be restored
// to the external source shape [2,6,8] - the desc must match the data the
// node actually produces (legacy parity: only the inserted output transposes
// get the external shape back).
TEST_F(TopKFusionPassTest, danglingIndicesKeepsInternalShape)
{
    SetPlatformInfo("Ascend910B1");
    std::vector<int64_t> dimsX{4, 6, 8}; // rank 3, dim = 0 -> non-last
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test_dangling_indices_shape");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto output = BuildTopKNodeWithConstK(graphBuilder, x, 2, true, 0);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 2, 0);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values}); // indices dangling
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopKD"), 1);
    EXPECT_EQ(CountNodeType(graph, "TransposeD"), 2);
    // The dangling indices output keeps the internal transposed layout [8,6,2].
    auto indicesShapes = CollectNodeTypeOutputShapes(graph, "TopKD", kOutputIndicesIdx);
    ASSERT_EQ(indicesShapes.size(), 1);
    EXPECT_EQ(indicesShapes[0], std::vector<int64_t>({8, 6, 2}));
    // The consumed values output (through its transpose) restores [2,6,8] at the
    // graph boundary: the final values output is the transpose's output.
    auto transShapes = CollectNodeTypeOutputShapes(graph, "TransposeD", kOutputValuesIdx);
    EXPECT_EQ(CountShape(transShapes, std::vector<int64_t>({2, 6, 8})), 1);
}

// Test 41: (review fix) the mirror case of Test 40 - only indices consumed:
// the dangling values output keeps the internal [8,6,2] layout.
TEST_F(TopKFusionPassTest, danglingValuesKeepsInternalShape)
{
    SetPlatformInfo("Ascend910B1");
    std::vector<int64_t> dimsX{4, 6, 8};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test_dangling_values_shape");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto output = BuildTopKNodeWithConstK(graphBuilder, x, 2, true, 0);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 2, 0);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.indices}); // values dangling
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopKD"), 1);
    EXPECT_EQ(CountNodeType(graph, "TransposeD"), 2);
    // The dangling values output keeps the internal transposed layout [8,6,2].
    auto valuesShapes = CollectNodeTypeOutputShapes(graph, "TopKD", kOutputValuesIdx);
    ASSERT_EQ(valuesShapes.size(), 1);
    EXPECT_EQ(valuesShapes[0], std::vector<int64_t>({8, 6, 2}));
}

// Test 42: (review fix) the TopKD backend support gate: a sort axis of length
// 1 (x=[2,1], dim=-1) is rejected by the tbe check_supported
// ("data of sort axis is 1") - the pass must stay silent so the original TopK
// keeps its fallback path, instead of building an unselectable TopKD.
TEST_F(TopKFusionPassTest, topkdSortAxisOneNotFused)
{
    SetPlatformInfo("Ascend910B1");
    std::vector<int64_t> dimsX{2, 1}; // last dim (sort axis) length = 1
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test_axis_one");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT, FORMAT_ND, dimsX);
    auto output = BuildTopKNodeWithConstK(graphBuilder, x, 1);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 1);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopK"), 1); // stays silent, backend rejects TopKD
    EXPECT_EQ(CountNodeType(graph, "TopKD"), 0);
}

// Test 43: (review fix) the TopKD backend support gate: input_size > 32768
// with a const k in (0, 8) is rejected on non-910B socs.
TEST_F(TopKFusionPassTest, topkdLargeInputSmallKNotFused)
{
    SetPlatformInfo("Ascend910_93");
    std::vector<int64_t> dimsX{64, 600}; // input_size = 38400 > 32768
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test_large_input_small_k");
    auto x = graphBuilder.CreateInput(kInputXIdx, "x", DT_FLOAT16, FORMAT_ND, dimsX);
    auto output = BuildTopKNodeWithConstK(graphBuilder, x, 4); // k = 4 in (0, 8)

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);
    TensorDesc kDesc;
    kDesc.SetDataType(DT_INT32);
    kDesc.SetShape(Shape(std::vector<int64_t>{1}));
    UpdateTopKNodeDesc(output, xDesc, kDesc, 4);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.values, output.indices});
    Status status = RunPass(graph);
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    EXPECT_EQ(CountNodeType(graph, "TopK"), 1); // stays silent on 910_93 (not 910B)
    EXPECT_EQ(CountNodeType(graph, "TopKD"), 0);
}
