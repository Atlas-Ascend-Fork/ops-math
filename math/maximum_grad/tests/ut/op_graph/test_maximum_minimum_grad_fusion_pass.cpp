/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"
#include "ge/es_graph_builder.h"
#include "ge/compliant_node_builder.h"
#include "es_math_ops.h"
#include "log/log.h"
#include "../../../op_graph/fusion_pass/maximum_minimum_grad_fusion_pass.h"

using namespace std;
using namespace ge;
using namespace fe;
using namespace fusion;
using namespace ops;

namespace {

// UT stub for rtGetSocSpec: the UT executable does not link the runtime lib, while the pass
// gates on NpuArch via this API. The stub feeds the arch from the ASCEND_NPU_ARCH env
// variable (same convention as drop_out_v3 / concat_d UTs). Weak definition: yields to any
// real one.
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

static void SetNpuArch(const char* arch) { (void)setenv("ASCEND_NPU_ARCH", arch, 1); }
const std::string kPassName = "MaximumMinimumGradFusionPass";
const std::vector<int64_t> kDims = {4L, 4L};
const std::string kMaximumGradScope = "gradients/Maximum_grad/GreaterEqual";
const std::string kMinimumGradScope = "gradients/Minimum_grad/LessEqual";
} // namespace

class MaximumMinimumGradFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase() { SetPlatform("Ascend950DT_9578"); }

    void SetUp() override { SetPlatform("Ascend950DT_9578"); }

    static void SetPlatform(const std::string& soc)
    {
        // 本机 CANN（platform_info.h）的 platform_info_map_ 为私有：
        // InitializePlatformInfo 从 CANN 平台 ini 加载对应 soc，
        // SetOptionalCompilationInfo 设定当前 soc（GetPlatformInfoWithOutSocVersion 依据它查询）。
        // 平台门控改走 NpuArch(经 rtGetSocSpec stub 从 ASCEND_NPU_ARCH 取值):
        // Ascend950 系 = DAV_3510, Ascend910_93 = DAV_2201(非目标)。
        (void)PlatformInfoManager::Instance().InitializePlatformInfo(soc);
        OptionalInfo optiCompilationInfo;
        optiCompilationInfo.soc_version = soc;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
        SetNpuArch(soc.rfind("Ascend950", 0) == 0 ? "3510" : "2201");
    }

    // Build dz/x1/x2 + zeros(Const) + cmp + Select(s) graph and run the pass.
    // form: "both" -> Select_dx + Select_dy; "dx" -> Select_dx only; "dy" -> Select_dy only.
    // zerosValue: value of the zeros Const (0 = valid gradient pattern).
    // swapDxPorts: swap t/e of Select_dx to build an invalid port-order pattern.
    std::shared_ptr<Graph> BuildGradGraphAndRunPass(const std::string& cmpOpType, const std::string& selectOpType,
                                                    const std::string& cmpNodeName, const std::string& form = "both",
                                                    int32_t zerosValue = 0, bool swapDxPorts = false,
                                                    bool mismatchX1Dtype = false)
    {
        auto graphBuilder = es::EsGraphBuilder("test");
        auto dz = graphBuilder.CreateInput(0, "dz", DT_INT32, FORMAT_ND, kDims);
        auto x1 = graphBuilder.CreateInput(1, "x1", mismatchX1Dtype ? DT_FLOAT : DT_INT32, FORMAT_ND, kDims);
        auto x2 = graphBuilder.CreateInput(2, "x2", DT_INT32, FORMAT_ND, kDims);
        auto zeros = graphBuilder.CreateConst(std::vector<int32_t>(16, zerosValue), kDims);

        auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

        // Select and SelectV2 share the same IR port names (condition/x1/x2 -> y),
        // matching the registered CANN IR definition.
        std::vector<std::string> selectIrInputs = {"condition", "x1", "x2"};
        std::string selectIrOutput = "y";

        auto cmpNode = es::CompliantNodeBuilder(graph)
                           .OpType(cmpOpType.c_str())
                           .Name(cmpNodeName.c_str())
                           .IrDefInputs({{"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                         {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                           .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                           .Build();

        const auto buildSelect = [graph, &selectOpType, &selectIrInputs, &selectIrOutput](const std::string& name) {
            return es::CompliantNodeBuilder(graph)
                .OpType(selectOpType.c_str())
                .Name(name.c_str())
                .IrDefInputs({{selectIrInputs[0], es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                              {selectIrInputs[1], es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                              {selectIrInputs[2], es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                .IrDefOutputs({{selectIrOutput, es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                .Build();
        };

        GNode selectDxNode;
        GNode selectDyNode;
        if (form != "dy") {
            selectDxNode = buildSelect("select_dx");
        }
        if (form != "dx") {
            selectDyNode = buildSelect("select_dy");
        }

        es::AddEdgeAndUpdatePeerDesc(*graph, *x1.GetProducer(), x1.GetProducerOutIndex(), cmpNode, 0);
        es::AddEdgeAndUpdatePeerDesc(*graph, *x2.GetProducer(), x2.GetProducerOutIndex(), cmpNode, 1);
        if (form != "dy") {
            es::AddEdgeAndUpdatePeerDesc(*graph, cmpNode, 0, selectDxNode, 0);
            if (!swapDxPorts) {
                es::AddEdgeAndUpdatePeerDesc(*graph, *dz.GetProducer(), dz.GetProducerOutIndex(), selectDxNode, 1);
                es::AddEdgeAndUpdatePeerDesc(*graph, *zeros.GetProducer(), zeros.GetProducerOutIndex(), selectDxNode,
                                             2);
            } else {
                es::AddEdgeAndUpdatePeerDesc(*graph, *zeros.GetProducer(), zeros.GetProducerOutIndex(), selectDxNode,
                                             1);
                es::AddEdgeAndUpdatePeerDesc(*graph, *dz.GetProducer(), dz.GetProducerOutIndex(), selectDxNode, 2);
            }
        }
        if (form != "dx") {
            es::AddEdgeAndUpdatePeerDesc(*graph, cmpNode, 0, selectDyNode, 0);
            es::AddEdgeAndUpdatePeerDesc(*graph, *zeros.GetProducer(), zeros.GetProducerOutIndex(), selectDyNode, 1);
            es::AddEdgeAndUpdatePeerDesc(*graph, *dz.GetProducer(), dz.GetProducerOutIndex(), selectDyNode, 2);
        }

        std::vector<es::EsTensorHolder> outputs;
        if (form == "both") {
            auto y1 = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(selectDxNode, 0);
            auto y2 = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(selectDyNode, 0);
            outputs = {es::EsTensorHolder(y1), es::EsTensorHolder(y2)};
        } else if (form == "dx") {
            auto y1 = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(selectDxNode, 0);
            outputs = {es::EsTensorHolder(y1)};
        } else {
            auto y2 = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(selectDyNode, 0);
            outputs = {es::EsTensorHolder(y2)};
        }
        std::shared_ptr<Graph> resultGraph = graphBuilder.BuildAndReset(outputs);

        CustomPassContext passContext;
        MaximumMinimumGradFusionPass pass;
        pass.Run(resultGraph, passContext);
        return resultGraph;
    }

    int CountNodeType(const std::shared_ptr<Graph>& graph, const char* typeName)
    {
        int count = 0;
        for (auto node : graph->GetAllNodes()) {
            AscendString type;
            node.GetType(type);
            if (type == typeName) {
                count++;
            }
        }
        return count;
    }

    bool HasNodeType(const std::shared_ptr<Graph>& graph, const char* typeName)
    {
        return CountNodeType(graph, typeName) > 0;
    }

    bool GetFusedGradAttrs(const std::shared_ptr<Graph>& graph, const char* typeName, bool& gradX, bool& gradY)
    {
        for (auto node : graph->GetAllNodes()) {
            AscendString type;
            node.GetType(type);
            if (type == typeName) {
                node.GetAttr("grad_x", gradX);
                node.GetAttr("grad_y", gradY);
                return true;
            }
        }
        return false;
    }
};

namespace {
// Patterns() 为 protected（PatternFusionPass 接口约束），测试侧用子类暴露
class PatternProbePass : public MaximumMinimumGradFusionPass {
public:
    using MaximumMinimumGradFusionPass::Patterns;
};
} // namespace

TEST_F(MaximumMinimumGradFusionPassTest, patternTest)
{
    PatternProbePass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_EQ(patterns.size(), 12UL);
}

TEST_F(MaximumMinimumGradFusionPassTest, fusionSuccessMaximumGradSelect)
{
    auto resultGraph = BuildGradGraphAndRunPass("GreaterEqual", "Select", kMaximumGradScope);
    EXPECT_TRUE(HasNodeType(resultGraph, "MaximumGrad"));
    EXPECT_FALSE(HasNodeType(resultGraph, "GreaterEqual"));
    EXPECT_FALSE(HasNodeType(resultGraph, "Select"));
    bool gradX = false;
    bool gradY = false;
    EXPECT_TRUE(GetFusedGradAttrs(resultGraph, "MaximumGrad", gradX, gradY));
    EXPECT_TRUE(gradX);
    EXPECT_TRUE(gradY);
}

TEST_F(MaximumMinimumGradFusionPassTest, fusionSuccessMinimumGradSelectV2)
{
    auto resultGraph = BuildGradGraphAndRunPass("LessEqual", "SelectV2", kMinimumGradScope);
    EXPECT_TRUE(HasNodeType(resultGraph, "MinimumGrad"));
    EXPECT_FALSE(HasNodeType(resultGraph, "LessEqual"));
    EXPECT_FALSE(HasNodeType(resultGraph, "SelectV2"));
    bool gradX = false;
    bool gradY = false;
    EXPECT_TRUE(GetFusedGradAttrs(resultGraph, "MinimumGrad", gradX, gradY));
    EXPECT_TRUE(gradX);
    EXPECT_TRUE(gradY);
}

TEST_F(MaximumMinimumGradFusionPassTest, fusionSuccessGradientTapeNaming)
{
    // TF2 GradientTape names backward nodes "gradient_tape/GreaterEqual" with
    // no per-op "_grad" scope; the pass must still recognize them as autodiff.
    auto resultGraph = BuildGradGraphAndRunPass("GreaterEqual", "SelectV2", "gradient_tape/GreaterEqual");
    EXPECT_TRUE(HasNodeType(resultGraph, "MaximumGrad"));
    EXPECT_FALSE(HasNodeType(resultGraph, "GreaterEqual"));
    EXPECT_FALSE(HasNodeType(resultGraph, "SelectV2"));
    bool gradX = false;
    bool gradY = false;
    EXPECT_TRUE(GetFusedGradAttrs(resultGraph, "MaximumGrad", gradX, gradY));
    EXPECT_TRUE(gradX);
    EXPECT_TRUE(gradY);
}

TEST_F(MaximumMinimumGradFusionPassTest, fusionSuccessGradientTapeIndexedNaming)
{
    // A second tape in the same trace is named "gradient_tape_1/...".
    auto resultGraph = BuildGradGraphAndRunPass("LessEqual", "Select", "model/gradient_tape_1/LessEqual");
    EXPECT_TRUE(HasNodeType(resultGraph, "MinimumGrad"));
    EXPECT_FALSE(HasNodeType(resultGraph, "LessEqual"));
    EXPECT_FALSE(HasNodeType(resultGraph, "Select"));
}

TEST_F(MaximumMinimumGradFusionPassTest, fusionSuccessDxOnly)
{
    auto resultGraph = BuildGradGraphAndRunPass("GreaterEqual", "Select", kMaximumGradScope, "dx");
    EXPECT_TRUE(HasNodeType(resultGraph, "MaximumGrad"));
    EXPECT_FALSE(HasNodeType(resultGraph, "GreaterEqual"));
    EXPECT_FALSE(HasNodeType(resultGraph, "Select"));
    bool gradX = false;
    bool gradY = false;
    EXPECT_TRUE(GetFusedGradAttrs(resultGraph, "MaximumGrad", gradX, gradY));
    EXPECT_TRUE(gradX);
    EXPECT_FALSE(gradY);
}

TEST_F(MaximumMinimumGradFusionPassTest, fusionSuccessDyOnly)
{
    auto resultGraph = BuildGradGraphAndRunPass("GreaterEqual", "Select", kMaximumGradScope, "dy");
    EXPECT_TRUE(HasNodeType(resultGraph, "MaximumGrad"));
    EXPECT_FALSE(HasNodeType(resultGraph, "GreaterEqual"));
    EXPECT_FALSE(HasNodeType(resultGraph, "Select"));
    bool gradX = false;
    bool gradY = false;
    EXPECT_TRUE(GetFusedGradAttrs(resultGraph, "MaximumGrad", gradX, gradY));
    EXPECT_FALSE(gradX);
    EXPECT_TRUE(gradY);
}

TEST_F(MaximumMinimumGradFusionPassTest, dyOnlySelectV2)
{
    auto resultGraph = BuildGradGraphAndRunPass("LessEqual", "SelectV2", kMinimumGradScope, "dy");
    EXPECT_TRUE(HasNodeType(resultGraph, "MinimumGrad"));
    bool gradX = false;
    bool gradY = false;
    EXPECT_TRUE(GetFusedGradAttrs(resultGraph, "MinimumGrad", gradX, gradY));
    EXPECT_FALSE(gradX);
    EXPECT_TRUE(gradY);
}

// 12 pattern 全矩阵补齐: GE/LE x Select/SelectV2 x both/dx/dy 中此前缺的 5 个组合。
TEST_F(MaximumMinimumGradFusionPassTest, maximumGradSelectV2DxOnly)
{
    auto resultGraph = BuildGradGraphAndRunPass("GreaterEqual", "SelectV2", kMaximumGradScope, "dx");
    EXPECT_TRUE(HasNodeType(resultGraph, "MaximumGrad"));
    EXPECT_FALSE(HasNodeType(resultGraph, "GreaterEqual"));
    EXPECT_FALSE(HasNodeType(resultGraph, "SelectV2"));
    bool gradX = false;
    bool gradY = false;
    EXPECT_TRUE(GetFusedGradAttrs(resultGraph, "MaximumGrad", gradX, gradY));
    EXPECT_TRUE(gradX);
    EXPECT_FALSE(gradY);
}

TEST_F(MaximumMinimumGradFusionPassTest, maximumGradSelectV2DyOnly)
{
    auto resultGraph = BuildGradGraphAndRunPass("GreaterEqual", "SelectV2", kMaximumGradScope, "dy");
    EXPECT_TRUE(HasNodeType(resultGraph, "MaximumGrad"));
    EXPECT_FALSE(HasNodeType(resultGraph, "GreaterEqual"));
    EXPECT_FALSE(HasNodeType(resultGraph, "SelectV2"));
    bool gradX = false;
    bool gradY = false;
    EXPECT_TRUE(GetFusedGradAttrs(resultGraph, "MaximumGrad", gradX, gradY));
    EXPECT_FALSE(gradX);
    EXPECT_TRUE(gradY);
}

TEST_F(MaximumMinimumGradFusionPassTest, minimumGradSelectDxOnly)
{
    auto resultGraph = BuildGradGraphAndRunPass("LessEqual", "Select", kMinimumGradScope, "dx");
    EXPECT_TRUE(HasNodeType(resultGraph, "MinimumGrad"));
    EXPECT_FALSE(HasNodeType(resultGraph, "LessEqual"));
    EXPECT_FALSE(HasNodeType(resultGraph, "Select"));
    bool gradX = false;
    bool gradY = false;
    EXPECT_TRUE(GetFusedGradAttrs(resultGraph, "MinimumGrad", gradX, gradY));
    EXPECT_TRUE(gradX);
    EXPECT_FALSE(gradY);
}

TEST_F(MaximumMinimumGradFusionPassTest, minimumGradSelectDyOnly)
{
    auto resultGraph = BuildGradGraphAndRunPass("LessEqual", "Select", kMinimumGradScope, "dy");
    EXPECT_TRUE(HasNodeType(resultGraph, "MinimumGrad"));
    EXPECT_FALSE(HasNodeType(resultGraph, "LessEqual"));
    EXPECT_FALSE(HasNodeType(resultGraph, "Select"));
    bool gradX = false;
    bool gradY = false;
    EXPECT_TRUE(GetFusedGradAttrs(resultGraph, "MinimumGrad", gradX, gradY));
    EXPECT_FALSE(gradX);
    EXPECT_TRUE(gradY);
}

TEST_F(MaximumMinimumGradFusionPassTest, minimumGradSelectV2DxOnly)
{
    auto resultGraph = BuildGradGraphAndRunPass("LessEqual", "SelectV2", kMinimumGradScope, "dx");
    EXPECT_TRUE(HasNodeType(resultGraph, "MinimumGrad"));
    EXPECT_FALSE(HasNodeType(resultGraph, "LessEqual"));
    EXPECT_FALSE(HasNodeType(resultGraph, "SelectV2"));
    bool gradX = false;
    bool gradY = false;
    EXPECT_TRUE(GetFusedGradAttrs(resultGraph, "MinimumGrad", gradX, gradY));
    EXPECT_TRUE(gradX);
    EXPECT_FALSE(gradY);
}

TEST_F(MaximumMinimumGradFusionPassTest, unsupportedPlatformFail)
{
    SetPlatform("Ascend910_93");
    auto resultGraph = BuildGradGraphAndRunPass("GreaterEqual", "Select", kMaximumGradScope);
    EXPECT_FALSE(HasNodeType(resultGraph, "MaximumGrad"));
    EXPECT_TRUE(HasNodeType(resultGraph, "GreaterEqual"));
    EXPECT_TRUE(HasNodeType(resultGraph, "Select"));
}

TEST_F(MaximumMinimumGradFusionPassTest, scopeMismatchFail)
{
    auto resultGraph = BuildGradGraphAndRunPass("GreaterEqual", "Select", "foo/bar/GreaterEqual");
    EXPECT_FALSE(HasNodeType(resultGraph, "MaximumGrad"));
    EXPECT_TRUE(HasNodeType(resultGraph, "GreaterEqual"));
}

TEST_F(MaximumMinimumGradFusionPassTest, nonZeroZerosFail)
{
    auto resultGraph = BuildGradGraphAndRunPass("GreaterEqual", "Select", kMaximumGradScope, "both", 1);
    EXPECT_FALSE(HasNodeType(resultGraph, "MaximumGrad"));
    EXPECT_TRUE(HasNodeType(resultGraph, "GreaterEqual"));
}

// 检视意见: 三输入 dtype 不一致(grads/x2=DT_INT32, x1=DT_FLOAT)须在匹配阶段拒绝,
// 避免生成必然 InferShape/InferDataType 失败的坏替换图。
TEST_F(MaximumMinimumGradFusionPassTest, inputDtypeMismatchFail)
{
    auto resultGraph = BuildGradGraphAndRunPass("GreaterEqual", "Select", kMaximumGradScope, "both", 0, false, true);
    EXPECT_FALSE(HasNodeType(resultGraph, "MaximumGrad"));
    EXPECT_TRUE(HasNodeType(resultGraph, "GreaterEqual"));
    EXPECT_TRUE(HasNodeType(resultGraph, "Select"));
}

TEST_F(MaximumMinimumGradFusionPassTest, nonZeroZerosDxOnlyFail)
{
    auto resultGraph = BuildGradGraphAndRunPass("GreaterEqual", "Select", kMaximumGradScope, "dx", 1);
    EXPECT_FALSE(HasNodeType(resultGraph, "MaximumGrad"));
    EXPECT_TRUE(HasNodeType(resultGraph, "GreaterEqual"));
}

TEST_F(MaximumMinimumGradFusionPassTest, swappedDxPortsFail)
{
    auto resultGraph = BuildGradGraphAndRunPass("GreaterEqual", "Select", kMaximumGradScope, "both", 0, true);
    EXPECT_FALSE(HasNodeType(resultGraph, "MaximumGrad"));
    EXPECT_TRUE(HasNodeType(resultGraph, "GreaterEqual"));
}
