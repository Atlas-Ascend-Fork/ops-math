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
#include <cstdlib>
#include <set>
#include <vector>

#include "es_math_ops.h"
#include "ge/es_graph_builder.h"
#include "ge/ge_utils.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "platform/soc_spec.h"
#include "version/ge-compiler_version.h"
#include "hard_max_fusion_pass.h"

// 运行时接口声明：NpuArch 与 GE 版本查询（aclsysGetVersionNum 为弱符号，旧环境未导出时安全降级）
extern "C" {
typedef int32_t rtError_t;
static const int32_t RT_ERROR_NONE = 0;
rtError_t rtGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen);
__attribute__((weak)) int32_t aclsysGetVersionNum(char* pkgName, int32_t* versionNum);
}

using namespace ge;
using namespace fe;
using namespace fusion;

namespace ops {
namespace es = ge::es;

static const std::string kPassName = "HardMaxPass";
static constexpr int64_t kCaptureHardMax = 0;

#define GE_COMPILER_VERSION_900 90000000

#if GE_COMPILER_VERSION_NUM >= GE_COMPILER_VERSION_900
// GE 版本 >= 9.0.0 用 kCompatibleInherited 同名顶替内建 pass，否则回退 kBeforeInferShape
static CustomPassStage GetHardMaxPassStage()
{
    int32_t version = 0;
    if (aclsysGetVersionNum != nullptr) {
        aclsysGetVersionNum(const_cast<char*>("ge_compiler"), &version);
    }
    if (version >= GE_COMPILER_VERSION_900) {
        return CustomPassStage::kCompatibleInherited;
    }
    return CustomPassStage::kBeforeInferShape;
}
#endif

// 与 Run 一致的运行时版本判定：GE 运行时可用且不低于 9.0.0。
// Patterns 入口也用它守卫，避免低版本运行时提前走到 ES 构图接口
static bool IsSupportedRuntimeVersion()
{
    if (aclsysGetVersionNum == nullptr) {
        return false;
    }
    int32_t version = 0;
    aclsysGetVersionNum(const_cast<char*>("ge_compiler"), &version);
    return version >= GE_COMPILER_VERSION_900;
}

// 连接 srcHolder 的输出到 dstNode 的 dstIdx 输入，producer 为空或连边失败时打日志并返回 false
static bool AddEdgeChecked(Graph& graph, const es::EsTensorHolder& srcHolder, GNode& dstNode, int32_t dstIdx)
{
    GNode* srcNode = srcHolder.GetProducer();
    if (srcNode == nullptr) {
        OP_LOGE(kPassName.c_str(), "Edge source producer is null, dst input index [%d].", dstIdx);
        return false;
    }
    if (es::AddEdgeAndUpdatePeerDesc(graph, *srcNode, srcHolder.GetProducerOutIndex(), dstNode, dstIdx) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to add edge for dst input index [%d].", dstIdx);
        return false;
    }
    return true;
}

// 与旧版 IsRegBase 完全一致：NpuArch 白名单判别 regbase，决定 OneHot/OneHotD 分支
static bool IsRegBasePlatform()
{
    char npuArchVal[32] = {0};
    if (rtGetSocSpec("version", "NpuArch", npuArchVal, sizeof(npuArchVal)) != RT_ERROR_NONE) {
        OP_LOGW(kPassName.c_str(), "Get NpuArch from soc spec failed.");
        return false;
    }
    const auto npuArch = static_cast<NpuArch>(std::strtol(npuArchVal, nullptr, 10));
    // regbase 白名单：DAV_3510（950/350）、DAV_5102（MC62）、DAV_9201（960DT）
    static const std::set<NpuArch> regbaseArch = {NpuArch::DAV_3510, NpuArch::DAV_5102, NpuArch::DAV_9201};
    return regbaseArch.count(npuArch) > 0;
}

// on/off 常量按输入 dtype 匹配，保证 OneHot 输出 dtype 与 HardMax 输出一致
static es::EsTensorHolder CreateOnOffValue(es::EsGraphBuilder& builder, bool on, DataType inputDtype)
{
    if (inputDtype == DT_FLOAT16) {
        const uint16_t fp16Bits = on ? 0x3C00U : 0x0000U; // IEEE754 半精度 1.0/0.0 位模式
        return builder.CreateConstV2<uint16_t>({fp16Bits}, {}, DT_FLOAT16);
    }
    return builder.CreateScalar(on ? 1.0F : 0.0F);
}

// 构建 ArgMaxD 节点（dimension 为 attr，AI Core 算子），desc 对齐老版 ArgMaxD_For_HardMax：
// 输出 = 输入去 axis 维、DT_INT32，format 随输入。连边失败返回 false
static bool BuildArgMaxDNode(Graph& graph, const es::EsTensorHolder& input, int64_t axis, const TensorDesc& inputDesc,
                             GNode& argMaxD)
{
    std::vector<int64_t> indicesDims = inputDesc.GetShape().GetDims();
    indicesDims.erase(indicesDims.begin() + static_cast<std::ptrdiff_t>(axis));
    argMaxD = es::CompliantNodeBuilder(&graph)
                  .OpType("ArgMaxD")
                  .Name("ArgMaxD")
                  .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                  .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                  .IrDefAttrs({{"dimension", es::CompliantNodeBuilder::kEsAttrRequired, "Int", es::CreateFrom(axis)}})
                  .InstanceOutputDataType("y", DT_INT32)
                  .InstanceOutputShape("y", indicesDims)
                  .InstanceOutputOriginShape("y", indicesDims)
                  .InstanceOutputFormat("y", inputDesc.GetFormat())
                  .InstanceOutputOriginFormat("y", inputDesc.GetFormat())
                  .Build();
    return AddEdgeChecked(graph, input, argMaxD, 0);
}

// 对齐老版 CheckOpSupported(ArgMaxD) 的运行时探测：ArgMaxD 为 AI Core 算子，用新框架等价接口
// GeUtils::CheckNodeSupportOnAicore 查询当前平台 ops-info 注册情况，支持则选 ArgMaxD，否则回退
// ArgMaxV2（老版 CreateArgMaxNode 探测回退语义）。探测节点承载与正式构建完全一致的
// dtype/shape/format desc，置于一次性探测图中，不污染替换图。
static bool IsArgMaxDSupported(int64_t axis, const TensorDesc& inputDesc)
{
    auto probeBuilder = es::EsGraphBuilder("argMaxDProbe");
    std::vector<int64_t> probeDims = inputDesc.GetShape().GetDims();
    auto probeInput = probeBuilder.CreateInput(0, "x", inputDesc.GetDataType(), inputDesc.GetFormat(), probeDims);
    Graph* probeGraph = probeBuilder.GetCGraphBuilder()->GetGraph();
    GNode probeNode;
    if (!BuildArgMaxDNode(*probeGraph, probeInput, axis, inputDesc, probeNode)) {
        OP_LOGW(kPassName.c_str(), "Failed to build ArgMaxD probe node. Fall back to ArgMaxV2.");
        return false;
    }
    bool isSupported = false;
    AscendString unsupportedReason;
    const Status checkStatus = GeUtils::CheckNodeSupportOnAicore(probeNode, isSupported, unsupportedReason);
    // AscendString 默认构造时 GetString() 可能为 nullptr，不能直接传给 %s
    const char* reasonStr = (unsupportedReason.GetString() == nullptr) ? "" : unsupportedReason.GetString();
    if (checkStatus != SUCCESS) {
        OP_LOGW(kPassName.c_str(),
                "CheckNodeSupportOnAicore query failed, status=%d, reason=%s. Fall back to ArgMaxV2.",
                static_cast<int32_t>(checkStatus), reasonStr);
        return false;
    }
    if (!isSupported) {
        OP_LOGI(kPassName.c_str(), "ArgMaxD not supported: %s. Fall back to ArgMaxV2.", reasonStr);
    }
    return isSupported;
}

std::vector<PatternUniqPtr> HardMaxPass::Patterns()
{
    std::vector<PatternUniqPtr> patternGraphs;
    // 与 Run 一致的版本守卫：低版本运行时不构建 pattern，避免提前走到 ES 构图接口
    if (!IsSupportedRuntimeVersion()) {
        OP_LOGD(kPassName.c_str(), "GE runtime version is below 9.0.0, skip pass.");
        return patternGraphs;
    }
    auto graphBuilder = es::EsGraphBuilder(kPassName.c_str());
    auto x = graphBuilder.CreateInput(0, "x");
    Graph* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    // HardMax 无 ES 接口，用 CompliantNodeBuilder 按 IR 定义构造 pattern 节点
    GNode hardMax = es::CompliantNodeBuilder(graph)
                        .OpType("HardMax")
                        .Name("HardMax")
                        .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .Build();
    if (!AddEdgeChecked(*graph, x, hardMax, 0)) {
        OP_LOGE(kPassName.c_str(), "Failed to add edge to HardMax pattern node.");
        return patternGraphs;
    }
    auto* hardMaxOutput = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(hardMax, 0);
    if (hardMaxOutput == nullptr) {
        OP_LOGE(kPassName.c_str(), "Failed to get HardMax output holder.");
        return patternGraphs;
    }
    auto output = es::EsTensorHolder(hardMaxOutput);
    auto patternGraph = graphBuilder.BuildAndReset({output});
    auto pattern = std::make_unique<Pattern>(std::move(*patternGraph));
    pattern->CaptureTensor({*output.GetProducer(), 0});
    patternGraphs.emplace_back(std::move(pattern));
    return patternGraphs;
}

// 过滤条件：dtype 仅 FP16/FP32，axis 归一化后须落在 [0, rank) 内
bool HardMaxPass::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    NodeIo nodeIo;
    if (matchResult->GetCapturedTensor(kCaptureHardMax, nodeIo) != SUCCESS) {
        return false;
    }
    TensorDesc inputDesc;
    if (nodeIo.node.GetInputDesc(0, inputDesc) != SUCCESS) {
        return false;
    }
    const DataType inputDtype = inputDesc.GetDataType();
    if (inputDtype != DT_FLOAT16 && inputDtype != DT_FLOAT) {
        return false;
    }
    int64_t axis = 0;
    if (nodeIo.node.GetAttr("axis", axis) != SUCCESS) {
        return false;
    }
    const int64_t rank = static_cast<int64_t>(inputDesc.GetShape().GetDimNum());
    if (axis < 0) {
        axis += rank;
    }
    return axis >= 0 && axis < rank;
}

std::unique_ptr<Graph> HardMaxPass::Replacement(const std::unique_ptr<MatchResult>& matchResult)
{
    std::vector<SubgraphInput> subgraphInputs;
    matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs);
    if (subgraphInputs.empty()) {
        return nullptr;
    }
    NodeIo nodeIo;
    if (matchResult->GetCapturedTensor(kCaptureHardMax, nodeIo) != SUCCESS) {
        return nullptr;
    }
    TensorDesc inputDesc;
    if (nodeIo.node.GetInputDesc(0, inputDesc) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get input desc of HardMax in Replacement.");
        return nullptr;
    }
    int64_t axis = -1;
    if (nodeIo.node.GetAttr("axis", axis) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get attr axis of HardMax in Replacement.");
        return nullptr;
    }
    const int64_t rank = static_cast<int64_t>(inputDesc.GetShape().GetDimNum());
    if (axis < 0) {
        axis += rank;
    }
    const int64_t depth = inputDesc.GetShape().GetDim(static_cast<size_t>(axis));

    auto builder = es::EsGraphBuilder("replacement");
    auto rX = builder.CreateInput(0, "x", inputDesc.GetDataType(), inputDesc.GetFormat(),
                                  inputDesc.GetShape().GetDims());
    auto onValue = CreateOnOffValue(builder, true, inputDesc.GetDataType());
    auto offValue = CreateOnOffValue(builder, false, inputDesc.GetDataType());

    // 两级选型对齐老版 Fusion() 的正交决策，互不干涉：
    // 决策一（老版 IsRegBase）：regbase（950 系）用 OneHot，其余用 OneHotD
    const bool useOneHot = IsRegBasePlatform();
    // 决策二（老版 CheckOpSupported）：运行时探测 ArgMaxD，支持则选之，否则回退 ArgMaxV2；
    // ArgMaxD 为 V1 tbe 算子，仅 8.5.0 等低版本注册，9.x 探测恒回退 ArgMaxV2
    const bool useArgMaxD = IsArgMaxDSupported(axis, inputDesc);

    es::EsTensorHolder indices;
    if (useArgMaxD) {
        Graph* graph = builder.GetCGraphBuilder()->GetGraph();
        GNode argMaxD;
        if (!BuildArgMaxDNode(*graph, rX, axis, inputDesc, argMaxD)) {
            OP_LOGE(kPassName.c_str(), "Failed to build ArgMaxD node in Replacement.");
            return nullptr;
        }
        auto* argMaxDOutput = builder.GetCGraphBuilder()->GetTensorHolderFromNode(argMaxD, 0);
        if (argMaxDOutput == nullptr) {
            OP_LOGE(kPassName.c_str(), "Failed to get ArgMaxD output holder.");
            return nullptr;
        }
        indices = es::EsTensorHolder(argMaxDOutput);
    } else {
        auto rAxis = builder.CreateScalar(static_cast<int32_t>(axis));
        indices = es::ArgMaxV2(rX, rAxis, DT_INT32);
    }

    es::EsTensorHolder output;
    if (useOneHot) {
        // regbase：OneHot（depth 为输入）。OneHot 无 ES 接口，用 CompliantNodeBuilder 构建，
        // 不引入 one_hot 目录 proto 依赖，宿主 CMakeLists 零改动
        auto rDepth = builder.CreateScalar(static_cast<int32_t>(depth));
        auto* graph = builder.GetCGraphBuilder()->GetGraph();
        auto oneHot = es::CompliantNodeBuilder(graph)
                          .OpType("OneHot")
                          .Name("OneHot")
                          .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                        {"depth", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                        {"on_value", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                        {"off_value", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                          .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .IrDefAttrs(
                              {{"axis", es::CompliantNodeBuilder::kEsAttrOptional, "Int", es::CreateFrom(axis)}})
                          .InstanceOutputDataType("y", inputDesc.GetDataType())
                          .InstanceOutputShape("y", inputDesc.GetShape().GetDims())
                          .InstanceOutputOriginShape("y", inputDesc.GetShape().GetDims())
                          .InstanceOutputFormat("y", inputDesc.GetFormat())
                          .InstanceOutputOriginFormat("y", inputDesc.GetFormat())
                          .Build();
        if (!AddEdgeChecked(*graph, indices, oneHot, 0) || !AddEdgeChecked(*graph, rDepth, oneHot, 1) ||
            !AddEdgeChecked(*graph, onValue, oneHot, 2) || !AddEdgeChecked(*graph, offValue, oneHot, 3)) {
            OP_LOGE(kPassName.c_str(), "Failed to build OneHot input edges.");
            return nullptr;
        }
        auto* oneHotOutput = builder.GetCGraphBuilder()->GetTensorHolderFromNode(oneHot, 0);
        if (oneHotOutput == nullptr) {
            OP_LOGE(kPassName.c_str(), "Failed to get OneHot output holder.");
            return nullptr;
        }
        output = es::EsTensorHolder(oneHotOutput);
    } else {
        output = es::OneHotD(indices, onValue, offValue, depth, axis);
    }
    auto replacement = builder.BuildAndReset({output});
    std::vector<Shape> inputShapes;
    for (const auto& subgraphInput : subgraphInputs) {
        auto boundary = subgraphInput.GetAllInputs().at(0);
        TensorDesc boundaryDesc;
        boundary.node.GetInputDesc(boundary.index, boundaryDesc);
        inputShapes.emplace_back(boundaryDesc.GetShape());
    }
    if (GeUtils::InferShape(*replacement, inputShapes) != SUCCESS) {
        return nullptr;
    }
    return replacement;
}

Status HardMaxPass::Run(GraphPtr& graph, CustomPassContext& passContext)
{
    int32_t geVersion = 0;
    const bool versionQueryOk = (aclsysGetVersionNum != nullptr);
    if (versionQueryOk) {
        aclsysGetVersionNum(const_cast<char*>("ge_compiler"), &geVersion);
    } else {
        OP_LOGW(kPassName.c_str(),
                "aclsysGetVersionNum is not resolved, GE runtime version unknown, defer to the built-in pass.");
    }
#if GE_COMPILER_VERSION_NUM >= GE_COMPILER_VERSION_900
    // 运行期低于 9.0.0 时让位给内建 pass，避免新旧 pass 重复融合
    if (!versionQueryOk || geVersion < GE_COMPILER_VERSION_900) {
        OP_LOGD(kPassName.c_str(), "GE runtime version %d < 9.0.0, defer to the built-in pass.", geVersion);
        return GRAPH_NOT_CHANGED;
    }
#else
    OP_LOGD(kPassName.c_str(),
            "compiled with pre-9.0.0 ge headers, kCompatibleInherited unavailable, defer to the built-in pass.");
    return GRAPH_NOT_CHANGED;
#endif
    // 直调 Run（如 UT）时补齐 pass 名，保证融合统计上报正常
    // SetPassName/GetPassName 为 9.0.0 新增接口（D4 场景），必须在版本守卫之后调用
    if (passContext.GetPassName().GetLength() == 0) {
        passContext.SetPassName(kPassName.c_str());
    }
    return PatternFusionPass::Run(graph, passContext);
}

// 注册名与旧版一致，经 kCompatibleInherited 同名顶替内建 HardMaxPass
REG_FUSION_PASS(HardMaxPass)
    .Stage(
#if GE_COMPILER_VERSION_NUM >= GE_COMPILER_VERSION_900
        GetHardMaxPassStage());
#else
        CustomPassStage::kBeforeInferShape);
#endif
} // namespace ops
