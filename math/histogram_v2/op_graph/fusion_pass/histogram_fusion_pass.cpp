/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file histogram_fusion_pass.cpp
 * \brief histogram fusion pass (Histogram --> HistogramV2)
 *
 * Pattern:
 *              x                     x, min, max
 *              |                     |
 *        Histogram    ==>    HistogramV2
 *              |                     |
 *              y                     y
 *
 * Key differences:
 * - Histogram: min/max are attributes (float32, 精度天花板为 IR 层结构性限制)
 * - HistogramV2: min/max are input tensors
 * - min/max tensors dtype should match x's dtype
 * - y_dtype attr is selected from x dtype (DT_FLOAT for float/f16)
 *
 * CUDA 语义对齐（min/max/bins 为 attr，编译期已知，全部静态决策）:
 * - min == max（含默认 0/0、同号 ±inf）触发 auto-range: 插 ReduceMin/ReduceMax
 *   子图重算数据范围；常数数据重算后仍相等时由 kernel ExpandMinMaxIfEqual 按
 *   CPU #163506 语义做 nextafter/±1 展开兜底。
 * - min > max / bins <= 0 / 有限性 / int64 转换溢出: 响亮报错（OP_LOGE + 返回
 *   nullptr，残留 Histogram 无 kernel，图编译失败，绝不静默出错误结果）。
 * - 标量超出 x dtype 值域（M3a）: x 与标量共同提升到能容纳标量的最窄类型
 *   （对齐 aclnn WidenDtypeForScalars）。
 * - 空输入: 静态 numel == 0 时不建 reduce，HistogramV2 kernel 输出全零
 *   （符合 CUDA 空输入语义）。
 * - 输出 dtype: HistogramV2 输出域为 {int32, float}，尾部 Cast 回源 Histogram
 *   输出 desc dtype（对齐 torch histc out dtype == self dtype 的使用方式）。
 */

#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include "es_math_ops.h"
#include "ge/es_graph_builder.h"
#include "platform/platform_info.h"
#include "ge/ge_utils.h"
#include "log/log.h"
#include "histogram_fusion_pass.h"

using namespace ge;
using namespace fe;
using namespace fusion;

namespace ops {

static const std::string kPassName = "HistogramFusionPass";
static const int64_t kCaptureHistogramNode = 0l;

std::vector<PatternUniqPtr> HistogramFusionPass::Patterns()
{
    OP_LOGD(kPassName.c_str(), "Enter Patterns for HistogramFusionPass");
    std::vector<PatternUniqPtr> patternGraphs;

    auto graphBuilder = es::EsGraphBuilder(kPassName.c_str());

    auto x = graphBuilder.CreateInput(0);

    auto output = es::Histogram(x, 100, 0.0f, 0.0f);

    auto graph = graphBuilder.BuildAndReset({output});

    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*output.GetProducer(), 0});

    patternGraphs.emplace_back(std::move(pattern));
    return patternGraphs;
}

bool HistogramFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& match_result)
{
    OP_LOGD(kPassName.c_str(), "Enter MeetRequirements for HistogramFusionPass");

    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Get platformInfo failed.");
        return false;
    }
    const std::string soc = platformInfo.str_info.short_soc_version;
    // 910b/310p 等旧芯片不走融合（HistogramV2 新 kernel 仅 regbase 系列有）；960DT/960PR
    // 与 950/350 同属 arch35 家族，def 已注册且 SIMT tiling 覆盖（DAV_9201/9202），同步放开
    if (soc != "Ascend950" && soc != "Ascend350" && soc != "Ascend960DT" && soc != "Ascend960PR") {
        OP_LOGD(kPassName.c_str(), "Platform %s is not supported, only Ascend950/Ascend350/Ascend960DT/Ascend960PR.",
                soc.c_str());
        return false;
    }

    NodeIo histogramNodeIo;
    if (match_result->GetCapturedTensor(kCaptureHistogramNode, histogramNodeIo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to GetCaptured tensor");
        return false;
    }
    AscendString nodeType;
    histogramNodeIo.node.GetType(nodeType);
    std::string typeStr = nodeType.GetString();
    if (typeStr != "Histogram") {
        OP_LOGD(kPassName.c_str(), "Node type %s is not Histogram, skip.", typeStr.c_str());
        return false;
    }

    TensorDesc inputDesc;
    histogramNodeIo.node.GetInputDesc(0, inputDesc);
    DataType inputDtype = inputDesc.GetDataType();
    if (inputDtype != DT_FLOAT16 && inputDtype != DT_FLOAT && inputDtype != DT_BF16 && inputDtype != DT_INT64 &&
        inputDtype != DT_INT32 && inputDtype != DT_INT16 && inputDtype != DT_INT8 && inputDtype != DT_UINT8) {
        OP_LOGD(kPassName.c_str(), "Input dtype %d not supported, skip.", inputDtype);
        return false;
    }

    TensorDesc outputDesc;
    histogramNodeIo.node.GetOutputDesc(0, outputDesc);
    DataType outputDtype = outputDesc.GetDataType();
    // Histogram proto 输出全集 {fp16, fp32, bf16, int64, int32, int16, int8, uint8}；HistogramV2 输出域
    // 为 {int32, float}，其余由 Replacement 尾部 Cast 兜回源输出 desc dtype。
    if (outputDtype != DT_FLOAT16 && outputDtype != DT_FLOAT && outputDtype != DT_BF16 && outputDtype != DT_INT64 &&
        outputDtype != DT_INT32 && outputDtype != DT_INT16 && outputDtype != DT_INT8 && outputDtype != DT_UINT8) {
        OP_LOGD(kPassName.c_str(), "Output dtype %d is not supported by Histogram, skip fusion.", outputDtype);
        return false;
    }

    return true;
}

namespace {
// 与 HistogramV2 proto 输入集一致
bool IsIntDtype(ge::DataType dtype)
{
    return dtype == ge::DT_INT64 || dtype == ge::DT_INT32 || dtype == ge::DT_INT16 || dtype == ge::DT_INT8 ||
           dtype == ge::DT_UINT8;
}

// CUDA checked_convert 语义: float payload 向零截断转 int64; NaN 或 |v| >= 2^63 失败
bool TryConvertToInt64(float value, int64_t& out)
{
    constexpr float kInt64HiF32 = 9223372036854775808.0f; // 2^63, float 精确可表示
    if (std::isnan(value) || value >= kInt64HiF32 || value < -kInt64HiF32) {
        return false;
    }
    out = static_cast<int64_t>(std::trunc(value));
    return true;
}

// 整数 x: 标量(截断后的 int64)超出 x 值域时的提升决策, 移植自 aclnn WidenIntDtypeForScalars
ge::DataType WidenIntDtype(int64_t minValue, int64_t maxValue, int64_t xLo, int64_t xHi)
{
    if (minValue >= xLo && maxValue <= xHi) {
        return ge::DT_UNDEFINED; // 标量在 x 值域内, 无需提升
    }
    if (minValue >= INT32_MIN && maxValue <= INT32_MAX) {
        return ge::DT_INT32;
    }
    return ge::DT_INT64;
}

// M3a 提升决策表(移植自 aclnn WidenDtypeForScalars): int64 无更宽类型, fp32 不需要提升
ge::DataType WidenDtypeForScalars(ge::DataType xDtype, int64_t minI64, int64_t maxI64, float minF32, float maxF32)
{
    switch (xDtype) {
        case ge::DT_INT8:
            return WidenIntDtype(minI64, maxI64, INT8_MIN, INT8_MAX);
        case ge::DT_UINT8:
            return WidenIntDtype(minI64, maxI64, 0, UINT8_MAX);
        case ge::DT_INT16:
            return WidenIntDtype(minI64, maxI64, INT16_MIN, INT16_MAX);
        case ge::DT_INT32:
            return WidenIntDtype(minI64, maxI64, INT32_MIN, INT32_MAX);
        case ge::DT_FLOAT16: {
            // fp16 是 NPU 扩展支持(CUDA 不支持), 标量超 fp16 值域(65504)时提升到 fp32
            constexpr float kFp16Max = 65504.0f;
            if (std::fabs(minF32) > kFp16Max || std::fabs(maxF32) > kFp16Max) {
                return ge::DT_FLOAT;
            }
            return ge::DT_UNDEFINED;
        }
        case ge::DT_BF16: {
            // bf16 同理(NPU 扩展), 标量超 bf16 值域(~3.39e38)时提升到 fp32
            constexpr float kBf16Max = 3.3895314e38f;
            if (std::fabs(minF32) > kBf16Max || std::fabs(maxF32) > kBf16Max) {
                return ge::DT_FLOAT;
            }
            return ge::DT_UNDEFINED;
        }
        default:
            return ge::DT_UNDEFINED;
    }
}

// 经 CompliantNodeBuilder 按 IR 定义建 Cast/ReduceMin/ReduceMax 节点（仿 hard_max_fusion_pass.cpp
// 先例）：不依赖 per-op es 头/符号——仓内 es 聚合头随 --ops 过滤裁剪、CI 的 toolkit 布局又与本机
// 不一致，框架级 builder 两边都稳。
static es::EsTensorHolder HolderFromNode(es::EsGraphBuilder& builder, const GNode& node)
{
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

static bool AddEdgeChecked(Graph* graph, const es::EsTensorHolder& srcHolder, GNode& dstNode, int32_t dstIdx)
{
    auto srcNode = srcHolder.GetProducer();
    if (srcNode == nullptr) {
        return false;
    }
    return es::AddEdgeAndUpdatePeerDesc(*graph, *srcNode, srcHolder.GetProducerOutIndex(), dstNode, dstIdx) == SUCCESS;
}

// nodeSeq 用于生成图内唯一节点名: 同名节点会让 CopyComputeGraph 按名 Relink 边时串边失败
static es::EsTensorHolder BuildCastNode(es::EsGraphBuilder& builder, const es::EsTensorHolder& x, ge::DataType dstType,
                                        const std::vector<int64_t>& inShape, int& nodeSeq)
{
    Graph* graph = builder.GetCGraphBuilder()->GetGraph();
    const std::string name = "Cast_" + std::to_string(nodeSeq++);
    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType("Cast")
                     .Name(name.c_str())
                     .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                     .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .IrDefAttrs({{"dst_type", es::CompliantNodeBuilder::kEsAttrRequired, "Int",
                                   es::CreateFrom(static_cast<int64_t>(dstType))}})
                     .InstanceOutputDataType("y", dstType)
                     .InstanceOutputShape("y", inShape)
                     .InstanceOutputOriginShape("y", inShape)
                     .InstanceOutputFormat("y", FORMAT_ND)
                     .InstanceOutputOriginFormat("y", FORMAT_ND)
                     .Build();
    if (!AddEdgeChecked(graph, x, node, 0)) {
        OP_LOGE(kPassName.c_str(), "Failed to add edge to Cast node.");
    }
    return HolderFromNode(builder, node);
}

static es::EsTensorHolder BuildReduceNode(es::EsGraphBuilder& builder, const char* opType, const es::EsTensorHolder& x,
                                          const es::EsTensorHolder& axes, int& nodeSeq)
{
    Graph* graph = builder.GetCGraphBuilder()->GetGraph();
    const std::string name = std::string(opType) + "_" + std::to_string(nodeSeq++);
    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType(opType)
                     .Name(name.c_str())
                     .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"axes", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                     .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .IrDefAttrs(
                         {{"keep_dims", es::CompliantNodeBuilder::kEsAttrOptional, "Bool", es::CreateFrom(false)},
                          {"noop_with_empty_axes", es::CompliantNodeBuilder::kEsAttrOptional, "Bool",
                           es::CreateFrom(true)}})
                     .InstanceOutputShape("y", {})
                     .InstanceOutputOriginShape("y", {})
                     .InstanceOutputFormat("y", FORMAT_ND)
                     .InstanceOutputOriginFormat("y", FORMAT_ND)
                     .Build();
    if (!AddEdgeChecked(graph, x, node, 0) || !AddEdgeChecked(graph, axes, node, 1)) {
        OP_LOGE(kPassName.c_str(), "Failed to add edge to %s node.", opType);
    }
    return HolderFromNode(builder, node);
}

// 按目标 dtype 创建标量 const: 有原生 CreateScalar 重载的直接用, 其余先建邻近类型再 Cast
// (值已被提升决策保证在目标值域内, Cast 无损; 常量折叠可消除该 Cast 节点)
es::EsTensorHolder CreateTypedScalar(es::EsGraphBuilder& builder, float fVal, int64_t iVal, ge::DataType dtype,
                                     int& nodeSeq)
{
    switch (dtype) {
        case ge::DT_FLOAT:
            return builder.CreateScalar(fVal);
        case ge::DT_INT32:
            return builder.CreateScalar(static_cast<int32_t>(iVal));
        case ge::DT_INT64:
            return builder.CreateScalar(iVal);
        case ge::DT_FLOAT16:
            return BuildCastNode(builder, builder.CreateScalar(fVal), ge::DT_FLOAT16, {}, nodeSeq);
        case ge::DT_BF16:
            return BuildCastNode(builder, builder.CreateScalar(fVal), ge::DT_BF16, {}, nodeSeq);
        default: // DT_INT8 / DT_UINT8 / DT_INT16
            return BuildCastNode(builder, builder.CreateScalar(static_cast<int32_t>(iVal)), dtype, {}, nodeSeq);
    }
}

// 静态形状求元素数; 含动态维(负数)返回 -1 表示未知
int64_t StaticNumel(const std::vector<int64_t>& dims)
{
    int64_t numel = 1;
    for (int64_t dim : dims) {
        if (dim < 0) {
            return -1;
        }
        numel *= dim;
    }
    return numel;
}
} // namespace

std::unique_ptr<Graph> HistogramFusionPass::Replacement(const std::unique_ptr<MatchResult>& match_result)
{
    OP_LOGD(kPassName.c_str(), "Enter Replacement for HistogramFusionPass");

    std::vector<SubgraphInput> subgraphInputs;
    match_result->ToSubgraphBoundary()->GetAllInputs(subgraphInputs);

    std::vector<Shape> inputShapes;
    std::vector<DataType> inputDtypes;
    std::vector<Format> inputFormats;
    GetInputsInfo(subgraphInputs, inputShapes, inputDtypes, inputFormats);

    NodeIo histogramNodeIo;
    if (match_result->GetCapturedTensor(kCaptureHistogramNode, histogramNodeIo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to GetCaptured tensor in Replacement");
        return nullptr;
    }

    int64_t bins = 100;
    histogramNodeIo.node.GetAttr("bins", bins);
    OP_LOGD(kPassName.c_str(), "bins: %ld", bins);

    float minVal = 0.0f;
    histogramNodeIo.node.GetAttr("min", minVal);

    float maxVal = 0.0f;
    histogramNodeIo.node.GetAttr("max", maxVal);

    DataType inputDtype = inputDtypes[0];
    const bool xIsInt = IsIntDtype(inputDtype);
    OP_LOGD(kPassName.c_str(), "input dtype: %d, min: %f, max: %f", inputDtype, minVal, maxVal);

    // 语义校验(对齐 CUDA dispatch 顺序: acc_type 转换先于大小比较):
    // 返回 nullptr 时残留 Histogram 节点在 950/350 上无 kernel, 图编译响亮失败, 不会静默出错误结果。
    if (bins <= 0) {
        OP_LOGE(kPassName.c_str(), "The value of bins must be greater than 0.");
        return nullptr;
    }

    int64_t minI64 = 0;
    int64_t maxI64 = 0;
    if (xIsInt) {
        // CUDA 语义: 整数输入的 bounds 先经 checked_convert 转 acc_type(int64), 溢出/NaN 响亮报错
        if (!TryConvertToInt64(minVal, minI64) || !TryConvertToInt64(maxVal, maxI64)) {
            OP_LOGE(kPassName.c_str(), "value cannot be converted to type int64_t without overflow");
            return nullptr;
        }
    } else {
        // 浮点路径有限性校验: min==max==同号±inf 走 auto-range(对齐 aclnn), 其余 inf/NaN 报错
        const bool sameSignInfEqual = (minVal == maxVal) && std::isinf(minVal);
        if (!sameSignInfEqual &&
            (std::isnan(minVal) || std::isnan(maxVal) || std::isinf(minVal) || std::isinf(maxVal))) {
            OP_LOGE(kPassName.c_str(), "range of [%f, %f] is not finite", minVal, maxVal);
            return nullptr;
        }
    }

    const bool minGtMax = xIsInt ? (minI64 > maxI64) : (minVal > maxVal);
    if (minGtMax) {
        OP_LOGE(kPassName.c_str(), "max must be larger than min");
        return nullptr;
    }

    // HistogramV2 输出域 {int32, float}: 浮点 x 出 fp32, 整数 x 出 int32
    const DataType yDtype = (inputDtype == DT_FLOAT || inputDtype == DT_FLOAT16 || inputDtype == DT_BF16) ? DT_FLOAT :
                                                                                                            DT_INT32;
    TensorDesc outputDesc;
    histogramNodeIo.node.GetOutputDesc(0, outputDesc);
    const DataType origOutDtype = outputDesc.GetDataType();

    auto replaceGraphBuilder = es::EsGraphBuilder("replacement");

    std::vector<int64_t> xDims;
    for (size_t i = 0; i < inputShapes[0].GetDimNum(); i++) {
        xDims.push_back(inputShapes[0].GetDim(i));
    }

    auto rX = replaceGraphBuilder.CreateInput(0, "x", inputDtypes[0], inputFormats[0], xDims);

    const bool minEqMax = xIsInt ? (minI64 == maxI64) : (minVal == maxVal);
    const bool staticEmpty = (StaticNumel(xDims) == 0);

    es::EsTensorHolder histOut;
    int nodeSeq = 0; // 本图内 Cast/Reduce 节点命名序号, 保证唯一
    if (minEqMax && !staticEmpty) {
        // auto-range: min==max 时按 CUDA 语义用数据实际范围(aminmax)作为统计上下限;
        // 常数数据重算后仍相等, 由 kernel ExpandMinMaxIfEqual 做 nextafter/±1 展开兜底。
        // rank0 标量输入的数据范围即其自身, 直接用 x 作 min/max, 不建 reduce(空 axes 边界)。
        es::EsTensorHolder rMin;
        es::EsTensorHolder rMax;
        if (xDims.empty()) {
            rMin = rX;
            rMax = rX;
        } else {
            std::vector<int64_t> axes(xDims.size());
            for (size_t i = 0; i < xDims.size(); i++) {
                axes[i] = static_cast<int64_t>(i);
            }
            auto axesConst = replaceGraphBuilder.CreateConst(axes, {static_cast<int64_t>(axes.size())});
            rMin = BuildReduceNode(replaceGraphBuilder, "ReduceMin", rX, axesConst, nodeSeq);
            rMax = BuildReduceNode(replaceGraphBuilder, "ReduceMax", rX, axesConst, nodeSeq);
        }
        histOut = es::HistogramV2(rX, rMin, rMax, bins, yDtype);
    } else {
        if (staticEmpty) {
            // 静态空输入: 不建 reduce(空张量归约会报错), 标量值不参与统计; kernel 输出全零,
            // 符合 CUDA 空输入语义。标量按 0 处理即可。
            minVal = 0.0f;
            maxVal = 0.0f;
            minI64 = 0;
            maxI64 = 0;
        }
        // M3a: 标量超 x 值域时 x 与标量共同提升到能容纳标量的最窄类型(对齐 aclnn)
        const DataType widenDtype = WidenDtypeForScalars(inputDtype, minI64, maxI64, minVal, maxVal);
        DataType effDtype = inputDtype;
        es::EsTensorHolder effX = rX;
        if (widenDtype != DT_UNDEFINED) {
            effX = BuildCastNode(replaceGraphBuilder, rX, widenDtype, xDims, nodeSeq);
            effDtype = widenDtype;
            OP_LOGD(kPassName.c_str(), "Widen x and scalars to dtype %d for out-of-range scalars", widenDtype);
        }
        auto rMin = CreateTypedScalar(replaceGraphBuilder, minVal, minI64, effDtype, nodeSeq);
        auto rMax = CreateTypedScalar(replaceGraphBuilder, maxVal, maxI64, effDtype, nodeSeq);
        histOut = es::HistogramV2(effX, rMin, rMax, bins, yDtype);
    }

    // 输出 dtype 回源 Histogram 输出 desc(对齐 torch histc out dtype == self dtype)
    es::EsTensorHolder output = histOut;
    if (origOutDtype != yDtype) {
        output = BuildCastNode(replaceGraphBuilder, histOut, origOutDtype, {bins}, nodeSeq);
    }

    std::vector<es::EsTensorHolder> outputs = {output};
    GraphUniqPtr replaceGraph = replaceGraphBuilder.BuildAndReset(outputs);
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Infershape for replacement failed.");
        return nullptr;
    }

    return replaceGraph;
}

static void GetInputsInfo(const std::vector<SubgraphInput>& subgraphInputs, std::vector<Shape>& inputShapes,
                          std::vector<DataType>& inputDtypes, std::vector<Format>& inputFormats)
{
    for (const auto& subgraphInput : subgraphInputs) {
        auto matchNode = subgraphInput.GetAllInputs().at(0);
        TensorDesc tensorDesc;
        matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
        inputShapes.emplace_back(tensorDesc.GetShape());
        inputDtypes.emplace_back(tensorDesc.GetDataType());
        inputFormats.emplace_back(tensorDesc.GetFormat());
    }
}

static Status InferShape(const GraphUniqPtr& replaceGraph, const std::vector<SubgraphInput>& subgraphInputs)
{
    OP_LOGD(kPassName.c_str(), "Begin infershape for replacements.");
    std::vector<Shape> inputShapes;
    for (const auto& subgraphInput : subgraphInputs) {
        auto matchNode = subgraphInput.GetAllInputs().at(0);
        TensorDesc tensorDesc;
        matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
        inputShapes.emplace_back(tensorDesc.GetShape());
    }
    return GeUtils::InferShape(*replaceGraph, inputShapes);
}

REG_FUSION_PASS(HistogramFusionPass).Stage(CustomPassStage::kAfterInferShape);

} // namespace ops
