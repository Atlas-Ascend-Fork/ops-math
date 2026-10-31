/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file topk_fusion_pass.cpp
 * \brief TopK fusion pass, rebuilt from the legacy fe::TopKFusionPass (canndev).
 *        Full-platform migration covering every chip family the legacy pass serves
 *        (within the ops-math build/deploy matrix).
 *
 *        Two platform families, split by the operator information library that
 *        actually registers the target ops (measured in the shipped op_info):
 *
 *        1. regbase (NpuArch DAV_3510 / DAV_5102: Ascend950/350, MC62/630):
 *           the new AscendC system registers TopKV2 (x + k inputs, int32/int64 k,
 *           indices_dtype attr) and the two-input Transpose (x + perm const input).
 *             TopK/TopKV2 --> TopKV2 (k kept as an input, dim = -1);
 *             dim != -1  --> Transpose --> TopKV2 --> Transpose.
 *           On DAV_5102 the legacy BigGuard gates apply (11-dtype whitelist,
 *           sorted=false, int64 const k, indices_dtype=DT_INT64); DAV_3510 keeps
 *           the conservative 3-dtype narrowing.
 *
 *        2. legacy tbe platforms (NpuArch DAV_2201/3002/2002/3004/3102:
 *           Ascend910/910B/910_93, Ascend310B/310P, Ascend031, Ascend610Lite):
 *           the old tbe op_info registers TopKD (x + assist_seq inputs, k/sorted/
 *           dim/largest attrs), TopKV2 (x + k inputs, no indices_dtype attr) and
 *           the attribute-based TransposeD (perm attr).
 *             const k  --> TopKD (k becomes an attr + assist_seq input);
 *             dynamic k --> TopKV2 (k kept as an input; int64 k desc converted to
 *                           int32, aligned with the legacy pass);
 *             dim != -1 --> TransposeD --> fused op --> TransposeD.
 *           The x dtype whitelist follows the per-soc op_info reality (see
 *           MeetRequirements step 3); socs without any TopK-family entry (e.g.
 *           Ascend035, dav_3505) and unidentifiable socs stay silent. Ascend031
 *           has no TopKV2 entry, so its dynamic-k path does not fuse.
 *           The assist_seq is 2048 fp16 values [0..2047]; on Ascend310B/031 with
 *           a const k and an unknown-shape x the tik flavour is used instead
 *           (4096 + 4096 fp16: values and their fp16 rounding gaps, RNE).
 *
 *        All descs of the replacement graph are set manually (the legacy pass does
 *        the same, no full-graph infershape): running a full-graph InferShape on
 *        the replacement graph would invoke the registered TopKV2 infershape
 *        (InferShapeForTopKV2D, InputsDataDependency({1})), which needs the const
 *        value of k - unavailable on the replacement graph's Data inputs - and
 *        reports "k is nullptr" on a real runtime that has the op impl loaded.
 *
 *        Not carried over from the legacy pass (outside the ops-math build/deploy
 *        matrix, soc_arch_variables.cmake): the Ascend310-only SegmentSort/
 *        MultiMerge multi-core branch and the Ascend310 dynamic-k TopKV2D target
 *        (ascend310 is not a build target of this repo; those paths keep being
 *        served by the legacy canndev package). The legacy defensive int64-x
 *        desc conversion in the TopKV2 branch is dead code (int64 x only passes
 *        the BigGuard dtype gate) and is not carried over either.
 *
 * Compatibility processing (D1 scenario, aligned with DropOutV3/Permute/Random*
 * passes): the pass inherits the legacy built-in TopKFusionPass slot and runs
 * at the kCompatibleInherited stage (born in GE 9.0.0), so that on supported
 * devices it adaptively takes over the legacy canndev pass. Version safety strategy:
 * compile-time macro guard + runtime version check + overall silence:
 *   - compile-time: pass built with pre-9.0.0 toolkit falls back to the old
 *     kBeforeInferShape stage so the kCompatibleInherited enum is not referenced
 *   - runtime: ge_compiler < 9.0.0 registers on the old stage and idles (no
 *     patterns, no fusion); ge_compiler in [9.0.0, 9.1.0) keeps the stage but
 *     skips pattern building (ES pattern framework readiness)
 */

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "es_math_ops.h"
#include "platform/platform_info.h"
#include "platform/soc_spec.h"
#include "log/log.h"
#include "securec.h"
#include "version/ge-compiler_version.h"

#include "topk_fusion_pass.h"
#include "common/inc/op_graph/fusion_pass/fusion_pass_common.h"

// Declare the runtime soc spec query (implemented in libruntime): the pass gates on
// NpuArch via this API, aligned with the OPP support_fusion_pass whitelist keys.
extern "C" {
typedef int32_t rtError_t;
static const int32_t RT_ERROR_NONE = 0;
rtError_t rtGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen);
}

using namespace ge;
using namespace fe;
using namespace ge::fusion;

namespace ops {
namespace es = ge::es;

// D1 scenario: uses kCompatibleInherited stage (9.0.0+).
// Strategy: compile-time macro guard + runtime version check + overall silence.
#define GE_COMPILER_VERSION_900 90000000

namespace {
#if GE_COMPILER_VERSION_NUM >= GE_COMPILER_VERSION_900
CustomPassStage GetTopKFusionPassStage()
{
    int32_t version = 0;
    if (aclsysGetVersionNum) {
        aclsysGetVersionNum(const_cast<char*>("ge_compiler"), &version);
    }
    if (version >= GE_COMPILER_VERSION_900) {
        return CustomPassStage::kCompatibleInherited;
    }
    return CustomPassStage::kBeforeInferShape;
}
#endif
} // namespace

namespace {
const std::string kPassName = "TopKFusionPass";
const int64_t kCaptureTensorIdx = 0L;
// Input/output index of the TopK family ops.
constexpr int32_t kInputXIdx = 0;
constexpr int32_t kInputKIdx = 1;
constexpr int32_t kOutputValuesIdx = 0;
constexpr int32_t kOutputIndicesIdx = 1;
constexpr int64_t kDefaultDim = -1L;
// The legacy pass builds a regular assist_seq of 2048 fp16 values [0, 1, ..., 2047]
// and a tik flavour of 4096 + 4096 fp16 values (values then fp16 rounding gaps).
constexpr int64_t kAssistSeqLen = 2048L;
// TopKD backend check_supported thresholds (ops_legacy/dynamic/top_k_d.py):
// input_size > 32768 with a const k in (0, 8) is rejected on non-910B socs.
constexpr int64_t kMaxTopKDInputSize = 32768L;
constexpr int64_t kMinTopKDKThreshold = 8L;
constexpr int64_t kTikAssistHalfLen = 4096L;

// Max length of the NpuArch string returned by rtGetSocSpec (e.g. "3510").
constexpr uint32_t kNpuArchValLen = 32U;

// Query the runtime NpuArch via the runtime soc spec (reads the driver/CANN config, not the
// PlatformInfoManager state). Returns false when the query fails; npuArch is only written on
// success.
bool GetRuntimeNpuArch(NpuArch& npuArch)
{
    char npuArchVal[kNpuArchValLen] = {0};
    if (rtGetSocSpec("version", "NpuArch", npuArchVal, sizeof(npuArchVal)) != RT_ERROR_NONE) {
        OP_LOGW(kPassName.c_str(), "Get NpuArch from soc spec failed.");
        return false;
    }
    npuArch = static_cast<NpuArch>(std::strtol(npuArchVal, nullptr, 10));
    return true;
}

// regbase archs (the legacy IsRegBase() set): the new AscendC op_info registers
// TopKV2 (x + k) and the two-input Transpose (x + perm const) there.
bool IsRegbaseArch(NpuArch npuArch) { return npuArch == NpuArch::DAV_3510 || npuArch == NpuArch::DAV_5102; }

// legacy tbe archs: the old op_info registers TopKD / TopKV2 / TransposeD there.
bool IsLegacyArch(NpuArch npuArch)
{
    static const std::set<NpuArch> legacyArchs = {NpuArch::DAV_2201, NpuArch::DAV_3002, NpuArch::DAV_2002,
                                                  NpuArch::DAV_3004, NpuArch::DAV_3102};
    return legacyArchs.count(npuArch) > 0;
}

// Ascend031 (DAV_3004) has no TopKV2 entry in its op_info: the dynamic-k path must
// not fuse there (the legacy pass generated an unselectable TopKV2 on 031, which
// is a legacy defect; this rebuild stays silent instead).
bool IsTopKV2Registered(NpuArch npuArch)
{
    return IsRegbaseArch(npuArch) || (IsLegacyArch(npuArch) && npuArch != NpuArch::DAV_3004);
}

// The legacy soc names recognized on legacy archs (short_soc_version of the
// PlatformInfo). Unrecognized legacy socs stay silent: their op_info coverage of
// the TopK family is unknown.
bool IsKnownLegacySoc(const std::string& shortSocVersion)
{
    if (shortSocVersion.rfind("Ascend910B", 0) == 0 || shortSocVersion == "Ascend910_93" ||
        shortSocVersion == "Ascend910") {
        return true; // DAV_2201 family
    }
    if (shortSocVersion.rfind("Ascend310B", 0) == 0 || shortSocVersion.rfind("Ascend310P", 0) == 0) {
        return true; // DAV_3002 / DAV_2002
    }
    if (shortSocVersion == "Ascend031" || shortSocVersion.rfind("Ascend610", 0) == 0) {
        return true; // DAV_3004 / DAV_3102
    }
    return false;
}

// The tik assist_seq flavour applies on Ascend310B/Ascend031 with a const k and an
// unknown-shape x (legacy is_tik gate reachable inside the TopKD branch; the
// Ascend310-only legs are outside the deploy matrix of this repo).
bool IsTikAssistSoc(const std::string& shortSocVersion)
{
    return shortSocVersion.rfind("Ascend310B", 0) == 0 || shortSocVersion == "Ascend031";
}

// x dtype gate for the regbase family: the full 11-dtype list on BOTH regbase archs,
// identical to the legacy BigGuard branch (kTopkSupportDtype in canndev, applied to
// DAV_3510 and DAV_5102 alike via IsRegBase()) and to the TopKV2 IR declaration
// (RealNumberType). The op_info reality agrees: ascend950 registers the same 11
// x dtypes for TopKV2 as mc62. The earlier conservative 3-dtype narrowing for
// DAV_3510 was a shipped decision dropped in favour of legacy alignment.
const std::set<DataType>& GetRegbaseXDtypeWhitelist()
{
    static const std::set<DataType> kRegbaseDtype = {DT_INT64, DT_INT32, DT_INT16,   DT_INT8,  DT_UINT32, DT_UINT16,
                                                     DT_UINT8, DT_BF16,  DT_FLOAT16, DT_FLOAT, DT_UINT64};
    return kRegbaseDtype;
}

// x dtype gate for the legacy family, following the per-soc op_info reality:
// - Ascend910B* / Ascend910_93 (TopKD & TopKV2): float16/float/bfloat16;
// - Ascend910 (910A, both entries fp16-only): float16;
// - Ascend310B*: TopKD (const k) float16-only, TopKV2 (dynamic k) float16/float;
// - Ascend310P* / Ascend610* (both entries): float16/float;
// - Ascend031: TopKD (const k) float16-only (dynamic k does not fuse at all).
bool IsLegacyXDtypeAllowed(NpuArch npuArch, const std::string& shortSocVersion, DataType xDtype, bool isKConst)
{
    static const std::set<DataType> kFp16Only = {DT_FLOAT16};
    static const std::set<DataType> kFp16F32 = {DT_FLOAT16, DT_FLOAT};
    static const std::set<DataType> kFp16F32Bf16 = {DT_FLOAT16, DT_FLOAT, DT_BF16};
    if (shortSocVersion.rfind("Ascend910B", 0) == 0 || shortSocVersion == "Ascend910_93") {
        return kFp16F32Bf16.count(xDtype) > 0;
    }
    if (shortSocVersion == "Ascend910") {
        return kFp16Only.count(xDtype) > 0;
    }
    if (shortSocVersion.rfind("Ascend310B", 0) == 0) {
        return (isKConst ? kFp16Only : kFp16F32).count(xDtype) > 0;
    }
    if (shortSocVersion.rfind("Ascend310P", 0) == 0 || shortSocVersion.rfind("Ascend610", 0) == 0) {
        return kFp16F32.count(xDtype) > 0;
    }
    if (shortSocVersion == "Ascend031") {
        return isKConst && kFp16Only.count(xDtype) > 0;
    }
    (void)npuArch;
    return false;
}

// True when the node's indices output is int64 but the indices_dtype attr is not DT_INT64 yet.
// The legacy BigGuard branch sets this attr on DAV_5102; the rebuilt TopKV2 carries it so that
// re-matching the pass's own output is idempotent (no infinite replacement loop).
bool IndicesDtypeNeedsFix(const GNode& topkNode)
{
    TensorDesc indicesDesc;
    if (topkNode.GetOutputDesc(kOutputIndicesIdx, indicesDesc) != SUCCESS) {
        return false;
    }
    if (indicesDesc.GetDataType() != DT_INT64) {
        return false;
    }
    int64_t indicesDtype = static_cast<int64_t>(DT_INT32);
    return !(topkNode.GetAttr("indices_dtype", indicesDtype) == GRAPH_SUCCESS &&
             indicesDtype == static_cast<int64_t>(DT_INT64));
}

// Build the fp16 bit pattern of a small integer that is exactly representable in
// fp16 (0..2048 arbitrary values, then multiples of the fp16 step; the sign is
// honoured for the tik rounding gaps, which can be -1). This replaces the legacy
// AssitHelp() helper that relied on fp16_t.
uint16_t IntToFp16Bits(const int64_t value)
{
    if (value == 0) {
        return static_cast<uint16_t>(0U);
    }
    const bool isNeg = (value < 0);
    const uint32_t magnitude = isNeg ? static_cast<uint32_t>(-value) : static_cast<uint32_t>(value);
    uint32_t mantissa = magnitude;
    int32_t exponent = 0;
    while (mantissa > 1U) {
        mantissa >>= 1U;
        exponent += 1;
    }
    // magnitude = 1.xxx * 2^exponent, fp16 bias is 15, fraction width is 10.
    const uint16_t exponentBits = static_cast<uint16_t>(exponent + 15);
    const uint32_t fraction = magnitude - (1U << exponent);
    uint16_t fractionBits = 0U;
    if (exponent <= 10) {
        fractionBits = static_cast<uint16_t>(fraction << (10 - exponent));
    } else {
        // exponent 11/12 (magnitude in (2048, 4096] / 4096]): the fraction is wider
        // than 10 bits, the caller must pass a value already rounded to the fp16 step.
        fractionBits = static_cast<uint16_t>(fraction >> (exponent - 10));
    }
    const uint16_t signBit = isNeg ? static_cast<uint16_t>(0x8000U) : static_cast<uint16_t>(0U);
    return static_cast<uint16_t>(signBit | (exponentBits << 10) | fractionBits);
}

// Round a non-negative integer of the tik assist range (0..4095) to the nearest
// fp16-representable integer, ties to even (same as the fp16_t int conversion the
// legacy AssitHelp() relied on): exact up to 2048, then a step of 2.
int64_t RoundToFp16Representable(const int64_t value)
{
    if (value <= 2048) {
        return value;
    }
    int64_t half = value >> 1;
    if ((value & 1) != 0 && (half & 1) != 0) {
        half += 1;
    }
    return half << 1;
}

// Build the assist_seq const data:
// - regular flavour: 2048 fp16 values [0, 1, ..., 2047] (all exactly representable);
// - tik flavour (Ascend310B/031, const k, unknown-shape x): 4096 fp16 values
//   [0..4095] rounded to fp16 (RNE) followed by 4096 fp16 rounding gaps
//   (i - int32(fp16(i))), same as the legacy AssitHelp(n, out, false).
std::vector<uint16_t> GenAssistSeqData(const bool isTik)
{
    if (!isTik) {
        std::vector<uint16_t> assistData(static_cast<size_t>(kAssistSeqLen), 0U);
        for (int64_t i = 0; i < kAssistSeqLen; ++i) {
            assistData[static_cast<size_t>(i)] = IntToFp16Bits(i);
        }
        return assistData;
    }
    std::vector<uint16_t> assistData(static_cast<size_t>(kTikAssistHalfLen * 2), 0U);
    for (int64_t i = 0; i < kTikAssistHalfLen; ++i) {
        const int64_t rounded = RoundToFp16Representable(i);
        assistData[static_cast<size_t>(i)] = IntToFp16Bits(rounded);
        assistData[static_cast<size_t>(i + kTikAssistHalfLen)] = IntToFp16Bits(i - rounded);
    }
    return assistData;
}

// True when the shape contains an unknown dim (-1) or unknown rank (-2), the
// equivalent of the legacy GeShape::IsUnknownShape used by the tik gate.
bool IsUnknownShape(const std::vector<int64_t>& dims)
{
    return std::any_of(dims.begin(), dims.end(), [](int64_t dim) { return dim < 0; });
}

// True when the shape is an unknown-RANK shape (contains the -2 marker), the
// equivalent of the legacy GeShape::IsUnknownRankShape used by the legacy dim
// range check: the rank itself is undecidable there, so the check is skipped.
bool IsUnknownRankShape(const std::vector<int64_t>& dims)
{
    return std::any_of(dims.begin(), dims.end(), [](int64_t dim) { return dim == -2; });
}

// True when the output port has downstream consumers, the equivalent of the
// legacy IsInsertTransposeForOutput (out-anchor peer check): output-side
// transposes are only inserted for consumed outputs.
bool HasOutConsumer(const GNode& node, const int64_t outIndex)
{
    return !node.GetOutDataNodesAndPortIndexs(static_cast<int32_t>(outIndex)).empty();
}

// True when the desc looks unset: ge::TensorDesc default-constructs to
// DT_FLOAT with an empty shape, and ATC-converted graphs may leave an input
// desc in exactly that state (the fused node then dispatches with a float k
// and fails with "The data type DT_FLOAT of the input k is not supported").
// A genuinely-set desc is excluded: a 0-dim scalar k (empty dims, valid int
// dtype) is the ONNX TopK K form and must be used as-is.
bool LooksUnsetDesc(const TensorDesc& tensorDesc)
{
    if (tensorDesc.GetDataType() == DT_UNDEFINED) {
        return true;
    }
    return tensorDesc.GetDataType() == DT_FLOAT && tensorDesc.GetShape().GetDims().empty();
}

// Resolve the TensorDesc of a subgraph input's first matched input.
// Fallback: when the desc looks unset, resolve it from the output desc of the
// SAME input port's producer. The legacy condition (shape-empty fallback with
// a hard-coded port-0 read) mishandled both directions: it treated the valid
// 0-dim scalar k as unset and then read the x producer, polluting the
// replacement graph's k Data desc with the x dtype (DT_FLOAT) and failing the
// ATC dispatch of the fused node (topk_v2_TopK_*).
TensorDesc ResolveSubgraphInputTensorDesc(const SubgraphInput& subgraphInput)
{
    auto matchNode = subgraphInput.GetAllInputs().at(0);
    TensorDesc tensorDesc;
    matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
    if (LooksUnsetDesc(tensorDesc)) {
        auto srcInfo = matchNode.node.GetInDataNodesAndPortIndexs(matchNode.index);
        if (srcInfo.first != nullptr) {
            GNode srcNode = *srcInfo.first;
            srcNode.GetOutputDesc(srcInfo.second, tensorDesc);
        }
    }
    return tensorDesc;
}

void GetInputsInfo(const std::vector<SubgraphInput>& subgraphInputs, std::vector<Shape>& inputShapes,
                   std::vector<DataType>& inputDtypes, std::vector<Format>& inputFormats)
{
    for (const auto& subgraphInput : subgraphInputs) {
        TensorDesc tensorDesc = ResolveSubgraphInputTensorDesc(subgraphInput);
        inputShapes.emplace_back(tensorDesc.GetShape());
        inputDtypes.emplace_back(tensorDesc.GetDataType());
        inputFormats.emplace_back(tensorDesc.GetFormat());
    }
}

// 把输入源tensor的输出desc回填到目标节点的输入desc, 避免节点输入desc停留在CompliantNodeBuilder默认的 DT_FLOAT
bool UpdateNodeInputDesc(GNode& targetNode, int32_t targetInputIndex, const es::EsTensorHolder& source)
{
    GNode sourceNode = *source.GetProducer();
    TensorDesc sourceDesc;
    if (sourceNode.GetOutputDesc(source.GetProducerOutIndex(), sourceDesc) != SUCCESS) {
        return false;
    }
    return targetNode.UpdateInputDesc(targetInputIndex, sourceDesc) == GRAPH_SUCCESS;
}

bool UpdateNodeInputDescFromNode(GNode& targetNode, int32_t targetInputIndex, const GNode& sourceNode,
                                 int32_t sourceOutputIndex)
{
    TensorDesc sourceDesc;
    if (sourceNode.GetOutputDesc(sourceOutputIndex, sourceDesc) != SUCCESS) {
        return false;
    }
    return targetNode.UpdateInputDesc(targetInputIndex, sourceDesc) == GRAPH_SUCCESS;
}

// Build a Transpose node (x + perm const inputs) using CompliantNodeBuilder for the
// regbase platforms: their operator information library only registers the two-input
// Transpose (perm as the second input), NOT the attribute-based TransposeD - the same
// op the legacy pass inserts via AddDynamicTransposeBefore/AfterNode on BigGuard.
// The perm input is an int64 const tensor, aligned with the legacy Vec2ConstNode.
es::EsTensorHolder BuildTransposeNode(es::EsGraphBuilder& replaceGraphBuilder, const std::string& nodeName,
                                      GNode inputNode, int32_t inputIndex, const std::vector<int64_t>& permAttr)
{
    auto* graph = replaceGraphBuilder.GetCGraphBuilder()->GetGraph();
    auto transposeNode = es::CompliantNodeBuilder(graph)
                             .OpType("Transpose")
                             .Name(nodeName.c_str())
                             .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                           {"perm", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                             .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                             .Build();
    if (es::AddEdgeAndUpdatePeerDesc(*graph, inputNode, inputIndex, transposeNode, 0) != GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to add edge for Transpose x input.");
        return es::EsTensorHolder();
    }
    auto permConst = replaceGraphBuilder.CreateConst(permAttr, {static_cast<int64_t>(permAttr.size())});
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *permConst.GetProducer(), permConst.GetProducerOutIndex(), transposeNode,
                                     1) != GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to add edge for Transpose perm input.");
        return es::EsTensorHolder();
    }
    auto output = replaceGraphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(transposeNode, 0);
    GNode transposeGNode = *es::EsTensorHolder(output).GetProducer();
    if (!UpdateNodeInputDescFromNode(transposeGNode, 0, inputNode, inputIndex) ||
        !UpdateNodeInputDesc(transposeGNode, 1, permConst)) {
        OP_LOGE(kPassName.c_str(), "Failed to update Transpose input descs.");
        return es::EsTensorHolder();
    }
    // Legacy parity: AddDynamicTransposeBefore/AfterNode declare the perm input as
    // an op-infer depend (SetOpInferDepends({"perm"})). The new framework headers
    // do not export OpDesc::SetOpInferDepends, but its implementation is exactly
    // SetAttr("_op_infer_depends", <string list>) (verified against libgraph_base:
    // the setter writes the "_op_infer_depends" attribute) - set it directly so
    // the GE infershape of the rewritten Transpose resolves the perm input the
    // same way as the legacy-inserted nodes.
    std::vector<ge::AscendString> inferDepends{ge::AscendString("perm")};
    if (transposeGNode.SetAttr(ge::AscendString("_op_infer_depends"), inferDepends) != ge::GRAPH_SUCCESS) {
        OP_LOGD(kPassName.c_str(), "Failed to set _op_infer_depends on Transpose (non-fatal).");
    }
    return es::EsTensorHolder(output);
}

// Build a TransposeD node (perm as an attribute) using CompliantNodeBuilder for the
// legacy tbe platforms: their op_info registers the attribute-based TransposeD
// (input x, output y, attr perm), the same op the legacy pass inserts via
// PatternFusionUtil::InsertSingleNode on non-BigGuard platforms.
es::EsTensorHolder BuildTransposeDNode(es::EsGraphBuilder& replaceGraphBuilder, const std::string& nodeName,
                                       GNode inputNode, int32_t inputIndex, const std::vector<int64_t>& permAttr)
{
    auto* graph = replaceGraphBuilder.GetCGraphBuilder()->GetGraph();
    auto transposeDNode = es::CompliantNodeBuilder(graph)
                              .OpType("TransposeD")
                              .Name(nodeName.c_str())
                              .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                              .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                              .IrDefAttrs({
                                  {"perm", es::CompliantNodeBuilder::kEsAttrRequired, "ListInt",
                                   es::CreateFrom(permAttr)},
                              })
                              .Build();
    if (es::AddEdgeAndUpdatePeerDesc(*graph, inputNode, inputIndex, transposeDNode, 0) != GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to add edge for TransposeD input.");
        return es::EsTensorHolder();
    }
    auto output = replaceGraphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(transposeDNode, 0);
    GNode transposeDGNode = *es::EsTensorHolder(output).GetProducer();
    if (!UpdateNodeInputDescFromNode(transposeDGNode, 0, inputNode, inputIndex)) {
        OP_LOGE(kPassName.c_str(), "Failed to update TransposeD input desc.");
        return es::EsTensorHolder();
    }
    return es::EsTensorHolder(output);
}

// Build a TopKV2 node (x, k inputs; sorted/dim/largest attrs, plus indices_dtype on
// regbase only) using CompliantNodeBuilder, since there is no in-repo ES API for TopKV2.
// The regbase op_info declares the indices_dtype attr (DT_INT32 default,
// DT_INT64 for int64 indices outputs, legacy BigGuard behaviour); the legacy tbe
// op_info has no such attr (attr.list = sorted,dim,largest), so withRegbaseAttrs
// must be false there. sort_policy is carried over STRICTLY as canndev does (the
// legacy in-place SetType keeps the source attribute set untouched): the rebuilt
// node declares it if and only if the source node carries it - a source without
// the attr keeps producing a node without it, so the kernel stays on the IR
// default (0) exactly like the legacy path (writing a defaulted value here would
// silently switch the kernel algorithm for attr-less sources). The dim attr keeps
// the source value on the last-dim path and is forced to -1 otherwise
// (the legacy pass only rewrites it when the transposes are inserted).
bool BuildTopKV2Node(es::EsGraphBuilder& builder, const es::EsTensorHolder& x, const es::EsTensorHolder& k, bool sorted,
                     bool largest, int64_t dim, int64_t indicesDtype, bool hasSortPolicy, int64_t sortPolicy,
                     bool withRegbaseAttrs, es::EsTensorHolder& valuesHolder, es::EsTensorHolder& indicesHolder)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    es::CompliantNodeBuilder nodeBuilder(graph);
    nodeBuilder.OpType("TopKV2")
        .Name("topk_v2")
        .IrDefInputs({
            {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"k", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
        })
        .IrDefOutputs({
            {"values", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
            {"indices", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
        });
    if (withRegbaseAttrs) {
        std::vector<es::CompliantNodeBuilder::IrAttrDef> regbaseAttrs = {
            {"sorted", es::CompliantNodeBuilder::kEsAttrRequired, "Bool", es::CreateFrom(sorted)},
            {"dim", es::CompliantNodeBuilder::kEsAttrRequired, "Int", es::CreateFrom(dim)},
            {"largest", es::CompliantNodeBuilder::kEsAttrRequired, "Bool", es::CreateFrom(largest)},
            {"indices_dtype", es::CompliantNodeBuilder::kEsAttrRequired, "Int", es::CreateFrom(indicesDtype)},
        };
        if (hasSortPolicy) {
            regbaseAttrs.push_back(
                {"sort_policy", es::CompliantNodeBuilder::kEsAttrRequired, "Int", es::CreateFrom(sortPolicy)});
        }
        (void)nodeBuilder.IrDefAttrs(regbaseAttrs);
    } else {
        (void)nodeBuilder.IrDefAttrs({
            {"sorted", es::CompliantNodeBuilder::kEsAttrRequired, "Bool", es::CreateFrom(sorted)},
            {"dim", es::CompliantNodeBuilder::kEsAttrRequired, "Int", es::CreateFrom(dim)},
            {"largest", es::CompliantNodeBuilder::kEsAttrRequired, "Bool", es::CreateFrom(largest)},
        });
    }
    auto topkV2Node = nodeBuilder.Build();
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), topkV2Node, kInputXIdx) !=
        GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to add edge for TopKV2 x input.");
        return false;
    }
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *k.GetProducer(), k.GetProducerOutIndex(), topkV2Node, kInputKIdx) !=
        GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to add edge for TopKV2 k input.");
        return false;
    }
    valuesHolder = es::EsTensorHolder(
        builder.GetCGraphBuilder()->GetTensorHolderFromNode(topkV2Node, kOutputValuesIdx));
    indicesHolder = es::EsTensorHolder(
        builder.GetCGraphBuilder()->GetTensorHolderFromNode(topkV2Node, kOutputIndicesIdx));
    GNode topkV2GNode = *valuesHolder.GetProducer();
    if (!UpdateNodeInputDesc(topkV2GNode, kInputXIdx, x) || !UpdateNodeInputDesc(topkV2GNode, kInputKIdx, k)) {
        OP_LOGE(kPassName.c_str(), "Failed to update TopKV2 input descs.");
        return false;
    }
    return true;
}

// Build a TopKD node (x + assist_seq inputs; k/sorted/dim/largest attrs) for the legacy
// tbe platforms: their op_info declares input0=x, input1=assist_seq (both required,
// fp16/ND) and the required int attr k - the same target the legacy pass builds via
// ConstToAttrWithNode (k moves from an input to an attr).
bool BuildTopKDNode(es::EsGraphBuilder& builder, const es::EsTensorHolder& x, const es::EsTensorHolder& assistSeq,
                    int64_t kValue, bool sorted, int64_t dim, bool largest, es::EsTensorHolder& valuesHolder,
                    es::EsTensorHolder& indicesHolder)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    auto topkDNode = es::CompliantNodeBuilder(graph)
                         .OpType("TopKD")
                         .Name("topk_d")
                         .IrDefInputs({
                             {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                             {"assist_seq", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                         })
                         .IrDefOutputs({
                             {"values", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
                             {"indices", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
                         })
                         .IrDefAttrs({
                             {"k", es::CompliantNodeBuilder::kEsAttrRequired, "Int", es::CreateFrom(kValue)},
                             {"sorted", es::CompliantNodeBuilder::kEsAttrRequired, "Bool", es::CreateFrom(sorted)},
                             {"dim", es::CompliantNodeBuilder::kEsAttrRequired, "Int", es::CreateFrom(dim)},
                             {"largest", es::CompliantNodeBuilder::kEsAttrRequired, "Bool", es::CreateFrom(largest)},
                         })
                         .Build();
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), topkDNode, kInputXIdx) !=
        GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to add edge for TopKD x input.");
        return false;
    }
    if (es::AddEdgeAndUpdatePeerDesc(*graph, *assistSeq.GetProducer(), assistSeq.GetProducerOutIndex(), topkDNode, 1) !=
        GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to add edge for TopKD assist_seq input.");
        return false;
    }
    valuesHolder = es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(topkDNode, kOutputValuesIdx));
    indicesHolder = es::EsTensorHolder(
        builder.GetCGraphBuilder()->GetTensorHolderFromNode(topkDNode, kOutputIndicesIdx));
    GNode topkDGNode = *valuesHolder.GetProducer();
    if (!UpdateNodeInputDesc(topkDGNode, kInputXIdx, x) || !UpdateNodeInputDesc(topkDGNode, 1, assistSeq)) {
        OP_LOGE(kPassName.c_str(), "Failed to update TopKD input descs.");
        return false;
    }
    return true;
}

// Generate the perm list that swaps dimAim with the last dim: [0,1,..,n-1] with
// perm[dimAim] and perm[n-1] exchanged, same as the legacy PermVecGen().
std::vector<int64_t> GenSwapLastDimPerm(int64_t dimSize, int64_t dimAim)
{
    std::vector<int64_t> perm;
    perm.reserve(static_cast<size_t>(dimSize));
    for (int64_t i = 0; i < dimSize; ++i) {
        perm.push_back(i);
    }
    std::swap(perm[static_cast<size_t>(dimAim)], perm[static_cast<size_t>(dimSize - 1)]);
    return perm;
}

// Swap dims[dimAim] and dims.back() in-place, used to compute transposed shapes.
// All intermediate descs are set manually (no full-graph infershape, same as the
// legacy pass), so the transposed shapes of the intermediate nodes must be computed
// here.
void SwapLastDim(std::vector<int64_t>& dims, int64_t dimAim)
{
    std::swap(dims[static_cast<size_t>(dimAim)], dims.back());
}

// Read the const value of the k input. int32 is parsed everywhere; int64 is
// additionally parsed (kept as-is into regbase TopKV2 whose k is IndexNumberType -
// legacy BigGuard behaviour on DAV_5102, and into the TopKD k attr where the value
// semantics are identical) - callers gate the dtype in MeetRequirements.
bool GetConstKValue(const GNode& topkNode, int64_t& kValue)
{
    Tensor kTensor;
    if (topkNode.GetInputConstData(kInputKIdx, kTensor) != GRAPH_SUCCESS) {
        return false;
    }
    TensorDesc kDesc;
    if (topkNode.GetInputDesc(kInputKIdx, kDesc) != SUCCESS) {
        return false;
    }
    const uint8_t* kData = kTensor.GetData();
    if (kData == nullptr) {
        return false;
    }
    if (kDesc.GetDataType() == DT_INT32) {
        int32_t val = 0;
        if (memcpy_s(&val, sizeof(int32_t), kData, sizeof(int32_t)) != EOK) {
            return false;
        }
        kValue = static_cast<int64_t>(val);
        return true;
    }
    if (kDesc.GetDataType() == DT_INT64) {
        if (memcpy_s(&kValue, sizeof(int64_t), kData, sizeof(int64_t)) != EOK) {
            return false;
        }
        return true;
    }
    return false;
}
} // namespace

std::vector<PatternUniqPtr> TopKFusionPass::Patterns()
{
    OP_LOGD(kPassName.c_str(), "Enter Patterns for TopKFusionPass.");
    std::vector<PatternUniqPtr> patternGraphs;

    // Version compatibility (D1 overall silence): on a runtime older than the
    // target version the ES pattern framework may not be ready, skip pattern
    // building so the pass stays silent instead of failing.
    if (!IsTargetVersion()) {
        OP_LOGD(kPassName.c_str(), "GE runtime version < %d, skip pass.", GE_COMPILER_VERSION_910);
        return patternGraphs;
    }

    // Pattern 1: match the TopK op (2 inputs: x, k; 2 outputs: values, indices).
    {
        auto graphBuilder = es::EsGraphBuilder(kPassName.c_str());
        auto x = graphBuilder.CreateInput(kInputXIdx, "x");
        auto k = graphBuilder.CreateInput(kInputKIdx, "k");
        auto output = es::TopK(x, k);
        auto builtGraph = graphBuilder.BuildAndReset({output.values, output.indices});
        auto pattern = std::make_unique<Pattern>(std::move(*builtGraph));
        pattern->CaptureTensor({*output.values.GetProducer(), kCaptureTensorIdx});
        patternGraphs.emplace_back(std::move(pattern));
    }

    // Pattern 2: match the TopKV2 op, which is converted to TopK and handled
    // the same way in the legacy pass.
    {
        auto graphBuilder = es::EsGraphBuilder(kPassName.c_str());
        auto x = graphBuilder.CreateInput(kInputXIdx, "x");
        auto k = graphBuilder.CreateInput(kInputKIdx, "k");
        es::EsTensorHolder valuesHolder;
        es::EsTensorHolder indicesHolder;
        if (!BuildTopKV2Node(graphBuilder, x, k, true, true, kDefaultDim, static_cast<int64_t>(DT_INT32), true, 0, true,
                             valuesHolder, indicesHolder)) {
            OP_LOGE(kPassName.c_str(), "Failed to build TopKV2 pattern node.");
            return patternGraphs;
        }
        auto builtGraph = graphBuilder.BuildAndReset({valuesHolder, indicesHolder});
        auto pattern = std::make_unique<Pattern>(std::move(*builtGraph));
        pattern->CaptureTensor({*valuesHolder.GetProducer(), kCaptureTensorIdx});
        patternGraphs.emplace_back(std::move(pattern));
    }

    return patternGraphs;
}

bool TopKFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    OP_LOGD(kPassName.c_str(), "Enter MeetRequirements for TopKFusionPass.");

    // Version compatibility (D1 overall silence): a package built by 9.x but
    // running on an older ge_compiler registers on the old stage and idles
    // here, deferring to the legacy built-in pass.
    int32_t version = 0;
    if (aclsysGetVersionNum) {
        aclsysGetVersionNum(const_cast<char*>("ge_compiler"), &version);
    }
    if (version < GE_COMPILER_VERSION_900) {
        OP_LOGD(kPassName.c_str(), "GE runtime version %d < 9.0.0, skip pass.", version);
        return false;
    }

    // 1. Platform info must be available (legacy pass fails fusion when it is not).
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Get platform_info failed.");
        return false;
    }

    // 1b. Platform gating by the operator information library reality:
    //     - regbase (DAV_3510/DAV_5102): TopKV2 + two-input Transpose (new AscendC
    //       system), legacy BigGuard gates on DAV_5102;
    //     - legacy tbe archs (DAV_2201/3002/2002/3004/3102): TopKD/TopKV2/TransposeD,
    //       with the soc identified via short_soc_version (unknown legacy socs and
    //       archs without any TopK-family entry stay silent);
    //     - everything else: not covered by this pass.
    NpuArch npuArch = static_cast<NpuArch>(0);
    if (!GetRuntimeNpuArch(npuArch)) {
        OP_LOGW(kPassName.c_str(), "Get NpuArch from soc spec failed, skip.");
        return false;
    }
    const bool isRegbase = IsRegbaseArch(npuArch);
    const bool isLegacy = IsLegacyArch(npuArch);
    if (!isRegbase && !isLegacy) {
        OP_LOGI(kPassName.c_str(), "Platform NpuArch %u has no TopK-family op_info entry, skip.",
                static_cast<uint32_t>(npuArch));
        return false;
    }
    const std::string shortSocVersion = platformInfo.str_info.short_soc_version;
    if (isLegacy && !IsKnownLegacySoc(shortSocVersion)) {
        OP_LOGI(kPassName.c_str(), "Legacy soc [%s] is not in the known TopK whitelist, skip.",
                shortSocVersion.c_str());
        return false;
    }
    // (isDav5102 removed: the legacy BigGuard semantics (= IsRegBase, both regbase
    // archs) now cover the sorted / int64-k / indices_dtype gates uniformly.)

    // 2. Get the captured TopK/TopKV2 node.
    NodeIo topkNodeIo;
    if (matchResult->GetCapturedTensor(kCaptureTensorIdx, topkNodeIo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get captured tensor.");
        return false;
    }
    GNode topkNode = topkNodeIo.node;

    AscendString nodeType;
    if (topkNode.GetType(nodeType) != GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get node type.");
        return false;
    }
    if (nodeType != "TopK" && nodeType != "TopKV2") {
        OP_LOGD(kPassName.c_str(), "Node type %s is not TopK/TopKV2, skip.", nodeType.GetString());
        return false;
    }

    // 2b. Legacy behaviour (L563-582): the source values output desc must carry a
    //     non-empty shape (the dim_info_out check on the output-0 desc of the
    //     matched node) - a missing or empty output desc aborts the fusion.
    TensorDesc srcValuesDescCheck;
    if (topkNode.GetOutputDesc(kOutputValuesIdx, srcValuesDescCheck) != SUCCESS ||
        srcValuesDescCheck.GetShape().GetDims().empty()) {
        OP_LOGD(kPassName.c_str(), "The values output desc of the source node is missing or empty, skip.");
        return false;
    }

    // 3. x dtype check, split by family (see the whitelist helpers for the
    //    per-soc rationale): regbase keeps the shipped split (3510 narrowed,
    //    5102 BigGuard 11-dtype); legacy follows the per-soc op_info dtype lists.
    TensorDesc xDesc;
    if (topkNode.GetInputDesc(kInputXIdx, xDesc) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get input x desc.");
        return false;
    }
    DataType xDtype = xDesc.GetDataType();

    // 5 (early). Constness of k decides the legacy target op (TopKD vs TopKV2),
    // which also decides the applicable dtype list on some legacy socs.
    int64_t kValue = 0;
    const bool isKConst = GetConstKValue(topkNode, kValue);

    if (isRegbase) {
        if (GetRegbaseXDtypeWhitelist().count(xDtype) == 0) {
            OP_LOGD(kPassName.c_str(), "The dtype of input x is not suitable for aicore, skip.");
            return false;
        }
    } else {
        if (!IsLegacyXDtypeAllowed(npuArch, shortSocVersion, xDtype, isKConst)) {
            OP_LOGD(kPassName.c_str(), "The dtype of input x is not supported by the target op on %s, skip.",
                    shortSocVersion.c_str());
            return false;
        }
    }

    // 4. The value of sorted cannot be false on aicore (legacy behaviour), except on
    //    the regbase archs where the legacy BigGuard branch (= IsRegBase, 3510 AND
    //    5102 alike) allows it (the TopKV2 kernel gates it itself).
    bool sorted = true;
    (void)topkNode.GetAttr("sorted", sorted);
    if (!sorted && !isRegbase) {
        OP_LOGD(kPassName.c_str(), "The value of sorted must be true on aicore, skip.");
        return false;
    }

    // 5. k input check (legacy behaviour): a const k must be int32 or int64 (the
    //    legacy pass parses both); a non-const k is kept as an input of TopKV2
    //    (dynamic-k path). DAV_3510 keeps the shipped int32-only narrowing for const
    //    k; DAV_5102 additionally allows int64 (kept as-is, legacy BigGuard
    //    behaviour). On Ascend031 the dynamic-k path does not fuse (no TopKV2 entry
    //    in its op_info). A dynamic int64 k is additionally rejected on the legacy
    //    platforms: the legacy pass faked the op_info match by rewriting the in-place
    //    k desc to int32, which cannot be carried over to the replacement-graph
    //    framework (the source k producer keeps its int64 desc), so fusing would risk
    //    an op selection failure - staying silent falls back to the AICPU TopK.
    //    ATC-converted graphs may leave the node's k input desc unset (a
    //    default-constructed desc reads DT_FLOAT with an empty shape): resolve
    //    it from the k producer's output desc before gating; if no usable desc
    //    is available at all, refuse to fuse - building a node with a garbage
    //    k desc would only fail the dispatch later (ATC "The data type
    //    DT_FLOAT of the input k is not supported" on the fused node).
    TensorDesc kDesc;
    bool hasKDesc = (topkNode.GetInputDesc(kInputKIdx, kDesc) == SUCCESS);
    if (!hasKDesc || LooksUnsetDesc(kDesc)) {
        auto kSrcInfo = topkNode.GetInDataNodesAndPortIndexs(kInputKIdx);
        if (kSrcInfo.first != nullptr) {
            GNode kSrcNode = *kSrcInfo.first;
            hasKDesc = (kSrcNode.GetOutputDesc(kSrcInfo.second, kDesc) == SUCCESS);
        }
    }
    if (hasKDesc) {
        Tensor kTensor;
        const bool isKConstRaw = (topkNode.GetInputConstData(kInputKIdx, kTensor) == GRAPH_SUCCESS);
        if (isKConstRaw) {
            const bool isKInt32 = (kDesc.GetDataType() == DT_INT32);
            // int64 const k: kept as-is on BOTH regbase archs (the legacy BigGuard
            // branch only rewrites the int64 k desc on non-regbase platforms; the
            // ascend950 op_info registers k as int32/int64, same as mc62) and on the
            // legacy archs (the TopKD k attr holds the parsed value, dtype-agnostic).
            const bool isKInt64Ok = (isRegbase || isLegacy);
            if (!isKInt32 && !(isKInt64Ok && kDesc.GetDataType() == DT_INT64)) {
                OP_LOGD(kPassName.c_str(), "K only supports int32/int64 on aicore, skip.");
                return false;
            }
        } else if (isLegacy && kDesc.GetDataType() == DT_INT64) {
            OP_LOGD(kPassName.c_str(), "Dynamic int64 k is not carry-overable on legacy platforms, skip.");
            return false;
        }
    } else {
        OP_LOGD(kPassName.c_str(), "No usable k input desc (neither the node desc nor the producer desc), skip.");
        return false;
    }
    if (!isKConst && !IsTopKV2Registered(npuArch)) {
        OP_LOGD(kPassName.c_str(), "Dynamic k has no TopKV2 entry on this platform, skip.");
        return false;
    }

    // 6. dim check: normalize negative dim and validate it against the x rank.
    // 6. dim check: normalize negative dim and validate it against the x rank.
    //    Legacy behaviour: the range check is skipped for unknown-rank shapes
    //    (dim_aim >= dim_size && !IsUnknownRankShape(dim_info) in the legacy
    //    pass) - the rank is undecidable there, so the fusion continues.
    auto xDims = xDesc.GetShape().GetDims();
    const int64_t dimSize = static_cast<int64_t>(xDims.size());
    if (dimSize < 1) {
        OP_LOGD(kPassName.c_str(), "The dim size of input x is invalid, skip.");
        return false;
    }
    int64_t dimAim = kDefaultDim;
    (void)topkNode.GetAttr("dim", dimAim);
    if (dimAim < 0) {
        dimAim += dimSize;
    }
    if (dimAim >= dimSize && !IsUnknownRankShape(xDims)) {
        OP_LOGD(kPassName.c_str(), "Dim %ld is out of range of rank %ld, skip.", dimAim, dimSize);
        return false;
    }

    // 6a. TopKD support gate (legacy parity: the legacy pass calls
    //     CheckOpSupported(topkd fusion_desc) before ConstToAttrWithNode and stays
    //     silent when the backend rejects; the static dtype/platform whitelists
    //     above cannot cover the backend shape/k constraints). This is the static
    //     equivalent of the tbe check_supported in
    //     ops_legacy/dynamic/top_k_d.py, all branches:
    //     - the sort axis (dimAim-th dim of x, which becomes the last dim after
    //       the swap) must be longer than 1 on non-310 socs
    //       ("data of sort axis is 1"); an unknown dim consults the shape range
    //       upper bound, an unresolvable one stays silent (the backend returns
    //       "Unknown" there);
    //     - input_size > 32768 with a const k in (0, 8) is rejected on non-910B
    //       socs (the backend's performance trade-off; Ascend910B is exempt).
    //     The Ascend310-only k/dim-size caps are not carried over: 310 is outside
    //     the build/deploy matrix of this repo.
    if (isLegacy && isKConst) {
        const int64_t axisDim = (dimAim >= 0 && dimAim < dimSize) ? xDims[static_cast<size_t>(dimAim)] : -1;
        if (axisDim == 1) {
            OP_LOGD(kPassName.c_str(), "The data of sort axis is 1, the TopKD backend rejects it, skip.");
            return false;
        }
        if (axisDim < 0) {
            std::vector<std::pair<int64_t, int64_t>> xRange;
            (void)xDesc.GetShapeRange(xRange);
            const bool rangeKnown = (xRange.size() == static_cast<size_t>(dimSize) &&
                                     xRange[static_cast<size_t>(dimAim)].second >= 0);
            if (rangeKnown && xRange[static_cast<size_t>(dimAim)].second == 1) {
                OP_LOGD(kPassName.c_str(), "The range of sort axis is 1, the TopKD backend rejects it, skip.");
                return false;
            }
        }
        int64_t inputSize = 1;
        bool inputSizeKnown = true;
        for (const int64_t dim : xDims) {
            if (dim < 0) {
                inputSizeKnown = false;
                break;
            }
            inputSize *= dim;
        }
        const bool isSoc910B = shortSocVersion.rfind("Ascend910B", 0) == 0;
        if (inputSizeKnown && inputSize > kMaxTopKDInputSize && kValue > 0 && kValue < kMinTopKDKThreshold &&
            !isSoc910B) {
            OP_LOGD(kPassName.c_str(),
                    "Input size %ld (> %ld) with a const k in (0, %ld) is rejected by the TopKD backend, skip.",
                    inputSize, kMaxTopKDInputSize, kMinTopKDKThreshold);
            return false;
        }
    }

    // 6b. Anti-loop / idempotency guard for TopKV2 sources (the replacement target
    //     type itself), aligned with the legacy no-op cases:
    //     - dynamic k + last dim: the legacy pass only re-set the type in place
    //       (no structural change), and the rebuilt TopKV2 always has dim = -1,
    //       so the pass's own output re-matches here and must terminate;
    //     - dynamic k + non-last dim: the legacy pass still inserts the transposes
    //       (the TopKV2 kernel only serves the last-dim path), so it is processed;
    //     - const k on regbase + last dim: already in the target form (TopKV2 with
    //       dim = -1), skip; a pending indices_dtype fix (int64 indices output
    //       without the DT_INT64 attr) is still processed and the rebuilt node
    //       carries the fixed attr so the next match skips (idempotent);
    //     - const k on legacy (last dim included): the legacy ConstToAttrWithNode
    //       rebuilds a native TopKV2 source into TopKD as well - and TopKD is never
    //       re-matched by the patterns, so no guard is needed there.
    if (nodeType == "TopKV2") {
        if (!isKConst) {
            if (dimAim == dimSize - 1 && !(isRegbase && IndicesDtypeNeedsFix(topkNode))) {
                OP_LOGD(kPassName.c_str(), "TopKV2 with dynamic k on the last dim keeps as is, skip.");
                return false;
            }
        } else if (isRegbase && dimAim == dimSize - 1 && !IndicesDtypeNeedsFix(topkNode)) {
            OP_LOGD(kPassName.c_str(), "TopKV2 with const k on the last dim keeps as is, skip.");
            return false;
        }
    }

    return true;
}

GraphUniqPtr TopKFusionPass::Replacement(const std::unique_ptr<MatchResult>& matchResult)
{
    OP_LOGD(kPassName.c_str(), "Enter Replacement for TopKFusionPass.");

    // 1. Get the captured TopK/TopKV2 node and read its attributes.
    NodeIo topkNodeIo;
    if (matchResult->GetCapturedTensor(kCaptureTensorIdx, topkNodeIo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get captured tensor in Replacement.");
        return nullptr;
    }
    GNode topkNode = topkNodeIo.node;

    bool sorted = true;
    bool largest = true;
    (void)topkNode.GetAttr("sorted", sorted);
    (void)topkNode.GetAttr("largest", largest);
    // sort_policy is carried over STRICTLY as canndev does (the legacy in-place
    // SetType keeps the source attribute set): the rebuilt node declares the attr
    // if and only if the source carries it - an attr-less source (a TopK node, or
    // an older TopKV2 graph) keeps producing an attr-less node, and the kernel
    // then runs on the IR default (0) exactly like the legacy path.
    int64_t sortPolicy = 0;
    const bool hasSortPolicy = (topkNode.GetAttr("sort_policy", sortPolicy) == GRAPH_SUCCESS);

    // The x input desc of the source node, used to build the descs of the
    // intermediate nodes of the replacement graph.
    TensorDesc xDesc;
    if (topkNode.GetInputDesc(kInputXIdx, xDesc) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get input x desc.");
        return nullptr;
    }

    // 2. Get subgraph inputs info (x and k).
    std::vector<SubgraphInput> subgraphInputs;
    matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs);
    std::vector<Shape> inputShapes;
    std::vector<DataType> inputDtypes;
    std::vector<Format> inputFormats;
    GetInputsInfo(subgraphInputs, inputShapes, inputDtypes, inputFormats);
    if (inputShapes.size() < 2) {
        OP_LOGE(kPassName.c_str(), "Subgraph inputs size %zu is less than 2.", inputShapes.size());
        return nullptr;
    }

    // 3. Decide the target op family (same gates as MeetRequirements):
    //    - regbase: TopKV2 (k kept as an input, regbase IR with indices_dtype);
    //    - legacy + const k: TopKD (k becomes an attr + assist_seq input);
    //    - legacy + dynamic k: TopKV2 (legacy IR without indices_dtype).
    int64_t kValue = 0;
    const bool isKConst = GetConstKValue(topkNode, kValue);
    NpuArch npuArch = static_cast<NpuArch>(0);
    if (!GetRuntimeNpuArch(npuArch)) {
        OP_LOGE(kPassName.c_str(), "Get NpuArch from soc spec failed.");
        return nullptr;
    }
    const bool isRegbase = IsRegbaseArch(npuArch);
    const bool isLegacy = IsLegacyArch(npuArch);
    const bool useTopKD = isLegacy && isKConst;

    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Get platform_info failed.");
        return nullptr;
    }
    const std::string shortSocVersion = platformInfo.str_info.short_soc_version;

    // 4. Normalize dim and decide whether transposes are needed. The dim attr of the
    //    fused node keeps the source value on the last-dim path and is forced to -1
    //    otherwise (legacy behaviour: the value is only rewritten when the
    //    transposes are inserted).
    auto xDims = inputShapes[0].GetDims();
    const int64_t dimSize = static_cast<int64_t>(xDims.size());
    int64_t dimAim = kDefaultDim;
    (void)topkNode.GetAttr("dim", dimAim);
    const int64_t srcDimAttr = dimAim;
    if (dimAim < 0) {
        dimAim += dimSize;
    }
    const bool isLastDim = (dimAim == dimSize - 1);
    const int64_t targetDim = isLastDim ? srcDimAttr : kDefaultDim;
    std::vector<int64_t> perm;
    if (!isLastDim) {
        // Legacy second dim check (L844, PARAM_INVALID): reachable only after the
        // unknown-rank exemption in MeetRequirements lets an out-of-range dim
        // through - the legacy pass aborts here before the swap; mirror it to
        // guard the perm/shape swap indices (otherwise UB on the small vector).
        if (dimAim < 0 || dimAim >= dimSize) {
            OP_LOGE(kPassName.c_str(), "Dim index is out of shape range.");
            return nullptr;
        }
        perm = GenSwapLastDimPerm(dimSize, dimAim);
    }
    // The shape range of the source x desc, swapped the same way as the shape for
    // the intermediate descs (the legacy pass propagates the range through every
    // manually set desc; it stays empty for fully static shapes).
    std::vector<std::pair<int64_t, int64_t>> xShapeRange;
    (void)xDesc.GetShapeRange(xShapeRange);

    // 5. Build the replacement graph. The k Data input is always declared so the
    //    replacement input count matches the match boundary (graph_rewriter
    //    requires equality); on the TopKV2 paths it feeds the fused node, on the
    //    TopKD path k becomes an attr and the Data stays dangling (it is bypassed
    //    during the copy and the source k const becomes isolated, pruned later).
    auto replaceGraphBuilder = es::EsGraphBuilder("replacement");
    auto rX = replaceGraphBuilder.CreateInput(kInputXIdx, "x", inputDtypes[0], inputFormats[0], xDims);
    auto rK = replaceGraphBuilder.CreateInput(kInputKIdx, "k", inputDtypes[1], inputFormats[1],
                                              inputShapes[1].GetDims());

    // 5.1 Optional input-side transpose: swap dimAim with the last dim. regbase
    //     uses the two-input Transpose (x + perm const, the only transposition op
    //     in its op_info); legacy uses the attribute-based TransposeD. All
    //     intermediate descs are set manually (no full-graph infershape).
    es::EsTensorHolder fusedInput = rX;
    std::vector<int64_t> fusedInputDims = xDims;
    if (!isLastDim) {
        SwapLastDim(fusedInputDims, dimAim);
        es::EsTensorHolder transIn;
        if (isRegbase) {
            transIn = BuildTransposeNode(replaceGraphBuilder, "transpose_in", *rX.GetProducer(),
                                         rX.GetProducerOutIndex(), perm);
        } else {
            transIn = BuildTransposeDNode(replaceGraphBuilder, "transpose_d_in", *rX.GetProducer(),
                                          rX.GetProducerOutIndex(), perm);
        }
        if (transIn.GetCTensorHolder() == nullptr) {
            OP_LOGE(kPassName.c_str(), "Failed to build input transpose node.");
            return nullptr;
        }
        TensorDesc transInDesc = xDesc;
        transInDesc.SetShape(Shape(fusedInputDims));
        transInDesc.SetOriginShape(Shape(fusedInputDims));
        // Swapped shape range, same as the legacy SetShapeRange on the input transpose.
        std::vector<std::pair<int64_t, int64_t>> transInRange = xShapeRange;
        if (transInRange.size() == static_cast<size_t>(dimSize)) {
            std::swap(transInRange[static_cast<size_t>(dimAim)], transInRange.back());
        }
        (void)transInDesc.SetShapeRange(transInRange);
        GNode transInNode = *transIn.GetProducer();
        if (transInNode.UpdateOutputDesc(0, transInDesc) != GRAPH_SUCCESS) {
            OP_LOGE(kPassName.c_str(), "Failed to update input transpose output desc.");
            return nullptr;
        }
        fusedInput = transIn;
    }

    // 5.2 Build the fused op with dim forced to -1 (regbase TopKV2 / legacy TopKD
    //     for const k / legacy TopKV2 for dynamic k) and set its output descs
    //     manually (no full-graph infershape: the registered TopKV2 infershape
    //     needs the const value of k, unavailable on the replacement graph's Data
    //     inputs, and would fail with "k is nullptr" on a real runtime).
    es::EsTensorHolder fusedValues;
    es::EsTensorHolder fusedIndices;
    // indices output dtype: regbase TopKV2 follows the source indices output
    // (int64 -> DT_INT64, legacy BigGuard behaviour); legacy targets are int32-only
    // in their op_info (the legacy pass also converts the indices desc to int32 for
    // an int64 k).
    DataType fusedIndicesDtype = DT_INT32;
    if (isRegbase) {
        TensorDesc srcIndicesDescForDtype;
        if (topkNode.GetOutputDesc(kOutputIndicesIdx, srcIndicesDescForDtype) == SUCCESS &&
            srcIndicesDescForDtype.GetDataType() == DT_INT64) {
            fusedIndicesDtype = DT_INT64;
        }
    }
    if (useTopKD) {
        // Legacy const-k path: TopKD (x + assist_seq inputs, k/sorted/dim/largest
        // attrs). The assist_seq flavour follows the legacy tik gate (310B/031 with
        // an unknown-shape x use 4096+4096 fp16, otherwise 2048 fp16). k leaves the
        // subgraph boundary as an attr (rK above stays dangling for the boundary
        // count), same as the legacy ConstToAttrWithNode.
        const bool isTikAssist = isKConst && IsUnknownShape(xDims) && IsTikAssistSoc(shortSocVersion);
        const std::vector<uint16_t> assistData = GenAssistSeqData(isTikAssist);
        const int64_t assistLen = isTikAssist ? (kTikAssistHalfLen * 2) : kAssistSeqLen;
        // The assist MUST be a legacy "Constant" node (SetWeights + SetType(...,
        // "Constant") in the legacy pass), NOT the repo-default "Const": the tbe
        // Pre-compile of TopKD (ops_legacy/dynamic/top_k_d.py) treats Constant as a
        // weight-class input with a dedicated channel; a plain "Const" node goes
        // through the ordinary input path and the pre-compile crashes (empty python
        // stack, E40021 on Ascend910_93 - verified: Const fails, Constant passes).
        auto* replaceGraph = replaceGraphBuilder.GetCGraphBuilder()->GetGraph();
        TensorDesc assistDesc(Shape(std::vector<int64_t>{assistLen}), FORMAT_ND, DT_FLOAT16);
        Tensor assistTensor(assistDesc, reinterpret_cast<uint8_t*>(const_cast<uint16_t*>(assistData.data())),
                            assistData.size() * sizeof(uint16_t));
        auto assistNode = es::CompliantNodeBuilder(replaceGraph)
                              .OpType("Constant")
                              .Name("topk_assist_constant")
                              .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                              .IrDefAttrs({{"value", es::CompliantNodeBuilder::kEsAttrOptional, "VT_TENSOR",
                                            es::CreateFrom(assistTensor)}})
                              .InstanceOutputShape("y", std::vector<int64_t>{assistLen})
                              .InstanceOutputDataType("y", DT_FLOAT16)
                              .InstanceOutputFormat("y", FORMAT_ND)
                              .Build();
        (void)assistNode.UpdateOutputDesc(0, assistDesc);
        es::EsTensorHolder assistHolder(replaceGraphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(assistNode, 0));
        if (!BuildTopKDNode(replaceGraphBuilder, fusedInput, assistHolder, kValue, sorted, targetDim, largest,
                            fusedValues, fusedIndices)) {
            OP_LOGE(kPassName.c_str(), "Failed to build TopKD node.");
            return nullptr;
        }
    } else {
        // TopKV2 path: k stays an input. The regbase TopKV2 IR declares the
        // indices_dtype attr; sort_policy is declared iff the source carries it
        // (strict canndev parity: the legacy in-place SetType preserves the source
        // attribute set - an attr-less source keeps an attr-less node and the
        // kernel runs on the IR default); the legacy IR has neither attr
        // (attr.list = sorted,dim,largest).
        if (!BuildTopKV2Node(replaceGraphBuilder, fusedInput, rK, sorted, largest, targetDim,
                             static_cast<int64_t>(fusedIndicesDtype), hasSortPolicy, sortPolicy, isRegbase, fusedValues,
                             fusedIndices)) {
            OP_LOGE(kPassName.c_str(), "Failed to build TopKV2 node.");
            return nullptr;
        }
    }
    // Manually set the fused op's output descs (same as the legacy
    // UpdateTopKV2NodeOutput path): the transposed shape with the last dim
    // replaced by k. For a const k the value is known; for a dynamic k the source
    // TopK output's dimAim-th dim is exactly k (kept as-is, possibly -1).
    std::vector<int64_t> fusedOutDims = fusedInputDims;
    if (!fusedOutDims.empty()) {
        if (isKConst) {
            fusedOutDims.back() = kValue;
        } else {
            TensorDesc srcValuesDescForShape;
            if (topkNode.GetOutputDesc(kOutputValuesIdx, srcValuesDescForShape) == SUCCESS) {
                auto srcDims = srcValuesDescForShape.GetShape().GetDims();
                if (srcDims.size() == static_cast<size_t>(dimSize)) {
                    fusedOutDims.back() = srcDims[static_cast<size_t>(dimAim)];
                } else {
                    fusedOutDims.back() = -1;
                }
            } else {
                fusedOutDims.back() = -1;
            }
        }
    }
    // The values desc inherits dtype/format from the source TopK values output (fallback:
    // the x desc), and only the shape is recomputed. The shape range follows the legacy
    // propagation: the swapped x range with the last dim capped by k (const k) or taken
    // from the source output's dimAim-th range pair (dynamic k).
    TensorDesc fusedValuesDesc = xDesc;
    TensorDesc srcValuesDescForDtype;
    if (topkNode.GetOutputDesc(kOutputValuesIdx, srcValuesDescForDtype) == SUCCESS) {
        fusedValuesDesc = srcValuesDescForDtype;
    }
    fusedValuesDesc.SetShape(Shape(fusedOutDims));
    fusedValuesDesc.SetOriginShape(Shape(fusedOutDims));
    std::vector<std::pair<int64_t, int64_t>> fusedShapeRange = xShapeRange;
    if (fusedShapeRange.size() == static_cast<size_t>(dimSize)) {
        if (!isLastDim) {
            std::swap(fusedShapeRange[static_cast<size_t>(dimAim)], fusedShapeRange.back());
        }
        if (isKConst) {
            fusedShapeRange.back() = {kValue, kValue};
        } else {
            std::vector<std::pair<int64_t, int64_t>> srcValuesRange;
            if (topkNode.GetOutputDesc(kOutputValuesIdx, srcValuesDescForDtype) == SUCCESS &&
                srcValuesDescForDtype.GetShapeRange(srcValuesRange) == GRAPH_SUCCESS &&
                srcValuesRange.size() == static_cast<size_t>(dimSize)) {
                fusedShapeRange.back() = srcValuesRange[static_cast<size_t>(dimAim)];
            } else {
                fusedShapeRange.back() = {-1, -1};
            }
        }
    } else {
        fusedShapeRange.clear();
    }
    (void)fusedValuesDesc.SetShapeRange(fusedShapeRange);
    TensorDesc fusedIndicesDesc = fusedValuesDesc;
    fusedIndicesDesc.SetDataType(fusedIndicesDtype);
    GNode fusedNode = *fusedValues.GetProducer();
    if (fusedNode.UpdateOutputDesc(kOutputValuesIdx, fusedValuesDesc) != GRAPH_SUCCESS ||
        fusedNode.UpdateOutputDesc(kOutputIndicesIdx, fusedIndicesDesc) != GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to update fused op output descs.");
        return nullptr;
    }

    // 5.3 Optional output-side transpose nodes for values and indices,
    //     swapping dimAim with the last dim back (family split as 5.1).
    //     Legacy behaviour differs per family: the regbase (BigGuard) leg inserts
    //     BOTH output transposes unconditionally (InsertTransposeForTopkV2Node),
    //     while the legacy tbe leg only inserts for consumed outputs
    //     (IsInsertTransposeForOutput, L950/L991).
    es::EsTensorHolder finalValues = fusedValues;
    es::EsTensorHolder finalIndices = fusedIndices;
    // Consumer split (also reused by step 6): the regbase (BigGuard) leg inserts
    // BOTH output transposes unconditionally, the legacy tbe leg only for
    // consumed outputs (IsInsertTransposeForOutput, L950/L991).
    const bool hasValuesConsumer = isRegbase || HasOutConsumer(topkNode, kOutputValuesIdx);
    const bool hasIndicesConsumer = isRegbase || HasOutConsumer(topkNode, kOutputIndicesIdx);
    if (!isLastDim) {
        es::EsTensorHolder transOutValues;
        es::EsTensorHolder transOutIndices;
        if (hasValuesConsumer) {
            if (isRegbase) {
                transOutValues = BuildTransposeNode(replaceGraphBuilder, "transpose_out_values",
                                                    *fusedValues.GetProducer(), fusedValues.GetProducerOutIndex(),
                                                    perm);
            } else {
                transOutValues = BuildTransposeDNode(replaceGraphBuilder, "transpose_d_out_values",
                                                     *fusedValues.GetProducer(), fusedValues.GetProducerOutIndex(),
                                                     perm);
            }
            if (transOutValues.GetCTensorHolder() == nullptr) {
                OP_LOGE(kPassName.c_str(), "Failed to build output transpose node for values.");
                return nullptr;
            }
            finalValues = transOutValues;
        }
        if (hasIndicesConsumer) {
            if (isRegbase) {
                transOutIndices = BuildTransposeNode(replaceGraphBuilder, "transpose_out_indices",
                                                     *fusedIndices.GetProducer(), fusedIndices.GetProducerOutIndex(),
                                                     perm);
            } else {
                transOutIndices = BuildTransposeDNode(replaceGraphBuilder, "transpose_d_out_indices",
                                                      *fusedIndices.GetProducer(), fusedIndices.GetProducerOutIndex(),
                                                      perm);
            }
            if (transOutIndices.GetCTensorHolder() == nullptr) {
                OP_LOGE(kPassName.c_str(), "Failed to build output transpose node for indices.");
                return nullptr;
            }
            finalIndices = transOutIndices;
        }
    }

    // 6. Restore the output descs of the source TopK node on the final output
    //    nodes BEFORE BuildAndReset (the graph is finalized by BuildAndReset and
    //    must not be modified afterwards), so the fused subgraph keeps identical
    //    output shape/dtype. Only the outputs that actually went through an
    //    output-side transpose (i.e. the real external outputs) get the source
    //    desc restored: an unconsumed output without its transpose keeps the
    //    fused node's INTERNAL desc (the transposed last-dim layout set in 5.2b)
    //    - overwriting it with the external shape here would leave a desc that
    //    disagrees with the data the node actually produces (legacy parity: the
    //    legacy pass sets both internal outputs to the transposed shape and only
    //    restores the external shape on the newly inserted output transposes).
    TensorDesc srcValuesDesc;
    TensorDesc srcIndicesDesc;
    if (topkNode.GetOutputDesc(kOutputValuesIdx, srcValuesDesc) != SUCCESS ||
        topkNode.GetOutputDesc(kOutputIndicesIdx, srcIndicesDesc) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get source TopK output descs.");
        return nullptr;
    }
    // Legacy int64-k note: the legacy pass converts the indices output desc to
    // int32 together with the k desc, so the restored desc is normalized the same way.
    if (isLegacy && srcIndicesDesc.GetDataType() == DT_INT64) {
        srcIndicesDesc.SetDataType(DT_INT32);
    }
    // Same consumer split as 5.3: on the last-dim path both outputs are the
    // fused node itself and their descs already match the source (5.2b), so
    // both are restored; on the non-last-dim path only the transposed
    // (consumed) outputs are external and get restored.
    const bool restoreValuesDesc = isLastDim || hasValuesConsumer;
    const bool restoreIndicesDesc = isLastDim || hasIndicesConsumer;
    GNode finalValuesNode = *finalValues.GetProducer();
    GNode finalIndicesNode = *finalIndices.GetProducer();
    if (restoreValuesDesc) {
        if (finalValuesNode.UpdateOutputDesc(finalValues.GetProducerOutIndex(), srcValuesDesc) != GRAPH_SUCCESS) {
            OP_LOGE(kPassName.c_str(), "Failed to update final values output desc.");
            return nullptr;
        }
    }
    if (restoreIndicesDesc) {
        if (finalIndicesNode.UpdateOutputDesc(finalIndices.GetProducerOutIndex(), srcIndicesDesc) != GRAPH_SUCCESS) {
            OP_LOGE(kPassName.c_str(), "Failed to update final indices output desc.");
            return nullptr;
        }
    }

    GraphUniqPtr replaceGraph = replaceGraphBuilder.BuildAndReset({finalValues, finalIndices});
    if (replaceGraph == nullptr) {
        OP_LOGE(kPassName.c_str(), "Failed to build replacement graph.");
        return nullptr;
    }

    OP_LOGI(kPassName.c_str(), "TopKFusionPass fusion success.");
    return replaceGraph;
}

// D1 registration: inherit the legacy built-in TopKFusionPass slot at the
// kCompatibleInherited stage on GE 9.0.0+; on pre-9.0.0 toolkits the enum
// value does not exist, so fall back to the old kBeforeInferShape stage (the
// runtime version checks keep the pass silent there).
#if GE_COMPILER_VERSION_NUM >= GE_COMPILER_VERSION_900
REG_FUSION_PASS(TopKFusionPass).Stage(GetTopKFusionPassStage());
#else
REG_FUSION_PASS(TopKFusionPass).Stage(CustomPassStage::kBeforeInferShape);
#endif

} // namespace ops
