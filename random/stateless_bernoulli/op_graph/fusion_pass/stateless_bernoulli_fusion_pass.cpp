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
 * \file stateless_bernoulli_fusion_pass.cpp
 * \brief bernoulli fusion pass (StatelessBernoulliV2 --> StatelessBernoulli)
 */

#include <vector>
#include <string>
#include "es_math_ops.h"
#include "platform/platform_info.h"
#include "platform/soc_spec.h"
#include "ge/ge_utils.h"
#include "log/log.h"
#include "version/ge-compiler_version.h"
#include "stateless_bernoulli_fusion_pass.h"
#include "common/inc/op_graph/fusion_pass/fusion_pass_common.h"
#include "random/random_common/op_graph/random_fusion_pass_common.h"

using namespace ge;
using namespace fe;

namespace ge::fusion {

namespace {
const std::string kPassName = "BernoulliFusionPass";
constexpr int64_t kCaptureIdxV2Node = 0l;

constexpr size_t kIdxProb = 0;
constexpr size_t kIdxSeed = 1;
constexpr size_t kIdxOffset = 2;

void UpdateNodeOutputDesc(es::EsTensorHolder& tensor, DataType dtype, const Shape& shape, Format fmt)
{
    TensorDesc desc;
    desc.SetDataType(dtype);
    desc.SetShape(shape);
    desc.SetFormat(fmt);
    tensor.GetProducer()->UpdateOutputDesc(0, desc);
}
} // namespace

std::vector<PatternUniqPtr> BernoulliFusionPass::Patterns()
{
    OP_LOGI(kPassName.c_str(), "Enter Patterns");
    std::vector<PatternUniqPtr> patternGraphs;

    if (!IsTargetVersion()) {
        OP_LOGD(kPassName.c_str(), "GE runtime version < %d, skip pass.", GE_COMPILER_VERSION_910);
        return patternGraphs;
    }

    auto graphBuilder = es::EsGraphBuilder(kPassName.c_str());
    auto x = graphBuilder.CreateInput(0);
    auto seed = graphBuilder.CreateInput(1);
    auto offset = graphBuilder.CreateInput(2);

    auto output = es::StatelessBernoulliV2(x, seed, offset);
    auto graph = graphBuilder.BuildAndReset({output});

    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*output.GetProducer(), 0});
    patternGraphs.emplace_back(std::move(pattern));
    return patternGraphs;
}

bool BernoulliFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    OP_LOGI(kPassName.c_str(), "Enter MeetRequirements");

    int32_t version = 0;
    if (!random_fusion_pass_common::IsVersion900(version)) {
        OP_LOGD(kPassName.c_str(), "GE runtime version %d < 9.0.0, skip pass.", version);
        return false;
    }

    if (!random_fusion_pass_common::IsTargetPlatformNpuArch(
            kPassName, {NpuArch::DAV_2201, NpuArch::DAV_3510, NpuArch::DAV_9201, NpuArch::DAV_9202})) {
        return false;
    }

    NodeIo v2NodeIo;
    if (matchResult->GetCapturedTensor(kCaptureIdxV2Node, v2NodeIo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get captured tensor of StatelessBernoulliV2 node.");
        return false;
    }

    AscendString nodeTypeStr;
    v2NodeIo.node.GetType(nodeTypeStr);
    if (std::string(nodeTypeStr.GetString()) != "StatelessBernoulliV2") {
        return false;
    }

    TensorDesc inputDesc;
    v2NodeIo.node.GetInputDesc(0, inputDesc);
    if (!random_fusion_pass_common::CheckDtype(inputDesc.GetDataType(), {DT_FLOAT16, DT_FLOAT})) {
        return false;
    }

    auto dimNum = inputDesc.GetShape().GetDimNum();
    if (dimNum == 0) {
        OP_LOGI(kPassName.c_str(), "Input shape dimNum is 0 (unknown rank), not supported by ES API CreateConst");
        return false;
    }
    return true;
}

std::unique_ptr<Graph> BernoulliFusionPass::Replacement(const std::unique_ptr<MatchResult>& matchResult)
{
    OP_LOGI(kPassName.c_str(), "Enter Replacement");

    NodeIo v2NodeIo;
    if (matchResult->GetCapturedTensor(kCaptureIdxV2Node, v2NodeIo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get captured tensor of StatelessBernoulliV2 node in Replacement.");
        return nullptr;
    }

    DataType dtype = DT_FLOAT;
    v2NodeIo.node.GetAttr("dtype", dtype);
    if (dtype == DT_UNDEFINED) {
        dtype = DT_FLOAT;
    }

    TensorDesc probDesc;
    v2NodeIo.node.GetInputDesc(kIdxProb, probDesc);
    TensorDesc seedDesc;
    v2NodeIo.node.GetInputDesc(kIdxSeed, seedDesc);
    TensorDesc offsetDesc;
    v2NodeIo.node.GetInputDesc(kIdxOffset, offsetDesc);

    auto replaceGraphBuilder = es::EsGraphBuilder("replacement");
    auto rProb = replaceGraphBuilder.CreateInput(kIdxProb, "prob", probDesc.GetDataType(), probDesc.GetFormat(),
                                                 random_fusion_pass_common::GetShapeDims(probDesc.GetShape()));
    rProb.SetFormat(probDesc.GetFormat());
    auto rSeed = replaceGraphBuilder.CreateInput(kIdxSeed, "seed", seedDesc.GetDataType(), seedDesc.GetFormat(),
                                                 random_fusion_pass_common::GetShapeDims(seedDesc.GetShape()));
    rSeed.SetFormat(seedDesc.GetFormat());
    auto rOffset = replaceGraphBuilder.CreateInput(kIdxOffset, "offset", offsetDesc.GetDataType(),
                                                   offsetDesc.GetFormat(),
                                                   random_fusion_pass_common::GetShapeDims(offsetDesc.GetShape()));
    rOffset.SetFormat(offsetDesc.GetFormat());

    std::vector<int64_t> shapeValue = random_fusion_pass_common::GetShapeDims(probDesc.GetShape());
    std::vector<int64_t> shapeDims = {static_cast<int64_t>(shapeValue.size())};
    auto rShape = replaceGraphBuilder.CreateConst(shapeValue, shapeDims);

    auto output = es::StatelessBernoulli(rShape, rProb, rSeed, rOffset, dtype);

    UpdateNodeOutputDesc(rProb, probDesc.GetDataType(), probDesc.GetShape(), probDesc.GetFormat());
    UpdateNodeOutputDesc(rSeed, seedDesc.GetDataType(), seedDesc.GetShape(), seedDesc.GetFormat());
    UpdateNodeOutputDesc(rOffset, offsetDesc.GetDataType(), offsetDesc.GetShape(), offsetDesc.GetFormat());
    UpdateNodeOutputDesc(output, dtype, probDesc.GetShape(), probDesc.GetFormat());

    std::vector<SubgraphInput> subgraphInputs;
    matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs);
    GraphUniqPtr replaceGraph = replaceGraphBuilder.BuildAndReset({output});
    if (random_fusion_pass_common::InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to infer shape for StatelessBernoulliV2 replacement graph.");
        return nullptr;
    }
    return replaceGraph;
}

#if GE_COMPILER_VERSION_NUM >= GE_COMPILER_VERSION_900
REG_FUSION_PASS(BernoulliFusionPass).Stage(random_fusion_pass_common::GetRandomFusionPassStage());
#else
REG_FUSION_PASS(BernoulliFusionPass).Stage(CustomPassStage::kBeforeInferShape);
#endif

} // namespace ge::fusion
