/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file random_uniform_fusion_pass.cpp
 * \brief RandomUniform fusion pass (RandomUniform --> RandomUniformV2)
 *
 *            shape                                       shape     offset(variable)
 *              |                                           |       /  |
 *         RandomUniform          ==>                 RandomUniformV2  |
 *              |                                           |       \  |
 *            output                                      output     offset
 */

#include <cstdlib>
#include <vector>
#include <string>
#include <set>
#include "es_math_ops.h"
#include "platform/platform_info.h"
#include "platform/soc_spec.h"
#include "ge/ge_utils.h"
#include "log/log.h"
#include "version/ge-compiler_version.h"
#include "random_uniform_fusion_pass.h"
#include "common/inc/op_graph/fusion_pass/fusion_pass_common.h"
#include "random/random_common/op_graph/random_fusion_pass_common.h"

using namespace ge;
using namespace ge::fusion;
using namespace fe;

namespace ops {

static const std::string kPassName = "RandomUniformFusionPass";
static const int64_t kCaptureIdxNode = 0L;
static const std::set<DataType> kAicoreDtypeSupportList = {DT_FLOAT, DT_FLOAT16, DT_BF16};

std::vector<PatternUniqPtr> RandomUniformFusionPass::Patterns()
{
    OP_LOGD(kPassName.c_str(), "Enter Patterns");
    std::vector<PatternUniqPtr> patternGraphs;

    if (!IsTargetVersion()) {
        OP_LOGD(kPassName.c_str(), "GE runtime version < %d, skip pass.", GE_COMPILER_VERSION_910);
        return patternGraphs;
    }

    auto graphBuilder = es::EsGraphBuilder(kPassName.c_str());
    auto shape = graphBuilder.CreateInput(0);
    auto output = es::RandomUniform(shape, DT_FLOAT, 0, 0);
    auto graph = graphBuilder.BuildAndReset({output});

    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*output.GetProducer(), 0});

    patternGraphs.emplace_back(std::move(pattern));
    return patternGraphs;
}

bool RandomUniformFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    OP_LOGD(kPassName.c_str(), "Enter MeetRequirements");

    int32_t version = 0;
    if (!random_fusion_pass_common::IsVersion900(version)) {
        OP_LOGD(kPassName.c_str(), "GE runtime version %d < 9.0.0, skip pass.", version);
        return false;
    }

    // Platform check - NpuArch based (DAV_3510 covers both Ascend950 and Ascend350)
    if (!random_fusion_pass_common::IsTargetPlatformNpuArch(kPassName)) {
        return false;
    }

    NodeIo nodeIo;
    if (matchResult->GetCapturedTensor(kCaptureIdxNode, nodeIo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to GetCaptured tensor.");
        return false;
    }
    AscendString nodeType;
    nodeIo.node.GetType(nodeType);
    if (std::string(nodeType.GetString()) != "RandomUniform") {
        OP_LOGD(kPassName.c_str(), "Node type is not RandomUniform, skip.");
        return false;
    }

    DataType dtype = DT_FLOAT;
    nodeIo.node.GetAttr("dtype", dtype);
    if (kAicoreDtypeSupportList.count(dtype) == 0) {
        OP_LOGD(kPassName.c_str(), "Dtype %d not supported.", static_cast<int>(dtype));
        return false;
    }

    return true;
}

std::unique_ptr<Graph> RandomUniformFusionPass::Replacement(const std::unique_ptr<MatchResult>& matchResult)
{
    OP_LOGD(kPassName.c_str(), "Enter Replacement");

    std::vector<SubgraphInput> subgraphInputs;
    matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs);

    std::vector<Shape> inputShapes;
    std::vector<DataType> inputDtypes;
    std::vector<Format> inputFormats;
    random_fusion_pass_common::GetInputsInfo(subgraphInputs, inputShapes, inputDtypes, inputFormats);

    NodeIo nodeIo;
    if (matchResult->GetCapturedTensor(kCaptureIdxNode, nodeIo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to GetCaptured tensor in Replacement.");
        return nullptr;
    }

    DataType dtype = DT_FLOAT;
    nodeIo.node.GetAttr("dtype", dtype);
    int64_t dtypeInt = static_cast<int64_t>(dtype);

    int64_t seed = 0;
    nodeIo.node.GetAttr("seed", seed);

    int64_t seed2 = 0;
    nodeIo.node.GetAttr("seed2", seed2);

    auto replaceGraphBuilder = es::EsGraphBuilder("replacement");
    auto rShape = replaceGraphBuilder.CreateInput(0, "shape", inputDtypes[0], inputFormats[0],
                                                  inputShapes[0].GetDims());

    AscendString nodeName;
    nodeIo.node.GetName(nodeName);
    std::string varName = std::string(nodeName.GetString()) + "/offsetVariable";

    TensorDesc offsetDesc(Shape({1}), FORMAT_ND, DT_INT64);

    auto rOffset = replaceGraphBuilder.CreateVariable(1, varName.c_str());
    random_fusion_pass_common::InitOffsetVariable(rOffset, offsetDesc);

    auto v2Output = es::RandomUniformV2(rShape, rOffset, dtypeInt, seed, seed2);
    GNode v2NodePtr = *v2Output.y.GetProducer();

    TensorDesc shapeInputDesc(inputShapes[0], inputFormats[0], inputDtypes[0]);
    v2NodePtr.UpdateInputDesc(0, shapeInputDesc);
    v2NodePtr.UpdateInputDesc(1, offsetDesc);

    TensorDesc outputYDesc;
    nodeIo.node.GetOutputDesc(0, outputYDesc);
    v2NodePtr.UpdateOutputDesc(0, outputYDesc);
    v2NodePtr.UpdateOutputDesc(1, offsetDesc);

    GraphUniqPtr replaceGraph = replaceGraphBuilder.BuildAndReset({v2Output.y});

    return replaceGraph;
}

#if GE_COMPILER_VERSION_NUM >= GE_COMPILER_VERSION_900
REG_FUSION_PASS(RandomUniformFusionPass).Stage(random_fusion_pass_common::GetRandomFusionPassStage());
#else
REG_FUSION_PASS(RandomUniformFusionPass).Stage(CustomPassStage::kBeforeInferShape);
#endif

} // namespace ops
