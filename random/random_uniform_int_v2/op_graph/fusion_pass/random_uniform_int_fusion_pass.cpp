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
 * \file random_uniform_int_fusion_pass.cpp
 * \brief RandomUniformInt fusion pass (RandomUniformInt --> RandomUniformIntV2)
 *
 *      shape  min  max                        shape  min  max  offset(variable)
 *         \    |    /                            \    |    /    /   |
 *      RandomUniformInt          ==>          RandomUniformIntV2    |
 *              |                                   |          \     |
 *              y                                   y           offset
 */

#include <vector>
#include <string>
#include <set>
#include "es_math_ops.h"
#include "platform/platform_info.h"
#include "ge/ge_utils.h"
#include "log/log.h"
// 版本兼容: 编译宏头(GE_COMPILER_VERSION_NUM), 用于判断 toolkit 版本
#include "version/ge-compiler_version.h"
#include "random_uniform_int_fusion_pass.h"
#include "random/random_common/op_graph/random_fusion_pass_common.h"

using namespace ge;
using namespace fe;
using namespace fusion;

namespace ops {

static const std::string kPassName = "RandomUniformIntFusionPass";
static const int64_t kCaptureIdxNode = 0L;
static const std::set<DataType> kAicoreDtypeSupportList = {DT_INT32, DT_INT64};

std::vector<PatternUniqPtr> RandomUniformIntFusionPass::Patterns()
{
    OP_LOGD(kPassName.c_str(), "Enter Patterns");
    std::vector<PatternUniqPtr> patternGraphs;

    auto graphBuilder = es::EsGraphBuilder(kPassName.c_str());
    auto shape = graphBuilder.CreateInput(0);
    auto min = graphBuilder.CreateInput(1);
    auto max = graphBuilder.CreateInput(2);

    auto output = es::RandomUniformInt(shape, min, max);
    auto graph = graphBuilder.BuildAndReset({output});

    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*output.GetProducer(), 0});

    patternGraphs.emplace_back(std::move(pattern));
    return patternGraphs;
}

bool RandomUniformIntFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    OP_LOGD(kPassName.c_str(), "Enter MeetRequirements");

    // 版本兼容(D1 整体静默): 9.x 编出的包跑在 8.5.0 运行时上时,
    // pass 被注册到老 stage kBeforeInferShape, 此处空跑, 不发生融合。
    int32_t version = 0;
    if (!random_fusion_pass_common::IsVersion900(version)) {
        OP_LOGD(kPassName.c_str(), "ge_compiler runtime version %d < 9.0.0, skip fusion.", version);
        return false;
    }

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
    if (std::string(nodeType.GetString()) != "RandomUniformInt") {
        OP_LOGD(kPassName.c_str(), "Node type is not RandomUniformInt, skip.");
        return false;
    }

    TensorDesc minDesc;
    nodeIo.node.GetInputDesc(1, minDesc);
    if (kAicoreDtypeSupportList.count(minDesc.GetDataType()) == 0) {
        OP_LOGD(kPassName.c_str(), "Dtype %d not supported.", static_cast<int>(minDesc.GetDataType()));
        return false;
    }

    return true;
}

std::unique_ptr<Graph> RandomUniformIntFusionPass::Replacement(const std::unique_ptr<MatchResult>& matchResult)
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

    int64_t seed = 0;
    nodeIo.node.GetAttr("seed", seed);

    int64_t seed2 = 0;
    nodeIo.node.GetAttr("seed2", seed2);

    auto replaceGraphBuilder = es::EsGraphBuilder("replacement");

    auto rShape = replaceGraphBuilder.CreateInput(0, "shape", inputDtypes[0], inputFormats[0],
                                                  inputShapes[0].GetDims());
    auto rMin = replaceGraphBuilder.CreateInput(1, "min", inputDtypes[1], inputFormats[1], inputShapes[1].GetDims());
    auto rMax = replaceGraphBuilder.CreateInput(2, "max", inputDtypes[2], inputFormats[2], inputShapes[2].GetDims());

    AscendString nodeName;
    nodeIo.node.GetName(nodeName);
    std::string varName = std::string(nodeName.GetString()) + "/offsetVariable";

    TensorDesc offsetDesc(Shape({1}), FORMAT_ND, DT_INT64);

    auto rOffset = replaceGraphBuilder.CreateVariable(3, varName.c_str());
    random_fusion_pass_common::InitOffsetVariable(rOffset, offsetDesc);

    auto v2Output = es::RandomUniformIntV2(rShape, rMin, rMax, rOffset, seed, seed2);
    GNode v2NodePtr = *v2Output.y.GetProducer();

    TensorDesc shapeInputDesc(inputShapes[0], inputFormats[0], inputDtypes[0]);
    TensorDesc minInputDesc(inputShapes[1], inputFormats[1], inputDtypes[1]);
    TensorDesc maxInputDesc(inputShapes[2], inputFormats[2], inputDtypes[2]);
    v2NodePtr.UpdateInputDesc(0, shapeInputDesc);
    v2NodePtr.UpdateInputDesc(1, minInputDesc);
    v2NodePtr.UpdateInputDesc(2, maxInputDesc);
    v2NodePtr.UpdateInputDesc(3, offsetDesc);

    TensorDesc outputYDesc;
    nodeIo.node.GetOutputDesc(0, outputYDesc);
    v2NodePtr.UpdateOutputDesc(0, outputYDesc);
    v2NodePtr.UpdateOutputDesc(1, offsetDesc);

    GraphUniqPtr replaceGraph = replaceGraphBuilder.BuildAndReset({v2Output.y});
    return replaceGraph;
}

// 版本兼容(D1 整体静默): 注册 stage 编译宏保护
//   - GE_COMPILER_VERSION_NUM >= 9.0.0: 头文件含 kCompatibleInherited, 运行时再按版本选 stage
//   - 老 8.5.0 toolkit: 枚举不存在, 降级注册到 8.5.0 已有的 kBeforeInferShape(运行时 MeetRequirements 空跑)
#if GE_COMPILER_VERSION_NUM >= 90000000
REG_FUSION_PASS(RandomUniformIntFusionPass).Stage(random_fusion_pass_common::GetRandomFusionPassStage());
#else
REG_FUSION_PASS(RandomUniformIntFusionPass).Stage(CustomPassStage::kBeforeInferShape);
#endif

} // namespace ops
