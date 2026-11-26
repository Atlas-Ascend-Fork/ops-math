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
 * \file random_fusion_pass_common.h
 * \brief Shared helpers for random series fusion passes (version/platform gating, input info, offset variable).
 */
#ifndef RANDOM_FUSION_PASS_COMMON_H
#define RANDOM_FUSION_PASS_COMMON_H

#include <cstdint>
#include <cstdlib>
#include <vector>
#include <string>
#include <set>
#include <algorithm>
#include "platform/soc_spec.h"
#include "ge/ge_utils.h"
#include "ge/es_tensor_holder.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "log/log.h"
#include "version/ge-compiler_version.h"

extern "C" {
static const int32_t RT_ERROR_NONE = 0;
int32_t rtGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen);
}

// 与 common/inc/op_graph/fusion_pass/fusion_pass_common.h 保持一致的声明作用域（匿名命名空间），
// 同一 TU 内两者合并为同一实体，避免同名 extern "C" 函数形成重载集。
namespace {
extern "C" {
__attribute__((weak)) int32_t aclsysGetVersionNum(char* pkgName, int32_t* versionNum);
}
} // namespace

// D1 scenario: uses kCompatibleInherited stage (9.0.0+).
#ifndef GE_COMPILER_VERSION_900
#define GE_COMPILER_VERSION_900 90000000
#endif

namespace random_fusion_pass_common {

// Runtime check: GE compiler version >= 9.0.0. The resolved version is returned through `version`
// so callers can still log the concrete value for troubleshooting.
inline bool IsVersion900(int32_t& version)
{
    version = 0;
    if (aclsysGetVersionNum) {
        aclsysGetVersionNum(const_cast<char*>("ge_compiler"), &version);
    }
    return version >= GE_COMPILER_VERSION_900;
}

#if GE_COMPILER_VERSION_NUM >= GE_COMPILER_VERSION_900
// Select the registered stage from the runtime ge_compiler version.
// >= 9.0.0 -> kCompatibleInherited; otherwise kBeforeInferShape.
inline ge::CustomPassStage GetRandomFusionPassStage()
{
    int32_t version = 0;
    if (IsVersion900(version)) {
        return ge::CustomPassStage::kCompatibleInherited;
    }
    return ge::CustomPassStage::kBeforeInferShape;
}
#endif

// Platform gate based on NpuArch. The default set covers DAV_3510 (Ascend950/Ascend350)
// and DAV_9201/DAV_9202 (Ascend960DT/Ascend960PR); passes supporting extra archs
// (e.g. DAV_2201 for Ascend910_93) pass their own supported set.
inline bool IsTargetPlatformNpuArch(const std::string& passName, const std::set<NpuArch>& supportedArch)
{
    char npuArchVal[32] = {0};
    if (rtGetSocSpec("version", "NpuArch", npuArchVal, sizeof(npuArchVal)) != RT_ERROR_NONE) {
        OP_LOGW(passName.c_str(), "Get NpuArch from soc spec failed, skip whole graph.");
        return false;
    }
    const auto npuArch = static_cast<NpuArch>(std::strtol(npuArchVal, nullptr, 10));
    const bool isSupported = (supportedArch.count(npuArch) > 0);
    if (!isSupported) {
        OP_LOGI(passName.c_str(), "Platform NpuArch %u is not supported, skip.", static_cast<uint32_t>(npuArch));
    }
    return isSupported;
}

inline bool IsTargetPlatformNpuArch(const std::string& passName)
{
    static const std::set<NpuArch> supportedArch = {NpuArch::DAV_3510, NpuArch::DAV_9201, NpuArch::DAV_9202};
    return IsTargetPlatformNpuArch(passName, supportedArch);
}

// Check whether the dtype is in the pass supported dtype list.
inline bool CheckDtype(ge::DataType dtype, const std::vector<ge::DataType>& validTypes)
{
    return std::find(validTypes.begin(), validTypes.end(), dtype) != validTypes.end();
}

// Convert a ge::Shape to a dims vector for the ES graph builder APIs.
inline std::vector<int64_t> GetShapeDims(const ge::Shape& shape)
{
    std::vector<int64_t> dims;
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        dims.push_back(shape.GetDim(i));
    }
    return dims;
}

// Collect input shapes from the subgraph boundary.
inline void GetInputShapes(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                           std::vector<ge::Shape>& inputShapes)
{
    for (const auto& subgraphInput : subgraphInputs) {
        auto matchNode = subgraphInput.GetAllInputs().at(0);
        ge::TensorDesc tensorDesc;
        matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
        inputShapes.emplace_back(tensorDesc.GetShape());
    }
}

// Collect input shapes/dtypes/formats from the subgraph boundary.
inline void GetInputsInfo(const std::vector<ge::fusion::SubgraphInput>& subgraphInputs,
                          std::vector<ge::Shape>& inputShapes, std::vector<ge::DataType>& inputDtypes,
                          std::vector<ge::Format>& inputFormats)
{
    GetInputShapes(subgraphInputs, inputShapes);
    for (const auto& subgraphInput : subgraphInputs) {
        auto matchNode = subgraphInput.GetAllInputs().at(0);
        ge::TensorDesc tensorDesc;
        matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
        inputDtypes.emplace_back(tensorDesc.GetDataType());
        inputFormats.emplace_back(tensorDesc.GetFormat());
    }
}

// Infer the shape of the replacement graph from the subgraph inputs.
inline ge::Status InferShape(const ge::fusion::GraphUniqPtr& replaceGraph,
                             const std::vector<ge::fusion::SubgraphInput>& subgraphInputs)
{
    std::vector<ge::Shape> inputShapes;
    GetInputShapes(subgraphInputs, inputShapes);
    return ge::GeUtils::InferShape(*replaceGraph, inputShapes);
}

// Initialize the offset variable node (constant 0, int64) required by the V2 interface.
inline void InitOffsetVariable(const ge::es::EsTensorHolder& rOffset, const ge::TensorDesc& offsetDesc)
{
    auto varNode = rOffset.GetProducer();
    if (varNode != nullptr) {
        varNode->UpdateOutputDesc(0, offsetDesc);
        int64_t initValue = 0;
        ge::Tensor initTensor(offsetDesc, reinterpret_cast<uint8_t*>(&initValue), sizeof(int64_t));
        varNode->SetAttr("init_value", initTensor);
    }
}

} // namespace random_fusion_pass_common

#endif // RANDOM_FUSION_PASS_COMMON_H
