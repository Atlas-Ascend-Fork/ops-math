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
#include <map>
#include <string>
#include <vector>

#include "ge_api.h"
#include "ge_api_types.h"

#include "inv_grad_geir_test_common.h"

namespace {

struct DynamicScenario {
    const char* name;
    std::vector<int64_t> declaredShape;
    std::vector<std::vector<int64_t>> concreteShapes;
};

bool RunScenario(ge::Session& session, uint32_t graphId, const DynamicScenario& scenario)
{
    std::printf("Scenario %s, declared shape %s\n", scenario.name,
                inv_grad_geir_test::ShapeToJson(scenario.declaredShape).c_str());

    const std::string graphName = std::string("inv_grad_") + scenario.name;
    ge::Graph graph(graphName.c_str());
    if (!inv_grad_geir_test::BuildGraph(graph, scenario.declaredShape, ge::DT_FLOAT, scenario.name)) {
        return false;
    }
    std::map<ge::AscendString, ge::AscendString> graphOptions = {
        {"ge.exec.dynamicGraphExecuteMode", "dynamic_execute"}};
    if (!(scenario.declaredShape.size() == 1U && scenario.declaredShape[0] == -2)) {
        graphOptions.emplace("ge.exec.dataInputsShapeRange", "[1~16,1~16],[1~16,1~16]");
    }
    const ge::Status addStatus = session.AddGraph(graphId, graph, graphOptions);
    if (addStatus != ge::SUCCESS) {
        std::printf("AddGraph failed for %s, status=%d\n", scenario.name, addStatus);
        return false;
    }

    size_t passed = 0;
    for (const std::vector<int64_t>& shape : scenario.concreteShapes) {
        std::printf("Run concrete shape %s\n", inv_grad_geir_test::ShapeToJson(shape).c_str());
        ge::Tensor x;
        ge::Tensor grad;
        if (!inv_grad_geir_test::MakeFilledTensor(shape, ge::DT_FLOAT, 2.0, x) ||
            !inv_grad_geir_test::MakeFilledTensor(shape, ge::DT_FLOAT, 3.0, grad)) {
            std::printf("Make input tensors failed for %s\n", inv_grad_geir_test::ShapeToJson(shape).c_str());
            return false;
        }
        std::vector<ge::Tensor> outputs;
        const ge::Status runStatus = session.RunGraph(graphId, {x, grad}, outputs);
        if (runStatus != ge::SUCCESS) {
            std::printf("RunGraph failed for %s, status=%d\n", inv_grad_geir_test::ShapeToJson(shape).c_str(),
                        runStatus);
            return false;
        }
        // 打印 RunGraph 状态与实际输出元数据，区分「RunGraph 失败」与
        // 「RunGraph 成功但输出 TensorDesc 未刷新导致的校验失败」（F28 可观测性）
        std::printf("RunGraph status=%d, outputs=%zu\n", runStatus, outputs.size());
        if (!outputs.empty()) {
            const ge::TensorDesc& outputDesc = outputs[0].GetTensorDesc();
            std::printf("Output metadata: shape=%s, dtype=%s\n",
                        inv_grad_geir_test::ShapeToJson(outputDesc.GetShape()).c_str(),
                        inv_grad_geir_test::DtypeName(outputDesc.GetDataType()));
        }
        if (!inv_grad_geir_test::ValidateOutput(outputs, shape, ge::DT_FLOAT, -12.0)) {
            std::printf("ValidateOutput failed for %s: expected shape=%s, dtype=float32, value=-12\n",
                        inv_grad_geir_test::ShapeToJson(shape).c_str(), inv_grad_geir_test::ShapeToJson(shape).c_str());
            return false;
        }
        ++passed;
        std::printf("Shape, dtype and values PASSED for %s\n", inv_grad_geir_test::ShapeToJson(shape).c_str());
    }

    std::printf("Scenario %s summary: %zu/%zu passed\n", scenario.name, passed, scenario.concreteShapes.size());
    return passed == scenario.concreteShapes.size();
}

} // namespace

int main()
{
    std::map<ge::AscendString, ge::AscendString> globalOptions = {
        {"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}, {"ge.exec.dynamicInput", "1"}};
    if (ge::GEInitialize(globalOptions) != ge::SUCCESS) {
        std::printf("InvGrad dynamic GEIR initialization failure\n");
        return -1;
    }

    const DynamicScenario unknownDim = {"unknown_dim_minus_1", {-1, -1}, {{4, 2}, {1, 8}, {3, 5}}};
    const DynamicScenario unknownRank = {"unknown_rank_minus_2", {-2}, {{8}, {4, 2}, {2, 3, 4}}};

    bool dimPassed = false;
    {
        std::map<ge::AscendString, ge::AscendString> sessionOptions;
        ge::Session session(sessionOptions);
        dimPassed = RunScenario(session, 1U, unknownDim);
    }
    bool rankPassed = false;
    if (dimPassed) {
        std::map<ge::AscendString, ge::AscendString> sessionOptions;
        ge::Session session(sessionOptions);
        rankPassed = RunScenario(session, 2U, unknownRank);
    }
    const ge::Status finalizeStatus = ge::GEFinalize();
    if (!dimPassed || !rankPassed || finalizeStatus != ge::SUCCESS) {
        return -1;
    }
    std::printf("InvGrad dynamic GEIR verification PASSED (-1 and -2)\n");
    return 0;
}
