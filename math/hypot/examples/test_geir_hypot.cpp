/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * @file test_geir_hypot.cpp
 * Hypot graph-mode (GE IR) call example.
 *
 * Builds a single-op graph  Data(self, other) -> Hypot -> out  aligned with
 * op_graph/hypot_proto.h (REG_OP(Hypot)), then runs it via GE Session
 * AddGraph/RunGraph. Inputs are FLOAT32 with broadcast shapes
 * self (4,2) all 3.0 and other (1,2) all 4.0, so every output element must be 5.0.
 *
 * Usage: ./test_geir_hypot [deviceId]   (default deviceId = 0)
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <map>
#include <string>
#include <vector>

#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

// op::Data (graph input placeholder) comes from the GE built-in op proto header.
#include "array_ops.h"
// Custom proto of this operator (REG_OP(Hypot)), consistent with op_graph registration;
// provides ge::op::Hypot.
#include "../op_graph/hypot_proto.h"

#define FAILED (-1)
#define SUCCESS 0

using namespace ge;

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

static std::string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64] = {0};
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S", localtime(&timep));
    return std::string(tmp);
}

// Fill a FLOAT32 tensor with a constant value.
static int32_t GenConstFloatData(const std::vector<int64_t>& shapes, Tensor& tensor, TensorDesc& desc, float value)
{
    desc.SetRealDimCnt(static_cast<uint32_t>(shapes.size()));
    size_t size = 1;
    for (auto d : shapes) {
        size *= static_cast<size_t>(d);
    }
    std::vector<float> buf(size, value);
    tensor = Tensor(desc, reinterpret_cast<uint8_t*>(buf.data()), size * sizeof(float));
    return SUCCESS;
}

// Build the single-op graph: placeholder(self), placeholder(other) -> Hypot -> out (broadcast (4,2)x(1,2)).
static int BuildHypotGraph(std::vector<Tensor>& graphInputs, std::vector<Operator>& inputs,
                           std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    const DataType dtype = DT_FLOAT;

    std::vector<int64_t> selfShape = {4, 2};
    std::vector<int64_t> otherShape = {1, 2};
    TensorDesc selfDesc(ge::Shape(selfShape), FORMAT_ND, dtype);
    TensorDesc otherDesc(ge::Shape(otherShape), FORMAT_ND, dtype);
    selfDesc.SetPlacement(ge::kPlacementHost);
    otherDesc.SetPlacement(ge::kPlacementHost);

    auto placeholderSelf = op::Data("self_placeholder").set_attr_index(0);
    placeholderSelf.update_input_desc_x(selfDesc);
    placeholderSelf.update_output_desc_y(selfDesc);

    auto placeholderOther = op::Data("other_placeholder").set_attr_index(1);
    placeholderOther.update_input_desc_x(otherDesc);
    placeholderOther.update_output_desc_y(otherDesc);

    // Runtime data: self = 3.0, other = 4.0 -> out = hypot(3,4) = 5.0 everywhere.
    Tensor selfTensor;
    Tensor otherTensor;
    ret = GenConstFloatData(selfShape, selfTensor, selfDesc, 3.0F);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: Generate self data failed\n", GetTime().c_str());
        return FAILED;
    }
    ret = GenConstFloatData(otherShape, otherTensor, otherDesc, 4.0F);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: Generate other data failed\n", GetTime().c_str());
        return FAILED;
    }
    graphInputs.push_back(selfTensor);
    graphInputs.push_back(otherTensor);

    // Hypot node (OpType consistent with REG_OP(Hypot)).
    auto hypot = op::Hypot("hypot_0");
    hypot.set_input_self(placeholderSelf);
    hypot.set_input_other(placeholderOther);
    hypot.update_input_desc_self(selfDesc);
    hypot.update_input_desc_other(otherDesc);
    TensorDesc outDesc(ge::Shape(selfShape), FORMAT_ND, dtype); // broadcast result (4,2)
    hypot.update_output_desc_out(outDesc);

    graph.AddOp(placeholderSelf);
    graph.AddOp(placeholderOther);
    graph.AddOp(hypot);

    inputs.push_back(placeholderSelf);
    inputs.push_back(placeholderOther);
    outputs.push_back(hypot);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    (void)argc;
    const char* deviceId = (argc > 1) ? argv[1] : "0";
    LOG_PRINT("=== Hypot GE IR (graph mode) example ===\n");
    LOG_PRINT("deviceId: %s\n", deviceId);

    const char* graphName = "hypot_geir_test";
    Graph graph(graphName);
    std::vector<Tensor> graphInputs;
    std::vector<Operator> inputs;
    std::vector<Operator> outputs;

    LOG_PRINT("%s - INFO - [XIR]: GEInitialize\n", GetTime().c_str());
    std::map<AscendString, AscendString> globalOptions = {
        {"ge.exec.deviceId", deviceId},
        {"ge.graphRunMode", "1"},
    };
    Status ret = ge::GEInitialize(globalOptions);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: GEInitialize failed\n", GetTime().c_str());
        return FAILED;
    }

    ret = BuildHypotGraph(graphInputs, inputs, outputs, graph);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: Build graph failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }
    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    LOG_PRINT("%s - INFO - [XIR]: create Session\n", GetTime().c_str());
    std::map<AscendString, AscendString> buildOptions;
    ge::Session* session = new (std::nothrow) Session(buildOptions);
    if (session == nullptr) {
        LOG_PRINT("%s - ERROR - [XIR]: create Session failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }

    uint32_t graphId = 0;
    std::map<AscendString, AscendString> graphOptions;
    ret = session->AddGraph(graphId, graph, graphOptions);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: AddGraph failed\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return FAILED;
    }
    LOG_PRINT("%s - INFO - [XIR]: AddGraph success\n", GetTime().c_str());

    LOG_PRINT("%s - INFO - [XIR]: RunGraph\n", GetTime().c_str());
    std::vector<Tensor> graphOutputs;
    ret = session->RunGraph(graphId, graphInputs, graphOutputs);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: RunGraph failed\n", GetTime().c_str());
        ge::AscendString errorMsg = ge::GEGetErrorMsgV2();
        LOG_PRINT("%s - ERROR - [XIR]: GE error message: %s\n", GetTime().c_str(), errorMsg.GetString());
        delete session;
        ge::GEFinalize();
        return FAILED;
    }
    LOG_PRINT("%s - INFO - [XIR]: RunGraph success\n", GetTime().c_str());

    // Print outputs and verify against golden (hypot(3,4) = 5).
    bool pass = true;
    for (size_t i = 0; i < graphOutputs.size(); ++i) {
        const float* out = reinterpret_cast<const float*>(graphOutputs[i].GetData());
        int64_t numel = graphOutputs[i].GetTensorDesc().GetShape().GetShapeSize();
        for (int64_t j = 0; j < numel; ++j) {
            double golden = std::hypot(3.0, 4.0); // 5.0
            bool ok = std::fabs(static_cast<double>(out[j]) - golden) <= 1e-5;
            if (!ok) {
                pass = false;
            }
            LOG_PRINT("    out[%ld] = %f  golden=%f  %s\n", j, out[j], golden, ok ? "OK" : "MISMATCH");
        }
    }

    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: GEFinalize failed\n", GetTime().c_str());
        return FAILED;
    }
    LOG_PRINT("%s - INFO - [XIR]: done\n", GetTime().c_str());
    LOG_PRINT("Hypot GE IR example overall: %s\n", pass ? "PASS" : "FAIL");
    return pass ? SUCCESS : FAILED;
}
