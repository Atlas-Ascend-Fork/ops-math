/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include "array_ops.h"
#include "ge_api.h"
#include "../op_graph/trace_proto.h"

namespace {
constexpr int64_t ROWS = 4;
constexpr int64_t COLS = 4;
constexpr float INPUT_VALUE = 2.0F;

int ReportError(const char* operation, ge::Status status)
{
    std::cerr << operation << " failed, status=" << status << ": " << ge::GEGetErrorMsgV2().GetString() << std::endl;
    return 1;
}

int RunTrace()
{
    std::vector<float> values(ROWS * COLS, INPUT_VALUE);
    ge::TensorDesc inputDesc(ge::Shape({ROWS, COLS}), ge::FORMAT_ND, ge::DT_FLOAT);
    inputDesc.SetRealDimCnt(2);
    inputDesc.SetPlacement(ge::kPlacementHost);
    ge::Tensor input(inputDesc);
    auto status = input.SetData(reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(float));
    if (status != ge::GRAPH_SUCCESS) {
        return ReportError("Tensor::SetData", status);
    }

    auto data = ge::op::Data("x").set_attr_index(0);
    data.update_input_desc_x(inputDesc);
    auto trace = ge::op::Trace("trace").set_input_x(data);
    trace.update_output_desc_y(ge::TensorDesc(ge::Shape(std::vector<int64_t>{}), ge::FORMAT_ND, ge::DT_FLOAT));
    ge::Graph graph("trace_example");
    status = graph.AddOp(data);
    if (status != ge::GRAPH_SUCCESS) {
        return ReportError("Graph::AddOp", status);
    }
    graph.SetInputs({data}).SetOutputs({trace});

    std::map<ge::AscendString, ge::AscendString> options;
    ge::Session session(options);
    status = session.AddGraph(0, graph);
    if (status != ge::SUCCESS) {
        return ReportError("Session::AddGraph", status);
    }
    std::vector<ge::Tensor> outputs;
    status = session.RunGraph(0, {input}, outputs);
    if (status != ge::SUCCESS) {
        return ReportError("Session::RunGraph", status);
    }
    if (outputs.size() != 1 || outputs[0].GetTensorDesc().GetDataType() != ge::DT_FLOAT ||
        outputs[0].GetTensorDesc().GetShape().GetDimNum() != 0 || outputs[0].GetData() == nullptr ||
        outputs[0].GetSize() < sizeof(float)) {
        std::cerr << "Trace must return one FLOAT scalar" << std::endl;
        return 1;
    }
    float actual = 0.0F;
    std::memcpy(&actual, outputs[0].GetData(), sizeof(actual));
    const float expected = std::min(ROWS, COLS) * INPUT_VALUE;
    if (!std::isfinite(actual) || std::fabs(actual - expected) > 1.0e-6F) {
        std::cerr << "Trace mismatch: actual=" << actual << ", expected=" << expected << std::endl;
        return 1;
    }
    const char* outputPath = "tc_ge_irrun_test_0008_npu_output_0.bin";
    std::ofstream outputFile(outputPath, std::ios::binary);
    outputFile.write(reinterpret_cast<const char*>(&actual), sizeof(actual));
    outputFile.close();
    if (!outputFile) {
        std::cerr << "Cannot write Trace result to " << outputPath << std::endl;
        return 1;
    }
    std::cout << "Trace result=" << actual << ", expected=" << expected << "; Precision is ok" << std::endl;
    return 0;
}
} // namespace

int main()
{
    std::map<ge::AscendString, ge::AscendString> options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    auto status = ge::GEInitialize(options);
    if (status != ge::SUCCESS) {
        return ReportError("GEInitialize", status);
    }
    int result = 1;
    try {
        result = RunTrace();
    } catch (const std::exception& error) {
        std::cerr << "Trace example failed: " << error.what() << std::endl;
    }
    // Session and its tensors leave scope before the GE runtime is finalized.
    status = ge::GEFinalize();
    if (status != ge::SUCCESS) {
        return ReportError("GEFinalize", status);
    }
    return result;
}
