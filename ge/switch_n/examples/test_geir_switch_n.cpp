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
 * \file test_geir_switch_n.cpp
 * \brief Graph-mode example for the SwitchN operator (RunGraph + BuildModel).
 */
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_api.h"
#include "ge_ir_build.h"
#include "../op_graph/switch_n_proto.h"
#include "../../data/op_graph/data_proto.h"

using namespace ge;

namespace {
constexpr int kFailed = -1;
constexpr int kSuccess = 0;
constexpr uint32_t kDeviceId = 0;
constexpr DataType kDtype = DT_FLOAT;
constexpr size_t kOutputNum = 3U;

struct RunContext {
    std::map<AscendString, AscendString> session_options;
    uint32_t graph_id = 0;
};

std::vector<std::pair<int64_t, int64_t>> MakeRange(const std::vector<int64_t>& shape)
{
    std::vector<std::pair<int64_t, int64_t>> range;
    for (auto d : shape) {
        range.emplace_back(d < 0 ? 1 : d, d < 0 ? 10 : d);
    }
    return range;
}

int BuildGraph(Graph& graph, const std::vector<int64_t>& data_shape, std::vector<Operator>& graph_inputs,
               Operator& graph_output, bool is_dynamic)
{
    TensorDesc data_desc(Shape(data_shape), FORMAT_ND, kDtype);
    data_desc.SetPlacement(kPlacementHost);
    if (is_dynamic) {
        data_desc.SetShapeRange(MakeRange(data_shape));
    }
    auto data = op::Data("data").set_attr_index(0);
    data.update_input_desc_x(data_desc);
    data.update_output_desc_y(data_desc);

    TensorDesc pred_desc(Shape(std::vector<int64_t>{}), FORMAT_ND, DT_INT64);
    pred_desc.SetPlacement(kPlacementHost);
    auto pred = op::Data("pred_value").set_attr_index(1);
    pred.update_input_desc_x(pred_desc);
    pred.update_output_desc_y(pred_desc);

    auto swn = op::SwitchN("switch_n_op");
    swn.set_input_data(data);
    swn.set_input_pred_value(pred);
    swn.create_dynamic_output_output(kOutputNum);

    graph.AddOp(data);
    graph.AddOp(pred);
    graph.AddOp(swn);
    graph_inputs = {data, pred};
    graph_output = swn;
    return kSuccess;
}

int CheckInferShape(const Graph& graph, const std::vector<int64_t>& expected_dims)
{
    Operator op_obj;
    if (graph.FindOpByName("switch_n_op", op_obj) != GRAPH_SUCCESS) {
        std::cerr << "CheckInferShape: FindOpByName failed" << std::endl;
        return kFailed;
    }
    for (size_t output = 0; output < kOutputNum; ++output) {
        const auto desc = op_obj.GetOutputDesc(output);
        const auto shape = desc.GetShape();
        std::cout << "InferShape[output" << output << "]: dim_num=" << shape.GetDimNum();
        for (size_t dim = 0; dim < shape.GetDims().size(); ++dim) {
            std::cout << " dim" << dim << "=" << shape.GetDim(dim);
        }
        std::cout << " shape_size=" << shape.GetShapeSize() << " dtype=" << desc.GetDataType() << std::endl;
        if (shape.GetDims() != expected_dims || desc.GetDataType() != kDtype) {
            std::cerr << "InferShape[output" << output << "]: shape or dtype mismatch" << std::endl;
            return kFailed;
        }
    }
    return kSuccess;
}

int CheckOutput(const std::vector<Tensor>& outputs, const std::vector<int64_t>& expected_dims,
                const std::vector<float>& expected_values, int64_t pred_value)
{
    // The graph exposes only the selected port, so an inactive port cannot pass as an empty output.
    if (outputs.size() != 1U) {
        std::cerr << "CheckOutput: expected 1 output, got " << outputs.size() << std::endl;
        return kFailed;
    }
    const auto desc = outputs[0].GetTensorDesc();
    if (desc.GetShape().GetDims() != expected_dims || desc.GetDataType() != kDtype ||
        desc.GetShape().GetShapeSize() != static_cast<int64_t>(expected_values.size())) {
        std::cerr << "CheckOutput: shape, element count or dtype mismatch" << std::endl;
        return kFailed;
    }
    if (outputs[0].GetData() == nullptr || outputs[0].GetSize() < expected_values.size() * sizeof(float)) {
        std::cerr << "CheckOutput: data buffer is missing or too small" << std::endl;
        return kFailed;
    }
    for (size_t i = 0; i < expected_values.size(); ++i) {
        float value = 0.0f;
        std::memcpy(&value, outputs[0].GetData() + i * sizeof(float), sizeof(value));
        if (value != expected_values[i]) {
            std::cerr << "CheckOutput: element " << i << " expected=" << expected_values[i] << " actual=" << value
                      << std::endl;
            return kFailed;
        }
    }
    std::cout << "SwitchNBranchCheck success: pred_value=" << pred_value << " elements=" << expected_values.size()
              << std::endl;
    return kSuccess;
}

int RunScenario(RunContext& ctx, const std::string& name, const std::vector<int64_t>& shape, bool is_dynamic,
                int64_t pred_value)
{
    std::cout << "=== Scenario: " << name << " ===" << std::endl;

    Graph graph(("switch_n_" + name).c_str());
    std::vector<Operator> graph_inputs;
    Operator graph_output;
    if (BuildGraph(graph, shape, graph_inputs, graph_output, is_dynamic) != kSuccess) {
        std::cerr << name << " FAILED: BuildGraph" << std::endl;
        return kFailed;
    }
    graph.SetInputs(graph_inputs);
    graph.SetOutputs(
        std::vector<std::pair<Operator, std::vector<size_t>>>{{graph_output, {static_cast<size_t>(pred_value)}}});

    std::vector<int64_t> actual;
    for (auto d : shape) {
        actual.push_back(d < 0 ? 2 : d);
    }
    size_t cnt = 1;
    for (auto d : actual) {
        cnt *= static_cast<size_t>(d);
    }

    Session session(ctx.session_options);
    uint32_t gid = ctx.graph_id++;
    if (session.AddGraph(gid, graph) != SUCCESS) {
        std::cerr << name << " FAILED: AddGraph" << std::endl;
        return kFailed;
    }
    std::vector<float> dvals(cnt);
    for (size_t i = 0; i < cnt; ++i) {
        dvals[i] = static_cast<float>(i + 1U);
    }
    Tensor data_tensor(TensorDesc(Shape(actual), FORMAT_ND, kDtype), reinterpret_cast<const uint8_t*>(dvals.data()),
                       cnt * sizeof(float));
    Tensor pred_tensor(TensorDesc(Shape(std::vector<int64_t>{}), FORMAT_ND, DT_INT64),
                       reinterpret_cast<const uint8_t*>(&pred_value), sizeof(pred_value));
    std::vector<Tensor> inputs = {data_tensor, pred_tensor};
    std::vector<Tensor> outputs;

    bool run_ok = false;
    Status ret = session.RunGraph(gid, inputs, outputs);
    if (ret != SUCCESS) {
        std::cerr << name << ": RunGraph failed (ret=" << ret << ")" << std::endl;
    } else if (outputs.empty()) {
        std::cerr << name << ": RunGraph success but output.size()==0" << std::endl;
    } else {
        std::cout << name << ": RunGraph success, outputs=" << outputs.size() << std::endl;
        run_ok = CheckOutput(outputs, actual, dvals, pred_value) == kSuccess;
    }

    Graph build_graph(("switch_n_build_" + name).c_str());
    std::vector<Operator> build_inputs;
    Operator build_output;
    if (BuildGraph(build_graph, shape, build_inputs, build_output, is_dynamic) != kSuccess) {
        std::cerr << name << " FAILED: BuildGraph for build" << std::endl;
        return kFailed;
    }
    build_graph.SetInputs(build_inputs);
    build_graph.SetOutputs(
        std::vector<std::pair<Operator, std::vector<size_t>>>{{build_output, {static_cast<size_t>(pred_value)}}});

    ModelBufferData model;
    std::map<AscendString, AscendString> build_opts;
    if (aclgrphBuildModel(build_graph, build_opts, model) != SUCCESS) {
        std::cerr << name << " FAILED: BuildModel" << std::endl;
        return kFailed;
    }
    std::cout << name << ": BuildModel success, size=" << model.length << std::endl;

    std::string om_file = name;
    if (aclgrphSaveModel(om_file.c_str(), model) != SUCCESS) {
        std::cerr << name << " FAILED: SaveModel" << std::endl;
        return kFailed;
    }
    std::cout << name << ": SaveModel success, file=" << om_file << std::endl;

    if (run_ok) {
        if (CheckInferShape(graph, actual) != kSuccess) {
            return kFailed;
        }
        std::cout << name << " PASSED" << std::endl;
        return kSuccess;
    }
    std::cerr << name << " NOT PASSED (RunGraph did not produce valid output)" << std::endl;
    return kFailed;
}
} // namespace

int main()
{
    std::map<AscendString, AscendString> global_options = {
        {"ge.exec.deviceId", std::to_string(kDeviceId).c_str()},
        {"ge.graphRunMode", "1"},
    };
    if (GEInitialize(global_options) != SUCCESS) {
        std::cerr << "GEInitialize failed" << std::endl;
        return kFailed;
    }

    RunContext ctx;
    ctx.session_options["ge.jit_compile"] = "1";
    int ret = kSuccess;
    for (size_t output = 0; output < kOutputNum; ++output) {
        const auto pred_value = static_cast<int64_t>(output);
        const auto suffix = "_output" + std::to_string(output);
        if (RunScenario(ctx, "static" + suffix, {2, 3}, false, pred_value) != kSuccess) {
            ret = kFailed;
        }
        if (RunScenario(ctx, "dynamic_minus1" + suffix, {-1, 3}, true, pred_value) != kSuccess) {
            ret = kFailed;
        }
        if (RunScenario(ctx, "dynamic_minus2" + suffix, {-2}, true, pred_value) != kSuccess) {
            ret = kFailed;
        }
    }

    GEFinalize();
    if (ret == kSuccess) {
        std::cout << "All scenarios PASSED" << std::endl;
    } else {
        std::cout << "Some scenarios FAILED" << std::endl;
    }
    return ret;
}
