/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_api.h"
#include "ge_ir_build.h"
#include "ops_proto_math.h"

namespace ge {
REG_OP(Data)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(Data)

        REG_OP(Identity)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(Identity)
}

using namespace ge;

namespace {
constexpr int kFailed = -1;
constexpr int kSuccess = 0;
constexpr uint32_t kDeviceId = 0;
constexpr DataType kDtype = DT_FLOAT;

struct RunContext {
    std::map<AscendString, AscendString> session_options;
    uint32_t graph_id = 0;
};

// ControlTrigger has no data I/O — it is only a placeholder for control edges,
// similar to NoOp but always produces a live control output even when some
// control inputs are dead. The example builds a Data → Identity → output data
// path and adds ControlTrigger to the same graph.
int BuildControlTriggerGraph(Graph& graph, const std::vector<int64_t>& shape, std::vector<Operator>& graph_inputs,
                             Operator& graph_output, bool is_dynamic)
{
    auto input_data = op::Data("input").set_attr_index(0);
    TensorDesc idesc(Shape(shape), FORMAT_ND, kDtype);
    if (is_dynamic) {
        std::vector<std::pair<int64_t, int64_t>> range;
        for (auto d : shape) {
            range.emplace_back(d < 0 ? 1 : d, d < 0 ? 10 : d);
        }
        idesc.SetShapeRange(range);
    }
    input_data.update_input_desc_x(idesc);
    input_data.update_output_desc_y(idesc);

    auto identity = op::Identity("identity");
    identity.set_input_x(input_data);
    identity.update_output_desc_y(TensorDesc(Shape(shape), FORMAT_ND, kDtype));

    auto ctrl_trigger = op::ControlTrigger("control_trigger");

    graph.AddOp(input_data);
    graph.AddOp(identity);
    graph.AddOp(ctrl_trigger);
    graph_inputs = {input_data};
    graph_output = identity;
    return kSuccess;
}

int CheckInferShape(const Operator& op, const std::vector<int64_t>& expected_dims)
{
    TensorDesc desc = op.GetOutputDesc(0);
    Shape shape = desc.GetShape();
    std::cout << "InferShape: dim_num=" << shape.GetDimNum();
    for (size_t i = 0; i < shape.GetDims().size(); i++) {
        std::cout << " dim" << i << "=" << shape.GetDim(i);
    }
    std::cout << " shape_size=" << shape.GetShapeSize() << " dtype=" << desc.GetDataType() << std::endl;
    return kSuccess;
}

int CheckOutput(const std::vector<Tensor>& outputs, const std::vector<int64_t>& expected_dims)
{
    if (outputs.empty()) {
        std::cerr << "CheckOutput: output is empty (output.size()==0)" << std::endl;
        return kFailed;
    }
    const auto& tensor = outputs[0];
    const auto& desc = tensor.GetTensorDesc();
    Shape shape = desc.GetShape();
    std::cout << "Output: dim_num=" << shape.GetDimNum();
    for (size_t i = 0; i < shape.GetDims().size(); i++) {
        std::cout << " dim" << i << "=" << shape.GetDim(i);
    }
    std::cout << " shape_size=" << shape.GetShapeSize() << " dtype=" << desc.GetDataType() << std::endl;

    if (shape.GetDimNum() != expected_dims.size()) {
        std::cerr << "CheckOutput: dim_num mismatch, expected=" << expected_dims.size()
                  << " actual=" << shape.GetDimNum() << std::endl;
        return kFailed;
    }
    for (size_t i = 0; i < expected_dims.size(); i++) {
        int64_t actual_dim = shape.GetDim(i);
        if (actual_dim == -1) {
            continue;
        }
        if (actual_dim != expected_dims[i]) {
            std::cerr << "CheckOutput: dim" << i << " mismatch, expected=" << expected_dims[i]
                      << " actual=" << actual_dim << std::endl;
            return kFailed;
        }
    }
    return kSuccess;
}

int RunScenario(RunContext& ctx, const std::string& name, const std::vector<int64_t>& shape, bool is_dynamic)
{
    std::cout << "=== Scenario: " << name << " ===" << std::endl;

    // --- Run scenario: build graph, add to session, execute on device ---
    Graph graph(("ct_rg_" + name).c_str());
    std::vector<Operator> graph_inputs;
    Operator graph_output;
    if (BuildControlTriggerGraph(graph, shape, graph_inputs, graph_output, is_dynamic) != kSuccess) {
        std::cerr << name << " FAILED: BuildControlTriggerGraph for RunGraph" << std::endl;
        return kFailed;
    }
    graph.SetInputs(graph_inputs).SetOutputs({{graph_output, std::vector<size_t>{0}}});

    Session session(ctx.session_options);
    uint32_t gid = ctx.graph_id++;
    if (session.AddGraph(gid, graph) != SUCCESS) {
        std::cerr << name << " FAILED: AddGraph" << std::endl;
        return kFailed;
    }
    CheckInferShape(graph_output, shape);

    std::vector<int64_t> actual;
    for (auto d : shape) {
        actual.push_back(d < 0 ? 2 : d);
    }
    std::vector<int64_t> max_shape;
    for (auto d : shape) {
        max_shape.push_back(d < 0 ? 10 : d);
    }
    size_t max_cnt = 1;
    for (auto d : max_shape) {
        max_cnt *= static_cast<size_t>(d);
    }
    std::vector<float> idata(max_cnt, 1.0f);
    Tensor itensor(TensorDesc(Shape(actual), FORMAT_ND, kDtype), reinterpret_cast<const uint8_t*>(idata.data()),
                   max_cnt * sizeof(float));

    std::vector<Tensor> inputs = {itensor};
    std::vector<Tensor> outputs;

    bool run_ok = false;
    Status ret = session.RunGraph(gid, inputs, outputs);
    if (ret != SUCCESS) {
        std::cerr << name << ": RunGraph failed (ret=" << ret << ")" << std::endl;
    } else if (outputs.empty()) {
        std::cerr << name << ": RunGraph success but output.size()==0" << std::endl;
    } else {
        std::cout << name << ": RunGraph success" << std::endl;
        if (CheckOutput(outputs, actual) == kSuccess) {
            run_ok = true;
        }
    }

    // --- BuildModel scenario: compile independent graph to om model ---
    Graph build_graph(("ct_bm_" + name).c_str());
    std::vector<Operator> build_inputs;
    Operator build_output;
    if (BuildControlTriggerGraph(build_graph, shape, build_inputs, build_output, is_dynamic) != kSuccess) {
        std::cerr << name << " FAILED: BuildControlTriggerGraph for build" << std::endl;
        return kFailed;
    }
    build_graph.SetInputs(build_inputs).SetOutputs({{build_output, std::vector<size_t>{0}}});

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
    int ret = kSuccess;

    // Scenario 1: static shape {2,3}, run + buildmodel
    if (RunScenario(ctx, "static", {2, 3}, false) != kSuccess) {
        ret = kFailed;
    }
    // Scenario 2: dynamic shape {-1,3}, dim0 unknown (range 1~10), run + buildmodel
    if (RunScenario(ctx, "dynamic_minus1", {-1, 3}, true) != kSuccess) {
        ret = kFailed;
    }
    // Scenario 3: dynamic shape {-2}, rank unknown (range 1~10), run + buildmodel
    if (RunScenario(ctx, "dynamic_minus2", {-2}, true) != kSuccess) {
        ret = kFailed;
    }

    GEFinalize();
    if (ret == kSuccess) {
        std::cout << "All scenarios PASSED" << std::endl;
    } else {
        std::cout << "Some scenarios FAILED" << std::endl;
    }
    return ret;
}
