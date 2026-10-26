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

// body subgraph: Data(0)=counter (int32 scalar), Data(1)=input (passthrough).
// The For pass wraps the original body inside a PartitionedCall node with 2 inputs
// (counter + dynamic input). Data(0) maps to counter, Data(1) maps to the dynamic
// input via default parent-input mapping. With start=0, limit=1, delta=1 the loop
// runs one iteration, so the output equals the input passed through the body.
Graph MakeBodyGraph(const std::string& name, const std::vector<int64_t>& shape)
{
    Graph sub(name.c_str());
    auto counter = op::Data((name + "_counter").c_str()).set_attr_index(0);
    TensorDesc counter_desc(Shape(), FORMAT_ND, DT_INT32);
    counter.update_input_desc_x(counter_desc);
    counter.update_output_desc_y(counter_desc);

    auto data = op::Data((name + "_data").c_str()).set_attr_index(1);
    TensorDesc desc(Shape(shape), FORMAT_ND, kDtype);
    data.update_input_desc_x(desc);
    data.update_output_desc_y(desc);

    sub.AddOp(counter);
    sub.AddOp(data);
    sub.SetInputs({counter, data}).SetOutputs({data});
    return sub;
}

int BuildForGraph(Graph& graph, const std::vector<int64_t>& shape, std::vector<Operator>& graph_inputs,
                  Operator& graph_output, bool is_dynamic)
{
    auto start_data = op::Data("start").set_attr_index(0);
    TensorDesc scalar_desc(Shape(), FORMAT_ND, DT_INT32);
    start_data.update_input_desc_x(scalar_desc);
    start_data.update_output_desc_y(scalar_desc);

    auto limit_data = op::Data("limit").set_attr_index(1);
    limit_data.update_input_desc_x(scalar_desc);
    limit_data.update_output_desc_y(scalar_desc);

    auto delta_data = op::Data("delta").set_attr_index(2);
    delta_data.update_input_desc_x(scalar_desc);
    delta_data.update_output_desc_y(scalar_desc);

    auto input_data = op::Data("input").set_attr_index(3);
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

    auto for_op = op::For("for");
    for_op.DynamicInputRegister("input", 1);
    for_op.DynamicOutputRegister("output", 1);
    for_op.set_input_start(start_data);
    for_op.set_input_limit(limit_data);
    for_op.set_input_delta(delta_data);
    for_op.set_dynamic_input_input(0, input_data);
    for_op.update_dynamic_output_desc_output(0, TensorDesc(Shape(shape), FORMAT_ND, kDtype));

    for_op.set_subgraph_builder_body([shape]() -> Graph { return MakeBodyGraph("body", shape); });

    graph.AddOp(start_data);
    graph.AddOp(limit_data);
    graph.AddOp(delta_data);
    graph.AddOp(input_data);
    graph.AddOp(for_op);
    graph_inputs = {start_data, limit_data, delta_data, input_data};
    graph_output = for_op;
    return kSuccess;
}

int CheckInferShape(const Operator& op, const std::vector<int64_t>& expected_dims)
{
    TensorDesc desc = op.GetDynamicOutputDesc("output", 0);
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
        if (shape.GetDim(i) != expected_dims[i]) {
            std::cerr << "CheckOutput: dim" << i << " mismatch, expected=" << expected_dims[i]
                      << " actual=" << shape.GetDim(i) << std::endl;
            return kFailed;
        }
    }
    return kSuccess;
}

int RunScenario(RunContext& ctx, const std::string& name, const std::vector<int64_t>& shape, bool is_dynamic)
{
    std::cout << "=== Scenario: " << name << " ===" << std::endl;

    // --- Run scenario: build graph, add to session, execute on device ---
    Graph graph(("for_rg_" + name).c_str());
    std::vector<Operator> graph_inputs;
    Operator graph_output;
    if (BuildForGraph(graph, shape, graph_inputs, graph_output, is_dynamic) != kSuccess) {
        std::cerr << name << " FAILED: BuildForGraph for RunGraph" << std::endl;
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

    // start=0, limit=1, delta=1 → loop runs one iteration, output = input passthrough
    int32_t start_val = 0;
    int32_t limit_val = 1;
    int32_t delta_val = 1;
    Tensor start_tensor(TensorDesc(Shape(), FORMAT_ND, DT_INT32), reinterpret_cast<const uint8_t*>(&start_val),
                        sizeof(int32_t));
    Tensor limit_tensor(TensorDesc(Shape(), FORMAT_ND, DT_INT32), reinterpret_cast<const uint8_t*>(&limit_val),
                        sizeof(int32_t));
    Tensor delta_tensor(TensorDesc(Shape(), FORMAT_ND, DT_INT32), reinterpret_cast<const uint8_t*>(&delta_val),
                        sizeof(int32_t));

    std::vector<int64_t> actual;
    for (auto d : shape) {
        actual.push_back(d < 0 ? 2 : d);
    }
    size_t cnt = 1;
    for (auto d : actual) {
        cnt *= static_cast<size_t>(d);
    }
    std::vector<float> idata(cnt, 1.0f);
    Tensor itensor(TensorDesc(Shape(actual), FORMAT_ND, kDtype), reinterpret_cast<const uint8_t*>(idata.data()),
                   cnt * sizeof(float));

    std::vector<Tensor> inputs = {start_tensor, limit_tensor, delta_tensor, itensor};
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
    Graph build_graph(("for_bm_" + name).c_str());
    std::vector<Operator> build_inputs;
    Operator build_output;
    if (BuildForGraph(build_graph, shape, build_inputs, build_output, is_dynamic) != kSuccess) {
        std::cerr << name << " FAILED: BuildForGraph for build" << std::endl;
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
