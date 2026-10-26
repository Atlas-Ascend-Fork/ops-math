/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Note: the DT_INT64 output of the "dtype" attribute path is derived by the
// InferDataType registered in the ops-math ophost deliverable. Run this
// example in an environment where that deliverable is loaded (e.g. after
// package installation); with only the legacy registration GE falls back to
// a DT_INT32 output and the dtype check below does not pass.

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
constexpr DataType kOutDtype = DT_INT64;

struct RunContext {
    std::map<AscendString, AscendString> session_options;
    uint32_t graph_id = 0;
};

// Scenario configuration: input desc shape (with -1/-2 for dynamic dims),
// whether SetShapeRange is required, and the dims of the tensor actually fed
// at RunGraph time. Size must report the element count of the fed tensor.
struct ScenarioConfig {
    std::vector<int64_t> shape;
    bool is_dynamic = false;
    std::vector<int64_t> actual;
};

// Size takes one tensor input x and outputs the number of elements of x as a
// scalar. x is fed by a Data node, so the graph is Data(x) -> Size -> y.
int BuildSizeGraph(Graph& graph, const ScenarioConfig& cfg, std::vector<Operator>& graph_inputs, Operator& graph_output)
{
    auto input_data = op::Data("input").set_attr_index(0);
    TensorDesc idesc(Shape(cfg.shape), FORMAT_ND, kDtype);
    if (cfg.is_dynamic) {
        std::vector<std::pair<int64_t, int64_t>> range;
        for (auto d : cfg.shape) {
            range.emplace_back(d < 0 ? 1 : d, d < 0 ? 10 : d);
        }
        idesc.SetShapeRange(range);
    }
    input_data.update_input_desc_x(idesc);
    input_data.update_output_desc_y(idesc);

    auto size_op = op::Size("size");
    size_op.set_input_x(input_data);
    size_op.set_attr_dtype(static_cast<int64_t>(kOutDtype));
    TensorDesc odesc(Shape(), FORMAT_ND, kOutDtype);
    size_op.update_output_desc_y(odesc);

    graph.AddOp(input_data);
    graph.AddOp(size_op);
    graph_inputs = {input_data};
    graph_output = size_op;
    return kSuccess;
}

// Size output is always a scalar, so expected dim_num is 0 for every scenario.
int CheckInferShape(const Operator& op)
{
    TensorDesc desc = op.GetOutputDescByName("y");
    Shape shape = desc.GetShape();
    std::cout << "InferShape: dim_num=" << shape.GetDimNum() << " shape_size=" << shape.GetShapeSize()
              << " dtype=" << desc.GetDataType() << std::endl;
    if (shape.GetDimNum() != 0) {
        std::cerr << "CheckInferShape: output is not a scalar, dim_num=" << shape.GetDimNum() << std::endl;
        return kFailed;
    }
    if (desc.GetDataType() != kOutDtype) {
        std::cerr << "CheckInferShape: dtype mismatch, expected=" << kOutDtype << " actual=" << desc.GetDataType()
                  << std::endl;
        return kFailed;
    }
    return kSuccess;
}

int CheckOutput(const std::vector<Tensor>& outputs, int64_t expected_value)
{
    if (outputs.empty()) {
        std::cerr << "CheckOutput: output is empty (output.size()==0)" << std::endl;
        return kFailed;
    }
    const auto& tensor = outputs[0];
    const auto& desc = tensor.GetTensorDesc();
    Shape shape = desc.GetShape();
    std::cout << "Output: dim_num=" << shape.GetDimNum() << " shape_size=" << shape.GetShapeSize()
              << " dtype=" << desc.GetDataType() << std::endl;
    if (shape.GetDimNum() != 0) {
        std::cerr << "CheckOutput: output is not a scalar, dim_num=" << shape.GetDimNum() << std::endl;
        return kFailed;
    }
    if (desc.GetDataType() != kOutDtype) {
        std::cerr << "CheckOutput: dtype mismatch, expected=" << kOutDtype << " actual=" << desc.GetDataType()
                  << std::endl;
        return kFailed;
    }
    const uint8_t* data = tensor.GetData();
    if (data == nullptr) {
        std::cerr << "CheckOutput: output data is null" << std::endl;
        return kFailed;
    }
    const int64_t value = *reinterpret_cast<const int64_t*>(data);
    std::cout << "Output: value=" << value << " expected=" << expected_value << std::endl;
    if (value != expected_value) {
        std::cerr << "CheckOutput: value mismatch, expected=" << expected_value << " actual=" << value << std::endl;
        return kFailed;
    }
    return kSuccess;
}

int RunScenario(RunContext& ctx, const std::string& name, const ScenarioConfig& cfg)
{
    std::cout << "=== Scenario: " << name << " ===" << std::endl;

    // --- Run scenario: build graph, add to session, execute on device ---
    Graph graph(("size_rg_" + name).c_str());
    std::vector<Operator> graph_inputs;
    Operator graph_output;
    if (BuildSizeGraph(graph, cfg, graph_inputs, graph_output) != kSuccess) {
        std::cerr << name << " FAILED: BuildSizeGraph for RunGraph" << std::endl;
        return kFailed;
    }
    graph.SetInputs(graph_inputs).SetOutputs({graph_output});

    Session session(ctx.session_options);
    uint32_t gid = ctx.graph_id++;
    if (session.AddGraph(gid, graph) != SUCCESS) {
        std::cerr << name << " FAILED: AddGraph" << std::endl;
        return kFailed;
    }
    if (CheckInferShape(graph_output) != kSuccess) {
        std::cerr << name << " FAILED: CheckInferShape" << std::endl;
        return kFailed;
    }

    size_t cnt = 1;
    for (auto d : cfg.actual) {
        cnt *= static_cast<size_t>(d);
    }
    std::vector<float> idata(cnt, 1.0f);
    Tensor itensor(TensorDesc(Shape(cfg.actual), FORMAT_ND, kDtype), reinterpret_cast<const uint8_t*>(idata.data()),
                   cnt * sizeof(float));

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
        if (CheckOutput(outputs, static_cast<int64_t>(cnt)) == kSuccess) {
            run_ok = true;
        }
    }

    // --- BuildModel scenario: compile independent graph to om model ---
    Graph build_graph(("size_bm_" + name).c_str());
    std::vector<Operator> build_inputs;
    Operator build_output;
    if (BuildSizeGraph(build_graph, cfg, build_inputs, build_output) != kSuccess) {
        std::cerr << name << " FAILED: BuildSizeGraph for build" << std::endl;
        return kFailed;
    }
    build_graph.SetInputs(build_inputs).SetOutputs({build_output});

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

    // Scenario 1: static shape {2,3}, Size must report 2*3=6.
    ScenarioConfig static_cfg = {{2, 3}, false, {2, 3}};
    if (RunScenario(ctx, "static", static_cfg) != kSuccess) {
        ret = kFailed;
    }
    // Scenario 2: dynamic shape {-1,3} (dim0 range 1~10), actual fed dims
    // {2,3} -> Size must report 6.
    ScenarioConfig dyn1_cfg = {{-1, 3}, true, {2, 3}};
    if (RunScenario(ctx, "dynamic_minus1", dyn1_cfg) != kSuccess) {
        ret = kFailed;
    }
    // Scenario 3: dynamic shape {-2} (rank unknown, range 1~10), actual fed
    // dims {2,3} -> Size must report 6.
    ScenarioConfig dyn2_cfg = {{-2}, true, {2, 3}};
    if (RunScenario(ctx, "dynamic_minus2", dyn2_cfg) != kSuccess) {
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
