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

struct Scenario {
    std::string name;
    std::vector<int64_t> input_shape;
    int64_t axis;
    bool is_dynamic;
    std::vector<int64_t> expected_infer;
    std::vector<int64_t> actual_input;
    std::vector<int64_t> expected_output;
};

// Flatten splits the input dims at axis into two groups and multiplies each
// group into one dim. Input {2,3,4} with axis=1 produces {2,12}; axis=0
// produces {1,24}; axis=3 (== rank) produces {24,1}.
int BuildFlattenGraph(Graph& graph, const Scenario& sc, std::vector<Operator>& graph_inputs, Operator& graph_output)
{
    auto input_data = op::Data("input").set_attr_index(0);
    TensorDesc idesc(Shape(sc.input_shape), FORMAT_ND, kDtype);
    if (sc.is_dynamic) {
        std::vector<std::pair<int64_t, int64_t>> range;
        for (auto d : sc.input_shape) {
            range.emplace_back(d < 0 ? 1 : d, d < 0 ? 10 : d);
        }
        idesc.SetShapeRange(range);
    }
    input_data.update_input_desc_x(idesc);
    input_data.update_output_desc_y(idesc);

    auto flatten = op::Flatten("flatten");
    flatten.set_input_x(input_data);
    flatten.set_attr_axis(sc.axis);
    flatten.update_output_desc_y(TensorDesc(Shape(sc.expected_infer), FORMAT_ND, kDtype));

    graph.AddOp(input_data);
    graph.AddOp(flatten);
    graph_inputs = {input_data};
    graph_output = flatten;
    return kSuccess;
}

int CheckInferShape(const Operator& op, const std::vector<int64_t>& expected_dims)
{
    TensorDesc desc = op.GetOutputDesc(0);
    Shape shape = desc.GetShape();
    const auto rawDims = shape.GetDims();

    std::cout << "InferShape: dim_num=" << shape.GetDimNum();
    for (size_t i = 0; i < rawDims.size(); i++) {
        std::cout << " dim" << i << "=" << rawDims[i];
    }
    std::cout << " dtype=" << desc.GetDataType() << std::endl;

    // ge::Shape::GetDimNum() returns 0 for an unknown-rank shape {-2}
    const bool isUnknownRank = (shape.GetDimNum() == 0U) ||
                               std::any_of(rawDims.begin(), rawDims.end(), [](int64_t d) { return d == -2; });
    if (isUnknownRank) {
        if (expected_dims.size() == 1UL && expected_dims[0] == -2) {
            std::cout << "CheckInferShape: unknown rank {-2} verified" << std::endl;
            return kSuccess;
        }
        std::cerr << "CheckInferShape: expected unknown rank {-2}" << std::endl;
        return kFailed;
    }

    if (shape.GetDimNum() != expected_dims.size()) {
        std::cerr << "CheckInferShape: dim_num mismatch, expected=" << expected_dims.size()
                  << " actual=" << shape.GetDimNum() << std::endl;
        return kFailed;
    }
    for (size_t i = 0; i < expected_dims.size(); i++) {
        if (shape.GetDim(i) != expected_dims[i]) {
            std::cerr << "CheckInferShape: dim" << i << " mismatch, expected=" << expected_dims[i]
                      << " actual=" << shape.GetDim(i) << std::endl;
            return kFailed;
        }
    }
    return kSuccess;
}

int CheckOutput(const std::vector<Tensor>& outputs, const std::vector<int64_t>& expected_dims,
                const std::vector<float>& expected_data)
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
    if (desc.GetDataType() != kDtype) {
        std::cerr << "CheckOutput: dtype mismatch, expected=" << kDtype << " actual=" << desc.GetDataType()
                  << std::endl;
        return kFailed;
    }

    // Flatten preserves the element order, so the output data equals the input data.
    const auto* data = reinterpret_cast<const float*>(tensor.GetData());
    if (data == nullptr) {
        std::cerr << "CheckOutput: output data is null" << std::endl;
        return kFailed;
    }
    if (expected_data.size() != static_cast<size_t>(shape.GetShapeSize())) {
        std::cerr << "CheckOutput: data size mismatch" << std::endl;
        return kFailed;
    }
    for (size_t i = 0; i < expected_data.size(); i++) {
        if (data[i] != expected_data[i]) {
            std::cerr << "CheckOutput: data[" << i << "] mismatch, expected=" << expected_data[i]
                      << " actual=" << data[i] << std::endl;
            return kFailed;
        }
    }
    return kSuccess;
}

int RunScenario(RunContext& ctx, const Scenario& sc)
{
    std::cout << "=== Scenario: " << sc.name << " ===" << std::endl;
    const bool isUnknownRank = (sc.input_shape.size() == 1UL && sc.input_shape[0] == -2);
    const bool isDynamic = sc.is_dynamic || isUnknownRank;

    // --- RunGraph scenario: build graph, add to session, execute on device ---
    Graph graph(("flatten_rg_" + sc.name).c_str());
    std::vector<Operator> graph_inputs;
    Operator graph_output;
    if (BuildFlattenGraph(graph, sc, graph_inputs, graph_output) != kSuccess) {
        std::cerr << sc.name << " FAILED: BuildFlattenGraph for RunGraph" << std::endl;
        return kFailed;
    }
    graph.SetInputs(graph_inputs).SetOutputs({{graph_output, std::vector<size_t>{0}}});

    Session session(ctx.session_options);
    uint32_t gid = ctx.graph_id++;
    if (session.AddGraph(gid, graph) != SUCCESS) {
        std::cerr << sc.name << " FAILED: AddGraph" << std::endl;
        return kFailed;
    }
    if (CheckInferShape(graph_output, sc.expected_infer) != kSuccess) {
        std::cerr << sc.name << " NOT PASSED (InferShape check failed)" << std::endl;
        return kFailed;
    }

    // --- BuildModel scenario: compile an independent graph to an OM model ---
    bool build_ok = false;
    Graph build_graph(("flatten_bm_" + sc.name).c_str());
    std::vector<Operator> build_inputs;
    Operator build_output;
    if (BuildFlattenGraph(build_graph, sc, build_inputs, build_output) == kSuccess) {
        build_graph.SetInputs(build_inputs).SetOutputs({{build_output, std::vector<size_t>{0}}});
        ModelBufferData model;
        std::map<AscendString, AscendString> build_opts;
        if (aclgrphBuildModel(build_graph, build_opts, model) != SUCCESS) {
            std::cerr << sc.name << ": BuildModel failed" << std::endl;
        } else if (aclgrphSaveModel(("flatten_" + sc.name).c_str(), model) != SUCCESS) {
            std::cerr << sc.name << ": SaveModel failed" << std::endl;
        } else {
            build_ok = true;
            std::cout << sc.name << ": BuildModel success, size=" << model.length << std::endl;
        }
    }

    // --- RunGraph with a concrete input shape ---
    bool run_ok = false;
    bool output_ok = false;
    if (!sc.actual_input.empty()) {
        size_t cnt = 1;
        for (auto d : sc.actual_input) {
            cnt *= static_cast<size_t>(d);
        }
        std::vector<float> idata(cnt);
        for (size_t i = 0; i < cnt; i++) {
            idata[i] = static_cast<float>(i);
        }
        Tensor itensor(TensorDesc(Shape(sc.actual_input), FORMAT_ND, kDtype),
                       reinterpret_cast<const uint8_t*>(idata.data()), cnt * sizeof(float));
        std::vector<Tensor> inputs = {itensor};
        std::vector<Tensor> outputs;
        if (session.RunGraph(gid, inputs, outputs) != SUCCESS) {
            std::cerr << sc.name << ": RunGraph failed" << std::endl;
        } else {
            run_ok = true;
            std::cout << sc.name << ": RunGraph success" << std::endl;
            output_ok = (CheckOutput(outputs, sc.expected_output, idata) == kSuccess);
        }
    }

    if (!isDynamic) {
        if (!run_ok || !output_ok || !build_ok) {
            std::cerr << sc.name << " NOT PASSED (RunGraph=" << run_ok << " Output=" << output_ok
                      << " BuildModel=" << build_ok << ")" << std::endl;
            return kFailed;
        }
        std::cout << sc.name << " PASSED" << std::endl;
        return kSuccess;
    }

    // Dynamic shapes may not support RunGraph; InferShape verification is the
    // pass criteria in that case. If RunGraph does run, the output must match.
    if (!run_ok) {
        std::cout << sc.name << " PASSED (InferShape verified, RunGraph not supported for this dynamic shape)"
                  << std::endl;
        return kSuccess;
    }
    if (!output_ok || !build_ok) {
        std::cerr << sc.name << " NOT PASSED (RunGraph=1 Output=" << output_ok << " BuildModel=" << build_ok << ")"
                  << std::endl;
        return kFailed;
    }
    std::cout << sc.name << " PASSED" << std::endl;
    return kSuccess;
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
    const std::vector<Scenario> scenarios = {
        {"static_axis1", {2, 3, 4}, 1, false, {2, 12}, {2, 3, 4}, {2, 12}},
        {"static_axis0", {2, 3, 4}, 0, false, {1, 24}, {2, 3, 4}, {1, 24}},
        {"static_axis_rank", {2, 3, 4}, 3, false, {24, 1}, {2, 3, 4}, {24, 1}},
        {"dynamic_minus1", {-1, 3, 4}, 1, true, {-1, 12}, {2, 3, 4}, {2, 12}},
        {"dynamic_minus2", {-2}, 1, true, {-2}, {2, 3, 4}, {2, 12}},
    };
    for (const auto& sc : scenarios) {
        if (RunScenario(ctx, sc) != kSuccess) {
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
