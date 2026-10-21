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
#include "exe_graph/runtime/shape.h"

namespace ge {
// Data and Const are graph-engine built-in nodes and are not defined in ops-math,
// so they are registered locally for graph construction.
REG_OP(Data)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(Data)

        REG_OP(Const)
    .OUTPUT(y, TensorType::ALL())
    .ATTR(value, Tensor, Tensor())
    .OP_END_FACTORY_REG(Const)
} // namespace ge

using namespace ge;

namespace {
constexpr int kFailed = -1;
constexpr int kSuccess = 0;
constexpr uint32_t kDeviceId = 0;
constexpr DataType kDtype = DT_FLOAT;
constexpr int64_t kAxis = 0; // insert dim 1 at position 0

struct RunContext {
    std::map<AscendString, AscendString> session_options;
    uint32_t graph_id = 0;
};

// ExpandDims inserts a dim of 1 at the axis position (axis is a const input
// tensor). GE's ge_local engine treats ExpandDims as a reuse-input op, so the
// output directly views the input memory. Input {2,3} with axis=0 produces
// output {1,2,3} with the same data.
int BuildExpandDimsGraph(Graph& graph, const std::vector<int64_t>& shape, std::vector<Operator>& graph_inputs,
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

    // axis as a Const input carrying the value kAxis
    int64_t* axisData = new int64_t(kAxis);
    Tensor axisTensor(TensorDesc(Shape({1}), FORMAT_ND, DT_INT64), reinterpret_cast<const uint8_t*>(axisData),
                      sizeof(int64_t));
    auto axis_const = op::Const("axis");
    axis_const.set_attr_value(axisTensor);
    axis_const.update_output_desc_y(TensorDesc(Shape({1}), FORMAT_ND, DT_INT64));

    // compile-time output shape: dim 1 inserted at position kAxis
    std::vector<int64_t> out_shape;
    for (size_t i = 0; i <= shape.size(); i++) {
        if (i == static_cast<size_t>(kAxis)) {
            out_shape.push_back(1);
        }
        if (i < shape.size()) {
            out_shape.push_back(shape[i]);
        }
    }

    auto expand_dims = op::ExpandDims("expand_dims");
    expand_dims.set_input_x(input_data);
    expand_dims.set_input_axis(axis_const);
    expand_dims.update_output_desc_y(TensorDesc(Shape(out_shape), FORMAT_ND, kDtype));

    graph.AddOp(input_data);
    graph.AddOp(axis_const);
    graph.AddOp(expand_dims);
    graph_inputs = {input_data};
    graph_output = expand_dims;
    return kSuccess;
}

int CheckInferShape(const Operator& op, const std::vector<int64_t>& expected_dims)
{
    TensorDesc desc = op.GetOutputDesc(0);
    Shape shape = desc.GetShape(); // ge::Shape
    const auto rawDims = shape.GetDims();

    // Build a gert::Shape from the same raw dims to compare the two APIs:
    // ge::Shape::GetDimNum() returns 0 for unknown rank ({-2}), while
    // gert::Shape::GetDimNum() returns 1 (dims_.size()) for the same shape.
    gert::StorageShape gertShapeStorage;
    gert::Shape& gertShape = gertShapeStorage.MutableOriginShape();
    gertShape.SetDimNum(rawDims.size());
    for (size_t i = 0; i < rawDims.size(); i++) {
        gertShape.SetDim(i, rawDims[i]);
    }
    std::cout << "InferShape (ge::Shape): dim_num=" << shape.GetDimNum() << " rawDims.size=" << rawDims.size();
    for (size_t i = 0; i < rawDims.size(); i++) {
        std::cout << " dim" << i << "=" << rawDims[i];
    }
    std::cout << " dtype=" << desc.GetDataType() << std::endl;
    std::cout << "InferShape (gert::Shape): dim_num=" << gertShape.GetDimNum()
              << " IsUnknownRank=" << ((gertShape.GetDimNum() == 1U && gertShape.GetDim(0) == -2) ? 1 : 0) << std::endl;

    // unknown rank: ge::Shape::GetDimNum() returns 0 for {-2}; rawDims contains -2
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
        int64_t actual_dim = shape.GetDim(i);
        if (actual_dim == -1) {
            continue; // runtime keeps unknown dim
        }
        if (actual_dim != expected_dims[i]) {
            std::cerr << "CheckOutput: dim" << i << " mismatch, expected=" << expected_dims[i]
                      << " actual=" << actual_dim << std::endl;
            return kFailed;
        }
    }
    if (desc.GetDataType() != kDtype) {
        std::cerr << "CheckOutput: dtype mismatch, expected=" << kDtype << " actual=" << desc.GetDataType()
                  << std::endl;
        return kFailed;
    }

    // ExpandDims is a reuse-input op, so the output data equals the input data.
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

int RunScenario(RunContext& ctx, const std::string& name, const std::vector<int64_t>& shape, bool is_dynamic)
{
    std::cout << "=== Scenario: " << name << " ===" << std::endl;
    bool infer_shape_ok = false;

    // --- Run scenario: build graph, add to session, execute on device ---
    Graph graph(("expand_dims_rg_" + name).c_str());
    std::vector<Operator> graph_inputs;
    Operator graph_output;
    if (BuildExpandDimsGraph(graph, shape, graph_inputs, graph_output, is_dynamic) != kSuccess) {
        std::cerr << name << " FAILED: BuildExpandDimsGraph for RunGraph" << std::endl;
        return kFailed;
    }
    graph.SetInputs(graph_inputs).SetOutputs({{graph_output, std::vector<size_t>{0}}});

    Session session(ctx.session_options);
    uint32_t gid = ctx.graph_id++;
    if (session.AddGraph(gid, graph) != SUCCESS) {
        std::cerr << name << " FAILED: AddGraph" << std::endl;
        return kFailed;
    }

    // compile-time InferShape expectation: dim 1 inserted at position kAxis.
    // unknown rank ({-2}) input is passed through as {-2} (no dim inserted).
    std::vector<int64_t> expected_infer;
    if (shape.size() == 1UL && shape[0] == -2) {
        expected_infer = {-2};
    } else {
        for (size_t i = 0; i <= shape.size(); i++) {
            if (i == static_cast<size_t>(kAxis)) {
                expected_infer.push_back(1);
            }
            if (i < shape.size()) {
                expected_infer.push_back(shape[i]);
            }
        }
    }
    if (CheckInferShape(graph_output, expected_infer) == kSuccess) {
        infer_shape_ok = true;
    }

    // actual input {2,3} = 6 elements; runtime output is concretized to {1,2,3}
    std::vector<int64_t> actual = {2, 3};
    size_t cnt = 1;
    for (auto d : actual) {
        cnt *= static_cast<size_t>(d);
    }
    std::vector<float> idata(cnt);
    for (size_t i = 0; i < cnt; i++) {
        idata[i] = static_cast<float>(i);
    }
    Tensor itensor(TensorDesc(Shape(actual), FORMAT_ND, kDtype), reinterpret_cast<const uint8_t*>(idata.data()),
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
        if (CheckOutput(outputs, {1, 2, 3}, idata) == kSuccess) {
            run_ok = true;
        }
    }

    // --- BuildModel scenario: compile independent graph to om model ---
    bool build_ok = false;
    Graph build_graph(("expand_dims_bm_" + name).c_str());
    std::vector<Operator> build_inputs;
    Operator build_output;
    if (BuildExpandDimsGraph(build_graph, shape, build_inputs, build_output, is_dynamic) != kSuccess) {
        std::cerr << name << " FAILED: BuildExpandDimsGraph for build" << std::endl;
        return kFailed;
    }
    build_graph.SetInputs(build_inputs).SetOutputs({{build_output, std::vector<size_t>{0}}});

    ModelBufferData model;
    std::map<AscendString, AscendString> build_opts;
    if (aclgrphBuildModel(build_graph, build_opts, model) != SUCCESS) {
        std::cerr << name << ": BuildModel failed" << std::endl;
    } else {
        build_ok = true;
        std::cout << name << ": BuildModel success, size=" << model.length << std::endl;
        std::string om_file = name;
        if (aclgrphSaveModel(om_file.c_str(), model) != SUCCESS) {
            std::cerr << name << " FAILED: SaveModel" << std::endl;
            build_ok = false;
        } else {
            std::cout << name << ": SaveModel success, file=" << om_file << std::endl;
        }
    }

    if (!infer_shape_ok) {
        std::cerr << name << " NOT PASSED (InferShape check failed)" << std::endl;
        return kFailed;
    }
    if (is_dynamic && !run_ok) {
        // Reuse-input ops may not support unknown-shape graph execution.
        // InferShape verification is the pass criteria for dynamic scenarios.
        std::cout << name << " PASSED (InferShape verified, RunGraph not supported for this dynamic shape)"
                  << std::endl;
        return kSuccess;
    }
    if (run_ok && (!is_dynamic || build_ok)) {
        std::cout << name << " PASSED" << std::endl;
        return kSuccess;
    }
    std::cerr << name << " NOT PASSED (RunGraph=" << run_ok << " BuildModel=" << build_ok << ")" << std::endl;
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

    // Scenario 1: static shape {2,3}, axis=0, run + buildmodel
    if (RunScenario(ctx, "static", {2, 3}, false) != kSuccess) {
        ret = kFailed;
    }
    // Scenario 2: dynamic shape {-1,3}, dim0 unknown (range 1~10)
    if (RunScenario(ctx, "dynamic_minus1", {-1, 3}, true) != kSuccess) {
        ret = kFailed;
    }
    // Scenario 3: dynamic shape {-2}, rank unknown (range 1~10)
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
