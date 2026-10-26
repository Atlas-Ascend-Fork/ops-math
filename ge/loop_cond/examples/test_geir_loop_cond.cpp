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
constexpr DataType kDtype = DT_BOOL;
const std::vector<int64_t> kScalarShape = {};

struct RunContext {
    std::map<AscendString, AscendString> session_options;
    uint32_t graph_id = 0;
};

int BuildGraph(Graph& graph, const std::vector<int64_t>& shape, std::vector<Operator>& graph_inputs,
               Operator& graph_output, bool is_dynamic)
{
    auto data = op::Data("input").set_attr_index(0);
    TensorDesc desc(Shape(shape), FORMAT_ND, kDtype);
    if (is_dynamic) {
        std::vector<std::pair<int64_t, int64_t>> range;
        for (auto d : shape) {
            range.emplace_back(d < 0 ? 1 : d, d < 0 ? 10 : d);
        }
        desc.SetShapeRange(range);
    }
    data.update_input_desc_x(desc);
    data.update_output_desc_y(desc);

    auto loop_cond = op::LoopCond("loop_cond");
    loop_cond.set_input_x(data);
    TensorDesc out_desc(Shape(shape), FORMAT_ND, kDtype);
    if (is_dynamic) {
        std::vector<std::pair<int64_t, int64_t>> range;
        for (auto d : shape) {
            range.emplace_back(d < 0 ? 1 : d, d < 0 ? 10 : d);
        }
        out_desc.SetShapeRange(range);
    }
    loop_cond.update_output_desc_y(out_desc);

    graph.AddOp(data);
    graph.AddOp(loop_cond);
    graph_inputs = {data};
    graph_output = loop_cond;
    return kSuccess;
}

int CheckInferShape(const Graph& graph, const std::vector<int64_t>& expected_dims)
{
    Operator op_obj;
    if (graph.FindOpByName("loop_cond", op_obj) != GRAPH_SUCCESS) {
        std::cerr << "CheckInferShape: FindOpByName failed" << std::endl;
        return kFailed;
    }
    TensorDesc desc = op_obj.GetOutputDescByName("y");
    Shape shape = desc.GetShape();
    // ge::Shape::GetDimNum() returns the number of VALID dims (0 for the unknown rank {-2}, see the official
    // doc in graph/tensor.h), so the literal dims from GetDims() are compared to stay consistent with
    // expected_dims (e.g. {-2} has 1 literal dim).
    const std::vector<int64_t> dims = shape.GetDims();
    std::cout << "InferShape: dim_num=" << dims.size();
    for (size_t i = 0; i < dims.size(); i++) {
        std::cout << " dim" << i << "=" << dims[i];
    }
    std::cout << " shape_size=" << shape.GetShapeSize() << " dtype=" << desc.GetDataType() << std::endl;
    if (dims.size() != expected_dims.size()) {
        std::cerr << "CheckInferShape: expected dim_num=" << expected_dims.size() << " actual=" << dims.size()
                  << std::endl;
        return kFailed;
    }
    for (size_t i = 0; i < expected_dims.size(); i++) {
        if (dims[i] != expected_dims[i]) {
            std::cerr << "CheckInferShape: dim" << i << " expected=" << expected_dims[i] << " actual=" << dims[i]
                      << std::endl;
            return kFailed;
        }
    }
    if (desc.GetDataType() != kDtype) {
        std::cerr << "CheckInferShape: expected dtype=" << kDtype << " actual=" << desc.GetDataType() << std::endl;
        return kFailed;
    }
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
    std::cout << "Output: dim_num=" << shape.GetDims().size();
    for (size_t i = 0; i < shape.GetDims().size(); i++) {
        std::cout << " dim" << i << "=" << shape.GetDims()[i];
    }
    std::cout << " shape_size=" << shape.GetShapeSize() << " dtype=" << desc.GetDataType() << std::endl;
    const std::vector<int64_t> dims = shape.GetDims();
    if (dims.size() != expected_dims.size() || desc.GetDataType() != kDtype) {
        std::cerr << "CheckOutput: expected dim_num=" << expected_dims.size() << " dtype=" << kDtype << std::endl;
        return kFailed;
    }
    for (size_t i = 0; i < expected_dims.size(); i++) {
        if (dims[i] != expected_dims[i]) {
            std::cerr << "CheckOutput: dim" << i << " expected=" << expected_dims[i] << " actual=" << dims[i]
                      << std::endl;
            return kFailed;
        }
    }
    return kSuccess;
}

int RunScenario(RunContext& ctx, const std::string& name, const std::vector<int64_t>& shape, bool is_dynamic)
{
    std::cout << "=== Scenario: " << name << " ===" << std::endl;

    // --- Run scenario: build graph, add to session, execute on device ---
    Graph graph(("loop_cond_" + name).c_str());
    std::vector<Operator> graph_inputs;
    Operator graph_output;
    if (BuildGraph(graph, shape, graph_inputs, graph_output, is_dynamic) != kSuccess) {
        std::cerr << name << " FAILED: BuildGraph" << std::endl;
        return kFailed;
    }
    graph.SetInputs(graph_inputs).SetOutputs({graph_output});

    Session session(ctx.session_options);
    uint32_t gid = ctx.graph_id++;
    if (session.AddGraph(gid, graph) != SUCCESS) {
        std::cerr << name << " FAILED: AddGraph" << std::endl;
        return kFailed;
    }
    if (CheckInferShape(graph, shape) != kSuccess) {
        std::cerr << name << " FAILED: CheckInferShape" << std::endl;
        return kFailed;
    }

    // LoopCond input is a boolean scalar, so the real tensor fed on RunGraph is always a scalar.
    std::vector<uint8_t> idata(1, 1);
    Tensor itensor(TensorDesc(Shape(kScalarShape), FORMAT_ND, kDtype), idata.data(), idata.size());

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
        if (CheckOutput(outputs, kScalarShape) == kSuccess) {
            run_ok = true;
        }
    }

    // --- BuildModel scenario: compile independent graph to om model ---
    Graph build_graph(("loop_cond_build_" + name).c_str());
    std::vector<Operator> build_inputs;
    Operator build_output;
    if (BuildGraph(build_graph, shape, build_inputs, build_output, is_dynamic) != kSuccess) {
        std::cerr << name << " FAILED: BuildGraph for build" << std::endl;
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

    std::string om_file = "loop_cond_" + name;
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

int RunNeg1Scenario(RunContext& ctx)
{
    std::cout << "=== Scenario: dynamic_minus1 (designed rejection) ===" << std::endl;
    // LoopCond only accepts a scalar (rank-0) or unknown-rank(-2) input. A {-1} input is rank-1 and must be
    // rejected by InferShape, so this scenario expects AddGraph/RunGraph to fail as designed.
    Graph graph("loop_cond_dynamic_minus1");
    std::vector<Operator> graph_inputs;
    Operator graph_output;
    if (BuildGraph(graph, {-1}, graph_inputs, graph_output, true) != kSuccess) {
        std::cerr << "dynamic_minus1 FAILED: BuildGraph" << std::endl;
        return kFailed;
    }
    graph.SetInputs(graph_inputs).SetOutputs({graph_output});

    Session session(ctx.session_options);
    uint32_t gid = ctx.graph_id++;
    if (session.AddGraph(gid, graph) != SUCCESS) {
        std::cout << "dynamic_minus1: AddGraph rejected non-scalar {-1} input as designed" << std::endl;
        std::cout << "dynamic_minus1 PASSED" << std::endl;
        return kSuccess;
    }

    std::vector<uint8_t> idata(1, 1);
    Tensor itensor(TensorDesc(Shape(kScalarShape), FORMAT_ND, kDtype), idata.data(), idata.size());
    std::vector<Tensor> inputs = {itensor};
    std::vector<Tensor> outputs;
    if (session.RunGraph(gid, inputs, outputs) != SUCCESS) {
        std::cout << "dynamic_minus1: RunGraph rejected non-scalar {-1} input as designed" << std::endl;
        std::cout << "dynamic_minus1 PASSED" << std::endl;
        return kSuccess;
    }
    std::cerr << "dynamic_minus1 FAILED: LoopCond unexpectedly accepted non-scalar {-1} input" << std::endl;
    return kFailed;
}

int RunMinus2Scenario(RunContext& ctx)
{
    std::cout << "=== Scenario: dynamic_minus2 (infershape only) ===" << std::endl;
    // LoopCond is a while-loop control-flow meta operator. A {-2} unknown-rank graph takes the RT2 online
    // execution path, where no lowering func is registered for LoopCond ("Cannot find lowering func for
    // node type:[LoopCond]"), so standalone RunGraph/BuildModel cannot succeed outside a real while-loop
    // structure. Following the NextIteration precedent, only AddGraph + InferShape are verified here;
    // real execution is covered by the static scenario.
    Graph graph("loop_cond_dynamic_minus2");
    std::vector<Operator> graph_inputs;
    Operator graph_output;
    if (BuildGraph(graph, {-2}, graph_inputs, graph_output, false) != kSuccess) {
        std::cerr << "dynamic_minus2 FAILED: BuildGraph" << std::endl;
        return kFailed;
    }
    graph.SetInputs(graph_inputs).SetOutputs({graph_output});

    Session session(ctx.session_options);
    uint32_t gid = ctx.graph_id++;
    if (session.AddGraph(gid, graph) != SUCCESS) {
        std::cerr << "dynamic_minus2 FAILED: AddGraph" << std::endl;
        return kFailed;
    }
    if (CheckInferShape(graph, {-2}) != kSuccess) {
        std::cerr << "dynamic_minus2 FAILED: CheckInferShape" << std::endl;
        return kFailed;
    }

    std::cout << "dynamic_minus2 PASSED" << std::endl;
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

    // Scenario 1: static scalar shape {}, run + buildmodel
    if (RunScenario(ctx, "static", kScalarShape, false) != kSuccess) {
        ret = kFailed;
    }
    // Scenario 2: dynamic shape {-1} (rank-1) is rejected by design, LoopCond only accepts scalar input
    if (RunNeg1Scenario(ctx) != kSuccess) {
        ret = kFailed;
    }
    // Scenario 3: dynamic shape {-2} (rank unknown). LoopCond has no RT2 lowering func outside a while
    // loop, so only AddGraph + InferShape are verified (see RunMinus2Scenario).
    if (RunMinus2Scenario(ctx) != kSuccess) {
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
