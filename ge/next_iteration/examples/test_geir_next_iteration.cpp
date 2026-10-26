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

    auto next_iteration = op::NextIteration("next_iteration");
    next_iteration.set_input_x(data);
    TensorDesc out_desc(Shape(shape), FORMAT_ND, kDtype);
    if (is_dynamic) {
        std::vector<std::pair<int64_t, int64_t>> range;
        for (auto d : shape) {
            range.emplace_back(d < 0 ? 1 : d, d < 0 ? 10 : d);
        }
        out_desc.SetShapeRange(range);
    }
    next_iteration.update_output_desc_y(out_desc);

    graph.AddOp(data);
    graph.AddOp(next_iteration);
    graph_inputs = {data};
    graph_output = next_iteration;
    return kSuccess;
}

int CheckInferShape(const Graph& graph, const std::vector<int64_t>& expected_dims)
{
    Operator op_obj;
    if (graph.FindOpByName("next_iteration", op_obj) != GRAPH_SUCCESS) {
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

int RunScenario(RunContext& ctx, const std::string& name, const std::vector<int64_t>& shape, bool is_dynamic)
{
    std::cout << "=== Scenario: " << name << " ===" << std::endl;

    // NextIteration is a while-loop control-flow operator. A standalone NextIteration node without the
    // Enter/Merge/Switch/LoopCond loop structure makes GE control-flow processing insert ge_global_step
    // and fail in engine partitioning (SplitNodeInputs) at RunGraph/BuildModel time. Following the
    // QueueData precedent, only AddGraph + InferShape are verified here.
    Graph graph(("next_iteration_" + name).c_str());
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

    std::cout << name << " PASSED" << std::endl;
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

    // Scenario 1: static shape {2,3}, addgraph + infershape
    if (RunScenario(ctx, "static", {2, 3}, false) != kSuccess) {
        ret = kFailed;
    }
    // Scenario 2: dynamic shape {-1,3}, dim0 unknown (range 1~10), addgraph + infershape
    if (RunScenario(ctx, "dynamic_minus1", {-1, 3}, true) != kSuccess) {
        ret = kFailed;
    }
    // Scenario 3: dynamic shape {-2}, rank unknown (range 1~10), addgraph + infershape
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
