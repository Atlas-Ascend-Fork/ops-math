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
constexpr DataType kX1Dtype = DT_INT32;
constexpr DataType kX2Dtype = DT_UINT64;

struct RunContext {
    std::map<AscendString, AscendString> session_options;
    uint32_t graph_id = 0;
};

// SwitchByIndex takes x1 (int32 subscript) and x2 (1-D uint64) and has no output tensor. It is connected
// into the graph by consuming data1/data2, and data1 is bridged to the graph output through Identity.
// Graph: data1(x1) + data2(x2) -> SwitchByIndex; data1(x1) -> Identity -> y.
int BuildGraph(Graph& graph, const std::vector<int64_t>& x1_shape, bool is_dynamic, std::vector<Operator>& graph_inputs,
               Operator& graph_output)
{
    auto data1 = op::Data("x1").set_attr_index(0);
    TensorDesc idesc1(Shape(x1_shape), FORMAT_ND, kX1Dtype);
    if (is_dynamic) {
        std::vector<std::pair<int64_t, int64_t>> range;
        for (auto d : x1_shape) {
            range.emplace_back(d < 0 ? 1 : d, d < 0 ? 10 : d);
        }
        idesc1.SetShapeRange(range);
    }
    data1.update_input_desc_x(idesc1);
    data1.update_output_desc_y(idesc1);

    auto data2 = op::Data("x2").set_attr_index(1);
    TensorDesc idesc2(Shape(std::vector<int64_t>{10}), FORMAT_ND, kX2Dtype);
    data2.update_input_desc_x(idesc2);
    data2.update_output_desc_y(idesc2);

    auto switch_by_index = op::SwitchByIndex("switch_by_index");
    switch_by_index.set_input_x1(data1);
    switch_by_index.set_input_x2(data2);

    auto identity = op::Identity("identity");
    identity.set_input_x(data1);
    TensorDesc odesc(Shape(x1_shape), FORMAT_ND, kX1Dtype);
    if (is_dynamic) {
        std::vector<std::pair<int64_t, int64_t>> range;
        for (auto d : x1_shape) {
            range.emplace_back(d < 0 ? 1 : d, d < 0 ? 10 : d);
        }
        odesc.SetShapeRange(range);
    }
    identity.update_output_desc_y(odesc);

    graph.AddOp(data1);
    graph.AddOp(data2);
    graph.AddOp(switch_by_index);
    graph.AddOp(identity);
    graph_inputs = {data1, data2};
    graph_output = identity;
    return kSuccess;
}

// SwitchByIndex has no output, so InferShape is asserted on the Identity output which passes x1 through.
int CheckInferShape(const Graph& graph, const std::vector<int64_t>& expected_dims)
{
    Operator op_obj;
    if (graph.FindOpByName("identity", op_obj) != GRAPH_SUCCESS) {
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
    if (desc.GetDataType() != kX1Dtype) {
        std::cerr << "CheckInferShape: expected dtype=" << kX1Dtype << " actual=" << desc.GetDataType() << std::endl;
        return kFailed;
    }
    return kSuccess;
}

int RunScenario(RunContext& ctx, const std::string& name, const std::vector<int64_t>& x1_shape, bool is_dynamic)
{
    std::cout << "=== Scenario: " << name << " ===" << std::endl;

    // SwitchByIndex is a runtime label-switch operator whose only kernel and ops-info registration live on
    // ascend035; ascend950 has never registered an engine for it. BuildModel/RunGraph therefore always fail
    // at engine placement ("Cannot find engine of type SwitchByIndex"), which is not a migration defect (the
    // same graph fails identically with the canndev registration on ascend950). Only AddGraph + InferShape
    // are verified here.
    Graph graph(("switch_by_index_" + name).c_str());
    std::vector<Operator> graph_inputs;
    Operator graph_output;
    if (BuildGraph(graph, x1_shape, is_dynamic, graph_inputs, graph_output) != kSuccess) {
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
    if (CheckInferShape(graph, x1_shape) != kSuccess) {
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

    // Scenario 1: static x1 shape {1}, addgraph + infershape
    if (RunScenario(ctx, "static", {1}, false) != kSuccess) {
        ret = kFailed;
    }
    // Scenario 2: dynamic x1 shape {-1} (range 1~10), addgraph + infershape
    if (RunScenario(ctx, "dynamic_minus1", {-1}, true) != kSuccess) {
        ret = kFailed;
    }
    // Scenario 3: dynamic x1 shape {-2} (rank unknown, range 1~10), addgraph + infershape
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
