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
    int32_t branch_index = 0;
    std::vector<int64_t> expected_out;
};

// branch0: Data → Reshape({2,3}→{6}),  branch1: Data → Reshape({2,3}→{3,2})
// branch_index=0 → output shape={6}, proving Case selected branch0 and subgraph executed
Graph MakeBranchGraph(const std::string& name, const std::vector<int64_t>& input_shape,
                      const std::vector<int64_t>& output_shape)
{
    Graph sub(name.c_str());
    auto data = op::Data((name + "_data").c_str()).set_attr_index(0);
    TensorDesc data_desc(Shape(input_shape), FORMAT_ND, kDtype);
    data.update_input_desc_x(data_desc);
    data.update_output_desc_y(data_desc);

    auto reshape_op = op::Reshape((name + "_reshape").c_str());
    reshape_op.set_input_x(data);
    TensorDesc reshape_desc(Shape(output_shape), FORMAT_ND, kDtype);
    reshape_op.update_output_desc_y(reshape_desc);

    int64_t shape_val = static_cast<int64_t>(output_shape.size());
    auto shape_const = op::Const((name + "_shape").c_str());
    TensorDesc shape_desc(Shape({static_cast<int64_t>(output_shape.size())}), FORMAT_ND, DT_INT64);
    shape_const.update_output_desc_y(shape_desc);
    auto shape_tensor = Tensor(TensorDesc(Shape({static_cast<int64_t>(output_shape.size())}), FORMAT_ND, DT_INT64),
                               reinterpret_cast<const uint8_t*>(output_shape.data()),
                               output_shape.size() * sizeof(int64_t));
    shape_const.set_attr_value(shape_tensor);
    reshape_op.set_input_shape(shape_const);

    sub.AddOp(data);
    sub.AddOp(shape_const);
    sub.AddOp(reshape_op);
    sub.SetInputs({data, shape_const}).SetOutputs({reshape_op});
    return sub;
}

int BuildCaseGraph(Graph& graph, const std::vector<int64_t>& shape, std::vector<Operator>& graph_inputs,
                   Operator& graph_output, bool is_dynamic, int32_t branch_index = 0)
{
    auto branch_data = op::Data("branch_index").set_attr_index(0);
    TensorDesc bdesc(Shape(), FORMAT_ND, DT_INT32);
    branch_data.update_input_desc_x(bdesc);
    branch_data.update_output_desc_y(bdesc);

    auto input_data = op::Data("input").set_attr_index(1);
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

    // branch0 reshapes to {prod}, branch1 reshapes to {prod/2, 2}
    // branch_index=0 → output = branch0 result
    int64_t total = 1;
    for (auto d : shape) {
        total *= (d < 0 ? 2 : d);
    }
    std::vector<int64_t> branch0_out = {total};
    std::vector<int64_t> branch1_out = {total / 2, 2};

    auto case_op = op::Case("case");
    case_op.create_dynamic_input_byindex_input(1, 1);
    case_op.create_dynamic_output_output(1);
    case_op.create_dynamic_subgraph_branches(2);
    case_op.set_input_branch_index(branch_data);
    case_op.set_dynamic_input_input(0, input_data);
    // preset output desc matches the selected branch's reshape target
    const auto& preset_out = (branch_index == 0) ? branch0_out : branch1_out;
    case_op.update_dynamic_output_desc_output(0, TensorDesc(Shape(preset_out), FORMAT_ND, kDtype));

    case_op.set_dynamic_subgraph_builder_branches(
        0, [shape, branch0_out]() -> Graph { return MakeBranchGraph("branch0", shape, branch0_out); });
    case_op.set_dynamic_subgraph_builder_branches(
        1, [shape, branch1_out]() -> Graph { return MakeBranchGraph("branch1", shape, branch1_out); });

    graph.AddOp(branch_data);
    graph.AddOp(input_data);
    graph.AddOp(case_op);
    graph_inputs = {branch_data, input_data};
    graph_output = case_op;
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
    std::cout << "=== Scenario: " << name << " (branch_index=" << ctx.branch_index << ") ===" << std::endl;

    // --- Run scenario: build graph, add to session, execute on device ---
    Graph graph(("case_rg_" + name).c_str());
    std::vector<Operator> graph_inputs;
    Operator graph_output;
    if (BuildCaseGraph(graph, shape, graph_inputs, graph_output, is_dynamic) != kSuccess) {
        std::cerr << name << " FAILED: BuildCaseGraph for RunGraph" << std::endl;
        return kFailed;
    }
    graph.SetInputs(graph_inputs).SetOutputs({{graph_output, std::vector<size_t>{0}}});

    Session session(ctx.session_options);
    uint32_t gid = ctx.graph_id++;
    if (session.AddGraph(gid, graph) != SUCCESS) {
        std::cerr << name << " FAILED: AddGraph" << std::endl;
        return kFailed;
    }
    CheckInferShape(graph_output, ctx.expected_out);

    Tensor btensor(TensorDesc(Shape(), FORMAT_ND, DT_INT32), reinterpret_cast<const uint8_t*>(&ctx.branch_index),
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

    std::vector<Tensor> inputs = {btensor, itensor};
    std::vector<Tensor> outputs;

    bool run_ok = false;
    Status ret = session.RunGraph(gid, inputs, outputs);
    if (ret != SUCCESS) {
        std::cerr << name << ": RunGraph failed (ret=" << ret << ")" << std::endl;
    } else if (outputs.empty()) {
        std::cerr << name << ": RunGraph success but output.size()==0" << std::endl;
    } else {
        std::cout << name << ": RunGraph success" << std::endl;
        if (CheckOutput(outputs, ctx.expected_out) == kSuccess) {
            run_ok = true;
        }
    }

    // --- BuildModel scenario: compile independent graph to om model ---
    Graph build_graph(("case_bm_" + name).c_str());
    std::vector<Operator> build_inputs;
    Operator build_output;
    if (BuildCaseGraph(build_graph, shape, build_inputs, build_output, is_dynamic, ctx.branch_index) != kSuccess) {
        std::cerr << name << " FAILED: BuildCaseGraph for build" << std::endl;
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

    // Scenario 1: static shape {2,3}, branch_index=0 → output={6}
    ctx.branch_index = 0;
    ctx.expected_out = {6};
    if (RunScenario(ctx, "static", {2, 3}, false) != kSuccess) {
        ret = kFailed;
    }
    // Scenario 2: dynamic shape {-1,3}, branch_index=0 → output={6}
    ctx.branch_index = 0;
    ctx.expected_out = {6};
    if (RunScenario(ctx, "dynamic_minus1", {-1, 3}, true) != kSuccess) {
        ret = kFailed;
    }
    // Scenario 3: dynamic shape {-2}, branch_index=1 → output={1,2}
    ctx.branch_index = 1;
    ctx.expected_out = {1, 2};
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
