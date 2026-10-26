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
 * \file test_geir_switch.cpp
 * \brief Graph-mode example for the Switch operator (RunGraph + BuildModel).
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
#include "../op_graph/switch_proto.h"
#include "../../data/op_graph/data_proto.h"
#include "../../merge/op_graph/merge_proto.h"
#include "../../const/op_graph/const_proto.h"
#include "../../../math/add/op_graph/add_proto.h"

using namespace ge;

namespace {
constexpr int kFailed = -1;
constexpr int kSuccess = 0;
constexpr uint32_t kDeviceId = 0;
constexpr DataType kDtype = DT_FLOAT;
constexpr float kFalseBranchOffset = 10.0f;
constexpr float kTrueBranchOffset = 20.0f;

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

    TensorDesc pred_desc(Shape(std::vector<int64_t>{}), FORMAT_ND, DT_BOOL);
    pred_desc.SetPlacement(kPlacementHost);
    auto pred = op::Data("pred").set_attr_index(1);
    pred.update_input_desc_x(pred_desc);
    pred.update_output_desc_y(pred_desc);

    auto sw = op::Switch("switch_op");
    sw.set_input_data(data);
    sw.set_input_pred(pred);

    // Distinct arithmetic on each branch makes the selected route observable.
    TensorDesc offset_desc(Shape(std::vector<int64_t>{}), FORMAT_ND, kDtype);
    Tensor false_offset_tensor(offset_desc, reinterpret_cast<const uint8_t*>(&kFalseBranchOffset), sizeof(float));
    Tensor true_offset_tensor(offset_desc, reinterpret_cast<const uint8_t*>(&kTrueBranchOffset), sizeof(float));
    auto false_offset = op::Const("false_offset").set_attr_value(false_offset_tensor);
    auto true_offset = op::Const("true_offset").set_attr_value(true_offset_tensor);
    false_offset.update_output_desc_y(offset_desc);
    true_offset.update_output_desc_y(offset_desc);

    auto false_branch = op::Add("false_branch_add");
    false_branch.set_input_x1(sw, "output_false");
    false_branch.set_input_x2(false_offset);
    auto true_branch = op::Add("true_branch_add");
    true_branch.set_input_x1(sw, "output_true");
    true_branch.set_input_x2(true_offset);

    auto merge = op::Merge("switch_merge");
    merge.create_dynamic_input_x(2);
    merge.set_dynamic_input_x(0, false_branch);
    merge.set_dynamic_input_x(1, true_branch);

    graph.AddOp(data);
    graph.AddOp(pred);
    graph.AddOp(sw);
    graph.AddOp(false_offset);
    graph.AddOp(true_offset);
    graph.AddOp(false_branch);
    graph.AddOp(true_branch);
    graph.AddOp(merge);
    graph_inputs = {data, pred};
    graph_output = merge;
    return kSuccess;
}

int CheckInferShape(const Graph& graph, const std::vector<int64_t>& expected_dims)
{
    Operator op_obj;
    if (graph.FindOpByName("switch_op", op_obj) != GRAPH_SUCCESS) {
        std::cerr << "CheckInferShape: FindOpByName failed" << std::endl;
        return kFailed;
    }
    // output_false = index 0, output_true = index 1
    const char* outs[] = {"output_false", "output_true"};
    for (size_t o = 0; o < 2; o++) {
        TensorDesc desc = op_obj.GetOutputDesc(o);
        Shape shape = desc.GetShape();
        std::cout << "InferShape[" << outs[o] << "]: dim_num=" << shape.GetDimNum();
        for (size_t i = 0; i < shape.GetDims().size(); i++) {
            std::cout << " dim" << i << "=" << shape.GetDim(i);
        }
        std::cout << " shape_size=" << shape.GetShapeSize() << " dtype=" << desc.GetDataType() << std::endl;
        if (desc.GetDataType() != kDtype) {
            std::cerr << "InferShape[" << outs[o] << "]: dtype mismatch" << std::endl;
            return kFailed;
        }
        if (shape.GetDimNum() != expected_dims.size()) {
            std::cerr << "InferShape[" << outs[o] << "]: dim_num mismatch" << std::endl;
            return kFailed;
        }
        for (size_t i = 0; i < expected_dims.size(); i++) {
            if (shape.GetDim(i) != expected_dims[i]) {
                std::cerr << "InferShape[" << outs[o] << "]: dim" << i << " mismatch" << std::endl;
                return kFailed;
            }
        }
    }
    return kSuccess;
}

int CheckOutput(const std::vector<Tensor>& outputs, const std::vector<int64_t>& expected_dims,
                const std::vector<float>& expected_values, bool pred)
{
    if (outputs.size() != 1U) {
        std::cerr << "CheckOutput: expected 1 output, got " << outputs.size() << std::endl;
        return kFailed;
    }
    const auto data_desc = outputs[0].GetTensorDesc();
    if (data_desc.GetShape().GetDims() != expected_dims || data_desc.GetDataType() != kDtype ||
        data_desc.GetShape().GetShapeSize() != static_cast<int64_t>(expected_values.size())) {
        std::cerr << "CheckOutput: data shape or dtype mismatch" << std::endl;
        return kFailed;
    }
    const size_t expected_bytes = expected_values.size() * sizeof(float);
    if (outputs[0].GetSize() < expected_bytes || outputs[0].GetData() == nullptr) {
        std::cerr << "CheckOutput: data buffer is missing or too small" << std::endl;
        return kFailed;
    }
    const float branch_offset = pred ? kTrueBranchOffset : kFalseBranchOffset;
    for (size_t i = 0; i < expected_values.size(); ++i) {
        float value = 0.0f;
        std::memcpy(&value, outputs[0].GetData() + i * sizeof(float), sizeof(value));
        const float expected = expected_values[i] + branch_offset;
        if (value != expected) {
            std::cerr << "CheckOutput: element " << i << " expected=" << expected << " actual=" << value << std::endl;
            return kFailed;
        }
    }
    std::cout << "SwitchBranchCheck success: pred=" << (pred ? "true" : "false") << " offset=" << branch_offset
              << " elements=" << expected_values.size() << std::endl;
    return kSuccess;
}

int RunScenario(RunContext& ctx, const std::string& name, const std::vector<int64_t>& shape, bool is_dynamic, bool pred)
{
    std::cout << "=== Scenario: " << name << " ===" << std::endl;

    Graph graph(("switch_" + name).c_str());
    std::vector<Operator> graph_inputs;
    Operator graph_output;
    if (BuildGraph(graph, shape, graph_inputs, graph_output, is_dynamic) != kSuccess) {
        std::cerr << name << " FAILED: BuildGraph" << std::endl;
        return kFailed;
    }
    graph.SetInputs(graph_inputs);
    graph.SetOutputs(std::vector<std::pair<Operator, std::vector<size_t>>>{{graph_output, {0U}}});

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
    uint8_t pred_val = pred ? 1 : 0;
    Tensor pred_tensor(TensorDesc(Shape(std::vector<int64_t>{}), FORMAT_ND, DT_BOOL), &pred_val, sizeof(pred_val));
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
        run_ok = CheckOutput(outputs, actual, dvals, pred) == kSuccess;
    }

    Graph build_graph(("switch_build_" + name).c_str());
    std::vector<Operator> build_inputs;
    Operator build_output;
    if (BuildGraph(build_graph, shape, build_inputs, build_output, is_dynamic) != kSuccess) {
        std::cerr << name << " FAILED: BuildGraph for build" << std::endl;
        return kFailed;
    }
    build_graph.SetInputs(build_inputs);
    build_graph.SetOutputs(std::vector<std::pair<Operator, std::vector<size_t>>>{{build_output, {0U}}});

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
    // Specialize execution to the concrete input shapes for legacy control flow.
    ctx.session_options["ge.jit_compile"] = "1";
    int ret = kSuccess;
    for (bool pred : {false, true}) {
        const std::string suffix = pred ? "_true" : "_false";
        if (RunScenario(ctx, "static" + suffix, {2, 3}, false, pred) != kSuccess) {
            ret = kFailed;
        }
        if (RunScenario(ctx, "dynamic_minus1" + suffix, {-1, 3}, true, pred) != kSuccess) {
            ret = kFailed;
        }
        if (RunScenario(ctx, "dynamic_minus2" + suffix, {-2}, true, pred) != kSuccess) {
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
