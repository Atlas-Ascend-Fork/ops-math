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
// Data is a graph-engine built-in node and is not defined in ops-math,
// so it is registered locally for graph construction.
REG_OP(Data).INPUT(x, TensorType::ALL()).OUTPUT(y, TensorType::ALL()).ATTR(index, Int, 0).OP_END_FACTORY_REG(Data)
} // namespace ge

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

// TransShape changes the output shape to the "outShape" attribute without
// changing the underlying data. GE replaces the TransShape node with a
// MemcpyAsync node during graph preparation (ReplaceTransShapePass), so the
// output data equals the input data. Input {2,3,4} (24 elements) with
// outShape={6,4} produces output {6,4}.
//
// Note: only a static-shape scenario is covered. The output shape is always
// the outShape attribute (a compile-time ListInt), regardless of the input
// shape, so dynamic inputs (-1/-2) exercise no extra InferShape logic. The
// reshape also requires input element count == output element count, which
// cannot be verified at compile time for unknown-shape inputs.
int BuildTransShapeGraph(Graph& graph, const std::vector<int64_t>& shape, std::vector<Operator>& graph_inputs,
                         Operator& graph_output)
{
    auto input_data = op::Data("input").set_attr_index(0);
    TensorDesc idesc(Shape(shape), FORMAT_ND, kDtype);
    input_data.update_input_desc_x(idesc);
    input_data.update_output_desc_y(idesc);

    auto trans_shape = op::TransShape("trans_shape");
    trans_shape.set_input_x(input_data);
    trans_shape.set_attr_outShape(std::vector<int64_t>{6, 4});
    trans_shape.update_output_desc_y(TensorDesc(Shape({6, 4}), FORMAT_ND, kDtype));

    graph.AddOp(input_data);
    graph.AddOp(trans_shape);
    graph_inputs = {input_data};
    graph_output = trans_shape;
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

    // TransShape is replaced by MemcpyAsync, so the output data equals the input data.
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

int RunScenario(RunContext& ctx, const std::string& name, const std::vector<int64_t>& shape)
{
    std::cout << "=== Scenario: " << name << " ===" << std::endl;

    // --- Run scenario: build graph, add to session, execute on device ---
    Graph graph(("trans_shape_rg_" + name).c_str());
    std::vector<Operator> graph_inputs;
    Operator graph_output;
    if (BuildTransShapeGraph(graph, shape, graph_inputs, graph_output) != kSuccess) {
        std::cerr << name << " FAILED: BuildTransShapeGraph for RunGraph" << std::endl;
        return kFailed;
    }
    graph.SetInputs(graph_inputs).SetOutputs({{graph_output, std::vector<size_t>{0}}});

    Session session(ctx.session_options);
    uint32_t gid = ctx.graph_id++;
    if (session.AddGraph(gid, graph) != SUCCESS) {
        std::cerr << name << " FAILED: AddGraph" << std::endl;
        return kFailed;
    }
    // output shape always equals outShape={6,4}
    if (CheckInferShape(graph_output, {6, 4}) != kSuccess) {
        return kFailed;
    }

    // actual input: {2,3,4} = 24 elements, matches outShape {6,4} = 24 elements
    std::vector<int64_t> actual = {2, 3, 4};
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
        if (CheckOutput(outputs, {6, 4}, idata) == kSuccess) {
            run_ok = true;
        }
    }

    // --- BuildModel scenario: compile independent graph to om model ---
    Graph build_graph(("trans_shape_bm_" + name).c_str());
    std::vector<Operator> build_inputs;
    Operator build_output;
    if (BuildTransShapeGraph(build_graph, shape, build_inputs, build_output) != kSuccess) {
        std::cerr << name << " FAILED: BuildTransShapeGraph for build" << std::endl;
        return kFailed;
    }
    build_graph.SetInputs(build_inputs).SetOutputs({{build_output, std::vector<size_t>{0}}});

    ModelBufferData model;
    std::map<AscendString, AscendString> build_opts;
    if (aclgrphBuildModel(build_graph, build_opts, model) != SUCCESS) {
        std::cerr << name << ": BuildModel failed" << std::endl;
    } else {
        std::cout << name << ": BuildModel success, size=" << model.length << std::endl;
        std::string om_file = name;
        if (aclgrphSaveModel(om_file.c_str(), model) != SUCCESS) {
            std::cerr << name << " FAILED: SaveModel" << std::endl;
        } else {
            std::cout << name << ": SaveModel success, file=" << om_file << std::endl;
        }
    }

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

    // Scenario: static shape {2,3,4}, outShape={6,4}, run + buildmodel.
    // Dynamic-shape scenarios are intentionally not covered: the output shape
    // is always the outShape attribute (compile-time ListInt) regardless of the
    // input shape, and the reshape requires input element count == output
    // element count, which cannot be guaranteed for unknown-shape inputs.
    if (RunScenario(ctx, "static", {2, 3, 4}) != kSuccess) {
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
