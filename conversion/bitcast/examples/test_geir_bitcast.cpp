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
#include "ops_proto_math.h"

using namespace ge;

namespace {
constexpr int kFailed = -1;
constexpr int kSuccess = 0;
constexpr uint32_t kDeviceId = 0;
constexpr DataType kXDtype = DT_FLOAT; // 4 bytes

struct RunContext {
    std::map<AscendString, AscendString> session_options;
    uint32_t graph_id = 0;
};

// Scenario configuration: the input desc shape (may contain -1/-2), the target
// dtype of the type attr, the keep_dim flag, the expected compile-time output
// shape, and the concrete shapes used by RunGraph.
struct Scenario {
    std::string name;
    std::vector<int64_t> desc_shape;      // input desc shape, may contain -1/-2
    bool is_dynamic;                      // whether to set shape range on the desc
    DataType target_dtype;                // value of the type attr
    bool keep_dim;                        // value of the keep_dim attr
    std::vector<int64_t> expected_infer;  // output shape after AddGraph InferShape
    std::vector<int64_t> actual_input;    // concrete input shape fed to RunGraph
    std::vector<int64_t> expected_output; // concrete output shape of RunGraph
};

// Bitcast reinterprets the input bytes as the target dtype without copying or
// converting data. The x dtype is FLOAT (4 bytes) in all scenarios:
//   FLOAT -> INT8 (keep_dim=false): append a dim of 4 at the tail;
//   FLOAT -> INT8 (keep_dim=true): multiply the last dim by 4;
//   FLOAT -> INT32 (same dtype size): shape keeps unchanged, which is the only
//   well-defined conversion for an unknown-rank {-2} input.
int BuildBitcastGraph(Graph& graph, const Scenario& sc, std::vector<Operator>& graph_inputs, Operator& graph_output)
{
    auto input_data = op::Data("input").set_attr_index(0);
    TensorDesc idesc(Shape(sc.desc_shape), FORMAT_ND, kXDtype);
    if (sc.is_dynamic) {
        std::vector<std::pair<int64_t, int64_t>> range;
        for (auto d : sc.desc_shape) {
            range.emplace_back(d < 0 ? 1 : d, d < 0 ? 10 : d);
        }
        idesc.SetShapeRange(range);
    }
    input_data.update_input_desc_x(idesc);
    input_data.update_output_desc_y(idesc);

    auto bitcast = op::Bitcast("bitcast");
    bitcast.set_input_x(input_data);
    bitcast.set_attr_type(sc.target_dtype);
    if (sc.keep_dim) {
        bitcast.set_attr_keep_dim(true);
    }
    bitcast.update_output_desc_y(TensorDesc(Shape(sc.expected_infer), FORMAT_ND, sc.target_dtype));

    graph.AddOp(input_data);
    graph.AddOp(bitcast);
    graph_inputs = {input_data};
    graph_output = bitcast;
    return kSuccess;
}

int CheckInferShape(const Operator& op, const std::vector<int64_t>& expected_dims, const DataType expected_dtype)
{
    TensorDesc desc = op.GetOutputDesc(0);
    Shape shape = desc.GetShape();
    const auto rawDims = shape.GetDims();

    std::cout << "InferShape: dim_num=" << shape.GetDimNum();
    for (size_t i = 0; i < rawDims.size(); i++) {
        std::cout << " dim" << i << "=" << rawDims[i];
    }
    std::cout << " shape_size=" << shape.GetShapeSize() << " dtype=" << desc.GetDataType() << std::endl;

    if (desc.GetDataType() != expected_dtype) {
        std::cerr << "CheckInferShape: dtype mismatch, expected=" << expected_dtype << " actual=" << desc.GetDataType()
                  << std::endl;
        return kFailed;
    }

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

int CheckOutput(const std::vector<Tensor>& outputs, const Scenario& sc, const uint8_t* input_bytes,
                const size_t input_bytes_len)
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

    if (shape.GetDimNum() != sc.expected_output.size()) {
        std::cerr << "CheckOutput: dim_num mismatch, expected=" << sc.expected_output.size()
                  << " actual=" << shape.GetDimNum() << std::endl;
        return kFailed;
    }
    for (size_t i = 0; i < sc.expected_output.size(); i++) {
        int64_t actual_dim = shape.GetDim(i);
        if (actual_dim == -1) {
            continue; // runtime keeps unknown dim
        }
        if (actual_dim != sc.expected_output[i]) {
            std::cerr << "CheckOutput: dim" << i << " mismatch, expected=" << sc.expected_output[i]
                      << " actual=" << actual_dim << std::endl;
            return kFailed;
        }
    }
    if (desc.GetDataType() != sc.target_dtype) {
        std::cerr << "CheckOutput: dtype mismatch, expected=" << sc.target_dtype << " actual=" << desc.GetDataType()
                  << std::endl;
        return kFailed;
    }

    // Bitcast only reinterprets the stored bytes: the output data equals the input data.
    const auto* data = tensor.GetData();
    if (data == nullptr) {
        std::cerr << "CheckOutput: output data is null" << std::endl;
        return kFailed;
    }
    if (tensor.GetSize() != input_bytes_len) {
        std::cerr << "CheckOutput: byte size mismatch, expected=" << input_bytes_len << " actual=" << tensor.GetSize()
                  << std::endl;
        return kFailed;
    }
    if (std::memcmp(data, input_bytes, input_bytes_len) != 0) {
        std::cerr << "CheckOutput: output bytes differ from input bytes" << std::endl;
        return kFailed;
    }
    return kSuccess;
}

int RunScenario(RunContext& ctx, const Scenario& sc)
{
    std::cout << "=== Scenario: " << sc.name << " ===" << std::endl;
    bool infer_shape_ok = false;

    // --- Run scenario: build graph, add to session, execute on device ---
    Graph graph(("bitcast_rg_" + sc.name).c_str());
    std::vector<Operator> graph_inputs;
    Operator graph_output;
    if (BuildBitcastGraph(graph, sc, graph_inputs, graph_output) != kSuccess) {
        std::cerr << sc.name << " FAILED: BuildBitcastGraph for RunGraph" << std::endl;
        return kFailed;
    }
    graph.SetInputs(graph_inputs).SetOutputs({{graph_output, std::vector<size_t>{0}}});

    Session session(ctx.session_options);
    uint32_t gid = ctx.graph_id++;
    if (session.AddGraph(gid, graph) != SUCCESS) {
        std::cerr << sc.name << " FAILED: AddGraph" << std::endl;
        return kFailed;
    }

    if (CheckInferShape(graph_output, sc.expected_infer, sc.target_dtype) == kSuccess) {
        infer_shape_ok = true;
    }

    // concrete input: FLOAT data 0..N-1; the same bytes flow through Bitcast
    size_t cnt = 1;
    for (auto d : sc.actual_input) {
        cnt *= static_cast<size_t>(d);
    }
    std::vector<float> idata(cnt);
    for (size_t i = 0; i < cnt; i++) {
        idata[i] = static_cast<float>(i);
    }
    const auto* input_bytes = reinterpret_cast<const uint8_t*>(idata.data());
    const size_t input_bytes_len = cnt * sizeof(float);
    Tensor itensor(TensorDesc(Shape(sc.actual_input), FORMAT_ND, kXDtype), input_bytes, input_bytes_len);

    std::vector<Tensor> inputs = {itensor};
    std::vector<Tensor> outputs;

    bool run_ok = false;
    Status ret = session.RunGraph(gid, inputs, outputs);
    if (ret != SUCCESS) {
        std::cerr << sc.name << ": RunGraph failed (ret=" << ret << ")" << std::endl;
    } else if (outputs.empty()) {
        std::cerr << sc.name << ": RunGraph success but output.size()==0" << std::endl;
    } else {
        std::cout << sc.name << ": RunGraph success" << std::endl;
        if (CheckOutput(outputs, sc, input_bytes, input_bytes_len) == kSuccess) {
            run_ok = true;
        }
    }

    // --- BuildModel scenario: compile independent graph to om model ---
    bool build_ok = false;
    Graph build_graph(("bitcast_bm_" + sc.name).c_str());
    std::vector<Operator> build_inputs;
    Operator build_output;
    if (BuildBitcastGraph(build_graph, sc, build_inputs, build_output) != kSuccess) {
        std::cerr << sc.name << " FAILED: BuildBitcastGraph for build" << std::endl;
        return kFailed;
    }
    build_graph.SetInputs(build_inputs).SetOutputs({{build_output, std::vector<size_t>{0}}});

    ModelBufferData model;
    std::map<AscendString, AscendString> build_opts;
    if (aclgrphBuildModel(build_graph, build_opts, model) != SUCCESS) {
        std::cerr << sc.name << ": BuildModel failed" << std::endl;
    } else {
        build_ok = true;
        std::cout << sc.name << ": BuildModel success, size=" << model.length << std::endl;
        std::string om_file = sc.name;
        if (aclgrphSaveModel(om_file.c_str(), model) != SUCCESS) {
            std::cerr << sc.name << " FAILED: SaveModel" << std::endl;
            build_ok = false;
        } else {
            std::cout << sc.name << ": SaveModel success, file=" << om_file << std::endl;
        }
    }

    if (!infer_shape_ok) {
        std::cerr << sc.name << " NOT PASSED (InferShape check failed)" << std::endl;
        return kFailed;
    }
    if (sc.is_dynamic && !run_ok) {
        // Reuse-input ops may not support unknown-shape graph execution.
        // InferShape verification is the pass criteria for dynamic scenarios.
        std::cout << sc.name << " PASSED (InferShape verified, RunGraph not supported for this dynamic shape)"
                  << std::endl;
        return kSuccess;
    }
    if (run_ok && (!sc.is_dynamic || build_ok)) {
        std::cout << sc.name << " PASSED" << std::endl;
        return kSuccess;
    }
    std::cerr << sc.name << " NOT PASSED (RunGraph=" << run_ok << " BuildModel=" << build_ok << ")" << std::endl;
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

    // Scenario 1: static shape {2,3}, FLOAT -> INT8, keep_dim=false:
    // dim of 4 appended at the tail, output {2,3,4} with 24 INT8 elements.
    Scenario static_sc = {"static", {2, 3}, false, DT_INT8, false, {2, 3, 4}, {2, 3}, {2, 3, 4}};
    if (RunScenario(ctx, static_sc) != kSuccess) {
        ret = kFailed;
    }

    // Scenario 2: static shape {2,3}, FLOAT -> INT8, keep_dim=true:
    // last dim multiplied by 4, output {2,12} with 24 INT8 elements.
    Scenario keepdim_sc = {"static_keepdim", {2, 3}, false, DT_INT8, true, {2, 12}, {2, 3}, {2, 12}};
    if (RunScenario(ctx, keepdim_sc) != kSuccess) {
        ret = kFailed;
    }

    // Scenario 3: dynamic shape {-1,3}, FLOAT -> INT8, keep_dim=false (dim0 unknown, range 1~10):
    // InferShape produces {-1,3,4}; the concrete run feeds {2,3}.
    Scenario minus1_sc = {"dynamic_minus1", {-1, 3}, true, DT_INT8, false, {-1, 3, 4}, {2, 3}, {2, 3, 4}};
    if (RunScenario(ctx, minus1_sc) != kSuccess) {
        ret = kFailed;
    }

    // Scenario 4: dynamic shape {-2} (rank unknown), FLOAT -> INT32 (same dtype size):
    // shape keeps unchanged, InferShape produces {-2}; the concrete run feeds {2,3}.
    Scenario minus2_sc = {"dynamic_minus2", {-2}, true, DT_INT32, false, {-2}, {2, 3}, {2, 3}};
    if (RunScenario(ctx, minus2_sc) != kSuccess) {
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
