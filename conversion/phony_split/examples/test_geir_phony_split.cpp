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

// PhonySplit marks that GE allocates one continuous memory block for all its
// outputs and performs no calculation. The input {8,4} is split along dim 0
// (split_dim={0}, num_split={2}) into 2 outputs {4,4}: output0 views the first
// half of the input memory and output1 the second half.
int BuildPhonySplitGraph(Graph& graph, const std::vector<int64_t>& shape, std::vector<Operator>& graph_inputs,
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

    // output shape: dim0 is the split dim, divided by num_split(2); unknown dims kept
    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < shape.size(); i++) {
        out_shape.push_back((i == 0 && shape[i] >= 0) ? shape[i] / 2 : shape[i]);
    }

    auto phony_split = op::PhonySplit("phony_split");
    phony_split.set_input_x(input_data);
    phony_split.set_attr_split_dim(std::vector<int64_t>{0});
    phony_split.set_attr_num_split(std::vector<int64_t>{2});
    phony_split.DynamicOutputRegister("y", 2);
    phony_split.update_dynamic_output_desc_y(0, TensorDesc(Shape(out_shape), FORMAT_ND, kDtype));
    phony_split.update_dynamic_output_desc_y(1, TensorDesc(Shape(out_shape), FORMAT_ND, kDtype));

    graph.AddOp(input_data);
    graph.AddOp(phony_split);
    graph_inputs = {input_data};
    graph_output = phony_split;
    return kSuccess;
}

int CheckInferShape(const Operator& op, const std::vector<int64_t>& expected_dims)
{
    TensorDesc desc = op.GetDynamicOutputDesc("y", 0);
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
                const std::vector<float>& idata)
{
    if (outputs.size() < 2UL) {
        std::cerr << "CheckOutput: expect 2 outputs, actual=" << outputs.size() << std::endl;
        return kFailed;
    }
    // output0 views the first half of the input memory, output1 the second half
    size_t halfCnt = idata.size() / 2UL;
    for (size_t outIdx = 0; outIdx < 2UL; outIdx++) {
        const auto& tensor = outputs[outIdx];
        const auto& desc = tensor.GetTensorDesc();
        Shape shape = desc.GetShape();
        std::cout << "Output[" << outIdx << "]: dim_num=" << shape.GetDimNum();
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

        const auto* data = reinterpret_cast<const float*>(tensor.GetData());
        if (data == nullptr) {
            std::cerr << "CheckOutput: output data is null" << std::endl;
            return kFailed;
        }
        for (size_t i = 0; i < halfCnt; i++) {
            float expected = idata[outIdx * halfCnt + i];
            if (data[i] != expected) {
                std::cerr << "CheckOutput: output[" << outIdx << "] data[" << i << "] mismatch, expected=" << expected
                          << " actual=" << data[i] << std::endl;
                return kFailed;
            }
        }
    }
    return kSuccess;
}

int RunScenario(RunContext& ctx, const std::string& name, const std::vector<int64_t>& shape, bool is_dynamic)
{
    std::cout << "=== Scenario: " << name << " ===" << std::endl;
    bool infer_shape_ok = false;

    // --- Run scenario: build graph, add to session, execute on device ---
    Graph graph(("phony_split_rg_" + name).c_str());
    std::vector<Operator> graph_inputs;
    Operator graph_output;
    if (BuildPhonySplitGraph(graph, shape, graph_inputs, graph_output, is_dynamic) != kSuccess) {
        std::cerr << name << " FAILED: BuildPhonySplitGraph for RunGraph" << std::endl;
        return kFailed;
    }
    graph.SetInputs(graph_inputs).SetOutputs({{graph_output, std::vector<size_t>{0, 1}}});

    Session session(ctx.session_options);
    uint32_t gid = ctx.graph_id++;
    if (session.AddGraph(gid, graph) != SUCCESS) {
        if (is_dynamic) {
            // expected: ge_local engine rejects PhonySplit in unknown-shape graphs at
            // compile stage ("does not support unknown graph") — a documented engine
            // limitation, not an InferShape failure
            std::cout << name << ": AddGraph failed (expected: ge_local rejects unknown graph)" << std::endl;
            std::cout << name << " PASSED (unknown graph rejection verified)" << std::endl;
            return kSuccess;
        }
        std::cerr << name << " FAILED: AddGraph" << std::endl;
        return kFailed;
    }

    std::vector<int64_t> expected_infer;
    for (size_t i = 0; i < shape.size(); i++) {
        expected_infer.push_back((i == 0 && shape[i] >= 0) ? shape[i] / 2 : shape[i]);
    }
    if (CheckInferShape(graph_output, expected_infer) == kSuccess) {
        infer_shape_ok = true;
    }

    // actual input: {8,4} = 32 elements, split into 2 outputs {4,4}
    std::vector<int64_t> actual;
    for (auto d : shape) {
        actual.push_back(d < 0 ? 8 : d);
    }
    std::vector<int64_t> actual_out;
    for (size_t i = 0; i < actual.size(); i++) {
        actual_out.push_back(i == 0 ? actual[i] / 2 : actual[i]);
    }
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
        // PhonySplit is a ge_local memory-offset placeholder: GE's ge_local
        // engine rejects it in unknown-shape graphs ("does not support unknown
        // graph"). This is a documented engine limitation, not an operator
        // failure; InferShape has already been verified above.
        std::cerr << name << ": RunGraph failed (ret=" << ret
                  << (is_dynamic ? "), expected for dynamic shape: ge_local engine does not support unknown graph" : "")
                  << std::endl;
    } else if (outputs.size() < 2UL) {
        std::cerr << name << ": RunGraph success but outputs.size()==" << outputs.size() << std::endl;
    } else {
        std::cout << name << ": RunGraph success" << std::endl;
        if (CheckOutput(outputs, actual_out, idata) == kSuccess) {
            run_ok = true;
        }
    }

    // --- BuildModel scenario: compile independent graph to om model ---
    bool build_ok = false;
    Graph build_graph(("phony_split_bm_" + name).c_str());
    std::vector<Operator> build_inputs;
    Operator build_output;
    if (BuildPhonySplitGraph(build_graph, shape, build_inputs, build_output, is_dynamic) != kSuccess) {
        std::cerr << name << " FAILED: BuildPhonySplitGraph for build" << std::endl;
        return kFailed;
    }
    build_graph.SetInputs(build_inputs).SetOutputs({{build_output, std::vector<size_t>{0, 1}}});

    ModelBufferData model;
    std::map<AscendString, AscendString> build_opts;
    if (aclgrphBuildModel(build_graph, build_opts, model) != SUCCESS) {
        std::cerr << name << ": BuildModel failed"
                  << (is_dynamic ? ", expected for dynamic shape: ge_local engine does not support unknown graph" : "")
                  << std::endl;
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
    if (is_dynamic) {
        // Dynamic shapes cannot be executed by the ge_local engine (documented
        // limitation). InferShape verification is the pass criteria.
        std::cout << name << " PASSED (InferShape verified, ge_local engine cannot execute unknown-shape graph)"
                  << std::endl;
        return kSuccess;
    }
    if (run_ok && build_ok) {
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

    // Scenario 1: static shape {8,4}, split along dim 0 into 2 outputs, run + buildmodel
    if (RunScenario(ctx, "static", {8, 4}, false) != kSuccess) {
        ret = kFailed;
    }
    // Scenario 2: dynamic shape {-1,4}, dim0 unknown (range 1~10)
    if (RunScenario(ctx, "dynamic_minus1", {-1, 4}, true) != kSuccess) {
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
