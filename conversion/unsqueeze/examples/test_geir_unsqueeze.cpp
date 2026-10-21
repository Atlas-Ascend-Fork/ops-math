/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include "assert.h"

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "ge_ir_build.h"

#include "../../../ge/data/op_graph/data_proto.h"
#include "../op_graph/unsqueeze_proto.h"

#define FAILED -1
#define SUCCESS 0

using std::map;
using std::string;
using std::vector;
using namespace ge;

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return string(tmp);
}

// static shape: x={2,3}, axes={1}, insert dim 1 at axis 1, y={2,1,3}
ge::Graph BuildGraph(std::vector<ge::Tensor>& input)
{
    ge::Graph graph("test_unsqueeze_ir");
    ge::DataType inDtype = ge::DT_FLOAT;
    vector<int64_t> xShape = {2, 3};

    auto data_node = op::Data("data_node");
    data_node.set_attr_index(0);
    ge::TensorDesc data_desc = ge::TensorDesc(ge::Shape(xShape), ge::FORMAT_ND, inDtype);
    data_desc.SetPlacement(ge::kPlacementHost);
    data_desc.SetFormat(ge::FORMAT_ND);
    data_node.update_input_desc_x(data_desc);

    auto unsqueeze_node = op::Unsqueeze("unsqueeze_node");
    unsqueeze_node.set_input_x(data_node);
    unsqueeze_node.set_attr_axes(vector<int64_t>{1});

    float* pData = new float[6];
    for (int i = 0; i < 6; i++) {
        pData[i] = 1.0f;
    }
    ge::Tensor tensor_data(data_desc, (uint8_t*)pData, 24);
    input.push_back(tensor_data);

    std::vector<ge::Operator> inputs{data_node};
    std::vector<ge::Operator> outputs{unsqueeze_node};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(data_node);
    graph.AddOp(unsqueeze_node);
    return graph;
}

// dynamic shape(-1): x={-1,3}, axes={1}, y={2,1,3}; feed real shape {2,3} when RunGraph
ge::Graph BuildDynamicShapeGraphDimNeg1(std::vector<ge::Tensor>& input)
{
    ge::Graph graph("test_unsqueeze_dynamic_dim_neg1");
    ge::DataType inDtype = ge::DT_FLOAT;
    vector<int64_t> xShape = {-1, 3};

    auto data_node = op::Data("data_node");
    data_node.set_attr_index(0);
    ge::TensorDesc data_desc = ge::TensorDesc(ge::Shape(xShape), ge::FORMAT_ND, inDtype);
    data_desc.SetPlacement(ge::kPlacementHost);
    data_desc.SetFormat(ge::FORMAT_ND);
    data_desc.SetShapeRange({{1, 10}, {3, 3}});
    data_node.update_input_desc_x(data_desc);

    auto unsqueeze_node = op::Unsqueeze("unsqueeze_node");
    unsqueeze_node.set_input_x(data_node);
    unsqueeze_node.set_attr_axes(vector<int64_t>{1});

    float* pData = new float[6];
    for (int i = 0; i < 6; i++) {
        pData[i] = 1.0f;
    }
    ge::TensorDesc real_desc = ge::TensorDesc(ge::Shape({2, 3}), ge::FORMAT_ND, inDtype);
    ge::Tensor tensor_data(real_desc, (uint8_t*)pData, 24);
    input.push_back(tensor_data);

    std::vector<ge::Operator> inputs{data_node};
    std::vector<ge::Operator> outputs{unsqueeze_node};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(data_node);
    graph.AddOp(unsqueeze_node);
    return graph;
}

// dynamic shape(-2): x={-2}, axes={1}, y={2,1,3}; feed real shape {2,3} when RunGraph
ge::Graph BuildDynamicShapeGraphDimNeg2(std::vector<ge::Tensor>& input)
{
    ge::Graph graph("test_unsqueeze_dynamic_dim_neg2");
    ge::DataType inDtype = ge::DT_FLOAT;
    vector<int64_t> xShape = {-2};

    auto data_node = op::Data("data_node");
    data_node.set_attr_index(0);
    ge::TensorDesc data_desc = ge::TensorDesc(ge::Shape(xShape), ge::FORMAT_ND, inDtype);
    data_desc.SetPlacement(ge::kPlacementHost);
    data_desc.SetFormat(ge::FORMAT_ND);
    // unknown rank(-2) has unknown dim num, no shape range is set;
    // the real input shape {2,3} is fed on RunGraph
    data_node.update_input_desc_x(data_desc);

    auto unsqueeze_node = op::Unsqueeze("unsqueeze_node");
    unsqueeze_node.set_input_x(data_node);
    unsqueeze_node.set_attr_axes(vector<int64_t>{1});

    float* pData = new float[6];
    for (int i = 0; i < 6; i++) {
        pData[i] = 1.0f;
    }
    ge::TensorDesc real_desc = ge::TensorDesc(ge::Shape({2, 3}), ge::FORMAT_ND, inDtype);
    ge::Tensor tensor_data(real_desc, (uint8_t*)pData, 24);
    input.push_back(tensor_data);

    std::vector<ge::Operator> inputs{data_node};
    std::vector<ge::Operator> outputs{unsqueeze_node};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(data_node);
    graph.AddOp(unsqueeze_node);
    return graph;
}

// the output tensor of RunGraph is the contractual result
bool CheckOutputTensor(const ge::Tensor& output, const vector<int64_t>& expected)
{
    auto desc = output.GetTensorDesc();
    auto shape = desc.GetShape();
    printf("%s - INFO - [XIR]: output desc: dim_num=%zu", GetTime().c_str(), shape.GetDimNum());
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        printf(" dim%zu=%ld", i, shape.GetDim(i));
    }
    printf(" shape_size=%ld dtype=%d | expected: dim_num=%zu", shape.GetShapeSize(), desc.GetDataType(),
           expected.size());
    for (size_t i = 0; i < expected.size(); i++) {
        printf(" dim%zu=%ld", i, expected[i]);
    }
    printf(" dtype=%d\n", ge::DT_FLOAT);

    if (shape.GetDimNum() != expected.size()) {
        printf("%s - ERROR - [XIR]: Output shape FAILED: expected dim_num=%zu\n", GetTime().c_str(), expected.size());
        return false;
    }
    for (size_t i = 0; i < expected.size(); i++) {
        if (shape.GetDim(i) != expected[i]) {
            printf("%s - ERROR - [XIR]: Output shape FAILED: dim%zu expected=%ld actual=%ld\n", GetTime().c_str(), i,
                   expected[i], shape.GetDim(i));
            return false;
        }
    }
    if (desc.GetDataType() != ge::DT_FLOAT) {
        printf("%s - ERROR - [XIR]: Output dtype FAILED: expected=%d actual=%d\n", GetTime().c_str(), ge::DT_FLOAT,
               desc.GetDataType());
        return false;
    }
    return true;
}

struct RunContext {
    const map<ge::AscendString, ge::AscendString>& session_options;
    uint32_t graph_id;
};

// runtime_shape is the expected full shape of the RunGraph output tensor
struct ExpectedResult {
    vector<int64_t> runtime_shape;
    int64_t size;
};

bool RunScenario(const string& tag, ge::Graph (*builder)(vector<ge::Tensor>&), const ExpectedResult& expected,
                 RunContext& ctx, const string& om_name)
{
    printf("%s - INFO - [XIR]: === %s ===\n", GetTime().c_str(), tag.c_str());

    vector<ge::Tensor> input;
    ge::Graph graph = builder(input);

    ge::Session* session = new ge::Session(ctx.session_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create session failed\n", GetTime().c_str());
        return false;
    }
    ge::Status ret = session->AddGraph(ctx.graph_id, graph, ctx.session_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        return false;
    }

    vector<ge::Tensor> output;
    ret = session->RunGraph(ctx.graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: RunGraph failed\n", GetTime().c_str());
        delete session;
        return false;
    }
    printf("%s - INFO - [XIR]: RunGraph success, output count=%zu\n", GetTime().c_str(), output.size());

    if (output.empty() || !CheckOutputTensor(output[0], expected.runtime_shape)) {
        delete session;
        return false;
    }
    // Unsqueeze is a view op: output values should be the same as input values
    {
        const auto* out_data = reinterpret_cast<const float*>(output[0].GetData());
        const auto* in_data = reinterpret_cast<const float*>(input[0].GetData());
        bool values_ok = true;
        for (int64_t i = 0; i < expected.size; i++) {
            if (out_data[i] != in_data[i]) {
                printf("%s - ERROR - [XIR]: Output value verification FAILED: out[%ld]=%.4f expected=%.4f\n",
                       GetTime().c_str(), i, out_data[i], in_data[i]);
                values_ok = false;
                break;
            }
        }
        if (!values_ok) {
            delete session;
            return false;
        }
        printf("%s - INFO - [XIR]: Output value verification success (view: output == input)\n", GetTime().c_str());
    }
    delete session;

    vector<ge::Tensor> input_om;
    ge::Graph graph_om = builder(input_om);
    ge::ModelBufferData model_buffer;
    ret = aclgrphBuildModel(graph_om, ctx.session_options, model_buffer);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: BuildModel failed\n", GetTime().c_str());
        return false;
    }
    aclgrphSaveModel(om_name.c_str(), model_buffer);
    printf("%s - INFO - [XIR]: Save OM success: %s\n", GetTime().c_str(), om_name.c_str());
    printf("%s - INFO - [XIR]: %s PASSED\n", GetTime().c_str(), tag.c_str());
    ctx.graph_id++;
    return true;
}

int main(int argc, char* argv[])
{
    printf("%s - INFO - [XIR]: Start to initialize ge\n", GetTime().c_str());
    std::map<ge::AscendString, ge::AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    ge::Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge success\n", GetTime().c_str());

    std::map<ge::AscendString, ge::AscendString> session_options;
    RunContext ctx{session_options, 0};

    // static scenario shape is determined at compile stage;
    // dynamic scenarios feed {2,3} at runtime, output is {2,1,3} after inserting dim 1 at axis 1
    ExpectedResult static_expected{{2, 1, 3}, 6};
    ExpectedResult neg1_expected{{2, 1, 3}, 6};
    ExpectedResult neg2_expected{{2, 1, 3}, 6};

    if (!RunScenario("static", BuildGraph, static_expected, ctx, "./unsqueeze_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("dynamic(-1)", BuildDynamicShapeGraphDimNeg1, neg1_expected, ctx,
                     "./unsqueeze_dynamic_neg1_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("dynamic(-2)", BuildDynamicShapeGraphDimNeg2, neg2_expected, ctx,
                     "./unsqueeze_dynamic_neg2_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    ge::GEFinalize();
    printf("%s - INFO - [XIR]: Finalize success\n", GetTime().c_str());
    return SUCCESS;
}
