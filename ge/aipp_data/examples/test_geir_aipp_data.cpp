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

#include "../../data/op_graph/data_proto.h"
#include "../op_graph/aipp_data_proto.h"

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

// static shape: data={2,3}; dynamic scenarios use a dynamic desc with the same fed real shape {2,3}
ge::Graph BuildGraph(std::vector<ge::Tensor>& input, const vector<int64_t>& xShape, const string& graphName)
{
    ge::Graph graph(graphName.c_str());
    ge::DataType inDtype = ge::DT_UINT8;

    auto data_node = op::Data("data_node");
    data_node.set_attr_index(0);
    ge::TensorDesc data_desc = ge::TensorDesc(ge::Shape(xShape), ge::FORMAT_ND, inDtype);
    data_desc.SetPlacement(ge::kPlacementHost);
    data_desc.SetFormat(ge::FORMAT_ND);
    if (xShape[0] == -1) {
        data_desc.SetShapeRange({{1, 10}, {3, 3}});
    }
    // unknown rank(-2) has unknown dim num, no shape range is set
    data_node.update_input_desc_x(data_desc);

    auto aipp_node = op::AippData("aipp_data_node");
    aipp_node.set_input_data(data_node);
    ge::TensorDesc aipp_desc = ge::TensorDesc(ge::Shape(xShape), ge::FORMAT_ND, inDtype);
    aipp_node.update_input_desc_data(aipp_desc);

    uint8_t* pData = new uint8_t[6];
    for (int i = 0; i < 6; i++) {
        pData[i] = 1;
    }
    ge::TensorDesc real_desc = ge::TensorDesc(ge::Shape({2, 3}), ge::FORMAT_ND, inDtype);
    ge::Tensor tensor_data(real_desc, (uint8_t*)pData, 6);
    input.push_back(tensor_data);
    // AippData acts as a framework-injected graph input (dynamic AIPP adds one input node at last):
    // RunGraph needs one more fed tensor for the aipp slot, otherwise it fails
    uint8_t* pAippData = new uint8_t[6];
    for (int i = 0; i < 6; i++) {
        pAippData[i] = 1;
    }
    ge::Tensor aipp_tensor_data(real_desc, (uint8_t*)pAippData, 6);
    input.push_back(aipp_tensor_data);

    std::vector<ge::Operator> inputs{data_node};
    std::vector<ge::Operator> outputs{aipp_node};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(data_node);
    graph.AddOp(aipp_node);
    return graph;
}

// AippData is a passthrough node: the output shape/dtype/values are the same as the fed input
bool CheckOutput(const vector<ge::Tensor>& output, const vector<ge::Tensor>& input)
{
    if (output.size() != 1U) {
        printf("%s - ERROR - [XIR]: output count=%zu, expected 1\n", GetTime().c_str(), output.size());
        return false;
    }
    auto desc = output[0].GetTensorDesc();
    auto shape = desc.GetShape();
    printf("%s - INFO - [XIR]: output desc: dim_num=%zu", GetTime().c_str(), shape.GetDimNum());
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        printf(" dim%zu=%ld", i, shape.GetDim(i));
    }
    printf(" shape_size=%ld dtype=%d\n", shape.GetShapeSize(), desc.GetDataType());

    if (shape.GetDimNum() != 2 || shape.GetDim(0) != 2 || shape.GetDim(1) != 3) {
        printf("%s - ERROR - [XIR]: Output shape FAILED: expected {2,3}\n", GetTime().c_str());
        return false;
    }
    if (shape.GetShapeSize() != 6) {
        printf("%s - ERROR - [XIR]: Output shape_size FAILED: expected 6\n", GetTime().c_str());
        return false;
    }
    if (desc.GetDataType() != ge::DT_UINT8) {
        printf("%s - ERROR - [XIR]: Output dtype FAILED: expected=%d actual=%d\n", GetTime().c_str(), ge::DT_UINT8,
               desc.GetDataType());
        return false;
    }
    const uint8_t* outData = output[0].GetData();
    const uint8_t* inData = input[0].GetData();
    if (outData == nullptr || inData == nullptr) {
        printf("%s - ERROR - [XIR]: Output data is null\n", GetTime().c_str());
        return false;
    }
    for (int64_t i = 0; i < 6; i++) {
        if (outData[i] != inData[i]) {
            printf("%s - ERROR - [XIR]: Output value verification FAILED: out[%ld]=%d expected=%d\n", GetTime().c_str(),
                   i, outData[i], inData[i]);
            return false;
        }
    }
    printf("%s - INFO - [XIR]: Output value verification success (passthrough: output == input)\n", GetTime().c_str());
    return true;
}

struct RunContext {
    const map<ge::AscendString, ge::AscendString>& session_options;
    uint32_t graph_id;
};

bool RunScenario(const string& tag, const vector<int64_t>& xShape, const string& graphName, const string& omName,
                 RunContext& ctx)
{
    printf("%s - INFO - [XIR]: === %s ===\n", GetTime().c_str(), tag.c_str());

    vector<ge::Tensor> input;
    ge::Graph graph = BuildGraph(input, xShape, graphName);

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
        printf("%s - ERROR - [XIR]: Run graph on NPU failed\n", GetTime().c_str());
        delete session;
        return false;
    }
    printf("%s - INFO - [XIR]: Run graph on NPU success, output count=%zu\n", GetTime().c_str(), output.size());
    delete session;

    if (!CheckOutput(output, input)) {
        return false;
    }

    vector<ge::Tensor> inputOm;
    ge::Graph graphOm = BuildGraph(inputOm, xShape, graphName);
    ge::ModelBufferData model_buffer;
    ret = aclgrphBuildModel(graphOm, ctx.session_options, model_buffer);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Build model failed\n", GetTime().c_str());
        return false;
    }
    aclgrphSaveModel(omName.c_str(), model_buffer);
    printf("%s - INFO - [XIR]: Save OM success: %s\n", GetTime().c_str(), omName.c_str());
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

    // all scenarios feed the real shape {2,3}: the passthrough output keeps {2,3}
    if (!RunScenario("static", vector<int64_t>{2, 3}, "test_aipp_data_ir", "./aipp_data_model.om", ctx)) {
        ge::GEFinalize();
        return FAILED;
    }
    if (!RunScenario("dynamic(-1)", vector<int64_t>{-1, 3}, "test_aipp_data_dynamic_dim_neg1",
                     "./aipp_data_dynamic_neg1_model.om", ctx)) {
        ge::GEFinalize();
        return FAILED;
    }
    if (!RunScenario("dynamic(-2)", vector<int64_t>{-2}, "test_aipp_data_dynamic_dim_neg2",
                     "./aipp_data_dynamic_neg2_model.om", ctx)) {
        ge::GEFinalize();
        return FAILED;
    }

    ge::GEFinalize();
    printf("%s - INFO - [XIR]: Finalize success\n", GetTime().c_str());
    return SUCCESS;
}
