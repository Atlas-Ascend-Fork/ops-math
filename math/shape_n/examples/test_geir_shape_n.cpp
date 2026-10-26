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

#include "../op_graph/shape_n_proto.h"
#include "../../../ge/data/op_graph/data_proto.h"

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

// 构图：Data(x0), Data(x1) --> ShapeN --> y0, y1（2 个动态输入/输出实例）
ge::Graph BuildShapeNGraph(std::vector<ge::Tensor>& input, const vector<int64_t>& x0Shape,
                           const vector<int64_t>& x1Shape, const ge::DataType outDtype, const bool setDtypeAttr)
{
    ge::Graph graph("test_shape_n_ir");
    ge::DataType inDtype = ge::DT_FLOAT;

    auto data_node0 = op::Data("data_node0");
    data_node0.set_attr_index(0);
    ge::TensorDesc data_desc0 = ge::TensorDesc(ge::Shape(x0Shape), ge::FORMAT_ND, inDtype);
    data_desc0.SetPlacement(ge::kPlacementHost);
    data_desc0.SetFormat(ge::FORMAT_ND);
    data_node0.update_input_desc_x(data_desc0);

    auto data_node1 = op::Data("data_node1");
    data_node1.set_attr_index(1);
    ge::TensorDesc data_desc1 = ge::TensorDesc(ge::Shape(x1Shape), ge::FORMAT_ND, inDtype);
    data_desc1.SetPlacement(ge::kPlacementHost);
    data_desc1.SetFormat(ge::FORMAT_ND);
    data_node1.update_input_desc_x(data_desc1);

    float* pData = new float[6];
    for (int i = 0; i < 6; i++) {
        pData[i] = 1.0f;
    }
    float* pData1 = new float[24];
    for (int i = 0; i < 24; i++) {
        pData1[i] = 1.0f;
    }
    ge::TensorDesc real_desc0 = ge::TensorDesc(ge::Shape({2, 3}), ge::FORMAT_ND, inDtype);
    ge::Tensor tensor_data0(real_desc0, (uint8_t*)pData, 24);
    input.push_back(tensor_data0);
    ge::TensorDesc real_desc1 = ge::TensorDesc(ge::Shape({2, 3, 4}), ge::FORMAT_ND, inDtype);
    ge::Tensor tensor_data1(real_desc1, (uint8_t*)pData1, 96);
    input.push_back(tensor_data1);

    auto shape_n_op = op::ShapeN("shape_n_node");
    shape_n_op.create_dynamic_input_x(2, false);
    shape_n_op.create_dynamic_output_y(2);
    shape_n_op.set_dynamic_input_x(0, data_node0);
    shape_n_op.set_dynamic_input_x(1, data_node1);
    if (setDtypeAttr) {
        shape_n_op.set_attr_dtype(static_cast<int64_t>(outDtype));
    }
    // 初始 desc，AddGraph 后 InferShape/InferDataType 会更新该 desc
    shape_n_op.UpdateDynamicOutputDesc("y", 0, ge::TensorDesc(ge::Shape({2}), ge::FORMAT_ND, outDtype));
    shape_n_op.UpdateDynamicOutputDesc("y", 1, ge::TensorDesc(ge::Shape({3}), ge::FORMAT_ND, outDtype));

    std::vector<ge::Operator> inputs{data_node0, data_node1};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
    outputs.emplace_back(shape_n_op, std::vector<size_t>{0, 1});
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(data_node0);
    graph.AddOp(data_node1);
    graph.AddOp(shape_n_op);
    return graph;
}

ge::Graph BuildStaticGraph(std::vector<ge::Tensor>& input)
{
    // 静态 shape：dtype 属性显式设置为 DT_INT64，验证 InferDataType 属性路径
    return BuildShapeNGraph(input, {2, 3}, {2, 3, 4}, ge::DT_INT64, true);
}

ge::Graph BuildDynamicShapeGraphDimNeg1(std::vector<ge::Tensor>& input)
{
    // 动态 shape(-1)：dtype 属性缺省，验证 InferDataType 默认值路径（DT_INT32）
    ge::Graph graph = BuildShapeNGraph(input, {-1, 3}, {2, -1, 4}, ge::DT_INT32, false);
    ge::Operator data_op;
    graph.FindOpByName("data_node0", data_op);
    auto data_desc = data_op.GetInputDesc(0);
    data_desc.SetShapeRange({{1, 10}, {3, 3}});
    data_op.UpdateInputDesc("x", data_desc);

    ge::Operator data_op1;
    graph.FindOpByName("data_node1", data_op1);
    auto data_desc1 = data_op1.GetInputDesc(0);
    data_desc1.SetShapeRange({{2, 2}, {1, 5}, {4, 4}});
    data_op1.UpdateInputDesc("x", data_desc1);
    return graph;
}

ge::Graph BuildDynamicShapeGraphDimNeg2(std::vector<ge::Tensor>& input)
{
    // 动态 shape(-2)：未知秩，无 shape range，真实 shape 由 RunGraph 喂入
    return BuildShapeNGraph(input, {-2}, {-2}, ge::DT_INT32, false);
}

struct ExpectedResult {
    vector<int64_t> dims;
    ge::DataType dtype;
    vector<int64_t> values;
};

struct ScenarioExpected {
    ExpectedResult output0;
    ExpectedResult output1;
};

bool CheckInferShape(const ge::Graph& graph, const ScenarioExpected& expected)
{
    ge::Operator shape_n_op;
    graph.FindOpByName("shape_n_node", shape_n_op);
    const vector<vector<int64_t>> expectedDims = {expected.output0.dims, expected.output1.dims};
    for (size_t outIdx = 0; outIdx < expectedDims.size(); outIdx++) {
        auto desc = shape_n_op.GetOutputDesc(outIdx);
        auto shape = desc.GetShape();
        printf("%s - INFO - [XIR]: InferShape output[%zu]: dim_num=%zu", GetTime().c_str(), outIdx, shape.GetDimNum());
        for (size_t i = 0; i < shape.GetDimNum(); i++) {
            printf(" dim%zu=%ld", i, shape.GetDim(i));
        }
        printf(" dtype=%d | expected: dim_num=%zu dtype=%d\n", desc.GetDataType(), expectedDims[outIdx].size(),
               expected.output0.dtype);

        if (shape.GetDimNum() != expectedDims[outIdx].size()) {
            printf("%s - ERROR - [XIR]: InferShape FAILED: output[%zu] expected dim_num=%zu\n", GetTime().c_str(),
                   outIdx, expectedDims[outIdx].size());
            return false;
        }
        for (size_t i = 0; i < expectedDims[outIdx].size(); i++) {
            if (shape.GetDim(i) != expectedDims[outIdx][i]) {
                printf("%s - ERROR - [XIR]: InferShape FAILED: output[%zu] dim%zu expected=%ld actual=%ld\n",
                       GetTime().c_str(), outIdx, i, expectedDims[outIdx][i], shape.GetDim(i));
                return false;
            }
        }
        if (desc.GetDataType() != expected.output0.dtype) {
            printf("%s - ERROR - [XIR]: InferDataType FAILED: output[%zu] expected=%d actual=%d\n", GetTime().c_str(),
                   outIdx, expected.output0.dtype, desc.GetDataType());
            return false;
        }
    }
    return true;
}

bool CheckSingleOutput(const ge::Tensor& output, const ExpectedResult& expected, const size_t outIdx)
{
    auto out_shape = output.GetTensorDesc().GetShape();
    printf("%s - INFO - [XIR]: output[%zu] dim_num=%zu", GetTime().c_str(), outIdx, out_shape.GetDimNum());
    for (size_t d = 0; d < out_shape.GetDimNum(); d++) {
        printf(" dim%zu=%ld", d, out_shape.GetDim(d));
    }
    printf(" shape_size=%ld dtype=%d\n", out_shape.GetShapeSize(), output.GetTensorDesc().GetDataType());

    if (out_shape.GetDimNum() != expected.dims.size()) {
        printf("%s - ERROR - [XIR]: Output verification FAILED: output[%zu] expected dim_num=%zu\n", GetTime().c_str(),
               outIdx, expected.dims.size());
        return false;
    }
    for (size_t i = 0; i < expected.dims.size(); i++) {
        if (out_shape.GetDim(i) != expected.dims[i]) {
            printf("%s - ERROR - [XIR]: Output verification FAILED: output[%zu] dim%zu expected=%ld actual=%ld\n",
                   GetTime().c_str(), outIdx, i, expected.dims[i], out_shape.GetDim(i));
            return false;
        }
    }
    if (output.GetTensorDesc().GetDataType() != expected.dtype) {
        printf("%s - ERROR - [XIR]: Output verification FAILED: output[%zu] dtype expected=%d actual=%d\n",
               GetTime().c_str(), outIdx, expected.dtype, output.GetTensorDesc().GetDataType());
        return false;
    }
    // ShapeN 输出张量的值即对应输入的各维度值（int32/int64 存储）
    if (expected.dtype == ge::DT_INT32) {
        const int32_t* data = reinterpret_cast<const int32_t*>(output.GetData());
        for (size_t i = 0; i < expected.values.size(); i++) {
            if (static_cast<int64_t>(data[i]) != expected.values[i]) {
                printf("%s - ERROR - [XIR]: Output verification FAILED: output[%zu] value%zu expected=%ld actual=%d\n",
                       GetTime().c_str(), outIdx, i, expected.values[i], data[i]);
                return false;
            }
        }
    } else if (expected.dtype == ge::DT_INT64) {
        const int64_t* data = reinterpret_cast<const int64_t*>(output.GetData());
        for (size_t i = 0; i < expected.values.size(); i++) {
            if (data[i] != expected.values[i]) {
                printf("%s - ERROR - [XIR]: Output verification FAILED: output[%zu] value%zu expected=%ld actual=%ld\n",
                       GetTime().c_str(), outIdx, i, expected.values[i], data[i]);
                return false;
            }
        }
    }
    return true;
}

bool CheckOutput(const std::vector<ge::Tensor>& output, const ScenarioExpected& expected)
{
    if (output.size() != 2U) {
        printf("%s - ERROR - [XIR]: Output verification FAILED: output count=%zu expected=2\n", GetTime().c_str(),
               output.size());
        return false;
    }
    if (!CheckSingleOutput(output[0], expected.output0, 0)) {
        return false;
    }
    if (!CheckSingleOutput(output[1], expected.output1, 1)) {
        return false;
    }
    return true;
}

struct RunContext {
    const map<ge::AscendString, ge::AscendString>& session_options;
    uint32_t graph_id;
};

bool RunScenario(const string& tag, ge::Graph (*builder)(vector<ge::Tensor>&), const ScenarioExpected& expected,
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

    if (!CheckOutput(output, expected)) {
        delete session;
        return false;
    }

    if (!CheckInferShape(graph, expected)) {
        delete session;
        return false;
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

    // 静态 shape：输入 {2,3}/{2,3,4}，输出 {2}/{3}，dtype 属性 DT_INT64
    ScenarioExpected static_expected;
    static_expected.output0 = {{2}, ge::DT_INT64, {2, 3}};
    static_expected.output1 = {{3}, ge::DT_INT64, {2, 3, 4}};
    // 动态(-1)：输入 {-1,3}/{2,-1,4} 喂入 {2,3}/{2,3,4}，输出 {2}/{3}，dtype 缺省 DT_INT32
    ScenarioExpected neg1_expected;
    neg1_expected.output0 = {{2}, ge::DT_INT32, {2, 3}};
    neg1_expected.output1 = {{3}, ge::DT_INT32, {2, 3, 4}};
    // 动态(-2)：输入 {-2}/{-2} 喂入 {2,3}/{2,3,4}，输出 {2}/{3}，dtype 缺省 DT_INT32
    ScenarioExpected neg2_expected;
    neg2_expected.output0 = {{2}, ge::DT_INT32, {2, 3}};
    neg2_expected.output1 = {{3}, ge::DT_INT32, {2, 3, 4}};

    if (!RunScenario("static", BuildStaticGraph, static_expected, ctx, "./shape_n_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("dynamic(-1)", BuildDynamicShapeGraphDimNeg1, neg1_expected, ctx,
                     "./shape_n_dynamic_neg1_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("dynamic(-2)", BuildDynamicShapeGraphDimNeg2, neg2_expected, ctx,
                     "./shape_n_dynamic_neg2_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    ge::GEFinalize();
    printf("%s - INFO - [XIR]: Finalize success\n", GetTime().c_str());
    return SUCCESS;
}
