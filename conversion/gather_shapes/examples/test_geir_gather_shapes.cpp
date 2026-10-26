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

#include "../op_graph/gather_shapes_proto.h"
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

// 构图：Data(x0), Data(x1) --> GatherShapes(axes) --> shape
ge::Graph BuildGatherShapesGraph(std::vector<ge::Tensor>& input, const vector<int64_t>& x0Shape,
                                 const vector<int64_t>& x1Shape, const std::vector<std::vector<int64_t>>& axes,
                                 const ge::DataType outDtype, const bool setDtypeAttr)
{
    ge::Graph graph("test_gather_shapes_ir");
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

    auto gather_shapes_op = op::GatherShapes("gather_shapes_node");
    gather_shapes_op.create_dynamic_input_x(2, false);
    gather_shapes_op.set_dynamic_input_x(0, data_node0);
    gather_shapes_op.set_dynamic_input_x(1, data_node1);
    gather_shapes_op.set_attr_axes(axes);
    if (setDtypeAttr) {
        gather_shapes_op.set_attr_dtype(static_cast<int64_t>(outDtype));
    }
    // 初始 desc，AddGraph 后 InferShape/InferDataType 会更新该 desc
    gather_shapes_op.update_output_desc_shape(
        ge::TensorDesc(ge::Shape({static_cast<int64_t>(axes.size())}), ge::FORMAT_ND, outDtype));

    std::vector<ge::Operator> inputs{data_node0, data_node1};
    std::vector<ge::Operator> outputs{gather_shapes_op};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(data_node0);
    graph.AddOp(data_node1);
    graph.AddOp(gather_shapes_op);
    return graph;
}

ge::Graph BuildStaticGraph(std::vector<ge::Tensor>& input)
{
    // 静态 shape：axes 收集两个输入的 4 个维度，dtype 属性显式设置为 DT_INT64
    return BuildGatherShapesGraph(input, {2, 3}, {2, 3, 4}, {{0, 0}, {0, 1}, {1, 1}, {1, 2}}, ge::DT_INT64, true);
}

ge::Graph BuildDynamicShapeGraphDimNeg1(std::vector<ge::Tensor>& input)
{
    // 动态 shape(-1)：dtype 属性缺省，验证 InferDataType 默认值路径（DT_INT32）
    ge::Graph graph = BuildGatherShapesGraph(input, {-1, 3}, {2, 3, 4}, {{0, 0}, {0, 1}, {1, 2}}, ge::DT_INT32, false);
    ge::Operator data_op;
    graph.FindOpByName("data_node0", data_op);
    auto data_desc = data_op.GetInputDesc(0);
    data_desc.SetShapeRange({{1, 10}, {3, 3}});
    data_op.UpdateInputDesc("x", data_desc);
    return graph;
}

ge::Graph BuildDynamicShapeGraphDimNeg2(std::vector<ge::Tensor>& input)
{
    // 动态 shape(-2)：未知秩，无 shape range，真实 shape 由 RunGraph 喂入
    return BuildGatherShapesGraph(input, {-2}, {2, 3, 4}, {{0, 0}, {0, 1}}, ge::DT_INT32, false);
}

struct ExpectedResult {
    vector<int64_t> dims;
    ge::DataType dtype;
    vector<int64_t> values;
};

bool CheckInferShape(const ge::Graph& graph, const vector<int64_t>& expectedDims, const ge::DataType expectedDtype)
{
    ge::Operator gather_shapes_op;
    graph.FindOpByName("gather_shapes_node", gather_shapes_op);
    auto desc = gather_shapes_op.GetOutputDesc(0);
    auto shape = desc.GetShape();
    printf("%s - INFO - [XIR]: InferShape: dim_num=%zu", GetTime().c_str(), shape.GetDimNum());
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        printf(" dim%zu=%ld", i, shape.GetDim(i));
    }
    printf(" dtype=%d | expected: dim_num=%zu", desc.GetDataType(), expectedDims.size());
    for (size_t i = 0; i < expectedDims.size(); i++) {
        printf(" dim%zu=%ld", i, expectedDims[i]);
    }
    printf(" dtype=%d\n", expectedDtype);

    if (shape.GetDimNum() != expectedDims.size()) {
        printf("%s - ERROR - [XIR]: InferShape FAILED: expected dim_num=%zu\n", GetTime().c_str(), expectedDims.size());
        return false;
    }
    for (size_t i = 0; i < expectedDims.size(); i++) {
        if (shape.GetDim(i) != expectedDims[i]) {
            printf("%s - ERROR - [XIR]: InferShape FAILED: dim%zu expected=%ld actual=%ld\n", GetTime().c_str(), i,
                   expectedDims[i], shape.GetDim(i));
            return false;
        }
    }
    if (desc.GetDataType() != expectedDtype) {
        printf("%s - ERROR - [XIR]: InferDataType FAILED: expected=%d actual=%d\n", GetTime().c_str(), expectedDtype,
               desc.GetDataType());
        return false;
    }
    return true;
}

bool CheckOutput(const std::vector<ge::Tensor>& output, const ExpectedResult& expected)
{
    if (output.empty()) {
        printf("%s - ERROR - [XIR]: Output verification FAILED: output is empty\n", GetTime().c_str());
        return false;
    }
    auto out_shape = output[0].GetTensorDesc().GetShape();
    printf("%s - INFO - [XIR]: output[0] dim_num=%zu", GetTime().c_str(), out_shape.GetDimNum());
    for (size_t d = 0; d < out_shape.GetDimNum(); d++) {
        printf(" dim%zu=%ld", d, out_shape.GetDim(d));
    }
    printf(" shape_size=%ld dtype=%d\n", out_shape.GetShapeSize(), output[0].GetTensorDesc().GetDataType());

    if (out_shape.GetDimNum() != expected.dims.size()) {
        printf("%s - ERROR - [XIR]: Output verification FAILED: expected dim_num=%zu\n", GetTime().c_str(),
               expected.dims.size());
        return false;
    }
    for (size_t i = 0; i < expected.dims.size(); i++) {
        if (out_shape.GetDim(i) != expected.dims[i]) {
            printf("%s - ERROR - [XIR]: Output verification FAILED: dim%zu expected=%ld actual=%ld\n",
                   GetTime().c_str(), i, expected.dims[i], out_shape.GetDim(i));
            return false;
        }
    }
    if (output[0].GetTensorDesc().GetDataType() != expected.dtype) {
        printf("%s - ERROR - [XIR]: Output verification FAILED: dtype expected=%d actual=%d\n", GetTime().c_str(),
               expected.dtype, output[0].GetTensorDesc().GetDataType());
        return false;
    }
    // GatherShapes 输出张量的值为 axes 收集到的各输入维度值（int32/int64 存储）
    if (expected.dtype == ge::DT_INT32) {
        const int32_t* data = reinterpret_cast<const int32_t*>(output[0].GetData());
        for (size_t i = 0; i < expected.values.size(); i++) {
            if (static_cast<int64_t>(data[i]) != expected.values[i]) {
                printf("%s - ERROR - [XIR]: Output verification FAILED: value%zu expected=%ld actual=%d\n",
                       GetTime().c_str(), i, expected.values[i], data[i]);
                return false;
            }
        }
    } else if (expected.dtype == ge::DT_INT64) {
        const int64_t* data = reinterpret_cast<const int64_t*>(output[0].GetData());
        for (size_t i = 0; i < expected.values.size(); i++) {
            if (data[i] != expected.values[i]) {
                printf("%s - ERROR - [XIR]: Output verification FAILED: value%zu expected=%ld actual=%ld\n",
                       GetTime().c_str(), i, expected.values[i], data[i]);
                return false;
            }
        }
    }
    return true;
}

struct RunContext {
    const map<ge::AscendString, ge::AscendString>& session_options;
    uint32_t graph_id;
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

    if (!CheckOutput(output, expected)) {
        delete session;
        return false;
    }

    if (!CheckInferShape(graph, expected.dims, expected.dtype)) {
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

    // 静态 shape：axes={{0,0},{0,1},{1,1},{1,2}}，输入 {2,3}/{2,3,4}，输出 {4}=[2,3,3,4]，dtype 属性 DT_INT64
    ExpectedResult static_expected{{4}, ge::DT_INT64, {2, 3, 3, 4}};
    // 动态(-1)：axes={{0,0},{0,1},{1,2}}，输入 {-1,3}/{2,3,4} 喂入 {2,3}，输出 {3}=[2,3,4]，dtype 缺省 DT_INT32
    ExpectedResult neg1_expected{{3}, ge::DT_INT32, {2, 3, 4}};
    // 动态(-2)：axes={{0,0},{0,1}}，输入 {-2}/{2,3,4} 喂入 {2,3}，输出 {2}=[2,3]，dtype 缺省 DT_INT32
    ExpectedResult neg2_expected{{2}, ge::DT_INT32, {2, 3}};

    if (!RunScenario("static", BuildStaticGraph, static_expected, ctx, "./gather_shapes_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("dynamic(-1)", BuildDynamicShapeGraphDimNeg1, neg1_expected, ctx,
                     "./gather_shapes_dynamic_neg1_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("dynamic(-2)", BuildDynamicShapeGraphDimNeg2, neg2_expected, ctx,
                     "./gather_shapes_dynamic_neg2_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    ge::GEFinalize();
    printf("%s - INFO - [XIR]: Finalize success\n", GetTime().c_str());
    return SUCCESS;
}
