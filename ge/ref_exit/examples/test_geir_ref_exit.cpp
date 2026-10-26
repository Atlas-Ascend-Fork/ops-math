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
#include "ge_api.h"
#include "ge_ir_build.h"

#include "../../data/op_graph/data_proto.h"
#include "../op_graph/ref_exit_proto.h"

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

ge::Graph BuildGraph(std::vector<ge::Tensor>& input)
{
    ge::Graph graph("test_ref_exit_ir");
    ge::DataType inDtype = ge::DT_FLOAT;
    vector<int64_t> xShape = {2, 3};

    auto data_node = op::Data("data_node");
    data_node.set_attr_index(0);
    ge::TensorDesc data_desc = ge::TensorDesc(ge::Shape(xShape), ge::FORMAT_ND, inDtype);
    data_desc.SetPlacement(ge::kPlacementHost);
    data_desc.SetFormat(ge::FORMAT_ND);
    data_node.update_input_desc_x(data_desc);
    data_node.update_output_desc_y(data_desc);

    float* pData = new float[6];
    for (int i = 0; i < 6; i++) {
        pData[i] = 1.0f;
    }
    ge::Tensor tensor_data(data_desc, (uint8_t*)pData, 24);
    input.push_back(tensor_data);

    auto ref_exit_node = op::RefExit("ref_exit_node");
    ref_exit_node.set_input_x(data_node);

    std::vector<ge::Operator> inputs{data_node};
    std::vector<ge::Operator> outputs{ref_exit_node};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(data_node);
    graph.AddOp(ref_exit_node);
    return graph;
}

ge::Graph BuildDynamicShapeGraphDimNeg1(std::vector<ge::Tensor>& input)
{
    ge::Graph graph("test_ref_exit_dynamic_dim_neg1");
    ge::DataType inDtype = ge::DT_FLOAT;
    vector<int64_t> xShape = {-1, 3};

    auto data_node = op::Data("data_node");
    data_node.set_attr_index(0);
    ge::TensorDesc data_desc = ge::TensorDesc(ge::Shape(xShape), ge::FORMAT_ND, inDtype);
    data_desc.SetPlacement(ge::kPlacementHost);
    data_desc.SetFormat(ge::FORMAT_ND);
    data_desc.SetShapeRange({{1, 10}, {3, 3}});
    data_node.update_input_desc_x(data_desc);
    data_node.update_output_desc_y(data_desc);

    float* pData = new float[6];
    for (int i = 0; i < 6; i++) {
        pData[i] = 1.0f;
    }
    ge::TensorDesc real_desc = ge::TensorDesc(ge::Shape({2, 3}), ge::FORMAT_ND, inDtype);
    ge::Tensor tensor_data(real_desc, (uint8_t*)pData, 24);
    input.push_back(tensor_data);

    auto ref_exit_node = op::RefExit("ref_exit_node");
    ref_exit_node.set_input_x(data_node);

    std::vector<ge::Operator> inputs{data_node};
    std::vector<ge::Operator> outputs{ref_exit_node};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(data_node);
    graph.AddOp(ref_exit_node);
    return graph;
}

ge::Graph BuildDynamicShapeGraphDimNeg2(std::vector<ge::Tensor>& input)
{
    ge::Graph graph("test_ref_exit_dynamic_dim_neg2");
    ge::DataType inDtype = ge::DT_FLOAT;
    vector<int64_t> xShape = {-2};

    auto data_node = op::Data("data_node");
    data_node.set_attr_index(0);
    ge::TensorDesc data_desc = ge::TensorDesc(ge::Shape(xShape), ge::FORMAT_ND, inDtype);
    data_desc.SetPlacement(ge::kPlacementHost);
    data_desc.SetFormat(ge::FORMAT_ND);
    // unknown rank(-2) has unknown dim num, no shape range is set; the real shape is fed on RunGraph
    data_node.update_input_desc_x(data_desc);
    data_node.update_output_desc_y(data_desc);

    float* pData = new float[6];
    for (int i = 0; i < 6; i++) {
        pData[i] = 1.0f;
    }
    ge::TensorDesc real_desc = ge::TensorDesc(ge::Shape({2, 3}), ge::FORMAT_ND, inDtype);
    ge::Tensor tensor_data(real_desc, (uint8_t*)pData, 24);
    input.push_back(tensor_data);

    auto ref_exit_node = op::RefExit("ref_exit_node");
    ref_exit_node.set_input_x(data_node);

    std::vector<ge::Operator> inputs{data_node};
    std::vector<ge::Operator> outputs{ref_exit_node};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(data_node);
    graph.AddOp(ref_exit_node);
    return graph;
}

bool CheckInferShape(const ge::Graph& graph)
{
    // 显式触发推导，独立校验本算子的 InferShape/InferDataType 正确性：
    // 将上游 Data 节点显式设置的输出 desc 传播到本算子输入（不依赖 AddGraph 的内部处理时序），
    // 再显式调用 InferShapeAndType —— 经 OperatorFactory V2 桥接调用安装目录 ophost 库中
    // IMPL_OP_INFERSHAPE 注册的推导函数，断言直通语义（输出 shape/dtype 与输入一致）。
    ge::Operator data_op;
    if (graph.FindOpByName("data_node", data_op) != ge::GRAPH_SUCCESS) {
        printf("%s - ERROR - [XIR]: CheckInferShape: FindOpByName(data_node) failed\n", GetTime().c_str());
        return false;
    }
    ge::Operator ref_exit_op;
    if (graph.FindOpByName("ref_exit_node", ref_exit_op) != ge::GRAPH_SUCCESS) {
        printf("%s - ERROR - [XIR]: CheckInferShape: FindOpByName(ref_exit_node) failed\n", GetTime().c_str());
        return false;
    }
    (void)ref_exit_op.UpdateInputDesc("x", data_op.GetOutputDesc(0));
    if (ref_exit_op.InferShapeAndType() != ge::GRAPH_SUCCESS) {
        printf("%s - ERROR - [XIR]: InferShapeAndType failed\n", GetTime().c_str());
        return false;
    }
    auto output_desc = ref_exit_op.GetOutputDesc(0);
    auto input_desc = ref_exit_op.GetInputDesc(0);
    auto shape = output_desc.GetShape();
    auto in_shape = input_desc.GetShape();
    printf("%s - INFO - [XIR]: InferShape: dim_num=%zu", GetTime().c_str(), shape.GetDimNum());
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        printf(" dim%zu=%ld", i, shape.GetDim(i));
    }
    printf(" dtype=%d | input: dim_num=%zu", output_desc.GetDataType(), in_shape.GetDimNum());
    for (size_t i = 0; i < in_shape.GetDimNum(); i++) {
        printf(" dim%zu=%ld", i, in_shape.GetDim(i));
    }
    printf("\n");

    if (shape.GetDims() != in_shape.GetDims()) {
        printf("%s - ERROR - [XIR]: InferShape FAILED: output dims != input dims (pass-through violated)\n",
               GetTime().c_str());
        return false;
    }
    if (output_desc.GetDataType() != ge::DT_FLOAT) {
        printf("%s - ERROR - [XIR]: InferDataType FAILED: expected=%d actual=%d\n", GetTime().c_str(), ge::DT_FLOAT,
               output_desc.GetDataType());
        return false;
    }
    return true;
}

struct RunContext {
    const map<ge::AscendString, ge::AscendString>& session_options;
    uint32_t graph_id;
};

// RefExit 为控制流标记算子：历史交付无 kernel（老芯片 ini 中 opFile.value=Null），
// 仅在完整 while 循环结构（Enter/Merge/Switch/NextIteration/LoopCond）中被图优化重构处理，
// 孤立构图（Data -> RefExit -> output）无执行语义，RunGraph/BuildModel 不作为判定标准。
// 本 example 判定标准：1) AddGraph 成功（proto 注册/加载链路 + 图预处理）；
// 2) CheckInferShape 显式断言（RT2.0 InferShape/InferDataType 交付件核心）。
bool RunScenario(const string& tag, ge::Graph (*builder)(vector<ge::Tensor>&), RunContext& ctx, const string& om_name)
{
    printf("%s - INFO - [XIR]: === %s ===\n", GetTime().c_str(), tag.c_str());

    vector<ge::Tensor> input;
    ge::Graph graph = builder(input);

    ge::Session* session = new ge::Session(ctx.session_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create session failed\n", GetTime().c_str());
        return false;
    }
    if (session->AddGraph(ctx.graph_id, graph, ctx.session_options) != SUCCESS) {
        printf("%s - ERROR - [XIR]: AddGraph failed\n", GetTime().c_str());
        delete session;
        return false;
    }
    printf("%s - INFO - [XIR]: AddGraph success\n", GetTime().c_str());
    delete session;

    if (!CheckInferShape(graph)) {
        return false;
    }
    printf("%s - INFO - [XIR]: InferShape/InferDataType verification success (pass-through: output == input)\n",
           GetTime().c_str());

    // 信息性输出：RunGraph（控制流标记算子孤立构图预期不可执行，仅记录结果，不作为判定）
    {
        ge::Session* run_session = new ge::Session(ctx.session_options);
        if (run_session != nullptr && run_session->AddGraph(ctx.graph_id, graph, ctx.session_options) == SUCCESS) {
            vector<ge::Tensor> output;
            ge::Status ret = run_session->RunGraph(ctx.graph_id, input, output);
            if (ret == SUCCESS && !output.empty()) {
                auto out_shape = output[0].GetTensorDesc().GetShape();
                printf(
                    "%s - INFO - [XIR]: RunGraph success (informational), output dim_num=%zu shape_size=%ld dtype=%d\n",
                    GetTime().c_str(), out_shape.GetDimNum(), out_shape.GetShapeSize(),
                    output[0].GetTensorDesc().GetDataType());
            } else {
                printf("%s - INFO - [XIR]: RunGraph not executable for standalone RefExit "
                       "(informational, expected: control-flow marker op without kernel requires complete "
                       "while-loop structure), ret=%u output_count=%zu\n",
                       GetTime().c_str(), ret, output.size());
            }
        }
        delete run_session;
    }

    // 信息性输出：BuildModel（同因可能失败，仅记录结果，不作为判定）
    {
        vector<ge::Tensor> input_om;
        ge::Graph graph_om = builder(input_om);
        ge::ModelBufferData model_buffer;
        ge::Status ret = aclgrphBuildModel(graph_om, ctx.session_options, model_buffer);
        if (ret == SUCCESS) {
            aclgrphSaveModel(om_name.c_str(), model_buffer);
            printf("%s - INFO - [XIR]: BuildModel & Save OM success (informational): %s\n", GetTime().c_str(),
                   om_name.c_str());
        } else {
            printf("%s - INFO - [XIR]: BuildModel not successful for standalone RefExit (informational, "
                   "not a failure criterion)\n",
                   GetTime().c_str());
        }
    }

    printf("%s - INFO - [XIR]: %s PASSED (verified: AddGraph + InferShape/InferDataType; "
           "RunGraph/BuildModel informational for control-flow marker op)\n",
           GetTime().c_str(), tag.c_str());
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

    if (!RunScenario("static", BuildGraph, ctx, "./ref_exit_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("dynamic(-1)", BuildDynamicShapeGraphDimNeg1, ctx, "./ref_exit_dynamic_neg1_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("dynamic(-2)", BuildDynamicShapeGraphDimNeg2, ctx, "./ref_exit_dynamic_neg2_model.om")) {
        ge::GEFinalize();
        return FAILED;
    }

    ge::GEFinalize();
    printf("%s - INFO - [XIR]: Finalize success\n", GetTime().c_str());
    return SUCCESS;
}
