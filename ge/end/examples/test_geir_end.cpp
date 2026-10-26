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
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <map>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"

#include "elewise_calculation_ops.h"
#include "../../data/op_graph/data_proto.h"
#include "../op_graph/end_proto.h"
#include "../../placeholder/op_graph/placeholder_proto.h"

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

// 配对构图（End→PlaceHolder，peerIndex 匹配）为引擎切分边界的真实语义；
// 独立 End 节点会使编译在 EnginePartitioner 内崩溃，配对后 AddGraph 成功且 RunGraph 优雅失败
static ge::Graph BuildGraphImpl(const string& graph_name, const vector<int64_t>& x_shape, bool negativeInitDesc = false)
{
    ge::Graph graph(graph_name.c_str());
    ge::DataType inDtype = ge::DT_FLOAT;

    auto data_node = op::Data("data_node");
    data_node.set_attr_index(0);
    ge::TensorDesc data_desc = ge::TensorDesc(ge::Shape(x_shape), ge::FORMAT_ND, inDtype);
    data_desc.SetPlacement(ge::kPlacementHost);
    data_desc.SetFormat(ge::FORMAT_ND);
    if (x_shape[0] == -1) {
        data_desc.SetShapeRange({{1, 10}, {3, 3}});
    }
    data_node.update_input_desc_x(data_desc);
    data_node.update_output_desc_y(data_desc);

    auto end_node = op::End("end_node");
    end_node.set_attr_peerIndex(0);
    end_node.set_attr_parentOpType("");
    end_node.set_input_x(data_node);
    end_node.update_input_desc_x(data_desc);
    // 判别场景：y 初值故意设错（shape+dtype 双错），读回正确值只能来自推导覆盖而非初值回声；
    // 已实证边两侧不一致不阻塞 AddGraph，若后续版本增加该校验可降级为仅设错 shape（{9,9}+DT_FLOAT）
    ge::TensorDesc end_init_desc = data_desc;
    if (negativeInitDesc) {
        end_init_desc = ge::TensorDesc(ge::Shape({9, 9}), ge::FORMAT_ND, ge::DT_INT32);
    }
    end_node.update_output_desc_y(end_init_desc);

    auto placeholder_node = op::PlaceHolder("placeholder_node");
    placeholder_node.set_attr_peerIndex(0);
    placeholder_node.set_attr_parentId("");
    placeholder_node.set_attr_parentOpType("");
    placeholder_node.set_attr_anchorIndex(0);
    placeholder_node.set_input_x(end_node);
    placeholder_node.update_input_desc_x(data_desc);
    placeholder_node.update_output_desc_y(data_desc);

    auto add_node = op::Add("add_node");
    add_node.set_input_x1(placeholder_node);
    add_node.set_input_x2(placeholder_node);
    add_node.update_input_desc_x1(data_desc);
    add_node.update_input_desc_x2(data_desc);
    add_node.update_output_desc_y(data_desc);

    std::vector<ge::Operator> inputs{data_node};
    std::vector<ge::Operator> outputs{add_node};
    graph.SetInputs(inputs).SetOutputs(outputs);
    graph.AddOp(data_node);
    graph.AddOp(end_node);
    graph.AddOp(placeholder_node);
    graph.AddOp(add_node);
    return graph;
}

ge::Graph BuildGraph(std::vector<ge::Tensor>& input)
{
    ge::Graph graph = BuildGraphImpl("test_end_ir", {2, 3});

    float pData[6] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    ge::TensorDesc real_desc = ge::TensorDesc(ge::Shape({2, 3}), ge::FORMAT_ND, ge::DT_FLOAT);
    ge::Tensor tensor_data(real_desc, (uint8_t*)pData, 24);
    input.push_back(tensor_data);
    return graph;
}

ge::Graph BuildDynamicShapeGraphDimNeg1(std::vector<ge::Tensor>& input)
{
    ge::Graph graph = BuildGraphImpl("test_end_dynamic_dim_neg1", {-1, 3});

    float pData[6] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    ge::TensorDesc real_desc = ge::TensorDesc(ge::Shape({2, 3}), ge::FORMAT_ND, ge::DT_FLOAT);
    ge::Tensor tensor_data(real_desc, (uint8_t*)pData, 24);
    input.push_back(tensor_data);
    return graph;
}

ge::Graph BuildDynamicShapeGraphDimNeg2(std::vector<ge::Tensor>& input)
{
    ge::Graph graph = BuildGraphImpl("test_end_dynamic_dim_neg2", {-2});

    float pData[6] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    ge::TensorDesc real_desc = ge::TensorDesc(ge::Shape({2, 3}), ge::FORMAT_ND, ge::DT_FLOAT);
    ge::Tensor tensor_data(real_desc, (uint8_t*)pData, 24);
    input.push_back(tensor_data);
    return graph;
}

ge::Graph BuildNegativeInitialDescGraph(std::vector<ge::Tensor>& input)
{
    ge::Graph graph = BuildGraphImpl("test_end_negative_init", {2, 3}, true);

    float pData[6] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    ge::TensorDesc real_desc = ge::TensorDesc(ge::Shape({2, 3}), ge::FORMAT_ND, ge::DT_FLOAT);
    ge::Tensor tensor_data(real_desc, (uint8_t*)pData, 24);
    input.push_back(tensor_data);
    return graph;
}

struct RunContext {
    const map<ge::AscendString, ge::AscendString>& session_options;
    uint32_t graph_id;
};

struct ExpectedResult {
    vector<int64_t> shape;
    bool unknownRank;
};

bool CheckInferShape(const ge::Graph& graph, const ExpectedResult& expected)
{
    ge::Operator op_node;
    if (graph.FindOpByName("end_node", op_node) != ge::GRAPH_SUCCESS) {
        printf("%s - ERROR - [XIR]: FindOpByName end_node failed\n", GetTime().c_str());
        return false;
    }
    // 显式触发推导：经 CallInferFunc 桥接 V2 注册表，验证 ops-math 的 RT2.0 注册（RT1.0 未迁移）
    ge::Status ret = op_node.InferShapeAndType();
    if (ret != ge::GRAPH_SUCCESS) {
        printf("%s - ERROR - [XIR]: InferShapeAndType failed, ret=%u\n", GetTime().c_str(), ret);
        return false;
    }
    auto output_desc = op_node.GetOutputDesc(0);
    auto shape = output_desc.GetShape();
    printf("%s - INFO - [XIR]: InferShape: dim_num=%zu", GetTime().c_str(), shape.GetDimNum());
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        printf(" dim%zu=%ld", i, shape.GetDim(i));
    }
    printf(" shape_size=%ld dtype=%d | expected: ", shape.GetShapeSize(), output_desc.GetDataType());
    if (expected.unknownRank) {
        printf("unknown rank (dim_num=0 or dim0=-2)");
    } else {
        printf("dim_num=%zu", expected.shape.size());
        for (size_t i = 0; i < expected.shape.size(); i++) {
            printf(" dim%zu=%ld", i, expected.shape[i]);
        }
    }
    printf(" dtype=%d\n", ge::DT_FLOAT);

    if (expected.unknownRank) {
        // 未知秩输入可能被 GE 规范化为 dim_num=0 或保留 {-2}，两种均证明透传，避免锚定规范化细节
        bool rankOk = (shape.GetDimNum() == 0U) || (shape.GetDimNum() == 1U && shape.GetDim(0) == -2);
        if (!rankOk) {
            printf("%s - ERROR - [XIR]: InferShape FAILED: unknown rank expected (dim_num=0 or dim0=-2), actual "
                   "dim_num=%zu\n",
                   GetTime().c_str(), shape.GetDimNum());
            return false;
        }
    } else {
        if (shape.GetDimNum() != expected.shape.size()) {
            printf("%s - ERROR - [XIR]: InferShape FAILED: expected dim_num=%zu\n", GetTime().c_str(),
                   expected.shape.size());
            return false;
        }
        for (size_t i = 0; i < expected.shape.size(); i++) {
            if (shape.GetDim(i) != expected.shape[i]) {
                printf("%s - ERROR - [XIR]: InferShape FAILED: dim%zu expected=%ld actual=%ld\n", GetTime().c_str(), i,
                       expected.shape[i], shape.GetDim(i));
                return false;
            }
        }
    }
    if (output_desc.GetDataType() != ge::DT_FLOAT) {
        printf("%s - ERROR - [XIR]: InferDataType FAILED: expected=%d actual=%d\n", GetTime().c_str(), ge::DT_FLOAT,
               output_desc.GetDataType());
        return false;
    }
    return true;
}

bool RunScenario(const string& tag, ge::Graph (*builder)(vector<ge::Tensor>&), const ExpectedResult& expected,
                 RunContext& ctx)
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
    if (ret != ge::GRAPH_SUCCESS) {
        printf("%s - ERROR - [XIR]: AddGraph failed, ret=%u\n", GetTime().c_str(), ret);
        delete session;
        ctx.graph_id++;
        return false;
    }
    printf("%s - INFO - [XIR]: AddGraph success (End IR registered and accepted in paired graph)\n", GetTime().c_str());

    vector<ge::Tensor> output;
    ret = session->RunGraph(ctx.graph_id, input, output);
    if (ret == ge::GRAPH_SUCCESS) {
        printf("%s - INFO - [XIR]: RunGraph success, output count=%zu\n", GetTime().c_str(), output.size());
    } else {
        // 预期行为：内部节点用户图不可执行，但编译准备阶段已触发推导并回写 desc
        printf("%s - INFO - [XIR]: RunGraph not executable standalone (expected for engine-partition internal "
               "node)\n",
               GetTime().c_str());
    }
    delete session;
    ctx.graph_id++;

    if (!CheckInferShape(graph, expected)) {
        return false;
    }
    printf("%s - INFO - [XIR]: %s PASSED (End is a GE engine-partition internal node paired with PlaceHolder via "
           "peerIndex; user graphs cannot compile/run it standalone, so verified via AddGraph validation and "
           "InferShape/InferDataType assertions that output y copies input x)\n",
           GetTime().c_str(), tag.c_str());
    return true;
}

int main(int argc, char* argv[])
{
    printf("%s - INFO - [XIR]: Start to initialize ge\n", GetTime().c_str());
    std::map<ge::AscendString, ge::AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    ge::Status ret = ge::GEInitialize(global_options);
    if (ret != ge::GRAPH_SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge success\n", GetTime().c_str());

    std::map<ge::AscendString, ge::AscendString> session_options;
    RunContext ctx{session_options, 0};

    // 静态 {2,3}；动态 {-1,3} 透传未知维；{-2} 未知秩接受两种表示；negative 场景 y 初值设错 {9,9}+DT_INT32，
    // 断言推导修正为 {2,3}+DT_FLOAT
    ExpectedResult static_expected{{2, 3}, false};
    ExpectedResult dynamic_neg1_expected{{-1, 3}, false};
    ExpectedResult dynamic_neg2_expected{{}, true};
    ExpectedResult negative_expected{{2, 3}, false};

    if (!RunScenario("static", BuildGraph, static_expected, ctx)) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("dynamic(-1)", BuildDynamicShapeGraphDimNeg1, dynamic_neg1_expected, ctx)) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("dynamic(-2)", BuildDynamicShapeGraphDimNeg2, dynamic_neg2_expected, ctx)) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!RunScenario("negative-initial-desc", BuildNegativeInitialDescGraph, negative_expected, ctx)) {
        ge::GEFinalize();
        return FAILED;
    }

    ge::GEFinalize();
    printf("%s - INFO - [XIR]: Finalize success\n", GetTime().c_str());
    return SUCCESS;
}
