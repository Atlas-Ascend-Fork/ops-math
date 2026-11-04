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
 * \file test_geir_merge.cpp
 * \brief Graph construction and AddGraph validation for Merge.
 * 参考SwitchByIndex示例，仅验证构图、节点连接结构和AddGraph注册。
 * 形状和类型推导由独立UT覆盖，本示例不验证是否触发V2推导回调。
 * 不进行模型编译、模型保存或图执行，不校验输出数据、分支选择或索引数值。
 * 原运行时索引不符合预期的问题尚未闭环，本示例不覆盖该问题。
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
#include "../op_graph/merge_proto.h"
#include "../../data/op_graph/data_proto.h"
#include "../../switch/op_graph/switch_proto.h"
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

    auto sw = op::Switch("merge_input_switch");
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

    auto merge = op::Merge("merge_op");
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

int CheckGraphStructure(const Graph& graph)
{
    Operator node;
    if (graph.FindOpByName("merge_op", node) != GRAPH_SUCCESS || node.GetInputsSize() != 2U ||
        node.GetOutputsSize() != 2U) {
        std::cerr << "CheckGraphStructure: target node or input/output count mismatch" << std::endl;
        return kFailed;
    }
    return kSuccess;
}

int RunScenario(RunContext& ctx, const std::string& name, const std::vector<int64_t>& shape, bool is_dynamic)
{
    std::cout << "=== Scenario: " << name << " ===" << std::endl;
    Graph graph(("merge_" + name).c_str());
    std::vector<Operator> graph_inputs;
    Operator graph_output;
    if (BuildGraph(graph, shape, graph_inputs, graph_output, is_dynamic) != kSuccess) {
        std::cerr << name << " FAILED: BuildGraph" << std::endl;
        return kFailed;
    }
    graph.SetInputs(graph_inputs).SetOutputs({graph_output});

    if (CheckGraphStructure(graph) != kSuccess) {
        std::cerr << name << " FAILED: CheckGraphStructure" << std::endl;
        return kFailed;
    }

    Session session(ctx.session_options);
    const uint32_t gid = ctx.graph_id++;
    if (session.AddGraph(gid, graph) != SUCCESS) {
        std::cerr << name << " FAILED: AddGraph" << std::endl;
        return kFailed;
    }
    std::cout << name << ": AddGraph success" << std::endl;

    // 仅报告构图和注册成功，不将其解释为推导、图编译或执行成功。
    std::cout << name << " ADD_GRAPH_ONLY PASSED" << std::endl;
    return kSuccess;
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

    std::cout << "MODE=ADD_GRAPH_ONLY; graph compilation/execution and V2 callback invocation are not verified."
              << std::endl;
    RunContext ctx;
    int ret = kSuccess;
    if (RunScenario(ctx, "static", {2, 3}, false) != kSuccess) {
        ret = kFailed;
    }
    if (RunScenario(ctx, "dynamic_minus1", {-1, 3}, true) != kSuccess) {
        ret = kFailed;
    }
    if (RunScenario(ctx, "dynamic_minus2", {-2}, true) != kSuccess) {
        ret = kFailed;
    }

    GEFinalize();
    if (ret == kSuccess) {
        std::cout << "All graph construction/AddGraph scenarios PASSED; inference, compilation and execution were not "
                     "checked."
                  << std::endl;
    } else {
        std::cout << "Some scenarios FAILED" << std::endl;
    }
    return ret;
}
