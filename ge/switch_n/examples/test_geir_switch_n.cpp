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
 * \file test_geir_switch_n.cpp
 * \brief Graph construction and AddGraph validation for SwitchN.
 * 参考SwitchByIndex示例，仅验证构图、节点连接结构和AddGraph注册。
 * 形状和类型推导由独立UT覆盖，本示例不验证是否触发V2推导回调。
 * 不进行模型编译、模型保存或图执行，不校验输出数据、分支选择或索引数值。
 * 原引擎分配失败的问题尚未闭环，AddGraph成功不代表该问题已修复。
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
#include "../op_graph/switch_n_proto.h"
#include "../../data/op_graph/data_proto.h"

using namespace ge;

namespace {
constexpr int kFailed = -1;
constexpr int kSuccess = 0;
constexpr uint32_t kDeviceId = 0;
constexpr DataType kDtype = DT_FLOAT;
constexpr size_t kOutputNum = 3U;

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

    TensorDesc pred_desc(Shape(std::vector<int64_t>{}), FORMAT_ND, DT_INT64);
    pred_desc.SetPlacement(kPlacementHost);
    auto pred = op::Data("pred_value").set_attr_index(1);
    pred.update_input_desc_x(pred_desc);
    pred.update_output_desc_y(pred_desc);

    auto swn = op::SwitchN("switch_n_op");
    swn.set_input_data(data);
    swn.set_input_pred_value(pred);
    swn.create_dynamic_output_output(kOutputNum);

    graph.AddOp(data);
    graph.AddOp(pred);
    graph.AddOp(swn);
    graph_inputs = {data, pred};
    graph_output = swn;
    return kSuccess;
}

int CheckGraphStructure(const Graph& graph)
{
    Operator node;
    if (graph.FindOpByName("switch_n_op", node) != GRAPH_SUCCESS || node.GetInputsSize() != 2U ||
        node.GetOutputsSize() != 3U) {
        std::cerr << "CheckGraphStructure: target node or input/output count mismatch" << std::endl;
        return kFailed;
    }
    return kSuccess;
}

int RunScenario(RunContext& ctx, const std::string& name, const std::vector<int64_t>& shape, bool is_dynamic,
                int64_t output_index)
{
    std::cout << "=== Scenario: " << name << " ===" << std::endl;
    Graph graph(("switch_n_" + name).c_str());
    std::vector<Operator> graph_inputs;
    Operator graph_output;
    if (BuildGraph(graph, shape, graph_inputs, graph_output, is_dynamic) != kSuccess) {
        std::cerr << name << " FAILED: BuildGraph" << std::endl;
        return kFailed;
    }
    graph.SetInputs(graph_inputs);
    // 此处仅选择图输出端口，不验证运行时分支选择。
    graph.SetOutputs(
        std::vector<std::pair<Operator, std::vector<size_t>>>{{graph_output, {static_cast<size_t>(output_index)}}});

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
    for (size_t output = 0; output < kOutputNum; ++output) {
        const auto output_index = static_cast<int64_t>(output);
        const auto suffix = "_output" + std::to_string(output);
        if (RunScenario(ctx, "static" + suffix, {2, 3}, false, output_index) != kSuccess) {
            ret = kFailed;
        }
        if (RunScenario(ctx, "dynamic_minus1" + suffix, {-1, 3}, true, output_index) != kSuccess) {
            ret = kFailed;
        }
        if (RunScenario(ctx, "dynamic_minus2" + suffix, {-2}, true, output_index) != kSuccess) {
            ret = kFailed;
        }
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
