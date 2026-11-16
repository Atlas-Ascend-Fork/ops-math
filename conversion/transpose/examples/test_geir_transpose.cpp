/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <cstring>
#include <ctime>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../op_graph/transpose_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

namespace ge {
REG_OP(Data).INPUT(x, TensorType::ALL()).OUTPUT(y, TensorType::ALL()).ATTR(index, Int, 0).OP_END_FACTORY_REG(Data);
REG_OP(Const).OUTPUT(y, TensorType::ALL()).ATTR(value, Tensor, Tensor()).OP_END_FACTORY_REG(Const);
} // namespace ge

namespace {
string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64] = {0};
    struct tm tm_info;
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime_r(&timep, &tm_info));
    return tmp;
}

int32_t WriteDataToFile(const string& bin_file, uint64_t data_size, const uint8_t* input_data)
{
    FILE* fp = fopen(bin_file.c_str(), "wb");
    if (fp == nullptr) {
        printf("Failed to open file: %s\n", bin_file.c_str());
        return FAILED;
    }
    size_t written = fwrite(input_data, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    if (written != data_size) {
        printf("Failed to write file: %s\n", bin_file.c_str());
        return FAILED;
    }
    return SUCCESS;
}

template <typename T>
ge::Tensor CreateTensor(vector<int64_t> shape, ge::DataType d_type, const vector<T>& values)
{
    TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, d_type);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetRealDimCnt(shape.size());
    size_t size = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        size *= static_cast<size_t>(shape[i]);
    }
    if (values.size() != size) {
        printf("%s - ERROR - [XIR]: Generate input tensor failed\n", GetTime().c_str());
        return ge::Tensor{};
    }

    return ge::Tensor(desc, reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(T));
}

ge::Operator CreateInputNode(size_t i, TensorDesc desc)
{
    auto node = op::Data("placeholder" + std::to_string(i)).set_attr_index(i);
    node.update_input_desc_x(desc);
    node.update_output_desc_y(desc);
    return node;
}

// perm 为数据依赖输入，形状推导阶段需要其取值，因此用 Const 节点提供。
ge::Operator CreateConstNode(const string& name, TensorDesc desc, const vector<int32_t>& values)
{
    auto node = op::Const(name);
    ge::Tensor tensor(desc, reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(int32_t));
    node.SetAttr("value", tensor);
    node.update_output_desc_y(desc);
    return node;
}

int CreateOppInGraph(vector<ge::Tensor>& input, vector<Operator>& inputs, vector<Operator>& outputs, Graph& graph)
{
    ge::DataType d_type = ge::DT_FLOAT;
    auto node_transpose = op::Transpose("transpose");

    vector<int64_t> x_shape = {2, 3};
    vector<int64_t> perm_shape = {2};
    vector<int64_t> y_shape = {3, 2};

    auto tensor_x = CreateTensor<float>(x_shape, d_type, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
    if (tensor_x.GetSize() == 0) {
        return FAILED;
    }
    auto node_x = CreateInputNode(0, tensor_x.GetTensorDesc());
    input.push_back(tensor_x);
    graph.AddOp(node_x);
    inputs.push_back(node_x);
    node_transpose.set_input_x(node_x);
    node_transpose.update_input_desc_x(tensor_x.GetTensorDesc());

    TensorDesc perm_desc(ge::Shape(perm_shape), ge::FORMAT_ND, ge::DT_INT32);
    perm_desc.SetPlacement(ge::kPlacementHost);
    perm_desc.SetRealDimCnt(perm_shape.size());
    auto node_perm = CreateConstNode("perm", perm_desc, {1, 0});
    graph.AddOp(node_perm);
    node_transpose.set_input_perm(node_perm);
    node_transpose.update_input_desc_perm(perm_desc);

    TensorDesc y_desc(ge::Shape(y_shape), ge::FORMAT_ND, d_type);
    node_transpose.update_output_desc_y(y_desc);
    outputs.push_back(node_transpose);
    return SUCCESS;
}

int SaveInput(const vector<ge::Tensor>& input)
{
    for (size_t i = 0; i < input.size(); ++i) {
        string input_file = "./tc_ge_irrun_example_npu_input_" + std::to_string(i) + ".bin";
        const uint8_t* input_data = input[i].GetData();
        int64_t shape_size = input[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t data_size = static_cast<uint32_t>(shape_size) *
                             ge::GetSizeByDataType(input[i].GetTensorDesc().GetDataType());
        if (WriteDataToFile(input_file, data_size, input_data) != SUCCESS) {
            return FAILED;
        }
    }
    return SUCCESS;
}

int SaveOutput(const vector<ge::Tensor>& output)
{
    for (size_t i = 0; i < output.size(); ++i) {
        string output_file = "./tc_ge_irrun_example_npu_output_" + std::to_string(i) + ".bin";
        const uint8_t* output_data = output[i].GetData();
        int64_t shape_size = output[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t data_size = static_cast<uint32_t>(shape_size) *
                             ge::GetSizeByDataType(output[i].GetTensorDesc().GetDataType());
        if (WriteDataToFile(output_file, data_size, output_data) != SUCCESS) {
            return FAILED;
        }

        const float* result = reinterpret_cast<const float*>(output_data);
        for (int64_t j = 0; j < shape_size; j++) {
            LOG_PRINT("result[%ld] is: %f\n", j, result[j]);
        }
    }
    return SUCCESS;
}
} // namespace

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    const char* graph_name = "tc_ge_irrun_transpose";
    Graph graph(graph_name);
    vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }

    vector<Operator> inputs;
    vector<Operator> outputs;
    ret = CreateOppInGraph(input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    map<AscendString, AscendString> build_options;
    std::unique_ptr<ge::Session> session(new (std::nothrow) Session(build_options));
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }

    map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }

    vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Run graph success\n", GetTime().c_str());

    ret = SaveInput(input);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Save graph input failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }

    ret = SaveOutput(output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Save graph output failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }

    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: GE Finalize failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Run graph over\n", GetTime().c_str());
    return SUCCESS;
}
