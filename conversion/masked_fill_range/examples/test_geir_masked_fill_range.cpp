/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "assert.h"

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "experiment_ops.h"
#include "nn_other.h"
#include "../op_graph/masked_fill_range_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;
#define ADD_INPUT(inputIndex, inputName, inputDtype, inputShape, value)                                           \
    vector<int64_t> placeholder##inputIndex##_shape = inputShape;                                                 \
    auto placeholder##inputIndex = op::Data("placeholder" + inputIndex).set_attr_index(0);                        \
    TensorDesc placeholder##inputIndex##_desc = TensorDesc(ge::Shape(placeholder##inputIndex##_shape), FORMAT_ND, \
                                                           inputDtype);                                           \
    placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                              \
    placeholder##inputIndex##_desc.SetFormat(FORMAT_ND);                                                          \
    Tensor tensor_placeholder##inputIndex;                                                                        \
    ret = GenInputData(placeholder##inputIndex##_shape, tensor_placeholder##inputIndex,                           \
                       placeholder##inputIndex##_desc, inputDtype, value);                                        \
    if (ret != SUCCESS) {                                                                                         \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                            \
        return FAILED;                                                                                            \
    }                                                                                                             \
    placeholder##inputIndex.update_input_desc_x(placeholder##inputIndex##_desc);                                  \
    input.push_back(tensor_placeholder##inputIndex);                                                              \
    graph.AddOp(placeholder##inputIndex);                                                                         \
    maskedFillRange1.set_input_##inputName(placeholder##inputIndex);                                              \
    inputs.push_back(placeholder##inputIndex)

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                       \
    TensorDesc outputName##outputIndex##_desc = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
    maskedFillRange1.update_output_desc_##outputName(outputName##outputIndex##_desc)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

uint32_t GetDataTypeSize(DataType dt)
{
    uint32_t dilation = 1;
    uint32_t oneByte = 1;
    uint32_t twoByte = 2;
    uint32_t fourByte = 4;
    uint32_t eightByte = 8;

    if (dt == ge::DT_FLOAT) {
        dilation = fourByte;
    } else if (dt == ge::DT_FLOAT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_BF16) {
        dilation = twoByte;
    } else if (dt == ge::DT_INT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_UINT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_INT32) {
        dilation = fourByte;
    } else if (dt == ge::DT_UINT32) {
        dilation = fourByte;
    } else if (dt == ge::DT_INT64) {
        dilation = eightByte;
    } else if (dt == ge::DT_UINT64) {
        dilation = eightByte;
    } else if (dt == ge::DT_INT8) {
        dilation = oneByte;
    } else if (dt == ge::DT_UINT8) {
        dilation = oneByte;
    } else if (dt == ge::DT_BOOL) {
        dilation = oneByte;
    } else if (dt == ge::DT_DOUBLE) {
        dilation = eightByte;
    } else if (dt == ge::DT_COMPLEX64) {
        dilation = eightByte;
    }
    return dilation;
}

// start/end 为 int32 窗口输入：按实际 dtype 生成，避免浮点位重解释产生越界窗口
int32_t GenInputData(const vector<int64_t>& shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc,
                     DataType data_type, int value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (size_t i = 0; i < shapes.size(); i++) {
        if (shapes[i] < 0 ||
            (shapes[i] != 0 && size > std::numeric_limits<size_t>::max() / static_cast<size_t>(shapes[i]))) {
            printf("%s - ERROR - [XIR]: Invalid or overflowing input shape\n", GetTime().c_str());
            return FAILED;
        }
        size *= static_cast<size_t>(shapes[i]);
    }
    const size_t element_size = GetDataTypeSize(data_type);
    if (element_size != 0 && size > std::numeric_limits<size_t>::max() / element_size) {
        printf("%s - ERROR - [XIR]: Input byte size overflow\n", GetTime().c_str());
        return FAILED;
    }
    const size_t data_len = size * element_size;
    // Tensor(desc, const uint8_t*, size) 重载对 const 缓冲必然深拷贝（构造后 Tensor
    // 自持数据），故用 vector 局部对象代替裸 new：出作用域自动释放，修复每次调用
    // 泄漏 size * sizeof(T) 的问题
    if (data_type == ge::DT_INT32) {
        std::vector<int32_t> pData(size, static_cast<int32_t>(value));
        input_tensor = Tensor(input_tensor_desc, reinterpret_cast<const uint8_t*>(pData.data()), data_len);
    } else {
        std::vector<float> pData(size, static_cast<float>(value));
        input_tensor = Tensor(input_tensor_desc, reinterpret_cast<const uint8_t*>(pData.data()), data_len);
    }
    return SUCCESS;
}

int32_t WriteDataToFile(const string& bin_file, uint64_t data_size, uint8_t* inputData)
{
    FILE* fp = fopen(bin_file.c_str(), "wb");
    if (fp == nullptr) {
        printf("%s - ERROR - [XIR]: Open file %s failed\n", GetTime().c_str(), bin_file.c_str());
        return FAILED;
    }
    size_t written = fwrite(inputData, sizeof(uint8_t), data_size, fp);
    if (written != static_cast<size_t>(data_size)) {
        printf("%s - ERROR - [XIR]: Write file %s incomplete: %lu/%lu\n", GetTime().c_str(), bin_file.c_str(),
               static_cast<unsigned long>(written), static_cast<unsigned long>(data_size));
        fclose(fp);
        return FAILED;
    }
    fclose(fp);
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    // 自定义代码：添加单算子定义到图中
    auto maskedFillRange1 = op::MaskedFillRange("maskedFillRange1");
    std::vector<int64_t> xShape = {4, 8};
    ADD_INPUT(1, x, inDtype, xShape, 1);
    // 2 轮填充，窗口批数 4 与 x 行数一致：每轮各行均取 [2, 6) 窗口
    std::vector<int64_t> startShape = {2, 4};
    ADD_INPUT(2, start, DT_INT32, startShape, 2);
    std::vector<int64_t> endShape = {2, 4};
    ADD_INPUT(3, end, DT_INT32, endShape, 6);
    std::vector<int64_t> valueShape = {2};
    ADD_INPUT(4, value, inDtype, valueShape, 10);
    std::vector<int64_t> yShape = {4, 8};
    ADD_OUTPUT(1, y, inDtype, yShape);
    maskedFillRange1.set_attr_axis(-1);
    outputs.push_back(maskedFillRange1);
    // 添加完毕
    return SUCCESS;
}

int ExecuteGraph(Graph& graph, std::vector<ge::Tensor>& input, std::vector<ge::Tensor>& output)
{
    // jit_compile=0：直接匹配 OPP 预编 binary（binList），不走在线编译 fallback
    // （自定义算子在线编译产物 kernel-store 注册链在 950 上未打通，须预编匹配）
    std::map<AscendString, AscendString> build_options = {{"ge.jit_compile", "0"}};
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    std::unique_ptr<Session> session(new Session(build_options));
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    Status ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: AddGraph failed\n", GetTime().c_str());
        session.reset();
        GEFinalize();
        return FAILED;
    }

    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());
    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());
    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        ge::AscendString errMsg = ge::GEGetErrorMsgV2();
        printf("%s - ERROR - [XIR]: RunGraph failed: %s\n", GetTime().c_str(), errMsg.GetString());
        ge::AscendString warnMsg = ge::GEGetWarningMsgV2();
        printf("%s - ERROR - [XIR]: RunGraph warning: %s\n", GetTime().c_str(), warnMsg.GetString());
        session.reset();
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    return SUCCESS;
}

void ProcessIOData(std::vector<ge::Tensor>& input, std::vector<ge::Tensor>& output)
{
    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        std::cout << "input " << i << " dtype :  " << input[i].GetTensorDesc().GetDataType() << std::endl;
        string input_file = "./tc_ge_irrun_test_0008_npu_input_" + std::to_string(i) + ".bin";
        uint8_t* input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th input, input shape size =" << input_shape << std::endl;
        uint32_t data_size = input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        WriteDataToFile(input_file, data_size, input_data_i);
    }

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << output_shape << std::endl;
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile(output_file, data_size, output_data_i);
        // x=(4,8) 全 1，每轮各行 [2,6) 窗口用 value=10 覆盖：期望每行 [1, 1, 10, 10, 10, 10, 1, 1]
        if (output[i].GetTensorDesc().GetDataType() == DT_FLOAT) {
            const float* result = reinterpret_cast<const float*>(output_data_i);
            for (int64_t j = 0; j < output_shape; j++) {
                LOG_PRINT("result[%ld] is: %f\n", j, result[j]);
            }
        } else {
            std::cout << "skip float value printing for non-float output" << std::endl;
        }
    }
}

int main(int argc, char* argv[])
{
    const char* graph_name = "tc_ge_irrun_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    // argv[1] 无功能用途，仅可选回显：无参运行时跳过（修复 argc<2 越界读）
    if (argc > 1) {
        std::cout << argv[1] << std::endl;
    }

    DataType inDtype = DT_FLOAT;
    std::cout << inDtype << std::endl;

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }
    // 执行图计算
    std::vector<ge::Tensor> output;
    if (ExecuteGraph(graph, input, output) != SUCCESS) {
        return FAILED;
    }

    // 处理输入输出数据
    ProcessIOData(input, output);

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    std::cout << "Error message: " << error_str << std::endl;
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    std::cout << "Warning message: " << warning_str << std::endl;
    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
