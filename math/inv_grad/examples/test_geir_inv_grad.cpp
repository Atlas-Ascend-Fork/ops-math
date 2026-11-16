/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <math.h>
#include <memory>
#include <string.h>
#include <cstring>
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

#include "array_ops.h"
#include "nn_other.h"
#include "../op_graph/inv_grad_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;
#define ADD_INPUT(intputIndex, intputName, intputDtype, inputShape)                                                 \
    vector<int64_t> placeholder##intputIndex##_shape = inputShape;                                                  \
    auto placeholder##intputIndex = op::Data("placeholder" + intputIndex).set_attr_index(0);                        \
    TensorDesc placeholder##intputIndex##_desc = TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, \
                                                            intputDtype);                                           \
    placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost);                                               \
    placeholder##intputIndex##_desc.SetFormat(FORMAT_ND);                                                           \
    Tensor tensor_placeholder##intputIndex;                                                                         \
    ret = GenOnesData(placeholder##intputIndex##_shape, tensor_placeholder##intputIndex,                            \
                      placeholder##intputIndex##_desc, intputDtype, 2);                                             \
    if (ret != SUCCESS) {                                                                                           \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                              \
        return FAILED;                                                                                              \
    }                                                                                                               \
    placeholder##intputIndex.update_input_desc_x(placeholder##intputIndex##_desc);                                  \
    input.push_back(tensor_placeholder##intputIndex);                                                               \
    graph.AddOp(placeholder##intputIndex);                                                                          \
    invGrad.set_input_##intputName(placeholder##intputIndex);                                                       \
    inputs.push_back(placeholder##intputIndex)

#define ADD_CONST_INPUT(intputIndex, intputName, intputDtype, inputShape)                                           \
    vector<int64_t> placeholder##intputIndex##_shape = inputShape;                                                  \
    auto placeholder##intputIndex = op::Const("placeholder" + intputIndex);                                         \
    TensorDesc placeholder##intputIndex##_desc = TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, \
                                                            intputDtype);                                           \
    placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost);                                               \
    placeholder##intputIndex##_desc.SetFormat(FORMAT_ND);                                                           \
    Tensor tensor_placeholder##intputIndex;                                                                         \
    ret = GenOnesData(placeholder##intputIndex##_shape, tensor_placeholder##intputIndex,                            \
                      placeholder##intputIndex##_desc, intputDtype, 2);                                             \
    if (ret != SUCCESS) {                                                                                           \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                              \
        return FAILED;                                                                                              \
    }                                                                                                               \
    placeholder##intputIndex.SetAttr("value", tensor_placeholder##intputIndex);                                     \
    placeholder##intputIndex.update_output_desc_y(placeholder##intputIndex##_desc);                                 \
    graph.AddOp(placeholder##intputIndex);                                                                          \
    invGrad.set_input_##intputName(placeholder##intputIndex);                                                       \
    invGrad.update_input_desc_##intputName(placeholder##intputIndex##_desc);                                        \
    inputs.push_back(placeholder##intputIndex)

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                       \
    TensorDesc outputName##outputIndex##_desc = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
    invGrad.update_output_desc_##outputName(outputName##outputIndex##_desc)

#define LOG_PRINT(message, ...) printf(message, ##__VA_ARGS__)

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
    }
    return dilation;
}

int32_t GenOnesDataFloat32(const std::vector<int64_t>& shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc,
                           float value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t byteSizeFloat32 = 4;
    uint32_t data_len = size * byteSizeFloat32;
    // Tensor(desc, data, size) 为拷贝构造，缓冲区所有权归调用方：
    // unique_ptr RAII 保证任何路径（含分配失败早退）都不泄漏
    std::unique_ptr<float[]> pData(new (std::nothrow) float[size]);
    if (pData == nullptr) {
        return FAILED;
    }
    for (size_t i = 0; i < size; ++i) {
        pData[i] = value;
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData.get()), data_len);
    return SUCCESS;
}

int32_t GenOnesData(const std::vector<int64_t>& shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc,
                    DataType data_type, int value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * GetDataTypeSize(data_type);
    // 按 dtype 写真实值：DT_FLOAT 写 float 位型，保证“已知输入”与张量 dtype 一致；
    // 分配失败先判空再写入并返回 FAILED，RAII 保证不泄漏
    if (data_type == ge::DT_FLOAT) {
        std::unique_ptr<float[]> floatData(new (std::nothrow) float[size]);
        if (floatData == nullptr) {
            return FAILED;
        }
        for (size_t i = 0; i < size; ++i) {
            floatData[i] = static_cast<float>(value);
        }
        input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(floatData.get()), data_len);
        return SUCCESS;
    }
    // 申请按元素数 size 计的 int32 数组（pData 只用前 size 个元素填值，
    // 余下字节由 Tensor 自身拷贝边界保证），不再按 data_len 字节误申请 4 倍内存
    std::unique_ptr<int32_t[]> pData(new (std::nothrow) int32_t[size]);
    if (pData == nullptr) {
        return FAILED;
    }
    for (uint32_t i = 0; i < size; ++i) {
        pData[i] = value;
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData.get()), data_len);
    return SUCCESS;
}

int32_t WriteDataToFile(const std::string& bin_file, uint64_t data_size, const uint8_t* inputData)
{
    if (data_size > 0 && inputData == nullptr) {
        printf("%s - ERROR - [XIR]: Write file %s failed, data buffer is null\n", GetTime().c_str(), bin_file.c_str());
        return FAILED;
    }
    FILE* fp = fopen(bin_file.c_str(), "w");
    if (fp == nullptr) {
        printf("%s - ERROR - [XIR]: Open file %s failed\n", GetTime().c_str(), bin_file.c_str());
        return FAILED;
    }
    size_t written = fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    if (written != static_cast<size_t>(data_size)) {
        printf("%s - ERROR - [XIR]: Write file %s incomplete, %zu of %lu bytes\n", GetTime().c_str(), bin_file.c_str(),
               written, static_cast<unsigned long>(data_size));
        return FAILED;
    }
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    // 自定义代码：添加单算子定义到图中
    auto invGrad = op::InvGrad("inv_grad");

    std::vector<int64_t> xShape = {4, 2};
    ADD_INPUT(1, x, inDtype, xShape);
    ADD_INPUT(2, grad, inDtype, xShape);

    ADD_OUTPUT(1, y, inDtype, xShape);

    outputs.push_back(invGrad);
    // 添加完毕
    return SUCCESS;
}

int InitializeAndSetupGraph(Graph& graph, std::vector<ge::Tensor>& input, DataType inDtype)
{
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

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    return SUCCESS;
}

int ExecuteGraph(Graph& graph, const std::vector<ge::Tensor>& input, std::vector<ge::Tensor>& output)
{
    std::map<AscendString, AscendString> build_options = {};
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    // Session 由 unique_ptr RAII 持有，任何路径（含失败早退）都不泄漏；失败路径在
    // GEFinalize 前显式释放，保持先析构 Session、再反初始化 GE 的顺序
    std::unique_ptr<ge::Session> session(new (std::nothrow) Session(build_options));

    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    Status ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Session add ir compute graph to ir session failed, ret=%d\n", GetTime().c_str(),
               ret);
        session.reset();
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());
    std::string file_path = "./dump";
    graphStatus dumpRet = aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());
    if (dumpRet != GRAPH_SUCCESS) {
        printf("%s - WARNING - [XIR]: Dump graph to txt failed, ret=%d (non-fatal)\n", GetTime().c_str(), dumpRet);
    }
    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());

    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        session.reset();
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    return SUCCESS;
}

int32_t ProcessIOData(const std::vector<ge::Tensor>& input, const std::vector<ge::Tensor>& output)
{
    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        std::cout << "input " << i << " dtype :  " << input[i].GetTensorDesc().GetDataType() << std::endl;
        string input_file = "./tc_ge_irrun_test_0008_npu_input_" + std::to_string(i) + ".bin";
        const uint8_t* input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th input, input shape size =" << input_shape << std::endl;
        uint32_t data_size = input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        if (WriteDataToFile(input_file, data_size, input_data_i) != SUCCESS) {
            return FAILED;
        }
    }

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";
        const uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << output_shape << std::endl;
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        if (WriteDataToFile(output_file, data_size, output_data_i) != SUCCESS) {
            return FAILED;
        }
        const uint8_t* resultData = output_data_i;
        for (int64_t j = 0; j < output_shape; j++) {
            LOG_PRINT("result[%ld] is: %u\n", j, resultData[j]);
        }
    }
    return SUCCESS;
}

int32_t ValidateOutputValues(const std::vector<ge::Tensor>& output, DataType expectedDtype,
                             const std::vector<int64_t>& expectedShape, float expectedValue)
{
    if (output.size() != 1) {
        printf("%s - ERROR - [XIR]: Validate output failed, output tensor count is %zu, expected 1\n",
               GetTime().c_str(), output.size());
        return FAILED;
    }
    const ge::TensorDesc& outputDesc = output[0].GetTensorDesc();
    if (outputDesc.GetDataType() != expectedDtype) {
        printf("%s - ERROR - [XIR]: Validate output failed, output dtype is %d, expected %d\n", GetTime().c_str(),
               static_cast<int>(outputDesc.GetDataType()), static_cast<int>(expectedDtype));
        return FAILED;
    }
    if (outputDesc.GetShape().GetDimNum() != expectedShape.size()) {
        printf("%s - ERROR - [XIR]: Validate output failed, output dim num is %zu, expected %zu\n", GetTime().c_str(),
               outputDesc.GetShape().GetDimNum(), expectedShape.size());
        return FAILED;
    }
    for (size_t i = 0; i < expectedShape.size(); i++) {
        if (outputDesc.GetShape().GetDim(i) != expectedShape[i]) {
            printf("%s - ERROR - [XIR]: Validate output failed, output dim[%zu] is %ld, expected %ld\n",
                   GetTime().c_str(), i, outputDesc.GetShape().GetDim(i), expectedShape[i]);
            return FAILED;
        }
    }
    const int64_t elementNum = outputDesc.GetShape().GetShapeSize();
    const uint8_t* outputData = output[0].GetData();
    if (elementNum > 0 && outputData == nullptr) {
        printf("%s - ERROR - [XIR]: Validate output failed, output data is null\n", GetTime().c_str());
        return FAILED;
    }
    for (int64_t j = 0; j < elementNum; j++) {
        // 按位取值（与 inv_grad_geir_test_common.h 的按位视图一致）：
        // memcpy 到局部 float 后逐元素比较，不做指针类型双关
        float actual = 0.0F;
        std::memcpy(&actual, outputData + j * static_cast<int64_t>(sizeof(float)), sizeof(actual));
        if (fabs(actual - expectedValue) > 1.0e-5F) {
            printf("%s - ERROR - [XIR]: Validate output failed, result[%ld] is %f, expected %f\n", GetTime().c_str(), j,
                   static_cast<double>(actual), static_cast<double>(expectedValue));
            return FAILED;
        }
    }
    printf("%s - INFO - [XIR]: Validate output shape, dtype and %ld values success, all equal %f\n", GetTime().c_str(),
           elementNum, static_cast<double>(expectedValue));
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    // 1、创建图对象
    const char* graph_name = "tc_ge_irrun_inv_grad_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    DataType inDtype = DT_FLOAT;

    std::cout << inDtype << std::endl;

    // 初始化和设置图
    if (InitializeAndSetupGraph(graph, input, inDtype) != SUCCESS) {
        return FAILED;
    }

    // 执行图计算
    std::vector<ge::Tensor> output;
    if (ExecuteGraph(graph, input, output) != SUCCESS) {
        return FAILED;
    }

    // 处理输入输出数据
    if (ProcessIOData(input, output) != SUCCESS) {
        printf("%s - ERROR - [XIR]: Process io data failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }

    // 已知输入 x=2.0、grad=2.0（ADD_INPUT 固定填 2），
    // InvGrad 期望 y = -grad * x * x = -8.0，真实校验后才报精度
    const std::vector<int64_t> expectedShape = {4, 2};
    const float expectedValue = -8.0F;
    if (ValidateOutputValues(output, inDtype, expectedShape, expectedValue) != SUCCESS) {
        printf("%s - ERROR - [XIR]: Precision validation failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    std::cout << "Error message: " << error_str << std::endl;
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    std::cout << "Warning message: " << warning_str << std::endl;
    printf("%s - INFO - [XIR]: Precision is ok\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    Status ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
