/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_geir_maximum_grad.cpp
 * \brief MinimumGrad算子GE IR调用示例
 *
 * 验证公式：
 *   y1 = grads * (x2 >= x1)
 *   y2 = grads * NOT(x2 >= x1)
 * 输入数据按奇偶交错生成，使y1/y2均非全零。
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
#include "array_ops.h"
#include "ge_ir_build.h"

#include "../../op_graph/minimum_grad_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

// dataPattern: 0=grads(全1), 1=x1(偶数位1), 2=x2(偶数位0), 使mask交错
#define ADD_INPUT(intputIndex, intputName, intputDtype, inputShape, dataPattern)                             \
    do {                                                                                                     \
        vector<int64_t> placeholder##intputIndex##_shape = inputShape;                                       \
        auto placeholder##intputIndex = op::Data("placeholder" #intputIndex).set_attr_index(0);              \
        TensorDesc placeholder##intputIndex##_desc = TensorDesc(ge::Shape(placeholder##intputIndex##_shape), \
                                                                FORMAT_ND, intputDtype);                     \
        placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost);                                    \
        placeholder##intputIndex##_desc.SetFormat(FORMAT_ND);                                                \
        Tensor tensor_placeholder##intputIndex;                                                              \
        ret = GenInputDataFloat32(placeholder##intputIndex##_shape, tensor_placeholder##intputIndex,         \
                                  placeholder##intputIndex##_desc, dataPattern);                             \
        if (ret != SUCCESS) {                                                                                \
            printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                   \
            return FAILED;                                                                                   \
        }                                                                                                    \
        placeholder##intputIndex.update_input_desc_x(placeholder##intputIndex##_desc);                       \
        placeholder##intputIndex.update_output_desc_y(placeholder##intputIndex##_desc);                      \
        input.push_back(tensor_placeholder##intputIndex);                                                    \
        graph.AddOp(placeholder##intputIndex);                                                               \
        minimumGrad1.set_input_##intputName(placeholder##intputIndex);                                       \
        inputs.push_back(placeholder##intputIndex);                                                          \
    } while (0)

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                           \
    do {                                                                                                        \
        TensorDesc outputName##outputIndex##_desc = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
        minimumGrad1.update_output_desc_##outputName(outputName##outputIndex##_desc);                           \
    } while (0)

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

// dataPattern: 0=grads全1; 1=偶数位1奇数位0(x1); 2=偶数位0奇数位1(x2)
// 仅支持 DT_FLOAT: 数据按 float 生成, 强转字节宽度只在本 dtype 下正确。
int32_t GenInputDataFloat32(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc,
                            int32_t dataPattern)
{
    if (input_tensor_desc.GetDataType() != DT_FLOAT) {
        printf("%s - ERROR - [XIR]: GenInputDataFloat32 only supports DT_FLOAT\n", GetTime().c_str());
        return FAILED;
    }
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    const uint32_t byteSizeFloat32 = 4;
    uint32_t data_len = size * byteSizeFloat32;
    // Tensor 的 const uint8_t* 构造接口会拷贝数据, 局部 vector 出作用域自动释放,
    // 避免裸 new 泄漏。
    std::vector<float> pData(size);
    for (size_t i = 0; i < size; ++i) {
        float value = 1.0f;
        if (dataPattern == 1) {
            value = (i % 2 == 0) ? 1.0f : 0.0f;
        } else if (dataPattern == 2) {
            value = (i % 2 == 0) ? 0.0f : 1.0f;
        }
        pData[i] = value;
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<const uint8_t*>(pData.data()), data_len);
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t* inputData)
{
    FILE* fp = fopen(bin_file.c_str(), "w");
    fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    // 自定义代码：添加单算子定义到图中
    auto minimumGrad1 = op::MinimumGrad("minimumGrad1");
    std::vector<int64_t> xShape = {32, 64};

    ADD_INPUT(1, grads, inDtype, xShape, 0);
    ADD_INPUT(2, x1, inDtype, xShape, 1);
    ADD_INPUT(3, x2, inDtype, xShape, 2);

    // 属性：同时输出y1和y2
    minimumGrad1.set_attr_grad_x(true);
    minimumGrad1.set_attr_grad_y(true);

    // 输出描述：y1/y2与广播后shape一致
    ADD_OUTPUT(1, y1, inDtype, xShape);
    ADD_OUTPUT(2, y2, inDtype, xShape);

    outputs.push_back(minimumGrad1);
    // 添加完毕
    return SUCCESS;
}

void SaveInputOutput(std::vector<ge::Tensor>& input, std::vector<ge::Tensor>& output)
{
    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        std::cout << "input " << i << " dtype :  " << input[i].GetTensorDesc().GetDataType() << std::endl;
        string input_file = "./tc_ge_irrun_min_grad_npu_input_" + std::to_string(i) + ".bin";
        uint8_t* input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th input, input shape size =" << input_shape << std::endl;
        uint32_t data_size = input_shape * ge::GetSizeByDataType(input[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)input_file.c_str(), data_size, input_data_i);
    }

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./tc_ge_irrun_min_grad_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << output_shape << std::endl;
        uint32_t data_size = output_shape * ge::GetSizeByDataType(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)output_file.c_str(), data_size, output_data_i);
        float* resultData = (float*)output_data_i;
        for (int64_t j = 0; j < output_shape; j++) {
            printf("result[%ld] is: %f\n", j, resultData[j]);
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
        printf("%s - ERROR - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

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

    std::map<AscendString, AscendString> build_options = {

    };
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session* session = new Session(build_options);

    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {

    };
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);

    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());
    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());
    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    SaveInputOutput(input, output);

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    std::cout << "Error message: " << error_str << std::endl;
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    std::cout << "Warning message: " << warning_str << std::endl;
    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
