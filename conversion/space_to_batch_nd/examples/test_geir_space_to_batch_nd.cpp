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
#include <memory>
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
#include "../op_graph/space_to_batch_nd_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

// ge::Tensor does not take ownership of external buffers, so keep them alive until graph execution finishes.
static std::vector<std::unique_ptr<uint8_t[]>> g_input_data_buffers;

#define ADD_SEQUENCE_INPUT(intputIndex, intputName, intputDtype, inputShape)                                        \
    vector<int64_t> placeholder##intputIndex##_shape = inputShape;                                                  \
    auto placeholder##intputIndex = op::Data("placeholder" + intputIndex).set_attr_index(0);                        \
    TensorDesc placeholder##intputIndex##_desc = TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, \
                                                            intputDtype);                                           \
    placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost);                                               \
    placeholder##intputIndex##_desc.SetFormat(FORMAT_ND);                                                           \
    Tensor tensor_placeholder##intputIndex;                                                                         \
    ret = GenSequenceData(placeholder##intputIndex##_shape, tensor_placeholder##intputIndex,                        \
                          placeholder##intputIndex##_desc, intputDtype);                                            \
    if (ret != SUCCESS) {                                                                                           \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                              \
        return FAILED;                                                                                              \
    }                                                                                                               \
    placeholder##intputIndex.update_input_desc_x(placeholder##intputIndex##_desc);                                  \
    input.push_back(tensor_placeholder##intputIndex);                                                               \
    graph.AddOp(placeholder##intputIndex);                                                                          \
    node.set_input_##intputName(placeholder##intputIndex);                                                          \
    inputs.push_back(placeholder##intputIndex);

#define ADD_INT_INPUT(intputIndex, intputName, intputDtype, inputShape, value)                                      \
    vector<int64_t> placeholder##intputIndex##_shape = inputShape;                                                  \
    auto placeholder##intputIndex = op::Data("placeholder" + intputIndex).set_attr_index(0);                        \
    TensorDesc placeholder##intputIndex##_desc = TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, \
                                                            intputDtype);                                           \
    placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost);                                               \
    placeholder##intputIndex##_desc.SetFormat(FORMAT_ND);                                                           \
    Tensor tensor_placeholder##intputIndex;                                                                         \
    ret = GenOnesData(placeholder##intputIndex##_shape, tensor_placeholder##intputIndex,                            \
                      placeholder##intputIndex##_desc, intputDtype, value);                                         \
    if (ret != SUCCESS) {                                                                                           \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                              \
        return FAILED;                                                                                              \
    }                                                                                                               \
    placeholder##intputIndex.update_input_desc_x(placeholder##intputIndex##_desc);                                  \
    input.push_back(tensor_placeholder##intputIndex);                                                               \
    graph.AddOp(placeholder##intputIndex);                                                                          \
    node.set_input_##intputName(placeholder##intputIndex);                                                          \
    inputs.push_back(placeholder##intputIndex);

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                       \
    TensorDesc outputName##outputIndex##_desc = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
    node.update_output_desc_##outputName(outputName##outputIndex##_desc);

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

static constexpr size_t MAX_ELEM_BYTES = 16;

uint32_t GetDataTypeSize(DataType dt)
{
    int size = ge::GetSizeByDataType(dt);
    return size > 0 ? static_cast<uint32_t>(size) : 0U;
}

int32_t FillElementData(uint8_t* dst, size_t elem_size, int64_t value)
{
    if (dst == nullptr || elem_size == 0U || elem_size > MAX_ELEM_BYTES) {
        printf("%s - ERROR - [XIR]: %s %s: invalid element size %zu\n", GetTime().c_str(), __FILE__, __func__,
               elem_size);
        return FAILED;
    }
    // Use a full-width zero-initialized buffer so bytes beyond int64_t are deterministic, even for 16B types.
    uint8_t tmp[MAX_ELEM_BYTES] = {0};
    size_t copy_size = elem_size < sizeof(value) ? elem_size : sizeof(value);
    memcpy(tmp, &value, copy_size);
    memcpy(dst, tmp, elem_size);
    return SUCCESS;
}

int32_t GenSequenceData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    size_t elem_size = GetDataTypeSize(data_type);
    if (elem_size == 0U || elem_size > MAX_ELEM_BYTES) {
        printf("%s - ERROR - [XIR]: %s %s: unsupported data type %d\n", GetTime().c_str(), __FILE__, __func__,
               static_cast<int32_t>(data_type));
        return FAILED;
    }
    size_t data_len = size * elem_size;
    std::unique_ptr<uint8_t[]> data(new (std::nothrow) uint8_t[data_len]);
    if (data == nullptr) {
        printf("%s - ERROR - [XIR]: %s %s: allocate input data memory failed\n", GetTime().c_str(), __FILE__, __func__);
        return FAILED;
    }
    for (size_t i = 0; i < size; ++i) {
        if (FillElementData(data.get() + i * elem_size, elem_size, static_cast<int64_t>(i)) != SUCCESS) {
            return FAILED;
        }
    }
    input_tensor = Tensor(input_tensor_desc, data.get(), data_len);
    g_input_data_buffers.push_back(std::move(data));
    return SUCCESS;
}

int32_t GenOnesData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type,
                    const vector<int32_t>& value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    size_t elem_size = GetDataTypeSize(data_type);
    if (elem_size == 0U || elem_size > MAX_ELEM_BYTES) {
        printf("%s - ERROR - [XIR]: %s %s: unsupported data type %d\n", GetTime().c_str(), __FILE__, __func__,
               static_cast<int32_t>(data_type));
        return FAILED;
    }
    size_t data_len = size * elem_size;
    std::unique_ptr<uint8_t[]> data(new (std::nothrow) uint8_t[data_len]);
    if (data == nullptr) {
        printf("%s - ERROR - [XIR]: %s %s: allocate input data memory failed\n", GetTime().c_str(), __FILE__, __func__);
        return FAILED;
    }
    for (size_t i = 0; i < size; ++i) {
        int64_t tmp = (i < value.size()) ? static_cast<int64_t>(value[i]) : 0;
        if (FillElementData(data.get() + i * elem_size, elem_size, tmp) != SUCCESS) {
            return FAILED;
        }
    }
    input_tensor = Tensor(input_tensor_desc, data.get(), data_len);
    g_input_data_buffers.push_back(std::move(data));
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t* inputData)
{
    FILE* fp = fopen(bin_file.c_str(), "wb");
    if (fp == nullptr) {
        printf("%s - ERROR - [XIR]: open file %s failed\n", GetTime().c_str(), bin_file.c_str());
        return FAILED;
    }
    if (data_size > 0) {
        if (inputData == nullptr) {
            printf("%s - ERROR - [XIR]: input data is null for file %s\n", GetTime().c_str(), bin_file.c_str());
            fclose(fp);
            return FAILED;
        }
        size_t written = fwrite(inputData, sizeof(uint8_t), data_size, fp);
        if (written != data_size) {
            printf("%s - ERROR - [XIR]: write file %s failed, expect %llu bytes but wrote %zu bytes\n",
                   GetTime().c_str(), bin_file.c_str(), static_cast<unsigned long long>(data_size), written);
            fclose(fp);
            return FAILED;
        }
    }
    if (fclose(fp) != 0) {
        printf("%s - ERROR - [XIR]: close file %s failed\n", GetTime().c_str(), bin_file.c_str());
        return FAILED;
    }
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    auto node = op::SpaceToBatchND("node");
    std::vector<int64_t> xShape = {1, 4, 4, 2};
    std::vector<int64_t> yShape = {4, 2, 2, 2};
    std::vector<int64_t> blockShapeShape = {2};
    std::vector<int32_t> blockShapeValue = {2, 2};
    std::vector<int64_t> paddingsShape = {2, 2};
    std::vector<int32_t> paddingsValue = {0, 0, 0, 0};

    ADD_SEQUENCE_INPUT(1, x, inDtype, xShape);

    ADD_INT_INPUT(2, block_shape, DT_INT32, blockShapeShape, blockShapeValue);
    ADD_INT_INPUT(3, paddings, DT_INT32, paddingsShape, paddingsValue);

    ADD_OUTPUT(1, y, inDtype, yShape);
    outputs.push_back(node);
    return SUCCESS;
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

    const char* arg = argc > 1 ? argv[1] : "";
    std::cout << arg << std::endl;

    DataType inDtype = DT_INT32;
    std::cout << inDtype << std::endl;

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        GEFinalize();
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
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {

    };
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Session add ir compute graph to ir session failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }

    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());
    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());
    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        std::cout << "input " << i << " dtype :  " << input[i].GetTensorDesc().GetDataType() << std::endl;
        string input_file = "./tc_ge_irrun_test_0008_npu_input_" + std::to_string(i) + ".bin";
        uint8_t* input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th input, input shape size =" << input_shape << std::endl;
        uint32_t data_size = input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        ret = WriteDataToFile((const char*)input_file.c_str(), data_size, input_data_i);
        if (ret != SUCCESS) {
            printf("%s - ERROR - [XIR]: write input bin failed\n", GetTime().c_str());
            delete session;
            GEFinalize();
            return FAILED;
        }
    }

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << output_shape << std::endl;
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        ret = WriteDataToFile((const char*)output_file.c_str(), data_size, output_data_i);
        if (ret != SUCCESS) {
            printf("%s - ERROR - [XIR]: write output bin failed\n", GetTime().c_str());
            delete session;
            GEFinalize();
            return FAILED;
        }
        int32_t* resultData = (int32_t*)output_data_i;
        for (int64_t j = 0; j < output_shape; j++) {
            LOG_PRINT("result[%ld] is: %u\n", j, resultData[j]);
        }
    }

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    std::cout << "Error message: " << error_str << std::endl;
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    std::cout << "Warning message: " << warning_str << std::endl;
    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
