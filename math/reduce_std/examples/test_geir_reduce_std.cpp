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
 * \file test_geir_reduce_std.cpp
 * \brief GE IR (graph mode) calling example for ReduceStd.
 *
 * Builds and runs a single-op graph for ReduceStd through the ge::Operator API
 * using the REG_OP registration from op_graph/reduce_std_proto.h.
 *
 *   ReduceStd: x[2,4] -> y1[2] (std), y2[2] (mean)  (reduce along dim=[1])
 *
 * Attributes:
 *   dim / unbiased / keepdim
 *
 * Input x (float32):
 *   row0 = [1, 2, 3, 4], row1 = [2, 4, 6, 8]
 *
 * Expected outputs (unbiased=true, keepdim=false):
 *   y1 = [sqrt(5/3), sqrt(20/3)] = [1.290994, 2.581989]
 *   y2 = [2.5, 5.0]
 *
 * NOTE: this example includes ONLY the local op_graph/reduce_std_proto.h and
 * deliberately does NOT include all_ops.h to avoid duplicate REG_OP.
 */

#include <cmath>
#include <iostream>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "graph/error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "../op_graph/reduce_std_proto.h"

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
    if (dt == ge::DT_FLOAT) {
        return 4;
    } else if (dt == ge::DT_FLOAT16) {
        return 2;
    }
    return 4;
}

int32_t GenX(const vector<int64_t>& shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType dt)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * GetDataTypeSize(dt);
    float* pData = new (std::nothrow) float[size];
    if (pData == nullptr) {
        return FAILED;
    }
    // x = [[1, 2, 3, 4], [2, 4, 6, 8]]
    const float vals[8] = {1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 4.0f, 6.0f, 8.0f};
    for (size_t i = 0; i < size; ++i) {
        pData[i] = (i < 8) ? vals[i] : 0.0f;
    }
    input_tensor = Tensor(input_tensor_desc, (uint8_t*)pData, data_len);
    delete[] pData;
    return SUCCESS;
}

int32_t WriteDataToFile(const string& bin_file, uint64_t data_size, const uint8_t* inputData)
{
    FILE* fp = fopen(bin_file.c_str(), "wb");
    if (fp == nullptr) {
        return FAILED;
    }
    size_t written = fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    if (written != data_size) {
        return FAILED;
    }
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    auto reduceStd = op::ReduceStd("reduce_std_0");

    // Attributes: dim=[1], unbiased=true, keepdim=false.
    std::vector<int64_t> dimAttr = {1};
    reduceStd.set_attr_dim(dimAttr);
    reduceStd.set_attr_unbiased(true);
    reduceStd.set_attr_keepdim(false);

    // x: [2, 4]
    std::vector<int64_t> xShape = {2, 4};
    auto placeholderX = op::Data("placeholder_x").set_attr_index(0);
    TensorDesc xDesc = TensorDesc(ge::Shape(xShape), FORMAT_ND, inDtype);
    xDesc.SetPlacement(ge::kPlacementHost);
    xDesc.SetFormat(FORMAT_ND);

    Tensor tensorX;
    ret = GenX(xShape, tensorX, xDesc, inDtype);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate x data failed\n", GetTime().c_str());
        return FAILED;
    }
    placeholderX.update_input_desc_x(xDesc);
    placeholderX.update_output_desc_y(xDesc);
    input.push_back(tensorX);
    graph.AddOp(placeholderX);
    reduceStd.set_input_x(placeholderX);
    inputs.push_back(placeholderX);

    // y1 (std): reduce dim=[1], keepdim=false -> [2]
    std::vector<int64_t> y1Shape = {2};
    TensorDesc y1Desc = TensorDesc(ge::Shape(y1Shape), FORMAT_ND, inDtype);
    reduceStd.update_output_desc_y1(y1Desc);

    // y2 (mean): reduce dim=[1], keepdim=false -> [2]
    std::vector<int64_t> y2Shape = {2};
    TensorDesc y2Desc = TensorDesc(ge::Shape(y2Shape), FORMAT_ND, inDtype);
    reduceStd.update_output_desc_y2(y2Desc);

    outputs.push_back(reduceStd);
    return SUCCESS;
}

int SaveInputOutput(std::vector<ge::Tensor>& input, std::vector<ge::Tensor>& output)
{
    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        std::cout << "input " << i << " dtype :  " << input[i].GetTensorDesc().GetDataType() << std::endl;
        string input_file = "./tc_ge_irrun_reduce_std_npu_input_" + std::to_string(i) + ".bin";
        uint8_t* input_data_i = input[i].GetData();
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
        string output_file = "./tc_ge_irrun_reduce_std_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << output_shape << std::endl;
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        if (WriteDataToFile(output_file, data_size, output_data_i) != SUCCESS) {
            return FAILED;
        }
        float* resultData = (float*)output_data_i;
        for (int64_t j = 0; j < output_shape; j++) {
            LOG_PRINT("result[%ld] is: %f\n", j, resultData[j]);
        }
    }
    return SUCCESS;
}

bool CheckOutput(const std::vector<ge::Tensor>& output)
{
    if (output.size() != 2) {
        printf("Output count mismatch: expected 2 (y1, y2), actual=%zu\n", output.size());
        return false;
    }
    if (output[0].GetTensorDesc().GetDataType() != DT_FLOAT || output[1].GetTensorDesc().GetDataType() != DT_FLOAT) {
        printf("Output dtype mismatch: expected FLOAT for y1 and y2\n");
        return false;
    }
    if (output[0].GetTensorDesc().GetShape().GetDims() != std::vector<int64_t>({2}) ||
        output[1].GetTensorDesc().GetShape().GetDims() != std::vector<int64_t>({2})) {
        printf("Output shape mismatch: expected [2] for y1 and y2\n");
        return false;
    }
    if (output[0].GetData() == nullptr || output[1].GetData() == nullptr) {
        printf("Output data is null\n");
        return false;
    }
    const float expectedStd[2] = {1.2909944f, 2.5819889f};
    const float expectedMean[2] = {2.5f, 5.0f};
    const float* stdValues = reinterpret_cast<const float*>(output[0].GetData());
    const float* meanValues = reinterpret_cast<const float*>(output[1].GetData());
    for (size_t i = 0; i < 2; ++i) {
        if (std::fabs(stdValues[i] - expectedStd[i]) > 1e-3f || std::fabs(meanValues[i] - expectedMean[i]) > 1e-3f) {
            printf("Output mismatch: index=%zu, std actual=%f expected=%f, mean actual=%f expected=%f\n", i,
                   stdValues[i], expectedStd[i], meanValues[i], expectedMean[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "tc_ge_irrun_reduce_std";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: GEInitialize failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    if (argc > 1) {
        std::cout << argv[1] << std::endl;
    }

    DataType inDtype = DT_FLOAT;
    std::cout << "Input dtype: " << inDtype << std::endl;

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create graph failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session* session = new (std::nothrow) Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: AddGraph failed, rc=%u\n", GetTime().c_str(), ret);
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
        printf("%s - ERROR - [XIR]: RunGraph failed, rc=%u\n", GetTime().c_str(), ret);
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    if (!CheckOutput(output) || SaveInputOutput(input, output) != SUCCESS) {
        printf("%s - ERROR - [XIR]: Result verification or file write failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }

    delete session;

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString() == nullptr ? "" : error_msg.GetString());
    std::cout << "Error message: " << error_str << std::endl;
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString() == nullptr ? "" : warning_msg.GetString());
    std::cout << "Warning message: " << warning_str << std::endl;
    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: GEFinalize failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
