/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_geir_add_mat_mat_elements.cpp
 * @brief GE IR 图模式调用示例 - AddMatMatElements
 *
 * 数学公式: c = c * beta + alpha * a * b
 *
 * 通过 GE Session 构建计算图，编译并执行 AddMatMatElements 算子。
 */

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include "assert.h"

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "../op_graph/add_mat_mat_elements_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

bool IsKnownScenario(const string& scenario)
{
    return scenario == "valid" || scenario == "input_c_fractal_nz" || scenario == "input_b_fractal_nz" ||
           scenario == "input_alpha_fractal_nz" || scenario == "input_a_nchw" || scenario == "output_bool" ||
           scenario == "output_fractal_nz" || scenario == "output_wrong_shape" || scenario == "output_wrong_rank";
}

bool ExpectsGraphRejection(const string& scenario) { return scenario == "input_alpha_fractal_nz"; }

Format GetInputFormat(const string& scenario, int inputIndex)
{
    if ((scenario == "input_c_fractal_nz" && inputIndex == 1) ||
        (scenario == "input_b_fractal_nz" && inputIndex == 3) ||
        (scenario == "input_alpha_fractal_nz" && inputIndex == 5)) {
        return FORMAT_FRACTAL_NZ;
    }
    if (scenario == "input_a_nchw" && inputIndex == 2) {
        return FORMAT_NCHW;
    }
    return FORMAT_ND;
}

vector<int64_t> GetDataShape(const string& scenario)
{
    return scenario == "input_a_nchw" ? vector<int64_t>{2, 2, 2, 2, 2, 2, 2} : vector<int64_t>{4, 8};
}

#define ADD_INPUT(inputIndex, inputName, inputDtype, inputShape)                                          \
    vector<int64_t> placeholder##inputIndex##_shape = inputShape;                                         \
    auto placeholder##inputIndex = op::Data("placeholder" #inputIndex).set_attr_index(inputIndex - 1);    \
    Format placeholder##inputIndex##_format = GetInputFormat(scenario, inputIndex);                       \
    TensorDesc placeholder##inputIndex##_desc = TensorDesc(ge::Shape(placeholder##inputIndex##_shape),    \
                                                           placeholder##inputIndex##_format, inputDtype); \
    placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                      \
    placeholder##inputIndex##_desc.SetFormat(placeholder##inputIndex##_format);                           \
    Tensor tensor_placeholder##inputIndex;                                                                \
    ret = GenInputData(placeholder##inputIndex##_shape, tensor_placeholder##inputIndex,                   \
                       placeholder##inputIndex##_desc, inputDtype, inputIndex);                           \
    if (ret != SUCCESS) {                                                                                 \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                    \
        return FAILED;                                                                                    \
    }                                                                                                     \
    placeholder##inputIndex.update_input_desc_x(placeholder##inputIndex##_desc);                          \
    input.push_back(tensor_placeholder##inputIndex);                                                      \
    graph.AddOp(placeholder##inputIndex);                                                                 \
    addMatOp.set_input_##inputName(placeholder##inputIndex);                                              \
    inputs.push_back(placeholder##inputIndex);

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                       \
    TensorDesc outputName##outputIndex##_desc = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
    addMatOp.update_output_desc_##outputName(outputName##outputIndex##_desc);

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

int32_t GenInputData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type,
                     int inputIndex)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    size_t storageSize = size;
    if (input_tensor_desc.GetFormat() == FORMAT_FRACTAL_NZ && storageSize < 128) {
        storageSize = 128;
    }
    uint32_t data_len = storageSize * GetDataTypeSize(data_type);
    float* pData = new (std::nothrow) float[storageSize];
    if (pData == nullptr) {
        return FAILED;
    }

    // inputIndex: 1=c, 2=a, 3=b, 4=beta(scalar), 5=alpha(scalar)
    if (inputIndex == 1) {
        // c: 1.0
        for (size_t i = 0; i < storageSize; ++i)
            pData[i] = 1.0f;
    } else if (inputIndex == 2) {
        // a: 2.0
        for (size_t i = 0; i < storageSize; ++i)
            pData[i] = 2.0f;
    } else if (inputIndex == 3) {
        // b: 3.0
        for (size_t i = 0; i < storageSize; ++i)
            pData[i] = 3.0f;
    } else if (inputIndex == 4) {
        // beta = 0.5
        for (size_t i = 0; i < storageSize; ++i)
            pData[i] = 0.5f;
    } else {
        // alpha = 1.5
        for (size_t i = 0; i < storageSize; ++i)
            pData[i] = 1.5f;
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), data_len);
    delete[] pData;
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, const string& scenario, std::vector<ge::Tensor>& input,
                     std::vector<Operator>& inputs, std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    auto addMatOp = op::AddMatMatElements("add_mat_mat_elements_1");
    std::vector<int64_t> xShape = GetDataShape(scenario);
    std::vector<int64_t> scalarShape = {1};

    ADD_INPUT(1, c, inDtype, xShape);
    ADD_INPUT(2, a, inDtype, xShape);
    ADD_INPUT(3, b, inDtype, xShape);
    ADD_INPUT(4, beta, inDtype, scalarShape);
    ADD_INPUT(5, alpha, inDtype, scalarShape);
    DataType outputDtype = scenario == "output_bool" ? DT_BOOL : inDtype;
    Format outputFormat = scenario == "output_fractal_nz" ? FORMAT_FRACTAL_NZ : FORMAT_ND;
    std::vector<int64_t> outputShape = xShape;
    if (scenario == "output_wrong_shape") {
        outputShape = {999, 999};
    } else if (scenario == "output_wrong_rank") {
        outputShape = {2, 1, 4};
    }
    TensorDesc outputDesc = TensorDesc(ge::Shape(outputShape), outputFormat, outputDtype);
    addMatOp.update_output_desc_c(outputDesc);

    outputs.push_back(addMatOp);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "add_mat_mat_elements_ge_ir_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;
    string deviceId = argc > 3 ? argv[3] : "0";

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", deviceId.c_str()},
                                                           {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    DataType inDtype = DT_FLOAT;
    if (argc > 1) {
        std::string dtypeArg(argv[1]);
        if (dtypeArg == "fp16") {
            inDtype = DT_FLOAT16;
        } else if (dtypeArg != "fp32") {
            printf("%s - ERROR - [XIR]: Unsupported dtype '%s'; expected fp16 or fp32\n", GetTime().c_str(),
                   dtypeArg.c_str());
            GEFinalize();
            return FAILED;
        }
    }
    string scenario = argc > 2 ? argv[2] : "valid";
    if (!IsKnownScenario(scenario)) {
        printf("%s - ERROR - [XIR]: Unsupported scenario '%s'\n", GetTime().c_str(), scenario.c_str());
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Using dtype = %d, scenario = %s, device = %s\n", GetTime().c_str(),
           static_cast<int>(inDtype), scenario.c_str(), deviceId.c_str());

    ret = CreateOppInGraph(inDtype, scenario, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create op in graph failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        return FAILED;
    }

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        ge::AscendString errorMsg = ge::GEGetErrorMsgV2();
        printf("%s - ERROR - [XIR]: AddGraph failed, ret=%u, message=%s\n", GetTime().c_str(), ret,
               errorMsg.GetString());
        delete session;
        GEFinalize();
        return ExpectsGraphRejection(scenario) ? SUCCESS : FAILED;
    }

    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        ge::AscendString errorMsg = ge::GEGetErrorMsgV2();
        printf("%s - ERROR - [XIR]: RunGraph failed, ret=%u, message=%s\n", GetTime().c_str(), ret,
               errorMsg.GetString());
        delete session;
        GEFinalize();
        return ExpectsGraphRejection(scenario) ? SUCCESS : FAILED;
    }
    if (ExpectsGraphRejection(scenario)) {
        printf("%s - ERROR - [XIR]: scenario %s unexpectedly executed successfully\n", GetTime().c_str(),
               scenario.c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    int output_num = output.size();
    bool validationFailed = false;
    for (int i = 0; i < output_num; i++) {
        TensorDesc actualOutputDesc = output[i].GetTensorDesc();
        int64_t output_shape = actualOutputDesc.GetShape().GetShapeSize();
        printf("%s - INFO - [XIR]: output %d shape size = %ld, dtype = %d, format = %d\n", GetTime().c_str(), i,
               output_shape, static_cast<int>(actualOutputDesc.GetDataType()),
               static_cast<int>(actualOutputDesc.GetFormat()));

        if (actualOutputDesc.GetDataType() != inDtype) {
            printf("%s - ERROR - [XIR]: output dtype was not inferred from c\n", GetTime().c_str());
            validationFailed = true;
        }
        if (scenario != "output_fractal_nz" && actualOutputDesc.GetShape().GetDims() != GetDataShape(scenario)) {
            printf("%s - ERROR - [XIR]: output shape was not inferred from c\n", GetTime().c_str());
            validationFailed = true;
        }

        if (scenario != "output_fractal_nz" && actualOutputDesc.GetDataType() == DT_FLOAT && output_shape > 0) {
            float* outData = reinterpret_cast<float*>(output[i].GetData());
            // expected: c*beta + alpha*a*b = 1*0.5 + 1.5*2*3 = 9.5
            float expected = 1.0f * 0.5f + 1.5f * 2.0f * 3.0f;
            int64_t printNum = (output_shape < 10) ? output_shape : 10;
            for (int64_t j = 0; j < printNum; j++) {
                printf("  output[%ld] = %.6f (expected = %.6f)\n", j, outData[j], expected);
            }
            for (int64_t j = 0; j < output_shape; j++) {
                if (std::fabs(outData[j] - expected) > 1e-6f) {
                    printf("%s - ERROR - [XIR]: output value mismatch at index %ld\n", GetTime().c_str(), j);
                    validationFailed = true;
                    break;
                }
            }
        } else if (scenario == "output_fractal_nz") {
            printf("  output uses a transformed physical layout; raw ND value comparison is not applicable\n");
        }
    }

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    if (!error_str.empty()) {
        std::cout << "Error message: " << error_str << std::endl;
    }

    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS || validationFailed) {
        return FAILED;
    }
    return SUCCESS;
}
