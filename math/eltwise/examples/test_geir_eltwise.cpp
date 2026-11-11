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
 * @file test_geir_eltwise.cpp
 * @brief GE IR graph-mode example - Eltwise (DYNAMIC_INPUT)
 *
 * Three modes supported:
 *   mode=0 (PRODUCT): out = x0 * x1 * ... * x_{n-1}
 *   mode=1 (SUM):     out = c0*x0 + c1*x1 + ... + c_{n-1}*x_{n-1}
 *   mode=2 (MAX):     out = max(x0, x1, ..., x_{n-1})
 *
 * This example uses N=2 inputs with mode=1 (SUM, default coeffs = 1.0),
 * which yields out = x0 + x1 (= 2.0 for all-ones inputs).
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

#include "../op_graph/eltwise_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define ADD_DYNAMIC_INPUT(inputIndex, inputDtype, inputShape, dataName)                            \
    do {                                                                                           \
        vector<int64_t> shape##dataName = inputShape;                                              \
        auto placeholder##dataName = op::Data("placeholder" #dataName).set_attr_index(inputIndex); \
        TensorDesc desc##dataName(ge::Shape(shape##dataName), FORMAT_ND, inputDtype);              \
        desc##dataName.SetFormat(FORMAT_ND);                                                       \
        placeholder##dataName.update_input_desc_x(desc##dataName);                                 \
        placeholder##dataName.update_output_desc_y(desc##dataName);                                \
        Tensor tensor##dataName;                                                                   \
        ret = GenInputData(shape##dataName, tensor##dataName, desc##dataName, inputDtype);         \
        if (ret != SUCCESS) {                                                                      \
            printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());         \
            return FAILED;                                                                         \
        }                                                                                          \
        eltwiseOp.UpdateDynamicInputDesc("x", inputIndex, desc##dataName);                         \
        eltwiseOp.set_dynamic_input_x(inputIndex, placeholder##dataName);                          \
        input.push_back(tensor##dataName);                                                         \
        graph.AddOp(placeholder##dataName);                                                        \
        inputs.push_back(placeholder##dataName);                                                   \
    } while (0)

#define ADD_OUTPUT(outputName, outputDtype, outputShape)                                           \
    do {                                                                                           \
        TensorDesc outputName##_desc = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
        eltwiseOp.update_output_desc_##outputName(outputName##_desc);                              \
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
    } else if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) {
        return 2;
    }
    return 4;
}

constexpr uint16_t FLOAT16_ONE_BITS = 0x3C00;
constexpr uint16_t BFLOAT16_ONE_BITS = 0x3F80;
constexpr uint32_t BF16_TO_FP32_SHIFT = 16U;
constexpr uint16_t FP16_SIGN_MASK = 0x8000U;
constexpr uint32_t FP16_EXP_MASK = 0x1FU;
constexpr uint32_t FP16_MANT_MASK = 0x3FFU;
constexpr uint32_t FP16_EXP_SHIFT = 10U;
constexpr uint32_t FP16_MANT_TO_FP32_SHIFT = 13U;
constexpr uint32_t FP16_TO_FP32_SIGN_SHIFT = 16U;
constexpr uint32_t FP32_EXP_BIAS_DELTA = 112U;
constexpr uint32_t FP32_MANT_SHIFT = 23U;
constexpr uint32_t FP32_INF_BITS = 0x7F800000U;
constexpr float FP16_DENORM_SCALE = 16777216.0f;

static float Bf16ToFp32(uint16_t bf16Bits)
{
    uint32_t fp32Bits = static_cast<uint32_t>(bf16Bits) << BF16_TO_FP32_SHIFT;
    float out;
    __builtin_memcpy(&out, &fp32Bits, sizeof(float));
    return out;
}

static float Fp16ToFp32(uint16_t fp16Bits)
{
    const uint32_t sign = static_cast<uint32_t>(fp16Bits & FP16_SIGN_MASK) << FP16_TO_FP32_SIGN_SHIFT;
    const uint32_t exp = (fp16Bits >> FP16_EXP_SHIFT) & FP16_EXP_MASK;
    const uint32_t mant = fp16Bits & FP16_MANT_MASK;
    uint32_t fp32Bits;
    if (exp == FP16_EXP_MASK) {
        fp32Bits = sign | FP32_INF_BITS | (mant << FP16_MANT_TO_FP32_SHIFT);
    } else if (exp == 0) {
        if (mant == 0) {
            fp32Bits = sign;
        } else {
            float denorm = static_cast<float>(mant) / FP16_DENORM_SCALE;
            return sign ? -denorm : denorm;
        }
    } else {
        fp32Bits = sign | ((exp + FP32_EXP_BIAS_DELTA) << FP32_MANT_SHIFT) | (mant << FP16_MANT_TO_FP32_SHIFT);
    }
    float out;
    __builtin_memcpy(&out, &fp32Bits, sizeof(float));
    return out;
}

int32_t GenInputData(vector<int64_t> shapes, Tensor& inputTensor, TensorDesc& inputTensorDesc, DataType dataType)
{
    inputTensorDesc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t dataLen = size * GetDataTypeSize(dataType);
    std::vector<uint8_t> pData(dataLen);

    // All inputs filled with 1.0 in the target dtype to make verification simple.
    if (dataType == DT_FLOAT16) {
        uint16_t* fp16Data = reinterpret_cast<uint16_t*>(pData.data());
        for (size_t i = 0; i < size; ++i) {
            fp16Data[i] = FLOAT16_ONE_BITS;
        }
    } else if (dataType == DT_BF16) {
        uint16_t* bf16Data = reinterpret_cast<uint16_t*>(pData.data());
        for (size_t i = 0; i < size; ++i) {
            bf16Data[i] = BFLOAT16_ONE_BITS;
        }
    } else {
        float* fp32Data = reinterpret_cast<float*>(pData.data());
        for (size_t i = 0; i < size; ++i) {
            fp32Data[i] = 1.0f;
        }
    }
    inputTensor = Tensor(inputTensorDesc, pData.data(), dataLen);
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    constexpr int64_t kInputNum = 2;

    // Create Eltwise node with N=2 dynamic inputs
    auto eltwiseOp = op::Eltwise("eltwise_1").create_dynamic_input_x(kInputNum, false);
    std::vector<int64_t> xShape = {4, 4, 4};

    // Add 2 dynamic inputs (all-ones)
    ADD_DYNAMIC_INPUT(0, inDtype, xShape, 1);
    ADD_DYNAMIC_INPUT(1, inDtype, xShape, 2);

    // Attributes: N=2 (required by proto.h), mode=1 (SUM), coeff=default (all 1.0)
    eltwiseOp.set_attr_N(kInputNum);
    eltwiseOp.set_attr_mode(1);
    eltwiseOp.set_attr_coeff({1.0f, 1.0f});

    // Output
    ADD_OUTPUT(y, inDtype, xShape);

    outputs.push_back(eltwiseOp);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "eltwise_ge_ir_test";
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

    DataType inDtype = DT_FLOAT;
    if (argc > 1) {
        std::string dtypeArg(argv[1]);
        if (dtypeArg == "fp16") {
            inDtype = DT_FLOAT16;
        } else if (dtypeArg == "bf16") {
            inDtype = DT_BF16;
        }
    }
    printf("%s - INFO - [XIR]: Using dtype = %d\n", GetTime().c_str(), static_cast<int>(inDtype));

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create op in graph failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session* session = new Session(build_options);

    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph to session failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }

    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
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

    // Verify output: mode=1 SUM of 2 all-ones tensors => 2.0 per element
    int output_num = output.size();
    printf("%s - INFO - [XIR]: Number of outputs: %d\n", GetTime().c_str(), output_num);
    for (int i = 0; i < output_num; i++) {
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        printf("%s - INFO - [XIR]: output %d shape size = %ld, dtype = %d\n", GetTime().c_str(), i, output_shape,
               static_cast<int>(output[i].GetTensorDesc().GetDataType()));

        if (output_shape > 0) {
            const DataType outDtype = output[i].GetTensorDesc().GetDataType();
            uint8_t* outData = output[i].GetData();
            int64_t printNum = (output_shape < 10) ? output_shape : 10;
            const float expectedSum = 2.0f;
            for (int64_t j = 0; j < printNum; j++) {
                float val = 0.0f;
                if (outDtype == DT_FLOAT16) {
                    val = Fp16ToFp32(reinterpret_cast<uint16_t*>(outData)[j]);
                } else if (outDtype == DT_BF16) {
                    val = Bf16ToFp32(reinterpret_cast<uint16_t*>(outData)[j]);
                } else {
                    val = reinterpret_cast<float*>(outData)[j];
                }
                printf("  output[%ld] = %.6f (expected = %.6f)\n", j, val, expectedSum);
            }
        }
    }

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    if (!error_str.empty()) {
        std::cout << "Error message: " << error_str << std::endl;
    }

    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
