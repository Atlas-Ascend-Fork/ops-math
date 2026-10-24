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
 * @file test_geir_asin_grad.cpp
 * @brief GE IR 图模式调用示例 - AsinGrad
 *
 * 数学公式: z = dy / sqrt(1 - y * y)
 *
 * 通过 GE Session 构建计算图，编译并执行 AsinGrad 算子。
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

#include "../op_graph/asin_grad_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define ADD_INPUT(inputIndex, inputName, inputDtype, inputShape)                                                  \
    vector<int64_t> placeholder##inputIndex##_shape = inputShape;                                                 \
    auto placeholder##inputIndex = op::Data("placeholder" #inputIndex).set_attr_index(inputIndex - 1);            \
    TensorDesc placeholder##inputIndex##_desc = TensorDesc(ge::Shape(placeholder##inputIndex##_shape), FORMAT_ND, \
                                                           inputDtype);                                           \
    placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                              \
    placeholder##inputIndex##_desc.SetFormat(FORMAT_ND);                                                          \
    Tensor tensor_placeholder##inputIndex;                                                                        \
    ret = GenInputData(placeholder##inputIndex##_shape, tensor_placeholder##inputIndex,                           \
                       placeholder##inputIndex##_desc, inputDtype, inputIndex);                                   \
    if (ret != SUCCESS) {                                                                                         \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                            \
        return FAILED;                                                                                            \
    }                                                                                                             \
    placeholder##inputIndex.update_input_desc_x(placeholder##inputIndex##_desc);                                  \
    input.push_back(tensor_placeholder##inputIndex);                                                              \
    graph.AddOp(placeholder##inputIndex);                                                                         \
    asinGradOp.set_input_##inputName(placeholder##inputIndex);                                                    \
    inputs.push_back(placeholder##inputIndex);

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                       \
    TensorDesc outputName##outputIndex##_desc = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
    asinGradOp.update_output_desc_##outputName(outputName##outputIndex##_desc);

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

// 与仓内样例（math/asin/examples/test_aclnn_asin_fp16.cpp）一致的位级转换。
static inline uint16_t FloatToFp16Bits(float value)
{
    uint32_t bits = *reinterpret_cast<uint32_t*>(&value);
    uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0) {
        return sign; // 采样值不会进入非规格化区间，直接置零
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u); // 溢出为 inf
    }
    return static_cast<uint16_t>(sign | (static_cast<uint16_t>(exponent) << 10) | (mantissa >> 13));
}

static inline uint16_t FloatToBf16Bits(float value)
{
    uint32_t bits = *reinterpret_cast<uint32_t*>(&value);
    if ((bits & 0x7f800000u) == 0x7f800000u) {
        return static_cast<uint16_t>(bits >> 16); // inf / NaN 原样保留
    }
    bits += 0x7fffu + ((bits >> 16) & 1u); // round to nearest even
    return static_cast<uint16_t>(bits >> 16);
}

int32_t GenInputData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type,
                     int inputIndex)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    // 缓冲宽度必须与 TensorDesc 声明的 dtype 一致（fp16/bf16 为 2 字节/元素），
    // 否则 Tensor 只会复制并重新解释 float 缓冲的前半段字节。
    uint32_t data_len = size * GetDataTypeSize(data_type);
    std::vector<uint8_t> data(data_len, 0);
    for (size_t i = 0; i < size; ++i) {
        // y: 线性分布的值，shape {4,4,4} 时实际为 [-0.5, 0.13]；
        // dy: 常量梯度 1.0。两者均按所选 dtype 写入。
        float value = (inputIndex == 1) ? (static_cast<float>(i) * 0.01f - 0.5f) : 1.0f;
        if (data_type == ge::DT_FLOAT) {
            reinterpret_cast<float*>(data.data())[i] = value;
        } else if (data_type == ge::DT_FLOAT16) {
            reinterpret_cast<uint16_t*>(data.data())[i] = FloatToFp16Bits(value);
        } else if (data_type == ge::DT_BF16) {
            reinterpret_cast<uint16_t*>(data.data())[i] = FloatToBf16Bits(value);
        } else {
            return FAILED;
        }
    }
    input_tensor = Tensor(input_tensor_desc, data.data(), data_len);
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    auto asinGradOp = op::AsinGrad("asin_grad_1");
    std::vector<int64_t> xShape = {4, 4, 4};

    ADD_INPUT(1, y, inDtype, xShape);
    ADD_INPUT(2, dy, inDtype, xShape);
    ADD_OUTPUT(1, z, inDtype, xShape);

    outputs.push_back(asinGradOp);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "asin_grad_ge_ir_test";
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
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session failed\n", GetTime().c_str());
        return FAILED;
    }

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph to session failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }

    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        printf("%s - INFO - [XIR]: output %d shape size = %ld, dtype = %d\n", GetTime().c_str(), i, output_shape,
               static_cast<int>(output[i].GetTensorDesc().GetDataType()));

        if (output[i].GetTensorDesc().GetDataType() == DT_FLOAT && output_shape > 0) {
            float* outData = reinterpret_cast<float*>(output[i].GetData());
            int64_t printNum = (output_shape < 10) ? output_shape : 10;
            for (int64_t j = 0; j < printNum; j++) {
                float y = static_cast<float>(j) * 0.01f - 0.5f;
                float expected = 1.0f / std::sqrt(1.0f - y * y);
                printf("  output[%ld] = %.6f (expected = %.6f)\n", j, outData[j], expected);
            }
        }
    }

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    if (!error_str.empty()) {
        std::cout << "Error message: " << error_str << std::endl;
    }

    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    return SUCCESS;
}
