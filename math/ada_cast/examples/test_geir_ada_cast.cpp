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
#include <cstdio>
#include <ctime>
#include <string.h>
#include <stdint.h>
#include <new>
#include <vector>
#include <string>
#include <map>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "nn_other.h"
#include "../op_graph/ada_cast_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

// 参考 libnnopbase fp16_t(opbase src/op_common/op_host/util/fp16.cpp)的示例级手写实现,
// 避免示例对 libnnopbase 的链接依赖; 含 denormal 与指数上/下溢处理
constexpr uint16_t FP16_EXP_BIAS = 15;
constexpr uint16_t FP16_MAN_LEN = 10;
constexpr uint16_t FP16_MAN_HIDE_BIT = 0x0400;
constexpr uint16_t FP16_MAX_MAN = 0x03FF;
constexpr uint32_t FP32_EXP_BIAS = 127;
constexpr uint32_t FP32_MAN_LEN = 23;

static float Fp16BitsToFloat(uint16_t fpVal)
{
    uint16_t hfSign = static_cast<uint16_t>((fpVal >> 15) & 1);
    int16_t hfExp = static_cast<int16_t>((fpVal >> FP16_MAN_LEN) & 0x1F);
    uint16_t hfMan = static_cast<uint16_t>(fpVal & FP16_MAX_MAN);
    if (hfExp > 0) {
        hfMan = static_cast<uint16_t>(hfMan | FP16_MAN_HIDE_BIT);
    } else {
        hfExp = 1;
    }
    while (hfMan != 0 && (hfMan & FP16_MAN_HIDE_BIT) == 0) {
        hfMan = static_cast<uint16_t>(hfMan << 1);
        hfExp = static_cast<int16_t>(hfExp - 1);
    }
    uint32_t eRet = 0;
    uint32_t mRet = 0;
    if (hfMan != 0) {
        eRet = static_cast<uint32_t>(hfExp - FP16_EXP_BIAS + FP32_EXP_BIAS);
        mRet = static_cast<uint32_t>(hfMan & FP16_MAX_MAN) << (FP32_MAN_LEN - FP16_MAN_LEN);
    }
    uint32_t fVal = (static_cast<uint32_t>(hfSign) << 31) | (eRet << FP32_MAN_LEN) | mRet;
    float ret;
    (void)memcpy(&ret, &fVal, sizeof(ret));
    return ret;
}

string GetTime()
{
    time_t timep;
    time(&timep);
    struct tm tmTmp = {0};
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime_r(&timep, &tmTmp));
    return tmp;
}

int32_t GenUint16Data(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (size_t i = 0; i < shapes.size(); i++) {
        size *= static_cast<size_t>(shapes[i]);
    }
    size_t data_len = size * sizeof(uint16_t);
    uint16_t* pData = new (std::nothrow) uint16_t[size];
    if (pData == nullptr) {
        return FAILED;
    }
    for (size_t i = 0; i < size; i++) {
        *(pData + i) = static_cast<uint16_t>(i % 256);
    }
    input_tensor = Tensor(input_tensor_desc, (uint8_t*)pData, data_len);
    delete[] pData; // Tensor 构造会拷贝数据，裸分配即用即还
    return SUCCESS;
}

int CreateOppInGraph(std::vector<ge::Tensor>& input, std::vector<Operator>& inputs, std::vector<Operator>& outputs,
                     Graph& graph)
{
    Status ret = SUCCESS;
    auto adaCast = op::AdaCast("ada_cast");
    adaCast.set_attr_pixel(65535);
    std::vector<int64_t> xShape = {4, 8};

    vector<int64_t> placeholder1_shape = xShape;
    auto placeholder1 = op::Data("placeholder").set_attr_index(0);
    TensorDesc placeholder1_desc = TensorDesc(ge::Shape(placeholder1_shape), FORMAT_ND, DT_UINT16);
    placeholder1_desc.SetPlacement(ge::kPlacementHost);
    placeholder1_desc.SetFormat(FORMAT_ND);
    Tensor tensor_placeholder1;
    ret = GenUint16Data(placeholder1_shape, tensor_placeholder1, placeholder1_desc);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());
        return FAILED;
    }
    placeholder1.update_input_desc_x(placeholder1_desc);
    placeholder1.update_output_desc_y(placeholder1_desc);
    input.push_back(tensor_placeholder1);
    graph.AddOp(placeholder1);
    adaCast.set_input_x(placeholder1);
    inputs.push_back(placeholder1);

    outputs.push_back(adaCast);
    return SUCCESS;
}

int main()
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

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    ret = CreateOppInGraph(input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create graph failed\n", GetTime().c_str());
        (void)ge::GEFinalize();
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    ge::Session* session = new (std::nothrow) Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create session failed\n", GetTime().c_str());
        (void)ge::GEFinalize();
        return FAILED;
    }

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        (void)ge::GEFinalize();
        return FAILED;
    }

    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        (void)ge::GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    size_t output_num = output.size();
    for (size_t i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        int64_t output_size = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "output element num = " << output_size << std::endl;
        uint8_t* output_data_i = output[i].GetData();
        if (output_data_i == nullptr || output_size < 1) {
            printf("%s - ERROR - [XIR]: Output %zu has no data\n", GetTime().c_str(), i);
            continue;
        }
        // AdaCast 输出 dtype 固定为 DT_FLOAT16，按 fp16 位型转 float 后打印
        uint16_t* outputF16 = reinterpret_cast<uint16_t*>(output_data_i);
        // x[j] = j (j < 256)，期望 out[j] = Cast_FP16(j × (1 / 65535))
        size_t printNum = static_cast<size_t>(output_size) < 4 ? static_cast<size_t>(output_size) : 4;
        for (size_t j = 0; j < printNum; j++) {
            printf("result[%zu] is: %f\n", j, Fp16BitsToFloat(outputF16[j]));
        }
    }

    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Finalize ge failed\n", GetTime().c_str());
        return FAILED;
    }
    return SUCCESS;
}
