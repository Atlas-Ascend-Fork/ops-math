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
#include "../op_graph/complex_abs_proto.h"

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

static uint16_t FloatToFp16Bits(float fVal)
{
    uint32_t ui32V;
    (void)memcpy(&ui32V, &fVal, sizeof(ui32V));
    uint16_t sRet = static_cast<uint16_t>((ui32V >> 31) & 1);
    uint32_t eF = (ui32V >> FP32_MAN_LEN) & 0xFF;
    uint32_t mF = ui32V & 0x007FFFFFu;
    uint16_t eRet;
    uint16_t manRet;
    if (eF > 0x8Fu) { // 指数上溢 -> inf
        eRet = 0x1E;
        manRet = FP16_MAX_MAN;
    } else if (eF <= 0x70u) { // 指数下溢 -> denormal/0
        eRet = 0;
        if (eF >= 0x67u) {
            uint64_t mTmp = static_cast<uint64_t>(mF | 0x00800000u) << (eF - 0x67u);
            manRet = static_cast<uint16_t>(mTmp >> FP32_MAN_LEN);
            if (((mTmp >> (FP32_MAN_LEN - 1)) & 1) != 0) {
                manRet = static_cast<uint16_t>(manRet + 1);
            }
        } else if (eF == 0x66u && mF > 0) {
            manRet = 1;
        } else {
            manRet = 0;
        }
    } else { // 常规: 就近舍入, 尾数进位越过隐藏位则指数+1
        eRet = static_cast<uint16_t>(eF - 0x70u);
        manRet = static_cast<uint16_t>(mF >> (FP32_MAN_LEN - FP16_MAN_LEN));
        if (((mF >> (FP32_MAN_LEN - FP16_MAN_LEN - 1)) & 1) != 0) {
            manRet = static_cast<uint16_t>(manRet + 1);
        }
        if ((manRet & FP16_MAN_HIDE_BIT) != 0) {
            eRet = static_cast<uint16_t>(eRet + 1);
        }
    }
    return static_cast<uint16_t>((sRet << 15) | (eRet << FP16_MAN_LEN) | (manRet & FP16_MAX_MAN));
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

int32_t GenComplex32Data(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, float real,
                         float imag)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (size_t i = 0; i < shapes.size(); i++) {
        size *= static_cast<size_t>(shapes[i]);
    }
    size_t totalHalfs = size * 2; // COMPLEX32 = 2 × FP16
    size_t data_len = totalHalfs * sizeof(uint16_t);
    uint16_t* pData = new (std::nothrow) uint16_t[totalHalfs];
    if (pData == nullptr) {
        return FAILED;
    }
    const uint16_t realBits = FloatToFp16Bits(real);
    const uint16_t imagBits = FloatToFp16Bits(imag);
    for (size_t i = 0; i < size; i++) {
        // COMPLEX32 内存布局：[real_fp16, imag_fp16] 交错
        *(pData + i * 2) = realBits;
        *(pData + i * 2 + 1) = imagBits;
    }
    input_tensor = Tensor(input_tensor_desc, (uint8_t*)pData, data_len);
    delete[] pData; // Tensor 构造会拷贝数据，裸分配即用即还
    return SUCCESS;
}

int CreateOppInGraph(std::vector<ge::Tensor>& input, std::vector<Operator>& inputs, std::vector<Operator>& outputs,
                     Graph& graph)
{
    Status ret = SUCCESS;
    auto complexAbs = op::ComplexAbs("complex_abs");
    std::vector<int64_t> xShape = {4, 2};

    vector<int64_t> placeholder1_shape = xShape;
    auto placeholder1 = op::Data("placeholder").set_attr_index(0);
    TensorDesc placeholder1_desc = TensorDesc(ge::Shape(placeholder1_shape), FORMAT_ND, DT_COMPLEX32);
    placeholder1_desc.SetPlacement(ge::kPlacementHost);
    placeholder1_desc.SetFormat(FORMAT_ND);
    Tensor tensor_placeholder1;
    ret = GenComplex32Data(placeholder1_shape, tensor_placeholder1, placeholder1_desc, 1.0F, 2.0F);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());
        return FAILED;
    }
    placeholder1.update_input_desc_x(placeholder1_desc);
    placeholder1.update_output_desc_y(placeholder1_desc);
    input.push_back(tensor_placeholder1);
    graph.AddOp(placeholder1);
    complexAbs.set_input_x(placeholder1);
    inputs.push_back(placeholder1);

    outputs.push_back(complexAbs);
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
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    ret = CreateOppInGraph(input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        (void)ge::GEFinalize();
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session* session = new (std::nothrow) Session(build_options);

    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        (void)ge::GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Session add ir compute graph to ir session failed\n", GetTime().c_str());
        delete session;
        (void)ge::GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
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
        std::cout << "this is " << i << "th output, output element num =" << output_size << std::endl;
        uint8_t* output_data_i = output[i].GetData();
        if (output_data_i == nullptr || output_size < 1) {
            printf("%s - ERROR - [XIR]: Output %zu has no data\n", GetTime().c_str(), i);
            continue;
        }
        // COMPLEX32 输入时输出 dtype 为 DT_FLOAT16，按 fp16 位型转 float 后打印
        uint16_t* outputF16 = reinterpret_cast<uint16_t*>(output_data_i);
        // 每个复数元素为 (1.0, 2.0)，期望 |1.0 + 2.0i| = sqrt(5) ≈ 2.236
        size_t printNum = static_cast<size_t>(output_size) < 4 ? static_cast<size_t>(output_size) : 4;
        for (size_t j = 0; j < printNum; j++) {
            printf("result[%zu] is: %f (expected: sqrt(5) = 2.236)\n", j, Fp16BitsToFloat(outputF16[j]));
        }
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
