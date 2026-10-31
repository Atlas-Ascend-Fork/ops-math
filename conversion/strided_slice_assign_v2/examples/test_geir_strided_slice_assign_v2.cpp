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

#include "experiment_ops.h"
#include "nn_other.h"
#include "../op_graph/strided_slice_assign_v2_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;
// #intputIndex 为字符串化，生成 "placeholder1" 等合法 op 名（用 + 会变成指针算术）；
// Data op 需同时设置 input / output desc。
// fillValue 为全张量填充值：var 与 input_value 取不同值才能区分"切片被写入"与"kernel 空跑"。
// 注意 GenOnesData 的 fp16/bf16 分支只支持 1.0 / 2.0 两种位编码。
#define ADD_INPUT(intputIndex, intputName, intputDtype, inputShape, fillValue)                                      \
    vector<int64_t> placeholder##intputIndex##_shape = inputShape;                                                  \
    auto placeholder##intputIndex = op::Data("placeholder" #intputIndex).set_attr_index(0);                         \
    TensorDesc placeholder##intputIndex##_desc = TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, \
                                                            intputDtype);                                           \
    placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost);                                               \
    placeholder##intputIndex##_desc.SetFormat(FORMAT_ND);                                                           \
    Tensor tensor_placeholder##intputIndex;                                                                         \
    ret = GenOnesData(placeholder##intputIndex##_shape, tensor_placeholder##intputIndex,                            \
                      placeholder##intputIndex##_desc, intputDtype, fillValue);                                     \
    if (ret != SUCCESS) {                                                                                           \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                              \
        return FAILED;                                                                                              \
    }                                                                                                               \
    placeholder##intputIndex.update_input_desc_x(placeholder##intputIndex##_desc);                                  \
    placeholder##intputIndex.update_output_desc_y(placeholder##intputIndex##_desc);                                 \
    input.push_back(tensor_placeholder##intputIndex);                                                               \
    graph.AddOp(placeholder##intputIndex);                                                                          \
    add1.set_input_##intputName(placeholder##intputIndex);                                                          \
    inputs.push_back(placeholder##intputIndex);

// begin / end / strides / axes 在 OpDef 中声明为 .ValueDepend(REQUIRED)（见 op_host/..._def.cpp），
// 即取值参与 tiling / infershape，GE 要求必须挂在 Const 节点上；用 Data 占位会直接
// "assign engine failed: value depend of input[begin] is required, but this input is not
// linked to a const or constant node"，故这三个宏只能走 Const。
// 逐元素取值不同，长度须等于 shape 元素总数，故走 GenInt64Data。
// 调用处传初始化列表时必须用圆括号包裹，如 (std::vector<int64_t>{1, 0})，
// 否则列表内的逗号会被预处理器当作宏参数分隔符。
#define ADD_CONST_INPUT(intputIndex, intputName, intputDtype, inputShape, values)                                   \
    vector<int64_t> placeholder##intputIndex##_shape = inputShape;                                                  \
    auto placeholder##intputIndex = op::Const("placeholder" #intputIndex);                                          \
    TensorDesc placeholder##intputIndex##_desc = TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, \
                                                            intputDtype);                                           \
    placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost);                                               \
    placeholder##intputIndex##_desc.SetFormat(FORMAT_ND);                                                           \
    Tensor tensor_placeholder##intputIndex;                                                                         \
    ret = GenInt64Data(placeholder##intputIndex##_shape, tensor_placeholder##intputIndex,                           \
                       placeholder##intputIndex##_desc, values);                                                    \
    if (ret != SUCCESS) {                                                                                           \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                              \
        return FAILED;                                                                                              \
    }                                                                                                               \
    placeholder##intputIndex.SetAttr("value", tensor_placeholder##intputIndex);                                     \
    placeholder##intputIndex.update_output_desc_y(placeholder##intputIndex##_desc);                                 \
    graph.AddOp(placeholder##intputIndex);                                                                          \
    add1.set_input_##intputName(placeholder##intputIndex);                                                          \
    add1.update_input_desc_##intputName(placeholder##intputIndex##_desc);                                           \
    inputs.push_back(placeholder##intputIndex);

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                       \
    TensorDesc outputName##outputIndex##_desc = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
    add1.update_output_desc_##outputName(outputName##outputIndex##_desc);

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
    }
    return dilation;
}

int32_t GenOnesDataFloat32(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, float value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t byteSizeFloat32 = 4;
    uint32_t data_len = size * byteSizeFloat32;
    float* pData = new (std::nothrow) float[size];

    for (size_t i = 0; i < size; ++i) {
        *(pData + i) = value;
    }
    input_tensor = Tensor(input_tensor_desc, (uint8_t*)pData, data_len);
    return SUCCESS;
}

int32_t GenOnesData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type,
                    int value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    // 按元素类型分配并完整初始化 size 个元素，data_len 为实际字节数。
    // 原实现按字节数分配 int32 数组且只写 size 个 int32：DT_INT64 上半内存未初始化，
    // DT_FLOAT/FP16/BF16 写入的是 int32 位模式而非真实数值。
    switch (data_type) {
        case DT_INT64: {
            int64_t* pData = new (std::nothrow) int64_t[size];
            if (pData == nullptr) {
                printf("%s - ERROR - [XIR]: Alloc int64 data failed (size=%zu)\n", GetTime().c_str(), size);
                return FAILED;
            }
            for (size_t i = 0; i < size; ++i) {
                pData[i] = value;
            }
            input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), size * sizeof(int64_t));
            return SUCCESS;
        }
        case DT_INT32: {
            int32_t* pData = new (std::nothrow) int32_t[size];
            if (pData == nullptr) {
                printf("%s - ERROR - [XIR]: Alloc int32 data failed (size=%zu)\n", GetTime().c_str(), size);
                return FAILED;
            }
            for (size_t i = 0; i < size; ++i) {
                pData[i] = value;
            }
            input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), size * sizeof(int32_t));
            return SUCCESS;
        }
        case DT_INT16: {
            int16_t* pData = new (std::nothrow) int16_t[size];
            if (pData == nullptr) {
                printf("%s - ERROR - [XIR]: Alloc int16 data failed (size=%zu)\n", GetTime().c_str(), size);
                return FAILED;
            }
            for (size_t i = 0; i < size; ++i) {
                pData[i] = static_cast<int16_t>(value);
            }
            input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), size * sizeof(int16_t));
            return SUCCESS;
        }
        case DT_FLOAT: {
            float* pData = new (std::nothrow) float[size];
            if (pData == nullptr) {
                printf("%s - ERROR - [XIR]: Alloc float data failed (size=%zu)\n", GetTime().c_str(), size);
                return FAILED;
            }
            for (size_t i = 0; i < size; ++i) {
                pData[i] = static_cast<float>(value);
            }
            input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), size * sizeof(float));
            return SUCCESS;
        }
        case DT_FLOAT16:
        case DT_BF16: {
            // 1.0 / 2.0 的位编码：fp16 = 0x3C00 / 0x4000，bf16 = 0x3F80 / 0x4000（2.0 恰好相同）
            if (value != 1 && value != 2) {
                printf("%s - ERROR - [XIR]: GenOnesData fp16/bf16 only supports value=1 or 2\n", GetTime().c_str());
                return FAILED;
            }
            const uint16_t bits = (data_type == DT_FLOAT16) ? ((value == 1) ? 0x3C00 : 0x4000) :
                                                              ((value == 1) ? 0x3F80 : 0x4000);
            uint16_t* pData = new (std::nothrow) uint16_t[size];
            if (pData == nullptr) {
                printf("%s - ERROR - [XIR]: Alloc fp16/bf16 data failed (size=%zu)\n", GetTime().c_str(), size);
                return FAILED;
            }
            for (size_t i = 0; i < size; ++i) {
                pData[i] = bits;
            }
            input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), size * sizeof(uint16_t));
            return SUCCESS;
        }
        default:
            printf("%s - ERROR - [XIR]: GenOnesData unsupported dtype=%d\n", GetTime().c_str(),
                   static_cast<int>(data_type));
            return FAILED;
    }
}

// 逐元素填充 int64 常量：begin/end/strides/axes 各元素取值不同，GenOnesData 只能填同一值，不适用。
int32_t GenInt64Data(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc,
                     const std::vector<int64_t>& values)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    if (values.size() != size) {
        printf("%s - ERROR - [XIR]: int64 data count mismatch (need %zu, got %zu)\n", GetTime().c_str(), size,
               values.size());
        return FAILED;
    }
    int64_t* pData = new (std::nothrow) int64_t[size];
    if (pData == nullptr) {
        printf("%s - ERROR - [XIR]: Alloc int64 data failed (size=%zu)\n", GetTime().c_str(), size);
        return FAILED;
    }
    for (size_t i = 0; i < size; ++i) {
        pData[i] = values[i];
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), size * sizeof(int64_t));
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t* inputData)
{
    FILE* fp;
    fp = fopen(bin_file.c_str(), "w");
    fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    // 自定义代码：添加单算子定义到图中
    auto add1 = op::StridedSliceAssignV2("add1");

    // 规模约束：arch35 当前仅实现 T3 Scatter（KEY 30/31）。Host 侧 S0-7 判据②规定
    // sliceElemNum <= SSA_SMALL_SHAPE_THRESHOLD(1024) 时改判 KEY 10（T1 Hybrid），
    // 而 T1 的 kernel 分支已下线为空占位 ⇒ 切片静默不写、输出与 var 完全相同。
    // 故本用例的切片元素数须 > 1024，且最内维步长非 1（否则命中判据④走 T2，同样已下线）。
    std::vector<int64_t> varRefShape = {4, 1024};
    std::vector<int64_t> inputValueShape = {3, 512};
    std::vector<int64_t> attrShape = {2};

    // var 填 1.0、input_value 填 2.0：切片区域 var[1:4:1, 0:1024:2] 被写成 2.0，其余位置保持 1.0，
    // 输出因此可区分"写入成功"与"kernel 空跑"。
    ADD_INPUT(1, var, inDtype, varRefShape, 1);
    ADD_INPUT(2, input_value, inDtype, inputValueShape, 2);

    // begin / end / strides / axes 在 OpDef 中声明为 idxDataType = DT_INT64 且 ValueDepend(REQUIRED)
    // （见 op_host/..._def.cpp），故必须以 DT_INT64 Const 常量形式提供：
    //   · DT_INT32 会被 OpDef 的 dtype 校验直接拒绝；
    //   · Data 占位会因 "value depend ... not linked to a const" 选不到引擎。
    // 值列表用圆括号包裹，避免初始化列表内的逗号被预处理器当成宏参数分隔符。
    ADD_CONST_INPUT(3, begin, DT_INT64, attrShape, (std::vector<int64_t>{1, 0}));
    ADD_CONST_INPUT(4, end, DT_INT64, attrShape, (std::vector<int64_t>{4, 1024}));
    ADD_CONST_INPUT(5, strides, DT_INT64, attrShape, (std::vector<int64_t>{1, 2}));
    ADD_CONST_INPUT(6, axes, DT_INT64, attrShape, (std::vector<int64_t>{0, 1}));

    ADD_OUTPUT(1, var, inDtype, varRefShape);
    outputs.push_back(add1);
    // 添加完毕
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

    if (argc > 1) {
        std::cout << argv[1] << std::endl;
    }
    char* endptr;

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
        WriteDataToFile((const char*)input_file.c_str(), data_size, input_data_i);
    }

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << output_shape << std::endl;
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)output_file.c_str(), data_size, output_data_i);
        uint8_t* resultData = (uint8_t*)output_data_i;
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
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
