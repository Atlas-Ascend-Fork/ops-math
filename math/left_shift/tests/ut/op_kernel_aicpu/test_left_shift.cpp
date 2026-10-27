/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"

#include <type_traits>

using namespace std;
using namespace aicpu;

// Align with torch.bitwise_left_shift.
template <typename T>
T LeftShiftExpect(T value, T shift)
{
    constexpr int64_t bit_width = static_cast<int64_t>(sizeof(T)) * CHAR_BIT;
    int64_t shift_bits = static_cast<int64_t>(static_cast<std::make_signed_t<T>>(shift));
    if (shift_bits < 0 || shift_bits >= bit_width) {
        return static_cast<T>(0);
    }
    return static_cast<T>(static_cast<std::make_unsigned_t<T>>(value) << shift_bits);
}

class TEST_LEFTSHIFT_UT : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "left_shift_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "left_shift_test TearDown\n" << endl; }
};

template <typename T>
void LeftCalcExpectWithDiffShape(const NodeDef& node_def, T expect_out[])
{
    auto input0 = node_def.MutableInputs(0);
    T* input0_data = (T*)input0->GetData();
    auto input1 = node_def.MutableInputs(1);
    T* input1_data = (T*)input1->GetData();
    int64_t input0_num = input0->NumElements();
    int64_t input1_num = input1->NumElements();

    if (input0_num > input1_num) {
        for (int64_t j = 0; j < input0_num; ++j) {
            int64_t i = j % input1_num;
            expect_out[j] = LeftShiftExpect(input0_data[j], input1_data[i]);
        }
    } else {
        for (int64_t j = 0; j < input1_num; ++j) {
            int64_t i = j % input0_num;
            expect_out[j] = LeftShiftExpect(input0_data[i], input1_data[j]);
        }
    }
}

#define CREATE_NODEDEF(shapes, data_types, datas)                    \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
    NodeDefBuilder(node_def.get(), "LeftShift", "LeftShift")         \
        .Input({"x", data_types[0], shapes[0], datas[0]})            \
        .Input({"y", data_types[1], shapes[1], datas[1]})            \
        .Output({"z", data_types[2], shapes[2], datas[2]})

TEST_F(TEST_LEFTSHIFT_UT, DATA_TYPE_INT8_SUCC)
{
    vector<DataType> data_types = {DT_INT8, DT_INT8, DT_INT8};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    const int8_t input1[6] = {1, 2, 3, 4, 5, 6};
    const int8_t input2[6] = {0, 1, 2, 3, 0, 1};
    int8_t output[6] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int8_t output_exp[6] = {1, 4, 12, 32, 5, 12};
    bool compare = CompareResult(output, output_exp, 6);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_LEFTSHIFT_UT, DATA_TYPE_UINT8_SUCC)
{
    vector<DataType> data_types = {DT_UINT8, DT_UINT8, DT_UINT8};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    const uint8_t input1[6] = {1, 2, 3, 4, 5, 6};
    const uint8_t input2[6] = {0, 1, 1, 2, 2, 0};
    uint8_t output[6] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    uint8_t output_exp[6] = {1, 4, 6, 16, 20, 6};
    bool compare = CompareResult(output, output_exp, 6);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_LEFTSHIFT_UT, DATA_TYPE_INT16_SUCC)
{
    vector<DataType> data_types = {DT_INT16, DT_INT16, DT_INT16};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    const int16_t input1[6] = {1, 2, 3, 4, 5, 6};
    const int16_t input2[6] = {0, 1, 2, 3, 4, 5};
    int16_t output[6] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int16_t output_exp[6] = {1, 4, 12, 32, 80, 192};
    bool compare = CompareResult(output, output_exp, 6);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_LEFTSHIFT_UT, DATA_TYPE_UINT16_SUCC)
{
    vector<DataType> data_types = {DT_UINT16, DT_UINT16, DT_UINT16};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    const uint16_t input1[6] = {1, 2, 3, 4, 5, 6};
    const uint16_t input2[6] = {1, 2, 3, 4, 5, 6};
    uint16_t output[6] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    uint16_t output_exp[6] = {2, 8, 24, 64, 160, 384};
    bool compare = CompareResult(output, output_exp, 6);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_LEFTSHIFT_UT, DATA_TYPE_INT32_SUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    const int32_t input1[6] = {1, 2, 3, 4, 5, 6};
    const int32_t input2[6] = {0, 1, 2, 3, 4, 5};
    int32_t output[6] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t output_exp[6] = {1, 4, 12, 32, 80, 192};
    bool compare = CompareResult(output, output_exp, 6);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_LEFTSHIFT_UT, DATA_TYPE_UINT32_SUCC)
{
    vector<DataType> data_types = {DT_UINT32, DT_UINT32, DT_UINT32};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    const uint32_t input1[6] = {1, 2, 3, 4, 5, 6};
    const uint32_t input2[6] = {0, 1, 2, 3, 4, 5};
    uint32_t output[6] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    uint32_t output_exp[6] = {1, 4, 12, 32, 80, 192};
    bool compare = CompareResult(output, output_exp, 6);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_LEFTSHIFT_UT, DATA_TYPE_INT64_SUCC)
{
    vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    const int64_t input1[6] = {1, 2, 3, 4, 5, 6};
    const int64_t input2[6] = {0, 1, 2, 3, 4, 5};
    int64_t output[6] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int64_t output_exp[6] = {1, 4, 12, 32, 80, 192};
    bool compare = CompareResult(output, output_exp, 6);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_LEFTSHIFT_UT, DATA_TYPE_UINT64_SUCC)
{
    vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_UINT64};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    const uint64_t input1[6] = {1, 2, 3, 4, 5, 6};
    const uint64_t input2[6] = {0, 1, 2, 3, 4, 5};
    uint64_t output[6] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    uint64_t output_exp[6] = {1, 4, 12, 32, 80, 192};
    bool compare = CompareResult(output, output_exp, 6);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_LEFTSHIFT_UT, DATA_TYPE_INT32_DIFFERENT_SHAPE_SUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 3}, {3}, {2, 3}};
    int32_t input1[6] = {1, 2, 3, 4, 5, 6};
    int32_t input2[3] = {1, 2, 0};
    int32_t output[6] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t output_exp[6] = {2, 8, 3, 8, 20, 6};
    LeftCalcExpectWithDiffShape<int32_t>(*node_def.get(), output_exp);

    bool compare = CompareResult(output, output_exp, 6);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_LEFTSHIFT_UT, DATA_TYPE_INT32_X_ONE_ELEMENT_SUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{1}, {3}, {3}};
    int32_t input1[1] = {3};
    int32_t input2[3] = {1, 2, 3};
    int32_t output[3] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t output_exp[3] = {6, 12, 24};
    LeftCalcExpectWithDiffShape<int32_t>(*node_def.get(), output_exp);

    bool compare = CompareResult(output, output_exp, 3);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_LEFTSHIFT_UT, DATA_TYPE_INT32_Y_ONE_ELEMENT_SUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{3}, {1}, {3}};
    int32_t input1[3] = {1, 2, 3};
    int32_t input2[1] = {2};
    int32_t output[3] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t output_exp[3] = {4, 8, 12};
    LeftCalcExpectWithDiffShape<int32_t>(*node_def.get(), output_exp);

    bool compare = CompareResult(output, output_exp, 3);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_LEFTSHIFT_UT, INPUT2_NEGATIVE_OR_GREATER)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
    int32_t input1[6] = {1, 1, 1, 1, 1, 1};
    int32_t input2[6] = {2, -2, 32, 0, 20, 40};
    int32_t output[6] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    // consistent with torch.bitwise_left_shift: negative shift or shift not less
    // than the bit width yields 0.
    int32_t output_exp[6] = {4, 0, 0, 1, 1 << 20, 0};

    bool compare = CompareResult(output, output_exp, 6);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_LEFTSHIFT_UT, LARGE_DATA_NOBCAST_SUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{64, 32}, {64, 32}, {64, 32}};

    constexpr uint64_t input1_size = 64 * 32;
    int32_t input1[input1_size];
    for (int64_t i = 0; i < input1_size; ++i) {
        input1[i] = static_cast<int32_t>((i + 100) % 1000);
    }

    constexpr uint64_t input2_size = 64 * 32;
    int32_t input2[input2_size];
    for (int64_t i = 0; i < input2_size; ++i) {
        input2[i] = static_cast<int32_t>((i + 1) % 4);
    }

    constexpr uint64_t output_size = 64 * 32;
    int32_t output[output_size] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t output_exp[output_size];
    for (int64_t i = 0; i < output_size; ++i) {
        output_exp[i] = input1[i] << input2[i];
    }

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_LEFTSHIFT_UT, LARGE_DATA_BCAST_SUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{64, 32}, {32}, {64, 32}};

    constexpr uint64_t input1_size = 64 * 32;
    int32_t input1[input1_size];
    for (int64_t i = 0; i < input1_size; ++i) {
        input1[i] = static_cast<int32_t>((i + 100) % 1000);
    }

    constexpr uint64_t input2_size = 32;
    int32_t input2[input2_size];
    for (int64_t i = 0; i < input2_size; ++i) {
        input2[i] = static_cast<int32_t>((i + 1) % 4);
    }

    constexpr uint64_t output_size = 64 * 32;
    int32_t output[output_size] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t output_exp[output_size];
    for (int64_t i = 0; i < output_size; ++i) {
        int64_t idx = i % input2_size;
        output_exp[i] = input1[i] << input2[idx];
    }

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
}

// exception instance
TEST_F(TEST_LEFTSHIFT_UT, INPUT_SHAPE_EXCEPTION)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 2, 4}, {2, 2, 3}, {2, 2, 4}};
    int32_t input1[16] = {(int32_t)1};
    int32_t input2[12] = {(int32_t)0};
    int32_t output[16] = {(int32_t)0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_LEFTSHIFT_UT, INPUT_DTYPE_EXCEPTION)
{
    vector<DataType> data_types = {DT_INT32, DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
    int32_t input1[22] = {(int32_t)1};
    int64_t input2[22] = {(int64_t)0};
    int64_t output[22] = {(int64_t)0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_LEFTSHIFT_UT, INPUT_NULL_EXCEPTION)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
    int64_t output[22] = {(int64_t)0};
    vector<void*> datas = {(void*)nullptr, (void*)nullptr, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_LEFTSHIFT_UT, INPUT_BOOL_UNSUPPORT)
{
    vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL};
    vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
    bool input1[22] = {(bool)1};
    bool input2[22] = {(bool)0};
    bool output[22] = {(bool)0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
