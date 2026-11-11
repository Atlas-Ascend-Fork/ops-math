/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <cmath>
#include <limits>
#include <filesystem>
#include <type_traits>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

#include "../../../op_kernel/atan2.cpp"

using namespace std;

template <typename T>
__global__ __aicore__ void Atan2BoundaryKernel(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(Atan2TilingData);
    GET_TILING_DATA_WITH_STRUCT(Atan2TilingData, tilingData, tiling);
    NsAtan2::Atan2<T> op;
    op.Init(x1, x2, y, &tilingData);
    op.Process();
}

template <typename T>
void CheckZeroBoundaries(float tolerance, uint32_t dataCount = 160, uint32_t inputOffset = 0, uint32_t tileSize = 64)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::array<std::array<float, 2>, 36> inputs = {
        {{0.0f, -1.0f}, {-0.0f, -1.0f}, {0.0f, 1.0f},     {-0.0f, 1.0f},     {1.0f, 0.0f},      {-1.0f, 0.0f},
         {1.0f, -0.0f}, {-1.0f, -0.0f}, {0.0f, 0.0f},     {-0.0f, 0.0f},     {0.0f, -0.0f},     {-0.0f, -0.0f},
         {1.0f, 1.0f},  {-1.0f, 1.0f},  {1.0f, -1.0f},    {-1.0f, -1.0f},    {2.0f, 3.0f},      {-2.0f, 3.0f},
         {2.0f, -3.0f}, {-2.0f, -3.0f}, {1e-36f, 2e-36f}, {-1e-36f, 2e-36f}, {1e-36f, -2e-36f}, {-1e-36f, -2e-36f},
         {nan, 0.0f},   {0.0f, nan},    {nan, 1.0f},      {1.0f, nan},       {-nan, 0.0f},      {0.0f, -nan},
         {-nan, 1.0f},  {1.0f, -nan},   {nan, nan},       {-nan, nan},       {nan, -nan},       {-nan, -nan}}};
    const size_t bytes = dataCount * sizeof(T);
    T* x1 = reinterpret_cast<T*>(AscendC::GmAlloc(bytes));
    T* x2 = reinterpret_cast<T*>(AscendC::GmAlloc(bytes));
    T* output = reinterpret_cast<T*>(AscendC::GmAlloc(bytes));
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    ASSERT_NE(output, nullptr);
    Atan2TilingData tiling{};
    tiling.smallCoreDataNum = dataCount;
    tiling.bigCoreDataNum = dataCount;
    tiling.finalBigTileNum = (dataCount + tileSize - 1) / tileSize;
    tiling.finalSmallTileNum = (dataCount + tileSize - 1) / tileSize;
    tiling.tileDataNum = tileSize;
    tiling.smallTailDataNum = dataCount % tileSize;
    tiling.bigTailDataNum = dataCount % tileSize;
    tiling.tailBlockNum = 1;
    tiling.atanTmpSize = 8 * tileSize * sizeof(float);
    auto* tilingBuffer = reinterpret_cast<Atan2TilingData*>(AscendC::GmAlloc(sizeof(tiling)));
    ASSERT_NE(tilingBuffer, nullptr);
    *tilingBuffer = tiling;
    for (uint32_t i = 0; i < dataCount; ++i) {
        x1[i] = static_cast<T>(inputs[(i + inputOffset) % inputs.size()][0]);
        x2[i] = static_cast<T>(inputs[(i + inputOffset) % inputs.size()][1]);
        output[i] = static_cast<T>(-123.0f);
    }

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = Atan2BoundaryKernel<T>;
    ICPU_RUN_KF(func, 1, reinterpret_cast<uint8_t*>(x1), reinterpret_cast<uint8_t*>(x2),
                reinterpret_cast<uint8_t*>(output), reinterpret_cast<uint8_t*>(tilingBuffer));
    for (uint32_t i = 0; i < dataCount; ++i) {
        const float numerator = static_cast<float>(x1[i]);
        const float denominator = static_cast<float>(x2[i]);
        // The operator documents a canonical zero for all origins.
        const float expected = (numerator == 0.0f && denominator == 0.0f) ? 0.0f : std::atan2(numerator, denominator);
        const float actual = static_cast<float>(output[i]);
        SCOPED_TRACE(i);
        if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(actual));
        } else {
            EXPECT_NEAR(actual, expected, tolerance);
            if (expected == 0.0f) {
                EXPECT_EQ(std::signbit(actual), std::signbit(expected));
            }
        }
    }
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(output);
    AscendC::GmFree(tilingBuffer);
}

template <typename T>
void CheckDefaultUbTile(float tolerance)
{
    // Match the host budget against the active CPU model, including the
    // Select API reservation, then exercise a full tile and a short tail.
    constexpr uint32_t ubFactor = std::is_same<T, float>::value ? 19 : 16;
    constexpr uint32_t tileSize = ((AscendC::TOTAL_UB_SIZE - 8 * 1024) / sizeof(float) / ubFactor / 64) * 64;
    CheckZeroBoundaries<T>(tolerance, tileSize + 32, 0, tileSize);
}

TEST(Atan2ZeroBoundaryTest, float32)
{
    CheckZeroBoundaries<float>(1e-4f);
    CheckZeroBoundaries<float>(1e-4f, 8);
    CheckZeroBoundaries<float>(1e-4f, 8, 24);
    CheckDefaultUbTile<float>(1e-4f);
}

TEST(Atan2ZeroBoundaryTest, float16)
{
    CheckZeroBoundaries<half>(5e-3f);
    CheckZeroBoundaries<half>(5e-3f, 16);
    CheckZeroBoundaries<half>(5e-3f, 16, 24);
    CheckDefaultUbTile<half>(5e-3f);
}

TEST(Atan2ZeroBoundaryTest, bfloat16)
{
    CheckZeroBoundaries<bfloat16_t>(2e-2f);
    CheckZeroBoundaries<bfloat16_t>(2e-2f, 16);
    CheckZeroBoundaries<bfloat16_t>(2e-2f, 16, 24);
    CheckDefaultUbTile<bfloat16_t>(2e-2f);
}

class Atan2Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "atan2_test SetUp" << std::endl;
        // The build may shorten __FILE__. CMake supplies the fixture path
        // independently of both compiler path rewriting and the working directory.
        const std::filesystem::path sourceData(ATAN2_TEST_DATA_DIR);
        std::error_code error;
        std::filesystem::copy(
            sourceData, "atan2_data",
            std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing, error);
        ASSERT_FALSE(error) << "Cannot copy Atan2 fixtures from " << sourceData << ": " << error.message();
    }
    static void TearDownTestCase() { std::cout << "atan2_test TearDown" << std::endl; }
};

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

TEST_F(Atan2Test, test_case_float16_1)
{
    uint32_t blockDim = 40;
    // 调用 python 生成测试数据
    ASSERT_EQ(system("cd ./atan2_data/ && python3 gen_data.py '(128, 64)' 'float16'"), 0);
    uint32_t dataCount = 128 * 64;
    size_t inputByteSize = dataCount * sizeof(half);

    std::string x1_fileName = "./atan2_data/float16_input_x1_t_atan2.bin";
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 256));
    ASSERT_NE(x1, nullptr);
    ASSERT_TRUE(ReadFile(x1_fileName, inputByteSize, x1, inputByteSize));

    std::string x2_fileName = "./atan2_data/float16_input_x2_t_atan2.bin";
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 256));
    ASSERT_NE(x2, nullptr);
    ASSERT_TRUE(ReadFile(x2_fileName, inputByteSize, x2, inputByteSize));

    size_t outputByteSize = dataCount * sizeof(half);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 256));
    ASSERT_NE(y, nullptr);

    size_t workspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(Atan2TilingData));
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    Atan2TilingData* tilingData = reinterpret_cast<Atan2TilingData*>(tiling);

    // Split all 8192 elements into 32 big cores and 8 small cores, using
    // a small tile to stay within the CPU model's UB capacity.
    tilingData->smallCoreDataNum = 192;
    tilingData->bigCoreDataNum = 208;
    tilingData->finalBigTileNum = 1;
    tilingData->finalSmallTileNum = 1;
    tilingData->tileDataNum = 256;
    tilingData->smallTailDataNum = 192;
    tilingData->bigTailDataNum = 208;
    tilingData->tailBlockNum = 32;
    tilingData->atanTmpSize = 8192;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = atan2<ELEMENTWISE_TPL_SCH_MODE_0>;
    ICPU_RUN_KF(func, blockDim, x1, x2, y, workspace, (uint8_t*)(tilingData));

    std::string fileName = "./atan2_data/float16_output_t_atan2.bin";
    ASSERT_TRUE(WriteFile(fileName, y, outputByteSize));

    AscendC::GmFree((void*)(x1));
    AscendC::GmFree((void*)(x2));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    // 调用 python 进行精度比对
    ASSERT_EQ(system("cd ./atan2_data/ && python3 compare_data.py 'float16'"), 0);
}
