/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_is_inf.cpp
 * \brief
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_host/is_inf_tiling_def.h"
#include "tiling_case_executor.h"

extern "C" __global__ __aicore__ void is_inf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class IsInfTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "is_inf_test SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "is_inf_test TearDown" << std::endl; }
};

namespace {
constexpr uint8_t OUTPUT_POISON = 0xAA;
constexpr size_t DATA_BLOCK = 32;
constexpr uint32_t CORE_NUM = 20;
constexpr uint64_t UB_SIZE = 192 * 1024;

size_t CeilAlign(size_t value) { return (value + DATA_BLOCK - 1) / DATA_BLOCK * DATA_BLOCK; }

void RunNullWorkspaceCase(ge::DataType dataType, size_t elementBytes, uint32_t positiveInf, uint32_t negativeInf,
                          uint32_t nanValue, uint32_t finiteValue, int64_t dataCount)
{
    optiling::IsInfCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("IsInf", {{{{dataCount}, {dataCount}}, dataType, ge::FORMAT_ND}},
                                              {{{{dataCount}, {dataCount}}, ge::DT_BOOL, ge::FORMAT_ND}}, &compileInfo,
                                              CORE_NUM, UB_SIZE);
    TilingInfo tilingInfo;
    EXPECT_EQ(ExecuteTiling(tilingContextPara, tilingInfo), true);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    ASSERT_EQ(tilingInfo.workspaceSizes[0], 0U);
    ASSERT_GT(tilingInfo.tilingDataSize, 0U);

    const size_t inputBytes = static_cast<size_t>(dataCount) * elementBytes;
    const size_t outputBytes = static_cast<size_t>(dataCount);
    std::vector<uint8_t> input(inputBytes, 0);
    std::vector<uint8_t> expected(outputBytes, 0);
    for (int64_t i = 0; i < dataCount; ++i) {
        uint32_t value = finiteValue;
        if (i % 4 == 0) {
            value = positiveInf;
        } else if (i % 4 == 1) {
            value = negativeInf;
        } else if (i % 4 == 2) {
            value = nanValue;
        }
        if (elementBytes == sizeof(uint16_t)) {
            const uint16_t value16 = static_cast<uint16_t>(value);
            std::memcpy(input.data() + static_cast<size_t>(i) * elementBytes, &value16, sizeof(value16));
        } else {
            std::memcpy(input.data() + static_cast<size_t>(i) * elementBytes, &value, sizeof(value));
        }
        expected[static_cast<size_t>(i)] = i % 4 < 2 ? 1U : 0U;
    }

    uint8_t* x = static_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(inputBytes)));
    uint8_t* y = static_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(outputBytes)));
    uint8_t* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(CeilAlign(tilingInfo.tilingDataSize)));
    ASSERT_NE(x, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(tiling, nullptr);
    std::memcpy(x, input.data(), inputBytes);
    std::memset(y, OUTPUT_POISON, CeilAlign(outputBytes));
    std::memset(tiling, 0, CeilAlign(tilingInfo.tilingDataSize));
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);

    // Static ACLNN dispatch passes nullptr when the tiling workspace slot is zero.
    uint8_t* workspace = nullptr;
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(is_inf, tilingInfo.blockNum, x, y, workspace, tiling);

    size_t poisonCount = 0;
    size_t mismatchCount = 0;
    size_t firstMismatch = outputBytes;
    for (size_t i = 0; i < outputBytes; ++i) {
        poisonCount += y[i] == OUTPUT_POISON ? 1U : 0U;
        if (y[i] != expected[i]) {
            ++mismatchCount;
            if (firstMismatch == outputBytes) {
                firstMismatch = i;
            }
        }
    }
    EXPECT_EQ(poisonCount, 0U) << "kernel left the output untouched";
    EXPECT_EQ(mismatchCount, 0U) << "first mismatch at element " << firstMismatch;

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(tiling);
}
} // namespace

TEST_F(IsInfTest, test_case_float16_1)
{
    RunNullWorkspaceCase(ge::DT_FLOAT16, sizeof(uint16_t), 0x7c00U, 0xfc00U, 0x7e00U, 0x3c00U, 128 * 64);
}

TEST_F(IsInfTest, test_case_float32_1)
{
    RunNullWorkspaceCase(ge::DT_FLOAT, sizeof(float), 0x7f800000U, 0xff800000U, 0x7fc00000U, 0x3f800000U, 256 * 33);
}

TEST_F(IsInfTest, test_case_bfloat16_1)
{
    RunNullWorkspaceCase(ge::DT_BF16, sizeof(uint16_t), 0x7f80U, 0xff80U, 0x7fc0U, 0x3f80U, 256 * 33);
}

TEST_F(IsInfTest, test_case_float32_null_workspace_small)
{
    RunNullWorkspaceCase(ge::DT_FLOAT, sizeof(float), 0x7f800000U, 0xff800000U, 0x7fc00000U, 0x3f800000U, 8);
}
