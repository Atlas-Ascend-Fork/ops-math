/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_case_executor.h"
#include "../../../op_kernel/fused_exponential_tiling_data.h"
#include "../../../op_host/fused_exponential_utils.h"

namespace {
struct CompileInfo {};
gert::TilingContextPara MakeContext(int64_t length, ge::DataType dtype, CompileInfo* info, uint32_t cores = 40,
                                    uint64_t ubBytes = 192 * 1024, bool zero = false, int64_t exponent = -8)
{
    gert::StorageShape shape = {{length}, {length}};
    return gert::TilingContextPara(
        "FusedExponential", {{shape, dtype, ge::FORMAT_ND}}, {{shape, dtype, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("scale", Ops::Math::AnyValue::CreateFrom<float>(-0.75f)),
         gert::TilingContextPara::OpAttr("exponent", Ops::Math::AnyValue::CreateFrom<int64_t>(exponent)),
         gert::TilingContextPara::OpAttr("zero", Ops::Math::AnyValue::CreateFrom<bool>(zero))},
        info, cores, ubBytes);
}

class FusedExponentialTilingBoundary : public testing::TestWithParam<int64_t> {};
TEST_P(FusedExponentialTilingBoundary, CoversInputWithoutOverlapOrExtraWorkspace)
{
    CompileInfo compileInfo;
    const auto length = GetParam();
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}) {
        auto context = MakeContext(length, dtype, &compileInfo);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(context, info));
        ASSERT_EQ(info.tilingDataSize, sizeof(FusedExponentialTilingData));
        FusedExponentialTilingData data{};
        std::memcpy(&data, info.tilingData.get(), sizeof(data));
        EXPECT_EQ(data.length, length);
        ASSERT_GE(info.blockNum, 1U);
        ASSERT_LE(info.blockNum, 40U);
        EXPECT_EQ(data.coreLength % 64, 0U);
        EXPECT_GE(data.coreLength * info.blockNum, data.length);
        if (length > 0) {
            EXPECT_LT(data.coreLength * (info.blockNum - 1), data.length);
        }
        EXPECT_GT(data.tileLength, 0U);
        EXPECT_EQ(data.tileLength % 64, 0U);
        EXPECT_LE(data.tileLength * 32ULL + 8192, 192 * 1024ULL);
        EXPECT_EQ(data.scale, -0.75f);
        EXPECT_EQ(data.exponent, -8);
        EXPECT_EQ(data.zero, 0U);
        const float epsilon = dtype == ge::DT_FLOAT16 ? 0x1p-11f : (dtype == ge::DT_BF16 ? 0x1p-8f : 0x1p-24f);
        EXPECT_EQ(data.epsilon, epsilon);
        EXPECT_EQ(info.tilingKey, 0U);
        ASSERT_EQ(info.workspaceSizes.size(), 1U);
        EXPECT_EQ(info.workspaceSizes[0], 0U);
    }
}
INSTANTIATE_TEST_SUITE_P(Edges, FusedExponentialTilingBoundary,
                         testing::Values(0, 1, 63, 64, 65, 1023, 1024, 1025, 4095, 4096, 4097, 1000000));

TEST(FusedExponentialTiling, HonorsPlatformAndInfinityAttribute)
{
    CompileInfo compileInfo;
    auto context = MakeContext(1000000, ge::DT_FLOAT, &compileInfo, 8, 32 * 1024, true);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    FusedExponentialTilingData data{};
    ASSERT_EQ(info.tilingDataSize, sizeof(data));
    std::memcpy(&data, info.tilingData.get(), sizeof(data));
    EXPECT_EQ(info.blockNum, 8U);
    EXPECT_EQ(data.zero, 1U);
    EXPECT_LE(data.tileLength * 32ULL + 8192, 32 * 1024ULL);
}

TEST(FusedExponentialTiling, RejectsInsufficientUb)
{
    CompileInfo compileInfo;
    auto context = MakeContext(128, ge::DT_FLOAT, &compileInfo, 40, 8 * 1024);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}

TEST(FusedExponentialTiling, RejectsNegativeAndOverflowingShape)
{
    CompileInfo compileInfo;
    for (int64_t length : std::vector<int64_t>{-1, std::numeric_limits<int64_t>::max()}) {
        auto context = MakeContext(length, ge::DT_FLOAT, &compileInfo);
        TilingInfo info;
        EXPECT_FALSE(ExecuteTiling(context, info));
    }
}

TEST(FusedExponentialTiling, RejectsUnsupportedDtypeAndExponent)
{
    CompileInfo compileInfo;
    auto invalidType = MakeContext(65, ge::DT_INT32, &compileInfo);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(invalidType, info));
    for (int64_t exponent : std::vector<int64_t>{-1025, 1074, std::numeric_limits<int64_t>::max()}) {
        auto context = MakeContext(65, ge::DT_FLOAT, &compileInfo, 40, 192 * 1024, false, exponent);
        EXPECT_FALSE(ExecuteTiling(context, info));
    }
}

TEST(FusedExponentialTiling, AcceptsExtremeDoubleExponents)
{
    CompileInfo compileInfo;
    for (int64_t exponent : {-1024, 1073}) {
        auto context = MakeContext(65, ge::DT_FLOAT, &compileInfo, 40, 192 * 1024, false, exponent);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(context, info));
        FusedExponentialTilingData data{};
        std::memcpy(&data, info.tilingData.get(), sizeof(data));
        EXPECT_EQ(data.exponent, exponent);
    }
}

TEST(FusedExponentialShapeArithmetic, RejectsProductOverflowBeforeMultiplication)
{
    uint64_t count = 123;
    gert::Shape overflow = {std::numeric_limits<int64_t>::max(), 2};
    EXPECT_FALSE(fused_exponential::GetElementCount(overflow, 4, count));
    gert::Shape empty = {std::numeric_limits<int64_t>::max(), 2, 0};
    EXPECT_TRUE(fused_exponential::GetElementCount(empty, 4, count));
    EXPECT_EQ(count, 0U);
    gert::Shape invalidEmpty = {0, -1};
    EXPECT_FALSE(fused_exponential::GetElementCount(invalidEmpty, 4, count));
    gert::Shape scalar = {};
    EXPECT_TRUE(fused_exponential::GetElementCount(scalar, 4, count));
    EXPECT_EQ(count, 1U);
}
} // namespace
