/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"
#include <cstdint>
#include <limits>

// 门控实现位于 L2 的 cpp 中；测试直接链接同一实现，不再依赖独立策略头文件。
namespace SelectTopKPolicy {
bool IsRadixShapeSupported(int64_t axis, int64_t k);
}

namespace {
constexpr int64_t FIRST_AXIS_BOUNDARY = 12000000;
constexpr int64_t SECOND_AXIS_BOUNDARY = 100000000;
constexpr int64_t FIRST_RATIO_BOUNDARY_K = 120000;
constexpr int64_t SECOND_RATIO_BOUNDARY_K = 2000000;
constexpr int64_t FIRST_SUPPORTED_K = 8;
using SelectTopKPolicy::IsRadixShapeSupported;

// PR 14011 的下限包含 K=8；K=7 和空输出不能误入 Radix。
TEST(SelectTopKRadixPolicy, IncludesNewKBoundary)
{
    EXPECT_TRUE(IsRadixShapeSupported(FIRST_AXIS_BOUNDARY, FIRST_SUPPORTED_K));
    EXPECT_FALSE(IsRadixShapeSupported(FIRST_AXIS_BOUNDARY, FIRST_SUPPORTED_K - 1));
    EXPECT_FALSE(IsRadixShapeSupported(FIRST_AXIS_BOUNDARY, 0));
}

// 第一档轴长包含边界，但比例等号必须留给 SelectTopK。
TEST(SelectTopKRadixPolicy, KeepsFirstAxisAndRatioBoundaries)
{
    EXPECT_FALSE(IsRadixShapeSupported(FIRST_AXIS_BOUNDARY - 1, FIRST_SUPPORTED_K));
    EXPECT_FALSE(IsRadixShapeSupported(FIRST_AXIS_BOUNDARY, FIRST_RATIO_BOUNDARY_K));
    EXPECT_TRUE(IsRadixShapeSupported(FIRST_AXIS_BOUNDARY, FIRST_RATIO_BOUNDARY_K - 1));
}

// 第二档从 N=1e8 开始采用更宽的比例，不能提前套用。
TEST(SelectTopKRadixPolicy, SwitchesRatioAtSecondAxisBoundary)
{
    EXPECT_FALSE(IsRadixShapeSupported(SECOND_AXIS_BOUNDARY - 1, SECOND_RATIO_BOUNDARY_K - 1));
    EXPECT_TRUE(IsRadixShapeSupported(SECOND_AXIS_BOUNDARY, SECOND_RATIO_BOUNDARY_K - 1));
    EXPECT_FALSE(IsRadixShapeSupported(SECOND_AXIS_BOUNDARY, SECOND_RATIO_BOUNDARY_K));
}

// 本接口仍保留原有 INT32 轴长上限，不借门控变更扩大公开能力。
TEST(SelectTopKRadixPolicy, RetainsAxisAndPaddingLimits)
{
    constexpr int64_t MAX_AXIS = std::numeric_limits<int32_t>::max();
    EXPECT_TRUE(IsRadixShapeSupported(MAX_AXIS, FIRST_SUPPORTED_K));
    EXPECT_FALSE(IsRadixShapeSupported(MAX_AXIS + 1, FIRST_SUPPORTED_K));
    EXPECT_FALSE(IsRadixShapeSupported(FIRST_AXIS_BOUNDARY, FIRST_AXIS_BOUNDARY + 1));
    EXPECT_FALSE(IsRadixShapeSupported(-1, FIRST_SUPPORTED_K));
}
} // namespace
