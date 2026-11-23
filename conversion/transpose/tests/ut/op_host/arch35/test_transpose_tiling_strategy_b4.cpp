/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_transpose_tiling_strategy_b4.cpp
 * \brief B4(10009) 策略 UT：直调策略（逐字段断言）+ 端到端命中
 */

#include "test_transpose_tiling_strategy_common.h"
#include "../../../../op_host/arch35/transpose_tiling_strategy_b4.h"

using namespace transpose_strategy_ut;

namespace {
gert::TilingContextPara MakePara(optiling::TransposeCompilerInfo& compileInfo)
{
    static int64_t permValue[4] = {1, 0, 3, 2};
    gert::TilingContextPara::TensorDescription x({{36, 204, 26, 30}, {36, 204, 26, 30}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &permValue);
    gert::TilingContextPara::TensorDescription out({{204, 36, 30, 26}, {204, 36, 30, 26}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    return gert::TilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
}
} // namespace

TEST(TransposeB4Strategy, e2e_hits_this_strategy)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    TilingInfo e2e;
    ASSERT_TRUE(ExecuteTiling(para, e2e));
    EXPECT_EQ(e2e.tilingKey, TK_B4);
    EXPECT_EQ(e2e.blockNum, 48);
    const auto* data = GetTilingData<TransposeB4TilingData>(e2e);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->permSize, 4);
}

// 单元覆盖：极小 4D 4bit shape
TEST(TransposeB4Strategy, direct_tiny_shape)
{
    auto compileInfo = MakeCompilerInfo();
    int64_t permValue[4] = {1, 0, 3, 2};
    gert::TilingContextPara::TensorDescription x({{2, 2, 4, 8}, {2, 2, 4, 8}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &permValue);
    gert::TilingContextPara::TensorDescription out({{2, 2, 8, 4}, {2, 2, 8, 4}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara para("Transpose", {x, perm}, {out}, &compileInfo);
    ShapeInfo si = MakeShape({2, 2, 4, 8}, {1, 0, 3, 2}, 1004);

    auto bundle = BuildTilingContext(para);
    optiling::TransposeB4Strategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_B4));
    EXPECT_EQ(bundle.context->GetBlockDim(), 1);
}

// B4 独立模板失败（workspace 缺失）→ 策略硬失败（不 fallback）
TEST(TransposeB4Strategy, direct_failure_is_hard)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    ShapeInfo si = MakeShape({2, 2, 4, 8}, {1, 0, 3, 2}, ge::kDataTypeSizeBitOffset + 4);
    auto bundle = BuildTilingContext(para, false);
    optiling::TransposeB4Strategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    EXPECT_EQ(strategy.Execute(), TilingStatus::FAILURE);
}

/* ============================================================================
 * B4(10009)：超 uint32 shape + 10×ubSize UB 占用
 * ========================================================================== */
TEST(TransposeB4Strategy, b4_large_shape_and_ub_budget)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeDummyPara(compileInfo);
    const int64_t eleLen = ge::kDataTypeSizeBitOffset + 4;
    const std::vector<int64_t> in = {8008, 77560, 2, 4, 6};
    const std::vector<int64_t> perm = {3, 2, 1, 4, 0};
    ShapeInfo si = MakeShape(in, perm, eleLen);

    auto bundle = BuildTilingContext(para);
    optiling::TransposeB4Strategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_B4));
    const auto* data = GetTilingData<TransposeB4TilingData>(bundle.context);
    ASSERT_NE(data, nullptr);

    EXPECT_EQ(data->permSize, 5);
    // 超大 shape 的跨步必须用 int64 完整保留（不截断到 uint32）。
    bool hasAboveUint32 = false;
    for (int64_t stride : data->gmInStride) {
        hasAboveUint32 = hasAboveUint32 || stride > static_cast<int64_t>(kUint32Max);
    }
    for (int64_t stride : data->gmOutStride) {
        hasAboveUint32 = hasAboveUint32 || stride > static_cast<int64_t>(kUint32Max);
    }
    for (int64_t stride : data->moveUbStride) {
        hasAboveUint32 = hasAboveUint32 || stride > static_cast<int64_t>(kUint32Max);
    }
    EXPECT_TRUE(hasAboveUint32);

    // kernel：TQue 2×ubSize + cast/transpose 两个 TBuf 各 4×ubSize。
    EXPECT_GT(data->ubSize, 0);
    EXPECT_EQ(data->ubSize % kUbBlockSize, 0);
    EXPECT_LE(static_cast<int64_t>(data->ubSize) * 10, kUbSize);
}

// 边界：仅 4bit 且尾轴转置命中
TEST(TransposeB4Strategy, b4_dtype_and_last_axis_boundary)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeDummyPara(compileInfo);
    auto bundle = BuildTilingContext(para);

    ShapeInfo hit = MakeShape({2, 4, 8}, {0, 2, 1}, ge::kDataTypeSizeBitOffset + 4);
    optiling::TransposeB4Strategy hitStrategy(bundle.context, MakePlatformInfo(), MakeOpParam(hit));
    EXPECT_TRUE(hitStrategy.IsSupport());

    ShapeInfo notB4 = MakeShape({2, 4, 8}, {0, 2, 1}, 1);
    optiling::TransposeB4Strategy notB4Strategy(bundle.context, MakePlatformInfo(), MakeOpParam(notB4));
    EXPECT_FALSE(notB4Strategy.IsSupport());

    ShapeInfo lastNotTransposed = MakeShape({2, 4, 8}, {1, 0, 2}, ge::kDataTypeSizeBitOffset + 4);
    optiling::TransposeB4Strategy lastStrategy(bundle.context, MakePlatformInfo(), MakeOpParam(lastNotTransposed));
    EXPECT_FALSE(lastStrategy.IsSupport());
}
