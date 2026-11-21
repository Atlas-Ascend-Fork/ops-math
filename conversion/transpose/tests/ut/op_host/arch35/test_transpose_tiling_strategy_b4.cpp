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
    const auto* data = GetTilingData<TransposeB4TilingData>(e2e);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->permSize, 4);
    EXPECT_GT(data->useCoreNum, 0);
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
    EXPECT_GT(bundle.context->GetBlockDim(), 0U);
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
