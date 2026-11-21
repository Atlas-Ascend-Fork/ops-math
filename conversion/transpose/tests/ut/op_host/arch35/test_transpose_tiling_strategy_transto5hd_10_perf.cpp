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
 * \file test_transpose_tiling_strategy_transto5hd_10_perf.cpp
 * \brief TRANSTO5HD_10_PERF(10017) 策略 UT：端到端命中 + tile 几何/回落
 */

#include "test_transpose_tiling_strategy_common.h"
#include "../../../../op_host/arch35/transpose_tiling_strategy_transto5hd_10_perf.h"

using namespace transpose_strategy_ut;

namespace {
// NDDMA_BIG_DIM path: 6D with isLastAxisTranspose=true, GATHER may fail then NDDMA_BIG_DIM

/* ==================== 准入/回落分支补充用例 ==================== */
gert::TilingContextPara Make2DPara(optiling::TransposeCompilerInfo& compileInfo)
{
    static int64_t permValue[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{512, 512}, {512, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &permValue);
    gert::TilingContextPara::TensorDescription out({{512, 512}, {512, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    return gert::TilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
}
} // namespace

// related（融合内联）场景不命中
TEST(TransposeTransTo5HD10PerfStrategy, direct_related_not_support)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = Make2DPara(compileInfo);
    ShapeInfo si = MakeShape({512, 512}, {1, 0}, 2);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeTransTo5HD10PerfStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si, true));
    EXPECT_FALSE(strategy.IsSupport());
}

// 非特化架构（DAV_9201）→ IsSupport 拒绝（准入判断只在 IsSupport）
TEST(TransposeTransTo5HD10PerfStrategy, direct_arch_not_support)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = Make2DPara(compileInfo);
    ShapeInfo si = MakeShape({512, 512}, {1, 0}, 2);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeTransTo5HD10PerfStrategy strategy(
        bundle.context, MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_9201)), MakeOpParam(si));
    EXPECT_FALSE(strategy.IsSupport());
}

// 准入（IsSupport）：dim != 2
TEST(TransposeTransTo5HD10PerfStrategy, direct_reject_dim)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = Make2DPara(compileInfo);
    ShapeInfo si = MakeShape({16, 32, 32}, {1, 0, 2}, 2);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeTransTo5HD10PerfStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    EXPECT_FALSE(strategy.IsSupport());
}

// 准入（IsSupport）：dim==2 但 perm 非 [1,0]
TEST(TransposeTransTo5HD10PerfStrategy, direct_reject_perm)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = Make2DPara(compileInfo);
    ShapeInfo si = MakeShape({512, 512}, {0, 1}, 2);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeTransTo5HD10PerfStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    EXPECT_FALSE(strategy.IsSupport());
}

// 准入（IsSupport）：dtype 位宽非 16/32bit（8bit）
TEST(TransposeTransTo5HD10PerfStrategy, direct_reject_dtype)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = Make2DPara(compileInfo);
    ShapeInfo si = MakeShape({512, 512}, {1, 0}, 1);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeTransTo5HD10PerfStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    EXPECT_FALSE(strategy.IsSupport());
}

// 准入（IsSupport）：m 非 16 对齐
TEST(TransposeTransTo5HD10PerfStrategy, direct_reject_not_aligned)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = Make2DPara(compileInfo);
    ShapeInfo si = MakeShape({510, 512}, {1, 0}, 2);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeTransTo5HD10PerfStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    EXPECT_FALSE(strategy.IsSupport());
}

// 准入（IsSupport）：m 非正
TEST(TransposeTransTo5HD10PerfStrategy, direct_reject_non_positive_m)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = Make2DPara(compileInfo);
    ShapeInfo si = MakeShape({0, 512}, {1, 0}, 2);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeTransTo5HD10PerfStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    EXPECT_FALSE(strategy.IsSupport());
}

// 准入（IsSupport）：总数据量未达 DAV_3510 门槛（16x16 fp16 = 512B < 512KB）
TEST(TransposeTransTo5HD10PerfStrategy, direct_reject_small_volume)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = Make2DPara(compileInfo);
    ShapeInfo si = MakeShape({16, 16}, {1, 0}, 2);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeTransTo5HD10PerfStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    EXPECT_FALSE(strategy.IsSupport());
}

TEST(TransposeTransTo5HD10PerfStrategy, transto5hd_10_perf_route)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 253952;

    int64_t perm_value[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{512, 512}, {512, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{512, 512}, {512, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_TRANSTO5HD_10_PERF));
    EXPECT_EQ(tilingInfo.blockNum, 40U);
    const auto* tilingData = GetTilingData<TransposeTransTo5HD10PerfTilingData>(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->m, 512);
    EXPECT_EQ(tilingData->n, 512);
    EXPECT_EQ(tilingData->coreNum, 40);
    // m=512 > cap(256): tileM 取 256; 248KB UB 预算下 tileN 取最大 2 的幂次 128
    EXPECT_EQ(tilingData->tileM, 256);
    EXPECT_EQ(tilingData->tileN, 128);
}

// 2D perm [1, 0], fp16, m=16: tileM 跟随 m, 避免 tile 退化成 16x128 的 4KB 小块
TEST(TransposeTransTo5HD10PerfStrategy, transto5hd_10_perf_small_m_tile)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 253952;

    int64_t perm_value[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{16, 262144}, {16, 262144}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{262144, 16}, {262144, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_TRANSTO5HD_10_PERF));
    const auto* tilingData = GetTilingData<TransposeTransTo5HD10PerfTilingData>(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->m, 16);
    EXPECT_EQ(tilingData->n, 262144);
    EXPECT_EQ(tilingData->tileM, 16);
    EXPECT_EQ(tilingData->tileN, 1552);
}

// 2D perm [1, 0], fp16, 窄列 (m=131072, n=16): 收起多余 tileN, 改为行方向切分,
// tileM 按 "行块数≈k*coreNum" 放大以保证多核负载均衡
TEST(TransposeTransTo5HD10PerfStrategy, transto5hd_10_perf_narrow_n_tile)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 253952;

    int64_t perm_value[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{131072, 16}, {131072, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{16, 131072}, {16, 131072}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_TRANSTO5HD_10_PERF));
    const auto* tilingData = GetTilingData<TransposeTransTo5HD10PerfTilingData>(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->m, 131072);
    EXPECT_EQ(tilingData->n, 16);
    EXPECT_EQ(tilingData->tileM, 1648);
    EXPECT_EQ(tilingData->tileN, 16);
}

// 2D perm [1, 0], fp16, m/n 均 16 对齐但不是 tile 整数倍 (1040x1040):
// 行、列方向均产生尾块, 覆盖 kernel 非 full-tile 的搬运/转置路径
TEST(TransposeTransTo5HD10PerfStrategy, transto5hd_10_perf_tail_tile)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 253952;

    int64_t perm_value[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{1040, 1040}, {1040, 1040}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{1040, 1040}, {1040, 1040}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_TRANSTO5HD_10_PERF));
    const auto* tilingData = GetTilingData<TransposeTransTo5HD10PerfTilingData>(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->m, 1040);
    EXPECT_EQ(tilingData->n, 1040);
    EXPECT_EQ(tilingData->coreNum, 40);
    // tileM=256 / tileN=128: 1040 既不是 256 也不是 128 的整数倍, 行/列均有尾块
    EXPECT_EQ(tilingData->tileM, 256);
    EXPECT_EQ(tilingData->tileN, 128);
}

// 2D perm [1, 0] 但 n 非 16 对齐: 10017 不适用, 回落主流程
TEST(TransposeTransTo5HD10PerfStrategy, transto5hd_10_perf_not_aligned_fallback)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 253952;

    int64_t perm_value[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{512, 510}, {512, 510}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{510, 512}, {510, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    // 10017 因 n 非 16 对齐被拒 → 尾轴转置交由 GATHER(10006)
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_GATHER));
    const auto* gatherData = GetTilingData<TransposeGatherTilingData>(tilingInfo);
    ASSERT_NE(gatherData, nullptr);
    EXPECT_GT(gatherData->usedCoreCnt, 0U);
}

// UB 预算不足以容纳一个 16x16 tile → 回落
TEST(TransposeTransTo5HD10PerfStrategy, direct_ub_budget_fallback)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = Make2DPara(compileInfo);
    ShapeInfo si = MakeShape({512, 512}, {1, 0}, 2);
    auto bundle = BuildTilingContext(para);
    optiling::TransposePlatformInfo platform = MakePlatformInfo();
    platform.ubSize = 8192;
    optiling::TransposeTransTo5HD10PerfStrategy strategy(bundle.context, platform, MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    EXPECT_EQ(strategy.Execute(), TilingStatus::FALLBACK);
}

// TilingData 容量不足 → SetTilingData 失败 → FAILURE（硬错误）
TEST(TransposeTransTo5HD10PerfStrategy, direct_tiling_data_insufficient)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = Make2DPara(compileInfo);
    para.tilingDataSize_ = 4;
    ShapeInfo si = MakeShape({512, 512}, {1, 0}, 2);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeTransTo5HD10PerfStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    EXPECT_EQ(strategy.Execute(), TilingStatus::FAILURE);
}
