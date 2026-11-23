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
 * \file test_transpose_tiling_strategy_n_last.cpp
 * \brief N_LAST(10004) 策略 UT
 *
 * 覆盖：普通/关联入口统一 紧凑 TransposeNLastTilingData，
 * 以及 N_LAST 分核因子回溯/退轴/Rate 分支。
 */

#include "test_transpose_tiling_strategy_common.h"
#include "../../../../op_host/arch35/transpose_tiling_strategy_n_last.h"

using namespace transpose_strategy_ut;

namespace {
gert::TilingContextPara MakePara(optiling::TransposeCompilerInfo& compileInfo)
{
    static int64_t permValue[3] = {1, 0, 2};
    gert::TilingContextPara::TensorDescription x({{100, 200, 64}, {100, 200, 64}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &permValue);
    gert::TilingContextPara::TensorDescription out({{200, 100, 64}, {200, 100, 64}}, ge::DT_FLOAT, ge::FORMAT_ND);
    return gert::TilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
}

optiling::TransposePlatformInfo MakeNLastPlatform(int64_t ubSize)
{
    optiling::TransposePlatformInfo platform = MakePlatformInfo();
    platform.ubSize = ubSize;
    return platform;
}
} // namespace

// 体积小于平台阈值时 N_LAST 不命中：
//   - 非 SIMT 平台由 NDDMA/NDDMA_BIG_DIM 承接；
//   - SIMT 平台由后续 SMALL_SHAPE 承接（N_LAST 注册在前，但体积门槛使其让位）。
TEST(TransposeNLastStrategy, is_support_false_below_threshold)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    // 384KB < 4MB，尾轴 48 不转置且 >= 32
    ShapeInfo si = MakeShape({32, 64, 48}, {1, 0, 2}, 4);
    auto bundle = BuildTilingContext(para);

    optiling::TransposePlatformInfo nonSimtPlatform = MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5162),
                                                                       false);
    optiling::TransposeNLastStrategy nonSimtStrategy(bundle.context, nonSimtPlatform, MakeOpParam(si));
    EXPECT_FALSE(nonSimtStrategy.IsSupport());

    // SIMT 平台同样受体积门槛约束，小 shape 交由后续 SMALL_SHAPE 承接
    optiling::TransposeNLastStrategy simtStrategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    EXPECT_FALSE(simtStrategy.IsSupport());
}

// DAV_5102 的 N_LAST 体积阈值为 400KB（与旧版 EntryTilingTemplate 一致）
TEST(TransposeNLastStrategy, is_support_dav_5102_volume_threshold_boundary)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    auto bundle = BuildTilingContext(para);
    constexpr int64_t kThreshold = 400000;
    auto platform = MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5102), true);

    // 尾轴不转置、尾轴 >= 32；体积恰好等于 400KB → 命中
    ShapeInfo hit = MakeShape({1, kThreshold / 4}, {0, 1}, 4);
    optiling::TransposeNLastStrategy hitStrategy(bundle.context, platform, MakeOpParam(hit));
    EXPECT_TRUE(hitStrategy.IsSupport());

    // 低于 400KB → 不命中
    ShapeInfo miss = MakeShape({1, kThreshold / 4 - 1}, {0, 1}, 4);
    optiling::TransposeNLastStrategy missStrategy(bundle.context, platform, MakeOpParam(miss));
    EXPECT_FALSE(missStrategy.IsSupport());
}

// 非 SIMT 平台体积门控边界：恰好等于 4MB 命中，低于 1 个元素（4 字节）不命中
TEST(TransposeNLastStrategy, is_support_non_simt_volume_threshold_boundary)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    auto bundle = BuildTilingContext(para);
    constexpr int64_t kThreshold = 4000000; // 非 SIMT 平台写死的 4MB 阈值
    auto platform = MakePlatformInfo(static_cast<int64_t>(Ops::Base::DAV_5162), false);

    // 尾轴不转置（perm={0,1}）、尾轴元素数 >= 32，体积恰好等于阈值 → 命中
    ShapeInfo hit = MakeShape({1, kThreshold / 4}, {0, 1}, 4);
    optiling::TransposeNLastStrategy hitStrategy(bundle.context, platform, MakeOpParam(hit));
    EXPECT_TRUE(hitStrategy.IsSupport());

    ShapeInfo miss = MakeShape({1, kThreshold / 4 - 1}, {0, 1}, 4);
    optiling::TransposeNLastStrategy missStrategy(bundle.context, platform, MakeOpParam(miss));
    EXPECT_FALSE(missStrategy.IsSupport());
}

// 尾轴转置或尾维 < 32 时不命中
TEST(TransposeNLastStrategy, is_support_rejects_last_transpose_and_small_tail)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    auto bundle = BuildTilingContext(para);

    // 尾轴转置 → 不命中
    ShapeInfo lastTranspose = MakeShape({32, 64, 48}, {0, 2, 1}, 4);
    optiling::TransposeNLastStrategy transposeStrategy(bundle.context, MakePlatformInfo(), MakeOpParam(lastTranspose));
    EXPECT_FALSE(transposeStrategy.IsSupport());

    // 尾维 < 32 → 不命中
    ShapeInfo smallTail = MakeShape({32, 64, 16}, {1, 0, 2}, 4);
    optiling::TransposeNLastStrategy smallTailStrategy(bundle.context, MakePlatformInfo(), MakeOpParam(smallTail));
    EXPECT_FALSE(smallTailStrategy.IsSupport());
}

// 普通入口 5D 尾轴不变 → 走 10004，紧凑结构
TEST(TransposeNLastStrategy, e2e_hits_n_last_regular_compact)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[5] = {1, 0, 3, 2, 4};
    gert::TilingContextPara::TensorDescription x({{10, 10, 10, 10, 128}, {10, 10, 10, 10, 128}}, ge::DT_FLOAT,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{10, 10, 10, 10, 128}, {10, 10, 10, 10, 128}}, ge::DT_FLOAT,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_N_LAST));
    EXPECT_EQ(tilingInfo.blockNum, 40);
    const auto* data = GetTilingData<TransposeNLastTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->blkFactor, 2);
    EXPECT_EQ(data->ubSize, 196608);
    EXPECT_EQ(data->inUbFactor, 1);
}

// 关联入口 5D：统一使用紧凑 紧凑 TransposeNLastTilingData，走 10004
TEST(TransposeNLastStrategy, related_entry_uses_compact_tiling_data)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    ShapeInfo si = MakeShape({10, 10, 10, 10, 128}, {1, 0, 3, 2, 4}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeNLastStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si, true));
    ASSERT_TRUE(strategy.IsSupport());
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_N_LAST));
    EXPECT_EQ(bundle.context->GetBlockDim(), 48);
    const auto* data = GetTilingData<TransposeNLastTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->blkFactor, 1);
    EXPECT_EQ(data->ubSize, 253952);
    EXPECT_EQ(data->inUbFactor, 2);
}

// 尾轴放不下 UB 且核利用率不足 → Rate 回溯并命中因子（普通 5D，走紧凑 10004）
TEST(TransposeNLastStrategy, direct_rate_branch_unreachable_by_e2e)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    ShapeInfo si = MakeShape({1, 1, 1, 1, 100000}, {0, 1, 2, 3, 4}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeNLastStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_N_LAST));
    EXPECT_EQ(bundle.context->GetBlockDim(), 44);
    const auto* data = GetTilingData<TransposeNLastTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->inUbFactor, 2325);
    EXPECT_GE(data->inCutIndex, 0);
}

// 尾轴可整体吸收、核利用率不足 → Rate 回溯分支（普通 5D 紧凑 10004）
TEST(TransposeNLastStrategy, direct_rate_absorb_branch_unreachable_by_e2e)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    ShapeInfo si = MakeShape({1, 1, 2, 3, 32}, {0, 1, 3, 2, 4}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeNLastStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_N_LAST));
    EXPECT_EQ(bundle.context->GetBlockDim(), 48);
    const auto* data = GetTilingData<TransposeNLastTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->inUbFactor, 4);
}

// inUbFactor 无效 → 退轴重切（普通 5D 紧凑 10004）
TEST(TransposeNLastStrategy, direct_revert_axis_unreachable_by_e2e)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    ShapeInfo si = MakeShape({1, 1, 1, 64, 33}, {0, 1, 2, 3, 4}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeNLastStrategy strategy(bundle.context, MakeNLastPlatform(288), MakeOpParam(si));
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_N_LAST));
    EXPECT_EQ(bundle.context->GetBlockDim(), static_cast<uint32_t>(compileInfo.coreNum));
    const auto* data = GetTilingData<TransposeNLastTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->ubSize, 288);
    EXPECT_EQ(data->inUbFactor, 17);
    EXPECT_GE(data->inCutIndex, 0);
}

// 块数不足以铺满核 → 每核 1 个 block（普通 5D 紧凑 10004）
TEST(TransposeNLastStrategy, direct_block_less_than_corenum_unreachable_by_e2e)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    ShapeInfo si = MakeShape({1, 1, 1, 1, 32}, {0, 1, 2, 3, 4}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeNLastStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetBlockDim(), 32U);
    const auto* data = GetTilingData<TransposeNLastTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->blkFactor, 1);
    EXPECT_EQ(data->blkTailFactor, 0);
}

// 普通 10004 紧凑码流容量不足：SetTilingData 返回 FAILURE
TEST(TransposeNLastStrategy, direct_tiling_data_insufficient)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    para.tilingDataSize_ = 8;
    // 5 维尾轴不变 → 普通 10004（紧凑）
    ShapeInfo si = MakeShape({10, 10, 10, 10, 128}, {1, 0, 3, 2, 4}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeNLastStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    EXPECT_EQ(strategy.Execute(), TilingStatus::FAILURE);
}

/* ============================================================================
 * N_LAST(10004)：超大 2D shape（首轴 > uint32）+ 双缓冲 UB 占用
 * ========================================================================== */
TEST(TransposeNLastStrategy, n_last_large_shape_ub_budget)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeDummyPara(compileInfo);
    const int64_t eleLen = 4;
    const int64_t tail = 64;
    ShapeInfo si = MakeShape({kHugeDim, tail}, {0, 1}, eleLen);

    auto bundle = BuildTilingContext(para);
    optiling::TransposeNLastStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    const auto* data = GetTilingData<TransposeNLastTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);

    ASSERT_GE(data->permSize, 2);
    ASSERT_GE(data->inCutIndex, 0);
    ASSERT_LT(data->inCutIndex, data->permSize);
    EXPECT_EQ(data->inputShape[0], kHugeDim);
    EXPECT_GT(data->inputShape[0], static_cast<int64_t>(kUint32Max));
    EXPECT_GT(data->inUbFactor, 0);

    // kernel：local_[0] | local_[1] 各占 ubSize/2；单块实际数据量不得超过该预算。
    const int64_t bufferBytes = data->ubSize / 2;
    const int64_t mainElems = static_cast<int64_t>(data->inUbFactor) *
                              Product(data->inputShape, data->inCutIndex + 1, data->permSize);
    const int64_t mainBytes = AlignUp(mainElems * eleLen, kUbBlockSize);
    EXPECT_LE(mainBytes, bufferBytes);
    if (data->inTailFactor != 0) {
        const int64_t tailElems = static_cast<int64_t>(data->inTailFactor) *
                                  Product(data->inputShape, data->inCutIndex + 1, data->permSize);
        EXPECT_LE(AlignUp(tailElems * eleLen, kUbBlockSize), bufferBytes);
        EXPECT_LT(data->inTailFactor, data->inUbFactor);
    }
    EXPECT_EQ(data->ubSize, kUbSize);
}

// 边界：非 SIMT 平台体积阈值 4MB，等于阈值命中、少 1 字节不命中；尾轴 31/32 边界
TEST(TransposeNLastStrategy, n_last_volume_and_tail_boundary)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeDummyPara(compileInfo);
    auto bundle = BuildTilingContext(para);

    ShapeInfo hit = MakeShape({15625, 64}, {0, 1}, 4); // 4,000,000B
    optiling::TransposeNLastStrategy hitStrategy(bundle.context, MakePlatformInfo(), MakeOpParam(hit));
    EXPECT_TRUE(hitStrategy.IsSupport());

    ShapeInfo miss = MakeShape({15625, 64}, {0, 1}, 4);
    miss.totalVolumeActual -= 1; // 少 4B
    optiling::TransposeNLastStrategy missStrategy(bundle.context, MakePlatformInfo(), MakeOpParam(miss));
    EXPECT_FALSE(missStrategy.IsSupport());

    ShapeInfo tail31 = MakeShape({15625, 31}, {0, 1}, 4);
    tail31.totalVolumeActual = 4000000;
    optiling::TransposeNLastStrategy tail31Strategy(bundle.context, MakePlatformInfo(), MakeOpParam(tail31));
    EXPECT_FALSE(tail31Strategy.IsSupport());

    ShapeInfo tail32 = MakeShape({15625, 32}, {0, 1}, 4);
    tail32.totalVolumeActual = 4000000;
    optiling::TransposeNLastStrategy tail32Strategy(bundle.context, MakePlatformInfo(), MakeOpParam(tail32));
    EXPECT_TRUE(tail32Strategy.IsSupport());
}
