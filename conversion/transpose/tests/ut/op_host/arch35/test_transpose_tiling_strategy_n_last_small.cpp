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
 * \file test_transpose_tiling_strategy_n_last_small.cpp
 * \brief TK_N_LAST_SMALL(10014) 策略 UT
 *
 * 覆盖：普通/关联入口统一 72B TransposeNLastSmallTilingData，
 * 以及 3/4 维尾轴不变小搬运的命中场景。
 */

#include <limits>

#include "test_transpose_tiling_strategy_common.h"
#include "../../../../op_host/arch35/transpose_tiling_strategy_n_last_small.h"

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
} // namespace

// 普通入口 3D 尾轴不变 → 10014，72B 紧凑结构
TEST(TransposeNLastSmallStrategy, e2e_hits_n_last_small)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    TilingInfo e2e;
    ASSERT_TRUE(ExecuteTiling(para, e2e));
    EXPECT_EQ(e2e.tilingKey, static_cast<uint64_t>(TK_N_LAST_SMALL));
    EXPECT_EQ(e2e.blockNum, 34);
    const auto* data = GetTilingData<TransposeNLastSmallTilingData>(e2e);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->rowBytes, 64U * 4U);
    EXPECT_EQ(data->aTiles * data->bTiles * data->cTiles, data->blocksPerCore * e2e.blockNum + data->extraBlocks);
}

// 关联入口 3D 尾轴不变：允许命中 10014，写 72B TransposeNLastSmallTilingData
TEST(TransposeNLastSmallStrategy, related_entry_hits_n_last_small_compact)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    ShapeInfo si = MakeShape({100, 200, 64}, {1, 0, 2}, 4);
    auto bundle = BuildTilingContext(para);
    optiling::TransposeNLastSmallStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si, true));
    ASSERT_TRUE(strategy.IsSupport());
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_N_LAST_SMALL));
    EXPECT_EQ(bundle.context->GetBlockDim(), 34);
    const auto* data = GetTilingData<TransposeNLastSmallTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->rowBytes, 64U * 4U);
}

TEST(TransposeNLastSmallStrategy, transpose_tiling_n_last_4d_no_last_transpose)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    // perm = {1, 0, 2, 3}: 只交换前两轴, 尾轴保持不转置。MergeAxis 后尾轴 (30*64=1920)
    // 仍不参与转置且 >= 32; 总量 <= int32 max（约 2GiB）→ 10014。
    int64_t perm_value[4] = {1, 0, 2, 3};
    gert::TilingContextPara::TensorDescription x({{30, 20, 30, 64}, {30, 20, 30, 64}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{20, 30, 30, 64}, {20, 30, 30, 64}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_N_LAST_SMALL));
    EXPECT_EQ(tilingInfo.tilingDataSize, sizeof(TransposeNLastSmallTilingData));
}

TEST(TransposeNLastSmallStrategy, transpose_tiling_n_last_large_3d)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {1, 0, 2};
    gert::TilingContextPara::TensorDescription x({{100, 200, 64}, {100, 200, 64}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{200, 100, 64}, {200, 100, 64}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_N_LAST_SMALL));
}

TEST(TransposeNLastSmallStrategy, transpose_tiling_n_last_large_4d)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[4] = {1, 0, 2, 3};
    gert::TilingContextPara::TensorDescription x({{200, 100, 64, 128}, {200, 100, 64, 128}}, ge::DT_FLOAT,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{100, 200, 64, 128}, {100, 200, 64, 128}}, ge::DT_FLOAT,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_N_LAST_SMALL));
}

TEST(TransposeNLastSmallStrategy, transpose_tiling_n_last_multiples_split)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {1, 0, 2};
    gert::TilingContextPara::TensorDescription x({{500, 400, 64}, {500, 400, 64}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{400, 500, 64}, {400, 500, 64}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_N_LAST_SMALL));
}

TEST(TransposeNLastSmallStrategy, transpose_tiling_n_last_small_corenum)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 4;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {1, 0, 2};
    gert::TilingContextPara::TensorDescription x({{10, 5, 128}, {10, 5, 128}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{5, 10, 128}, {5, 10, 128}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_N_LAST_SMALL));
}

/* ============================================================================
 * N_LAST_SMALL(10014)：int32 总字节上界 + 单块/分块 UB 占用
 * ========================================================================== */
TEST(TransposeNLastSmallStrategy, n_last_small_byte_boundary)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeDummyPara(compileInfo);
    auto bundle = BuildTilingContext(para);
    const int64_t eleLen = 4;
    const int64_t maxElems = (static_cast<int64_t>(std::numeric_limits<int32_t>::max()) - 1) / eleLen;

    ShapeInfo hit = MakeShape({1, 1, 32}, {0, 1, 2}, eleLen);
    hit.totalVolumeActual = maxElems;
    optiling::TransposeNLastSmallStrategy hitStrategy(bundle.context, MakePlatformInfo(), MakeOpParam(hit));
    EXPECT_TRUE(hitStrategy.IsSupport());

    ShapeInfo miss = hit;
    miss.totalVolumeActual = maxElems + 1;
    optiling::TransposeNLastSmallStrategy missStrategy(bundle.context, MakePlatformInfo(), MakeOpParam(miss));
    EXPECT_FALSE(missStrategy.IsSupport());
}

TEST(TransposeNLastSmallStrategy, n_last_small_ub_budget)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeDummyPara(compileInfo);
    auto bundle = BuildTilingContext(para);

    // 单块模式：srcLoopStrideBytes != 0，仅一块 srcTileStride 大小的 buffer。
    ShapeInfo single = MakeShape({100, 200, 64}, {1, 0, 2}, 4);
    optiling::TransposeNLastSmallStrategy singleStrategy(bundle.context, MakePlatformInfo(), MakeOpParam(single));
    ASSERT_TRUE(singleStrategy.IsSupport());
    ASSERT_EQ(singleStrategy.Execute(), TilingStatus::SUCCESS);
    const auto* singleData = GetTilingData<TransposeNLastSmallTilingData>(bundle.context);
    ASSERT_NE(singleData, nullptr);
    EXPECT_EQ(singleData->bufferOffset, 0U);
    EXPECT_NE(singleData->srcLoopStrideBytes, 0U);
    EXPECT_LE(singleData->srcTileStride, static_cast<uint32_t>(kUbSize));
    EXPECT_LE(singleData->tailSrcBytes, singleData->srcTileStride);

    // 分块模式：双缓冲，每块 bufferOffset 字节。
    auto bundle2 = BuildTilingContext(para);
    ShapeInfo multi = MakeShape({1000, 1000, 64}, {1, 0, 2}, 4);
    optiling::TransposeNLastSmallStrategy multiStrategy(bundle2.context, MakePlatformInfo(), MakeOpParam(multi));
    ASSERT_TRUE(multiStrategy.IsSupport());
    ASSERT_EQ(multiStrategy.Execute(), TilingStatus::SUCCESS);
    const auto* multiData = GetTilingData<TransposeNLastSmallTilingData>(bundle2.context);
    ASSERT_NE(multiData, nullptr);
    EXPECT_GT(multiData->bufferOffset, 0U);
    EXPECT_EQ(multiData->bufferOffset % kUbBlockSize, 0U);
    EXPECT_LE(static_cast<int64_t>(multiData->bufferOffset) * 2, kUbSize);
    EXPECT_LE(multiData->tileA, multiData->aTiles);
    EXPECT_LE(multiData->tileB, 1000U);
    const int64_t tileRowBytes = AlignUp(multiData->tileBytes, kUbBlockSize);
    EXPECT_LE(static_cast<int64_t>(multiData->tileA) * multiData->tileB * tileRowBytes,
              static_cast<int64_t>(multiData->bufferOffset));
}
