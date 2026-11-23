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
 * \file test_transpose_tiling_strategy_tensor_move.cpp
 * \brief TENSOR_MOVE(10000) 策略 UT：直调策略（逐字段断言）+ 端到端命中
 */

#include "test_transpose_tiling_strategy_common.h"
#include "../../../../op_host/arch35/transpose_tiling_strategy_tensor_move.h"

using namespace transpose_strategy_ut;

namespace {
gert::TilingContextPara MakePara(optiling::TransposeCompilerInfo& compileInfo)
{
    static int64_t permValue[1] = {0};
    gert::TilingContextPara::TensorDescription x({{100000}, {100000}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &permValue);
    gert::TilingContextPara::TensorDescription out({{100000}, {100000}}, ge::DT_FLOAT, ge::FORMAT_ND);
    return gert::TilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
}
} // namespace

// 4D perm {3,2,1,0} (full reverse), covers NDDMA NDDMA_CUT_2 path

// 3 参构造：context + platform + opParam
TEST(TransposeTensorMoveStrategy, direct_three_arg_ctor)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    auto bundle = BuildTilingContext(para);
    // 默认 shapeInfo（dim=0），TENSOR_MOVE 要求 dim==1，故不命中
    optiling::TransposeTensorMoveStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(ShapeInfo{}));
    EXPECT_FALSE(strategy.IsSupport());
}

TEST(TransposeTensorMoveStrategy, e2e_hits_this_strategy)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    TilingInfo e2e;
    ASSERT_TRUE(ExecuteTiling(para, e2e));
    EXPECT_EQ(e2e.tilingKey, TK_TENSOR_MOVE);
    EXPECT_EQ(e2e.blockNum, 25U);
    ASSERT_EQ(e2e.tilingDataSize, sizeof(TransposeTensorMoveTilingData));
    const auto* data = GetTilingData<TransposeTensorMoveTilingData>(e2e);
    ASSERT_NE(data, nullptr);
    // 普通入口动态主块分核：total=100000, eleLen=4, coreNum=48, ubSize=253952。
    EXPECT_EQ(data->mainBlockElements, 4096U);
    EXPECT_EQ(data->mainBlocksPerCore, 0U);
    EXPECT_EQ(data->extraBlockCores, 24U);
    EXPECT_EQ(data->tailElements, 1696U);
}

TEST(TransposeTensorMoveStrategy, transpose_tiling_tensor_move_1d)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[1] = {0};
    gert::TilingContextPara::TensorDescription x({{100000}, {100000}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{100000}, {100000}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_TENSOR_MOVE));
}

// Small shape (total volume bytes < threshold), expect SMALL_SHAPE tiling key 10001

TEST(TransposeTensorMoveStrategy, transpose_tiling_all_ones_shape)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[4] = {3, 0, 1, 2};
    gert::TilingContextPara::TensorDescription x({{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_TENSOR_MOVE));
}

// Invalid init (coreNum = 0), expect failure

TEST(TransposeTensorMoveStrategy, transpose_tiling_identity_perm_2d)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[2] = {0, 1};
    gert::TilingContextPara::TensorDescription x({{100, 200}, {100, 200}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{100, 200}, {100, 200}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_TENSOR_MOVE));
}

// Output shape mismatch with expected shape (y shape wrong), expect failure

TEST(TransposeTensorMoveStrategy, transpose_tiling_tensor_move_small_volume)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[1] = {0};
    gert::TilingContextPara::TensorDescription x({{10}, {10}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{10}, {10}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_TENSOR_MOVE));
}

// 4D transpose with perm {0,1,2,3} (identity permutation), expect TENSOR_MOVE

TEST(TransposeTensorMoveStrategy, transpose_tiling_identity_perm_4d)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[4] = {0, 1, 2, 3};
    gert::TilingContextPara::TensorDescription x({{10, 20, 30, 40}, {10, 20, 30, 40}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{10, 20, 30, 40}, {10, 20, 30, 40}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    // Identity perm reduces to 1D -> TENSOR_MOVE
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_TENSOR_MOVE));
}

// 关联算子入口统一使用 24B 紧凑 TransposeTensorMoveTilingData
TEST(TransposeTensorMoveStrategy, related_entry_uses_compact_tiling)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakePara(compileInfo);
    auto bundle = BuildTilingContext(para);
    ShapeInfo si = MakeShape({100000}, {0}, 4);
    optiling::TransposeTensorMoveStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si, true));
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    EXPECT_EQ(bundle.context->GetTilingKey(), static_cast<uint64_t>(TK_TENSOR_MOVE));
    EXPECT_EQ(bundle.context->GetBlockDim(), 25U);
    const auto* data = GetTilingData<TransposeTensorMoveTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);
    // 与普通入口一致：动态主块分核，coreNum=25, mainBlockElements=4096, extraBlockCores=24, tail=1696。
    EXPECT_EQ(data->mainBlockElements, 4096U);
    EXPECT_EQ(data->mainBlocksPerCore, 0U);
    EXPECT_EQ(data->extraBlockCores, 24U);
    EXPECT_EQ(data->tailElements, 1696U);
}

// TilingData 容量不足：SetTilingData 无法取得 TilingData，不写码流但 tiling 本身成功
TEST(TransposeTensorMoveStrategy, direct_tiling_data_insufficient)
{
    auto compileInfo = MakeCompilerInfo();
    int64_t permValue[1] = {0};
    gert::TilingContextPara::TensorDescription x({{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, permValue);
    gert::TilingContextPara::TensorDescription out({{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara para("Transpose", {x, perm}, {out}, &compileInfo);
    para.tilingDataSize_ = 8;
    TilingInfo tilingInfo;
    // 码流容量不足：SetTilingData 返回 FAILURE，整体 tiling 失败且不产出 TilingData
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingDataSize, 0U);
}

/* ============================================================================
 * TENSOR_MOVE(10000)：超大一维 shape + 双缓冲（含 bank 错位）UB 占用
 * ========================================================================== */
TEST(TransposeTensorMoveStrategy, tensor_move_large_shape_ub_budget)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeDummyPara(compileInfo);
    const int64_t eleLen = 1;
    const int64_t totalElements = kHugeDim;
    ShapeInfo si = MakeShape({totalElements}, {0}, eleLen);

    auto bundle = BuildTilingContext(para);
    optiling::TransposeTensorMoveStrategy strategy(bundle.context, MakePlatformInfo(), MakeOpParam(si));
    ASSERT_TRUE(strategy.IsSupport());
    ASSERT_EQ(strategy.Execute(), TilingStatus::SUCCESS);
    const auto* data = GetTilingData<TransposeTensorMoveTilingData>(bundle.context);
    ASSERT_NE(data, nullptr);

    const uint64_t blockDim = bundle.context->GetBlockDim();
    EXPECT_GT(blockDim, 0U);
    EXPECT_GT(data->mainBlockElements, 0U);
    EXPECT_LT(data->tailElements, data->mainBlockElements);

    // kernel：静态双缓冲，第二块额外错开 BANK_OFFSET_BYTES。
    const uint64_t ubUsage = static_cast<uint64_t>(data->mainBlockElements) * static_cast<uint64_t>(eleLen) * 2U +
                             TransposeTensorMoveTilingData::BANK_OFFSET_BYTES;
    EXPECT_LE(ubUsage, static_cast<uint64_t>(kUbSize));

    // kernel 实际处理元素数必须与 shape 完全一致（超大 shape 走 uint64 分块/偏移）。
    const uint64_t fullBlocks = data->mainBlocksPerCore * blockDim + static_cast<uint64_t>(data->extraBlockCores);
    EXPECT_EQ(fullBlocks * data->mainBlockElements + data->tailElements, static_cast<uint64_t>(totalElements));
    EXPECT_GT(data->mainBlocksPerCore, 0U);
}

// 边界：dim==1 命中，dim==2 不命中（TENSOR_MOVE 只承接纯搬运）
TEST(TransposeTensorMoveStrategy, tensor_move_dim_boundary)
{
    auto compileInfo = MakeCompilerInfo();
    auto para = MakeDummyPara(compileInfo);
    auto bundle = BuildTilingContext(para);

    ShapeInfo dim1 = MakeShape({1024}, {0}, 4);
    optiling::TransposeTensorMoveStrategy hit(bundle.context, MakePlatformInfo(), MakeOpParam(dim1));
    EXPECT_TRUE(hit.IsSupport());

    ShapeInfo dim2 = MakeShape({32, 32}, {0, 1}, 4);
    optiling::TransposeTensorMoveStrategy miss(bundle.context, MakePlatformInfo(), MakeOpParam(dim2));
    EXPECT_FALSE(miss.IsSupport());
}
