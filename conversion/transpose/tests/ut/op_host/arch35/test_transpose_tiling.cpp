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
 * \file test_transpose_tiling.cpp
 * \brief transpose tiling ut test
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/transpose_tiling_arch35.h"
#include "../../../../op_kernel/arch35/transpose_tiling_key.h"

using namespace ge;
using namespace Transpose;

class TransposeTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TransposeTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TransposeTiling TearDown" << std::endl; }
};

// 1D 纯搬运：普通入口使用 24B 紧凑 TilingData（与 kernel 紧凑布局一致）
TEST_F(TransposeTiling, transpose_tiling_tensor_move_compact_tiling_data)
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
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_TENSOR_MOVE));
    EXPECT_EQ(tilingInfo.blockNum, 25U);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(TransposeTensorMoveTilingData));
    const auto* data = reinterpret_cast<const TransposeTensorMoveTilingData*>(tilingInfo.tilingData.get());
    // 动态主块分核：coreNum=25, mainBlockElements=4096, extraBlockCores=24, tail=1696
    EXPECT_EQ(data->mainBlockElements, 4096U);
    EXPECT_EQ(data->mainBlocksPerCore, 0U);
    EXPECT_EQ(data->extraBlockCores, 24U);
    EXPECT_EQ(data->tailElements, 1696U);
}

// 3D 尾轴不变：普通入口走 10014，72B 紧凑 TilingData
TEST_F(TransposeTiling, transpose_tiling_n_last_small_compact_tiling_data)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {1, 0, 2};
    gert::TilingContextPara::TensorDescription x({{30, 20, 64}, {30, 20, 64}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{20, 30, 64}, {20, 30, 64}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_N_LAST_SMALL));
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(TransposeNLastSmallTilingData));
    const auto* data = reinterpret_cast<const TransposeNLastSmallTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->rowBytes, 64U * 4U);
}

// 5D 尾轴不变：普通入口走 10004 紧凑 TilingData
TEST_F(TransposeTiling, transpose_tiling_n_last_regular_compact_tiling_data)
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
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(TransposeNLastTilingData));
    const auto* data = reinterpret_cast<const TransposeNLastTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->permSize, 5);
}

TEST_F(TransposeTiling, transpose_tiling_perm_int32_dtype)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int32_t perm_value[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{100, 200}, {100, 200}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{200, 100}, {200, 100}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
}

// Negative perm values in tiling, expect success with resolved perm

TEST_F(TransposeTiling, transpose_tiling_negative_perm)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[2] = {-1, -2};
    gert::TilingContextPara::TensorDescription x({{10, 20}, {10, 20}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{20, 10}, {20, 10}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(success);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE_23D));
    EXPECT_EQ(tilingInfo.blockNum, 1U);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(TransposeSmallShape23DTilingData));
    const auto* data = reinterpret_cast<const TransposeSmallShape23DTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(data->dim, 2);
    EXPECT_EQ(data->inputShape[0], 10);
    EXPECT_EQ(data->inputShape[1], 20);
    EXPECT_EQ(data->perm[0], 1);
    EXPECT_EQ(data->perm[1], 0);
    // 200 元素 x fp32 = 800B < 1024B 对齐单元，Host 分核退化为单核。
}

// Invalid perm dtype (DT_INT16, not int32/int64), expect failure

TEST_F(TransposeTiling, transpose_tiling_invalid_perm_dtype)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int16_t perm_value[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{10, 20}, {10, 20}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT16, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{20, 10}, {20, 10}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_FALSE(success);
}

// Perm size mismatch with x dim count, expect failure

TEST_F(TransposeTiling, transpose_tiling_perm_size_mismatch)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[3] = {0, 1, 2};
    gert::TilingContextPara::TensorDescription x({{10, 20}, {10, 20}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{10, 20}, {10, 20}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_FALSE(success);
}

// Perm value out of valid range (perm[i] >= dimSize), expect failure

TEST_F(TransposeTiling, transpose_tiling_perm_value_out_of_range)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[2] = {2, 0};
    gert::TilingContextPara::TensorDescription x({{10, 20}, {10, 20}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{20, 10}, {20, 10}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_FALSE(success);
}

// Perm value still negative after normalization (perm[i] < -dimSize), expect failure

TEST_F(TransposeTiling, transpose_tiling_perm_value_negative_out_of_range)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[2] = {-3, 0};
    gert::TilingContextPara::TensorDescription x({{10, 20}, {10, 20}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{20, 10}, {20, 10}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_FALSE(success);
}

// Duplicated values in perm, expect failure

TEST_F(TransposeTiling, transpose_tiling_perm_value_duplicated)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[4] = {0, 2, 2, 1};
    gert::TilingContextPara::TensorDescription x({{10, 20, 30, 40}, {10, 20, 30, 40}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{10, 30, 30, 20}, {10, 30, 30, 20}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_FALSE(success);
}

// All-ones shape input, expect TENSOR_MOVE (reduced to 1D)

TEST_F(TransposeTiling, transpose_tiling_invalid_core_num_zero)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 0;
    compileInfo.ubSize = 196608;

    int64_t perm_value[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{10, 20}, {10, 20}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{20, 10}, {20, 10}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_FALSE(success);
}

// Invalid init (ubSize = 0), expect failure

TEST_F(TransposeTiling, transpose_tiling_invalid_ub_size_zero)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 0;

    int64_t perm_value[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{10, 20}, {10, 20}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{20, 10}, {20, 10}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_FALSE(success);
}

// FP16 dtype, 3D transpose perm {0,2,1}, expect success

TEST_F(TransposeTiling, transpose_tiling_output_shape_mismatch)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{10, 20}, {10, 20}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    // Wrong output shape: should be {20, 10} but we give {10, 30}
    gert::TilingContextPara::TensorDescription out({{10, 30}, {10, 30}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_FALSE(success);
}

// INT64 dtype 4D large transpose, covers int64 perm data path

TEST_F(TransposeTiling, transpose_tiling_zero_shape_value)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    int64_t perm_value[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{0, 20}, {0, 20}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{20, 0}, {20, 0}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    bool success = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_FALSE(success);
}

/* ==================== 参数校验补充用例 ==================== */

namespace {
optiling::TransposeCompilerInfo MakeCompileInfo(int64_t coreNum = 40, int64_t ubSize = 196608)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = coreNum;
    compileInfo.ubSize = ubSize;
    return compileInfo;
}
} // namespace

// perm 元素个数大于最大轴数（8）→ GetPerm 直接拒绝
TEST_F(TransposeTiling, transpose_tiling_perm_size_over_max_int64)
{
    auto compileInfo = MakeCompileInfo();
    int64_t permValue[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    gert::TilingContextPara::TensorDescription x({{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{9}, {9}}, ge::DT_INT64, ge::FORMAT_ND, true, permValue);
    gert::TilingContextPara::TensorDescription out({{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara para("Transpose", {x, perm}, {out}, &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}

// perm 元素个数大于最大轴数（8）→ int32 perm 路径同样被拒绝
TEST_F(TransposeTiling, transpose_tiling_perm_size_over_max_int32)
{
    auto compileInfo = MakeCompileInfo();
    int32_t permValue[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    gert::TilingContextPara::TensorDescription x({{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{9}, {9}}, ge::DT_INT32, ge::FORMAT_ND, true, permValue);
    gert::TilingContextPara::TensorDescription out({{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara para("Transpose", {x, perm}, {out}, &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}

// perm 元素个数为 0（<1）→ GetPerm 拒绝
TEST_F(TransposeTiling, transpose_tiling_perm_size_zero)
{
    auto compileInfo = MakeCompileInfo();
    int64_t permValue[1] = {0};
    gert::TilingContextPara::TensorDescription x({{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, permValue);
    gert::TilingContextPara::TensorDescription out({{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara para("Transpose", {x, perm}, {out}, &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}

// perm 无数据（const tensor 数据为空）→ GetPerm 拒绝
TEST_F(TransposeTiling, transpose_tiling_perm_data_nullptr)
{
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara::TensorDescription x({{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, nullptr);
    gert::TilingContextPara::TensorDescription out({{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara para("Transpose", {x, perm}, {out}, &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}

// x 的 dtype 无法识别（元素字节数 <= 0）→ 参数校验失败
TEST_F(TransposeTiling, transpose_tiling_invalid_x_dtype)
{
    auto compileInfo = MakeCompileInfo();
    int64_t permValue[2] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{2, 3}, {2, 3}}, ge::DT_UNDEFINED, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, permValue);
    gert::TilingContextPara::TensorDescription out({{3, 2}, {3, 2}}, ge::DT_UNDEFINED, ge::FORMAT_ND);
    gert::TilingContextPara para("Transpose", {x, perm}, {out}, &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}
