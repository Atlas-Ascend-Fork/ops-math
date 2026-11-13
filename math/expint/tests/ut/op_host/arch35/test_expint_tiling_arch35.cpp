/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ge;
using namespace std;

// ExpintCompileInfo is an empty struct private to the tiling TU; a dummy non-null
// pointer satisfies the faker (the tiling func never reads its content).
static int g_dummyCompileInfo = 0;

// Faker defaults: coreNum=64, ubSize=262144, ubBlockSize=32.
// FP32 (bufferNum=7):     ubFactor = FloorAlign(FloorDiv((262144-8192)/4, 7), 32) = 9056
// FP16/BF16 (bufferNum=9): ubFactor = FloorAlign(FloorDiv((262144-8192)/4, 9), 32) = 7040
// Copy cap: maxUbFactor = FloorAlign(65535/4, 32) = 16352

// Asserts a successful tiling: ExpintTilingData = {totalNum, blockFactor, ubFactor},
// workspace {0}, and blockDim = usedCoreNum.
static void ExpectTilingSuccess(const gert::TilingContextPara& para, int64_t totalNum, int64_t blockFactor,
                                int64_t ubFactor, size_t blockDim)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ASSERT_NE(info.tilingData, nullptr);
    const int64_t* data = reinterpret_cast<const int64_t*>(info.tilingData.get());
    EXPECT_EQ(data[0], totalNum);
    EXPECT_EQ(data[1], blockFactor);
    EXPECT_EQ(data[2], ubFactor);
    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_EQ(info.workspaceSizes[0], 0U);
    EXPECT_EQ(info.blockNum, blockDim);
}

static void ExpectTilingReject(const gert::TilingContextPara& para)
{
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

class ExpintTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ExpintTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ExpintTiling TearDown" << std::endl; }
};

TEST_F(ExpintTiling, expint_tiling_fp32_nd_success)
{
    gert::TilingContextPara tilingContextPara("Expint",
                                              {
                                                  {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &g_dummyCompileInfo);
    ExpectTilingSuccess(tilingContextPara, 16, 32, 9056, 1);
}

TEST_F(ExpintTiling, expint_tiling_fp16_nd_success)
{
    gert::TilingContextPara tilingContextPara("Expint",
                                              {
                                                  {{{16}, {16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{16}, {16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &g_dummyCompileInfo);
    ExpectTilingSuccess(tilingContextPara, 16, 32, 7040, 1);
}

TEST_F(ExpintTiling, expint_tiling_bf16_nd_success)
{
    gert::TilingContextPara tilingContextPara("Expint",
                                              {
                                                  {{{16}, {16}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{16}, {16}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              &g_dummyCompileInfo);
    ExpectTilingSuccess(tilingContextPara, 16, 32, 7040, 1);
}

TEST_F(ExpintTiling, expint_tiling_scalar_promoted_to_one)
{
    gert::TilingContextPara tilingContextPara("Expint",
                                              {
                                                  {{}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &g_dummyCompileInfo);
    ExpectTilingSuccess(tilingContextPara, 1, 32, 9056, 1);
}

TEST_F(ExpintTiling, expint_tiling_empty_tensor_short_circuit)
{
    gert::TilingContextPara tilingContextPara("Expint",
                                              {
                                                  {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &g_dummyCompileInfo);
    ExpectTilingSuccess(tilingContextPara, 0, 0, 0, 1);
}

TEST_F(ExpintTiling, expint_tiling_multi_core_partition)
{
    gert::TilingContextPara tilingContextPara("Expint",
                                              {
                                                  {{{2048}, {2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2048}, {2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &g_dummyCompileInfo);
    ExpectTilingSuccess(tilingContextPara, 2048, 32, 9056, 64);
}

TEST_F(ExpintTiling, expint_tiling_ub_factor_capped_by_copy_limit)
{
    gert::TilingContextPara tilingContextPara("Expint",
                                              {
                                                  {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &g_dummyCompileInfo, 64, 524288);
    ExpectTilingSuccess(tilingContextPara, 16, 32, 16352, 1);
}

TEST_F(ExpintTiling, expint_tiling_public_format_nchw_success)
{
    gert::TilingContextPara tilingContextPara("Expint",
                                              {
                                                  {{{2, 8}, {2, 8}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
                                              },
                                              {
                                                  {{{2, 8}, {2, 8}}, ge::DT_FLOAT, ge::FORMAT_NCHW},
                                              },
                                              &g_dummyCompileInfo);
    ExpectTilingSuccess(tilingContextPara, 16, 32, 9056, 1);
}

// NOTE: TilingContextPara::TensorDescription carries a single format field fed to both
// origin and storage, so the origin/storage-divergent branches of IsPrivateFormat cannot be
// covered independently here (faker limitation); their behavior was verified on-device via
// TTK logs (Invalid_Argument_Tensor_Format with reason: private formats are not supported).

TEST_F(ExpintTiling, expint_tiling_private_format_fractal_nz_reject)
{
    gert::TilingContextPara tilingContextPara("Expint",
                                              {
                                                  {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ},
                                              },
                                              {
                                                  {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ},
                                              },
                                              &g_dummyCompileInfo);
    ExpectTilingReject(tilingContextPara);
}

TEST_F(ExpintTiling, expint_tiling_private_format_fractal_z_reject)
{
    gert::TilingContextPara tilingContextPara("Expint",
                                              {
                                                  {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_Z},
                                              },
                                              {
                                                  {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_Z},
                                              },
                                              &g_dummyCompileInfo);
    ExpectTilingReject(tilingContextPara);
}

TEST_F(ExpintTiling, expint_tiling_double_dtype_reject)
{
    gert::TilingContextPara tilingContextPara("Expint",
                                              {
                                                  {{{16}, {16}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{16}, {16}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                              },
                                              &g_dummyCompileInfo);
    ExpectTilingReject(tilingContextPara);
}

TEST_F(ExpintTiling, expint_tiling_int32_dtype_reject)
{
    gert::TilingContextPara tilingContextPara("Expint",
                                              {
                                                  {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{16}, {16}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              &g_dummyCompileInfo);
    ExpectTilingReject(tilingContextPara);
}

TEST_F(ExpintTiling, expint_tiling_nine_dim_reject)
{
    gert::TilingContextPara tilingContextPara(
        "Expint",
        {
            {{{1, 1, 1, 1, 1, 1, 1, 2, 2}, {1, 1, 1, 1, 1, 1, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 1, 1, 1, 1, 1, 2, 2}, {1, 1, 1, 1, 1, 1, 1, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &g_dummyCompileInfo);
    ExpectTilingReject(tilingContextPara);
}

TEST_F(ExpintTiling, expint_tiling_output_elem_mismatch_reject)
{
    gert::TilingContextPara tilingContextPara("Expint",
                                              {
                                                  {{{4, 4}, {4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{5, 4}, {5, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &g_dummyCompileInfo);
    ExpectTilingReject(tilingContextPara);
}

TEST_F(ExpintTiling, expint_tiling_zero_core_num_reject)
{
    gert::TilingContextPara tilingContextPara("Expint",
                                              {
                                                  {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &g_dummyCompileInfo, 0);
    ExpectTilingReject(tilingContextPara);
}

TEST_F(ExpintTiling, expint_tiling_zero_ub_size_reject)
{
    gert::TilingContextPara tilingContextPara("Expint",
                                              {
                                                  {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &g_dummyCompileInfo, 64, 0);
    ExpectTilingReject(tilingContextPara);
}
