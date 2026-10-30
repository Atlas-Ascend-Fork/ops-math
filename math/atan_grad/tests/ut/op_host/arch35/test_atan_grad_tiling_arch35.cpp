/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

namespace {
struct AtanGradUtCompileInfo {};
} // namespace

class AtanGradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AtanGradTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AtanGradTilingTest TearDown" << std::endl; }
};

// 正常场景：fp32 基本形状
TEST_F(AtanGradTilingTest, atan_grad_tiling_fp32_basic)
{
    AtanGradUtCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("AtanGrad",
                                              {
                                                  {{{32, 32}, {32, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{32, 32}, {32, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{32, 32}, {32, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_GE(tilingInfo.blockNum, 1);
}

// README 约束：支持空 Tensor（元素个数为0），tiling 放行（blockDim=1）
TEST_F(AtanGradTilingTest, atan_grad_tiling_empty_tensor)
{
    AtanGradUtCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("AtanGrad",
                                              {
                                                  {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.blockNum, 1);
}

// 私有格式拦截：y=FRACTAL_NZ / dy=FRACTAL_NZ / z=FRACTAL_NZ
// （FRACTAL_NZ 等私有格式内存布局非线性，kernel 线性寻址会越界/错乱）
TEST_F(AtanGradTilingTest, atan_grad_tiling_fractal_nz_intercept)
{
    AtanGradUtCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("AtanGrad",
                                              {
                                                  {{{32, 32}, {32, 32}}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ},
                                                  {{{32, 32}, {32, 32}}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ},
                                              },
                                              {
                                                  {{{32, 32}, {32, 32}}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ},
                                              },
                                              &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 私有格式拦截：仅 dy 为 FRACTAL_NZ（y/z 为 ND）
TEST_F(AtanGradTilingTest, atan_grad_tiling_dy_fractal_nz_intercept)
{
    AtanGradUtCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("AtanGrad",
                                              {
                                                  {{{32, 32}, {32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{32, 32}, {32, 32}}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ},
                                              },
                                              {
                                                  {{{32, 32}, {32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 私有格式拦截：NC1HWC0
TEST_F(AtanGradTilingTest, atan_grad_tiling_nc1hwc0_intercept)
{
    AtanGradUtCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("AtanGrad",
                                              {
                                                  {{{32, 32}, {32, 32}}, ge::DT_FLOAT, ge::FORMAT_NC1HWC0},
                                                  {{{32, 32}, {32, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{32, 32}, {32, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// dtype 白名单拦截：y=bool（dy/z 同为 bool，一致性检查无法拦截，由白名单拦截）
TEST_F(AtanGradTilingTest, atan_grad_tiling_dtype_bool_intercept)
{
    AtanGradUtCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("AtanGrad",
                                              {
                                                  {{{32, 32}, {32, 32}}, ge::DT_BOOL, ge::FORMAT_ND},
                                                  {{{32, 32}, {32, 32}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{32, 32}, {32, 32}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// dtype 白名单拦截：y=complex128
TEST_F(AtanGradTilingTest, atan_grad_tiling_dtype_complex128_intercept)
{
    AtanGradUtCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("AtanGrad",
                                              {
                                                  {{{32, 32}, {32, 32}}, ge::DT_COMPLEX128, ge::FORMAT_ND},
                                                  {{{32, 32}, {32, 32}}, ge::DT_COMPLEX128, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{32, 32}, {32, 32}}, ge::DT_COMPLEX128, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// dtype 一致性拦截：y=fp16, dy=bool
TEST_F(AtanGradTilingTest, atan_grad_tiling_dtype_mismatch_intercept)
{
    AtanGradUtCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("AtanGrad",
                                              {
                                                  {{{32, 32}, {32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{32, 32}, {32, 32}}, ge::DT_BOOL, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{32, 32}, {32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// dtype 一致性拦截：z 与 y/dy 不一致
TEST_F(AtanGradTilingTest, atan_grad_tiling_output_dtype_mismatch_intercept)
{
    AtanGradUtCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("AtanGrad",
                                              {
                                                  {{{32, 32}, {32, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{32, 32}, {32, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{32, 32}, {32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// shape 一致性拦截：dy 与 y shape 不同（不支持广播）
TEST_F(AtanGradTilingTest, atan_grad_tiling_shape_mismatch_intercept)
{
    AtanGradUtCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("AtanGrad",
                                              {
                                                  {{{32, 32}, {32, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{64, 64}, {64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{32, 32}, {32, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
