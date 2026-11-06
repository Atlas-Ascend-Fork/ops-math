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
 * \file test_xdivy_tiling.cpp
 * \brief Xdivy Tiling UT — 2入1出 broadcast
 */
#include "math/xdivy/op_host/arch35/xdivy_tiling_arch35.h"
#include "math/xdivy/op_kernel/arch35/xdivy_struct.h"
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class XdivyTilingTest : public testing::Test {};

// ============================================================
// PadAndSqueeze (2 cases)
// ============================================================

// case1: 同形 2D — 无 broadcast, shape=(4,8)
TEST_F(XdivyTilingTest, case1_same_shape_2d)
{
    optiling::XdivyCompileInfo ci = {56, 256 * 1024};
    gert::TilingContextPara ctx("Xdivy",
                                {{{{{4, 8}, {4, 8}}, DT_FLOAT, FORMAT_ND}, {{{4, 8}, {4, 8}}, DT_FLOAT, FORMAT_ND}}},
                                {{{{{4, 8}, {4, 8}}, DT_FLOAT, FORMAT_ND}}}, &ci);
    string expect = "2 4 1 4 1 1 1 0 2 87360 1 1 4 8 2 1 1 1 4 8 1 1 4 8 0 0 8 1 0 0 8 1 1 1 4 8 0 0 8 1 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{16777216}});
}

// case2: broadcast — x=(4,1), y=(1,8) → max_bro=(4,8)
TEST_F(XdivyTilingTest, case2_broadcast_2d)
{
    optiling::XdivyCompileInfo ci = {56, 256 * 1024};
    gert::TilingContextPara ctx("Xdivy",
                                {{{{{4, 1}, {4, 1}}, DT_FLOAT, FORMAT_ND}, {{{1, 8}, {1, 8}}, DT_FLOAT, FORMAT_ND}}},
                                {{{{{4, 8}, {4, 8}}, DT_FLOAT, FORMAT_ND}}}, &ci);
    string expect = "2 4 1 4 1 1 1 0 2 87360 1 1 4 8 2 1 1 1 4 1 1 1 1 8 0 0 1 0 0 0 0 1 1 1 4 8 0 0 8 1 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{16777216}});
}

// ============================================================
// FindSplitAxis (2 cases)
// ============================================================

// case3: 末轴溢出 — shape=(100000,), FP32 P=3, per_buf_elems=21840
TEST_F(XdivyTilingTest, case3_last_dim_split_fp32)
{
    optiling::XdivyCompileInfo ci = {56, 256 * 1024};
    gert::TilingContextPara ctx(
        "Xdivy", {{{{{100000}, {100000}}, DT_FLOAT, FORMAT_ND}, {{{100000}, {100000}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{100000}, {100000}}, DT_FLOAT, FORMAT_ND}}}, &ci);
    string expect = "3 21840 5 12640 5 5 1 0 1 87360 1 1 1 100000 2 1 1 1 1 100000 1 1 1 100000 0 0 0 1 0 0 0 1 1 1 1 "
                    "100000 0 0 0 1 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{16777216}});
}

// case4: 全量入UB — shape=(64,32,8), FP32 P=3, 16384≤21840
TEST_F(XdivyTilingTest, case4_no_split_fp32)
{
    optiling::XdivyCompileInfo ci = {56, 256 * 1024};
    gert::TilingContextPara ctx(
        "Xdivy",
        {{{{{64, 32, 8}, {64, 32, 8}}, DT_FLOAT, FORMAT_ND}, {{{64, 32, 8}, {64, 32, 8}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{64, 32, 8}, {64, 32, 8}}, DT_FLOAT, FORMAT_ND}}}, &ci);
    string
        expect = "1 64 1 64 1 1 1 0 3 87360 1 64 32 8 2 1 1 64 32 8 1 64 32 8 0 256 8 1 0 256 8 1 1 64 32 8 0 256 8 1 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{16777216}});
}

// ============================================================
// MultiCoreSplit (1 case)
// ============================================================

// case5: 多核 — shape=(7,13,200,100), axis=2
TEST_F(XdivyTilingTest, case5_multi_core)
{
    optiling::XdivyCompileInfo ci = {56, 256 * 1024};
    gert::TilingContextPara ctx("Xdivy",
                                {{{{{7, 13, 200, 100}, {7, 13, 200, 100}}, DT_FLOAT, FORMAT_ND},
                                  {{{7, 13, 200, 100}, {7, 13, 200, 100}}, DT_FLOAT, FORMAT_ND}}},
                                {{{{{7, 13, 200, 100}, {7, 13, 200, 100}}, DT_FLOAT, FORMAT_ND}}}, &ci);
    string expect = "1 1 13 1 56 91 1 35 4 87360 7 13 200 100 2 1 7 13 200 100 7 13 200 100 260000 20000 100 1 260000 "
                    "20000 100 1 7 13 200 100 260000 20000 100 1 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{16777216}});
}

// ============================================================
// dtype (2 cases)
// ============================================================

// case6: FP16 — P=4, per_buf_bytes=65536, per_buf_elems=16384
TEST_F(XdivyTilingTest, case6_fp16_split)
{
    optiling::XdivyCompileInfo ci = {56, 256 * 1024};
    gert::TilingContextPara ctx(
        "Xdivy", {{{{{50000}, {50000}}, DT_FLOAT16, FORMAT_ND}, {{{50000}, {50000}}, DT_FLOAT16, FORMAT_ND}}},
        {{{{{50000}, {50000}}, DT_FLOAT16, FORMAT_ND}}}, &ci);
    string expect = "3 16384 4 848 4 4 1 0 1 65536 1 1 1 50000 2 1 1 1 1 50000 1 1 1 50000 0 0 0 1 0 0 0 1 1 1 1 50000 "
                    "0 0 0 1 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{16777216}});
}

// case7: 标量 broadcast — x=(), y=(100,) → max_bro=(100,)
TEST_F(XdivyTilingTest, case7_scalar_broadcast)
{
    optiling::XdivyCompileInfo ci = {56, 256 * 1024};
    gert::TilingContextPara ctx("Xdivy", {{{{{}, {}}, DT_FLOAT, FORMAT_ND}, {{{100}, {100}}, DT_FLOAT, FORMAT_ND}}},
                                {{{{{100}, {100}}, DT_FLOAT, FORMAT_ND}}}, &ci);
    string expect = "3 100 1 100 1 1 1 0 1 87360 1 1 1 100 2 1 1 1 1 1 1 1 1 100 0 0 0 0 0 0 0 1 1 1 1 100 0 0 0 1 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{16777216}});
}

// ============================================================
// 加固校验 (8 cases): 空 Tensor / dtype 一致性 / rank 上限 / 负维 / 溢出 / 广播
// ============================================================

// case8: 空 Tensor 同形 — (0,100000)：输出 0 元素走空分支，
// 覆盖原 MultiCoreSplit total_tiles=0、num_cores=0 时的 0/0 整除零缺陷
TEST_F(XdivyTilingTest, case8_empty_tensor_same_shape)
{
    optiling::XdivyCompileInfo ci = {56, 256 * 1024};
    gert::TilingContextPara ctx(
        "Xdivy",
        {{{{{0, 100000}, {0, 100000}}, DT_FLOAT, FORMAT_ND}, {{{0, 100000}, {0, 100000}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{0, 100000}, {0, 100000}}, DT_FLOAT, FORMAT_ND}}}, &ci);
    ExecuteTestCase(ctx, GRAPH_SUCCESS, GET_TPL_TILING_KEY(XDIVY_RANK_4, XDIVY_USER_DEF_EMPTY), "0 ", {{16777216}});
}

// case9: 空 Tensor 0-vs-1 广播 — x1=(0,), x2=(1,) → 广播结果 (0,)。
// 覆盖原 PadAndSqueeze max(0,1)=1 导致 Kernel 对 0 元素张量越界读写的缺陷
TEST_F(XdivyTilingTest, case9_empty_tensor_zero_vs_one_broadcast)
{
    optiling::XdivyCompileInfo ci = {56, 256 * 1024};
    gert::TilingContextPara ctx("Xdivy", {{{{{0}, {0}}, DT_FLOAT, FORMAT_ND}, {{{1}, {1}}, DT_FLOAT, FORMAT_ND}}},
                                {{{{{0}, {0}}, DT_FLOAT, FORMAT_ND}}}, &ci);
    ExecuteTestCase(ctx, GRAPH_SUCCESS, GET_TPL_TILING_KEY(XDIVY_RANK_4, XDIVY_USER_DEF_EMPTY), "0 ", {{16777216}});
}

// case10: 混合 dtype — x1=FLOAT, x2=FLOAT16：README 要求 x1/x2/y dtype 一致，
// Kernel 以 DTYPE_X1 单模板实例化两输入，混合组合需在 Tiling 入口拒绝
TEST_F(XdivyTilingTest, case10_mixed_dtype_rejected)
{
    optiling::XdivyCompileInfo ci = {56, 256 * 1024};
    gert::TilingContextPara ctx("Xdivy",
                                {{{{{4, 8}, {4, 8}}, DT_FLOAT, FORMAT_ND}, {{{4, 8}, {4, 8}}, DT_FLOAT16, FORMAT_ND}}},
                                {{{{{4, 8}, {4, 8}}, DT_FLOAT, FORMAT_ND}}}, &ci);
    ExecuteTestCase(ctx, GRAPH_FAILED);
}

// case11: rank 超上限 — 9 维输入：README 约束维度 ≤ 8，TilingData 定长数组
// 容量为 kRank=8，超限需显式拒绝，覆盖原负下标越界写 TilingData 的缺陷
TEST_F(XdivyTilingTest, case11_rank_gt_8_rejected)
{
    optiling::XdivyCompileInfo ci = {56, 256 * 1024};
    gert::TilingContextPara ctx("Xdivy",
                                {{{{{2, 2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2, 2}}, DT_FLOAT, FORMAT_ND},
                                  {{{2, 2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2, 2}}, DT_FLOAT, FORMAT_ND}}},
                                {{{{{2, 2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2, 2}}, DT_FLOAT, FORMAT_ND}}},
                                &ci);
    ExecuteTestCase(ctx, GRAPH_FAILED);
}

// case12: 负维度 — (-1,4)：未知维不应以负值进入 Tiling 算术
TEST_F(XdivyTilingTest, case12_negative_dim_rejected)
{
    optiling::XdivyCompileInfo ci = {56, 256 * 1024};
    gert::TilingContextPara ctx("Xdivy",
                                {{{{{-1, 4}, {-1, 4}}, DT_FLOAT, FORMAT_ND}, {{{4, 4}, {4, 4}}, DT_FLOAT, FORMAT_ND}}},
                                {{{{{-1, 4}, {-1, 4}}, DT_FLOAT, FORMAT_ND}}}, &ci);
    ExecuteTestCase(ctx, GRAPH_FAILED);
}

// case13: 元素总数溢出 — (2^62,2)：单输入可分配描述但 numel 超 int64，
// 覆盖原 FindSplitAxis/MultiCoreSplit 无保护乘加产生负值计划参数的缺陷
TEST_F(XdivyTilingTest, case13_numel_overflow_rejected)
{
    optiling::XdivyCompileInfo ci = {56, 256 * 1024};
    gert::TilingContextPara ctx("Xdivy",
                                {{{{{4611686018427387904, 2}, {4611686018427387904, 2}}, DT_FLOAT, FORMAT_ND},
                                  {{{4611686018427387904, 2}, {4611686018427387904, 2}}, DT_FLOAT, FORMAT_ND}}},
                                {{{{{4611686018427387904, 2}, {4611686018427387904, 2}}, DT_FLOAT, FORMAT_ND}}}, &ci);
    ExecuteTestCase(ctx, GRAPH_FAILED);
}

// case14: 广播结果不一致 — x1=(4,1), x2=(1,8) 广播应为 (4,8)，伪造 y=(4,4) 需拒绝
TEST_F(XdivyTilingTest, case14_broadcast_result_mismatch_rejected)
{
    optiling::XdivyCompileInfo ci = {56, 256 * 1024};
    gert::TilingContextPara ctx("Xdivy",
                                {{{{{4, 1}, {4, 1}}, DT_FLOAT, FORMAT_ND}, {{{1, 8}, {1, 8}}, DT_FLOAT, FORMAT_ND}}},
                                {{{{{4, 4}, {4, 4}}, DT_FLOAT, FORMAT_ND}}}, &ci);
    ExecuteTestCase(ctx, GRAPH_FAILED);
}

// case15: 输入不可广播 — x1=(4,2), x2=(3,2)
TEST_F(XdivyTilingTest, case15_incompatible_broadcast_rejected)
{
    optiling::XdivyCompileInfo ci = {56, 256 * 1024};
    gert::TilingContextPara ctx("Xdivy",
                                {{{{{4, 2}, {4, 2}}, DT_FLOAT, FORMAT_ND}, {{{3, 2}, {3, 2}}, DT_FLOAT, FORMAT_ND}}},
                                {{{{{4, 2}, {4, 2}}, DT_FLOAT, FORMAT_ND}}}, &ci);
    ExecuteTestCase(ctx, GRAPH_FAILED);
}
