/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class MaskedFillRangeInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MaskedFillRangeInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MaskedFillRangeInfershape TearDown" << std::endl; }
};

// 属性构造辅助：axis 必填
static std::vector<gert::InfershapeContextPara::OpAttr> MakeAxisAttr(int64_t axis)
{
    return {gert::InfershapeContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(axis))};
}

// 基础 1D：y 与 x 同形
TEST_F(MaskedFillRangeInfershape, infer_shape_1d)
{
    gert::InfershapeContextPara para("MaskedFillRange",
                                     {
                                         {{{37}, {37}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 1}, {2, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{2, 1}, {2, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     MakeAxisAttr(0));
    std::vector<std::vector<int64_t>> expectOutputShape = {{37}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 2D 负轴 axis=-1：归一化后合法
TEST_F(MaskedFillRangeInfershape, infer_shape_2d_negative_axis)
{
    gert::InfershapeContextPara para("MaskedFillRange",
                                     {
                                         {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     MakeAxisAttr(-1));
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 8}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// axis=-dim 合法：canndev Verifier 的 std::abs(axis) 会误杀此用例，本实现按归一化放行
TEST_F(MaskedFillRangeInfershape, infer_shape_2d_axis_neg_dim)
{
    gert::InfershapeContextPara para("MaskedFillRange",
                                     {
                                         {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     MakeAxisAttr(-2));
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 8}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// unknown rank（-2）传播：shape 未知时跳过 axis 越界判定
TEST_F(MaskedFillRangeInfershape, infer_shape_unknown_rank)
{
    gert::InfershapeContextPara para("MaskedFillRange",
                                     {
                                         {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 1}, {2, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{2, 1}, {2, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     MakeAxisAttr(-1));
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// x 与 value dtype 不一致：拒绝（canndev Verifier 校验 1）
TEST_F(MaskedFillRangeInfershape, x_value_dtype_mismatch)
{
    gert::InfershapeContextPara para("MaskedFillRange",
                                     {
                                         {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     MakeAxisAttr(-1));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// start 与 end dtype 不一致：拒绝（canndev Verifier 校验 2）
TEST_F(MaskedFillRangeInfershape, start_end_dtype_mismatch)
{
    gert::InfershapeContextPara para("MaskedFillRange",
                                     {
                                         {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{2, 4}, {2, 4}}, ge::DT_INT64, ge::FORMAT_ND},
                                         {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     MakeAxisAttr(-1));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// axis 越界：拒绝（canndev Verifier 校验 3）
TEST_F(MaskedFillRangeInfershape, axis_out_of_range)
{
    gert::InfershapeContextPara para("MaskedFillRange",
                                     {
                                         {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     MakeAxisAttr(2));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// start 与 end dtype 相等但均非 INT32：拒绝（INT32 前置校验，tiling 校验的
// 图构建期前置化；与 start_end_dtype_mismatch 区分——该例 dtype 不等命中校验 2，
// 本例 dtype 相等穿透校验 2 后恰命中新增分支）
TEST_F(MaskedFillRangeInfershape, start_end_dtype_not_int32)
{
    gert::InfershapeContextPara para("MaskedFillRange",
                                     {
                                         {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 4}, {2, 4}}, ge::DT_INT64, ge::FORMAT_ND},
                                         {{{2, 4}, {2, 4}}, ge::DT_INT64, ge::FORMAT_ND},
                                         {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     MakeAxisAttr(-1));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
