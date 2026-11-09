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
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/scale_tiling_arch35.h"

using namespace std;

class ScaleTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ScaleTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ScaleTilingTest TearDown" << std::endl; }
};

TEST_F(ScaleTilingTest, scale_tiling_float_with_bias)
{
    optiling::ScaleCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Scale",
        {
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        {1, 1, 1}, {1}, &compileInfo);
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(result);
    EXPECT_EQ(tilingInfo.blockNum, 1);
}

TEST_F(ScaleTilingTest, scale_tiling_float_no_bias)
{
    optiling::ScaleCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Scale",
        {
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        {1, 1, 0}, {1}, &compileInfo);
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(result);
    EXPECT_EQ(tilingInfo.blockNum, 1);
}

TEST_F(ScaleTilingTest, scale_tiling_float16_with_bias)
{
    optiling::ScaleCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Scale",
        {
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        {1, 1, 1}, {1}, &compileInfo);
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(result);
}

TEST_F(ScaleTilingTest, scale_tiling_bf16)
{
    optiling::ScaleCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Scale",
        {
            {{{4, 8}, {4, 8}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{4, 8}, {4, 8}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        {1, 1, 1}, {1}, &compileInfo);
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(result);
}

TEST_F(ScaleTilingTest, scale_tiling_scalar_scale)
{
    optiling::ScaleCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Scale",
        {
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        {1, 1, 1}, {1}, &compileInfo);
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(result);
}

TEST_F(ScaleTilingTest, scale_tiling_1d)
{
    optiling::ScaleCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Scale",
        {
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        {1, 1, 0}, {1}, &compileInfo);
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(result);
}

TEST_F(ScaleTilingTest, scale_tiling_4d)
{
    optiling::ScaleCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Scale",
        {
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 5}, {4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 5}, {4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        {1, 1, 1}, {1}, &compileInfo);
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(result);
}

TEST_F(ScaleTilingTest, scale_tiling_empty_tensor)
{
    optiling::ScaleCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Scale",
        {
            {{{0, 5}, {0, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{0, 5}, {0, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{0, 5}, {0, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        {1, 1, 0}, {1}, &compileInfo);
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(result);
}

TEST_F(ScaleTilingTest, scale_tiling_unsupported_dtype)
{
    optiling::ScaleCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Scale",
        {
            {{{3, 5}, {3, 5}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{3, 5}, {3, 5}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        {1, 1, 0}, {1}, &compileInfo);
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_FALSE(result);
}

TEST_F(ScaleTilingTest, scale_tiling_scale_from_blob_false)
{
    optiling::ScaleCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Scale",
        {
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(false))},
        {1, 1, 1}, {1}, &compileInfo);
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(result);
}

TEST_F(ScaleTilingTest, scale_tiling_negative_axis)
{
    optiling::ScaleCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Scale",
        {
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        {1, 1, 0}, {1}, &compileInfo);
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(result);
}

TEST_F(ScaleTilingTest, scale_tiling_rejects_overflowing_shape_metadata)
{
    for (int64_t outer : {int64_t(1) << 62, std::numeric_limits<int64_t>::max()}) {
        optiling::ScaleCompileInfo compileInfo = {64, 262144};
        gert::TilingContextPara tilingContextPara(
            "Scale",
            {
                {{{outer, 2}, {outer, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            },
            {{{{outer, 2}, {outer, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
            {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
             gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
             gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
            {1, 1, 0}, {1}, &compileInfo);
        TilingInfo tilingInfo;
        EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
    }
}

TEST_F(ScaleTilingTest, scale_r2_reject_axis_high)
{
    optiling::ScaleCompileInfo compileInfo{64, 262144};
    gert::TilingContextPara para(
        "Scale", {{{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        {1, 1, 0}, {1}, &compileInfo);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

TEST_F(ScaleTilingTest, scale_r2_reject_axes_low)
{
    optiling::ScaleCompileInfo compileInfo{64, 262144};
    gert::TilingContextPara para(
        "Scale", {{{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(-2)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(false))},
        {1, 1, 0}, {1}, &compileInfo);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

TEST_F(ScaleTilingTest, scale_r2_reject_expand_x)
{
    optiling::ScaleCompileInfo compileInfo{64, 262144};
    gert::TilingContextPara para(
        "Scale", {{{{1, 4}, {1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{1, 4}, {1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        {1, 1, 0}, {1}, &compileInfo);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

TEST_F(ScaleTilingTest, scale_r2_reject_zero_scale)
{
    optiling::ScaleCompileInfo compileInfo{64, 262144};
    gert::TilingContextPara para(
        "Scale", {{{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        {1, 1, 0}, {1}, &compileInfo);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

TEST_F(ScaleTilingTest, scale_r2_reject_rank0)
{
    optiling::ScaleCompileInfo compileInfo{64, 262144};
    gert::TilingContextPara para(
        "Scale", {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        {1, 1, 0}, {1}, &compileInfo);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

TEST_F(ScaleTilingTest, scale_geir_rank9)
{
    optiling::ScaleCompileInfo compileInfo{64, 262144};
    gert::TilingContextPara para(
        "Scale",
        {{{{2, 1, 1, 1, 1, 1, 1, 1, 1}, {2, 1, 1, 1, 1, 1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 1, 1, 1, 1, 1, 1, 1, 1}, {2, 1, 1, 1, 1, 1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        {1, 1, 0}, {1}, &compileInfo);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

TEST_F(ScaleTilingTest, scale_r2_reject_window)
{
    optiling::ScaleCompileInfo compileInfo{64, 262144};
    gert::TilingContextPara para(
        "Scale", {{{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        {1, 1, 0}, {1}, &compileInfo);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

TEST_F(ScaleTilingTest, scale_r2_reject_bias_shape)
{
    optiling::ScaleCompileInfo compileInfo{64, 262144};
    gert::TilingContextPara para(
        "Scale",
        {{{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        {1, 1, 1}, {1}, &compileInfo);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

TEST_F(ScaleTilingTest, scale_rejects_unconverted_mixed_dtype)
{
    optiling::ScaleCompileInfo compileInfo{64, 262144};
    gert::TilingContextPara para(
        "Scale", {{{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{4}, {4}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        {1, 1, 0}, {1}, &compileInfo);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

TEST_F(ScaleTilingTest, scale_geir_noblob_4d_singleton_broadcast)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}) {
        for (int64_t axis : {int64_t(0), int64_t(-4)}) {
            optiling::ScaleCompileInfo compileInfo{64, 262144};
            gert::TilingContextPara para(
                "Scale",
                {{{{2, 4, 4, 4}, {2, 4, 4, 4}}, dtype, ge::FORMAT_ND},
                 {{{1, 4, 1, 1}, {1, 4, 1, 1}}, dtype, ge::FORMAT_ND},
                 {{{1, 4, 1, 1}, {1, 4, 1, 1}}, dtype, ge::FORMAT_ND}},
                {{{{2, 4, 4, 4}, {2, 4, 4, 4}}, dtype, ge::FORMAT_ND}},
                {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(axis)),
                 gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
                 gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(false))},
                {1, 1, 1}, {1}, &compileInfo);
            TilingInfo info;
            EXPECT_TRUE(ExecuteTiling(para, info));
            EXPECT_GT(info.blockNum, 0);
        }
    }
}

TEST_F(ScaleTilingTest, scale_geir_singleton_broadcast_keeps_blob_window_strict)
{
    optiling::ScaleCompileInfo compileInfo{64, 262144};
    gert::TilingContextPara para(
        "Scale",
        {{{{2, 4, 4, 4}, {2, 4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1, 4, 1, 1}, {1, 4, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 4, 4, 4}, {2, 4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(4)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        {1, 1, 0}, {1}, &compileInfo);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

TEST_F(ScaleTilingTest, scale_reject_rank9_with_empty_bias)
{
    optiling::ScaleCompileInfo compileInfo{64, 262144};
    gert::TilingContextPara para(
        "Scale",
        {{{{2, 2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(8)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        {1, 1, 1}, {1}, &compileInfo);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

TEST_F(ScaleTilingTest, scale_a3_nonblob_rank3_singleton)
{
    optiling::ScaleCompileInfo compileInfo{64, 262144};
    gert::TilingContextPara para(
        "Scale",
        {{{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{1, 4}, {1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(false))},
        {1, 1, 0}, {1}, &compileInfo);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

TEST_F(ScaleTilingTest, scale_a3_legacy_expanded_nonblob)
{
    for (int64_t channels : {int64_t(3), int64_t(7)}) {
        optiling::ScaleCompileInfo compileInfo{64, 262144};
        gert::TilingContextPara para(
            "Scale",
            {{{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
             {{{1, channels, 1, 1}, {1, channels, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
            {{{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
            {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
             gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
             gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(false))},
            {1, 1, 0}, {1}, &compileInfo);
        TilingInfo info;
        EXPECT_EQ(ExecuteTiling(para, info), channels == 3);
    }
}

TEST_F(ScaleTilingTest, scale_a3_dynamic_trailing_singleton_supported)
{
    optiling::ScaleCompileInfo compileInfo{64, 262144};
    gert::TilingContextPara para(
        "Scale",
        {{{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{3, 1}, {3, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(false))},
        {1, 1, 0}, {1}, &compileInfo);
    TilingInfo info;
    EXPECT_TRUE(ExecuteTiling(para, info));
}

TEST_F(ScaleTilingTest, scale_a3_legacy_expanded_4d_windows)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}) {
        for (bool blob : {false, true}) {
            for (bool valid : {false, true}) {
                const gert::StorageShape x{{2, 3, 4, 5}, {2, 3, 4, 5}};
                const gert::StorageShape scale = blob ? gert::StorageShape{{1, valid ? 3 : 1, 4, 1},
                                                                           {1, valid ? 3 : 1, 4, 1}} :
                                                        gert::StorageShape{{2, 3, valid ? 4 : 7, 1},
                                                                           {2, 3, valid ? 4 : 7, 1}};
                optiling::ScaleCompileInfo compileInfo{64, 262144};
                gert::TilingContextPara para(
                    "Scale", {{x, dtype, ge::FORMAT_ND}, {scale, dtype, ge::FORMAT_ND}, {scale, dtype, ge::FORMAT_ND}},
                    {{x, dtype, ge::FORMAT_ND}},
                    {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(blob ? 1 : 0)),
                     gert::TilingContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
                     gert::TilingContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(blob))},
                    {1, 1, 1}, {1}, &compileInfo);
                TilingInfo info;
                EXPECT_EQ(ExecuteTiling(para, info), valid);
            }
        }
    }
}
