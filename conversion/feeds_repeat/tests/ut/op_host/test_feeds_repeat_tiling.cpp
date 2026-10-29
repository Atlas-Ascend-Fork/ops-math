/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/feeds_repeat_tiling.h"

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace gert;
using namespace optiling;

class FeedsRepeatTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "FeedsRepeat Tiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "FeedsRepeat Tiling TearDown" << std::endl; }
};

TEST_F(FeedsRepeatTiling, test_tiling_fp16_int32)
{
    optiling::FeedsRepeatCompileInfo compileInfo = {48, 196608};

    gert::TilingContextPara tilingContextPara(
        "FeedsRepeat",
        {{{{4, 5, 6, 7}, {4, 5, 6, 7}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{
               4,
           },
           {
               4,
           }},
          ge::DT_INT32,
          ge::FORMAT_ND}},
        {
            {{{15, 5, 6, 7}, {15, 5, 6, 7}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("output_feeds_size", Ops::Math::AnyValue::CreateFrom<int64_t>(15))},
        &compileInfo);
    uint64_t expectTilingKey = 2;
    string expectTilingData = "34359738372 210 224 64 16 0 15 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FeedsRepeatTiling, test_tiling_fp32_int32)
{
    optiling::FeedsRepeatCompileInfo compileInfo = {48, 196608};

    gert::TilingContextPara tilingContextPara(
        "FeedsRepeat",
        {{{{4, 5, 6, 7}, {4, 5, 6, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{
               4,
           },
           {
               4,
           }},
          ge::DT_INT32,
          ge::FORMAT_ND}},
        {
            {{{15, 5, 6, 7}, {15, 5, 6, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("output_feeds_size", Ops::Math::AnyValue::CreateFrom<int64_t>(15))},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    string expectTilingData = "34359738372 210 216 64 16 0 15 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FeedsRepeatTiling, test_tiling_bf16_int32)
{
    optiling::FeedsRepeatCompileInfo compileInfo = {48, 196608};

    gert::TilingContextPara tilingContextPara(
        "FeedsRepeat",
        {{{{4, 50, 60, 7}, {4, 50, 60, 7}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{
               4,
           },
           {
               4,
           }},
          ge::DT_INT32,
          ge::FORMAT_ND}},
        {
            {{{15, 50, 60, 7}, {15, 50, 60, 7}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("output_feeds_size", Ops::Math::AnyValue::CreateFrom<int64_t>(15))},
        &compileInfo);
    uint64_t expectTilingKey = 3;
    string expectTilingData = "34359738372 21000 21008 64 16 0 15 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FeedsRepeatTiling, test_tiling_fp32_int64)
{
    optiling::FeedsRepeatCompileInfo compileInfo = {48, 196608};

    gert::TilingContextPara tilingContextPara(
        "FeedsRepeat",
        {{{{48, 5, 6, 7}, {48, 5, 6, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{
               48,
           },
           {
               48,
           }},
          ge::DT_INT64,
          ge::FORMAT_ND}},
        {
            {{{100, 5, 6, 7}, {100, 5, 6, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("output_feeds_size", Ops::Math::AnyValue::CreateFrom<int64_t>(15))},
        &compileInfo);
    uint64_t expectTilingKey = 101;
    string expectTilingData = "206158430256 210 216 64 1 16 15 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FeedsRepeatTiling, test_tiling_fp16_int64)
{
    optiling::FeedsRepeatCompileInfo compileInfo = {48, 196608};

    gert::TilingContextPara tilingContextPara(
        "FeedsRepeat",
        {{{{50, 5, 6, 7}, {50, 5, 6, 7}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{
               50,
           },
           {
               50,
           }},
          ge::DT_INT64,
          ge::FORMAT_ND}},
        {
            {{{50, 5, 6, 7}, {50, 5, 6, 7}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("output_feeds_size", Ops::Math::AnyValue::CreateFrom<int64_t>(15))},
        &compileInfo);
    uint64_t expectTilingKey = 102;
    string expectTilingData = "240518168626 210 224 64 1 14 15 1 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(FeedsRepeatTiling, test_tiling_bf16_int64)
{
    optiling::FeedsRepeatCompileInfo compileInfo = {48, 196608};

    gert::TilingContextPara tilingContextPara(
        "FeedsRepeat",
        {{{{100, 5, 6, 7}, {100, 5, 6, 7}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{
               100,
           },
           {
               100,
           }},
          ge::DT_INT64,
          ge::FORMAT_ND}},
        {
            {{{50, 5, 6, 7}, {50, 5, 6, 7}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("output_feeds_size", Ops::Math::AnyValue::CreateFrom<int64_t>(15))},
        &compileInfo);
    uint64_t expectTilingKey = 103;
    string expectTilingData = "446676598884 210 224 64 0 0 15 1 36 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// feeds 仅支持 fp32/fp16/bf16，int32 feeds 返回失败
TEST_F(FeedsRepeatTiling, test_tiling_int32_int64)
{
    optiling::FeedsRepeatCompileInfo compileInfo = {48, 196608};

    gert::TilingContextPara tilingContextPara(
        "FeedsRepeat",
        {{{{48, 5, 6, 7}, {48, 5, 6, 7}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{
               48,
           },
           {
               48,
           }},
          ge::DT_INT64,
          ge::FORMAT_ND}},
        {
            {{{100, 5, 6, 7}, {100, 5, 6, 7}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("output_feeds_size", Ops::Math::AnyValue::CreateFrom<int64_t>(15))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ===== 非法 shapes / 容量（EZ0010 / EZ0015） =====

TEST_F(FeedsRepeatTiling, test_tiling_neg_zero_batch)
{
    optiling::FeedsRepeatCompileInfo compileInfo = {48, 196608};

    gert::TilingContextPara tilingContextPara(
        "FeedsRepeat",
        {{{{0, 5, 6, 7}, {0, 5, 6, 7}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{
               0,
           },
           {
               0,
           }},
          ge::DT_INT32,
          ge::FORMAT_ND}},
        {
            {{{15, 5, 6, 7}, {15, 5, 6, 7}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("output_feeds_size", Ops::Math::AnyValue::CreateFrom<int64_t>(15))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FeedsRepeatTiling, test_tiling_neg_length_mismatch)
{
    optiling::FeedsRepeatCompileInfo compileInfo = {48, 196608};

    gert::TilingContextPara tilingContextPara(
        "FeedsRepeat",
        {{{{4, 5, 6, 7}, {4, 5, 6, 7}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{
               5,
           },
           {
               5,
           }},
          ge::DT_INT32,
          ge::FORMAT_ND}},
        {
            {{{15, 5, 6, 7}, {15, 5, 6, 7}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("output_feeds_size", Ops::Math::AnyValue::CreateFrom<int64_t>(15))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FeedsRepeatTiling, test_tiling_neg_repeat_times_dtype_invalid)
{
    optiling::FeedsRepeatCompileInfo compileInfo = {48, 196608};

    gert::TilingContextPara tilingContextPara(
        "FeedsRepeat",
        {{{{4, 5, 6, 7}, {4, 5, 6, 7}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{
               4,
           },
           {
               4,
           }},
          ge::DT_FLOAT,
          ge::FORMAT_ND}},
        {
            {{{15, 5, 6, 7}, {15, 5, 6, 7}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("output_feeds_size", Ops::Math::AnyValue::CreateFrom<int64_t>(15))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(FeedsRepeatTiling, test_tiling_neg_repeat_times_too_large)
{
    optiling::FeedsRepeatCompileInfo compileInfo = {48, 196608};

    gert::TilingContextPara tilingContextPara(
        "FeedsRepeat",
        {{{{31728, 1}, {31728, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{
               31728,
           },
           {
               31728,
           }},
          ge::DT_INT32,
          ge::FORMAT_ND}},
        {
            {{{31728, 1}, {31728, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("output_feeds_size", Ops::Math::AnyValue::CreateFrom<int64_t>(15))},
        &compileInfo, 64, 262144);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
TEST_F(FeedsRepeatTiling, test_tiling_elem_row_align_overflow)
{
    optiling::FeedsRepeatCompileInfo compileInfo = {48, 196608};

    gert::TilingContextPara tilingContextPara(
        "FeedsRepeat",
        {{{{1, INT64_MAX}, {1, INT64_MAX}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{
               1,
           },
           {
               1,
           }},
          ge::DT_INT32,
          ge::FORMAT_ND}},
        {
            {{{15, INT64_MAX}, {15, INT64_MAX}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("output_feeds_size", Ops::Math::AnyValue::CreateFrom<int64_t>(15))},
        &compileInfo);
    // elem_row=INT64_MAX 使 elem_row_aligned 的向上对齐加法溢出，tiling 应拒绝
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
TEST_F(FeedsRepeatTiling, test_tiling_elem_row_product_overflow)
{
    optiling::FeedsRepeatCompileInfo compileInfo = {48, 196608};

    gert::TilingContextPara tilingContextPara(
        "FeedsRepeat",
        {{{{1, 4, INT64_MAX / 2}, {1, 4, INT64_MAX / 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{
               1,
           },
           {
               1,
           }},
          ge::DT_INT32,
          ge::FORMAT_ND}},
        {
            {{{15, 4, INT64_MAX / 2}, {15, 4, INT64_MAX / 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("output_feeds_size", Ops::Math::AnyValue::CreateFrom<int64_t>(15))},
        &compileInfo);
    // 4 * (INT64_MAX/2) 超出 int64 表示范围，逐维累乘守卫应拒绝
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
TEST_F(FeedsRepeatTiling, test_tiling_rank_zero_rejected)
{
    optiling::FeedsRepeatCompileInfo compileInfo = {48, 196608};

    gert::TilingContextPara tilingContextPara(
        "FeedsRepeat",
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{
               1,
           },
           {
               1,
           }},
          ge::DT_INT32,
          ge::FORMAT_ND}},
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("output_feeds_size", Ops::Math::AnyValue::CreateFrom<int64_t>(1))},
        &compileInfo);
    // rank-0 输入必须在读取 GetDim(0) 之前被秩校验拒绝，不得发生越界读
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
TEST_F(FeedsRepeatTiling, test_tiling_times_ub_boundary_rejected)
{
    // 2*15864*8 + 128 + 8192 = 253952，恰触 faker 平台 UB（253952）时拒绝；
    // 15344*8 侧放行，由设备用例见证同一公式的放行边界
    optiling::FeedsRepeatCompileInfo compileInfo = {48, 196608};

    gert::TilingContextPara tilingContextPara(
        "FeedsRepeat",
        {{{{15864, 1}, {15864, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{
               15864,
           },
           {
               15864,
           }},
          ge::DT_INT64,
          ge::FORMAT_ND}},
        {
            {{{16, 1}, {16, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("output_feeds_size", Ops::Math::AnyValue::CreateFrom<int64_t>(16))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
