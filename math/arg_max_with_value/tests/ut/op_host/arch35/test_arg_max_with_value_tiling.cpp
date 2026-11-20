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
 * \file test_arg_max_with_value_tiling.cpp
 * \brief ArgMaxWithValue Tiling UT for ascend950 (arch35)
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/arg_common_base_tiling.h"

using namespace std;
using namespace ge;

class ArgMaxWithValueTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ArgMaxWithValueTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ArgMaxWithValueTiling TearDown" << std::endl; }
};

static optiling::ArgOpsCompileInfo MakeCompileInfo()
{
    optiling::ArgOpsCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 262144;
    compileInfo.with_value = true;
    compileInfo.vRegSize = 256;
    return compileInfo;
}

static void RunTiling(const std::string& opName, const std::vector<gert::TilingContextPara::TensorDescription>& inputs,
                      const std::vector<gert::TilingContextPara::TensorDescription>& outputs, int64_t dimension,
                      bool keepDims)
{
    optiling::ArgOpsCompileInfo compileInfo = MakeCompileInfo();
    gert::TilingContextPara tilingContextPara(
        opName, inputs, outputs,
        {gert::TilingContextPara::OpAttr("dimension", Ops::Math::AnyValue::CreateFrom<int64_t>(dimension)),
         gert::TilingContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(keepDims))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ok) << "ArgMaxWithValue tiling failed, dim=" << dimension << " keepDims=" << keepDims;
}

TEST_F(ArgMaxWithValueTiling, tiling_2d_dim0_non_last_axis_fp16)
{
    RunTiling("ArgMaxWithValue", {{{{4, 2}, {4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
              {{{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND}, {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, 0, false);
}

TEST_F(ArgMaxWithValueTiling, tiling_2d_dim1_last_axis_fp16)
{
    RunTiling("ArgMaxWithValue", {{{{4, 2}, {4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
              {{{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND}, {{{4}, {4}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, 1, false);
}

TEST_F(ArgMaxWithValueTiling, tiling_3d_dim2_last_axis_keepdims_fp16)
{
    RunTiling("ArgMaxWithValue", {{{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
              {{{{2, 3, 1}, {2, 3, 1}}, ge::DT_INT32, ge::FORMAT_ND},
               {{{2, 3, 1}, {2, 3, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
              2, true);
}

TEST_F(ArgMaxWithValueTiling, tiling_1d_dim0_last_axis_fp16)
{
    RunTiling("ArgMaxWithValue", {{{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
              {{{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, 0, false);
}

TEST_F(ArgMaxWithValueTiling, tiling_large_last_axis_fp32)
{
    RunTiling("ArgMaxWithValue", {{{{1024, 2048}, {1024, 2048}}, ge::DT_FLOAT, ge::FORMAT_ND}},
              {{{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND}, {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND}}, 1,
              false);
}

TEST_F(ArgMaxWithValueTiling, tiling_3d_dim0_first_axis_keepdims_int32)
{
    RunTiling(
        "ArgMaxWithValue", {{{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{1, 3, 4}, {1, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND}, {{{1, 3, 4}, {1, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND}},
        0, true);
}

TEST_F(ArgMaxWithValueTiling, tiling_4d_dim1_non_last_axis_bf16)
{
    RunTiling(
        "ArgMaxWithValue", {{{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{2, 4, 5}, {2, 4, 5}}, ge::DT_INT32, ge::FORMAT_ND}, {{{2, 4, 5}, {2, 4, 5}}, ge::DT_BF16, ge::FORMAT_ND}},
        1, false);
}

TEST_F(ArgMaxWithValueTiling, tiling_2d_last_axis_int64)
{
    RunTiling("ArgMaxWithValue", {{{{64, 64}, {64, 64}}, ge::DT_INT64, ge::FORMAT_ND}},
              {{{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND}, {{{64}, {64}}, ge::DT_INT64, ge::FORMAT_ND}}, 1, false);
}

TEST_F(ArgMaxWithValueTiling, tiling_3d_neg_dim_last_axis_fp16)
{
    RunTiling("ArgMaxWithValue", {{{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
              {{{{2, 3, 1}, {2, 3, 1}}, ge::DT_INT32, ge::FORMAT_ND},
               {{{2, 3, 1}, {2, 3, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
              -1, true);
}

TEST_F(ArgMaxWithValueTiling, tiling_3d_dim1_mid_axis_fp32)
{
    RunTiling("ArgMaxWithValue", {{{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}},
              {{{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND}, {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND}}, 1,
              false);
}

TEST_F(ArgMaxWithValueTiling, tiling_copy_only_rsize_1)
{
    RunTiling("ArgMaxWithValue", {{{{4, 1}, {4, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
              {{{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND}, {{{4}, {4}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, 1, false);
}

TEST_F(ArgMaxWithValueTiling, tiling_copy_only_large_sum)
{
    RunTiling("ArgMaxWithValue", {{{{100, 1}, {100, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
              {{{{100}, {100}}, ge::DT_INT32, ge::FORMAT_ND}, {{{100}, {100}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, 1,
              false);
}

TEST_F(ArgMaxWithValueTiling, tiling_copy_only_single)
{
    RunTiling("ArgMaxWithValue", {{{{1, 1}, {1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
              {{{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND}, {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, 1, false);
}

TEST_F(ArgMaxWithValueTiling, tiling_ra_cut_next_only)
{
    RunTiling(
        "ArgMaxWithValue", {{{{50, 100000}, {50, 100000}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{100000}, {100000}}, ge::DT_INT32, ge::FORMAT_ND}, {{{100000}, {100000}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, 0,
        false);
}

TEST_F(ArgMaxWithValueTiling, tiling_ar_gather_last_axis_large_a)
{
    RunTiling("ArgMaxWithValue", {{{{8192, 100}, {8192, 100}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
              {{{{8192}, {8192}}, ge::DT_INT32, ge::FORMAT_ND}, {{{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, 1,
              false);
}

TEST_F(ArgMaxWithValueTiling, tiling_ra_cut_a_first_axis)
{
    RunTiling("ArgMaxWithValue", {{{{5, 100}, {5, 100}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
              {{{{100}, {100}}, ge::DT_INT32, ge::FORMAT_ND}, {{{100}, {100}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, 0,
              false);
}

TEST_F(ArgMaxWithValueTiling, tiling_ara_gather_mid_axis)
{
    RunTiling("ArgMaxWithValue", {{{{256, 3, 256}, {256, 3, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
              {{{{256, 256}, {256, 256}}, ge::DT_INT32, ge::FORMAT_ND},
               {{{256, 256}, {256, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
              1, false);
}

TEST_F(ArgMaxWithValueTiling, tiling_group_reduce_large_r)
{
    RunTiling("ArgMaxWithValue", {{{{50, 8192}, {50, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
              {{{{50}, {50}}, ge::DT_INT32, ge::FORMAT_ND}, {{{50}, {50}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, 1, false);
}

TEST_F(ArgMaxWithValueTiling, tiling_ara_cut_a_and_next_a_normal)
{
    RunTiling("ArgMaxWithValue", {{{{2, 100, 4000}, {2, 100, 4000}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
              {{{{2, 4000}, {2, 4000}}, ge::DT_INT32, ge::FORMAT_ND},
               {{{2, 4000}, {2, 4000}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
              1, false);
}

TEST_F(ArgMaxWithValueTiling, tiling_ara_mode4_cut_ra)
{
    RunTiling("ArgMaxWithValue", {{{{2, 2000, 2}, {2, 2000, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
              {{{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND}, {{{2, 2}, {2, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, 1,
              false);
}

TEST_F(ArgMaxWithValueTiling, tiling_ara_mode2_cut_a)
{
    RunTiling("ArgMaxWithValue", {{{{100, 3, 40}, {100, 3, 40}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
              {{{{100, 40}, {100, 40}}, ge::DT_INT32, ge::FORMAT_ND},
               {{{100, 40}, {100, 40}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
              1, false);
}

TEST_F(ArgMaxWithValueTiling, tiling_ara_mode3_cut_r)
{
    RunTiling(
        "ArgMaxWithValue", {{{{100, 8000, 8}, {100, 8000, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{{100, 8}, {100, 8}}, ge::DT_INT32, ge::FORMAT_ND}, {{{100, 8}, {100, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, 1,
        false);
}

TEST_F(ArgMaxWithValueTiling, tiling_ra_cut_next_and_r)
{
    RunTiling("ArgMaxWithValue", {{{{200, 30000}, {200, 30000}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
              {{{{30000}, {30000}}, ge::DT_INT32, ge::FORMAT_ND}, {{{30000}, {30000}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
              0, false);
}

TEST_F(ArgMaxWithValueTiling, tiling_ara_gather_divide_by_a_nexta)
{
    RunTiling("ArgMaxWithValue", {{{{2, 3, 8192}, {2, 3, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
              {{{{2, 8192}, {2, 8192}}, ge::DT_INT32, ge::FORMAT_ND},
               {{{2, 8192}, {2, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
              1, false);
}

TEST_F(ArgMaxWithValueTiling, tiling_group_reduce_ra_branch)
{
    RunTiling("ArgMaxWithValue", {{{{1, 8192, 64}, {1, 8192, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
              {{{{1, 64}, {1, 64}}, ge::DT_INT32, ge::FORMAT_ND}, {{{1, 64}, {1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
              1, false);
}

TEST_F(ArgMaxWithValueTiling, tiling_ar_cut_r_large_rsize)
{
    RunTiling("ArgMaxWithValue", {{{{64, 70000}, {64, 70000}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
              {{{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND}, {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND}}, 1, false);
}

// ===== half路径单块载入元素数须≤uint16上限的看护 =====
// 序列化布局与ArgMaxWithValueTilingData字段声明序一致（自然对齐）
static const uint64_t UINT16_MAX_VALUE = 65535;
static const uint64_t HALF_ELEMENT_PER_BLOCK = 16; // 32B / sizeof(half)

struct ArgMaxTilingDataView {
    uint64_t tilingKey;
    uint64_t aRaMode;
    uint64_t aSize;
    uint64_t cutASize;
    uint64_t rSize;
    uint64_t cutRSize;
    uint64_t nextASize;
    uint64_t isRaSplit;
};

static ArgMaxTilingDataView RunTilingAndDecode(uint64_t coreNum, uint64_t ubSize,
                                               const std::vector<gert::TilingContextPara::TensorDescription>& inputs,
                                               const std::vector<gert::TilingContextPara::TensorDescription>& outputs,
                                               int64_t dimension)
{
    optiling::ArgOpsCompileInfo compileInfo;
    compileInfo.coreNum = coreNum;
    compileInfo.ubSize = ubSize;
    compileInfo.with_value = true;
    compileInfo.vRegSize = 256;
    gert::TilingContextPara tilingContextPara(
        "ArgMaxWithValue", inputs, outputs,
        {gert::TilingContextPara::OpAttr("dimension", Ops::Math::AnyValue::CreateFrom<int64_t>(dimension)),
         gert::TilingContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false))},
        &compileInfo);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_TRUE(ok);
    const uint8_t* raw = tilingInfo.tilingData.get();
    ArgMaxTilingDataView view;
    view.aSize = *reinterpret_cast<const uint64_t*>(raw + 0);
    view.cutASize = *reinterpret_cast<const uint16_t*>(raw + 8);
    view.rSize = *reinterpret_cast<const uint64_t*>(raw + 16);
    view.cutRSize = *reinterpret_cast<const uint16_t*>(raw + 24);
    view.nextASize = *reinterpret_cast<const uint64_t*>(raw + 32);
    view.tilingKey = *reinterpret_cast<const uint64_t*>(raw + 96);
    view.aRaMode = *reinterpret_cast<const uint64_t*>(raw + 104);
    view.isRaSplit = *reinterpret_cast<const uint8_t*>(raw + 116);
    return view;
}

static uint64_t CeilAlignVal(uint64_t value, uint64_t align) { return (value + align - 1) / align * align; }

// TilingKey 20001: 单块元素数 = cutASize * rSize
TEST_F(ArgMaxWithValueTiling, tiling_key20001_fp16_single_block_within_uint16)
{
    auto inputs = std::vector<gert::TilingContextPara::TensorDescription>{
        {{{170000, 21}, {170000, 21}}, ge::DT_FLOAT16, ge::FORMAT_ND}};
    auto outputs = std::vector<gert::TilingContextPara::TensorDescription>{
        {{{170000}, {170000}}, ge::DT_INT32, ge::FORMAT_ND}, {{{170000}, {170000}}, ge::DT_FLOAT16, ge::FORMAT_ND}};
    for (uint64_t ubSize : {262144, 393216}) {
        auto view = RunTilingAndDecode(48, ubSize, inputs, outputs, 1);
        EXPECT_EQ(view.tilingKey, 20001);
        EXPECT_LE(view.cutASize * view.rSize, UINT16_MAX_VALUE);
    }
}

// ARA MODE3: 单块元素数 = cutRSize * nextASize
TEST_F(ArgMaxWithValueTiling, tiling_ara_mode3_fp16_single_block_within_uint16)
{
    auto inputs = std::vector<gert::TilingContextPara::TensorDescription>{
        {{{64, 7000, 15}, {64, 7000, 15}}, ge::DT_FLOAT16, ge::FORMAT_ND}};
    auto outputs = std::vector<gert::TilingContextPara::TensorDescription>{
        {{{64, 15}, {64, 15}}, ge::DT_INT32, ge::FORMAT_ND}, {{{64, 15}, {64, 15}}, ge::DT_FLOAT16, ge::FORMAT_ND}};
    for (uint64_t ubSize : {262144, 393216}) {
        auto view = RunTilingAndDecode(48, ubSize, inputs, outputs, 1);
        EXPECT_EQ(view.tilingKey, 10002); // ARA_CUT_A
        EXPECT_EQ(view.aRaMode, 103);     // ARA_MODE3
        EXPECT_LE(view.cutRSize * view.nextASize, UINT16_MAX_VALUE);
    }
}

// GROUP_REDUCE AR分支(nextA>1): 单块元素数 = aSize * align(cutRSize * nextASize)
TEST_F(ArgMaxWithValueTiling, tiling_group_reduce_fp16_single_block_within_uint16)
{
    auto inputs = std::vector<gert::TilingContextPara::TensorDescription>{
        {{{1, 100000, 63}, {1, 100000, 63}}, ge::DT_FLOAT16, ge::FORMAT_ND}};
    auto outputs = std::vector<gert::TilingContextPara::TensorDescription>{
        {{{1, 63}, {1, 63}}, ge::DT_INT32, ge::FORMAT_ND}, {{{1, 63}, {1, 63}}, ge::DT_FLOAT16, ge::FORMAT_ND}};
    for (uint64_t ubSize : {262144, 393216}) {
        auto view = RunTilingAndDecode(48, ubSize, inputs, outputs, 1);
        EXPECT_EQ(view.tilingKey, 30001); // GROUP_REDUCE
        EXPECT_EQ(view.isRaSplit, 0);
        uint64_t blockElems = view.aSize * CeilAlignVal(view.cutRSize * view.nextASize, HALF_ELEMENT_PER_BLOCK);
        EXPECT_LE(blockElems, UINT16_MAX_VALUE);
    }
}
