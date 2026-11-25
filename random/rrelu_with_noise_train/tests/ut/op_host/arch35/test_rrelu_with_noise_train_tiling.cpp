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
 * \file test_rrelu_with_noise_train_tiling.cpp
 * \brief RreluWithNoiseTrain tiling UT
 */

#include <iostream>
#include <gtest/gtest.h>
#include <vector>
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/rrelu_with_noise_train_tiling_arch35.h"

// expectTilingData 为 RandomUnifiedSimtTilingDataStruct 按 int64 逐词内存序列化（共 68 词）：
// [0..9] 公共头部（usedCoreNum/outputSize/seed/offset/ubSize(未填充)/extraInt64Param1/
//       prob|extraFloat32Param1 打包/from/range/fromFp32|toFp32 打包），
// [10] splitBlockCount，[11..50] splitBlocks[8]（每块 {numel, gmOffset, grid, totalThreads, kernelOffset}），
// [51] probDims(int32+padding)，[52..59] probSizes[8]，[60..67] probStrides[8]（本算子不使用，全 0）
// 本算子 lower/upper 填入 fromFp32/toFp32 槽位（第 9 词低 32 位为 lower 位模式、高 32 位为 upper 位模式）；
// seed/offset 为 INT64 输入张量(各 1 元素), tiling 侧恒置 0(kernel 运行时从 GM 读取),
// splitBlock 的 kernelOffset 仅累积块前缀(单块为 0), 运行时流位置 = kernelOffset + realOffset。
// 单 tilingKey=100（dtype 变体由构建期 DTYPE_X 承担，fp16/bf16/fp32 期望值一致）。
// words[16..50]: splitBlocks[1..7] 全 0（7 块 × 5 词 = 35 个 0）
#define BLOCKS_TAIL_35 "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
// words[21..50]: splitBlocks[2..7] 全 0（6 块 × 5 词 = 30 个 0）
#define BLOCKS_TAIL_30 "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
// words[51..67]: probDims/probSizes/probStrides（17 个 0）
#define PROB_TAIL_17 "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "

class RreluWithNoiseTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RreluWithNoiseTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "RreluWithNoiseTiling TearDown" << std::endl; }
};

// fp32 基础用例: 单 splitBlock, 默认区间
// usedCoreNum=64, block0={300000, 0, 624(封顶), 159744, 0(前缀, 运行时加 realOffset)}
TEST_F(RreluWithNoiseTiling, rrelu_with_noise_tiling_fp32)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 131072};
    gert::TilingContextPara tilingContextPara(
        "RreluWithNoiseTrain",
        {{{{300000}, {300000}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND}},
        {{{{300000}, {300000}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{300000}, {300000}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("lower", Ops::Math::AnyValue::CreateFrom<float>(0.125f)),
         gert::TilingContextPara::OpAttr("upper", Ops::Math::AnyValue::CreateFrom<float>(0.333333f))},
        &compileInfo);

    uint64_t expectTilingKey = 100;
    // word[9] = 0.125f(0x3E000000) | 0.333333f(0x3EAAAA9F)<<32; word[2]/[3] = 0(seed/offset 桩)
    string expectTilingData = "64 300000 0 0 0 0 0 0 0 4515609177309052928 1 300000 0 624 159744 0 " BLOCKS_TAIL_35
        PROB_TAIL_17;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// fp16: tilingKey 仍为 100, tiling 数据与 fp32 完全一致(DTYPE_X 构建期变体)
TEST_F(RreluWithNoiseTiling, rrelu_with_noise_tiling_fp16)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 131072};
    gert::TilingContextPara tilingContextPara(
        "RreluWithNoiseTrain",
        {{{{300000}, {300000}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND}},
        {{{{300000}, {300000}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{300000}, {300000}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("lower", Ops::Math::AnyValue::CreateFrom<float>(0.125f)),
         gert::TilingContextPara::OpAttr("upper", Ops::Math::AnyValue::CreateFrom<float>(0.333333f))},
        &compileInfo);

    uint64_t expectTilingKey = 100;
    string expectTilingData = "64 300000 0 0 0 0 0 0 0 4515609177309052928 1 300000 0 624 159744 0 " BLOCKS_TAIL_35
        PROB_TAIL_17;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// bf16: tilingKey 仍为 100, tiling 数据与 fp32 完全一致
TEST_F(RreluWithNoiseTiling, rrelu_with_noise_tiling_bf16)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 131072};
    gert::TilingContextPara tilingContextPara(
        "RreluWithNoiseTrain",
        {{{{300000}, {300000}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND}},
        {{{{300000}, {300000}}, ge::DT_BF16, ge::FORMAT_ND}, {{{300000}, {300000}}, ge::DT_BF16, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("lower", Ops::Math::AnyValue::CreateFrom<float>(0.125f)),
         gert::TilingContextPara::OpAttr("upper", Ops::Math::AnyValue::CreateFrom<float>(0.333333f))},
        &compileInfo);

    uint64_t expectTilingKey = 100;
    string expectTilingData = "64 300000 0 0 0 0 0 0 0 4515609177309052928 1 300000 0 624 159744 0 " BLOCKS_TAIL_35
        PROB_TAIL_17;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 多维 shape + 自定义区间: grid 未触顶(199 < 624)
// word[9] = -0.5f(0xBF000000) | 2.0f(0x40000000)<<32
TEST_F(RreluWithNoiseTiling, rrelu_with_noise_tiling_fp32_multidim_custom_range)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 131072};
    gert::TilingContextPara tilingContextPara(
        "RreluWithNoiseTrain",
        {{{{37, 86, 16}, {37, 86, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND}},
        {{{{37, 86, 16}, {37, 86, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{37, 86, 16}, {37, 86, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("lower", Ops::Math::AnyValue::CreateFrom<float>(-0.5f)),
         gert::TilingContextPara::OpAttr("upper", Ops::Math::AnyValue::CreateFrom<float>(2.0f))},
        &compileInfo);

    uint64_t expectTilingKey = 100;
    // numel=37*86*16=50912, grid=ceil(50912/256)=199(未封顶), totalThreads=199*256=50944
    string expectTilingData = "64 50912 0 0 0 0 0 0 0 4611686021631836160 1 50912 0 199 50944 0 " BLOCKS_TAIL_35
        PROB_TAIL_17;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 超大 shape 触发多 splitBlock(fp32 字节跨度超 2^31): numel=536870914=2^29+2
// splitBlockCount=2, 两块各 268435457; kernelOffset 仅累积块前缀(realOffset 运行时加):
// 块0 = C(536870914)=3364; 块1 = 3364 + C(268435457)=3364+1684=5048
TEST_F(RreluWithNoiseTiling, rrelu_with_noise_tiling_fp32_multi_split_blocks)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 131072};
    gert::TilingContextPara tilingContextPara(
        "RreluWithNoiseTrain",
        {{{{536870914}, {536870914}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND}},
        {{{{536870914}, {536870914}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{536870914}, {536870914}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("lower", Ops::Math::AnyValue::CreateFrom<float>(0.125f)),
         gert::TilingContextPara::OpAttr("upper", Ops::Math::AnyValue::CreateFrom<float>(0.75f))},
        &compileInfo);

    uint64_t expectTilingKey = 100;
    // word[9] = 0.125f(0x3E000000) | 0.75f(0x3F400000)<<32
    string expectTilingData = "64 536870914 0 0 0 0 0 0 0 4557642823939129344 2 "
                              "268435457 0 624 159744 3364 268435457 268435457 624 159744 5048 " BLOCKS_TAIL_30
                                  PROB_TAIL_17;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 正例: lower > upper 不再拦截(不限制值域), 值原样填入 fromFp32/toFp32
TEST_F(RreluWithNoiseTiling, rrelu_with_noise_tiling_lower_greater_than_upper)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 131072};
    gert::TilingContextPara tilingContextPara(
        "RreluWithNoiseTrain",
        {{{{300000}, {300000}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND}},
        {{{{300000}, {300000}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{300000}, {300000}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("lower", Ops::Math::AnyValue::CreateFrom<float>(0.5f)),
         gert::TilingContextPara::OpAttr("upper", Ops::Math::AnyValue::CreateFrom<float>(0.125f))},
        &compileInfo);

    uint64_t expectTilingKey = 100;
    // word[9] = 0.5f(0x3F000000) | 0.125f(0x3E000000)<<32
    string expectTilingData = "64 300000 0 0 0 0 0 0 0 4467570831408496640 1 300000 0 624 159744 0 " BLOCKS_TAIL_35
        PROB_TAIL_17;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 负例: x 为不支持的 dtype(DOUBLE)
TEST_F(RreluWithNoiseTiling, rrelu_with_noise_tiling_unsupported_dtype)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 131072};
    gert::TilingContextPara tilingContextPara(
        "RreluWithNoiseTrain",
        {{{{300000}, {300000}}, ge::DT_DOUBLE, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND}},
        {{{{300000}, {300000}}, ge::DT_DOUBLE, ge::FORMAT_ND}, {{{300000}, {300000}}, ge::DT_DOUBLE, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("lower", Ops::Math::AnyValue::CreateFrom<float>(0.125f)),
         gert::TilingContextPara::OpAttr("upper", Ops::Math::AnyValue::CreateFrom<float>(0.333333f))},
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 负例: seed 张量 dtype 非 INT64
TEST_F(RreluWithNoiseTiling, rrelu_with_noise_tiling_seed_wrong_dtype)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 131072};
    gert::TilingContextPara tilingContextPara(
        "RreluWithNoiseTrain",
        {{{{300000}, {300000}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND}},
        {{{{300000}, {300000}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{300000}, {300000}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("lower", Ops::Math::AnyValue::CreateFrom<float>(0.125f)),
         gert::TilingContextPara::OpAttr("upper", Ops::Math::AnyValue::CreateFrom<float>(0.333333f))},
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 负例: offset 张量元素数非 1
TEST_F(RreluWithNoiseTiling, rrelu_with_noise_tiling_offset_wrong_shape)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 131072};
    gert::TilingContextPara tilingContextPara(
        "RreluWithNoiseTrain",
        {{{{300000}, {300000}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND}},
        {{{{300000}, {300000}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{300000}, {300000}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("lower", Ops::Math::AnyValue::CreateFrom<float>(0.125f)),
         gert::TilingContextPara::OpAttr("upper", Ops::Math::AnyValue::CreateFrom<float>(0.333333f))},
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 负例: noise shape 与 y shape 不一致
TEST_F(RreluWithNoiseTiling, rrelu_with_noise_tiling_noise_shape_mismatch)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 131072};
    gert::TilingContextPara tilingContextPara(
        "RreluWithNoiseTrain",
        {{{{300000}, {300000}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND}},
        {{{{300000}, {300000}}, ge::DT_FLOAT, ge::FORMAT_ND}, {{{299999}, {299999}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("lower", Ops::Math::AnyValue::CreateFrom<float>(0.125f)),
         gert::TilingContextPara::OpAttr("upper", Ops::Math::AnyValue::CreateFrom<float>(0.333333f))},
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
