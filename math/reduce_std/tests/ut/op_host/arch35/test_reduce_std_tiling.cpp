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
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../../reduce_var/op_host/arch35/reduce_var_tiling.h"

using namespace std;
using namespace ge;

class ReduceStdTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ReduceStdTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ReduceStdTiling TearDown" << std::endl; }
};

TEST_F(ReduceStdTiling, ReduceStd_test_tiling_001)
{
    optiling::ReduceVarCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "ReduceStd",
        {
            {{{13, 5, 9, 13, 7, 13}, {13, 5, 9, 13, 7, 13}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{13, 5, 9, 13, 7, 13}, {13, 5, 9, 13, 7, 13}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{13, 5, 9, 13, 7, 13}, {13, 5, 9, 13, 7, 13}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 2, 3, 4, 5})),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 12350;
    string expectTilingData = "1 3 2 9 169 832 19 5 45568 512 64 3959346946488926208 1 13 5 10647 0 0 0 0 0 692055 "
                              "53235 10647 1 0 0 0 0 0 5 5 1 1 0 0 0 0 0 0 0 1 16384 3959346947410783506 ";
    std::vector<size_t> expectWorkspaces = {16809984};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ReduceStdTiling, ReduceStd_test_tiling_002)
{
    // unbiased = true -> correction = 1，tiling key 与 reduce_var/reduce_std_v2 同构
    optiling::ReduceVarCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "ReduceStd",
        {
            {{{64, 10240}, {64, 10240}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{64, 1}, {64, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1}, {64, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 5142;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ReduceStdTiling, ReduceStd_test_invalid_dtype_bf16)
{
    // ReduceStd 只支持 fp16/fp32，bf16 虽能通过 ReduceVar 的 tiling 校验，但应被 ReduceStd 显式拒绝
    optiling::ReduceVarCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "ReduceStd",
        {
            {{{64, 10240}, {64, 10240}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{64, 1}, {64, 1}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{64, 1}, {64, 1}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
         gert::TilingContextPara::OpAttr("unbiased", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
