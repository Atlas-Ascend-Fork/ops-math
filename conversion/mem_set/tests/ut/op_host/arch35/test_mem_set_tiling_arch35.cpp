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
 * \file test_mem_set_tiling_arch35.cpp
 * \brief UT for MemSet arch35 tiling
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/mem_set_tiling_arch35.h"

using namespace std;

class MemSetTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MemSetTilingArch35 SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MemSetTilingArch35 TearDown" << std::endl; }
};

TEST_F(MemSetTilingArch35, test_single_int32)
{
    optiling::MemSetCompileInfoArch35 compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "MemSet",
        {
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("sizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({512})),
         gert::TilingContextPara::OpAttr("dtypes",
                                         Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({ge::DT_INT32})),
         gert::TilingContextPara::OpAttr("values_int", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({7})),
         gert::TilingContextPara::OpAttr("values_float", Ops::Math::AnyValue::CreateFrom<std::vector<float>>({0.0f}))},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(MemSetTilingArch35, test_two_tensors_int32_float)
{
    optiling::MemSetCompileInfoArch35 compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "MemSet",
        {
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("sizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1024, 2048})),
         gert::TilingContextPara::OpAttr(
             "dtypes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({ge::DT_INT32, ge::DT_FLOAT})),
         gert::TilingContextPara::OpAttr("values_int", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({5})),
         gert::TilingContextPara::OpAttr("values_float", Ops::Math::AnyValue::CreateFrom<std::vector<float>>({1.5f}))},
        &compileInfo);
    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(MemSetTilingArch35, test_int8_multi_core)
{
    optiling::MemSetCompileInfoArch35 compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "MemSet",
        {
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("sizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({66560})),
         gert::TilingContextPara::OpAttr("dtypes",
                                         Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({ge::DT_INT8})),
         gert::TilingContextPara::OpAttr("values_int", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
         gert::TilingContextPara::OpAttr("values_float", Ops::Math::AnyValue::CreateFrom<std::vector<float>>({0.0f}))},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(MemSetTilingArch35, test_invalid_dtype)
{
    optiling::MemSetCompileInfoArch35 compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "MemSet",
        {
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("sizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({512})),
         gert::TilingContextPara::OpAttr("dtypes",
                                         Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({ge::DT_DOUBLE})),
         gert::TilingContextPara::OpAttr("values_int", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})),
         gert::TilingContextPara::OpAttr("values_float", Ops::Math::AnyValue::CreateFrom<std::vector<float>>({0.0f}))},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(MemSetTilingArch35, test_size_not_divisible_by_dtype)
{
    optiling::MemSetCompileInfoArch35 compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "MemSet",
        {
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("sizes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({513})),
         gert::TilingContextPara::OpAttr("dtypes",
                                         Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({ge::DT_INT32})),
         gert::TilingContextPara::OpAttr("values_int", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})),
         gert::TilingContextPara::OpAttr("values_float", Ops::Math::AnyValue::CreateFrom<std::vector<float>>({0.0f}))},
        &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}
