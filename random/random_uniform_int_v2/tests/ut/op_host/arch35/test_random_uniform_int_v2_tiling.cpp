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
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "err_mgr.h"
#include <map>
#include "../../../../op_host/arch35/random_uniform_int_v2_tiling_arch35.h"

using namespace std;
using namespace ge;

class RandomUniformIntV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RandomUniformIntV2 SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "RandomUniformIntV2 TearDown" << std::endl; }
};

TEST_F(RandomUniformIntV2Tiling, random_uniform_int_v2_tiling_950_int32_int32)
{
    optiling::RandomUniformIntV2CompileInfo compileInfo = {64, 196608};
    gert::StorageShape shape_shape = {{2}, {2}};
    gert::StorageShape min_shape = {{1}, {1}};
    gert::StorageShape max_shape = {{1}, {1}};
    gert::StorageShape offset_shape = {{1}, {1}};
    gert::StorageShape out_shape = {{32, 512}, {32, 512}};
    auto seed = Ops::Math::AnyValue::CreateFrom<int64_t>(10);
    auto seed2 = Ops::Math::AnyValue::CreateFrom<int64_t>(5);

    vector<int32_t> shape_value = {32, 512};
    vector<int32_t> min_value = {2147483646};
    vector<int32_t> max_value = {2147483647};
    vector<int64_t> offset_value = {0};

    gert::TilingContextPara tilingContextPara(
        "RandomUniformIntV2",
        {{shape_shape, ge::DT_INT32, ge::FORMAT_ND, true, shape_value.data()},
         {min_shape, ge::DT_INT32, ge::FORMAT_ND, true, min_value.data()},
         {max_shape, ge::DT_INT32, ge::FORMAT_ND, true, max_value.data()},
         {offset_shape, ge::DT_INT64, ge::FORMAT_ND, true, offset_value.data()}},
        {{out_shape, ge::DT_INT32, ge::FORMAT_ND}, {offset_shape, ge::DT_INT64, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("seed", seed), gert::TilingContextPara::OpAttr("seed2", seed2)}, &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "64 256 256 10240 10 5 16384 1 2147483646 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RandomUniformIntV2Tiling, random_uniform_int_v2_tiling_950_int32_int64)
{
    optiling::RandomUniformIntV2CompileInfo compileInfo = {64, 196608};
    gert::StorageShape shape_shape = {{2}, {2}};
    gert::StorageShape min_shape = {{1}, {1}};
    gert::StorageShape max_shape = {{1}, {1}};
    gert::StorageShape offset_shape = {{1}, {1}};
    gert::StorageShape out_shape = {{32, 512}, {32, 512}};
    auto seed = Ops::Math::AnyValue::CreateFrom<int64_t>(10);
    auto seed2 = Ops::Math::AnyValue::CreateFrom<int64_t>(5);

    vector<int32_t> shape_value = {32, 512};
    vector<int64_t> min_value = {2147483648};
    vector<int64_t> max_value = {2147483649};
    vector<int64_t> offset_value = {0};

    gert::TilingContextPara tilingContextPara(
        "RandomUniformIntV2",
        {{shape_shape, ge::DT_INT32, ge::FORMAT_ND, true, shape_value.data()},
         {min_shape, ge::DT_INT64, ge::FORMAT_ND, true, min_value.data()},
         {max_shape, ge::DT_INT64, ge::FORMAT_ND, true, max_value.data()},
         {offset_shape, ge::DT_INT64, ge::FORMAT_ND, true, offset_value.data()}},
        {{out_shape, ge::DT_INT64, ge::FORMAT_ND}, {offset_shape, ge::DT_INT64, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("seed", seed), gert::TilingContextPara::OpAttr("seed2", seed2)}, &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "64 256 256 5120 10 5 16384 1 2147483648 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RandomUniformIntV2Tiling, random_uniform_int_v2_tiling_950_int64_int32)
{
    optiling::RandomUniformIntV2CompileInfo compileInfo = {64, 196608};
    gert::StorageShape shape_shape = {{2}, {2}};
    gert::StorageShape min_shape = {{1}, {1}};
    gert::StorageShape max_shape = {{1}, {1}};
    gert::StorageShape offset_shape = {{1}, {1}};
    gert::StorageShape out_shape = {{32, 512}, {32, 512}};
    auto seed = Ops::Math::AnyValue::CreateFrom<int64_t>(10);
    auto seed2 = Ops::Math::AnyValue::CreateFrom<int64_t>(5);

    vector<int64_t> shape_value = {32, 512};
    vector<int32_t> min_value = {2};
    vector<int32_t> max_value = {5};
    vector<int64_t> offset_value = {0};

    gert::TilingContextPara tilingContextPara(
        "RandomUniformIntV2",
        {{shape_shape, ge::DT_INT64, ge::FORMAT_ND, true, shape_value.data()},
         {min_shape, ge::DT_INT32, ge::FORMAT_ND, true, min_value.data()},
         {max_shape, ge::DT_INT32, ge::FORMAT_ND, true, max_value.data()},
         {offset_shape, ge::DT_INT64, ge::FORMAT_ND, true, offset_value.data()}},
        {{out_shape, ge::DT_INT32, ge::FORMAT_ND}, {offset_shape, ge::DT_INT64, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("seed", seed), gert::TilingContextPara::OpAttr("seed2", seed2)}, &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "64 256 256 10240 10 5 16384 3 2 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(RandomUniformIntV2Tiling, random_uniform_int_v2_tiling_950_int64_int64)
{
    optiling::RandomUniformIntV2CompileInfo compileInfo = {64, 196608};
    gert::StorageShape shape_shape = {{2}, {2}};
    gert::StorageShape min_shape = {{1}, {1}};
    gert::StorageShape max_shape = {{1}, {1}};
    gert::StorageShape offset_shape = {{1}, {1}};
    gert::StorageShape out_shape = {{32, 512}, {32, 512}};
    auto seed = Ops::Math::AnyValue::CreateFrom<int64_t>(10);
    auto seed2 = Ops::Math::AnyValue::CreateFrom<int64_t>(5);

    vector<int64_t> shape_value = {32, 512};
    vector<int64_t> min_value = {2};
    vector<int64_t> max_value = {5};
    vector<int64_t> offset_value = {0};

    gert::TilingContextPara tilingContextPara(
        "RandomUniformIntV2",
        {{shape_shape, ge::DT_INT64, ge::FORMAT_ND, true, shape_value.data()},
         {min_shape, ge::DT_INT64, ge::FORMAT_ND, true, min_value.data()},
         {max_shape, ge::DT_INT64, ge::FORMAT_ND, true, max_value.data()},
         {offset_shape, ge::DT_INT64, ge::FORMAT_ND, true, offset_value.data()}},
        {{out_shape, ge::DT_INT64, ge::FORMAT_ND}, {offset_shape, ge::DT_INT64, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("seed", seed), gert::TilingContextPara::OpAttr("seed2", seed2)}, &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "64 256 256 5120 10 5 16384 3 2 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

namespace {
void ExpectParameterFailure(int scenario, const char* code, const std::map<std::string, std::string>& expected)
{
    optiling::RandomUniformIntV2CompileInfo compileInfo = {64, 196608};
    gert::StorageShape shapeShape = scenario == 1 ? gert::StorageShape{{1, 2}, {1, 2}} : gert::StorageShape{{2}, {2}};
    gert::StorageShape scalarShape = {{1}, {1}};
    gert::StorageShape outputShape = {{32, 512}, {32, 512}};
    std::vector<int64_t> shapeValue = {32, 512};
    std::vector<int32_t> minValue = {5};
    std::vector<int32_t> maxValue = {scenario == 0 ? 5 : 8};
    std::vector<int64_t> offsetValue = {0};
    auto seed = Ops::Math::AnyValue::CreateFrom<int64_t>(10);
    auto seed2 = Ops::Math::AnyValue::CreateFrom<int64_t>(5);
    gert::TilingContextPara para(
        "RandomUniformIntV2",
        {{shapeShape, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
         {scalarShape, ge::DT_INT32, ge::FORMAT_ND, true, minValue.data()},
         {scalarShape, ge::DT_INT32, ge::FORMAT_ND, true, maxValue.data()},
         {scalarShape, ge::DT_INT64, ge::FORMAT_ND, true, offsetValue.data()}},
        {{outputShape, scenario == 2 ? ge::DT_INT64 : ge::DT_INT32, ge::FORMAT_ND},
         {scalarShape, ge::DT_INT64, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("seed", seed), gert::TilingContextPara::OpAttr("seed2", seed2)}, &compileInfo);
    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    (void)error_message::GetErrMgrRawErrorMessages();
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
    const auto reports = error_message::GetErrMgrRawErrorMessages();
    ASSERT_EQ(reports.size(), 1U);
    ASSERT_NE(reports[0].error_id, nullptr);
    EXPECT_STREQ(reports[0].error_id.get(), code);
    ASSERT_EQ(reports[0].args_key.size(), reports[0].args_value.size());
    ASSERT_EQ(reports[0].args_key.size(), expected.size());
    std::map<std::string, std::string> actual;
    for (size_t i = 0; i < reports[0].args_key.size(); ++i) {
        ASSERT_NE(reports[0].args_key[i], nullptr);
        ASSERT_NE(reports[0].args_value[i], nullptr);
        ASSERT_TRUE(actual.emplace(reports[0].args_key[i].get(), reports[0].args_value[i].get()).second);
    }
    EXPECT_EQ(actual, expected);
}
} // namespace

TEST_F(RandomUniformIntV2Tiling, equal_min_max_preserves_root_error)
{
    ExpectParameterFailure(0, "EZ0027",
                           {{"op_name", "RandomUniformIntV2"},
                            {"param_names", "max, min"},
                            {"incorrect_values", "5, 5"},
                            {"reason", "max value must be greater than min value"}});
}

TEST_F(RandomUniformIntV2Tiling, shape_rank_preserves_root_error)
{
    ExpectParameterFailure(1, "EZ0012",
                           {{"op_name", "RandomUniformIntV2"},
                            {"param_name", "input shape"},
                            {"incorrect_dim", "2"},
                            {"reason", "must be 1D tensor"}});
}

TEST_F(RandomUniformIntV2Tiling, output_dtype_preserves_root_error)
{
    ExpectParameterFailure(2, "EZ0021",
                           {{"op_name", "RandomUniformIntV2"},
                            {"param_names", "out, min"},
                            {"incorrect_dtypes", "DT_INT64, DT_INT32"},
                            {"reason", "out dtype must be the same as min dtype"}});
}
