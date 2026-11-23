/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <type_traits>
#include <utility>
#include <map>
#include <gtest/gtest.h>
#include "err_mgr.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/stateless_randperm_tiling_arch35.h"

#include "../../../../op_kernel/stateless_randperm_struct.h"
#include "../../../../../../math/sort/op_kernel/arch35/sort_tiling_data.h"

using namespace std;
using namespace ge;

static void ExpectRawError(const char* code, const char* value)
{
    auto reports = error_message::GetErrMgrRawErrorMessages();
    ASSERT_EQ(reports.size(), 1U);
    ASSERT_NE(reports[0].error_id.get(), nullptr);
    EXPECT_STREQ(reports[0].error_id.get(), code);
    ASSERT_EQ(reports[0].args_key.size(), 4U);
    ASSERT_EQ(reports[0].args_value.size(), 4U);
    std::map<std::string, std::string> args;
    for (size_t i = 0; i < reports[0].args_key.size(); ++i) {
        ASSERT_NE(reports[0].args_key[i].get(), nullptr);
        ASSERT_NE(reports[0].args_value[i].get(), nullptr);
        args.emplace(reports[0].args_key[i].get(), reports[0].args_value[i].get());
    }
    ASSERT_EQ(args.size(), 4U);
    EXPECT_EQ(args.at("op_name"), "StatelessRandperm");
    EXPECT_EQ(args.at("param_name"), "output tensor y");
    EXPECT_EQ(args.at("incorrect_value"), value);
    EXPECT_EQ(args.at("reason"), "the max permuted value (n-1) must not be greater than max value of y dtype");
}

static void ExpectRawDtype(const char* param, const char* dtype, const char* reason)
{
    auto reports = error_message::GetErrMgrRawErrorMessages();
    ASSERT_EQ(reports.size(), 1U);
    ASSERT_NE(reports[0].error_id.get(), nullptr);
    EXPECT_STREQ(reports[0].error_id.get(), "EZ0020");
    ASSERT_EQ(reports[0].args_key.size(), 4U);
    ASSERT_EQ(reports[0].args_value.size(), 4U);
    std::map<std::string, std::string> args;
    for (size_t i = 0; i < reports[0].args_key.size(); ++i) {
        ASSERT_NE(reports[0].args_key[i].get(), nullptr);
        ASSERT_NE(reports[0].args_value[i].get(), nullptr);
        args.emplace(reports[0].args_key[i].get(), reports[0].args_value[i].get());
    }
    ASSERT_EQ(args.size(), 4U);
    EXPECT_EQ(args.at("op_name"), "StatelessRandperm");
    EXPECT_EQ(args.at("param_name"), param);
    EXPECT_EQ(args.at("incorrect_dtype"), dtype);
    EXPECT_EQ(args.at("reason"), reason);
}

template <typename StructType, std::size_t Index>
struct StructMemberType;

#define DEFINE_STRUCT_MEMBER_TYPE(StructType, Index, Member)      \
    template <>                                                   \
    struct StructMemberType<StructType, Index> {                  \
        using type = decltype(std::declval<StructType>().Member); \
    };

template <typename T1, typename T2, std::size_t I>
struct CompareStructMembers {
    static constexpr bool
        value = std::is_same_v<typename StructMemberType<T1, I>::type, typename StructMemberType<T2, I>::type> &&
                CompareStructMembers<T1, T2, I - 1>::value;
};

template <typename T1, typename T2>
struct CompareStructMembers<T1, T2, 0> {
    static constexpr bool
        value = std::is_same_v<typename StructMemberType<T1, 0>::type, typename StructMemberType<T2, 0>::type>;
};

DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingData, 0, numTileDataSize)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingData, 1, unsortedDimParallel)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingData, 2, lastDimTileNum)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingData, 3, sortLoopTimes)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingData, 4, lastDimNeedCore)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingData, 5, keyParams0)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingData, 6, keyParams1)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingData, 7, keyParams2)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingData, 8, keyParams3)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingData, 9, keyParams4)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingData, 10, keyParams5)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingData, 11, tmpUbSize)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingData, 12, lastAxisNum)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingData, 13, unsortedDimNum)

DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingDataForRandperm, 0, numTileDataSize)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingDataForRandperm, 1, unsortedDimParallel)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingDataForRandperm, 2, lastDimTileNum)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingDataForRandperm, 3, sortLoopTimes)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingDataForRandperm, 4, lastDimNeedCore)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingDataForRandperm, 5, keyParams0)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingDataForRandperm, 6, keyParams1)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingDataForRandperm, 7, keyParams2)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingDataForRandperm, 8, keyParams3)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingDataForRandperm, 9, keyParams4)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingDataForRandperm, 10, keyParams5)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingDataForRandperm, 11, tmpUbSize)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingDataForRandperm, 12, lastAxisNum)
DEFINE_STRUCT_MEMBER_TYPE(SortRegBaseTilingDataForRandperm, 13, unsortedDimNum)

bool CheckTilingDataDefinitions()
{
    constexpr size_t size1 = sizeof(SortRegBaseTilingData);
    constexpr size_t size2 = sizeof(SortRegBaseTilingDataForRandperm);

    // 结构体字节长度是否相等
    if (size2 > size1) {
        std::cout << "Error: SortRegBaseTilingDataForRandperm size (" << size2
                  << ") larger than SortRegBaseTilingData size (" << size1 << ")" << std::endl;
        return false;
    }

    // 结构体每个成员数据类型是否相等
    constexpr size_t count1 = 14;
    constexpr bool
        typesMatch = CompareStructMembers<SortRegBaseTilingData, SortRegBaseTilingDataForRandperm, count1 - 1>::value;

    if (!typesMatch) {
        std::cout << "Error: Struct member type is not the same." << std::endl;
        return false;
    }

    std::cout << "Success: Struct definitions are compatible." << std::endl;
    return true;
}

class StatelessRandpermTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "StatelessRandperm SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "StatelessRandperm TearDown" << std::endl; }
};

TEST_F(StatelessRandpermTiling, stateless_randperm_tiling_950_001)
{
    optiling::StatelessRandpermCompileInfo compileInfo = {64, 196608};
    gert::StorageShape n_shape = {{1}, {1}};
    gert::StorageShape seed_shape = {{1}, {1}};
    gert::StorageShape offset_shape = {{1}, {1}};
    gert::StorageShape out_shape = {{7}, {7}};
    auto dtype = Ops::Math::AnyValue::CreateFrom<int64_t>(0);

    vector<int64_t> shape_value = {7};
    vector<int64_t> seed_value = {7};
    vector<int64_t> offset_value = {7};

    gert::TilingContextPara tilingContextPara("StatelessRandperm",
                                              {{n_shape, ge::DT_INT64, ge::FORMAT_ND, true, shape_value.data()},
                                               {seed_shape, ge::DT_INT64, ge::FORMAT_ND, true, seed_value.data()},
                                               {offset_shape, ge::DT_INT64, ge::FORMAT_ND, true, offset_value.data()}},
                                              {{out_shape, ge::DT_INT64, ge::FORMAT_ND}},
                                              {gert::TilingContextPara::OpAttr("dtype", dtype)}, &compileInfo);
    uint64_t expectTilingKey = 16843008;
    string expectTilingData = "4294967304 4294967297 4294967297 70 30064771073 0 0 0 30064771072 30064771072 7 "
                              "4294967303 4294967297 137438953473 34359738400 0 0 7 1 ";
    std::vector<size_t> expectWorkspaces = {16777286};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 该用例拦截Sort单方面修改tilingdata，防止StatelessRandperm功能异常
TEST_F(StatelessRandpermTiling, stateless_randperm_tiling_950_struct_check)
{
    bool result = CheckTilingDataDefinitions();
    EXPECT_TRUE(result);
}

struct RandpermBoundaryCase {
    vector<int64_t> nValue;
    vector<int64_t> seedValue;
    vector<int64_t> offsetValue;
    optiling::StatelessRandpermCompileInfo compileInfo;
    gert::TilingContextPara para;

    RandpermBoundaryCase(int64_t n, ge::DataType dtype)
        : nValue{n},
          seedValue{1},
          offsetValue{0},
          compileInfo{64, 196608},
          para("StatelessRandperm",
               {{gert::StorageShape({1}, {1}), ge::DT_INT64, ge::FORMAT_ND, true, nValue.data()},
                {gert::StorageShape({1}, {1}), ge::DT_INT64, ge::FORMAT_ND, true, seedValue.data()},
                {gert::StorageShape({1}, {1}), ge::DT_INT64, ge::FORMAT_ND, true, offsetValue.data()}},
               {{gert::StorageShape({n}, {n}), dtype, ge::FORMAT_ND}},
               {gert::TilingContextPara::OpAttr("dtype",
                                                Ops::Math::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(dtype)))},
               &compileInfo)
    {}
};

static gert::TilingContextPara MakeNegativeCase(ge::DataType nType, ge::DataType seedType, ge::DataType offsetType,
                                                int64_t attrDtype)
{
    static int64_t n = 7;
    static int64_t seed = 1;
    static int64_t offset = 0;
    static optiling::StatelessRandpermCompileInfo info{64, 196608};
    return gert::TilingContextPara(
        "StatelessRandperm",
        {{gert::StorageShape({1}, {1}), nType, ge::FORMAT_ND, true, &n},
         {gert::StorageShape({1}, {1}), seedType, ge::FORMAT_ND, true, &seed},
         {gert::StorageShape({1}, {1}), offsetType, ge::FORMAT_ND, true, &offset}},
        {{gert::StorageShape({7}, {7}), ge::DT_INT64, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("layout", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
         gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(attrDtype))},
        &info);
}

TEST_F(StatelessRandpermTiling, stateless_randperm_raw_formal_failures)
{
    struct Case {
        gert::TilingContextPara para;
        const char* param;
        const char* dtype;
        const char* reason;
    };
    static const char*
        attrReason = "[attr]dtype only supports int64, int32, int16, uint8, int8, float32, float16, bfloat16";
    std::vector<Case> cases = {
        {MakeNegativeCase(ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, static_cast<int64_t>(ge::DT_DOUBLE)), "attr dtype",
         "DT_DOUBLE", attrReason},
        {MakeNegativeCase(ge::DT_FLOAT, ge::DT_INT64, ge::DT_INT64, static_cast<int64_t>(ge::DT_INT64)),
         "input tensor n", "DT_FLOAT", "input n dtype should be int64"},
        {MakeNegativeCase(ge::DT_INT64, ge::DT_FLOAT, ge::DT_INT64, static_cast<int64_t>(ge::DT_INT64)),
         "input tensor seed", "DT_FLOAT", "input seed dtype should be int64"},
        {MakeNegativeCase(ge::DT_INT64, ge::DT_INT64, ge::DT_FLOAT, static_cast<int64_t>(ge::DT_INT64)),
         "input tensor offset", "DT_FLOAT", "input offset dtype should be int64"}};
    for (auto& item : cases) {
        ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
        error_message::GetErrMgrRawErrorMessages();
        ExecuteTestCase(item.para, ge::GRAPH_FAILED);
        ExpectRawDtype(item.param, item.dtype, item.reason);
    }
}

// 输出为0..n-1的排列，最大输出值为n-1：n-1可被dtype表示时应放行，n-1溢出时应拒绝
TEST_F(StatelessRandpermTiling, stateless_randperm_tiling_950_int8_boundary)
{
    RandpermBoundaryCase acceptCase(128, ge::DT_INT8);
    RandpermBoundaryCase rejectCase(129, ge::DT_INT8);
    ExecuteTestCase(acceptCase.para, ge::GRAPH_SUCCESS, 16843010, std::vector<size_t>{16779264});
    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    error_message::GetErrMgrRawErrorMessages();
    ExecuteTestCase(rejectCase.para, ge::GRAPH_FAILED);
    ExpectRawError("EZ0026", "129");
}

TEST_F(StatelessRandpermTiling, stateless_randperm_tiling_950_uint8_boundary)
{
    RandpermBoundaryCase acceptCase(256, ge::DT_UINT8);
    RandpermBoundaryCase rejectCase(257, ge::DT_UINT8);
    ExecuteTestCase(acceptCase.para, ge::GRAPH_SUCCESS, 16843010, std::vector<size_t>{16781312});
    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    error_message::GetErrMgrRawErrorMessages();
    ExecuteTestCase(rejectCase.para, ge::GRAPH_FAILED);
    ExpectRawError("EZ0026", "257");
}

TEST_F(StatelessRandpermTiling, stateless_randperm_tiling_950_int16_boundary)
{
    RandpermBoundaryCase acceptCase(32768, ge::DT_INT16);
    RandpermBoundaryCase rejectCase(32769, ge::DT_INT16);
    ExecuteTestCase(acceptCase.para, ge::GRAPH_SUCCESS, 16908547, std::vector<size_t>{18292736});
    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    error_message::GetErrMgrRawErrorMessages();
    ExecuteTestCase(rejectCase.para, ge::GRAPH_FAILED);
    ExpectRawError("EZ0026", "32769");
}

// 浮点类型按连续整数精确表示上限校验：最大输出值n-1超出精度范围（会舍入产生重复值）时应拒绝
TEST_F(StatelessRandpermTiling, stateless_randperm_tiling_950_float16_boundary)
{
    RandpermBoundaryCase acceptCase(2049, ge::DT_FLOAT16);
    RandpermBoundaryCase rejectCase(2050, ge::DT_FLOAT16);
    ExecuteTestCase(acceptCase.para, ge::GRAPH_SUCCESS, 16843010, std::vector<size_t>{16810000});
    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    error_message::GetErrMgrRawErrorMessages();
    ExecuteTestCase(rejectCase.para, ge::GRAPH_FAILED);
    ExpectRawError("EZ0026", "2050");
}

TEST_F(StatelessRandpermTiling, stateless_randperm_tiling_950_bfloat16_boundary)
{
    RandpermBoundaryCase acceptCase(257, ge::DT_BF16);
    RandpermBoundaryCase rejectCase(258, ge::DT_BF16);
    ExecuteTestCase(acceptCase.para, ge::GRAPH_SUCCESS, 16843010, std::vector<size_t>{16781328});
    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    error_message::GetErrMgrRawErrorMessages();
    ExecuteTestCase(rejectCase.para, ge::GRAPH_FAILED);
    ExpectRawError("EZ0026", "258");
}

TEST_F(StatelessRandpermTiling, stateless_randperm_tiling_950_float_boundary)
{
    RandpermBoundaryCase acceptCase(16777217, ge::DT_FLOAT);
    RandpermBoundaryCase rejectCase(16777218, ge::DT_FLOAT);
    // 大规模n的sort workspace依赖运行环境平台信息查询结果，不校验环境相关workspace
    TilingInfo info;
    EXPECT_TRUE(ExecuteTiling(acceptCase.para, info));
    EXPECT_EQ(info.tilingKey, 16908547);
    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    error_message::GetErrMgrRawErrorMessages();
    ExecuteTestCase(rejectCase.para, ge::GRAPH_FAILED);
    ExpectRawError("EZ0026", "16777218");
}
