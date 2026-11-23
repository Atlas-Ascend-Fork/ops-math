/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_confusion_transpose_d_tiling_arch35.cpp
 * \brief
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/confusion_transpose_d_tiling_arch35.h"
#include "conversion/transpose/op_kernel/arch35/transpose_tiling_data.h"
#include "conversion/transpose/op_kernel/arch35/transpose_tiling_key.h"

using namespace ge;

namespace {
/**
 * @brief 从 TilingInfo 取实际 TilingData 并按类型断言（不使用序列化字符串）
 *
 * ConfusionTransposeD 的 TilingData 由 Transpose 侧统一写入，关联入口按简化后维数
 * 选择 10012（TransposeSmallShape2DTilingData）或 10011（<3> 精简结构）。
 */
template <typename T>
static const T* GetTilingData(const TilingInfo& tilingInfo)
{
    EXPECT_NE(tilingInfo.tilingData, nullptr);
    EXPECT_EQ(tilingInfo.tilingDataSize, sizeof(T));
    if (tilingInfo.tilingData == nullptr || tilingInfo.tilingDataSize != sizeof(T)) {
        return nullptr;
    }
    return reinterpret_cast<const T*>(tilingInfo.tilingData.get());
}
} // namespace

class ConfusionTransposeDTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ConfusionTransposeDTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ConfusionTransposeDTilingTest TearDown" << std::endl; }
};

TEST_F(ConfusionTransposeDTilingTest, ConfusionTransposeDTiling_Not_Transpose_First)
{
    // compile info
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara(
        "ConfusionTransposeD",
        {
            {{{21, 27, 14, 6}, {21, 27, 14, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{162, 14, 21}, {162, 14, 21}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("perm", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 0})),
         gert::TilingContextPara::OpAttr("shape", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({21, 162, 14})),
         gert::TilingContextPara::OpAttr("transpose_first", Ops::Math::AnyValue::CreateFrom<bool>(false))},
        &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    // 简化后为 2D perm=[1,0] 且数据量 186KB ∈ [64KB, 4MB)：命中 SmallShape2D(10012)
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE_2D));
    EXPECT_EQ(tilingInfo.blockNum, 64U);
    const auto* data = GetTilingData<Transpose::TransposeSmallShape2DTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->rows, 21U);
    EXPECT_EQ(data->cols, 2268U);
    EXPECT_EQ(data->chunkCols, 32U);
    EXPECT_EQ(data->totalWork, 71U);
}

TEST_F(ConfusionTransposeDTilingTest, ConfusionTransposeDTiling_Transpose_First)
{
    // compile info
    optiling::TransposeCompilerInfo compileInfo;

    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara(
        "ConfusionTransposeD",
        {
            {{{6, 31, 18, 4}, {6, 31, 18, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{6, 124, 18}, {6, 124, 18}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("perm", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({3, 2, 0, 1})),
         gert::TilingContextPara::OpAttr("shape", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({6, 124, 18})),
         gert::TilingContextPara::OpAttr("transpose_first", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    // 简化后为 3D：命中 SmallShape23D(10011)，各维取值互不相同，便于发现维序错误
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE_23D));
    EXPECT_EQ(tilingInfo.blockNum, 53U);
    const auto* data = GetTilingData<Transpose::TransposeSmallShape23DTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->dim, 3);
    EXPECT_THAT(data->inputShape, ::testing::ElementsAreArray(std::vector<int32_t>{186, 18, 4}));
    EXPECT_THAT(data->perm, ::testing::ElementsAreArray(std::vector<int8_t>{2, 1, 0}));
    EXPECT_EQ(data->blkFactor, 256U);
    EXPECT_EQ(data->blkTailFactor, 80U);
}

TEST_F(ConfusionTransposeDTilingTest, ConfusionTransposeDTiling_Fail_01)
{
    // compile info
    optiling::TransposeCompilerInfo compileInfo;

    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara(
        "ConfusionTransposeD",
        {
            {{{21, 27, 16, 16}, {21, 27, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{3, 189, 16, 16}, {3, 189, 16, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("perm", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})),
         gert::TilingContextPara::OpAttr("shape", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({3024, 48})),
         gert::TilingContextPara::OpAttr("transpose_first", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectTilingKey = 10001;
    std::vector<size_t> expectWorkspaces = {optiling::DEFAULT_WORKSPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(ConfusionTransposeDTilingTest, ConfusionTransposeDTiling_Fail_02)
{
    // compile info
    optiling::TransposeCompilerInfo compileInfo;

    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara(
        "ConfusionTransposeD",
        {
            {{{6, 31, 18, 4}, {6, 31, 18, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{6, 124, 18}, {6, 124, 18}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("perm", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({5, 2, 0, 1})),
         gert::TilingContextPara::OpAttr("shape", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({6, 124, 18})),
         gert::TilingContextPara::OpAttr("transpose_first", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);

    uint64_t expectTilingKey = 10001;
    std::vector<size_t> expectWorkspaces = {optiling::DEFAULT_WORKSPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(ConfusionTransposeDTilingTest, ConfusionTransposeDTiling_Fail_03)
{
    // compile info
    optiling::TransposeCompilerInfo compileInfo;

    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara(
        "ConfusionTransposeD",
        {
            {{{21, 27, 14, 6}, {21, 27, 14, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{162, 14, 21}, {162, 14, 21}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("perm", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 0})),
         gert::TilingContextPara::OpAttr("shape", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({21, 162, 13})),
         gert::TilingContextPara::OpAttr("transpose_first", Ops::Math::AnyValue::CreateFrom<bool>(false))},
        &compileInfo);

    uint64_t expectTilingKey = 10001;
    std::vector<size_t> expectWorkspaces = {optiling::DEFAULT_WORKSPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(ConfusionTransposeDTilingTest, ConfusionTransposeDTiling_Fail_04)
{
    // compile info
    optiling::TransposeCompilerInfo compileInfo;

    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;

    gert::TilingContextPara tilingContextPara(
        "ConfusionTransposeD",
        {
            {{{0, 27, 14, 6}, {0, 27, 14, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{162, 14, 0}, {162, 14, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("perm", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 0})),
         gert::TilingContextPara::OpAttr("shape", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 162, 14})),
         gert::TilingContextPara::OpAttr("transpose_first", Ops::Math::AnyValue::CreateFrom<bool>(false))},
        &compileInfo);

    uint64_t expectTilingKey = 10001;
    std::vector<size_t> expectWorkspaces = {optiling::DEFAULT_WORKSPACE_SIZE};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}
