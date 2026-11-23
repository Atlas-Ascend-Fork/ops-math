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
 * \file test_depth_to_space_tiling.cpp
 * \brief Unit tests for DepthToSpace tiling
 */
#include <cstdint>
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

#include "../../../../op_host/arch35/depth_to_space_tiling_arch35.h"
#include "conversion/transpose/op_kernel/arch35/transpose_tiling_data.h"
#include "conversion/transpose/op_kernel/arch35/transpose_tiling_key.h"

using namespace std;
using namespace ge;

namespace {
/**
 * @brief 从 TilingInfo 取实际 TilingData 并按类型断言（不使用序列化字符串）
 *
 * DepthToSpace 的 TilingData 由 Transpose 侧统一写入，关联入口 4-8D 固定使用
 * TransposeSmallShape48DTilingData；此处同时校验码流大小，便于定位布局不一致问题。
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

class DepthToSpaceTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DepthToSpaceTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "DepthToSpaceTiling TearDown" << std::endl; }
};

// Test case 1: NHWC format with DCR mode (default scenario)
// Input shape: [2, 2, 3, 12], block_size=2
// Output shape: [2, 4, 6, 3]
TEST_F(DepthToSpaceTiling, test_nhwc_dcr_float16_success_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "DepthToSpace",
        {
            {{{2, 2, 3, 12}, {2, 2, 3, 12}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {{{2, 4, 6, 3}, {2, 4, 6, 3}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("DCR")),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE));
    EXPECT_EQ(tilingInfo.blockNum, 1U);
    // 关联入口 4-8D 统一使用 TransposeSmallShape48DTilingData（精简结构）
    const auto* data = GetTilingData<Transpose::TransposeSmallShape48DTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    // [2,2,3,12] NHWC + DCR 展开并简化后：inShape {4,3,2,6}, perm {0,2,1,3}
    // 各维取值互不相同，可及时发现维序/轴映射错误；未用槽位 shape 补 0、perm 补 -1
    EXPECT_EQ(data->dim, 4);
    EXPECT_THAT(data->inputShape, ::testing::ElementsAreArray(std::vector<int32_t>{4, 3, 2, 6, 0, 0, 0, 0}));
    EXPECT_THAT(data->perm, ::testing::ElementsAreArray(std::vector<int8_t>{0, 2, 1, 3, -1, -1, -1, -1}));
    EXPECT_EQ(data->blkFactor, 144U);
    EXPECT_EQ(data->blkTailFactor, 0U);
}

// Test case 2: NCHW format with DCR mode
// Input shape: [1, 63, 2, 4], block_size=3
// Output shape: [1, 7, 6, 12]
TEST_F(DepthToSpaceTiling, test_nchw_dcr_float16_success_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "DepthToSpace",
        {
            {{{1, 63, 2, 4}, {1, 63, 2, 4}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
        },
        {
            {{{1, 7, 6, 12}, {1, 7, 6, 12}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(3)),
         gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("DCR")),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NCHW"))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE));
    EXPECT_EQ(tilingInfo.blockNum, 1U);
    const auto* data = GetTilingData<Transpose::TransposeSmallShape48DTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    // [1,63,2,4] NCHW + DCR 展开并简化后：inShape {3,3,14,4}, perm {2,0,3,1}
    // 注：DCR 展开后 block_size 在输入轴中天然出现两次，故前两维相同
    EXPECT_EQ(data->dim, 4);
    EXPECT_THAT(data->inputShape, ::testing::ElementsAreArray(std::vector<int32_t>{3, 3, 14, 4, 0, 0, 0, 0}));
    EXPECT_THAT(data->perm, ::testing::ElementsAreArray(std::vector<int8_t>{2, 0, 3, 1, -1, -1, -1, -1}));
    EXPECT_EQ(data->blkFactor, 504U);
    EXPECT_EQ(data->blkTailFactor, 0U);
}

// Test case 3: NHWC format with CRD mode
// Input shape: [2, 3, 4, 20], block_size=2
// Output shape: [2, 6, 8, 5]
TEST_F(DepthToSpaceTiling, test_nhwc_crd_float16_success_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "DepthToSpace",
        {
            {{{2, 3, 4, 20}, {2, 3, 4, 20}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {{{2, 6, 8, 5}, {2, 6, 8, 5}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("CRD")),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE));
    EXPECT_EQ(tilingInfo.blockNum, 1U);
    const auto* data = GetTilingData<Transpose::TransposeSmallShape48DTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    // [2,3,4,20] NHWC + CRD 展开并简化后：inShape {6,4,5,2,2}, perm {0,3,1,4,2}（5D）
    // 注：CRD 展开后 block_size 在输入轴中天然出现两次，故末两维相同
    EXPECT_EQ(data->dim, 5);
    EXPECT_THAT(data->inputShape, ::testing::ElementsAreArray(std::vector<int32_t>{6, 4, 5, 2, 2, 0, 0, 0}));
    EXPECT_THAT(data->perm, ::testing::ElementsAreArray(std::vector<int8_t>{0, 3, 1, 4, 2, -1, -1, -1}));
    EXPECT_EQ(data->blkFactor, 480U);
    EXPECT_EQ(data->blkTailFactor, 0U);
}

// Test case 4: NCHW format with CRD mode
// Input shape: [1, 20, 3, 4], block_size=2
// Output shape: [1, 5, 6, 8]
TEST_F(DepthToSpaceTiling, test_nchw_crd_float16_success_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "DepthToSpace",
        {
            {{{1, 20, 3, 4}, {1, 20, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
        },
        {
            {{{1, 5, 6, 8}, {1, 5, 6, 8}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("CRD")),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NCHW"))},
        &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(TK_SMALL_SHAPE));
    EXPECT_EQ(tilingInfo.blockNum, 1U);
    const auto* data = GetTilingData<Transpose::TransposeSmallShape48DTilingData>(tilingInfo);
    ASSERT_NE(data, nullptr);
    // [1,20,3,4] NCHW + CRD 展开并简化后：inShape {5,2,2,3,4}, perm {0,3,1,4,2}（5D）
    // 注：CRD 展开后 block_size 在输入轴中天然出现两次，故中间两维相同
    EXPECT_EQ(data->dim, 5);
    EXPECT_THAT(data->inputShape, ::testing::ElementsAreArray(std::vector<int32_t>{5, 2, 2, 3, 4, 0, 0, 0}));
    EXPECT_THAT(data->perm, ::testing::ElementsAreArray(std::vector<int8_t>{0, 3, 1, 4, 2, -1, -1, -1}));
    EXPECT_EQ(data->blkFactor, 240U);
    EXPECT_EQ(data->blkTailFactor, 0U);
}

// Test case 5: Failure case - invalid block_size (block_size < 2)
TEST_F(DepthToSpaceTiling, test_invalid_block_size_failed_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "DepthToSpace",
        {
            {{{1, 2, 2, 8}, {1, 2, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {{{1, 4, 4, 2}, {1, 4, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("DCR")),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))},
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Test case 6: Failure case - depth not divisible by block_size^2
// Input shape: [1, 2, 2, 7], block_size=2
// C=7, block_size^2=4, 7 % 4 != 0
TEST_F(DepthToSpaceTiling, test_invalid_depth_failed_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "DepthToSpace",
        {
            {{{1, 2, 2, 7}, {1, 2, 2, 7}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {{{1, 4, 4, 1}, {1, 4, 4, 1}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("DCR")),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))},
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Failure case: input format NHWC and data_format attr NCHW mismatch, expect failure
TEST_F(DepthToSpaceTiling, test_format_mismatch_failed_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "DepthToSpace",
        {
            {{{1, 2, 2, 8}, {1, 2, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {{{1, 4, 4, 2}, {1, 4, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("DCR")),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NCHW"))},
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Failure case: x and y have different storage formats (NHWC vs NCHW), expect failure
TEST_F(DepthToSpaceTiling, test_xy_format_mismatch_failed_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "DepthToSpace",
        {
            {{{1, 2, 2, 8}, {1, 2, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {{{1, 2, 4, 4}, {1, 2, 4, 4}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("DCR")),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))},
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Failure case: input shape is 3D (not 4D), expect failure
TEST_F(DepthToSpaceTiling, test_invalid_dims_failed_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "DepthToSpace",
        {
            {{{1, 2, 8}, {1, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {{{1, 4, 2}, {1, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("DCR")),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))},
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Failure case: data_format attr is invalid string (not NCHW or NHWC), expect failure
TEST_F(DepthToSpaceTiling, test_invalid_data_format_failed_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "DepthToSpace",
        {
            {{{1, 2, 2, 8}, {1, 2, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {{{1, 4, 4, 2}, {1, 4, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("DCR")),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("ND"))},
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Failure case: mode attr is invalid string (not DCR or CRD), expect failure
TEST_F(DepthToSpaceTiling, test_invalid_mode_failed_001)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    gert::TilingContextPara tilingContextPara(
        "DepthToSpace",
        {
            {{{1, 2, 2, 8}, {1, 2, 2, 8}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {
            {{{1, 4, 4, 2}, {1, 4, 4, 2}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
        },
        {gert::TilingContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("INVALID")),
         gert::TilingContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))},
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
