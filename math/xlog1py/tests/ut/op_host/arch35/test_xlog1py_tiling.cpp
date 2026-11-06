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
#include "../../../../op_host/arch35/xlog1py_tiling_arch35.h"
#include "../../../../op_kernel/arch35/xlog1py_tiling_struct.h"

namespace {
static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dimensions)
{
    gert::StorageShape storageShape;
    auto& originShape = storageShape.MutableOriginShape();
    auto& runtimeShape = storageShape.MutableStorageShape();
    originShape.SetDimNum(dimensions.size());
    runtimeShape.SetDimNum(dimensions.size());
    for (size_t i = 0; i < dimensions.size(); ++i) {
        originShape.SetDim(i, dimensions[i]);
        runtimeShape.SetDim(i, dimensions[i]);
    }
    return storageShape;
}

static gert::TilingContextPara MakeTilingContext(const std::vector<std::vector<int64_t>>& inputShapes,
                                                 const std::vector<int64_t>& outputShape, ge::DataType dataType,
                                                 optiling::Xlog1pyCompileInfo* compileInfo)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    for (const auto& shape : inputShapes) {
        inputs.emplace_back(MakeStorageShape(shape), dataType, ge::FORMAT_ND);
    }
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        {MakeStorageShape(outputShape), dataType, ge::FORMAT_ND},
    };
    return gert::TilingContextPara("Xlog1py", inputs, outputs, compileInfo);
}
} // namespace

class Xlog1pyTiling : public ::testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Xlog1pyTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "Xlog1pyTiling TearDown" << std::endl; }
};

TEST_F(Xlog1pyTiling, float32_broadcast_success)
{
    optiling::Xlog1pyCompileInfo compileInfo = {64, 262144};
    auto context = MakeTilingContext({{2, 1, 4}, {1, 3, 4}}, {2, 3, 4}, ge::DT_FLOAT, &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    ASSERT_GT(info.tilingDataSize, 0u);
    const auto* data = reinterpret_cast<const Xlog1pyTilingData<4>*>(info.tilingData.get());
    EXPECT_EQ(data->rank, 3);
    EXPECT_EQ(data->num_inputs, 2);
    EXPECT_EQ(data->num_outputs, 1);
}

TEST_F(Xlog1pyTiling, float16_same_shape_success)
{
    optiling::Xlog1pyCompileInfo compileInfo = {64, 262144};
    auto context = MakeTilingContext({{8, 1024}, {8, 1024}}, {8, 1024}, ge::DT_FLOAT16, &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    ASSERT_GT(info.tilingDataSize, 0u);
    const auto* data = reinterpret_cast<const Xlog1pyTilingData<4>*>(info.tilingData.get());
    EXPECT_EQ(data->rank, 2);
    EXPECT_EQ(data->num_inputs, 2);
}

TEST_F(Xlog1pyTiling, broadcast_incompatible_fail)
{
    optiling::Xlog1pyCompileInfo compileInfo = {64, 262144};
    auto context = MakeTilingContext({{2, 3}, {2, 4}}, {2, 4}, ge::DT_FLOAT, &compileInfo);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(Xlog1pyTiling, unsupported_dtype_fail)
{
    optiling::Xlog1pyCompileInfo compileInfo = {64, 262144};
    auto context = MakeTilingContext({{8, 1024}, {8, 1024}}, {8, 1024}, ge::DT_DOUBLE, &compileInfo);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(Xlog1pyTiling, dtype_mismatch_fail)
{
    optiling::Xlog1pyCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara context("Xlog1py",
                                    {{MakeStorageShape({8, 1024}), ge::DT_FLOAT, ge::FORMAT_ND},
                                     {MakeStorageShape({8, 1024}), ge::DT_FLOAT16, ge::FORMAT_ND}},
                                    {{MakeStorageShape({8, 1024}), ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(Xlog1pyTiling, output_dtype_mismatch_fail)
{
    optiling::Xlog1pyCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara context("Xlog1py",
                                    {{MakeStorageShape({8, 1024}), ge::DT_FLOAT, ge::FORMAT_ND},
                                     {MakeStorageShape({8, 1024}), ge::DT_FLOAT, ge::FORMAT_ND}},
                                    {{MakeStorageShape({8, 1024}), ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(Xlog1pyTiling, rank_over_8_fail)
{
    optiling::Xlog1pyCompileInfo compileInfo = {64, 262144};
    auto context = MakeTilingContext({{1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 2, 3, 4, 5, 6, 7, 8, 9}},
                                     {1, 2, 3, 4, 5, 6, 7, 8, 9}, ge::DT_FLOAT, &compileInfo);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

// rank 5~8 走 DoTilingAndSet<8>() 分支（Xlog1pyTilingData<8>，容量 8 = kMaxTilingRank）。
// 断言 tilingDataSize>=sizeof(Xlog1pyTilingData<8>) 用于证明 rank 5~8 未误用 <4> 结构导致越界。

TEST_F(Xlog1pyTiling, rank5_broadcast_success)
{
    optiling::Xlog1pyCompileInfo compileInfo = {64, 262144};
    auto context = MakeTilingContext({{1, 2, 3, 4, 5}, {2, 1, 3, 1, 5}}, {2, 2, 3, 4, 5}, ge::DT_FLOAT, &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    ASSERT_GE(info.tilingDataSize, sizeof(Xlog1pyTilingData<8>));
    const auto* data = reinterpret_cast<const Xlog1pyTilingData<8>*>(info.tilingData.get());
    EXPECT_EQ(data->rank, 5);
    EXPECT_EQ(data->num_inputs, 2);
    EXPECT_EQ(data->num_outputs, 1);
    EXPECT_EQ(data->max_bro_shape[0], 1);
    EXPECT_EQ(data->max_bro_shape[1], 1);
    EXPECT_EQ(data->max_bro_shape[2], 1);
    EXPECT_EQ(data->max_bro_shape[3], 2);
    EXPECT_EQ(data->max_bro_shape[4], 2);
    EXPECT_EQ(data->max_bro_shape[5], 3);
    EXPECT_EQ(data->max_bro_shape[6], 4);
    EXPECT_EQ(data->max_bro_shape[7], 5);
}

TEST_F(Xlog1pyTiling, rank6_same_shape_success)
{
    optiling::Xlog1pyCompileInfo compileInfo = {64, 262144};
    auto context = MakeTilingContext({{2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2}}, {2, 2, 2, 2, 2, 2}, ge::DT_FLOAT,
                                     &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    ASSERT_GE(info.tilingDataSize, sizeof(Xlog1pyTilingData<8>));
    const auto* data = reinterpret_cast<const Xlog1pyTilingData<8>*>(info.tilingData.get());
    EXPECT_EQ(data->rank, 6);
    EXPECT_EQ(data->num_inputs, 2);
    EXPECT_EQ(data->num_outputs, 1);
    EXPECT_EQ(data->max_bro_shape[0], 1);
    EXPECT_EQ(data->max_bro_shape[1], 1);
    for (int d = 2; d < 8; ++d)
        EXPECT_EQ(data->max_bro_shape[d], 2);
}

TEST_F(Xlog1pyTiling, rank7_same_shape_success)
{
    optiling::Xlog1pyCompileInfo compileInfo = {64, 262144};
    auto context = MakeTilingContext({{2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2}}, {2, 2, 2, 2, 2, 2, 2},
                                     ge::DT_FLOAT, &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    ASSERT_GE(info.tilingDataSize, sizeof(Xlog1pyTilingData<8>));
    const auto* data = reinterpret_cast<const Xlog1pyTilingData<8>*>(info.tilingData.get());
    EXPECT_EQ(data->rank, 7);
    EXPECT_EQ(data->num_inputs, 2);
    EXPECT_EQ(data->num_outputs, 1);
    EXPECT_EQ(data->max_bro_shape[0], 1);
    for (int d = 1; d < 8; ++d)
        EXPECT_EQ(data->max_bro_shape[d], 2);
}

TEST_F(Xlog1pyTiling, rank8_same_shape_success)
{
    optiling::Xlog1pyCompileInfo compileInfo = {64, 262144};
    auto context = MakeTilingContext({{2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2}}, {2, 2, 2, 2, 2, 2, 2, 2},
                                     ge::DT_FLOAT, &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    ASSERT_GE(info.tilingDataSize, sizeof(Xlog1pyTilingData<8>));
    const auto* data = reinterpret_cast<const Xlog1pyTilingData<8>*>(info.tilingData.get());
    EXPECT_EQ(data->rank, 8);
    EXPECT_EQ(data->num_inputs, 2);
    EXPECT_EQ(data->num_outputs, 1);
    for (int d = 0; d < 8; ++d)
        EXPECT_EQ(data->max_bro_shape[d], 2);
}
