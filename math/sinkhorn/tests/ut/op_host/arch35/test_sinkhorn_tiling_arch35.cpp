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
#include <limits>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "math/sinkhorn/op_host/sinkhorn_tiling.h"

using namespace std;
using namespace ge;

namespace {
gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dimensions)
{
    gert::StorageShape shape;
    auto& originShape = shape.MutableOriginShape();
    auto& storageShape = shape.MutableStorageShape();
    originShape.SetDimNum(dimensions.size());
    storageShape.SetDimNum(dimensions.size());
    for (size_t i = 0; i < dimensions.size(); ++i) {
        originShape.SetDim(i, dimensions[i]);
        storageShape.SetDim(i, dimensions[i]);
    }
    return shape;
}
} // namespace

class SinkhornTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SinkhornTilingArch35 SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SinkhornTilingArch35 TearDown" << std::endl; }
};

TEST_F(SinkhornTilingArch35, sinkhorn_tiling_parse)
{
    optiling::SinkhornCompileInfo compileInfo = {72, 16 * 1024 * 1024, 253952};
    gert::TilingContextPara tilingContextPara(
        "Sinkhorn",
        {
            {{{48, 2}, {48, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{48, 2}, {48, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("tol", Ops::Math::AnyValue::CreateFrom<float>(0.0001)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "1 48 96 1 48 96 0 0 0 0 0 0 2680 5360 48 2 8 953267991 ";
    std::vector<size_t> expectWorkspaces = {16777920};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(SinkhornTilingArch35, sinkhorn_tiling_one_by_one_workspace_is_cacheline_aligned)
{
    const std::vector<std::tuple<ge::DataType, uint64_t>> cases = {
        {ge::DT_FLOAT, 0}, {ge::DT_FLOAT16, 1}, {ge::DT_BF16, 27}};
    const std::string expectedTilingData = "1 1 1 1 1 1 0 0 0 0 0 0 2680 2680 1 1 8 953267991 ";
    for (const auto& [dtype, expectedKey] : cases) {
        optiling::SinkhornCompileInfo compileInfo = {72, 16 * 1024 * 1024, 253952};
        gert::TilingContextPara tilingContextPara(
            "Sinkhorn", {{{{1, 1}, {1, 1}}, dtype, ge::FORMAT_ND}}, {{{{1, 1}, {1, 1}}, dtype, ge::FORMAT_ND}},
            {gert::TilingContextPara::OpAttr("tol", Ops::Math::AnyValue::CreateFrom<float>(0.0001))}, &compileInfo);
        ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedKey, expectedTilingData,
                        {16 * 1024 * 1024 + 128});
    }
}

TEST_F(SinkhornTilingArch35, sinkhorn_tiling_empty_row)
{
    optiling::SinkhornCompileInfo compileInfo = {72, 16 * 1024 * 1024, 253952};
    gert::TilingContextPara tilingContextPara(
        "Sinkhorn",
        {
            {{{0, 2}, {0, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{0, 2}, {0, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("tol", Ops::Math::AnyValue::CreateFrom<float>(0.0001)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 2 0 953267991 ";
    std::vector<size_t> expectWorkspaces = {16777288};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(SinkhornTilingArch35, sinkhorn_tiling_all_float_tol_values_accepted)
{
    const std::vector<float> tolValues = {0.0f,
                                          -0.0f,
                                          -0.0001f,
                                          std::numeric_limits<float>::infinity(),
                                          -std::numeric_limits<float>::infinity(),
                                          std::numeric_limits<float>::quiet_NaN()};
    for (const float tol : tolValues) {
        optiling::SinkhornCompileInfo compileInfo = {72, 16 * 1024 * 1024, 253952};
        gert::TilingContextPara tilingContextPara(
            "Sinkhorn",
            {
                {{{8, 2}, {8, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            },
            {
                {{{8, 2}, {8, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            },
            {
                gert::TilingContextPara::OpAttr("tol", Ops::Math::AnyValue::CreateFrom<float>(tol)),
            },
            &compileInfo);
        ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, std::vector<size_t>{16 * 1024 * 1024 + 192});
    }
}

TEST_F(SinkhornTilingArch35, sinkhorn_tiling_default_tol_accepted)
{
    optiling::SinkhornCompileInfo compileInfo = {72, 16 * 1024 * 1024, 253952};
    gert::TilingContextPara tilingContextPara("Sinkhorn", {{{{8, 2}, {8, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{{{8, 2}, {8, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}}, {}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, std::vector<size_t>{16 * 1024 * 1024 + 192});
}

TEST_F(SinkhornTilingArch35, sinkhorn_tiling_zero_aiv_rejected)
{
    optiling::SinkhornCompileInfo compileInfo = {0, 16 * 1024 * 1024, 253952};
    gert::TilingContextPara tilingContextPara(
        "Sinkhorn", {{{{8, 2}, {8, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{8, 2}, {8, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("tol", Ops::Math::AnyValue::CreateFrom<float>(0.0001))}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(SinkhornTilingArch35, sinkhorn_tiling_invalid_shape_rejected)
{
    const std::vector<std::vector<int64_t>> invalidShapes = {
        {}, {-1, 2}, {8}, {2, 3, 4}, {2, 4169}, {1048576, 4096}, {std::numeric_limits<int64_t>::max(), 4096}};
    for (const auto& shape : invalidShapes) {
        optiling::SinkhornCompileInfo compileInfo = {72, 16 * 1024 * 1024, 253952};
        std::vector<gert::TilingContextPara::TensorDescription> inputs = {
            {MakeStorageShape(shape), ge::DT_FLOAT, ge::FORMAT_ND}};
        std::vector<gert::TilingContextPara::TensorDescription> outputs = {
            {MakeStorageShape(shape), ge::DT_FLOAT, ge::FORMAT_ND}};
        gert::TilingContextPara tilingContextPara(
            "Sinkhorn", inputs, outputs,
            {gert::TilingContextPara::OpAttr("tol", Ops::Math::AnyValue::CreateFrom<float>(0.0001))}, &compileInfo);
        ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
    }
}

TEST_F(SinkhornTilingArch35, sinkhorn_tiling_ge_direct_col_4097_accepted)
{
    optiling::SinkhornCompileInfo compileInfo = {72, 16 * 1024 * 1024, 253952};
    gert::TilingContextPara tilingContextPara(
        "Sinkhorn", {{{{2, 4097}, {2, 4097}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 4097}, {2, 4097}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("tol", Ops::Math::AnyValue::CreateFrom<float>(0.0001))}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, std::vector<size_t>{16842880});
}

TEST_F(SinkhornTilingArch35, sinkhorn_tiling_row_above_10000_accepted)
{
    optiling::SinkhornCompileInfo compileInfo = {72, 16 * 1024 * 1024, 253952};
    gert::TilingContextPara tilingContextPara(
        "Sinkhorn", {{{{10001, 2}, {10001, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{10001, 2}, {10001, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("tol", Ops::Math::AnyValue::CreateFrom<float>(0.0001))}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, std::vector<size_t>{16902336});
}

TEST_F(SinkhornTilingArch35, sinkhorn_tiling_insufficient_ub_rejected)
{
    optiling::SinkhornCompileInfo compileInfo = {72, 16 * 1024 * 1024, 1024};
    gert::TilingContextPara tilingContextPara(
        "Sinkhorn", {{{{8, 4096}, {8, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{8, 4096}, {8, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("tol", Ops::Math::AnyValue::CreateFrom<float>(0.0001))}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(SinkhornTilingArch35, sinkhorn_tiling_zero_tile_row_rejected)
{
    optiling::SinkhornCompileInfo compileInfo = {72, 16 * 1024 * 1024, 2048};
    gert::TilingContextPara tilingContextPara(
        "Sinkhorn", {{{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("tol", Ops::Math::AnyValue::CreateFrom<float>(0.0001))}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(SinkhornTilingArch35, sinkhorn_tiling_data_shared_abi_size)
{
    // arch22 与 arch35 共享同一份序列化 ABI（op_host/sinkhorn_tiling.h）：
    // 17 个 uint64_t + 1 个 float：字段裸加为 17*8 + 4 = 140 字节，但结构体对齐要求为
    // 8（最大成员对齐），故尾部补齐到 8 的倍数 => 实际序列化尺寸 144 字节。
    // 实测值 144（此前 README/spec/CONTRACT_DECISIONS 写的 140 是未计 tail padding 的
    // 字段裸加，属文档错误，已同批更正）。
    // 两侧 tiling 各自写入、kernel 各自读取，字段顺序/类型/偏移必须逐字一致，
    // 否则 host 与 device 对同一块 tiling buffer 的解释会静默错位。
    // GetDataSize() 是运行期方法（tilingdata_base.h:86 `size_t GetDataSize() const;`），
    // 无法 static_assert，故以 UT 钉死；README 与 spec.yaml 声明的 140 由此断言背书。
    optiling::SinkhornTilingData tilingData;
    EXPECT_EQ(tilingData.GetDataSize(), 144U);
}

TEST_F(SinkhornTilingArch35, sinkhorn_tiling_data_capacity_boundary)
{
    optiling::SinkhornCompileInfo compileInfo = {72, 16 * 1024 * 1024, 253952};
    for (const auto capacity : {143U, 144U, 145U}) {
        gert::TilingContextPara para(
            "Sinkhorn", {{{{8, 2}, {8, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
            {{{{8, 2}, {8, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
            {gert::TilingContextPara::OpAttr("tol", Ops::Math::AnyValue::CreateFrom<float>(0.0001F))}, &compileInfo, 72,
            253952, capacity);
        if (capacity < 144U) {
            ExecuteTestCase(para, ge::GRAPH_FAILED, 0, "", {});
        } else {
            ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0, std::vector<size_t>{16 * 1024 * 1024 + 192});
        }
    }
}
