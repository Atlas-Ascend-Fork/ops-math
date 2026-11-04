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
#include <cstring>
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

// arch22（Option A）独立公式复刻，用于机器核验不变式（与产品代码解耦，按 spec §5.2/§5.3 重写）
constexpr uint64_t kAivNum = 72;
constexpr uint64_t kSysWs = 16 * 1024 * 1024;
constexpr uint64_t kUbSize = 253952;
constexpr uint64_t kRowBlock = 32;
constexpr uint64_t kNarrowColThreshold = 16;
constexpr uint64_t kNarrowColRowsPerCore = 512;

uint64_t AlignUpU(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }
uint64_t SizeOfT(ge::DataType dt) { return dt == ge::DT_FLOAT16 ? 2 : 4; }
uint64_t UbBytesArch22(uint64_t tr, uint64_t ca, uint64_t s)
{
    return 2 * tr * ca * s + 4 * AlignUpU(tr * s, kRowBlock) + 5 * ca * s;
}
// I1：totalColAligned == AlignUp32(C*sizeofT)/sizeofT
uint64_t ExpectColAligned(uint64_t c, uint64_t s) { return AlignUpU(c * s, kRowBlock) / s; }
// I2：Option A user workspace（非空 AlignUp64）
uint64_t ExpectUserWs(uint64_t r, uint64_t c, uint64_t nb)
{
    uint64_t ws = 64 + 4 * r + 4 * nb * c + 4 * c + 4 * c;
    return (r > 0 && c > 0) ? AlignUpU(ws, 64) : ws;
}

struct TilingFields {
    uint64_t tileRow = 0;
    uint64_t totalRow = 0;
    uint64_t totalCol = 0;
    uint64_t totalColAligned = 0;
};

TilingFields ParseFields(const TilingInfo& info)
{
    TilingFields f;
    const uint8_t* b = info.tilingData.get();
    std::memcpy(&f.tileRow, b + 96, sizeof(uint64_t));
    std::memcpy(&f.totalRow, b + 112, sizeof(uint64_t));
    std::memcpy(&f.totalCol, b + 120, sizeof(uint64_t));
    std::memcpy(&f.totalColAligned, b + 128, sizeof(uint64_t));
    return f;
}
} // namespace

class SinkhornTilingArch22 : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SinkhornTilingArch22 SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SinkhornTilingArch22 TearDown" << std::endl; }
};

TEST_F(SinkhornTilingArch22, sinkhorn_tiling_parse)
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
    // Option A：tileRow=2696（arch35 为 2680，去 +64 padding/scratch 后增大）
    string expectTilingData = "1 48 96 1 48 96 0 0 0 0 0 0 2696 5392 48 2 8 953267991 ";
    std::vector<size_t> expectWorkspaces = {16777536};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(SinkhornTilingArch22, sinkhorn_tiling_one_by_one_workspace_is_cacheline_aligned)
{
    // Option A + Y-A：FP16 直调 sizeOfDataType=2 ⇒ totalColAligned=16、tileRow=2994；
    // FP32/BF16 sizeOfDataType=4 ⇒ totalColAligned=8、tileRow=2696。三者 user workspace 同为 128（FP32 计）。
    const std::vector<std::tuple<ge::DataType, uint64_t, std::string>> cases = {
        {ge::DT_FLOAT, 0, "1 1 1 1 1 1 0 0 0 0 0 0 2696 2696 1 1 8 953267991 "},
        {ge::DT_FLOAT16, 1, "1 1 1 1 1 1 0 0 0 0 0 0 2994 2994 1 1 16 953267991 "},
        {ge::DT_BF16, 27, "1 1 1 1 1 1 0 0 0 0 0 0 2696 2696 1 1 8 953267991 "}};
    for (const auto& [dtype, expectedKey, expectedTilingData] : cases) {
        optiling::SinkhornCompileInfo compileInfo = {72, 16 * 1024 * 1024, 253952};
        gert::TilingContextPara tilingContextPara(
            "Sinkhorn", {{{{1, 1}, {1, 1}}, dtype, ge::FORMAT_ND}}, {{{{1, 1}, {1, 1}}, dtype, ge::FORMAT_ND}},
            {gert::TilingContextPara::OpAttr("tol", Ops::Math::AnyValue::CreateFrom<float>(0.0001))}, &compileInfo);
        ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedKey, expectedTilingData,
                        {16 * 1024 * 1024 + 128});
    }
}

TEST_F(SinkhornTilingArch22, sinkhorn_tiling_empty_row)
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
    // Option A 空张量：numBlocks=1 ⇒ user ws = 64 + 0 + 4*1*2 + 4*2 + 4*2 = 88（arch35 grouped 为 72）
    std::vector<size_t> expectWorkspaces = {16777304};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(SinkhornTilingArch22, sinkhorn_tiling_all_float_tol_values_accepted)
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
        // Option A (8,2)：user ws = AlignUp64(64+32+8+16)=128
        ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, std::vector<size_t>{16 * 1024 * 1024 + 128});
    }
}

TEST_F(SinkhornTilingArch22, sinkhorn_tiling_default_tol_accepted)
{
    optiling::SinkhornCompileInfo compileInfo = {72, 16 * 1024 * 1024, 253952};
    gert::TilingContextPara tilingContextPara("Sinkhorn", {{{{8, 2}, {8, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{{{8, 2}, {8, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}}, {}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, std::vector<size_t>{16 * 1024 * 1024 + 128});
}

TEST_F(SinkhornTilingArch22, sinkhorn_tiling_zero_aiv_rejected)
{
    optiling::SinkhornCompileInfo compileInfo = {0, 16 * 1024 * 1024, 253952};
    gert::TilingContextPara tilingContextPara(
        "Sinkhorn", {{{{8, 2}, {8, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{8, 2}, {8, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("tol", Ops::Math::AnyValue::CreateFrom<float>(0.0001))}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(SinkhornTilingArch22, sinkhorn_tiling_invalid_shape_rejected)
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

TEST_F(SinkhornTilingArch22, sinkhorn_tiling_ge_direct_col_4097_accepted)
{
    optiling::SinkhornCompileInfo compileInfo = {72, 16 * 1024 * 1024, 253952};
    gert::TilingContextPara tilingContextPara(
        "Sinkhorn", {{{{2, 4097}, {2, 4097}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{2, 4097}, {2, 4097}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("tol", Ops::Math::AnyValue::CreateFrom<float>(0.0001))}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, std::vector<size_t>{16826496});
}

TEST_F(SinkhornTilingArch22, sinkhorn_tiling_row_above_10000_accepted)
{
    optiling::SinkhornCompileInfo compileInfo = {72, 16 * 1024 * 1024, 253952};
    gert::TilingContextPara tilingContextPara(
        "Sinkhorn", {{{{10001, 2}, {10001, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{10001, 2}, {10001, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("tol", Ops::Math::AnyValue::CreateFrom<float>(0.0001))}, &compileInfo);
    // Option A (10001,2) NB=20：user ws = AlignUp64(64+40004+160+16)=40256（arch35 grouped 为 125120）
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, std::vector<size_t>{16817472});
}

TEST_F(SinkhornTilingArch22, sinkhorn_tiling_insufficient_ub_rejected)
{
    optiling::SinkhornCompileInfo compileInfo = {72, 16 * 1024 * 1024, 1024};
    gert::TilingContextPara tilingContextPara(
        "Sinkhorn", {{{{8, 4096}, {8, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{8, 4096}, {8, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("tol", Ops::Math::AnyValue::CreateFrom<float>(0.0001))}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(SinkhornTilingArch22, sinkhorn_tiling_zero_tile_row_rejected)
{
    // Option A 无 +64 padding/scratch，fixedBytes 较小；ubSize=400 时 (8,8) 仍无法容纳 1 行 ⇒ tileRow==0 拒绝
    optiling::SinkhornCompileInfo compileInfo = {72, 16 * 1024 * 1024, 400};
    gert::TilingContextPara tilingContextPara(
        "Sinkhorn", {{{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("tol", Ops::Math::AnyValue::CreateFrom<float>(0.0001))}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}

// I1 机器核验：totalColAligned == AlignUp32(totalCol*sizeof(T))/sizeof(T)，FP16(Y-A) 与 FP32/BF16 分化
TEST_F(SinkhornTilingArch22, sinkhorn_tiling_colaligned_invariant)
{
    const std::vector<uint64_t> cols = {1, 2, 7, 8, 9, 15, 16, 17, 4096, 4168};
    const std::vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    for (const auto dt : dtypes) {
        const uint64_t s = SizeOfT(dt);
        for (const uint64_t c : cols) {
            optiling::SinkhornCompileInfo compileInfo = {kAivNum, kSysWs, kUbSize};
            auto shape = MakeStorageShape({8, static_cast<int64_t>(c)});
            gert::TilingContextPara para(
                "Sinkhorn", {{shape, dt, ge::FORMAT_ND}}, {{shape, dt, ge::FORMAT_ND}},
                {gert::TilingContextPara::OpAttr("tol", Ops::Math::AnyValue::CreateFrom<float>(0.0001f))},
                &compileInfo);
            TilingInfo info;
            ASSERT_TRUE(ExecuteTiling(para, info)) << "dtype=" << dt << " col=" << c;
            auto f = ParseFields(info);
            EXPECT_EQ(f.totalColAligned, ExpectColAligned(c, s)) << "dtype=" << dt << " col=" << c;
        }
    }
}

// I2 机器核验：user workspace == sysWs + Option A 公式；blockNum == 期望核数
TEST_F(SinkhornTilingArch22, sinkhorn_tiling_workspace_formula_invariant)
{
    const std::vector<std::pair<uint64_t, uint64_t>> shapes = {{1, 1},     {8, 8},    {48, 2},
                                                               {64, 4096}, {2, 4168}, {10001, 2}};
    for (const auto& [r, c] : shapes) {
        optiling::SinkhornCompileInfo compileInfo = {kAivNum, kSysWs, kUbSize};
        auto shape = MakeStorageShape({static_cast<int64_t>(r), static_cast<int64_t>(c)});
        gert::TilingContextPara para(
            "Sinkhorn", {{shape, ge::DT_FLOAT, ge::FORMAT_ND}}, {{shape, ge::DT_FLOAT, ge::FORMAT_ND}},
            {gert::TilingContextPara::OpAttr("tol", Ops::Math::AnyValue::CreateFrom<float>(0.0001f))}, &compileInfo);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(para, info)) << "shape=(" << r << "," << c << ")";
        auto f = ParseFields(info);
        // 期望核数：min(ceil(R/tileRow), aivNum)，窄列再按 512 行/核覆盖
        uint64_t ubNum = (r + f.tileRow - 1) / f.tileRow;
        uint64_t nb = std::min(ubNum, kAivNum);
        if (c < kNarrowColThreshold) {
            nb = std::min((r + kNarrowColRowsPerCore - 1) / kNarrowColRowsPerCore, kAivNum);
        }
        EXPECT_EQ(info.blockNum, nb) << "shape=(" << r << "," << c << ")";
        ASSERT_EQ(info.workspaceSizes.size(), 1u);
        EXPECT_EQ(info.workspaceSizes[0], kSysWs + ExpectUserWs(r, c, nb)) << "shape=(" << r << "," << c << ")";
    }
}

// I3 机器核验：floor(ubSize*0.85) >= CalculateUbBytesArch22(tileRow, totalColAligned, sizeofT)，且 exact-fit
TEST_F(SinkhornTilingArch22, sinkhorn_tiling_ub_fit_invariant)
{
    const std::vector<std::pair<uint64_t, uint64_t>> shapes = {{1, 1}, {8, 8}, {64, 4096}, {2, 4168}, {10001, 2}};
    const std::vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    const uint64_t usable = static_cast<uint64_t>(kUbSize * 0.85f);
    for (const auto dt : dtypes) {
        const uint64_t s = SizeOfT(dt);
        for (const auto& [r, c] : shapes) {
            optiling::SinkhornCompileInfo compileInfo = {kAivNum, kSysWs, kUbSize};
            auto shape = MakeStorageShape({static_cast<int64_t>(r), static_cast<int64_t>(c)});
            gert::TilingContextPara para(
                "Sinkhorn", {{shape, dt, ge::FORMAT_ND}}, {{shape, dt, ge::FORMAT_ND}},
                {gert::TilingContextPara::OpAttr("tol", Ops::Math::AnyValue::CreateFrom<float>(0.0001f))},
                &compileInfo);
            TilingInfo info;
            ASSERT_TRUE(ExecuteTiling(para, info)) << "dtype=" << dt << " shape=(" << r << "," << c << ")";
            auto f = ParseFields(info);
            EXPECT_LE(UbBytesArch22(f.tileRow, f.totalColAligned, s), usable)
                << "dtype=" << dt << " shape=(" << r << "," << c << ") tileRow=" << f.tileRow;
            // exact-fit：再加一行必然超出
            EXPECT_GT(UbBytesArch22(f.tileRow + 1, f.totalColAligned, s), usable)
                << "dtype=" << dt << " shape=(" << r << "," << c << ")";
        }
    }
}

TEST_F(SinkhornTilingArch22, sinkhorn_tiling_data_shared_abi_size)
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

TEST_F(SinkhornTilingArch22, sinkhorn_tiling_data_capacity_boundary)
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
            ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0, std::vector<size_t>{16 * 1024 * 1024 + 128});
        }
    }
}
