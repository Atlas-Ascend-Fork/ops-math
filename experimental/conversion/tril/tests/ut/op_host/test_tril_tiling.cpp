/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <cstring>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_kernel/tril_tiling_data.h"
#include "tiling_case_executor.h"

namespace {
constexpr uint64_t UB_SIZE = 196608;
constexpr uint64_t CORE_NUM = 32;
struct TrilTestCompileInfo {};

bool RunTiling(const std::vector<int64_t>& dims, int64_t diagonal, TilingInfo& info, ge::DataType dtype = ge::DT_FLOAT)
{
    gert::StorageShape shape;
    for (auto dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    TrilTestCompileInfo compileInfo;
    gert::TilingContextPara para("Tril", {{shape, dtype, ge::FORMAT_ND}}, {{shape, dtype, ge::FORMAT_ND}},
                                 {{"diagonal", Ops::Math::AnyValue::CreateFrom<int64_t>(diagonal)}}, &compileInfo,
                                 CORE_NUM, UB_SIZE);
    return ExecuteTiling(para, info) && info.tilingDataSize == sizeof(TrilTilingData);
}

const TrilTilingData& Data(const TilingInfo& info)
{
    return *reinterpret_cast<const TrilTilingData*>(info.tilingData.get());
}

bool SameTiling(const TilingInfo& lhs, const TilingInfo& rhs)
{
    return lhs.tilingKey == rhs.tilingKey && lhs.blockNum == rhs.blockNum && lhs.workspaceSizes == rhs.workspaceSizes &&
           lhs.tilingDataSize == rhs.tilingDataSize &&
           std::memcmp(lhs.tilingData.get(), rhs.tilingData.get(), lhs.tilingDataSize) == 0;
}
} // namespace

TEST(TrilTiling, RepeatedBatchInput)
{
    TilingInfo first;
    TilingInfo second;
    ASSERT_TRUE(RunTiling({2, 3, 4}, 0, first));
    ASSERT_TRUE(RunTiling({2, 3, 4}, 0, second));
    EXPECT_EQ(Data(second).matrixNum, 2);
    EXPECT_EQ(Data(second).lastTileLength, 24U);
    EXPECT_TRUE(SameTiling(first, second));
}

TEST(TrilTiling, BatchThenTwoDimensionalInput)
{
    TilingInfo batch;
    TilingInfo matrix;
    ASSERT_TRUE(RunTiling({2, 3, 4}, 0, batch));
    ASSERT_TRUE(RunTiling({5, 6}, 0, matrix));
    EXPECT_EQ(Data(matrix).matrixNum, 1);
    EXPECT_EQ(Data(matrix).totalLengthAligned, 32U);
    EXPECT_EQ(Data(matrix).lastTileLength, 32U);
}

TEST(TrilTiling, EmptyInputBetweenCalls)
{
    TilingInfo first;
    TilingInfo empty;
    TilingInfo second;
    ASSERT_TRUE(RunTiling({2, 3, 4}, 0, first));
    ASSERT_TRUE(RunTiling({0, 5}, 0, empty));
    TrilTilingData zeros{};
    EXPECT_EQ(std::memcmp(empty.tilingData.get(), &zeros, sizeof(zeros)), 0);
    ASSERT_TRUE(RunTiling({2, 3, 4}, 0, second));
    EXPECT_TRUE(SameTiling(first, second));
}

TEST(TrilTiling, FourBuffersThenTwoBuffers)
{
    TilingInfo reference;
    TilingInfo fourBuffers;
    TilingInfo twoBuffers;
    ASSERT_TRUE(RunTiling({200, 200}, -200, reference));
    ASSERT_TRUE(RunTiling({200, 200}, 200, fourBuffers));
    ASSERT_TRUE(RunTiling({200, 200}, -200, twoBuffers));
    EXPECT_EQ(twoBuffers.tilingKey, 2);
    EXPECT_EQ(Data(twoBuffers).loopCnt, 2);
    EXPECT_TRUE(SameTiling(reference, twoBuffers));
}

TEST(TrilTiling, FastPathThenRegularPath)
{
    TilingInfo reference;
    TilingInfo fast;
    TilingInfo regular;
    ASSERT_TRUE(RunTiling({200, 200}, 0, reference));
    ASSERT_TRUE(RunTiling({192, 64}, 0, fast));
    ASSERT_EQ(fast.tilingKey, 4);
    ASSERT_TRUE(RunTiling({200, 200}, 0, regular));
    EXPECT_EQ(regular.tilingKey, 1);
    EXPECT_EQ(Data(regular).loopCnt, 2);
    EXPECT_TRUE(SameTiling(reference, regular));
}

TEST(TrilTiling, FourBufferTileFitsUb)
{
    TilingInfo info;
    ASSERT_TRUE(RunTiling({200, 200}, 200, info));
    ASSERT_EQ(info.tilingKey, 3);
    const auto& data = Data(info);
    EXPECT_EQ(data.loopCnt, 4);
    EXPECT_GT(data.fullTileLength, 0U);
    EXPECT_GT(data.lastTileLength, 0U);
    EXPECT_LE(data.lastTileLength, data.fullTileLength);
    EXPECT_LE(static_cast<uint64_t>(data.fullTileLength) * data.typeSize * 4, UB_SIZE);
    EXPECT_EQ(static_cast<uint64_t>(data.fullTileLength) * (data.loopCnt - 1) + data.lastTileLength,
              data.totalLengthAligned);
}

TEST(TrilTiling, DataTypeSwitch)
{
    TilingInfo half;
    TilingInfo single;
    TilingInfo halfAgain;
    ASSERT_TRUE(RunTiling({2, 3, 4}, 0, half, ge::DT_FLOAT16));
    ASSERT_TRUE(RunTiling({5, 6}, 0, single));
    ASSERT_TRUE(RunTiling({2, 3, 4}, 0, halfAgain, ge::DT_FLOAT16));
    EXPECT_EQ(Data(single).typeSize, 4U);
    EXPECT_TRUE(SameTiling(half, halfAgain));
}

TEST(TrilTiling, ConcurrentIndependentContexts)
{
    const std::vector<std::vector<int64_t>> shapes = {{2, 3, 4}, {5, 6}, {200, 200}, {192, 64}};
    const std::vector<int64_t> diagonals = {0, 0, 200, 0};
    std::vector<TilingInfo> expected(shapes.size());
    for (size_t i = 0; i < shapes.size(); ++i) {
        ASSERT_TRUE(RunTiling(shapes[i], diagonals[i], expected[i]));
    }
    std::atomic<bool> start{false};
    std::atomic<bool> success{true};
    std::vector<std::thread> threads;
    for (size_t i = 0; i < shapes.size(); ++i) {
        threads.emplace_back([&, i] {
            while (!start.load()) {
                std::this_thread::yield();
            }
            for (size_t repeat = 0; repeat < 50; ++repeat) {
                TilingInfo actual;
                if (!RunTiling(shapes[i], diagonals[i], actual) || !SameTiling(expected[i], actual)) {
                    success.store(false);
                    break;
                }
            }
        });
    }
    start.store(true);
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_TRUE(success.load());
}
