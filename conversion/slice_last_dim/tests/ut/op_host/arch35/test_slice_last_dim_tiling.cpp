/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "conversion/slice_last_dim/op_kernel/arch35/slice_last_dim_tiling_data.h"

namespace {
// Documented upper bound (2^40) of the NdDma loopSrcStride field. Independently defined here rather
// than imported from the tiling implementation, so the boundary tests below stay non-co-sourced.
constexpr int64_t kMaxStride = static_cast<int64_t>(1) << 40;

// Runs the real tiling callback and decodes SliceLastDimTilingData on success.
bool RunTiling(const gert::TilingContextPara& para, TilingInfo& tilingInfo, SliceLastDimTilingData& td)
{
    if (!ExecuteTiling(para, tilingInfo)) {
        return false;
    }
    if (tilingInfo.tilingData == nullptr || tilingInfo.tilingDataSize < sizeof(SliceLastDimTilingData)) {
        return false;
    }
    std::memcpy(&td, tilingInfo.tilingData.get(), sizeof(td));
    return true;
}
} // namespace

class SliceLastDimTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SliceLastDimTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SliceLastDimTiling TearDown" << std::endl; }
};

TEST_F(SliceLastDimTilingTest, slice_last_dim_tiling_basic)
{
    SliceLastDimCompileInfo compileInfo;
    gert::StorageShape inShape = {{4, 16}, {4, 16}};
    gert::StorageShape outShape = {{4, 8}, {4, 8}};
    gert::TilingContextPara tilingContextPara("SliceLastDim", {{inShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {{outShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {
                                                  {"start", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
                                                  {"end", Ops::Math::AnyValue::CreateFrom<int64_t>(8)},
                                                  {"stride", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_GT(tilingInfo.blockNum, 0);
}

TEST_F(SliceLastDimTilingTest, slice_last_dim_tiling_with_stride)
{
    SliceLastDimCompileInfo compileInfo;
    gert::StorageShape inShape = {{2, 3, 100}, {2, 3, 100}};
    gert::StorageShape outShape = {{2, 3, 50}, {2, 3, 50}};
    gert::TilingContextPara tilingContextPara("SliceLastDim", {{inShape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {{outShape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              {
                                                  {"start", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
                                                  {"end", Ops::Math::AnyValue::CreateFrom<int64_t>(100)},
                                                  {"stride", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    ASSERT_GT(tilingInfo.blockNum, 0);
}

// A zero non-last dim (outerSize == 0) must return a legal minimal tiling (totalCount == 0 /
// blockDim == 1) instead of a 0/0 division in the tiling search.
TEST_F(SliceLastDimTilingTest, slice_last_dim_tiling_degenerate_outer_zero_first)
{
    SliceLastDimCompileInfo compileInfo;
    gert::StorageShape inShape = {{0, 6}, {0, 6}};
    gert::StorageShape outShape = {{0, 6}, {0, 6}};
    gert::TilingContextPara tilingContextPara("SliceLastDim", {{inShape, ge::DT_INT8, ge::FORMAT_ND}},
                                              {{outShape, ge::DT_INT8, ge::FORMAT_ND}},
                                              {
                                                  {"start", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
                                                  {"end", Ops::Math::AnyValue::CreateFrom<int64_t>(6)},
                                                  {"stride", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    SliceLastDimTilingData td;
    ASSERT_TRUE(RunTiling(tilingContextPara, tilingInfo, td));
    ASSERT_EQ(tilingInfo.blockNum, 1);
    ASSERT_EQ(td.outerSize, 0);
    ASSERT_EQ(td.lastDimIn, 6);
    ASSERT_EQ(td.lastDimOut, 6);
    ASSERT_EQ(td.totalCount, 0);
    ASSERT_EQ(td.ubFactor, 1);
}

// A zero dim in the middle yields the same degenerate-output contract.
TEST_F(SliceLastDimTilingTest, slice_last_dim_tiling_degenerate_outer_zero_mid)
{
    SliceLastDimCompileInfo compileInfo;
    gert::StorageShape inShape = {{2, 0, 5}, {2, 0, 5}};
    gert::StorageShape outShape = {{2, 0, 5}, {2, 0, 5}};
    gert::TilingContextPara tilingContextPara("SliceLastDim", {{inShape, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{outShape, ge::DT_INT32, ge::FORMAT_ND}},
                                              {
                                                  {"start", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
                                                  {"end", Ops::Math::AnyValue::CreateFrom<int64_t>(5)},
                                                  {"stride", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    SliceLastDimTilingData td;
    ASSERT_TRUE(RunTiling(tilingContextPara, tilingInfo, td));
    ASSERT_EQ(tilingInfo.blockNum, 1);
    ASSERT_EQ(td.outerSize, 0);
    ASSERT_EQ(td.lastDimIn, 5);
    ASSERT_EQ(td.lastDimOut, 5);
    ASSERT_EQ(td.totalCount, 0);
    ASSERT_EQ(td.ubFactor, 1);
}

// An empty slice (end <= start, lastDimOut == 0) returns the same minimal legal tiling; the early
// exit also makes tiling O(1) regardless of outerSize.
TEST_F(SliceLastDimTilingTest, slice_last_dim_tiling_empty_slice_early_exit)
{
    SliceLastDimCompileInfo compileInfo;
    gert::StorageShape inShape = {{4, 16}, {4, 16}};
    gert::StorageShape outShape = {{4, 0}, {4, 0}};
    gert::TilingContextPara tilingContextPara("SliceLastDim", {{inShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {{outShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {
                                                  {"start", Ops::Math::AnyValue::CreateFrom<int64_t>(8)},
                                                  {"end", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
                                                  {"stride", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    SliceLastDimTilingData td;
    ASSERT_TRUE(RunTiling(tilingContextPara, tilingInfo, td));
    ASSERT_EQ(tilingInfo.blockNum, 1);
    ASSERT_EQ(td.outerSize, 4);
    ASSERT_EQ(td.lastDimOut, 0);
    ASSERT_EQ(td.totalCount, 0);
    ASSERT_EQ(td.ubFactor, 1);
}

// Opposite-sign int64 extremes must clamp into [0, lastDimIn] (start -> lastDimIn, end -> 0) and
// produce the empty-output tiling without overflow.
TEST_F(SliceLastDimTilingTest, slice_last_dim_tiling_extreme_clamp)
{
    SliceLastDimCompileInfo compileInfo;
    gert::StorageShape inShape = {{2, 10}, {2, 10}};
    gert::StorageShape outShape = {{2, 0}, {2, 0}};
    gert::TilingContextPara tilingContextPara("SliceLastDim", {{inShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {{outShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {
                                                  {"start", Ops::Math::AnyValue::CreateFrom<int64_t>(INT64_MAX)},
                                                  {"end", Ops::Math::AnyValue::CreateFrom<int64_t>(INT64_MIN)},
                                                  {"stride", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    SliceLastDimTilingData td;
    ASSERT_TRUE(RunTiling(tilingContextPara, tilingInfo, td));
    ASSERT_EQ(td.start, 10);
    ASSERT_EQ(td.lastDimIn, 10);
    ASSERT_EQ(td.lastDimOut, 0);
    ASSERT_EQ(td.totalCount, 0);
    ASSERT_EQ(tilingInfo.blockNum, 1);
}

// Negative indices within [-lastDimIn, 0) normalize by adding lastDimIn.
TEST_F(SliceLastDimTilingTest, slice_last_dim_tiling_negative_indices_in_range)
{
    SliceLastDimCompileInfo compileInfo;
    gert::StorageShape inShape = {{4, 16}, {4, 16}};
    gert::StorageShape outShape = {{4, 2}, {4, 2}};
    gert::TilingContextPara tilingContextPara("SliceLastDim", {{inShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {{outShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {
                                                  {"start", Ops::Math::AnyValue::CreateFrom<int64_t>(-4)},
                                                  {"end", Ops::Math::AnyValue::CreateFrom<int64_t>(-2)},
                                                  {"stride", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    SliceLastDimTilingData td;
    ASSERT_TRUE(RunTiling(tilingContextPara, tilingInfo, td));
    ASSERT_EQ(td.start, 12);
    ASSERT_EQ(td.lastDimOut, 2);
    ASSERT_GT(td.totalCount, 0);
}

// Out-of-range negatives clamp to [0, lastDimIn] on both ends.
TEST_F(SliceLastDimTilingTest, slice_last_dim_tiling_negative_indices_out_of_range)
{
    SliceLastDimCompileInfo compileInfo;
    gert::StorageShape inShape = {{4, 16}, {4, 16}};
    gert::StorageShape outShape = {{4, 8}, {4, 8}};
    gert::TilingContextPara tilingContextPara("SliceLastDim", {{inShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {{outShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {
                                                  {"start", Ops::Math::AnyValue::CreateFrom<int64_t>(-100)},
                                                  {"end", Ops::Math::AnyValue::CreateFrom<int64_t>(100)},
                                                  {"stride", Ops::Math::AnyValue::CreateFrom<int64_t>(2)},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    SliceLastDimTilingData td;
    ASSERT_TRUE(RunTiling(tilingContextPara, tilingInfo, td));
    ASSERT_EQ(td.start, 0);
    ASSERT_EQ(td.lastDimOut, 8);
    ASSERT_EQ(td.stride, 2);
    ASSERT_GT(td.totalCount, 0);
}

// Stride upper bound — a non-degenerate output with stride >= 2^40 must be rejected (NdDma
// loopSrcStride documented range [0, 2^40)).
TEST_F(SliceLastDimTilingTest, slice_last_dim_tiling_stride_upper_bound_reject)
{
    SliceLastDimCompileInfo compileInfo;
    gert::StorageShape inShape = {{2, 10}, {2, 10}};
    gert::StorageShape outShape = {{2, 1}, {2, 1}};
    gert::TilingContextPara tilingContextPara("SliceLastDim", {{inShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {{outShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {
                                                  {"start", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
                                                  {"end", Ops::Math::AnyValue::CreateFrom<int64_t>(10)},
                                                  {"stride", Ops::Math::AnyValue::CreateFrom<int64_t>(kMaxStride)},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Boundary — stride == 2^40 - 1 (largest documented value) must stay allowed.
TEST_F(SliceLastDimTilingTest, slice_last_dim_tiling_stride_upper_bound_max_allowed)
{
    SliceLastDimCompileInfo compileInfo;
    gert::StorageShape inShape = {{2, 10}, {2, 10}};
    gert::StorageShape outShape = {{2, 1}, {2, 1}};
    gert::TilingContextPara tilingContextPara("SliceLastDim", {{inShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {{outShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {
                                                  {"start", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
                                                  {"end", Ops::Math::AnyValue::CreateFrom<int64_t>(10)},
                                                  {"stride", Ops::Math::AnyValue::CreateFrom<int64_t>(kMaxStride - 1)},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    SliceLastDimTilingData td;
    ASSERT_TRUE(RunTiling(tilingContextPara, tilingInfo, td));
    ASSERT_EQ(td.lastDimOut, 1);
    ASSERT_EQ(td.stride, kMaxStride - 1);
    ASSERT_GT(td.totalCount, 0);
}

// An empty output (end <= start) with a huge stride is legal: stride is not transferred for a
// degenerate output, so it must be accepted (not rejected by the stride upper-bound check).
TEST_F(SliceLastDimTilingTest, slice_last_dim_tiling_stride_oversize_empty_inert)
{
    SliceLastDimCompileInfo compileInfo;
    gert::StorageShape inShape = {{2, 10}, {2, 10}};
    gert::StorageShape outShape = {{2, 0}, {2, 0}};
    gert::TilingContextPara tilingContextPara("SliceLastDim", {{inShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {{outShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {
                                                  {"start", Ops::Math::AnyValue::CreateFrom<int64_t>(10)},
                                                  {"end", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
                                                  {"stride", Ops::Math::AnyValue::CreateFrom<int64_t>(kMaxStride)},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    SliceLastDimTilingData td;
    ASSERT_TRUE(RunTiling(tilingContextPara, tilingInfo, td));
    ASSERT_EQ(tilingInfo.blockNum, 1);
    ASSERT_EQ(td.lastDimOut, 0);
    ASSERT_EQ(td.totalCount, 0);
    ASSERT_EQ(td.stride, kMaxStride);
}

// Dtype whitelist — DT_BOOL must be rejected at the tiling entry.
TEST_F(SliceLastDimTilingTest, slice_last_dim_tiling_dtype_bool_reject)
{
    SliceLastDimCompileInfo compileInfo;
    gert::StorageShape inShape = {{4, 16}, {4, 16}};
    gert::StorageShape outShape = {{4, 8}, {4, 8}};
    gert::TilingContextPara tilingContextPara("SliceLastDim", {{inShape, ge::DT_BOOL, ge::FORMAT_ND}},
                                              {{outShape, ge::DT_BOOL, ge::FORMAT_ND}},
                                              {
                                                  {"start", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
                                                  {"end", Ops::Math::AnyValue::CreateFrom<int64_t>(8)},
                                                  {"stride", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Dtype whitelist — DT_COMPLEX128 must be rejected (a byte size > 0 is not enough).
TEST_F(SliceLastDimTilingTest, slice_last_dim_tiling_dtype_complex128_reject)
{
    SliceLastDimCompileInfo compileInfo;
    gert::StorageShape inShape = {{4, 16}, {4, 16}};
    gert::StorageShape outShape = {{4, 8}, {4, 8}};
    gert::TilingContextPara tilingContextPara("SliceLastDim", {{inShape, ge::DT_COMPLEX128, ge::FORMAT_ND}},
                                              {{outShape, ge::DT_COMPLEX128, ge::FORMAT_ND}},
                                              {
                                                  {"start", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
                                                  {"end", Ops::Math::AnyValue::CreateFrom<int64_t>(8)},
                                                  {"stride", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// Optional-stride contract — a missing stride attribute defaults to 1 and tiling succeeds.
TEST_F(SliceLastDimTilingTest, slice_last_dim_tiling_stride_attr_absent)
{
    SliceLastDimCompileInfo compileInfo;
    gert::StorageShape inShape = {{4, 16}, {4, 16}};
    gert::StorageShape outShape = {{4, 8}, {4, 8}};
    gert::TilingContextPara tilingContextPara("SliceLastDim", {{inShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {{outShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                              {
                                                  {"start", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
                                                  {"end", Ops::Math::AnyValue::CreateFrom<int64_t>(8)},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    SliceLastDimTilingData td;
    ASSERT_TRUE(RunTiling(tilingContextPara, tilingInfo, td));
    ASSERT_EQ(td.stride, 1);
    ASSERT_EQ(td.lastDimOut, 8);
    ASSERT_GT(tilingInfo.blockNum, 0);
}
