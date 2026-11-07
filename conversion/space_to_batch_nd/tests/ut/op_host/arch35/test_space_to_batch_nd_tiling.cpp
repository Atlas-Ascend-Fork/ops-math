/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/space_to_batch_nd/op_kernel/arch35/space_to_batch_nd_tiling_data.h"
#include <iostream>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class SpaceToBatchNDTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SpaceToBatchNDTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SpaceToBatchNDTilingTest TearDown" << std::endl; }
};

namespace {
SpaceToBatchNDCompileInfo compileInfo;
}

// ---------- 1D spatial with padding ----------

TEST_F(SpaceToBatchNDTilingTest, Spatial1D_WithPadding)
{
    // in [279, 465, 385], bs=[3], pad=[[1,2]]
    // padded spatial = 468, out spatial = 156, batch = 279*3 = 837
    // out [837, 156, 385]
    std::vector<int32_t> blockShapeValues = {3};
    std::vector<int32_t> paddingsValues = {1, 2};
    gert::TilingContextPara para("SpaceToBatchND",
                                 {
                                     {{{279, 465, 385}, {279, 465, 385}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                     {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                 },
                                 {
                                     {{{837, 156, 385}, {837, 156, 385}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.workspaceSizes, std::vector<int64_t>{0});

    auto* td = reinterpret_cast<SpaceToBatchNDTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->numSpatialDims, 1);
    EXPECT_EQ(td->rank, 3);
    EXPECT_EQ(td->outShape[0], 837);
    EXPECT_EQ(td->outShape[1], 156);
    EXPECT_EQ(td->outShape[2], 385);
    EXPECT_EQ(td->blockShape[0], 3);
    EXPECT_EQ(td->padTop[0], 1);
    EXPECT_EQ(td->padBottom[0], 2);
    EXPECT_GE(td->totalCount, 1);
    EXPECT_GE(td->perCoreCount, 1);
    EXPECT_GE(td->ubFactor, 1);
    EXPECT_LE(td->ubAxis, 2);
    EXPECT_NE(info.tilingKey, static_cast<uint64_t>(-1));
}

// ---------- 2D spatial no padding ----------

TEST_F(SpaceToBatchNDTilingTest, Spatial2D_NoPadding)
{
    // in [2, 4, 4, 3], bs=[2,2], pad=0
    // batch = 2*4 = 8, out_spatial = [2, 2]
    // out [8, 2, 2, 3]
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> paddingsValues = {0, 0, 0, 0};
    gert::TilingContextPara para("SpaceToBatchND",
                                 {
                                     {{{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                     {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                 },
                                 {
                                     {{{8, 2, 2, 3}, {8, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    auto* td = reinterpret_cast<SpaceToBatchNDTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->numSpatialDims, 2);
    EXPECT_EQ(td->rank, 4);
    EXPECT_EQ(td->outShape[0], 8);
    EXPECT_EQ(td->outShape[1], 2);
    EXPECT_EQ(td->outShape[2], 2);
    EXPECT_EQ(td->outShape[3], 3);
    EXPECT_EQ(td->blockShape[0], 2);
    EXPECT_EQ(td->blockShape[1], 2);
}

// ---------- same shape with double dtype ----------

TEST_F(SpaceToBatchNDTilingTest, DoubleDtypeShouldPass)
{
    // in [2, 4, 4, 3], bs=[2,2], pad=0; only the element dtype differs from Spatial2D_NoPadding
    // out [8, 2, 2, 3]
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> paddingsValues = {0, 0, 0, 0};
    gert::TilingContextPara para("SpaceToBatchND",
                                 {
                                     {{{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                     {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                     {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                 },
                                 {
                                     {{{8, 2, 2, 3}, {8, 2, 2, 3}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                 },
                                 &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    auto* td = reinterpret_cast<SpaceToBatchNDTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->rank, 4);
}

// ---------- same shape with complex32 dtype ----------

TEST_F(SpaceToBatchNDTilingTest, Complex32DtypeShouldPass)
{
    // in [2, 4, 4, 3], bs=[2,2], pad=0; only the element dtype differs from Spatial2D_NoPadding
    // out [8, 2, 2, 3]
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> paddingsValues = {0, 0, 0, 0};
    gert::TilingContextPara para("SpaceToBatchND",
                                 {
                                     {{{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::DT_COMPLEX32, ge::FORMAT_ND},
                                     {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                     {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                 },
                                 {
                                     {{{8, 2, 2, 3}, {8, 2, 2, 3}}, ge::DT_COMPLEX32, ge::FORMAT_ND},
                                 },
                                 &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    auto* td = reinterpret_cast<SpaceToBatchNDTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->rank, 4);
}

// ---------- same shape with complex64 dtype ----------

TEST_F(SpaceToBatchNDTilingTest, Complex64DtypeShouldPass)
{
    // in [2, 4, 4, 3], bs=[2,2], pad=0; only the element dtype differs from Spatial2D_NoPadding
    // out [8, 2, 2, 3]
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> paddingsValues = {0, 0, 0, 0};
    gert::TilingContextPara para("SpaceToBatchND",
                                 {
                                     {{{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                     {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                     {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                 },
                                 {
                                     {{{8, 2, 2, 3}, {8, 2, 2, 3}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                 },
                                 &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    auto* td = reinterpret_cast<SpaceToBatchNDTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->rank, 4);
}

// ---------- 2D spatial with padding ----------

TEST_F(SpaceToBatchNDTilingTest, Spatial2D_WithPadding)
{
    // in [1, 4, 4, 3], bs=[2,2], pad=[[1,1],[1,1]]
    // padded = [6,6], out = [4, 3, 3, 3]
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> paddingsValues = {1, 1, 1, 1};
    gert::TilingContextPara para("SpaceToBatchND",
                                 {
                                     {{{1, 4, 4, 3}, {1, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                     {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                 },
                                 {
                                     {{{4, 3, 3, 3}, {4, 3, 3, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    auto* td = reinterpret_cast<SpaceToBatchNDTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->outShape[0], 4);
    EXPECT_EQ(td->outShape[1], 3);
    EXPECT_EQ(td->outShape[2], 3);
    EXPECT_EQ(td->outShape[3], 3);
    EXPECT_EQ(td->padTop[0], 1);
    EXPECT_EQ(td->padBottom[0], 1);
    EXPECT_EQ(td->padTop[1], 1);
    EXPECT_EQ(td->padBottom[1], 1);
}

// ---------- 4D spatial large batch ----------

TEST_F(SpaceToBatchNDTilingTest, Spatial4D_LargeBatch)
{
    // in [24, 28, 44, 42, 40], bs=[2,2,2,2], pad=0
    // batch = 24*16 = 384, out_spatial = [14, 22, 21, 20]
    // out [384, 14, 22, 21, 20, 1]
    std::vector<int32_t> blockShapeValues = {2, 2, 2, 2};
    std::vector<int32_t> paddingsValues = {0, 0, 0, 0, 0, 0, 0, 0};
    gert::TilingContextPara para(
        "SpaceToBatchND",
        {
            {{{24, 28, 44, 42, 40}, {24, 28, 44, 42, 40}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{4, 2}, {4, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
        },
        {
            {{{384, 14, 22, 21, 20, 1}, {384, 14, 22, 21, 20, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    auto* td = reinterpret_cast<SpaceToBatchNDTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->numSpatialDims, 4);
    EXPECT_EQ(td->rank, 6);
    EXPECT_EQ(td->outShape[0], 384);
    EXPECT_EQ(td->outShape[1], 14);
    EXPECT_EQ(td->outShape[2], 22);
    EXPECT_EQ(td->outShape[3], 21);
    EXPECT_EQ(td->outShape[4], 20);
    EXPECT_EQ(td->outShape[5], 1);
    EXPECT_GE(td->totalCount, 1);
}

// ---------- Negative padding should fail ----------

TEST_F(SpaceToBatchNDTilingTest, NegativePaddingShouldFail)
{
    std::vector<int32_t> blockShapeValues = {2};
    std::vector<int32_t> paddingsValues = {-1, 0};
    gert::TilingContextPara para("SpaceToBatchND",
                                 {
                                     {{{2, 4, 4}, {2, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                     {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                 },
                                 {
                                     {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);

    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// ---------- x with unsupported float8 dtype should fail ----------

TEST_F(SpaceToBatchNDTilingTest, Float8DtypeShouldFail)
{
    // float8 不在 OpDef 支持列表，tiling 应拒绝
    // same shape as Spatial2D_NoPadding but x/y are declared DT_FLOAT8_E4M3FN
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> paddingsValues = {0, 0, 0, 0};
    gert::TilingContextPara para("SpaceToBatchND",
                                 {
                                     {{{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                     {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                     {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                 },
                                 {
                                     {{{8, 2, 2, 3}, {8, 2, 2, 3}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
                                 },
                                 &compileInfo);

    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// ---------- x with non-ND storage format should fail ----------

TEST_F(SpaceToBatchNDTilingTest, XFractalNzFormatShouldFail)
{
    // same as Spatial2D_NoPadding but x is declared FRACTAL_NZ; only FORMAT_ND is supported
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> paddingsValues = {0, 0, 0, 0};
    gert::TilingContextPara para("SpaceToBatchND",
                                 {
                                     {{{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ},
                                     {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                     {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                 },
                                 {
                                     {{{8, 2, 2, 3}, {8, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);

    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// ---------- block_shape with non-ND storage format should fail ----------

TEST_F(SpaceToBatchNDTilingTest, BlockShapeFractalNzFormatShouldFail)
{
    // same as Spatial2D_NoPadding but block_shape is declared FRACTAL_NZ; only FORMAT_ND is supported
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> paddingsValues = {0, 0, 0, 0};
    gert::TilingContextPara para("SpaceToBatchND",
                                 {
                                     {{{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ, true, blockShapeValues.data()},
                                     {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                 },
                                 {
                                     {{{8, 2, 2, 3}, {8, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);

    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// ---------- output dtype mismatch should fail ----------

TEST_F(SpaceToBatchNDTilingTest, OutputDtypeMismatchShouldFail)
{
    // same as Spatial2D_NoPadding but y declares DT_UINT16 while x is DT_FLOAT
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> paddingsValues = {0, 0, 0, 0};
    gert::TilingContextPara para("SpaceToBatchND",
                                 {
                                     {{{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                     {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                 },
                                 {
                                     {{{8, 2, 2, 3}, {8, 2, 2, 3}}, ge::DT_UINT16, ge::FORMAT_ND},
                                 },
                                 &compileInfo);

    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// ---------- output with non-ND storage format should fail ----------

TEST_F(SpaceToBatchNDTilingTest, OutputFractalNzFormatShouldFail)
{
    // same as Spatial2D_NoPadding but y is declared FRACTAL_NZ; only FORMAT_ND is supported
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> paddingsValues = {0, 0, 0, 0};
    gert::TilingContextPara para("SpaceToBatchND",
                                 {
                                     {{{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                     {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                 },
                                 {
                                     {{{8, 2, 2, 3}, {8, 2, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ},
                                 },
                                 &compileInfo);

    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// ---------- Paddings shape [M,1] should fail ----------

TEST_F(SpaceToBatchNDTilingTest, PaddingsShapeOneColumnShouldFail)
{
    // x [2,3,3], bs=[2], paddings declared [1,1] with a single value {1}
    // paddings must be [M,2], so [1,1] must be rejected instead of silently reading padBottom as 0
    std::vector<int32_t> blockShapeValues = {2};
    std::vector<int32_t> paddingsValues = {1};
    gert::TilingContextPara para("SpaceToBatchND",
                                 {
                                     {{{2, 3, 3}, {2, 3, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                     {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                 },
                                 {
                                     {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);

    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// ---------- Paddings first dim != numSpatialDims should fail ----------

TEST_F(SpaceToBatchNDTilingTest, PaddingsFirstDimMismatchShouldFail)
{
    // x [2,3,3], bs=[2], paddings declared [2,2] but first dim 2 != M=1
    std::vector<int32_t> blockShapeValues = {2};
    std::vector<int32_t> paddingsValues = {0, 0, 0, 0};
    gert::TilingContextPara para("SpaceToBatchND",
                                 {
                                     {{{2, 3, 3}, {2, 3, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                     {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                 },
                                 {
                                     {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);

    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// ---------- Block_shape length equals x rank should fail ----------

TEST_F(SpaceToBatchNDTilingTest, BlockShapeLengthEqualsRankShouldFail)
{
    // x [2,3,3] (rank=3), block_shape length M=3 equals rank N=3
    // constraint requires M < N
    std::vector<int32_t> blockShapeValues = {2, 2, 2};
    std::vector<int32_t> paddingsValues = {0, 0, 0, 0, 0, 0};
    gert::TilingContextPara para("SpaceToBatchND",
                                 {
                                     {{{2, 3, 3}, {2, 3, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                     {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                 },
                                 {
                                     {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);

    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// ---------- Block_shape length greater than x rank should fail ----------

TEST_F(SpaceToBatchNDTilingTest, BlockShapeLengthGreaterThanRankShouldFail)
{
    // x [2,3,3] (rank=3), block_shape length M=4 greater than rank N=3
    std::vector<int32_t> blockShapeValues = {2, 2, 2, 2};
    std::vector<int32_t> paddingsValues = {0, 0, 0, 0, 0, 0, 0, 0};
    gert::TilingContextPara para("SpaceToBatchND",
                                 {
                                     {{{2, 3, 3}, {2, 3, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                     {{{4, 2}, {4, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                 },
                                 {
                                     {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);

    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// ---------- Block_shape length 257 must be rejected before uint8_t truncation ----------

TEST_F(SpaceToBatchNDTilingTest, BlockShapeLength257ShouldFail)
{
    // x [2,3,3] (rank=3), block_shape length M=257. Truncated to uint8_t it becomes 1 and would
    // silently pass the 1..7 range check while ignoring the remaining 256 block_shape entries.
    // paddings is declared [1,2] (the truncated M) so the truncation bypass is actually exercised.
    std::vector<int32_t> blockShapeValues(257, 1);
    std::vector<int32_t> paddingsValues = {0, 0};
    gert::TilingContextPara para("SpaceToBatchND",
                                 {
                                     {{{2, 3, 3}, {2, 3, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{257}, {257}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                     {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                 },
                                 {
                                     {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);

    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// ---------- Block_shape length 258 must be rejected before uint8_t truncation ----------

TEST_F(SpaceToBatchNDTilingTest, BlockShapeLength258ShouldFail)
{
    // x [2,3,3] (rank=3), block_shape length M=258. Truncated to uint8_t it becomes 2 and would
    // silently pass the 1..7 range check while ignoring the remaining 256 block_shape entries.
    // paddings is declared [2,2] (the truncated M) so the truncation bypass is actually exercised.
    std::vector<int32_t> blockShapeValues(258, 1);
    std::vector<int32_t> paddingsValues = {0, 0, 0, 0};
    gert::TilingContextPara para("SpaceToBatchND",
                                 {
                                     {{{2, 3, 3}, {2, 3, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{258}, {258}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                     {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                 },
                                 {
                                     {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);

    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// ---------- Rank 9 input should fail (input rank upper bound is 8) ----------

TEST_F(SpaceToBatchNDTilingTest, Rank9ShouldFail)
{
    std::vector<int32_t> blockShapeValues = {1, 1, 1, 1, 1, 1, 1};
    std::vector<int32_t> paddingsValues = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    gert::TilingContextPara para(
        "SpaceToBatchND",
        {
            {{{1, 2, 2, 2, 2, 2, 2, 2, 1}, {1, 2, 2, 2, 2, 2, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{7}, {7}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{7, 2}, {7, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);

    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// ---------- Rank 8 input should pass with merged rank 9 ----------

TEST_F(SpaceToBatchNDTilingTest, Rank8ShouldPass)
{
    // in [1, 2, 2, 2, 2, 2, 2, 2], bs=[1,1,1,1,1,1,1], pad=0
    // max input rank 8, merged rank = numSpatialDims + 2 = 9
    // out [1, 2, 2, 2, 2, 2, 2, 2, 1]
    std::vector<int32_t> blockShapeValues = {1, 1, 1, 1, 1, 1, 1};
    std::vector<int32_t> paddingsValues = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    gert::TilingContextPara para(
        "SpaceToBatchND",
        {
            {{{1, 2, 2, 2, 2, 2, 2, 2}, {1, 2, 2, 2, 2, 2, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{7}, {7}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{7, 2}, {7, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
        },
        {
            {{{1, 2, 2, 2, 2, 2, 2, 2, 1}, {1, 2, 2, 2, 2, 2, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    auto* td = reinterpret_cast<SpaceToBatchNDTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->rank, 9);
    EXPECT_EQ(td->numSpatialDims, 7);
}

// ---------- Empty tensor: batch dim is 0 ----------

TEST_F(SpaceToBatchNDTilingTest, EmptyBatchShouldPass)
{
    // in [0, 4, 4], bs=[2,2], pad=0
    // batch = 0*4 = 0, out_spatial = [2, 2], merged out [0, 2, 2, 1]
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> paddingsValues = {0, 0, 0, 0};
    gert::TilingContextPara para("SpaceToBatchND",
                                 {
                                     {{{0, 4, 4}, {0, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                     {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                 },
                                 {
                                     {{{0, 2, 2, 1}, {0, 2, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    auto* td = reinterpret_cast<SpaceToBatchNDTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->rank, 4);
    EXPECT_EQ(td->totalCount, 0);
    EXPECT_EQ(info.blockNum, 1);
}

// ---------- Empty tensor: spatial dim is 0 ----------

TEST_F(SpaceToBatchNDTilingTest, EmptySpatialShouldPass)
{
    // in [2, 0, 4], bs=[2,2], pad=0
    // batch = 2*4 = 8, out_spatial = [0, 2], merged out [8, 0, 2, 1]
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> paddingsValues = {0, 0, 0, 0};
    gert::TilingContextPara para("SpaceToBatchND",
                                 {
                                     {{{2, 0, 4}, {2, 0, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                     {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                 },
                                 {
                                     {{{8, 0, 2, 1}, {8, 0, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    auto* td = reinterpret_cast<SpaceToBatchNDTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->rank, 4);
    EXPECT_EQ(td->totalCount, 0);
    EXPECT_EQ(info.blockNum, 1);
}

// ---------- Empty tensor: inner (tail) dim is 0 ----------

TEST_F(SpaceToBatchNDTilingTest, EmptyInnerShouldPass)
{
    // in [2, 4, 4, 0], bs=[2,2], pad=0
    // batch = 2*4 = 8, out_spatial = [2, 2], innerSize = 0, merged out [8, 2, 2, 0]
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> paddingsValues = {0, 0, 0, 0};
    gert::TilingContextPara para("SpaceToBatchND",
                                 {
                                     {{{2, 4, 4, 0}, {2, 4, 4, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                     {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                 },
                                 {
                                     {{{8, 2, 2, 0}, {8, 2, 2, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    auto* td = reinterpret_cast<SpaceToBatchNDTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->rank, 4);
    EXPECT_EQ(td->totalCount, 0);
    EXPECT_EQ(info.blockNum, 1);
}

// ---------- Fallback when DoCacheLineTiling returns startAxis=0 (rank 3) ----------

TEST_F(SpaceToBatchNDTilingTest, StartAxisZeroFallbackRank3)
{
    // in [1048576, 1, 1], bs=[1], pad=0
    // out [1048576, 1, 1], cache-line threshold reached at ax=0 so startAxis=0,
    // candidate loop never runs and the fallback must pick lastAxis=2.
    std::vector<int32_t> blockShapeValues = {1};
    std::vector<int32_t> paddingsValues = {0, 0};
    gert::TilingContextPara para("SpaceToBatchND",
                                 {
                                     {{{1048576, 1, 1}, {1048576, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                     {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                 },
                                 {
                                     {{{1048576, 1, 1}, {1048576, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    auto* td = reinterpret_cast<SpaceToBatchNDTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->rank, 3);
    EXPECT_EQ(td->ubAxis, 2);
    EXPECT_EQ(td->totalCount, 1048576);
    EXPECT_GE(td->ubFactor, 1);
    EXPECT_GE(info.blockNum, 1);
}

// ---------- Fallback when DoCacheLineTiling returns startAxis=0 (rank 4) ----------

TEST_F(SpaceToBatchNDTilingTest, StartAxisZeroFallbackRank4)
{
    // in [1048576, 1, 1, 1], bs=[1,1], pad=0
    // out [1048576, 1, 1, 1], cache-line threshold reached at ax=0 so startAxis=0,
    // candidate loop never runs and the fallback must pick lastAxis=3.
    std::vector<int32_t> blockShapeValues = {1, 1};
    std::vector<int32_t> paddingsValues = {0, 0, 0, 0};
    gert::TilingContextPara para("SpaceToBatchND",
                                 {
                                     {{{1048576, 1, 1, 1}, {1048576, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                     {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                 },
                                 {
                                     {{{1048576, 1, 1, 1}, {1048576, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    auto* td = reinterpret_cast<SpaceToBatchNDTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->rank, 4);
    EXPECT_EQ(td->ubAxis, 3);
    EXPECT_EQ(td->totalCount, 1048576);
}

// ---------- padded spatial addition overflow should fail ----------

TEST_F(SpaceToBatchNDTilingTest, PaddedDimAddOverflowShouldFail)
{
    // spatial = INT64_MAX - 1, pad = [1, 1] -> spatial + padTop + padBottom overflows int64
    std::vector<int32_t> blockShapeValues = {2};
    std::vector<int32_t> paddingsValues = {1, 1};
    gert::TilingContextPara para(
        "SpaceToBatchND",
        {
            {{{1, std::numeric_limits<int64_t>::max() - 1, 4}, {1, std::numeric_limits<int64_t>::max() - 1, 4}},
             ge::DT_FLOAT,
             ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);

    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// ---------- batch * block_shape overflow should fail ----------

TEST_F(SpaceToBatchNDTilingTest, BatchProductOverflowShouldFail)
{
    // batch = INT64_MAX, block_shape = [2] -> batchProd = batch * blockShape overflows int64
    std::vector<int32_t> blockShapeValues = {2};
    std::vector<int32_t> paddingsValues = {0, 0};
    gert::TilingContextPara para(
        "SpaceToBatchND",
        {
            {{{std::numeric_limits<int64_t>::max(), 4}, {std::numeric_limits<int64_t>::max(), 4}},
             ge::DT_FLOAT,
             ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);

    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// ---------- inner size product overflow should fail ----------

TEST_F(SpaceToBatchNDTilingTest, InnerSizeOverflowShouldFail)
{
    // inner dims are [INT64_MAX, 2], their product overflows int64
    std::vector<int32_t> blockShapeValues = {1};
    std::vector<int32_t> paddingsValues = {0, 0};
    gert::TilingContextPara para(
        "SpaceToBatchND",
        {
            {{{1, 4, std::numeric_limits<int64_t>::max(), 2}, {1, 4, std::numeric_limits<int64_t>::max(), 2}},
             ge::DT_FLOAT,
             ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);

    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// ---------- total count exceeding int64 should fail at SetTilingData ----------

TEST_F(SpaceToBatchNDTilingTest, TotalCountExceedsInt64ShouldFail)
{
    // out shape [2^62, 4096, 1]: the UB limit caps factor at 2048, so the largest candidate has
    // CeilDiv(4096, 2048) = 2 and totalCount = 2^62 * 2 = 2^63, still representable as uint64 but
    // not as int64, so SetTilingData must reject it.
    std::vector<int32_t> blockShapeValues = {1};
    std::vector<int32_t> paddingsValues = {0, 0};
    gert::TilingContextPara para(
        "SpaceToBatchND",
        {
            {{{4611686018427387904LL, 4096, 1}, {4611686018427387904LL, 4096, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
            {{{1, 2}, {1, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);

    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

// ---------- Empty input spatial dim but non-empty output after padding ----------

TEST_F(SpaceToBatchNDTilingTest, EmptySpatialWithPaddingShouldTile)
{
    // in [2, 0, 4], bs=[2,2], pad=[[1,1],[1,1]]
    // padded spatial = [2, 6], out_spatial = [1, 3], batch = 2*2*2 = 8
    // out [8, 1, 3, 1] has 24 elements, so tiling must NOT take the empty no-op path
    std::vector<int32_t> blockShapeValues = {2, 2};
    std::vector<int32_t> paddingsValues = {1, 1, 1, 1};
    gert::TilingContextPara para("SpaceToBatchND",
                                 {
                                     {{{2, 0, 4}, {2, 0, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, blockShapeValues.data()},
                                     {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND, true, paddingsValues.data()},
                                 },
                                 {
                                     {{{8, 1, 3, 1}, {8, 1, 3, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));

    auto* td = reinterpret_cast<SpaceToBatchNDTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->rank, 4);
    EXPECT_EQ(td->numSpatialDims, 2);
    EXPECT_EQ(td->outShape[0], 8);
    EXPECT_EQ(td->outShape[1], 1);
    EXPECT_EQ(td->outShape[2], 3);
    EXPECT_EQ(td->outShape[3], 1);
    EXPECT_GT(td->totalCount, 0);
    EXPECT_GE(info.blockNum, 1);
}
