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
 * \file test_transpose_tiling.cpp
 * \brief broadcast_to tiling ut test
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <array>
#include <iostream>
#include <vector>
#include <cstring>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/transpose_tiling_arch35.h"

using namespace ge;

class TransposeB4Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TransposeB4Tiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TransposeB4Tiling TearDown" << std::endl; }
};

// 场景：B4 4D 输入 [36,204,26,30]，perm={1,0,3,2}，交换最后两轴，输出 [204,36,30,26]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_4d_perm_1032)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[4] = {1, 0, 3, 2};
    gert::TilingContextPara::TensorDescription x({{36, 204, 26, 30}, {36, 204, 26, 30}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{204, 36, 30, 26}, {204, 36, 30, 26}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    std::vector<size_t> expectWorkspaces = {0};
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    EXPECT_EQ(tilingInfo.blockNum, 63);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Transpose::TransposeB4TilingData));
    const auto* tilingData = reinterpret_cast<const Transpose::TransposeB4TilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(tilingData->gmShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 1, 1, 1, 1, 1, 12, 68}));
    EXPECT_THAT(tilingData->gmInStride,
                ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 477360, 2340}));
    EXPECT_THAT(tilingData->gmOutStride,
                ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 2340, 84240}));
    EXPECT_THAT(tilingData->ubShape, ::testing::ElementsAreArray(std::vector<int64_t>{3, 3, 26, 30}));
    EXPECT_THAT(tilingData->ubInStride, ::testing::ElementsAreArray(std::vector<int64_t>{159120, 780, 30, 1}));
    EXPECT_THAT(tilingData->ubOutStride, ::testing::ElementsAreArray(std::vector<int64_t>{780, 28080, 1, 26}));
    EXPECT_THAT(tilingData->moveUbStride, ::testing::ElementsAreArray(std::vector<int64_t>{28080, 780, 26, 1}));
    EXPECT_THAT(tilingData->ubPerm, ::testing::ElementsAreArray(std::vector<int64_t>{1, 0, 3, 2}));
    EXPECT_THAT(tilingData->ubShapeId, ::testing::ElementsAreArray(std::vector<int64_t>{0, 1, 2, 3}));
    EXPECT_EQ(tilingData->cutAxisIdx01, 0);
    EXPECT_EQ(tilingData->cutAxisIdx02, 1);
    EXPECT_EQ(tilingData->cutAxisTail01, 3);
    EXPECT_EQ(tilingData->cutAxisTail02, 3);
    EXPECT_EQ(tilingData->outCutId, 0);
    EXPECT_EQ(tilingData->inCutId, 1);
    EXPECT_EQ(tilingData->outerAxisNums, 2);
    EXPECT_EQ(tilingData->innerAxisNums, 4);
    EXPECT_EQ(tilingData->dimsPerCore, 13);
    EXPECT_EQ(tilingData->dimsPerCoreTail, 10);
    EXPECT_EQ(tilingData->ubSize, 25376);
    EXPECT_EQ(tilingData->permSize, 4);
    EXPECT_THAT(tilingData->perm, ::testing::ElementsAreArray(std::vector<int64_t>{1, 0, 3, 2, 0, 0, 0, 0}));
}

// 场景：B4 4D 极小 shape 输入 [2,2,4,8]，perm={1,0,3,2}，交换最后两轴，输出 [2,2,8,4]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_4d_perm_1032_tiny)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[4] = {1, 0, 3, 2};
    gert::TilingContextPara::TensorDescription x({{2, 2, 4, 8}, {2, 2, 4, 8}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{2, 2, 8, 4}, {2, 2, 8, 4}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    std::vector<size_t> expectWorkspaces = {0};
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    EXPECT_EQ(tilingInfo.blockNum, 1);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Transpose::TransposeB4TilingData));
    const auto* tilingData = reinterpret_cast<const Transpose::TransposeB4TilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(tilingData->gmShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 1, 1, 1, 1, 1, 1, 1}));
    EXPECT_THAT(tilingData->gmInStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 1}));
    EXPECT_THAT(tilingData->gmOutStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 2}));
    EXPECT_THAT(tilingData->ubShape, ::testing::ElementsAreArray(std::vector<int64_t>{2, 2, 4, 8}));
    EXPECT_THAT(tilingData->ubInStride, ::testing::ElementsAreArray(std::vector<int64_t>{64, 32, 8, 1}));
    EXPECT_THAT(tilingData->ubOutStride, ::testing::ElementsAreArray(std::vector<int64_t>{32, 64, 1, 4}));
    EXPECT_THAT(tilingData->moveUbStride, ::testing::ElementsAreArray(std::vector<int64_t>{64, 32, 4, 1}));
    EXPECT_THAT(tilingData->ubPerm, ::testing::ElementsAreArray(std::vector<int64_t>{1, 0, 3, 2}));
    EXPECT_THAT(tilingData->ubShapeId, ::testing::ElementsAreArray(std::vector<int64_t>{0, 1, 2, 3}));
    EXPECT_EQ(tilingData->cutAxisIdx01, -1);
    EXPECT_EQ(tilingData->cutAxisIdx02, -1);
    EXPECT_EQ(tilingData->cutAxisTail01, -1);
    EXPECT_EQ(tilingData->cutAxisTail02, -1);
    EXPECT_EQ(tilingData->outCutId, -1);
    EXPECT_EQ(tilingData->inCutId, -1);
    EXPECT_EQ(tilingData->outerAxisNums, 1);
    EXPECT_EQ(tilingData->innerAxisNums, 4);
    EXPECT_EQ(tilingData->dimsPerCore, 1);
    EXPECT_EQ(tilingData->dimsPerCoreTail, 1);
    EXPECT_EQ(tilingData->ubSize, 25376);
    EXPECT_EQ(tilingData->permSize, 4);
    EXPECT_THAT(tilingData->perm, ::testing::ElementsAreArray(std::vector<int64_t>{1, 0, 3, 2, 0, 0, 0, 0}));
}

// 场景：B4 4D 输入 [2,628,4,8]，perm={1,0,3,2}，批量维为 2 且中间轴较大的尾轴转置，输出 [628,2,8,4]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_4d_perm_1032_batch_large)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[4] = {1, 0, 3, 2};
    gert::TilingContextPara::TensorDescription x({{2, 628, 4, 8}, {2, 628, 4, 8}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{628, 2, 8, 4}, {628, 2, 8, 4}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    std::vector<size_t> expectWorkspaces = {0};
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    EXPECT_EQ(tilingInfo.blockNum, 53);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Transpose::TransposeB4TilingData));
    const auto* tilingData = reinterpret_cast<const Transpose::TransposeB4TilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(tilingData->gmShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 1, 1, 1, 1, 1, 1, 105}));
    EXPECT_THAT(tilingData->gmInStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 192}));
    EXPECT_THAT(tilingData->gmOutStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 384}));
    EXPECT_THAT(tilingData->ubShape, ::testing::ElementsAreArray(std::vector<int64_t>{2, 6, 4, 8}));
    EXPECT_THAT(tilingData->ubInStride, ::testing::ElementsAreArray(std::vector<int64_t>{20096, 32, 8, 1}));
    EXPECT_THAT(tilingData->ubOutStride, ::testing::ElementsAreArray(std::vector<int64_t>{32, 64, 1, 4}));
    EXPECT_THAT(tilingData->moveUbStride, ::testing::ElementsAreArray(std::vector<int64_t>{64, 32, 4, 1}));
    EXPECT_THAT(tilingData->ubPerm, ::testing::ElementsAreArray(std::vector<int64_t>{1, 0, 3, 2}));
    EXPECT_THAT(tilingData->ubShapeId, ::testing::ElementsAreArray(std::vector<int64_t>{0, 1, 2, 3}));
    EXPECT_EQ(tilingData->cutAxisIdx01, 1);
    EXPECT_EQ(tilingData->cutAxisIdx02, -1);
    EXPECT_EQ(tilingData->cutAxisTail01, 4);
    EXPECT_EQ(tilingData->cutAxisTail02, -1);
    EXPECT_EQ(tilingData->outCutId, 0);
    EXPECT_EQ(tilingData->inCutId, 0);
    EXPECT_EQ(tilingData->outerAxisNums, 1);
    EXPECT_EQ(tilingData->innerAxisNums, 4);
    EXPECT_EQ(tilingData->dimsPerCore, 2);
    EXPECT_EQ(tilingData->dimsPerCoreTail, 1);
    EXPECT_EQ(tilingData->ubSize, 25376);
    EXPECT_EQ(tilingData->permSize, 4);
    EXPECT_THAT(tilingData->perm, ::testing::ElementsAreArray(std::vector<int64_t>{1, 0, 3, 2, 0, 0, 0, 0}));
}

// 场景：B4 2D 输入 [2,628]，perm={1,0}，标准二维转置，输出 [628,2]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_2d_perm_10)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[4] = {1, 0};
    gert::TilingContextPara::TensorDescription x({{2, 628}, {2, 628}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{628, 2}, {628, 2}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    std::vector<size_t> expectWorkspaces = {0};
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    EXPECT_EQ(tilingInfo.blockNum, 1);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Transpose::TransposeB4TilingData));
    const auto* tilingData = reinterpret_cast<const Transpose::TransposeB4TilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(tilingData->gmShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 1, 1, 1, 1, 1, 1, 1}));
    EXPECT_THAT(tilingData->gmInStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 1}));
    EXPECT_THAT(tilingData->gmOutStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 2}));
    EXPECT_THAT(tilingData->ubShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 1, 2, 628}));
    EXPECT_THAT(tilingData->ubInStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 628, 1}));
    EXPECT_THAT(tilingData->ubOutStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 1, 2}));
    EXPECT_THAT(tilingData->moveUbStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 2, 1}));
    EXPECT_THAT(tilingData->ubPerm, ::testing::ElementsAreArray(std::vector<int64_t>{0, 1, 3, 2}));
    EXPECT_THAT(tilingData->ubShapeId, ::testing::ElementsAreArray(std::vector<int64_t>{-1, -1, 0, 1}));
    EXPECT_EQ(tilingData->cutAxisIdx01, -1);
    EXPECT_EQ(tilingData->cutAxisIdx02, -1);
    EXPECT_EQ(tilingData->cutAxisTail01, -1);
    EXPECT_EQ(tilingData->cutAxisTail02, -1);
    EXPECT_EQ(tilingData->outCutId, -1);
    EXPECT_EQ(tilingData->inCutId, -1);
    EXPECT_EQ(tilingData->outerAxisNums, 1);
    EXPECT_EQ(tilingData->innerAxisNums, 2);
    EXPECT_EQ(tilingData->dimsPerCore, 1);
    EXPECT_EQ(tilingData->dimsPerCoreTail, 1);
    EXPECT_EQ(tilingData->ubSize, 25376);
    EXPECT_EQ(tilingData->permSize, 2);
    EXPECT_THAT(tilingData->perm, ::testing::ElementsAreArray(std::vector<int64_t>{1, 0, 0, 0, 0, 0, 0, 0}));
}

// 场景：B4 6D 输入 [10,20,6,2,8,10]，perm={1,0,3,2,5,4}，三组相邻轴两两互换，输出 [20,10,2,6,10,8]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_6d_perm_103254)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[6] = {1, 0, 3, 2, 5, 4};
    gert::TilingContextPara::TensorDescription x({{10, 20, 6, 2, 8, 10}, {10, 20, 6, 2, 8, 10}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{6}, {6}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{20, 10, 2, 6, 10, 8}, {20, 10, 2, 6, 10, 8}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    std::vector<size_t> expectWorkspaces = {0};
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    EXPECT_EQ(tilingInfo.blockNum, 50);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Transpose::TransposeB4TilingData));
    const auto* tilingData = reinterpret_cast<const Transpose::TransposeB4TilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(tilingData->gmShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 1, 1, 1, 1, 1, 10, 20}));
    EXPECT_THAT(tilingData->gmInStride,
                ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 19200, 960}));
    EXPECT_THAT(tilingData->gmOutStride,
                ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 960, 9600}));
    EXPECT_THAT(tilingData->ubShape, ::testing::ElementsAreArray(std::vector<int64_t>{6, 2, 8, 10}));
    EXPECT_THAT(tilingData->ubInStride, ::testing::ElementsAreArray(std::vector<int64_t>{160, 80, 10, 1}));
    EXPECT_THAT(tilingData->ubOutStride, ::testing::ElementsAreArray(std::vector<int64_t>{80, 480, 1, 8}));
    EXPECT_THAT(tilingData->moveUbStride, ::testing::ElementsAreArray(std::vector<int64_t>{480, 80, 8, 1}));
    EXPECT_THAT(tilingData->ubPerm, ::testing::ElementsAreArray(std::vector<int64_t>{1, 0, 3, 2}));
    EXPECT_THAT(tilingData->ubShapeId, ::testing::ElementsAreArray(std::vector<int64_t>{2, 3, 4, 5}));
    EXPECT_EQ(tilingData->cutAxisIdx01, -1);
    EXPECT_EQ(tilingData->cutAxisIdx02, -1);
    EXPECT_EQ(tilingData->cutAxisTail01, -1);
    EXPECT_EQ(tilingData->cutAxisTail02, -1);
    EXPECT_EQ(tilingData->outCutId, -1);
    EXPECT_EQ(tilingData->inCutId, -1);
    EXPECT_EQ(tilingData->outerAxisNums, 2);
    EXPECT_EQ(tilingData->innerAxisNums, 4);
    EXPECT_EQ(tilingData->dimsPerCore, 4);
    EXPECT_EQ(tilingData->dimsPerCoreTail, 4);
    EXPECT_EQ(tilingData->ubSize, 25376);
    EXPECT_EQ(tilingData->permSize, 6);
    EXPECT_THAT(tilingData->perm, ::testing::ElementsAreArray(std::vector<int64_t>{1, 0, 3, 2, 5, 4, 0, 0}));
}

// 场景：B4 4D 输入 [36,204,26,50]，perm={1,0,2,3}，尾轴不转置且后两轴合并后尾轴较大，输出 [204,36,26,50]。
// 期望：走三/四维尾轴不变搬移模板，tiling key=10014（N_LAST_SMALL）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_4d_perm_1023_n_last)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[4] = {1, 0, 2, 3};
    gert::TilingContextPara::TensorDescription x({{36, 204, 26, 50}, {36, 204, 26, 50}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{204, 36, 26, 50}, {204, 36, 26, 50}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(10014));
    EXPECT_EQ(tilingInfo.blockNum, 64);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Transpose::TransposeNLastSmallTilingData));
    EXPECT_EQ(tilingInfo.workspaceSizes, std::vector<int64_t>{0});
    Transpose::TransposeNLastSmallTilingData data{};
    std::memcpy(&data, tilingInfo.tilingData.get(), sizeof(data));
    EXPECT_EQ(data.rowBytes, 26 * 50 / 2);
    EXPECT_EQ(data.outDstStrideBytes, (36 - 1) * data.rowBytes);
    EXPECT_EQ(data.aTiles * data.bTiles * data.cTiles, data.blocksPerCore * tilingInfo.blockNum + data.extraBlocks);
}

// 场景：B4 4D 大 shape 输入 [36,204,2600,3020]，perm={1,0,3,2}，交换最后两轴，输出 [204,36,3020,2600]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_4d_perm_1032_huge)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[4] = {1, 0, 3, 2};
    gert::TilingContextPara::TensorDescription x({{36, 204, 2600, 3020}, {36, 204, 2600, 3020}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{204, 36, 3020, 2600}, {204, 36, 3020, 2600}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    std::vector<size_t> expectWorkspaces = {0};
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    EXPECT_EQ(tilingInfo.blockNum, 64);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Transpose::TransposeB4TilingData));
    const auto* tilingData = reinterpret_cast<const Transpose::TransposeB4TilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(tilingData->gmShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 1, 1, 1, 36, 204, 20, 32}));
    EXPECT_THAT(tilingData->gmInStride,
                ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 1601808000, 7852000, 398640, 96}));
    EXPECT_THAT(tilingData->gmOutStride,
                ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 7852000, 282672000, 132, 249600}));
    EXPECT_THAT(tilingData->ubShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 1, 132, 96}));
    EXPECT_THAT(tilingData->ubInStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 3020, 1}));
    EXPECT_THAT(tilingData->ubOutStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 1, 2600}));
    EXPECT_THAT(tilingData->moveUbStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 2600, 1}));
    EXPECT_THAT(tilingData->ubPerm, ::testing::ElementsAreArray(std::vector<int64_t>{0, 1, 3, 2}));
    EXPECT_THAT(tilingData->ubShapeId, ::testing::ElementsAreArray(std::vector<int64_t>{-1, -1, 2, 3}));
    EXPECT_EQ(tilingData->cutAxisIdx01, 2);
    EXPECT_EQ(tilingData->cutAxisIdx02, 3);
    EXPECT_EQ(tilingData->cutAxisTail01, 92);
    EXPECT_EQ(tilingData->cutAxisTail02, 44);
    EXPECT_EQ(tilingData->outCutId, 2);
    EXPECT_EQ(tilingData->inCutId, 3);
    EXPECT_EQ(tilingData->outerAxisNums, 4);
    EXPECT_EQ(tilingData->innerAxisNums, 2);
    EXPECT_EQ(tilingData->dimsPerCore, 73440);
    EXPECT_EQ(tilingData->dimsPerCoreTail, 73440);
    EXPECT_EQ(tilingData->ubSize, 25376);
    EXPECT_EQ(tilingData->permSize, 4);
    EXPECT_THAT(tilingData->perm, ::testing::ElementsAreArray(std::vector<int64_t>{1, 0, 3, 2, 0, 0, 0, 0}));
}

// 场景：B4 4D 大 shape 输入 [36,204,2600,3020]，perm={3,2,1,0}，四轴完全逆序，输出 [3020,2600,204,36]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_4d_perm_3210_reverse)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[4] = {3, 2, 1, 0};
    gert::TilingContextPara::TensorDescription x({{36, 204, 2600, 3020}, {36, 204, 2600, 3020}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{3020, 2600, 204, 36}, {3020, 2600, 204, 36}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    std::vector<size_t> expectWorkspaces = {0};
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    EXPECT_EQ(tilingInfo.blockNum, 64);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Transpose::TransposeB4TilingData));
    const auto* tilingData = reinterpret_cast<const Transpose::TransposeB4TilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(tilingData->gmShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 1, 1, 1, 1, 51, 2600, 16}));
    EXPECT_THAT(tilingData->gmInStride,
                ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 31408000, 3020, 192}));
    EXPECT_THAT(tilingData->gmOutStride,
                ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 144, 7344, 3666124800}));
    EXPECT_THAT(tilingData->ubShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 36, 4, 192}));
    EXPECT_THAT(tilingData->ubInStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 1601808000, 7852000, 1}));
    EXPECT_THAT(tilingData->ubOutStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 1, 36, 19094400}));
    EXPECT_THAT(tilingData->moveUbStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 19094400, 36, 1}));
    EXPECT_THAT(tilingData->ubPerm, ::testing::ElementsAreArray(std::vector<int64_t>{0, 3, 2, 1}));
    EXPECT_THAT(tilingData->ubShapeId, ::testing::ElementsAreArray(std::vector<int64_t>{-1, 0, 1, 3}));
    EXPECT_EQ(tilingData->cutAxisIdx01, 1);
    EXPECT_EQ(tilingData->cutAxisIdx02, 3);
    EXPECT_EQ(tilingData->cutAxisTail01, 4);
    EXPECT_EQ(tilingData->cutAxisTail02, 140);
    EXPECT_EQ(tilingData->outCutId, 0);
    EXPECT_EQ(tilingData->inCutId, 2);
    EXPECT_EQ(tilingData->outerAxisNums, 3);
    EXPECT_EQ(tilingData->innerAxisNums, 3);
    EXPECT_EQ(tilingData->dimsPerCore, 33150);
    EXPECT_EQ(tilingData->dimsPerCoreTail, 33150);
    EXPECT_EQ(tilingData->ubSize, 25376);
    EXPECT_EQ(tilingData->permSize, 4);
    EXPECT_THAT(tilingData->perm, ::testing::ElementsAreArray(std::vector<int64_t>{3, 2, 1, 0, 0, 0, 0, 0}));
}

// 场景：B4 4D 输入 [36,204,26,66]，perm={1,0,2,3}，尾轴不转置且后两轴合并后尾轴较大，输出 [204,36,26,66]。
// 期望：走三/四维尾轴不变搬移模板，tiling key=10014（N_LAST_SMALL）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_4d_perm_1023_n_last_tail66)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[4] = {1, 0, 2, 3};
    gert::TilingContextPara::TensorDescription x({{36, 204, 26, 66}, {36, 204, 26, 66}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{204, 36, 26, 66}, {204, 36, 26, 66}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(10014));
    EXPECT_EQ(tilingInfo.blockNum, 64);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Transpose::TransposeNLastSmallTilingData));
    EXPECT_EQ(tilingInfo.workspaceSizes, std::vector<int64_t>{0});
    Transpose::TransposeNLastSmallTilingData data{};
    std::memcpy(&data, tilingInfo.tilingData.get(), sizeof(data));
    EXPECT_EQ(data.rowBytes, 26 * 66 / 2);
    EXPECT_EQ(data.outDstStrideBytes, (36 - 1) * data.rowBytes);
    EXPECT_EQ(data.aTiles * data.bTiles * data.cTiles, data.blocksPerCore * tilingInfo.blockNum + data.extraBlocks);
}

// 场景：B4 3D 输入 [8,38,4]，perm={0,2,1}，仅交换后两轴（尾轴转置），输出 [8,4,38]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_3d_perm_021)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{8, 38, 4}, {8, 38, 4}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{8, 4, 38}, {8, 4, 38}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    std::vector<size_t> expectWorkspaces = {0};
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    EXPECT_EQ(tilingInfo.blockNum, 1);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Transpose::TransposeB4TilingData));
    const auto* tilingData = reinterpret_cast<const Transpose::TransposeB4TilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(tilingData->gmShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 1, 1, 1, 1, 1, 1, 1}));
    EXPECT_THAT(tilingData->gmInStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 1}));
    EXPECT_THAT(tilingData->gmOutStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 8}));
    EXPECT_THAT(tilingData->ubShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 8, 38, 4}));
    EXPECT_THAT(tilingData->ubInStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 152, 4, 1}));
    EXPECT_THAT(tilingData->ubOutStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 152, 1, 38}));
    EXPECT_THAT(tilingData->moveUbStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 152, 38, 1}));
    EXPECT_THAT(tilingData->ubPerm, ::testing::ElementsAreArray(std::vector<int64_t>{0, 1, 3, 2}));
    EXPECT_THAT(tilingData->ubShapeId, ::testing::ElementsAreArray(std::vector<int64_t>{-1, 0, 1, 2}));
    EXPECT_EQ(tilingData->cutAxisIdx01, -1);
    EXPECT_EQ(tilingData->cutAxisIdx02, -1);
    EXPECT_EQ(tilingData->cutAxisTail01, -1);
    EXPECT_EQ(tilingData->cutAxisTail02, -1);
    EXPECT_EQ(tilingData->outCutId, -1);
    EXPECT_EQ(tilingData->inCutId, -1);
    EXPECT_EQ(tilingData->outerAxisNums, 1);
    EXPECT_EQ(tilingData->innerAxisNums, 3);
    EXPECT_EQ(tilingData->dimsPerCore, 1);
    EXPECT_EQ(tilingData->dimsPerCoreTail, 1);
    EXPECT_EQ(tilingData->ubSize, 25376);
    EXPECT_EQ(tilingData->permSize, 3);
    EXPECT_THAT(tilingData->perm, ::testing::ElementsAreArray(std::vector<int64_t>{0, 2, 1, 0, 0, 0, 0, 0}));
}

// 场景：B4 3D 输入 [672,7,83]，perm={1,2,0}，尾轴参与转置，输出 [7,83,672]。
// 期望：B4 双切分无法为该 shape 找到合法切分轴，tiling 返回失败（GRAPH_FAILED），不接受该 shape。
TEST_F(TransposeB4Tiling, b4_3d_perm_120_dual_cut_fail)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[3] = {1, 2, 0};
    gert::TilingContextPara::TensorDescription x({{672, 7, 83}, {672, 7, 83}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{7, 83, 672}, {7, 83, 672}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    auto bundle = BuildTilingContext(tilingContextPara);
    EXPECT_EQ(RunTiling(bundle, tilingContextPara), ge::GRAPH_FAILED);
}

// 场景：FLOAT4_E1M2 3D 输入 [32,8,55480]，perm={2,0,1}，4bit 类型的尾轴转置，输出 [55480,32,8]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_3d_e1m2_perm_201)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[3] = {2, 0, 1};
    gert::TilingContextPara::TensorDescription x({{32, 8, 55480}, {32, 8, 55480}}, ge::DT_FLOAT4_E1M2, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{55480, 32, 8}, {55480, 32, 8}}, ge::DT_FLOAT4_E1M2, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    std::vector<size_t> expectWorkspaces = {0};
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    EXPECT_EQ(tilingInfo.blockNum, 58);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Transpose::TransposeB4TilingData));
    const auto* tilingData = reinterpret_cast<const Transpose::TransposeB4TilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(tilingData->gmShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 1, 1, 1, 1, 1, 1, 289}));
    EXPECT_THAT(tilingData->gmInStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 192}));
    EXPECT_THAT(tilingData->gmOutStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 49152}));
    EXPECT_THAT(tilingData->ubShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 1, 256, 192}));
    EXPECT_THAT(tilingData->ubInStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 55480, 1}));
    EXPECT_THAT(tilingData->ubOutStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 1, 256}));
    EXPECT_THAT(tilingData->moveUbStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 256, 1}));
    EXPECT_THAT(tilingData->ubPerm, ::testing::ElementsAreArray(std::vector<int64_t>{0, 1, 3, 2}));
    EXPECT_THAT(tilingData->ubShapeId, ::testing::ElementsAreArray(std::vector<int64_t>{-1, -1, 0, 1}));
    EXPECT_EQ(tilingData->cutAxisIdx01, 1);
    EXPECT_EQ(tilingData->cutAxisIdx02, -1);
    EXPECT_EQ(tilingData->cutAxisTail01, 184);
    EXPECT_EQ(tilingData->cutAxisTail02, -1);
    EXPECT_EQ(tilingData->outCutId, 0);
    EXPECT_EQ(tilingData->inCutId, 0);
    EXPECT_EQ(tilingData->outerAxisNums, 1);
    EXPECT_EQ(tilingData->innerAxisNums, 2);
    EXPECT_EQ(tilingData->dimsPerCore, 5);
    EXPECT_EQ(tilingData->dimsPerCoreTail, 4);
    EXPECT_EQ(tilingData->ubSize, 25376);
    EXPECT_EQ(tilingData->permSize, 2);
    EXPECT_THAT(tilingData->perm, ::testing::ElementsAreArray(std::vector<int64_t>{1, 0, 0, 0, 0, 0, 0, 0}));
}

// 场景：FLOAT4_E2M1 3D 输入 [3,15968,6]，perm={0,2,1}，4bit 类型的尾轴转置，输出 [3,6,15968]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_3d_e2m1_perm_021)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{3, 15968, 6}, {3, 15968, 6}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{3, 6, 15968}, {3, 6, 15968}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    std::vector<size_t> expectWorkspaces = {0};
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    EXPECT_EQ(tilingInfo.blockNum, 62);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Transpose::TransposeB4TilingData));
    const auto* tilingData = reinterpret_cast<const Transpose::TransposeB4TilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(tilingData->gmShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 1, 1, 1, 1, 1, 3, 41}));
    EXPECT_THAT(tilingData->gmInStride,
                ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 95808, 2376}));
    EXPECT_THAT(tilingData->gmOutStride,
                ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 95808, 396}));
    EXPECT_THAT(tilingData->ubShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 1, 396, 6}));
    EXPECT_THAT(tilingData->ubInStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 6, 1}));
    EXPECT_THAT(tilingData->ubOutStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 1, 15968}));
    EXPECT_THAT(tilingData->moveUbStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 15968, 1}));
    EXPECT_THAT(tilingData->ubPerm, ::testing::ElementsAreArray(std::vector<int64_t>{0, 1, 3, 2}));
    EXPECT_THAT(tilingData->ubShapeId, ::testing::ElementsAreArray(std::vector<int64_t>{-1, -1, 1, 2}));
    EXPECT_EQ(tilingData->cutAxisIdx01, 1);
    EXPECT_EQ(tilingData->cutAxisIdx02, -1);
    EXPECT_EQ(tilingData->cutAxisTail01, 128);
    EXPECT_EQ(tilingData->cutAxisTail02, -1);
    EXPECT_EQ(tilingData->outCutId, 1);
    EXPECT_EQ(tilingData->inCutId, 1);
    EXPECT_EQ(tilingData->outerAxisNums, 2);
    EXPECT_EQ(tilingData->innerAxisNums, 2);
    EXPECT_EQ(tilingData->dimsPerCore, 2);
    EXPECT_EQ(tilingData->dimsPerCoreTail, 1);
    EXPECT_EQ(tilingData->ubSize, 25376);
    EXPECT_EQ(tilingData->permSize, 3);
    EXPECT_THAT(tilingData->perm, ::testing::ElementsAreArray(std::vector<int64_t>{0, 2, 1, 0, 0, 0, 0, 0}));
}

// 场景：B4 4D 大 shape 输入 [3690,204,2600,3020]，perm={1,3,2,0}，输出 [204,3020,2600,3690]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_4d_perm_1320_large)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[4] = {1, 3, 2, 0};
    gert::TilingContextPara::TensorDescription x({{3690, 204, 2600, 3020}, {3690, 204, 2600, 3020}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{204, 3020, 2600, 3690}, {204, 3020, 2600, 3690}},
                                                   ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    std::vector<size_t> expectWorkspaces = {0};
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    EXPECT_EQ(tilingInfo.blockNum, 64);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Transpose::TransposeB4TilingData));
    const auto* tilingData = reinterpret_cast<const Transpose::TransposeB4TilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(tilingData->gmShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 1, 1, 1, 28, 204, 2600, 32}));
    EXPECT_THAT(tilingData->gmInStride,
                ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 211438656000, 7852000, 3020, 96}));
    EXPECT_THAT(tilingData->gmOutStride,
                ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 132, 28973880000, 3690, 921024000}));
    EXPECT_THAT(tilingData->ubShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 1, 132, 96}));
    EXPECT_THAT(tilingData->ubInStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 1601808000, 1}));
    EXPECT_THAT(tilingData->ubOutStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 1, 9594000}));
    EXPECT_THAT(tilingData->moveUbStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 9594000, 1}));
    EXPECT_THAT(tilingData->ubPerm, ::testing::ElementsAreArray(std::vector<int64_t>{0, 1, 3, 2}));
    EXPECT_THAT(tilingData->ubShapeId, ::testing::ElementsAreArray(std::vector<int64_t>{-1, -1, 0, 3}));
    EXPECT_EQ(tilingData->cutAxisIdx01, 0);
    EXPECT_EQ(tilingData->cutAxisIdx02, 3);
    EXPECT_EQ(tilingData->cutAxisTail01, 126);
    EXPECT_EQ(tilingData->cutAxisTail02, 44);
    EXPECT_EQ(tilingData->outCutId, 0);
    EXPECT_EQ(tilingData->inCutId, 3);
    EXPECT_EQ(tilingData->outerAxisNums, 4);
    EXPECT_EQ(tilingData->innerAxisNums, 2);
    EXPECT_EQ(tilingData->dimsPerCore, 7425600);
    EXPECT_EQ(tilingData->dimsPerCoreTail, 7425600);
    EXPECT_EQ(tilingData->ubSize, 25376);
    EXPECT_EQ(tilingData->permSize, 4);
    EXPECT_THAT(tilingData->perm, ::testing::ElementsAreArray(std::vector<int64_t>{1, 3, 2, 0, 0, 0, 0, 0}));
}

// 场景：B4 4D 高瘦 shape 输入 [67785,18,4,2]，perm={1,3,0,2}，输出 [18,2,67785,4]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_4d_perm_1302_tall)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[4] = {1, 3, 0, 2};
    gert::TilingContextPara::TensorDescription x({{67785, 18, 4, 2}, {67785, 18, 4, 2}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{18, 2, 67785, 4}, {18, 2, 67785, 4}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    std::vector<size_t> expectWorkspaces = {0};
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    EXPECT_EQ(tilingInfo.blockNum, 64);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Transpose::TransposeB4TilingData));
    const auto* tilingData = reinterpret_cast<const Transpose::TransposeB4TilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(tilingData->gmShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 1, 1, 1, 1, 1, 5649, 18}));
    EXPECT_THAT(tilingData->gmInStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 1728, 8}));
    EXPECT_THAT(tilingData->gmOutStride,
                ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 48, 542280}));
    EXPECT_THAT(tilingData->ubShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 12, 4, 2}));
    EXPECT_THAT(tilingData->ubInStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 144, 2, 1}));
    EXPECT_THAT(tilingData->ubOutStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 4, 1, 271140}));
    EXPECT_THAT(tilingData->moveUbStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 271140, 4, 1}));
    EXPECT_THAT(tilingData->ubPerm, ::testing::ElementsAreArray(std::vector<int64_t>{0, 3, 1, 2}));
    EXPECT_THAT(tilingData->ubShapeId, ::testing::ElementsAreArray(std::vector<int64_t>{-1, 0, 2, 3}));
    EXPECT_EQ(tilingData->cutAxisIdx01, 0);
    EXPECT_EQ(tilingData->cutAxisIdx02, -1);
    EXPECT_EQ(tilingData->cutAxisTail01, 9);
    EXPECT_EQ(tilingData->cutAxisTail02, -1);
    EXPECT_EQ(tilingData->outCutId, 0);
    EXPECT_EQ(tilingData->inCutId, 0);
    EXPECT_EQ(tilingData->outerAxisNums, 2);
    EXPECT_EQ(tilingData->innerAxisNums, 3);
    EXPECT_EQ(tilingData->dimsPerCore, 1589);
    EXPECT_EQ(tilingData->dimsPerCoreTail, 1575);
    EXPECT_EQ(tilingData->ubSize, 25376);
    EXPECT_EQ(tilingData->permSize, 4);
    EXPECT_THAT(tilingData->perm, ::testing::ElementsAreArray(std::vector<int64_t>{1, 3, 0, 2, 0, 0, 0, 0}));
}

// 场景：B4 3D 输入 [630,8112,40]，perm={1,0,2}，尾轴不转置，输出 [8112,630,40]。
// 期望：按 int8 视图处理后尾轴（40/2=20）小于 32，走 NDDMA 双轴切分模板，
//       tiling key=10003（NDDMA_CUT_2）；tiling 成功。
TEST_F(TransposeB4Tiling, b4_3d_perm_102_cut_twice)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[3] = {1, 0, 2};
    gert::TilingContextPara::TensorDescription x({{630, 8112, 40}, {630, 8112, 40}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{8112, 630, 40}, {8112, 630, 40}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, info));
    ASSERT_EQ(info.tilingKey, 10003);
    EXPECT_EQ(info.blockNum, static_cast<uint64_t>(compileInfo.coreNum));
    ASSERT_EQ(info.tilingDataSize, sizeof(Transpose::TransposeNddmaCut2TilingData));
    Transpose::TransposeNddmaCut2TilingData data{};
    std::memcpy(&data, info.tilingData.get(), sizeof(data));
    EXPECT_EQ(info.workspaceSizes, std::vector<int64_t>{0});
    EXPECT_EQ(data.blkFactor, 10);
    EXPECT_EQ(data.blkTailFactor, 10);
    EXPECT_EQ(data.inUbFactor, 25);
    EXPECT_EQ(data.outUbFactor, 315);
}

// 场景：B4 3D 小 shape 输入 [456,848,8]，perm={1,0,2}，尾轴不转置，输出 [848,456,8]。
// 期望：B4 双切分不适用（尾轴不转置），4bit 按 int8 视图后数据量低于小 shape 阈值，
//       走 10011 二/三维 SIMT 24B 精简 TilingData。
TEST_F(TransposeB4Tiling, b4_3d_perm_102_small_shape)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[3] = {1, 0, 2};
    gert::TilingContextPara::TensorDescription x({{456, 848, 8}, {456, 848, 8}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{848, 456, 8}, {848, 456, 8}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<uint64_t>(10011));
    EXPECT_EQ(tilingInfo.blockNum, 63U);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Transpose::TransposeSmallShape23DTilingData));
    const auto* tilingData = reinterpret_cast<const Transpose::TransposeSmallShape23DTilingData*>(
        tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->dim, 3);
    // 4bit 视为 int8, 尾轴元素数按 2 个一字节折半: 8 -> 4
    EXPECT_THAT(tilingData->inputShape, ::testing::ElementsAreArray(std::vector<int32_t>{456, 848, 4}));
    EXPECT_THAT(tilingData->perm, ::testing::ElementsAreArray(std::vector<int8_t>{1, 0, 2}));
}

// 场景：B4 3D 输入 [456,847,8]，perm={0,2,1}，首轴为偶、中间轴为奇的尾轴转置，输出 [456,8,847]。
// 期望：B4 双切分无法为该 shape 找到合法切分轴，tiling 返回失败（GRAPH_FAILED），不接受该 shape。
TEST_F(TransposeB4Tiling, b4_3d_perm_021_dual_cut_fail)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{456, 847, 8}, {456, 847, 8}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{456, 8, 847}, {456, 8, 847}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    auto bundle = BuildTilingContext(tilingContextPara);
    EXPECT_EQ(RunTiling(bundle, tilingContextPara), ge::GRAPH_FAILED);
}

// 场景：B4 3D 输入 [457,848,8]，perm={0,2,1}，与失败用例相反的首轴为奇、中间轴为偶，输出 [457,8,848]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_3d_perm_021_odd_first_dim)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[3] = {0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{457, 848, 8}, {457, 848, 8}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{457, 8, 848}, {457, 8, 848}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    std::vector<size_t> expectWorkspaces = {0};
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    EXPECT_EQ(tilingInfo.blockNum, 63);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Transpose::TransposeB4TilingData));
    const auto* tilingData = reinterpret_cast<const Transpose::TransposeB4TilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(tilingData->gmShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 1, 1, 1, 1, 1, 457, 3}));
    EXPECT_THAT(tilingData->gmInStride,
                ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 6784, 3168}));
    EXPECT_THAT(tilingData->gmOutStride,
                ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 6784, 396}));
    EXPECT_THAT(tilingData->ubShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 1, 396, 8}));
    EXPECT_THAT(tilingData->ubInStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 8, 1}));
    EXPECT_THAT(tilingData->ubOutStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 1, 848}));
    EXPECT_THAT(tilingData->moveUbStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 848, 1}));
    EXPECT_THAT(tilingData->ubPerm, ::testing::ElementsAreArray(std::vector<int64_t>{0, 1, 3, 2}));
    EXPECT_THAT(tilingData->ubShapeId, ::testing::ElementsAreArray(std::vector<int64_t>{-1, -1, 1, 2}));
    EXPECT_EQ(tilingData->cutAxisIdx01, 1);
    EXPECT_EQ(tilingData->cutAxisIdx02, -1);
    EXPECT_EQ(tilingData->cutAxisTail01, 56);
    EXPECT_EQ(tilingData->cutAxisTail02, -1);
    EXPECT_EQ(tilingData->outCutId, 1);
    EXPECT_EQ(tilingData->inCutId, 1);
    EXPECT_EQ(tilingData->outerAxisNums, 2);
    EXPECT_EQ(tilingData->innerAxisNums, 2);
    EXPECT_EQ(tilingData->dimsPerCore, 22);
    EXPECT_EQ(tilingData->dimsPerCoreTail, 7);
    EXPECT_EQ(tilingData->ubSize, 25376);
    EXPECT_EQ(tilingData->permSize, 3);
    EXPECT_THAT(tilingData->perm, ::testing::ElementsAreArray(std::vector<int64_t>{0, 2, 1, 0, 0, 0, 0, 0}));
}

// 场景：B4 4D 高瘦 shape 输入 [67785,17,4,2]，perm={1,3,0,2}，输出 [17,2,67785,4]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_4d_perm_1302_tall_batch17)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[4] = {1, 3, 0, 2};
    gert::TilingContextPara::TensorDescription x({{67785, 17, 4, 2}, {67785, 17, 4, 2}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{17, 2, 67785, 4}, {17, 2, 67785, 4}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    std::vector<size_t> expectWorkspaces = {0};
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    EXPECT_EQ(tilingInfo.blockNum, 64);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Transpose::TransposeB4TilingData));
    const auto* tilingData = reinterpret_cast<const Transpose::TransposeB4TilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(tilingData->gmShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 1, 1, 1, 1, 1, 5649, 17}));
    EXPECT_THAT(tilingData->gmInStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 1632, 8}));
    EXPECT_THAT(tilingData->gmOutStride,
                ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 48, 542280}));
    EXPECT_THAT(tilingData->ubShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 12, 4, 2}));
    EXPECT_THAT(tilingData->ubInStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 136, 2, 1}));
    EXPECT_THAT(tilingData->ubOutStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 4, 1, 271140}));
    EXPECT_THAT(tilingData->moveUbStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 271140, 4, 1}));
    EXPECT_THAT(tilingData->ubPerm, ::testing::ElementsAreArray(std::vector<int64_t>{0, 3, 1, 2}));
    EXPECT_THAT(tilingData->ubShapeId, ::testing::ElementsAreArray(std::vector<int64_t>{-1, 0, 2, 3}));
    EXPECT_EQ(tilingData->cutAxisIdx01, 0);
    EXPECT_EQ(tilingData->cutAxisIdx02, -1);
    EXPECT_EQ(tilingData->cutAxisTail01, 9);
    EXPECT_EQ(tilingData->cutAxisTail02, -1);
    EXPECT_EQ(tilingData->outCutId, 0);
    EXPECT_EQ(tilingData->inCutId, 0);
    EXPECT_EQ(tilingData->outerAxisNums, 2);
    EXPECT_EQ(tilingData->innerAxisNums, 3);
    EXPECT_EQ(tilingData->dimsPerCore, 1501);
    EXPECT_EQ(tilingData->dimsPerCoreTail, 1470);
    EXPECT_EQ(tilingData->ubSize, 25376);
    EXPECT_EQ(tilingData->permSize, 4);
    EXPECT_THAT(tilingData->perm, ::testing::ElementsAreArray(std::vector<int64_t>{1, 3, 0, 2, 0, 0, 0, 0}));
}

// 场景：B4 4D 输入 [6775,17,48,20]，perm={1,0,3,2}，小批量交换最后两轴，输出 [17,6775,20,48]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_4d_perm_1032_small_batch)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[4] = {1, 0, 3, 2};
    gert::TilingContextPara::TensorDescription x({{6775, 17, 48, 20}, {6775, 17, 48, 20}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{17, 6775, 20, 48}, {17, 6775, 20, 48}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    std::vector<size_t> expectWorkspaces = {0};
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    EXPECT_EQ(tilingInfo.blockNum, 64);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Transpose::TransposeB4TilingData));
    const auto* tilingData = reinterpret_cast<const Transpose::TransposeB4TilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(tilingData->gmShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 1, 1, 1, 1, 1, 2259, 6}));
    EXPECT_THAT(tilingData->gmInStride,
                ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 48960, 2880}));
    EXPECT_THAT(tilingData->gmOutStride,
                ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 2880, 19512000}));
    EXPECT_THAT(tilingData->ubShape, ::testing::ElementsAreArray(std::vector<int64_t>{3, 3, 48, 20}));
    EXPECT_THAT(tilingData->ubInStride, ::testing::ElementsAreArray(std::vector<int64_t>{16320, 960, 20, 1}));
    EXPECT_THAT(tilingData->ubOutStride, ::testing::ElementsAreArray(std::vector<int64_t>{960, 6504000, 1, 48}));
    EXPECT_THAT(tilingData->moveUbStride, ::testing::ElementsAreArray(std::vector<int64_t>{6504000, 960, 48, 1}));
    EXPECT_THAT(tilingData->ubPerm, ::testing::ElementsAreArray(std::vector<int64_t>{1, 0, 3, 2}));
    EXPECT_THAT(tilingData->ubShapeId, ::testing::ElementsAreArray(std::vector<int64_t>{0, 1, 2, 3}));
    EXPECT_EQ(tilingData->cutAxisIdx01, 0);
    EXPECT_EQ(tilingData->cutAxisIdx02, 1);
    EXPECT_EQ(tilingData->cutAxisTail01, 1);
    EXPECT_EQ(tilingData->cutAxisTail02, 2);
    EXPECT_EQ(tilingData->outCutId, 0);
    EXPECT_EQ(tilingData->inCutId, 1);
    EXPECT_EQ(tilingData->outerAxisNums, 2);
    EXPECT_EQ(tilingData->innerAxisNums, 4);
    EXPECT_EQ(tilingData->dimsPerCore, 212);
    EXPECT_EQ(tilingData->dimsPerCoreTail, 198);
    EXPECT_EQ(tilingData->ubSize, 25376);
    EXPECT_EQ(tilingData->permSize, 4);
    EXPECT_THAT(tilingData->perm, ::testing::ElementsAreArray(std::vector<int64_t>{1, 0, 3, 2, 0, 0, 0, 0}));
}

// 场景：B4 5D 输入 [8,97,8,494,2]，perm={3,2,1,4,0}，输出 [494,8,97,2,8]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_5d_perm_32140)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[5] = {3, 2, 1, 4, 0};
    gert::TilingContextPara::TensorDescription x({{8, 97, 8, 494, 2}, {8, 97, 8, 494, 2}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{494, 8, 97, 2, 8}, {494, 8, 97, 2, 8}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    std::vector<size_t> expectWorkspaces = {0};
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    EXPECT_EQ(tilingInfo.blockNum, 64);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Transpose::TransposeB4TilingData));
    const auto* tilingData = reinterpret_cast<const Transpose::TransposeB4TilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(tilingData->gmShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 1, 1, 1, 1, 97, 8, 42}));
    EXPECT_THAT(tilingData->gmInStride,
                ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 7904, 988, 24}));
    EXPECT_THAT(tilingData->gmOutStride,
                ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 16, 1552, 148992}));
    EXPECT_THAT(tilingData->ubShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 8, 12, 2}));
    EXPECT_THAT(tilingData->ubInStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 766688, 2, 1}));
    EXPECT_THAT(tilingData->ubOutStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 1, 12416, 8}));
    EXPECT_THAT(tilingData->moveUbStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 12416, 8, 1}));
    EXPECT_THAT(tilingData->ubPerm, ::testing::ElementsAreArray(std::vector<int64_t>{0, 2, 3, 1}));
    EXPECT_THAT(tilingData->ubShapeId, ::testing::ElementsAreArray(std::vector<int64_t>{-1, 0, 3, 4}));
    EXPECT_EQ(tilingData->cutAxisIdx01, 3);
    EXPECT_EQ(tilingData->cutAxisIdx02, -1);
    EXPECT_EQ(tilingData->cutAxisTail01, 2);
    EXPECT_EQ(tilingData->cutAxisTail02, -1);
    EXPECT_EQ(tilingData->outCutId, 2);
    EXPECT_EQ(tilingData->inCutId, 2);
    EXPECT_EQ(tilingData->outerAxisNums, 3);
    EXPECT_EQ(tilingData->innerAxisNums, 3);
    EXPECT_EQ(tilingData->dimsPerCore, 510);
    EXPECT_EQ(tilingData->dimsPerCoreTail, 462);
    EXPECT_EQ(tilingData->ubSize, 25376);
    EXPECT_EQ(tilingData->permSize, 5);
    EXPECT_THAT(tilingData->perm, ::testing::ElementsAreArray(std::vector<int64_t>{3, 2, 1, 4, 0, 0, 0, 0}));
}

// 场景：B4 4D 输入 [54,9,14,46]，perm={2,1,3,0}，输出 [14,9,46,54]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_4d_perm_2130)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[4] = {2, 1, 3, 0};
    gert::TilingContextPara::TensorDescription x({{54, 9, 14, 46}, {54, 9, 14, 46}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{14, 9, 46, 54}, {14, 9, 46, 54}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    std::vector<size_t> expectWorkspaces = {0};
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    EXPECT_EQ(tilingInfo.blockNum, 63);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Transpose::TransposeB4TilingData));
    const auto* tilingData = reinterpret_cast<const Transpose::TransposeB4TilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(tilingData->gmShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 1, 1, 1, 1, 1, 7, 9}));
    EXPECT_THAT(tilingData->gmInStride,
                ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 46368, 644}));
    EXPECT_THAT(tilingData->gmOutStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 8, 2484}));
    EXPECT_THAT(tilingData->ubShape, ::testing::ElementsAreArray(std::vector<int64_t>{8, 1, 14, 46}));
    EXPECT_THAT(tilingData->ubInStride, ::testing::ElementsAreArray(std::vector<int64_t>{5796, 644, 46, 1}));
    EXPECT_THAT(tilingData->ubOutStride, ::testing::ElementsAreArray(std::vector<int64_t>{1, 2484, 22356, 54}));
    EXPECT_THAT(tilingData->moveUbStride, ::testing::ElementsAreArray(std::vector<int64_t>{22356, 2484, 54, 1}));
    EXPECT_THAT(tilingData->ubPerm, ::testing::ElementsAreArray(std::vector<int64_t>{2, 1, 3, 0}));
    EXPECT_THAT(tilingData->ubShapeId, ::testing::ElementsAreArray(std::vector<int64_t>{0, 1, 2, 3}));
    EXPECT_EQ(tilingData->cutAxisIdx01, 0);
    EXPECT_EQ(tilingData->cutAxisIdx02, 1);
    EXPECT_EQ(tilingData->cutAxisTail01, 6);
    EXPECT_EQ(tilingData->cutAxisTail02, 1);
    EXPECT_EQ(tilingData->outCutId, 0);
    EXPECT_EQ(tilingData->inCutId, 1);
    EXPECT_EQ(tilingData->outerAxisNums, 2);
    EXPECT_EQ(tilingData->innerAxisNums, 4);
    EXPECT_EQ(tilingData->dimsPerCore, 1);
    EXPECT_EQ(tilingData->dimsPerCoreTail, 1);
    EXPECT_EQ(tilingData->ubSize, 25376);
    EXPECT_EQ(tilingData->permSize, 4);
    EXPECT_THAT(tilingData->perm, ::testing::ElementsAreArray(std::vector<int64_t>{2, 1, 3, 0, 0, 0, 0, 0}));
}

// 场景：B4 4D 输入 [404,128,3,2]，perm={3,0,2,1}，输出 [2,404,3,128]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_4d_perm_3021)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[4] = {3, 0, 2, 1};
    gert::TilingContextPara::TensorDescription x({{404, 128, 3, 2}, {404, 128, 3, 2}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{2, 404, 3, 128}, {2, 404, 3, 128}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    std::vector<size_t> expectWorkspaces = {0};
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    EXPECT_EQ(tilingInfo.blockNum, 51);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Transpose::TransposeB4TilingData));
    const auto* tilingData = reinterpret_cast<const Transpose::TransposeB4TilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(tilingData->gmShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 1, 1, 1, 1, 1, 1, 202}));
    EXPECT_THAT(tilingData->gmInStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 1536}));
    EXPECT_THAT(tilingData->gmOutStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 768}));
    EXPECT_THAT(tilingData->ubShape, ::testing::ElementsAreArray(std::vector<int64_t>{2, 128, 3, 2}));
    EXPECT_THAT(tilingData->ubInStride, ::testing::ElementsAreArray(std::vector<int64_t>{768, 6, 2, 1}));
    EXPECT_THAT(tilingData->ubOutStride, ::testing::ElementsAreArray(std::vector<int64_t>{384, 1, 128, 155136}));
    EXPECT_THAT(tilingData->moveUbStride, ::testing::ElementsAreArray(std::vector<int64_t>{155136, 384, 128, 1}));
    EXPECT_THAT(tilingData->ubPerm, ::testing::ElementsAreArray(std::vector<int64_t>{3, 0, 2, 1}));
    EXPECT_THAT(tilingData->ubShapeId, ::testing::ElementsAreArray(std::vector<int64_t>{0, 1, 2, 3}));
    EXPECT_EQ(tilingData->cutAxisIdx01, 0);
    EXPECT_EQ(tilingData->cutAxisIdx02, -1);
    EXPECT_EQ(tilingData->cutAxisTail01, 2);
    EXPECT_EQ(tilingData->cutAxisTail02, -1);
    EXPECT_EQ(tilingData->outCutId, 0);
    EXPECT_EQ(tilingData->inCutId, 0);
    EXPECT_EQ(tilingData->outerAxisNums, 1);
    EXPECT_EQ(tilingData->innerAxisNums, 4);
    EXPECT_EQ(tilingData->dimsPerCore, 4);
    EXPECT_EQ(tilingData->dimsPerCoreTail, 2);
    EXPECT_EQ(tilingData->ubSize, 25376);
    EXPECT_EQ(tilingData->permSize, 4);
    EXPECT_THAT(tilingData->perm, ::testing::ElementsAreArray(std::vector<int64_t>{3, 0, 2, 1, 0, 0, 0, 0}));
}

// 场景：B4 5D 输入 [8,77560,2,4,6]，perm={3,2,1,4,0}，输出 [4,2,77560,6,8]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_5d_perm_32140_wide)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[5] = {3, 2, 1, 4, 0};
    gert::TilingContextPara::TensorDescription x({{8, 77560, 2, 4, 6}, {8, 77560, 2, 4, 6}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{4, 2, 77560, 6, 8}, {4, 2, 77560, 6, 8}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    std::vector<size_t> expectWorkspaces = {0};
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    EXPECT_EQ(tilingInfo.blockNum, 64);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Transpose::TransposeB4TilingData));
    const auto* tilingData = reinterpret_cast<const Transpose::TransposeB4TilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(tilingData->gmShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 1, 1, 1, 1, 1, 25854, 2}));
    EXPECT_THAT(tilingData->gmInStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 144, 24}));
    EXPECT_THAT(tilingData->gmOutStride,
                ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 144, 3722880}));
    EXPECT_THAT(tilingData->ubShape, ::testing::ElementsAreArray(std::vector<int64_t>{8, 3, 4, 6}));
    EXPECT_THAT(tilingData->ubInStride, ::testing::ElementsAreArray(std::vector<int64_t>{3722880, 48, 6, 1}));
    EXPECT_THAT(tilingData->ubOutStride, ::testing::ElementsAreArray(std::vector<int64_t>{1, 48, 7445760, 8}));
    EXPECT_THAT(tilingData->moveUbStride, ::testing::ElementsAreArray(std::vector<int64_t>{7445760, 48, 8, 1}));
    EXPECT_THAT(tilingData->ubPerm, ::testing::ElementsAreArray(std::vector<int64_t>{2, 1, 3, 0}));
    EXPECT_THAT(tilingData->ubShapeId, ::testing::ElementsAreArray(std::vector<int64_t>{0, 1, 3, 4}));
    EXPECT_EQ(tilingData->cutAxisIdx01, 1);
    EXPECT_EQ(tilingData->cutAxisIdx02, -1);
    EXPECT_EQ(tilingData->cutAxisTail01, 1);
    EXPECT_EQ(tilingData->cutAxisTail02, -1);
    EXPECT_EQ(tilingData->outCutId, 0);
    EXPECT_EQ(tilingData->inCutId, 0);
    EXPECT_EQ(tilingData->outerAxisNums, 2);
    EXPECT_EQ(tilingData->innerAxisNums, 4);
    EXPECT_EQ(tilingData->dimsPerCore, 808);
    EXPECT_EQ(tilingData->dimsPerCoreTail, 804);
    EXPECT_EQ(tilingData->ubSize, 25376);
    EXPECT_EQ(tilingData->permSize, 5);
    EXPECT_THAT(tilingData->perm, ::testing::ElementsAreArray(std::vector<int64_t>{3, 2, 1, 4, 0, 0, 0, 0}));
}

// 场景：B4 5D 输入 [2,2875,147,8,4]，perm={2,4,1,0,3}，输出 [147,4,2875,2,8]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_5d_perm_24103)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[5] = {2, 4, 1, 0, 3};
    gert::TilingContextPara::TensorDescription x({{2, 2875, 147, 8, 4}, {2, 2875, 147, 8, 4}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{147, 4, 2875, 2, 8}, {147, 4, 2875, 2, 8}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    std::vector<size_t> expectWorkspaces = {0};
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    EXPECT_EQ(tilingInfo.blockNum, 64);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Transpose::TransposeB4TilingData));
    const auto* tilingData = reinterpret_cast<const Transpose::TransposeB4TilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(tilingData->gmShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 1, 1, 1, 1, 1, 480, 147}));
    EXPECT_THAT(tilingData->gmInStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 28224, 32}));
    EXPECT_THAT(tilingData->gmOutStride,
                ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 96, 184000}));
    EXPECT_THAT(tilingData->ubShape, ::testing::ElementsAreArray(std::vector<int64_t>{2, 6, 8, 4}));
    EXPECT_THAT(tilingData->ubInStride, ::testing::ElementsAreArray(std::vector<int64_t>{13524000, 4704, 4, 1}));
    EXPECT_THAT(tilingData->ubOutStride, ::testing::ElementsAreArray(std::vector<int64_t>{8, 16, 1, 46000}));
    EXPECT_THAT(tilingData->moveUbStride, ::testing::ElementsAreArray(std::vector<int64_t>{46000, 16, 8, 1}));
    EXPECT_THAT(tilingData->ubPerm, ::testing::ElementsAreArray(std::vector<int64_t>{3, 1, 0, 2}));
    EXPECT_THAT(tilingData->ubShapeId, ::testing::ElementsAreArray(std::vector<int64_t>{0, 1, 3, 4}));
    EXPECT_EQ(tilingData->cutAxisIdx01, 1);
    EXPECT_EQ(tilingData->cutAxisIdx02, -1);
    EXPECT_EQ(tilingData->cutAxisTail01, 1);
    EXPECT_EQ(tilingData->cutAxisTail02, -1);
    EXPECT_EQ(tilingData->outCutId, 0);
    EXPECT_EQ(tilingData->inCutId, 0);
    EXPECT_EQ(tilingData->outerAxisNums, 2);
    EXPECT_EQ(tilingData->innerAxisNums, 4);
    EXPECT_EQ(tilingData->dimsPerCore, 1103);
    EXPECT_EQ(tilingData->dimsPerCoreTail, 1071);
    EXPECT_EQ(tilingData->ubSize, 25376);
    EXPECT_EQ(tilingData->permSize, 5);
    EXPECT_THAT(tilingData->perm, ::testing::ElementsAreArray(std::vector<int64_t>{2, 4, 1, 0, 3, 0, 0, 0}));
}

// 场景：B4 5D 超大 shape 输入 [8008,77560,2,4,6]，perm={3,2,1,4,0}，输出 [4,2,77560,6,8008]。
// 期望：尾轴转置命中 B4 双切分模板，tiling key=10009（B4）；
//       tiling 成功。
TEST_F(TransposeB4Tiling, b4_5d_perm_32140_huge)
{
    optiling::TransposeCompilerInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 248 * 1024;

    int64_t perm_value[5] = {3, 2, 1, 4, 0};
    gert::TilingContextPara::TensorDescription x({{8008, 77560, 2, 4, 6}, {8008, 77560, 2, 4, 6}}, ge::DT_FLOAT4_E2M1,
                                                 ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription perm({{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND, true, &perm_value);
    gert::TilingContextPara::TensorDescription out({{4, 2, 77560, 6, 8008}, {4, 2, 77560, 6, 8008}}, ge::DT_FLOAT4_E2M1,
                                                   ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara("Transpose", {x, perm}, {out}, &compileInfo);
    uint64_t expectTilingKey = 10009;
    std::vector<size_t> expectWorkspaces = {0};
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    EXPECT_EQ(tilingInfo.blockNum, 64);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(Transpose::TransposeB4TilingData));
    const auto* tilingData = reinterpret_cast<const Transpose::TransposeB4TilingData*>(tilingInfo.tilingData.get());
    EXPECT_THAT(tilingData->gmShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 1, 1, 1, 1, 42, 77560, 2}));
    EXPECT_THAT(tilingData->gmInStride,
                ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 714792960, 48, 24}));
    EXPECT_THAT(tilingData->gmOutStride,
                ::testing::ElementsAreArray(std::vector<int64_t>{0, 0, 0, 0, 0, 192, 48048, 3726602880}));
    EXPECT_THAT(tilingData->ubShape, ::testing::ElementsAreArray(std::vector<int64_t>{1, 192, 4, 6}));
    EXPECT_THAT(tilingData->ubInStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 3722880, 6, 1}));
    EXPECT_THAT(tilingData->ubOutStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 1, 7453205760, 8008}));
    EXPECT_THAT(tilingData->moveUbStride, ::testing::ElementsAreArray(std::vector<int64_t>{0, 7453205760, 8008, 1}));
    EXPECT_THAT(tilingData->ubPerm, ::testing::ElementsAreArray(std::vector<int64_t>{0, 2, 3, 1}));
    EXPECT_THAT(tilingData->ubShapeId, ::testing::ElementsAreArray(std::vector<int64_t>{-1, 0, 3, 4}));
    EXPECT_EQ(tilingData->cutAxisIdx01, 0);
    EXPECT_EQ(tilingData->cutAxisIdx02, -1);
    EXPECT_EQ(tilingData->cutAxisTail01, 136);
    EXPECT_EQ(tilingData->cutAxisTail02, -1);
    EXPECT_EQ(tilingData->outCutId, 0);
    EXPECT_EQ(tilingData->inCutId, 0);
    EXPECT_EQ(tilingData->outerAxisNums, 3);
    EXPECT_EQ(tilingData->innerAxisNums, 3);
    EXPECT_EQ(tilingData->dimsPerCore, 101798);
    EXPECT_EQ(tilingData->dimsPerCoreTail, 101766);
    EXPECT_EQ(tilingData->ubSize, 25376);
    EXPECT_EQ(tilingData->permSize, 5);
    EXPECT_THAT(tilingData->perm, ::testing::ElementsAreArray(std::vector<int64_t>{3, 2, 1, 4, 0, 0, 0, 0}));
}
