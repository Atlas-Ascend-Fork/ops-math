/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <cstdint>
#include <limits>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_kernel/reduce_min_v2.h"

namespace {
template <typename T>
void RunReduceMinV2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    NsReduceMinV2::ReduceMinV2<T> op;
    op.Init(x, y, workspace, reinterpret_cast<ReduceMinV2TilingData*>(tiling));
    op.Process();
}

template <typename T>
void CheckReduction(uint32_t length, uint32_t tileLength, uint32_t blockDim = 1, int32_t axes = -1)
{
    const uint32_t blockElements = 32 / sizeof(T);
    const uint32_t alignedLength = (length + blockElements - 1) / blockElements * blockElements;
    auto* x = static_cast<T*>(AscendC::GmAlloc(alignedLength * sizeof(T)));
    const uint32_t outputLength = axes == 0 ? length : 1;
    const uint32_t workspaceLength = outputLength * NsReduceMinV2::SLOT_STRIDE;
    auto* y = static_cast<T*>(AscendC::GmAlloc((outputLength * sizeof(T) + 31) / 32 * 32));
    auto* workspace = static_cast<float*>(AscendC::GmAlloc(workspaceLength * sizeof(float)));
    auto* tiling = static_cast<ReduceMinV2TilingData*>(AscendC::GmAlloc(sizeof(ReduceMinV2TilingData)));
    // A smaller value in physical padding must never affect the logical minimum.
    std::fill_n(x, alignedLength, static_cast<T>(-100));
    std::fill_n(x, length, static_cast<T>(5));
    y[0] = static_cast<T>(-999);
    std::fill_n(workspace, workspaceLength, std::numeric_limits<float>::infinity());
    *tiling = {};
    const uint32_t smallBlocks = alignedLength / blockElements / blockDim;
    tiling->tailBlockNum = alignedLength / blockElements % blockDim;
    tiling->smallCoreDataNum = smallBlocks * blockElements;
    tiling->bigCoreDataNum = (smallBlocks + 1) * blockElements;
    tiling->tileDataNum = tileLength;
    tiling->finalSmallTileNum = (tiling->smallCoreDataNum + tileLength - 1) / tileLength;
    tiling->finalBigTileNum = (tiling->bigCoreDataNum + tileLength - 1) / tileLength;
    tiling->smallTailDataNum = (tiling->smallCoreDataNum - 1) % tileLength + 1;
    tiling->bigTailDataNum = (tiling->bigCoreDataNum - 1) % tileLength + 1;
    tiling->dataTypeId = sizeof(T) == 2 ? 1 : 0;
    tiling->rows = 1;
    tiling->cols = length;
    tiling->axes = static_cast<uint32_t>(axes);
    tiling->keepdims = 1;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(RunReduceMinV2<T>, blockDim, reinterpret_cast<GM_ADDR>(x), reinterpret_cast<GM_ADDR>(y),
                reinterpret_cast<GM_ADDR>(workspace), reinterpret_cast<GM_ADDR>(tiling));
    for (uint32_t i = 0; i < outputLength; ++i) {
        EXPECT_FLOAT_EQ(static_cast<float>(y[i]), 5.0f) << "length=" << length << ", tileLength=" << tileLength;
    }
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST(ReduceMinV2, AxesAllFloatPadding)
{
    for (uint32_t length : {1, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33}) {
        CheckReduction<float>(length, 16);
    }
}

TEST(ReduceMinV2, AxesAllHalfPadding)
{
    for (uint32_t length : {1, 5, 15, 16, 17, 31, 32, 33, 63, 64, 65}) {
        CheckReduction<half>(length, 32);
    }
}
TEST(ReduceMinV2, AxesAllMultipleCores)
{
    CheckReduction<float>(85, 16, 3);
    CheckReduction<float>(97, 16, 4);
    CheckReduction<half>(161, 32, 3);
}

TEST(ReduceMinV2, AxesZeroAndOne)
{
    for (int32_t axes : {0, 1}) {
        CheckReduction<float>(5, 16, 1, axes);
        CheckReduction<float>(8, 16, 1, axes);
        CheckReduction<half>(17, 32, 1, axes);
    }
}
} // namespace
