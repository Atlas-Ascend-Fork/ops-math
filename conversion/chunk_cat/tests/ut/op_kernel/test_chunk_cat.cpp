
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_chunk_cat.cpp
 * \brief
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_kernel/chunk_cat_tiling_data.h"

extern "C" __global__ __aicore__ void chunk_cat(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

namespace {
using Shape = std::array<int64_t, 2>;
constexpr float OUTPUT_SENTINEL = -12345.0f;
constexpr size_t GUARD_ELEMENTS = 8;

struct GmDeleter {
    void operator()(uint8_t* ptr) const { AscendC::GmFree(ptr); }
};
using GmBuffer = std::unique_ptr<uint8_t, GmDeleter>;

GmBuffer AllocateGm(size_t size)
{
    return GmBuffer(static_cast<uint8_t*>(AscendC::GmAlloc(std::max(size, size_t{32}))));
}

// ICPU_RUN_KF does not propagate a failed worker's exit status. The parent must
// check a shared marker written only after the kernel and its sync checks return.
__global__ __aicore__ void ChunkCatWithCompletion(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling,
                                                  GM_ADDR completed)
{
    chunk_cat(x, y, workspace, tiling);
    AscendC::CheckSyncState();
    *reinterpret_cast<__gm__ uint32_t*>(completed) = 1;
}

void RunChunkCatCase(const std::vector<Shape>& shapes, const std::vector<float>& expected)
{
    constexpr int64_t numChunks = 2;
    constexpr int64_t elementsPerBlock = 32 / sizeof(float);
    std::vector<GmBuffer> inputs;
    int64_t outputCol = 0;
    float nextValue = 1.0f;
    bool isAllAlign = true;
    bool isHalfAlign = true;
    for (const auto& shape : shapes) {
        const size_t elementCount = shape[0] * shape[1];
        inputs.push_back(AllocateGm(elementCount * sizeof(float)));
        ASSERT_NE(inputs.back().get(), nullptr);
        auto* data = reinterpret_cast<float*>(inputs.back().get());
        for (size_t i = 0; i < elementCount; ++i) {
            data[i] = nextValue++;
        }
        const int64_t tensorCol = ((shape[0] + numChunks - 1) / numChunks) * shape[1];
        outputCol += tensorCol;
        isAllAlign = isAllAlign && tensorCol % elementsPerBlock == 0;
        isHalfAlign = isHalfAlign && tensorCol % (elementsPerBlock / 2) == 0;
    }
    ASSERT_EQ(expected.size(), static_cast<size_t>(numChunks * outputCol));

    // ListTensor layout: pointer-table offset, rank/index + shape for each input,
    // then the input pointers. These cases use rank 2, including empty shapes.
    const size_t inputNum = shapes.size();
    const size_t pointerOffset = 1 + 3 * inputNum;
    auto x = AllocateGm((pointerOffset + inputNum) * sizeof(uint64_t));
    ASSERT_NE(x.get(), nullptr);
    auto* list = reinterpret_cast<uint64_t*>(x.get());
    list[0] = pointerOffset * sizeof(uint64_t);
    for (size_t i = 0; i < inputNum; ++i) {
        list[1 + 3 * i] = uint64_t{2} | (static_cast<uint64_t>(i == 0 ? inputNum : i) << 32);
        list[2 + 3 * i] = shapes[i][0];
        list[3 + 3 * i] = shapes[i][1];
        list[pointerOffset + i] = reinterpret_cast<uintptr_t>(inputs[i].get());
    }

    auto y = AllocateGm((expected.size() + GUARD_ELEMENTS) * sizeof(float));
    auto workspace = AllocateGm(16 * 1024 * 1024);
    auto tiling = AllocateGm(sizeof(ChunkCatTilingData));
    auto completed = AllocateGm(sizeof(uint32_t));
    ASSERT_NE(y.get(), nullptr);
    ASSERT_NE(workspace.get(), nullptr);
    ASSERT_NE(tiling.get(), nullptr);
    ASSERT_NE(completed.get(), nullptr);
    auto* output = reinterpret_cast<float*>(y.get());
    std::fill_n(output, expected.size() + GUARD_ELEMENTS, OUTPUT_SENTINEL);
    *reinterpret_cast<uint32_t*>(completed.get()) = 0;

    auto* tilingData = reinterpret_cast<ChunkCatTilingData*>(tiling.get());
    *tilingData = {};
    tilingData->isAllAlign = isAllAlign;
    tilingData->isHalfAlign = isHalfAlign;
    tilingData->inputNum = inputNum;
    tilingData->inUbSize = 32768;
    tilingData->outUbSize = 65536;
    tilingData->blockRowNum = 1;
    tilingData->blockColNum = 1;
    tilingData->ubRowFactor = outputCol == 0 ? 1 : numChunks;
    tilingData->ubColFactor = outputCol == 0 ?
                                  1 :
                                  ((outputCol + elementsPerBlock - 1) / elementsPerBlock) * elementsPerBlock;
    tilingData->dim = 0;
    tilingData->numChunk = numChunks;
    tilingData->outputRow = numChunks;
    tilingData->outputCol = outputCol;
    tilingData->blockRowFactor = numChunks;
    tilingData->tailBlockRowFactor = numChunks;
    tilingData->blockColFactor = outputCol;
    tilingData->tailBlockColFactor = outputCol;

    // One AIV block matches the single-block tiling above; do not launch a second
    // vector sub-block using the CPU debugger's default mixed-core mode.
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(ChunkCatWithCompletion, 1, x.get(), y.get(), workspace.get(), tiling.get(), completed.get());

    ASSERT_EQ(*reinterpret_cast<uint32_t*>(completed.get()), 1U) << "ChunkCat worker did not complete";
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(output[i], expected[i]) << "output index " << i;
    }
    for (size_t i = expected.size(); i < expected.size() + GUARD_ELEMENTS; ++i) {
        EXPECT_FLOAT_EQ(output[i], OUTPUT_SENTINEL) << "guard index " << i;
    }
}
} // namespace

class chunk_cat_test : public testing::Test {};

TEST_F(chunk_cat_test, test_case_fp32_nonempty)
{
    std::vector<float> expected(16);
    std::iota(expected.begin(), expected.end(), 1.0f);
    RunChunkCatCase({{4, 4}}, expected);
}

TEST_F(chunk_cat_test, test_case_empty_output_tiling_no_div0)
{
    // SetEmptyTiling must result in zero UB loops, with a valid input descriptor.
    RunChunkCatCase({{0, 3}}, {});
}

TEST_F(chunk_cat_test, test_case_mixed_empty_tensors)
{
    const std::vector<float> expected = {1, 2,  3,  4,  5,  6,  7,  8,  17, 18, 19, 20, 21, 22, 23, 24,
                                         9, 10, 11, 12, 13, 14, 15, 16, 25, 26, 27, 28, 29, 30, 31, 32};
    for (const auto& emptyShape : std::vector<Shape>{{0, 4}, {4, 0}}) {
        for (size_t position = 0; position < 3; ++position) {
            SCOPED_TRACE(testing::Message()
                         << "empty shape [" << emptyShape[0] << "," << emptyShape[1] << "], position " << position);
            std::vector<Shape> shapes{{4, 4}, {4, 4}};
            shapes.insert(shapes.begin() + position, emptyShape);
            RunChunkCatCase(shapes, expected);
        }
    }
}
