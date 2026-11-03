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
 * \file test_tensor_redirect.cpp
 * \brief TensorRedirect op_kernel UT
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>
#include "tikicpulib.h"

#include "../../../op_kernel/tensor_redirect.cpp"

namespace {

constexpr uint32_t NUM_BLOCKS = 1;
constexpr size_t ALIGN_BYTES = 32;
constexpr size_t WORKSPACE_BYTES = 16 * 1024 * 1024;

size_t Align32(size_t size) { return (size + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES; }

class GmScope {
public:
    ~GmScope()
    {
        for (auto* buffer : buffers_) {
            if (buffer != nullptr) {
                AscendC::GmFree(buffer);
            }
        }
    }
    uint8_t* Allocate(size_t size)
    {
        if (count_ == buffers_.size()) {
            return nullptr;
        }
        auto* buffer = static_cast<uint8_t*>(AscendC::GmAlloc(size));
        buffers_[count_++] = buffer;
        return buffer;
    }

private:
    std::array<uint8_t*, 4> buffers_{};
    size_t count_ = 0;
};

class TensorRedirectKernelTest : public testing::Test {};

TEST_F(TensorRedirectKernelTest, empty_tensor_returns_before_zero_length_init_buffer)
{
    GmScope resources;
    const size_t dataBytes = ALIGN_BYTES;
    const size_t tilingBytes = Align32(sizeof(TensorRedirectTilingData));
    auto* x = resources.Allocate(dataBytes);
    auto* y = resources.Allocate(dataBytes);
    auto* workspace = resources.Allocate(WORKSPACE_BYTES);
    auto* tiling = resources.Allocate(tilingBytes);
    ASSERT_NE(x, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    std::fill_n(x, dataBytes, 0xA5);
    std::fill_n(y, dataBytes, 0x5A);
    std::fill_n(tiling, tilingBytes, 0); // usedCoreNum=0, ubFactor=0: host 的空 Tensor tiling

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF((tensor_redirect<0>), NUM_BLOCKS, x, y, workspace, tiling);

    for (size_t i = 0; i < dataBytes; ++i) {
        EXPECT_EQ(y[i], 0x5A) << "empty tensor must not touch output, byte=" << i;
    }
}

// Ascend950PR_9599 的 CANN 9.2 tikicpulib 尚不能正确回写设备侧 DataCopyPad 结果；
// 此处只验证正常搬运路径可执行且不破坏输入。位级精度由真机 TTK binary 用例覆盖。
TEST_F(TensorRedirectKernelTest, normal_one_block_executes_without_crash)
{
    GmScope resources;
    constexpr int64_t elementCount = 16;
    const size_t dataBytes = Align32(elementCount * sizeof(int32_t));
    const size_t tilingBytes = Align32(sizeof(TensorRedirectTilingData));
    auto* x = resources.Allocate(dataBytes);
    auto* y = resources.Allocate(dataBytes);
    auto* workspace = resources.Allocate(WORKSPACE_BYTES);
    auto* tiling = resources.Allocate(tilingBytes);
    ASSERT_NE(x, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    for (size_t i = 0; i < dataBytes; ++i) {
        x[i] = static_cast<uint8_t>((i * 37U + 11U) & 0xFFU);
    }
    std::fill_n(y, dataBytes, 0);
    std::fill_n(tiling, tilingBytes, 0);
    auto* tilingData = reinterpret_cast<TensorRedirectTilingData*>(tiling);
    tilingData->usedCoreNum = 1;
    tilingData->blockFactor = 1;
    tilingData->tailBlockFactor = 1;
    tilingData->ubFactor = elementCount;
    tilingData->tailBlockTailUbFactor = elementCount;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF((tensor_redirect<0>), NUM_BLOCKS, x, y, workspace, tiling);

    for (size_t i = 0; i < dataBytes; ++i) {
        EXPECT_EQ(x[i], static_cast<uint8_t>((i * 37U + 11U) & 0xFFU));
    }
}

} // namespace
