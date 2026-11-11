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
#include "../../../op_kernel/atan2_tiling_data.h"
#include "../../../op_kernel/atan2_tiling_key.h"
#include "../../../op_host/atan2_tiling_schedule.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

// 1. 先定义 namespace 和 mock 的 CompileInfo 结构体
namespace optiling {
struct Atan2CompileInfo {};
} // namespace optiling

// 2. 然后再使用 using namespace
using namespace optiling;

class Atan2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "Atan2Tiling SetUp" << endl; }

    static void TearDownTestCase() { cout << "Atan2Tiling TearDown " << endl; }
};

static void VerifyTiling(uint64_t ubSize, const string& expectTilingData, ge::DataType dtype = ge::DT_FLOAT16)
{
    optiling::Atan2CompileInfo compileInfo;

    // atan2 是双输入 (x1, x2)，单输出 (y)
    gert::TilingContextPara tilingContextPara("Atan2",
                                              {
                                                  // Input 0: x1
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, dtype, ge::FORMAT_ND},
                                                  // Input 1: x2
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, dtype, ge::FORMAT_ND},
                                              },
                                              {
                                                  // Output 0: y
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, dtype, ge::FORMAT_ND},
                                              },
                                              &compileInfo, 64, ubSize);

    uint64_t expectTilingKey = 0; // ELEMENTWISE_TPL_SCH_MODE_0

    std::vector<size_t> expectWorkspaces = {16777728};

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// FP16 buffers use 64 bytes per element plus 8192 bytes for Select scratch.
// Keep both budgets inside a tile interval so SDK-reserved bytes do not make
// the expected 64-element alignment depend on an exact capacity boundary.
TEST_F(Atan2Tiling, ascend910b_test_tiling_fp16_001)
{
    VerifyTiling(254U * 1024U, "128 144 1 1 3904 128 144 0 124928 ");
}

TEST_F(Atan2Tiling, ascend910b_test_tiling_fp16_smaller_ub)
{
    VerifyTiling(126U * 1024U, "128 144 1 1 1856 128 144 0 59392 ");
}

TEST_F(Atan2Tiling, ascend910b_test_tiling_fp32)
{
    // 1 * 64 * 2 * 64 = 8192 elements, split across 64 vector cores.
    VerifyTiling(254U * 1024U, "128 136 1 1 3264 128 136 0 104448 ", ge::DT_FLOAT);
}

TEST_F(Atan2Tiling, ascend910b_test_tiling_bf16)
{
    VerifyTiling(254U * 1024U, "128 144 1 1 3904 128 144 0 124928 ", ge::DT_BF16);
}

TEST_F(Atan2Tiling, schedule_balances_dma_blocks_and_keeps_tile_budget)
{
    for (uint32_t bytes : {2U, 4U}) {
        for (uint64_t elements : {1U, 17U, 160U, 16384U, 65569U}) {
            for (uint32_t cores : {1U, 8U, 64U}) {
                Atan2TilingData data{};
                uint32_t blocks = 0;
                constexpr uint64_t ub = 254U * 1024U;
                ASSERT_TRUE(optiling::atan2::BuildSchedule(elements, bytes, ub, cores, data, blocks));
                const uint64_t paddedElements = ((elements * bytes + 31) / 32) * 32 / bytes;
                const uint64_t covered = data.tailBlockNum * data.bigCoreDataNum +
                                         (blocks - data.tailBlockNum) * data.smallCoreDataNum;
                EXPECT_EQ(covered, paddedElements);
                EXPECT_LE(blocks, cores);
                EXPECT_EQ(data.bigCoreDataNum - data.smallCoreDataNum, 32 / bytes);
                EXPECT_EQ((data.finalSmallTileNum - 1) * data.tileDataNum + data.smallTailDataNum,
                          data.smallCoreDataNum);
                EXPECT_EQ((data.finalBigTileNum - 1) * data.tileDataNum + data.bigTailDataNum, data.bigCoreDataNum);
                EXPECT_EQ(data.tileDataNum % 64, 0);
                const uint64_t usedBytes = data.tileDataNum * (6 * bytes + 13 * sizeof(float)) + 8192;
                EXPECT_LE(usedBytes, ub);
                EXPECT_GT(usedBytes + 64 * (6 * bytes + 13 * sizeof(float)), ub);
            }
        }
    }
}

TEST_F(Atan2Tiling, schedule_rejects_insufficient_resources_and_kernel_offset_overflow)
{
    Atan2TilingData data{};
    uint32_t blocks = 0;
    EXPECT_FALSE(optiling::atan2::BuildSchedule(64, 4, 8192 + 64 * 76 - 1, 64, data, blocks));
    ASSERT_TRUE(optiling::atan2::BuildSchedule(64, 4, 8192 + 64 * 76, 64, data, blocks));
    EXPECT_EQ(data.tileDataNum, 64);
    EXPECT_EQ(data.atanTmpSize, 2048U);
    EXPECT_FALSE(optiling::atan2::BuildSchedule(64, 4, 256 * 1024, 0, data, blocks));
    EXPECT_FALSE(optiling::atan2::BuildSchedule(64, 1, 256 * 1024, 64, data, blocks));
    EXPECT_FALSE(optiling::atan2::BuildSchedule(0, 4, 256 * 1024, 64, data, blocks));
    EXPECT_FALSE(optiling::atan2::BuildSchedule(1ULL << 32, 4, 256 * 1024, 64, data, blocks));
}
