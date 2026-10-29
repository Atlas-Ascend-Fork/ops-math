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
 * \file test_acosh_tiling_arch35.cpp
 * \brief Acosh Tiling UT (arch35 / Ascend950)
 *
 * Covers paths in op_host/arch35/acosh_tiling_arch35.cpp:
 *   1) Dtype paths: FP32 / FP16 / BF16 均走同一 tiling 流水，dtype 仅影响
 *      blockFactor 对齐粒度(32B 按 dtypeSize 换算: FP32=8 元素 / FP16,BF16=16 元素)。
 *   2) 多核切分(大 shape) / 单核尾块(小 shape) / 非对齐尾块（blockNum=usedCoreNum 校验）。
 *   3) 空 Tensor(totalNum==0)早返回,字段全零,BlockDim=1。
 *   4) OI-14 回归锚点: blockFactor 按 32B 字节边界换算元素当量对齐
 *      (修复前把 32 字节当 32 元素, FP32 等效 128B 过对齐, usedCoreNum 被低估;
 *       锚点用例 totalNum=3855 取自 T10 探针实测值)。
 *   5) 失败路径(ExecuteTiling 返回 false):
 *        - input rank>8 / output rank>8          → GRAPH_FAILED
 *        - 负 shape size(动态维 -1 乘积为负)      → GRAPH_FAILED
 *        - input/output shape 不一致              → GRAPH_FAILED
 *        - 不支持 dtype(DT_DOUBLE)               → GRAPH_FAILED
 *        - output dtype 与 input 不一致          → GRAPH_FAILED
 *        - coreNum==0 / ubSize==0 / ubFactor 过小 → GRAPH_FAILED
 *
 * Acosh tiling 不调用 SetTilingKey(dtype 不进 TilingKey),faker 环境下
 * tilingKey 输出槽为未初始化内存——故不走 ExecuteTestCase 的 key 断言,
 * 采用 ExecuteTiling + TilingData 字段直断言(bessel_i0e 同款先例)。
 *
 * Platform constants (UT faker): coreNum=64, ubSize=262144, ubBlockSize=32.
 *
 * Formulas (faker 平台):
 *   ubFactor    = FloorAlign(FloorDiv(262144, 4*8), 32/4) = FloorAlign(8192, 8) = 8192
 *   blockAlign  = max(32/dtypeSize, 1)  (FP32: 8, FP16/BF16: 16)
 *   blockFactor = CeilAlign(CeilDiv(totalNum, 64), blockAlign)
 *   blockNum    = CeilDiv(totalNum, blockFactor)
 */

#include <cstdint>
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_kernel/arch35/acosh_tiling_data.h"

namespace AcoshUT {
using namespace std;
using namespace ge;

static const std::string OP_NAME = "Acosh";

struct AcoshCompileInfo {};
} // namespace AcoshUT

using namespace AcoshUT;

class AcoshTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AcoshTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AcoshTilingTest TearDown" << std::endl; }
};

// 成功用例公共断言: workspace={0} + tiling data 字段 + blockNum
static void AssertAcoshTiling(const gert::TilingContextPara& para, int64_t expectTotalNum, int64_t expectBlockFactor,
                              int64_t expectUbFactor, size_t expectBlockNum)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo)) << "Acosh tiling should succeed";

    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0) << "workspace[0] should be WS_SYS_SIZE(0)";

    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(AcoshTilingData));
    auto* tilingData = reinterpret_cast<const AcoshTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->totalNum, expectTotalNum) << "totalNum mismatch";
    EXPECT_EQ(tilingData->blockFactor, expectBlockFactor) << "blockFactor mismatch";
    EXPECT_EQ(tilingData->ubFactor, expectUbFactor) << "ubFactor mismatch";
    EXPECT_EQ(tilingInfo.blockNum, expectBlockNum) << "blockNum(usedCoreNum) mismatch";
}

// ===========================================================================
// 1) FP32 multi-core aligned — 8192 elem {1,64,2,64}
//    blockFactor=CeilAlign(CeilDiv(8192,64)=128, 8)=128, blockNum=CeilDiv(8192,128)=64
// ===========================================================================
TEST_F(AcoshTilingTest, test_tiling_fp32_multi_core_001)
{
    AcoshCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(OP_NAME,
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}, // x
                                              },
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}, // y
                                              },
                                              &compileInfo);
    AssertAcoshTiling(tilingContextPara, 8192, 128, 8192, 64);
}

// ===========================================================================
// 2) FP16 multi-core aligned — blockAlign=16, CeilAlign(128,16)=128
// ===========================================================================
TEST_F(AcoshTilingTest, test_tiling_fp16_multi_core_002)
{
    AcoshCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(OP_NAME,
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    AssertAcoshTiling(tilingContextPara, 8192, 128, 8192, 64);
}

// ===========================================================================
// 3) BF16 multi-core aligned — blockAlign=16
// ===========================================================================
TEST_F(AcoshTilingTest, test_tiling_bf16_multi_core_003)
{
    AcoshCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(OP_NAME,
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    AssertAcoshTiling(tilingContextPara, 8192, 128, 8192, 64);
}

// ===========================================================================
// 4) FP32 small + single-core + tail — {7} elem
//    blockFactor=CeilAlign(CeilDiv(7,64)=1, 8)=8, blockNum=CeilDiv(7,8)=1
// ===========================================================================
TEST_F(AcoshTilingTest, test_tiling_fp32_small_tail_004)
{
    AcoshCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(OP_NAME,
                                              {
                                                  {{{7}, {7}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{7}, {7}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    AssertAcoshTiling(tilingContextPara, 7, 8, 8192, 1);
}

// ===========================================================================
// 5) FP16 unaligned single-core — {17} elem; blockAlign=16
//    blockFactor=CeilAlign(1,16)=16, blockNum=CeilDiv(17,16)=2
// ===========================================================================
TEST_F(AcoshTilingTest, test_tiling_fp16_unalign_005)
{
    AcoshCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(OP_NAME,
                                              {
                                                  {{{17}, {17}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{17}, {17}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    AssertAcoshTiling(tilingContextPara, 17, 16, 8192, 2);
}

// ===========================================================================
// 6) FP32 large multi-core — {416910} elem
//    blockFactor=CeilAlign(CeilDiv(416910,64)=6515, 8)=6520, blockNum=CeilDiv(416910,6520)=64
// ===========================================================================
TEST_F(AcoshTilingTest, test_tiling_fp32_large_multi_core_006)
{
    AcoshCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(OP_NAME,
                                              {
                                                  {{{416910}, {416910}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{416910}, {416910}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    AssertAcoshTiling(tilingContextPara, 416910, 6520, 8192, 64);
}

// ===========================================================================
// 7) OI-14 回归锚点 FP16 — {3855} elem(T10 探针同 totalNum)
//    blockFactor=CeilAlign(CeilDiv(3855,64)=61, 16)=64, blockNum=CeilDiv(3855,64)=61
//    修复前(32 字节当 32 元素)CeilAlign(61,32)=96——回归即 FAIL
// ===========================================================================
TEST_F(AcoshTilingTest, test_tiling_fp16_oi14_anchor_007)
{
    AcoshCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(OP_NAME,
                                              {
                                                  {{{3855}, {3855}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3855}, {3855}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    AssertAcoshTiling(tilingContextPara, 3855, 64, 8192, 61);
}

// ===========================================================================
// 8) OI-14 回归锚点 FP32 — {3855} elem
//    blockFactor=CeilAlign(61, 8)=64(修复前 CeilAlign(61,32)=96), blockNum=61
// ===========================================================================
TEST_F(AcoshTilingTest, test_tiling_fp32_oi14_anchor_008)
{
    AcoshCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(OP_NAME,
                                              {
                                                  {{{3855}, {3855}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3855}, {3855}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    AssertAcoshTiling(tilingContextPara, 3855, 64, 8192, 61);
}

// ===========================================================================
// 9) Empty tensor FP32 — early-return branch: 字段全零, BlockDim=1
// ===========================================================================
TEST_F(AcoshTilingTest, test_tiling_empty_fp32_009)
{
    AcoshCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(OP_NAME,
                                              {
                                                  {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    AssertAcoshTiling(tilingContextPara, 0, 0, 0, 1);
}

// ===========================================================================
// 10) Empty tensor BF16 multi-dim — early-return
// ===========================================================================
TEST_F(AcoshTilingTest, test_tiling_empty_bf16_2d_010)
{
    AcoshCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(OP_NAME,
                                              {
                                                  {{{0, 8}, {0, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{0, 8}, {0, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    AssertAcoshTiling(tilingContextPara, 0, 0, 0, 1);
}

// ===========================================================================
// 11) FAIL: input rank>8 (9D) → GRAPH_FAILED
// ===========================================================================
TEST_F(AcoshTilingTest, test_tiling_fail_input_rank_011)
{
    AcoshCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        OP_NAME,
        {
            {{{2, 2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo)) << "input rank>8 should be rejected";
}

// ===========================================================================
// 12) FAIL: output rank>8 (input 2D, output 9D) → GRAPH_FAILED
// ===========================================================================
TEST_F(AcoshTilingTest, test_tiling_fail_output_rank_012)
{
    AcoshCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        OP_NAME,
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo)) << "output rank>8 should be rejected";
}

// ===========================================================================
// 13) FAIL: input/output shape mismatch → GRAPH_FAILED
// ===========================================================================
TEST_F(AcoshTilingTest, test_tiling_fail_shape_mismatch_013)
{
    AcoshCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(OP_NAME,
                                              {
                                                  {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo)) << "shape mismatch should be rejected";
}

// ===========================================================================
// 14) FAIL: unsupported dtype (DT_DOUBLE, AICPU 域 dtype 不入 arch35 tiling) → GRAPH_FAILED
// ===========================================================================
TEST_F(AcoshTilingTest, test_tiling_fail_unsupported_dtype_014)
{
    AcoshCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(OP_NAME,
                                              {
                                                  {{{8}, {8}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8}, {8}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo)) << "unsupported dtype should be rejected";
}

// ===========================================================================
// 15) FAIL: output dtype mismatch (x=FP32, y=FP16) → GRAPH_FAILED
// ===========================================================================
TEST_F(AcoshTilingTest, test_tiling_fail_output_dtype_mismatch_015)
{
    AcoshCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(OP_NAME,
                                              {
                                                  {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo)) << "output dtype mismatch should be rejected";
}

// ===========================================================================
// 16) FAIL: negative shape size (动态维 -1 乘积为负, {-1,4}→-4) → GRAPH_FAILED
// ===========================================================================
TEST_F(AcoshTilingTest, test_tiling_fail_negative_shape_size_016)
{
    AcoshCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(OP_NAME,
                                              {
                                                  {{{-1, 4}, {-1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{-1, 4}, {-1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo)) << "negative shape size should be rejected";
}

// ===========================================================================
// 17) FAIL: coreNum==0 (faker 平台参数) → GRAPH_FAILED
// ===========================================================================
TEST_F(AcoshTilingTest, test_tiling_fail_core_num_zero_017)
{
    AcoshCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(OP_NAME,
                                              {
                                                  {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo,
                                              0, // coreNum=0
                                              262144);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo)) << "coreNum==0 should be rejected";
}

// ===========================================================================
// 18) FAIL: ubSize==0 (faker 平台参数) → GRAPH_FAILED
// ===========================================================================
TEST_F(AcoshTilingTest, test_tiling_fail_ub_size_zero_018)
{
    AcoshCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(OP_NAME,
                                              {
                                                  {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo,
                                              64, // coreNum
                                              0); // ubSize=0
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo)) << "ubSize==0 should be rejected";
}

// ===========================================================================
// 19) FAIL: ubFactor 过小 (ubSize=32 → FloorAlign(FloorDiv(32,32)=1, 8)=0) → GRAPH_FAILED
// ===========================================================================
TEST_F(AcoshTilingTest, test_tiling_fail_ub_factor_too_small_019)
{
    AcoshCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(OP_NAME,
                                              {
                                                  {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo,
                                              64,  // coreNum
                                              32); // ubSize=32
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo)) << "ubFactor too small should be rejected";
}
