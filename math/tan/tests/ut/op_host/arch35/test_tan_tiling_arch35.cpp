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
 * \file test_tan_tiling_arch35.cpp
 * \brief Tan Tiling UT (arch35 / Ascend950)
 *
 * Covers paths in op_host/arch35/tan_tiling_arch35.cpp:
 *   1) Dtype paths: FP16 / FP32 / BF16 / INT32 map to dataTypeId 0/1/2/3 and
 *      TilingKey = GET_TPL_TILING_KEY(TAN_TPL_SCH_MODE_{0..3}) (schMode is the only
 *      declared template argument, 2 bits, UI_LIST {0,1,2,3} — key == schMode value).
 *   2) 核切分: perCoreElements = THREAD_NUM * MIN_ELEMENTS_PER_THREAD = 2048;
 *      needCoreNum = CeilDiv(totalElements, 2048) 且钳制到 coreNum(faker=64)。
 *   3) 空 Tensor 守卫: totalElements==0 时 blockDim=1
 *      (避免 SetBlockDim(0) 被框架以 INVALID_TILING 拒绝)。
 *   4) 累乘上界保护: shape 维度累乘溢出 int64 时返回 GRAPH_FAILED;
 *      单维 INT64_MAX 边界(乘积恰不溢出)正常通过且走 int64 全链钳制。
 *   5) 失败路径(ExecuteTiling 返回 false):
 *        - 不支持 dtype(DT_DOUBLE)          → GRAPH_FAILED
 *        - coreNum==0(GetPlatformInfo 校验) → GRAPH_FAILED
 *
 * Tan tiling 调用 SetTilingKey(faker 环境下 TilingInfo.tilingKey 可读),
 * 采用 ExecuteTiling + TilingData 字段直断言(Acosh/bessel_i0e 同款先例)。
 *
 * Platform constants (UT faker): coreNum=64。
 *
 * Formulas (faker 平台):
 *   perCoreElements = 512 * 4 = 2048
 *   needCoreNum     = CeilDiv(totalElements, 2048), clamped to [0, 64]
 *   blockDim        = (totalElements == 0) ? 1 : needCoreNum
 */

#include <cstdint>
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_kernel/arch35/tan_tiling_data.h"
#include "../../../../op_kernel/arch35/tan_tiling_key.h"

namespace TanUT {
using namespace std;
using namespace ge;

static const std::string OP_NAME = "Tan";

struct TanCompileInfo {};
} // namespace TanUT

using namespace TanUT;

// faker 平台核数与核切分常量(op_host/arch35/tan_tiling_arch35.cpp)
static constexpr int64_t TAN_UT_CORE_NUM = 64;         // faker coreNum
static constexpr int64_t TAN_UT_PER_CORE_ELEMS = 2048; // THREAD_NUM(512) * MIN_ELEMENTS_PER_THREAD(4)

class TanTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TanTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TanTilingTest TearDown" << std::endl; }
};

// 成功用例公共断言: workspace={0} + tiling data 字段 + blockDim(blockNum) + tilingKey
static void AssertTanTiling(const gert::TilingContextPara& para, int64_t expectTotalElements, int32_t expectNeedCoreNum,
                            int32_t expectDataTypeId, uint64_t expectTilingKey)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo)) << "Tan tiling should succeed";

    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0) << "workspace[0] should be WS_SYS_SIZE(0)";

    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(TanTilingData));
    auto* tilingData = reinterpret_cast<const TanTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->totalElements, expectTotalElements) << "totalElements mismatch";
    EXPECT_EQ(tilingData->perCoreElements, TAN_UT_PER_CORE_ELEMS) << "perCoreElements mismatch";
    EXPECT_EQ(tilingData->needCoreNum, expectNeedCoreNum) << "needCoreNum mismatch";
    EXPECT_EQ(tilingData->dataType, expectDataTypeId) << "dataType id mismatch";

    // 空 Tensor 守卫: totalElements==0 时 blockDim=1; 否则 blockDim=needCoreNum
    const size_t expectBlockDim = (expectTotalElements == 0) ? 1 : static_cast<size_t>(expectNeedCoreNum);
    EXPECT_EQ(tilingInfo.blockNum, expectBlockDim) << "blockDim mismatch";

    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey) << "tilingKey mismatch";
}

// ===========================================================================
// 1) FP32 multi-core — {1,64,2,64}=8192 elem: needCoreNum=4, dataTypeId=1
//    TilingKey = GET_TPL_TILING_KEY(TAN_TPL_SCH_MODE_1)
// ===========================================================================
TEST_F(TanTilingTest, test_tiling_fp32_multi_core_001)
{
    TanCompileInfo compileInfo;
    gert::TilingContextPara para(OP_NAME,
                                 {
                                     {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}, // x
                                 },
                                 {
                                     {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}, // y
                                 },
                                 &compileInfo);
    AssertTanTiling(para, 8192, 4, 1, GET_TPL_TILING_KEY(TAN_TPL_SCH_MODE_1));
}

// ===========================================================================
// 2) FP16 — same shape: dataTypeId=0, TilingKey = GET_TPL_TILING_KEY(TAN_TPL_SCH_MODE_0)
// ===========================================================================
TEST_F(TanTilingTest, test_tiling_fp16_multi_core_002)
{
    TanCompileInfo compileInfo;
    gert::TilingContextPara para(OP_NAME,
                                 {
                                     {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    AssertTanTiling(para, 8192, 4, 0, GET_TPL_TILING_KEY(TAN_TPL_SCH_MODE_0));
}

// ===========================================================================
// 3) BF16 — dataTypeId=2, TilingKey = GET_TPL_TILING_KEY(TAN_TPL_SCH_MODE_2)
// ===========================================================================
TEST_F(TanTilingTest, test_tiling_bf16_multi_core_003)
{
    TanCompileInfo compileInfo;
    gert::TilingContextPara para(OP_NAME,
                                 {
                                     {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    AssertTanTiling(para, 8192, 4, 2, GET_TPL_TILING_KEY(TAN_TPL_SCH_MODE_2));
}

// ===========================================================================
// 4) INT32 — dataTypeId=3, TilingKey = GET_TPL_TILING_KEY(TAN_TPL_SCH_MODE_3)
//    (int32 入出同 dtype 直通契约的 tiling 侧 dtype 分派)
// ===========================================================================
TEST_F(TanTilingTest, test_tiling_int32_multi_core_004)
{
    TanCompileInfo compileInfo;
    gert::TilingContextPara para(OP_NAME,
                                 {
                                     {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    AssertTanTiling(para, 8192, 4, 3, GET_TPL_TILING_KEY(TAN_TPL_SCH_MODE_3));
}

// ===========================================================================
// 5) FP32 small tail — {7} elem: needCoreNum=1, blockDim=1
// ===========================================================================
TEST_F(TanTilingTest, test_tiling_fp32_small_tail_005)
{
    TanCompileInfo compileInfo;
    gert::TilingContextPara para(OP_NAME,
                                 {
                                     {{{7}, {7}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{7}, {7}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    AssertTanTiling(para, 7, 1, 1, GET_TPL_TILING_KEY(TAN_TPL_SCH_MODE_1));
}

// ===========================================================================
// 6) FP32 exact multiple — {2048} elem: needCoreNum=1(整除无尾块)
// ===========================================================================
TEST_F(TanTilingTest, test_tiling_fp32_exact_multiple_006)
{
    TanCompileInfo compileInfo;
    gert::TilingContextPara para(OP_NAME,
                                 {
                                     {{{2048}, {2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{2048}, {2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    AssertTanTiling(para, 2048, 1, 1, GET_TPL_TILING_KEY(TAN_TPL_SCH_MODE_1));
}

// ===========================================================================
// 7) FP32 large multi-core clamp — {204800} elem: CeilDiv=100 > coreNum(64) → 钳制 64
//    (钳制先行: int64 全链比较后再收敛 int32)
// ===========================================================================
TEST_F(TanTilingTest, test_tiling_fp32_large_clamp_007)
{
    TanCompileInfo compileInfo;
    gert::TilingContextPara para(OP_NAME,
                                 {
                                     {{{204800}, {204800}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{204800}, {204800}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    AssertTanTiling(para, 204800, 64, 1, GET_TPL_TILING_KEY(TAN_TPL_SCH_MODE_1));
}

// ===========================================================================
// 8) FP32 2D product — {1024,2}=2048 elem: 多维累乘正确性
// ===========================================================================
TEST_F(TanTilingTest, test_tiling_fp32_2d_product_008)
{
    TanCompileInfo compileInfo;
    gert::TilingContextPara para(OP_NAME,
                                 {
                                     {{{1024, 2}, {1024, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1024, 2}, {1024, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    AssertTanTiling(para, 2048, 1, 1, GET_TPL_TILING_KEY(TAN_TPL_SCH_MODE_1));
}

// ===========================================================================
// 9) FP32 8维常规点 — {2,2,2,2,2,2,2,2}=256 elem: tiling 全维数族采样
// ===========================================================================
TEST_F(TanTilingTest, test_tiling_fp32_rank8_009)
{
    TanCompileInfo compileInfo;
    gert::TilingContextPara para(
        OP_NAME,
        {
            {{{2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    AssertTanTiling(para, 256, 1, 1, GET_TPL_TILING_KEY(TAN_TPL_SCH_MODE_1));
}

// ===========================================================================
// 10) Empty tensor FP32 — {0}: totalElements==0 → blockDim=1(空 Tensor 守卫,
//     避免 SetBlockDim(0) 被框架以 INVALID_TILING 拒绝); tiling 数据按 dtype
//     正常设置(totalElements=0, perCoreElements=2048, needCoreNum=0, dataTypeId=1)
// ===========================================================================
TEST_F(TanTilingTest, test_tiling_empty_fp32_010)
{
    TanCompileInfo compileInfo;
    gert::TilingContextPara para(OP_NAME,
                                 {
                                     {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    AssertTanTiling(para, 0, 0, 1, GET_TPL_TILING_KEY(TAN_TPL_SCH_MODE_1));
}

// ===========================================================================
// 11) Empty tensor FP16 multi-dim — {0,8}: 空轴在首位, 其后维乘 0 仍为 0 → blockDim=1
// ===========================================================================
TEST_F(TanTilingTest, test_tiling_empty_fp16_2d_011)
{
    TanCompileInfo compileInfo;
    gert::TilingContextPara para(OP_NAME,
                                 {
                                     {{{0, 8}, {0, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{0, 8}, {0, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    AssertTanTiling(para, 0, 0, 0, GET_TPL_TILING_KEY(TAN_TPL_SCH_MODE_0));
}

// ===========================================================================
// 12) FAIL: 累乘上界保护 — {3037000500, 3037000500} 乘积 9.22e18
//     超 INT64_MAX: 第二维处 totalElements(3037000500) > INT64_MAX/dim(3037000499) →
//     GRAPH_FAILED(溢出前置拒绝, 防止回绕为负值破坏核切分)
// ===========================================================================
TEST_F(TanTilingTest, test_tiling_fail_shape_product_overflow_012)
{
    TanCompileInfo compileInfo;
    gert::TilingContextPara para(
        OP_NAME,
        {
            {{{3037000500, 3037000500}, {3037000500, 3037000500}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{3037000500, 3037000500}, {3037000500, 3037000500}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo)) << "shape product overflow int64 should be rejected";
}

// ===========================================================================
// 13) 溢出守卫边界通过 — {INT64_MAX} 单维: 乘积恰为 INT64_MAX 不溢出(守卫不触发);
//     需核数 CeilDiv(INT64_MAX, 2048) 钳制到 64(int64 全链窄化路径)
// ===========================================================================
TEST_F(TanTilingTest, test_tiling_int64_max_single_dim_013)
{
    TanCompileInfo compileInfo;
    gert::TilingContextPara para(OP_NAME,
                                 {
                                     {{{INT64_MAX}, {INT64_MAX}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{INT64_MAX}, {INT64_MAX}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    AssertTanTiling(para, INT64_MAX, 64, 1, GET_TPL_TILING_KEY(TAN_TPL_SCH_MODE_1));
}

// ===========================================================================
// 14) FAIL: 不支持 dtype(DT_DOUBLE, AICPU 域 dtype 不入 arch35 tiling 分派) → GRAPH_FAILED
// ===========================================================================
TEST_F(TanTilingTest, test_tiling_fail_unsupported_dtype_014)
{
    TanCompileInfo compileInfo;
    gert::TilingContextPara para(OP_NAME,
                                 {
                                     {{{8}, {8}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{8}, {8}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo)) << "unsupported dtype should be rejected";
}

// ===========================================================================
// 15) FAIL: coreNum==0(faker 平台参数注入, GetPlatformInfo 校验) → GRAPH_FAILED
// ===========================================================================
TEST_F(TanTilingTest, test_tiling_fail_core_num_zero_015)
{
    TanCompileInfo compileInfo;
    gert::TilingContextPara para(OP_NAME,
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
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo)) << "coreNum==0 should be rejected";
}

// ===========================================================================
// 16) FAIL: 不支持 dtype(DT_COMPLEX64) → GRAPH_FAILED(dtype else 分支双 dtype 取样)
// ===========================================================================
TEST_F(TanTilingTest, test_tiling_fail_unsupported_dtype_c64_016)
{
    TanCompileInfo compileInfo;
    gert::TilingContextPara para(OP_NAME,
                                 {
                                     {{{8}, {8}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{8}, {8}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo)) << "complex64 dtype should be rejected";
}

// 参数化核数对账(钳制分支双向): 204800 elem 在 coreNum=100 下不钳制(needCoreNum=100)
TEST_F(TanTilingTest, test_tiling_fp32_no_clamp_when_core_sufficient_017)
{
    TanCompileInfo compileInfo;
    gert::TilingContextPara para(OP_NAME,
                                 {
                                     {{{204800}, {204800}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{204800}, {204800}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo,
                                 100, // coreNum=100 ≥ CeilDiv(204800, 2048)=100 → 恰不钳制
                                 262144);
    AssertTanTiling(para, 204800, 100, 1, GET_TPL_TILING_KEY(TAN_TPL_SCH_MODE_1));
}
