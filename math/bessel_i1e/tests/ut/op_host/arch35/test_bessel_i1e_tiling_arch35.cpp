/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0(the "License").
 * Please refer to the License for details.You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_bessel_i1e_tiling_arch35.cpp
 * \brief Tiling UT for BesselI1e (arch35), running the delivered tiling function
 *        through the faker framework (coreNum=64, ubSize=262144, UB block 32B).
 *
 * Real tiling contract under the faker platform parameters:
 *   totalNum    = numel(x) (rank-0 treated as 1 element)
 *   blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), 8)
 *   ubFactor    = FloorAlign(min(FloorDiv((ubSize - 8192) / 4, bufferNum),
 *                                FloorAlign(65535 / 4, 8)), 8)
 *                with bufferNum = 12 (fp32) / 13 (fp16, bf16)
 *   => ubFactor(fp32) = FloorAlign(5290, 8) = 5288
 *      ubFactor(fp16/bf16) = FloorAlign(4883, 8) = 4880
 */

#include <gtest/gtest.h>
#include <iostream>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_kernel/arch35/bessel_i1e_tiling_data.h"

namespace BesselI1eUT {
using namespace std;
using namespace ge;

static const std::string OP_NAME = "BesselI1e";

// BesselI1eCompileInfo is an empty struct declared inside the tiling source;
// the tiling function reads coreNum/ubSize from the platform info faked below.
struct BesselI1eCompileInfo {};

class BesselI1eTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BesselI1eTilingTest SetUp." << std::endl; }

    static void TearDownTestCase() { std::cout << "BesselI1eTilingTest TearDown." << std::endl; }
};

TEST_F(BesselI1eTilingTest, tiling_fp32_basic)
{
    BesselI1eCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(OP_NAME,
                                              {
                                                  {{{2048}, {2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2048}, {2048}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo, 64, 262144, 4096);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo)) << "Tiling should succeed for FP32 [2048]";

    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0) << "Workspace should be 0 for elementwise op";

    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(BesselI1eTilingData));
    auto* tilingData = reinterpret_cast<const BesselI1eTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->totalNum, 2048) << "totalNum should equal total elements";
    EXPECT_EQ(tilingData->blockFactor, 32) << "CeilAlign(CeilDiv(2048,64),8) = 32";
    EXPECT_EQ(tilingData->ubFactor, 5288) << "FloorAlign(FloorDiv(63488,12),8) = 5288";
    EXPECT_EQ(tilingData->blockFactor % 8, 0) << "blockFactor must be 8-element aligned";
    EXPECT_EQ(tilingData->ubFactor % 8, 0) << "ubFactor must be 8-element aligned";
    EXPECT_EQ(tilingInfo.blockNum, 64) << "CeilDiv(2048,32) = 64 cores";
}

TEST_F(BesselI1eTilingTest, tiling_fp16_basic)
{
    BesselI1eCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(OP_NAME,
                                              {
                                                  {{{4096}, {4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4096}, {4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              &compileInfo, 64, 262144, 4096);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo)) << "Tiling should succeed for FP16 [4096]";

    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);

    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(BesselI1eTilingData));
    auto* tilingData = reinterpret_cast<const BesselI1eTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->totalNum, 4096);
    EXPECT_EQ(tilingData->blockFactor, 64) << "CeilAlign(CeilDiv(4096,64),8) = 64";
    EXPECT_EQ(tilingData->ubFactor, 4880) << "FloorAlign(FloorDiv(63488,13),8) = 4880";
    EXPECT_EQ(tilingInfo.blockNum, 64) << "CeilDiv(4096,64) = 64 cores";
}

TEST_F(BesselI1eTilingTest, tiling_bf16_basic)
{
    BesselI1eCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(OP_NAME,
                                              {
                                                  {{{8192}, {8192}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8192}, {8192}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              &compileInfo, 64, 262144, 4096);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo)) << "Tiling should succeed for BF16 [8192]";

    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);

    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(BesselI1eTilingData));
    auto* tilingData = reinterpret_cast<const BesselI1eTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->totalNum, 8192);
    EXPECT_EQ(tilingData->blockFactor, 128) << "CeilAlign(CeilDiv(8192,64),8) = 128";
    EXPECT_EQ(tilingData->ubFactor, 4880) << "BF16 shares the FP16 buffer budget (13 shares)";
    EXPECT_EQ(tilingInfo.blockNum, 64) << "CeilDiv(8192,128) = 64 cores";
}

TEST_F(BesselI1eTilingTest, tiling_empty_tensor)
{
    BesselI1eCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(OP_NAME,
                                              {
                                                  {{{0, 3}, {0, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{0, 3}, {0, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo, 64, 262144, 4096);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo)) << "Tiling should succeed for empty tensor";

    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);

    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(BesselI1eTilingData));
    auto* tilingData = reinterpret_cast<const BesselI1eTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->totalNum, 0) << "totalNum should be 0 for empty tensor";
    EXPECT_EQ(tilingData->blockFactor, 0);
    EXPECT_EQ(tilingData->ubFactor, 0);
    EXPECT_EQ(tilingInfo.blockNum, 1) << "Empty tensor runs on a single block";
}

TEST_F(BesselI1eTilingTest, tiling_rank0_scalar)
{
    BesselI1eCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(OP_NAME,
                                              {
                                                  {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo, 64, 262144, 4096);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo)) << "Tiling should succeed for rank-0 scalar";

    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);

    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(BesselI1eTilingData));
    auto* tilingData = reinterpret_cast<const BesselI1eTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->totalNum, 1) << "Scalar is normalized to 1 element";
    EXPECT_EQ(tilingData->blockFactor, 8) << "CeilAlign(CeilDiv(1,64),8) = 8";
    EXPECT_GT(tilingData->ubFactor, 0);
    EXPECT_EQ(tilingInfo.blockNum, 1) << "CeilDiv(1,8) = 1 core";
}

TEST_F(BesselI1eTilingTest, tiling_fp32_large)
{
    BesselI1eCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(OP_NAME,
                                              {
                                                  {{{100000}, {100000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{100000}, {100000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo, 64, 262144, 4096);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo)) << "Tiling should succeed for FP32 [100000]";

    auto* tilingData = reinterpret_cast<const BesselI1eTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->totalNum, 100000);
    EXPECT_EQ(tilingData->blockFactor, 1568) << "CeilAlign(CeilDiv(100000,64),8) = 1568";
    EXPECT_EQ(tilingData->ubFactor, 5288);
    EXPECT_EQ(tilingInfo.blockNum, 64) << "CeilDiv(100000,1568) = 64 cores";
}

TEST_F(BesselI1eTilingTest, tiling_fp32_multidim)
{
    BesselI1eCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(OP_NAME,
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo, 64, 262144, 4096);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo)) << "Tiling should succeed for 4-D input";

    auto* tilingData = reinterpret_cast<const BesselI1eTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->totalNum, 8192) << "1*64*2*64 = 8192";
    EXPECT_EQ(tilingData->blockFactor, 128);
    EXPECT_EQ(tilingData->ubFactor, 5288);
}

TEST_F(BesselI1eTilingTest, tiling_dtype_buffer_budget)
{
    // FP32 uses 12 UB shares while FP16/BF16 use 13, so the fp32 ubFactor is
    // the larger one; blockFactor depends only on totalNum/coreNum.
    BesselI1eCompileInfo compileInfo;

    gert::TilingContextPara fp32Para(OP_NAME,
                                     {
                                         {{{8192}, {8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{8192}, {8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     &compileInfo, 64, 262144, 4096);
    gert::TilingContextPara fp16Para(OP_NAME,
                                     {
                                         {{{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{8192}, {8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     },
                                     &compileInfo, 64, 262144, 4096);
    gert::TilingContextPara bf16Para(OP_NAME,
                                     {
                                         {{{8192}, {8192}}, ge::DT_BF16, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{8192}, {8192}}, ge::DT_BF16, ge::FORMAT_ND},
                                     },
                                     &compileInfo, 64, 262144, 4096);

    TilingInfo fp32Info, fp16Info, bf16Info;
    ASSERT_TRUE(ExecuteTiling(fp32Para, fp32Info));
    ASSERT_TRUE(ExecuteTiling(fp16Para, fp16Info));
    ASSERT_TRUE(ExecuteTiling(bf16Para, bf16Info));

    auto* fp32Data = reinterpret_cast<const BesselI1eTilingData*>(fp32Info.tilingData.get());
    auto* fp16Data = reinterpret_cast<const BesselI1eTilingData*>(fp16Info.tilingData.get());
    auto* bf16Data = reinterpret_cast<const BesselI1eTilingData*>(bf16Info.tilingData.get());

    EXPECT_EQ(fp32Data->totalNum, 8192);
    EXPECT_EQ(fp16Data->totalNum, 8192);
    EXPECT_EQ(bf16Data->totalNum, 8192);

    EXPECT_EQ(fp32Data->blockFactor, fp16Data->blockFactor);
    EXPECT_EQ(fp16Data->blockFactor, bf16Data->blockFactor);

    EXPECT_EQ(fp16Data->ubFactor, bf16Data->ubFactor) << "FP16 and BF16 share the 13-share budget";
    EXPECT_GT(fp32Data->ubFactor, fp16Data->ubFactor) << "FP32 has the 12-share budget (larger ubFactor)";

    // The tiling function never calls SetTilingKey, so the reported key is the
    // framework default, which differs across CANN versions (0 on 9.0.0, a fixed
    // non-zero value on 9.3.0). The testable contract is that dtype is not
    // encoded into the key: all dtypes of the same op report the same value.
    EXPECT_EQ(fp32Info.tilingKey, fp16Info.tilingKey) << "dtype must not be encoded in the tiling key";
    EXPECT_EQ(fp16Info.tilingKey, bf16Info.tilingKey) << "dtype must not be encoded in the tiling key";
}

TEST_F(BesselI1eTilingTest, tiling_unsupported_dtype_double)
{
    BesselI1eCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(OP_NAME,
                                              {
                                                  {{{1024}, {1024}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024}, {1024}}, ge::DT_DOUBLE, ge::FORMAT_ND},
                                              },
                                              &compileInfo, 64, 262144, 4096);

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo)) << "Tiling should fail for DT_DOUBLE";
}

TEST_F(BesselI1eTilingTest, tiling_unsupported_dtype_int32)
{
    BesselI1eCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(OP_NAME,
                                              {
                                                  {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024}, {1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              &compileInfo, 64, 262144, 4096);

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo)) << "Tiling should fail for DT_INT32";
}

TEST_F(BesselI1eTilingTest, tiling_rank_exceeds_8)
{
    BesselI1eCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        OP_NAME,
        {
            {{{1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo, 64, 262144, 4096);

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo)) << "Tiling should fail for rank 9 (> 8)";
}

TEST_F(BesselI1eTilingTest, tiling_unsupported_format)
{
    for (const ge::Format format : {ge::FORMAT_FRACTAL_Z, ge::FORMAT_NC1HWC0, ge::FORMAT_NCHW, ge::FORMAT_NHWC}) {
        BesselI1eCompileInfo compileInfo;
        gert::TilingContextPara tilingContextPara(OP_NAME,
                                                  {
                                                      {{{1024}, {1024}}, ge::DT_FLOAT, format},
                                                  },
                                                  {
                                                      {{{1024}, {1024}}, ge::DT_FLOAT, format},
                                                  },
                                                  &compileInfo, 64, 262144, 4096);
        TilingInfo tilingInfo;
        EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo))
            << "Tiling should fail for unsupported format " << static_cast<int32_t>(format);
    }
}

TEST_F(BesselI1eTilingTest, tiling_input_output_size_mismatch)
{
    BesselI1eCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(OP_NAME,
                                              {
                                                  {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo, 64, 262144, 4096);

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo))
        << "Tiling should fail when input and output element counts differ";
}

} // namespace BesselI1eUT
