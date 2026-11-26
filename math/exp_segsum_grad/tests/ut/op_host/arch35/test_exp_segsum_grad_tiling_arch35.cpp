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
 * \file test_exp_segsum_grad_tiling_arch35.cpp
 * \brief ExpSegsumGrad tiling UT for arch35 / Ascend950.
 */

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "platform/platform_info.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/exp_segsum_grad_tiling_arch35.h"
#include "../../../../op_kernel/arch35/exp_segsum_grad_tiling_data.h"

using namespace ge;
using namespace gert;
using optiling::ExpSegsumGradCompileInfoArch35;

#define EXP_SEGSUM_GRAD_ARCH35_STR_IMPL(x) #x
#define EXP_SEGSUM_GRAD_ARCH35_STR(x) EXP_SEGSUM_GRAD_ARCH35_STR_IMPL(x)

namespace {
constexpr uint64_t EXP_SEGSUM_GRAD_ARCH35_TILING_KEY_SMALL = 1;
constexpr uint64_t EXP_SEGSUM_GRAD_ARCH35_TILING_KEY_SHORT = 2;
constexpr size_t EXP_SEGSUM_GRAD_A2_WORKSPACE_SIZE = 32 * 1024 * 1024;
constexpr size_t EXP_SEGSUM_GRAD_ARCH35_SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr int64_t EXP_SEGSUM_GRAD_ARCH35_FP32_SLIDE_SIZE = 10912;

bool IsAscend950Build() { return std::string(EXP_SEGSUM_GRAD_ARCH35_STR(BUILD_SOC_VERSION)) == "ascend950"; }

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (auto dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

gert::TilingContextPara MakeTilingPara(const std::vector<int64_t>& dims, ge::DataType dtype, void* compileInfo)
{
    auto shape = MakeStorageShape(dims);
    return gert::TilingContextPara("ExpSegsumGrad",
                                   {
                                       {shape, dtype, ge::FORMAT_ND},
                                       {shape, dtype, ge::FORMAT_ND},
                                   },
                                   {
                                       {shape, dtype, ge::FORMAT_ND},
                                   },
                                   compileInfo);
}

void ExpectBatchBounds(const ExpSegsumGradTilingDataArch35& tiling, int64_t batches, uint32_t expectNeedCoreNum)
{
    ASSERT_EQ(tiling.needCoreNum, expectNeedCoreNum);
    if (batches == 0) {
        for (uint16_t i = 0; i < EXP_SEGSUM_GRAD_MAX_CORE_ARCH35; ++i) {
            EXPECT_EQ(tiling.batchStart[i], 0);
            EXPECT_EQ(tiling.batchEnd[i], 0);
        }
        return;
    }

    auto averageBatches = (batches + EXP_SEGSUM_GRAD_MAX_CORE_ARCH35 - 1) / EXP_SEGSUM_GRAD_MAX_CORE_ARCH35;
    for (uint32_t i = 0; i < expectNeedCoreNum; ++i) {
        EXPECT_EQ(tiling.batchStart[i], static_cast<int32_t>(i * averageBatches));
        EXPECT_EQ(tiling.batchEnd[i], static_cast<int32_t>(std::min<int64_t>((i + 1) * averageBatches, batches)));
    }
    for (uint32_t i = expectNeedCoreNum; i < EXP_SEGSUM_GRAD_MAX_CORE_ARCH35; ++i) {
        EXPECT_EQ(tiling.batchStart[i], 0);
        EXPECT_EQ(tiling.batchEnd[i], 0);
    }
}

const ExpSegsumGradTilingDataArch35* GetArch35TilingData(const TilingInfo& tilingInfo)
{
    EXPECT_GE(tilingInfo.tilingDataSize, sizeof(ExpSegsumGradTilingDataArch35));
    return reinterpret_cast<const ExpSegsumGradTilingDataArch35*>(tilingInfo.tilingData.get());
}
} // namespace

class ExpSegsumGradTilingArch35Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        fe::OptionalInfos optiCompilation;
        optiCompilation.Init();
        optiCompilation.SetSocVersion("Ascend950");
        fe::PlatformInfoManager::GeInstance().SetOptionalCompilationInfo(optiCompilation);
    }

    static void TearDownTestCase()
    {
        fe::OptionalInfos optiCompilation;
        optiCompilation.Init();
        optiCompilation.SetSocVersion("soc_version");
        fe::PlatformInfoManager::GeInstance().SetOptionalCompilationInfo(optiCompilation);
    }

    void SetUp() override
    {
        if (!IsAscend950Build()) {
            GTEST_SKIP() << "ExpSegsumGrad arch35 tiling UT only runs in ascend950 build.";
        }
    }
};

TEST_F(ExpSegsumGradTilingArch35Test, small_fp32_uses_arch35_tiling_data)
{
    ExpSegsumGradCompileInfoArch35 compileInfo = {};
    auto tilingContextPara = MakeTilingPara({1, 1, 12, 12}, ge::DT_FLOAT, &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));

    EXPECT_EQ(tilingInfo.tilingKey, EXP_SEGSUM_GRAD_ARCH35_TILING_KEY_SMALL);
    EXPECT_EQ(tilingInfo.blockNum, 1);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0],
              EXP_SEGSUM_GRAD_A2_WORKSPACE_SIZE + EXP_SEGSUM_GRAD_ARCH35_SYS_WORKSPACE_SIZE);

    auto tiling = GetArch35TilingData(tilingInfo);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->batches, 1);
    EXPECT_EQ(tiling->tailDimLength, 12);
    EXPECT_EQ(tiling->slideSize, EXP_SEGSUM_GRAD_ARCH35_FP32_SLIDE_SIZE);
    ExpectBatchBounds(*tiling, 1, 1);
}

TEST_F(ExpSegsumGradTilingArch35Test, multi_batch_splits_across_ascend950_cores)
{
    ExpSegsumGradCompileInfoArch35 compileInfo = {};
    auto tilingContextPara = MakeTilingPara({65, 1, 12, 12}, ge::DT_FLOAT, &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));

    EXPECT_EQ(tilingInfo.tilingKey, EXP_SEGSUM_GRAD_ARCH35_TILING_KEY_SMALL);
    EXPECT_EQ(tilingInfo.blockNum, 33);

    auto tiling = GetArch35TilingData(tilingInfo);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->batches, 65);
    EXPECT_EQ(tiling->tailDimLength, 12);
    EXPECT_EQ(tiling->slideSize, EXP_SEGSUM_GRAD_ARCH35_FP32_SLIDE_SIZE);
    ExpectBatchBounds(*tiling, 65, 33);
}

TEST_F(ExpSegsumGradTilingArch35Test, empty_batch_keeps_valid_block_dim)
{
    ExpSegsumGradCompileInfoArch35 compileInfo = {};
    auto tilingContextPara = MakeTilingPara({0, 1, 12, 12}, ge::DT_FLOAT, &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));

    EXPECT_EQ(tilingInfo.tilingKey, EXP_SEGSUM_GRAD_ARCH35_TILING_KEY_SMALL);
    EXPECT_EQ(tilingInfo.blockNum, 1);

    auto tiling = GetArch35TilingData(tilingInfo);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->batches, 0);
    EXPECT_EQ(tiling->tailDimLength, 12);
    EXPECT_EQ(tiling->slideSize, 0);
    ExpectBatchBounds(*tiling, 0, 1);
}

namespace {
void ExpectAlignedBatchBounds(const ExpSegsumGradTilingDataArch35& tiling, int64_t batches, int64_t length,
                              int64_t dtypeSize, int64_t expectedCores)
{
    ASSERT_EQ(tiling.needCoreNum, expectedCores);
    int64_t next = 0;
    for (int64_t core = 0; core < tiling.needCoreNum; ++core) {
        EXPECT_EQ(tiling.batchStart[core], next);
        EXPECT_EQ(tiling.batchStart[core] * length * dtypeSize % 32, 0);
        EXPECT_GT(tiling.batchEnd[core], tiling.batchStart[core]);
        next = tiling.batchEnd[core];
        if (core + 1 < tiling.needCoreNum) {
            EXPECT_EQ(next * length * dtypeSize % 32, 0);
        }
    }
    EXPECT_EQ(next, batches);
    for (int64_t core = tiling.needCoreNum; core < EXP_SEGSUM_GRAD_MAX_CORE_ARCH35; ++core) {
        EXPECT_EQ(tiling.batchStart[core], 0);
        EXPECT_EQ(tiling.batchEnd[core], 0);
    }
}
} // namespace

TEST_F(ExpSegsumGradTilingArch35Test, short_fp32_does_not_round_each_core_to_64_batches)
{
    ExpSegsumGradCompileInfoArch35 compileInfo = {};
    auto para = MakeTilingPara({4097, 1, 16, 16}, ge::DT_FLOAT, &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, EXP_SEGSUM_GRAD_ARCH35_TILING_KEY_SHORT);
    EXPECT_EQ(info.blockNum, 64);
    auto tiling = GetArch35TilingData(info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_GT(tiling->slideSize, 0);
    ExpectAlignedBatchBounds(*tiling, 4097, 16, 4, 64);
}

TEST_F(ExpSegsumGradTilingArch35Test, short_fp32_length16_fits_192_kib_ub)
{
    ExpSegsumGradCompileInfoArch35 compileInfo = {};
    auto para = MakeTilingPara({4097, 1, 16, 16}, ge::DT_FLOAT, &compileInfo);
    para.ubSize_ = 192 * 1024;
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, EXP_SEGSUM_GRAD_ARCH35_TILING_KEY_SHORT);
    auto tiling = GetArch35TilingData(info);
    ASSERT_NE(tiling, nullptr);
    // For FP32 AR with 16 columns on Ascend 950, ReduceSum needs zero temporary bytes.
    // The two inputs, prefix and output use 3136 bytes per batch: 62 fit in 192 KiB, 63 do not.
    EXPECT_EQ(tiling->slideSize, 62);
    ExpectAlignedBatchBounds(*tiling, 4097, 16, 4, 64);
}

TEST_F(ExpSegsumGradTilingArch35Test, short_fp32_odd_length_uses_eight_batch_alignment)
{
    ExpSegsumGradCompileInfoArch35 compileInfo = {};
    auto para = MakeTilingPara({4097, 1, 3, 3}, ge::DT_FLOAT, &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, EXP_SEGSUM_GRAD_ARCH35_TILING_KEY_SHORT);
    auto tiling = GetArch35TilingData(info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_GT(tiling->slideSize, 0);
    EXPECT_EQ(tiling->slideSize % 8, 0);
    ExpectAlignedBatchBounds(*tiling, 4097, 3, 4, 57);
}

TEST_F(ExpSegsumGradTilingArch35Test, short_bf16_odd_length_uses_sixteen_batch_alignment)
{
    ExpSegsumGradCompileInfoArch35 compileInfo = {};
    auto para = MakeTilingPara({4097, 1, 3, 3}, ge::DT_BF16, &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, EXP_SEGSUM_GRAD_ARCH35_TILING_KEY_SHORT);
    auto tiling = GetArch35TilingData(info);
    ASSERT_NE(tiling, nullptr);
    EXPECT_GT(tiling->slideSize, 0);
    EXPECT_EQ(tiling->slideSize % 16, 0);
    ExpectAlignedBatchBounds(*tiling, 4097, 3, 2, 52);
}

TEST_F(ExpSegsumGradTilingArch35Test, short_path_starts_at_64_batches_per_core)
{
    ExpSegsumGradCompileInfoArch35 compileInfo = {};
    auto para = MakeTilingPara({4096, 1, 16, 16}, ge::DT_FLOAT, &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, EXP_SEGSUM_GRAD_ARCH35_TILING_KEY_SHORT);
    EXPECT_EQ(info.blockNum, 64);
}

TEST_F(ExpSegsumGradTilingArch35Test, short_path_keeps_small_workloads_on_existing_kernel)
{
    ExpSegsumGradCompileInfoArch35 compileInfo = {};
    auto para = MakeTilingPara({4095, 1, 16, 16}, ge::DT_FLOAT, &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, EXP_SEGSUM_GRAD_ARCH35_TILING_KEY_SMALL);
}
