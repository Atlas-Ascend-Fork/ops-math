/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <dlfcn.h>
#include <map>
#include <string>
#include <gtest/gtest.h>
#include "platform/platform_infos_def.h"
#include "tiling_case_executor.h"
#include "../../../../op_kernel/arch35/sinh_tiling_data.h"

namespace {
// 当前手写 tiling 不读取编译信息，仅需 ComputeNodeInfo 非空。
struct CompileInfo {};
CompileInfo compileInfo;

using SinhTilingFunction = ge::graphStatus (*)(gert::TilingContext*);

SinhTilingFunction GetSinhTilingFunction()
{
    void* handle = dlopen("libophost_math_ut.so", RTLD_NOW | RTLD_NOLOAD);
    if (handle == nullptr) {
        ADD_FAILURE() << dlerror();
        return nullptr;
    }
    auto function = reinterpret_cast<SinhTilingFunction>(dlsym(handle, "SinhTilingTestEntry"));
    if (function == nullptr) {
        ADD_FAILURE() << dlerror();
    }
    dlclose(handle);
    return function;
}

gert::TilingContextPara MakeContext(const gert::StorageShape& shape, ge::DataType dtype = ge::DT_FLOAT,
                                    ge::Format format = ge::FORMAT_ND, uint64_t coreNum = 64, uint64_t ubSize = 262144)
{
    return {"Sinh", {{shape, dtype, format}}, {{shape, dtype, format}}, &compileInfo, coreNum, ubSize};
}

bool ExecuteSinhTiling(const gert::TilingContextPara& para, TilingInfo& info)
{
    // The common executor does not populate vector_reg_width for nonempty inputs.
    fe::PlatFormInfos platform;
    platform.Init();
    std::map<std::string, std::string> soc = {{"ai_core_cnt", std::to_string(para.coreNum_)},
                                              {"vector_core_cnt", std::to_string(para.coreNum_)},
                                              {"core_type_list", "AICore"}};
    std::map<std::string, std::string> spec = {{"ub_size", std::to_string(para.ubSize_)}, {"vector_reg_width", "256"}};
    std::map<std::string, std::string> version = {{"NpuArch", "3510"}, {"Short_SoC_version", "Ascend950"}};
    platform.SetPlatformRes("SoCInfo", soc);
    platform.SetPlatformRes("AICoreSpec", spec);
    platform.SetPlatformRes("version", version);
    platform.SetCoreNumByCoreType("AICore");

    const auto& x = para.inputTensorDesc_.front();
    const auto& y = para.outputTensorDesc_.front();
    gert::Tensor input(x.shape_, gert::StorageFormat(x.format_, x.format_, gert::ExpandDimsType()),
                       gert::TensorPlacement::kOnHost, x.dtype_, nullptr);
    gert::Tensor output(y.shape_, gert::StorageFormat(y.format_, y.format_, gert::ExpandDimsType()),
                        gert::TensorPlacement::kOnHost, y.dtype_, nullptr);
    auto data = gert::TilingData::CreateCap(para.tilingDataSize_);
    auto workspace = gert::ContinuousVector::Create<size_t>(1);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .InputTensors({&input})
                      .OutputTensors({&output})
                      .SetOpType("Sinh")
                      .CompileInfo(para.compileInfo_)
                      .PlatformInfo(&platform)
                      .TilingData(data.get())
                      .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()))
                      .Build();
    const auto tiling = GetSinhTilingFunction();
    auto* context = holder.GetContext();
    if (tiling == nullptr || tiling(context) != ge::GRAPH_SUCCESS) {
        return false;
    }
    info.tilingKey = context->GetTilingKey();
    info.blockNum = context->GetBlockDim();
    info.workspaceSizes = {static_cast<int64_t>(context->GetWorkspaceSizes(1)[0])};
    const auto* rawData = context->GetRawTilingData();
    info.tilingDataSize = rawData->GetDataSize();
    info.tilingData = std::make_unique<uint8_t[]>(info.tilingDataSize);
    std::memcpy(info.tilingData.get(), rawData->GetData(), info.tilingDataSize);
    return true;
}

SinhTilingData ReadTilingData(const TilingInfo& info)
{
    EXPECT_EQ(info.tilingDataSize, sizeof(SinhTilingData));
    SinhTilingData data{};
    if (info.tilingDataSize != sizeof(data) || info.tilingData == nullptr) {
        return data;
    }
    std::memcpy(&data, info.tilingData.get(), sizeof(data));
    return data;
}

void CheckEmptyTiling(const gert::StorageShape& shape)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        TilingInfo info;
        ASSERT_TRUE(ExecuteSinhTiling(MakeContext(shape, dtype), info));
        EXPECT_EQ(info.tilingKey, 0U);
        EXPECT_EQ(info.blockNum, 1U);
        EXPECT_EQ(info.workspaceSizes, std::vector<int64_t>({0}));
        const SinhTilingData data = ReadTilingData(info);
        EXPECT_EQ(data.totalLength, 0U);
        EXPECT_EQ(data.blockLength, 0U);
        EXPECT_EQ(data.tileLength, 0U);
    }
}
} // namespace

TEST(SinhEmptyTilingTest, ZeroLengthVector) { CheckEmptyTiling({{0}, {0}}); }

TEST(SinhEmptyTilingTest, ZeroOnFirstAxis) { CheckEmptyTiling({{0, 2, 3}, {0, 2, 3}}); }

TEST(SinhEmptyTilingTest, ZeroOnMiddleAxis) { CheckEmptyTiling({{2, 0, 3}, {2, 0, 3}}); }

TEST(SinhEmptyTilingTest, ZeroOnLastAxis) { CheckEmptyTiling({{2, 3, 0}, {2, 3, 0}}); }

TEST(SinhEmptyTilingTest, MultipleZeroAxes) { CheckEmptyTiling({{0, 0, 0}, {0, 0, 0}}); }

TEST(SinhEmptyTilingTest, EmptyInputWithNonEmptyOutputFails)
{
    gert::TilingContextPara context{"Sinh",
                                    {{{{2, 0, 3}, {2, 0, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                    {{{{2, 1, 3}, {2, 1, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                    &compileInfo};
    TilingInfo info;
    EXPECT_FALSE(ExecuteSinhTiling(context, info));
}

TEST(SinhEmptyTilingTest, ScalarStillContainsOneElement)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteSinhTiling(MakeContext({{}, {}}), info));
    const SinhTilingData data = ReadTilingData(info);
    EXPECT_EQ(data.totalLength, 1U);
    EXPECT_GE(data.blockLength, 1U);
    EXPECT_GE(data.tileLength, 1U);
    EXPECT_GE(info.blockNum, 1U);
    EXPECT_EQ(info.tilingKey, 0U);
    EXPECT_EQ(info.workspaceSizes, std::vector<int64_t>({0}));
}

TEST(SinhEmptyTilingTest, RankEightAccepted)
{
    gert::StorageShape shape = {{1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1}};
    TilingInfo info;
    ASSERT_TRUE(ExecuteSinhTiling(MakeContext(shape), info));
    EXPECT_EQ(ReadTilingData(info).totalLength, 1U);
}

TEST(SinhEmptyTilingTest, RankNineRejected)
{
    gert::StorageShape shape = {{1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}};
    TilingInfo info;
    EXPECT_FALSE(ExecuteSinhTiling(MakeContext(shape), info));
}

TEST(SinhEmptyTilingTest, NchwStorageFormatAccepted)
{
    // P1-2 回归锁：tiling 不再拦截非 ND 公有存储格式，连续 NCHW 按平展长度计算。
    gert::StorageShape shape = {{2, 3, 4, 5}, {2, 3, 4, 5}};
    TilingInfo info;
    ASSERT_TRUE(ExecuteSinhTiling(MakeContext(shape, ge::DT_FLOAT, ge::FORMAT_NCHW), info));
    EXPECT_EQ(ReadTilingData(info).totalLength, 120U);
    EXPECT_EQ(info.tilingKey, 0U);
    EXPECT_EQ(info.workspaceSizes, std::vector<int64_t>({0}));
}

TEST(SinhEmptyTilingTest, FourGigElementBoundaryNoOverflow)
{
    // P2-35b host 侧验证：uint64 平展乘积与切分计算在 4G(2^32) 元素界点不溢出。
    // 2GB+ 用例已按 GPU 参考不可回传移出比对范围，界点数学由本 UT 承担。
    for (uint64_t elements : {uint64_t(4294967295ULL), uint64_t(4294967296ULL), uint64_t(4294967297ULL)}) {
        SCOPED_TRACE(static_cast<unsigned long long>(elements));
        gert::StorageShape shape = {{static_cast<int64_t>(elements)}, {static_cast<int64_t>(elements)}};
        TilingInfo info;
        ASSERT_TRUE(ExecuteSinhTiling(MakeContext(shape, ge::DT_FLOAT16, ge::FORMAT_ND, 128U), info));
        const SinhTilingData data = ReadTilingData(info);
        EXPECT_EQ(data.totalLength, elements);
        EXPECT_GE(data.blockLength * info.blockNum, elements);
        EXPECT_LE(info.blockNum, 128U);
        EXPECT_EQ(info.workspaceSizes, std::vector<int64_t>({0}));
    }
}
