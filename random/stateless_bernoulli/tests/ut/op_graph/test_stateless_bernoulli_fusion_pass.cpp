/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include "gtest/gtest.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"
#include "ge/es_graph_builder.h"
#include "es_math_ops.h"
#include "log/log.h"
#include "../../../op_graph/fusion_pass/stateless_bernoulli_fusion_pass.h"

using namespace std;
using namespace ge;
using namespace fe;
using namespace fusion;

namespace {
const std::string kPassName = "BernoulliFusionPass";
const char* const kSupportedArchDav2201 = "2201";   // Ascend910_93
const char* const kSupportedArchDav3510 = "3510";   // Ascend950/Ascend350
const char* const kSupportedArchDav9201 = "9201";   // Ascend960DT
const char* const kSupportedArchDav9202 = "9202";   // Ascend960PR
const char* const kUnsupportedArchDav2003 = "2003"; // 不支持 arch 兜底

// UT stub for rtGetSocSpec: the UT executable does not link the runtime lib, while the pass gates
// on NpuArch via this API. The stub feeds the arch from the ASCEND_NPU_ARCH env variable.
// Weak definition: yields to any real one.
extern "C" {
typedef int32_t rtError_t;
static const int32_t RT_ERROR_NONE = 0;
rtError_t rtGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen);
}

__attribute__((weak)) rtError_t rtGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen)
{
    const char* arch = std::getenv("ASCEND_NPU_ARCH");
    if (arch == nullptr || std::strcmp(label, "version") != 0 || std::strcmp(key, "NpuArch") != 0) {
        return static_cast<rtError_t>(1);
    }
    (void)std::snprintf(val, maxLen, "%s", arch);
    return RT_ERROR_NONE;
}

// UT stub for aclsysGetVersionNum: pass 侧以弱符号声明并按 GE 编译器版本门控, 部分 UT 环境该符号
// 不可解析, 会导致融合逻辑整体被跳过。固定返回满足门槛的版本号。
extern "C" {
__attribute__((weak)) int32_t aclsysGetVersionNum(char* pkgName, int32_t* versionNum);
}

__attribute__((weak)) int32_t aclsysGetVersionNum(char* pkgName, int32_t* versionNum)
{
    if (versionNum == nullptr) {
        return -1;
    }
    *versionNum = 90100000; // GE_COMPILER_VERSION_910
    return 0;
}

static void SetNpuArch(const char* arch) { (void)setenv("ASCEND_NPU_ARCH", arch, 1); }

// soc 字符串到 NpuArch 的映射(对齐真实平台上报): Ascend350 与 Ascend950 同为 DAV_3510
// (摘自 !7874 增量: CheckFusionOnPlatform 原硬编码三元仅覆盖 960DT/960PR,
//  无法承载 350/950 成功用例)
static const char* ArchFromSoc(const std::string& soc)
{
    if (soc == "Ascend910_93") {
        return kSupportedArchDav2201;
    }
    if (soc == "Ascend950" || soc == "Ascend350") {
        return kSupportedArchDav3510;
    }
    if (soc == "Ascend960DT") {
        return kSupportedArchDav9201;
    }
    if (soc == "Ascend960PR") {
        return kSupportedArchDav9202;
    }
    return kUnsupportedArchDav2003;
}
} // namespace

class BernoulliFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend910_93";
        optiCompilationInfo.soc_version = "Ascend910_93";
        PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
        SetNpuArch(kSupportedArchDav2201);
    }

    void SetUp() override
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend910_93";
        optiCompilationInfo.soc_version = "Ascend910_93";
        PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
        SetNpuArch(kSupportedArchDav2201);
    }

    void SetPlatform950()
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend950";
        optiCompilationInfo.soc_version = "Ascend950";
        PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
        SetNpuArch(kSupportedArchDav3510);
    }

    void CheckFusionOnPlatform(const std::string& soc, DataType dtype)
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = soc;
        optiCompilationInfo.soc_version = soc;
        PlatformInfoManager::Instance().platform_info_map_[soc] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
        SetNpuArch(ArchFromSoc(soc));

        const std::vector<int64_t> dimsX{-1, 3, 5, 4};
        const std::vector<int64_t> dimsSeed{2};
        auto graphBuilder = es::EsGraphBuilder("test");
        auto x = graphBuilder.CreateInput(0, "x", dtype, FORMAT_NCHW, dimsX);
        auto seed = graphBuilder.CreateInput(1, "seed", DT_INT64, FORMAT_ND, dimsSeed);
        auto offset = graphBuilder.CreateInput(2, "offset", DT_INT64, FORMAT_ND, dimsSeed);
        auto output = es::StatelessBernoulliV2(x, seed, offset, dtype);

        TensorDesc xDesc(Shape(dimsX), FORMAT_NCHW, dtype);
        TensorDesc seedDesc(Shape(dimsSeed), FORMAT_ND, DT_INT64);
        x.GetProducer()->UpdateOutputDesc(0, xDesc);
        seed.GetProducer()->UpdateOutputDesc(0, seedDesc);
        offset.GetProducer()->UpdateOutputDesc(0, seedDesc);
        output.GetProducer()->UpdateInputDesc(0, xDesc);
        output.GetProducer()->UpdateInputDesc(1, seedDesc);
        output.GetProducer()->UpdateInputDesc(2, seedDesc);
        output.GetProducer()->UpdateOutputDesc(0, xDesc);

        std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});
        ASSERT_NE(graph, nullptr);
        CustomPassContext passContext;
        BernoulliFusionPass pass;
        Status status = pass.Run(graph, passContext);
        EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

        size_t fusedNodeCount = 0;
        size_t originalNodeCount = 0;
        for (auto node : graph->GetAllNodes()) {
            AscendString type;
            node.GetType(type);
            if (type == "StatelessBernoulli") {
                ++fusedNodeCount;
            } else if (type == "StatelessBernoulliV2") {
                ++originalNodeCount;
            }
        }
        EXPECT_EQ(fusedNodeCount, 1U);
        EXPECT_EQ(originalNodeCount, 0U);
    }

    void SetPlatform91093()
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend910_93";
        optiCompilationInfo.soc_version = "Ascend910_93";
        PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
        SetNpuArch(kSupportedArchDav2201);
    }
};

TEST_F(BernoulliFusionPassTest, patternTest)
{
    BernoulliFusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

TEST_F(BernoulliFusionPassTest, unsupportedPlatformFail)
{
    PlatformInfo platformInfo;
    OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910";
    optiCompilationInfo.soc_version = "Ascend910";
    PlatformInfoManager::Instance().platform_info_map_["Ascend910"] = platformInfo;
    PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    SetNpuArch("2002");

    std::vector<int64_t> dimsX{-1, 3, 5, 4};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, shapeX.GetDims());
    auto seed = graphBuilder.CreateInput(1, "seed", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});
    auto offset = graphBuilder.CreateInput(2, "offset", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});

    auto output = es::StatelessBernoulliV2(x, seed, offset);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    xDesc.SetFormat(FORMAT_NCHW);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);

    TensorDesc seedDesc;
    seed.GetProducer()->GetOutputDesc(0, seedDesc);
    seedDesc.SetDataType(DT_INT64);
    seedDesc.SetShape(Shape(std::vector<int64_t>{2}));
    seedDesc.SetFormat(FORMAT_ND);
    seed.GetProducer()->UpdateOutputDesc(0, seedDesc);

    TensorDesc offsetDesc;
    offset.GetProducer()->GetOutputDesc(0, offsetDesc);
    offsetDesc.SetDataType(DT_INT64);
    offsetDesc.SetShape(Shape(std::vector<int64_t>{2}));
    offsetDesc.SetFormat(FORMAT_ND);
    offset.GetProducer()->UpdateOutputDesc(0, offsetDesc);

    output.GetProducer()->UpdateInputDesc(0, xDesc);
    output.GetProducer()->UpdateInputDesc(1, seedDesc);
    output.GetProducer()->UpdateInputDesc(2, offsetDesc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    BernoulliFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(BernoulliFusionPassTest, fusionFailArchUnavailable)
{
    // 清空 ASCEND_NPU_ARCH，触发 CheckPlatform 中 rtGetSocSpec 失败出口
    (void)unsetenv("ASCEND_NPU_ARCH");

    std::vector<int64_t> dimsX{-1, 3, 5, 4};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, shapeX.GetDims());
    auto seed = graphBuilder.CreateInput(1, "seed", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});
    auto offset = graphBuilder.CreateInput(2, "offset", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});

    auto output = es::StatelessBernoulliV2(x, seed, offset);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    xDesc.SetFormat(FORMAT_NCHW);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);

    TensorDesc seedDesc;
    seed.GetProducer()->GetOutputDesc(0, seedDesc);
    seedDesc.SetDataType(DT_INT64);
    seedDesc.SetShape(Shape(std::vector<int64_t>{2}));
    seedDesc.SetFormat(FORMAT_ND);
    seed.GetProducer()->UpdateOutputDesc(0, seedDesc);

    TensorDesc offsetDesc;
    offset.GetProducer()->GetOutputDesc(0, offsetDesc);
    offsetDesc.SetDataType(DT_INT64);
    offsetDesc.SetShape(Shape(std::vector<int64_t>{2}));
    offsetDesc.SetFormat(FORMAT_ND);
    offset.GetProducer()->UpdateOutputDesc(0, offsetDesc);

    output.GetProducer()->UpdateInputDesc(0, xDesc);
    output.GetProducer()->UpdateInputDesc(1, seedDesc);
    output.GetProducer()->UpdateInputDesc(2, offsetDesc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    BernoulliFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(BernoulliFusionPassTest, fusionSuccessFp16On960DT) { CheckFusionOnPlatform("Ascend960DT", DT_FLOAT16); }

TEST_F(BernoulliFusionPassTest, fusionSuccessFp32On960DT) { CheckFusionOnPlatform("Ascend960DT", DT_FLOAT); }

TEST_F(BernoulliFusionPassTest, fusionSuccessFp16On960PR) { CheckFusionOnPlatform("Ascend960PR", DT_FLOAT16); }

TEST_F(BernoulliFusionPassTest, fusionSuccessFp32On960PR) { CheckFusionOnPlatform("Ascend960PR", DT_FLOAT); }

// Ascend350 与 Ascend950 的 NpuArch 同为 DAV_3510, 平台门控按 arch 判断后 350 上融合必须生效
// (摘自 !7874 增量: pass 侧门控修复已由 !7937 合入, 此处补齐 350 正面回归锚点)
TEST_F(BernoulliFusionPassTest, fusionSuccessFp16On350) { CheckFusionOnPlatform("Ascend350", DT_FLOAT16); }

TEST_F(BernoulliFusionPassTest, fusionSuccessFp32On350) { CheckFusionOnPlatform("Ascend350", DT_FLOAT); }

TEST_F(BernoulliFusionPassTest, fusionSuccessFp16On950)
{
    SetPlatform950();

    std::vector<int64_t> dimsX{-1, 3, 5, 4};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, shapeX.GetDims());
    auto seed = graphBuilder.CreateInput(1, "seed", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});
    auto offset = graphBuilder.CreateInput(2, "offset", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});

    auto output = es::StatelessBernoulliV2(x, seed, offset);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    xDesc.SetFormat(FORMAT_NCHW);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);

    TensorDesc seedDesc;
    seed.GetProducer()->GetOutputDesc(0, seedDesc);
    seedDesc.SetDataType(DT_INT64);
    seedDesc.SetShape(Shape(std::vector<int64_t>{2}));
    seedDesc.SetFormat(FORMAT_ND);
    seed.GetProducer()->UpdateOutputDesc(0, seedDesc);

    TensorDesc offsetDesc;
    offset.GetProducer()->GetOutputDesc(0, offsetDesc);
    offsetDesc.SetDataType(DT_INT64);
    offsetDesc.SetShape(Shape(std::vector<int64_t>{2}));
    offsetDesc.SetFormat(FORMAT_ND);
    offset.GetProducer()->UpdateOutputDesc(0, offsetDesc);
    TensorDesc outputDesc;
    output.GetProducer()->GetOutputDesc(0, outputDesc);
    outputDesc.SetDataType(DT_FLOAT16);
    outputDesc.SetShape(shapeX);
    outputDesc.SetFormat(FORMAT_NCHW);
    output.GetProducer()->UpdateOutputDesc(0, outputDesc);

    output.GetProducer()->UpdateInputDesc(0, xDesc);
    output.GetProducer()->UpdateInputDesc(1, seedDesc);
    output.GetProducer()->UpdateInputDesc(2, offsetDesc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    BernoulliFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool isFound = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "StatelessBernoulli") {
            isFound = true;
            break;
        }
    }
    EXPECT_TRUE(isFound);
}

TEST_F(BernoulliFusionPassTest, fusionSuccessFp16On91093)
{
    SetPlatform91093();

    std::vector<int64_t> dimsX{-1, 3, 5, 4};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, shapeX.GetDims());
    auto seed = graphBuilder.CreateInput(1, "seed", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});
    auto offset = graphBuilder.CreateInput(2, "offset", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});

    auto output = es::StatelessBernoulliV2(x, seed, offset);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    xDesc.SetFormat(FORMAT_NCHW);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);

    TensorDesc seedDesc;
    seed.GetProducer()->GetOutputDesc(0, seedDesc);
    seedDesc.SetDataType(DT_INT64);
    seedDesc.SetShape(Shape(std::vector<int64_t>{2}));
    seedDesc.SetFormat(FORMAT_ND);
    seed.GetProducer()->UpdateOutputDesc(0, seedDesc);

    TensorDesc offsetDesc;
    offset.GetProducer()->GetOutputDesc(0, offsetDesc);
    offsetDesc.SetDataType(DT_INT64);
    offsetDesc.SetShape(Shape(std::vector<int64_t>{2}));
    offsetDesc.SetFormat(FORMAT_ND);
    offset.GetProducer()->UpdateOutputDesc(0, offsetDesc);

    output.GetProducer()->UpdateInputDesc(0, xDesc);
    output.GetProducer()->UpdateInputDesc(1, seedDesc);
    output.GetProducer()->UpdateInputDesc(2, offsetDesc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    BernoulliFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool isFound = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "StatelessBernoulli") {
            isFound = true;
            break;
        }
    }
    EXPECT_TRUE(isFound);
}

TEST_F(BernoulliFusionPassTest, fusionSuccessFp32)
{
    SetPlatform950();

    std::vector<int64_t> dimsX{-1, 3, 5, 4};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NCHW, shapeX.GetDims());
    auto seed = graphBuilder.CreateInput(1, "seed", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});
    auto offset = graphBuilder.CreateInput(2, "offset", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});

    auto output = es::StatelessBernoulliV2(x, seed, offset);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT);
    xDesc.SetShape(shapeX);
    xDesc.SetFormat(FORMAT_NCHW);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);

    TensorDesc seedDesc;
    seed.GetProducer()->GetOutputDesc(0, seedDesc);
    seedDesc.SetDataType(DT_INT64);
    seedDesc.SetShape(Shape(std::vector<int64_t>{2}));
    seedDesc.SetFormat(FORMAT_ND);
    seed.GetProducer()->UpdateOutputDesc(0, seedDesc);

    TensorDesc offsetDesc;
    offset.GetProducer()->GetOutputDesc(0, offsetDesc);
    offsetDesc.SetDataType(DT_INT64);
    offsetDesc.SetShape(Shape(std::vector<int64_t>{2}));
    offsetDesc.SetFormat(FORMAT_ND);
    offset.GetProducer()->UpdateOutputDesc(0, offsetDesc);

    output.GetProducer()->UpdateInputDesc(0, xDesc);
    output.GetProducer()->UpdateInputDesc(1, seedDesc);
    output.GetProducer()->UpdateInputDesc(2, offsetDesc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    BernoulliFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool isFound = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "StatelessBernoulli") {
            isFound = true;
            break;
        }
    }
    EXPECT_TRUE(isFound);
}

TEST_F(BernoulliFusionPassTest, fusionSuccessInt64Dtype)
{
    SetPlatform950();

    std::vector<int64_t> dimsX{-1, 3, 5, 4};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, shapeX.GetDims());
    auto seed = graphBuilder.CreateInput(1, "seed", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});
    auto offset = graphBuilder.CreateInput(2, "offset", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});

    auto output = es::StatelessBernoulliV2(x, seed, offset, DT_INT64);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    xDesc.SetFormat(FORMAT_NCHW);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);

    TensorDesc seedDesc;
    seed.GetProducer()->GetOutputDesc(0, seedDesc);
    seedDesc.SetDataType(DT_INT64);
    seedDesc.SetShape(Shape(std::vector<int64_t>{2}));
    seedDesc.SetFormat(FORMAT_ND);
    seed.GetProducer()->UpdateOutputDesc(0, seedDesc);

    TensorDesc offsetDesc;
    offset.GetProducer()->GetOutputDesc(0, offsetDesc);
    offsetDesc.SetDataType(DT_INT64);
    offsetDesc.SetShape(Shape(std::vector<int64_t>{2}));
    offsetDesc.SetFormat(FORMAT_ND);
    offset.GetProducer()->UpdateOutputDesc(0, offsetDesc);

    output.GetProducer()->UpdateInputDesc(0, xDesc);
    output.GetProducer()->UpdateInputDesc(1, seedDesc);
    output.GetProducer()->UpdateInputDesc(2, offsetDesc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    BernoulliFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool isFound = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "StatelessBernoulli") {
            isFound = true;
            break;
        }
    }
    EXPECT_TRUE(isFound);
}

TEST_F(BernoulliFusionPassTest, fusionFailUnknownRank)
{
    SetPlatform950();

    std::vector<int64_t> dimsX{-2};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, shapeX.GetDims());
    auto seed = graphBuilder.CreateInput(1, "seed", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});
    auto offset = graphBuilder.CreateInput(2, "offset", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});

    auto output = es::StatelessBernoulliV2(x, seed, offset, DT_INT64);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    xDesc.SetFormat(FORMAT_NCHW);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);

    TensorDesc seedDesc;
    seed.GetProducer()->GetOutputDesc(0, seedDesc);
    seedDesc.SetDataType(DT_INT64);
    seedDesc.SetShape(Shape(std::vector<int64_t>{2}));
    seedDesc.SetFormat(FORMAT_ND);
    seed.GetProducer()->UpdateOutputDesc(0, seedDesc);

    TensorDesc offsetDesc;
    offset.GetProducer()->GetOutputDesc(0, offsetDesc);
    offsetDesc.SetDataType(DT_INT64);
    offsetDesc.SetShape(Shape(std::vector<int64_t>{2}));
    offsetDesc.SetFormat(FORMAT_ND);
    offset.GetProducer()->UpdateOutputDesc(0, offsetDesc);

    output.GetProducer()->UpdateInputDesc(0, xDesc);
    output.GetProducer()->UpdateInputDesc(1, seedDesc);
    output.GetProducer()->UpdateInputDesc(2, offsetDesc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    BernoulliFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(BernoulliFusionPassTest, fusionSuccessLargeShape)
{
    SetPlatform950();

    std::vector<int64_t> dimsX{-1, 7, 8, 6};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_NCHW, shapeX.GetDims());
    auto seed = graphBuilder.CreateInput(1, "seed", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});
    auto offset = graphBuilder.CreateInput(2, "offset", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});

    auto output = es::StatelessBernoulliV2(x, seed, offset, DT_INT64);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT);
    xDesc.SetShape(shapeX);
    xDesc.SetFormat(FORMAT_NCHW);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);

    TensorDesc seedDesc;
    seed.GetProducer()->GetOutputDesc(0, seedDesc);
    seedDesc.SetDataType(DT_INT64);
    seedDesc.SetShape(Shape(std::vector<int64_t>{2}));
    seedDesc.SetFormat(FORMAT_ND);
    seed.GetProducer()->UpdateOutputDesc(0, seedDesc);

    TensorDesc offsetDesc;
    offset.GetProducer()->GetOutputDesc(0, offsetDesc);
    offsetDesc.SetDataType(DT_INT64);
    offsetDesc.SetShape(Shape(std::vector<int64_t>{2}));
    offsetDesc.SetFormat(FORMAT_ND);
    offset.GetProducer()->UpdateOutputDesc(0, offsetDesc);

    output.GetProducer()->UpdateInputDesc(0, xDesc);
    output.GetProducer()->UpdateInputDesc(1, seedDesc);
    output.GetProducer()->UpdateInputDesc(2, offsetDesc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    BernoulliFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool isFound = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "StatelessBernoulli") {
            isFound = true;
            break;
        }
    }
    EXPECT_TRUE(isFound);
}

TEST_F(BernoulliFusionPassTest, fusionSuccessHighDim)
{
    SetPlatform950();

    std::vector<int64_t> dimsX{-1, 3, 5, 4, 4, 2, 2};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, shapeX.GetDims());
    auto seed = graphBuilder.CreateInput(1, "seed", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});
    auto offset = graphBuilder.CreateInput(2, "offset", DT_INT64, FORMAT_ND, std::vector<int64_t>{2});

    auto output = es::StatelessBernoulliV2(x, seed, offset, DT_INT64);

    TensorDesc xDesc;
    x.GetProducer()->GetOutputDesc(0, xDesc);
    xDesc.SetDataType(DT_FLOAT16);
    xDesc.SetShape(shapeX);
    xDesc.SetFormat(FORMAT_NCHW);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);

    TensorDesc seedDesc;
    seed.GetProducer()->GetOutputDesc(0, seedDesc);
    seedDesc.SetDataType(DT_INT64);
    seedDesc.SetShape(Shape(std::vector<int64_t>{2}));
    seedDesc.SetFormat(FORMAT_ND);
    seed.GetProducer()->UpdateOutputDesc(0, seedDesc);

    TensorDesc offsetDesc;
    offset.GetProducer()->GetOutputDesc(0, offsetDesc);
    offsetDesc.SetDataType(DT_INT64);
    offsetDesc.SetShape(Shape(std::vector<int64_t>{2}));
    offsetDesc.SetFormat(FORMAT_ND);
    offset.GetProducer()->UpdateOutputDesc(0, offsetDesc);

    output.GetProducer()->UpdateInputDesc(0, xDesc);
    output.GetProducer()->UpdateInputDesc(1, seedDesc);
    output.GetProducer()->UpdateInputDesc(2, offsetDesc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output});

    CustomPassContext passContext;
    BernoulliFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool isFound = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "StatelessBernoulli") {
            isFound = true;
            break;
        }
    }
    EXPECT_TRUE(isFound);
}
