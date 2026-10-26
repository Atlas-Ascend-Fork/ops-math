/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdlib>
#include <cstring>
#include <string>

#include "gtest/gtest.h"
#include "ge/es_graph_builder.h"
#include "ge/ge_utils.h"
#include "graph/tensor.h"
#include "es_math_ops.h"
#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "../../../op_graph/fusion_pass/hard_max_fusion_pass.h"

using namespace ge;
using namespace fe;
using namespace fusion;
using namespace ops;

namespace {
const char* const kRegbaseArchDav3510 = "3510";
const char* const kRegbaseArchDav9201 = "9201";
const char* const kNonRegbaseArchDav2201 = "2201";

// UT 打桩：按 ASCEND_NPU_ARCH 环境变量注入架构（同 concat_d UT 约定），弱定义可被真实实现覆盖
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

static void SetNpuArch(const char* arch) { (void)setenv("ASCEND_NPU_ARCH", arch, 1); }

// UT 桩：强定义覆盖运行库实现，按 ASCEND_GE_VERSION 环境变量注入 GE 版本（默认 9.1.0）。
// 用于覆盖低版本运行时的 defer 分支；测试内用完立即恢复，避免影响同进程其他用例。
extern "C" {
int32_t aclsysGetVersionNum(char* pkgName, int32_t* versionNum);
}

int32_t aclsysGetVersionNum(char* pkgName, int32_t* versionNum)
{
    (void)pkgName;
    const char* ver = std::getenv("ASCEND_GE_VERSION");
    *versionNum = (ver == nullptr) ? 90100000 : std::atoi(ver);
    return 0;
}

static void SetGeVersion(const char* version) { (void)setenv("ASCEND_GE_VERSION", version, 1); }

static void SetArgMaxDSupported(const char* supported) { (void)setenv("ASCEND_ARGMAXD_SUPPORTED", supported, 1); }
} // namespace

// UT 桩：强定义覆盖 GE 运行库的 ArgMaxD 探测实现，按 ASCEND_ARGMAXD_SUPPORTED 注入结果，
// 使 ArgMaxD 命中与回退两条分支都能确定性覆盖，不再依赖本机 ops-info 注册情况。
namespace ge {
Status GeUtils::CheckNodeSupportOnAicore(const GNode& node, bool& isSupported, AscendString& unsupportedReason)
{
    (void)node;
    const char* supported = std::getenv("ASCEND_ARGMAXD_SUPPORTED");
    isSupported = (supported != nullptr && std::strcmp(supported, "1") == 0);
    unsupportedReason = isSupported ? AscendString() : AscendString("ArgMaxD not registered on stubbed platform");
    return SUCCESS;
}
} // namespace ge

class TestableHardMaxPass : public HardMaxPass {
public:
    using HardMaxPass::Patterns;
};

class HardMaxPassTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        SetPlatform("Ascend950", "Ascend950");
        SetNpuArch(kRegbaseArchDav3510);
    }

    void SetUp() override
    {
        SetPlatform("Ascend950", "Ascend950");
        SetNpuArch(kRegbaseArchDav3510);
        SetGeVersion("90100000");
        // 默认构造"ArgMaxD 不支持"，保证回退分支确定性；命中分支由用例显式置 1
        SetArgMaxDSupported("0");
    }

    static void SetPlatform(const std::string& shortSoc, const std::string& socVersion)
    {
        PlatformInfo platformInfo;
        OptionalInfo optionalInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = shortSoc;
        optionalInfo.soc_version = socVersion;
        PlatformInfoManager::Instance().platform_info_map_[socVersion] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
    }

    std::shared_ptr<Graph> BuildGraph(DataType dtype, int64_t axis)
    {
        std::vector<int64_t> dims{2, 3, 4};
        auto builder = es::EsGraphBuilder("hardMaxTest");
        auto x = builder.CreateInput(0, "x", dtype, FORMAT_ND, dims);
        Graph* graph = builder.GetCGraphBuilder()->GetGraph();
        GNode hardMax = es::CompliantNodeBuilder(graph)
                            .OpType("HardMax")
                            .Name("HardMax")
                            .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                            .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                            .Build();
        es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), hardMax, 0);
        auto output = es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(hardMax, 0));
        output.SetAttrForNode("axis", axis);
        TensorDesc desc(Shape(dims), FORMAT_ND, dtype);
        hardMax.UpdateInputDesc(0, desc);
        hardMax.UpdateOutputDesc(0, desc);
        return builder.BuildAndReset({output});
    }

    void ExpectFusion(DataType dtype, int64_t axis, const char* npuArch, const std::string& shortSoc,
                      const char* argMaxType, const char* oneHotType, int64_t onValueIdx)
    {
        SetNpuArch(npuArch);
        SetPlatform(shortSoc, shortSoc);
        auto graph = BuildGraph(dtype, axis);
        CustomPassContext passContext;
        HardMaxPass pass;
        EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);
        // >=9.0.0 路径：pass 名已被补齐（直调 Run 场景）
        EXPECT_EQ(std::string(passContext.GetPassName().GetString()), "HardMaxPass");
        bool hasHardMax = false;
        bool hasArgMax = false;
        bool hasOneHot = false;
        int64_t argMaxDimension = -1;
        int64_t oneHotDepth = -1;
        int64_t oneHotAxis = -999;
        GNode argMaxNode;
        GNode oneHotNode;
        for (const auto& node : graph->GetAllNodes()) {
            AscendString type;
            node.GetType(type);
            hasHardMax |= type == "HardMax";
            hasArgMax |= type == argMaxType;
            hasOneHot |= type == oneHotType;
            if (type == argMaxType) {
                node.GetAttr("dimension", argMaxDimension);
                argMaxNode = node;
            }
            if (type == oneHotType) {
                node.GetAttr("depth", oneHotDepth);
                node.GetAttr("axis", oneHotAxis);
                TensorDesc onValueDesc;
                node.GetInputDesc(onValueIdx, onValueDesc);
                EXPECT_EQ(onValueDesc.GetDataType(), dtype);
                oneHotNode = node;
            }
        }
        EXPECT_FALSE(hasHardMax);
        EXPECT_TRUE(hasArgMax);
        EXPECT_TRUE(hasOneHot);
        const std::vector<int64_t> graphDims{2, 3, 4}; // 与 BuildGraph 固定 rank=3 shape 一致
        const int64_t expectedAxis = (axis < 0) ? axis + static_cast<int64_t>(graphDims.size()) : axis;
        const int64_t expectedDepth = graphDims[static_cast<size_t>(expectedAxis)];
        if (std::string(argMaxType) == "ArgMaxD") {
            // BuildArgMaxDNode：dimension attr 与输出 shape（去 axis 维）/dtype 均需与老版一致
            EXPECT_EQ(argMaxDimension, expectedAxis);
            TensorDesc argMaxOutDesc;
            ASSERT_EQ(argMaxNode.GetOutputDesc(0, argMaxOutDesc), GRAPH_SUCCESS);
            EXPECT_EQ(argMaxOutDesc.GetDataType(), DT_INT32);
            std::vector<int64_t> expectedIndicesDims = graphDims;
            expectedIndicesDims.erase(expectedIndicesDims.begin() + static_cast<std::ptrdiff_t>(expectedAxis));
            EXPECT_EQ(argMaxOutDesc.GetShape().GetDims(), expectedIndicesDims);
        }
        if (std::string(oneHotType) == "OneHotD") {
            EXPECT_EQ(oneHotDepth, expectedDepth);
            EXPECT_EQ(oneHotAxis, expectedAxis);
        }
        if (std::string(oneHotType) == "OneHot") {
            EXPECT_EQ(oneHotAxis, expectedAxis);
            // regbase 的 depth 是 OneHot 第 2 个输入（index 1）的常量，校验其值等于 axis 维大小，
            // 避免 depth 取错维（如误用 rank 而非 axis 维）测不出来
            Tensor depthTensor;
            ASSERT_EQ(oneHotNode.GetInputConstData(1, depthTensor), GRAPH_SUCCESS);
            ASSERT_EQ(depthTensor.GetDataType(), DT_INT32);
            int32_t depthValue = -1;
            if (depthTensor.GetData() != nullptr) {
                std::memcpy(&depthValue, depthTensor.GetData(), sizeof(int32_t));
            }
            EXPECT_EQ(depthValue, expectedDepth);
        }
    }
};

TEST_F(HardMaxPassTest, patternBuilds)
{
    TestableHardMaxPass pass;
    EXPECT_EQ(pass.Patterns().size(), 1U);
}

TEST_F(HardMaxPassTest, fusionSuccessFp32)
{
    ExpectFusion(DT_FLOAT, 1, kRegbaseArchDav3510, "Ascend950", "ArgMaxV2", "OneHot", 2);
}

TEST_F(HardMaxPassTest, fusionSuccessFp16NegativeAxis)
{
    ExpectFusion(DT_FLOAT16, -1, kRegbaseArchDav3510, "Ascend950", "ArgMaxV2", "OneHot", 2);
}

TEST_F(HardMaxPassTest, fusionSuccessNonRegBase)
{
    // 910B（非 regbase）+ 显式构造"ArgMaxD 不支持"：回退 ArgMaxV2（对齐老版 CheckOpSupported 探测回退行为）
    SetArgMaxDSupported("0");
    ExpectFusion(DT_FLOAT16, 1, kNonRegbaseArchDav2201, "Ascend910B", "ArgMaxV2", "OneHotD", 1);
}

TEST_F(HardMaxPassTest, fusionSuccessNonRegBaseArgMaxDHit)
{
    // 910B（非 regbase）+ ArgMaxD 探测命中：覆盖 BuildArgMaxDNode 正向路径
    // （dimension attr、输出 shape/dtype），不再依赖本机 ops-info 注册情况
    SetArgMaxDSupported("1");
    ExpectFusion(DT_FLOAT16, 1, kNonRegbaseArchDav2201, "Ascend910B", "ArgMaxD", "OneHotD", 1);
}

TEST_F(HardMaxPassTest, fusionSuccessAscend350)
{
    // Ascend350 为 DAV_3510 但 short_soc_version 非 Ascend950，分支必须跟随 NpuArch 而非字符串
    ExpectFusion(DT_FLOAT, 1, kRegbaseArchDav3510, "Ascend350", "ArgMaxV2", "OneHot", 2);
}

TEST_F(HardMaxPassTest, fusionSuccessAscend960Dt)
{
    // Ascend960DT（regbase，DAV_9201）与 950 同属 regbase，应走同一分支（ArgMaxV2+OneHot）
    ExpectFusion(DT_FLOAT, 1, kRegbaseArchDav9201, "Ascend960DT", "ArgMaxV2", "OneHot", 2);
}

TEST_F(HardMaxPassTest, fusionSuccessAscend960DtFp16NegativeAxis)
{
    ExpectFusion(DT_FLOAT16, -1, kRegbaseArchDav9201, "Ascend960DT", "ArgMaxV2", "OneHot", 2);
}

TEST_F(HardMaxPassTest, unsupportedDtypeNotChanged)
{
    SetNpuArch(kRegbaseArchDav3510);
    auto graph = BuildGraph(DT_INT32, 1);
    CustomPassContext passContext;
    HardMaxPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
}

TEST_F(HardMaxPassTest, invalidAxisNotChanged)
{
    SetNpuArch(kRegbaseArchDav3510);
    auto graph = BuildGraph(DT_FLOAT, 3);
    CustomPassContext passContext;
    HardMaxPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
}

TEST_F(HardMaxPassTest, lowRuntimeVersionDeferNotChanged)
{
    // 8.5.0 运行时：版本守卫让位给内建 pass，不触碰 9.0.0 新增接口
    SetNpuArch(kRegbaseArchDav3510);
    SetGeVersion("80500000");
    auto graph = BuildGraph(DT_FLOAT, 1);
    CustomPassContext passContext;
    HardMaxPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
    SetGeVersion("90100000");
    bool hasHardMax = false;
    for (const auto& node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        hasHardMax |= type == "HardMax";
    }
    EXPECT_TRUE(hasHardMax);
    // defer 路径未执行 SetPassName 补齐，证明 9.0.0 接口只在版本守卫之后调用
    EXPECT_EQ(passContext.GetPassName().GetLength(), 0);
}
