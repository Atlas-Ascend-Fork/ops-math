/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_histogram_fusion_pass.cpp
 * \brief HistogramFusionPass (Histogram --> HistogramV2) 的 CUDA 语义对齐 UT
 *
 * 覆盖: min<max 正常透传 / min==max auto-range(ReduceMin+ReduceMax 子图) /
 * min>max、bins<=0、NaN/inf、int64 转换溢出的响亮失败 / M3a 标量超值域提升 /
 * 静态空输入 / 输出 dtype 尾部 Cast / 平台 gate。
 */

#include <iostream>
#include <vector>
#include "gtest/gtest.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"
#include "ge/es_graph_builder.h"
#include "ge/compliant_node_builder.h"
#include "es_math_ops.h"
#include "log/log.h"
#include "../../../op_graph/fusion_pass/histogram_fusion_pass.h"

using namespace std;
using namespace ge;
using namespace fe;
using namespace fusion;
using namespace ops;

class HistogramFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend950";
        optiCompilationInfo.soc_version = "Ascend950";
        PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    void SetUp() override { SetPlatform("Ascend950"); }

    void SetPlatform(const std::string& soc)
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = soc;
        optiCompilationInfo.soc_version = soc;
        PlatformInfoManager::Instance().platform_info_map_[soc] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    // 建 Histogram(attr bins/min/max) 节点并跑 pass; yDtype 为 Histogram 输出 desc dtype
    std::shared_ptr<Graph> BuildHistogramGraphAndRunPass(const std::vector<int64_t>& inputDims, int64_t bins,
                                                         float minVal, float maxVal, DataType xDtype = DT_FLOAT,
                                                         DataType yDtype = DT_FLOAT, Format format = FORMAT_ND)
    {
        Shape shapeX(inputDims);

        auto graphBuilder = es::EsGraphBuilder("test");
        auto x = graphBuilder.CreateInput(0, "x", xDtype, format, inputDims);

        auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
        auto histogramNode = es::CompliantNodeBuilder(graph)
                                 .OpType("Histogram")
                                 .Name("Histogram")
                                 .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                                 .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                                 .IrDefAttrs({
                                     {"bins", es::CompliantNodeBuilder::kEsAttrOptional, "Int", es::CreateFrom(bins)},
                                     {"min", es::CompliantNodeBuilder::kEsAttrOptional, "Float",
                                      es::CreateFrom(minVal)},
                                     {"max", es::CompliantNodeBuilder::kEsAttrOptional, "Float",
                                      es::CreateFrom(maxVal)},
                                 })
                                 .Build();

        es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), histogramNode, 0);

        TensorDesc xDesc;
        x.GetProducer()->GetOutputDesc(0, xDesc);
        xDesc.SetDataType(xDtype);
        xDesc.SetShape(shapeX);
        xDesc.SetFormat(format);
        x.GetProducer()->UpdateOutputDesc(0, xDesc);

        TensorDesc yDesc;
        yDesc.SetDataType(yDtype);
        yDesc.SetShape(Shape({bins}));
        yDesc.SetFormat(format);
        histogramNode.UpdateInputDesc(0, xDesc);
        histogramNode.UpdateOutputDesc(0, yDesc);

        auto y = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(histogramNode, 0);
        std::shared_ptr<Graph> resultGraph = graphBuilder.BuildAndReset(
            std::vector<es::EsTensorHolder>{es::EsTensorHolder(y)});

        CustomPassContext passContext;
        HistogramFusionPass pass;
        pass.Run(resultGraph, passContext);

        return resultGraph;
    }

    int CountNodeType(const std::shared_ptr<Graph>& graph, const char* typeName)
    {
        int count = 0;
        for (auto node : graph->GetAllNodes()) {
            AscendString type;
            node.GetType(type);
            if (type == typeName) {
                count++;
            }
        }
        return count;
    }

    bool HasNodeType(const std::shared_ptr<Graph>& graph, const char* typeName)
    {
        return CountNodeType(graph, typeName) > 0;
    }

    // 读取图中 HistogramV2 节点的 y_dtype attr; es 层对默认值(DT_INT32)不落盘, 读不到视为默认值
    bool GetHistogramV2YDtype(const std::shared_ptr<Graph>& graph, DataType& yDtype)
    {
        for (auto node : graph->GetAllNodes()) {
            AscendString type;
            node.GetType(type);
            if (type == "HistogramV2") {
                if (node.GetAttr("y_dtype", yDtype) != SUCCESS) {
                    yDtype = DT_INT32; // proto 默认值
                }
                return true;
            }
        }
        return false;
    }
};

TEST_F(HistogramFusionPassTest, patternTest)
{
    HistogramFusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

// min<max fp32: 正常透传, 无 auto-range 子图, y_dtype=DT_FLOAT
TEST_F(HistogramFusionPassTest, normalRangeFp32)
{
    auto resultGraph = BuildHistogramGraphAndRunPass({4, 8}, 10, 0.0f, 100.0f, DT_FLOAT, DT_FLOAT);

    EXPECT_EQ(CountNodeType(resultGraph, "HistogramV2"), 1);
    EXPECT_FALSE(HasNodeType(resultGraph, "Histogram"));
    EXPECT_FALSE(HasNodeType(resultGraph, "ReduceMin"));
    EXPECT_FALSE(HasNodeType(resultGraph, "ReduceMax"));
    DataType yDtype = DT_UNDEFINED;
    EXPECT_TRUE(GetHistogramV2YDtype(resultGraph, yDtype));
    EXPECT_EQ(yDtype, DT_FLOAT);
}

// min<max int64: y_dtype=DT_INT32, 输出 desc 为 int32 时无尾部 Cast
TEST_F(HistogramFusionPassTest, normalRangeInt64)
{
    auto resultGraph = BuildHistogramGraphAndRunPass({4, 8}, 10, 0.0f, 100.0f, DT_INT64, DT_INT32);

    EXPECT_EQ(CountNodeType(resultGraph, "HistogramV2"), 1);
    DataType yDtype = DT_UNDEFINED;
    EXPECT_TRUE(GetHistogramV2YDtype(resultGraph, yDtype));
    EXPECT_EQ(yDtype, DT_INT32);
    EXPECT_FALSE(HasNodeType(resultGraph, "Cast"));
}

// min<max int32: 同上
TEST_F(HistogramFusionPassTest, normalRangeInt32)
{
    auto resultGraph = BuildHistogramGraphAndRunPass({16}, 4, -10.0f, 10.0f, DT_INT32, DT_INT32);

    EXPECT_EQ(CountNodeType(resultGraph, "HistogramV2"), 1);
    DataType yDtype = DT_UNDEFINED;
    EXPECT_TRUE(GetHistogramV2YDtype(resultGraph, yDtype));
    EXPECT_EQ(yDtype, DT_INT32);
}

// min==max=0(默认 auto-range) fp32: ReduceMin+ReduceMax+HistogramV2 子图
TEST_F(HistogramFusionPassTest, autoRangeFp32)
{
    auto resultGraph = BuildHistogramGraphAndRunPass({4, 8}, 100, 0.0f, 0.0f, DT_FLOAT, DT_FLOAT);

    EXPECT_EQ(CountNodeType(resultGraph, "ReduceMin"), 1);
    EXPECT_EQ(CountNodeType(resultGraph, "ReduceMax"), 1);
    EXPECT_EQ(CountNodeType(resultGraph, "HistogramV2"), 1);
    EXPECT_FALSE(HasNodeType(resultGraph, "Histogram"));
}

// min==max int8: auto-range 子图, 不做 M3a 提升(标量值被忽略)
TEST_F(HistogramFusionPassTest, autoRangeInt8)
{
    auto resultGraph = BuildHistogramGraphAndRunPass({4, 8}, 3, 5.0f, 5.0f, DT_INT8, DT_INT32);

    EXPECT_EQ(CountNodeType(resultGraph, "ReduceMin"), 1);
    EXPECT_EQ(CountNodeType(resultGraph, "ReduceMax"), 1);
    EXPECT_EQ(CountNodeType(resultGraph, "HistogramV2"), 1);
    EXPECT_FALSE(HasNodeType(resultGraph, "Cast"));
}

// min==max==+inf / -inf: 走 auto-range(对齐 aclnn 同号 inf 相等重算语义), 不报错
TEST_F(HistogramFusionPassTest, autoRangeInfEqual)
{
    const float inf = std::numeric_limits<float>::infinity();
    auto graphPosInf = BuildHistogramGraphAndRunPass({8}, 4, inf, inf, DT_FLOAT, DT_FLOAT);
    EXPECT_EQ(CountNodeType(graphPosInf, "HistogramV2"), 1);
    EXPECT_TRUE(HasNodeType(graphPosInf, "ReduceMin"));

    auto graphNegInf = BuildHistogramGraphAndRunPass({8}, 4, -inf, -inf, DT_FLOAT, DT_FLOAT);
    EXPECT_EQ(CountNodeType(graphNegInf, "HistogramV2"), 1);
    EXPECT_TRUE(HasNodeType(graphNegInf, "ReduceMax"));
}

// min>max fp32 / int32: 响亮失败(不融合, 残留 Histogram 在 950 上无 kernel -> 编译失败)
TEST_F(HistogramFusionPassTest, minGtMaxRejected)
{
    auto graphFp = BuildHistogramGraphAndRunPass({8}, 4, 10.0f, 1.0f, DT_FLOAT, DT_FLOAT);
    EXPECT_TRUE(HasNodeType(graphFp, "Histogram"));
    EXPECT_FALSE(HasNodeType(graphFp, "HistogramV2"));

    // int 路径按截断后 int64 比较: min=2.9->2, max=1.9->1, 2>1 报错
    auto graphInt = BuildHistogramGraphAndRunPass({8}, 4, 2.9f, 1.9f, DT_INT32, DT_INT32);
    EXPECT_TRUE(HasNodeType(graphInt, "Histogram"));
    EXPECT_FALSE(HasNodeType(graphInt, "HistogramV2"));
}

// 浮点 x + NaN/inf 标量(min!=max): 有限性校验失败
TEST_F(HistogramFusionPassTest, nonFiniteRangeRejected)
{
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();

    auto graphInf = BuildHistogramGraphAndRunPass({8}, 4, 0.0f, inf, DT_FLOAT, DT_FLOAT);
    EXPECT_TRUE(HasNodeType(graphInf, "Histogram"));
    EXPECT_FALSE(HasNodeType(graphInf, "HistogramV2"));

    auto graphNan = BuildHistogramGraphAndRunPass({8}, 4, nan, 1.0f, DT_FLOAT, DT_FLOAT);
    EXPECT_TRUE(HasNodeType(graphNan, "Histogram"));
    EXPECT_FALSE(HasNodeType(graphNan, "HistogramV2"));
}

// M3a: int8 x + 超值域标量 ±1000 -> x Cast 到 int32, min/max const 随之提升
TEST_F(HistogramFusionPassTest, widenInt8ToInt32)
{
    auto resultGraph = BuildHistogramGraphAndRunPass({8}, 3, -1000.0f, 1000.0f, DT_INT8, DT_INT32);

    EXPECT_EQ(CountNodeType(resultGraph, "HistogramV2"), 1);
    EXPECT_TRUE(HasNodeType(resultGraph, "Cast"));
}

// M3a: int8 x + ±5e19 -> 截断后仍超 int64? 不, 5e19 > 2^63 -> checked_convert 失败, 响亮报错
TEST_F(HistogramFusionPassTest, scalarOverflowRejected)
{
    auto resultGraph = BuildHistogramGraphAndRunPass({8}, 3, -5e19f, 5e19f, DT_INT8, DT_INT32);

    EXPECT_TRUE(HasNodeType(resultGraph, "Histogram"));
    EXPECT_FALSE(HasNodeType(resultGraph, "HistogramV2"));
}

// M3a: fp16 x + ±1e5 超 fp16 值域(65504) -> Cast 到 fp32
TEST_F(HistogramFusionPassTest, widenFp16ToFp32)
{
    auto resultGraph = BuildHistogramGraphAndRunPass({8}, 4, -100000.0f, 100000.0f, DT_FLOAT16, DT_FLOAT);

    EXPECT_EQ(CountNodeType(resultGraph, "HistogramV2"), 1);
    EXPECT_TRUE(HasNodeType(resultGraph, "Cast"));
}

// 静态空输入(shape 含 0) + min==max: 不建 reduce, HistogramV2 + const, kernel 输出全零
TEST_F(HistogramFusionPassTest, staticEmptyInputAutoRange)
{
    auto resultGraph = BuildHistogramGraphAndRunPass({0, 8}, 4, 0.0f, 0.0f, DT_FLOAT, DT_FLOAT);

    EXPECT_EQ(CountNodeType(resultGraph, "HistogramV2"), 1);
    EXPECT_FALSE(HasNodeType(resultGraph, "ReduceMin"));
    EXPECT_FALSE(HasNodeType(resultGraph, "ReduceMax"));
}

// bins=0 / bins=-1: 响亮失败
TEST_F(HistogramFusionPassTest, invalidBinsRejected)
{
    auto graphZero = BuildHistogramGraphAndRunPass({8}, 0, 0.0f, 1.0f, DT_FLOAT, DT_FLOAT);
    EXPECT_TRUE(HasNodeType(graphZero, "Histogram"));
    EXPECT_FALSE(HasNodeType(graphZero, "HistogramV2"));

    auto graphNeg = BuildHistogramGraphAndRunPass({8}, -1, 0.0f, 1.0f, DT_FLOAT, DT_FLOAT);
    EXPECT_TRUE(HasNodeType(graphNeg, "Histogram"));
    EXPECT_FALSE(HasNodeType(graphNeg, "HistogramV2"));
}

// 输出 desc 放宽: int8 x + y desc int8 -> HistogramV2(int32) 尾部 Cast 回 int8
TEST_F(HistogramFusionPassTest, tailCastToOriginalOutputDtype)
{
    auto resultGraph = BuildHistogramGraphAndRunPass({8}, 4, 0.0f, 10.0f, DT_INT8, DT_INT8);

    EXPECT_EQ(CountNodeType(resultGraph, "HistogramV2"), 1);
    EXPECT_TRUE(HasNodeType(resultGraph, "Cast"));
}

// 平台 gate: 910_93 不融合(仅 950/350/960DT/960PR)
TEST_F(HistogramFusionPassTest, unsupportedPlatformSkipped)
{
    SetPlatform("Ascend910_93");
    auto resultGraph = BuildHistogramGraphAndRunPass({8}, 4, 0.0f, 1.0f, DT_FLOAT, DT_FLOAT);

    EXPECT_TRUE(HasNodeType(resultGraph, "Histogram"));
    EXPECT_FALSE(HasNodeType(resultGraph, "HistogramV2"));
}

// 平台 gate: 960DT 与 950/350 同属 arch35 家族，放开融合（新芯片同步适配）
TEST_F(HistogramFusionPassTest, ascend960dtFused)
{
    SetPlatform("Ascend960DT");
    auto resultGraph = BuildHistogramGraphAndRunPass({8}, 4, 0.0f, 1.0f, DT_FLOAT, DT_FLOAT);

    EXPECT_FALSE(HasNodeType(resultGraph, "Histogram"));
    EXPECT_EQ(CountNodeType(resultGraph, "HistogramV2"), 1);
}

// rank0 标量输入 + min==max: 数据范围即自身, 直接用 x 作 min/max, 不建 reduce
TEST_F(HistogramFusionPassTest, rank0AutoRange)
{
    auto resultGraph = BuildHistogramGraphAndRunPass({}, 4, 0.0f, 0.0f, DT_FLOAT, DT_FLOAT);

    EXPECT_EQ(CountNodeType(resultGraph, "HistogramV2"), 1);
    EXPECT_FALSE(HasNodeType(resultGraph, "ReduceMin"));
    EXPECT_FALSE(HasNodeType(resultGraph, "ReduceMax"));
}
