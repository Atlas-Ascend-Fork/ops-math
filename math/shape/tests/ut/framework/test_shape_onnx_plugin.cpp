/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include "../../../framework/shape_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name, "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs)
{
    ge::Operator op_src = CreateOperator("src");
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

// 缺失 attribute 字段：start/end 使用 ONNX 默认值（0 / INT64_MAX），dtype 固定 DT_INT64
TEST(OnnxShapePluginTest, NoAttributeParseReturnsSuccessWithDefaults)
{
    ge::Operator op_src = CreateOperator("src");
    ge::Operator op_dest = CreateOperator("shape");

    EXPECT_EQ(domi::ParseParamsShape(op_src, op_dest), domi::SUCCESS);

    uint32_t dtype = ge::DT_UNDEFINED;
    EXPECT_EQ(op_dest.GetAttr("dtype", dtype), ge::GRAPH_SUCCESS);
    EXPECT_EQ(dtype, static_cast<uint32_t>(ge::DT_INT64));

    int64_t start = -1;
    int64_t end = -1;
    EXPECT_EQ(op_dest.GetAttr("start", start), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("end", end), ge::GRAPH_SUCCESS);
    EXPECT_EQ(start, 0);
    EXPECT_EQ(end, std::numeric_limits<int64_t>::max());
}

// 合法 JSON 且含 start/end INT 属性：属性值被正确解析
TEST(OnnxShapePluginTest, ValidStartEndAttributesParsed)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"start","type":2,"i":1},{"name":"end","type":2,"i":5}]})");
    ge::Operator op_dest = CreateOperator("shape");

    EXPECT_EQ(domi::ParseParamsShape(op_src, op_dest), domi::SUCCESS);

    int64_t start = -1;
    int64_t end = -1;
    EXPECT_EQ(op_dest.GetAttr("start", start), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("end", end), ge::GRAPH_SUCCESS);
    EXPECT_EQ(start, 1);
    EXPECT_EQ(end, 5);
}

// 空 attribute 数组：保持默认值并返回 SUCCESS
TEST(OnnxShapePluginTest, EmptyAttributeArrayKeepsDefaults)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[]})");
    ge::Operator op_dest = CreateOperator("shape");

    EXPECT_EQ(domi::ParseParamsShape(op_src, op_dest), domi::SUCCESS);

    int64_t start = -1;
    int64_t end = -1;
    EXPECT_EQ(op_dest.GetAttr("start", start), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("end", end), ge::GRAPH_SUCCESS);
    EXPECT_EQ(start, 0);
    EXPECT_EQ(end, std::numeric_limits<int64_t>::max());
}

// JSON 缺少 attribute 数组：解析失败
TEST(OnnxShapePluginTest, MissingAttributeArrayFails)
{
    ge::Operator op_src = CreateSourceOperator(R"({"other":[]})");
    ge::Operator op_dest = CreateOperator("shape");

    EXPECT_EQ(domi::ParseParamsShape(op_src, op_dest), domi::FAILED);
}

// 非法 JSON：解析失败
TEST(OnnxShapePluginTest, InvalidJsonFails)
{
    ge::Operator op_src = CreateSourceOperator("not a json");
    ge::Operator op_dest = CreateOperator("shape");

    EXPECT_EQ(domi::ParseParamsShape(op_src, op_dest), domi::FAILED);
}

// 非 INT 类型或缺失 i 字段的 start/end 条目：跳过并保持默认值
TEST(OnnxShapePluginTest, NonIntTypeAttributeIgnored)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"start","type":1,"f":1.5},{"name":"end","type":2},{"name":"other","i":7}]})");
    ge::Operator op_dest = CreateOperator("shape");

    EXPECT_EQ(domi::ParseParamsShape(op_src, op_dest), domi::SUCCESS);

    int64_t start = -1;
    int64_t end = -1;
    EXPECT_EQ(op_dest.GetAttr("start", start), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("end", end), ge::GRAPH_SUCCESS);
    EXPECT_EQ(start, 0);
    EXPECT_EQ(end, std::numeric_limits<int64_t>::max());
}

// name/original_type 属性回填：以源算子名与源类型为准
TEST(OnnxShapePluginTest, NameAndOriginalTypeSetFromSource)
{
    ge::Operator op_src = CreateOperator("src");
    ge::Operator op_dest = CreateOperator("shape");

    EXPECT_EQ(domi::ParseParamsShape(op_src, op_dest), domi::SUCCESS);

    std::string name;
    std::string original_type;
    EXPECT_EQ(op_dest.GetAttr("name", name), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("original_type", original_type), ge::GRAPH_SUCCESS);
    EXPECT_EQ(name, "src");
    EXPECT_EQ(original_type, "TestOp");
}

// 一对多展开：Data --> Shape --> StridedSliceV2，加上 4 个 Const 输入共 7 个节点；
// Graph::SetOutputs 会额外自动插入一个 NetOutput 节点，GetAllNodes 共 8 个
TEST(OnnxShapePluginTest, ParseOpToGraphExpandsToSubGraph)
{
    ge::Operator op = CreateOperator("shape_expand");
    op.SetAttr("name", std::string("node_x"));
    op.SetAttr("start", static_cast<int64_t>(1));
    op.SetAttr("end", static_cast<int64_t>(5));

    ge::Graph graph("expand_graph");
    EXPECT_EQ(domi::ParseOpToGraphShape(op, graph), domi::SUCCESS);

    std::vector<ge::GNode> nodes = graph.GetAllNodes();
    EXPECT_EQ(nodes.size(), 8U);

    std::set<std::string> node_types;
    for (const auto& node : nodes) {
        ge::AscendString type;
        EXPECT_EQ(node.GetType(type), ge::GRAPH_SUCCESS);
        node_types.insert(type.GetString());
    }
    EXPECT_NE(node_types.find("Data"), node_types.end());
    EXPECT_NE(node_types.find("Shape"), node_types.end());
    EXPECT_NE(node_types.find("Const"), node_types.end());
    EXPECT_NE(node_types.find("StridedSliceV2"), node_types.end());
}

// 缺少 name 属性时展开失败
TEST(OnnxShapePluginTest, ParseOpToGraphWithoutNameFails)
{
    ge::Operator op = CreateOperator("shape_expand");
    op.SetAttr("start", static_cast<int64_t>(1));
    op.SetAttr("end", static_cast<int64_t>(5));

    ge::Graph graph("expand_graph");
    EXPECT_EQ(domi::ParseOpToGraphShape(op, graph), domi::FAILED);
}
