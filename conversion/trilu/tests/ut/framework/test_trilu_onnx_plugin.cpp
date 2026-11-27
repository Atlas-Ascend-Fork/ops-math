/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>

#include <gtest/gtest.h>
#include "experiment/metadef/common/util/error_manager/error_manager.h"

// Match the SDK's Abseil string_view ABI when the header leaves it automatic.
#include "absl/base/options.h"
#pragma push_macro("ABSL_OPTION_USE_STD_STRING_VIEW")
#if ABSL_OPTION_USE_STD_STRING_VIEW == 2
#undef ABSL_OPTION_USE_STD_STRING_VIEW
#define ABSL_OPTION_USE_STD_STRING_VIEW 0
#endif

// The ONNX SDK builds protobuf in the ascend_private namespace.
#pragma push_macro("google")
#undef google
#define google ascend_private
#include "../../../framework/trilu_onnx_plugin.cpp"
#pragma pop_macro("google")
#pragma pop_macro("ABSL_OPTION_USE_STD_STRING_VIEW")

namespace {
ge::onnx::NodeProto MakeNode(const int inputCount)
{
    ge::onnx::NodeProto node;
    node.set_op_type("Trilu");
    for (int i = 0; i < inputCount; ++i) {
        node.add_input("x" + std::to_string(i));
    }
    return node;
}

// 断言 ParseParamsTrilu 失败且 ErrorManager 恰好产生一条指定错误码的首报（首错唯一性）。
void ExpectParseError(const ge::onnx::NodeProto& node, const char* errorCode)
{
    auto& errors = ErrorManager::GetInstance();
    ASSERT_EQ(errors.Init(), 0);
    (void)errors.GetRawErrorMessages();
    ge::Operator opDest("trilu", "Trilu");
    EXPECT_EQ(domi::ParseParamsTrilu(&node, opDest), domi::FAILED);
    const auto reports = errors.GetRawErrorMessages();
    ASSERT_EQ(reports.size(), 1U);
    EXPECT_EQ(reports[0].error_id, errorCode);
}
} // namespace

// 单输入（仅 x）：解析成功，upper 缺省为 1，输入注册数为 1。
TEST(OnnxTriluPluginTest, OneInputParsesSuccessfully)
{
    ge::onnx::NodeProto node = MakeNode(1);
    ge::Operator opDest("trilu", "Trilu");

    EXPECT_EQ(domi::ParseParamsTrilu(&node, opDest), domi::SUCCESS);

    int upper = 0;
    EXPECT_EQ(opDest.GetAttr("upper", upper), ge::GRAPH_SUCCESS);
    EXPECT_EQ(upper, 1);
    EXPECT_EQ(opDest.GetInputsSize(), 1U);
}

// 双输入（x 与 k）：解析成功，显式 upper=0 正确转写。
TEST(OnnxTriluPluginTest, TwoInputsParsesSuccessfully)
{
    ge::onnx::NodeProto node = MakeNode(2);
    auto* attr = node.add_attribute();
    attr->set_name("upper");
    attr->set_type(ge::onnx::AttributeProto::INT);
    attr->set_i(0);
    ge::Operator opDest("trilu", "Trilu");

    EXPECT_EQ(domi::ParseParamsTrilu(&node, opDest), domi::SUCCESS);

    int upper = 1;
    EXPECT_EQ(opDest.GetAttr("upper", upper), ge::GRAPH_SUCCESS);
    EXPECT_EQ(upper, 0);
    EXPECT_EQ(opDest.GetInputsSize(), 2U);
}

// 0 个输入：非法输入数，以 EZ0026 正式首报并返回 FAILED（此前仅记日志继续执行）。
TEST(OnnxTriluPluginTest, ZeroInputsReportsEz0026)
{
    ge::onnx::NodeProto node = MakeNode(0);
    ExpectParseError(node, "EZ0026");
}

// 3 个输入：超出 Trilu 支持的 1~2 输入范围，同样以 EZ0026 唯一首报。
TEST(OnnxTriluPluginTest, ThreeInputsReportsEz0026)
{
    ge::onnx::NodeProto node = MakeNode(3);
    ExpectParseError(node, "EZ0026");
}

// op_src 为空指针：沿用现有空指针防护，返回 FAILED。
TEST(OnnxTriluPluginTest, NullOpSrcReturnsFailed)
{
    ge::Operator opDest("trilu", "Trilu");
    EXPECT_EQ(domi::ParseParamsTrilu(nullptr, opDest), domi::FAILED);
}
