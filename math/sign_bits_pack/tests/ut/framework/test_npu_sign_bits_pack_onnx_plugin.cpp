/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

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
#include "../../../framework/npu_sign_bits_pack_onnx_plugin.cpp"
#pragma pop_macro("google")
#pragma pop_macro("ABSL_OPTION_USE_STD_STRING_VIEW")

TEST(OnnxNpuSignBitsPackPluginTest, PreservesInt64Size)
{
    constexpr int64_t int32Max = std::numeric_limits<int32_t>::max();
    constexpr int64_t sizes[] = {1, int32Max, int32Max + 1, (int64_t{1} << 32) + 1};

    for (const int64_t size : sizes) {
        SCOPED_TRACE(size);
        ge::onnx::NodeProto node;
        node.set_op_type("NPUSignBitsPack");
        auto* attr = node.add_attribute();
        attr->set_name("size");
        attr->set_type(ge::onnx::AttributeProto::INT);
        attr->set_i(size);

        ge::Operator op("sign_bits_pack", "SignBitsPack");
        EXPECT_EQ(domi::ParseParamsNpuSignBitsPack(&node, op), domi::SUCCESS);
        int64_t parsedSize = 0;
        EXPECT_EQ(op.GetAttr("size", parsedSize), ge::GRAPH_SUCCESS);
        EXPECT_EQ(parsedSize, size);
    }
}
