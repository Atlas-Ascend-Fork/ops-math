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

#include "infershape_case_executor.h"
#include "../../../op_kernel/select_top_k_config.h"

namespace {
constexpr int64_t OUTER_SIZE = 2;
constexpr int64_t AXIS_SIZE = 16;
constexpr int64_t INNER_SIZE = 4;
constexpr int64_t OUTPUT_K = 3;
constexpr int64_t UNKNOWN_DIM = -1;
constexpr int64_t INVALID_K = -1;

// 构造 L0 三维视图的推形上下文；可选区间输入缺省，两个输出从空形状开始推导。
gert::InfershapeContextPara MakeContext(const gert::StorageShape& inputShape, int64_t k)
{
    return gert::InfershapeContextPara("SelectTopK", {{inputShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                       {
                                           {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                           {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                       },
                                       {{"k", Ops::Math::AnyValue::CreateFrom<int64_t>(k)}}, {1, 0, 0}, {1, 1});
}
} // namespace

class SelectTopKInferShape : public testing::Test {};

// 只替换排序轴长度，值和索引输出均保留原来的 outer、inner。
TEST_F(SelectTopKInferShape, PreservesOuterAndInner)
{
    auto context = MakeContext({{OUTER_SIZE, AXIS_SIZE, INNER_SIZE}, {OUTER_SIZE, AXIS_SIZE, INNER_SIZE}}, OUTPUT_K);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS,
                    {{OUTER_SIZE, OUTPUT_K, INNER_SIZE}, {OUTER_SIZE, OUTPUT_K, INNER_SIZE}});
}

// 动态外维和轴后跨度保留为未知维，不能按静态非法尺寸拒绝。
TEST_F(SelectTopKInferShape, PreservesDynamicDimensions)
{
    auto context = MakeContext({{UNKNOWN_DIM, AXIS_SIZE, UNKNOWN_DIM}, {UNKNOWN_DIM, AXIS_SIZE, UNKNOWN_DIM}},
                               OUTPUT_K);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS,
                    {{UNKNOWN_DIM, OUTPUT_K, UNKNOWN_DIM}, {UNKNOWN_DIM, OUTPUT_K, UNKNOWN_DIM}});
}

// k=0 时两个输出的排序维为空，其余维度不变。
TEST_F(SelectTopKInferShape, AcceptsZeroK)
{
    auto context = MakeContext({{OUTER_SIZE, AXIS_SIZE, 1}, {OUTER_SIZE, AXIS_SIZE, 1}}, 0);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{OUTER_SIZE, 0, 1}, {OUTER_SIZE, 0, 1}});
}

// 扩展接口允许 k 超过原轴长度，输出仍按请求长度分配，由 Kernel 负责尾部填充。
TEST_F(SelectTopKInferShape, KeepsRequestedLengthForPadding)
{
    constexpr int64_t PADDED_K = AXIS_SIZE + 1;
    auto context = MakeContext({{OUTER_SIZE, AXIS_SIZE, 1}, {OUTER_SIZE, AXIS_SIZE, 1}}, PADDED_K);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{OUTER_SIZE, PADDED_K, 1}, {OUTER_SIZE, PADDED_K, 1}});
}

// 负 k 不允许进入形状推导。
TEST_F(SelectTopKInferShape, RejectsNegativeK)
{
    auto context = MakeContext({{OUTER_SIZE, AXIS_SIZE, 1}, {OUTER_SIZE, AXIS_SIZE, 1}}, INVALID_K);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

// L0 只接收三维视图，已知 inner 必须位于融合布局支持的范围内。
TEST_F(SelectTopKInferShape, RejectsInvalidInputLayouts)
{
    constexpr int64_t EXCESSIVE_INNER = SelectTopK::MAX_LAYOUT_INNER + 1;
    const std::vector<gert::StorageShape> invalidShapes = {
        {{OUTER_SIZE, AXIS_SIZE}, {OUTER_SIZE, AXIS_SIZE}},
        {{OUTER_SIZE, AXIS_SIZE, 0}, {OUTER_SIZE, AXIS_SIZE, 0}},
        {{OUTER_SIZE, AXIS_SIZE, EXCESSIVE_INNER}, {OUTER_SIZE, AXIS_SIZE, EXCESSIVE_INNER}},
    };
    for (size_t index = 0; index < invalidShapes.size(); ++index) {
        SCOPED_TRACE(index);
        auto context = MakeContext(invalidShapes[index], OUTPUT_K);
        ExecuteTestCase(context, ge::GRAPH_FAILED);
    }
}
