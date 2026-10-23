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

#include "any_value.h"
#include "infershape_case_executor.h"
#include "infershape_context_faker.h"

namespace {
constexpr int64_t kXDim0 = 2;
constexpr int64_t kXDim1 = 3;
constexpr int64_t kXDim2 = 4;

gert::InfershapeContextPara BuildContext(const std::vector<int64_t>& perm, const std::vector<int64_t>& shape,
                                         bool transposeFirst)
{
    return gert::InfershapeContextPara(
        "ConfusionTransposeD",
        {
            {{{kXDim0, kXDim1, kXDim2}, {kXDim0, kXDim1, kXDim2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"perm", Ops::Math::AnyValue::CreateFrom(perm)},
            {"shape", Ops::Math::AnyValue::CreateFrom(shape)},
            {"transpose_first", Ops::Math::AnyValue::CreateFrom(transposeFirst)},
        });
}
} // namespace

TEST(ConfusionTransposeDInferShapeTest, transpose_first_true_uses_shape_attr)
{
    auto context = BuildContext({0, 1, 2}, {4, 3, 2}, true);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{4, 3, 2}});
}

TEST(ConfusionTransposeDInferShapeTest, transpose_first_false_applies_perm_to_shape_attr)
{
    auto context = BuildContext({0, 2, 1}, {4, 3, 2}, false);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{4, 2, 3}});
}

TEST(ConfusionTransposeDInferShapeTest, transpose_first_false_perm_size_mismatch_failed)
{
    auto context = BuildContext({0, 1}, {4, 3, 2}, false);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(ConfusionTransposeDInferShapeTest, transpose_first_false_perm_out_of_range_failed)
{
    auto context = BuildContext({0, 1, 3}, {4, 3, 2}, false);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(ConfusionTransposeDInferShapeTest, transpose_first_false_perm_negative_failed)
{
    auto context = BuildContext({0, -1, 1}, {4, 3, 2}, false);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST(ConfusionTransposeDInferShapeTest, unknown_rank_input_passed_through)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ConfusionTransposeD",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"perm", Ops::Math::AnyValue::CreateFrom(std::vector<int64_t>{0, 1, 2})},
            {"shape", Ops::Math::AnyValue::CreateFrom(std::vector<int64_t>{4, 3, 2})},
            {"transpose_first", Ops::Math::AnyValue::CreateFrom(false)},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {{-2}});
}
