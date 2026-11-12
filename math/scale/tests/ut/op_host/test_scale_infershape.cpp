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
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class ScaleInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ScaleInferShapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ScaleInferShapeTest TearDown" << std::endl; }
};

TEST_F(ScaleInferShapeTest, infershape_float_2d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Scale",
        {
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
            gert::InfershapeContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
            gert::InfershapeContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ScaleInferShapeTest, infershape_float16_3d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Scale",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
            gert::InfershapeContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
            gert::InfershapeContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ScaleInferShapeTest, infershape_bf16_4d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Scale",
        {
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 5}, {4, 5}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 5}, {4, 5}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
            gert::InfershapeContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
            gert::InfershapeContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ScaleInferShapeTest, infershape_1d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Scale",
        {
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
            gert::InfershapeContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
            gert::InfershapeContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {128},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ScaleInferShapeTest, infershape_scalar_scale)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Scale",
        {
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
            gert::InfershapeContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
            gert::InfershapeContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(true)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

namespace {
void CheckShape(std::initializer_list<int64_t> xDims, std::initializer_list<int64_t> scaleDims, int64_t axis,
                int64_t axes, bool blob, ge::graphStatus status, const std::vector<int64_t>& expected = {})
{
    gert::InfershapeContextPara para(
        "Scale", {{{xDims, xDims}, ge::DT_FLOAT, ge::FORMAT_ND}, {{scaleDims, scaleDims}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {gert::InfershapeContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(axis)),
         gert::InfershapeContextPara::OpAttr("num_axes", Ops::Math::AnyValue::CreateFrom<int64_t>(axes)),
         gert::InfershapeContextPara::OpAttr("scale_from_blob", Ops::Math::AnyValue::CreateFrom<bool>(blob))});
    ExecuteTestCase(para, status,
                    status == ge::GRAPH_SUCCESS ? std::vector<std::vector<int64_t>>{expected} :
                                                  std::vector<std::vector<int64_t>>{});
}
} // namespace

TEST_F(ScaleInferShapeTest, nonblob_scalar) { CheckShape({3, 5}, {}, 0, 0, false, ge::GRAPH_SUCCESS, {3, 5}); }

TEST_F(ScaleInferShapeTest, a3_nonblob_singletons)
{
    CheckShape({2, 3, 4}, {1, 4}, 1, 1, false, ge::GRAPH_FAILED);
    CheckShape({2, 3, 4, 5}, {3, 1}, 1, 1, false, ge::GRAPH_SUCCESS, {2, 3, 4, 5});
}

TEST_F(ScaleInferShapeTest, legacy_expanded_4d_runtime_windows)
{
    CheckShape({2, 3, 4, 5}, {1, 1, 4, 1}, 2, 1, true, ge::GRAPH_SUCCESS, {2, 3, 4, 5});
    CheckShape({2, 3, 4, 5}, {1, 1, 4, 1}, -2, 1, true, ge::GRAPH_SUCCESS, {2, 3, 4, 5});
    CheckShape({2, 3, 4, 5}, {1, 1, 4, 5}, 2, -1, true, ge::GRAPH_SUCCESS, {2, 3, 4, 5});
    CheckShape({2, 3, 4, 5}, {1, 1, 1, 1}, 2, 0, true, ge::GRAPH_SUCCESS, {2, 3, 4, 5});
    CheckShape({2, 3, 4, 5}, {1, 3, 4, 1}, 1, 1, false, ge::GRAPH_SUCCESS, {2, 3, 4, 5});
    CheckShape({0, 3, 4, 5}, {1, 1, 4, 1}, 2, 1, true, ge::GRAPH_SUCCESS, {0, 3, 4, 5});
}

TEST_F(ScaleInferShapeTest, reject_invalid_expanded_runtime_windows)
{
    CheckShape({2, 3, 4, 5}, {2, 1, 4, 1}, 2, 1, true, ge::GRAPH_FAILED);
    CheckShape({2, 3, 4, 5}, {1, 1, 3, 1}, 2, 1, true, ge::GRAPH_FAILED);
    CheckShape({2, 3, 4, 5}, {1, 1, 4, 2}, 2, 1, true, ge::GRAPH_FAILED);
    CheckShape({2, 3, 4, 5}, {1, 1, 4, 1}, 2, 0, true, ge::GRAPH_FAILED);
    CheckShape({2, 3, 4, 5}, {2, 3, 4, 1}, 1, 1, false, ge::GRAPH_FAILED);
    CheckShape({2, 3, 4, 5}, {1, 1, 4, 1}, 4, 1, true, ge::GRAPH_FAILED);
    CheckShape({2, 3, 4, 5}, {1, 1, 4, 1}, 2, -2, true, ge::GRAPH_FAILED);
}

TEST_F(ScaleInferShapeTest, reject_before_empty_output_elimination)
{
    CheckShape({0, 3}, {4}, 1, 1, true, ge::GRAPH_FAILED);
    CheckShape({0, 3}, {3}, 1, 1, true, ge::GRAPH_SUCCESS, {0, 3});
    CheckShape({0, 3}, {3}, 2, 1, true, ge::GRAPH_FAILED);
    CheckShape({0, 3}, {3}, 1, -2, true, ge::GRAPH_FAILED);
    CheckShape({2, 3}, {}, 1, 0, true, ge::GRAPH_SUCCESS, {2, 3});
}

TEST_F(ScaleInferShapeTest, dynamic_and_high_rank)
{
    CheckShape({-2}, {3}, 1, 1, true, ge::GRAPH_SUCCESS, {-2});
    CheckShape({2, 3}, {-2}, 1, 1, true, ge::GRAPH_SUCCESS, {2, 3});
    CheckShape({-1, 3}, {3}, 1, 1, true, ge::GRAPH_SUCCESS, {-1, 3});
    CheckShape({2, 2, 2, 2, 2, 2, 2, 2}, {2}, 7, 1, true, ge::GRAPH_SUCCESS, {2, 2, 2, 2, 2, 2, 2, 2});
    CheckShape({2, 2, 2, 2, 2, 2, 2, 2, 2}, {2}, 8, 1, true, ge::GRAPH_FAILED);
    CheckShape({0, 2, 2, 2, 2, 2, 2, 2, 2}, {2}, 8, 1, true, ge::GRAPH_FAILED);
    CheckShape({-2}, {1, 1, 1, 1, 1, 1, 1, 1, 1}, 0, -1, true, ge::GRAPH_FAILED);
}

TEST_F(ScaleInferShapeTest, unknown_scale_rank_preserves_x_shape)
{
    CheckShape({-1, 3}, {-2}, 1, 1, true, ge::GRAPH_SUCCESS, {-1, 3});
    CheckShape({0, 3}, {-2}, 1, 1, true, ge::GRAPH_SUCCESS, {0, 3});
    CheckShape({2, 3, 4, 5}, {-2}, 1, 1, false, ge::GRAPH_SUCCESS, {2, 3, 4, 5});
    CheckShape({-2}, {-2}, 1, 1, true, ge::GRAPH_SUCCESS, {-2});
    CheckShape({2, 3}, {-2}, 2, 1, true, ge::GRAPH_FAILED);
}
