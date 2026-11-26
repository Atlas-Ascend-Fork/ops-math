/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_pack_v2_infershape.cpp
 * \brief PackV2 infershape UT.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class PackV2Infershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PackV2Infershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "PackV2Infershape TearDown" << std::endl; }
};

// PackV2 的 InferShape（op_host/pack_v2_infershape.cpp）行为：
// 输出 z 与输入 x 形状一致，仅最后一维翻倍；入参 y 与属性 d 不参与推导。
TEST_F(PackV2Infershape, infer_shape_2d_float)
{
    gert::InfershapeContextPara infershapeContextPara("PackV2",
                                                      {
                                                          {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });

    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 6}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PackV2Infershape, infer_shape_3d_float16)
{
    gert::InfershapeContextPara infershapeContextPara("PackV2",
                                                      {
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      });

    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PackV2Infershape, infer_shape_1d_int32)
{
    gert::InfershapeContextPara infershapeContextPara("PackV2",
                                                      {
                                                          {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      });

    std::vector<std::vector<int64_t>> expectOutputShape = {{10}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 空 tensor 边界：某一维为 0 时，翻倍后仍为 0，推导保持成功
TEST_F(PackV2Infershape, infer_shape_zero_dim_int16)
{
    gert::InfershapeContextPara infershapeContextPara("PackV2",
                                                      {
                                                          {{{0, 4}, {0, 4}}, ge::DT_INT16, ge::FORMAT_ND},
                                                          {{{0, 4}, {0, 4}}, ge::DT_INT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT16, ge::FORMAT_ND},
                                                      });

    std::vector<std::vector<int64_t>> expectOutputShape = {{0, 8}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 空 tensor 边界：最后一维为 0 时，0 * 2 = 0，推导保持成功
TEST_F(PackV2Infershape, infer_shape_trailing_zero_dim)
{
    gert::InfershapeContextPara infershapeContextPara("PackV2",
                                                      {
                                                          {{{4, 0}, {4, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{4, 0}, {4, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });

    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 动态维边界：最后一维为 -1 时，InferShape 执行 -1 * 2 = -2，推导保持成功
TEST_F(PackV2Infershape, infer_shape_dynamic_last_dim)
{
    gert::InfershapeContextPara infershapeContextPara("PackV2",
                                                      {
                                                          {{{2, -1}, {2, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, -1}, {2, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });

    std::vector<std::vector<int64_t>> expectOutputShape = {{2, -2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 区分“x 末维翻倍”与“x/y 末维拼接”：x 与 y 形状不同（{2,3} vs {2,5}），
// 验证 InferShapePackV2 仅依据 x 推导输出为 {2,6}（而非拼接语义的 {2,8}），y 不参与推导。
TEST_F(PackV2Infershape, infer_shape_distinct_y_shape_ignored)
{
    gert::InfershapeContextPara infershapeContextPara("PackV2",
                                                      {
                                                          {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 5}, {2, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });

    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 6}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 显式构造可选属性 d（不同取值）叠加不同 y 形状，验证属性 d 不改变推导结果
TEST_F(PackV2Infershape, infer_shape_with_attr_d_values)
{
    for (int64_t dVal : {static_cast<int64_t>(0), static_cast<int64_t>(1)}) {
        gert::InfershapeContextPara infershapeContextPara(
            "PackV2",
            {
                {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                {{{4, 5}, {4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            },
            {
                {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            },
            {
                gert::InfershapeContextPara::OpAttr("d", Ops::Math::AnyValue::CreateFrom<int64_t>(dVal)),
            });

        std::vector<std::vector<int64_t>> expectOutputShape = {{2, 6}};
        ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
    }
}

// 输入 x 缺失（null）时，InferShape 的空指针校验应返回失败
TEST_F(PackV2Infershape, infer_shape_null_input_x_fails)
{
    gert::InfershapeContextPara infershapeContextPara("PackV2",
                                                      {
                                                          {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    infershapeContextPara.nullInputIndices_ = {0};

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}
