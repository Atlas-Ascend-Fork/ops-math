/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0(the "License").
 * Please refer to the License for details.You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_bessel_i1e_infershape.cpp
 * \brief InferShape UT for BesselI1e: output shape is a direct copy of the input
 *        shape (elementwise op, rank 0~8, empty tensor and dynamic dims included).
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

namespace BesselI1eUT {
using namespace std;
using namespace ge;

static const std::string OP_NAME = "BesselI1e";

class BesselI1eInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BesselI1eInfershape SetUp." << std::endl; }

    static void TearDownTestCase() { std::cout << "BesselI1eInfershape TearDown." << std::endl; }
};

TEST_F(BesselI1eInfershape, infershape_fp32_1d)
{
    gert::InfershapeContextPara infershapeContextPara(OP_NAME,
                                                      {
                                                          {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1024},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BesselI1eInfershape, infershape_fp16_4d)
{
    gert::InfershapeContextPara infershapeContextPara(OP_NAME,
                                                      {
                                                          {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BesselI1eInfershape, infershape_bf16_2d)
{
    gert::InfershapeContextPara infershapeContextPara(OP_NAME,
                                                      {
                                                          {{{8, 16}, {8, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8, 16},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BesselI1eInfershape, infershape_rank0_scalar)
{
    gert::InfershapeContextPara infershapeContextPara(OP_NAME,
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BesselI1eInfershape, infershape_empty_tensor)
{
    gert::InfershapeContextPara infershapeContextPara(OP_NAME,
                                                      {
                                                          {{{0, 3}, {0, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {0, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BesselI1eInfershape, infershape_dynamic_shape)
{
    gert::InfershapeContextPara infershapeContextPara(
        OP_NAME,
        {
            {{{1, -1, -1, 64}, {1, -1, -1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, -1, -1, 64},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BesselI1eInfershape, infershape_max_rank_8d)
{
    gert::InfershapeContextPara infershapeContextPara(
        OP_NAME,
        {
            {{{2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 2, 2, 2, 2, 2, 2, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BesselI1eInfershape, infershape_fp16_rank0_scalar)
{
    gert::InfershapeContextPara infershapeContextPara(OP_NAME,
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BesselI1eInfershape, infershape_bf16_rank0_scalar)
{
    gert::InfershapeContextPara infershapeContextPara(OP_NAME,
                                                      {
                                                          {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BesselI1eInfershape, infershape_bf16_empty_tensor)
{
    gert::InfershapeContextPara infershapeContextPara(OP_NAME,
                                                      {
                                                          {{{0, 3}, {0, 3}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {0, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BesselI1eInfershape, infershape_bf16_dynamic_shape)
{
    gert::InfershapeContextPara infershapeContextPara(OP_NAME,
                                                      {
                                                          {{{-1, -1}, {-1, -1}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BesselI1eInfershape, infershape_fp16_max_rank_8d)
{
    gert::InfershapeContextPara infershapeContextPara(
        OP_NAME,
        {
            {{{2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 2, 2, 2, 2, 2, 2, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(BesselI1eInfershape, infershape_bf16_max_rank_8d)
{
    gert::InfershapeContextPara infershapeContextPara(
        OP_NAME,
        {
            {{{2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 2, 2, 2, 2, 2, 2, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

} // namespace BesselI1eUT
