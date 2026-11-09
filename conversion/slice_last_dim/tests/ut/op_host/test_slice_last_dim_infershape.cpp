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
#include <cstdint>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class SliceLastDimInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SliceLastDimInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SliceLastDimInfershape TearDown" << std::endl; }
};

TEST_F(SliceLastDimInfershape, slice_last_dim_basic_1d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SliceLastDim",
        {
            {{{10}, {10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("start", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
            gert::InfershapeContextPara::OpAttr("end", Ops::Math::AnyValue::CreateFrom<int64_t>(7)),
            gert::InfershapeContextPara::OpAttr("stride", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SliceLastDimInfershape, slice_last_dim_2d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SliceLastDim",
        {
            {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("start", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
            gert::InfershapeContextPara::OpAttr("end", Ops::Math::AnyValue::CreateFrom<int64_t>(4)),
            gert::InfershapeContextPara::OpAttr("stride", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SliceLastDimInfershape, slice_last_dim_3d_with_stride)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SliceLastDim",
        {
            {{{2, 3, 10}, {2, 3, 10}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("start", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
            gert::InfershapeContextPara::OpAttr("end", Ops::Math::AnyValue::CreateFrom<int64_t>(10)),
            gert::InfershapeContextPara::OpAttr("stride", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SliceLastDimInfershape, slice_last_dim_dynamic)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SliceLastDim",
        {
            {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("start", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
            gert::InfershapeContextPara::OpAttr("end", Ops::Math::AnyValue::CreateFrom<int64_t>(5)),
            gert::InfershapeContextPara::OpAttr("stride", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SliceLastDimInfershape, slice_last_dim_full_slice)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SliceLastDim",
        {
            {{{5, 100}, {5, 100}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("start", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
            gert::InfershapeContextPara::OpAttr("end", Ops::Math::AnyValue::CreateFrom<int64_t>(100)),
            gert::InfershapeContextPara::OpAttr("stride", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5, 100},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// stride == 0 must be rejected (division by zero on a non-empty slice).
TEST_F(SliceLastDimInfershape, slice_last_dim_infershape_stride_zero_reject)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SliceLastDim",
        {
            {{{2, 3, 10}, {2, 3, 10}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("start", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
            gert::InfershapeContextPara::OpAttr("end", Ops::Math::AnyValue::CreateFrom<int64_t>(5)),
            gert::InfershapeContextPara::OpAttr("stride", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// Negative stride must be rejected (it would produce a negative length that collides with the
// UNKNOWN_DIM sentinel).
TEST_F(SliceLastDimInfershape, slice_last_dim_infershape_stride_negative_reject)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SliceLastDim",
        {
            {{{4, 8}, {4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("start", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
            gert::InfershapeContextPara::OpAttr("end", Ops::Math::AnyValue::CreateFrom<int64_t>(5)),
            gert::InfershapeContextPara::OpAttr("stride", Ops::Math::AnyValue::CreateFrom<int64_t>(-2)),
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// Rank 0 (scalar) input is outside [1, 8] and must be rejected.
TEST_F(SliceLastDimInfershape, slice_last_dim_infershape_rank_zero_reject)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SliceLastDim",
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("start", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
            gert::InfershapeContextPara::OpAttr("end", Ops::Math::AnyValue::CreateFrom<int64_t>(10)),
            gert::InfershapeContextPara::OpAttr("stride", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// Rank 9 input is outside [1, 8] and must be rejected.
TEST_F(SliceLastDimInfershape, slice_last_dim_infershape_rank_nine_reject)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SliceLastDim",
        {
            {{{1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("start", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
            gert::InfershapeContextPara::OpAttr("end", Ops::Math::AnyValue::CreateFrom<int64_t>(5)),
            gert::InfershapeContextPara::OpAttr("stride", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// Unknown rank ([-2]) is passed through: the output keeps an unknown-rank descriptor ([-2]) and
// the real shape is derived at runtime from the actual input.
TEST_F(SliceLastDimInfershape, slice_last_dim_infershape_unknown_rank_passthrough)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SliceLastDim",
        {
            {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("start", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
            gert::InfershapeContextPara::OpAttr("end", Ops::Math::AnyValue::CreateFrom<int64_t>(10)),
            gert::InfershapeContextPara::OpAttr("stride", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Control case — a known-rank dynamic shape ([-1], single dynamic dim) keeps the dynamic
// capability: the unknown last dim propagates to the output.
TEST_F(SliceLastDimInfershape, slice_last_dim_infershape_dynamic_single_dim_supported)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SliceLastDim",
        {
            {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("start", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
            gert::InfershapeContextPara::OpAttr("end", Ops::Math::AnyValue::CreateFrom<int64_t>(5)),
            gert::InfershapeContextPara::OpAttr("stride", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Opposite-sign int64 extremes clamp into [0, lastDim] (start -> lastDim, end -> 0) and yield an
// empty last dim without overflow.
TEST_F(SliceLastDimInfershape, slice_last_dim_infershape_extreme_clamp)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SliceLastDim",
        {
            {{{2, 10}, {2, 10}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("start", Ops::Math::AnyValue::CreateFrom<int64_t>(INT64_MAX)),
            gert::InfershapeContextPara::OpAttr("end", Ops::Math::AnyValue::CreateFrom<int64_t>(INT64_MIN)),
            gert::InfershapeContextPara::OpAttr("stride", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 0},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Length computation stays overflow-free for extreme strides (e - s == 2, stride == INT64_MAX
// must produce length 1, not a wrapped negative).
TEST_F(SliceLastDimInfershape, slice_last_dim_infershape_overflow_free_length)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SliceLastDim",
        {
            {{{4, 10}, {4, 10}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("start", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
            gert::InfershapeContextPara::OpAttr("end", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
            gert::InfershapeContextPara::OpAttr("stride", Ops::Math::AnyValue::CreateFrom<int64_t>(INT64_MAX)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Negative indices within [-lastDim, 0) count from the end of the last dim.
TEST_F(SliceLastDimInfershape, slice_last_dim_infershape_negative_indices_in_range)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SliceLastDim",
        {
            {{{4, 8}, {4, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("start", Ops::Math::AnyValue::CreateFrom<int64_t>(-3)),
            gert::InfershapeContextPara::OpAttr("end", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)),
            gert::InfershapeContextPara::OpAttr("stride", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Out-of-range negatives clamp to [0, lastDim] on both ends.
TEST_F(SliceLastDimInfershape, slice_last_dim_infershape_negative_indices_out_of_range)
{
    gert::InfershapeContextPara infershapeContextPara(
        "SliceLastDim",
        {
            {{{5, 100}, {5, 100}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("start", Ops::Math::AnyValue::CreateFrom<int64_t>(-200)),
            gert::InfershapeContextPara::OpAttr("end", Ops::Math::AnyValue::CreateFrom<int64_t>(200)),
            gert::InfershapeContextPara::OpAttr("stride", Ops::Math::AnyValue::CreateFrom<int64_t>(3)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5, 34},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
