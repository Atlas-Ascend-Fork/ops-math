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

class SliceWithAxesInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SliceWithAxesInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SliceWithAxesInfershape TearDown" << std::endl; }
};

TEST_F(SliceWithAxesInfershape, slice_with_axes_1d_basic)
{
    int32_t offsets[] = {2};
    int32_t sizes[] = {5};
    gert::InfershapeContextPara infershapeContextPara(
        "SliceWithAxes",
        {
            {{{10}, {10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, offsets},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizes},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("axes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SliceWithAxesInfershape, slice_with_axes_2d_single_axis)
{
    int32_t offsets[] = {0};
    int32_t sizes[] = {4};
    gert::InfershapeContextPara infershapeContextPara(
        "SliceWithAxes",
        {
            {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, offsets},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizes},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("axes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SliceWithAxesInfershape, slice_with_axes_3d_multi_axes)
{
    int32_t offsets[] = {1, 5};
    int32_t sizes[] = {5, 10};
    gert::InfershapeContextPara infershapeContextPara(
        "SliceWithAxes",
        {
            {{{10, 20, 30}, {10, 20, 30}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, offsets},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, sizes},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("axes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 2})),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5, 20, 10},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SliceWithAxesInfershape, slice_with_axes_size_minus1)
{
    int32_t offsets[] = {30};
    int32_t sizes[] = {-1};
    gert::InfershapeContextPara infershapeContextPara(
        "SliceWithAxes",
        {
            {{{100}, {100}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, offsets},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizes},
        },
        {
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("axes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {70},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SliceWithAxesInfershape, slice_with_axes_int64_index)
{
    int64_t offsets[] = {10};
    int64_t sizes[] = {20};
    gert::InfershapeContextPara infershapeContextPara(
        "SliceWithAxes",
        {
            {{{50, 60}, {50, 60}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, offsets},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, sizes},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("axes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {20, 60},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Reproduce gitcode issue #13 (case SliceWithAxes_L0_021): offsets is int32 while size is int64.
// offsets=[2,2] stored as int32 bytes; reading them with int64 stride merges both elements into
// 0x0000000200000002 = 8589934594, so dim[2] becomes 25 - 8589934594 = -8589934569
// (the "invalid dimension, it must >= -2, but it is:-8589934569" E19999 error).
TEST_F(SliceWithAxesInfershape, slice_with_axes_issue13_mixed_dtype_offsets_int32_size_int64)
{
    int32_t offsets[] = {2, 2};
    int64_t sizes[] = {-1, -1};
    gert::InfershapeContextPara infershapeContextPara(
        "SliceWithAxes",
        {
            {{{19, 2, 25, 13, 8}, {19, 2, 25, 13, 8}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, offsets},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, sizes},
        },
        {
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("axes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 4})),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {19, 2, 23, 13, 6},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Reproduce gitcode issue #12 (case SliceWithAxes_L0_034): offsets is int64 while size is int32.
// offsets=[1,1,1,1] read with int32 stride yields [1,0,1,0], so dims 1/3 keep their offsets unset
// and the output becomes (5,109,4,1148) instead of (5,108,4,1147).
TEST_F(SliceWithAxesInfershape, slice_with_axes_issue12_mixed_dtype_offsets_int64_size_int32)
{
    int64_t offsets[] = {1, 1, 1, 1};
    int32_t sizes[] = {-1, -1, -1, -1};
    gert::InfershapeContextPara infershapeContextPara(
        "SliceWithAxes",
        {
            {{{6, 109, 5, 1148}, {6, 109, 5, 1148}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, offsets},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, sizes},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("axes",
                                                Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 1, 2, 3})),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {5, 108, 4, 1147},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Dynamic dims (-1) not targeted by axes pass through unchanged (case SliceWithAxes_L0_021 dyn shape).
TEST_F(SliceWithAxesInfershape, slice_with_axes_dynamic_dim_passthrough)
{
    int32_t offsets[] = {2, 2};
    int64_t sizes[] = {-1, -1};
    gert::InfershapeContextPara infershapeContextPara(
        "SliceWithAxes",
        {
            {{{19, -1, 25, -1, 8}, {19, -1, 25, -1, 8}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, offsets},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, sizes},
        },
        {
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("axes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 4})),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {19, -1, 23, -1, 6},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Axis targets an unknown dim (-1) with size -1: the sliced dim stays unknown.
TEST_F(SliceWithAxesInfershape, slice_with_axes_axis_on_unknown_dim)
{
    int32_t offsets[] = {2};
    int32_t sizes[] = {-1};
    gert::InfershapeContextPara infershapeContextPara(
        "SliceWithAxes",
        {
            {{{-1, 10}, {-1, 10}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, offsets},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizes},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("axes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 10},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// Unknown rank input ({-2}): output keeps unknown rank instead of touching GetDim(axis) out of range.
TEST_F(SliceWithAxesInfershape, slice_with_axes_unknown_rank_input)
{
    int32_t offsets[] = {2};
    int32_t sizes[] = {-1};
    gert::InfershapeContextPara infershapeContextPara(
        "SliceWithAxes",
        {
            {{{-2}, {-2}}, ge::DT_UINT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, offsets},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, sizes},
        },
        {
            {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("axes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0})),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
