/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "math/cross/op_host/arch35/cross_tiling.h"

using namespace std;
using namespace ge;

class CrossTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CrossTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "CrossTiling TearDown" << std::endl; }
};

TEST_F(CrossTiling, Cross_test_tiling_missing_dim_defaults_to_last_axis)
{
    // dim 属性缺失时按 IR 默认值 -1 回落，取最后一维（size 为 3）
    optiling::CrossCompileInfo compileInfo = {64};

    gert::TilingContextPara tilingContextPara("Cross",
                                              {
                                                  {{{2, 4, 3}, {2, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 4, 3}, {2, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 4, 3}, {2, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0, expectWorkspaces);
}

TEST_F(CrossTiling, Cross_test_tiling_rank_exceed_max_dim)
{
    // rank 为 MAX_DIM + 1（9 维）时应在写入定长数组前返回失败，而不是越界
    optiling::CrossCompileInfo compileInfo = {64};

    gert::TilingContextPara tilingContextPara(
        "Cross",
        {
            {{{3, 2, 3, 2, 3, 2, 3, 2, 3}, {3, 2, 3, 2, 3, 2, 3, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 1, 3, 1, 3, 1, 3, 1, 3}, {3, 1, 3, 1, 3, 1, 3, 1, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{3, 2, 3, 2, 3, 2, 3, 2, 3}, {3, 2, 3, 2, 3, 2, 3, 2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
