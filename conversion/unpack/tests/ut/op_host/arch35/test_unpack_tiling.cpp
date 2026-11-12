/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/unpack_tiling_arch35.h"

using namespace std;
using namespace ge;

class UnpackTiling : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

TEST_F(UnpackTiling, Unpack_test_tiling_001)
{
    optiling::UnpackCompileInfo compileInfo = {64, 253952};

    gert::TilingContextPara tilingContextPara(
        "Unpack",
        {
            {
                {{2, 3000, 1, 64}, {2, 3000, 1, 64}},
                ge::DT_FLOAT,
                ge::FORMAT_ND,
            },
        },
        {
            {{{3000, 1, 64}, {3000, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("num", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
            gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 100;
    // 等长场景走精简tiling数据 SplitVSlimTilingData(16个int64字段, tilingKey=100 等长纯搬运模板)
    string expectTilingData = "253952 2 192000 1 1 1 1 1 2 31744 1536 7 1 0 14 1 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
