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
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/tanh_grad_tiling_arch35.h"

using namespace std;
using namespace ge;
using namespace optiling;

class TanhGradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TanhGradTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TanhGradTilingTest TearDown" << std::endl; }
};

// ===== #6104: BroadcastTiling 失败 -> 外层 EZ1008（错误码治理验证用例）=====
// TilingContextPara 直接注入 ubSize=0 -> tanh_grad GetPlatformInfo 查得 ubSize=0
// -> opbase DoBrodcastTiling "ubSize can not be 0"(:572) 失败 -> BroadcastTiling FAILED
// -> tanh_grad DoOpTiling else 分支（修复点）：改前 OP_LOGE 隐式 EZ9999，改后 EZ1008 正式上报。
// 注：连续布局 shape 会被 DimensionCollapse 全并为一维，shapeLen 路径不可经 UT 触发。
TEST_F(TanhGradTilingTest, tanh_grad_tiling_broadcast_failed_6104)
{
    gert::StorageShape shape = {{32, 32}, {32, 32}};

    TanhGradCompileInfo compileInfo = {64, 253952};

    gert::TilingContextPara tilingContextPara(
        "TanhGrad", {{shape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {shape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {}, &compileInfo, 64, 0);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(TanhGradTilingTest, tanh_grad_tiling_test_001)
{
    gert::StorageShape shape = {{32, 32}, {32, 32}};

    TanhGradCompileInfo compileInfo = {64, 253952};

    gert::TilingContextPara tilingContextPara(
        "TanhGrad", {{shape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {shape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    uint64_t expectTilingKey = 100000001000100;
    string expectTilingData = "1 21760 1 1024 1 1 0 1 21760 1024 0 0 0 0 0 0 0 1024 0 0 0 0 0 0 0 1024 0 0 0 0 0 0 0 1 "
                              "0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
