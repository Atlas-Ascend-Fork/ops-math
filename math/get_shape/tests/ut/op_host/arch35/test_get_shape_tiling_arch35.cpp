/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN " AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../../op_host/arch35/get_shape_tiling_arch35.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "exe_graph/runtime/kernel_run_context.h"

using namespace std;

class GetShapeTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GetShapeTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "GetShapeTilingTest TearDown" << std::endl; }
};

TEST_F(GetShapeTilingTest, SingleInput_3D_Tensor)
{
    optiling::GetShapeCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GetShape",
                                              {
                                                  {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {1}, {1}, &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(GetShapeTilingTest, ScalarInput_Fail)
{
    optiling::GetShapeCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GetShape",
                                              {
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {1}, {1}, &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(GetShapeTilingTest, NullInputShape_Fail)
{
    optiling::GetShapeCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GetShape",
                                              {
                                                  {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {1}, {1}, &compileInfo);
    auto bundle = BuildTilingContext(tilingContextPara);
    ASSERT_NE(bundle.context, nullptr);
    auto* kernelCtx = reinterpret_cast<KernelRunContext*>(bundle.context);
    kernelCtx->values[0]->data.pointer = nullptr;
    EXPECT_EQ(RunTiling(bundle, tilingContextPara), ge::GRAPH_FAILED);
}

TEST_F(GetShapeTilingTest, UnknownRankStorageShape_Fail)
{
    optiling::GetShapeCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GetShape",
                                              {
                                                  {{{0}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {1}, {1}, &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(GetShapeTilingTest, TotalDimExceed128_Fail)
{
    optiling::GetShapeCompileInfo compileInfo;
    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    for (int i = 0; i < 17; ++i) {
        inputs.emplace_back(gert::StorageShape({1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 3, 4, 5, 6, 7, 8}), ge::DT_FLOAT,
                            ge::FORMAT_ND);
    }
    gert::TilingContextPara tilingContextPara("GetShape", inputs, {{{{136}, {136}}, ge::DT_INT32, ge::FORMAT_ND}}, {17},
                                              {1}, &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(GetShapeTilingTest, HighDimTensor)
{
    optiling::GetShapeCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "GetShape",
        {
            {{{1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 3, 4, 5, 6, 7, 8}}, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {1}, {1}, &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(GetShapeTilingTest, EmptyTensorList_Fail)
{
    optiling::GetShapeCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("GetShape", {},
                                              {
                                                  {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {0}, {1}, &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}

TEST_F(GetShapeTilingTest, DimNumExceed8_Fail)
{
    optiling::GetShapeCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "GetShape",
        {
            {{{1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 2, 3, 4, 5, 6, 7, 8, 9}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{9}, {9}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {1}, {1}, &compileInfo);
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectTilingKey, expectWorkspaces);
}
