/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_minimum_grad_tiling_arch35.cpp
 * \brief MinimumGrad 算子 arch35 Tiling 单元测试
 */

#include "atvoss/broadcast/broadcast_tiling.h"
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

namespace {
static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dimensions)
{
    gert::StorageShape storageShape;
    auto& originShape = storageShape.MutableOriginShape();
    auto& runtimeShape = storageShape.MutableStorageShape();
    originShape.SetDimNum(dimensions.size());
    runtimeShape.SetDimNum(dimensions.size());
    for (size_t i = 0; i < dimensions.size(); ++i) {
        originShape.SetDim(i, dimensions[i]);
        runtimeShape.SetDim(i, dimensions[i]);
    }
    return storageShape;
}

static Ops::Base::BroadcastCompileInfo MakeCompileInfo()
{
    Ops::Base::BroadcastCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    return compileInfo;
}

static gert::TilingContextPara MakeTilingContext(const std::vector<std::vector<int64_t>>& inputDims,
                                                 const std::vector<std::vector<int64_t>>& outputDims,
                                                 ge::DataType dataType, bool gradX = true, bool gradY = true)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    for (const auto& dims : inputDims) {
        inputs.push_back({MakeStorageShape(dims), dataType, ge::FORMAT_ND});
    }
    std::vector<gert::TilingContextPara::TensorDescription> outputs;
    for (const auto& dims : outputDims) {
        outputs.push_back({MakeStorageShape(dims), dataType, ge::FORMAT_ND});
    }
    auto compileInfo = MakeCompileInfo();
    return gert::TilingContextPara(
        "MinimumGrad", inputs, outputs,
        {gert::TilingContextPara::OpAttr("grad_x", Ops::Math::AnyValue::CreateFrom<bool>(gradX)),
         gert::TilingContextPara::OpAttr("grad_y", Ops::Math::AnyValue::CreateFrom<bool>(gradY))},
        &compileInfo);
}
} // namespace

class MinimumGradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MinimumGradTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MinimumGradTilingTest TearDown" << std::endl; }
};

// 双输出 + fp16 同 shape
TEST_F(MinimumGradTilingTest, tiling_both_fp16)
{
    auto para = MakeTilingContext({{8, 1024}, {8, 1024}, {8, 1024}}, {{8, 1024}, {8, 1024}}, ge::DT_FLOAT16);
    uint64_t expectTilingKey = 131079;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// 双输出 + fp16 三输入广播
TEST_F(MinimumGradTilingTest, tiling_both_fp16_broadcast)
{
    auto para = MakeTilingContext({{2, 1, 5}, {1, 3, 1}, {5}}, {{2, 3, 5}, {2, 3, 5}}, ge::DT_FLOAT16);
    uint64_t expectTilingKey = 131072;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// 仅 y1 输出（grad_x=True, grad_y=False）
TEST_F(MinimumGradTilingTest, tiling_y1_only_fp16)
{
    auto para = MakeTilingContext({{8, 1024}, {8, 1024}, {8, 1024}}, {{8, 1024}, {8, 1024}}, ge::DT_FLOAT16, true,
                                  false);
    uint64_t expectTilingKey = 8;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// 仅 y2 输出（grad_x=False, grad_y=True）
TEST_F(MinimumGradTilingTest, tiling_y2_only_fp16)
{
    auto para = MakeTilingContext({{8, 1024}, {8, 1024}, {8, 1024}}, {{8, 1024}, {8, 1024}}, ge::DT_FLOAT16, false,
                                  true);
    uint64_t expectTilingKey = 65544;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// 双输出 + fp32
TEST_F(MinimumGradTilingTest, tiling_both_fp32)
{
    auto para = MakeTilingContext({{8, 1024}, {8, 1024}, {8, 1024}}, {{8, 1024}, {8, 1024}}, ge::DT_FLOAT);
    uint64_t expectTilingKey = 131079;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// 双输出 + bf16
TEST_F(MinimumGradTilingTest, tiling_both_bf16)
{
    auto para = MakeTilingContext({{8, 1024}, {8, 1024}, {8, 1024}}, {{8, 1024}, {8, 1024}}, ge::DT_BF16);
    uint64_t expectTilingKey = 131079;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// 双输出 + int32
TEST_F(MinimumGradTilingTest, tiling_both_int32)
{
    auto para = MakeTilingContext({{8, 1024}, {8, 1024}, {8, 1024}}, {{8, 1024}, {8, 1024}}, ge::DT_INT32);
    uint64_t expectTilingKey = 131079;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// grad_x 与 grad_y 同时为 False：失败
TEST_F(MinimumGradTilingTest, tiling_both_grads_false_fail)
{
    auto para = MakeTilingContext({{8, 1024}, {8, 1024}, {8, 1024}}, {{8, 1024}, {8, 1024}}, ge::DT_FLOAT16, false,
                                  false);
    ExecuteTestCase(para, ge::GRAPH_FAILED, 0, std::vector<size_t>{});
}

// 不支持的 dtype（int8）：失败
TEST_F(MinimumGradTilingTest, tiling_unsupported_dtype_fail)
{
    auto para = MakeTilingContext({{8, 1024}, {8, 1024}, {8, 1024}}, {{8, 1024}, {8, 1024}}, ge::DT_INT8);
    ExecuteTestCase(para, ge::GRAPH_FAILED, 0, std::vector<size_t>{});
}

// 空 Tensor（dim=0）：走 schMode=999 自定义分支, 不落 BroadcastBaseTiling。
// tilingKey = (outputCountIdx << 16) | 14 —— 14 是 999 在 BRC_TEMP_SCH_MODE_KEY_DECL
// 取值表 (1,2,101,...,305,999) 里的序号; outputCountIdx: y1_only=0, y2_only=1, both=2。
TEST_F(MinimumGradTilingTest, tiling_empty_both)
{
    auto para = MakeTilingContext({{0}, {0}, {0}}, {{0}, {0}}, ge::DT_FLOAT16);
    uint64_t expectTilingKey = 131086;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(MinimumGradTilingTest, tiling_empty_y1_only)
{
    auto para = MakeTilingContext({{2, 0}, {2, 0}, {0}}, {{2, 0}, {2, 0}}, ge::DT_FLOAT16, true, false);
    uint64_t expectTilingKey = 14;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(MinimumGradTilingTest, tiling_empty_y2_only)
{
    auto para = MakeTilingContext({{0, 3}, {0, 3}, {3}}, {{0, 3}, {0, 3}}, ge::DT_FLOAT16, false, true);
    uint64_t expectTilingKey = 65550;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(MinimumGradTilingTest, tiling_empty_broadcast)
{
    // x2=(0,) 广播到 (2,0), 广播结果元素数为 0
    auto para = MakeTilingContext({{2, 0}, {2, 0}, {0}}, {{2, 0}, {2, 0}}, ge::DT_FLOAT16);
    uint64_t expectTilingKey = 131086;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// 维度超过 8（atvoss broadcast 上限）：失败
TEST_F(MinimumGradTilingTest, tiling_too_many_dims_fail)
{
    auto para = MakeTilingContext(
        {{1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 2, 3, 4, 5, 6, 7, 8, 9}},
        {{1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 2, 3, 4, 5, 6, 7, 8, 9}}, ge::DT_FLOAT16);
    ExecuteTestCase(para, ge::GRAPH_FAILED, 0, std::vector<size_t>{});
}

// 广播不兼容：失败
TEST_F(MinimumGradTilingTest, tiling_incompatible_shapes_fail)
{
    auto para = MakeTilingContext({{2, 3}, {4, 3}, {2, 3}}, {{2, 3}, {2, 3}}, ge::DT_FLOAT16);
    ExecuteTestCase(para, ge::GRAPH_FAILED, 0, std::vector<size_t>{});
}

// 输出 shape 与广播结果不一致：失败
TEST_F(MinimumGradTilingTest, tiling_output_shape_mismatch_fail)
{
    auto para = MakeTilingContext({{8, 1024}, {8, 1024}, {8, 1024}}, {{8, 512}, {8, 1024}}, ge::DT_FLOAT16);
    ExecuteTestCase(para, ge::GRAPH_FAILED, 0, std::vector<size_t>{});
}
