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
 * \file test_minimum_grad_infershape.cpp
 * \brief MinimumGrad 算子 InferShape 单元测试
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

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

static gert::InfershapeContextPara MakeInferShapeContext(const std::vector<std::vector<int64_t>>& inputShapes,
                                                         const std::vector<std::vector<int64_t>>& outputShapes,
                                                         ge::DataType dataType, bool gradX = true, bool gradY = true)
{
    std::vector<gert::InfershapeContextPara::TensorDescription> inputs;
    for (const auto& shape : inputShapes) {
        inputs.emplace_back(MakeStorageShape(shape), dataType, ge::FORMAT_ND);
    }
    std::vector<gert::InfershapeContextPara::TensorDescription> outputs;
    for (const auto& shape : outputShapes) {
        outputs.emplace_back(MakeStorageShape(shape), dataType, ge::FORMAT_ND);
    }
    return gert::InfershapeContextPara(
        "MinimumGrad", inputs, outputs,
        {gert::InfershapeContextPara::OpAttr("grad_x", Ops::Math::AnyValue::CreateFrom<bool>(gradX)),
         gert::InfershapeContextPara::OpAttr("grad_y", Ops::Math::AnyValue::CreateFrom<bool>(gradY))});
}
} // namespace

class MinimumGradInferShape : public ::testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MinimumGradInferShape SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "MinimumGradInferShape TearDown" << std::endl; }
};

// 双输出 + 相同 shape：y1 = y2 = 输入 shape
TEST_F(MinimumGradInferShape, same_shape_both_outputs)
{
    auto context = MakeInferShapeContext({{8, 1024}, {8, 1024}, {8, 1024}}, {{}, {}}, ge::DT_FLOAT);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{8, 1024}, {8, 1024}});
}

// 双输出 + 三输入广播：y1 = y2 = broadcast shape
TEST_F(MinimumGradInferShape, broadcast_shape_both_outputs)
{
    auto context = MakeInferShapeContext({{2, 1, 5}, {1, 3, 1}, {5}}, {{}, {}}, ge::DT_FLOAT16);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{2, 3, 5}, {2, 3, 5}});
}

// 标量与 tensor 广播
TEST_F(MinimumGradInferShape, scalar_broadcast)
{
    auto context = MakeInferShapeContext({{}, {7, 11}, {7, 11}}, {{}, {}}, ge::DT_FLOAT);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{7, 11}, {7, 11}});
}

// grad_x=True, grad_y=False：两个输出 shape 都写 broadcast shape（图模式要求）
TEST_F(MinimumGradInferShape, y1_only)
{
    auto context = MakeInferShapeContext({{4, 8}, {4, 8}, {8}}, {{}, {}}, ge::DT_FLOAT, true, false);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{4, 8}, {4, 8}});
}

// grad_x=False, grad_y=True：两个输出 shape 都写 broadcast shape（图模式要求）
TEST_F(MinimumGradInferShape, y2_only)
{
    auto context = MakeInferShapeContext({{4, 8}, {8}, {4, 8}}, {{}, {}}, ge::DT_FLOAT, false, true);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{4, 8}, {4, 8}});
}

// grad_x 与 grad_y 同时为 False：失败
TEST_F(MinimumGradInferShape, both_grads_false_fail)
{
    auto context = MakeInferShapeContext({{8, 8}, {8, 8}, {8, 8}}, {{}, {}}, ge::DT_FLOAT, false, false);
    ExecuteTestCase(context, ge::GRAPH_FAILED, {});
}

// 空 tensor（dim=0）：输出 shape 含 0 维, 与 TF 行为对齐（由 tiling/kernel 空分支处理）
TEST_F(MinimumGradInferShape, empty_tensor_success)
{
    // 三输入同含 0 维且互相兼容（0 与非 1 正数不可广播，同 TF 语义）
    auto context = MakeInferShapeContext({{0, 8}, {0, 8}, {0, 8}}, {{}, {}}, ge::DT_FLOAT);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{0, 8}, {0, 8}});
}

// 广播不兼容：失败
TEST_F(MinimumGradInferShape, incompatible_shapes_fail)
{
    auto context = MakeInferShapeContext({{2, 3}, {4, 3}, {2, 3}}, {{}, {}}, ge::DT_FLOAT);
    ExecuteTestCase(context, ge::GRAPH_FAILED, {});
}
