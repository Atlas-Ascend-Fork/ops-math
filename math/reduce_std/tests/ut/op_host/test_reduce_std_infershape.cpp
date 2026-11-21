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
#include "base/registry/op_impl_space_registry_v2.h"

class ReduceStd : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ReduceStd SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ReduceStd TearDown" << std::endl; }
};

static std::vector<int64_t> ToVector(const gert::Shape& shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);
    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

static void ExeTestCase(std::vector<std::vector<int64_t>> expectResults,
                        const std::vector<gert::StorageShape>& inputShapes, // 存储所有输入StorageShape参数
                        const std::vector<ge::DataType>& dtypes,            // 存储所有DataType参数
                        gert::StorageShape& outStorageShape, std::vector<int64_t> dim, bool unbiased, bool keepdim,
                        ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
{
    // 从vector中取出对应参数（保持原顺序）
    const auto& x1StorageShape = inputShapes[0];

    ge::DataType input1Dtype = dtypes[0];
    ge::DataType outputDtype = dtypes[0];

    /* make infershape context */
    std::vector<gert::Tensor*> inputTensors = {(gert::Tensor*)&x1StorageShape};

    std::vector<int64_t> batchshape = {};
    std::vector<gert::StorageShape*> outputShapes = {&outStorageShape};
    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("ReduceStd")
                             .NodeIoNum(1, 2)
                             .NodeInputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputTensors(inputTensors)
                             .OutputShapes(outputShapes)
                             .Attr("dim", dim)
                             .Attr("unbiased", unbiased)
                             .Attr("keepdim", keepdim)
                             .Build();

    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("ReduceStd")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    /* do infershape */
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
    for (size_t i = 0; i < expectResults.size(); i++) {
        EXPECT_EQ(ToVector(*contextHolder.GetContext()->GetOutputShape(i)), expectResults[i]);
    }
}

TEST_F(ReduceStd, ReduceStd_infershape_case_0)
{
    // 用vector存储同类型参数（顺序与原参数列表一致）
    std::vector<gert::StorageShape> inputShapes = {{{64, 10240}, {64, 10240}}};

    std::vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16};

    std::vector<std::vector<int64_t>> expectResult = {{64, 1}, {64, 1}};
    gert::StorageShape outStorageShape = {};

    // 简化后的函数调用
    ExeTestCase(expectResult, inputShapes, dtypes, outStorageShape, {1}, true, true, ge::GRAPH_SUCCESS);
}

TEST_F(ReduceStd, ReduceStd_infershape_case_1)
{
    // keepdim = false 时归约维不保留
    std::vector<gert::StorageShape> inputShapes = {{{4, 16, 8}, {4, 16, 8}}};

    std::vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT};

    std::vector<std::vector<int64_t>> expectResult = {{4, 8}, {4, 8}};
    gert::StorageShape outStorageShape = {};

    ExeTestCase(expectResult, inputShapes, dtypes, outStorageShape, {1}, false, false, ge::GRAPH_SUCCESS);
}

TEST_F(ReduceStd, ReduceStd_infershape_case_2)
{
    // dim 为空表示全部维度归约
    std::vector<gert::StorageShape> inputShapes = {{{4, 16, 8}, {4, 16, 8}}};

    std::vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT};

    std::vector<std::vector<int64_t>> expectResult = {{1, 1, 1}, {1, 1, 1}};
    gert::StorageShape outStorageShape = {};

    ExeTestCase(expectResult, inputShapes, dtypes, outStorageShape, {}, true, true, ge::GRAPH_SUCCESS);
}
