/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

class diagFlatInfer : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "diagFlatInfer Proto Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "diagFlatInfer Proto Test TearDown" << std::endl; }
};

static std::vector<int64_t> ToVectorForDiagFlat(const gert::Shape& shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);
    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

static void ExeTestCaseForDiagFlat(
    std::vector<std::vector<int64_t> > expectResults,
    const std::vector<gert::StorageShape>& inputShapes, // 被测算子输入 x 的 StorageShape（单输入）
    const std::vector<ge::DataType>& dtypes,            // x 与 y 的 DataType
    gert::StorageShape& outStorageShape, ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS, int64_t attr = 0)
{
    // DiagFlat 只有一个输入 x 和一个可选属性 diagonal（默认值 0）
    const auto& selfStorageShape = inputShapes[0];
    ge::DataType inputDtype = dtypes[0];
    ge::DataType outputDtype = dtypes[1];

    /* make infershape context */
    std::vector<gert::Tensor*> inputTensors = {(gert::Tensor*)&selfStorageShape};
    std::vector<gert::StorageShape*> outputShapes = {&outStorageShape};
    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("DiagFlat")
                             .NodeIoNum(1, 1)
                             .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputTensors(inputTensors)
                             .OutputShapes(outputShapes)
                             .Attr("diagonal", attr)
                             .Build();

    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("DiagFlat")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    /* do infershape */
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
    for (size_t i = 0; i < expectResults.size(); i++) {
        EXPECT_EQ(ToVectorForDiagFlat(*contextHolder.GetContext()->GetOutputShape(i)), expectResults[i]);
    }
}

TEST_F(diagFlatInfer, diag_flat_infershape_case_tiling_key_101)
{
    size_t size1 = 8;

    // 用vector存储同类型参数（顺序与原参数列表一致）
    std::vector<gert::StorageShape> inputShapes = {
        {{
             size1,
         },
         {
             size1,
         }} // x
    };
    std::vector<ge::DataType> dtypes = {
        ge::DT_FLOAT16, // x
        ge::DT_FLOAT16  // y
    };

    std::vector<int64_t> expectResult = {size1, size1};
    gert::StorageShape outStorageShape = {};
    int64_t attr = 0;
    // 简化后的函数调用
    ExeTestCaseForDiagFlat({expectResult}, inputShapes, dtypes, outStorageShape, ge::GRAPH_SUCCESS, attr);
}
