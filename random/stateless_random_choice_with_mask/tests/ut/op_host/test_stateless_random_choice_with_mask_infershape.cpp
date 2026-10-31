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
#include "op_infer_shape_range_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"

class StatelessRandomChoiceWithMaskProtoTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "StatelessRandomChoiceWithMaskProtoTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "StatelessRandomChoiceWithMaskProtoTest TearDown" << std::endl; }
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
                        const std::vector<gert::StorageShape>& inputShapes,
                        std::vector<gert::StorageShape*>& outStorageShape,
                        ge::graphStatus testCaseResult = ge::GRAPH_SUCCESS)
{
    const auto& xStorageShape = inputShapes[0];
    const auto& countStorageShape = inputShapes[1];
    const auto& seedStorageShape = inputShapes[2];
    const auto& offsetStorageShape = inputShapes[3];

    std::vector<gert::Tensor*> inputTensors = {(gert::Tensor*)&xStorageShape, (gert::Tensor*)&countStorageShape,
                                               (gert::Tensor*)&seedStorageShape, (gert::Tensor*)&offsetStorageShape};
    std::vector<gert::StorageShape*> outputShapes = outStorageShape;

    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("StatelessRandomChoiceWithMask")
                             .NodeIoNum(4, 2)
                             .NodeInputTd(0, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputTensors(inputTensors)
                             .OutputShapes(outputShapes)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("StatelessRandomChoiceWithMask")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), testCaseResult);
    for (size_t i = 0; i < expectResults.size(); i++) {
        EXPECT_EQ(ToVector(*contextHolder.GetContext()->GetOutputShape(i)), expectResults[i]);
    }
}

TEST_F(StatelessRandomChoiceWithMaskProtoTest, stateless_random_choice_with_mask_test_infershape_unknownrank)
{
    std::vector<gert::StorageShape> inputShapes = {{{-2}, {-2}}, {{-2}, {-2}}, {{-2}, {-2}}, {{-2}, {-2}}};
    std::vector<std::vector<int64_t>> expectResults = {{-2}, {-2}};
    std::vector<gert::StorageShape*> outStorageShape = {};

    ExeTestCase(expectResults, inputShapes, outStorageShape, ge::GRAPH_SUCCESS);
}

TEST_F(StatelessRandomChoiceWithMaskProtoTest, stateless_random_choice_with_mask_test_infershape_unknownshape)
{
    std::vector<gert::StorageShape> inputShapes = {{{-1}, {-1}}, {{-2}, {-2}}, {{-2}, {-2}}, {{-2}, {-2}}};
    std::vector<std::vector<int64_t>> expectResults = {{-1, -1}, {-1}};
    std::vector<gert::StorageShape*> outStorageShape = {};

    ExeTestCase(expectResults, inputShapes, outStorageShape, ge::GRAPH_SUCCESS);
}

TEST_F(StatelessRandomChoiceWithMaskProtoTest, stateless_random_choice_with_mask_test_infershape_success_1d)
{
    std::vector<gert::StorageShape> inputShapes = {{{3}, {3}}, {{1}, {1}}, {{1}, {1}}, {{1}, {1}}};
    std::vector<std::vector<int64_t>> expectResults = {{-1, 1}, {-1}};
    std::vector<gert::StorageShape*> outStorageShape = {};

    ExeTestCase(expectResults, inputShapes, outStorageShape, ge::GRAPH_SUCCESS);
}

TEST_F(StatelessRandomChoiceWithMaskProtoTest, stateless_random_choice_with_mask_test_infershape_success_2d)
{
    std::vector<gert::StorageShape> inputShapes = {{{3, 3}, {3, 3}}, {{1}, {1}}, {{1}, {1}}, {{1}, {1}}};
    std::vector<std::vector<int64_t>> expectResults = {{-1, 2}, {-1}};
    std::vector<gert::StorageShape*> outStorageShape = {};

    ExeTestCase(expectResults, inputShapes, outStorageShape, ge::GRAPH_SUCCESS);
}

TEST_F(StatelessRandomChoiceWithMaskProtoTest, stateless_random_choice_with_mask_test_infershape_success_3d)
{
    std::vector<gert::StorageShape> inputShapes = {{{3, 3, 1}, {3, 3, 1}}, {{1}, {1}}, {{1}, {1}}, {{1}, {1}}};
    std::vector<std::vector<int64_t>> expectResults = {{-1, 3}, {-1}};
    std::vector<gert::StorageShape*> outStorageShape = {};

    ExeTestCase(expectResults, inputShapes, outStorageShape, ge::GRAPH_SUCCESS);
}

TEST_F(StatelessRandomChoiceWithMaskProtoTest, stateless_random_choice_with_mask_test_infershape_success_4d)
{
    std::vector<gert::StorageShape> inputShapes = {{{3, 4, 5, 6}, {3, 4, 5, 6}}, {{1}, {1}}, {{1}, {1}}, {{1}, {1}}};
    std::vector<std::vector<int64_t>> expectResults = {{-1, 4}, {-1}};
    std::vector<gert::StorageShape*> outStorageShape = {};

    ExeTestCase(expectResults, inputShapes, outStorageShape, ge::GRAPH_SUCCESS);
}

TEST_F(StatelessRandomChoiceWithMaskProtoTest, stateless_random_choice_with_mask_test_infershape_success_5d)
{
    std::vector<gert::StorageShape> inputShapes = {
        {{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}}, {{1}, {1}}, {{1}, {1}}, {{1}, {1}}};
    std::vector<std::vector<int64_t>> expectResults = {{-1, 5}, {-1}};
    std::vector<gert::StorageShape*> outStorageShape = {};

    ExeTestCase(expectResults, inputShapes, outStorageShape, ge::GRAPH_SUCCESS);
}

static std::vector<int64_t> ShapeRangeToVec(const gert::Shape& shape)
{
    std::vector<int64_t> shapeVec;
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        shapeVec.push_back(shape.GetDim(i));
    }
    return shapeVec;
}

static gert::Tensor MakeHostTensor(std::initializer_list<int64_t> dims, ge::DataType dtype)
{
    return gert::Tensor(gert::StorageShape(dims, dims),
                        gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                        gert::TensorPlacement::kOnHost, dtype, nullptr);
}

static void ExeShapeRangeTestCase(int32_t countValue, std::initializer_list<int64_t> xMinDims,
                                  std::initializer_list<int64_t> xMaxDims, const std::vector<int64_t>& expectYMin,
                                  const std::vector<int64_t>& expectYMax, const std::vector<int64_t>& expectMaskMin,
                                  const std::vector<int64_t>& expectMaskMax)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("StatelessRandomChoiceWithMask");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape_range, nullptr);

    gert::OpInferShapeRangeContextBuilder builder;
    builder.OpType("StatelessRandomChoiceWithMask").OpName("StatelessRandomChoiceWithMask");
    builder.IONum(4, 2);
    builder.OutputTensorDesc(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(1, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND);

    gert::Tensor xMinTensor(gert::StorageShape(xMinDims, xMinDims),
                            gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                            gert::TensorPlacement::kOnHost, ge::DT_BOOL, nullptr);
    gert::Tensor xMaxTensor(gert::StorageShape(xMaxDims, xMaxDims),
                            gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                            gert::TensorPlacement::kOnHost, ge::DT_BOOL, nullptr);
    gert::Range<gert::Tensor> xRange(&xMinTensor, &xMaxTensor);

    gert::Tensor countTensor(gert::StorageShape({1}, {1}),
                             gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                             gert::TensorPlacement::kOnHost, ge::DT_INT32, &countValue);
    gert::Range<gert::Tensor> countRange(&countTensor, &countTensor);

    gert::Tensor seedTensor = MakeHostTensor({1}, ge::DT_INT64);
    gert::Range<gert::Tensor> seedRange(&seedTensor, &seedTensor);
    gert::Tensor offsetTensor = MakeHostTensor({1}, ge::DT_INT64);
    gert::Range<gert::Tensor> offsetRange(&offsetTensor, &offsetTensor);

    builder.InputTensorsRange({&xRange, &countRange, &seedRange, &offsetRange});
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(opImpl->infer_shape_range(context), ge::GRAPH_SUCCESS);

    auto* yRange = context->GetOutputShapeRange(0);
    ASSERT_NE(yRange, nullptr);
    ASSERT_NE(yRange->GetMin(), nullptr);
    ASSERT_NE(yRange->GetMax(), nullptr);
    EXPECT_EQ(ShapeRangeToVec(*(yRange->GetMin())), expectYMin);
    EXPECT_EQ(ShapeRangeToVec(*(yRange->GetMax())), expectYMax);

    auto* maskRange = context->GetOutputShapeRange(1);
    ASSERT_NE(maskRange, nullptr);
    ASSERT_NE(maskRange->GetMin(), nullptr);
    ASSERT_NE(maskRange->GetMax(), nullptr);
    EXPECT_EQ(ShapeRangeToVec(*(maskRange->GetMin())), expectMaskMin);
    EXPECT_EQ(ShapeRangeToVec(*(maskRange->GetMax())), expectMaskMax);
}

TEST_F(StatelessRandomChoiceWithMaskProtoTest, stateless_random_choice_with_mask_test_infershape_range_2d)
{
    // x max {3, 8} → shape size 24 > count 5, outputLengthMax = 24
    ExeShapeRangeTestCase(5, {3, 8}, {3, 8}, {0, 2}, {24, 2}, {0}, {24});
}

TEST_F(StatelessRandomChoiceWithMaskProtoTest, stateless_random_choice_with_mask_test_infershape_range_count_gt)
{
    // x max {2} → shape size 2 < count 10, outputLengthMax = 10
    ExeShapeRangeTestCase(10, {2}, {2}, {0, 1}, {10, 1}, {0}, {10});
}

TEST_F(StatelessRandomChoiceWithMaskProtoTest, stateless_random_choice_with_mask_test_output_shape_depend_on_compute)
{
    // 输出 y/mask shape 依赖计算（三类算子），注册链必须带 OutputShapeDependOnCompute 标记，
    // 否则 GE 不会给节点打 DEPEND_SHAPE_RANGE，rt v2 执行期无法按 shape range 分配输出内存。
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("StatelessRandomChoiceWithMask");
    ASSERT_NE(opImpl, nullptr);

    EXPECT_TRUE(opImpl->IsOutputShapeDependOnCompute());
    EXPECT_TRUE(opImpl->IsOutputShapeDependOnCompute(0));
    EXPECT_TRUE(opImpl->IsOutputShapeDependOnCompute(1));
}
