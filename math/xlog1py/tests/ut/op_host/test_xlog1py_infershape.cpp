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
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

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

static gert::InfershapeContextPara MakeInfershapeContext(const std::vector<std::vector<int64_t>>& inputShapes,
                                                         ge::DataType dataType)
{
    std::vector<gert::InfershapeContextPara::TensorDescription> inputs;
    for (const auto& shape : inputShapes) {
        inputs.emplace_back(MakeStorageShape(shape), dataType, ge::FORMAT_ND);
    }
    std::vector<gert::InfershapeContextPara::TensorDescription> outputs = {
        {MakeStorageShape({}), dataType, ge::FORMAT_ND},
    };
    return gert::InfershapeContextPara("Xlog1py", inputs, outputs);
}
} // namespace

class Xlog1pyInferShape : public ::testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Xlog1pyInferShape SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "Xlog1pyInferShape TearDown" << std::endl; }
};

TEST_F(Xlog1pyInferShape, same_shape)
{
    auto context = MakeInfershapeContext({{8, 1024}, {8, 1024}}, ge::DT_FLOAT);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{8, 1024}});
}

TEST_F(Xlog1pyInferShape, broadcast)
{
    auto context = MakeInfershapeContext({{2, 1, 4}, {1, 3, 4}}, ge::DT_FLOAT16);
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{2, 3, 4}});
}

TEST_F(Xlog1pyInferShape, incompatible_broadcast_fail)
{
    auto context = MakeInfershapeContext({{2, 3}, {4, 3}}, ge::DT_FLOAT);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(Xlog1pyInferShape, rank_over_8_fail)
{
    auto context = MakeInfershapeContext({{1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 2, 3, 4, 5, 6, 7, 8, 9}}, ge::DT_FLOAT);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

// InferDataType 校验：输出 dtype 与输入一致（float32）
TEST_F(Xlog1pyInferShape, infer_datatype_float_success)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("Xlog1py");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Xlog1py").OpName("Xlog1py");
    builder.IONum(2, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = opImpl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}

// InferDataType 校验：输出 dtype 与输入一致（float16）
TEST_F(Xlog1pyInferShape, infer_datatype_float16_success)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("Xlog1py");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Xlog1py").OpName("Xlog1py");
    builder.IONum(2, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = opImpl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

// InferDataType 校验：输出 dtype 与输入一致（bf16）
TEST_F(Xlog1pyInferShape, infer_datatype_bf16_success)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("Xlog1py");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Xlog1py").OpName("Xlog1py");
    builder.IONum(2, 1);
    builder.InputTensorDesc(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = opImpl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
}

// InferDataType 校验：input[0] 不支持 dtype（DT_DOUBLE）拒绝
TEST_F(Xlog1pyInferShape, infer_datatype_unsupported_dtype_fail)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("Xlog1py");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Xlog1py").OpName("Xlog1py");
    builder.IONum(2, 1);
    builder.InputTensorDesc(0, ge::DT_DOUBLE, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_DOUBLE, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = opImpl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// InferDataType 校验：input[1] 不支持 dtype（input[0]=float、input[1]=double）拒绝
TEST_F(Xlog1pyInferShape, infer_datatype_unsupported_dtype_input1_fail)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("Xlog1py");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Xlog1py").OpName("Xlog1py");
    builder.IONum(2, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_DOUBLE, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = opImpl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// InferDataType 校验：输入 dtype 不一致（float + float16）拒绝
TEST_F(Xlog1pyInferShape, infer_datatype_mixed_dtype_fail)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("Xlog1py");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Xlog1py").OpName("Xlog1py");
    builder.IONum(2, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = opImpl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
