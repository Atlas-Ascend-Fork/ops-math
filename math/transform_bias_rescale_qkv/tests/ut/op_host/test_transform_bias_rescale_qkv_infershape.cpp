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
#include <map>
#include <string>
#include "base/err_mgr.h"
#include "infershape_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class TransformBiasRescaleQkv : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TransformBiasRescaleQkv SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TransformBiasRescaleQkv TearDown" << std::endl; }
};

TEST_F(TransformBiasRescaleQkv, TransformBiasRescaleQkv_infershape_case_0)
{
    gert::StorageShape qkvStorageShape = {{3, 4, 144}, {3, 4, 144}};
    gert::StorageShape biasStorageShape = {{144}, {144}};
    gert::StorageShape outputShape = {{}, {}};

    ge::DataType qkvDtype = ge::DT_FLOAT16;
    ge::DataType biasDtype = ge::DT_FLOAT16;
    ge::DataType outputDtype = ge::DT_FLOAT16;

    /* make infershape context */
    std::vector<gert::Tensor*> inputTensors = {
        (gert::Tensor*)&qkvStorageShape,
        (gert::Tensor*)&biasStorageShape,
    };

    std::vector<gert::StorageShape*> outputShapes = {&outputShape, &outputShape, &outputShape};

    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("TransformBiasRescaleQkv")
                             .NodeIoNum(2, 3)
                             .NodeInputTd(0, qkvDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputTensors(inputTensors)
                             .OutputShapes(outputShapes)
                             .Attr("num_heads", (int64_t)3)
                             .Build();

    /* get infershape func */
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("TransformBiasRescaleQkv")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    /* do infershape */
    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), ge::GRAPH_SUCCESS);
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context->GetOutputShape(0), nullptr);
    EXPECT_EQ(context->GetOutputShape(0)->GetDimNum(), 4U);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(0), 3);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(1), 3);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(2), 4);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(3), 16);
}

TEST_F(TransformBiasRescaleQkv, TransformBiasRescaleQkv_infershape_case_1)
{
    gert::StorageShape qkvStorageShape = {{3, 4, 144}, {3, 4, 144}};
    gert::StorageShape biasStorageShape = {{144}, {144}};
    gert::StorageShape outputShape = {{}, {}};

    ge::DataType qkvDtype = ge::DT_FLOAT16;
    ge::DataType biasDtype = ge::DT_FLOAT16;
    ge::DataType outputDtype = ge::DT_FLOAT16;

    std::vector<gert::Tensor*> inputTensors = {
        (gert::Tensor*)&qkvStorageShape,
        (gert::Tensor*)&biasStorageShape,
    };

    std::vector<gert::StorageShape*> outputShapes = {&outputShape, &outputShape, &outputShape};

    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("TransformBiasRescaleQkv")
                             .NodeIoNum(2, 3)
                             .NodeInputTd(0, qkvDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputTensors(inputTensors)
                             .OutputShapes(outputShapes)
                             .Attr("num_heads", (int64_t)1)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("TransformBiasRescaleQkv")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    EXPECT_EQ(inferShapeFunc(contextHolder.GetContext()), ge::GRAPH_SUCCESS);
}

TEST_F(TransformBiasRescaleQkv, TransformBiasRescaleQkv_infershape_num_heads_zero_failed)
{
    gert::StorageShape qkvStorageShape = {{3, 4, 144}, {3, 4, 144}};
    gert::StorageShape biasStorageShape = {{144}, {144}};
    gert::StorageShape outputShape = {{}, {}};

    ge::DataType qkvDtype = ge::DT_FLOAT16;
    ge::DataType biasDtype = ge::DT_FLOAT16;
    ge::DataType outputDtype = ge::DT_FLOAT16;

    std::vector<gert::Tensor*> inputTensors = {
        (gert::Tensor*)&qkvStorageShape,
        (gert::Tensor*)&biasStorageShape,
    };

    std::vector<gert::StorageShape*> outputShapes = {&outputShape, &outputShape, &outputShape};

    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("TransformBiasRescaleQkv")
                             .NodeIoNum(2, 3)
                             .NodeInputTd(0, qkvDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputTensors(inputTensors)
                             .OutputShapes(outputShapes)
                             .Attr("num_heads", (int64_t)0)
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("TransformBiasRescaleQkv")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    error_message::GetErrMgrRawErrorMessages();
    ASSERT_EQ(inferShapeFunc(contextHolder.GetContext()), ge::GRAPH_FAILED);
    auto reports = error_message::GetErrMgrRawErrorMessages();
    ASSERT_EQ(reports.size(), 1U);
    ASSERT_NE(reports[0].error_id.get(), nullptr);
    ASSERT_EQ(reports[0].args_key.size(), 4U);
    ASSERT_EQ(reports[0].args_value.size(), 4U);
    ASSERT_NE(reports[0].args_value[0].get(), nullptr);
    ASSERT_NE(reports[0].args_value[1].get(), nullptr);
    ASSERT_NE(reports[0].args_value[2].get(), nullptr);
    ASSERT_NE(reports[0].args_value[3].get(), nullptr);
    std::map<std::string, std::string> args;
    for (size_t i = 0; i < reports[0].args_key.size(); ++i) {
        ASSERT_NE(reports[0].args_key[i].get(), nullptr);
        args.emplace(reports[0].args_key[i].get(), reports[0].args_value[i].get());
    }
    EXPECT_EQ(args["param_name"], "num_heads");
    EXPECT_EQ(args["op_name"], "TransformBiasRescaleQkv");
    EXPECT_EQ(args["incorrect_value"], "0");
    EXPECT_EQ(args["reason"], "num_heads must be greater than 0");
    EXPECT_EQ(std::string(reports[0].error_id.get()), "EZ0026");
}

TEST_F(TransformBiasRescaleQkv, TransformBiasRescaleQkv_infershape_num_heads_negative_failed)
{
    gert::StorageShape qkvStorageShape = {{3, 4, 144}, {3, 4, 144}};
    gert::StorageShape biasStorageShape = {{144}, {144}};
    gert::StorageShape outputShape = {{}, {}};

    ge::DataType qkvDtype = ge::DT_FLOAT16;
    ge::DataType biasDtype = ge::DT_FLOAT16;
    ge::DataType outputDtype = ge::DT_FLOAT16;

    std::vector<gert::Tensor*> inputTensors = {
        (gert::Tensor*)&qkvStorageShape,
        (gert::Tensor*)&biasStorageShape,
    };

    std::vector<gert::StorageShape*> outputShapes = {&outputShape, &outputShape, &outputShape};

    auto contextHolder = gert::InferShapeContextFaker()
                             .SetOpType("TransformBiasRescaleQkv")
                             .NodeIoNum(2, 3)
                             .NodeInputTd(0, qkvDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputTensors(inputTensors)
                             .OutputShapes(outputShapes)
                             .Attr("num_heads", (int64_t)(-1))
                             .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferShapeFunc = spaceRegistry->GetOpImpl("TransformBiasRescaleQkv")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    error_message::GetErrMgrRawErrorMessages();
    ASSERT_EQ(inferShapeFunc(contextHolder.GetContext()), ge::GRAPH_FAILED);
    auto reports = error_message::GetErrMgrRawErrorMessages();
    ASSERT_EQ(reports.size(), 1U);
    ASSERT_NE(reports[0].error_id.get(), nullptr);
    ASSERT_EQ(reports[0].args_key.size(), 4U);
    ASSERT_EQ(reports[0].args_value.size(), 4U);
    ASSERT_NE(reports[0].args_value[0].get(), nullptr);
    ASSERT_NE(reports[0].args_value[1].get(), nullptr);
    ASSERT_NE(reports[0].args_value[2].get(), nullptr);
    ASSERT_NE(reports[0].args_value[3].get(), nullptr);
    std::map<std::string, std::string> args;
    for (size_t i = 0; i < reports[0].args_key.size(); ++i) {
        ASSERT_NE(reports[0].args_key[i].get(), nullptr);
        args.emplace(reports[0].args_key[i].get(), reports[0].args_value[i].get());
    }
    EXPECT_EQ(args["param_name"], "num_heads");
    EXPECT_EQ(args["op_name"], "TransformBiasRescaleQkv");
    EXPECT_EQ(args["incorrect_value"], "-1");
    EXPECT_EQ(args["reason"], "num_heads must be greater than 0");
    EXPECT_EQ(std::string(reports[0].error_id.get()), "EZ0026");
}
