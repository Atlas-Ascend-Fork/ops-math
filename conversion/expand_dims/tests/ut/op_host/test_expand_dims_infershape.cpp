/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_expand_dims_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "op_infer_datatype_context_builder.h"
#include "exe_graph/runtime/kernel_run_context.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace {
// Build an axis tensor carrying real host data, so the data-dependent
// InferShape can read the axis value.
template <typename T>
gert::Tensor MakeAxisTensor(const gert::StorageShape& shape, ge::DataType dtype, T& value)
{
    return gert::Tensor(shape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                        gert::TensorPlacement::kOnHost, dtype, &value);
}

gert::Tensor MakeNullDataAxisTensor(const gert::StorageShape& shape, ge::DataType dtype)
{
    return gert::Tensor(shape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                        gert::TensorPlacement::kOnHost, dtype, nullptr);
}
} // namespace

class ExpandDimsTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ExpandDimsTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ExpandDimsTest TearDown" << std::endl; }

    static auto GetInferShapeFunc()
    {
        auto opImpl = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("ExpandDims");
        EXPECT_NE(opImpl, nullptr);
        return opImpl->infer_shape;
    }
};

// x {8,3,224,224}, axis=0: insert dim 1 at position 0. Expect output {1,8,3,224,224}.
TEST_F(ExpandDimsTest, expand_normal_axis_ok)
{
    auto inferShapeFunc = GetInferShapeFunc();
    gert::StorageShape xShape = {{8, 3, 224, 224}, {8, 3, 224, 224}};
    gert::StorageShape axisShape = {{1}, {1}};
    int64_t axisValue = 0;
    auto axisTensor = MakeAxisTensor<int64_t>(axisShape, ge::DT_INT64, axisValue);

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("ExpandDims")
                      .NodeIoNum(2, 1)
                      .InputTensors({(gert::Tensor*)&xShape, &axisTensor})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext()), ge::GRAPH_SUCCESS);
    const auto* yShape = holder.GetContext()->GetOutputShape(0);
    ASSERT_NE(yShape, nullptr);
    EXPECT_EQ(yShape->GetDimNum(), 5UL);
    EXPECT_EQ(yShape->GetDim(0), 1);
    EXPECT_EQ(yShape->GetDim(1), 8);
    EXPECT_EQ(yShape->GetDim(2), 3);
    EXPECT_EQ(yShape->GetDim(3), 224);
    EXPECT_EQ(yShape->GetDim(4), 224);
}

// negative axis: x {8,3,224,224}, axis=-5 (resolves to 0). Expect output {1,8,3,224,224}.
TEST_F(ExpandDimsTest, expand_negative_axis_ok)
{
    auto inferShapeFunc = GetInferShapeFunc();
    gert::StorageShape xShape = {{8, 3, 224, 224}, {8, 3, 224, 224}};
    gert::StorageShape axisShape = {{1}, {1}};
    int64_t axisValue = -5;
    auto axisTensor = MakeAxisTensor<int64_t>(axisShape, ge::DT_INT64, axisValue);

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("ExpandDims")
                      .NodeIoNum(2, 1)
                      .InputTensors({(gert::Tensor*)&xShape, &axisTensor})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext()), ge::GRAPH_SUCCESS);
    const auto* yShape = holder.GetContext()->GetOutputShape(0);
    ASSERT_NE(yShape, nullptr);
    EXPECT_EQ(yShape->GetDimNum(), 5UL);
    EXPECT_EQ(yShape->GetDim(0), 1);
    EXPECT_EQ(yShape->GetDim(1), 8);
    EXPECT_EQ(yShape->GetDim(2), 3);
    EXPECT_EQ(yShape->GetDim(3), 224);
    EXPECT_EQ(yShape->GetDim(4), 224);
}

// axis=5 out of upper bound [-(4+1), 4]. Expect failure.
TEST_F(ExpandDimsTest, expand_out_of_upper_bound_failed)
{
    auto inferShapeFunc = GetInferShapeFunc();
    gert::StorageShape xShape = {{8, 3, 224, 224}, {8, 3, 224, 224}};
    gert::StorageShape axisShape = {{1}, {1}};
    int64_t axisValue = 5;
    auto axisTensor = MakeAxisTensor<int64_t>(axisShape, ge::DT_INT64, axisValue);

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("ExpandDims")
                      .NodeIoNum(2, 1)
                      .InputTensors({(gert::Tensor*)&xShape, &axisTensor})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext()), ge::GRAPH_FAILED);
}

// axis=-6 out of lower bound [-(4+1), 4]. Expect failure.
TEST_F(ExpandDimsTest, expand_out_of_lower_bound_failed)
{
    auto inferShapeFunc = GetInferShapeFunc();
    gert::StorageShape xShape = {{8, 3, 224, 224}, {8, 3, 224, 224}};
    gert::StorageShape axisShape = {{1}, {1}};
    int64_t axisValue = -6;
    auto axisTensor = MakeAxisTensor<int64_t>(axisShape, ge::DT_INT64, axisValue);

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("ExpandDims")
                      .NodeIoNum(2, 1)
                      .InputTensors({(gert::Tensor*)&xShape, &axisTensor})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext()), ge::GRAPH_FAILED);
}

// int32 axis: x {8,3,224,224}, axis=-1 (resolves to 4). Expect output {8,3,224,224,1}.
TEST_F(ExpandDimsTest, expand_int32_axis_ok)
{
    auto inferShapeFunc = GetInferShapeFunc();
    gert::StorageShape xShape = {{8, 3, 224, 224}, {8, 3, 224, 224}};
    gert::StorageShape axisShape = {{1}, {1}};
    int32_t axisValue = -1;
    auto axisTensor = MakeAxisTensor<int32_t>(axisShape, ge::DT_INT32, axisValue);

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("ExpandDims")
                      .NodeIoNum(2, 1)
                      .InputTensors({(gert::Tensor*)&xShape, &axisTensor})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext()), ge::GRAPH_SUCCESS);
    const auto* yShape = holder.GetContext()->GetOutputShape(0);
    ASSERT_NE(yShape, nullptr);
    EXPECT_EQ(yShape->GetDimNum(), 5UL);
    EXPECT_EQ(yShape->GetDim(0), 8);
    EXPECT_EQ(yShape->GetDim(1), 3);
    EXPECT_EQ(yShape->GetDim(2), 224);
    EXPECT_EQ(yShape->GetDim(3), 224);
    EXPECT_EQ(yShape->GetDim(4), 1);
}

// axis dtype is DT_FLOAT (must be DT_INT32/DT_INT64). Expect failure.
TEST_F(ExpandDimsTest, expand_axis_dtype_invalid_failed)
{
    auto inferShapeFunc = GetInferShapeFunc();
    gert::StorageShape xShape = {{8, 3, 224, 224}, {8, 3, 224, 224}};
    gert::StorageShape axisShape = {{1}, {1}};
    float axisValue = 0.0f;
    auto axisTensor = MakeAxisTensor<float>(axisShape, ge::DT_FLOAT, axisValue);

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("ExpandDims")
                      .NodeIoNum(2, 1)
                      .InputTensors({(gert::Tensor*)&xShape, &axisTensor})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext()), ge::GRAPH_FAILED);
}

// unknown rank input {-2}: output keeps unknown rank.
TEST_F(ExpandDimsTest, expand_unknown_rank_ok)
{
    auto inferShapeFunc = GetInferShapeFunc();
    gert::StorageShape xShape = {{-2}, {-2}};
    gert::StorageShape axisShape = {{1}, {1}};
    auto axisTensor = MakeNullDataAxisTensor(axisShape, ge::DT_INT64);

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("ExpandDims")
                      .NodeIoNum(2, 1)
                      .InputTensors({(gert::Tensor*)&xShape, &axisTensor})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext()), ge::GRAPH_SUCCESS);
    const auto* yShape = holder.GetContext()->GetOutputShape(0);
    ASSERT_NE(yShape, nullptr);
    EXPECT_EQ(yShape->GetDimNum(), 1UL);
    EXPECT_EQ(yShape->GetDim(0), -2);
}

// unknown rank input {-2} with axis value AVAILABLE (runtime): output still keeps unknown rank
// (RT1.0 v1 semantics — inserting a dim into an unknown-rank shape is meaningless; the gert version
// would incorrectly produce {1, -2} in this case).
TEST_F(ExpandDimsTest, expand_unknown_rank_with_axis_value_ok)
{
    auto inferShapeFunc = GetInferShapeFunc();
    gert::StorageShape xShape = {{-2}, {-2}};
    gert::StorageShape axisShape = {{1}, {1}};
    int64_t axisValue = 0;
    auto axisTensor = MakeAxisTensor<int64_t>(axisShape, ge::DT_INT64, axisValue);

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("ExpandDims")
                      .NodeIoNum(2, 1)
                      .InputTensors({(gert::Tensor*)&xShape, &axisTensor})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext()), ge::GRAPH_SUCCESS);
    const auto* yShape = holder.GetContext()->GetOutputShape(0);
    ASSERT_NE(yShape, nullptr);
    EXPECT_EQ(yShape->GetDimNum(), 1UL);
    EXPECT_EQ(yShape->GetDim(0), -2);
}

// axis tensor is nullptr. Expect failure.
TEST_F(ExpandDimsTest, expand_no_input_axis_tensor_failed)
{
    auto inferShapeFunc = GetInferShapeFunc();
    gert::StorageShape xShape = {{8, 3, 224, 224}, {8, 3, 224, 224}};
    gert::StorageShape axisShape = {{1}, {1}};
    int64_t axisValue = 0;
    auto axisTensor = MakeAxisTensor<int64_t>(axisShape, ge::DT_INT64, axisValue);

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("ExpandDims")
                      .NodeIoNum(2, 1)
                      .InputTensors({(gert::Tensor*)&xShape, &axisTensor})
                      .Build();
    // set the axis input slot pointer to nullptr, same as nullInputIndices of ExecuteTestCase
    auto* kernelCtx = reinterpret_cast<KernelRunContext*>(holder.GetContext());
    ASSERT_LT(1U, kernelCtx->input_size);
    kernelCtx->values[1]->data.pointer = nullptr;

    EXPECT_EQ(inferShapeFunc(holder.GetContext()), ge::GRAPH_FAILED);
}

// axis tensor with 2 values ({2} shape). Expect failure.
TEST_F(ExpandDimsTest, expand_error_input_size_failed)
{
    auto inferShapeFunc = GetInferShapeFunc();
    gert::StorageShape xShape = {{8, 3, 224, 224}, {8, 3, 224, 224}};
    gert::StorageShape axisShape = {{2}, {2}};
    int64_t axisValue[2] = {0, 1};
    auto axisTensor = gert::Tensor(axisShape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                                   gert::TensorPlacement::kOnHost, ge::DT_INT64, axisValue);

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("ExpandDims")
                      .NodeIoNum(2, 1)
                      .InputTensors({(gert::Tensor*)&xShape, &axisTensor})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext()), ge::GRAPH_FAILED);
}

// axis tensor data is nullptr (axis value unknown): output rank is x rank + 1, all dims unknown.
TEST_F(ExpandDimsTest, expand_no_tensor_data_ok)
{
    auto inferShapeFunc = GetInferShapeFunc();
    gert::StorageShape xShape = {{8, 3, 224, 224}, {8, 3, 224, 224}};
    gert::StorageShape axisShape = {{1}, {1}};
    auto axisTensor = MakeNullDataAxisTensor(axisShape, ge::DT_INT64);

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("ExpandDims")
                      .NodeIoNum(2, 1)
                      .InputTensors({(gert::Tensor*)&xShape, &axisTensor})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext()), ge::GRAPH_SUCCESS);
    const auto* yShape = holder.GetContext()->GetOutputShape(0);
    ASSERT_NE(yShape, nullptr);
    EXPECT_EQ(yShape->GetDimNum(), 5UL);
    for (size_t i = 0; i < 5UL; i++) {
        EXPECT_EQ(yShape->GetDim(i), -1);
    }
}

// scalar axis (shape {}): valid single value. Expect output {1,8,3,224,224}.
TEST_F(ExpandDimsTest, expand_scalar_axis_ok)
{
    auto inferShapeFunc = GetInferShapeFunc();
    gert::StorageShape xShape = {{8, 3, 224, 224}, {8, 3, 224, 224}};
    gert::StorageShape axisShape = {{}, {}};
    int64_t axisValue = 0;
    auto axisTensor = MakeAxisTensor<int64_t>(axisShape, ge::DT_INT64, axisValue);

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("ExpandDims")
                      .NodeIoNum(2, 1)
                      .InputTensors({(gert::Tensor*)&xShape, &axisTensor})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext()), ge::GRAPH_SUCCESS);
    const auto* yShape = holder.GetContext()->GetOutputShape(0);
    ASSERT_NE(yShape, nullptr);
    EXPECT_EQ(yShape->GetDimNum(), 5UL);
    EXPECT_EQ(yShape->GetDim(0), 1);
    EXPECT_EQ(yShape->GetDim(1), 8);
    EXPECT_EQ(yShape->GetDim(2), 3);
    EXPECT_EQ(yShape->GetDim(3), 224);
    EXPECT_EQ(yShape->GetDim(4), 224);
}

// InferDataType copies the input x dtype to the output.
TEST_F(ExpandDimsTest, expand_infer_data_type_success)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry, nullptr);
    auto op_impl = space_registry->GetOpImpl("ExpandDims");
    ASSERT_NE(op_impl, nullptr);
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("ExpandDims").OpName("ExpandDims");
    builder.IONum(2, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto context_holder = builder.Build();
    auto* context = context_holder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = op_impl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}
