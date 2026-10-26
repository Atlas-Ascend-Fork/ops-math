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
#include <limits>
#include "register/op_impl_registry.h"
#include "infershape_context_faker.h"
#include "op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace {
template <class T>
gert::ContextHolder<gert::InferShapeContext> GetInferShapeContextHolder(
    std::unique_ptr<uint8_t[]>& tensor_holder, std::vector<T> target_dims, gert::StorageShape& x_shape,
    gert::StorageShape& y_shape, const gert::StorageShape& target_storage_shape, ge::DataType dt,
    const int64_t allowzero_attr = -1)
{
    auto target_size = target_dims.size();
    auto target_tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    auto tensor_data = reinterpret_cast<T*>(target_tensor + 1);
    new (target_tensor)
        gert::Tensor(target_storage_shape, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kFollowing, dt, tensor_data);
    for (size_t idx = 0; idx < target_size; ++idx) {
        tensor_data[idx] = target_dims[idx];
    }

    gert::InferShapeContextFaker faker;
    // attrs are positional: axis(0), num_axes(1), allowzero(2); when allowzero_attr is set, the
    // copy(0) or literal(non-zero) semantics of 0 is pinned explicitly
    auto& faker_chain = faker.SetOpType("Reshape").NodeIoNum(2, 1);
    if (allowzero_attr >= 0) {
        faker_chain.Attr("axis", static_cast<int64_t>(0))
            .Attr("num_axes", static_cast<int64_t>(-1))
            .Attr("allowzero", allowzero_attr);
    }
    auto holder = faker_chain.InputTensors({(gert::Tensor*)&x_shape, target_tensor}).OutputShapes({&y_shape}).Build();
    return holder;
}
} // namespace
class ReshapeUT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Reshape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "Reshape TearDown" << std::endl; }
};

TEST_F(ReshapeUT, Reshape_registered_success)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
}

// the shape size of the shape input is negative, rejected by the element num check
TEST_F(ReshapeUT, Reshape_shape_input_neg_size_failed)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    gert::StorageShape x_shape = {{4, 2}, {4, 2}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{-1}, {-1}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * 4]);
    auto target_tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    new (target_tensor) gert::Tensor(target_storage_shape, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kFollowing,
                                     ge::DT_INT64, reinterpret_cast<int64_t*>(target_tensor + 1));
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Reshape")
                      .NodeIoNum(2, 1)
                      .InputTensors({(gert::Tensor*)&x_shape, target_tensor})
                      .OutputShapes({&y_shape})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}
// the shape data is not got and the shape input desc is dynamic: the output shape is unknown
// rank, same as RT1.0
TEST_F(ReshapeUT, Reshape_shape_input_dynamic_desc_no_data)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    gert::StorageShape x_shape = {{4, 2}, {4, 2}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{-1}, {-1}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor)]);
    auto target_tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    new (target_tensor)
        gert::Tensor(target_storage_shape, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kFollowing, ge::DT_INT64, nullptr);
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Reshape")
                      .NodeIoNum(2, 1)
                      .InputTensors({(gert::Tensor*)&x_shape, target_tensor})
                      .OutputShapes({&y_shape})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), -2);
}

// x {-1, 0, 50} shape [5, 10, -1, 4], empty tensor x with negative dim, target has no 0 so the
// -1 dim is set to 0 directly, the negative dim of x does not block the inference
TEST_F(ReshapeUT, Rt2_EmptyTensor_x_neg_1_0_50_and_shape_5_10_neg_1_4)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int32_t> target_dims = {5, 10, -1, 4};
    gert::StorageShape x_shape = {{-1, 0, 50}, {-1, 0, 50}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{4}, {4}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int32_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int32_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT32);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 5);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 10);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 0);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(3), 4);
}

// x {5} shape [-1, -1, 5], two -1 error
TEST_F(ReshapeUT, Rt2_InferShapeFailed_ShapeInputHas2NegOne)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int32_t> target_dims = {-1, -1, 5};
    gert::StorageShape x_shape = {{5}, {5}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{3}, {3}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int32_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int32_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT32);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

// x {3, 8, 8} shape [3, 8, 8, 0], 0 copy index out of range
TEST_F(ReshapeUT, Rt2_InferShapeFailed_ShapeInputZeroIdxOutOfRange)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int32_t> target_dims = {3, 8, 8, 0};
    gert::StorageShape x_shape = {{3, 8, 8}, {3, 8, 8}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{4}, {4}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int32_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int32_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT32, 0);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

// x {4, 0, 50} shape [5, 10, 0, 4], empty tensor x keeps target dims with 0
TEST_F(ReshapeUT, Rt2_EmptyTensor_x_4_0_50_and_shape_5_10_0_4)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {5, 10, 0, 4};
    gert::StorageShape x_shape = {{4, 0, 50}, {4, 0, 50}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{4}, {4}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 5);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 10);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 0);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(3), 4);
}

// x {4, 0, 50} shape [5, 10, -1, 4], empty tensor x infers -1 to 0
TEST_F(ReshapeUT, Rt2_EmptyTensor_x_4_0_50_and_shape_5_10_neg_1_4)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {5, 10, -1, 4};
    gert::StorageShape x_shape = {{4, 0, 50}, {4, 0, 50}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{4}, {4}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 4);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 5);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 10);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 0);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(3), 4);
}

// x {4, 0, 50} shape [5, 10, -1, -1], two -1 error
TEST_F(ReshapeUT, Rt2_EmptyTensor_x_4_0_50_and_shape_5_10_neg_1_neg_1)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {5, 10, -1, -1};
    gert::StorageShape x_shape = {{4, 0, 50}, {4, 0, 50}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{4}, {4}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

// x {2, 2, 2} shape [5, 4], output shape size not equal to input
TEST_F(ReshapeUT, Rt2_InferShape_x_2_2_2_and_shape_5_4)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int32_t> target_dims = {5, 4};
    gert::StorageShape x_shape = {{2, 2, 2}, {2, 2, 2}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{2}, {2}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int32_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int32_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT32);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

// shape input contains no element, output is scalar
TEST_F(ReshapeUT, Rt2_ScalarShape)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    gert::StorageShape x_shape = {{1}, {1}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{0}, {0}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor)]);
    auto target_tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    new (target_tensor)
        gert::Tensor(target_storage_shape, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kFollowing, ge::DT_INT64, nullptr);
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Reshape")
                      .NodeIoNum(2, 1)
                      .InputTensors({(gert::Tensor*)&x_shape, target_tensor})
                      .OutputShapes({&y_shape})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 0);
}

// x {8, 3, 224, 224} shape [24, -1], -1 inferred by shape size
TEST_F(ReshapeUT, InferShapeOk)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {24, -1};
    gert::StorageShape x_shape = {{8, 3, 224, 224}, {8, 3, 224, 224}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{2}, {2}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 24);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 50176);
}

TEST_F(ReshapeUT, DataDependentOk)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    EXPECT_FALSE(space_registry->GetOpImpl("Reshape")->IsInputDataDependency(0));
    EXPECT_TRUE(space_registry->GetOpImpl("Reshape")->IsInputDataDependency(1));
}

// x is unknown rank(-2), output keeps the target dims as-is
TEST_F(ReshapeUT, Reshape_dynamic_x_unknown_rank)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {3, 2};
    gert::StorageShape x_shape = {{-2}, {-2}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{2}, {2}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 2);
}

// x is unknown rank(-2) and the target contains -1: -1 stays unknown, same combination as the
// dynamic(-2) scenario of the graph example
TEST_F(ReshapeUT, Reshape_dynamic_x_unknown_rank_with_neg_1)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {3, -1};
    gert::StorageShape x_shape = {{-2}, {-2}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{2}, {2}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), -1);
}

// x has unknown dim(-1), output keeps the target dims as-is and -1 stays -1
TEST_F(ReshapeUT, Reshape_dynamic_x_keep_neg_1)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {-1, 1};
    gert::StorageShape x_shape = {{-1, 3}, {-1, 3}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{2}, {2}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), -1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 1);
}

// x has unknown dim(-1) and target is known, output is the target dims
TEST_F(ReshapeUT, Reshape_dynamic_x_target_known)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {3, 2};
    gert::StorageShape x_shape = {{-1, 3}, {-1, 3}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{2}, {2}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 2);
}

// x is dynamic and target contains 0, 0 cannot copy from unknown-rank x ({-2} marker)
TEST_F(ReshapeUT, Reshape_dynamic_x_with_zero_failed)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {0, 3};
    gert::StorageShape x_shape = {{-2}, {-2}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{2}, {2}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64, 0);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

// x is dynamic and target contains 0, 0 copies the dim from x and unknown dim(-1) is kept as-is
TEST_F(ReshapeUT, Reshape_dynamic_x_copy_zero)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {0, 3};
    gert::StorageShape x_shape = {{-1, 3}, {-1, 3}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{2}, {2}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64, 0);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), -1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 3);
}

// x is dynamic, target contains both 0 and -1 in copy mode: the copied unknown dim stays -1 and
// counts as 1 in the size math, the -1 dim is inferred from the effective x size (1*3/1=3)
TEST_F(ReshapeUT, Reshape_dynamic_x_copy_zero_with_neg_1)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {0, -1};
    gert::StorageShape x_shape = {{-1, 3}, {-1, 3}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{2}, {2}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64, 0);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), -1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 3);
}

// x is dynamic and the target contains two -1 dims: rejected by the unconditional duplicate
// check, same as RT1.0
TEST_F(ReshapeUT, Reshape_dynamic_x_two_neg_1_failed)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {-1, -1};
    gert::StorageShape x_shape = {{-1, 3}, {-1, 3}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{2}, {2}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

// x is dynamic and the target dims overflow int64 when multiplied: rejected by the overflow
// check, same as RT1.0
TEST_F(ReshapeUT, Reshape_dynamic_x_target_overflow_failed)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max(), -1};
    gert::StorageShape x_shape = {{-1, 3}, {-1, 3}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{3}, {3}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

// x is dynamic, the only unknown dim is copied and the -1 dim is not divisible by the effective
// x size (3 % 2): rejected the same way as RT1.0
TEST_F(ReshapeUT, Reshape_dynamic_x_copy_zero_neg_1_not_divisible_failed)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {0, -1, 2};
    gert::StorageShape x_shape = {{-1, 3}, {-1, 3}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{3}, {3}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64, 0);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

// x is dynamic, allowzero is 1 and target contains both a literal 0 and -1: the -1 dim cannot be
// inferred as the product of the other dims is 0, RT1.0 rejects it as well
TEST_F(ReshapeUT, Reshape_dynamic_x_literal_zero_with_neg_1_failed)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {0, -1};
    gert::StorageShape x_shape = {{-1, 3}, {-1, 3}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{2}, {2}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64, 1);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

// x is dynamic and 0 copy index is out of the dim num of x
TEST_F(ReshapeUT, Reshape_dynamic_x_copy_zero_idx_out_of_range_failed)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {3, 2, 0};
    gert::StorageShape x_shape = {{-1, 3}, {-1, 3}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{3}, {3}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64, 0);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

// shape data is not got on compile stage, output rank is inferred from the shape of shape input
TEST_F(ReshapeUT, Reshape_shape_data_not_got_rank_known)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    gert::StorageShape x_shape = {{2, 6}, {2, 6}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{2}, {2}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor)]);
    auto target_tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    new (target_tensor)
        gert::Tensor(target_storage_shape, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kFollowing, ge::DT_INT64, nullptr);
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Reshape")
                      .NodeIoNum(2, 1)
                      .InputTensors({(gert::Tensor*)&x_shape, target_tensor})
                      .OutputShapes({&y_shape})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), -1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), -1);
}

// shape data is not got and the rank of shape input is unknown(-1, the RT1.0 UNKNOWN_SHAPE
// marker), output shape is unknown rank; both storage members are -1 so the case does not
// depend on which member GetInputShape reads
TEST_F(ReshapeUT, Reshape_shape_data_not_got_rank_unknown)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    gert::StorageShape x_shape = {{2, 6}, {2, 6}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{-1}, {-1}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor)]);
    auto target_tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    new (target_tensor)
        gert::Tensor(target_storage_shape, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kFollowing, ge::DT_INT64, nullptr);
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Reshape")
                      .NodeIoNum(2, 1)
                      .InputTensors({(gert::Tensor*)&x_shape, target_tensor})
                      .OutputShapes({&y_shape})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), -2);
}

// shape data is not got and the shape of shape input is scalar, output is scalar
TEST_F(ReshapeUT, Reshape_shape_data_not_got_scalar)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    gert::StorageShape x_shape = {{2, 6}, {2, 6}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{}, {}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor)]);
    auto target_tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    new (target_tensor)
        gert::Tensor(target_storage_shape, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kFollowing, ge::DT_INT64, nullptr);
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Reshape")
                      .NodeIoNum(2, 1)
                      .InputTensors({(gert::Tensor*)&x_shape, target_tensor})
                      .OutputShapes({&y_shape})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 0);
}

// shape data is not got and the rank of shape input is larger than 1
TEST_F(ReshapeUT, Reshape_shape_data_not_got_rank_larger_than_1_failed)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    gert::StorageShape x_shape = {{2, 6}, {2, 6}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{2, 3}, {2, 3}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor)]);
    auto target_tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    new (target_tensor)
        gert::Tensor(target_storage_shape, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kFollowing, ge::DT_INT64, nullptr);
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Reshape")
                      .NodeIoNum(2, 1)
                      .InputTensors({(gert::Tensor*)&x_shape, target_tensor})
                      .OutputShapes({&y_shape})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

// x {2, 3} shape [0, -1], 0 copies the dim from x and -1 is inferred
TEST_F(ReshapeUT, Reshape_copy_zero_success)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {0, -1};
    gert::StorageShape x_shape = {{2, 3}, {2, 3}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{2}, {2}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64, 0);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 3);
}

// static x and copy mode: the copied dim makes the output size (2*5) differ from the x size (6)
TEST_F(ReshapeUT, Reshape_copy_zero_size_mismatch_failed)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {0, 5};
    gert::StorageShape x_shape = {{2, 3}, {2, 3}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{2}, {2}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64, 0);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

// x {2, 3} shape [3, -2], target dim must be -1 or non-negative
TEST_F(ReshapeUT, Reshape_neg_dim_failed)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {3, -2};
    gert::StorageShape x_shape = {{2, 3}, {2, 3}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{2}, {2}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

// element num of shape input is larger than kMaxDimNum
TEST_F(ReshapeUT, Reshape_target_larger_than_kMaxDimNum_failed)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    gert::StorageShape x_shape = {{2, 3}, {2, 3}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{30}, {30}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor)]);
    auto target_tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    new (target_tensor)
        gert::Tensor(target_storage_shape, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kFollowing, ge::DT_INT64, nullptr);
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Reshape")
                      .NodeIoNum(2, 1)
                      .InputTensors({(gert::Tensor*)&x_shape, target_tensor})
                      .OutputShapes({&y_shape})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

// ONNX allowzero is 0: 0 in the shape input copies the corresponding x dim
TEST_F(ReshapeUT, Reshape_allowzero_0_copy_zero_success)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {0, 3};
    gert::StorageShape x_shape = {{2, 3}, {2, 3}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{2}, {2}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto target_tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    new (target_tensor) gert::Tensor(target_storage_shape, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kFollowing,
                                     ge::DT_INT64, reinterpret_cast<int64_t*>(target_tensor + 1));
    auto* tensor_data = reinterpret_cast<int64_t*>(target_tensor + 1);
    tensor_data[0] = target_dims[0];
    tensor_data[1] = target_dims[1];
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Reshape")
                      .NodeIoNum(2, 1)
                      .Attr("axis", static_cast<int64_t>(0))
                      .Attr("num_axes", static_cast<int64_t>(-1))
                      .Attr("allowzero", static_cast<int64_t>(0))
                      .InputTensors({(gert::Tensor*)&x_shape, target_tensor})
                      .OutputShapes({&y_shape})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 3);
}

// ONNX allowzero is 1: 0 is a literal dim value, the 0 size cannot match the non-empty x
TEST_F(ReshapeUT, Reshape_allowzero_1_zero_literal_failed)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {0, 3};
    gert::StorageShape x_shape = {{2, 3}, {2, 3}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{2}, {2}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto target_tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    new (target_tensor) gert::Tensor(target_storage_shape, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kFollowing,
                                     ge::DT_INT64, reinterpret_cast<int64_t*>(target_tensor + 1));
    auto* tensor_data = reinterpret_cast<int64_t*>(target_tensor + 1);
    tensor_data[0] = target_dims[0];
    tensor_data[1] = target_dims[1];
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Reshape")
                      .NodeIoNum(2, 1)
                      .Attr("axis", static_cast<int64_t>(0))
                      .Attr("num_axes", static_cast<int64_t>(-1))
                      .Attr("allowzero", static_cast<int64_t>(1))
                      .InputTensors({(gert::Tensor*)&x_shape, target_tensor})
                      .OutputShapes({&y_shape})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

// the most common real lowering layout: parser graphs carry no axis/num_axes values, the IR
// segment truncates to 0 and only the private attr is materialized
TEST_F(ReshapeUT, Reshape_private_attr_only_allowzero_0_copy_success)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {0, 3};
    gert::StorageShape x_shape = {{2, 3}, {2, 3}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{2}, {2}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto target_tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    new (target_tensor) gert::Tensor(target_storage_shape, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kFollowing,
                                     ge::DT_INT64, reinterpret_cast<int64_t*>(target_tensor + 1));
    auto* tensor_data = reinterpret_cast<int64_t*>(target_tensor + 1);
    tensor_data[0] = target_dims[0];
    tensor_data[1] = target_dims[1];
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Reshape")
                      .NodeIoNum(2, 1)
                      .Attr("allowzero", static_cast<int64_t>(0))
                      .InputTensors({(gert::Tensor*)&x_shape, target_tensor})
                      .OutputShapes({&y_shape})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 3);
}

// ONNX allowzero is 1 with -1: the literal 0 dim cannot be divided for the -1 inference
TEST_F(ReshapeUT, Reshape_allowzero_1_zero_with_neg_1_failed)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {0, -1};
    gert::StorageShape x_shape = {{2, 3}, {2, 3}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{2}, {2}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto target_tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    new (target_tensor) gert::Tensor(target_storage_shape, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kFollowing,
                                     ge::DT_INT64, reinterpret_cast<int64_t*>(target_tensor + 1));
    auto* tensor_data = reinterpret_cast<int64_t*>(target_tensor + 1);
    tensor_data[0] = target_dims[0];
    tensor_data[1] = target_dims[1];
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Reshape")
                      .NodeIoNum(2, 1)
                      .Attr("axis", static_cast<int64_t>(0))
                      .Attr("num_axes", static_cast<int64_t>(-1))
                      .Attr("allowzero", static_cast<int64_t>(1))
                      .InputTensors({(gert::Tensor*)&x_shape, target_tensor})
                      .OutputShapes({&y_shape})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

// ONNX allowzero is 1 and x is dynamic: 0 is a literal dim value instead of copying the unknown dim
// x is dynamic and the target contains a literal 0: the output size is 0, which can never match
// the unknown input size, RT1.0 rejects it by the positive output size check as well
TEST_F(ReshapeUT, Reshape_allowzero_1_dynamic_x_literal_zero_failed)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {0, 5};
    gert::StorageShape x_shape = {{-1, 3}, {-1, 3}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{2}, {2}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto target_tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    new (target_tensor) gert::Tensor(target_storage_shape, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kFollowing,
                                     ge::DT_INT64, reinterpret_cast<int64_t*>(target_tensor + 1));
    auto* tensor_data = reinterpret_cast<int64_t*>(target_tensor + 1);
    tensor_data[0] = target_dims[0];
    tensor_data[1] = target_dims[1];
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Reshape")
                      .NodeIoNum(2, 1)
                      .Attr("axis", static_cast<int64_t>(0))
                      .Attr("num_axes", static_cast<int64_t>(-1))
                      .Attr("allowzero", static_cast<int64_t>(1))
                      .InputTensors({(gert::Tensor*)&x_shape, target_tensor})
                      .OutputShapes({&y_shape})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

// allowzero is 2: any non-zero value means the literal 0, the output size 0 differs from 6
TEST_F(ReshapeUT, Reshape_allowzero_2_zero_literal_failed)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {0, 6};
    gert::StorageShape x_shape = {{2, 3}, {2, 3}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{2}, {2}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64, 2);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

// the framework attrs (axis/num_axes) with default values do not disturb the normal path
TEST_F(ReshapeUT, Reshape_default_attrs_keep_normal_path)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {3, 2};
    gert::StorageShape x_shape = {{2, 3}, {2, 3}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{2}, {2}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto target_tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    new (target_tensor) gert::Tensor(target_storage_shape, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, gert::kFollowing,
                                     ge::DT_INT64, reinterpret_cast<int64_t*>(target_tensor + 1));
    auto* tensor_data = reinterpret_cast<int64_t*>(target_tensor + 1);
    tensor_data[0] = target_dims[0];
    tensor_data[1] = target_dims[1];
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("Reshape")
                      .NodeIoNum(2, 1)
                      .Attr("axis", static_cast<int64_t>(0))
                      .Attr("num_axes", static_cast<int64_t>(-1))
                      .Attr("allowzero", static_cast<int64_t>(0))
                      .InputTensors({(gert::Tensor*)&x_shape, target_tensor})
                      .OutputShapes({&y_shape})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 2);
}

// no attr is set: 0 is a literal dim value by default (RT1.0 and TF semantics), the 0 size
// cannot match the non-empty x
TEST_F(ReshapeUT, Reshape_no_attr_zero_literal_failed)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {0, 3};
    gert::StorageShape x_shape = {{2, 3}, {2, 3}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{2}, {2}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

// empty tensor x, target has -1 and no 0: the -1 dim is set to 0 directly, the output stays empty
TEST_F(ReshapeUT, Reshape_empty_tensor_neg_1_no_zero)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {-1, 2};
    gert::StorageShape x_shape = {{2, 0, 3}, {2, 0, 3}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{2}, {2}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 0);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 2);
}

// empty tensor x, target has neither 0 nor -1: the non-empty target cannot come from an empty x
TEST_F(ReshapeUT, Reshape_empty_tensor_no_zero_no_neg_1_failed)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {2, 3};
    gert::StorageShape x_shape = {{2, 0, 3}, {2, 0, 3}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{2}, {2}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64, -1);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

// empty tensor x and copy mode: the target 0 keeps the literal value, the output stays empty
TEST_F(ReshapeUT, Reshape_empty_x_allowzero_0_target_with_zero)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {0, 3};
    gert::StorageShape x_shape = {{2, 0, 5}, {2, 0, 5}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{2}, {2}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64, 0);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 0);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 3);
}

// empty tensor x, target has both 0 and -1: the -1 dim is inferred by the truncated integer
// division of the non-zero products (10/3=3), the remainder is discarded as RT1.0 does
TEST_F(ReshapeUT, Reshape_empty_tensor_neg_1_truncated_division)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {0, -1, 3};
    gert::StorageShape x_shape = {{2, 0, 5}, {2, 0, 5}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{3}, {3}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 0);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 3);
}

// shape input contains no element: only a scalar x (size is 1) can be reshaped to a scalar;
// stricter than RT1.0, which outputs a scalar unconditionally when the target is empty
TEST_F(ReshapeUT, Reshape_scalar_shape_mismatch_failed)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {};
    gert::StorageShape x_shape = {{2, 3}, {2, 3}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{0}, {0}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor)]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

// x is a true scalar (rank 0), the target contains 1 element
TEST_F(ReshapeUT, Reshape_scalar_x)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {1};
    gert::StorageShape x_shape = {{}, {}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{1}, {1}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64, -1);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 1);
}

// shape input has 0 elements with readable data: a scalar x (size 1) is reshaped to a scalar
TEST_F(ReshapeUT, Reshape_scalar_shape_data_present)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {};
    gert::StorageShape x_shape = {{}, {}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{0}, {0}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64, -1);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 0);
}

// the dtype of the shape input must be int32 or int64, other dtypes are rejected
TEST_F(ReshapeUT, Reshape_shape_input_dtype_invalid_failed)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int32_t> target_dims = {2, 6};
    gert::StorageShape x_shape = {{4, 3}, {4, 3}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{2}, {2}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int32_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int32_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_FLOAT);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

// the product of the target dims overflows int64
TEST_F(ReshapeUT, Reshape_output_size_overflow_failed)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {std::numeric_limits<int64_t>::max(), 2};
    gert::StorageShape x_shape = {{4, 6}, {4, 6}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{2}, {2}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

// empty tensor x, target contains a negative dim other than -1: rejected by the explicit
// negative validation instead of the misleading overflow check
TEST_F(ReshapeUT, Reshape_empty_tensor_neg_dim_failed)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {-5, 2};
    gert::StorageShape x_shape = {{2, 0, 3}, {2, 0, 3}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{2}, {2}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * target_dims.size()]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

// empty tensor x, shape input contains no element: rejected the same way as RT1.0, whose empty
// tensor process has no branch for an empty target (no 0 and no -1)
TEST_F(ReshapeUT, Reshape_empty_tensor_to_scalar_failed)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto infer_shape_func = space_registry->GetOpImpl("Reshape")->infer_shape;

    std::vector<int64_t> target_dims = {};
    gert::StorageShape x_shape = {{2, 0, 3}, {2, 0, 3}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape target_storage_shape = {{0}, {0}};
    auto tensor_holder = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor)]);
    auto holder = GetInferShapeContextHolder<int64_t>(tensor_holder, target_dims, x_shape, y_shape,
                                                      target_storage_shape, ge::DT_INT64);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(ReshapeUT, Reshape_infer_datatype)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("Reshape"), nullptr);
    auto op_impl = space_registry->GetOpImpl("Reshape");
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Reshape").OpName("Reshape");
    builder.IONum(2, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto context_holder = builder.Build();
    auto* context = context_holder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = op_impl->infer_datatype(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    // output dtype is same as input x
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}
