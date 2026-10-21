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
#include "register/op_impl_registry.h"
#include "infershape_context_faker.h"
#include "op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace {
template <class T>
gert::ContextHolder<gert::InferShapeContext> GetKernelRunContextHolder(
    std::unique_ptr<uint8_t[]>& input_tensor_holder, std::vector<T> axes, gert::StorageShape& input_shape,
    gert::StorageShape& output_shape, const gert::StorageShape& axis_storage_shape,
    const gert::StorageFormat& axis_storage_fmt, ge::DataType dt)
{
    auto axes_shape_size = axes.size();
    auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
    auto tensor_data = reinterpret_cast<T*>(input_tensor + 1);
    new (input_tensor) gert::Tensor(axis_storage_shape, axis_storage_fmt, gert::kFollowing, dt, tensor_data);
    for (size_t idx = 0; idx < axes_shape_size; ++idx) {
        tensor_data[idx] = axes[idx];
    }

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SqueezeV3")
                      .NodeIoNum(2, 1)
                      .InputTensors({(gert::Tensor*)&input_shape, input_tensor})
                      .OutputShapes({&output_shape})
                      .Build();
    return holder;
}
} // namespace
class SqueezeV3UT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SqueezeV3 SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SqueezeV3 TearDown" << std::endl; }
};

// x is not data dependent, the axes input data is read on compile stage
TEST_F(SqueezeV3UT, SqueezeV3_data_dependency)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("SqueezeV3"), nullptr);
    EXPECT_FALSE(space_registry->GetOpImpl("SqueezeV3")->IsInputDataDependency(0));
    EXPECT_TRUE(space_registry->GetOpImpl("SqueezeV3")->IsInputDataDependency(1));
}

TEST_F(SqueezeV3UT, SqueezeV3_unknown_rank_with_axes)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV3")->infer_shape;

    std::vector<int64_t> axes = {-5, 3};
    gert::StorageShape input_shape = {{-2}, {-2}};
    gert::StorageShape output_shape = {{}, {}};
    gert::StorageShape axis_storage_shape = {{2}, {2}};
    gert::StorageFormat axis_storage_fmt = {ge::FORMAT_ND, ge::FORMAT_ND, {}};
    ge::DataType axis_dt = ge::DT_INT64;
    auto input_tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * axes.size()]);
    ASSERT_NE(input_tensor_holder, nullptr);
    auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
    auto tensor_data = reinterpret_cast<int64_t*>(input_tensor + 1);
    new (input_tensor) gert::Tensor(axis_storage_shape, axis_storage_fmt, gert::kFollowing, ge::DT_INT64, tensor_data);
    tensor_data[0] = axes[0];
    tensor_data[1] = axes[1];
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SqueezeV3")
                      .NodeIoNum(2, 1)
                      .InputTensors({(gert::Tensor*)&input_shape, input_tensor})
                      .OutputShapes({&output_shape})
                      .Build();
    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), -2);
}

// x has an unknown dim and axes points to a determinate 1 dim: the 1 dim is squeezed and
// the unknown dim is kept, same combination as the dynamic(-1) scenario of the graph example
TEST_F(SqueezeV3UT, SqueezeV3_one_axis_with_unknown_dim)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV3")->infer_shape;

    std::vector<int64_t> axes = {1};
    gert::StorageShape input_shape = {{-1, 1, 3}, {-1, 1, 3}};
    gert::StorageShape output_shape = {{}, {}};
    gert::StorageShape axis_storage_shape = {{1}, {1}};
    gert::StorageFormat axis_storage_fmt = {ge::FORMAT_ND, ge::FORMAT_ND, {}};
    auto input_tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * axes.size()]);
    auto holder = GetKernelRunContextHolder<int64_t>(input_tensor_holder, axes, input_shape, output_shape,
                                                     axis_storage_shape, axis_storage_fmt, ge::DT_INT64);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), -1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 3);
}

// readable empty axes and no 1-dims in x: all dims are kept as-is
TEST_F(SqueezeV3UT, SqueezeV3_no_1dims_passthrough)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV3")->infer_shape;

    std::vector<int64_t> axes = {};
    gert::StorageShape input_shape = {{2, 3}, {2, 3}};
    gert::StorageShape output_shape = {{}, {}};
    gert::StorageShape axis_storage_shape = {{0}, {0}};
    gert::StorageFormat axis_storage_fmt = {ge::FORMAT_ND, ge::FORMAT_ND, {}};
    auto input_tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * axes.size()]);
    auto holder = GetKernelRunContextHolder<int64_t>(input_tensor_holder, axes, input_shape, output_shape,
                                                     axis_storage_shape, axis_storage_fmt, ge::DT_INT64);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 3);
}

// readable empty axes and all dims of x are 1: the output is a scalar
TEST_F(SqueezeV3UT, SqueezeV3_all_1dims_to_scalar)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV3")->infer_shape;

    std::vector<int64_t> axes = {};
    gert::StorageShape input_shape = {{1, 1, 1}, {1, 1, 1}};
    gert::StorageShape output_shape = {{}, {}};
    gert::StorageShape axis_storage_shape = {{0}, {0}};
    gert::StorageFormat axis_storage_fmt = {ge::FORMAT_ND, ge::FORMAT_ND, {}};
    auto input_tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * axes.size()]);
    auto holder = GetKernelRunContextHolder<int64_t>(input_tensor_holder, axes, input_shape, output_shape,
                                                     axis_storage_shape, axis_storage_fmt, ge::DT_INT64);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 0);
}

// the axes input is constrained to DT_INT64 by the proto, other dtypes are rejected
// instead of being misread as int64
TEST_F(SqueezeV3UT, SqueezeV3_axes_dtype_invalid_failed)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV3")->infer_shape;

    std::vector<int64_t> axes = {1};
    gert::StorageShape input_shape = {{2, 1, 3}, {2, 1, 3}};
    gert::StorageShape output_shape = {{}, {}};
    gert::StorageShape axis_storage_shape = {{1}, {1}};
    gert::StorageFormat axis_storage_fmt = {ge::FORMAT_ND, ge::FORMAT_ND, {}};
    auto input_tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * axes.size()]);
    auto holder = GetKernelRunContextHolder<int64_t>(input_tensor_holder, axes, input_shape, output_shape,
                                                     axis_storage_shape, axis_storage_fmt, ge::DT_INT32);

    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_FAILED);
}

TEST_F(SqueezeV3UT, SqueezeV3_no_axes_input)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV3")->infer_shape;
    gert::StorageShape input_shape = {{1, 3, 224, 224, 1}, {1, 3, 224, 224, 1}};
    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SqueezeV3")
                      .NodeIoNum(1, 1)
                      .InputTensors({(gert::Tensor*)&input_shape})
                      .OutputShapes({&output_shape})
                      .Build();
    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 224);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 224);
}

TEST_F(SqueezeV3UT, SqueezeV3_no_axis_input_buf)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV3")->infer_shape;
    gert::StorageShape input_shape = {{1, 3, 224, 224}, {1, 3, 224, 224}};
    gert::StorageShape output_shape = {{}, {}};
    gert::StorageShape axis_storage_shape = {{1}, {1}};
    gert::StorageFormat axis_storage_fmt = {ge::FORMAT_ND, ge::FORMAT_ND, {}};
    int64_t input_shape_size = 0;
    auto input_tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * input_shape_size]);
    ASSERT_NE(input_tensor_holder, nullptr);
    auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
    new (input_tensor) gert::Tensor(axis_storage_shape, axis_storage_fmt, gert::kFollowing, ge::DT_INT64, nullptr);

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SqueezeV3")
                      .NodeIoNum(2, 1)
                      .InputTensors({(gert::Tensor*)&input_shape, input_tensor})
                      .OutputShapes({&output_shape})
                      .Build();
    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), -2);
}

TEST_F(SqueezeV3UT, SqueezeV3_empty_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV3")->infer_shape;

    std::vector<int64_t> axes = {};
    gert::StorageShape input_shape = {{1, 3, 2, -1, 5}, {1, 3, 2, -1, 5}};
    gert::StorageShape output_shape = {{}, {}};
    gert::StorageShape axis_storage_shape = {{0}, {0}};
    gert::StorageFormat axis_storage_fmt = {ge::FORMAT_ND, ge::FORMAT_ND, {}};
    ge::DataType axis_dt = ge::DT_INT64;
    auto input_tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * axes.size()]);
    auto holder = GetKernelRunContextHolder<int64_t>(input_tensor_holder, axes, input_shape, output_shape,
                                                     axis_storage_shape, axis_storage_fmt, axis_dt);
    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 1);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), -2);
}

TEST_F(SqueezeV3UT, SqueezeV3_static_shape_with_empty_axes)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV3")->infer_shape;

    // axes is a readable empty list (connected, size 0, data addr non-null), input shape is
    // known: axes_empty is true via (has_axes_data && axes_size == 0U), squeeze all dims of 1
    std::vector<int64_t> axes = {};
    gert::StorageShape input_shape = {{1, 3, 1, 1, 5}, {1, 3, 1, 1, 5}};
    gert::StorageShape output_shape = {{}, {}};
    gert::StorageShape axis_storage_shape = {{0}, {0}};
    gert::StorageFormat axis_storage_fmt = {ge::FORMAT_ND, ge::FORMAT_ND, {}};
    ge::DataType axis_dt = ge::DT_INT64;
    auto input_tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * axes.size()]);
    auto holder = GetKernelRunContextHolder<int64_t>(input_tensor_holder, axes, input_shape, output_shape,
                                                     axis_storage_shape, axis_storage_fmt, axis_dt);
    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 5);
}

TEST_F(SqueezeV3UT, SqueezeV3_axis_out_of_upper_bound)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV3")->infer_shape;

    std::vector<int64_t> axes = {5};
    gert::StorageShape input_shape = {{1, 3, 2, -1, 5}, {1, 3, 2, -1, 5}};
    gert::StorageShape output_shape = {{}, {}};
    gert::StorageShape axis_storage_shape = {{1}, {1}};
    gert::StorageFormat axis_storage_fmt = {ge::FORMAT_ND, ge::FORMAT_ND, {}};
    ge::DataType axis_dt = ge::DT_INT64;
    auto input_tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * axes.size()]);
    auto holder = GetKernelRunContextHolder<int64_t>(input_tensor_holder, axes, input_shape, output_shape,
                                                     axis_storage_shape, axis_storage_fmt, axis_dt);
    EXPECT_NE(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
}

TEST_F(SqueezeV3UT, SqueezeV3_axis_out_of_lower_bound)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV3")->infer_shape;

    std::vector<int64_t> axes = {-6};
    gert::StorageShape input_shape = {{1, 3, 2, -1, 5}, {1, 3, 2, -1, 5}};
    gert::StorageShape output_shape = {{}, {}};
    gert::StorageShape axis_storage_shape = {{1}, {1}};
    gert::StorageFormat axis_storage_fmt = {ge::FORMAT_ND, ge::FORMAT_ND, {}};
    ge::DataType axis_dt = ge::DT_INT64;
    auto input_tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * axes.size()]);
    auto holder = GetKernelRunContextHolder<int64_t>(input_tensor_holder, axes, input_shape, output_shape,
                                                     axis_storage_shape, axis_storage_fmt, axis_dt);
    EXPECT_NE(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
}

TEST_F(SqueezeV3UT, SqueezeV3_error_dim)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV3")->infer_shape;

    std::vector<int64_t> axes = {2};
    gert::StorageShape input_shape = {{1, 3, 2, -1, 5}, {1, 3, 2, -1, 5}};
    gert::StorageShape output_shape = {{}, {}};
    gert::StorageShape axis_storage_shape = {{1}, {1}};
    gert::StorageFormat axis_storage_fmt = {ge::FORMAT_ND, ge::FORMAT_ND, {}};
    ge::DataType axis_dt = ge::DT_INT64;
    auto input_tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * axes.size()]);
    auto holder = GetKernelRunContextHolder<int64_t>(input_tensor_holder, axes, input_shape, output_shape,
                                                     axis_storage_shape, axis_storage_fmt, axis_dt);
    EXPECT_NE(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
}

TEST_F(SqueezeV3UT, SqueezeV3_normal_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV3")->infer_shape;

    std::vector<int64_t> axes = {0, 3};
    gert::StorageShape input_shape = {{1, 3, 2, 1, 5}, {1, 3, 2, 1, 5}};
    gert::StorageShape output_shape = {{}, {}};
    gert::StorageShape axis_storage_shape = {{2}, {2}};
    gert::StorageFormat axis_storage_fmt = {ge::FORMAT_ND, ge::FORMAT_ND, {}};
    ge::DataType axis_dt = ge::DT_INT64;
    auto input_tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * axes.size()]);
    auto holder = GetKernelRunContextHolder<int64_t>(input_tensor_holder, axes, input_shape, output_shape,
                                                     axis_storage_shape, axis_storage_fmt, axis_dt);
    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 5);
}

TEST_F(SqueezeV3UT, SqueezeV3_negative_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV3")->infer_shape;

    std::vector<int64_t> axes = {-5, 3};
    gert::StorageShape input_shape = {{1, 3, 2, 1, 5}, {1, 3, 2, 1, 5}};
    gert::StorageShape output_shape = {{}, {}};
    gert::StorageShape axis_storage_shape = {{2}, {2}};
    gert::StorageFormat axis_storage_fmt = {ge::FORMAT_ND, ge::FORMAT_ND, {}};
    ge::DataType axis_dt = ge::DT_INT64;
    auto input_tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * axes.size()]);
    auto holder = GetKernelRunContextHolder<int64_t>(input_tensor_holder, axes, input_shape, output_shape,
                                                     axis_storage_shape, axis_storage_fmt, axis_dt);
    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 5);
    // the axes input may feed other nodes, its data must stay unchanged after infershape
    const auto axes_data_after = reinterpret_cast<const int64_t*>(input_tensor_holder.get() + sizeof(gert::Tensor));
    for (size_t idx = 0U; idx < axes.size(); ++idx) {
        EXPECT_EQ(axes_data_after[idx], axes[idx]);
    }
}

TEST_F(SqueezeV3UT, SqueezeV3_unknown_dims)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV3")->infer_shape;

    std::vector<int64_t> axes = {0, 3, 4};
    gert::StorageShape input_shape = {{1, 3, 2, -1, -1, 5}, {1, 3, 2, -1, -1, 5}};
    gert::StorageShape output_shape = {{}, {}};
    gert::StorageShape axis_storage_shape = {{3}, {3}};
    gert::StorageFormat axis_storage_fmt = {ge::FORMAT_ND, ge::FORMAT_ND, {}};
    ge::DataType axis_dt = ge::DT_INT64;
    auto input_tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * axes.size()]);
    auto holder = GetKernelRunContextHolder<int64_t>(input_tensor_holder, axes, input_shape, output_shape,
                                                     axis_storage_shape, axis_storage_fmt, axis_dt);
    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 5);
}

TEST_F(SqueezeV3UT, SqueezeV3_repeated_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV3")->infer_shape;

    std::vector<int64_t> axes = {0, 3, 3, 4};
    gert::StorageShape input_shape = {{1, 3, 2, -1, -1, 5}, {1, 3, 2, -1, -1, 5}};
    gert::StorageShape output_shape = {{}, {}};
    gert::StorageShape axis_storage_shape = {{4}, {4}};
    gert::StorageFormat axis_storage_fmt = {ge::FORMAT_ND, ge::FORMAT_ND, {}};
    ge::DataType axis_dt = ge::DT_INT64;
    auto input_tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * axes.size()]);
    auto holder = GetKernelRunContextHolder<int64_t>(input_tensor_holder, axes, input_shape, output_shape,
                                                     axis_storage_shape, axis_storage_fmt, axis_dt);
    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 5);
}

TEST_F(SqueezeV3UT, SqueezeV3_unsorted_axis)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = space_registry->GetOpImpl("SqueezeV3")->infer_shape;

    std::vector<int64_t> axes = {4, 0, 3};
    gert::StorageShape input_shape = {{1, 3, 2, -1, -1, 5}, {1, 3, 2, -1, -1, 5}};
    gert::StorageShape output_shape = {{}, {}};
    gert::StorageShape axis_storage_shape = {{3}, {3}};
    gert::StorageFormat axis_storage_fmt = {ge::FORMAT_ND, ge::FORMAT_ND, {}};
    ge::DataType axis_dt = ge::DT_INT64;
    auto input_tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * axes.size()]);
    auto holder = GetKernelRunContextHolder<int64_t>(input_tensor_holder, axes, input_shape, output_shape,
                                                     axis_storage_shape, axis_storage_fmt, axis_dt);
    EXPECT_EQ(infer_shape_func(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDimNum(), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(0), 3);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(1), 2);
    EXPECT_EQ(holder.GetContext()->GetOutputShape(0)->GetDim(2), 5);
    // the axes input may feed other nodes, its data must stay unchanged after infershape
    const auto axes_data_after = reinterpret_cast<const int64_t*>(input_tensor_holder.get() + sizeof(gert::Tensor));
    for (size_t idx = 0U; idx < axes.size(); ++idx) {
        EXPECT_EQ(axes_data_after[idx], axes[idx]);
    }
}

TEST_F(SqueezeV3UT, SqueezeV3_registered_success)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("SqueezeV3"), nullptr);
}

TEST_F(SqueezeV3UT, SqueezeV3_infer_datatype)
{
    auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(space_registry->GetOpImpl("SqueezeV3"), nullptr);
    auto op_impl = space_registry->GetOpImpl("SqueezeV3");
    ASSERT_NE(op_impl->infer_datatype, nullptr);

    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("SqueezeV3").OpName("SqueezeV3");
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
