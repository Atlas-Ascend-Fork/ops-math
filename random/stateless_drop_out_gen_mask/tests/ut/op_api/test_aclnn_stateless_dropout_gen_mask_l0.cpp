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
#include <cstdint>
#include <vector>

#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "random/stateless_drop_out_gen_mask/op_api/stateless_dropout_gen_mask.h"

using namespace op;
using namespace std;

namespace {
const int64_t DATA_SIZE = 64;
const int64_t BIT_NUMBER = 128;
const int64_t UINT8_BIT_NUMBER = 8;

// 与 op_api 实现保持一致的 mask 字节数推导，用于校验输出 shape
int64_t ExpectMaskByteSize(const std::vector<int64_t>& dims)
{
    int64_t size = 1;
    for (auto dim : dims) {
        size *= dim;
    }
    return (size + BIT_NUMBER - 1) / BIT_NUMBER * BIT_NUMBER / UINT8_BIT_NUMBER;
}
} // namespace

class StatelessDropoutGenMaskL0Test : public ::testing::Test {
public:
    void SetUp() override
    {
        auto executor = &exe_;
        auto uniqueExecutor = CREATE_EXECUTOR();
        uniqueExecutor.ReleaseTo(executor);
    }

    void TearDown() override
    {
        for (auto tensor : tensors_) {
            aclDestroyTensor(tensor);
        }
        tensors_.clear();
        for (auto array : intArrays_) {
            aclDestroyIntArray(array);
        }
        intArrays_.clear();
        delete exe_;
        exe_ = nullptr;
        // 复位平台信息，避免用例间相互污染
        SetPlatformNpuArch(NpuArch::DAV_2201);
    }

protected:
    aclTensor* CreateTensor(const std::vector<int64_t>& shape, aclDataType dtype)
    {
        auto tensor = aclCreateTensor(shape.data(), shape.size(), dtype, nullptr, 0, ACL_FORMAT_ND, shape.data(),
                                      shape.size(), data_);
        tensors_.push_back(tensor);
        return tensor;
    }

    const aclTensor* GenMask(const std::vector<int64_t>& dims, aclDataType seedDtype = ACL_INT64)
    {
        auto shape = aclCreateIntArray(dims.data(), dims.size());
        intArrays_.push_back(shape);
        auto prob = CreateTensor({1}, ACL_FLOAT);
        auto seed = CreateTensor({1}, seedDtype);
        auto seed1 = CreateTensor({1}, seedDtype);
        auto offset = CreateTensor({1}, ACL_INT64);
        return l0op::StatelessDropoutGenMask(shape, prob, seed, seed1, offset, exe_);
    }

    void CheckOut(const aclTensor* out, const std::vector<int64_t>& dims)
    {
        ASSERT_NE(out, nullptr);
        EXPECT_EQ(out->GetDataType(), DataType::DT_UINT8);
        EXPECT_EQ(out->GetViewShape().GetDimNum(), 1);
        EXPECT_EQ(out->GetViewShape().GetDim(0), ExpectMaskByteSize(dims));
    }

    aclOpExecutor* exe_ = nullptr;
    std::vector<aclTensor*> tensors_;
    std::vector<aclIntArray*> intArrays_;
    int64_t data_[DATA_SIZE] = {0};
};

// Ascend950（DAV_3510）走 AICore 分支
TEST_F(StatelessDropoutGenMaskL0Test, L0_StatelessDropoutGenMask_AICore_Arch3510)
{
    SetPlatformNpuArch(NpuArch::DAV_3510);
    const std::vector<int64_t> dims = {2, 3, 4};
    auto out = GenMask(dims);
    CheckOut(out, dims);
}

// Ascend960dt（DAV_9201）走 AICore 分支
TEST_F(StatelessDropoutGenMaskL0Test, L0_StatelessDropoutGenMask_AICore_Arch9201)
{
    SetPlatformNpuArch(NpuArch::DAV_9201);
    const std::vector<int64_t> dims = {8, 8};
    auto out = GenMask(dims, ACL_INT32);
    CheckOut(out, dims);
}

// Ascend960pr（DAV_9202）走 AICore 分支
TEST_F(StatelessDropoutGenMaskL0Test, L0_StatelessDropoutGenMask_AICore_Arch9202)
{
    SetPlatformNpuArch(NpuArch::DAV_9202);
    const std::vector<int64_t> dims = {128};
    auto out = GenMask(dims);
    CheckOut(out, dims);
}

// Ascend910B（DAV_2201）不在支持列表内，走 AICPU 分支
TEST_F(StatelessDropoutGenMaskL0Test, L0_StatelessDropoutGenMask_AiCpu_Arch2201)
{
    SetPlatformNpuArch(NpuArch::DAV_2201);
    const std::vector<int64_t> dims = {4, 5, 6};
    auto out = GenMask(dims);
    CheckOut(out, dims);
}

// Ascend910A（DAV_1001）不在支持列表内，走 AICPU 分支
TEST_F(StatelessDropoutGenMaskL0Test, L0_StatelessDropoutGenMask_AiCpu_Arch1001)
{
    SetPlatformNpuArch(NpuArch::DAV_1001);
    const std::vector<int64_t> dims = {1, 1, 1, 1};
    auto out = GenMask(dims, ACL_INT32);
    CheckOut(out, dims);
}

// 元素个数不能被 128 整除时按 bit 向上对齐
TEST_F(StatelessDropoutGenMaskL0Test, L0_StatelessDropoutGenMask_AlignUp)
{
    SetPlatformNpuArch(NpuArch::DAV_3510);
    const std::vector<int64_t> dims = {129};
    auto out = GenMask(dims);
    CheckOut(out, dims);
}

// 空 shape 时按 1 个元素处理
TEST_F(StatelessDropoutGenMaskL0Test, L0_StatelessDropoutGenMask_EmptyShape)
{
    SetPlatformNpuArch(NpuArch::DAV_3510);
    const std::vector<int64_t> dims = {};
    auto out = GenMask(dims);
    CheckOut(out, dims);
}

// 大 shape 覆盖 inferShape 累乘
TEST_F(StatelessDropoutGenMaskL0Test, L0_StatelessDropoutGenMask_LargeShape)
{
    SetPlatformNpuArch(NpuArch::DAV_3510);
    const std::vector<int64_t> dims = {1024, 1024, 4};
    auto out = GenMask(dims);
    CheckOut(out, dims);
}
