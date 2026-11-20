/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include "gtest/gtest.h"

#include "conversion/fill_diagonal_v2/op_api/aclnn_fill_diagonal.h"
#include "opdev/platform.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;

class l2_inplace_fill_diagonal_aicore_test : public testing::Test {
protected:
    void SetUp() override { SetPlatformSocVersion(SocVersion::ASCEND910B); }
    void TearDown() override { SetPlatformSocVersion(SocVersion::ASCEND910B); }
};

TEST_F(l2_inplace_fill_diagonal_aicore_test, case_aicpu_default_platform)
{
    auto self_tensor_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalar_desc = ScalarDesc(1.0f);
    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, false), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_fill_diagonal_aicore_test, case_aicore_950_nd)
{
    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto self_tensor_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalar_desc = ScalarDesc(1.0f);
    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, true), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_fill_diagonal_aicore_test, case_aicore_950_bool_tiny_float)
{
    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto self_tensor_desc = TensorDesc({3, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(1e-8f);
    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, false), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_fill_diagonal_aicore_test, case_aicore_950_nchw_format)
{
    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    auto scalar_desc = ScalarDesc(1.0f);
    auto ut = OP_API_UT(aclnnInplaceFillDiagonal, INPUT(self_tensor_desc, scalar_desc, false), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
