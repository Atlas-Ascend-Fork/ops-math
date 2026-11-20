/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include "gtest/gtest.h"

#include "opdev/platform.h"
#include "../../../op_host/op_api/aclnn_is_inf.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_is_inf_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_is_inf_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2_is_inf_test TearDown" << endl; }
};

// 正常场景：x 与 out 的 shape 一致
TEST_F(l2_is_inf_test, case_01_float)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto xDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInf, INPUT(xDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_is_inf_test, case_02_float16)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto xDesc = TensorDesc({1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({1, 2, 2}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInf, INPUT(xDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_is_inf_test, case_03_bfloat16)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto xDesc = TensorDesc({3, 2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({3, 2}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInf, INPUT(xDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 空Tensor：x 与 out 均为空，正常返回
TEST_F(l2_is_inf_test, case_04_empty_tensor)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto xDesc = TensorDesc({0, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({0, 4}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInf, INPUT(xDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 异常场景：x=[1,4]，out=[4]，shape 不一致，应返回 161002
TEST_F(l2_is_inf_test, case_05_out_shape_mismatch_rank)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto xDesc = TensorDesc({1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({4}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInf, INPUT(xDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：形状元素个数相同但维度排列不同，同样不一致
TEST_F(l2_is_inf_test, case_06_out_shape_mismatch_dim_order)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto xDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({3, 2}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInf, INPUT(xDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：out 的数据类型不是 BOOL
TEST_F(l2_is_inf_test, case_07_out_dtype_invalid)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto xDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInf, INPUT(xDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：x 的数据类型不在支持范围内
TEST_F(l2_is_inf_test, case_08_x_dtype_invalid)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto xDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInf, INPUT(xDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 310P 场景：FLOAT16 在支持范围内
TEST_F(l2_is_inf_test, case_09_310p_float16_success)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND310P);

    auto xDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInf, INPUT(xDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 310P 场景：BFLOAT16 不在支持范围内
TEST_F(l2_is_inf_test, case_10_310p_bfloat16_invalid)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND310P);

    auto xDesc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIsInf, INPUT(xDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
