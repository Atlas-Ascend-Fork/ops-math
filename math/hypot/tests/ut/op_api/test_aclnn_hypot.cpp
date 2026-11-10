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

#include "opdev/platform.h"
#include "../../../op_api/aclnn_hypot.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_hypot_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_hypot_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2_hypot_test TearDown" << endl; }
};

TEST_F(l2_hypot_test, case_01_float)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto x1Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{3.0, 5.0, 8.0, 7.0});
    auto x2Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{4.0, 12.0, 6.0, 24.0});
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHypot, INPUT(x1Desc, x2Desc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_hypot_test, case_02_float16)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto x1Desc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{3.0, 5.0, 8.0, 7.0});
    auto x2Desc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{4.0, 12.0, 6.0, 24.0});
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHypot, INPUT(x1Desc, x2Desc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_hypot_test, case_03_bfloat16)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto x1Desc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{3.0, 5.0, 8.0, 7.0});
    auto x2Desc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{4.0, 12.0, 6.0, 24.0});
    auto outDesc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHypot, INPUT(x1Desc, x2Desc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_hypot_test, case_04_mixed_float_float16)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto x1Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{3.0, 5.0, 8.0, 7.0});
    auto x2Desc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{4.0, 12.0, 6.0, 24.0});
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHypot, INPUT(x1Desc, x2Desc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_hypot_test, case_05_mixed_float16_bfloat16)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto x1Desc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{3.0, 5.0, 8.0, 7.0});
    auto x2Desc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{4.0, 12.0, 6.0, 24.0});
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHypot, INPUT(x1Desc, x2Desc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_hypot_test, case_06_broadcast_right_pad)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto x1Desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto x2Desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHypot, INPUT(x1Desc, x2Desc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_hypot_test, case_07_broadcast_cross)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto x1Desc = TensorDesc({1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto x2Desc = TensorDesc({3, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto outDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHypot, INPUT(x1Desc, x2Desc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_hypot_test, case_08_empty_tensor)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto x1Desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHypot, INPUT(x1Desc, x2Desc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_hypot_test, case_09_not_contiguous)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto x1Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {2, 2}).Value(vector<float>{1, 2, 3, 4});
    auto x2Desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {2, 2}).Value(vector<float>{1, 2, 3, 4});
    auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHypot, INPUT(x1Desc, x2Desc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_hypot_test, case_10_out_shape_mismatch)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto x1Desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHypot, INPUT(x1Desc, x2Desc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_hypot_test, case_11_broadcast_incompatible)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto x1Desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHypot, INPUT(x1Desc, x2Desc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_hypot_test, case_12_x1_null)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto x1Desc = nullptr;
    auto x2Desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHypot, INPUT(x1Desc, x2Desc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_hypot_test, case_13_x2_null)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto x1Desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = nullptr;
    auto outDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHypot, INPUT(x1Desc, x2Desc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_hypot_test, case_14_out_null)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto x1Desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = nullptr;

    auto ut = OP_API_UT(aclnnHypot, INPUT(x1Desc, x2Desc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_hypot_test, case_15_dim_gt_8)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto x1Desc = TensorDesc({3, 2, 1, 2, 1, 3, 2, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({3, 2, 1, 2, 1, 3, 2, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({3, 2, 1, 2, 1, 3, 2, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHypot, INPUT(x1Desc, x2Desc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_hypot_test, case_16_dtype_unsupported)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2201);

    auto x1Desc = TensorDesc({3, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto x2Desc = TensorDesc({3, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outDesc = TensorDesc({3, 3}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHypot, INPUT(x1Desc, x2Desc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
