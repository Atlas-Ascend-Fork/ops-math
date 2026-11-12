/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <vector>
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_pdist_forward.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

namespace op {
extern SocVersion g_socVersion;
extern NpuArch g_npuArchOverride;
} // namespace op

using namespace std;

class l2_pdist_forward_test : public testing::Test {
protected:
    // mock 平台（g_socVersion/g_npuArchOverride）为进程级全局量，正常用例将 arch 切至 DAV_3510；
    // SetUp/TearDown 精确保存并还原两个字段（对齐 test_aclnn_sin.cpp 既有模式，不经 setter），
    // 防止泄漏给同进程内后续执行的套件（如 pow 的 out dtype 白名单校验仅 DAV_2201 生效）
    void SetUp() override
    {
        savedSocVersion_ = op::g_socVersion;
        savedNpuArchOverride_ = op::g_npuArchOverride;
    }

    void TearDown() override
    {
        // Restore both stub fields without the setters' coupled updates.
        op::g_socVersion = savedSocVersion_;
        op::g_npuArchOverride = savedNpuArchOverride_;
    }

    static void SetUpTestCase() { cout << "pdist_forward_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "pdist_forward_test TearDown" << endl; }

private:
    op::SocVersion savedSocVersion_{op::SocVersion::RESERVED_VERSION};
    NpuArch savedNpuArchOverride_{NpuArch::DAV_RESV};
};

// self为空指针
TEST_F(l2_pdist_forward_test, case_forward_self_nullptr)
{
    auto p_desc = ScalarDesc(2.0f);
    auto out_desc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnPdistForward, INPUT((aclTensor*)nullptr, p_desc), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

// p为空指针
TEST_F(l2_pdist_forward_test, case_forward_p_nullptr)
{
    auto self_desc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnPdistForward, INPUT(self_desc, (aclScalar*)nullptr), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

// out为空指针
TEST_F(l2_pdist_forward_test, case_forward_out_nullptr)
{
    auto self_desc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto p_desc = ScalarDesc(2.0f);
    auto ut = OP_API_UT(aclnnPdistForward, INPUT(self_desc, p_desc), OUTPUT((aclTensor*)nullptr));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

// self不支持的数据类型
TEST_F(l2_pdist_forward_test, case_forward_self_dtype_not_support)
{
    auto self_desc = TensorDesc({4, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto p_desc = ScalarDesc(2.0f);
    auto out_desc = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnPdistForward, INPUT(self_desc, p_desc), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// out不支持的数据类型
TEST_F(l2_pdist_forward_test, case_forward_out_dtype_not_support)
{
    auto self_desc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto p_desc = ScalarDesc(2.0f);
    auto out_desc = TensorDesc({6}, ACL_INT64, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnPdistForward, INPUT(self_desc, p_desc), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self与out数据类型不一致
TEST_F(l2_pdist_forward_test, case_forward_dtype_not_match)
{
    auto self_desc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto p_desc = ScalarDesc(2.0f);
    auto out_desc = TensorDesc({6}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnPdistForward, INPUT(self_desc, p_desc), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self维度不为2
TEST_F(l2_pdist_forward_test, case_forward_self_wrong_dim)
{
    auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto p_desc = ScalarDesc(2.0f);
    auto out_desc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnPdistForward, INPUT(self_desc, p_desc), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// out维度不为1
TEST_F(l2_pdist_forward_test, case_forward_out_wrong_dim)
{
    auto self_desc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto p_desc = ScalarDesc(2.0f);
    auto out_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnPdistForward, INPUT(self_desc, p_desc), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// out的shape与期望的N*(N-1)/2不匹配
TEST_F(l2_pdist_forward_test, case_forward_out_shape_not_match)
{
    auto self_desc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto p_desc = ScalarDesc(2.0f);
    auto out_desc = TensorDesc({7}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnPdistForward, INPUT(self_desc, p_desc), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// p为负数
TEST_F(l2_pdist_forward_test, case_forward_p_negative)
{
    auto self_desc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto p_desc = ScalarDesc(-1.0f);
    auto out_desc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnPdistForward, INPUT(self_desc, p_desc), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 正常场景：float类型，p=2.0（欧式距离）
TEST_F(l2_pdist_forward_test, case_forward_float_p2)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    auto self_desc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto p_desc = ScalarDesc(2.0f);
    auto out_desc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnPdistForward, INPUT(self_desc, p_desc), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景：float类型，p=1.0（曼哈顿距离）
TEST_F(l2_pdist_forward_test, case_forward_float_p1)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    auto self_desc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto p_desc = ScalarDesc(1.0f);
    auto out_desc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnPdistForward, INPUT(self_desc, p_desc), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景：float16类型
TEST_F(l2_pdist_forward_test, case_forward_float16)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    auto self_desc = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto p_desc = ScalarDesc(1.5f);
    auto out_desc = TensorDesc({28}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnPdistForward, INPUT(self_desc, p_desc), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景：p=0
TEST_F(l2_pdist_forward_test, case_forward_p_zero)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    auto self_desc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto p_desc = ScalarDesc(0.0f);
    auto out_desc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnPdistForward, INPUT(self_desc, p_desc), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景：p为正无穷
TEST_F(l2_pdist_forward_test, case_forward_p_inf)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    auto self_desc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto p_desc = ScalarDesc(INFINITY);
    auto out_desc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnPdistForward, INPUT(self_desc, p_desc), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景：只有一行数据，输出空tensor
TEST_F(l2_pdist_forward_test, case_forward_single_row)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    auto self_desc = TensorDesc({1, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto p_desc = ScalarDesc(2.0f);
    auto out_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnPdistForward, INPUT(self_desc, p_desc), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_EQ(workspace_size, 0);
}

// 正常场景：两行数据，输出1个距离值
TEST_F(l2_pdist_forward_test, case_forward_two_rows)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    auto self_desc = TensorDesc({2, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto p_desc = ScalarDesc(2.0f);
    auto out_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnPdistForward, INPUT(self_desc, p_desc), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景：空tensor输入（第二维为0），走FillScalar分支
TEST_F(l2_pdist_forward_test, case_forward_empty_input)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    auto self_desc = TensorDesc({4, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto p_desc = ScalarDesc(2.0f);
    auto out_desc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnPdistForward, INPUT(self_desc, p_desc), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
