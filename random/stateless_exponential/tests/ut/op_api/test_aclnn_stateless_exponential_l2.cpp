/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_stateless_exponential_l2.cpp
 * \brief aclnnStatelessExponentialTensor L2 UT
 *
 * 覆盖 op_api/aclnn_stateless_exponential.cpp 的入参校验与构图：
 *   正常域: fp32/fp16/bf16、小值 lambd、subnormal lambd、非零 seed/offset、空 self、多维
 *   极小值: double 合法 > 0 的极小 lambd 在 aclnn 层按 double 原值校验放行(构图通过)，
 *           double->fp32 下溢为 0 后的拒绝发生在 tiling 阶段(由 op_host UT 覆盖)
 *   异常域: lambd<=0、nullptr、不支持的 dtype、seed/offset 维度非 1、self 超 8 维
 */
#include "gtest/gtest.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "random/stateless_exponential/op_api/aclnn_stateless_exponential.h"

using namespace op;
using namespace std;

class l2_stateless_exponential_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_stateless_exponential_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2_stateless_exponential_test TearDown" << endl; }
};

// ======================== 正常场景 ========================

TEST_F(l2_stateless_exponential_test, case_normal_float32)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto seedDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offsetDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    int64_t offset = 0;
    double lambd = 1.0;
    auto ut = OP_API_UT(aclnnStatelessExponentialTensor, INPUT(selfDesc, seedDesc, offsetDesc, offset, lambd),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_stateless_exponential_test, case_normal_float16)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto seedDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offsetDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    int64_t offset = 0;
    double lambd = 1.0;
    auto ut = OP_API_UT(aclnnStatelessExponentialTensor, INPUT(selfDesc, seedDesc, offsetDesc, offset, lambd),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_stateless_exponential_test, case_normal_bfloat16)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
    auto seedDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offsetDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    int64_t offset = 0;
    double lambd = 1.0;
    auto ut = OP_API_UT(aclnnStatelessExponentialTensor, INPUT(selfDesc, seedDesc, offsetDesc, offset, lambd),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_stateless_exponential_test, case_normal_lambd_small)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto seedDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offsetDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    int64_t offset = 0;
    double lambd = 1e-10; // double 与 fp32 均可精确表示
    auto ut = OP_API_UT(aclnnStatelessExponentialTensor, INPUT(selfDesc, seedDesc, offsetDesc, offset, lambd),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_stateless_exponential_test, case_normal_lambd_subnormal)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto seedDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offsetDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    int64_t offset = 0;
    double lambd = 1e-38; // fp32 subnormal, 下发后保留非 0, 可正常构图
    auto ut = OP_API_UT(aclnnStatelessExponentialTensor, INPUT(selfDesc, seedDesc, offsetDesc, offset, lambd),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_stateless_exponential_test, case_normal_nonzero_seed_offset)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto seedDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offsetDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    int64_t offset = 8;
    double lambd = 1.0;
    auto ut = OP_API_UT(aclnnStatelessExponentialTensor, INPUT(selfDesc, seedDesc, offsetDesc, offset, lambd),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_stateless_exponential_test, case_normal_empty_self)
{
    auto selfDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto seedDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offsetDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    int64_t offset = 0;
    double lambd = 1.0;
    auto ut = OP_API_UT(aclnnStatelessExponentialTensor, INPUT(selfDesc, seedDesc, offsetDesc, offset, lambd),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_stateless_exponential_test, case_normal_multi_dim)
{
    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto seedDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offsetDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    int64_t offset = 0;
    double lambd = 2.0;
    auto ut = OP_API_UT(aclnnStatelessExponentialTensor, INPUT(selfDesc, seedDesc, offsetDesc, offset, lambd),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// ======================== 极小值场景 ========================

// 极小值 lambd(double 合法 > 0): aclnn 层按 double 原值校验放行、构图通过;
// double->fp32 下溢为 0 后被 tiling 层拒绝(该阶段由 op_host UT 覆盖)
TEST_F(l2_stateless_exponential_test, case_lambd_extreme_small_1em46)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto seedDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offsetDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    int64_t offset = 0;
    double lambd = 1e-46; // < 2^-150, double->fp32 下溢为 0
    auto ut = OP_API_UT(aclnnStatelessExponentialTensor, INPUT(selfDesc, seedDesc, offsetDesc, offset, lambd),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_stateless_exponential_test, case_lambd_extreme_small_1em300)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto seedDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offsetDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    int64_t offset = 0;
    double lambd = 1e-300; // 极端极小值, double 有效但 fp32 必下溢为 0
    auto ut = OP_API_UT(aclnnStatelessExponentialTensor, INPUT(selfDesc, seedDesc, offsetDesc, offset, lambd),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// ======================== 异常场景 ========================

TEST_F(l2_stateless_exponential_test, case_self_nullptr)
{
    auto seedDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offsetDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    int64_t offset = 0;
    double lambd = 1.0;
    auto ut = OP_API_UT(aclnnStatelessExponentialTensor,
                        INPUT(((aclTensor*)nullptr), seedDesc, offsetDesc, offset, lambd), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_stateless_exponential_test, case_seed_nullptr)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto offsetDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    int64_t offset = 0;
    double lambd = 1.0;
    auto ut = OP_API_UT(aclnnStatelessExponentialTensor,
                        INPUT(selfDesc, ((aclTensor*)nullptr), offsetDesc, offset, lambd), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_stateless_exponential_test, case_offset_nullptr)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto seedDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    int64_t offset = 0;
    double lambd = 1.0;
    auto ut = OP_API_UT(aclnnStatelessExponentialTensor,
                        INPUT(selfDesc, seedDesc, ((aclTensor*)nullptr), offset, lambd), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_stateless_exponential_test, case_lambd_zero)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto seedDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offsetDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    int64_t offset = 0;
    double lambd = 0.0;
    auto ut = OP_API_UT(aclnnStatelessExponentialTensor, INPUT(selfDesc, seedDesc, offsetDesc, offset, lambd),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_stateless_exponential_test, case_lambd_negative)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto seedDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offsetDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    int64_t offset = 0;
    double lambd = -1.0;
    auto ut = OP_API_UT(aclnnStatelessExponentialTensor, INPUT(selfDesc, seedDesc, offsetDesc, offset, lambd),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_stateless_exponential_test, case_self_dtype_int32)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto seedDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offsetDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    int64_t offset = 0;
    double lambd = 1.0;
    auto ut = OP_API_UT(aclnnStatelessExponentialTensor, INPUT(selfDesc, seedDesc, offsetDesc, offset, lambd),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_stateless_exponential_test, case_seed_dtype_int32)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto seedDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    auto offsetDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    int64_t offset = 0;
    double lambd = 1.0;
    auto ut = OP_API_UT(aclnnStatelessExponentialTensor, INPUT(selfDesc, seedDesc, offsetDesc, offset, lambd),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_stateless_exponential_test, case_seed_dim_not_1)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto seedDesc = TensorDesc({1, 1}, ACL_INT64, ACL_FORMAT_ND);
    auto offsetDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    int64_t offset = 0;
    double lambd = 1.0;
    auto ut = OP_API_UT(aclnnStatelessExponentialTensor, INPUT(selfDesc, seedDesc, offsetDesc, offset, lambd),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_stateless_exponential_test, case_self_9dim)
{
    auto selfDesc = TensorDesc({2, 3, 2, 2, 3, 2, 3, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto seedDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offsetDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    int64_t offset = 0;
    double lambd = 1.0;
    auto ut = OP_API_UT(aclnnStatelessExponentialTensor, INPUT(selfDesc, seedDesc, offsetDesc, offset, lambd),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
