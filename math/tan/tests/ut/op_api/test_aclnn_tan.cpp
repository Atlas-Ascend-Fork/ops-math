/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_tan.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;

class l2_tan_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "tan_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "tan_test TearDown" << endl; }
};

// test aicore type: FLOAT
TEST_F(l2_tan_test, case_tan_for_float_type)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.5, 1.5);
    auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnTan, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test aicore type: FLOAT16
TEST_F(l2_tan_test, case_tan_for_float16_type)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1.5, 1.5);
    auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnTan, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test aicore type: DOUBLE
TEST_F(l2_tan_test, case_tan_for_double_type)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1.5, 1.5);
    auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnTan, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test aicore type: BF16
TEST_F(l2_tan_test, case_tan_for_bf16_type)
{
    auto self_tensor_desc = TensorDesc({3, 3, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1.5, 1.5);
    auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnTan, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test 1d tensor
TEST_F(l2_tan_test, case_tan_for_1d_tensor)
{
    auto self_tensor_desc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.5, 1.5);
    auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnTan, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test 5d tensor
TEST_F(l2_tan_test, case_tan_for_5d_tensor)
{
    auto self_tensor_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.5, 1.5);
    auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnTan, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test empty tensor
TEST_F(l2_tan_test, case_tan_for_empty_tensor)
{
    auto self_tensor_desc = TensorDesc({0, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.5, 1.5);
    auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnTan, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test nullptr input
TEST_F(l2_tan_test, case_tan_nullptr_input)
{
    auto ut = OP_API_UT(aclnnTan, INPUT((aclTensor*)nullptr), OUTPUT((aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// test nullptr output
TEST_F(l2_tan_test, case_tan_nullptr_output)
{
    auto self_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.5, 1.5);
    auto ut = OP_API_UT(aclnnTan, INPUT(self_tensor_desc), OUTPUT(nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// test mismatched shape
TEST_F(l2_tan_test, case_tan_mismatched_shape)
{
    auto self_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.5, 1.5);
    auto out_tensor_desc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnTan, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// int32 入出同 dtype 被拒: 整型 self 归一为 FLOAT 后参与出参校验,
// 浮点转整型不被允许, 入参校验期以 161002 拒绝
TEST_F(l2_tan_test, case_tan_int32_in_out_direct)
{
    auto self_desc = TensorDesc({1, 16}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-100, 100);
    auto out_desc = TensorDesc({1, 16}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnTan, INPUT(self_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 整型 self 归一 FLOAT 路径(NEED_CAST_TO_FLOAT_DTYPE_LIST): int8 self → float out,
// Compute 内 l0op::Cast(int8→float) 后按 out dtype 收敛
TEST_F(l2_tan_test, case_tan_int8_to_float)
{
    auto self_desc = TensorDesc({1, 16}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-100, 100);
    auto out_desc = TensorDesc({1, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnTan, INPUT(self_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 浮点 self 转整型 out 被拒: 出参转换校验不允许浮点转整型(FLOAT→INT16),
// 入参校验期以 161002 拒绝
TEST_F(l2_tan_test, case_tan_out_dtype_out_of_support_domain)
{
    auto self_desc = TensorDesc({1, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.5, 1.5);
    auto out_desc = TensorDesc({1, 16}, ACL_INT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnTan, INPUT(self_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self dtype 不在支持列表(910B 列表无 UINT64) → 入参校验期拒绝
TEST_F(l2_tan_test, case_tan_self_dtype_not_supported)
{
    auto self_desc = TensorDesc({1, 16}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 100);
    auto out_desc = TensorDesc({1, 16}, ACL_UINT64, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnTan, INPUT(self_desc), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// test 9d input tensor
TEST_F(l2_tan_test, case_tan_for_9d_input_tensor)
{
    auto self_tensor_desc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.5, 1.5);
    auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnTan, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test 9d output tensor
TEST_F(l2_tan_test, case_tan_for_9d_output_tensor)
{
    auto self_tensor_desc = TensorDesc({1, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.5, 1.5);
    auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnTan, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test 10d tensor
TEST_F(l2_tan_test, case_tan_for_10d_tensor)
{
    auto self_tensor_desc = TensorDesc({1, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.5, 1.5);
    auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnTan, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 非 910B/910_93/RegBase 平台路由(ASCEND910): GetDtypeSupportList 走 ASCEND910 列表,
// FLOAT 合法 + l0op::Tan 走 AICORE 支持列表(TanAiCore 路由)
TEST_F(l2_tan_test, case_tan_platform_910_aicore_route)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910);
    {
        auto self_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.5, 1.5);
        auto out_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
        auto ut = OP_API_UT(aclnnTan, INPUT(self_desc), OUTPUT(out_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
}

// ASCEND910 平台 BF16 入参拒绝(910 列表不含 BF16; 910B 列表含)——平台门控双向
TEST_F(l2_tan_test, case_tan_platform_910_bf16_rejected)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910);
    {
        auto self_desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1.5, 1.5);
        auto out_desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnTan, INPUT(self_desc), OUTPUT(out_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
}

// aclnnInplaceTan 基本路径: FLOAT 就地计算成功(CheckParamsInplace + 空 executor 构造)
TEST_F(l2_tan_test, case_tan_inplace_float)
{
    auto self_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.5, 1.5);
    auto ut = OP_API_UT(aclnnInplaceTan, INPUT(self_desc), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnInplaceTan int32 就地被拒: 整型 self 归一为 FLOAT 后, 出参转换校验
// 不允许浮点转整型(FLOAT→INT32), 以 161002 拒绝
TEST_F(l2_tan_test, case_tan_inplace_int32)
{
    auto self_desc = TensorDesc({1, 16}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-100, 100);
    auto ut = OP_API_UT(aclnnInplaceTan, INPUT(self_desc), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// aclnnInplaceTan 空 tensor 早返回(selfRef->IsEmpty() → workspaceSize=0 直接成功)
TEST_F(l2_tan_test, case_tan_inplace_empty_tensor)
{
    auto self_desc = TensorDesc({0, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.5, 1.5);
    auto ut = OP_API_UT(aclnnInplaceTan, INPUT(self_desc), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// aclnnInplaceTan INT8 拒绝: 整型 self 归一为 FLOAT 后, 出参转换校验
// 不允许浮点转整型(FLOAT→INT8), 以 161002 拒绝
TEST_F(l2_tan_test, case_tan_inplace_int8_out_of_domain)
{
    auto self_desc = TensorDesc({1, 16}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-100, 100);
    auto ut = OP_API_UT(aclnnInplaceTan, INPUT(self_desc), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// aclnnInplaceTan 9d tensor 就地计算
TEST_F(l2_tan_test, case_tan_inplace_9d_tensor)
{
    auto self_desc = TensorDesc({1, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.5, 1.5);
    auto ut = OP_API_UT(aclnnInplaceTan, INPUT(self_desc), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
