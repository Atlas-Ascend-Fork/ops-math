/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

#include "../../../op_api/aclnn_linalg_cholesky.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_linalg_cholesky_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "aclnnLinalgCholesky_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "aclnnLinalgCholesky_test TearDown" << std::endl; }
};

// checkNotNull
TEST_F(l2_linalg_cholesky_test, ascend910B_case_1)
{
    auto out_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnLinalgCholesky, INPUT((aclTensor*)nullptr, true), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid
TEST_F(l2_linalg_cholesky_test, ascend910B_case_2)
{
    auto self_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto out_desc = TensorDesc({3, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto ut = OP_API_UT(aclnnLinalgCholesky, INPUT(self_desc, true), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape
TEST_F(l2_linalg_cholesky_test, ascend910B_case_3)
{
    auto self_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto out_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto ut = OP_API_UT(aclnnLinalgCholesky, INPUT(self_desc, true), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape
TEST_F(l2_linalg_cholesky_test, ascend910B_case_4)
{
    auto self_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto out_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto ut = OP_API_UT(aclnnLinalgCholesky, INPUT(self_desc, false), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径
TEST_F(l2_linalg_cholesky_test, ascend910B_case_5)
{
    auto self_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto out_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto ut = OP_API_UT(aclnnLinalgCholesky, INPUT(self_desc, true), OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_linalg_cholesky_test, rejects_empty_non_square_matrix)
{
    auto tensor = TensorDesc({0, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnLinalgCholesky, INPUT(tensor, false), OUTPUT(tensor));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_linalg_cholesky_test, rejects_empty_matrix_over_size_limit)
{
    auto tensor = TensorDesc({0, 8193, 8193}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnLinalgCholesky, INPUT(tensor, false), OUTPUT(tensor));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_linalg_cholesky_test, valid_empty_matrices_require_no_workspace)
{
    for (const auto& shape : std::vector<std::vector<int64_t>>{{0, 0}, {3, 0, 0}, {0, 8192, 8192}}) {
        auto tensor = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnLinalgCholesky, INPUT(tensor, true), OUTPUT(tensor));
        uint64_t workspaceSize = 123;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
        EXPECT_EQ(workspaceSize, 0);
    }
}

TEST_F(l2_linalg_cholesky_test, ascend910B_case_6_null_workspace_or_executor)
{
    auto self = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
    auto out = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(aclnnLinalgCholeskyGetWorkspaceSize(self.get(), true, out.get(), nullptr, &executor),
              ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(aclnnLinalgCholeskyGetWorkspaceSize(self.get(), true, out.get(), &workspaceSize, nullptr),
              ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_linalg_cholesky_test, ascend910B_case_8_format_mismatch)
{
    auto ut = OP_API_UT(aclnnLinalgCholesky, INPUT(TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND), true),
                        OUTPUT(TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW)));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

// Base formats used by the framework remain accepted at the ACLNN boundary.
TEST_F(l2_linalg_cholesky_test, accepts_ncl_base_format)
{
    auto self = TensorDesc({2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto out = TensorDesc({2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto ut = OP_API_UT(aclnnLinalgCholesky, INPUT(self, false), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}
