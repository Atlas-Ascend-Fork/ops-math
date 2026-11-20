/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "math/arg_max_with_value/op_api/aclnn_max_dim.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

class l2_argmax_with_value_indices_dtype_test : public testing::Test {};

TEST_F(l2_argmax_with_value_indices_dtype_test, aclnnMaxDim_3_4_float_nd_float_indices)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto indices_tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out_tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMaxDim, INPUT(self_tensor_desc, dim, keepdim),
                        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_argmax_with_value_indices_dtype_test, aclnnMaxDim_3_4_float_nd_bool_indices)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto indices_tensor_desc = TensorDesc({4}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out_tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMaxDim, INPUT(self_tensor_desc, dim, keepdim),
                        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_argmax_with_value_indices_dtype_test, aclnnMaxDim_3_4_float_nd_int32_indices)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto indices_tensor_desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out_tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMaxDim, INPUT(self_tensor_desc, dim, keepdim),
                        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_argmax_with_value_indices_dtype_test, aclnnMaxDim_empty_float_nd_float_indices)
{
    const int64_t dim = 0;
    const bool keepdim = false;

    auto self_tensor_desc = TensorDesc({0, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto indices_tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto out_tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMaxDim, INPUT(self_tensor_desc, dim, keepdim),
                        OUTPUT(out_tensor_desc, indices_tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
