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

#include "../../../../op_api/aclnn_bernoulli.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_bernoulli_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "bernoulli_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "bernoulli_test TearDown" << endl; }
};

struct BernoulliOutputLayout {
    vector<int64_t> shape;
    vector<int64_t> strides;
    vector<int64_t> storageShape;
    int64_t offset;
    aclnnStatus expected;
};

class l2_bernoulli_output_layout_test : public testing::TestWithParam<BernoulliOutputLayout> {};

TEST_P(l2_bernoulli_output_layout_test, tensor_output)
{
    const auto& layout = GetParam();
    auto self = TensorDesc(layout.shape, ACL_FLOAT16, ACL_FORMAT_ND);
    auto prob = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0.5f, 0.5f);
    auto out = TensorDesc(layout.shape, ACL_FLOAT16, ACL_FORMAT_ND, layout.strides, layout.offset, layout.storageShape);
    auto ut = OP_API_UT(aclnnBernoulliTensor, INPUT(self, prob, 53073689, 4), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), layout.expected);
}

TEST_P(l2_bernoulli_output_layout_test, scalar_output)
{
    const auto& layout = GetParam();
    auto self = TensorDesc(layout.shape, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc(layout.shape, ACL_FLOAT16, ACL_FORMAT_ND, layout.strides, layout.offset, layout.storageShape);
    auto ut = OP_API_UT(aclnnBernoulli, INPUT(self, ScalarDesc(0.5f), 53073689, 4), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), layout.expected);
}

TEST_P(l2_bernoulli_output_layout_test, inplace_tensor_output)
{
    const auto& layout = GetParam();
    auto self = TensorDesc(layout.shape, ACL_FLOAT16, ACL_FORMAT_ND, layout.strides, layout.offset,
                           layout.storageShape);
    auto prob = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0.5f, 0.5f);
    auto ut = OP_API_UT(aclnnInplaceBernoulliTensor, INPUT(self, prob, 53073689, 4), OUTPUT());
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), layout.expected);
}

TEST_P(l2_bernoulli_output_layout_test, inplace_scalar_output)
{
    const auto& layout = GetParam();
    auto self = TensorDesc(layout.shape, ACL_FLOAT16, ACL_FORMAT_ND, layout.strides, layout.offset,
                           layout.storageShape);
    auto ut = OP_API_UT(aclnnInplaceBernoulli, INPUT(self, ScalarDesc(0.5f), 53073689, 4), OUTPUT());
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), layout.expected);
}

INSTANTIATE_TEST_SUITE_P(output_layouts, l2_bernoulli_output_layout_test,
                         testing::Values(BernoulliOutputLayout{{875, 205}, {0, 0}, {1}, 0, ACLNN_ERR_PARAM_INVALID},
                                         BernoulliOutputLayout{{4, 5}, {0, 1}, {5}, 0, ACLNN_ERR_PARAM_INVALID},
                                         BernoulliOutputLayout{{4, 5}, {1, 0}, {4}, 0, ACLNN_ERR_PARAM_INVALID},
                                         BernoulliOutputLayout{{5, 4}, {1, 5}, {4, 5}, 0, ACLNN_SUCCESS},
                                         BernoulliOutputLayout{{4, 5}, {11, 2}, {44}, 1, ACLNN_SUCCESS},
                                         BernoulliOutputLayout{{1, 5}, {0, 1}, {5}, 0, ACLNN_SUCCESS},
                                         BernoulliOutputLayout{{5, 1}, {1, 0}, {5}, 0, ACLNN_SUCCESS},
                                         BernoulliOutputLayout{{0, 5}, {0, 0}, {0}, 0, ACLNN_SUCCESS},
                                         BernoulliOutputLayout{{}, {}, {}, 0, ACLNN_SUCCESS}));

TEST_F(l2_bernoulli_test, test_nc_out_broadcast_regression)
{
    auto self = TensorDesc({875, 205}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-0.00006103515625f, -0.00006103515625f);
    auto prob = TensorDesc({875, 1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0.5f, 0.5f);
    auto out = TensorDesc({875, 205}, ACL_FLOAT16, ACL_FORMAT_ND, {0, 0}, 0, {1});
    auto ut = OP_API_UT(aclnnBernoulliTensor, INPUT(self, prob, 53073689, 4), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_bernoulli_test, test_broadcast_inputs_non_overlapping_out)
{
    // Only write targets are checked: broadcasting either read-only input is valid.
    auto self = TensorDesc({5, 4}, ACL_FLOAT16, ACL_FORMAT_ND, {0, 0}, 0, {1});
    auto prob = TensorDesc({5, 4}, ACL_FLOAT16, ACL_FORMAT_ND, {0, 0}, 0, {1}).ValueRange(0.5f, 0.5f);
    auto out = TensorDesc({5, 4}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 5}, 0, {4, 5});
    auto ut = OP_API_UT(aclnnBernoulliTensor, INPUT(self, prob, 53073689, 4), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(l2_bernoulli_test, invalid_probability_broadcast_is_rejected_before_executor_creation)
{
    for (const auto& shape : vector<vector<int64_t>>{{2, 8, 8}, {8, 2}, {0}}) {
        auto self = TensorDesc({8, 8}, ACL_FLOAT, ACL_FORMAT_ND);
        auto prob = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
        auto out = TensorDesc({8, 8}, ACL_FLOAT, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnBernoulliTensor, INPUT(self, prob, 6, 8), OUTPUT(out));
        uint64_t workspaceSize = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
        auto inplace = OP_API_UT(aclnnInplaceBernoulliTensor, INPUT(self, prob, 6, 8), OUTPUT());
        EXPECT_EQ(inplace.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
    }
}

// Test for precision
TEST_F(l2_bernoulli_test, test_float_nchw)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW)
                           .ValueRange(-2, 2)
                           .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.0001, 0.0001);
    auto scalar_desc = ScalarDesc(1.0f);
    auto ut = OP_API_UT(aclnnBernoulli, INPUT(tensor_desc, scalar_desc, 0, 0), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test for precision
TEST_F(l2_bernoulli_test, test_float16_nchw)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW)
                           .ValueRange(-2, 2)
                           .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.0001, 0.0001);
    auto scalar_desc = ScalarDesc(1.0f);
    auto ut = OP_API_UT(aclnnBernoulli, INPUT(tensor_desc, scalar_desc, 0, 0), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test for precision
TEST_F(l2_bernoulli_test, test_int8_nchw)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_INT8, ACL_FORMAT_NCHW)
                           .ValueRange(-2, 2)
                           .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.0001, 0.0001);
    auto scalar_desc = ScalarDesc(1.0f);
    auto ut = OP_API_UT(aclnnBernoulli, INPUT(tensor_desc, scalar_desc, 0, 0), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test for precision
TEST_F(l2_bernoulli_test, test_int32_nchw)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_INT32, ACL_FORMAT_NCHW)
                           .ValueRange(-2, 2)
                           .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.0001, 0.0001);
    auto scalar_desc = ScalarDesc(1.0f);
    auto ut = OP_API_UT(aclnnBernoulli, INPUT(tensor_desc, scalar_desc, 0, 0), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test for precision
TEST_F(l2_bernoulli_test, test_uint8_nchw)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_UINT8, ACL_FORMAT_NCHW)
                           .ValueRange(-2, 2)
                           .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.0001, 0.0001);
    auto scalar_desc = ScalarDesc(1.0f);
    auto ut = OP_API_UT(aclnnBernoulli, INPUT(tensor_desc, scalar_desc, 0, 0), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test for precision
TEST_F(l2_bernoulli_test, test_bool_nchw)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_BOOL, ACL_FORMAT_NCHW)
                           .ValueRange(-2, 2)
                           .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.0001, 0.0001);
    auto scalar_desc = ScalarDesc(1.0f);
    auto ut = OP_API_UT(aclnnBernoulli, INPUT(tensor_desc, scalar_desc, 0, 0), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test for precision
TEST_F(l2_bernoulli_test, ascend910B_test_bf16_nchw)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_BF16, ACL_FORMAT_NCHW)
                           .ValueRange(-2, 2)
                           .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.0001, 0.0001);
    auto scalar_desc = ScalarDesc(1.0f);
    auto ut = OP_API_UT(aclnnBernoulli, INPUT(tensor_desc, scalar_desc, 0, 0), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test for precision
TEST_F(l2_bernoulli_test, test_float_nhwc)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_NHWC)
                           .ValueRange(-2, 2)
                           .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.0001, 0.0001);
    auto scalar_desc = ScalarDesc(1.0f);
    auto ut = OP_API_UT(aclnnBernoulli, INPUT(tensor_desc, scalar_desc, 0, 0), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test for precision
TEST_F(l2_bernoulli_test, test_float_nd)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                           .ValueRange(-2, 2)
                           .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.0001, 0.0001);
    auto scalar_desc = ScalarDesc(1.0f);
    auto ut = OP_API_UT(aclnnBernoulli, INPUT(tensor_desc, scalar_desc, 0, 0), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test for precision
TEST_F(l2_bernoulli_test, test_float_nc1hwc0)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_NC1HWC0)
                           .ValueRange(-2, 2)
                           .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.0001, 0.0001);
    auto scalar_desc = ScalarDesc(1.0f);
    auto ut = OP_API_UT(aclnnBernoulli, INPUT(tensor_desc, scalar_desc, 0, 0), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Test for precision
TEST_F(l2_bernoulli_test, test_float_hwcn)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_HWCN)
                           .ValueRange(-2, 2)
                           .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.0001, 0.0001);
    auto scalar_desc = ScalarDesc(1.0f);
    auto ut = OP_API_UT(aclnnBernoulli, INPUT(tensor_desc, scalar_desc, 0, 0), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test for precision
TEST_F(l2_bernoulli_test, test_float_ndhwc)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_NDHWC)
                           .ValueRange(-2, 2)
                           .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.0001, 0.0001);
    auto scalar_desc = ScalarDesc(1.0f);
    auto ut = OP_API_UT(aclnnBernoulli, INPUT(tensor_desc, scalar_desc, 0, 0), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test for precision
TEST_F(l2_bernoulli_test, test_empty_tensor)
{
    auto tensor_desc = TensorDesc({4, 0, 4}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.0001, 0.0001);
    auto scalar_desc = ScalarDesc(1.0f);
    auto ut = OP_API_UT(aclnnBernoulli, INPUT(tensor_desc, scalar_desc, 0, 0), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test for non-continuous
TEST_F(l2_bernoulli_test, test_non_continuous)
{
    auto tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.0001, 0.0001);
    auto scalar_desc = ScalarDesc(1.0f);
    auto ut = OP_API_UT(aclnnBernoulli, INPUT(tensor_desc, scalar_desc, 0, 0), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 异常场景：double、9维、返回ACLNN_ERR_PARAM_INVALID
TEST_F(l2_bernoulli_test, test_err_long_shape)
{
    auto selfDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(selfDesc).Precision(0.0001, 0.0001);
    auto scalar_desc = ScalarDesc(1.0f);
    auto ut = OP_API_UT(aclnnBernoulli, INPUT(selfDesc, scalar_desc, 0, 0), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Test for precision
TEST_F(l2_bernoulli_test, ascend910B_test_bf16_nchw_prob02)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_BF16, ACL_FORMAT_NCHW)
                           .ValueRange(-2, 2)
                           .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.0001, 0.0001);
    auto scalar_desc = ScalarDesc(0.2f);
    auto ut = OP_API_UT(aclnnBernoulli, INPUT(tensor_desc, scalar_desc, 0, 0), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACL_SUCCESS);
}

// Test for precision
TEST_F(l2_bernoulli_test, ascend910B_test_fp16_nchw_prob02)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW)
                           .ValueRange(-2, 2)
                           .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.0001, 0.0001);
    auto scalar_desc = ScalarDesc(0.2f);
    auto ut = OP_API_UT(aclnnBernoulli, INPUT(tensor_desc, scalar_desc, 0, 0), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACL_SUCCESS);
}

// Test for precision
TEST_F(l2_bernoulli_test, ascend910B_test_float_nchw_prob02)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW)
                           .ValueRange(-2, 2)
                           .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.0001, 0.0001);
    auto scalar_desc = ScalarDesc(0.9f);
    auto ut = OP_API_UT(aclnnBernoulli, INPUT(tensor_desc, scalar_desc, 0, 0), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACL_SUCCESS);
}

// 异常场景：double、9维、返回ACLNN_ERR_PARAM_INVALID
TEST_F(l2_bernoulli_test, test_err_long_shape_tensor)
{
    auto selfDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(selfDesc).Precision(0.0001, 0.0001);
    auto prob_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBernoulliTensor, INPUT(selfDesc, prob_desc, 0, 0), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Test for prob tensor
TEST_F(l2_bernoulli_test, test_prob_tensor)
{
    auto tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.0001, 0.0001);
    auto prob_desc = TensorDesc({4, 4}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBernoulliTensor, INPUT(tensor_desc, prob_desc, 0, 0), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Test 910D
// TEST_F(l2_bernoulli_test, ascend910D_case_tensor_prob_tensor) {
//   auto tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-2, 2);
//   auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.0001, 0.0001);
//   auto prob_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND);
//   auto ut = OP_API_UT(aclnnBernoulliTensor, INPUT(tensor_desc, prob_desc, 0, 0), OUTPUT(out_tensor_desc));

//   // SAMPLE: only test GetWorkspaceSize
//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_bernoulli_test, ascend910D_case_float_shape_tensor) {
//   auto selfDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND);
//   auto out_tensor_desc = TensorDesc(selfDesc).Precision(0.0001, 0.0001);
//   auto prob_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND);
//   auto ut = OP_API_UT(aclnnBernoulliTensor, INPUT(selfDesc, prob_desc, 0, 0), OUTPUT(out_tensor_desc));

//   // SAMPLE: only test GetWorkspaceSize
//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }
