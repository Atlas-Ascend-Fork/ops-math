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
#include <array>
#include <limits>
#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_uniform.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/inner/types.h"

using namespace std;

class l2_uniform_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_uniform_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2_uniform_test TearDown" << endl; }
};

TEST_F(l2_uniform_test, aclnnUniform_3_4_float_nd)
{
    const vector<int64_t>& self_shape = {3, 4};
    aclDataType self_dtype = ACL_FLOAT;
    aclFormat self_format = ACL_FORMAT_ND;

    double from = 2;
    double to = 4;
    uint64_t seed = 0;
    uint64_t offset = 0;

    const vector<int64_t>& out_shape = {3, 4};
    aclDataType out_dtype = ACL_FLOAT;
    aclFormat out_format = ACL_FORMAT_ND;

    auto self_tensor_desc = TensorDesc(self_shape, self_dtype, self_format);

    auto ut = OP_API_UT(aclnnInplaceUniform, INPUT(self_tensor_desc, from, to, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACL_SUCCESS);
}

TEST_F(l2_uniform_test, aclnnUniform_5_4_6_float16_nd)
{
    double from = 2;
    double to = 4;
    uint64_t seed = 0;
    uint64_t offset = 0;

    auto self_tensor_desc = TensorDesc({5, 4, 6}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceUniform, INPUT(self_tensor_desc, from, to, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACL_SUCCESS);
}

// 异常场景：from 为 NaN, NaN 使比较恒为 false 绕过 from > to 校验, 应返回参数错误
TEST_F(l2_uniform_test, aclnnUniform_5_4_6_from_nan)
{
    double from = std::numeric_limits<double>::quiet_NaN();
    double to = 1;
    uint64_t seed = 0;
    uint64_t offset = 0;

    auto self_tensor_desc = TensorDesc({5, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceUniform, INPUT(self_tensor_desc, from, to, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：to 为 NaN, NaN 使比较恒为 false 绕过 from > to 校验, 应返回参数错误
TEST_F(l2_uniform_test, aclnnUniform_5_4_6_to_nan)
{
    double from = 0;
    double to = std::numeric_limits<double>::quiet_NaN();
    uint64_t seed = 0;
    uint64_t offset = 0;

    auto self_tensor_desc = TensorDesc({5, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceUniform, INPUT(self_tensor_desc, from, to, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：from 为 -inf, 超出数据类型表示范围, 应返回参数错误
TEST_F(l2_uniform_test, aclnnUniform_5_4_6_from_neg_inf)
{
    double from = -std::numeric_limits<double>::infinity();
    double to = 1;
    uint64_t seed = 0;
    uint64_t offset = 0;

    auto self_tensor_desc = TensorDesc({5, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceUniform, INPUT(self_tensor_desc, from, to, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：to 为 +inf, 超出数据类型表示范围, 应返回参数错误
TEST_F(l2_uniform_test, aclnnUniform_5_4_6_to_pos_inf)
{
    double from = 0;
    double to = std::numeric_limits<double>::infinity();
    uint64_t seed = 0;
    uint64_t offset = 0;

    auto self_tensor_desc = TensorDesc({5, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceUniform, INPUT(self_tensor_desc, from, to, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：int8 [0, 1000), 右端点 to-1=999 超出 int8 上界 127, 应返回参数错误
TEST_F(l2_uniform_test, aclnnUniform_5_4_6_int8_to_out_of_dtype_range)
{
    double from = 0;
    double to = 1000;
    uint64_t seed = 0;
    uint64_t offset = 0;

    auto self_tensor_desc = TensorDesc({5, 4, 6}, ACL_INT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceUniform, INPUT(self_tensor_desc, from, to, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：int8 [-129, 0), from 超出 int8 下界 -128, 应返回参数错误
TEST_F(l2_uniform_test, aclnnUniform_5_4_6_int8_from_out_of_dtype_range)
{
    double from = -129;
    double to = 0;
    uint64_t seed = 0;
    uint64_t offset = 0;

    auto self_tensor_desc = TensorDesc({5, 4, 6}, ACL_INT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceUniform, INPUT(self_tensor_desc, from, to, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：float16 [-65505, 0), from 超出 float16 表示范围, 应返回参数错误
TEST_F(l2_uniform_test, aclnnUniform_5_4_6_float16_from_out_of_dtype_range)
{
    double from = -65505;
    double to = 0;
    uint64_t seed = 0;
    uint64_t offset = 0;

    auto self_tensor_desc = TensorDesc({5, 4, 6}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceUniform, INPUT(self_tensor_desc, from, to, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACLNN_ERR_PARAM_INVALID);
}

// 正常场景：int8 满量程 [-128, 128), 右端点按 to-1=127 校验, 不应被 dtype 范围校验误拦,
// 与 aclnnInplaceRandom 对同一区间的接受口径一致
TEST_F(l2_uniform_test, aclnnUniform_int8_full_dtype_range)
{
    double from = -128;
    double to = 128;
    uint64_t seed = 0;
    uint64_t offset = 0;

    auto self_tensor_desc = TensorDesc({3, 4}, ACL_INT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceUniform, INPUT(self_tensor_desc, from, to, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACL_SUCCESS);
}

// 正常场景：uint8 满量程 [0, 256), 右端点按 to-1=255 校验, 不应被 dtype 范围校验误拦
TEST_F(l2_uniform_test, aclnnUniform_uint8_full_dtype_range)
{
    double from = 0;
    double to = 256;
    uint64_t seed = 0;
    uint64_t offset = 0;

    auto self_tensor_desc = TensorDesc({3, 4}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceUniform, INPUT(self_tensor_desc, from, to, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACL_SUCCESS);
}

// 异常场景：Tensor 变体 from 为 NaN, 应返回参数错误
TEST_F(l2_uniform_test, aclnnUniformTensor_5_4_6_from_nan)
{
    double from = std::numeric_limits<double>::quiet_NaN();
    double to = 1;
    uint64_t offset = 0;

    auto self_tensor_desc = TensorDesc({5, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto seed_tensor_desc = TensorDesc({1}, ACL_UINT64, ACL_FORMAT_ND);
    auto offset_tensor_desc = TensorDesc({1}, ACL_UINT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceUniformTensor,
                        INPUT(self_tensor_desc, from, to, seed_tensor_desc, offset_tensor_desc, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：Tensor 变体 to 为 +inf, 应返回参数错误
TEST_F(l2_uniform_test, aclnnUniformTensor_5_4_6_to_pos_inf)
{
    double from = 0;
    double to = std::numeric_limits<double>::infinity();
    uint64_t offset = 0;

    auto self_tensor_desc = TensorDesc({5, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto seed_tensor_desc = TensorDesc({1}, ACL_UINT64, ACL_FORMAT_ND);
    auto offset_tensor_desc = TensorDesc({1}, ACL_UINT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceUniformTensor,
                        INPUT(self_tensor_desc, from, to, seed_tensor_desc, offset_tensor_desc, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：Tensor 变体 int8 [0, 1000), 右端点 to-1=999 超出 int8 上界 127, 应返回参数错误
TEST_F(l2_uniform_test, aclnnUniformTensor_5_4_6_int8_to_out_of_dtype_range)
{
    double from = 0;
    double to = 1000;
    uint64_t offset = 0;

    auto self_tensor_desc = TensorDesc({5, 4, 6}, ACL_INT8, ACL_FORMAT_ND);
    auto seed_tensor_desc = TensorDesc({1}, ACL_UINT64, ACL_FORMAT_ND);
    auto offset_tensor_desc = TensorDesc({1}, ACL_UINT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceUniformTensor,
                        INPUT(self_tensor_desc, from, to, seed_tensor_desc, offset_tensor_desc, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_uniform_test, aclnnUniform_5_4_6_invaild_from_to)
{
    double from = 3;
    double to = 2;
    uint64_t seed = 0;
    uint64_t offset = 0;

    auto self_tensor_desc = TensorDesc({5, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceUniform, INPUT(self_tensor_desc, from, to, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_uniform_test, aclnnUniform_5_4_6_input_nullptr)
{
    double from = 2;
    double to = 3;
    uint64_t seed = 0;
    uint64_t offset = 0;

    auto self_tensor_desc = TensorDesc({5, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceUniform, INPUT((aclTensor*)nullptr, from, to, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_uniform_test, aclnnUniform_5_4_6_invalid_dtype)
{
    double from = 2;
    double to = 3;
    uint64_t seed = 0;
    uint64_t offset = 0;

    auto self_tensor_desc = TensorDesc({5, 4, 6}, ACL_UINT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceUniform, INPUT(self_tensor_desc, from, to, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_uniform_test, aclnnUniform_5_4_6_invalid_shape)
{
    double from = 2;
    double to = 3;
    uint64_t seed = 0;
    uint64_t offset = 0;

    auto self_tensor_desc = TensorDesc({5, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceUniform, INPUT(self_tensor_desc, from, to, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(acl_ret, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_uniform_test, aclnnUniform_5_4_6_invalid_shape_9_dims)
{
    double from = 2;
    double to = 3;
    uint64_t seed = 0;
    uint64_t offset = 0;

    auto self_tensor_desc = TensorDesc({5, 4, 6, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceUniform, INPUT(self_tensor_desc, from, to, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_uniform_test, ascend910B2_aclnnUniform_3_4_float_nd)
{
    const vector<int64_t>& self_shape = {3, 4};
    aclDataType self_dtype = ACL_FLOAT;
    aclFormat self_format = ACL_FORMAT_ND;

    double from = 2;
    double to = 4;
    uint64_t seed = 0;
    uint64_t offset = 0;

    const vector<int64_t>& out_shape = {3, 4};
    aclDataType out_dtype = ACL_FLOAT;
    aclFormat out_format = ACL_FORMAT_ND;

    auto self_tensor_desc = TensorDesc(self_shape, self_dtype, self_format);

    auto ut = OP_API_UT(aclnnInplaceUniform, INPUT(self_tensor_desc, from, to, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACL_SUCCESS);
}

// 正常场景910B SocVersion bf16
TEST_F(l2_uniform_test, ascend910B2_aclnnUniform_0to1_3_4_bf16_nd)
{
    const vector<int64_t>& self_shape = {3, 4};
    aclDataType self_dtype = ACL_BF16;
    aclFormat self_format = ACL_FORMAT_ND;

    double from = 0;
    double to = 1;
    uint64_t seed = 0;
    uint64_t offset = 0;

    const vector<int64_t>& out_shape = {3, 4};
    aclDataType out_dtype = ACL_BF16;
    aclFormat out_format = ACL_FORMAT_ND;

    auto self_tensor_desc = TensorDesc(self_shape, self_dtype, self_format);

    auto ut = OP_API_UT(aclnnInplaceUniform, INPUT(self_tensor_desc, from, to, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACL_SUCCESS);
}

// 正常场景: int32 负数区间, 端点在 dtype 表示范围内不被校验误拦, 先 Floor 再 Cast 构建成功
TEST_F(l2_uniform_test, ascend910B2_aclnnUniform_neg2to0_int32_nd)
{
    double from = -2;
    double to = 0;
    uint64_t seed = 0;
    uint64_t offset = 0;

    auto self_tensor_desc = TensorDesc({3, 4}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceUniform, INPUT(self_tensor_desc, from, to, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACL_SUCCESS);
}

TEST_F(l2_uniform_test, ascend910B2_aclnnUniform_0to1_3_4_fp16_nd)
{
    const vector<int64_t>& self_shape = {3, 4};
    aclDataType self_dtype = ACL_FLOAT16;
    aclFormat self_format = ACL_FORMAT_ND;

    double from = 0;
    double to = 1;
    uint64_t seed = 0;
    uint64_t offset = 0;

    const vector<int64_t>& out_shape = {3, 4};
    aclDataType out_dtype = ACL_FLOAT16;
    aclFormat out_format = ACL_FORMAT_ND;

    auto self_tensor_desc = TensorDesc(self_shape, self_dtype, self_format);

    auto ut = OP_API_UT(aclnnInplaceUniform, INPUT(self_tensor_desc, from, to, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACL_SUCCESS);
}

// ======================== aclnnInplaceUniform 新增用例 ========================

// double类型走StatelessRandomUniformV2 fallback路径
TEST_F(l2_uniform_test, aclnnUniform_3_4_double_nd)
{
    double from = 2;
    double to = 4;
    uint64_t seed = 0;
    uint64_t offset = 0;

    auto self_tensor_desc = TensorDesc({3, 4}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceUniform, INPUT(self_tensor_desc, from, to, seed, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACL_SUCCESS);
}

// ======================== aclnnInplaceUniformTensor 新增用例 ========================

// 正常场景：float类型
TEST_F(l2_uniform_test, aclnnUniformTensor_3_4_float_nd)
{
    double from = 2;
    double to = 4;
    uint64_t offset = 0;

    auto self_tensor_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto seed_tensor_desc = TensorDesc({1}, ACL_UINT64, ACL_FORMAT_ND);
    auto offset_tensor_desc = TensorDesc({1}, ACL_UINT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceUniformTensor,
                        INPUT(self_tensor_desc, from, to, seed_tensor_desc, offset_tensor_desc, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACL_SUCCESS);
}

// 正常场景：float16类型
TEST_F(l2_uniform_test, aclnnUniformTensor_5_4_6_float16_nd)
{
    double from = 0;
    double to = 1;
    uint64_t offset = 0;

    auto self_tensor_desc = TensorDesc({5, 4, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto seed_tensor_desc = TensorDesc({1}, ACL_UINT64, ACL_FORMAT_ND);
    auto offset_tensor_desc = TensorDesc({1}, ACL_UINT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceUniformTensor,
                        INPUT(self_tensor_desc, from, to, seed_tensor_desc, offset_tensor_desc, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACL_SUCCESS);
}

// 正常场景：bf16类型
TEST_F(l2_uniform_test, aclnnUniformTensor_3_4_bf16_nd)
{
    double from = 0;
    double to = 1;
    uint64_t offset = 0;

    auto self_tensor_desc = TensorDesc({3, 4}, ACL_BF16, ACL_FORMAT_ND);
    auto seed_tensor_desc = TensorDesc({1}, ACL_UINT64, ACL_FORMAT_ND);
    auto offset_tensor_desc = TensorDesc({1}, ACL_UINT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceUniformTensor,
                        INPUT(self_tensor_desc, from, to, seed_tensor_desc, offset_tensor_desc, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACL_SUCCESS);
}

// 正常场景：double类型，走StatelessRandomUniformV2 fallback路径
TEST_F(l2_uniform_test, aclnnUniformTensor_3_4_double_nd)
{
    double from = 2;
    double to = 4;
    uint64_t offset = 0;

    auto self_tensor_desc = TensorDesc({3, 4}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto seed_tensor_desc = TensorDesc({1}, ACL_UINT64, ACL_FORMAT_ND);
    auto offset_tensor_desc = TensorDesc({1}, ACL_UINT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceUniformTensor,
                        INPUT(self_tensor_desc, from, to, seed_tensor_desc, offset_tensor_desc, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACL_SUCCESS);
}

// 异常场景：from > to
TEST_F(l2_uniform_test, aclnnUniformTensor_5_4_6_invalid_from_to)
{
    double from = 3;
    double to = 2;
    uint64_t offset = 0;

    auto self_tensor_desc = TensorDesc({5, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto seed_tensor_desc = TensorDesc({1}, ACL_UINT64, ACL_FORMAT_ND);
    auto offset_tensor_desc = TensorDesc({1}, ACL_UINT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceUniformTensor,
                        INPUT(self_tensor_desc, from, to, seed_tensor_desc, offset_tensor_desc, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：self为nullptr
TEST_F(l2_uniform_test, aclnnUniformTensor_5_4_6_input_nullptr)
{
    double from = 2;
    double to = 3;
    uint64_t offset = 0;

    auto seed_tensor_desc = TensorDesc({1}, ACL_UINT64, ACL_FORMAT_ND);
    auto offset_tensor_desc = TensorDesc({1}, ACL_UINT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceUniformTensor,
                        INPUT((aclTensor*)nullptr, from, to, seed_tensor_desc, offset_tensor_desc, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常场景：不支持的数据类型UINT32
TEST_F(l2_uniform_test, aclnnUniformTensor_5_4_6_invalid_dtype)
{
    double from = 2;
    double to = 3;
    uint64_t offset = 0;

    auto self_tensor_desc = TensorDesc({5, 4, 6}, ACL_UINT32, ACL_FORMAT_ND);
    auto seed_tensor_desc = TensorDesc({1}, ACL_UINT64, ACL_FORMAT_ND);
    auto offset_tensor_desc = TensorDesc({1}, ACL_UINT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceUniformTensor,
                        INPUT(self_tensor_desc, from, to, seed_tensor_desc, offset_tensor_desc, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：超过8维
TEST_F(l2_uniform_test, aclnnUniformTensor_5_4_6_invalid_shape_9_dims)
{
    double from = 2;
    double to = 3;
    uint64_t offset = 0;

    auto self_tensor_desc = TensorDesc({5, 4, 6, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto seed_tensor_desc = TensorDesc({1}, ACL_UINT64, ACL_FORMAT_ND);
    auto offset_tensor_desc = TensorDesc({1}, ACL_UINT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceUniformTensor,
                        INPUT(self_tensor_desc, from, to, seed_tensor_desc, offset_tensor_desc, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACLNN_ERR_PARAM_INVALID);
}

// 正常场景：带非零offset
TEST_F(l2_uniform_test, aclnnUniformTensor_3_4_float_with_offset)
{
    double from = 0;
    double to = 1;
    uint64_t offset = 100;

    auto self_tensor_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto seed_tensor_desc = TensorDesc({1}, ACL_UINT64, ACL_FORMAT_ND);
    auto offset_tensor_desc = TensorDesc({1}, ACL_UINT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceUniformTensor,
                        INPUT(self_tensor_desc, from, to, seed_tensor_desc, offset_tensor_desc, offset), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus acl_ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(acl_ret, ACL_SUCCESS);
}
