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

#include "random/drop_out_v3/op_api/aclnn_dropout_v3.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
#include "op_common/op_host/util/op_const_def.h"

using namespace op;
using namespace std;

/*!
 * \file test_aclnn_drop_out_v3.cpp
 * \brief aclnnDropoutV3 / aclnnDropoutV3Tensor L2 UT
 *        覆盖 op_api/aclnn_dropout_v3.cpp 的入参校验与构图：
 *        标量接口(p/seed/offset)与 Tensor 接口(seedTensor/offsetTensor)各自的
 *        空指针 / dtype / 维度 / shape / p 取值校验，及 arch 相关 dtype 支持列表分支。
 */

class l2_dropout_v3_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "dropout_v3_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "dropout_v3_test TearDown" << endl; }
};

static inline int64_t ComputeByteSizeAlign(const vector<int64_t>& shape)
{
    int64_t num = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        num *= shape[i];
    }
    return (num + 128 - 1) / 128 * 128 / 8;
}

TEST_F(l2_dropout_v3_test, DAV_3510_89_case_001)
{
    const vector<int64_t> shape = {10, 16, 9, 8, 5, 5};
    auto input_desc = TensorDesc(shape, ACL_FLOAT16, ACL_FORMAT_ND);
    double p = 0;
    int64_t seed = 1234;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3, INPUT(input_desc, (aclTensor*)nullptr, p, seed, offset),
                        OUTPUT(out_desc, mask_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_dropout_v3_test, DAV_3510_89_case_002)
{
    const vector<int64_t> shape = {2, 0, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_NCL);
    double p = 1;
    int64_t seed = 213;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3, INPUT(input_desc, (aclTensor*)nullptr, p, seed, offset),
                        OUTPUT(out_desc, mask_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// input nullptr
TEST_F(l2_dropout_v3_test, DAV_3510_89_case_003)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    double p = 1;
    int64_t seed = 1234;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3, INPUT((aclTensor*)nullptr, (aclTensor*)nullptr, p, seed, offset),
                        OUTPUT(out_desc, mask_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// out nullptr
TEST_F(l2_dropout_v3_test, DAV_3510_89_case_004)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    double p = 1;
    int64_t seed = 1234;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3, INPUT(input_desc, (aclTensor*)nullptr, p, seed, offset),
                        OUTPUT((aclTensor*)nullptr, mask_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// mask nullptr
TEST_F(l2_dropout_v3_test, DAV_3510_89_case_005)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    double p = 1;
    int64_t seed = 1234;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3, INPUT(input_desc, (aclTensor*)nullptr, p, seed, offset),
                        OUTPUT(out_desc, (aclTensor*)nullptr));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// check dtype can cast
TEST_F(l2_dropout_v3_test, DAV_3510_89_case_006)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT16, ACL_FORMAT_ND);
    double p = 1;
    int64_t seed = 1234;
    int64_t offset = 0;
    auto out_desc = TensorDesc(shape, ACL_INT32, ACL_FORMAT_ND);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3, INPUT(input_desc, (aclTensor*)nullptr, p, seed, offset),
                        OUTPUT(out_desc, mask_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// check dtype valid
TEST_F(l2_dropout_v3_test, DAV_3510_89_case_007)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_INT32, ACL_FORMAT_ND);
    double p = 1;
    int64_t seed = 1234;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3, INPUT(input_desc, (aclTensor*)nullptr, p, seed, offset),
                        OUTPUT(out_desc, mask_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// check dtype valid
TEST_F(l2_dropout_v3_test, DAV_3510_89_case_008)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    double p = 1;
    int64_t seed = 1234;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3, INPUT(input_desc, (aclTensor*)nullptr, p, seed, offset),
                        OUTPUT(out_desc, mask_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// check p
TEST_F(l2_dropout_v3_test, DAV_3510_89_case_009)
{
    const vector<int64_t> shape = {2, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_NCL);
    double p = 1.5;
    int64_t seed = 213;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3, INPUT(input_desc, (aclTensor*)nullptr, p, seed, offset),
                        OUTPUT(out_desc, mask_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// check shape
TEST_F(l2_dropout_v3_test, DAV_3510_89_case_010)
{
    const vector<int64_t> shape = {2, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_NCL);
    double p = 0.5;
    int64_t seed = 213;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({32}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3, INPUT(input_desc, (aclTensor*)nullptr, p, seed, offset),
                        OUTPUT(out_desc, mask_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// check shape
TEST_F(l2_dropout_v3_test, DAV_3510_89_case_011)
{
    const vector<int64_t> shape = {2, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_NCL);
    double p = 0.5;
    int64_t seed = 213;
    int64_t offset = 0;
    auto out_desc = TensorDesc({2, 3, 1}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3, INPUT(input_desc, (aclTensor*)nullptr, p, seed, offset),
                        OUTPUT(out_desc, mask_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// check shape
TEST_F(l2_dropout_v3_test, DAV_3510_89_case_012)
{
    const vector<int64_t> shape = {2, 9, 3, 4, 5, 6, 7, 8};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_NCL);
    double p = 0.3;
    int64_t seed = 213;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3, INPUT(input_desc, (aclTensor*)nullptr, p, seed, offset),
                        OUTPUT(out_desc, mask_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// check shape
TEST_F(l2_dropout_v3_test, DAV_3510_89_case_013)
{
    const vector<int64_t> shape = {2, 9, 3, 4, 5, 6, 7, 8, 10};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_NCL);
    double p = 0.3;
    int64_t seed = 213;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3, INPUT(input_desc, (aclTensor*)nullptr, p, seed, offset),
                        OUTPUT(out_desc, mask_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_dropout_v3_test, DAV_3510_89_case_014)
{
    const vector<int64_t> shape = {10, 16, 9, 8, 5, 5};
    auto input_desc = TensorDesc(shape, ACL_FLOAT16, ACL_FORMAT_ND);
    double p = 1;
    int64_t seed = 1234;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3, INPUT(input_desc, (aclTensor*)nullptr, p, seed, offset),
                        OUTPUT(out_desc, mask_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_dropout_v3_test, DAV_3510_89_case_015)
{
    const vector<int64_t> shape = {10, 16, 9, 8, 5, 5};
    auto input_desc = TensorDesc(shape, ACL_BF16, ACL_FORMAT_ND);
    double p = 1;
    int64_t seed = 1234;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3, INPUT(input_desc, (aclTensor*)nullptr, p, seed, offset),
                        OUTPUT(out_desc, mask_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// ===================== 标量接口补充覆盖 =====================

// optionalNoiseShape 非空（预留参数要求传空指针）→ ACLNN_ERR_PARAM_INVALID
TEST_F(l2_dropout_v3_test, case_fail_noise_shape_not_null)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto noise_shape_desc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
    double p = 0.5;
    int64_t seed = 1234;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3, INPUT(input_desc, noise_shape_desc, p, seed, offset),
                        OUTPUT(out_desc, mask_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// p < 0 → ACLNN_ERR_PARAM_INVALID
TEST_F(l2_dropout_v3_test, case_fail_p_negative)
{
    const vector<int64_t> shape = {2, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    double p = -0.1;
    int64_t seed = 213;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3, INPUT(input_desc, (aclTensor*)nullptr, p, seed, offset),
                        OUTPUT(out_desc, mask_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 二段式接口直接调用：executor 为空指针时返回 ACLNN_ERR_PARAM_NULLPTR
TEST_F(l2_dropout_v3_test, case_phase2_drop_out_v3)
{
    aclnnStatus aclRet = aclnnDropoutV3(nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_dropout_v3_test, case_phase2_drop_out_v3_tensor)
{
    aclnnStatus aclRet = aclnnDropoutV3Tensor(nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ===================== arch 相关 dtype 支持列表分支 =====================

class l2_dropout_v3_arch_test : public testing::Test {
protected:
    NpuArch savedArch_ = Ops::Base::DAV_2201;

    void SetUp() override { savedArch_ = GetCurrentPlatformInfo().GetCurNpuArch(); }

    void TearDown() override { SetPlatformNpuArch(savedArch_); }
};

// DAV_1001：dtype 支持列表为 {FLOAT, FLOAT16}，FP16 构图成功
TEST_F(l2_dropout_v3_arch_test, case_dav_1001_float16_success)
{
    SetPlatformNpuArch(Ops::Base::DAV_1001);
    const vector<int64_t> shape = {10, 16, 9, 8, 5, 5};
    auto input_desc = TensorDesc(shape, ACL_FLOAT16, ACL_FORMAT_ND);
    double p = 0;
    int64_t seed = 1234;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3, INPUT(input_desc, (aclTensor*)nullptr, p, seed, offset),
                        OUTPUT(out_desc, mask_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// DAV_1001：BF16 不在支持列表 → ACLNN_ERR_PARAM_INVALID
TEST_F(l2_dropout_v3_arch_test, case_dav_1001_bfloat16_fail)
{
    SetPlatformNpuArch(Ops::Base::DAV_1001);
    const vector<int64_t> shape = {10, 16, 9, 8, 5, 5};
    auto input_desc = TensorDesc(shape, ACL_BF16, ACL_FORMAT_ND);
    double p = 1;
    int64_t seed = 1234;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3, INPUT(input_desc, (aclTensor*)nullptr, p, seed, offset),
                        OUTPUT(out_desc, mask_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 未列举 arch（DAV_2002）走 default 分支：FLOAT 构图成功
TEST_F(l2_dropout_v3_arch_test, case_default_arch_float_success)
{
    SetPlatformNpuArch(Ops::Base::DAV_2002);
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    double p = 0.5;
    int64_t seed = 1234;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3, INPUT(input_desc, (aclTensor*)nullptr, p, seed, offset),
                        OUTPUT(out_desc, mask_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// ===================== aclnnDropoutV3Tensor 接口 =====================

class l2_dropout_v3_tensor_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "dropout_v3_tensor_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "dropout_v3_tensor_test TearDown" << endl; }
};

// case 1: float32 / p=0.5，seed/offset 以 Tensor 形式传入
TEST_F(l2_dropout_v3_tensor_test, case_tensor_success_float32)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto seed_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offset_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    double p = 0.5;
    int64_t offset = 4;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3Tensor, INPUT(input_desc, (aclTensor*)nullptr, p, seed_desc, offset_desc, offset),
                        OUTPUT(out_desc, mask_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// case 2: float16 / p=0.5
TEST_F(l2_dropout_v3_tensor_test, case_tensor_success_float16)
{
    const vector<int64_t> shape = {10, 16, 9, 8, 5, 5};
    auto input_desc = TensorDesc(shape, ACL_FLOAT16, ACL_FORMAT_ND);
    auto seed_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offset_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    double p = 0.5;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3Tensor, INPUT(input_desc, (aclTensor*)nullptr, p, seed_desc, offset_desc, offset),
                        OUTPUT(out_desc, mask_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// case 3: bfloat16 / p=0.5
TEST_F(l2_dropout_v3_tensor_test, case_tensor_success_bfloat16)
{
    const vector<int64_t> shape = {10, 16, 9, 8, 5, 5};
    auto input_desc = TensorDesc(shape, ACL_BF16, ACL_FORMAT_ND);
    auto seed_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offset_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    double p = 0.5;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3Tensor, INPUT(input_desc, (aclTensor*)nullptr, p, seed_desc, offset_desc, offset),
                        OUTPUT(out_desc, mask_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// case 4: p=0，mask 全 1，out 直通 input
TEST_F(l2_dropout_v3_tensor_test, case_tensor_success_p_zero)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto seed_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offset_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    double p = 0;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3Tensor, INPUT(input_desc, (aclTensor*)nullptr, p, seed_desc, offset_desc, offset),
                        OUTPUT(out_desc, mask_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// case 5: p=1，out 全 0，mask 全 0
TEST_F(l2_dropout_v3_tensor_test, case_tensor_success_p_one)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT16, ACL_FORMAT_ND);
    auto seed_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offset_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    double p = 1;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3Tensor, INPUT(input_desc, (aclTensor*)nullptr, p, seed_desc, offset_desc, offset),
                        OUTPUT(out_desc, mask_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// case 6: 空 tensor，workspace=0，返回成功
TEST_F(l2_dropout_v3_tensor_test, case_tensor_success_empty)
{
    const vector<int64_t> shape = {2, 0, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto seed_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offset_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    double p = 0.5;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3Tensor, INPUT(input_desc, (aclTensor*)nullptr, p, seed_desc, offset_desc, offset),
                        OUTPUT(out_desc, mask_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// case 7: input 空指针 → ACLNN_ERR_PARAM_NULLPTR
TEST_F(l2_dropout_v3_tensor_test, case_tensor_fail_null_input)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto seed_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offset_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    double p = 0.5;
    int64_t offset = 0;
    auto out_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3Tensor,
                        INPUT((aclTensor*)nullptr, (aclTensor*)nullptr, p, seed_desc, offset_desc, offset),
                        OUTPUT(out_desc, mask_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// case 8: seedTensor 空指针 → ACLNN_ERR_PARAM_NULLPTR
TEST_F(l2_dropout_v3_tensor_test, case_tensor_fail_null_seed)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto offset_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    double p = 0.5;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3Tensor,
                        INPUT(input_desc, (aclTensor*)nullptr, p, (aclTensor*)nullptr, offset_desc, offset),
                        OUTPUT(out_desc, mask_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// case 9: offsetTensor 空指针 → ACLNN_ERR_PARAM_NULLPTR
TEST_F(l2_dropout_v3_tensor_test, case_tensor_fail_null_offset)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto seed_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    double p = 0.5;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3Tensor,
                        INPUT(input_desc, (aclTensor*)nullptr, p, seed_desc, (aclTensor*)nullptr, offset),
                        OUTPUT(out_desc, mask_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// case 10: out 空指针 → ACLNN_ERR_PARAM_NULLPTR
TEST_F(l2_dropout_v3_tensor_test, case_tensor_fail_null_out)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto seed_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offset_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    double p = 0.5;
    int64_t offset = 0;
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3Tensor, INPUT(input_desc, (aclTensor*)nullptr, p, seed_desc, offset_desc, offset),
                        OUTPUT((aclTensor*)nullptr, mask_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// case 11: maskOut 空指针 → ACLNN_ERR_PARAM_NULLPTR
TEST_F(l2_dropout_v3_tensor_test, case_tensor_fail_null_mask)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto seed_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offset_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    double p = 0.5;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);

    auto ut = OP_API_UT(aclnnDropoutV3Tensor, INPUT(input_desc, (aclTensor*)nullptr, p, seed_desc, offset_desc, offset),
                        OUTPUT(out_desc, (aclTensor*)nullptr));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// case 12: optionalNoiseShape 非空 → ACLNN_ERR_PARAM_INVALID
TEST_F(l2_dropout_v3_tensor_test, case_tensor_fail_noise_shape_not_null)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto noise_shape_desc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
    auto seed_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offset_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    double p = 0.5;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3Tensor, INPUT(input_desc, noise_shape_desc, p, seed_desc, offset_desc, offset),
                        OUTPUT(out_desc, mask_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// case 13: p 超出 [0, 1] → ACLNN_ERR_PARAM_INVALID
TEST_F(l2_dropout_v3_tensor_test, case_tensor_fail_p_invalid)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto seed_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offset_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    double p = 1.5;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3Tensor, INPUT(input_desc, (aclTensor*)nullptr, p, seed_desc, offset_desc, offset),
                        OUTPUT(out_desc, mask_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// case 14: seedTensor dtype 非 INT64 → ACLNN_ERR_PARAM_INVALID
TEST_F(l2_dropout_v3_tensor_test, case_tensor_fail_seed_dtype)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto seed_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    auto offset_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    double p = 0.5;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3Tensor, INPUT(input_desc, (aclTensor*)nullptr, p, seed_desc, offset_desc, offset),
                        OUTPUT(out_desc, mask_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// case 15: offsetTensor dtype 非 INT64 → ACLNN_ERR_PARAM_INVALID
TEST_F(l2_dropout_v3_tensor_test, case_tensor_fail_offset_dtype)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto seed_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offset_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    double p = 0.5;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3Tensor, INPUT(input_desc, (aclTensor*)nullptr, p, seed_desc, offset_desc, offset),
                        OUTPUT(out_desc, mask_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// case 16: seedTensor 维度非 1 → ACLNN_ERR_PARAM_INVALID
TEST_F(l2_dropout_v3_tensor_test, case_tensor_fail_seed_dim)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto seed_desc = TensorDesc({1, 1}, ACL_INT64, ACL_FORMAT_ND);
    auto offset_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    double p = 0.5;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3Tensor, INPUT(input_desc, (aclTensor*)nullptr, p, seed_desc, offset_desc, offset),
                        OUTPUT(out_desc, mask_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// case 17: offsetTensor 维度非 1 → ACLNN_ERR_PARAM_INVALID
TEST_F(l2_dropout_v3_tensor_test, case_tensor_fail_offset_dim)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto seed_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offset_desc = TensorDesc({1, 1}, ACL_INT64, ACL_FORMAT_ND);
    double p = 0.5;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3Tensor, INPUT(input_desc, (aclTensor*)nullptr, p, seed_desc, offset_desc, offset),
                        OUTPUT(out_desc, mask_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// case 18: maskOut 维度非 1 → ACLNN_ERR_PARAM_INVALID
TEST_F(l2_dropout_v3_tensor_test, case_tensor_fail_mask_dim)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto seed_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offset_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    double p = 0.5;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape), 1}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3Tensor, INPUT(input_desc, (aclTensor*)nullptr, p, seed_desc, offset_desc, offset),
                        OUTPUT(out_desc, mask_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// case 19: input dtype 非法（INT32）→ ACLNN_ERR_PARAM_INVALID
TEST_F(l2_dropout_v3_tensor_test, case_tensor_fail_input_dtype)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_INT32, ACL_FORMAT_ND);
    auto seed_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offset_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    double p = 0.5;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3Tensor, INPUT(input_desc, (aclTensor*)nullptr, p, seed_desc, offset_desc, offset),
                        OUTPUT(out_desc, mask_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// case 20: maskOut dtype 非法（FLOAT）→ ACLNN_ERR_PARAM_INVALID
TEST_F(l2_dropout_v3_tensor_test, case_tensor_fail_mask_dtype)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto seed_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offset_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    double p = 0.5;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3Tensor, INPUT(input_desc, (aclTensor*)nullptr, p, seed_desc, offset_desc, offset),
                        OUTPUT(out_desc, mask_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// case 21: out dtype 不可从 input cast（FP16 → INT32）→ ACLNN_ERR_PARAM_INVALID
TEST_F(l2_dropout_v3_tensor_test, case_tensor_fail_out_dtype_not_castable)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT16, ACL_FORMAT_ND);
    auto seed_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offset_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    double p = 0.5;
    int64_t offset = 0;
    auto out_desc = TensorDesc(shape, ACL_INT32, ACL_FORMAT_ND);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3Tensor, INPUT(input_desc, (aclTensor*)nullptr, p, seed_desc, offset_desc, offset),
                        OUTPUT(out_desc, mask_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// case 22: out 与 input shape 不一致 → ACLNN_ERR_PARAM_INVALID
TEST_F(l2_dropout_v3_tensor_test, case_tensor_fail_out_shape)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto seed_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offset_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    double p = 0.5;
    int64_t offset = 0;
    auto out_desc = TensorDesc({2, 3, 1, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3Tensor, INPUT(input_desc, (aclTensor*)nullptr, p, seed_desc, offset_desc, offset),
                        OUTPUT(out_desc, mask_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// case 23: maskOut 元素个数不满足 align(inputSize, 128) / 8 → ACLNN_ERR_PARAM_INVALID
TEST_F(l2_dropout_v3_tensor_test, case_tensor_fail_mask_size)
{
    const vector<int64_t> shape = {2, 16, 1, 3};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto seed_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offset_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    double p = 0.5;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({32}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3Tensor, INPUT(input_desc, (aclTensor*)nullptr, p, seed_desc, offset_desc, offset),
                        OUTPUT(out_desc, mask_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// case 24: input 超 8 维 → ACLNN_ERR_PARAM_INVALID
TEST_F(l2_dropout_v3_tensor_test, case_tensor_fail_max_dim)
{
    const vector<int64_t> shape = {2, 9, 3, 4, 5, 6, 7, 8, 10};
    auto input_desc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto seed_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    auto offset_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    double p = 0.3;
    int64_t offset = 0;
    auto out_desc = TensorDesc(input_desc);
    auto mask_desc = TensorDesc({ComputeByteSizeAlign(shape)}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDropoutV3Tensor, INPUT(input_desc, (aclTensor*)nullptr, p, seed_desc, offset_desc, offset),
                        OUTPUT(out_desc, mask_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
