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
#include "opdev/platform.h"

#include "math/div/op_api/aclnn_div.h"

#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_div_mod_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "div_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "div_test TearDown" << endl; }
};

// 测试所有数据格式支持
TEST_F(l2_div_mod_test, case_dtype_all_format)
{
    vector<aclFormat> format_list{ACL_FORMAT_ND,   ACL_FORMAT_NCHW,  ACL_FORMAT_NHWC,
                                  ACL_FORMAT_HWCN, ACL_FORMAT_NDHWC, ACL_FORMAT_NCDHW};
    for (auto format : format_list) {
        auto self_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, format).ValueRange(10, 100);
        auto other_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, format).ValueRange(10, 100);
        auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, format);

        int mode = 2;
        auto ut = OP_API_UT(aclnnDivMod, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);

        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

// 测试类型转换支持
TEST_F(l2_div_mod_test, case_dtype_cast_support)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    int mode = 0;

    auto ut = OP_API_UT(aclnnDivMod, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试非连续支持
TEST_F(l2_div_mod_test, case_NonContiguous)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND, {1, 2}, 0, {3, 2})
                                .Value(vector<float>{-10.5, -20, -30, -40, 50, 60});
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND, {1, 2}, 0, {3, 2})
                                 .Value(vector<float>{-2, -3, -2, -4, 2, 4});
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    int mode = 0;

    auto ut = OP_API_UT(aclnnDivMod, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试broadcast
TEST_F(l2_div_mod_test, case_broadcast)
{
    auto self_tensor_desc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 100);
    auto other_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 100);
    auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);
    int mode = 1;

    auto ut = OP_API_UT(aclnnDivMod, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试空tensor
TEST_F(l2_div_mod_test, case_empty_tensors)
{
    auto self_tensor_desc = TensorDesc({3, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto other_tensor_desc = TensorDesc({3, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);
    int mode = 2;
    auto ut = OP_API_UT(aclnnDivMod, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试不支持类型CheckDtypeValid
TEST_F(l2_div_mod_test, case_CheckDtypeValid)
{
    auto tensor_desc = TensorDesc({10, 5}, ACL_UINT32, ACL_FORMAT_ND);
    int mode = 1;
    auto ut = OP_API_UT(aclnnDivMod, INPUT(tensor_desc, tensor_desc, mode), OUTPUT(tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试入参空指针
TEST_F(l2_div_mod_test, case_nullptr)
{
    auto tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int mode = 2;
    auto ut = OP_API_UT(aclnnDivMod, INPUT((aclTensor*)nullptr, (aclTensor*)nullptr, mode), OUTPUT(tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 测试other为scalar,mode为2立即数输入
TEST_F(l2_div_mod_test, case_other_scalar_floor_support)
{
    auto self_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(10, 100);
    auto other_tensor_desc = ScalarDesc(2.0f);
    auto out_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    int mode = 2;

    auto ut = OP_API_UT(aclnnDivMods, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试other为scalar,mode为1立即数输入
TEST_F(l2_div_mod_test, case_other_scalar_trunc_support)
{
    auto self_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(10, 100);
    auto other_tensor_desc = ScalarDesc(2.0f);
    auto out_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    int mode = 1;

    auto ut = OP_API_UT(aclnnDivMods, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试other为scalar超过8维的tensor
TEST_F(l2_div_mod_test, case_scalar_shape_dim_9)
{
    auto self_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto other_tensor_desc = ScalarDesc(2.0f);
    auto out_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCHW);
    int mode = 2;

    auto ut = OP_API_UT(aclnnDivMods, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试超过8维的tensor
TEST_F(l2_div_mod_test, case_shape_dim_9)
{
    auto self_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCHW);
    int mode = 2;

    auto ut = OP_API_UT(aclnnDivMod, INPUT(self_tensor_desc, self_tensor_desc, mode), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试div_:other为scalar立即数输入
TEST_F(l2_div_mod_test, case_inplace_other_scalar_support)
{
    auto self_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(10, 100);
    auto other_tensor_desc = ScalarDesc(2.0f);
    int mode = 1;

    auto ut = OP_API_UT(aclnnInplaceDivMods, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试div_:other为tensor输入
TEST_F(l2_div_mod_test, case_inplace_other_support)
{
    auto self_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(10, 100);
    auto other_tensor_desc = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(10, 100);
    int mode = 2;

    auto ut = OP_API_UT(aclnnInplaceDivMod, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_div_mod_test, case_complex_mods_1_invalid)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(1, 2);
    auto other_tensor_desc = ScalarDesc(2.0f);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    int mode = 1;

    auto ut = OP_API_UT(aclnnDivMods, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_div_mod_test, case_complex_mods_2_invalid)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(1, 2);
    auto other_tensor_desc = ScalarDesc(2.0f);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).Precision(0.001, 0.001);
    int mode = 2;

    auto ut = OP_API_UT(aclnnDivMods, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_div_mod_test, case_complex_mod_1_invalid)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(1, 2);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(1, 2);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    int mode = 2;

    auto ut = OP_API_UT(aclnnDivMod, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_div_mod_test, case_complex_mod_2_invalid)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(1, 2);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(1, 2);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).Precision(0.001, 0.001);
    int mode = 2;

    auto ut = OP_API_UT(aclnnDivMod, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试mode异常值拦截
TEST_F(l2_div_mod_test, case_mode_invalid)
{
    auto self_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto other_tensor_desc = ScalarDesc(2.0f);
    auto out_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCHW);
    int mode = 3;

    auto ut = OP_API_UT(aclnnDivMods, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Ascend950PR_9589
// 测试场景mode为0时数据类型支持
TEST_F(l2_div_mod_test, Ascend950PR_89_case_real_div_dtype_modeNone)
{
    vector<aclDataType> dtype_list{ACL_FLOAT16};
    for (auto dtype : dtype_list) {
        auto self_tensor_desc = TensorDesc({4, 5}, dtype, ACL_FORMAT_ND).ValueRange(10, 100);
        auto other_tensor_desc = ScalarDesc(2);
        auto out_tensor_desc = TensorDesc({4, 5}, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
        int mode = 0;

        auto ut = OP_API_UT(aclnnDivMods, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_div_mod_test, Ascend950PR_89_case_divmods_trunc_div)
{
    vector<aclDataType> dtype_list{ACL_INT8};
    for (auto dtype : dtype_list) {
        auto self_tensor_desc = TensorDesc({4, 5}, dtype, ACL_FORMAT_ND).ValueRange(10, 100);
        auto other_tensor_desc = ScalarDesc(2);
        auto out_tensor_desc = TensorDesc({4, 5}, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
        int mode = 1;

        auto ut = OP_API_UT(aclnnDivMods, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_div_mod_test, Ascend950PR_89_case_divmods_floor_div)
{
    vector<aclDataType> dtype_list{ACL_INT16};
    for (auto dtype : dtype_list) {
        auto self_tensor_desc = TensorDesc({4, 5}, dtype, ACL_FORMAT_ND).ValueRange(10, 100);
        auto other_tensor_desc = ScalarDesc(2);
        auto out_tensor_desc = TensorDesc({4, 5}, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
        int mode = 2;

        auto ut = OP_API_UT(aclnnDivMods, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_div_mod_test, Ascend950PR_89_case_divmod_trunc_div)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(1, 2);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(1, 2);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    int mode = 1;

    auto ut = OP_API_UT(aclnnDivMod, INPUT(self_tensor_desc, other_tensor_desc, mode), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

namespace op {
extern SocVersion g_socVersion;
extern NpuArch g_npuArchOverride;
} // namespace op

class l2_div_mod_regbase_out_dtype_test : public testing::Test {
protected:
    void SetUp() override
    {
        savedSoc_ = op::g_socVersion;
        savedArch_ = op::g_npuArchOverride;
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    }
    void TearDown() override
    {
        // Restore the raw stub state to avoid the setters changing the paired field.
        op::g_socVersion = savedSoc_;
        op::g_npuArchOverride = savedArch_;
    }

private:
    op::SocVersion savedSoc_;
    NpuArch savedArch_;
};

TEST_F(l2_div_mod_regbase_out_dtype_test, tensor_supports_unsigned_out_for_integer_computation)
{
    for (auto arch : {NpuArch::DAV_3510}) {
        op::SetPlatformNpuArch(arch);
        for (int mode : {1, 2}) {
            for (auto inDtype : {ACL_INT8, ACL_INT16, ACL_INT32, ACL_INT64, ACL_UINT8}) {
                for (auto outDtype : {ACL_UINT16, ACL_UINT32, ACL_UINT64}) {
                    for (auto dims : {vector<int64_t>{2, 3}, vector<int64_t>{0}}) {
                        SCOPED_TRACE(testing::Message() << "arch=" << static_cast<int>(arch) << " mode=" << mode
                                                        << " outDtype=" << outDtype << " dim0=" << dims[0]);
                        auto self = TensorDesc(dims, inDtype, ACL_FORMAT_ND).ValueRange(6, 12);
                        auto other = TensorDesc(dims, inDtype, ACL_FORMAT_ND).ValueRange(2, 3);
                        auto out = TensorDesc(dims, outDtype, ACL_FORMAT_ND);
                        auto ut = OP_API_UT(aclnnDivMod, INPUT(self, other, mode), OUTPUT(out));
                        uint64_t workspaceSize = 0;
                        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
                    }
                }
            }
        }
    }
}

TEST_F(l2_div_mod_regbase_out_dtype_test, scalar_supports_unsigned_out_for_integer_computation)
{
    for (auto arch : {NpuArch::DAV_3510}) {
        op::SetPlatformNpuArch(arch);
        for (int mode : {1, 2}) {
            for (auto inDtype : {ACL_INT8, ACL_INT16, ACL_INT32, ACL_INT64, ACL_UINT8}) {
                for (auto outDtype : {ACL_UINT16, ACL_UINT32, ACL_UINT64}) {
                    for (auto dims : {vector<int64_t>{2, 3}, vector<int64_t>{0}}) {
                        SCOPED_TRACE(testing::Message() << "arch=" << static_cast<int>(arch) << " mode=" << mode
                                                        << " outDtype=" << outDtype << " dim0=" << dims[0]);
                        auto self = TensorDesc(dims, inDtype, ACL_FORMAT_ND).ValueRange(6, 12);
                        auto other = (inDtype == ACL_INT8)  ? ScalarDesc(int8_t(2)) :
                                     (inDtype == ACL_INT16) ? ScalarDesc(int16_t(2)) :
                                     (inDtype == ACL_INT32) ? ScalarDesc(int32_t(2)) :
                                     (inDtype == ACL_UINT8) ? ScalarDesc(uint8_t(2)) :
                                                              ScalarDesc(int64_t(2));
                        auto out = TensorDesc(dims, outDtype, ACL_FORMAT_ND);
                        auto ut = OP_API_UT(aclnnDivMods, INPUT(self, other, mode), OUTPUT(out));
                        uint64_t workspaceSize = 0;
                        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
                    }
                }
            }
        }
    }
}

TEST_F(l2_div_mod_regbase_out_dtype_test, unsigned_out_is_not_unconditionally_supported)
{
    for (auto arch : {NpuArch::DAV_3510}) {
        op::SetPlatformNpuArch(arch);
        for (auto outDtype : {ACL_UINT16, ACL_UINT32, ACL_UINT64}) {
            for (int mode : {0, 2}) {
                SCOPED_TRACE(testing::Message()
                             << "arch=" << static_cast<int>(arch) << " mode=" << mode << " outDtype=" << outDtype);
                auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(6, 12);
                auto other = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(2, 3);
                auto scalar = ScalarDesc(2.0f);
                auto out = TensorDesc({2, 3}, outDtype, ACL_FORMAT_ND);
                auto tensorUt = OP_API_UT(aclnnDivMod, INPUT(self, other, mode), OUTPUT(out));
                auto scalarUt = OP_API_UT(aclnnDivMods, INPUT(self, scalar, mode), OUTPUT(out));
                uint64_t workspaceSize = 0;
                EXPECT_EQ(tensorUt.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
                EXPECT_EQ(scalarUt.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
            }
        }
    }
}

TEST_F(l2_div_mod_regbase_out_dtype_test, trunc_float_unsigned_out_uses_cast_path)
{
    for (auto arch : {NpuArch::DAV_3510}) {
        op::SetPlatformNpuArch(arch);
        for (auto outDtype : {ACL_UINT16, ACL_UINT32, ACL_UINT64}) {
            for (auto dims : {vector<int64_t>{2, 3}, vector<int64_t>{0}}) {
                SCOPED_TRACE(testing::Message()
                             << "arch=" << static_cast<int>(arch) << " outDtype=" << outDtype << " dim0=" << dims[0]);
                auto self = TensorDesc(dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(6, 12);
                auto other = TensorDesc(dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(2, 3);
                auto scalar = ScalarDesc(2.0f);
                auto out = TensorDesc(dims, outDtype, ACL_FORMAT_ND);
                auto tensorUt = OP_API_UT(aclnnDivMod, INPUT(self, other, 1), OUTPUT(out));
                auto scalarUt = OP_API_UT(aclnnDivMods, INPUT(self, scalar, 1), OUTPUT(out));
                uint64_t workspaceSize = 0;
                EXPECT_EQ(tensorUt.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
                EXPECT_EQ(scalarUt.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
            }
        }
    }
}

TEST_F(l2_div_mod_regbase_out_dtype_test, bool_inputs_support_unsigned_out)
{
    for (auto arch : {NpuArch::DAV_3510}) {
        op::SetPlatformNpuArch(arch);
        for (int mode : {1, 2}) {
            auto self = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(1, 1);
            auto other = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(1, 1);
            auto scalar = ScalarDesc(true);
            auto out = TensorDesc({2, 3}, ACL_UINT16, ACL_FORMAT_ND);
            auto tensorUt = OP_API_UT(aclnnDivMod, INPUT(self, other, mode), OUTPUT(out));
            auto scalarUt = OP_API_UT(aclnnDivMods, INPUT(self, scalar, mode), OUTPUT(out));
            uint64_t workspaceSize = 0;
            EXPECT_EQ(tensorUt.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
            EXPECT_EQ(scalarUt.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
        }
    }
}

TEST_F(l2_div_mod_regbase_out_dtype_test, negative_integer_inputs_support_unsigned_out)
{
    for (auto arch : {NpuArch::DAV_3510}) {
        op::SetPlatformNpuArch(arch);
        for (int mode : {1, 2}) {
            for (auto outDtype : {ACL_UINT16, ACL_UINT32, ACL_UINT64}) {
                SCOPED_TRACE(testing::Message()
                             << "arch=" << static_cast<int>(arch) << " mode=" << mode << " outDtype=" << outDtype);
                auto self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-12, -6);
                auto other = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(2, 3);
                auto scalar = ScalarDesc(int32_t(2));
                auto out = TensorDesc({2, 3}, outDtype, ACL_FORMAT_ND);
                auto tensorUt = OP_API_UT(aclnnDivMod, INPUT(self, other, mode), OUTPUT(out));
                auto scalarUt = OP_API_UT(aclnnDivMods, INPUT(self, scalar, mode), OUTPUT(out));
                uint64_t workspaceSize = 0;
                EXPECT_EQ(tensorUt.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
                EXPECT_EQ(scalarUt.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
            }
        }
    }
}
