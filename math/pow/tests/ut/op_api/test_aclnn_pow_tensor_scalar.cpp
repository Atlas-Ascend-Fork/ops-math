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

#include "../../../op_api/aclnn_pow.h"

#include "opdev/platform.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_common/op_host/util/op_const_def.h"

using namespace std;

class l2_pow_tensor_scalar_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "pow_tensor_scalar_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "pow_tensor_scalar_test TearDown" << endl; }
};

TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_null_0)
{
    // input
    const vector<int64_t>& selfShape = {0};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {0};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(0, 1);
    auto expScalarDesc = ScalarDesc(3);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_null_1)
{
    // input
    const vector<int64_t>& selfShape = {1, 11, 34, 22, 19, 6, 5, 1};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {1, 11, 34, 22, 19, 6, 5, 1};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(0, 1);
    auto expScalarDesc = ScalarDesc(3);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(nullptr, expScalarDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_nullptr2)
{
    // input
    const vector<int64_t>& selfShape = {1, 11, 34, 22, 19, 6, 5, 1};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {1, 11, 34, 22, 19, 6, 5, 1};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(0, 1);
    auto expScalarDesc = ScalarDesc(3);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(nullptr));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_1_1_1_2_2_2_1_1_float_nd_1_1_1_2_2_2_1_1_float_nd)
{
    // input
    const vector<int64_t>& selfShape = {1, 1, 1, 2, 2, 2, 1, 1};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {1, 1, 1, 2, 2, 2, 1, 1};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(0, 1);
    auto expScalarDesc = ScalarDesc(3);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_1_2_3_4_5_6_7_8_9_shape9_nd_1_2_3_4_5_6_7_8_9_shape9_nd)
{
    // input
    const vector<int64_t>& selfShape = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(0, 1);
    auto expScalarDesc = ScalarDesc(3);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_223_32_17_17_float16_nchw_223_1_1_17_float16_nchw)
{
    // left input
    const vector<int64_t>& selfShape = {223, 32, 17, 17};
    aclDataType selfDtype = ACL_FLOAT16;
    aclFormat selfFormat = ACL_FORMAT_NCHW;

    // output
    const vector<int64_t>& outShape = {223, 32, 17, 17};
    aclDataType outDtype = ACL_FLOAT16;
    aclFormat outFormat = ACL_FORMAT_NCHW;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(0, 1);
    auto expScalarDesc = ScalarDesc(2);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
/*
TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_11_32_1_12_float64_nchw_11_32_1_12_float64_nchw) {
  // input
  const vector<int64_t>& selfShape = {11,32,1,12};
  aclDataType selfDtype = ACL_DOUBLE;
  aclFormat selfFormat = ACL_FORMAT_ND;

  // output
  const vector<int64_t>& outShape = {11,32,1,12};
  aclDataType outDtype = ACL_DOUBLE;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(0,1);
  auto expScalarDesc = ScalarDesc(2);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}
 */

TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_1_1_16_129_int8_hwcn_1_1_16_129_int8_hwcn)
{
    // input
    const vector<int64_t>& selfShape = {1, 1, 16, 129};
    aclDataType selfDtype = ACL_INT8;
    aclFormat selfFormat = ACL_FORMAT_HWCN;
    // output
    const vector<int64_t>& outShape = {1, 1, 16, 129};
    aclDataType outDtype = ACL_INT8;
    aclFormat outFormat = ACL_FORMAT_HWCN;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto expScalarDesc = ScalarDesc(2);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_1_1_16_129_in32_hwcn_1_1_16_129_int32_hwcn)
{
    // input
    const vector<int64_t>& selfShape = {1, 1, 16, 129};
    aclDataType selfDtype = ACL_INT32;
    aclFormat selfFormat = ACL_FORMAT_HWCN;
    // output
    const vector<int64_t>& outShape = {1, 1, 16, 129};
    aclDataType outDtype = ACL_INT32;
    aclFormat outFormat = ACL_FORMAT_HWCN;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto expScalarDesc = ScalarDesc(2);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_1_1_16_129_in64_hwcn_1_1_16_129_int64_hwcn)
{
    // input
    const vector<int64_t>& selfShape = {1, 1, 16, 129};
    aclDataType selfDtype = ACL_INT64;
    aclFormat selfFormat = ACL_FORMAT_HWCN;
    // output
    const vector<int64_t>& outShape = {1, 1, 16, 129};
    aclDataType outDtype = ACL_INT64;
    aclFormat outFormat = ACL_FORMAT_HWCN;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto expScalarDesc = ScalarDesc(2);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_pow_tensor_scalar_test, Ascend950_aclnnPowTensorScalar_square_complex64)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);

    const vector<int64_t>& selfShape = {1, 1, 16, 129};
    aclDataType selfDtype = ACL_COMPLEX64;
    aclFormat selfFormat = ACL_FORMAT_HWCN;

    const vector<int64_t>& outShape = {1, 1, 16, 129};
    aclDataType outDtype = ACL_COMPLEX64;
    aclFormat outFormat = ACL_FORMAT_HWCN;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto expScalarDesc = ScalarDesc(2);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_pow_tensor_scalar_test, Ascend950_aclnnPowTensorScalar_square_complex128)
{
    op::SetPlatformNpuArch(Ops::Base::DAV_3510);

    const vector<int64_t>& selfShape = {1, 1, 16, 129};
    aclDataType selfDtype = ACL_COMPLEX128;
    aclFormat selfFormat = ACL_FORMAT_HWCN;

    const vector<int64_t>& outShape = {1, 1, 16, 129};
    aclDataType outDtype = ACL_COMPLEX128;
    aclFormat outFormat = ACL_FORMAT_HWCN;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto expScalarDesc = ScalarDesc(2);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    op::SetPlatformNpuArch(Ops::Base::DAV_2201);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_pow_tensor_scalar_test, Ascend950_out_dtype_check_not_applied)
{
    op::SetPlatformNpuArch(Ops::Base::DAV_3510);
    auto selfTensorDesc = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto expScalarDesc = ScalarDesc(2);
    auto outTensorDesc = TensorDesc({4, 8}, ACL_INT4, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    EXPECT_NE(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
    op::SetPlatformNpuArch(Ops::Base::DAV_2201);
}
/*
TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_1_1_16_129_complex64_hwcn_1_1_16_129_complex64_hwcn) {
  // input
  const vector<int64_t>& selfShape = {1,1,16,129};
  aclDataType selfDtype = ACL_COMPLEX64;
  aclFormat selfFormat = ACL_FORMAT_HWCN;
  // output
  const vector<int64_t>& outShape = {1,1,16,129};
  aclDataType outDtype = ACL_COMPLEX64;
  aclFormat outFormat = ACL_FORMAT_HWCN;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto expScalarDesc = ScalarDesc(2);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_1_1_16_129_complex128_hwcn_1_1_16_129_complex128_hwcn) {
  // input
  const vector<int64_t>& selfShape = {1,1,16,129};
  aclDataType selfDtype = ACL_COMPLEX128;
  aclFormat selfFormat = ACL_FORMAT_HWCN;
  // output
  const vector<int64_t>& outShape = {1,1,16,129};
  aclDataType outDtype = ACL_COMPLEX128;
  aclFormat outFormat = ACL_FORMAT_HWCN;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto expScalarDesc = ScalarDesc(2);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

}

TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_1_1_16_129_bf16_hwcn_1_1_16_129_bf16_hwcn) {
  // input
  const vector<int64_t>& selfShape = {1,1,16,129};
  aclDataType selfDtype = ACL_BF16;
  aclFormat selfFormat = ACL_FORMAT_HWCN;
  // output
  const vector<int64_t>& outShape = {1,1,16,129};
  aclDataType outDtype = ACL_BF16;
  aclFormat outFormat = ACL_FORMAT_HWCN;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto expScalarDesc = ScalarDesc(2);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
 */
TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_1_1_16_129_bool_hwcn_1_1_16_129_int64_hwcn)
{
    // input
    const vector<int64_t>& selfShape = {1, 1, 16, 129};
    aclDataType selfDtype = ACL_BOOL;
    aclFormat selfFormat = ACL_FORMAT_HWCN;
    // output
    const vector<int64_t>& outShape = {1, 1, 16, 129};
    aclDataType outDtype = ACL_INT64;
    aclFormat outFormat = ACL_FORMAT_HWCN;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto expScalarDesc = ScalarDesc(2);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_1_1_16_129_int16_hwcn_1_1_16_129_int16_hwcn)
{
    // input
    const vector<int64_t>& selfShape = {1, 1, 16, 129};
    aclDataType selfDtype = ACL_INT16;
    aclFormat selfFormat = ACL_FORMAT_HWCN;
    // output
    const vector<int64_t>& outShape = {1, 1, 16, 129};
    aclDataType outDtype = ACL_INT16;
    aclFormat outFormat = ACL_FORMAT_HWCN;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto expScalarDesc = ScalarDesc(2);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_1_1_16_129_undefined_hwcn_1_1_16_129_undefined_hwcn)
{
    // input
    const vector<int64_t>& selfShape = {1, 1, 16, 129};
    aclDataType selfDtype = ACL_DT_UNDEFINED;
    aclFormat selfFormat = ACL_FORMAT_HWCN;
    // output
    const vector<int64_t>& outShape = {1, 1, 16, 129};
    aclDataType outDtype = ACL_DT_UNDEFINED;
    aclFormat outFormat = ACL_FORMAT_HWCN;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto expScalarDesc = ScalarDesc(2);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

    auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
/*
TEST_F(l2_pow_tensor_scalar_test, aclnnPowTensorScalar_1_float_hwcn_1_complex64_hwcn) {
  // input
  const vector<int64_t>& selfShape = {1};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_HWCN;
  // output
  const vector<int64_t>& outShape = {1};
  aclDataType outDtype = ACL_COMPLEX64;
  aclFormat outFormat = ACL_FORMAT_HWCN;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  float src_value = 1.0f;
  auto expScalarDesc = aclCreateScalar(&src_value, aclDataType::ACL_COMPLEX64);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat);

  auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Ascend950PR_9589测试float16 tensor + complex scalar
TEST_F(l2_pow_tensor_scalar_test, Ascend950PR_89_aclnnPowTensorScalar_base_fp16_scalar_complex) {
  // input
  const vector<int64_t>& selfShape = {4, 5};
  aclDataType selfDtype = ACL_FLOAT16;
  aclFormat selfFormat = ACL_FORMAT_ND;

  // output
  const vector<int64_t>& outShape = {4, 5};
  aclDataType outDtype = ACL_COMPLEX64;
  aclFormat outFormat = ACL_FORMAT_ND;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(0,1);
  auto expScalarDesc = ScalarDesc(2.5 + 2.5j);
  auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(selfTensorDesc, expScalarDesc), OUTPUT(outTensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
} */

namespace op {
extern SocVersion g_socVersion;
extern NpuArch g_npuArchOverride;
} // namespace op

class l2_pow_scalar_empty_validation_test : public testing::Test {
protected:
    void SetUp() override
    {
        savedSoc_ = op::g_socVersion;
        savedArch_ = op::g_npuArchOverride;
        op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    }
    void TearDown() override
    {
        op::g_socVersion = savedSoc_;
        op::g_npuArchOverride = savedArch_;
    }
    void Check(const vector<int64_t>& shape, aclDataType dtype, const ScalarDesc& exponent, aclnnStatus expected)
    {
        auto self = TensorDesc(shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
        auto out = TensorDesc(shape, dtype, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnPowTensorScalar, INPUT(self, exponent), OUTPUT(out));
        uint64_t workspaceSize = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), expected);
        auto inplace = OP_API_UT(aclnnInplacePowTensorScalar, INPUT(self, exponent), OUTPUT());
        EXPECT_EQ(inplace.TestGetWorkspaceSize(&workspaceSize), expected);
        if (expected == ACL_SUCCESS && shape[0] == 0) {
            EXPECT_EQ(workspaceSize, 0U);
        }
    }

private:
    op::SocVersion savedSoc_;
    NpuArch savedArch_;
};

TEST_F(l2_pow_scalar_empty_validation_test, overflowing_float_exponent)
{
    for (auto shape : {vector<int64_t>{0, 2}, vector<int64_t>{2, 2}}) {
        Check(shape, ACL_FLOAT, ScalarDesc(1e308), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_pow_scalar_empty_validation_test, negative_integer_exponent)
{
    for (auto shape : {vector<int64_t>{0, 2}, vector<int64_t>{2, 2}}) {
        Check(shape, ACL_INT32, ScalarDesc(int32_t(-1)), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_pow_scalar_empty_validation_test, valid_empty_and_mixed_inputs)
{
    Check({0, 2}, ACL_FLOAT, ScalarDesc(2.0), ACL_SUCCESS);
    Check({0, 2}, ACL_INT32, ScalarDesc(int32_t(0)), ACL_SUCCESS);
    Check({0, 2}, ACL_FLOAT, ScalarDesc(int32_t(-1)), ACL_SUCCESS);
    Check({0, 2}, ACL_DOUBLE, ScalarDesc(1e308), ACL_SUCCESS);
}

TEST_F(l2_pow_scalar_empty_validation_test, legacy_platform_empty_behavior)
{
    for (auto soc : {op::SocVersion::ASCEND910B, op::SocVersion::ASCEND910_93}) {
        op::SetPlatformSocVersion(soc);
        Check({0, 2}, ACL_FLOAT, ScalarDesc(1e308), ACL_SUCCESS);
        Check({0, 2}, ACL_INT32, ScalarDesc(int32_t(-1)), ACL_SUCCESS);
    }
}
