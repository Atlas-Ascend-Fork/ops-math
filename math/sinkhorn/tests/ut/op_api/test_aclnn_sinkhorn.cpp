/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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

#include "../../../op_api/aclnn_sinkhorn.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
#include <cstdlib>
#include <ctime>
#include <limits>

using namespace std;

class l2_sinkhorn_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "sinkhorn_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "sinkhorn_test TearDown" << endl; }
};

TEST_F(l2_sinkhorn_test, aclnnSinkhorn_float_8_2)
{
    const vector<int64_t>& costShape = {8, 2};
    aclDataType dType = ACL_FLOAT;
    aclFormat dFormat = ACL_FORMAT_ND;

    const vector<int64_t>& pShape = {8, 2};

    auto costTensorDesc = TensorDesc(costShape, dType, dFormat)
                              .Value(vector<float>{0.45f, 0.48f, 0.65f, 0.68f, 0.68f, 0.10f, 0.84f, 0.22f, 0.37f, 0.71f,
                                                   0.13f, 0.59f, 0.66f, 0.40f, 0.47f, 0.82f});
    auto pTensorDesc = TensorDesc(pShape, dType, dFormat).ValidCount(16);
    auto tolScalarDesc = ScalarDesc(0.0001f);

    auto ut = OP_API_UT(aclnnSinkhorn, INPUT(costTensorDesc, tolScalarDesc), OUTPUT(pTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_sinkhorn_test, ascend910B2_aclnnSinkhorn_bf16_8_2)
{
    auto originalSocVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    const vector<int64_t>& costShape = {8, 2};
    aclDataType dType = ACL_BF16;
    aclFormat dFormat = ACL_FORMAT_ND;

    const vector<int64_t>& pShape = {8, 2};

    auto costTensorDesc = TensorDesc(costShape, dType, dFormat).ValueRange(0.0, 1.0);
    auto pTensorDesc = TensorDesc(pShape, dType, dFormat).ValidCount(16);
    auto tolScalarDesc = ScalarDesc(0.0001f);

    auto ut = OP_API_UT(aclnnSinkhorn, INPUT(costTensorDesc, tolScalarDesc), OUTPUT(pTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    op::SetPlatformSocVersion(originalSocVersion);
}

TEST_F(l2_sinkhorn_test, aclnnSinkhorn_float_empty_row)
{
    const vector<int64_t>& costShape = {0, 2};
    aclDataType dType = ACL_FLOAT;
    aclFormat dFormat = ACL_FORMAT_ND;

    const vector<int64_t>& pShape = {0, 2};

    auto costTensorDesc = TensorDesc(costShape, dType, dFormat).ValueRange(0.0, 1.0);
    auto pTensorDesc = TensorDesc(pShape, dType, dFormat).ValidCount(0);
    auto tolScalarDesc = ScalarDesc(0.0001f);

    auto ut = OP_API_UT(aclnnSinkhorn, INPUT(costTensorDesc, tolScalarDesc), OUTPUT(pTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_EQ(workspaceSize, 0);
}

TEST_F(l2_sinkhorn_test, aclnnSinkhorn_float_empty_col)
{
    const vector<int64_t>& costShape = {2, 0};
    aclDataType dType = ACL_FLOAT16;
    aclFormat dFormat = ACL_FORMAT_ND;

    const vector<int64_t>& pShape = {2, 0};

    auto costTensorDesc = TensorDesc(costShape, dType, dFormat).ValueRange(0.0, 1.0);
    auto pTensorDesc = TensorDesc(pShape, dType, dFormat).ValidCount(0);
    auto tolScalarDesc = ScalarDesc(0.0001f);

    auto ut = OP_API_UT(aclnnSinkhorn, INPUT(costTensorDesc, tolScalarDesc), OUTPUT(pTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_EQ(workspaceSize, 0);
}

TEST_F(l2_sinkhorn_test, aclnnSinkhorn_noncontiguous_output_rejected)
{
    const vector<int64_t>& shape = {3, 2};
    aclDataType dType = ACL_FLOAT;
    aclFormat dFormat = ACL_FORMAT_ND;

    auto costTensorDesc = TensorDesc(shape, dType, dFormat).ValueRange(0.0, 1.0);
    auto pTensorDesc = TensorDesc(shape, dType, dFormat, {4, 1}, 0, {3, 4}).ValidCount(6);
    auto tolScalarDesc = ScalarDesc(0.0001f);

    auto ut = OP_API_UT(aclnnSinkhorn, INPUT(costTensorDesc, tolScalarDesc), OUTPUT(pTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_sinkhorn_test, aclnnSinkhorn_low_precision_noncontiguous_output_accepted)
{
    const vector<int64_t> shape = {3, 2};
    auto cost = TensorDesc(shape, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
    auto output = TensorDesc(shape, ACL_FLOAT16, ACL_FORMAT_ND, {4, 1}, 0, {3, 4}).ValidCount(6);
    auto ut = OP_API_UT(aclnnSinkhorn, INPUT(cost, ScalarDesc(0.0001f)), OUTPUT(output));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

TEST_F(l2_sinkhorn_test, aclnnSinkhorn_null_tol_uses_default)
{
    const vector<int64_t>& shape = {4, 4};
    aclDataType dType = ACL_FLOAT;
    aclFormat dFormat = ACL_FORMAT_ND;

    auto costTensorDesc = TensorDesc(shape, dType, dFormat).ValueRange(0.0, 1.0);
    auto pTensorDesc = TensorDesc(shape, dType, dFormat).ValidCount(16);
    auto ut = OP_API_UT(aclnnSinkhorn, INPUT(costTensorDesc, (aclScalar*)nullptr), OUTPUT(pTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    auto explicitTol = ScalarDesc(0.0001f);
    auto explicitUt = OP_API_UT(aclnnSinkhorn, INPUT(costTensorDesc, explicitTol), OUTPUT(pTensorDesc));
    uint64_t explicitWorkspaceSize = 0;
    EXPECT_EQ(explicitUt.TestGetWorkspaceSize(&explicitWorkspaceSize), ACL_SUCCESS);
    EXPECT_EQ(workspaceSize, explicitWorkspaceSize);
}

TEST_F(l2_sinkhorn_test, aclnnSinkhorn_null_cost_rejected)
{
    const vector<int64_t>& shape = {4, 4};
    auto pTensorDesc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(16);
    auto tolScalarDesc = ScalarDesc(0.0001f);
    auto ut = OP_API_UT(aclnnSinkhorn, INPUT((aclTensor*)nullptr, tolScalarDesc), OUTPUT(pTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_sinkhorn_test, aclnnSinkhorn_null_output_rejected)
{
    const vector<int64_t>& shape = {4, 4};
    auto costTensorDesc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
    auto tolScalarDesc = ScalarDesc(0.0001f);
    auto ut = OP_API_UT(aclnnSinkhorn, INPUT(costTensorDesc, tolScalarDesc), OUTPUT((aclTensor*)nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_sinkhorn_test, aclnnSinkhorn_non_nd_format_rejected)
{
    const vector<int64_t>& shape = {4, 4};
    auto costTensorDesc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0.0, 1.0);
    auto pTensorDesc = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_NCHW).ValidCount(16);
    auto tolScalarDesc = ScalarDesc(0.0001f);
    auto ut = OP_API_UT(aclnnSinkhorn, INPUT(costTensorDesc, tolScalarDesc), OUTPUT(pTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_sinkhorn_test, aclnnSinkhorn_non_nd_output_format_rejected)
{
    const vector<int64_t> shape = {4, 4};
    auto cost = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
    auto output = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_NCHW).ValidCount(16);
    auto ut = OP_API_UT(aclnnSinkhorn, INPUT(cost, ScalarDesc(0.0001f)), OUTPUT(output));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_sinkhorn_test, ascend910B2_aclnnSinkhorn_float16_8_2)
{
    const vector<int64_t>& costShape = {8, 2};
    aclDataType dType = ACL_FLOAT16;
    aclFormat dFormat = ACL_FORMAT_ND;

    const vector<int64_t>& pShape = {8, 2};

    auto costTensorDesc = TensorDesc(costShape, dType, dFormat).ValueRange(0, 1);
    auto pTensorDesc = TensorDesc(pShape, dType, dFormat).ValidCount(16);
    auto tolScalarDesc = ScalarDesc(0.0001f);

    auto ut = OP_API_UT(aclnnSinkhorn, INPUT(costTensorDesc, tolScalarDesc), OUTPUT(pTensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_sinkhorn_test, aclnnSinkhorn_all_float_tol_values_accepted)
{
    const vector<int64_t> shape = {4, 4};
    const std::vector<float> tolValues = {0.0f,
                                          -0.0f,
                                          -0.0001f,
                                          std::numeric_limits<float>::infinity(),
                                          -std::numeric_limits<float>::infinity(),
                                          std::numeric_limits<float>::quiet_NaN()};
    for (float tol : tolValues) {
        auto cost = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
        auto output = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(16);
        auto ut = OP_API_UT(aclnnSinkhorn, INPUT(cost, ScalarDesc(tol)), OUTPUT(output));
        uint64_t workspaceSize = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
    }
}

TEST_F(l2_sinkhorn_test, aclnnSinkhorn_convertible_tol_dtype_accepted)
{
    const vector<int64_t> shape = {4, 4};
    auto cost = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
    auto output = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(16);
    auto float16Ut = OP_API_UT(aclnnSinkhorn, INPUT(cost, ScalarDesc(0.0001f, ACL_FLOAT16)), OUTPUT(output));
    auto doubleUt = OP_API_UT(aclnnSinkhorn, INPUT(cost, ScalarDesc(0.001)), OUTPUT(output));
    auto int64Ut = OP_API_UT(aclnnSinkhorn, INPUT(cost, ScalarDesc(int64_t{1})), OUTPUT(output));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(float16Ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
    EXPECT_EQ(doubleUt.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
    EXPECT_EQ(int64Ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

TEST_F(l2_sinkhorn_test, aclnnSinkhorn_invalid_rank_rejected)
{
    for (const vector<int64_t>& shape : {vector<int64_t>{}, vector<int64_t>{4}, vector<int64_t>{2, 3, 4}}) {
        auto cost = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
        auto output = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(24);
        auto ut = OP_API_UT(aclnnSinkhorn, INPUT(cost, ScalarDesc(0.0001f)), OUTPUT(output));
        uint64_t workspaceSize = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_sinkhorn_test, aclnnSinkhorn_output_rank_rejected)
{
    auto cost = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
    auto output = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(16);
    auto ut = OP_API_UT(aclnnSinkhorn, INPUT(cost, ScalarDesc(0.0001f)), OUTPUT(output));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_sinkhorn_test, aclnnSinkhorn_row_above_10000_accepted)
{
    const vector<int64_t> shape = {10001, 2};
    auto cost = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
    auto output = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(1);
    auto ut = OP_API_UT(aclnnSinkhorn, INPUT(cost, ScalarDesc(0.0001f)), OUTPUT(output));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

TEST_F(l2_sinkhorn_test, aclnnSinkhorn_col_above_4096_rejected)
{
    const vector<int64_t> shape = {2, 4097};
    auto cost = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
    auto output = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(1);
    auto ut = OP_API_UT(aclnnSinkhorn, INPUT(cost, ScalarDesc(0.0001f)), OUTPUT(output));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_sinkhorn_test, aclnnSinkhorn_element_count_over_uint32_rejected)
{
    const vector<int64_t> shape = {1048576, 4096};
    auto cost = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
    auto output = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(1);
    auto ut = OP_API_UT(aclnnSinkhorn, INPUT(cost, ScalarDesc(0.0001f)), OUTPUT(output));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_sinkhorn_test, aclnnSinkhorn_shape_mismatch_rejected)
{
    auto cost = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
    auto output = TensorDesc({4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(20);
    auto ut = OP_API_UT(aclnnSinkhorn, INPUT(cost, ScalarDesc(0.0001f)), OUTPUT(output));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_sinkhorn_test, aclnnSinkhorn_low_precision_input_float_output_accepted)
{
    const vector<int64_t> shape = {4, 4};
    auto cost = TensorDesc(shape, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
    auto output = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(16);
    auto ut = OP_API_UT(aclnnSinkhorn, INPUT(cost, ScalarDesc(0.0001f)), OUTPUT(output));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

TEST_F(l2_sinkhorn_test, aclnnSinkhorn_float_input_low_precision_output_rejected)
{
    const vector<int64_t> shape = {4, 4};
    auto cost = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
    auto output = TensorDesc(shape, ACL_FLOAT16, ACL_FORMAT_ND).ValidCount(16);
    auto ut = OP_API_UT(aclnnSinkhorn, INPUT(cost, ScalarDesc(0.0001f)), OUTPUT(output));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_sinkhorn_test, aclnnSinkhorn_unsupported_dtype_rejected)
{
    const vector<int64_t> shape = {4, 4};
    auto cost = TensorDesc(shape, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 1);
    auto output = TensorDesc(shape, ACL_INT32, ACL_FORMAT_ND).ValidCount(16);
    auto ut = OP_API_UT(aclnnSinkhorn, INPUT(cost, ScalarDesc(0.0001f)), OUTPUT(output));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_sinkhorn_test, ascend950_aclnnSinkhorn_bf16_accepted)
{
    auto originalSocVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    const vector<int64_t> shape = {8, 2};
    auto cost = TensorDesc(shape, ACL_BF16, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
    auto output = TensorDesc(shape, ACL_BF16, ACL_FORMAT_ND).ValidCount(16);
    auto ut = OP_API_UT(aclnnSinkhorn, INPUT(cost, ScalarDesc(0.0001f)), OUTPUT(output));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
    op::SetPlatformSocVersion(originalSocVersion);
}

TEST_F(l2_sinkhorn_test, unsupported_soc_aclnnSinkhorn_bf16_rejected)
{
    auto originalSocVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910);
    const vector<int64_t> shape = {8, 2};
    auto cost = TensorDesc(shape, ACL_BF16, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
    auto output = TensorDesc(shape, ACL_BF16, ACL_FORMAT_ND).ValidCount(16);
    auto ut = OP_API_UT(aclnnSinkhorn, INPUT(cost, ScalarDesc(0.0001f)), OUTPUT(output));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
    op::SetPlatformSocVersion(originalSocVersion);
}

TEST_F(l2_sinkhorn_test, aclnnSinkhorn_empty_with_nonfinite_tol_accepted)
{
    const vector<int64_t> shape = {0, 2};
    auto cost = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
    auto output = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(0);
    auto ut = OP_API_UT(aclnnSinkhorn, INPUT(cost, ScalarDesc(std::numeric_limits<float>::infinity())), OUTPUT(output));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
    EXPECT_EQ(workspaceSize, 0);
}
