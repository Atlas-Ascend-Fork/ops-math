/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_api/aclnn_fused_exponential.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

namespace op {
extern SocVersion g_socVersion;
extern NpuArch g_npuArchOverride;
} // namespace op

using namespace op;
namespace {
class FusedExponentialApi : public testing::TestWithParam<aclDataType> {};

TEST(FusedExponentialPlatform, RejectsUnsupportedArchitectureIncludingEmptyInput)
{
    const auto savedSoc = op::g_socVersion;
    const auto savedArch = op::g_npuArchOverride;
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910);
    for (int64_t length : {0, 65}) {
        auto input = TensorDesc({length}, ACL_FLOAT, ACL_FORMAT_ND);
        auto output = TensorDesc({length}, ACL_FLOAT, ACL_FORMAT_ND);
        auto scalar = ScalarDesc(1.0);
        auto test = OP_API_UT(aclnnFusedExponential, INPUT(input, scalar), OUTPUT(output));
        uint64_t workspace = 0;
        EXPECT_EQ(test.TestGetWorkspaceSize(&workspace), ACLNN_ERR_PARAM_INVALID);
    }
    op::g_socVersion = savedSoc;
    op::g_npuArchOverride = savedArch;
}
TEST_P(FusedExponentialApi, AcceptsSupportedDtypeAndScalar)
{
    auto input = TensorDesc({65}, GetParam(), ACL_FORMAT_ND);
    auto output = TensorDesc({65}, GetParam(), ACL_FORMAT_ND);
    for (double rate : {0.002, 1.0, 10000.0, std::numeric_limits<double>::infinity()}) {
        auto scalar = ScalarDesc(rate);
        auto test = OP_API_UT(aclnnFusedExponential, INPUT(input, scalar), OUTPUT(output));
        uint64_t workspace = 0;
        EXPECT_EQ(test.TestGetWorkspaceSize(&workspace), ACLNN_SUCCESS);
    }
}
TEST_P(FusedExponentialApi, EmptyTensorHasZeroWorkspace)
{
    auto input = TensorDesc({2, 0, 3}, GetParam(), ACL_FORMAT_ND);
    auto output = TensorDesc({2, 0, 3}, GetParam(), ACL_FORMAT_ND);
    auto scalar = ScalarDesc(1.0);
    auto test = OP_API_UT(aclnnFusedExponential, INPUT(input, scalar), OUTPUT(output));
    uint64_t workspace = 123;
    EXPECT_EQ(test.TestGetWorkspaceSize(&workspace), ACLNN_SUCCESS);
    EXPECT_EQ(workspace, 0U);
}
TEST_P(FusedExponentialApi, AcceptsScalarAndRankEight)
{
    for (const auto& shape : {std::vector<int64_t>{}, std::vector<int64_t>{1, 1, 1, 1, 1, 1, 1, 1}}) {
        auto input = TensorDesc(shape, GetParam(), ACL_FORMAT_ND);
        auto output = TensorDesc(shape, GetParam(), ACL_FORMAT_ND);
        auto scalar = ScalarDesc(1.0);
        auto test = OP_API_UT(aclnnFusedExponential, INPUT(input, scalar), OUTPUT(output));
        uint64_t workspace = 0;
        EXPECT_EQ(test.TestGetWorkspaceSize(&workspace), ACLNN_SUCCESS);
    }
}
INSTANTIATE_TEST_SUITE_P(Dtypes, FusedExponentialApi, testing::Values(ACL_FLOAT16, ACL_BF16, ACL_FLOAT));

TEST(FusedExponentialApiInvalid, RejectsDtypeAndShapeMismatch)
{
    auto scalar = ScalarDesc(1.0);
    auto input = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    for (auto output : {TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND), TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND)}) {
        auto test = OP_API_UT(aclnnFusedExponential, INPUT(input, scalar), OUTPUT(output));
        uint64_t workspace = 0;
        EXPECT_EQ(test.TestGetWorkspaceSize(&workspace), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST(FusedExponentialApiInvalid, RejectsUnsupportedDtype)
{
    auto input = TensorDesc({7}, ACL_INT32, ACL_FORMAT_ND);
    auto output = TensorDesc({7}, ACL_INT32, ACL_FORMAT_ND);
    auto scalar = ScalarDesc(1.0);
    auto test = OP_API_UT(aclnnFusedExponential, INPUT(input, scalar), OUTPUT(output));
    uint64_t workspace = 0;
    EXPECT_EQ(test.TestGetWorkspaceSize(&workspace), ACLNN_ERR_PARAM_INVALID);
}

TEST(FusedExponentialApiInvalid, RejectsRankNine)
{
    auto input = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalar = ScalarDesc(1.0);
    auto test = OP_API_UT(aclnnFusedExponential, INPUT(input, scalar), OUTPUT(output));
    uint64_t workspace = 0;
    EXPECT_EQ(test.TestGetWorkspaceSize(&workspace), ACLNN_ERR_PARAM_INVALID);
}

TEST(FusedExponentialApiInvalid, RejectsFloatScalar)
{
    auto input = TensorDesc({7}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output = TensorDesc({7}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalar = ScalarDesc(1.0f);
    auto test = OP_API_UT(aclnnFusedExponential, INPUT(input, scalar), OUTPUT(output));
    uint64_t workspace = 0;
    EXPECT_EQ(test.TestGetWorkspaceSize(&workspace), ACLNN_ERR_PARAM_INVALID);
}

TEST(FusedExponentialApiInvalid, RejectsNullInput)
{
    auto output = TensorDesc({7}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalar = ScalarDesc(1.0);
    auto test = OP_API_UT(aclnnFusedExponential, INPUT(static_cast<aclTensor*>(nullptr), scalar), OUTPUT(output));
    uint64_t workspace = 0;
    EXPECT_EQ(test.TestGetWorkspaceSize(&workspace), ACLNN_ERR_PARAM_NULLPTR);
}

TEST(FusedExponentialApiInvalid, RejectsNullOutput)
{
    auto input = TensorDesc({7}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalar = ScalarDesc(1.0);
    auto test = OP_API_UT(aclnnFusedExponential, INPUT(input, scalar), OUTPUT(static_cast<aclTensor*>(nullptr)));
    uint64_t workspace = 0;
    EXPECT_EQ(test.TestGetWorkspaceSize(&workspace), ACLNN_ERR_PARAM_NULLPTR);
}

TEST(FusedExponentialApiInvalid, RejectsNullScalar)
{
    auto input = TensorDesc({7}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output = TensorDesc({7}, ACL_FLOAT, ACL_FORMAT_ND);
    auto test = OP_API_UT(aclnnFusedExponential, INPUT(input, static_cast<aclScalar*>(nullptr)), OUTPUT(output));
    uint64_t workspace = 0;
    EXPECT_EQ(test.TestGetWorkspaceSize(&workspace), ACLNN_ERR_PARAM_NULLPTR);
}
} // namespace
