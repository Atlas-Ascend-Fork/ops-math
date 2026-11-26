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

#include "../../../op_api/aclnn_fill_tensor.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_fill_tensor_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "fill_tensor_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "fill_tensor_test TearDown" << endl; }
};

TEST_F(l2_fill_tensor_test, case_001_FLOAT)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1).Precision(0.0001, 0.0001);
    auto value_tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(8.0, 9.0).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_tensor_test, case_002_FLOAT16)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1).Precision(0.0001, 0.0001);
    auto value_tensor_desc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(8.0, 9.0).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_tensor_test, case_003_INT32)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto value_tensor_desc = TensorDesc({}, ACL_INT32, ACL_FORMAT_ND).ValueRange(8, 9);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_tensor_test, case_004_INT64)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto value_tensor_desc = TensorDesc({}, ACL_INT64, ACL_FORMAT_ND).ValueRange(8, 9);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_tensor_test, case_005_INT16)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto value_tensor_desc = TensorDesc({}, ACL_INT16, ACL_FORMAT_ND).ValueRange(8, 9);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_tensor_test, case_006_INT8)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto value_tensor_desc = TensorDesc({}, ACL_INT8, ACL_FORMAT_ND).ValueRange(8, 9);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_tensor_test, case_007_UINT8)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 10);
    auto value_tensor_desc = TensorDesc({}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(8, 9);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_tensor_test, case_008_BOOL)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-1, 2);
    auto value_tensor_desc = TensorDesc({}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 1);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_tensor_test, case_009_DOUBLE)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto value_tensor_desc = TensorDesc({}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(8.0, 9.0);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_tensor_test, case_010_COMPLEX64)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto value_tensor_desc = TensorDesc({}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(8.0, 9.0);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_tensor_test, case_011_COMPLEX128)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto value_tensor_desc = TensorDesc({}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(8.0, 9.0);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_tensor_test, case_012_NHWC)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_INT32, ACL_FORMAT_NHWC).ValueRange(-10, 10);
    auto value_tensor_desc = TensorDesc({}, ACL_INT32, ACL_FORMAT_ND).ValueRange(8, 9);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_tensor_test, case_013_NCHW)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
    auto value_tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(8.0, 9.0);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_tensor_test, case_014_NDHWC)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-10, 10);
    auto value_tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(8.0, 9.0);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_tensor_test, case_015_NCDHW)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-10, 10);
    auto value_tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(8.0, 9.0);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_tensor_test, case_016_NC1HWC0)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_INT32, ACL_FORMAT_NC1HWC0).ValueRange(-10, 10);
    auto value_tensor_desc = TensorDesc({}, ACL_INT32, ACL_FORMAT_ND).ValueRange(8, 9);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_tensor_test, case_017_EMPTY)
{
    auto self_tensor_desc = TensorDesc({2, 0, 3}, ACL_INT32, ACL_FORMAT_NHWC).ValueRange(-10, 10);
    auto value_tensor_desc = TensorDesc({}, ACL_INT32, ACL_FORMAT_ND).ValueRange(8, 9);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_tensor_test, case_018_CONTINUOUS)
{
    auto self_tensor_desc = TensorDesc({5, 4}, ACL_INT8, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-1, 10);
    auto value_tensor_desc = TensorDesc({}, ACL_INT8, ACL_FORMAT_ND).ValueRange(8, 9);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_tensor_test, case_019_MAX_DIM)
{
    auto self_tensor_desc = TensorDesc({2, 3, 2, 3, 2, 2, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto value_tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(8.0, 9.0);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fill_tensor_test, case_020_DIM_0)
{
    auto self_tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto value_tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(8.0, 9.0);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_tensor_test, case_021_DIM_1)
{
    auto self_tensor_desc = TensorDesc({100}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto value_tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(8.0, 9.0);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_tensor_test, case_022_DIM_3)
{
    auto self_tensor_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto value_tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(8.0, 9.0);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_tensor_test, case_023_DIM_4)
{
    auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto value_tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(8.0, 9.0);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_tensor_test, case_024_DIM_5)
{
    auto self_tensor_desc = TensorDesc({2, 3, 4, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto value_tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(8.0, 9.0);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_tensor_test, case_025_DIM_6)
{
    auto self_tensor_desc = TensorDesc({2, 3, 4, 5, 6, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto value_tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(8.0, 9.0);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_tensor_test, case_026_DIM_7)
{
    auto self_tensor_desc = TensorDesc({2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto value_tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(8.0, 9.0);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_tensor_test, case_027_DIM_8)
{
    auto self_tensor_desc = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto value_tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(8.0, 9.0);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_tensor_test, case_028_VALUE_1D)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto value_tensor_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(8.0, 9.0);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fill_tensor_test, case_029_VALUE_DIM_ERROR)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto value_tensor_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(8.0, 9.0);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fill_tensor_test, case_030_NOT_SUPPORT_DTYPE)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_COMPLEX32, ACL_FORMAT_ND).ValueRange(0, 10);
    auto value_tensor_desc = TensorDesc({}, ACL_COMPLEX32, ACL_FORMAT_ND).ValueRange(8, 9);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fill_tensor_test, case_031_SELF_NULLPTR)
{
    auto value_tensor_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(8.0, 9.0);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(nullptr, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_fill_tensor_test, case_032_VALUE_NULLPTR)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, nullptr), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_fill_tensor_test, case_033_DTYPE_CAST)
{
    auto self_tensor_desc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto value_tensor_desc = TensorDesc({}, ACL_INT32, ACL_FORMAT_ND).ValueRange(8, 9);

    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self_tensor_desc, value_tensor_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

class l2_fill_tensor_unsigned_test : public l2_fill_tensor_test, public testing::WithParamInterface<aclDataType> {};

TEST_P(l2_fill_tensor_unsigned_test, normal_scalar_and_single_element)
{
    for (const auto& valueShape : std::vector<std::vector<int64_t>>{{}, {1}}) {
        auto self = TensorDesc({8, 9}, GetParam(), ACL_FORMAT_ND).ValueRange(0, 10);
        auto value = TensorDesc(valueShape, GetParam(), ACL_FORMAT_ND).ValueRange(7, 7);
        auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self, value), OUTPUT());
        uint64_t workspaceSize = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
    }
}

TEST_P(l2_fill_tensor_unsigned_test, empty_tensor)
{
    auto self = TensorDesc({2, 0, 3}, GetParam(), ACL_FORMAT_ND).ValueRange(0, 10);
    auto value = TensorDesc({1}, GetParam(), ACL_FORMAT_ND).ValueRange(7, 7);
    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self, value), OUTPUT());
    uint64_t workspaceSize = 123;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
    EXPECT_EQ(workspaceSize, 0U);
}

TEST_P(l2_fill_tensor_unsigned_test, int32_value_conversion)
{
    auto self = TensorDesc({33}, GetParam(), ACL_FORMAT_ND).ValueRange(0, 10);
    auto value = TensorDesc({}, ACL_INT32, ACL_FORMAT_ND).ValueRange(7, 7);
    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self, value), OUTPUT());
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_P(l2_fill_tensor_unsigned_test, strided_offset_preserves_caller_metadata)
{
    auto self = TensorDesc({3, 5}, GetParam(), ACL_FORMAT_ND, {26, 2}, 11, {90}).ToAclTypeRawPtr();
    auto value = TensorDesc({1}, GetParam(), ACL_FORMAT_ND, {7}, 5, {18}).ToAclTypeRawPtr();
    const auto selfDtype = self->GetDataType();
    const auto selfShape = self->GetViewShape();
    const auto selfStorage = self->GetStorageShape();
    const auto selfOriginal = self->GetOriginalShape();
    const auto selfStride = self->GetViewStrides();
    const auto selfOffset = self->GetViewOffset();
    const auto selfStorageOffset = self->GetStorageOffset();
    const auto valueDtype = value->GetDataType();
    const auto valueShape = value->GetViewShape();
    const auto valueStorage = value->GetStorageShape();
    const auto valueOriginal = value->GetOriginalShape();
    const auto valueStride = value->GetViewStrides();
    const auto valueOffset = value->GetViewOffset();
    const auto valueStorageOffset = value->GetStorageOffset();
    // OpApiUt owns these ACL tensors and releases its executor after phase one.
    auto ut = OP_API_UT(aclnnInplaceFillTensor, INPUT(self, value), OUTPUT());
    uint64_t workspaceSize = 0;
    ASSERT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
    EXPECT_EQ(self->GetDataType(), selfDtype);
    EXPECT_EQ(self->GetViewShape(), selfShape);
    EXPECT_EQ(self->GetStorageShape(), selfStorage);
    EXPECT_EQ(self->GetOriginalShape(), selfOriginal);
    EXPECT_EQ(self->GetViewStrides(), selfStride);
    EXPECT_EQ(self->GetViewOffset(), selfOffset);
    EXPECT_EQ(self->GetStorageOffset(), selfStorageOffset);
    EXPECT_EQ(value->GetDataType(), valueDtype);
    EXPECT_EQ(value->GetViewShape(), valueShape);
    EXPECT_EQ(value->GetStorageShape(), valueStorage);
    EXPECT_EQ(value->GetOriginalShape(), valueOriginal);
    EXPECT_EQ(value->GetViewStrides(), valueStride);
    EXPECT_EQ(value->GetViewOffset(), valueOffset);
    EXPECT_EQ(value->GetStorageOffset(), valueStorageOffset);
}

INSTANTIATE_TEST_SUITE_P(UnsignedDtypes, l2_fill_tensor_unsigned_test,
                         testing::Values(ACL_UINT16, ACL_UINT32, ACL_UINT64));
