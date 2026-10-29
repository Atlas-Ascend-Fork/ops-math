/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class SpaceToDepthTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SpaceToDepthTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SpaceToDepthTest TearDown" << std::endl; }
};

TEST_F(SpaceToDepthTest, space_to_depth_infer_shape_nhwc)
{
    gert::StorageShape xShape = {{1, 2, 2, 4}, {1, 2, 2, 4}};
    gert::StorageShape yShape;
    gert::InfershapeContextPara infershapeContextPara(
        "SpaceToDepth", {{{xShape}, ge::DT_FLOAT, ge::FORMAT_NHWC}}, {{{yShape}, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {gert::InfershapeContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::InfershapeContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 1, 1, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SpaceToDepthTest, space_to_depth_infer_shape_nchw)
{
    gert::StorageShape xShape = {{1, 4, 2, 2}, {1, 4, 2, 2}};
    gert::StorageShape yShape;
    gert::InfershapeContextPara infershapeContextPara(
        "SpaceToDepth", {{{xShape}, ge::DT_FLOAT, ge::FORMAT_NCHW}}, {{{yShape}, ge::DT_FLOAT, ge::FORMAT_NCHW}},
        {gert::InfershapeContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::InfershapeContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NCHW"))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 16, 1, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SpaceToDepthTest, space_to_depth_infer_shape_dynamic_dim)
{
    gert::StorageShape xShape = {{1, -1, -1, 4}, {1, -1, -1, 4}};
    gert::StorageShape yShape;
    gert::InfershapeContextPara infershapeContextPara(
        "SpaceToDepth", {{{xShape}, ge::DT_FLOAT, ge::FORMAT_NHWC}}, {{{yShape}, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {gert::InfershapeContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::InfershapeContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, -1, -1, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SpaceToDepthTest, space_to_depth_infer_shape_unknown_rank)
{
    gert::StorageShape xShape = {{ge::UNKNOWN_DIM_NUM}, {ge::UNKNOWN_DIM_NUM}};
    gert::StorageShape yShape;
    gert::InfershapeContextPara infershapeContextPara(
        "SpaceToDepth", {{{xShape}, ge::DT_FLOAT, ge::FORMAT_NHWC}}, {{{yShape}, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {gert::InfershapeContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::InfershapeContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))});
    std::vector<std::vector<int64_t>> expectOutputShape = {{ge::UNKNOWN_DIM_NUM}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SpaceToDepthTest, space_to_depth_infer_shape_failed_block_size)
{
    gert::StorageShape xShape = {{1, 2, 2, 4}, {1, 2, 2, 4}};
    gert::InfershapeContextPara infershapeContextPara(
        "SpaceToDepth", {{{xShape}, ge::DT_FLOAT, ge::FORMAT_NHWC}}, {{{}, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {gert::InfershapeContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
         gert::InfershapeContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(SpaceToDepthTest, space_to_depth_infer_shape_failed_data_format)
{
    gert::StorageShape xShape = {{1, 2, 2, 4}, {1, 2, 2, 4}};
    gert::InfershapeContextPara infershapeContextPara(
        "SpaceToDepth", {{{xShape}, ge::DT_FLOAT, ge::FORMAT_NHWC}}, {{{}, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {gert::InfershapeContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::InfershapeContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NGHW"))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(SpaceToDepthTest, space_to_depth_infer_shape_failed_height_not_divisible)
{
    gert::StorageShape xShape = {{1, 3, 2, 4}, {1, 3, 2, 4}};
    gert::InfershapeContextPara infershapeContextPara(
        "SpaceToDepth", {{{xShape}, ge::DT_FLOAT, ge::FORMAT_NHWC}}, {{{}, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {gert::InfershapeContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::InfershapeContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(SpaceToDepthTest, space_to_depth_infer_shape_failed_width_not_divisible)
{
    gert::StorageShape xShape = {{1, 2, 3, 4}, {1, 2, 3, 4}};
    gert::InfershapeContextPara infershapeContextPara(
        "SpaceToDepth", {{{xShape}, ge::DT_FLOAT, ge::FORMAT_NHWC}}, {{{}, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {gert::InfershapeContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::InfershapeContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(SpaceToDepthTest, space_to_depth_infer_shape_failed_rank)
{
    gert::StorageShape xShape = {{1, 2, 2}, {1, 2, 2}};
    gert::InfershapeContextPara infershapeContextPara(
        "SpaceToDepth", {{{xShape}, ge::DT_FLOAT, ge::FORMAT_NHWC}}, {{{}, ge::DT_FLOAT, ge::FORMAT_NHWC}},
        {gert::InfershapeContextPara::OpAttr("block_size", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
         gert::InfershapeContextPara::OpAttr("data_format", Ops::Math::AnyValue::CreateFrom<std::string>("NHWC"))});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}
