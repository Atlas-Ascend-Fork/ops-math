/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_topk_infershape.cpp
 * \brief TopK InferShape and InferDataType UT.
 *        输出 shape = 输入 shape 的 dim 轴替换为 k；
 *        values.dtype = x.dtype，indices.dtype = k.dtype。
 */

#include <iostream>
#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

class TopKInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TopKInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TopKInfershape TearDown" << std::endl; }
};

// 属性按注册顺序传入：sorted(0)、largest(1)、dim(2)，infershape 按索引 2 取 dim
static std::vector<gert::InfershapeContextPara::OpAttr> MakeTopKAttrs(int64_t dim)
{
    return {gert::InfershapeContextPara::OpAttr("sorted", Ops::Math::AnyValue::CreateFrom<bool>(true)),
            gert::InfershapeContextPara::OpAttr("largest", Ops::Math::AnyValue::CreateFrom<bool>(true)),
            gert::InfershapeContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(dim))};
}

static void ExecuteTopKShapeCase(const std::initializer_list<int64_t>& xShape, int32_t k, int64_t dim,
                                 ge::graphStatus expect, const std::vector<std::vector<int64_t>>& expectOutputShape)
{
    gert::StorageShape xStorageShape(xShape, xShape);
    gert::InfershapeContextPara infershapeContextPara("TopK",
                                                      {
                                                          {xStorageShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &k},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      MakeTopKAttrs(dim));
    ExecuteTestCase(infershapeContextPara, expect, expectOutputShape);
}

TEST_F(TopKInfershape, topk_infer_shape_2d_dim_last)
{
    ExecuteTopKShapeCase({4, 6}, 2, -1, ge::GRAPH_SUCCESS, {{4, 2}, {4, 2}});
}

TEST_F(TopKInfershape, topk_infer_shape_2d_dim0)
{
    ExecuteTopKShapeCase({4, 6}, 3, 0, ge::GRAPH_SUCCESS, {{3, 6}, {3, 6}});
}

TEST_F(TopKInfershape, topk_infer_shape_3d_neg_dim)
{
    ExecuteTopKShapeCase({2, 3, 4}, 1, -2, ge::GRAPH_SUCCESS, {{2, 1, 4}, {2, 1, 4}});
}

TEST_F(TopKInfershape, topk_infer_shape_k_equals_dim_size)
{
    ExecuteTopKShapeCase({4}, 4, 0, ge::GRAPH_SUCCESS, {{4}, {4}});
}

TEST_F(TopKInfershape, topk_infer_shape_k_zero) { ExecuteTopKShapeCase({4}, 0, 0, ge::GRAPH_SUCCESS, {{0}, {0}}); }

TEST_F(TopKInfershape, topk_infer_shape_dynamic_dim)
{
    ExecuteTopKShapeCase({-1, -1}, 2, -1, ge::GRAPH_SUCCESS, {{-1, 2}, {-1, 2}});
}

TEST_F(TopKInfershape, topk_infer_shape_k_exceeds_dim_fail) { ExecuteTopKShapeCase({4}, 5, 0, ge::GRAPH_FAILED, {}); }

TEST_F(TopKInfershape, topk_infer_shape_negative_k_fail) { ExecuteTopKShapeCase({4}, -1, 0, ge::GRAPH_FAILED, {}); }

TEST_F(TopKInfershape, topk_infer_shape_dim_out_of_range_fail)
{
    ExecuteTopKShapeCase({4, 6}, 2, 2, ge::GRAPH_FAILED, {});
}

TEST_F(TopKInfershape, topk_infer_shape_neg_dim_out_of_range_fail)
{
    ExecuteTopKShapeCase({4, 6}, 2, -3, ge::GRAPH_FAILED, {});
}

// 编译态 k 槽位缺失（tensor 为空）→ 降级不报错
TEST_F(TopKInfershape, topk_k_tensor_null_degrades_to_unknown)
{
    gert::InfershapeContextPara infershapeContextPara("TopK",
                                                      {
                                                          {{{4, 6}, {4, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      MakeTopKAttrs(-1), {}, {}, {1});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {{4, -1}, {4, -1}});
}

// 编译态 k 非常量（数据不可读）→ 输出 dim 降级 -1
TEST_F(TopKInfershape, topk_k_data_null_degrades_to_unknown)
{
    gert::InfershapeContextPara infershapeContextPara("TopK",
                                                      {
                                                          {{{4, 6}, {4, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, nullptr},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      MakeTopKAttrs(-1));
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {{4, -1}, {4, -1}});
}

// 编译态 k 非常量 + dim=0 → 仅对应轴降级 -1，其余轴保留
TEST_F(TopKInfershape, topk_k_data_null_dim0_degrades_to_unknown)
{
    gert::InfershapeContextPara infershapeContextPara("TopK",
                                                      {
                                                          {{{4, 6}, {4, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, nullptr},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      MakeTopKAttrs(0));
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, {{-1, 6}, {-1, 6}});
}

TEST_F(TopKInfershape, topk_infer_datatype)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("TopK");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    const std::vector<ge::DataType> xDtypes = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    for (const auto& xDtype : xDtypes) {
        gert::OpInferDataTypeContextBuilder builder;
        builder.OpType("TopK").OpName("TopK");
        builder.IONum(2, 2);
        builder.InputTensorDesc(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.InputTensorDesc(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND);
        auto contextHolder = builder.Build();
        auto* context = contextHolder.GetContext();
        ASSERT_NE(context, nullptr);

        auto ret = opImpl->infer_datatype(context);
        EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
        EXPECT_EQ(context->GetOutputDataType(0), xDtype);
        EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT32);
    }
}
