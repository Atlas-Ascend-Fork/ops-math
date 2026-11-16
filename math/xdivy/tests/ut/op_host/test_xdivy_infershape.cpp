/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

using namespace ge;
using namespace gert;

class XdivyInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "XdivyInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "XdivyInfershape TearDown" << std::endl; }
};

static void RunXdivyCase(const std::vector<std::vector<int64_t>>& inShapes, const std::vector<ge::DataType>& inDtypes,
                         const std::vector<std::vector<int64_t>>& expectOut)
{
    std::vector<InfershapeContextPara::TensorDescription> inputs;
    for (size_t i = 0; i < inShapes.size(); i++) {
        gert::StorageShape s;
        for (int64_t d : inShapes[i]) {
            s.MutableOriginShape().AppendDim(d);
            s.MutableStorageShape().AppendDim(d);
        }
        inputs.emplace_back(s, inDtypes[i], ge::FORMAT_ND);
    }
    std::vector<InfershapeContextPara::TensorDescription> outputs;
    for (size_t i = 0; i < expectOut.size(); i++) {
        outputs.emplace_back(gert::StorageShape(), inDtypes[0], ge::FORMAT_ND);
    }
    InfershapeContextPara para("Xdivy", inputs, outputs);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOut);
}

// 常规场景
TEST_F(XdivyInfershape, xdivy_infershape_test_1)
{
    RunXdivyCase({{2, 3},
                  {
                      3,
                  }},
                 {ge::DT_FLOAT, ge::DT_FLOAT}, {{2, 3}});
}

// 部分未知shape（-1）场景：标准广播，未知维取对端已知维
TEST_F(XdivyInfershape, xdivy_infershape_partial_unknown)
{
    RunXdivyCase({{2, -1},
                  {
                      3,
                  }},
                 {ge::DT_FLOAT, ge::DT_FLOAT}, {{2, 3}});
}

// 全未知shape（-1）场景
TEST_F(XdivyInfershape, xdivy_infershape_all_unknown_dim)
{
    RunXdivyCase({{-1, -1}, {-1, -1}}, {ge::DT_FLOAT, ge::DT_FLOAT}, {{-1, -1}});
}

// 未知rank（-2）场景
TEST_F(XdivyInfershape, xdivy_infershape_unknown_rank)
{
    RunXdivyCase({{
                      -2,
                  },
                  {
                      -2,
                  }},
                 {ge::DT_FLOAT, ge::DT_FLOAT},
                 {{
                     -2,
                 }});
}
