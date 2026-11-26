/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "base/err_msg.h"
#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_kernel/arch35/reciprocal_grad_tiling_data.h"
#include "../../../../op_host/arch35/reciprocal_grad_tiling_arch35.h"

using namespace std;
using namespace ge;
using optiling::ReciprocalGradCompileInfo;

namespace {
// 安装包 error_code.json 未登记 EZ1008，但目标 libnnopbase.so 内嵌了该定义；Math Host UT 入口
// 未加载 nnopbase 注册配置。参照 ops-nn 的夹具做法，仅在本用例内注册同一 ErrMessage/Arglist，
// 使断言读取 ErrorManager 的正式记录，而不是只凭 stdout 证明调用。
constexpr const char kEz1008ErrorMessage[] = R"json({"error_info_list":[
    {"errClass":"Nnopbase Errors","errTitle":"Execution_Error",
     "ErrCode":"EZ1008",
     "ErrMessage":"Failed to execute operator %s. Reason: %s.",
     "Arglist":"opName, reason",
     "suggestion":{"Possible Cause":"N/A","Solution":"1.If the tiling function does not exist, check whether the tiling function is registered successfully. 2.If the tiling function fails to be executed, check the implementation logic of the tiling function."}}
]})json";
} // namespace

class ReciprocalGradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ReciprocalGradTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ReciprocalGradTiling TearDown" << std::endl; }
};

// ─── L0: 各 dtype 基本用例 ───

TEST_F(ReciprocalGradTilingTest, reciprocal_grad_fp32_basic)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 40, 196608, 4096);

    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ReciprocalGradTilingTest, reciprocal_grad_fp16_basic)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 40, 196608, 4096);

    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ReciprocalGradTilingTest, reciprocal_grad_bf16_basic)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{512, 512}, {512, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{512, 512}, {512, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{512, 512}, {512, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 40, 196608, 4096);

    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// ─── L0: 1D / 4D shape ───

TEST_F(ReciprocalGradTilingTest, reciprocal_grad_fp32_1d)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 40, 196608, 4096);

    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ReciprocalGradTilingTest, reciprocal_grad_fp32_4d)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{2, 4, 8, 16}, {2, 4, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 4, 8, 16}, {2, 4, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 4, 8, 16}, {2, 4, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 40, 196608, 4096);

    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ReciprocalGradTilingTest, reciprocal_grad_fp16_1d)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{2048}, {2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{2048}, {2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2048}, {2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 40, 196608, 4096);

    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ReciprocalGradTilingTest, reciprocal_grad_bf16_4d)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{2, 4, 8, 16}, {2, 4, 8, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{2, 4, 8, 16}, {2, 4, 8, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 4, 8, 16}, {2, 4, 8, 16}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 40, 196608, 4096);

    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// ─── L0: 不同 core/UB 配置 ───

TEST_F(ReciprocalGradTilingTest, reciprocal_grad_fp32_different_cores)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 16;
    compileInfo.ubSize = 196608;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{512, 512}, {512, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{512, 512}, {512, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{512, 512}, {512, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 16, 196608, 4096);

    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ReciprocalGradTilingTest, reciprocal_grad_fp32_small_ub)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 8;
    compileInfo.ubSize = 32768;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{256, 256}, {256, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{256, 256}, {256, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{256, 256}, {256, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 8, 32768, 4096);

    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// ─── L1: 大 shape ───

TEST_F(ReciprocalGradTilingTest, reciprocal_grad_fp16_large)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 40, 196608, 4096);

    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(ReciprocalGradTilingTest, reciprocal_grad_bf16_large)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 40, 196608, 4096);

    uint64_t expectTilingKey = 1;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// ─── L1: 反例测试 ───

TEST_F(ReciprocalGradTilingTest, reciprocal_grad_shape_mismatch_y_dy)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{512, 512}, {512, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 40, 196608, 4096);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ReciprocalGradTilingTest, reciprocal_grad_shape_mismatch_y_z)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{512, 512}, {512, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 40, 196608, 4096);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ReciprocalGradTilingTest, reciprocal_grad_dtype_mismatch_y_dy)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 40, 196608, 4096);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(ReciprocalGradTilingTest, reciprocal_grad_unsupported_dtype)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 40, 196608, 4096);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// DoTiling 失败边界：合法输入下 opbase 内部 tiling 失败时，算子层用正式 EZ1008 上报并保留原返回。
// ubSize=0 使 ElewiseTiling 的 ubFormer 为 0，稳定触发该分支。EZ1008 定义来自目标 libnnopbase.so，
// 本用例内注册后由 ErrorManager 正式记录，断言唯一正式码、参数与不再追加的包装 EZ9999。
TEST_F(ReciprocalGradTilingTest, reciprocal_grad_dotiling_failure_reports_ez1008)
{
    ReciprocalGradCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;

    gert::TilingContextPara tilingContextPara("ReciprocalGrad",
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 40, 0, 4096);

    ASSERT_EQ(error_message::RegisterFormatErrorMessage(kEz1008ErrorMessage, sizeof(kEz1008ErrorMessage) - 1), 0);

    auto& errors = ErrorManager::GetInstance();
    ASSERT_EQ(errors.Init(), 0);
    (void)errors.GetRawErrorMessages();
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
    const auto reports = errors.GetRawErrorMessages();

    size_t ez1008Count = 0;
    size_t ez9999Count = 0;
    size_t ez1008Index = reports.size();
    for (size_t i = 0; i < reports.size(); ++i) {
        const auto& report = reports[i];
        if (report.error_id == "EZ1008") {
            ++ez1008Count;
            ez1008Index = i;
        } else if (report.error_id == "EZ9999") {
            ++ez9999Count;
        }
        // 旧 OP_LOGE 包装会追加带此文本的 EZ9999；改用 WITHOUT_REPORT 后不得再出现。
        EXPECT_EQ(report.error_message.find("ReciprocalGrad: ElewiseBaseTiling DoTiling failed"), std::string::npos)
            << report.error_id << ": " << report.error_message;
    }

    ASSERT_EQ(ez1008Count, 1U);
    ASSERT_LT(ez1008Index, reports.size());
    const auto& ez1008 = reports[ez1008Index];
    const auto opName = ez1008.args_map.find("opName");
    ASSERT_NE(opName, ez1008.args_map.end());
    EXPECT_EQ(opName->second, "ReciprocalGrad");
    const auto reason = ez1008.args_map.find("reason");
    ASSERT_NE(reason, ez1008.args_map.end());
    EXPECT_NE(reason->second.find("stage=ReciprocalGradTiling::RunTiling"), std::string::npos) << reason->second;
    // ge::graphStatus 为 uint32_t，GRAPH_FAILED 在此记为 4294967295，证明 reason 保留了原始返回。
    EXPECT_NE(reason->second.find("ret=4294967295"), std::string::npos) << reason->second;
    EXPECT_NE(reason->second.find("outputDtype=DT_FLOAT"), std::string::npos) << reason->second;

    // 共享 opbase Elewise 基线的 3 条 EZ9999（ubFormer==0 首报 + 两层 DoTiling 包装）照实保留，
    // 不是当前算子追加；本算子唯一正式上报为上方的 EZ1008。
    EXPECT_EQ(ez9999Count, 3U);
}
