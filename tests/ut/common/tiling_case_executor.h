/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_MATH_DEV_TESTS_UT_COMMON_TILING_CASE_EXECUTOR_H
#define OPS_MATH_DEV_TESTS_UT_COMMON_TILING_CASE_EXECUTOR_H

#include <memory>
#include <string>
#include <vector>
#include "base/context_builder/context_holder.h"
#include "platform/platform_infos_def.h"
#include "tiling_context_faker.h"

using namespace std;

const string EMPTY_EXPECT_TILING_DATA = "EMPTY_EXPECT_TILING_DATA";

struct TilingInfo {
    int64_t tilingKey = -1;
    std::vector<int64_t> workspaceSizes;
    std::unique_ptr<uint8_t[]> tilingData;
    size_t tilingDataSize = 0;
    size_t blockNum = 0;
};

/**
 * @brief tiling 上下文及其依赖资源
 *
 * BuildTilingContext() 构造该 bundle，RunTiling() 与后续断言期间必须保持存活。
 * 被 context 以指针引用的对象（platformInfo/tilingData/workspace/tensors）使用
 * unique_ptr/容器持有，保证 bundle 返回时的地址稳定。
 */
struct TilingContextBundle {
    std::unique_ptr<fe::PlatFormInfos> platformInfo;
    std::unique_ptr<uint8_t[]> tilingData;
    std::unique_ptr<uint8_t[]> workspace;
    std::vector<std::unique_ptr<gert::Tensor>> inputTensors;
    std::vector<std::unique_ptr<gert::Tensor>> outputTensors;
    gert::ContextHolder<gert::TilingContext> holder;
    gert::TilingContext* context = nullptr;
};

/**
 * @brief 构造 tiling 执行上下文（张量/属性/平台信息/TilingData/Workspace 等）
 */
TilingContextBundle BuildTilingContext(const gert::TilingContextPara& tilingContextPara, bool withWorkspace = true);

/**
 * @brief 执行 tiling 函数（查找并调用 op 的 tiling 实现）
 */
ge::graphStatus RunTiling(TilingContextBundle& bundle, const gert::TilingContextPara& tilingContextPara);

void ExecuteTestCase(const gert::TilingContextPara& tilingContextPara, ge::graphStatus expectResult = ge::GRAPH_FAILED,
                     uint64_t expectTilingKey = 0, const string& expectTilingData = "",
                     const std::vector<size_t>& expectWorkspaces = {});

void ExecuteTestCase(const gert::TilingContextPara& tilingContextPara, ge::graphStatus expectResult,
                     uint64_t expectTilingKey, const std::vector<size_t>& expectWorkspaces);

void ExecuteTestCaseForEle(const gert::TilingContextPara& tilingContextPara, ge::graphStatus expectResult,
                           bool needCheckTilingKey, uint64_t expectTilingKey, bool needCheckTilingData,
                           const string& expectTilingData, const std::vector<size_t>& expectWorkspaces);

bool ExecuteTiling(const gert::TilingContextPara& tilingContextPara, TilingInfo& tilingInfo);

#endif // OPS_MATH_DEV_TESTS_UT_COMMON_TILING_CASE_EXECUTOR_H
