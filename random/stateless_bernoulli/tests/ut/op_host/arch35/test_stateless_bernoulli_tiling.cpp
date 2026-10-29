/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_stateless_bernoulli_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include <vector>
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/stateless_bernoulli_tiling_arch35.h"

class StatelessBernoulliTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "StatelessBernoulliTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "StatelessBernoulliTest TearDown" << std::endl; }
};

// 期望 tilingData 为 RandomUnifiedSimtTilingDataStruct 按 int64 逐词内存序列化（共 68 词）：
// [0..5] usedCoreNum/outputSize/seed/offset/ubSize/extraInt64Param1（场景标志不落盘, 恒 0），
// [6] prob+extraF32 打包，[7] from，[8] range，[9] fromFp32+toFp32 打包，[10] splitBlockCount，
// [11..50] splitBlocks[8]（numel/gmOffset/grid/totalThreads/kernelOffset），
// [51] probDims（int32+padding），[52..59] probSizes，[60..67] probStrides。
// 标量语义（probMode=1）不填写 probDims/probSizes/probStrides（kernel 不消费，保持 0）。
// workspace 为 0（prob 广播 stride 与快除参数改为 kernel 内 UB 传递，不走 workspace）。

// 用例1：缺省 probMode（不传属性）→ 张量语义，prob numel==1 也不再推断为标量
TEST_F(StatelessBernoulliTiling, stateless_bernoulli_tensor_default)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 253952};
    vector<int64_t> shapeValue = {1};
    vector<int64_t> seedValue = {2};
    vector<int64_t> offsetValue = {8};

    gert::TilingContextPara tilingContextPara("StatelessBernoulli",
                                              {
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, seedValue.data()},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetValue.data()},
                                              },
                                              {{{{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND}},
                                              {{"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)}}, &compileInfo);

    uint64_t expectTilingKey = 101; // 缺省张量语义 + prob 同 shape（dense）→ 101
    // probDims=1([51])，probSizes=[1,0..]([52])，probStrides=[1,0..]([60])
    string expectTilingData = "1 1 2 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
                              "0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 用例2：probMode=1（标量语义）→ splitBlocks 计算下发，T=256*min(ceil(n/256),78*8)
TEST_F(StatelessBernoulliTiling, stateless_bernoulli_scalar_probmode)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 253952};
    vector<int64_t> shapeValue = {1};
    vector<int64_t> seedValue = {2};
    vector<int64_t> offsetValue = {8};

    gert::TilingContextPara tilingContextPara("StatelessBernoulli",
                                              {
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, seedValue.data()},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetValue.data()},
                                              },
                                              {{{{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND}},
                                              {{"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
                                               {"probMode", Ops::Math::AnyValue::CreateFrom<int64_t>(1)}},
                                              &compileInfo);

    uint64_t expectTilingKey = 100;
    // splitBlocks[0] = {numel=1, gmOffset=0, grid=1, totalThreads=256, kernelOffset=8(单块无双重消费)}；
    // 标量语义不填写 probDims/probSizes/probStrides（[51..67] 全 0）
    string expectTilingData = "1 1 2 8 0 0 0 0 0 0 1 1 0 1 256 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
                              "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 用例3：prob 广播 stride（prob=[8] → out=[4,8]，右对齐，前导维 stride=0），缺省张量语义
TEST_F(StatelessBernoulliTiling, stateless_bernoulli_tensor_broadcast)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 253952};
    vector<int64_t> shapeValue = {4, 8};
    vector<int64_t> seedValue = {2};
    vector<int64_t> offsetValue = {8};

    gert::TilingContextPara tilingContextPara("StatelessBernoulli",
                                              {
                                                  {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
                                                  {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, seedValue.data()},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetValue.data()},
                                              },
                                              {{{{4, 8}, {4, 8}}, ge::DT_UINT8, ge::FORMAT_ND}},
                                              {{"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)}}, &compileInfo);

    uint64_t expectTilingKey = 102; // 张量语义 + prob[8]→out[4,8] 广播 → 102
    // probDims=2([51])，probSizes=[4,8,0..]([52,53])，probStrides=[0,1,0..]（[61]=1 尾维连续，前导广播 stride=0）
    string expectTilingData = "1 32 2 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
                              "0 0 0 0 0 0 0 0 2 4 8 0 0 0 0 0 0 0 1 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 用例4：probMode=1 + shape 输入为空 tensor（输出 rank 0 标量，outputSize=1）→
//        标量语义不做广播校验（prob rank 1 > 输出 rank 0 不再误报），正常下发 splitBlocks 单块
TEST_F(StatelessBernoulliTiling, stateless_bernoulli_scalar_empty_shape_tensor)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 253952};
    vector<int64_t> shapeValue = {}; // shape 输入 0 元素，输出推为 rank 0
    vector<int64_t> seedValue = {2};
    vector<int64_t> offsetValue = {8};
    int64_t shapeDummy = 0; // 空 const tensor 的 data 指针占位（0 元素不读取内容，仅要求非空）

    gert::TilingContextPara tilingContextPara("StatelessBernoulli",
                                              {
                                                  {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, &shapeDummy},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, seedValue.data()},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetValue.data()},
                                              },
                                              {{{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND}},
                                              {{"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
                                               {"probMode", Ops::Math::AnyValue::CreateFrom<int64_t>(1)}},
                                              &compileInfo);

    uint64_t expectTilingKey = 100;
    // outputSize=1（rank 0 的 shapeSize 为 1），splitBlocks[0] = {numel=1, gmOffset=0, grid=1,
    // totalThreads=256, kernelOffset=8}；标量语义不填写 prob 广播字段（[51..67] 全 0）
    string expectTilingData = "1 1 2 8 0 0 0 0 0 0 1 1 0 1 256 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
                              "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 用例5：probMode=1 + prob 非单元素张量 → 报错（标量语义下单元素约束校验）
TEST_F(StatelessBernoulliTiling, stateless_bernoulli_scalar_multi_elements_prob)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 253952};
    vector<int64_t> shapeValue = {4};
    vector<int64_t> seedValue = {2};
    vector<int64_t> offsetValue = {8};

    gert::TilingContextPara tilingContextPara("StatelessBernoulli",
                                              {
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
                                                  {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, seedValue.data()},
                                                  {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, offsetValue.data()},
                                              },
                                              {{{{4}, {4}}, ge::DT_UINT8, ge::FORMAT_ND}},
                                              {{"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
                                               {"probMode", Ops::Math::AnyValue::CreateFrom<int64_t>(1)}},
                                              &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
