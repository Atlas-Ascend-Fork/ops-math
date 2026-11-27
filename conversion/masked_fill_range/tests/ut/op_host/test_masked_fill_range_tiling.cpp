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
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class MaskedFillRangeTiling : public testing::Test {
public:
    static void SetUpTestCase() { cout << "MaskedFillRangeTiling SetUp" << endl; }
    static void TearDownTestCase() { cout << "MaskedFillRangeTiling TearDown" << endl; }
};

struct MaskedFillRangeCompileInfo {};

// 属性构造辅助：axis 必填（本分支 ut 框架 API：OpAttr + AnyValue）
static vector<gert::TilingContextPara::OpAttr> MakeAxisAttr(int64_t axis)
{
    return {gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(axis))};
}

// 2D [4,8] fp32，num=2、B=4：Win=1（rounds*batches=8<=64），blockDim=min(core,4)
TEST_F(MaskedFillRangeTiling, basic_2d_fp32)
{
    MaskedFillRangeCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MaskedFillRange",
                                              {
                                                  {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              MakeAxisAttr(-1), &compileInfo, 16, 262144, 4096);
    // TilingKey = Dtype(0=FP32, bit1-0) + Win(1, bit2) = 0b100 = 4
    // （Win=1 门限按窗口表跨度：(2-1)*4+4=8<=64）
    uint64_t expectTilingKey = 4;
    // TilingData: rows cols rounds batches winStride repeat fillMode
    string expectTilingData = "4 8 2 4 4 1 0 ";
    vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 3D [2,7,5] fp32 axis=1（ROW_C 行填充，TIK 语义）：三轴统一折叠
// rows=N*C=14, cols=D=5, repeat=C=7, batches=min(B,N)=2, fillMode=2；
// RowFill 恒 Win=0 -> key=0
TEST_F(MaskedFillRangeTiling, three_d_axis1_row_fill)
{
    MaskedFillRangeCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MaskedFillRange",
                                              {
                                                  {{{2, 7, 5}, {2, 7, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 7, 5}, {2, 7, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              MakeAxisAttr(1), &compileInfo, 16, 262144, 4096);
    uint64_t expectTilingKey = 0; // RowFill(fillMode=2) 恒 Win=0，Dtype=0(FP32)
    string expectTilingData = "14 5 2 2 2 7 2 ";
    vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 3D [2,7,5] fp32 axis=0（ROW_B 整片填充，TIK 语义）：折叠同 axis1 用例，
// 仅 fillMode=1；RowFill 恒 Win=0 -> key=0
TEST_F(MaskedFillRangeTiling, three_d_axis0_row_fill)
{
    MaskedFillRangeCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MaskedFillRange",
                                              {
                                                  {{{2, 7, 5}, {2, 7, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 7, 5}, {2, 7, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              MakeAxisAttr(0), &compileInfo, 16, 262144, 4096);
    uint64_t expectTilingKey = 0; // RowFill(fillMode=1) 恒 Win=0，Dtype=0(FP32)
    string expectTilingData = "14 5 2 2 2 7 1 ";
    vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 3D [16,32,5] fp16 axis=2：rows=512, repeat=32, batches=16；Win=0（fp16），key=Dtype1
TEST_F(MaskedFillRangeTiling, three_d_fp16_last_axis)
{
    MaskedFillRangeCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MaskedFillRange",
                                              {
                                                  {{{16, 32, 5}, {16, 32, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{2, 16}, {2, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 16}, {2, 16}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{16, 32, 5}, {16, 32, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              MakeAxisAttr(2), &compileInfo, 16, 262144, 4096);
    uint64_t expectTilingKey = 1; // Dtype=1(FP16), Win=0
    string expectTilingData = "512 5 2 16 16 32 0 ";
    vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// int8：Dtype=3，Win=0
TEST_F(MaskedFillRangeTiling, int8_key)
{
    MaskedFillRangeCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MaskedFillRange",
                                              {
                                                  {{{8, 64}, {8, 64}}, ge::DT_INT8, ge::FORMAT_ND},
                                                  {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 64}, {8, 64}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              MakeAxisAttr(-1), &compileInfo, 16, 262144, 4096);
    uint64_t expectTilingKey = 3;
    string expectTilingData = "8 64 2 8 8 1 0 ";
    vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// int32：Dtype=2，rounds*batches=16<=64 -> Win=1，key = 2 | (1<<2) = 6
TEST_F(MaskedFillRangeTiling, int32_win1_key)
{
    MaskedFillRangeCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MaskedFillRange",
                                              {
                                                  {{{8, 64}, {8, 64}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{8, 64}, {8, 64}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              MakeAxisAttr(-1), &compileInfo, 16, 262144, 4096);
    uint64_t expectTilingKey = 6;
    string expectTilingData = "8 64 2 8 8 1 0 ";
    vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// fp32 但 rounds*batches>64：退到 Win=0，key = 0。start 第二维 B=40
// （rounds*batches=80 ∈ (64,128]，保留 Win=1 退化测试意图；B=100 的
// rounds*batches=200>128 会被统一常驻元数据容量校验拒绝）
TEST_F(MaskedFillRangeTiling, fp32_large_meta_win0)
{
    MaskedFillRangeCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MaskedFillRange",
                                              {
                                                  {{{100, 64}, {100, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 40}, {2, 40}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 40}, {2, 40}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{100, 64}, {100, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              MakeAxisAttr(-1), &compileInfo, 16, 262144, 4096);
    uint64_t expectTilingKey = 0; // Dtype=0, Win=0（窗口表跨度 (2-1)*40+40=80>64）
    string expectTilingData = "100 64 2 40 40 1 0 ";
    vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 1D：rows=1、batches=min(B,1)
TEST_F(MaskedFillRangeTiling, one_d)
{
    MaskedFillRangeCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MaskedFillRange",
                                              {
                                                  {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4, 1}, {4, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{4, 1}, {4, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              MakeAxisAttr(0), &compileInfo, 16, 262144, 4096);
    uint64_t expectTilingKey = 4; // fp32 + Win=1（窗口表跨度 (4-1)*1+1=4<=64）
    string expectTilingData = "1 1024 4 1 1 1 0 ";
    vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// dtype 不一致（value 为 int32 而 x 为 float）：拒绝
TEST_F(MaskedFillRangeTiling, value_dtype_mismatch)
{
    MaskedFillRangeCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MaskedFillRange",
                                              {
                                                  {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              MakeAxisAttr(-1), &compileInfo, 16, 262144, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// start/end 非同形：拒绝
TEST_F(MaskedFillRangeTiling, start_end_shape_mismatch)
{
    MaskedFillRangeCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MaskedFillRange",
                                              {
                                                  {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 8}, {2, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              MakeAxisAttr(-1), &compileInfo, 16, 262144, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// axis 越界：拒绝
TEST_F(MaskedFillRangeTiling, axis_out_of_range)
{
    MaskedFillRangeCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MaskedFillRange",
                                              {
                                                  {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              MakeAxisAttr(2), &compileInfo, 16, 262144, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 负 axis 归一化（axis=-dim 合法，修复老实现 abs 误杀缺陷）
TEST_F(MaskedFillRangeTiling, negative_axis_full_range)
{
    MaskedFillRangeCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MaskedFillRange",
                                              {
                                                  {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              MakeAxisAttr(-2), &compileInfo, 16, 262144, 4096);
    uint64_t expectTilingKey = 4;
    string expectTilingData = "4 8 2 4 4 1 0 ";
    vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 4D 不支持：拒绝
TEST_F(MaskedFillRangeTiling, four_d_rejected)
{
    MaskedFillRangeCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MaskedFillRange",
                                              {
                                                  {{{2, 2, 2, 2}, {2, 2, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 2, 2, 2}, {2, 2, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              MakeAxisAttr(-1), &compileInfo, 16, 262144, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 空 leading 维（2D (0,8)）：rows==0 进空 tensor 分支 no-op 放行；
// batches 下钳 1（min(4,0)=0），blockDim 下钳 1；Win=1（span=(2-1)*4+1=5<=64）
TEST_F(MaskedFillRangeTiling, empty_leading_dim_2d)
{
    MaskedFillRangeCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MaskedFillRange",
                                              {
                                                  {{{0, 8}, {0, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{0, 8}, {0, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              MakeAxisAttr(-1), &compileInfo, 16, 262144, 4096);
    uint64_t expectTilingKey = 4; // fp32 + Win=1
    string expectTilingData = "0 8 2 1 4 1 0 ";
    vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 空 leading 维（3D (0,3,5)）：rows=0*C=0，no-op 放行；batches 下钳 1
// （min(2,0)=0），repeat=3；Win=1（span=(2-1)*2+1=3<=64）
TEST_F(MaskedFillRangeTiling, empty_leading_dim_3d)
{
    MaskedFillRangeCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MaskedFillRange",
                                              {
                                                  {{{0, 3, 5}, {0, 3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{0, 3, 5}, {0, 3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              MakeAxisAttr(-1), &compileInfo, 16, 262144, 4096);
    uint64_t expectTilingKey = 4; // fp32 + Win=1
    string expectTilingData = "0 5 2 1 2 3 0 ";
    vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 空 middle 维（3D (2,0,5)）：rows=2*0=0，no-op 放行；batches 名义值
// min(2,dim0Batch=2)=2 下钳不变（容量校验空/非空语义一致），repeat=C=0
// 为合法 tilingData 值（kernel 零跳不消费）；Win=1（span=(2-1)*2+2=4<=64）
TEST_F(MaskedFillRangeTiling, empty_mid_dim_3d)
{
    MaskedFillRangeCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MaskedFillRange",
                                              {
                                                  {{{2, 0, 5}, {2, 0, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 0, 5}, {2, 0, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              MakeAxisAttr(-1), &compileInfo, 16, 262144, 4096);
    uint64_t expectTilingKey = 4; // fp32 + Win=1
    string expectTilingData = "0 5 2 2 2 0 0 ";
    vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// rounds*batches 超过常驻元数据容量（2*100=200>128）：拒绝
// （容量校验已前置到 tiling 输出写入之前）
TEST_F(MaskedFillRangeTiling, meta_capacity_rejected)
{
    MaskedFillRangeCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MaskedFillRange",
                                              {
                                                  {{{100, 64}, {100, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 100}, {2, 100}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 100}, {2, 100}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{100, 64}, {100, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              MakeAxisAttr(-1), &compileInfo, 16, 262144, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 1D B>1（start {4,3}）：batches=min(B,1)=1，kernel 每轮仅读 start[i,0]
// （TIK 基准 _fill_mask_info_1d 只读 start_ub[0]，其余窗口列忽略）；
// winStride=3 保留 B 原值作 GM 窗口步长，防 batches 截断后索引错位
TEST_F(MaskedFillRangeTiling, one_d_multi_batch_windows_ignored)
{
    MaskedFillRangeCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MaskedFillRange",
                                              {
                                                  {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4, 3}, {4, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{4, 3}, {4, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              MakeAxisAttr(0), &compileInfo, 16, 262144, 4096);
    uint64_t expectTilingKey = 4; // fp32 + Win=1（窗口表跨度 (4-1)*3+1=10<=64）
    string expectTilingData = "1 1024 4 1 3 1 0 ";
    vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 1D B=100 大跨度（start {2,100}）：batches=1 但窗口表跨度 (2-1)*100+1=101>64，
// 须退到 Win=0（老门限 rounds*batches=2<=64 会误进 Win=1，窗口表越界 _TILE_R）
TEST_F(MaskedFillRangeTiling, one_d_large_b_win0)
{
    MaskedFillRangeCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MaskedFillRange",
                                              {
                                                  {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 100}, {2, 100}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 100}, {2, 100}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              MakeAxisAttr(0), &compileInfo, 16, 262144, 4096);
    uint64_t expectTilingKey = 0; // Dtype=0, Win=0（跨度 101>64）
    string expectTilingData = "1 1024 2 1 100 1 0 ";
    vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 2D B>N（start {2,6}，N=4）：batches=min(6,4)=4 而 winStride=6，窗口 (i,b) 的
// GM 步长必须取 6——若误用 batches=4 作步长会错位读后续窗口列
TEST_F(MaskedFillRangeTiling, two_d_b_gt_rows_stride)
{
    MaskedFillRangeCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("MaskedFillRange",
                                              {
                                                  {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 6}, {2, 6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2, 6}, {2, 6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              MakeAxisAttr(-1), &compileInfo, 16, 262144, 4096);
    uint64_t expectTilingKey = 4; // fp32 + Win=1（跨度 (2-1)*6+4=10<=64）
    string expectTilingData = "4 8 2 4 6 1 0 ";
    vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
