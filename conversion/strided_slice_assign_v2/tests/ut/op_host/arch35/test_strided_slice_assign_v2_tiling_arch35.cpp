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
 * \file test_strided_slice_assign_v2_tiling_arch35.cpp
 * \brief A5 (arch35) StridedSliceAssignV2 Tiling UT：S0 路由判据 + T1/T2/T3/T4 四模板切分
 *
 * 平台口径：Ascend950DT_9582（A5 / arch35 / NpuArch 3510）
 *   - AIV 核数 = 64，UB = 248KB = 253952 B
 *   - 与 op_host/arch35/strided_slice_assign_v2_tiling_arch35.h 中的常量一一对应（判据顺序：
 *     ① 混合正负 → T1；③ 极稀疏（窗口内有效元素 ≤ 1）→ T1；
 *     ④ 连续段（段长 ≥ 32B，n'≥3 时 256B；段长 32B 整倍数；段首 32B 对齐）→ T2；
 *     ② 小 shape（sliceElemNum ≤ 1024）→ T1；其余 → T3；var/slice 为空 → T4）。
 *
 * 切分结果核对：T1/T3 关键用例经 ExecuteTiling 断言 TilingData 字段（核间切分 / ubFactor /
 * 约简形状），其余只断言 tilingKey 与 blockDim；全字段以算子自身的 OP_LOGI（原始输入 /
 * 判据 / PrintTilingInfo）输出到 UT 日志，直接查日志核对即可。
 */

#include <iostream>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/strided_slice_assign_v2_tiling_arch35.h"

using namespace std;
using namespace ge;

namespace {

constexpr uint64_t SSA_UT_CORE_NUM = 64;    // A5 AIV 核数
constexpr uint64_t SSA_UT_UB_SIZE = 253952; // A5 UB 248KB
constexpr size_t SSA_UT_WORKSPACE = 0;      // 无 GM 中间结果，workspace 分配 0

optiling::StridedSliceAssignV2CompileInfo MakeCompileInfo()
{
    optiling::StridedSliceAssignV2CompileInfo info;
    info.totalCoreNum = static_cast<int32_t>(SSA_UT_CORE_NUM);
    info.ubSize = static_cast<int64_t>(SSA_UT_UB_SIZE);
    info.ubUsable = static_cast<int64_t>(SSA_UT_UB_SIZE);
    return info;
}

gert::StorageShape ShapeOf(const std::vector<int64_t>& dims)
{
    gert::StorageShape storageShape;
    for (int64_t dim : dims) {
        storageShape.MutableOriginShape().AppendDim(dim);
        storageShape.MutableStorageShape().AppendDim(dim);
    }
    return storageShape;
}

// begin/end/strides/axes 是「1 维、长度为 N」的常量张量，shape 必须是 {N} 而不是 N 个元素值
gert::StorageShape VecShape(size_t len) { return ShapeOf({static_cast<int64_t>(len)}); }

// ---------------------------------------------------------------------------
// 用例封装：持有 begin/end/strides/axes 的常量数据，保证 constValue_ 指针生命周期
// ---------------------------------------------------------------------------
class SsaTilingCase {
public:
    SsaTilingCase(std::vector<int64_t> varShape, std::vector<int64_t> valueShape, std::vector<int64_t> begin,
                  std::vector<int64_t> end, std::vector<int64_t> strides, std::vector<int64_t> axes = {},
                  ge::DataType dtype = ge::DT_FLOAT, bool constInputs = true, int64_t ubUsable = SSA_UT_UB_SIZE)
        : varShape_(std::move(varShape)),
          valueShape_(std::move(valueShape)),
          begin_(std::move(begin)),
          end_(std::move(end)),
          strides_(std::move(strides)),
          axes_(std::move(axes)),
          dtype_(dtype),
          constInputs_(constInputs)
    {
        compileInfo_ = MakeCompileInfo();
        // ubUsable 可注入非标准 UB 预算（T1 ubFactor 取整 / 下限用例）；tiling 侧优先读该字段
        compileInfo_.ubUsable = ubUsable;
        compileInfo_.ubSize = ubUsable;
        std::vector<gert::TilingContextPara::TensorDescription> inputs = {
            {ShapeOf(varShape_), dtype_, ge::FORMAT_ND},
            {ShapeOf(valueShape_), dtype_, ge::FORMAT_ND},
            {VecShape(begin_.size()), ge::DT_INT64, ge::FORMAT_ND, constInputs_, begin_.data()},
            {VecShape(end_.size()), ge::DT_INT64, ge::FORMAT_ND, constInputs_, end_.data()},
            {VecShape(strides_.size()), ge::DT_INT64, ge::FORMAT_ND, constInputs_, strides_.data()},
        };
        if (!axes_.empty()) {
            inputs.push_back({VecShape(axes_.size()), ge::DT_INT64, ge::FORMAT_ND, constInputs_, axes_.data()});
        }
        std::vector<gert::TilingContextPara::TensorDescription> outputs = {
            {ShapeOf(varShape_), dtype_, ge::FORMAT_ND},
        };
        para_ = std::make_unique<gert::TilingContextPara>("StridedSliceAssignV2", inputs, outputs, &compileInfo_,
                                                          SSA_UT_CORE_NUM, static_cast<uint64_t>(ubUsable));
    }

    const gert::TilingContextPara& Para() const { return *para_; }

private:
    std::vector<int64_t> varShape_;
    std::vector<int64_t> valueShape_;
    std::vector<int64_t> begin_;
    std::vector<int64_t> end_;
    std::vector<int64_t> strides_;
    std::vector<int64_t> axes_;
    ge::DataType dtype_;
    bool constInputs_;
    optiling::StridedSliceAssignV2CompileInfo compileInfo_;
    std::unique_ptr<gert::TilingContextPara> para_;
};

// ---------------------------------------------------------------------------
// T3 切分结果断言：4 参 ExecuteTestCase 只核对 tilingKey，这里经 ExecuteTiling 再核对
// ubAxis / ubFactor / tileElemNum / ubTotalCount 与 blockDim（择轴回归用）。
// ---------------------------------------------------------------------------
void ExpectScatterTiling(const gert::TilingContextPara& para, uint64_t expectKey, int64_t expectUbAxis,
                         int64_t expectUbFactor, int64_t expectTileElemNum, int64_t expectTotal, size_t expectBlockNum)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, expectKey);
    EXPECT_EQ(info.blockNum, expectBlockNum);
    ASSERT_GE(info.tilingDataSize, sizeof(StridedSliceAssignV2ScatterTilingData));
    const auto* td = reinterpret_cast<const StridedSliceAssignV2ScatterTilingData*>(info.tilingData.get());
    EXPECT_EQ(static_cast<int64_t>(td->ubAxis), expectUbAxis);
    EXPECT_EQ(static_cast<int64_t>(td->ubFactor), expectUbFactor);
    EXPECT_EQ(static_cast<int64_t>(td->tileElemNum), expectTileElemNum);
    EXPECT_EQ(static_cast<int64_t>(td->ubTotalCount), expectTotal);
    EXPECT_LE(static_cast<int64_t>(td->rows), optiling::SSA_MAX_DMA_BLOCK_COUNT); // blockCount = rows
}

// ---------------------------------------------------------------------------
// T1 切分结果断言：核对 tilingKey / blockDim / 核间三件套（sliceElemNum 不变，直接对
// elemPerCore / lastCoreElemNum）/ varBase / 约简形状（S'/T'，按 n' 长度逐维比对）/
// ubFactor（预算 (ubUsable−32768−512)/(2·elemSize) 向 16 元素粒度向下取整、下限 16；32768 为 T1 SIMT DCache 预留）。
// ---------------------------------------------------------------------------
void ExpectHybridTiling(const gert::TilingContextPara& para, uint64_t expectKey, size_t expectBlockNum,
                        int64_t expectElemPerCore, int64_t expectLastCoreElemNum, int64_t expectVarBase,
                        int64_t expectUbFactor, const std::vector<int64_t>& expectSliceShape,
                        const std::vector<int64_t>& expectAxisStep)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, expectKey);
    EXPECT_EQ(info.blockNum, expectBlockNum);
    ASSERT_GE(info.tilingDataSize, sizeof(StridedSliceAssignV2HybridTilingData));
    const auto* td = reinterpret_cast<const StridedSliceAssignV2HybridTilingData*>(info.tilingData.get());
    EXPECT_EQ(td->dimNum, static_cast<int64_t>(expectSliceShape.size()));
    EXPECT_EQ(td->sliceElemNum, expectElemPerCore * static_cast<int64_t>(expectBlockNum - 1) + expectLastCoreElemNum);
    EXPECT_EQ(static_cast<int64_t>(td->blockDim), static_cast<int64_t>(expectBlockNum));
    EXPECT_EQ(td->elemPerCore, expectElemPerCore);
    EXPECT_EQ(td->lastCoreElemNum, expectLastCoreElemNum);
    EXPECT_EQ(td->varBase, expectVarBase);
    EXPECT_EQ(static_cast<int64_t>(td->ubFactor), expectUbFactor);
    for (size_t d = 0; d < expectSliceShape.size(); d++) {
        EXPECT_EQ(td->sliceShape[d], expectSliceShape[d]);
        EXPECT_EQ(td->axisStep[d], expectAxisStep[d]);
    }
}

} // namespace

class StridedSliceAssignV2TilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "StridedSliceAssignV2TilingArch35 SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "StridedSliceAssignV2TilingArch35 TearDown" << std::endl; }
};

// ===========================================================================
// T3 Scatter（TPL 2）：正负不混合、非小 shape、非稀疏、连续段不足 cacheline
// ===========================================================================

// 1. 与 GE 例程同形：var{4,1024} ← var[1:4, 0:1024:2]，合轴后 n'=1、S'=[1536]、T'=[2]
//    预期：T3 / key=2；单任务即可容纳 ⇒ 只铺 1 核
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Scatter_StridedInnerDim)
{
    SsaTilingCase c({4, 1024}, {3, 512}, {1, 0}, {4, 1024}, {1, 2}, {0, 1});
    ExecuteTestCase(c.Para(), ge::GRAPH_SUCCESS, 2, std::vector<size_t>{SSA_UT_WORKSPACE});
}

// 2. 二维约简形状（不可合轴）：var{256,2048}, end 取 1025 ⇒ S'=[256,129]、T'=[2048,8]
//    预期：T3 / key=2；PickAxis 选内轴后由 Balance 内移到外轴并铺满 64 核
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Scatter_TwoDimSplit)
{
    SsaTilingCase c({256, 2048}, {256, 129}, {0, 0}, {256, 1025}, {1, 8}, {0, 1});
    ExecuteTestCase(c.Para(), ge::GRAPH_SUCCESS, 2, std::vector<size_t>{SSA_UT_WORKSPACE});
}

// 3. 全负方向 + varBase 未 32B 对齐 ⇒ 连判据④都不满足，落到 T3
//    var{64,256} 反向整取：合轴后 n'=1、S'=[16384]、T'=[-1]、varBase=16383
//    预期：T3 / key=2（SSA_DIR_ALL_NEGATIVE，方向已入 tiling data，不再入 key）
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Scatter_NegativeUnaligned)
{
    SsaTilingCase c({64, 256}, {64, 256}, {63, 255}, {-65, -257}, {-1, -1}, {0, 1});
    ExecuteTestCase(c.Para(), ge::GRAPH_SUCCESS, 2, std::vector<size_t>{SSA_UT_WORKSPACE});
}

// 4. 全负方向 n'=1 且单任务超 UB ⇒ 走 SplitF 闭式劈半
//    var{2,16384} 反向整取：S'=[32768]、T'=[-1]、varBase=32767 ⇒ 未对齐 ⇒ T3
//    Cost(0,32768) ≈ 655360 > ubBudget(252384) ⇒ f = SplitF() = 10928，total=3 ⇒ 3 核
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Scatter_NegativeSplitF)
{
    SsaTilingCase c({2, 16384}, {2, 16384}, {1, 16383}, {-3, -16385}, {-1, -1}, {0, 1});
    ExecuteTestCase(c.Para(), ge::GRAPH_SUCCESS, 2, std::vector<size_t>{SSA_UT_WORKSPACE});
}

// 5. n'=3 且切分轴停在 a=n'-2（k=2）⇒ 首次覆盖 rowStepVar ≠ 0 的行内迭代路径
//    var{8,288,2048}, strides{1,2,8}: 合轴后 S'=[8,72,256]、T'=[589824,4096,8]
//    判据：validNum=ceil(32/8)=4>1、contigRun=1 ⇒ T3
//    PickAxis: a=1、f=MaxF(1)=12 ⇒ total=8·ceil(72/12)=48、perCore=1、usedCore=48 = 0.75·64
//    ⇒ Balance 的 while 不进入（usedCore 已达 ratio），a 保持 n'-2 ⇒ rowStepVar[1]=4096
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Scatter_InnerRowIter)
{
    SsaTilingCase c({8, 288, 2048}, {8, 72, 256}, {0, 0, 0}, {8, 144, 2048}, {1, 2, 8}, {0, 1, 2});
    ExecuteTestCase(c.Para(), ge::GRAPH_SUCCESS, 2, std::vector<size_t>{SSA_UT_WORKSPACE});
}

// 6. n'=1 但任务数 > 核数且非整除 ⇒ 覆盖 Main/Tail 余量（ubPerCount·usedCore > ubTotalCount）
//    1 维 var{820041}、begin={1} ⇒ S'=[820040]、T'=[1]、varBase=1
//    判据：innerStep=1 但 (varBase·4)%32=4≠0 ⇒ 判据④落空 ⇒ T3（key 2，非空对齐路径）
//    SplitF: fub=12616 ⇒ m=ceil(820040/12616)=65 ⇒ f=12616 ⇒ total=65
//    ⇒ perCore=ceil(65/64)=2、usedCore=ceil(65/2)=33 ⇒ 33 核 × 2 任务 = 66 槽装 65 任务
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Scatter_MainTailRemainder)
{
    SsaTilingCase c({820041}, {820040}, {1}, {820041}, {1}, {0});
    ExecuteTestCase(c.Para(), ge::GRAPH_SUCCESS, 2, std::vector<size_t>{SSA_UT_WORKSPACE});
}

// ===========================================================================
// T3 择轴回归：a = n-2 时 rows = f（与 S[n-2] 无关）⇒ S[n-2] 超 blockCount 上限也照样可切，
// 只是 f 被 SSA_MAX_DMA_BLOCK_COUNT 封顶；a ≤ n-3 才因 rows = f·S[n-2] 被 MaxF 拒绝。
// ===========================================================================

// 7. 2 维：var{4096,320} 内维步长 6 ⇒ S'=[4096,54]、T'=[320,6]（320 ≠ 54·6 ⇒ 不满足合轴判据）
//    修复前：PickAxis 一见 S[0]=4096 > 4095 就 break ⇒ a=1、f=54（cols=54、rows=1）
//            ⇒ tile=54 元素、total=4096、blockDim=64
//    修复后：a=0、f=MaxF(0)=40（perF=54·20+4·1·320·4=6200，252128/6200=40；rows=f=40 ≤ 4095）
//            ⇒ tile=40·54=2160 元素、total=ceil(4096/40)=103、blockDim=52
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Scatter_OuterAxisBeyondBlockCount)
{
    SsaTilingCase c({4096, 320}, {4096, 54}, {0, 0}, {4096, 320}, {1, 6}, {0, 1});
    ExecuteTestCase(c.Para(), ge::GRAPH_SUCCESS, 2, std::vector<size_t>{SSA_UT_WORKSPACE});
    ExpectScatterTiling(c.Para(), /*key=*/2, /*ubAxis=*/0, /*ubFactor=*/40, /*tileElemNum=*/40 * 54,
                        /*total=*/103, /*blockNum=*/52);
}

// 8. 3 维：var{2048,5000,16}、end={2048,4999,16}、步长={1,1,3}
//    ⇒ S'=[2048,4999,6]、T'=[80000,16,3]（16 ≠ 6·3 且 80000 ≠ 4999·16 ⇒ 三维不合并）
//    修复前：S[1]=4999 > 4095 ⇒ 连 a=1 都不试 ⇒ a=2、f=6 ⇒ tile=6 元素、total=10237952、blockDim=64
//    修复后：a=1、f=MaxF(1)=670（perF=6·20+4·1·16·4=376，252128/376=670；rows=f ≤ 4095）
//            ⇒ tile=670·6=4020 元素、total=2048·ceil(4999/670)=16384、blockDim=64
//    a=0 因 S[1]=4999 > 4095 落在 blockCount（DmaLevelsFit 拒绝）⇒ MaxF(0)=0，循环止步
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Scatter_MidAxisBeyondBlockCount)
{
    SsaTilingCase c({2048, 5000, 16}, {2048, 4999, 6}, {0, 0, 0}, {2048, 4999, 16}, {1, 1, 3}, {0, 1, 2});
    ExecuteTestCase(c.Para(), ge::GRAPH_SUCCESS, 2, std::vector<size_t>{SSA_UT_WORKSPACE});
    ExpectScatterTiling(c.Para(), /*key=*/2, /*ubAxis=*/1, /*ubFactor=*/670, /*tileElemNum=*/670 * 6,
                        /*total=*/16384, /*blockNum=*/64);
}

// ===========================================================================
// T2 Contig（TPL 1）：连续段 ≥ 32B（n'≥3 时 256B）且段首 32B 对齐
// ===========================================================================

// 5. var{16,1024} 全切：合轴后 n'=1、S'=[16384]、T'=[1] ⇒ contigRun=16384 → 65536B ≥ 32B
//    预期：T2 / key=1；seg=1 → chunksPerSeg 由核数反推，每核 1 个 chunk
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Contig_FullSlice)
{
    SsaTilingCase c({16, 1024}, {16, 1024}, {0, 0}, {16, 1024}, {1, 1}, {0, 1});
    ExecuteTestCase(c.Para(), ge::GRAPH_SUCCESS, 1, std::vector<size_t>{SSA_UT_WORKSPACE});
}

// 6. 内轴部分切但外层不连续：var{256,2048}, end={256,1024} ⇒ S'=[256,1024]、T'=[2048,1]
//    预期：T2 / key=1；segNum=256，多段铺核
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Contig_MultiSeg)
{
    SsaTilingCase c({256, 2048}, {256, 1024}, {0, 0}, {256, 1024}, {1, 1}, {0, 1});
    ExecuteTestCase(c.Para(), ge::GRAPH_SUCCESS, 1, std::vector<size_t>{SSA_UT_WORKSPACE});
}

// 7. axes 缺省（OPTIONAL 不给）：前导轴切片，其余整轴 ⇒ S'=[8]、T'=[1]，连续 32B ⇒ T2 / key=1
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Contig_NoAxesInput)
{
    SsaTilingCase c({2, 8}, {1, 8}, {1}, {2}, {1});
    ExecuteTestCase(c.Para(), ge::GRAPH_SUCCESS, 1, std::vector<size_t>{SSA_UT_WORKSPACE});
}

// ===========================================================================
// T1 Hybrid（TPL 0）
// ===========================================================================

// 7. 小 shape（非连续）：var{8,16} 内维步长 2 ⇒ 合轴后 S'=[64]、T'=[2]（内维 step≠1 不满足判据④）
//    sliceElemNum=64 <= 1024 ⇒ 判据② T1 / key=0
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Hybrid_SmallShape)
{
    SsaTilingCase c({8, 16}, {8, 8}, {0, 0}, {8, 16}, {1, 2}, {0, 1});
    ExecuteTestCase(c.Para(), ge::GRAPH_SUCCESS, 0, std::vector<size_t>{SSA_UT_WORKSPACE});
}

// 8. 小 shape + 全负方向：var{2,4} 反向整取 ⇒ S'=[8]、T'=[-1]、dir=ALL_NEGATIVE
//    预期：T1 / key=0
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Hybrid_SmallNegative)
{
    SsaTilingCase c({2, 4}, {2, 4}, {1, 3}, {-3, -5}, {-1, -1}, {0, 1});
    ExecuteTestCase(c.Para(), ge::GRAPH_SUCCESS, 0, std::vector<size_t>{SSA_UT_WORKSPACE});
}

// 9. 正负混合：var{4,8}，轴0 负步长（S'=[4,8]、T'=[-8,1]）⇒ dir=MIXED
//    预期：T1 / key=0（判据①最先命中，与 shape 大小无关）
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Hybrid_MixedDirection)
{
    SsaTilingCase c({4, 8}, {4, 8}, {3, 0}, {-5, 8}, {-1, 1}, {0, 1});
    ExecuteTestCase(c.Para(), ge::GRAPH_SUCCESS, 0, std::vector<size_t>{SSA_UT_WORKSPACE});
}

// 10. 极稀疏：大 shape 但每 cacheline 仅 1 个有效元素
//     var{64,4096} fp32，内维步长 64 ⇒ 合轴后 S'=[4096]、T'=[64]
//     cacheLineElems=32、内维 step=64 != 1 且 innerShape>=32 ⇒ checkNum=32 ⇒ validNum=ceil(32/64)=1 <= 1
//     预期：T1 / key=0（判据③，注意判据②未命中：sliceElemNum=4096 > 1024）
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Hybrid_ExtremelySparse)
{
    SsaTilingCase c({64, 4096}, {64, 64}, {0, 0}, {64, 4096}, {1, 64}, {0, 1});
    ExecuteTestCase(c.Para(), ge::GRAPH_SUCCESS, 0, std::vector<size_t>{SSA_UT_WORKSPACE});
}

// ===========================================================================
// T1 切分结果断言（TilingData 全字段）：核间三件套 / varBase / 约简形状 S'/T' / ubFactor
// （预算 (ubUsable−512)/(2·elemSize)，向 16 元素粒度向下取整，下限 16）
// ===========================================================================

// 11.1 全负方向 3 维（与 ST 用例 allneg_unit_mid 同形）：
//      var{8,20,6}, begin=[7,15,5], end=[-9,13,-7], strides=[-1,-3,-1]
//      ⇒ S'=[8,1,6]、T'=[-120,-18,-1]（无合轴）、varBase=7·120+15·6+5=935
//      ⇒ sliceElemNum=48 ≤ 1024（判据②）⇒ T1 / key=0；单核铺满
//      ⇒ ubFactor(fp32) = AlignDown((253952−32768−512)/8, 16) = 27584
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Hybrid_AllNegativeFullFields)
{
    SsaTilingCase c({8, 20, 6}, {8, 1, 6}, {7, 15, 5}, {-9, 13, -7}, {-1, -3, -1}, {0, 1, 2});
    ExpectHybridTiling(c.Para(), /*key=*/0, /*blockNum=*/1, /*elemPerCore=*/48, /*lastCoreElemNum=*/48,
                       /*varBase=*/935, /*ubFactor=*/27584, /*sliceShape=*/{8, 1, 6},
                       /*axisStep=*/{-120, -18, -1});
}

// 11.2 正负混合 + 大切片多核切分（与 ST 用例 mixed_multitile 同形）：
//      var{350,200}, begin=[0,199], end=[350,-201], strides=[1,-1]
//      ⇒ S'=[350,200]、T'=[200,-1]（T' 异号不可合轴）、varBase=199、sliceElemNum=70000
//      ⇒ 判据①（混合）最先命中 ⇒ T1 / key=0（与 shape 大小、连续性无关）
//      ⇒ blockDim=clamp(CeilDiv(70000,1024),1,64)=64、elemPerCore=1094、lastCoreElemNum=1078
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Hybrid_MixedMultiCoreSplit)
{
    SsaTilingCase c({350, 200}, {350, 200}, {0, 199}, {350, -201}, {1, -1}, {0, 1});
    ExpectHybridTiling(c.Para(), /*key=*/0, /*blockNum=*/64, /*elemPerCore=*/1094, /*lastCoreElemNum=*/1078,
                       /*varBase=*/199, /*ubFactor=*/27584, /*sliceShape=*/{350, 200},
                       /*axisStep=*/{200, -1});
}

// 11.3 ubFactor 16 元素粒度：注入非 16 倍数的 UB 预算，锁 AlignDown 向下取整行为
//      var{8,16} 内维步长 2（合轴后 S'=[64]、T'=[2]，内维 step≠1 避免判据④抢跑），fp32，
//      ubUsable=253448 ⇒ 扣 SIMT DCache 后 (253448−32768−512)/8 = 27521（非 16 倍数）
//      ⇒ AlignDown(27521,16) = 27520
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Hybrid_UbFactorAlignDownTo16Elems)
{
    SsaTilingCase c({8, 16}, {8, 8}, {0, 0}, {8, 16}, {1, 2}, {0, 1}, ge::DT_FLOAT, true, /*ubUsable=*/253448);
    ExpectHybridTiling(c.Para(), /*key=*/0, /*blockNum=*/1, /*elemPerCore=*/64, /*lastCoreElemNum=*/64,
                       /*varBase=*/0, /*ubFactor=*/27520, /*sliceShape=*/{64}, /*axisStep=*/{2});
}

// 11.4 ubFactor 下限保护：预算不足以容纳一个对齐单元时钳到 16 元素
//      var{8,16} 内维步长 2，fp32，ubUsable=600 ⇒ (600−32768−512)/8 < 0 ⇒ AlignDown < 0 ⇒ Max(·,16)=16
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Hybrid_UbFactorFloorClamp)
{
    SsaTilingCase c({8, 16}, {8, 8}, {0, 0}, {8, 16}, {1, 2}, {0, 1}, ge::DT_FLOAT, true, /*ubUsable=*/600);
    ExpectHybridTiling(c.Para(), /*key=*/0, /*blockNum=*/1, /*elemPerCore=*/64, /*lastCoreElemNum=*/64,
                       /*varBase=*/0, /*ubFactor=*/16, /*sliceShape=*/{64}, /*axisStep=*/{2});
}

// 11.5 fp16 的 ubFactor 平台口径：AlignDown((253952−32768−512)/(2·2), 16) = 55168
//      （ST 预算推导口径：512B paramBuf + 32KB SIMT DCache + 2 槽 × 55168 × 2B = 253952B = 100% UB）
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Hybrid_UbFactorFp16)
{
    SsaTilingCase c({8, 16}, {8, 8}, {0, 0}, {8, 16}, {1, 2}, {0, 1}, ge::DT_FLOAT16);
    ExpectHybridTiling(c.Para(), /*key=*/0, /*blockNum=*/1, /*elemPerCore=*/64, /*lastCoreElemNum=*/64,
                       /*varBase=*/0, /*ubFactor=*/55168, /*sliceShape=*/{64}, /*axisStep=*/{2});
}

// 11.6 int8 的 ubFactor 平台口径：AlignDown((253952−32768−512)/(2·1), 16) = 110336
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Hybrid_UbFactorInt8)
{
    SsaTilingCase c({8, 16}, {8, 8}, {0, 0}, {8, 16}, {1, 2}, {0, 1}, ge::DT_INT8);
    ExpectHybridTiling(c.Para(), /*key=*/0, /*blockNum=*/1, /*elemPerCore=*/64, /*lastCoreElemNum=*/64,
                       /*varBase=*/0, /*ubFactor=*/110336, /*sliceShape=*/{64}, /*axisStep=*/{2});
}

// ===========================================================================
// T4 Empty（TPL 3）
// ===========================================================================

// 12. var 含 0 长度维 ⇒ 无元素可写
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Empty_VarZeroDim)
{
    SsaTilingCase c({0, 4}, {0, 4}, {0, 0}, {0, 4}, {1, 1}, {0, 1});
    ExecuteTestCase(c.Para(), ge::GRAPH_SUCCESS, 3, std::vector<size_t>{SSA_UT_WORKSPACE});
}

// 13. 切片区域为空（begin == end）⇒ 同样退化为空模板
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Empty_SliceEmpty)
{
    SsaTilingCase c({4, 8}, {0, 8}, {2, 0}, {2, 8}, {1, 1}, {0, 1});
    ExecuteTestCase(c.Para(), ge::GRAPH_SUCCESS, 3, std::vector<size_t>{SSA_UT_WORKSPACE});
}

// ===========================================================================
// 失败路径
// ===========================================================================

// 14. begin 非 const 输入（ValueDepend REQUIRED 未满足）⇒ 取不到常量值
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Failed_BeginNotConst)
{
    SsaTilingCase c({4, 8}, {4, 8}, {0, 0}, {4, 8}, {1, 1}, {0, 1}, ge::DT_FLOAT, /*constInputs=*/false);
    ExecuteTestCase(c.Para(), ge::GRAPH_FAILED);
}

// 15. strides 含 0 ⇒ 报错
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Failed_ZeroStride)
{
    SsaTilingCase c({4, 8}, {4, 8}, {0, 0}, {4, 8}, {1, 0}, {0, 1});
    ExecuteTestCase(c.Para(), ge::GRAPH_FAILED);
}

// 16. axes 归一化后重复（1 与 -1 在 2 维下同轴）⇒ 报错
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Failed_DuplicateAxes)
{
    SsaTilingCase c({4, 3}, {4, 3}, {1, 0}, {4, 2}, {2, 1}, {-1, 1});
    ExecuteTestCase(c.Para(), ge::GRAPH_FAILED);
}

// 17. axes 越界 ⇒ 报错
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Failed_AxesOutOfRange)
{
    SsaTilingCase c({4, 8}, {4, 8}, {1}, {4}, {1}, {99});
    ExecuteTestCase(c.Para(), ge::GRAPH_FAILED);
}

// 18. input_value 形状与切片形状不符（不支持广播）⇒ 报错
//     切片 S=[3,8]（begin={1,0} end={4,8}），value 给 {4,8} ⇒ 形状不匹配
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Failed_ValueShapeMismatch)
{
    SsaTilingCase c({4, 8}, {4, 8}, {1, 0}, {4, 8}, {1, 1}, {0, 1});
    ExecuteTestCase(c.Para(), ge::GRAPH_FAILED);
}

// 19. input_value 秩与 var 不等 ⇒ 报错
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Failed_ValueDimMismatch)
{
    SsaTilingCase c({4, 8}, {4}, {0, 0}, {4, 8}, {1, 1}, {0, 1});
    ExecuteTestCase(c.Para(), ge::GRAPH_FAILED);
}

// 19.1 不支持 dtype：uint8（elemSize 合法但不在白名单）⇒ 报错
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Failed_UnsupportedDtypeUint8)
{
    SsaTilingCase c({4, 8}, {4, 8}, {0, 0}, {4, 8}, {1, 1}, {0, 1}, ge::DT_UINT8);
    ExecuteTestCase(c.Para(), ge::GRAPH_FAILED);
}

// 19.2 不支持 dtype：bool ⇒ 报错
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Failed_UnsupportedDtypeBool)
{
    SsaTilingCase c({4, 8}, {4, 8}, {0, 0}, {4, 8}, {1, 1}, {0, 1}, ge::DT_BOOL);
    ExecuteTestCase(c.Para(), ge::GRAPH_FAILED);
}

// ===========================================================================
// 补充用例（白盒 CSV 挑选，覆盖负向/阈值/对齐回落/axes 子集等路由分支）
// ===========================================================================

// 20. 1D 反向整取：contigRun=1025 对齐 ⇒ T2 / key=1
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Contig_Negative1D)
{
    SsaTilingCase c({2048}, {1025}, {1024}, {-2049}, {-1}, {0});
    ExecuteTestCase(c.Para(), ge::GRAPH_SUCCESS, 1, std::vector<size_t>{SSA_UT_WORKSPACE});
}

// 21. 连续但首址 4B 非 32B 对齐 ⇒ 回落 T3 / key=2
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Scatter_AlignFallback1D)
{
    SsaTilingCase c({2048}, {2047}, {1}, {2048}, {1}, {0});
    ExecuteTestCase(c.Para(), ge::GRAPH_SUCCESS, 2, std::vector<size_t>{SSA_UT_WORKSPACE});
}

// 22. 非连续 axes 子集 [0,2]：contigRun=20B < 32B ⇒ T1 / key=0
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Hybrid_AxesSubset)
{
    SsaTilingCase c({10, 20, 30}, {2, 20, 5}, {2, 5}, {4, 10}, {1, 1}, {0, 2});
    ExecuteTestCase(c.Para(), ge::GRAPH_SUCCESS, 0, std::vector<size_t>{SSA_UT_WORKSPACE});
}

// 23. 阈值上界：sliceElem=1025 连续对齐 ⇒ T2 / key=1
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Contig_ThresholdUpper)
{
    SsaTilingCase c({1025}, {1025}, {0}, {1025}, {1}, {0});
    ExecuteTestCase(c.Para(), ge::GRAPH_SUCCESS, 1, std::vector<size_t>{SSA_UT_WORKSPACE});
}

// 24. 1D 全负反向小切片：varBase 未对齐 + sliceElem=32 ≤ 1024 ⇒ T1 / key=0
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Hybrid_AllNegative1D)
{
    SsaTilingCase c({32}, {32}, {31}, {-33}, {-1}, {0});
    ExecuteTestCase(c.Para(), ge::GRAPH_SUCCESS, 0, std::vector<size_t>{SSA_UT_WORKSPACE});
}

// 25. int32 3D 全连续：n'=3、contigRun=40B < 256B ⇒ T1 / key=0
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Hybrid_Int32_3D)
{
    SsaTilingCase c({10, 20, 30}, {2, 5, 10}, {2, 5, 10}, {4, 10, 20}, {1, 1, 1}, {0, 1, 2}, ge::DT_INT32);
    ExecuteTestCase(c.Para(), ge::GRAPH_SUCCESS, 0, std::vector<size_t>{SSA_UT_WORKSPACE});
}

// 26. 单元素张量：contigRun=4B < 32B ⇒ T1 / key=0
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Hybrid_SingleElem)
{
    SsaTilingCase c({1, 1}, {1, 1}, {0, 0}, {1, 1}, {1, 1}, {0, 1});
    ExecuteTestCase(c.Para(), ge::GRAPH_SUCCESS, 0, std::vector<size_t>{SSA_UT_WORKSPACE});
}

// 27. 多核尾核：1D 全连续 65793 元素不整除核数 ⇒ T2 / key=1
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Contig_MultiCoreTail)
{
    SsaTilingCase c({65793}, {65793}, {0}, {65793}, {1}, {0});
    ExecuteTestCase(c.Para(), ge::GRAPH_SUCCESS, 1, std::vector<size_t>{SSA_UT_WORKSPACE});
}

// 28. 2D 负 stride 合并（n'=1）：contigRun=9153 对齐 ⇒ T2 / key=1
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Contig_Negative2DMerged)
{
    SsaTilingCase c({16, 1017}, {9, 1017}, {8, 1016}, {-18, -1019}, {-1, -1}, {0, 1});
    ExecuteTestCase(c.Para(), ge::GRAPH_SUCCESS, 1, std::vector<size_t>{SSA_UT_WORKSPACE});
}

// 29. 正负 stride 混合大 shape：判据① ⇒ T1 / key=0
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Hybrid_MixedBigtile)
{
    SsaTilingCase c({4200, 1000}, {4200, 1000}, {0, 999}, {4200, -1001}, {1, -1}, {0, 1});
    ExecuteTestCase(c.Para(), ge::GRAPH_SUCCESS, 0, std::vector<size_t>{SSA_UT_WORKSPACE});
}

// 30. 1D 负 stride 非 32B 对齐：varBase=2049 ⇒ 回落 T3 / key=2
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Scatter_Negative1D)
{
    SsaTilingCase c({4096}, {2050}, {2049}, {-4097}, {-1}, {0});
    ExecuteTestCase(c.Para(), ge::GRAPH_SUCCESS, 2, std::vector<size_t>{SSA_UT_WORKSPACE});
}

// 31. 3D 非单位外层 stride [3,2,1]：contigRun=4096B ≥ 256B 且对齐 ⇒ T2 / key=1
TEST_F(StridedSliceAssignV2TilingArch35, ascend950_Contig_Ndim3)
{
    SsaTilingCase c({16, 8, 2048}, {6, 4, 1024}, {0, 0, 0}, {16, 8, 1024}, {3, 2, 1}, {0, 1, 2});
    ExecuteTestCase(c.Para(), ge::GRAPH_SUCCESS, 1, std::vector<size_t>{SSA_UT_WORKSPACE});
}
