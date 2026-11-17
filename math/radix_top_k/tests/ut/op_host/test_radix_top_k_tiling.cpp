/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/radix_top_k_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class RadixTopKTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RadixTopKTilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "RadixTopKTilingTest TearDown" << std::endl; }
};

namespace {

// 与 op_host/radix_top_k_tiling.cpp 常量对齐
constexpr uint64_t LARGE_SORT_LEN = 1440000000;
constexpr uint64_t INT32_SORT_LEN_MAX = 2147483647;
constexpr uint64_t CORE_NUM = 48;
constexpr uint64_t UB_SIZE = 196608;

// SelectTopK 键位与子门控窗口常量（与 op_host 一致）
constexpr uint64_t STK_BIT = 1ULL << 4;                // isSelectTopK 位于最高位（DECL 第 5 维）
constexpr uint64_t STK_MAX_K = 4096;                   // SelectTopK 窗口上界：k ≤ 4096
constexpr uint64_t STK_WINDOW_MIN_SORT_LEN = 12000000; // SelectTopK 窗口下界：sortLen ≥ 12e6
constexpr uint64_t STK_SLOT_ALIGN = 64;                // GM 归并槽位步长对齐（与 op_host/kernel 一致）
constexpr uint64_t SYS_WORKSPACE_SIZE = 16777216;      // GetLibApiWorkSpaceSize 在 UT 平台返回 16 MiB

static thread_local int32_t g_kValue = 0;

gert::TilingContextPara MakeRadixTopKTilingContext(const gert::StorageShape& xShape, ge::DataType xDtype,
                                                   int32_t kValue, bool sorted = true, int64_t dim = -1,
                                                   bool largest = true, int64_t indicesDtype = 3,
                                                   ge::DataType outIndicesDtype = ge::DT_INT32,
                                                   uint64_t ubSize = UB_SIZE)
{
    g_kValue = kValue;
    gert::StorageShape kShape({1}, {1});

    optiling::RadixTopKCompileInfo compileInfo;
    compileInfo.totalCoreNum = CORE_NUM;
    compileInfo.ubSizePlatForm = ubSize;

    auto* compileInfoPtr = new optiling::RadixTopKCompileInfo(compileInfo);

    return gert::TilingContextPara(
        "RadixTopK",
        {
            {xShape, xDtype, ge::FORMAT_ND},
            {kShape, ge::DT_INT32, ge::FORMAT_ND, true, &g_kValue},
        },
        {
            {xShape, xDtype, ge::FORMAT_ND},
            {xShape, outIndicesDtype, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("sorted", Ops::Math::AnyValue::CreateFrom<bool>(sorted)),
            gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(dim)),
            gert::TilingContextPara::OpAttr("largest", Ops::Math::AnyValue::CreateFrom<bool>(largest)),
            gert::TilingContextPara::OpAttr("indices_dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(indicesDtype)),
        },
        compileInfoPtr, /*coreNum=*/64, /*ubSize=*/ubSize);
}

// radix 键编码 bit0..bit3 = sorted/largest/isLargeShape/isInt64Shape，SelectTopK 占 bit4
uint64_t ComputeExpectedTilingKey(bool sorted, bool largest, uint64_t sortLen, bool isSelectTopK = false)
{
    uint64_t key = 0;
    key |= (static_cast<uint64_t>(sorted) << 0);
    key |= (static_cast<uint64_t>(largest) << 1);
    bool isLargeShape = (sortLen > LARGE_SORT_LEN);
    key |= (static_cast<uint64_t>(isLargeShape) << 2);
    // bit3 = isInt64Shape，与 GET_TPL_TILING_KEY 的五维顺序对应
    bool isInt64Shape = (sortLen > INT32_SORT_LEN_MAX);
    key |= (static_cast<uint64_t>(isInt64Shape) << 3);
    if (isSelectTopK) {
        // SelectTopK 单变体键：isLargeShape/isInt64Shape 钉为 0，仅由 sorted × largest 决定
        key = STK_BIT | static_cast<uint64_t>(sorted) | (static_cast<uint64_t>(largest) << 1);
    }
    return key;
}

// SelectTopK 键（isSelectTopK=1, isLargeShape=0, isInt64Shape=0）编码 ∈ {16, 17, 18, 19}
uint64_t ComputeSelectTopKKey(bool sorted, bool largest)
{
    return ComputeExpectedTilingKey(sorted, largest, /*sortLen=*/0, /*isSelectTopK=*/true);
}

bool IsRadixKey(uint64_t key) { return (key & STK_BIT) == 0; }
bool IsSelectTopKKey(uint64_t key) { return (key & STK_BIT) != 0; }

// radix 三分支名：bit3=1 ⇒ WsI64；bit2=1 ⇒ WS；否则 UB
const char* RadixBranchName(uint64_t key)
{
    if (((key >> 3) & 1ULL) != 0) {
        return "WsI64";
    }
    if (((key >> 2) & 1ULL) != 0) {
        return "WS";
    }
    return "UB";
}

// 逐 case 证据输出到 stdout（前缀 UT_EVIDENCE）与 RTK_UT_EVIDENCE_FILE 指定文件
void EmitEvidence(const string& caseName, uint64_t expectedKey, uint64_t actualKey, uint64_t stkKeyIfEnabled,
                  const string& branchOrVerdict)
{
    std::ostringstream oss;
    oss << "UT_EVIDENCE {\"case\":\"" << caseName << "\",\"expected_key\":" << expectedKey
        << ",\"actual_key\":" << actualKey << ",\"is_selecttopk\":" << (((actualKey >> 4) & 1ULL) ? "true" : "false")
        << ",\"branch\":\"" << branchOrVerdict << "\",\"stk_key_if_enabled\":" << stkKeyIfEnabled << "}";
    std::cout << oss.str() << std::endl;
    const char* evidenceFile = std::getenv("RTK_UT_EVIDENCE_FILE");
    if (evidenceFile != nullptr && evidenceFile[0] != '\0') {
        std::ofstream ofs(evidenceFile, std::ios::app);
        if (ofs.is_open()) {
            ofs << oss.str() << "\n";
        }
    }
}

void CheckTilingKey(const gert::TilingContextPara& tilingCtx, uint64_t expectKey)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingCtx, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectKey);
    EXPECT_EQ(tilingInfo.workspaceSizes.size(), 1);
}

void CheckTilingWithWs(const gert::TilingContextPara& tilingCtx, uint64_t expectKey, bool expectWsNonzero)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingCtx, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectKey);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    if (expectWsNonzero) {
        EXPECT_GT(tilingInfo.workspaceSizes[0], 0u) << "k 较小时应触发 workspace 分配 (needWorkspace=true)";
    } else {
        EXPECT_EQ(tilingInfo.workspaceSizes[0], 0u) << "k 较大时应复用 indices 内存 (needWorkspace=false)";
    }
}

// SelectTopK 目标窗口须回落 radix（bit4=0）；stkKeyIfEnabled 用于证明未误命中 SelectTopK 键
void CheckFallbackToRadix(const gert::TilingContextPara& tilingCtx, const string& caseName, uint64_t expectKey,
                          uint64_t stkKeyIfEnabled)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingCtx, tilingInfo)) << caseName;
    const uint64_t actualKey = static_cast<uint64_t>(tilingInfo.tilingKey);
    EmitEvidence(caseName, expectKey, actualKey, stkKeyIfEnabled, RadixBranchName(actualKey));
    EXPECT_EQ(actualKey, expectKey) << caseName << ": 子门控不满足时必须等于现网 radix 编码";
    EXPECT_TRUE(IsRadixKey(actualKey)) << caseName << ": bit4(isSelectTopK) 必须为 0（回落 radix）";
    EXPECT_FALSE(IsSelectTopKKey(actualKey)) << caseName << ": SelectTopK 键（bit4=1）对本输入不可达";
    EXPECT_NE(actualKey, stkKeyIfEnabled) << caseName << ": 不得命中 SelectTopK 键编码";
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1) << caseName;
}

// SelectTopK 目标窗口须命中 SelectTopK 键（bit4=1，编码 16..19）
void CheckSelectTopKRoute(const gert::TilingContextPara& tilingCtx, const string& caseName, uint64_t expectKey)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingCtx, tilingInfo)) << caseName;
    const uint64_t actualKey = static_cast<uint64_t>(tilingInfo.tilingKey);
    EmitEvidence(caseName, expectKey, actualKey, expectKey, "SelectTopK");
    EXPECT_EQ(actualKey, expectKey) << caseName << ": 目标窗口必须命中 SelectTopK 键";
    EXPECT_TRUE(IsSelectTopKKey(actualKey)) << caseName << ": bit4(isSelectTopK) 必须为 1";
    EXPECT_GE(actualKey, STK_BIT) << caseName << ": SelectTopK 键编码必须 ∈ 16..19";
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1) << caseName;
}

// radix 三分支回归：键编码 + 分支 + workspace 行为与现网一致
void CheckRadixBranchCase(const gert::TilingContextPara& tilingCtx, const string& caseName, uint64_t expectKey,
                          const char* expectBranch, bool expectWsNonzero, uint64_t stkKeyIfEnabled = 0)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingCtx, tilingInfo)) << caseName;
    const uint64_t actualKey = static_cast<uint64_t>(tilingInfo.tilingKey);
    EmitEvidence(caseName, expectKey, actualKey, stkKeyIfEnabled, RadixBranchName(actualKey));
    EXPECT_EQ(actualKey, expectKey) << caseName;
    EXPECT_TRUE(IsRadixKey(actualKey)) << caseName << ": radix 回归要求 bit4=0";
    EXPECT_STREQ(RadixBranchName(actualKey), expectBranch) << caseName;
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1) << caseName;
    if (expectWsNonzero) {
        EXPECT_GT(tilingInfo.workspaceSizes[0], 0u) << caseName;
    } else {
        EXPECT_EQ(tilingInfo.workspaceSizes[0], 0u) << caseName;
    }
}

} // namespace

// ---- UB 变体 fp16：1D shapes ----

TEST_F(RadixTopKTilingTest, ub_fp16_1d_128_k5)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{128}, {128}}, ge::DT_FLOAT16, 5);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(true, true, 128));
}

TEST_F(RadixTopKTilingTest, ub_fp16_1d_4096_k500)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{4096}, {4096}}, ge::DT_FLOAT16, 500);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(true, true, 4096));
}

TEST_F(RadixTopKTilingTest, ub_fp16_1d_65536_k4097)
{
    // k=4097 > STK_MAX_K，保持 radix UB 分支覆盖
    auto tilingCtx = MakeRadixTopKTilingContext({{65536}, {65536}}, ge::DT_FLOAT16, 4097);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(true, true, 65536));
}

// ---- UB 变体 fp16：2D/3D shapes ----

TEST_F(RadixTopKTilingTest, ub_fp16_2d_32x512_k6)
{
    // sortLen/k = 85 < 100，保持 radix UB 分支覆盖
    auto tilingCtx = MakeRadixTopKTilingContext({{32, 512}, {32, 512}}, ge::DT_FLOAT16, 6);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(true, true, 512));
}

TEST_F(RadixTopKTilingTest, ub_fp16_3d_4x128x256_k10)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{4, 128, 256}, {4, 128, 256}}, ge::DT_FLOAT16, 10);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(true, true, 256));
}

TEST_F(RadixTopKTilingTest, ub_fp16_bert_512x768_k10)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{512, 768}, {512, 768}}, ge::DT_FLOAT16, 10);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(true, true, 768));
}

// ---- UB 变体 bf16 ----

TEST_F(RadixTopKTilingTest, ub_bf16_1d_1024_k5)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{1024}, {1024}}, ge::DT_BF16, 5);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(true, true, 1024));
}

TEST_F(RadixTopKTilingTest, ub_bf16_2d_128x1024_k50)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{128, 1024}, {128, 1024}}, ge::DT_BF16, 50);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(true, true, 1024));
}

// ---- largest/sorted 组合对 tiling key 的影响 ----

TEST_F(RadixTopKTilingTest, ub_fp16_largest_false_sorted_true)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{512}, {512}}, ge::DT_FLOAT16, 10, true, -1, false);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(true, false, 512));
}

TEST_F(RadixTopKTilingTest, ub_fp16_largest_true_sorted_false)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{512}, {512}}, ge::DT_FLOAT16, 10, false, -1, true);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(false, true, 512));
}

TEST_F(RadixTopKTilingTest, ub_fp16_largest_false_sorted_false)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{512}, {512}}, ge::DT_FLOAT16, 10, false, -1, false);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(false, false, 512));
}

// ---- 边界 K 值 ----

TEST_F(RadixTopKTilingTest, ub_fp16_k_equals_sortlen)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{256}, {256}}, ge::DT_FLOAT16, 256);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(true, true, 256));
}

TEST_F(RadixTopKTilingTest, ub_fp16_k_exceeds_ws_threshold)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{4096}, {4096}}, ge::DT_FLOAT16, 385);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(true, true, 4096));
}

// ---- 极小 shape 边界 ----

TEST_F(RadixTopKTilingTest, ub_fp16_tiny_1_elem)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{1}, {1}}, ge::DT_FLOAT16, 1);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(true, true, 1));
}

TEST_F(RadixTopKTilingTest, ub_fp16_tiny_3_elems)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{3}, {3}}, ge::DT_FLOAT16, 2);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(true, true, 3));
}

TEST_F(RadixTopKTilingTest, ub_fp16_tiny_1x1)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{1, 1}, {1, 1}}, ge::DT_FLOAT16, 1);
    CheckTilingKey(tilingCtx, ComputeExpectedTilingKey(true, true, 1));
}

// ---- WS 变体（sortLen > LARGE_SORT_LEN，需要 workspace）----

TEST_F(RadixTopKTilingTest, ws_fp16_large_sort_len_k8192)
{
    // k=8192 > STK_MAX_K，保持 radix WS 覆盖
    constexpr uint64_t BIG_SORT_LEN = LARGE_SORT_LEN + 1;
    auto tilingCtx = MakeRadixTopKTilingContext(
        {{1, static_cast<int64_t>(BIG_SORT_LEN)}, {1, static_cast<int64_t>(BIG_SORT_LEN)}}, ge::DT_FLOAT16, 8192);
    uint64_t expectKey = ComputeExpectedTilingKey(true, true, BIG_SORT_LEN);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingCtx, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectKey);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_GT(tilingInfo.workspaceSizes[0], 0);
}

// ---- Workspace：k ≤ coreNum×6 时需要，否则复用 indices ----

TEST_F(RadixTopKTilingTest, ws_verify_small_k_needs_ws)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{512}, {512}}, ge::DT_FLOAT16, 5);
    CheckTilingWithWs(tilingCtx, ComputeExpectedTilingKey(true, true, 512), true);
}

TEST_F(RadixTopKTilingTest, ws_verify_large_k_reuses_indices)
{
    auto tilingCtx = MakeRadixTopKTilingContext({{4096}, {4096}}, ge::DT_FLOAT16, 500);
    CheckTilingWithWs(tilingCtx, ComputeExpectedTilingKey(true, true, 4096), false);
}

// ---- k=100 UB 变体：sortLen 足够大使 coreNum×6 > 100，触发 workspace ----

TEST_F(RadixTopKTilingTest, k100_ub_needs_ws)
{
    // sortLen/k = 81 < 100，保持 radix UB + workspace 覆盖
    auto tilingCtx = MakeRadixTopKTilingContext({{8192}, {8192}}, ge::DT_FLOAT16, 100);
    CheckTilingWithWs(tilingCtx, ComputeExpectedTilingKey(true, true, 8192), true);
}

// ---- int64 主路径（sortLen > 2^31）：tiling key 第 4 位 + 入口防御校验 ----

TEST_F(RadixTopKTilingTest, i64_shape_tiling_key_bit3)
{
    // sortLen = 2^31 ⇒ isInt64Shape（bit3）并蕴含 isLargeShape（bit2）；indices 须 int64 且申请 workspace
    constexpr uint64_t SORT_LEN = 2147483648ULL; // 2^31
    auto tilingCtx = MakeRadixTopKTilingContext(
        {{1, static_cast<int64_t>(SORT_LEN)}, {1, static_cast<int64_t>(SORT_LEN)}}, ge::DT_FLOAT16, 1001, true, -1,
        true, /*indicesDtype=*/9 /*DT_INT64*/,
        /*outIndicesDtype=*/ge::DT_INT64);
    CheckTilingWithWs(tilingCtx, ComputeExpectedTilingKey(true, true, SORT_LEN), true);
}

TEST_F(RadixTopKTilingTest, i64_shape_reject_int32_indices)
{
    // sortLen > 2^31 但 indices 为 int32 ⇒ tiling 必须返回 GRAPH_FAILED
    constexpr uint64_t SORT_LEN = 2147483648ULL;
    auto tilingCtx = MakeRadixTopKTilingContext(
        {{1, static_cast<int64_t>(SORT_LEN)}, {1, static_cast<int64_t>(SORT_LEN)}}, ge::DT_FLOAT16, 1001);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingCtx, tilingInfo)) << "sortLen 超 int32 上限且 indices 非 int64 时必须失败";
}

// ---- SelectTopK 路径 UT：TilingKey 分发与回落（radix 16 键 + SelectTopK 4 键）----

static_assert(STK_MAX_K == 4096 && STK_WINDOW_MIN_SORT_LEN == 12000000,
              "SelectTopK 子门控窗口常量必须与 DESIGN.md §3.4③ 一致");
static_assert(STK_BIT == 16, "isSelectTopK 必须位于 TilingKey 最高位（bit4）");

// ---------- 1) 键域编码：16 radix（bit4=0）+ 4 SelectTopK（bit4=1）= 20 键 ----------

TEST_F(RadixTopKTilingTest, stk_key_domain_encoding_20_keys)
{
    // 20 键域：radix 16 键（0..15）+ SelectTopK 4 键（16..19）
    for (int sorted = 0; sorted <= 1; ++sorted) {
        for (int largest = 0; largest <= 1; ++largest) {
            for (int large = 0; large <= 1; ++large) {
                for (int i64 = 0; i64 <= 1; ++i64) {
                    const uint64_t radixKey = static_cast<uint64_t>(sorted) | (static_cast<uint64_t>(largest) << 1) |
                                              (static_cast<uint64_t>(large) << 2) | (static_cast<uint64_t>(i64) << 3);
                    EXPECT_TRUE(IsRadixKey(radixKey)) << "radix 16 键必须 bit4=0";
                    EXPECT_LT(radixKey, 16u) << "radix 16 键编码范围 0..15";
                }
            }
        }
    }
    std::vector<uint64_t> stkKeys;
    for (int sorted = 0; sorted <= 1; ++sorted) {
        for (int largest = 0; largest <= 1; ++largest) {
            const uint64_t stkKey = ComputeSelectTopKKey(sorted == 1, largest == 1);
            EXPECT_EQ(stkKey, STK_BIT | static_cast<uint64_t>(sorted) | (static_cast<uint64_t>(largest) << 1))
                << "SelectTopK 键编码必须为 16 + sorted + 2*largest（isSelectTopK 位于最高位）";
            EXPECT_TRUE(IsSelectTopKKey(stkKey));
            stkKeys.push_back(stkKey);
        }
    }
    std::sort(stkKeys.begin(), stkKeys.end());
    const std::vector<uint64_t> expectStkKeys = {16, 17, 18, 19}; // 4 键集合
    EXPECT_EQ(stkKeys, expectStkKeys) << "SelectTopK 4 键编码集合必须为 {16,17,18,19}";
    EmitEvidence("stk_key_domain_encoding_20_keys", expectStkKeys[0], expectStkKeys[0], expectStkKeys[3],
                 "encoding_domain_radix16_bit4_0_plus_stk4_bit4_1");
}

// ---------- 2) SelectTopK 目标窗口内输入命中 SelectTopK 键 ----------
// 窗口：k ≤ 4096 ∧ sortLen ≥ 12e6 ∧ indices int32 ∧ sortLen ≤ 2^31-1

TEST_F(RadixTopKTilingTest, stk_route_bf16_s1_l1_k512_s12e6)
{
    auto tilingCtx = MakeRadixTopKTilingContext(
        {{static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}, {static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}}, ge::DT_BF16,
        512, /*sorted=*/true, -1, /*largest=*/true);
    CheckSelectTopKRoute(tilingCtx, "stk_route_bf16_s1_l1_k512_s12e6", ComputeSelectTopKKey(true, true));
}

TEST_F(RadixTopKTilingTest, stk_route_bf16_s1_l0_k512_s12e6)
{
    auto tilingCtx = MakeRadixTopKTilingContext(
        {{static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}, {static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}}, ge::DT_BF16,
        512, /*sorted=*/true, -1, /*largest=*/false);
    CheckSelectTopKRoute(tilingCtx, "stk_route_bf16_s1_l0_k512_s12e6", ComputeSelectTopKKey(true, false));
}

TEST_F(RadixTopKTilingTest, stk_route_bf16_s0_l1_k512_s12e6)
{
    auto tilingCtx = MakeRadixTopKTilingContext(
        {{static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}, {static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}}, ge::DT_BF16,
        512, /*sorted=*/false, -1, /*largest=*/true);
    CheckSelectTopKRoute(tilingCtx, "stk_route_bf16_s0_l1_k512_s12e6", ComputeSelectTopKKey(false, true));
}

TEST_F(RadixTopKTilingTest, stk_route_bf16_s0_l0_k512_s12e6)
{
    auto tilingCtx = MakeRadixTopKTilingContext(
        {{static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}, {static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}}, ge::DT_BF16,
        512, /*sorted=*/false, -1, /*largest=*/false);
    CheckSelectTopKRoute(tilingCtx, "stk_route_bf16_s0_l0_k512_s12e6", ComputeSelectTopKKey(false, false));
}

TEST_F(RadixTopKTilingTest, stk_route_bf16_k4096_s12e6)
{
    // 窗口上界 k = 4096，命中 SelectTopK 键（bit4=1）
    auto tilingCtx = MakeRadixTopKTilingContext(
        {{static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}, {static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}}, ge::DT_BF16,
        static_cast<int32_t>(STK_MAX_K), true, -1, true);
    CheckSelectTopKRoute(tilingCtx, "stk_route_bf16_k4096_s12e6", ComputeSelectTopKKey(true, true));
}

TEST_F(RadixTopKTilingTest, stk_route_bf16_k1001_s12e6)
{
    // k = 1001 命中 SelectTopK 键（bit4=1）
    auto tilingCtx = MakeRadixTopKTilingContext(
        {{static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}, {static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}}, ge::DT_BF16,
        1001, true, -1, true);
    CheckSelectTopKRoute(tilingCtx, "stk_route_bf16_k1001_s12e6", ComputeSelectTopKKey(true, true));
}

TEST_F(RadixTopKTilingTest, stk_route_bf16_2d_k1024_s12e6)
{
    // 2D batch=2 < totalCoreNum_ ⇒ stkCoreMode=0（行-核映射）
    auto tilingCtx = MakeRadixTopKTilingContext(
        {{2, static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}, {2, static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}},
        ge::DT_BF16, 1024, true, -1, true);
    CheckSelectTopKRoute(tilingCtx, "stk_route_bf16_2d_k1024_s12e6", ComputeSelectTopKKey(true, true));
}

TEST_F(RadixTopKTilingTest, stk_route_large_window_sortlen_15e8_k4096)
{
    // sortLen = 15e8 > LARGE_SORT_LEN 但 ≤ INT32_SORT_LEN_MAX ⇒ 仍命中 SelectTopK 键 19（非 radix WS 键 7）
    constexpr uint64_t SORT_LEN = 1500000000ULL;
    auto tilingCtx = MakeRadixTopKTilingContext(
        {{1, static_cast<int64_t>(SORT_LEN)}, {1, static_cast<int64_t>(SORT_LEN)}}, ge::DT_BF16, 4096, true, -1, true);
    CheckSelectTopKRoute(tilingCtx, "stk_route_large_window_sortlen_15e8_k4096", ComputeSelectTopKKey(true, true));
    EXPECT_NE(ComputeExpectedTilingKey(true, true, SORT_LEN), ComputeSelectTopKKey(true, true))
        << "SelectTopK 键（19）不得等于该输入回落 radix 时的键（7）";
}

// ---------- 2b) 子门控否定项：任一条不满足 ⇒ 回落 radix ----------

TEST_F(RadixTopKTilingTest, stk_fallback_fp16_k4097_s12e6)
{
    // k = 4097 > STK_MAX_K ⇒ fp16 同样回落 radix
    auto tilingCtx = MakeRadixTopKTilingContext(
        {{static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}, {static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}},
        ge::DT_FLOAT16, 4097, true, -1, true);
    CheckFallbackToRadix(tilingCtx, "stk_fallback_fp16_k4097_s12e6",
                         ComputeExpectedTilingKey(true, true, STK_WINDOW_MIN_SORT_LEN),
                         ComputeSelectTopKKey(true, true));
}

TEST_F(RadixTopKTilingTest, stk_route_fp16_k1024_s12e6)
{
    // fp16 与 bf16 同窗口 ⇒ 命中 SelectTopK 键
    auto tilingCtx = MakeRadixTopKTilingContext(
        {{static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}, {static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}},
        ge::DT_FLOAT16, 1024, true, -1, true);
    CheckSelectTopKRoute(tilingCtx, "stk_route_fp16_k1024_s12e6", ComputeSelectTopKKey(true, true));
}

TEST_F(RadixTopKTilingTest, stk_fallback_indices_int64_k512_s12e6)
{
    // indices 非 int32 ⇒ 不满足子门控，回落 radix（bit4=0）
    auto tilingCtx = MakeRadixTopKTilingContext(
        {{static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}, {static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}}, ge::DT_BF16,
        512, true, -1, true, /*indicesDtype=*/9 /*DT_INT64*/, /*outIndicesDtype=*/ge::DT_INT64);
    CheckFallbackToRadix(tilingCtx, "stk_fallback_indices_int64_k512_s12e6",
                         ComputeExpectedTilingKey(true, true, STK_WINDOW_MIN_SORT_LEN),
                         ComputeSelectTopKKey(true, true));
}

TEST_F(RadixTopKTilingTest, stk_fallback_k_beyond_4096_s12e6)
{
    // k = 4097 > STK_MAX_K ⇒ 回落 radix（bit4=0）
    auto tilingCtx = MakeRadixTopKTilingContext(
        {{static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}, {static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}}, ge::DT_BF16,
        4097, true, -1, true);
    CheckFallbackToRadix(tilingCtx, "stk_fallback_k_beyond_4096_s12e6",
                         ComputeExpectedTilingKey(true, true, STK_WINDOW_MIN_SORT_LEN),
                         ComputeSelectTopKKey(true, true));
}

TEST_F(RadixTopKTilingTest, stk_fallback_sortlen_below_window)
{
    // sortLen = 400000 < 12e6 ⇒ 窗口外，回落 radix
    constexpr uint64_t SORT_LEN = 400000ULL;
    auto tilingCtx = MakeRadixTopKTilingContext({{static_cast<int64_t>(SORT_LEN)}, {static_cast<int64_t>(SORT_LEN)}},
                                                ge::DT_BF16, 4096, true, -1, true);
    CheckFallbackToRadix(tilingCtx, "stk_fallback_sortlen_below_window", ComputeExpectedTilingKey(true, true, SORT_LEN),
                         ComputeSelectTopKKey(true, true));
}

TEST_F(RadixTopKTilingTest, stk_fallback_window_lower_bound_minus_one)
{
    // sortLen = 12e6 - 1：窗口下界为闭区间，差一个元素即回落 radix
    constexpr uint64_t SORT_LEN = STK_WINDOW_MIN_SORT_LEN - 1ULL;
    auto tilingCtx = MakeRadixTopKTilingContext({{static_cast<int64_t>(SORT_LEN)}, {static_cast<int64_t>(SORT_LEN)}},
                                                ge::DT_BF16, 512, true, -1, true);
    CheckFallbackToRadix(tilingCtx, "stk_fallback_window_lower_bound_minus_one",
                         ComputeExpectedTilingKey(true, true, SORT_LEN), ComputeSelectTopKKey(true, true));
}

TEST_F(RadixTopKTilingTest, stk_fallback_small_sortlen_high_ratio)
{
    // sortLen = 1e6、k = 1000（sortLen/k ≥ 100）：仍以需求窗口为准 ⇒ 回落 radix
    constexpr uint64_t SORT_LEN = 1000000ULL;
    auto tilingCtx = MakeRadixTopKTilingContext({{static_cast<int64_t>(SORT_LEN)}, {static_cast<int64_t>(SORT_LEN)}},
                                                ge::DT_BF16, 1000, true, -1, true);
    CheckFallbackToRadix(tilingCtx, "stk_fallback_small_sortlen_high_ratio",
                         ComputeExpectedTilingKey(true, true, SORT_LEN), ComputeSelectTopKKey(true, true));
}

TEST_F(RadixTopKTilingTest, stk_fallback_dim_not_last)
{
    // dim 不为 -1 且不为最后一维 ⇒ GetShapeAttrsInfo 必须返回 GRAPH_FAILED
    auto tilingCtx = MakeRadixTopKTilingContext({{64, 512}, {64, 512}}, ge::DT_BF16, 5, true, /*dim=*/0, true);
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingCtx, tilingInfo)) << "dim=0 而 rank-1=1 时必须失败";
}

// ---------- 2c) SelectTopK tiling 字段断言与 UB 不可用时的 fail-soft 回落 ----------

// 取出 tiling 输出中的 SelectTopK 字段；长度不足视为失败
static bool LoadTilingData(const TilingInfo& info, RadixTopKTilingData& out)
{
    if (info.tilingData == nullptr || info.tilingDataSize < sizeof(RadixTopKTilingData)) {
        return false;
    }
    memcpy(&out, info.tilingData.get(), sizeof(RadixTopKTilingData));
    return true;
}

// 断言 SelectTopK 档位、核映射、归并叉数与候选容量契约
static void CheckDsTilingFields(const TilingInfo& info, uint32_t expectCoreMode, uint32_t expectKLevel,
                                uint32_t expectMergeArity, uint64_t expectCoreNum)
{
    RadixTopKTilingData td{};
    ASSERT_TRUE(LoadTilingData(info, td)) << "tilingData 长度不足";
    EXPECT_EQ(td.stkCoreMode, expectCoreMode);
    EXPECT_EQ(td.stkKLevel, expectKLevel);
    EXPECT_EQ(td.stkMergeArity, expectMergeArity);
    EXPECT_EQ(td.coreNum, expectCoreNum);
    EXPECT_GT(td.stkBlockSize, 0ULL);
    EXPECT_GT(td.stkFilteredCapacity, 0ULL);
    EXPECT_EQ(td.stkSafeTrigger, td.kValue + td.stkFilteredCapacity);
    // 扫描一次整块追加不得溢出候选缓冲
    EXPECT_GE(td.stkCandidateCapacity, td.stkSafeTrigger + td.stkBlockSize);
}

TEST_F(RadixTopKTilingTest, stk_fields_bf16_k1024_s12e6)
{
    // batch=1 + 长序列 ⇒ 行内多核；k=1024 ⇒ kLevel=1、4 路归并
    auto tilingCtx = MakeRadixTopKTilingContext(
        {{static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}, {static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}}, ge::DT_BF16,
        1024);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingCtx, tilingInfo));
    EXPECT_TRUE(IsSelectTopKKey(tilingInfo.tilingKey));
    CheckDsTilingFields(tilingInfo, /*coreMode=*/1U, /*kLevel=*/1U, /*mergeArity=*/4U, CORE_NUM);
    // 行内多核（stkCoreMode==1）：归并槽位 workspace = 2 * coreNum * align64(k) * sizeof(int32)，另加系统 workspace
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], SYS_WORKSPACE_SIZE + 2ULL * CORE_NUM * 1024ULL * 4ULL);
}

TEST_F(RadixTopKTilingTest, stk_fields_bf16_k4096_s12e6)
{
    // k=4096 ⇒ kLevel=2、2 路归并（4k 候选放不进 UB）
    auto tilingCtx = MakeRadixTopKTilingContext(
        {{static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}, {static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}}, ge::DT_BF16,
        4096);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingCtx, tilingInfo));
    EXPECT_TRUE(IsSelectTopKKey(tilingInfo.tilingKey));
    CheckDsTilingFields(tilingInfo, 1U, 2U, 2U, CORE_NUM);
}

TEST_F(RadixTopKTilingTest, stk_fields_bf16_k128_s12e6)
{
    // k=128 ⇒ kLevel=0、核 0 单核串行归并（arity=0）
    auto tilingCtx = MakeRadixTopKTilingContext(
        {{static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}, {static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}}, ge::DT_BF16,
        128);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingCtx, tilingInfo));
    EXPECT_TRUE(IsSelectTopKKey(tilingInfo.tilingKey));
    CheckDsTilingFields(tilingInfo, 1U, 0U, 0U, CORE_NUM);
}

TEST_F(RadixTopKTilingTest, stk_fields_row_mode_batch48)
{
    // batch = totalCoreNum ⇒ 行-核映射，每核 1 行
    auto tilingCtx = MakeRadixTopKTilingContext(
        {{static_cast<int64_t>(CORE_NUM), static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)},
         {static_cast<int64_t>(CORE_NUM), static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}},
        ge::DT_BF16, 1024);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingCtx, tilingInfo));
    EXPECT_TRUE(IsSelectTopKKey(tilingInfo.tilingKey));
    CheckDsTilingFields(tilingInfo, 0U, 1U, 4U, CORE_NUM);
    RadixTopKTilingData td{};
    ASSERT_TRUE(LoadTilingData(tilingInfo, td));
    EXPECT_EQ(td.stkRowsPerCore, 1ULL);
    EXPECT_EQ(td.stkFormerCoreNum, CORE_NUM);
    EXPECT_EQ(td.stkTailCoreNum, 0ULL);
    // 整行整除（r=0）无余数行段化 ⇒ 不分配 GM 槽位归并，仅保留系统 workspace
    EXPECT_EQ(td.stkResidualRows, 0ULL);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], SYS_WORKSPACE_SIZE) << "纯整行行模式不应分配归并槽位 workspace";
}

TEST_F(RadixTopKTilingTest, stk_fields_row_mode_residual_ws)
{
    // batch = CORE_NUM+1 ⇒ 行模式余数行段化（r=1）：余数行由全核协作扫段 + GM 槽位树归并，
    // 必须分配归并槽位 workspace（修复前该场景 workspace=0，kernel WriteSlot 向 0 长 GM buffer 越界写）
    const int64_t batch = static_cast<int64_t>(CORE_NUM) + 1;
    auto tilingCtx = MakeRadixTopKTilingContext({{batch, static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)},
                                                 {batch, static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}},
                                                ge::DT_BF16, 1000);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingCtx, tilingInfo));
    EXPECT_TRUE(IsSelectTopKKey(tilingInfo.tilingKey));
    // k=1000 ⇒ kLevel=1（>512 且 ≤1024）、4 路归并；batch>coreNum ⇒ blockDim=coreNum
    CheckDsTilingFields(tilingInfo, 0U, 1U, 4U, CORE_NUM);
    RadixTopKTilingData td{};
    ASSERT_TRUE(LoadTilingData(tilingInfo, td));
    EXPECT_EQ(td.stkRowsPerCore, 1ULL);
    EXPECT_EQ(td.stkFullRows, CORE_NUM);
    EXPECT_EQ(td.stkResidualRows, 1ULL);
    EXPECT_GT(td.stkSegLen, 0ULL);
    // 余数行走 GM 槽位树归并：workspace = 2 * coreNum * align64(k) * sizeof(int32)
    constexpr uint64_t kAlign = (1000ULL + STK_SLOT_ALIGN - 1) / STK_SLOT_ALIGN * STK_SLOT_ALIGN; // = 1024
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], SYS_WORKSPACE_SIZE + 2ULL * CORE_NUM * kAlign * 4ULL)
        << "行模式余数行段化同样需要归并槽位 workspace";
}

TEST_F(RadixTopKTilingTest, stk_fallback_when_no_ub_tier)
{
    // ubSize=70000：SelectTopK 三档 UB 占用均超预算（最小档已超 100 KB），但 radix 小 tile 仍可容纳 ⇒ 回落 radix
    auto tilingCtx = MakeRadixTopKTilingContext(
        {{static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}, {static_cast<int64_t>(STK_WINDOW_MIN_SORT_LEN)}}, ge::DT_BF16,
        512, true, -1, true, /*indicesDtype=*/3, /*outIndicesDtype=*/ge::DT_INT32, /*ubSize=*/70000);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingCtx, tilingInfo));
    EXPECT_TRUE(IsRadixKey(tilingInfo.tilingKey));
    RadixTopKTilingData td{};
    ASSERT_TRUE(LoadTilingData(tilingInfo, td));
    EXPECT_EQ(td.stkCoreMode, 0U);
    EXPECT_EQ(td.stkBlockSize, 0ULL);
    EXPECT_EQ(td.stkCandidateCapacity, 0ULL);
    EXPECT_GT(td.coreNum, 0ULL);
    EXPECT_GT(td.formerTileLen, 0ULL);
}

// ---------- 3) radix 回归：16 键三分支编码与 workspace 行为不变 ----------

TEST_F(RadixTopKTilingTest, radix_regression_ub_branch_all_sorted_largest)
{
    // R-UB：4 键 = sorted | (largest<<1)，编码 0..3；k 大时复用 indices（ws=0）
    constexpr uint64_t SORT_LEN = 4096;
    for (int sorted = 0; sorted <= 1; ++sorted) {
        for (int largest = 0; largest <= 1; ++largest) {
            auto tilingCtx = MakeRadixTopKTilingContext({{4096}, {4096}}, ge::DT_FLOAT16, 512, sorted == 1, -1,
                                                        largest == 1);
            const string name = "radix_regression_ub_s" + std::to_string(sorted) + "_l" + std::to_string(largest);
            CheckRadixBranchCase(tilingCtx, name, ComputeExpectedTilingKey(sorted == 1, largest == 1, SORT_LEN), "UB",
                                 /*expectWsNonzero=*/false);
        }
    }
}

TEST_F(RadixTopKTilingTest, radix_regression_ws_branch_all_sorted_largest)
{
    // R-WS：sortLen > 1.44e9 ⇒ 键编码 4..7（bit2=1），需要 workspace
    constexpr uint64_t SORT_LEN = 1500000000ULL;
    for (int sorted = 0; sorted <= 1; ++sorted) {
        for (int largest = 0; largest <= 1; ++largest) {
            // k=5000 > STK_MAX_K，确保本回归仍走 radix WS 分支
            auto tilingCtx = MakeRadixTopKTilingContext(
                {{1, static_cast<int64_t>(SORT_LEN)}, {1, static_cast<int64_t>(SORT_LEN)}}, ge::DT_FLOAT16, 5000,
                sorted == 1, -1, largest == 1);
            const string name = "radix_regression_ws_s" + std::to_string(sorted) + "_l" + std::to_string(largest);
            CheckRadixBranchCase(tilingCtx, name, ComputeExpectedTilingKey(sorted == 1, largest == 1, SORT_LEN), "WS",
                                 /*expectWsNonzero=*/true);
        }
    }
}

TEST_F(RadixTopKTilingTest, radix_regression_i64_branch_all_sorted_largest)
{
    // R-I64：sortLen > 2^31-1（indices 必须 int64）⇒ 键编码 12..15
    constexpr uint64_t SORT_LEN = 2147483648ULL; // 2^31
    for (int sorted = 0; sorted <= 1; ++sorted) {
        for (int largest = 0; largest <= 1; ++largest) {
            auto tilingCtx = MakeRadixTopKTilingContext(
                {{1, static_cast<int64_t>(SORT_LEN)}, {1, static_cast<int64_t>(SORT_LEN)}}, ge::DT_FLOAT16, 1001,
                sorted == 1, -1, largest == 1, /*indicesDtype=*/9 /*DT_INT64*/,
                /*outIndicesDtype=*/ge::DT_INT64);
            const string name = "radix_regression_i64_s" + std::to_string(sorted) + "_l" + std::to_string(largest);
            CheckRadixBranchCase(tilingCtx, name, ComputeExpectedTilingKey(sorted == 1, largest == 1, SORT_LEN),
                                 "WsI64", /*expectWsNonzero=*/true);
        }
    }
}
