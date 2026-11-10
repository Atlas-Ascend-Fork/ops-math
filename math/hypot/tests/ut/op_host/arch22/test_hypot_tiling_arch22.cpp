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
 * \file test_hypot_tiling_arch22.cpp
 * \brief hypot arch22 tiling UT: template dispatch (TilingKey_FP32_OneDim / TilingKey_FP32_MultiDimBroadcast),
 *        scene decision, multi-core/UB split, scalar flag, failure branches
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../../op_host/arch22/hypot_tiling_arch22.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace NsHypot;

class HypotTilingArch22Test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HypotTilingArch22Test SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "HypotTilingArch22Test TearDown" << std::endl; }
};

namespace {
// TilingKey encoding: only BROADCAST_SCENE is encoded (dtypes come from the per-combination binary),
// so every dtype combination shares the same five keys.
constexpr uint64_t kKeyOneDim = 0;   // scene list index 0
constexpr uint64_t kKeyMultiDim = 1; // scene list index 1
constexpr uint64_t kKeyStruct = 2;   // scene list index 2
constexpr uint64_t kKeyStream = 3;   // scene list index 3
constexpr uint64_t kKeyVecIdx = 4;   // scene list index 4

// Ascend910B3-like fabric platform: 40 AIV cores, 192KB UB
constexpr int32_t kCoreNum = 40;
constexpr int64_t kUbSize = 196608;

const HypotTilingData* TD(const TilingInfo& info)
{
    return reinterpret_cast<const HypotTilingData*>(info.tilingData.get());
}
} // namespace

// pin the TilingKey encoding contract between host tiling and kernel ASCENDC_TPL_SEL: the key is
// the scene only, identical for every dtype combination
TEST_F(HypotTilingArch22Test, tiling_key_encoding_sanity)
{
    EXPECT_EQ(GET_TPL_TILING_KEY(HYPOT_SCENE_ONE_DIM), kKeyOneDim);
    EXPECT_EQ(GET_TPL_TILING_KEY(HYPOT_SCENE_MULTI_DIM_BRC), kKeyMultiDim);
    EXPECT_EQ(GET_TPL_TILING_KEY(HYPOT_SCENE_MULTI_DIM_STRUCT), kKeyStruct);
    EXPECT_EQ(GET_TPL_TILING_KEY(HYPOT_SCENE_MULTI_DIM_STREAM), kKeyStream);
    EXPECT_EQ(GET_TPL_TILING_KEY(HYPOT_SCENE_MULTI_DIM_VECIDX), kKeyVecIdx);
}

// [8,1,33] x [1,4,1] -> cross-3D double-sided broadcast -> STRUCT scene dispatch
TEST_F(HypotTilingArch22Test, tiling_fp32_struct_cross3d)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{8, 1, 33}, {8, 1, 33}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1, 4, 1}, {1, 4, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{8, 4, 33}, {8, 4, 33}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, kKeyStruct);
    const HypotTilingData* td = TD(info);
    EXPECT_EQ(td->totalLength, 1056U);
    EXPECT_EQ(td->scalarFlag, 0U);
    EXPECT_EQ(td->dimNum, 3U);
}

// [2,2,2,2,2,2,1,8] x [1,2,2,1,2,2,2,8] -> fused rank-6 with aligned ROW input -> STRUCT scene
TEST_F(HypotTilingArch22Test, tiling_fp32_struct_high_rank_row)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para(
        "Hypot",
        {
            {{{2, 2, 2, 2, 2, 2, 1, 8}, {2, 2, 2, 2, 2, 2, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 2, 2, 1, 2, 2, 2, 8}, {1, 2, 2, 1, 2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 2, 2, 2, 2, 2, 2, 8}, {2, 2, 2, 2, 2, 2, 2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, kKeyStruct);
    const HypotTilingData* td = TD(info);
    EXPECT_EQ(td->totalLength, 1024U);
    EXPECT_EQ(td->scalarFlag, 0U);
    EXPECT_GT(td->dimNum, 4U);
}

// [1,3,1,1] x [2,1,4,8] -> 192-element family: every small (<=1024) same-dtype
// multi-dim broadcast routes to the VECIDX scene (single-core whole-output Gather expansion)
TEST_F(HypotTilingArch22Test, tiling_fp32_tiny192_vecidx)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{1, 3, 1, 1}, {1, 3, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{2, 1, 4, 8}, {2, 1, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{2, 3, 4, 8}, {2, 3, 4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, kKeyVecIdx);
    const HypotTilingData* td = TD(info);
    EXPECT_EQ(td->totalLength, 192U);
    EXPECT_EQ(td->usedCoreNum, 1U);
    EXPECT_EQ(td->ubFormer, 192U);
}

// same shape fuses to 1 dim -> OneDim dispatch; small input shrinks ubFormer to the 256-element floor
TEST_F(HypotTilingArch22Test, tiling_fp32_onedim_same_shape)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, kKeyOneDim);
    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_EQ(info.workspaceSizes[0], 0U);
    EXPECT_EQ(info.blockNum, 1U);
    const HypotTilingData* td = TD(info);
    EXPECT_EQ(td->totalLength, 64U);
    EXPECT_EQ(td->scalarFlag, 0U);
    EXPECT_EQ(td->usedCoreNum, 1U);
    EXPECT_EQ(td->blockFormer, 1U);
    EXPECT_EQ(td->blockTail, 1U);
    EXPECT_EQ(td->ubFormer, 256U);
    EXPECT_EQ(td->ubOuter, 1U);
    EXPECT_EQ(td->ubTail, 64U);
}

// [1] x [2,3]: x1 fuses to numel 1 -> OneDim + scalarFlag bit0 (kernel Duplicate path contract)
TEST_F(HypotTilingArch22Test, tiling_fp32_onedim_scalar_input)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, kKeyOneDim);
    EXPECT_EQ(info.workspaceSizes[0], 0U);
    const HypotTilingData* td = TD(info);
    EXPECT_EQ(td->totalLength, 6U);
    EXPECT_EQ(td->scalarFlag, 0x1U);
    EXPECT_EQ(td->usedCoreNum, 1U);
    EXPECT_EQ(td->ubFormer, 256U);
    EXPECT_EQ(td->ubOuter, 1U);
    EXPECT_EQ(td->ubTail, 6U);
}

// large OneDim input feeds all 40 cores: ubFormer shrinks (32-elem steps) until ubOuter >= coreNum
TEST_F(HypotTilingArch22Test, tiling_fp32_onedim_multicore_split)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{102400}, {102400}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{102400}, {102400}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{102400}, {102400}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, kKeyOneDim);
    EXPECT_EQ(info.blockNum, 24U);
    const HypotTilingData* td = TD(info);
    EXPECT_EQ(td->totalLength, 102400U);
    EXPECT_EQ(td->usedCoreNum, 24U);
    EXPECT_EQ(td->blockFormer, 1U);
    EXPECT_EQ(td->blockTail, 1U);
    // Core-shrink keeps a useful 24-way split instead of forcing all 40 cores.
    EXPECT_EQ(td->ubFormer, 4320U);
    EXPECT_EQ(td->ubOuter, 24U);
    EXPECT_EQ(td->ubTail, 3040U);
}

// [3,1] x [1,4] -> MultiDimBroadcast dispatch; x1 innermost broadcast -> BRC_COL, x2 row repeat -> BRC_ROW
TEST_F(HypotTilingArch22Test, tiling_fp32_multidim_brc_3x1_1x4)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{3, 1}, {3, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1, 4}, {1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{3, 4}, {3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    // small (<=1024) same-dtype multi-dim broadcasts route to the VECIDX scene
    EXPECT_EQ(info.tilingKey, kKeyVecIdx);
    EXPECT_EQ(info.workspaceSizes[0], 0U);
    EXPECT_EQ(info.blockNum, 1U);
    const HypotTilingData* td = TD(info);
    EXPECT_EQ(td->totalLength, 12U);
    EXPECT_EQ(td->scalarFlag, 0U);
    EXPECT_EQ(td->dimNum, 2U);
    EXPECT_EQ(td->outShape[0], 3U);
    EXPECT_EQ(td->outShape[1], 4U);
    EXPECT_EQ(td->x1Strides[0], 1U);
    EXPECT_EQ(td->x1Strides[1], 0U);
    EXPECT_EQ(td->x2Strides[0], 0U);
    EXPECT_EQ(td->x2Strides[1], 1U);
    EXPECT_EQ(td->outStrides[0], 4U);
    EXPECT_EQ(td->outStrides[1], 1U);
    EXPECT_EQ(td->ubSplitAxis, 0U);
    EXPECT_EQ(td->dimProduct, 1U);
    EXPECT_EQ(td->innerLen, 4U);
    EXPECT_EQ(td->rowPitch, 8U);
    EXPECT_EQ(td->ubFormer, 12U); // VECIDX: single tile covering the whole output
    EXPECT_EQ(td->ubOuter, 1U);
    EXPECT_EQ(td->ubTail, 12U);
    EXPECT_EQ(td->usedCoreNum, 1U);
    EXPECT_EQ(td->blockFormer, 1U);
    EXPECT_EQ(td->blockTail, 1U);
    EXPECT_EQ(td->x1BrcMode, HYPOT_BRC_COL);
    EXPECT_EQ(td->x1LeadInner, 1U);
    EXPECT_EQ(td->x1BrcInner, 4U);
    EXPECT_EQ(td->x1SrcInner, 1U);
    EXPECT_EQ(td->x2BrcMode, HYPOT_BRC_ROW);
    EXPECT_EQ(td->x2LeadInner, 1U);
    EXPECT_EQ(td->x2BrcInner, 1U);
    EXPECT_EQ(td->x2SrcInner, 4U);
    EXPECT_EQ(td->x1TileLen, 3U);
    EXPECT_EQ(td->x2TileLen, 4U);
}

// [2,3] x [3] -> small broadcast routes to the VECIDX scene
TEST_F(HypotTilingArch22Test, tiling_fp32_multidim_brc_leftpad_2x3_3)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, kKeyVecIdx);
    EXPECT_EQ(info.blockNum, 1U);
    const HypotTilingData* td = TD(info);
    EXPECT_EQ(td->totalLength, 6U);
    EXPECT_EQ(td->dimNum, 2U);
    EXPECT_EQ(td->outShape[0], 2U);
    EXPECT_EQ(td->outShape[1], 3U);
    EXPECT_EQ(td->x1Strides[0], 3U);
    EXPECT_EQ(td->x1Strides[1], 1U);
    EXPECT_EQ(td->x2Strides[0], 0U);
    EXPECT_EQ(td->x2Strides[1], 1U);
    EXPECT_EQ(td->ubSplitAxis, 0U);
    EXPECT_EQ(td->innerLen, 3U);
    EXPECT_EQ(td->rowPitch, 8U);
    EXPECT_EQ(td->ubFormer, 6U); // VECIDX: single tile covering the whole output
    EXPECT_EQ(td->ubOuter, 1U);
    EXPECT_EQ(td->ubTail, 6U);
    EXPECT_EQ(td->usedCoreNum, 1U);
    EXPECT_EQ(td->x1BrcMode, HYPOT_BRC_NONE);
    EXPECT_EQ(td->x2BrcMode, HYPOT_BRC_ROW);
    EXPECT_EQ(td->x1TileLen, 6U);
    EXPECT_EQ(td->x2TileLen, 3U);
}

// empty tensor guard branch: both inputs share a 0 axis -> fused totalLength == 0 -> safe no-op config
// (the aclnn layer filters empty tensors before launch; this only guards against a direct launch)
TEST_F(HypotTilingArch22Test, tiling_fp32_empty_tensor)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{0, 3}, {0, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{0, 3}, {0, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{0, 3}, {0, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, kKeyOneDim);
    EXPECT_EQ(info.workspaceSizes[0], 0U);
    EXPECT_EQ(info.blockNum, 1U);
    const HypotTilingData* td = TD(info);
    EXPECT_EQ(td->totalLength, 0U);
    EXPECT_EQ(td->usedCoreNum, 1U);
    EXPECT_EQ(td->blockFormer, 0U);
    EXPECT_EQ(td->blockTail, 0U);
    EXPECT_EQ(td->ubTail, 0U);
    EXPECT_EQ(td->ubOuter, 0U);
}

// failure branch: rank > HYPOT_MAX_RANK (8) is rejected
TEST_F(HypotTilingArch22Test, tiling_failed_rank_over_limit)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para(
        "Hypot",
        {
            {{{1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED, 0, "");
}

// failure branch: UB budget <= 0 (ubSize <= HYPOT_UB_EXTRA_SIZE) is rejected
TEST_F(HypotTilingArch22Test, tiling_failed_ub_too_small)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, 8192};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED, 0, "");
}

// ---------------------------------------------------------------------------
// per-dtype dispatch coverage (the TilingKey is dtype-independent by design)
// ---------------------------------------------------------------------------

// FP16 OneDim dispatch (OneDim bytes-per-elem = 31)
TEST_F(HypotTilingArch22Test, tiling_fp16_onedim_dispatch)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, kKeyOneDim);
    const HypotTilingData* td = TD(info);
    EXPECT_EQ(td->totalLength, 1024U);
    EXPECT_EQ(td->scalarFlag, 0U);
    EXPECT_EQ(td->ubFormer, 320U);
    EXPECT_EQ(td->ubOuter, 4U);
    EXPECT_EQ(td->ubTail, 64U);
    EXPECT_EQ(td->usedCoreNum, 4U);
}

// BF16 OneDim dispatch
TEST_F(HypotTilingArch22Test, tiling_bf16_onedim_dispatch)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{8, 8}, {8, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                     {{{8, 8}, {8, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{8, 8}, {8, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, kKeyOneDim);
    const HypotTilingData* td = TD(info);
    EXPECT_EQ(td->totalLength, 64U);
    EXPECT_EQ(td->ubFormer, 256U);
    EXPECT_EQ(td->ubTail, 64U);
    EXPECT_EQ(td->usedCoreNum, 1U);
}

// FP16 MultiDimBroadcast dispatch: dtype branch + 51 bytes-per-elem budget path
TEST_F(HypotTilingArch22Test, tiling_fp16_multidim_dispatch)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{3, 1}, {3, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 4}, {1, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, kKeyVecIdx); // small broadcast -> VECIDX scene
    const HypotTilingData* td = TD(info);
    EXPECT_EQ(td->totalLength, 12U);
    EXPECT_EQ(td->dimNum, 2U);
    EXPECT_EQ(td->x1BrcMode, HYPOT_BRC_COL);
    EXPECT_EQ(td->x2BrcMode, HYPOT_BRC_ROW);
    EXPECT_EQ(td->usedCoreNum, 1U);
}

// BF16 MultiDimBroadcast dispatch
TEST_F(HypotTilingArch22Test, tiling_bf16_multidim_dispatch)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{3, 1}, {3, 1}}, ge::DT_BF16, ge::FORMAT_ND},
                                     {{{1, 4}, {1, 4}}, ge::DT_BF16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{3, 4}, {3, 4}}, ge::DT_BF16, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, kKeyVecIdx); // small broadcast -> VECIDX scene
    const HypotTilingData* td = TD(info);
    EXPECT_EQ(td->totalLength, 12U);
    EXPECT_EQ(td->dimNum, 2U);
    EXPECT_EQ(td->x1BrcMode, HYPOT_BRC_COL);
    EXPECT_EQ(td->x2BrcMode, HYPOT_BRC_ROW);
}

// ---------------------------------------------------------------------------
// STREAM scene: large (>= 1M elems) same-dtype half rank-2 single-sided broadcast
// ---------------------------------------------------------------------------

// [64,65536] x [64,1] fp16 -> COL-single stream: 8832-element tiles (21 B/elem budget, 128B
// preload reserved), row-major walk, broadcast side runs the scalar-b chain from the 64-element
// preload (no b tensor materialized).
TEST_F(HypotTilingArch22Test, tiling_fp16_stream_col)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{64, 65536}, {64, 65536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{64, 1}, {64, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{64, 65536}, {64, 65536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, kKeyStream);
    const HypotTilingData* td = TD(info);
    EXPECT_EQ(td->totalLength, 4194304U);
    EXPECT_EQ(td->dimNum, 2U);
    EXPECT_EQ(td->ubSplitAxis, 1U);
    EXPECT_EQ(td->innerLen, 1U);
    EXPECT_EQ(td->dimProduct, 64U);
    EXPECT_EQ(td->x1BrcMode, HYPOT_BRC_NONE);
    EXPECT_EQ(td->x2BrcMode, HYPOT_BRC_COL);
    EXPECT_EQ(td->ubFormer, 8832U);
    EXPECT_EQ(td->ubOuter, 8U);
    EXPECT_EQ(td->ubTail, 3712U);
    EXPECT_EQ(td->x2PreloadLen, 64U);
    EXPECT_EQ(td->usedCoreNum, 40U);
}

// [64,65536] x [1,65536] fp16 -> ROW-periodic stream: 6848-element tiles (27 B/elem budget, no
// preload, single-buffered staging queue), inner-major walk with the UB segment cache.
TEST_F(HypotTilingArch22Test, tiling_fp16_stream_row)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{64, 65536}, {64, 65536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 65536}, {1, 65536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{64, 65536}, {64, 65536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, kKeyStream);
    const HypotTilingData* td = TD(info);
    EXPECT_EQ(td->totalLength, 4194304U);
    EXPECT_EQ(td->x1BrcMode, HYPOT_BRC_NONE);
    EXPECT_EQ(td->x2BrcMode, HYPOT_BRC_ROW);
    EXPECT_EQ(td->ubFormer, 6848U);
    EXPECT_EQ(td->ubOuter, 10U);
    EXPECT_EQ(td->ubTail, 3904U);
    EXPECT_EQ(td->x2PreloadLen, 0U);
}

// role-swapped COL stream: [64,1] x [64,65536] bf16 -> broadcast side is x1
TEST_F(HypotTilingArch22Test, tiling_bf16_stream_col_swapped)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{64, 1}, {64, 1}}, ge::DT_BF16, ge::FORMAT_ND},
                                     {{{64, 65536}, {64, 65536}}, ge::DT_BF16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{64, 65536}, {64, 65536}}, ge::DT_BF16, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, kKeyStream);
    const HypotTilingData* td = TD(info);
    EXPECT_EQ(td->x1BrcMode, HYPOT_BRC_COL);
    EXPECT_EQ(td->x2BrcMode, HYPOT_BRC_NONE);
    EXPECT_EQ(td->x1PreloadLen, 64U);
    EXPECT_EQ(td->ubFormer, 8832U);
}

// FP32 never enters the STREAM scene (the template split is FP16/BF16 only)
TEST_F(HypotTilingArch22Test, tiling_fp32_stream_pattern_stays_multidim)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{64, 65536}, {64, 65536}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{64, 1}, {64, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{64, 65536}, {64, 65536}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, kKeyMultiDim);
}

// below the 1M-element stream floor the same pattern keeps the MULTI_DIM_BRC scene
TEST_F(HypotTilingArch22Test, tiling_fp16_stream_below_floor_stays_multidim)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{64, 8192}, {64, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{64, 1}, {64, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{64, 8192}, {64, 8192}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, kKeyMultiDim);
}

// ---------------- VECIDX scene: small same-dtype multi-dim broadcast ----------------

// [2,3,4,1] x [2,3,1,8] fp16 -> double-sided cross (fuses to [6,4,8] rank-3): 192-element
// whole-output tile, 1 core
TEST_F(HypotTilingArch22Test, tiling_fp16_vecidx_cross_rank4)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{2, 3, 4, 1}, {2, 3, 4, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 3, 1, 8}, {2, 3, 1, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{2, 3, 4, 8}, {2, 3, 4, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, kKeyVecIdx);
    const HypotTilingData* td = TD(info);
    EXPECT_EQ(td->totalLength, 192U);
    EXPECT_EQ(td->dimNum, 3U); // adjacent full axes fuse: [2,3] -> 6
    EXPECT_EQ(td->usedCoreNum, 1U);
    EXPECT_EQ(td->ubFormer, 192U);
    EXPECT_EQ(td->ubOuter, 1U);
}

// [2,1,2,2,2,1] x [2,2,2,1,2,8] bf16 -> double-sided cross rank-6 (the worst pre-vectorized-index
// family, ratio 6.6): 256-element whole-output tile, 1 core
TEST_F(HypotTilingArch22Test, tiling_bf16_vecidx_cross_rank6)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{2, 1, 2, 2, 2, 1}, {2, 1, 2, 2, 2, 1}}, ge::DT_BF16, ge::FORMAT_ND},
                                     {{{2, 2, 2, 1, 2, 8}, {2, 2, 2, 1, 2, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{2, 2, 2, 2, 2, 8}, {2, 2, 2, 2, 2, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, kKeyVecIdx);
    const HypotTilingData* td = TD(info);
    EXPECT_EQ(td->totalLength, 256U);
    EXPECT_EQ(td->dimNum, 6U);
    EXPECT_EQ(td->usedCoreNum, 1U);
}

// STRUCT keeps priority over VECIDX for its tuned cross-3D pattern (scene decided first)
TEST_F(HypotTilingArch22Test, tiling_fp32_struct_priority_over_vecidx)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{8, 1, 4}, {8, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1, 3, 1}, {1, 3, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{8, 3, 4}, {8, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, kKeyStruct);
    const HypotTilingData* td = TD(info);
    EXPECT_EQ(td->totalLength, 96U);
}

// dtype-aware UB budget: MultiDim FP32 (43 B/elem -> maxElem 4288) vs FP16 (51 B/elem -> maxElem 3648).
// [80,2100] inner row 2100: FP32 fits ubFormer=2 rows/tile, FP16 only 1 -> pinned distinct splits.
TEST_F(HypotTilingArch22Test, tiling_fp32_multidim_dtype_bytes)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{80, 2100}, {80, 2100}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{2100}, {2100}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{80, 2100}, {80, 2100}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, kKeyMultiDim);
    const HypotTilingData* td = TD(info);
    EXPECT_EQ(td->ubSplitAxis, 0U);
    EXPECT_EQ(td->innerLen, 2100U);
    EXPECT_EQ(td->rowPitch, 2104U); // ceil(2100, 8)
    EXPECT_EQ(td->ubFormer, 2U);    // 4288 / 2104 = 2
    EXPECT_EQ(td->ubOuter, 40U);
    EXPECT_EQ(td->ubTail, 2U);
    EXPECT_EQ(td->usedCoreNum, 40U);
    EXPECT_EQ(td->blockFormer, 1U);
    EXPECT_EQ(td->x1BrcMode, HYPOT_BRC_NONE);
    EXPECT_EQ(td->x2BrcMode, HYPOT_BRC_ROW);
}

TEST_F(HypotTilingArch22Test, tiling_fp16_multidim_dtype_bytes)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{80, 2100}, {80, 2100}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2100}, {2100}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{80, 2100}, {80, 2100}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, kKeyMultiDim);
    const HypotTilingData* td = TD(info);
    EXPECT_EQ(td->ubSplitAxis, 0U);
    EXPECT_EQ(td->innerLen, 2100U);
    EXPECT_EQ(td->rowPitch, 2104U);
    EXPECT_EQ(td->ubFormer, 2U);
    EXPECT_EQ(td->ubOuter, 40U);
    EXPECT_EQ(td->ubTail, 2U);
    EXPECT_EQ(td->usedCoreNum, 40U);
    EXPECT_EQ(td->blockFormer, 1U);
    EXPECT_EQ(td->x1BrcMode, HYPOT_BRC_NONE);
    EXPECT_EQ(td->x2BrcMode, HYPOT_BRC_ROW);
}

// axis merging: [4,3,8] x [1,3,8] fuses the two trailing full axes -> dimNum=2
TEST_F(HypotTilingArch22Test, tiling_fuse_adjacent_same_flag_axes)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{4, 3, 8}, {4, 3, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1, 3, 8}, {1, 3, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{4, 3, 8}, {4, 3, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    // 96-element single-sided broadcast routes to the VECIDX scene
    EXPECT_EQ(info.tilingKey, kKeyVecIdx);
    const HypotTilingData* td = TD(info);
    EXPECT_EQ(td->totalLength, 96U);
    EXPECT_EQ(td->dimNum, 2U);
    EXPECT_EQ(td->outShape[0], 4U);
    EXPECT_EQ(td->outShape[1], 24U);
    EXPECT_EQ(td->x1Strides[0], 24U);
    EXPECT_EQ(td->x1Strides[1], 1U);
    EXPECT_EQ(td->x2Strides[0], 0U); // broadcast axis stride = 0
    EXPECT_EQ(td->x2Strides[1], 1U);
    EXPECT_EQ(td->outStrides[0], 24U);
    EXPECT_EQ(td->outStrides[1], 1U);
    EXPECT_EQ(td->ubSplitAxis, 1U);
    EXPECT_EQ(td->innerLen, 1U);
    EXPECT_EQ(td->rowPitch, 8U);
    EXPECT_EQ(td->ubFormer, 96U); // VECIDX: single tile covering the whole output
    EXPECT_EQ(td->ubOuter, 1U);
    EXPECT_EQ(td->ubTail, 96U);
    EXPECT_EQ(td->usedCoreNum, 1U);
    EXPECT_EQ(td->x1BrcMode, HYPOT_BRC_NONE);
    EXPECT_EQ(td->x2BrcMode, HYPOT_BRC_NONE);
    EXPECT_EQ(td->x2SrcInner, 1U);
    EXPECT_EQ(td->x1TileLen, 24U);
    EXPECT_EQ(td->x2TileLen, 24U);
}

// BRC_GENERIC mode: x1=[1,3,1] has a split-axis broadcast AND an inner broadcast run -> generic
// index-mapping fallback; x2=[2,1,4] is a plain inner ROW expansion
TEST_F(HypotTilingArch22Test, tiling_brc_generic_mode)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{1, 3, 1}, {1, 3, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{2, 1, 4}, {2, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    // [1,3,1] x [2,1,4] is a cross-3D pattern ([R,1,I] x [1,A,1], totalLength 24 <= 4096):
    // dispatched to the isolated STRUCT scene (structured tiny plane path)
    EXPECT_EQ(info.tilingKey, kKeyStruct);
    const HypotTilingData* td = TD(info);
    EXPECT_EQ(td->totalLength, 24U);
    EXPECT_EQ(td->dimNum, 3U);
    EXPECT_EQ(td->ubSplitAxis, 0U);
    EXPECT_EQ(td->innerLen, 12U);
    EXPECT_EQ(td->rowPitch, 16U);
    EXPECT_EQ(td->x1BrcMode, HYPOT_BRC_GENERIC);
    EXPECT_EQ(td->x1LeadInner, 3U);
    EXPECT_EQ(td->x1BrcInner, 1U);
    EXPECT_EQ(td->x1SrcInner, 1U);
    EXPECT_EQ(td->x1TileLen, 3U);
    EXPECT_EQ(td->x2BrcMode, HYPOT_BRC_ROW);
    EXPECT_EQ(td->x2LeadInner, 1U);
    EXPECT_EQ(td->x2BrcInner, 3U);
    EXPECT_EQ(td->x2SrcInner, 4U);
    EXPECT_EQ(td->x2TileLen, 8U);
}

// scalarFlag bit1: x2 fuses to numel 1 -> OneDim + kernel Duplicate path for x2
TEST_F(HypotTilingArch22Test, tiling_scalar_flag_x2)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, kKeyOneDim);
    const HypotTilingData* td = TD(info);
    EXPECT_EQ(td->totalLength, 6U);
    EXPECT_EQ(td->scalarFlag, 0x2U);
    EXPECT_EQ(td->usedCoreNum, 1U);
    EXPECT_EQ(td->ubTail, 6U);
}

// both inputs scalar: scalarFlag = 0b11, degenerate 1-element OneDim
TEST_F(HypotTilingArch22Test, tiling_scalar_flag_both)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, kKeyOneDim);
    const HypotTilingData* td = TD(info);
    EXPECT_EQ(td->totalLength, 1U);
    EXPECT_EQ(td->scalarFlag, 0x3U);
    EXPECT_EQ(td->usedCoreNum, 1U);
    EXPECT_EQ(td->ubTail, 1U);
}

// scalar input surviving into the MultiDim scene: [1,1] x [3,1] keeps dimNum=2 (the out==1 axis
// clears both flags so the axes do not merge); x1 numel==1 takes the ComputeInputMode early return
TEST_F(HypotTilingArch22Test, tiling_multidim_scalar_input)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{1, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{3, 1}, {3, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{3, 1}, {3, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, kKeyOneDim);
    const HypotTilingData* td = TD(info);
    EXPECT_EQ(td->totalLength, 3U);
    EXPECT_EQ(td->dimNum, 0U);
    EXPECT_EQ(td->scalarFlag, 0x1U);
    EXPECT_EQ(td->ubSplitAxis, 0U);
    EXPECT_EQ(td->innerLen, 0U);
    EXPECT_EQ(td->x1TileLen, 0U);
    EXPECT_EQ(td->x2TileLen, 0U);
    EXPECT_EQ(td->usedCoreNum, 1U);
}

// MultiDim shrink loop floor: ubFormer already 1 but cores still starved -> stop via ubFormer <= 1
TEST_F(HypotTilingArch22Test, tiling_multidim_shrink_floor)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{2, 4000}, {2, 4000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{1, 4000}, {1, 4000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{2, 4000}, {2, 4000}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, kKeyMultiDim);
    const HypotTilingData* td = TD(info);
    EXPECT_EQ(td->ubSplitAxis, 1U);
    EXPECT_EQ(td->innerLen, 1U);
    EXPECT_EQ(td->rowPitch, 8U);
    EXPECT_EQ(td->ubFormer, 3999U);
    EXPECT_EQ(td->ubOuter, 2U);
    EXPECT_EQ(td->ubTail, 1U);
    EXPECT_EQ(td->usedCoreNum, 4U);
    EXPECT_EQ(td->x1BrcMode, HYPOT_BRC_NONE);
    EXPECT_EQ(td->x2BrcMode, HYPOT_BRC_NONE);
}

// failure branch: unsupported compute dtype (INT32) is rejected
TEST_F(HypotTilingArch22Test, tiling_failed_unsupported_dtype)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                     {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED, 0, "");
}

// ---------------- mixed-dtype dispatch: same scene-only TilingKey space ----------------

// [8,1] fp16 x [1,8] bf16 -> 64 elems: small cross broadcast routes to VECIDX for mixed exactly
// like same-dtype (the kernel stages each input in its own raw dtype)
TEST_F(HypotTilingArch22Test, tiling_mixed_small_accepted)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{8, 1}, {8, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1, 8}, {1, 8}}, ge::DT_BF16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, kKeyVecIdx);
    const HypotTilingData* td = TD(info);
    EXPECT_EQ(td->totalLength, 64U);
    EXPECT_EQ(td->x1Numel, 8U);
    EXPECT_EQ(td->x2Numel, 8U);
}

// mixed FP16/BF16 above 256 elements is a normal launch now (the kernel expands each input from
// its own raw dtype), not a failure: [32,32] fp16 x [32,1] bf16 -> 1024 elems -> MULTI_DIM_VECIDX
TEST_F(HypotTilingArch22Test, tiling_mixed_over_256_accepted)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{32, 32}, {32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{32, 1}, {32, 1}}, ge::DT_BF16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{32, 32}, {32, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, kKeyVecIdx);
    const HypotTilingData* td = TD(info);
    EXPECT_EQ(td->totalLength, 1024U);
    EXPECT_EQ(td->x1Numel, 1024U);
    EXPECT_EQ(td->x2Numel, 32U);
}

// mixed FP32 x FP16 large contiguous launch: ONE_DIM scene, same scene-only key
TEST_F(HypotTilingArch22Test, tiling_mixed_fp32_fp16_onedim)
{
    optiling::HypotCompileInfoArch22 compileInfo = {kCoreNum, kUbSize};
    gert::TilingContextPara para("Hypot",
                                 {
                                     {{{4096}, {4096}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{4096}, {4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{4096}, {4096}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 &compileInfo);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    EXPECT_EQ(info.tilingKey, kKeyOneDim);
    const HypotTilingData* td = TD(info);
    EXPECT_EQ(td->totalLength, 4096U);
}
