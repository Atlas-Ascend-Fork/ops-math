/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "weight_quant_preprocess_registry.h"
#include "common/inc/op_host/op_const_def_math.h"

#include <inttypes.h>

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include <initializer_list>
#include <string>
#include "log/log.h"
#include <inttypes.h>

#include "math/bitwise_and/op_api/bitwiseand.h"
#include "math/bitwise_or/op_api/bitwise_or.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/pad.h"
#include "aclnn_kernels/transpose.h"

#define LOGE_WITH_SCENARIO(err, fmt, ...)                                              \
    OP_LOGE(err, "[NpuArch=%u, DataFlow=%s] " fmt, static_cast<uint32_t>(ctx.npuArch), \
            QuantDataFlowToString(ctx.dataFlow), ##__VA_ARGS__)

namespace {

static constexpr size_t DIMS_1 = 1;
static constexpr size_t DIMS_2 = 2;
static constexpr size_t DIMS_3 = 3;
static constexpr size_t DIMS_4 = 4;
static constexpr size_t IDX_0 = 0;
static constexpr size_t IDX_1 = 1;
static constexpr size_t IDX_2 = 2;
static constexpr size_t IDX_3 = 3;
static constexpr size_t IDX_4 = 4;
static constexpr size_t SINGLE = 1;
static constexpr size_t DOUBLE = 2;
static constexpr int64_t KGROUP_SIZE_MX = 32;
static constexpr int64_t KGROUP_SIZE_HIF4 = 64;
static constexpr int64_t KGROUP_SIZE_INT4 = 0;
static constexpr int64_t KGROUP_SIZE_128 = 128;
static constexpr int64_t KGROUP_SIZE_256 = 256;
static constexpr int64_t NZ_16 = 16;
static constexpr int64_t NZ_C0_16 = 16;
static constexpr int64_t NZ_C0_32 = 32;
static constexpr int64_t NZ_C0_64 = 64;

static constexpr int64_t B4_NUMS_PER_BYTE = 2; // 1 字节打包 2 个 4-bit 值（INT4/FP4 通用）
// CheckOutWeightNz 的 weight 转置状态：决定 NZ 块序（转置 -> K 块在前；非转置 -> N 块在前）
static constexpr bool TRANS_WEIGHT = true;
static constexpr bool NON_TRANS_WEIGHT = false;

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    if (b == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CeilDiv divisor b must not be zero.");
        return INT64_MIN;
    }
    return (a + b - 1) / b;
}

static bool IsMMMxA8W4DataFlow(QuantContext& ctx)
{
    auto weightDtype = ctx.weight->GetDataType();
    auto scaleDtype = ctx.weightScale->GetDataType();
    auto weightViewShape = ctx.weight->GetViewShape();

    if (weightDtype == op::DataType::DT_FLOAT4_E2M1 && scaleDtype == op::DataType::DT_FLOAT8_E8M0 &&
        ctx.xDtype == op::DataType::DT_FLOAT8_E4M3FN && ctx.xScaleDtype == op::DataType::DT_FLOAT8_E8M0 &&
        weightViewShape.GetDimNum() == DIMS_2) {
        ctx.dataFlow = QuantDataFlow::MM_MX_A8W4;
        return true;
    }
    return false;
}

static bool IsGMMMxA8W4DataFlow(QuantContext& ctx)
{
    auto weightDtype = ctx.weight->GetDataType();
    auto scaleDtype = ctx.weightScale->GetDataType();
    auto weightViewShape = ctx.weight->GetViewShape();
    if (weightDtype == op::DataType::DT_FLOAT4_E2M1 && scaleDtype == op::DataType::DT_FLOAT8_E8M0 &&
        ctx.xDtype == op::DataType::DT_FLOAT8_E4M3FN && ctx.xScaleDtype == op::DataType::DT_FLOAT8_E8M0 &&
        weightViewShape.GetDimNum() == DIMS_3) {
        ctx.dataFlow = QuantDataFlow::GMM_MX_A8W4;
        return true;
    }
    return false;
}

// ===== A16S4/A16F4 judge 判定件 =====
// judge 按 scale 模式 × 转置 × 出 format 拆分，以下判定件负责组装；dtype 通过模板参数传入
// x dtype fp16/bf16 + weight 2D；A16W4 无 xScale 语义，xScale 由各 judge 以 IsXScaleUndefined 统一排除
template <op::DataType WType>
static bool IsA16W4Base(const QuantContext& ctx)
{
    bool xDtypeMatch = (ctx.xDtype == op::DataType::DT_FLOAT16 || ctx.xDtype == op::DataType::DT_BF16);
    return ctx.weight->GetDataType() == WType && xDtypeMatch && ctx.weight->GetViewShape().GetDimNum() == DIMS_2;
}

static bool IsXScaleUndefined(const QuantContext& ctx) { return ctx.xScaleDtype == op::DataType::DT_UNDEFINED; }

// 转置探测：最后两维 strides [1, size(-2)] 即转置（与 CheckWeightTrans 同一判定）
static bool IsWeightLastTwoDimsTrans(const QuantContext& ctx)
{
    const auto& viewShape = ctx.weight->GetViewShape();
    const auto& viewStrides = ctx.weight->GetViewStrides();
    int64_t dimNum = static_cast<int64_t>(viewShape.GetDimNum());
    return viewStrides[dimNum - IDX_2] == 1 && viewStrides[dimNum - IDX_1] == viewShape.GetDim(dimNum - IDX_2);
}

// 非转置探测：最后两维 strides [size(-1), 1]；与 IsWeightLastTwoDimsTrans 互补，均不命中 -> UNSUPPORTED
static bool IsWeightLastTwoDimsNonTrans(const QuantContext& ctx)
{
    const auto& viewShape = ctx.weight->GetViewShape();
    const auto& viewStrides = ctx.weight->GetViewStrides();
    int64_t dimNum = static_cast<int64_t>(viewShape.GetDimNum());
    return viewStrides[dimNum - IDX_1] == 1 && viewStrides[dimNum - IDX_2] == viewShape.GetDim(dimNum - IDX_1);
}

// scale 模式判定：per-tensor 单元素 / per-channel 1D 或 [1,N] 且 numel>1 / per-group 2D 且 G>1
static bool IsScalePerTensor(const QuantContext& ctx) { return ctx.weightScale->GetViewShape().GetShapeSize() == 1; }

static bool IsScalePerChannel(const QuantContext& ctx)
{
    const auto& scaleViewShape = ctx.weightScale->GetViewShape();
    int64_t scaleDim = static_cast<int64_t>(scaleViewShape.GetDimNum());
    return (scaleDim == DIMS_1 || (scaleDim == DIMS_2 && scaleViewShape.GetDim(IDX_0) == 1)) &&
           scaleViewShape.GetShapeSize() > 1;
}

static bool IsScalePerGroup(const QuantContext& ctx)
{
    const auto& scaleViewShape = ctx.weightScale->GetViewShape();
    return scaleViewShape.GetDimNum() == DIMS_2 && scaleViewShape.GetDim(IDX_0) > 1;
}

// out format 判定：NZ_C0_16 分形转换；ND/NCL 出（直拷）由转置 judge 命中，无需单独判定
static bool IsOutWeightNzC016(const QuantContext& ctx)
{
    return ctx.outWeight != nullptr && ctx.outWeight->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ_C0_16;
}

// A16S4 per-tensor：转置不验证（ND 直拷物理透传与转置无关），仅 ND 出
static bool IsMMA16S4PerTensorDataFlow(QuantContext& ctx)
{
    if (IsA16W4Base<op::DataType::DT_INT4>(ctx) && IsXScaleUndefined(ctx) && IsScalePerTensor(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16S4_PERTENSOR;
        return true;
    }
    return false;
}

// A16S4 per-channel：转置 → ND 直拷；非转置仅支持 NZ_C0_16 出（ND 出不支持，与 A16F4 对齐）
static bool IsMMA16S4PerChannelTransDataFlow(QuantContext& ctx)
{
    if (IsA16W4Base<op::DataType::DT_INT4>(ctx) && IsXScaleUndefined(ctx) && IsScalePerChannel(ctx) &&
        IsWeightLastTwoDimsTrans(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16S4_PERCHANNEL;
        return true;
    }
    return false;
}

static bool IsMMA16S4PerChannelNonTransNzDataFlow(QuantContext& ctx)
{
    if (IsA16W4Base<op::DataType::DT_INT4>(ctx) && IsXScaleUndefined(ctx) && IsScalePerChannel(ctx) &&
        !IsWeightLastTwoDimsTrans(ctx) && IsOutWeightNzC016(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16S4_PERCHANNEL;
        return true;
    }
    return false;
}

// A16S4 per-group：同 per-channel 的两路拆分（转置 ND 直拷 / 非转置 NZ_C0_16 转换）
static bool IsMMA16S4PerGroupTransDataFlow(QuantContext& ctx)
{
    if (IsA16W4Base<op::DataType::DT_INT4>(ctx) && IsXScaleUndefined(ctx) && IsScalePerGroup(ctx) &&
        IsWeightLastTwoDimsTrans(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16S4_PERGROUP;
        return true;
    }
    return false;
}

static bool IsMMA16S4PerGroupNonTransNzDataFlow(QuantContext& ctx)
{
    if (IsA16W4Base<op::DataType::DT_INT4>(ctx) && IsXScaleUndefined(ctx) && IsScalePerGroup(ctx) &&
        !IsWeightLastTwoDimsTrans(ctx) && IsOutWeightNzC016(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16S4_PERGROUP;
        return true;
    }
    return false;
}

// A16F4 per-group：FP4 weight + per-group scale [G, N]（G > 1）；NZ only，转置由 checks 的
// CheckWeightNotTrans 拒绝（保留明确报错）
static bool IsMMA16F4PerGroupDataFlow(QuantContext& ctx)
{
    auto scaleDtype = ctx.weightScale->GetDataType();
    bool scaleDtypeMatch = (scaleDtype == op::DataType::DT_FLOAT16 || scaleDtype == op::DataType::DT_BF16);

    if (IsA16W4Base<op::DataType::DT_FLOAT4_E2M1>(ctx) && scaleDtypeMatch && IsXScaleUndefined(ctx) &&
        IsScalePerGroup(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16F4_PERGROUP;
        return true;
    }
    return false;
}

// A16MXFP4 公共判定：FP4 weight + MX scale（E8M0，2D [K/32, N]，与 wqbmmv2 MX kernel 约定一致）
static bool IsMMA16MXF4Base(const QuantContext& ctx)
{
    auto scaleDtype = ctx.weightScale->GetDataType();

    return IsA16W4Base<op::DataType::DT_FLOAT4_E2M1>(ctx) && scaleDtype == op::DataType::DT_FLOAT8_E8M0 &&
           IsXScaleUndefined(ctx) && ctx.weightScale->GetViewShape().GetDimNum() == DIMS_2;
}

// A16MXFP4 转置（末两维严格转置）→ ND 直拷（wqbmmv2 MX kernel 支持 ND 转置输入）
static bool IsMMA16MXF4TransDataFlow(QuantContext& ctx)
{
    if (IsMMA16MXF4Base(ctx) && IsWeightLastTwoDimsTrans(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16MXF4;
        return true;
    }
    return false;
}

// A16MXFP4 非转置 + NZ_C0_16 出 → ND→NZ 分形转换（NZ 出仅支持非转置，转置由转置条目的
// OUT_WEIGHT_ND_CHECKS 拒绝）
static bool IsMMA16MXF4NonTransNzDataFlow(QuantContext& ctx)
{
    if (IsMMA16MXF4Base(ctx) && !IsWeightLastTwoDimsTrans(ctx) && IsOutWeightNzC016(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16MXF4;
        return true;
    }
    return false;
}

// ===== A16W8 judge 判定件 =====
// A16W8：x fp16/bf16 + weight INT8（1 字节元素，无 4-bit 打包约束）；
// scale 与 x dtype 一致（fp16/bf16），xScale 无语义统一排除
static bool IsA16INT8Base(const QuantContext& ctx)
{
    bool xDtypeMatch = (ctx.xDtype == op::DataType::DT_FLOAT16 || ctx.xDtype == op::DataType::DT_BF16);
    return ctx.weight->GetDataType() == op::DataType::DT_INT8 && xDtypeMatch &&
           ctx.weight->GetViewShape().GetDimNum() == DIMS_2;
}

static bool IsA16INT8ScaleDtypeMatch(const QuantContext& ctx)
{
    auto scaleDtype = ctx.weightScale->GetDataType();
    return scaleDtype == op::DataType::DT_FLOAT16 || scaleDtype == op::DataType::DT_BF16;
}

// A16INT8 per-tensor：转置/非转置均 ND 直拷（wqbmmv2 INT8-NZ 不支持 per-tensor，无 NZ 路径）
static bool IsMMA16INT8PerTensorDataFlow(QuantContext& ctx)
{
    if (IsA16INT8Base(ctx) && IsA16INT8ScaleDtypeMatch(ctx) && IsXScaleUndefined(ctx) && IsScalePerTensor(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16INT8_PERTENSOR;
        return true;
    }
    return false;
}

// A16INT8 per-channel：转置/非转置均 ND 直拷（物理透传，weight ND 场景不提供 NZ 转换路径）
static bool IsMMA16INT8PerChannelDataFlow(QuantContext& ctx)
{
    if (IsA16INT8Base(ctx) && IsA16INT8ScaleDtypeMatch(ctx) && IsXScaleUndefined(ctx) && IsScalePerChannel(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16INT8_PERCHANNEL;
        return true;
    }
    return false;
}

// A16INT8 per-group：转置/非转置均 ND 直拷（wqbmmv2 INT8-NZ 不支持 per-group）
static bool IsMMA16INT8PerGroupDataFlow(QuantContext& ctx)
{
    if (IsA16INT8Base(ctx) && IsA16INT8ScaleDtypeMatch(ctx) && IsXScaleUndefined(ctx) && IsScalePerGroup(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16INT8_PERGROUP;
        return true;
    }
    return false;
}

// A16FP8HIF8 per-channel：weight FLOAT8_E4M3FN/HIFLOAT8，仅 per-channel（C 量化），
// 转置/非转置均 ND 直拷（wqbmmv2 FP8 不支持 weightNZ）；offset 无透传 process，约束下放下游 wqbmmv2 拦截
static bool IsMMA16FP8HIF8PerChannelDataFlow(QuantContext& ctx)
{
    auto weightDtype = ctx.weight->GetDataType();
    bool weightDtypeMatch = (weightDtype == op::DataType::DT_FLOAT8_E4M3FN || weightDtype == op::DataType::DT_HIFLOAT8);
    bool xDtypeMatch = (ctx.xDtype == op::DataType::DT_FLOAT16 || ctx.xDtype == op::DataType::DT_BF16);
    auto scaleDtype = ctx.weightScale->GetDataType();
    bool scaleDtypeMatch = (scaleDtype == op::DataType::DT_FLOAT16 || scaleDtype == op::DataType::DT_BF16);

    if (weightDtypeMatch && xDtypeMatch && scaleDtypeMatch && IsXScaleUndefined(ctx) &&
        ctx.weight->GetViewShape().GetDimNum() == DIMS_2 && IsScalePerChannel(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16FP8HIF8_PERCHANNEL;
        return true;
    }
    return false;
}

// ===== A8F4 T-CG judge 判定件 =====
// A8F4 T-CG：FP4 weight + FP16/BF16 per-group scale {ceildiv(K,32), N}（G>1）+ FP8 x；
// 与 MX（scale=E8M0）、A16（x=F16/BF16）按 dtype 互斥
static bool IsMMA8F4TcgBase(const QuantContext& ctx)
{
    auto scaleDtype = ctx.weightScale->GetDataType();
    bool scaleDtypeMatch = (scaleDtype == op::DataType::DT_FLOAT16 || scaleDtype == op::DataType::DT_BF16);

    return ctx.weight->GetDataType() == op::DataType::DT_FLOAT4_E2M1 && scaleDtypeMatch &&
           ctx.xDtype == op::DataType::DT_FLOAT8_E4M3FN && IsXScaleUndefined(ctx) &&
           ctx.weight->GetViewShape().GetDimNum() == DIMS_2 && IsScalePerGroup(ctx);
}

// 转置 → ND 直拷（下游 T-CG ND 入口要求转置输入）
static bool IsMMA8F4TcgTransDataFlow(QuantContext& ctx)
{
    if (IsMMA8F4TcgBase(ctx) && IsWeightLastTwoDimsTrans(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A8F4_TCG;
        return true;
    }
    return false;
}

// 非转置 → ND→NZ 分形转换；outWeight 的 C0_32 format 由 checks 的 CheckOutWeightNz 校验（judge 不探测 format）
static bool IsMMA8F4TcgNonTransDataFlow(QuantContext& ctx)
{
    if (IsMMA8F4TcgBase(ctx) && IsWeightLastTwoDimsNonTrans(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A8F4_TCG;
        return true;
    }
    return false;
}

static bool IsMMMxA16W4DataFlow(QuantContext& ctx)
{
    auto weightDtype = ctx.weight->GetDataType();
    auto scaleDtype = ctx.weightScale->GetDataType();
    auto weightViewShape = ctx.weight->GetViewShape();
    if (weightDtype == op::DataType::DT_FLOAT4_E2M1 && scaleDtype == op::DataType::DT_FLOAT8_E8M0 &&
        (ctx.xDtype == op::DataType::DT_FLOAT16 || ctx.xDtype == op::DataType::DT_BF16) &&
        ctx.xScaleDtype == op::DataType::DT_UNDEFINED && weightViewShape.GetDimNum() == DIMS_2) {
        ctx.dataFlow = QuantDataFlow::MM_A16MXF4;
        return true;
    }
    return false;
}

static bool IsMMMxA16HIF4DataFlow(QuantContext& ctx)
{
    auto weightDtype = ctx.weight->GetDataType();
    auto scaleDtype = ctx.weightScale->GetDataType();
    auto weightViewShape = ctx.weight->GetViewShape();
    if (weightDtype == Ops::Math::DT_HIFLOAT4 && scaleDtype == Ops::Math::DT_HIFLOAT4_SCALE &&
        (ctx.xDtype == op::DataType::DT_FLOAT16 || ctx.xDtype == op::DataType::DT_BF16) &&
        ctx.xScaleDtype == op::DataType::DT_UNDEFINED && weightViewShape.GetDimNum() == DIMS_2) {
        ctx.dataFlow = QuantDataFlow::MM_MX_A16HIF4;
        return true;
    }
    return false;
}

static bool IsMMC064A16W4DataFlow(QuantContext& ctx)
{
    auto weightDtype = ctx.weight->GetDataType();
    auto scaleDtype = ctx.weightScale->GetDataType();
    auto weightViewShape = ctx.weight->GetViewShape();
    auto scaleViewShape = ctx.weightScale->GetViewShape();
    if ((weightDtype != op::DataType::DT_INT4 && weightDtype != op::DataType::DT_FLOAT4_E2M1) ||
        scaleDtype != ctx.xDtype || (ctx.xDtype != op::DataType::DT_FLOAT16 && ctx.xDtype != op::DataType::DT_BF16) ||
        weightViewShape.GetDimNum() != DIMS_2) {
        return false;
    }

    int64_t k = weightViewShape.GetDim(IDX_0);
    int64_t n = weightViewShape.GetDim(IDX_1);
    bool scale1d = scaleViewShape.GetDimNum() == DIMS_1 && scaleViewShape.GetDim(IDX_0) == n && n != SINGLE;
    bool scale2d = scaleViewShape.GetDimNum() == DIMS_2 &&
                   (scaleViewShape.GetDim(IDX_0) == SINGLE && scaleViewShape.GetDim(IDX_1) == n);
    bool matched = (scale1d || scale2d) && ctx.weightOffsetOptional == nullptr && k % NZ_C0_64 == 0 &&
                   n % NZ_C0_64 == 0;
    if (matched) {
        ctx.dataFlow = QuantDataFlow::MM_C064_A16W4;
    }
    return matched;
}

static bool IsMMA16S4PerTensorDataFlow_9201(QuantContext& ctx)
{
    auto weightViewShape = ctx.weight->GetViewShape();
    auto scaleViewShape = ctx.weightScale->GetViewShape();
    bool matched = ctx.weight->GetDataType() == op::DataType::DT_INT4 && ctx.weightScale->GetDataType() == ctx.xDtype &&
                   (ctx.xDtype == op::DataType::DT_FLOAT16 || ctx.xDtype == op::DataType::DT_BF16) &&
                   weightViewShape.GetDimNum() == DIMS_2 && scaleViewShape.GetDimNum() == DIMS_1 &&
                   scaleViewShape.GetDim(IDX_0) == SINGLE;
    if (matched) {
        ctx.dataFlow = QuantDataFlow::MM_A16S4_PERTENSOR;
    }
    return matched;
}

static bool IsMMA16S4PerChannelDataFlow(QuantContext& ctx)
{
    auto weightViewShape = ctx.weight->GetViewShape();
    if (ctx.weight->GetDataType() != op::DataType::DT_INT4 || ctx.weightScale->GetDataType() != ctx.xDtype ||
        (ctx.xDtype != op::DataType::DT_FLOAT16 && ctx.xDtype != op::DataType::DT_BF16) ||
        weightViewShape.GetDimNum() != DIMS_2) {
        return false;
    }
    int64_t n = weightViewShape.GetDim(IDX_1);
    auto scaleViewShape = ctx.weightScale->GetViewShape();
    bool scale1d = scaleViewShape.GetDimNum() == DIMS_1 && scaleViewShape.GetDim(IDX_0) == n && n != SINGLE;
    bool scale2d = scaleViewShape.GetDimNum() == DIMS_2 &&
                   (scaleViewShape.GetDim(IDX_0) == SINGLE && scaleViewShape.GetDim(IDX_1) == n);
    bool matched = scale1d || scale2d;
    if (matched) {
        ctx.dataFlow = QuantDataFlow::MM_A16S4_PERCHANNEL;
    }
    return matched;
}

static bool IsMMC064PerGroupA16W4DataFlow(QuantContext& ctx)
{
    auto weightDtype = ctx.weight->GetDataType();
    auto weightViewShape = ctx.weight->GetViewShape();
    auto scaleViewShape = ctx.weightScale->GetViewShape();
    if ((weightDtype != op::DataType::DT_INT4 && weightDtype != op::DataType::DT_FLOAT4_E2M1) ||
        ctx.weightScale->GetDataType() != ctx.xDtype ||
        (ctx.xDtype != op::DataType::DT_FLOAT16 && ctx.xDtype != op::DataType::DT_BF16) ||
        weightViewShape.GetDimNum() != DIMS_2 || scaleViewShape.GetDimNum() != DIMS_2) {
        return false;
    }
    int64_t k = weightViewShape.GetDim(IDX_0);
    int64_t n = weightViewShape.GetDim(IDX_1);
    bool perGroupShape = scaleViewShape.GetDim(IDX_1) == n && scaleViewShape.GetDim(IDX_0) != SINGLE;
    bool c064Group = ctx.kGroupSize == KGROUP_SIZE_HIF4 || ctx.kGroupSize == KGROUP_SIZE_128;
    bool matched = perGroupShape && c064Group && ctx.weightOffsetOptional == nullptr && k % NZ_C0_64 == 0 &&
                   n % NZ_C0_64 == 0 && k % ctx.kGroupSize == 0;
    if (matched) {
        ctx.dataFlow = QuantDataFlow::MM_C064_PerG_A16W4;
    }
    return matched;
}

static bool IsMMA16S4PerGroupDataFlow(QuantContext& ctx)
{
    auto weightViewShape = ctx.weight->GetViewShape();
    auto scaleViewShape = ctx.weightScale->GetViewShape();
    if (ctx.weight->GetDataType() != op::DataType::DT_INT4 || ctx.weightScale->GetDataType() != ctx.xDtype ||
        (ctx.xDtype != op::DataType::DT_FLOAT16 && ctx.xDtype != op::DataType::DT_BF16) ||
        weightViewShape.GetDimNum() != DIMS_2 || scaleViewShape.GetDimNum() != DIMS_2) {
        return false;
    }
    int64_t n = weightViewShape.GetDim(IDX_1);
    bool matched = scaleViewShape.GetDim(IDX_1) == n && scaleViewShape.GetDim(IDX_0) != SINGLE;
    if (matched) {
        ctx.dataFlow = QuantDataFlow::MM_A16S4_PERGROUP;
    }
    return matched;
}

static bool IsMMA16Fp4PerGroupDataFlow(QuantContext& ctx)
{
    auto weightViewShape = ctx.weight->GetViewShape();
    auto scaleViewShape = ctx.weightScale->GetViewShape();
    if (ctx.weight->GetDataType() != op::DataType::DT_FLOAT4_E2M1 || ctx.weightScale->GetDataType() != ctx.xDtype ||
        (ctx.xDtype != op::DataType::DT_FLOAT16 && ctx.xDtype != op::DataType::DT_BF16) ||
        weightViewShape.GetDimNum() != DIMS_2 || scaleViewShape.GetDimNum() != DIMS_2) {
        return false;
    }
    int64_t n = weightViewShape.GetDim(IDX_1);
    bool matched = scaleViewShape.GetDim(IDX_1) == n && scaleViewShape.GetDim(IDX_0) != SINGLE;
    if (matched) {
        ctx.dataFlow = QuantDataFlow::MM_A16F4_PERGROUP;
    }
    return matched;
}

static bool IsWeightNonTransposed(const aclTensor* weight)
{
    auto viewShape = weight->GetViewShape();
    auto viewStrides = weight->GetViewStrides();
    size_t dimNum = viewShape.GetDimNum();
    if (dimNum < DIMS_2) {
        return false;
    }
    size_t lastIdx = dimNum - IDX_1;
    size_t secondLastIdx = dimNum - IDX_2;
    return IsContiguous(weight) && viewStrides[lastIdx] == SINGLE &&
           viewStrides[secondLastIdx] == viewShape.GetDim(lastIdx);
}

static aclnnStatus CheckWeightNotEmpty(const QuantContext& ctx)
{
    OP_CHECK(
        !ctx.weight->IsEmpty(),
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            "weight_quant_preprocess", "weight", std::to_string(ctx.weight->GetViewShape().GetShapeSize()).c_str(),
            (std::string("weight must not be empty tensor when dataFlow is ") + QuantDataFlowToString(ctx.dataFlow))
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckWeightFormatND(const QuantContext& ctx)
{
    auto weightFormat = ctx.weight->GetStorageFormat();
    OP_CHECK(weightFormat == op::Format::FORMAT_ND || weightFormat == op::Format::FORMAT_NCL,
             OP_LOGE_FOR_INVALID_FORMAT("weight_quant_preprocess", "weight", op::ToString(weightFormat).GetString(),
                                        "ND or NCL"),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckWeightTrans(const QuantContext& ctx)
{
    auto viewShape = ctx.weight->GetViewShape();
    auto viewStrides = ctx.weight->GetViewStrides();

    int64_t dimNum = static_cast<int64_t>(viewShape.GetDimNum());
    int64_t lastIdx = dimNum - IDX_1;
    int64_t secondLastIdx = dimNum - IDX_2;

    OP_CHECK(
        viewStrides[secondLastIdx] == 1 && viewStrides[lastIdx] == viewShape.GetDim(secondLastIdx),
        OP_LOGE_FOR_INVALID_STRIDE(
            "weight_quant_preprocess", "weight", op::ToString(ctx.weight->GetViewStrides()).GetString(),
            (std::string("last two dims stride [1, ") + std::to_string(viewShape.GetDim(secondLastIdx)) + "]").c_str()),
        return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// A16S4 NZ 路径（per-channel / per-group）仅支持非转置 weight，转置返回错误
static aclnnStatus CheckWeightNotTrans(const QuantContext& ctx)
{
    auto viewShape = ctx.weight->GetViewShape();
    auto viewStrides = ctx.weight->GetViewStrides();

    int64_t dimNum = static_cast<int64_t>(viewShape.GetDimNum());
    int64_t lastIdx = dimNum - IDX_1;
    int64_t secondLastIdx = dimNum - IDX_2;

    OP_CHECK(viewStrides[lastIdx] == 1 && viewStrides[secondLastIdx] == viewShape.GetDim(lastIdx),
             OP_LOGE_FOR_INVALID_STRIDE(
                 "weight_quant_preprocess", "weight", op::ToString(ctx.weight->GetViewStrides()).GetString(),
                 (std::string("transposed weight is not supported for ") + QuantDataFlowToString(ctx.dataFlow) +
                  ", last two dims stride should be [" + std::to_string(viewShape.GetDim(lastIdx)) + ", 1]")
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// 紧凑 4-bit（INT4/FP4）每字节打包 2 个值：打包维（连续维）必须为偶数
// 非转置 [K,N] 沿 N 打包 -> N 为偶数；转置 [K,N] strides [1,K] 沿 K 打包 -> K 为偶数
static aclnnStatus CheckWeightPackingDimEven(const QuantContext& ctx)
{
    auto viewShape = ctx.weight->GetViewShape();
    auto viewStrides = ctx.weight->GetViewStrides();

    int64_t dimNum = static_cast<int64_t>(viewShape.GetDimNum());
    int64_t lastIdx = dimNum - IDX_1;
    int64_t secondLastIdx = dimNum - IDX_2;
    bool isTransposed = (viewStrides[secondLastIdx] == 1 && viewShape.GetDim(secondLastIdx) > 1);
    int64_t packingDim = isTransposed ? viewShape.GetDim(secondLastIdx) : viewShape.GetDim(lastIdx);

    OP_CHECK(packingDim % B4_NUMS_PER_BYTE == 0,
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                 "weight_quant_preprocess", "weight", op::ToString(viewShape).GetString(),
                 (std::string("the packing dim of 4-bit weight must be even when dataFlow is ") +
                  QuantDataFlowToString(ctx.dataFlow))
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// 紧凑 4-bit（INT4/FP4，1 字节打包 2 个值）weight 视图的物理寻址前提：
// 视图必须连续或末两维严格转置（其余 strides 模式会按错误的打包维寻址），
// 且 weight/outWeight 的 viewOffset 须为偶数（奇数偏移落在半字节边界，物理寻址错位）。
// ND 直拷（按字节物理透传）与 NZ 转换（TransData 搬运）都依赖此前提
static aclnnStatus CheckWeight4BitPackedView(const QuantContext& ctx)
{
    if (ctx.weight->GetDataType() != op::DataType::DT_INT4 &&
        ctx.weight->GetDataType() != op::DataType::DT_FLOAT4_E2M1) {
        return ACLNN_SUCCESS;
    }
    // outWeight 为空属参数错误，由后续 CheckOutWeightSameAsInput/CheckOutWeightSameBase
    // 报告；此处提前返回避免空指针
    if (ctx.outWeight == nullptr) {
        return ACLNN_SUCCESS;
    }
    auto viewShape = ctx.weight->GetViewShape();
    auto viewStrides = ctx.weight->GetViewStrides();
    int64_t dimNum = static_cast<int64_t>(viewShape.GetDimNum());
    int64_t lastIdx = dimNum - IDX_1;
    int64_t secondLastIdx = dimNum - IDX_2;
    bool isStrictTransposed = (viewStrides[secondLastIdx] == 1 && viewShape.GetDim(secondLastIdx) > 1 &&
                               viewStrides[lastIdx] == viewShape.GetDim(secondLastIdx));
    OP_CHECK(IsContiguous(ctx.weight) || isStrictTransposed,
             OP_LOGE_FOR_INVALID_STRIDE(
                 "weight_quant_preprocess", "weight", op::ToString(viewStrides).GetString(),
                 (std::string("contiguous or strictly transposed at last two dims when dataFlow is ") +
                  QuantDataFlowToString(ctx.dataFlow))
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(
        ctx.weight->GetViewOffset() % B4_NUMS_PER_BYTE == 0 && ctx.outWeight->GetViewOffset() % B4_NUMS_PER_BYTE == 0,
        OP_LOGE_FOR_INVALID_VALUE("weight_quant_preprocess", "weight/outWeight viewOffset",
                                  (std::to_string(ctx.weight->GetViewOffset()) + std::string("/") +
                                   std::to_string(ctx.outWeight->GetViewOffset()))
                                      .c_str(),
                                  "even (4-bit packs 2 values per byte, odd offset misaligns the nibble)"),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckWeightOffsetOptionalNull(const QuantContext& ctx)
{
    OP_CHECK(ctx.weightOffsetOptional == nullptr,
             LOGE_WITH_SCENARIO(ACLNN_ERR_PARAM_INVALID, "weightOffsetOptional must be nullptr."),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(ctx.outWeightOffsetOptional == nullptr,
             LOGE_WITH_SCENARIO(ACLNN_ERR_PARAM_INVALID, "outWeightOffsetOptional must be nullptr."),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutWeightNzStorageDim(const QuantContext& ctx)
{
    auto weightViewShape = ctx.weight->GetViewShape();
    auto outStorageShape = ctx.outWeight->GetStorageShape();
    OP_CHECK(outStorageShape.GetDimNum() == weightViewShape.GetDimNum() + 2,
             OP_LOGE_FOR_INVALID_SHAPEDIM("weight_quant_preprocess", "outWeight",
                                          std::to_string(outStorageShape.GetDimNum()).c_str(),
                                          std::to_string(weightViewShape.GetDimNum() + 2).c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

template <int64_t nzC0, op::Format outWeightFormat>
static aclnnStatus CheckOutWeightTransNz(const QuantContext& ctx)
{
    auto outFormat = ctx.outWeight->GetStorageFormat();
    OP_CHECK(outFormat == outWeightFormat,
             OP_LOGE_FOR_INVALID_FORMAT("weight_quant_preprocess", "outWeight", op::ToString(outFormat).GetString(),
                                        op::ToString(outWeightFormat).GetString()),
             return ACLNN_ERR_PARAM_INVALID);

    auto outStorageShape = ctx.outWeight->GetStorageShape();
    auto weightViewShape = ctx.weight->GetViewShape();
    size_t outStorageDim = outStorageShape.GetDimNum();
    size_t viewDim = weightViewShape.GetDimNum();

    int64_t k = weightViewShape.GetDim(viewDim - IDX_2);
    int64_t n = weightViewShape.GetDim(viewDim - IDX_1);
    int64_t expectedNBlocks = CeilDiv(n, NZ_16);
    int64_t expectedKBlocks = CeilDiv(k, nzC0);

    // NZ_C0_16（A16S4/A16F4 紧凑 4-bit）物理布局为 [N/16, K/nzC0, 16, nzC0]（N 块在前）；
    // NZ_C0_32（A8W4）保持 master 的 [K/nzC0, N/16, 16, nzC0]（K 块在前）布局约定
    bool nFirst = (outWeightFormat == op::Format::FORMAT_FRACTAL_NZ_C0_16);
    int64_t expectedBlocks4 = nFirst ? expectedNBlocks : expectedKBlocks;
    int64_t expectedBlocks3 = nFirst ? expectedKBlocks : expectedNBlocks;
    const char* layoutDesc = nFirst ? "{ceildiv(N, 16), ceildiv(K, nzC0), 16, nzC0}" :
                                      "{ceildiv(K, nzC0), ceildiv(N, 16), 16, nzC0}";
    OP_CHECK(
        outStorageShape.GetDim(outStorageDim - IDX_4) == expectedBlocks4 &&
            outStorageShape.GetDim(outStorageDim - IDX_3) == expectedBlocks3 &&
            outStorageShape.GetDim(outStorageDim - IDX_2) == NZ_16 &&
            outStorageShape.GetDim(outStorageDim - IDX_1) == nzC0,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            "weight_quant_preprocess", "outWeight, weight",
            (op::ToString(outStorageShape).GetString() + std::string(", ") + op::ToString(weightViewShape).GetString())
                .c_str(),
            (std::string("outWeight storage shape last four dims must be ") + layoutDesc +
             std::string(" when dataFlow is ") + QuantDataFlowToString(ctx.dataFlow))
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// out weight NZ 校验：format 匹配 + storageShape 末四维 == {ceildiv(c0轴, nzC0), ceildiv(另一轴, 16), 16, nzC0}
// transWeight 决定块序：非转置 N 块在前，转置 K 块在前
template <int64_t nzC0, op::Format outWeightFormat, bool transWeight>
static aclnnStatus CheckOutWeightNz(const QuantContext& ctx)
{
    auto outFormat = ctx.outWeight->GetStorageFormat();
    OP_CHECK(outFormat == outWeightFormat,
             OP_LOGE_FOR_INVALID_FORMAT("weight_quant_preprocess", "outWeight", op::ToString(outFormat).GetString(),
                                        op::ToString(outWeightFormat).GetString()),
             return ACLNN_ERR_PARAM_INVALID);

    auto outStorageShape = ctx.outWeight->GetStorageShape();
    auto weightViewShape = ctx.weight->GetViewShape();
    size_t outStorageDim = outStorageShape.GetDimNum();
    size_t viewDim = weightViewShape.GetDimNum();

    int64_t k = weightViewShape.GetDim(viewDim - IDX_2);
    int64_t n = weightViewShape.GetDim(viewDim - IDX_1);

    int64_t c0AxisSize = transWeight ? k : n;
    int64_t otherAxisSize = transWeight ? n : k;
    const char* layoutDesc = transWeight ? "{ceildiv(K, nzC0), ceildiv(N, 16), 16, nzC0}" :
                                           "{ceildiv(N, nzC0), ceildiv(K, 16), 16, nzC0}";
    OP_CHECK(
        outStorageShape.GetDim(outStorageDim - IDX_4) == CeilDiv(c0AxisSize, nzC0) &&
            outStorageShape.GetDim(outStorageDim - IDX_3) == CeilDiv(otherAxisSize, NZ_16) &&
            outStorageShape.GetDim(outStorageDim - IDX_2) == NZ_16 &&
            outStorageShape.GetDim(outStorageDim - IDX_1) == nzC0,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            "weight_quant_preprocess", "outWeight, weight",
            (op::ToString(outStorageShape).GetString() + std::string(", ") + op::ToString(weightViewShape).GetString())
                .c_str(),
            (std::string("outWeight storage shape last four dims must be ") + layoutDesc +
             std::string(" when dataFlow is ") + QuantDataFlowToString(ctx.dataFlow))
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// ===== out 与 input 一致性归一校验 =====
// 轻量一致性：nullptr 一致（都空或都不空）→ 都非空时 out 非 empty → viewShape ==。
// 单独用于 NZ 出 weight 等 format/storageShape/dtype 合法异于输入的场景
static aclnnStatus CheckOutSameBase(const QuantContext& ctx, const aclTensor* input, const aclTensor* output,
                                    const char* inputName, const char* outputName)
{
    OP_CHECK((input == nullptr) == (output == nullptr),
             LOGE_WITH_SCENARIO(ACLNN_ERR_PARAM_NULLPTR, "%s and %s must be both nullptr or both not nullptr.",
                                outputName, inputName),
             return ACLNN_ERR_PARAM_NULLPTR);
    if (input == nullptr) {
        return ACLNN_SUCCESS;
    }
    OP_CHECK(!output->IsEmpty(),
             OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                 "weight_quant_preprocess", outputName, std::to_string(output->GetViewShape().GetShapeSize()).c_str(),
                 (std::string(outputName) + " must not be empty tensor when dataFlow is " +
                  QuantDataFlowToString(ctx.dataFlow))
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    auto inViewShape = input->GetViewShape();
    auto outViewShape = output->GetViewShape();
    OP_CHECK(outViewShape == inViewShape,
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                 "weight_quant_preprocess", (std::string(outputName) + ", " + inputName).c_str(),
                 (op::ToString(outViewShape).GetString() + std::string(", ") + op::ToString(inViewShape).GetString())
                     .c_str(),
                 (std::string(outputName) + " and " + inputName + " must have the same viewShape when dataFlow is " +
                  QuantDataFlowToString(ctx.dataFlow))
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

// 完整一致性（直拷透传）：Base 之上再要求 format/dtype/storageShape 全 ==、连续性一致
static aclnnStatus CheckOutSameAsInput(const QuantContext& ctx, const aclTensor* input, const aclTensor* output,
                                       const char* inputName, const char* outputName)
{
    auto status = CheckOutSameBase(ctx, input, output, inputName, outputName);
    if (status != ACLNN_SUCCESS || input == nullptr) {
        return status;
    }
    // 连续性一致：ViewCopy 直拷要求源连续，非连续输入 + 连续输出（或反之）在此拦截，
    // 避免错误延后到 process 阶段才以内部错误码暴露（weight 转置直拷两侧同为非连续，不受影响）
    OP_CHECK(
        IsContiguous(input) == IsContiguous(output),
        LOGE_WITH_SCENARIO(ACLNN_ERR_PARAM_INVALID, "%s and %s must have the same contiguity.", outputName, inputName),
        return ACLNN_ERR_PARAM_INVALID);
    auto inFormat = input->GetStorageFormat();
    auto outFormat = output->GetStorageFormat();
    OP_CHECK(outFormat == inFormat,
             OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
                 "weight_quant_preprocess", (std::string(outputName) + ", " + inputName).c_str(),
                 (op::ToString(outFormat).GetString() + std::string(", ") + op::ToString(inFormat).GetString()).c_str(),
                 (std::string(outputName) + " and " + inputName + " must have the same format when dataFlow is " +
                  QuantDataFlowToString(ctx.dataFlow))
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    auto inDtype = input->GetDataType();
    auto outDtype = output->GetDataType();
    OP_CHECK(outDtype == inDtype,
             OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                 "weight_quant_preprocess", (std::string(outputName) + ", " + inputName).c_str(),
                 (op::ToString(outDtype).GetString() + std::string(", ") + op::ToString(inDtype).GetString()).c_str(),
                 (std::string(outputName) + " and " + inputName + " must have the same dtype when dataFlow is " +
                  QuantDataFlowToString(ctx.dataFlow))
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    auto inStorageShape = input->GetStorageShape();
    auto outStorageShape = output->GetStorageShape();
    OP_CHECK(
        outStorageShape == inStorageShape,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            "weight_quant_preprocess", (std::string(outputName) + ", " + inputName).c_str(),
            (op::ToString(outStorageShape).GetString() + std::string(", ") + op::ToString(inStorageShape).GetString())
                .c_str(),
            (std::string(outputName) + " and " + inputName + " must have the same storageShape when dataFlow is " +
             QuantDataFlowToString(ctx.dataFlow))
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

// out weight ND 直拷：与 weight 完全一致
static aclnnStatus CheckOutWeightSameAsInput(const QuantContext& ctx)
{
    return CheckOutSameAsInput(ctx, ctx.weight, ctx.outWeight, "weight", "outWeight");
}

static aclnnStatus CheckOutWeightScaleSameAsInput(const QuantContext& ctx)
{
    return CheckOutSameAsInput(ctx, ctx.weightScale, ctx.outWeightScale, "weightScale", "outWeightScale");
}

static aclnnStatus CheckOutWeightOffsetOptionalSameAsInput(const QuantContext& ctx)
{
    return CheckOutSameAsInput(ctx, ctx.weightOffsetOptional, ctx.outWeightOffsetOptional, "weightOffsetOptional",
                               "outWeightOffsetOptional");
}

static aclnnStatus CheckOutBiasOptionalSameAsInput(const QuantContext& ctx)
{
    return CheckOutSameAsInput(ctx, ctx.biasOptional, ctx.outBiasOptional, "biasOptional", "outBiasOptional");
}

// out weight NZ 出：只要求 nullptr/empty 一致 + viewShape 一致（format/storageShape 合法异于输入；
// dtype 一致性由 CheckOutWeightDtypeSame 单独承担）
static aclnnStatus CheckOutWeightSameBase(const QuantContext& ctx)
{
    return CheckOutSameBase(ctx, ctx.weight, ctx.outWeight, "weight", "outWeight");
}

// out weight NZ 出：dtype 与输入一致（真实场景 NZ 出 dtype 恒等于输入，不一致会在 process ViewCopy 失败）
static aclnnStatus CheckOutWeightDtypeSame(const QuantContext& ctx)
{
    auto weightDtype = ctx.weight->GetDataType();
    auto outWeightDtype = ctx.outWeight->GetDataType();
    OP_CHECK(outWeightDtype == weightDtype,
             OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                 "weight_quant_preprocess", "outWeight, weight",
                 (op::ToString(outWeightDtype).GetString() + std::string(", ") + op::ToString(weightDtype).GetString())
                     .c_str(),
                 (std::string("outWeight and weight must have the same dtype when dataFlow is ") +
                  QuantDataFlowToString(ctx.dataFlow))
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

template <size_t viewKIdx>
static aclnnStatus ProcessWeightTransNd2Nz(QuantContext& ctx)
{
    auto viewShape = ctx.weight->GetViewShape();
    auto dstFormat = ctx.outWeight->GetStorageFormat();

    op::Shape storageShape(viewShape);
    // torch 侧传入的 storageShape 为 1 维，需要手动构造转置后的 storageShape
    std::swap(storageShape[viewKIdx], storageShape[viewKIdx + IDX_1]);

    auto weightTensor = const_cast<aclTensor*>(ctx.weight);
    // TransData 要求输入连续 Tensor，设置所有 Shape 为转置
    weightTensor->SetViewShape(storageShape);
    weightTensor->SetOriginalShape(storageShape);
    weightTensor->SetStorageShape(storageShape);
    auto outTensor = const_cast<aclTensor*>(l0op::TransData(weightTensor, dstFormat, 0, ctx.executor));
    OP_CHECK(outTensor != nullptr,
             LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "TransData failed, storageShape=%s, dstFormat=%d.",
                                op::ToString(storageShape).GetString(), static_cast<int>(dstFormat)),
             return ACLNN_ERR_INNER_NULLPTR);
    // TransData 输出 storageFormat 为 (dstFormat, 0) 组合状态，因 ViewCopy 要求完全一致，需要重新设置
    outTensor->SetStorageFormat(dstFormat);
    outTensor->SetViewShape(viewShape);
    // SetViewShape 内部会隐式设置 viewStride 为非转置连续，重新设置 viewStride 为转置非连续
    op::Strides viewStrides(outTensor->GetViewStrides());
    OP_CHECK(static_cast<int64_t>(viewStrides.size()) > static_cast<int64_t>(viewKIdx + IDX_1),
             LOGE_WITH_SCENARIO(ACLNN_ERR_PARAM_INVALID, "viewStrides size too small for viewKIdx."),
             return ACLNN_ERR_PARAM_INVALID);
    viewStrides[viewKIdx] = 1;
    viewStrides[viewKIdx + IDX_1] = viewShape.GetDim(viewKIdx);
    outTensor->SetViewStrides(viewStrides);

    auto viewCopyResult = l0op::ViewCopy(outTensor, ctx.outWeight, ctx.executor);
    OP_CHECK(viewCopyResult != nullptr,
             LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR,
                                "ViewCopy failed, outTensor viewShape=%s format=%d, outWeight viewShape=%s format=%d.",
                                op::ToString(outTensor->GetViewShape()).GetString(),
                                static_cast<int>(outTensor->GetStorageFormat()),
                                op::ToString(ctx.outWeight->GetViewShape()).GetString(),
                                static_cast<int>(ctx.outWeight->GetStorageFormat())),
             return ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

// weight ND→NZ：viewKIdx 为 k 在 viewShape 的下标（MM=IDX_0，GMM=IDX_1，前面多 g 轴）；
// transWeight 决定喂入 TransData 的 shape 是否交换 k/n 轴：转置 -> K 块在前；非转置 -> N 块在前
template <size_t viewKIdx, bool transWeight>
static aclnnStatus ProcessWeightNd2Nz(QuantContext& ctx)
{
    auto viewShape = ctx.weight->GetViewShape();
    auto dstFormat = ctx.outWeight->GetStorageFormat();

    // INT4 和 FP4 一样，传入逻辑 shape [K, N]，不除以 2，runtime 会根据 dtype 自动计算物理大小
    op::Shape storageShape(viewShape);
    if (transWeight) {
        // torch 侧传入的 storageShape 为 1 维，需要手动构造转置后的 storageShape
        std::swap(storageShape[viewKIdx], storageShape[viewKIdx + IDX_1]);
    }

    auto weightTensor = const_cast<aclTensor*>(ctx.weight);
    // TransData 要求输入连续 Tensor，设置所有 Shape 为喂入的 storageShape
    weightTensor->SetViewShape(storageShape);
    weightTensor->SetOriginalShape(storageShape);
    weightTensor->SetStorageShape(storageShape);
    auto outTensor = const_cast<aclTensor*>(l0op::TransData(weightTensor, dstFormat, 0, ctx.executor));
    OP_CHECK(outTensor != nullptr,
             LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "TransData failed, storageShape=%s, dstFormat=%d.",
                                op::ToString(storageShape).GetString(), static_cast<int>(dstFormat)),
             return ACLNN_ERR_INNER_NULLPTR);
    // TransData 输出 storageFormat 为 (dstFormat, 0) 组合状态，因 ViewCopy 要求完全一致，需要重新设置
    outTensor->SetStorageFormat(dstFormat);
    outTensor->SetViewShape(viewShape);
    // SetViewShape 内部会隐式设置 viewStride 为非转置连续，重新设置为与输入一致的转置状态
    op::Strides viewStrides(outTensor->GetViewStrides());
    OP_CHECK(static_cast<int64_t>(viewStrides.size()) > static_cast<int64_t>(viewKIdx + IDX_1),
             LOGE_WITH_SCENARIO(ACLNN_ERR_PARAM_INVALID, "viewStrides size too small for viewKIdx."),
             return ACLNN_ERR_PARAM_INVALID);
    viewStrides[viewKIdx] = transWeight ? 1 : viewShape.GetDim(viewKIdx + IDX_1);
    viewStrides[viewKIdx + IDX_1] = transWeight ? viewShape.GetDim(viewKIdx) : 1;
    outTensor->SetViewStrides(viewStrides);

    auto viewCopyResult = l0op::ViewCopy(outTensor, ctx.outWeight, ctx.executor);
    OP_CHECK(viewCopyResult != nullptr,
             LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR,
                                "ViewCopy failed, outTensor viewShape=%s format=%d, outWeight viewShape=%s format=%d.",
                                op::ToString(outTensor->GetViewShape()).GetString(),
                                static_cast<int>(outTensor->GetStorageFormat()),
                                op::ToString(ctx.outWeight->GetViewShape()).GetString(),
                                static_cast<int>(ctx.outWeight->GetStorageFormat())),
             return ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

static aclnnStatus ProcessWeightScaleDirectCopy(QuantContext& ctx)
{
    auto srcScale = const_cast<aclTensor*>(ctx.weightScale);
    auto dstScale = ctx.outWeightScale;

    if (srcScale->GetDataType() == op::DataType::DT_FLOAT8_E8M0) {
        auto srcView = ctx.executor->CreateView(srcScale, srcScale->GetViewShape(), srcScale->GetViewOffset());
        OP_CHECK(srcView != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "CreateView weightScale failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
        srcView->SetDataType(op::DataType::DT_INT8);

        auto dstView = ctx.executor->CreateView(dstScale, dstScale->GetViewShape(), dstScale->GetViewOffset());
        OP_CHECK(dstView != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "CreateView outWeightScale failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
        dstView->SetDataType(op::DataType::DT_INT8);

        auto result = l0op::ViewCopy(srcView, dstView, ctx.executor);
        OP_CHECK(result != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "ViewCopy weightScale as INT8 failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
    } else if (srcScale->GetDataType() == Ops::Math::DT_HIFLOAT4_SCALE) {
        auto srcView = ctx.executor->CreateView(srcScale, srcScale->GetViewShape(), srcScale->GetViewOffset());
        OP_CHECK(srcView != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "CreateView weightScale failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
        srcView->SetDataType(op::DataType::DT_FLOAT);

        auto dstView = ctx.executor->CreateView(dstScale, dstScale->GetViewShape(), dstScale->GetViewOffset());
        OP_CHECK(dstView != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "CreateView outWeightScale failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
        dstView->SetDataType(op::DataType::DT_FLOAT);

        auto result = l0op::ViewCopy(srcView, dstView, ctx.executor);
        OP_CHECK(result != nullptr,
                 LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "ViewCopy weightScale as Float32 failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
    } else {
        auto result = l0op::ViewCopy(srcScale, dstScale, ctx.executor);
        OP_CHECK(result != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "ViewCopy weightScale failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus ProcessBiasDirectCopy(QuantContext& ctx)
{
    if (ctx.biasOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    auto result = l0op::ViewCopy(const_cast<aclTensor*>(ctx.biasOptional), ctx.outBiasOptional, ctx.executor);
    OP_CHECK(result != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "ViewCopy bias failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ProcessWeightOffsetDirectCopy(QuantContext& ctx)
{
    if (ctx.weightOffsetOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    auto result = l0op::ViewCopy(const_cast<aclTensor*>(ctx.weightOffsetOptional), ctx.outWeightOffsetOptional,
                                 ctx.executor);
    OP_CHECK(result != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "ViewCopy weightOffset failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ProcessWeightDirectCopy(QuantContext& ctx)
{
    auto srcWeight = const_cast<aclTensor*>(ctx.weight);
    auto dstWeight = ctx.outWeight;

    if (srcWeight->GetDataType() == op::DataType::DT_INT4 || srcWeight->GetDataType() == op::DataType::DT_FLOAT4_E2M1) {
        auto weightViewShape = srcWeight->GetViewShape();
        auto weightViewStrides = srcWeight->GetViewStrides();
        op::Shape packedShape(weightViewShape);
        size_t lastDimIdx = packedShape.GetDimNum() - 1;
        size_t secondLastDimIdx = packedShape.GetDimNum() - IDX_2;
        bool isTransposed = (weightViewStrides[secondLastDimIdx] == 1 && weightViewShape.GetDim(secondLastDimIdx) > 1);
        size_t packDimIdx = isTransposed ? secondLastDimIdx : lastDimIdx;
        packedShape.SetDim(packDimIdx, packedShape.GetDim(packDimIdx) / B4_NUMS_PER_BYTE);

        auto srcView = ctx.executor->CreateView(srcWeight, packedShape, srcWeight->GetViewOffset() / B4_NUMS_PER_BYTE);
        OP_CHECK(srcView != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "CreateView weight failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
        srcView->SetDataType(op::DataType::DT_UINT8);

        auto dstView = ctx.executor->CreateView(dstWeight, packedShape, dstWeight->GetViewOffset() / B4_NUMS_PER_BYTE);
        OP_CHECK(dstView != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "CreateView outWeight failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
        dstView->SetDataType(op::DataType::DT_UINT8);

        auto result = l0op::ViewCopy(srcView, dstView, ctx.executor);
        OP_CHECK(result != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "ViewCopy weight as UINT8 failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
        return ACLNN_SUCCESS;
    }

    auto result = l0op::ViewCopy(srcWeight, dstWeight, ctx.executor);
    OP_CHECK(result != nullptr, LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "ViewCopy weight failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckWeightKDivisibleByGroupSize(const QuantContext& ctx)
{
    OP_CHECK(
        ctx.kGroupSize != 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "aclnnWeightQuantPreprocessGetWorkspaceSize", "kGroupSize", std::to_string(ctx.kGroupSize).c_str(),
            (std::string("kGroupSize can not be zero for data flow ") + QuantDataFlowToString(ctx.dataFlow)).c_str()),
        return ACLNN_ERR_PARAM_INVALID);

    auto weightViewShape = ctx.weight->GetViewShape();
    int64_t k = weightViewShape.GetDim(weightViewShape.GetDimNum() - IDX_2);
    OP_CHECK(
        k % ctx.kGroupSize == 0,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON("aclnnWeightQuantPreprocessGetWorkspaceSize", "weight.K, kGroupSize",
                                               (std::to_string(k) + ", " + std::to_string(ctx.kGroupSize)).c_str(),
                                               (std::string("weight.K must be divisible by kGroupSize for data flow ") +
                                                QuantDataFlowToString(ctx.dataFlow))
                                                   .c_str()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckWeightLayoutA16(const QuantContext& ctx)
{
    auto viewShape = ctx.weight->GetViewShape();
    auto viewStrides = ctx.weight->GetViewStrides();
    OP_CHECK(viewShape.GetDimNum() == DIMS_2 && viewStrides.size() == DIMS_2,
             OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                 "aclnnWeightQuantPreprocessGetWorkspaceSize", "weight.rank, weight.viewStrides.size",
                 (std::to_string(viewShape.GetDimNum()) + "D" + ", " + std::to_string(viewStrides.size())).c_str(),
                 "weight must have rank 2D and exactly two view strides"),
             return ACLNN_ERR_PARAM_INVALID);
    int64_t k = viewShape.GetDim(IDX_0);
    int64_t n = viewShape.GetDim(IDX_1);
    bool nonTransposed = viewStrides[IDX_0] == n && viewStrides[IDX_1] == SINGLE;
    bool transposed = viewStrides[IDX_0] == SINGLE && viewStrides[IDX_1] == k;
    OP_CHECK(nonTransposed || transposed,
             OP_LOGE_FOR_INVALID_STRIDE("aclnnWeightQuantPreprocessGetWorkspaceSize", "weight",
                                        op::ToString(viewStrides).GetString(),
                                        "[N, 1] or [1, K], where [K, N] is the weight view shape"),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutWeightNz9201(const QuantContext& ctx)
{
    auto weightShape = ctx.weight->GetViewShape();
    auto storageShape = ctx.outWeight->GetStorageShape();
    auto viewStrides = ctx.outWeight->GetViewStrides();
    int64_t k = weightShape.GetDim(IDX_0);
    int64_t n = weightShape.GetDim(IDX_1);
    bool validStorage = storageShape.GetDim(IDX_0) == CeilDiv(n, NZ_16) &&
                        storageShape.GetDim(IDX_1) == CeilDiv(k, NZ_16) && storageShape.GetDim(IDX_2) == NZ_16 &&
                        storageShape.GetDim(IDX_3) == NZ_16;
    bool validStride = viewStrides.size() == DIMS_2 && viewStrides[IDX_0] == n && viewStrides[IDX_1] == SINGLE;
    OP_CHECK(ctx.outWeight->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ && validStorage && validStride,
             OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                 "aclnnWeightQuantPreprocessGetWorkspaceSize",
                 "outWeight.storageFormat, outWeight.storageShape, outWeight.viewStrides",
                 (std::string(op::ToString(ctx.outWeight->GetStorageFormat()).GetString()) + ", " +
                  op::ToString(storageShape).GetString() + ", " + op::ToString(viewStrides).GetString())
                     .c_str(),
                 "outWeight must have format FRACTAL_NZ, storage shape [ceildiv(N, 16), ceildiv(K, 16), "
                 "16, 16] and view strides [N, 1], where [K, N] is the weight view shape"),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckWeightScaleNotEmpty(const QuantContext& ctx)
{
    OP_CHECK(
        !ctx.weightScale->IsEmpty(),
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            "aclnnWeightQuantPreprocessGetWorkspaceSize", "weightScale",
            std::to_string(ctx.weightScale->GetViewShape().GetShapeSize()).c_str(),
            (std::string("weightScale can not be an empty tensor for data flow ") + QuantDataFlowToString(ctx.dataFlow))
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckWeightScaleFormatND(const QuantContext& ctx)
{
    auto scaleFormat = ctx.weightScale->GetStorageFormat();
    OP_CHECK(scaleFormat == op::Format::FORMAT_ND || scaleFormat == op::Format::FORMAT_NCL,
             OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON("aclnnWeightQuantPreprocessGetWorkspaceSize", "weightScale",
                                                    op::ToString(scaleFormat).GetString(),
                                                    "the format of weightScale must be ND or NCL"),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

template <size_t targetDim, int64_t scalePack = DOUBLE>
static aclnnStatus CheckWeightScaleMx(const QuantContext& ctx)
{
    auto scaleViewShape = ctx.weightScale->GetViewShape();
    size_t scaleViewDim = scaleViewShape.GetDimNum();

    OP_CHECK(
        scaleViewDim == targetDim,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            "aclnnWeightQuantPreprocessGetWorkspaceSize", "weightScale", (std::to_string(scaleViewDim) + "D").c_str(),
            (std::string("the rank of weightScale must be ") + std::to_string(targetDim) + "D").c_str()),
        return ACLNN_ERR_PARAM_INVALID);

    auto weightViewShape = ctx.weight->GetViewShape();
    size_t weightViewDim = weightViewShape.GetDimNum();
    OP_CHECK(
        scaleViewShape.GetDim(scaleViewDim - IDX_3) ==
                CeilDiv(weightViewShape.GetDim(weightViewDim - IDX_2), KGROUP_SIZE_MX * DOUBLE) &&
            scaleViewShape.GetDim(scaleViewDim - IDX_2) == weightViewShape.GetDim(weightViewDim - IDX_1) &&
            scaleViewShape.GetDim(scaleViewDim - IDX_1) == scalePack,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            "aclnnWeightQuantPreprocessGetWorkspaceSize", "weightScale, weight",
            (std::string(op::ToString(scaleViewShape).GetString()) + ", " + op::ToString(weightViewShape).GetString())
                .c_str(),
            (std::string("the last three dimensions of weightScale must be [ceildiv(K, 64), N, ") +
             std::to_string(scalePack) + "], where K and N are the last two dimensions of weight for data flow " +
             QuantDataFlowToString(ctx.dataFlow))
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckMxA16ScaleDim(const QuantContext& ctx)
{
    const aclTensor* scale = ctx.weightScale;
    size_t dimNum = scale->GetViewShape().GetDimNum();
    bool valid = (dimNum == DIMS_2 || dimNum == DIMS_3) && scale->GetViewStrides().size() == dimNum;
    OP_CHECK(valid,
             OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                 "aclnnWeightQuantPreprocessGetWorkspaceSize", "weightScale.rank, weightScale.viewStrides.size",
                 (std::to_string(dimNum) + "D" + ", " + std::to_string(scale->GetViewStrides().size())).c_str(),
                 "weightScale must have rank 2D or 3D and the number of view strides must equal its rank"),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckMxA16ScaleShape(const QuantContext& ctx)
{
    auto viewShape = ctx.weightScale->GetViewShape();
    int64_t k = ctx.weight->GetViewShape().GetDim(IDX_0);
    int64_t n = ctx.weight->GetViewShape().GetDim(IDX_1);
    int64_t groupNum = CeilDiv(k, KGROUP_SIZE_MX);
    if (viewShape.GetDimNum() == DIMS_2) {
        bool valid = viewShape.GetDim(IDX_0) == groupNum && viewShape.GetDim(IDX_1) == n;
        OP_CHECK(valid,
                 OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                     "aclnnWeightQuantPreprocessGetWorkspaceSize", "weightScale, weight",
                     (std::string(op::ToString(viewShape).GetString()) + ", " +
                      op::ToString(ctx.weight->GetViewShape()).GetString())
                         .c_str(),
                     (std::string("weightScale must have shape [ceildiv(K, 32), N], where [K, N] is the weight view "
                                  "shape for data flow ") +
                      QuantDataFlowToString(ctx.dataFlow))
                         .c_str()),
                 return ACLNN_ERR_PARAM_INVALID);
        return ACLNN_SUCCESS;
    }

    int64_t blockNum = CeilDiv(groupNum, DOUBLE);
    bool valid = viewShape.GetDim(IDX_0) == blockNum && viewShape.GetDim(IDX_1) == n &&
                 viewShape.GetDim(IDX_2) == DOUBLE;
    OP_CHECK(valid,
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                 "aclnnWeightQuantPreprocessGetWorkspaceSize", "weightScale, weight",
                 (std::string(op::ToString(viewShape).GetString()) + ", " +
                  op::ToString(ctx.weight->GetViewShape()).GetString())
                     .c_str(),
                 (std::string("weightScale must have shape [ceildiv(ceildiv(K, 32), 2), N, 2], where [K, N] is the "
                              "weight view shape for data flow ") +
                  QuantDataFlowToString(ctx.dataFlow))
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckMxA16ScaleLayout(const QuantContext& ctx)
{
    auto viewShape = ctx.weightScale->GetViewShape();
    auto viewStrides = ctx.weightScale->GetViewStrides();
    int64_t k = ctx.weight->GetViewShape().GetDim(IDX_0);
    int64_t n = ctx.weight->GetViewShape().GetDim(IDX_1);
    int64_t groupNum = CeilDiv(k, KGROUP_SIZE_MX);
    if (viewShape.GetDimNum() == DIMS_2) {
        bool groupFirst = viewStrides[IDX_0] == n && viewStrides[IDX_1] == SINGLE;
        bool channelFirst = viewStrides[IDX_0] == SINGLE && viewStrides[IDX_1] == groupNum;
        OP_CHECK(groupFirst || channelFirst,
                 OP_LOGE_FOR_INVALID_STRIDE(
                     "aclnnWeightQuantPreprocessGetWorkspaceSize", "weightScale", op::ToString(viewStrides).GetString(),
                     "[N, 1] or [1, G], where G = ceildiv(K, 32) and [K, N] is the weight view shape"),
                 return ACLNN_ERR_PARAM_INVALID);
        return ACLNN_SUCCESS;
    }
    int64_t blockNum = CeilDiv(groupNum, DOUBLE);
    int64_t lastDim = static_cast<int64_t>(DOUBLE);
    bool blockFirst = viewStrides[IDX_0] == n * lastDim && viewStrides[IDX_1] == lastDim &&
                      viewStrides[IDX_2] == SINGLE;
    bool channelFirst = viewStrides[IDX_0] == lastDim && viewStrides[IDX_1] == blockNum * lastDim &&
                        viewStrides[IDX_2] == SINGLE;
    OP_CHECK(blockFirst || channelFirst,
             OP_LOGE_FOR_INVALID_STRIDE(
                 "aclnnWeightQuantPreprocessGetWorkspaceSize", "weightScale", op::ToString(viewStrides).GetString(),
                 "[N * 2, 2, 1] or [2, B * 2, 1], where B = ceildiv(ceildiv(K, 32), 2) and [K, N] is "
                 "the weight view shape"),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

template <size_t targetIdx>
static aclnnStatus CheckWeightScaleTrans(const QuantContext& ctx)
{
    auto viewShape = ctx.weightScale->GetViewShape();
    auto viewStrides = ctx.weightScale->GetViewStrides();

    OP_CHECK(viewStrides[targetIdx + IDX_1] == viewStrides[targetIdx] * viewShape.GetDim(targetIdx),
             OP_LOGE_FOR_INVALID_STRIDE(
                 "aclnnWeightQuantPreprocessGetWorkspaceSize", "weightScale", op::ToString(viewStrides).GetString(),
                 (std::string("the stride at dimension ") + std::to_string(targetIdx + IDX_1) +
                  " must equal the stride at the preceding dimension multiplied by its view-shape dimension")
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOffsetNotEmpty(const QuantContext& ctx)
{
    if (ctx.weightOffsetOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    OP_CHECK(!ctx.weightOffsetOptional->IsEmpty(),
             OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                 "aclnnWeightQuantPreprocessGetWorkspaceSize", "weightOffsetOptional",
                 std::to_string(ctx.weightOffsetOptional->GetViewShape().GetShapeSize()).c_str(),
                 (std::string("weightOffsetOptional can not be an empty tensor for data flow ") +
                  QuantDataFlowToString(ctx.dataFlow))
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOffsetDtype(const QuantContext& ctx)
{
    if (ctx.weightOffsetOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    OP_CHECK(ctx.weightOffsetOptional->GetDataType() == ctx.xDtype,
             OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                 "aclnnWeightQuantPreprocessGetWorkspaceSize", "weightOffsetOptional, xDtype",
                 (std::string(op::ToString(ctx.weightOffsetOptional->GetDataType()).GetString()) + ", " +
                  op::ToString(ctx.xDtype).GetString())
                     .c_str(),
                 (std::string("the dtype of weightOffsetOptional must equal xDtype for data flow ") +
                  QuantDataFlowToString(ctx.dataFlow))
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOffsetFormat(const QuantContext& ctx)
{
    if (ctx.weightOffsetOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    OP_CHECK(
        ctx.weightOffsetOptional->GetStorageFormat() == op::Format::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON("aclnnWeightQuantPreprocessGetWorkspaceSize", "weightOffsetOptional",
                                               op::ToString(ctx.weightOffsetOptional->GetStorageFormat()).GetString(),
                                               "the format of weightOffsetOptional must be ND"),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

template <bool isOffset>
static aclnnStatus CheckPerTensorScaleOrOffset(const QuantContext& ctx)
{
    const aclTensor* tensor = isOffset ? ctx.weightOffsetOptional : ctx.weightScale;
    const char* tensorName = isOffset ? "weightOffsetOptional" : "weightScale";
    if (isOffset && tensor == nullptr) {
        return ACLNN_SUCCESS;
    }
    auto viewShape = tensor->GetViewShape();
    auto viewStrides = tensor->GetViewStrides();
    bool valid = viewShape.GetDimNum() == DIMS_1 && viewStrides.size() == DIMS_1 && viewShape.GetDim(IDX_0) == SINGLE &&
                 viewStrides[IDX_0] == SINGLE;
    OP_CHECK(
        valid,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            "aclnnWeightQuantPreprocessGetWorkspaceSize",
            (std::string(tensorName) + ".viewShape" + ", " + tensorName + ".viewStrides").c_str(),
            (std::string(op::ToString(viewShape).GetString()) + ", " + op::ToString(viewStrides).GetString()).c_str(),
            (std::string("the view shape must be [1] and view strides must be [1] for data flow ") +
             QuantDataFlowToString(ctx.dataFlow))
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

template <bool isOffset>
static aclnnStatus CheckPerChannelScaleOrOffset(const QuantContext& ctx)
{
    const aclTensor* tensor = isOffset ? ctx.weightOffsetOptional : ctx.weightScale;
    const char* tensorName = isOffset ? "weightOffsetOptional" : "weightScale";
    if (isOffset && tensor == nullptr) {
        return ACLNN_SUCCESS;
    }
    size_t dimNum = tensor->GetViewShape().GetDimNum();
    OP_CHECK(
        dimNum == DIMS_1 || dimNum == DIMS_2,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnWeightQuantPreprocessGetWorkspaceSize", tensorName,
                                                 (std::to_string(dimNum) + "D").c_str(), "the rank must be 1D or 2D"),
        return ACLNN_ERR_PARAM_INVALID);
    auto viewStrides = tensor->GetViewStrides();
    OP_CHECK(viewStrides.size() == dimNum,
             OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                 "aclnnWeightQuantPreprocessGetWorkspaceSize",
                 (std::string(tensorName) + ".viewStrides.size, " + tensorName + ".rank").c_str(),
                 (std::to_string(viewStrides.size()) + ", " + std::to_string(dimNum) + "D").c_str(),
                 "the number of view strides must equal the tensor rank"),
             return ACLNN_ERR_PARAM_INVALID);

    int64_t n = ctx.weight->GetViewShape().GetDim(IDX_1);
    auto viewShape = tensor->GetViewShape();
    bool valid = viewShape.GetDim(IDX_0) == n && viewStrides[IDX_0] == SINGLE;
    if (dimNum == DIMS_2) {
        bool shape1N = viewShape.GetDim(IDX_0) == SINGLE && viewShape.GetDim(IDX_1) == n;
        bool layoutN1 = viewStrides[IDX_0] == SINGLE && viewStrides[IDX_1] == SINGLE;
        bool layout1N = viewStrides[IDX_0] == n && viewStrides[IDX_1] == SINGLE;
        valid = shape1N && (layoutN1 || layout1N);
    }
    OP_CHECK(
        valid,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            "aclnnWeightQuantPreprocessGetWorkspaceSize",
            (std::string(tensorName) + ".viewShape" + ", " + tensorName + ".viewStrides").c_str(),
            (std::string(op::ToString(viewShape).GetString()) + ", " + op::ToString(viewStrides).GetString()).c_str(),
            (std::string("the view shape and strides must be [N] / [1], [1, N] / [1, 1] or [1, N] / [N, 1], "
                         "where N is the last dimension of weight for data flow ") +
             QuantDataFlowToString(ctx.dataFlow))
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

template <bool isOffset>
static aclnnStatus CheckPerGroupScaleOrOffset(const QuantContext& ctx)
{
    const aclTensor* tensor = isOffset ? ctx.weightOffsetOptional : ctx.weightScale;
    const char* tensorName = isOffset ? "weightOffsetOptional" : "weightScale";
    if (isOffset && tensor == nullptr) {
        return ACLNN_SUCCESS;
    }
    auto viewShape = tensor->GetViewShape();
    auto viewStrides = tensor->GetViewStrides();
    OP_CHECK(viewShape.GetDimNum() == DIMS_2 && viewStrides.size() == DIMS_2,
             OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                 "aclnnWeightQuantPreprocessGetWorkspaceSize",
                 (std::string(tensorName) + ".rank" + ", " + tensorName + ".viewStrides.size").c_str(),
                 (std::to_string(viewShape.GetDimNum()) + "D" + ", " + std::to_string(viewStrides.size())).c_str(),
                 "the tensor must have rank 2D and exactly two view strides"),
             return ACLNN_ERR_PARAM_INVALID);
    int64_t groupNum = CeilDiv(ctx.weight->GetViewShape().GetDim(IDX_0), ctx.kGroupSize);
    int64_t n = ctx.weight->GetViewShape().GetDim(IDX_1);
    bool validShape = viewShape.GetDim(IDX_0) == groupNum && viewShape.GetDim(IDX_1) == n;
    bool layoutGN = viewStrides[IDX_0] == n && viewStrides[IDX_1] == SINGLE;
    bool layoutNG = viewStrides[IDX_0] == SINGLE && viewStrides[IDX_1] == groupNum;
    OP_CHECK(
        validShape && (layoutGN || layoutNG),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            "aclnnWeightQuantPreprocessGetWorkspaceSize",
            (std::string(tensorName) + ".viewShape" + ", " + tensorName + ".viewStrides").c_str(),
            (std::string(op::ToString(viewShape).GetString()) + ", " + op::ToString(viewStrides).GetString()).c_str(),
            (std::string("the view shape must be [G, N] and view strides must be [N, 1] or [1, G], where G = "
                         "ceildiv(K, kGroupSize) and [K, N] is the weight view shape for data flow ") +
             QuantDataFlowToString(ctx.dataFlow))
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckKGroupSizeMx(const QuantContext& ctx)
{
    OP_CHECK(ctx.kGroupSize == KGROUP_SIZE_MX,
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnWeightQuantPreprocessGetWorkspaceSize", "kGroupSize",
                                                   std::to_string(ctx.kGroupSize).c_str(), "kGroupSize must be 32"),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

template <int64_t... allowedGroupSizes>
static aclnnStatus CheckGroupSize(const QuantContext& ctx)
{
    bool match = ((ctx.kGroupSize == allowedGroupSizes) || ...);
    std::string allowedStr = "kGroupSize must be one of ";
    std::string sep;
    for (int64_t sz : {allowedGroupSizes...}) {
        allowedStr += sep + std::to_string(sz);
        sep = ", ";
    }
    OP_CHECK(match,
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnWeightQuantPreprocessGetWorkspaceSize", "kGroupSize",
                                                   std::to_string(ctx.kGroupSize).c_str(), allowedStr.c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckWeightEven(const QuantContext& ctx)
{
    auto weightShape = ctx.weight->GetViewShape();
    int64_t k = weightShape.GetDim(IDX_0);
    int64_t n = weightShape.GetDim(IDX_1);
    OP_CHECK(k % DOUBLE == 0 && n % DOUBLE == 0,
             OP_LOGE_FOR_INVALID_VALUES_WITH_REASON("aclnnWeightQuantPreprocessGetWorkspaceSize", "weight.K, weight.N",
                                                    (std::to_string(k) + ", " + std::to_string(n)).c_str(),
                                                    "weight.K and weight.N must both be divisible by 2"),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutWeightNotNullEmpty(const QuantContext& ctx)
{
    OP_CHECK(ctx.outWeight != nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnWeightQuantPreprocessGetWorkspaceSize", "outWeight",
                                                      "outWeight can not be null"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(
        !ctx.outWeight->IsEmpty(),
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            "aclnnWeightQuantPreprocessGetWorkspaceSize", "outWeight",
            std::to_string(ctx.outWeight->GetViewShape().GetShapeSize()).c_str(),
            (std::string("outWeight can not be an empty tensor for data flow ") + QuantDataFlowToString(ctx.dataFlow))
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutWeightViewShapeSame(const QuantContext& ctx)
{
    auto weightViewShape = ctx.weight->GetViewShape();
    auto outViewShape = ctx.outWeight->GetViewShape();
    OP_CHECK(
        weightViewShape == outViewShape,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            "aclnnWeightQuantPreprocessGetWorkspaceSize", "outWeight, weight",
            (std::string(op::ToString(outViewShape).GetString()) + ", " + op::ToString(weightViewShape).GetString())
                .c_str(),
            (std::string("outWeight and weight must have the same view shape for data flow ") +
             QuantDataFlowToString(ctx.dataFlow))
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutWeightScaleNotNullEmpty(const QuantContext& ctx)
{
    OP_CHECK(ctx.outWeightScale != nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnWeightQuantPreprocessGetWorkspaceSize", "outWeightScale",
                                                      "outWeightScale can not be null"),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(!ctx.outWeightScale->IsEmpty(),
             OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                 "aclnnWeightQuantPreprocessGetWorkspaceSize", "outWeightScale",
                 std::to_string(ctx.outWeightScale->GetViewShape().GetShapeSize()).c_str(),
                 (std::string("outWeightScale can not be an empty tensor for data flow ") +
                  QuantDataFlowToString(ctx.dataFlow))
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutWeightScaleFormatND(const QuantContext& ctx)
{
    auto outScaleFormat = ctx.outWeightScale->GetStorageFormat();
    OP_CHECK(outScaleFormat == op::Format::FORMAT_ND || outScaleFormat == op::Format::FORMAT_NCL,
             OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON("aclnnWeightQuantPreprocessGetWorkspaceSize", "outWeightScale",
                                                    op::ToString(outScaleFormat).GetString(),
                                                    "the format of outWeightScale must be ND or NCL"),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutWeightScaleDtypeSame(const QuantContext& ctx)
{
    auto scaleDtype = ctx.weightScale->GetDataType();
    auto outScaleDtype = ctx.outWeightScale->GetDataType();
    OP_CHECK(outScaleDtype == scaleDtype,
             OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                 "aclnnWeightQuantPreprocessGetWorkspaceSize", "outWeightScale, weightScale",
                 (std::string(op::ToString(outScaleDtype).GetString()) + ", " + op::ToString(scaleDtype).GetString())
                     .c_str(),
                 (std::string("outWeightScale and weightScale must have the same dtype for data flow ") +
                  QuantDataFlowToString(ctx.dataFlow))
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutWeightScaleViewShapeSame(const QuantContext& ctx)
{
    auto scaleViewShape = ctx.weightScale->GetViewShape();
    auto outScaleViewShape = ctx.outWeightScale->GetViewShape();
    OP_CHECK(
        outScaleViewShape == scaleViewShape,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            "aclnnWeightQuantPreprocessGetWorkspaceSize", "outWeightScale, weightScale",
            (std::string(op::ToString(outScaleViewShape).GetString()) + ", " + op::ToString(scaleViewShape).GetString())
                .c_str(),
            (std::string("outWeightScale and weightScale must have the same view shape for data flow ") +
             QuantDataFlowToString(ctx.dataFlow))
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutWeightScaleStorageShapeSame(const QuantContext& ctx)
{
    auto scaleStorageShape = ctx.weightScale->GetStorageShape();
    auto outScaleStorageShape = ctx.outWeightScale->GetStorageShape();
    OP_CHECK(outScaleStorageShape == scaleStorageShape,
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                 "aclnnWeightQuantPreprocessGetWorkspaceSize", "outWeightScale.storageShape, weightScale.storageShape",
                 (std::string(op::ToString(outScaleStorageShape).GetString()) + ", " +
                  op::ToString(scaleStorageShape).GetString())
                     .c_str(),
                 (std::string("outWeightScale and weightScale must have the same storage shape for data flow ") +
                  QuantDataFlowToString(ctx.dataFlow))
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutOffsetNotEmpty(const QuantContext& ctx)
{
    if (ctx.weightOffsetOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    OP_CHECK(
        (ctx.weightOffsetOptional == nullptr) == (ctx.outWeightOffsetOptional == nullptr),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            "aclnnWeightQuantPreprocessGetWorkspaceSize", "weightOffsetOptional.isNull, outWeightOffsetOptional.isNull",
            (std::string((ctx.weightOffsetOptional == nullptr ? "true" : "false")) + ", " +
             (ctx.outWeightOffsetOptional == nullptr ? "true" : "false"))
                .c_str(),
            "weightOffsetOptional and outWeightOffsetOptional must both be null or both be non-null"),
        return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(!ctx.outWeightOffsetOptional->IsEmpty(),
             OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                 "aclnnWeightQuantPreprocessGetWorkspaceSize", "outWeightOffsetOptional",
                 std::to_string(ctx.outWeightOffsetOptional->GetViewShape().GetShapeSize()).c_str(),
                 (std::string("outWeightOffsetOptional can not be an empty tensor for data flow ") +
                  QuantDataFlowToString(ctx.dataFlow))
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutOffsetFormat(const QuantContext& ctx)
{
    if (ctx.weightOffsetOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    OP_CHECK(ctx.outWeightOffsetOptional->GetStorageFormat() == op::Format::FORMAT_ND,
             OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                 "aclnnWeightQuantPreprocessGetWorkspaceSize", "outWeightOffsetOptional",
                 op::ToString(ctx.outWeightOffsetOptional->GetStorageFormat()).GetString(),
                 "the format of outWeightOffsetOptional must be ND"),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutOffsetDtypeSame(const QuantContext& ctx)
{
    if (ctx.weightOffsetOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    OP_CHECK(
        ctx.outWeightOffsetOptional->GetDataType() == ctx.weightOffsetOptional->GetDataType(),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            "aclnnWeightQuantPreprocessGetWorkspaceSize", "outWeightOffsetOptional, weightOffsetOptional",
            (std::string(op::ToString(ctx.outWeightOffsetOptional->GetDataType()).GetString()) + ", " +
             op::ToString(ctx.weightOffsetOptional->GetDataType()).GetString())
                .c_str(),
            (std::string("outWeightOffsetOptional and weightOffsetOptional must have the same dtype for data flow ") +
             QuantDataFlowToString(ctx.dataFlow))
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

template <bool isOffset>
static aclnnStatus CheckOutPerTensorScaleOrOffset(const QuantContext& ctx)
{
    if (isOffset && ctx.weightOffsetOptional == nullptr) {
        return ACLNN_SUCCESS;
    }

    const aclTensor* target = isOffset ? ctx.outWeightOffsetOptional : ctx.outWeightScale;
    const char* tensorName = isOffset ? "outWeightOffsetOptional" : "outWeightScale";
    auto viewShape = target->GetViewShape();
    auto viewStrides = target->GetViewStrides();
    bool valid = viewShape.GetDimNum() == DIMS_1 && viewStrides.size() == DIMS_1 && viewShape.GetDim(IDX_0) == SINGLE &&
                 viewStrides[IDX_0] == SINGLE;
    OP_CHECK(
        valid,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            "aclnnWeightQuantPreprocessGetWorkspaceSize",
            (std::string(tensorName) + ".viewShape" + ", " + tensorName + ".viewStrides").c_str(),
            (std::string(op::ToString(viewShape).GetString()) + ", " + op::ToString(viewStrides).GetString()).c_str(),
            (std::string("the view shape must be [1] and view strides must be [1] for data flow ") +
             QuantDataFlowToString(ctx.dataFlow))
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

template <bool isOffset>
static aclnnStatus CheckOutPerChannelScaleOrOffset(const QuantContext& ctx)
{
    if (isOffset && ctx.weightOffsetOptional == nullptr) {
        return ACLNN_SUCCESS;
    }

    const aclTensor* source = isOffset ? ctx.weightOffsetOptional : ctx.weightScale;
    const aclTensor* target = isOffset ? ctx.outWeightOffsetOptional : ctx.outWeightScale;
    const char* tensorName = isOffset ? "outWeightOffsetOptional" : "outWeightScale";
    size_t dimNum = source->GetViewShape().GetDimNum();
    int64_t n = ctx.weight->GetViewShape().GetDim(IDX_1);
    auto viewShape = target->GetViewShape();
    auto viewStrides = target->GetViewStrides();
    OP_CHECK(viewShape.GetDimNum() == dimNum && viewStrides.size() == dimNum,
             OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                 "aclnnWeightQuantPreprocessGetWorkspaceSize",
                 (std::string(tensorName) + ".rank" + ", " + tensorName + ".viewStrides.size").c_str(),
                 (std::to_string(viewShape.GetDimNum()) + "D" + ", " + std::to_string(viewStrides.size())).c_str(),
                 (std::string("the output rank and the number of view strides must equal the input rank of ") +
                  std::to_string(dimNum) + "D")
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    bool valid = viewShape.GetDim(IDX_0) == n && viewStrides[IDX_0] == SINGLE;
    if (dimNum == DIMS_2) {
        valid = viewShape.GetDim(IDX_0) == SINGLE && viewShape.GetDim(IDX_1) == n && viewStrides[IDX_0] == n &&
                viewStrides[IDX_1] == SINGLE;
    }
    OP_CHECK(
        valid,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            "aclnnWeightQuantPreprocessGetWorkspaceSize",
            (std::string(tensorName) + ".viewShape" + ", " + tensorName + ".viewStrides").c_str(),
            (std::string(op::ToString(viewShape).GetString()) + ", " + op::ToString(viewStrides).GetString()).c_str(),
            (std::string("the view shape and strides must be [N] / [1] for a 1D input or [1, N] / [N, 1] for a "
                         "2D input, where N is the last dimension of weight for data flow ") +
             QuantDataFlowToString(ctx.dataFlow))
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutC064Scale(const QuantContext& ctx)
{
    size_t dimNum = ctx.weightScale->GetViewShape().GetDimNum();
    OP_CHECK(ctx.outWeightScale->GetDataType() == op::DataType::DT_UINT64,
             OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                 "aclnnWeightQuantPreprocessGetWorkspaceSize", "outWeightScale",
                 op::ToString(ctx.outWeightScale->GetDataType()).GetString(),
                 "the dtype of outWeightScale must be UINT64 for C064 per-channel quantization"),
             return ACLNN_ERR_PARAM_INVALID);
    int64_t n = ctx.weight->GetViewShape().GetDim(IDX_1);
    auto viewShape = ctx.outWeightScale->GetViewShape();
    auto viewStrides = ctx.outWeightScale->GetViewStrides();
    OP_CHECK(viewShape.GetDimNum() == dimNum && viewStrides.size() == dimNum,
             OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                 "aclnnWeightQuantPreprocessGetWorkspaceSize", "outWeightScale.rank, outWeightScale.viewStrides.size",
                 (std::to_string(viewShape.GetDimNum()) + "D" + ", " + std::to_string(viewStrides.size())).c_str(),
                 (std::string("the output rank and the number of view strides must equal the input rank of ") +
                  std::to_string(dimNum) + "D")
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    bool valid = viewShape.GetDim(IDX_0) == n && viewStrides[IDX_0] == SINGLE;
    if (dimNum == DIMS_2) {
        valid = viewShape.GetDim(IDX_0) == n && viewShape.GetDim(IDX_1) == SINGLE && viewStrides[IDX_0] == SINGLE &&
                viewStrides[IDX_1] == SINGLE;
    }
    OP_CHECK(
        valid,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            "aclnnWeightQuantPreprocessGetWorkspaceSize", "outWeightScale.viewShape, outWeightScale.viewStrides",
            (std::string(op::ToString(viewShape).GetString()) + ", " + op::ToString(viewStrides).GetString()).c_str(),
            (std::string("the view shape and strides must be [N] / [1] for a 1D input or [N, 1] / [1, 1] for a "
                         "2D input, where N is the last dimension of weight for data flow ") +
             QuantDataFlowToString(ctx.dataFlow))
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

template <bool isOffset>
static aclnnStatus CheckOutPerGroupScaleOrOffset(const QuantContext& ctx)
{
    if (isOffset && ctx.weightOffsetOptional == nullptr) {
        return ACLNN_SUCCESS;
    }

    const aclTensor* target = isOffset ? ctx.outWeightOffsetOptional : ctx.outWeightScale;
    const char* tensorName = isOffset ? "outWeightOffsetOptional" : "outWeightScale";
    int64_t groupNum = CeilDiv(ctx.weight->GetViewShape().GetDim(IDX_0), ctx.kGroupSize);
    int64_t n = ctx.weight->GetViewShape().GetDim(IDX_1);
    auto viewShape = target->GetViewShape();
    auto viewStrides = target->GetViewStrides();
    OP_CHECK(viewShape.GetDimNum() == DIMS_2 && viewStrides.size() == DIMS_2,
             OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                 "aclnnWeightQuantPreprocessGetWorkspaceSize",
                 (std::string(tensorName) + ".rank" + ", " + tensorName + ".viewStrides.size").c_str(),
                 (std::to_string(viewShape.GetDimNum()) + "D" + ", " + std::to_string(viewStrides.size())).c_str(),
                 "the tensor must have rank 2D and exactly two view strides"),
             return ACLNN_ERR_PARAM_INVALID);
    bool valid = viewShape.GetDim(IDX_0) == groupNum && viewShape.GetDim(IDX_1) == n && viewStrides[IDX_0] == n &&
                 viewStrides[IDX_1] == SINGLE;
    OP_CHECK(
        valid,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            "aclnnWeightQuantPreprocessGetWorkspaceSize",
            (std::string(tensorName) + ".viewShape" + ", " + tensorName + ".viewStrides").c_str(),
            (std::string(op::ToString(viewShape).GetString()) + ", " + op::ToString(viewStrides).GetString()).c_str(),
            (std::string("the view shape must be [G, N] and view strides must be [N, 1], where G = ceildiv(K, "
                         "kGroupSize) and [K, N] is the weight view shape for data flow ") +
             QuantDataFlowToString(ctx.dataFlow))
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutMxScale(const QuantContext& ctx)
{
    auto viewShape = ctx.outWeightScale->GetViewShape();
    auto viewStrides = ctx.outWeightScale->GetViewStrides();
    OP_CHECK(viewShape.GetDimNum() == DIMS_3 && viewStrides.size() == DIMS_3,
             OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                 "aclnnWeightQuantPreprocessGetWorkspaceSize", "outWeightScale.rank, outWeightScale.viewStrides.size",
                 (std::to_string(viewShape.GetDimNum()) + "D" + ", " + std::to_string(viewStrides.size())).c_str(),
                 "outWeightScale must have rank 3D and exactly three view strides"),
             return ACLNN_ERR_PARAM_INVALID);

    int64_t k = ctx.weight->GetViewShape().GetDim(IDX_0);
    int64_t n = ctx.weight->GetViewShape().GetDim(IDX_1);
    int64_t groupNum = CeilDiv(k, KGROUP_SIZE_MX);
    int64_t blockNum = CeilDiv(groupNum, DOUBLE);
    int64_t lastDim = static_cast<int64_t>(DOUBLE);
    int64_t channelStride = blockNum * lastDim;

    bool valid = viewShape.GetDim(IDX_0) == blockNum && viewShape.GetDim(IDX_1) == n &&
                 viewShape.GetDim(IDX_2) == lastDim && viewStrides[IDX_0] == lastDim &&
                 viewStrides[IDX_1] == channelStride && viewStrides[IDX_2] == SINGLE;
    OP_CHECK(
        valid,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            "aclnnWeightQuantPreprocessGetWorkspaceSize", "outWeightScale.viewShape, outWeightScale.viewStrides",
            (std::string(op::ToString(viewShape).GetString()) + ", " + op::ToString(viewStrides).GetString()).c_str(),
            (std::string("the view shape must be [B, N, 2] and view strides must be [2, B * 2, 1], where B = "
                         "ceildiv(ceildiv(K, 32), 2) and [K, N] is the weight view shape for data flow ") +
             QuantDataFlowToString(ctx.dataFlow))
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

template <size_t viewKIdx, bool transposedOutput, bool nz_16 = false>
static aclnnStatus ProcessWeightNd2Nz9201(const aclTensor* weight, const op::Shape& physicalShape, QuantContext& ctx)
{
    auto viewShape = ctx.weight->GetViewShape();
    auto dstFormat = nz_16 ? op::Format::FORMAT_FRACTAL_NZ_C0_16 : ctx.outWeight->GetStorageFormat();
    auto weightTensor = ctx.executor->CreateView(weight, physicalShape, weight->GetViewOffset());
    OP_CHECK(weightTensor != nullptr,
             LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "Create physical Weight view before TransData failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    weightTensor->SetViewShape(physicalShape);
    weightTensor->SetOriginalShape(physicalShape);
    weightTensor->SetStorageShape(physicalShape);

    auto outTensor = const_cast<aclTensor*>(l0op::TransData(weightTensor, dstFormat, 0, ctx.executor));
    OP_CHECK(outTensor != nullptr,
             LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "TransData Weight failed, physicalShape=%s, dstFormat=%d.",
                                op::ToString(physicalShape).GetString(), static_cast<int>(dstFormat)),
             return ACLNN_ERR_INNER_NULLPTR);

    outTensor->SetStorageFormat(ctx.outWeight->GetStorageFormat());
    outTensor->SetViewShape(viewShape);
    if constexpr (transposedOutput) {
        op::Strides viewStrides(outTensor->GetViewStrides());
        viewStrides[viewKIdx] = SINGLE;
        viewStrides[viewKIdx + IDX_1] = viewShape.GetDim(viewKIdx);
        outTensor->SetViewStrides(viewStrides);
    }

    auto result = l0op::ViewCopy(outTensor, ctx.outWeight, ctx.executor);
    OP_CHECK(result != nullptr,
             LOGE_WITH_SCENARIO(
                 ACLNN_ERR_INNER_NULLPTR, "ViewCopy NZ Weight failed, transposedOutput=%d, viewShape=%s, format=%d.",
                 static_cast<int>(transposedOutput), op::ToString(viewShape).GetString(), static_cast<int>(dstFormat)),
             return ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static const aclTensor* TransposeWeightInt4ViaFp4(const aclTensor* weight, QuantContext& ctx)
{
    const bool isInt4 = weight->GetDataType() == op::DataType::DT_INT4;
    const bool isFp4 = weight->GetDataType() == op::DataType::DT_FLOAT4_E2M1;
    OP_CHECK(isInt4 || isFp4,
             OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON("aclnnWeightQuantPreprocessGetWorkspaceSize", "weight",
                                                   op::ToString(weight->GetDataType()).GetString(),
                                                   "the dtype of weight must be INT4 or FLOAT4_E2M1"),
             return nullptr);
    auto fp4View = ctx.executor->CreateView(weight, weight->GetViewShape(), weight->GetViewOffset());
    OP_CHECK(fp4View != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Create FP4 view for B4 Weight failed."),
             return nullptr);
    fp4View->SetViewStrides(weight->GetViewStrides());
    fp4View->SetDataType(op::DataType::DT_FLOAT4_E2M1);

    auto bf16Weight = l0op::Cast(fp4View, op::DataType::DT_BF16, ctx.executor);
    OP_CHECK(bf16Weight != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Cast B4 Weight to BF16 failed."), return nullptr);
    int64_t permData[DIMS_2] = {IDX_1, IDX_0};
    auto perm = ctx.executor->AllocIntArray(permData, DIMS_2);
    OP_CHECK(perm != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Allocate Weight transpose permutation failed."),
             return nullptr);

    auto transposedBf16 = l0op::Transpose(bf16Weight, perm, ctx.executor);
    OP_CHECK(transposedBf16 != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Transpose BF16 Weight failed."),
             return nullptr);

    auto transposedWeight = l0op::Cast(transposedBf16, op::DataType::DT_FLOAT4_E2M1, ctx.executor);
    OP_CHECK(transposedWeight != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Cast transposed BF16 Weight to FP4 failed."), return nullptr);
    if (!isInt4) {
        return transposedWeight;
    }
    auto int4View = ctx.executor->CreateView(transposedWeight, transposedWeight->GetViewShape(),
                                             transposedWeight->GetViewOffset());
    OP_CHECK(int4View != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Restore INT4 view after Weight Transpose failed."),
             return nullptr);

    int4View->SetDataType(op::DataType::DT_INT4);
    return int4View;
}

static const aclTensor* PreprocessNonTransposedToTransposedWeight(QuantContext& ctx)
{
    auto preparedWeight = TransposeWeightInt4ViaFp4(ctx.weight, ctx);
    OP_CHECK(preparedWeight != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Preprocess non-transposed Weight to transposed ND failed."),
             return nullptr);
    return preparedWeight;
}

static const aclTensor* PreprocessTransposedToNonTransposedWeight(QuantContext& ctx)
{
    op::Shape physicalShape(ctx.weight->GetViewShape());
    std::swap(physicalShape[IDX_0], physicalShape[IDX_1]);

    auto physicalView = ctx.executor->CreateView(ctx.weight, physicalShape, ctx.weight->GetViewOffset());
    OP_CHECK(physicalView != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Create transposed Weight physical view failed."), return nullptr);

    return TransposeWeightInt4ViaFp4(physicalView, ctx);
}

static aclnnStatus ProcessWeightA16C016(QuantContext& ctx)
{
    const aclTensor* preparedWeight = ctx.weight;
    if (!IsWeightNonTransposed(ctx.weight)) {
        preparedWeight = PreprocessTransposedToNonTransposedWeight(ctx);
        CHECK_RET(preparedWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ProcessWeightNd2Nz9201<IDX_0, false, true>(preparedWeight, preparedWeight->GetViewShape(), ctx);
}

static aclnnStatus ProcessWeightA16C064(QuantContext& ctx)
{
    const aclTensor* preparedWeight = ctx.weight;
    op::Shape physicalShape(ctx.weight->GetViewShape());
    if (!IsWeightNonTransposed(ctx.weight)) {
        return ProcessWeightTransNd2Nz<IDX_0>(ctx);
    }
    preparedWeight = PreprocessNonTransposedToTransposedWeight(ctx);
    CHECK_RET(preparedWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    physicalShape = preparedWeight->GetViewShape();
    return ProcessWeightNd2Nz9201<IDX_0, true>(preparedWeight, physicalShape, ctx);
}

static aclnnStatus ProcessWeightMxFp4(QuantContext& ctx)
{
    const aclTensor* preparedWeight = ctx.weight;
    bool nonTransposed = IsWeightNonTransposed(ctx.weight);
    if (nonTransposed) {
        preparedWeight = PreprocessNonTransposedToTransposedWeight(ctx);
        CHECK_RET(preparedWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    op::Shape storageShape(preparedWeight->GetViewShape());
    if (!nonTransposed) {
        std::swap(storageShape[IDX_0], storageShape[IDX_1]);
    }

    return ProcessWeightNd2Nz9201<IDX_0, true>(preparedWeight, storageShape, ctx);
}

static aclnnStatus CopyToTargetLayout(const aclTensor* source, aclTensor* target, QuantContext& ctx)
{
    auto sourceView = ctx.executor->CreateView(source, target->GetViewShape(), source->GetViewOffset());
    OP_CHECK(sourceView != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Create source view with target layout failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    auto targetView = ctx.executor->CreateView(target, target->GetViewShape(), target->GetViewOffset());
    OP_CHECK(targetView != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Create target output byte view failed."),
             return ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::ViewCopy(sourceView, targetView, ctx.executor);
    OP_CHECK(result != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ViewCopy source with target layout failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus CopyMxScale(const aclTensor* source, QuantContext& ctx)
{
    auto outputView = ctx.executor->CreateView(ctx.outWeightScale, source->GetViewShape(),
                                               ctx.outWeightScale->GetViewOffset());
    OP_CHECK(outputView != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Create MX-G32 output byte view failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    outputView->SetDataType(op::DataType::DT_INT8);
    auto result = l0op::ViewCopy(source, outputView, ctx.executor);
    OP_CHECK(result != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Copy MX-G32 scale bytes failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ConvertMxScale3D(QuantContext& ctx)
{
    int64_t groupNum = CeilDiv(ctx.weight->GetViewShape().GetDim(IDX_0), KGROUP_SIZE_MX);
    int64_t blockNum = CeilDiv(groupNum, DOUBLE);
    int64_t n = ctx.weight->GetViewShape().GetDim(IDX_1);
    auto inputStrides = ctx.weightScale->GetViewStrides();
    bool blockFirst = inputStrides[IDX_0] == n * static_cast<int64_t>(DOUBLE) &&

                      inputStrides[IDX_1] == DOUBLE && inputStrides[IDX_2] == SINGLE;

    op::Shape inputShape;
    inputShape.AppendDim(blockFirst ? blockNum : n);
    inputShape.AppendDim(blockFirst ? n : blockNum);
    inputShape.AppendDim(DOUBLE);
    auto inputView = ctx.executor->CreateView(ctx.weightScale, inputShape, ctx.weightScale->GetViewOffset());
    OP_CHECK(inputView != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Create A16MXFp4 3-D byte view failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    inputView->SetDataType(op::DataType::DT_INT8);

    const aclTensor* scaleByChannel = inputView;
    if (blockFirst) {
        int64_t permData[DIMS_3] = {IDX_1, IDX_0, IDX_2};
        auto perm = ctx.executor->AllocIntArray(permData, DIMS_3);
        OP_CHECK(perm != nullptr,
                 OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Allocate A16MXFp4 3-D transpose permutation failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
        scaleByChannel = l0op::Transpose(inputView, perm, ctx.executor);
        OP_CHECK(scaleByChannel != nullptr,
                 OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Transpose A16MXFp4 scale to {N,blocks,2} failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
    }
    return CopyMxScale(scaleByChannel, ctx);
}

static aclnnStatus ConvertMxScale2DTo3D(QuantContext& ctx)
{
    int64_t groupNum = CeilDiv(ctx.weight->GetViewShape().GetDim(IDX_0), KGROUP_SIZE_MX);

    int64_t n = ctx.weight->GetViewShape().GetDim(IDX_1);
    auto inputStrides = ctx.weightScale->GetViewStrides();
    bool groupFirst = inputStrides[IDX_0] == n && inputStrides[IDX_1] == SINGLE;

    op::Shape inputShape;
    inputShape.AppendDim(groupFirst ? groupNum : n);
    inputShape.AppendDim(groupFirst ? n : groupNum);
    auto inputView = ctx.executor->CreateView(ctx.weightScale, inputShape, ctx.weightScale->GetViewOffset());
    OP_CHECK(inputView != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Create A16MXFp4 2-D byte view failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    inputView->SetDataType(op::DataType::DT_INT8);

    // 输出物理顺序固定为 {N,G}；仅 {G,N} 输入执行一次真实转置。
    const aclTensor* scaleByChannel = inputView;
    if (groupFirst) {
        int64_t permData[DIMS_2] = {IDX_1, IDX_0};
        auto perm = ctx.executor->AllocIntArray(permData, DIMS_2);
        OP_CHECK(perm != nullptr,
                 OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Allocate A16MXFp4 2-D transpose permutation failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
        scaleByChannel = l0op::Transpose(inputView, perm, ctx.executor);
        OP_CHECK(scaleByChannel != nullptr,
                 OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Transpose A16MXFp4 scale to {N,G} failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
    }

    // G 为奇数时仅在每个 N 对应的 G 尾部补一个原始零字节。
    if (groupNum % DOUBLE != 0) {
        int64_t paddingData[DIMS_2 * DOUBLE] = {
            0, 0, // N 不补
            0, 1  // G 尾部补 1
        };
        auto paddings = ctx.executor->AllocIntArray(paddingData, DIMS_2 * DOUBLE);
        OP_CHECK(paddings != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Allocate odd-G MX padding failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
        auto paddingTensor = ctx.executor->ConvertToTensor(paddings, op::DataType::DT_INT64);
        OP_CHECK(paddingTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Create odd-G MX padding tensor failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
        scaleByChannel = l0op::Pad(scaleByChannel, paddingTensor, ctx.executor);
        OP_CHECK(scaleByChannel != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Pad odd-G MX scale with zero failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
    }
    return CopyMxScale(scaleByChannel, ctx);
}

static aclnnStatus ProcessMxScale(QuantContext& ctx)
{
    if (ctx.weightScale->GetViewShape().GetDimNum() == DIMS_2) {
        return ConvertMxScale2DTo3D(ctx);
    }
    return ConvertMxScale3D(ctx);
}

static aclnnStatus ProcessWeightScaleA16C016(QuantContext& ctx)
{
    auto ret = CopyToTargetLayout(ctx.weightScale, ctx.outWeightScale, ctx);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (ctx.weightOffsetOptional != nullptr) {
        return CopyToTargetLayout(ctx.weightOffsetOptional, ctx.outWeightOffsetOptional, ctx);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus ProcessWeightScaleA16C064(QuantContext& ctx)
{
    const auto& scaleShape = ctx.weightScale->GetViewShape();
    op::Shape flatScaleShape;
    flatScaleShape.AppendDim(scaleShape.GetShapeSize());
    auto flatScale = ctx.executor->CreateView(ctx.weightScale, flatScaleShape, ctx.weightScale->GetViewOffset());
    OP_CHECK(flatScale != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Create flat A16W4 weightScale view failed."),
             return ACLNN_ERR_INNER_NULLPTR);

    auto scaleFp32 = l0op::Cast(flatScale, op::DataType::DT_FLOAT, ctx.executor);
    OP_CHECK(scaleFp32 != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Cast weightScale from FP16/BF16 to FP32 failed."),
             return ACLNN_ERR_INNER_NULLPTR);

    auto scaleInt32 = ctx.executor->CreateView(scaleFp32, scaleFp32->GetViewShape(), scaleFp32->GetViewOffset());
    OP_CHECK(scaleInt32 != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Create INT32 view for A16W4 weightScale failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    scaleInt32->SetDataType(op::DataType::DT_INT32);

    auto andMaskScalar = ctx.executor->AllocScalar(static_cast<int32_t>(0xFFFFE000U));
    OP_CHECK(andMaskScalar != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Alloc A16W4 scale AND mask failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    auto andMask = ctx.executor->ConvertToTensor(andMaskScalar, op::DataType::DT_INT32);
    OP_CHECK(andMask != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Convert A16W4 scale AND mask failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    auto maskedInt32 = l0op::BitwiseAnd(scaleInt32, andMask, ctx.executor);
    OP_CHECK(maskedInt32 != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BitwiseAnd A16W4 weightScale failed."),
             return ACLNN_ERR_INNER_NULLPTR);

    auto maskedUint32 = ctx.executor->CreateView(maskedInt32, maskedInt32->GetViewShape(),
                                                 maskedInt32->GetViewOffset());
    OP_CHECK(maskedUint32 != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Create UINT32 view for A16W4 weightScale failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    maskedUint32->SetDataType(op::DataType::DT_UINT32);
    auto encodedInt64 = l0op::Cast(maskedUint32, op::DataType::DT_INT64, ctx.executor);
    OP_CHECK(encodedInt64 != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Cast A16W4 scale bits to INT64 failed."),
             return ACLNN_ERR_INNER_NULLPTR);

    auto orMaskScalar = ctx.executor->AllocScalar(static_cast<int64_t>(UINT64_C(0x0000400000000000)));
    OP_CHECK(orMaskScalar != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Alloc A16W4 scale OR mask failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    auto orMask = ctx.executor->ConvertToTensor(orMaskScalar, op::DataType::DT_INT64);
    OP_CHECK(orMask != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Convert A16W4 scale OR mask failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    auto encodedInt64WithFlag = l0op::BitwiseOr(encodedInt64, orMask, ctx.executor);
    OP_CHECK(encodedInt64WithFlag != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BitwiseOr A16W4 weightScale failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    auto encodedScale = ctx.executor->CreateView(encodedInt64WithFlag, flatScaleShape,
                                                 encodedInt64WithFlag->GetViewOffset());
    OP_CHECK(encodedScale != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Create UINT64 A16W4 scale view failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    encodedScale->SetDataType(op::DataType::DT_UINT64);
    return CopyToTargetLayout(encodedScale, ctx.outWeightScale, ctx);
}

static aclnnStatus CopyPerGroupScaleOrOffsetToOutput(const aclTensor* source, aclTensor* target, int64_t g, int64_t n,
                                                     QuantContext& ctx)
{
    auto inputStrides = source->GetViewStrides();
    bool storageNG = inputStrides[IDX_0] == SINGLE && inputStrides[IDX_1] == g;
    const aclTensor* scaleOrOffsetGn = source;

    if (storageNG) {
        // 输入逻辑 shape 为 {G,N}；根据 N、G 重建连续物理 View {N,G}，不依赖输入 storageShape。
        op::Shape storageShape;
        storageShape.AppendDim(n);
        storageShape.AppendDim(g);
        auto storageView = ctx.executor->CreateView(source, storageShape, source->GetViewOffset());
        OP_CHECK(storageView != nullptr,
                 OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Create PerGroup parameter storage view failed."),
                 return ACLNN_ERR_INNER_NULLPTR);

        int64_t permData[DIMS_2] = {1, 0};
        auto perm = ctx.executor->AllocIntArray(permData, DIMS_2);
        OP_CHECK(perm != nullptr,
                 OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Allocate PerGroup parameter transpose permutation failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
        scaleOrOffsetGn = l0op::Transpose(storageView, perm, ctx.executor);
        OP_CHECK(scaleOrOffsetGn != nullptr,
                 OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Transpose PerGroup parameter from {N,G} to {G,N} failed."),
                 return ACLNN_ERR_INNER_NULLPTR);
    }
    auto result = l0op::ViewCopy(scaleOrOffsetGn, target, ctx.executor);
    OP_CHECK(result != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ViewCopy PerGroup parameter failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ProcessPerGroupScaleAndOffset(QuantContext& ctx)
{
    auto weightShape = ctx.weight->GetViewShape();
    int64_t g = CeilDiv(weightShape.GetDim(IDX_0), ctx.kGroupSize);
    int64_t n = weightShape.GetDim(IDX_1);
    auto ret = CopyPerGroupScaleOrOffsetToOutput(ctx.weightScale, ctx.outWeightScale, g, n, ctx);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (ctx.weightOffsetOptional != nullptr) {
        return CopyPerGroupScaleOrOffsetToOutput(ctx.weightOffsetOptional, ctx.outWeightOffsetOptional, g, n, ctx);
    }
    return ACLNN_SUCCESS;
}

} // namespace

// ===== A16S4/A16F4/A8W4-MX 检查组合：按粒度拼装，条目间无内部分流 =====
// 紧凑 4-bit 输入公共检查（打包维须为偶数；打包视图须连续/严格转置且偏移字节对齐，
// ND 直拷与 NZ TransData 共用此前提），ND 入 + NZ 出的条目均可复用
// scale/offset/bias 为透传参数：out 侧 same 校验已保证输出与输入完全一致，
// 其自身的 dtype/format/shape 约束 preprocess 不消费也不处理，统一下放下游（wqbmmv2/qbmmv4）拦截
const std::vector<CheckFunc> INPUT_4BIT_BASE_CHECKS = {CheckWeightNotEmpty, CheckWeightFormatND,
                                                       CheckWeightPackingDimEven, CheckWeight4BitPackedView};

// A16F4/A8W4-MX 条目无 offset 直拷 process：入参 offset 必须 nullptr，否则会被静默丢弃
const std::vector<CheckFunc> OFFSET_NULL_CHECK = {CheckWeightOffsetOptionalNull};

// out weight ND 直拷检查
const std::vector<CheckFunc> OUT_WEIGHT_ND_CHECKS = {CheckOutWeightSameAsInput};

// 非转置防御（judge 已保证非转置）：供 CombineChecks 条目在 OUT_WEIGHT_NZ_C016/C032_CHECKS 前插入
const std::vector<CheckFunc> WEIGHT_NOT_TRANS_CHECK = {CheckWeightNotTrans};

// out weight NZ_C0_16 分形检查（非转置防御由 WEIGHT_NOT_TRANS_CHECK 承担）
const std::vector<CheckFunc> OUT_WEIGHT_NZ_C016_CHECKS = {
    CheckOutWeightSameBase, CheckOutWeightDtypeSame, CheckOutWeightNzStorageDim,
    CheckOutWeightNz<NZ_C0_16, op::Format::FORMAT_FRACTAL_NZ_C0_16, NON_TRANS_WEIGHT>};

// out weight NZ_C0_32 分形检查（非转置 -> N 块在前）
const std::vector<CheckFunc> OUT_WEIGHT_NZ_C032_CHECKS = {
    CheckOutWeightSameBase, CheckOutWeightDtypeSame, CheckOutWeightNzStorageDim,
    CheckOutWeightNz<NZ_C0_32, op::Format::FORMAT_FRACTAL_NZ_C0_32, NON_TRANS_WEIGHT>};

// 输出 scale/offset/bias 与入参一致性检查（公共尾部）
const std::vector<CheckFunc> OUT_TAIL_CHECKS = {CheckOutWeightScaleSameAsInput, CheckOutWeightOffsetOptionalSameAsInput,
                                                CheckOutBiasOptionalSameAsInput};

// 输出 scale/bias 与入参一致性检查（offset 无直拷 process、必须 nullptr 的条目用此尾部）
const std::vector<CheckFunc> OUT_SCALE_BIAS_CHECKS = {CheckOutWeightScaleSameAsInput, CheckOutBiasOptionalSameAsInput};

template <typename... Groups>
static std::vector<CheckFunc> CombineChecks(const Groups&... groups)
{
    std::vector<CheckFunc> combined;
    (combined.insert(combined.end(), groups.begin(), groups.end()), ...);
    return combined;
}

const std::unordered_map<NpuArch, std::vector<DataFlowEntry>> NPU_DATA_FLOW_REGISTRY_MAP = {
    {NpuArch::DAV_3510,
     {{.judge = IsMMMxA8W4DataFlow,
       .checks = {CheckWeightNotEmpty, CheckWeightFormatND,
                  CheckWeightTrans,              // 校验 weight 最后 2 维是否转置
                  CheckWeightOffsetOptionalNull, // offset 无直拷 process，入参必须 nullptr，否则被静默丢弃
                  CheckOutWeightSameBase,        // out 与 input 的 nullptr/empty/viewShape 一致
                  CheckOutWeightDtypeSame,       // NZ 出 dtype 与输入一致
                  CheckOutWeightNzStorageDim,    // 输出 weight 为 NZ 格式，校验 storageShape 维度是否正确
                  // 校验输出 weight storageShape 符合 C0_32（转置 -> K 块在前）
                  CheckOutWeightNz<NZ_C0_32, op::Format::FORMAT_FRACTAL_NZ_C0_32, TRANS_WEIGHT>,
                  CheckOutWeightScaleSameAsInput, CheckOutBiasOptionalSameAsInput},
       .processes = {ProcessWeightNd2Nz<IDX_0, TRANS_WEIGHT>, // weight ND→NZ 转换，模板参为 k 下标与转置状态
                     ProcessWeightScaleDirectCopy, ProcessBiasDirectCopy}},
      {.judge = IsGMMMxA8W4DataFlow,
       .checks = {CheckWeightNotEmpty, CheckWeightFormatND,
                  CheckWeightTrans,              // 校验 weight 最后 2 维是否转置
                  CheckWeightOffsetOptionalNull, // offset 无直拷 process，入参必须 nullptr，否则被静默丢弃
                  CheckOutWeightSameBase,        // out 与 input 的 nullptr/empty/viewShape 一致
                  CheckOutWeightDtypeSame,       // NZ 出 dtype 与输入一致
                  CheckOutWeightNzStorageDim,    // 输出 weight 为 NZ 格式，校验 storageShape 维度是否正确
                  // 校验输出 weight storageShape 符合 C0_32（转置 -> K 块在前）
                  CheckOutWeightNz<NZ_C0_32, op::Format::FORMAT_FRACTAL_NZ_C0_32, TRANS_WEIGHT>,
                  CheckOutWeightScaleSameAsInput, CheckOutBiasOptionalSameAsInput},
       .processes = {ProcessWeightNd2Nz<IDX_1, TRANS_WEIGHT>, // weight ND→NZ 转换，模板参为 k 下标与转置状态
                     ProcessWeightScaleDirectCopy, ProcessBiasDirectCopy}},
      {.judge = IsMMA16S4PerTensorDataFlow,
       .checks = {CheckWeightNotEmpty, CheckWeightFormatND,
                  CheckWeightPackingDimEven, // 紧凑 4-bit 打包维须为偶数
                  CheckWeight4BitPackedView, // 4-bit 打包视图须连续/严格转置且偏移字节对齐
                  // per-tensor 不支持 NZ：outWeight 须与 weight 完全一致（format==），转置/非转置 weight 均直拷
                  CheckOutWeightSameAsInput, CheckOutWeightScaleSameAsInput, CheckOutWeightOffsetOptionalSameAsInput,
                  CheckOutBiasOptionalSameAsInput},
       .processes = {ProcessWeightDirectCopy, ProcessWeightScaleDirectCopy, ProcessWeightOffsetDirectCopy,
                     ProcessBiasDirectCopy}},
      // A16S4 per-channel 转置：ND 直拷（物理透传）
      {.judge = IsMMA16S4PerChannelTransDataFlow,
       .checks = CombineChecks(INPUT_4BIT_BASE_CHECKS, OUT_WEIGHT_ND_CHECKS, OUT_TAIL_CHECKS),
       .processes = {ProcessWeightDirectCopy, ProcessWeightScaleDirectCopy, ProcessWeightOffsetDirectCopy,
                     ProcessBiasDirectCopy}},
      // A16S4 per-channel 非转置 + NZ_C0_16 出：ND→NZ 转换（非转置仅支持 NZ 出）
      {.judge = IsMMA16S4PerChannelNonTransNzDataFlow,
       .checks = CombineChecks(INPUT_4BIT_BASE_CHECKS, WEIGHT_NOT_TRANS_CHECK, OUT_WEIGHT_NZ_C016_CHECKS,
                               OUT_TAIL_CHECKS),
       .processes = {ProcessWeightNd2Nz<IDX_0, NON_TRANS_WEIGHT>, ProcessWeightScaleDirectCopy,
                     ProcessWeightOffsetDirectCopy, ProcessBiasDirectCopy}},
      // A16S4 per-group 转置：ND 直拷（scale 形状须匹配 per-group 分组语义）
      {.judge = IsMMA16S4PerGroupTransDataFlow,
       .checks = CombineChecks(INPUT_4BIT_BASE_CHECKS, OUT_WEIGHT_ND_CHECKS, OUT_TAIL_CHECKS),
       .processes = {ProcessWeightDirectCopy, ProcessWeightScaleDirectCopy, ProcessWeightOffsetDirectCopy,
                     ProcessBiasDirectCopy}},
      // A16S4 per-group 非转置 + NZ_C0_16 出：ND→NZ 转换（非转置仅支持 NZ 出）
      {.judge = IsMMA16S4PerGroupNonTransNzDataFlow,
       .checks = CombineChecks(INPUT_4BIT_BASE_CHECKS, WEIGHT_NOT_TRANS_CHECK, OUT_WEIGHT_NZ_C016_CHECKS,
                               OUT_TAIL_CHECKS),
       .processes = {ProcessWeightNd2Nz<IDX_0, NON_TRANS_WEIGHT>, ProcessWeightScaleDirectCopy,
                     ProcessWeightOffsetDirectCopy, ProcessBiasDirectCopy}},
      // A16F4 per-group 非转置 + NZ_C0_16 出：ND→NZ 转换（NZ 出仅支持非转置）
      {.judge = IsMMA16F4PerGroupDataFlow,
       .checks = CombineChecks(INPUT_4BIT_BASE_CHECKS, WEIGHT_NOT_TRANS_CHECK, OFFSET_NULL_CHECK,
                               OUT_WEIGHT_NZ_C016_CHECKS, OUT_SCALE_BIAS_CHECKS),
       .processes = {ProcessWeightNd2Nz<IDX_0, NON_TRANS_WEIGHT>, ProcessWeightScaleDirectCopy, ProcessBiasDirectCopy}},
      // A16MXFP4 转置：ND 直拷（wqbmmv2 MX kernel 支持 ND 转置输入；offset 不支持必须 nullptr）
      {.judge = IsMMA16MXF4TransDataFlow,
       .checks = CombineChecks(INPUT_4BIT_BASE_CHECKS, OFFSET_NULL_CHECK, OUT_WEIGHT_ND_CHECKS, OUT_TAIL_CHECKS),
       .processes = {ProcessWeightDirectCopy, ProcessWeightScaleDirectCopy, ProcessBiasDirectCopy}},
      // A16MXFP4 非转置 + NZ_C0_16 出：ND→NZ 分形转换（NZ 出仅支持非转置）
      {.judge = IsMMA16MXF4NonTransNzDataFlow,
       .checks = CombineChecks(INPUT_4BIT_BASE_CHECKS, WEIGHT_NOT_TRANS_CHECK, OFFSET_NULL_CHECK,
                               OUT_WEIGHT_NZ_C016_CHECKS, OUT_SCALE_BIAS_CHECKS),
       .processes = {ProcessWeightNd2Nz<IDX_0, NON_TRANS_WEIGHT>, ProcessWeightScaleDirectCopy, ProcessBiasDirectCopy}},
      // ===== A16W8 数据流 =====
      // A16INT8 per-tensor：转置/非转置均 ND 直拷，offset/bias 透传
      {.judge = IsMMA16INT8PerTensorDataFlow,
       .checks = {CheckOutWeightSameAsInput, CheckOutWeightScaleSameAsInput, CheckOutWeightOffsetOptionalSameAsInput,
                  CheckOutBiasOptionalSameAsInput},
       .processes = {ProcessWeightDirectCopy, ProcessWeightScaleDirectCopy, ProcessWeightOffsetDirectCopy,
                     ProcessBiasDirectCopy}},
      // A16INT8 per-channel：转置/非转置均 ND 直拷（物理透传）
      {.judge = IsMMA16INT8PerChannelDataFlow,
       .checks = {CheckOutWeightSameAsInput, CheckOutWeightScaleSameAsInput, CheckOutWeightOffsetOptionalSameAsInput,
                  CheckOutBiasOptionalSameAsInput},
       .processes = {ProcessWeightDirectCopy, ProcessWeightScaleDirectCopy, ProcessWeightOffsetDirectCopy,
                     ProcessBiasDirectCopy}},
      // A16INT8 per-group：转置/非转置均 ND 直拷（wqbmmv2 INT8-NZ 不支持 per-group）
      {.judge = IsMMA16INT8PerGroupDataFlow,
       .checks = {CheckOutWeightSameAsInput, CheckOutWeightScaleSameAsInput, CheckOutWeightOffsetOptionalSameAsInput,
                  CheckOutBiasOptionalSameAsInput},
       .processes = {ProcessWeightDirectCopy, ProcessWeightScaleDirectCopy, ProcessWeightOffsetDirectCopy,
                     ProcessBiasDirectCopy}},
      // A16FP8HIF8 per-channel（E4M3FN/HIF8）：转置/非转置均 ND 直拷；offset 无透传 process（wqbmmv2 FP8 无 offset）
      {.judge = IsMMA16FP8HIF8PerChannelDataFlow,
       .checks = {CheckOutWeightSameAsInput, CheckOutWeightScaleSameAsInput, CheckOutWeightOffsetOptionalSameAsInput,
                  CheckOutBiasOptionalSameAsInput},
       .processes = {ProcessWeightDirectCopy, ProcessWeightScaleDirectCopy, ProcessBiasDirectCopy}},
      // ===== A8F4 T-CG 数据流 =====
      // 转置：ND 直拷（输入侧查非空/format/打包视图；打包奇偶/x2Offset 等下沉下游 V5 入口拦截）
      {.judge = IsMMA8F4TcgTransDataFlow,
       .checks = {CheckWeightNotEmpty, CheckWeightFormatND, CheckWeight4BitPackedView, CheckOutWeightSameAsInput,
                  CheckOutWeightScaleSameAsInput, CheckOutWeightOffsetOptionalSameAsInput,
                  CheckOutBiasOptionalSameAsInput},
       .processes = {ProcessWeightDirectCopy, ProcessWeightScaleDirectCopy, ProcessWeightOffsetDirectCopy,
                     ProcessBiasDirectCopy}},
      // 非转置 + C0_32 出：复用 MX 的 TransData 注册（非转置喂入使 c0 轴落在 N）；
      // x2Offset 由下游 WeightNz 入口拦截
      {.judge = IsMMA8F4TcgNonTransDataFlow,
       .checks = CombineChecks(INPUT_4BIT_BASE_CHECKS, OUT_WEIGHT_NZ_C032_CHECKS, OUT_TAIL_CHECKS),
       .processes = {ProcessWeightNd2Nz<IDX_0, NON_TRANS_WEIGHT>, ProcessWeightScaleDirectCopy,
                     ProcessWeightOffsetDirectCopy, ProcessBiasDirectCopy}}}},
    {NpuArch::DAV_9201,
     {{.judge = IsMMMxA8W4DataFlow,
       .checks =
           {CheckWeightNotEmpty, CheckWeightFormatND,
            CheckWeightTrans, // 校验 weight 最后 2 维是否转置
            CheckWeightScaleNotEmpty, CheckWeightScaleFormatND,
            CheckWeightScaleMx<DIMS_3>, // 校验 weightScale 的 viewShape 是否符合 Mx 场景的 3 维形式 (k/64, n, 2)
            CheckWeightScaleTrans<IDX_0>, // 校验 weightScale 的第 0 维和第 1 维是否转置
            CheckWeightOffsetOptionalNull,
            CheckKGroupSizeMx, // Mx 场景要求 kGroupSize 为 32
            CheckOutWeightSameBase, CheckOutWeightDtypeSame,
            CheckOutWeightNzStorageDim, // 输出 weight 为 NZ 格式，校验 storageShape 维度是否正确
            CheckOutWeightTransNz<NZ_C0_64, op::Format::FORMAT_FRACTAL_NZ>, // 校验输出 weight storageShape 符合 C0_64
            CheckOutWeightScaleNotNullEmpty, CheckOutWeightScaleFormatND, CheckOutWeightScaleDtypeSame,
            CheckOutWeightScaleViewShapeSame, CheckOutWeightScaleStorageShapeSame, CheckOutBiasOptionalSameAsInput},
       .processes = {ProcessWeightTransNd2Nz<IDX_0>, // 对 weight 进行 Nd2Nz 转换，参数表示 k 在 viewShape 中的下标
                     ProcessWeightScaleDirectCopy, ProcessBiasDirectCopy}},
      {.judge = IsMMMxA16HIF4DataFlow,
       .checks =
           {
               CheckWeightNotEmpty,
               CheckWeightFormatND,
               CheckWeightTrans, // 校验 weight 最后 2 维是否转置
               CheckWeightScaleNotEmpty,
               CheckWeightScaleFormatND,
               CheckWeightScaleMx<DIMS_3, SINGLE>, // 校验 weightScale 的 viewShape 是否符合 Mx 场景的 3 维形式 (k/64,
                                                   // n, 1)
               CheckWeightScaleTrans<IDX_0>, // 校验 weightScale 的第 0 维和第 1 维是否转置
               CheckWeightOffsetOptionalNull,
               CheckGroupSize<KGROUP_SIZE_HIF4>, // HIF4 场景要求 kGroupSize 为 64
               CheckOutWeightSameBase,
               CheckOutWeightDtypeSame,
               CheckOutWeightNzStorageDim, // 输出 weight 为 NZ 格式，校验 storageShape 维度是否正确
               CheckOutWeightTransNz<NZ_C0_64, op::Format::FORMAT_FRACTAL_NZ>, // 校验输出 weight storageShape 符合
                                                                               // C0_64
               CheckOutWeightScaleNotNullEmpty,
               CheckOutWeightScaleFormatND,
               CheckOutWeightScaleDtypeSame,
               CheckOutWeightScaleViewShapeSame,
               CheckOutWeightScaleStorageShapeSame,
               CheckOutBiasOptionalSameAsInput,
           },
       .processes = {ProcessWeightTransNd2Nz<IDX_0>, // 对 weight 进行 Nd2Nz 转换，参数表示 k 在 viewShape 中的下标
                     ProcessWeightScaleDirectCopy, ProcessBiasDirectCopy}},
      {.judge = IsMMC064A16W4DataFlow,
       .checks =
           {
               CheckWeightNotEmpty,
               CheckWeightFormatND,
               CheckWeightLayoutA16,
               CheckGroupSize<KGROUP_SIZE_INT4>,
               CheckOutWeightNotNullEmpty,
               CheckOutWeightDtypeSame,
               CheckOutWeightViewShapeSame,
               CheckOutWeightNzStorageDim,
               CheckOutWeightTransNz<NZ_C0_64, op::Format::FORMAT_FRACTAL_NZ>,
               CheckWeightScaleNotEmpty,
               CheckWeightScaleFormatND,
               CheckPerChannelScaleOrOffset<false>,
               CheckOutWeightScaleNotNullEmpty,
               CheckOutWeightScaleFormatND,
               CheckOutC064Scale,
               CheckWeightOffsetOptionalNull,
               CheckOutBiasOptionalSameAsInput,
           },
       .processes =
           {
               ProcessWeightA16C064,
               ProcessWeightScaleA16C064,
               ProcessBiasDirectCopy,
           }},
      {.judge = IsMMA16S4PerTensorDataFlow_9201,
       .checks =
           {
               CheckWeightNotEmpty,
               CheckWeightFormatND,
               CheckWeightLayoutA16,
               CheckGroupSize<KGROUP_SIZE_INT4>,
               CheckWeightEven,
               CheckOutWeightNotNullEmpty,
               CheckOutWeightDtypeSame,
               CheckOutWeightViewShapeSame,
               CheckOutWeightNzStorageDim,
               CheckOutWeightNz9201,
               CheckWeightScaleNotEmpty,
               CheckWeightScaleFormatND,
               CheckPerTensorScaleOrOffset<false>,
               CheckOutWeightScaleNotNullEmpty,
               CheckOutWeightScaleFormatND,
               CheckOutWeightScaleDtypeSame,
               CheckOutPerTensorScaleOrOffset<false>,
               CheckOffsetNotEmpty,
               CheckOffsetDtype,
               CheckOffsetFormat,
               CheckPerTensorScaleOrOffset<true>,
               CheckOutOffsetNotEmpty,
               CheckOutOffsetFormat,
               CheckOutOffsetDtypeSame,
               CheckOutPerTensorScaleOrOffset<true>,
               CheckOutBiasOptionalSameAsInput,
           },
       .processes =
           {
               ProcessWeightA16C016,
               ProcessWeightScaleA16C016,
               ProcessBiasDirectCopy,
           }},
      {.judge = IsMMA16S4PerChannelDataFlow,
       .checks =
           {
               CheckWeightNotEmpty,
               CheckWeightFormatND,
               CheckWeightLayoutA16,
               CheckGroupSize<KGROUP_SIZE_INT4>,
               CheckWeightEven,
               CheckOutWeightNotNullEmpty,
               CheckOutWeightDtypeSame,
               CheckOutWeightViewShapeSame,
               CheckOutWeightNzStorageDim,
               CheckOutWeightNz9201,
               CheckWeightScaleNotEmpty,
               CheckWeightScaleFormatND,
               CheckPerChannelScaleOrOffset<false>,
               CheckOutWeightScaleNotNullEmpty,
               CheckOutWeightScaleFormatND,
               CheckOutWeightScaleDtypeSame,
               CheckOutPerChannelScaleOrOffset<false>,
               CheckOffsetNotEmpty,
               CheckOffsetDtype,
               CheckOffsetFormat,
               CheckPerChannelScaleOrOffset<true>,
               CheckOutOffsetNotEmpty,
               CheckOutOffsetFormat,
               CheckOutOffsetDtypeSame,
               CheckOutPerChannelScaleOrOffset<true>,
               CheckOutBiasOptionalSameAsInput,
           },
       .processes =
           {
               ProcessWeightA16C016,
               ProcessWeightScaleA16C016,
               ProcessBiasDirectCopy,
           }},
      {.judge = IsMMC064PerGroupA16W4DataFlow,
       .checks =
           {
               CheckWeightNotEmpty,
               CheckWeightFormatND,
               CheckWeightLayoutA16,
               CheckGroupSize<KGROUP_SIZE_HIF4, KGROUP_SIZE_128>,
               CheckWeightKDivisibleByGroupSize,
               CheckOutWeightNotNullEmpty,
               CheckOutWeightDtypeSame,
               CheckOutWeightViewShapeSame,
               CheckOutWeightNzStorageDim,
               CheckOutWeightTransNz<NZ_C0_64, op::Format::FORMAT_FRACTAL_NZ>,
               CheckWeightScaleNotEmpty,
               CheckWeightScaleFormatND,
               CheckPerGroupScaleOrOffset<false>,
               CheckOutWeightScaleNotNullEmpty,
               CheckOutWeightScaleFormatND,
               CheckOutWeightScaleDtypeSame,
               CheckOutPerGroupScaleOrOffset<false>,
               CheckWeightOffsetOptionalNull,
               CheckOutBiasOptionalSameAsInput,
           },
       .processes =
           {
               ProcessWeightA16C064,
               ProcessPerGroupScaleAndOffset,
               ProcessBiasDirectCopy,
           }},
      {.judge = IsMMA16S4PerGroupDataFlow,
       .checks =
           {
               CheckWeightNotEmpty,
               CheckWeightFormatND,
               CheckWeightLayoutA16,
               CheckGroupSize<KGROUP_SIZE_MX, KGROUP_SIZE_HIF4, KGROUP_SIZE_128, KGROUP_SIZE_256>,
               CheckWeightEven,
               CheckOutWeightNotNullEmpty,
               CheckOutWeightDtypeSame,
               CheckOutWeightViewShapeSame,
               CheckOutWeightNzStorageDim,
               CheckOutWeightNz9201,
               CheckWeightScaleNotEmpty,
               CheckWeightScaleFormatND,
               CheckPerGroupScaleOrOffset<false>,
               CheckOutWeightScaleNotNullEmpty,
               CheckOutWeightScaleFormatND,
               CheckOutWeightScaleDtypeSame,
               CheckOutPerGroupScaleOrOffset<false>,
               CheckOffsetNotEmpty,
               CheckOffsetDtype,
               CheckOffsetFormat,
               CheckPerGroupScaleOrOffset<true>,
               CheckOutOffsetNotEmpty,
               CheckOutOffsetFormat,
               CheckOutOffsetDtypeSame,
               CheckOutPerGroupScaleOrOffset<true>,
               CheckOutBiasOptionalSameAsInput,
           },
       .processes =
           {
               ProcessWeightA16C016,
               ProcessPerGroupScaleAndOffset,
               ProcessBiasDirectCopy,
           }},
      {.judge = IsMMA16Fp4PerGroupDataFlow,
       .checks =
           {
               CheckWeightNotEmpty,
               CheckWeightFormatND,
               CheckWeightLayoutA16,
               CheckGroupSize<KGROUP_SIZE_MX, KGROUP_SIZE_HIF4, KGROUP_SIZE_128, KGROUP_SIZE_256>,
               CheckWeightEven,
               CheckOutWeightNotNullEmpty,
               CheckOutWeightDtypeSame,
               CheckOutWeightViewShapeSame,
               CheckOutWeightNzStorageDim,
               CheckOutWeightNz9201,
               CheckWeightScaleNotEmpty,
               CheckWeightScaleFormatND,
               CheckPerGroupScaleOrOffset<false>,
               CheckOutWeightScaleNotNullEmpty,
               CheckOutWeightScaleFormatND,
               CheckOutWeightScaleDtypeSame,
               CheckOutPerGroupScaleOrOffset<false>,
               CheckWeightOffsetOptionalNull,
               CheckOutBiasOptionalSameAsInput,
           },
       .processes =
           {
               ProcessWeightA16C016,
               ProcessPerGroupScaleAndOffset,
               ProcessBiasDirectCopy,
           }},
      {.judge = IsMMMxA16W4DataFlow,
       .checks =
           {
               CheckWeightNotEmpty,
               CheckWeightFormatND,
               CheckWeightLayoutA16,
               CheckKGroupSizeMx,
               CheckOutWeightNotNullEmpty,
               CheckOutWeightDtypeSame,
               CheckOutWeightViewShapeSame,
               CheckOutWeightNzStorageDim,
               CheckOutWeightTransNz<NZ_C0_64, op::Format::FORMAT_FRACTAL_NZ>,
               CheckWeightScaleNotEmpty,
               CheckWeightScaleFormatND,
               CheckMxA16ScaleDim,
               CheckMxA16ScaleShape,
               CheckMxA16ScaleLayout,
               CheckOutWeightScaleNotNullEmpty,
               CheckOutWeightScaleFormatND,
               CheckOutWeightScaleDtypeSame,
               CheckOutMxScale,
               CheckWeightOffsetOptionalNull,
               CheckOutBiasOptionalSameAsInput,
           },
       .processes = {
           ProcessWeightMxFp4,
           ProcessMxScale,
           ProcessBiasDirectCopy,
       }}}}};
