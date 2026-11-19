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
#include "weight_quant_preprocess_check.h"
#include "weight_quant_preprocess_process.h"

#include <inttypes.h>
#include <utility>

#include "common/inc/op_host/op_const_def_math.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"

namespace WeightQuantPreprocess {

int64_t CeilDiv(int64_t a, int64_t b)
{
    if (b == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CeilDiv divisor b must not be zero.");
        return INT64_MIN;
    }
    return (a + b - 1) / b;
}

static bool IsMMMxA8Base(const QuantContext& ctx)
{
    return ctx.xDtype == op::DataType::DT_FLOAT8_E4M3FN && ctx.xScaleDtype == op::DataType::DT_FLOAT8_E8M0 &&
           ctx.weight->GetViewShape().GetDimNum() == DIMS_2;
}

static bool IsMMMxA8W4DataFlow(QuantContext& ctx)
{
    if (IsMMMxA8Base(ctx) && ctx.weight->GetDataType() == op::DataType::DT_FLOAT4_E2M1 &&
        ctx.weightScale->GetDataType() == op::DataType::DT_FLOAT8_E8M0) {
        ctx.dataFlow = QuantDataFlow::MM_MX_A8W4;
        return true;
    }
    return false;
}

static bool IsMMMxA8HIF4DataFlow(QuantContext& ctx)
{
    if (IsMMMxA8Base(ctx) && ctx.weight->GetDataType() == Ops::Math::DT_HIFLOAT4 &&
        ctx.weightScale->GetDataType() == Ops::Math::DT_HIFLOAT4_SCALE) {
        ctx.dataFlow = QuantDataFlow::MM_MX_A8HIF4;
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
bool IsTensor2DTransposed(const aclTensor* tensor2D)
{
    auto viewShape = tensor2D->GetViewShape();
    auto viewStrides = tensor2D->GetViewStrides();
    size_t dimNum = viewShape.GetDimNum();
    size_t lastIdx = dimNum - IDX_1;
    size_t secondLastIdx = dimNum - IDX_2;
    return viewStrides[lastIdx] == viewShape.GetDim(secondLastIdx) && viewStrides[secondLastIdx] == SINGLE;
}

// 非转置探测：最后两维 strides [size(-1), 1]；与 IsTensor2DTransposed 互补，均不命中 -> UNSUPPORTED
bool IsTensor2DNonTransposed(const aclTensor* tensor2D)
{
    auto viewShape = tensor2D->GetViewShape();
    auto viewStrides = tensor2D->GetViewStrides();
    size_t dimNum = viewShape.GetDimNum();
    size_t lastIdx = dimNum - IDX_1;
    size_t secondLastIdx = dimNum - IDX_2;
    return viewStrides[lastIdx] == SINGLE && viewStrides[secondLastIdx] == viewShape.GetDim(lastIdx);
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
        IsTensor2DTransposed(ctx.weight)) {
        ctx.dataFlow = QuantDataFlow::MM_A16S4_PERCHANNEL;
        return true;
    }
    return false;
}

static bool IsMMA16S4PerChannelNonTransNzDataFlow(QuantContext& ctx)
{
    if (IsA16W4Base<op::DataType::DT_INT4>(ctx) && IsXScaleUndefined(ctx) && IsScalePerChannel(ctx) &&
        !IsTensor2DTransposed(ctx.weight) && IsOutWeightNzC016(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16S4_PERCHANNEL;
        return true;
    }
    return false;
}

// A16S4 per-group：同 per-channel 的两路拆分（转置 ND 直拷 / 非转置 NZ_C0_16 转换）
static bool IsMMA16S4PerGroupTransDataFlow(QuantContext& ctx)
{
    if (IsA16W4Base<op::DataType::DT_INT4>(ctx) && IsXScaleUndefined(ctx) && IsScalePerGroup(ctx) &&
        IsTensor2DTransposed(ctx.weight)) {
        ctx.dataFlow = QuantDataFlow::MM_A16S4_PERGROUP;
        return true;
    }
    return false;
}

static bool IsMMA16S4PerGroupNonTransNzDataFlow(QuantContext& ctx)
{
    if (IsA16W4Base<op::DataType::DT_INT4>(ctx) && IsXScaleUndefined(ctx) && IsScalePerGroup(ctx) &&
        !IsTensor2DTransposed(ctx.weight) && IsOutWeightNzC016(ctx)) {
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
    if (IsMMA16MXF4Base(ctx) && IsTensor2DTransposed(ctx.weight)) {
        ctx.dataFlow = QuantDataFlow::MM_A16MXF4;
        return true;
    }
    return false;
}

// A16MXFP4 非转置 + NZ_C0_16 出 → ND→NZ 分形转换（NZ 出仅支持非转置，转置由转置条目的
// OUT_WEIGHT_ND_CHECKS 拒绝）
static bool IsMMA16MXF4NonTransNzDataFlow(QuantContext& ctx)
{
    if (IsMMA16MXF4Base(ctx) && !IsTensor2DTransposed(ctx.weight) && IsOutWeightNzC016(ctx)) {
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
    if (IsMMA8F4TcgBase(ctx) && IsTensor2DTransposed(ctx.weight)) {
        ctx.dataFlow = QuantDataFlow::MM_A8F4_TCG;
        return true;
    }
    return false;
}

// 非转置 → ND→NZ 分形转换；outWeight 的 C0_32 format 由 checks 的 CheckOutWeightNz 校验（judge 不探测 format）
static bool IsMMA8F4TcgNonTransDataFlow(QuantContext& ctx)
{
    if (IsMMA8F4TcgBase(ctx) && IsTensor2DNonTransposed(ctx.weight)) {
        ctx.dataFlow = QuantDataFlow::MM_A8F4_TCG;
        return true;
    }
    return false;
}

static bool IsA16DAV9201Base(const QuantContext& ctx)
{
    return (ctx.xDtype == op::DataType::DT_FLOAT16 || ctx.xDtype == op::DataType::DT_BF16) &&
           ctx.weight->GetViewShape().GetDimNum() == DIMS_2 && ctx.xScaleDtype == op::DataType::DT_UNDEFINED;
}

static bool IsA16DAV9201ScaleSame(const QuantContext& ctx)
{
    return IsA16DAV9201Base(ctx) && ctx.weightScale->GetDataType() == ctx.xDtype;
}

bool IsB4TypeNormal(const aclTensor* tensorLocal)
{
    auto tensorDtype = tensorLocal->GetDataType();
    return tensorDtype == op::DataType::DT_INT4 || tensorDtype == op::DataType::DT_FLOAT4_E2M1;
}

static bool IsA16DAV9201Spec(const QuantContext& ctx)
{
    auto weightViewShape = ctx.weight->GetViewShape();
    return IsA16DAV9201ScaleSame(ctx) && ctx.weightOffsetOptional == nullptr &&
           weightViewShape.GetDim(IDX_0) % NZ_C0_64 == 0 && weightViewShape.GetDim(IDX_1) % NZ_C0_64 == 0;
}

static bool IsMMMxA16W4DataFlow(QuantContext& ctx)
{
    if (IsA16DAV9201Base(ctx) && ctx.weight->GetDataType() == op::DataType::DT_FLOAT4_E2M1 &&
        ctx.weightScale->GetDataType() == op::DataType::DT_FLOAT8_E8M0) {
        ctx.dataFlow = QuantDataFlow::MM_A16MXF4;
        return true;
    }
    return false;
}

static bool IsMMMxA16HIF4DataFlow(QuantContext& ctx)
{
    if (IsA16DAV9201Base(ctx) && ctx.weight->GetDataType() == Ops::Math::DT_HIFLOAT4 &&
        ctx.weightScale->GetDataType() == Ops::Math::DT_HIFLOAT4_SCALE) {
        ctx.dataFlow = QuantDataFlow::MM_MX_A16HIF4;
        return true;
    }
    return false;
}

static bool IsMMC064A16W4DataFlow(QuantContext& ctx)
{
    if (IsA16DAV9201Spec(ctx) && IsB4TypeNormal(ctx.weight) && IsScalePerChannel(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_C064_A16W4;
        return true;
    }
    return false;
}

static bool IsMMA16S4PerTensorDataFlow9201(QuantContext& ctx)
{
    if (IsA16DAV9201ScaleSame(ctx) && ctx.weight->GetDataType() == op::DataType::DT_INT4 && IsScalePerTensor(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16S4_PERTENSOR;
        return true;
    }
    return false;
}

static bool IsMMA16S4PerChannelDataFlow(QuantContext& ctx)
{
    if (IsA16DAV9201ScaleSame(ctx) && ctx.weight->GetDataType() == op::DataType::DT_INT4 && IsScalePerChannel(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16S4_PERCHANNEL;
        return true;
    }
    return false;
}

static bool IsMMC064PerGroupA16W4DataFlow(QuantContext& ctx)
{
    if (IsA16DAV9201Spec(ctx) && IsB4TypeNormal(ctx.weight) && IsScalePerGroup(ctx) &&
        (ctx.kGroupSize == KGROUP_SIZE_64 || ctx.kGroupSize == KGROUP_SIZE_128) &&
        ctx.weight->GetViewShape().GetDim(IDX_0) % ctx.kGroupSize == 0) {
        ctx.dataFlow = QuantDataFlow::MM_C064_PerG_A16W4;
        return true;
    }
    return false;
}

static bool IsMMA16S4PerGroupDataFlow(QuantContext& ctx)
{
    if (IsA16DAV9201ScaleSame(ctx) && ctx.weight->GetDataType() == op::DataType::DT_INT4 && IsScalePerGroup(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16S4_PERGROUP;
        return true;
    }
    return false;
}

static bool IsMMA16Fp4PerGroupDataFlow(QuantContext& ctx)
{
    if (IsA16DAV9201ScaleSame(ctx) && ctx.weight->GetDataType() == op::DataType::DT_FLOAT4_E2M1 &&
        IsScalePerGroup(ctx)) {
        ctx.dataFlow = QuantDataFlow::MM_A16F4_PERGROUP;
        return true;
    }
    return false;
}

bool IsTensor3DTransposed(const aclTensor* tensor3D)
{
    auto viewShape = tensor3D->GetViewShape();
    auto viewStrides = tensor3D->GetViewStrides();
    int64_t lastDim = viewShape.GetDim(IDX_2);
    return viewStrides[IDX_0] == lastDim && viewStrides[IDX_1] == viewShape.GetDim(IDX_0) * lastDim &&
           viewStrides[IDX_2] == SINGLE;
}

bool IsTensor3DNonTransposed(const aclTensor* tensor3D)
{
    auto viewShape = tensor3D->GetViewShape();
    auto viewStrides = tensor3D->GetViewStrides();
    int64_t lastDim = viewShape.GetDim(IDX_2);
    return viewStrides[IDX_0] == lastDim * viewShape.GetDim(IDX_1) && viewStrides[IDX_1] == lastDim &&
           viewStrides[IDX_2] == SINGLE;
}

} // namespace WeightQuantPreprocess

using namespace WeightQuantPreprocess;

// ===== A16S4/A16F4/A8W4-MX 检查组合：按粒度拼装，条目间无内部分流 =====
// 紧凑 4-bit 输入公共检查（打包维须为偶数；打包视图须连续/严格转置且偏移字节对齐，
// ND 直拷与 NZ TransData 共用此前提），ND 入 + NZ 出的条目均可复用
// scale/offset/bias 为透传参数：out 侧 same 校验已保证输出与输入完全一致，
// 其自身的 dtype/format/shape 约束 preprocess 不消费也不处理，统一下放下游（wqbmmv2/qbmmv4）拦截
static const std::vector<CheckFunc> INPUT_4BIT_BASE_CHECKS = {CheckWeightNotEmpty, CheckWeightFormatND,
                                                              CheckWeightPackingDimEven, CheckWeight4BitPackedView};

// A16F4/A8W4-MX 条目无 offset 直拷 process：入参 offset 必须 nullptr，否则会被静默丢弃
static const std::vector<CheckFunc> OFFSET_NULL_CHECK = {CheckWeightOffsetOptionalNull};

// out weight ND 直拷检查
static const std::vector<CheckFunc> OUT_WEIGHT_ND_CHECKS = {CheckOutWeightSameAsInput};

// 非转置防御（judge 已保证非转置）：供 CombineChecks 条目在 OUT_WEIGHT_NZ_C016/C032_CHECKS 前插入
static const std::vector<CheckFunc> WEIGHT_NOT_TRANS_CHECK = {CheckWeightNotTrans};

// out weight NZ_C0_16 分形检查（非转置防御由 WEIGHT_NOT_TRANS_CHECK 承担）
const std::vector<CheckFunc> OUT_WEIGHT_NZ_C016_CHECKS = {
    CheckOutWeightSameBase, CheckOutWeightDtypeSame, CheckOutWeightNzStorageDim,
    CheckOutWeightNz<NZ_C0_16, op::Format::FORMAT_FRACTAL_NZ_C0_16, false>};

// out weight NZ_C0_32 分形检查（非转置 -> N 块在前）
static const std::vector<CheckFunc> OUT_WEIGHT_NZ_C032_CHECKS = {
    CheckOutWeightSameBase, CheckOutWeightDtypeSame, CheckOutWeightNzStorageDim,
    CheckOutWeightNz<NZ_C0_32, op::Format::FORMAT_FRACTAL_NZ_C0_32, false>};

// 输出 scale/offset/bias 与入参一致性检查（公共尾部）
static const std::vector<CheckFunc> OUT_TAIL_CHECKS = {
    CheckOutWeightScaleSameAsInput, CheckOutWeightOffsetOptionalSameAsInput, CheckOutBiasOptionalSameAsInput};

// 输出 scale/bias 与入参一致性检查（offset 无直拷 process、必须 nullptr 的条目用此尾部）
static const std::vector<CheckFunc> OUT_SCALE_BIAS_CHECKS = {CheckOutWeightScaleSameAsInput,
                                                             CheckOutBiasOptionalSameAsInput};

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
                  CheckWeightTrans, // 校验 weight 最后 2 维是否转置
                  CheckWeightScaleNotEmpty, CheckWeightScaleFormatND,
                  CheckWeightScaleMx<DIMS_3>, // 校验 weightScale 的 viewShape 是否符合 Mx 场景的 3 维形式 (k/64, n, 2)
                  CheckWeightScaleTrans<IDX_0>, // 校验 weightScale 的第 0 维和第 1 维是否转置
                  CheckWeightOffsetOptionalNull, // offset 无直拷 process，入参必须 nullptr，否则被静默丢弃
                  CheckKGroupSizeMx, CheckOutWeightSameBase, CheckOutWeightDtypeSame, CheckOutWeightNzStorageDim,
                  CheckOutWeightNz<NZ_C0_32, op::Format::FORMAT_FRACTAL_NZ_C0_32, true>,
                  CheckOutWeightScaleNotNullEmpty, CheckOutWeightScaleFormatNz, CheckOutWeightScaleDtypeSame,
                  CheckOutWeightScaleViewShapeSame, CheckOutWeightScaleMxNzStorage, CheckOutBiasOptionalSameAsInput},
       .processes = {ProcessWeightNd2Nz<IDX_0, true>, // weight ND→NZ 转换，模板参为 k 下标与转置状态
                     ProcessWeightScaleNd2Nz, ProcessBiasDirectCopy}},
      {.judge = IsGMMMxA8W4DataFlow,
       .checks = {CheckWeightNotEmpty, CheckWeightFormatND,
                  CheckWeightTrans,              // 校验 weight 最后 2 维是否转置
                  CheckWeightOffsetOptionalNull, // offset 无直拷 process，入参必须 nullptr，否则被静默丢弃
                  CheckOutWeightSameBase,        // out 与 input 的 nullptr/empty/viewShape 一致
                  CheckOutWeightDtypeSame,       // NZ 出 dtype 与输入一致
                  CheckOutWeightNzStorageDim,    // 输出 weight 为 NZ 格式，校验 storageShape 维度是否正确
                  // 校验输出 weight storageShape 符合 C0_32（转置 -> K 块在前）
                  CheckOutWeightNz<NZ_C0_32, op::Format::FORMAT_FRACTAL_NZ_C0_32, true>, CheckOutWeightScaleSameAsInput,
                  CheckOutBiasOptionalSameAsInput},
       .processes = {ProcessWeightNd2Nz<IDX_1, true>, // weight ND→NZ 转换，模板参为 k 下标与转置状态
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
       .processes = {ProcessWeightNd2Nz<IDX_0, false>, ProcessWeightScaleDirectCopy, ProcessWeightOffsetDirectCopy,
                     ProcessBiasDirectCopy}},
      // A16S4 per-group 转置：ND 直拷（scale 形状须匹配 per-group 分组语义）
      {.judge = IsMMA16S4PerGroupTransDataFlow,
       .checks = CombineChecks(INPUT_4BIT_BASE_CHECKS, OUT_WEIGHT_ND_CHECKS, OUT_TAIL_CHECKS),
       .processes = {ProcessWeightDirectCopy, ProcessWeightScaleDirectCopy, ProcessWeightOffsetDirectCopy,
                     ProcessBiasDirectCopy}},
      // A16S4 per-group 非转置 + NZ_C0_16 出：ND→NZ 转换（非转置仅支持 NZ 出）
      {.judge = IsMMA16S4PerGroupNonTransNzDataFlow,
       .checks = CombineChecks(INPUT_4BIT_BASE_CHECKS, WEIGHT_NOT_TRANS_CHECK, OUT_WEIGHT_NZ_C016_CHECKS,
                               OUT_TAIL_CHECKS),
       .processes = {ProcessWeightNd2Nz<IDX_0, false>, ProcessWeightScaleDirectCopy, ProcessWeightOffsetDirectCopy,
                     ProcessBiasDirectCopy}},
      // A16F4 per-group 非转置 + NZ_C0_16 出：ND→NZ 转换（NZ 出仅支持非转置）
      {.judge = IsMMA16F4PerGroupDataFlow,
       .checks = CombineChecks(INPUT_4BIT_BASE_CHECKS, WEIGHT_NOT_TRANS_CHECK, OFFSET_NULL_CHECK,
                               OUT_WEIGHT_NZ_C016_CHECKS, OUT_SCALE_BIAS_CHECKS),
       .processes = {ProcessWeightNd2Nz<IDX_0, false>, ProcessWeightScaleDirectCopy, ProcessBiasDirectCopy}},
      // A16MXFP4 转置：ND 直拷（wqbmmv2 MX kernel 支持 ND 转置输入；offset 不支持必须 nullptr）
      {.judge = IsMMA16MXF4TransDataFlow,
       .checks = CombineChecks(INPUT_4BIT_BASE_CHECKS, OFFSET_NULL_CHECK, OUT_WEIGHT_ND_CHECKS, OUT_TAIL_CHECKS),
       .processes = {ProcessWeightDirectCopy, ProcessWeightScaleDirectCopy, ProcessBiasDirectCopy}},
      // A16MXFP4 非转置 + NZ_C0_16 出：ND→NZ 分形转换（NZ 出仅支持非转置）
      {.judge = IsMMA16MXF4NonTransNzDataFlow,
       .checks = CombineChecks(INPUT_4BIT_BASE_CHECKS, WEIGHT_NOT_TRANS_CHECK, OFFSET_NULL_CHECK,
                               OUT_WEIGHT_NZ_C016_CHECKS, OUT_SCALE_BIAS_CHECKS),
       .processes = {ProcessWeightNd2Nz<IDX_0, false>, ProcessWeightScaleDirectCopy, ProcessBiasDirectCopy}},
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
       .processes = {ProcessWeightNd2Nz<IDX_0, false>, ProcessWeightScaleDirectCopy, ProcessWeightOffsetDirectCopy,
                     ProcessBiasDirectCopy}}}},
    {NpuArch::DAV_9201,
     {{.judge = IsMMMxA8W4DataFlow,
       .checks = {CheckWeightNotEmpty, CheckWeightFormatND,
                  CheckWeightTrans, // 校验 weight 最后 2 维是否转置
                  CheckWeightOffsetOptionalNull, CheckOutWeightSameBase, CheckOutWeightDtypeSame,
                  CheckOutWeightNzStorageDim, // 输出 weight 为 NZ 格式，校验 storageShape 维度是否正确
                  CheckOutWeightNz<NZ_C0_64, op::Format::FORMAT_FRACTAL_NZ,
                                   true>, // 校验输出 weight storageShape 符合 C0_64
                  CheckOutWeightScaleSameAsInput, CheckOutBiasOptionalSameAsInput},
       .processes = {ProcessWeightNd2Nz<IDX_0, true>, // 对 weight 进行 Nd2Nz 转换，参数表示 k 在 viewShape 中的下标
                     ProcessWeightScaleDirectCopy, ProcessBiasDirectCopy}},
      {.judge = IsMMMxA8HIF4DataFlow,
       .checks = {CheckWeightNotEmpty, CheckWeightFormatND, CheckWeightTrans, CheckWeightOffsetOptionalNull,
                  CheckOutWeightSameBase, CheckOutWeightDtypeSame, CheckOutWeightNzStorageDim,
                  CheckOutWeightNz<NZ_C0_64, op::Format::FORMAT_FRACTAL_NZ, true>, CheckOutWeightScaleSameAsInput,
                  CheckOutBiasOptionalSameAsInput},
       .processes = {ProcessWeightNd2Nz<IDX_0, true>, ProcessWeightScaleDirectCopy, ProcessBiasDirectCopy}},
      {.judge = IsMMMxA16HIF4DataFlow,
       .checks = {CheckWeight<DIMS_2>, CheckWeightTrans, CheckOutWeight<true>,
                  CheckOutWeightNz<NZ_C0_64, op::Format::FORMAT_FRACTAL_NZ, true>, CheckWeightOffsetOptionalNull,
                  CheckOutWeightScaleSameAsInput, CheckOutBiasOptionalSameAsInput},
       .processes = {ProcessWeightNd2Nz<IDX_0, true>, ProcessWeightScaleDirectCopy, ProcessBiasDirectCopy}},
      {.judge = IsMMC064A16W4DataFlow,
       .checks = {CheckWeight<DIMS_2>, CheckOutWeight<true>,
                  CheckOutWeightNz<NZ_C0_64, op::Format::FORMAT_FRACTAL_NZ, true>, CheckWeightScale<DIMS_1, DIMS_2>,
                  CheckOutWeightScale<true, op::DataType::DT_UINT64>, CheckWeightOffsetOptionalNull,
                  CheckOutBiasOptionalSameAsInput},
       .processes = {ProcessWeightNd2NzNoShapeLimit<true, false>, ProcessF16ToUint64, ProcessBiasDirectCopy}},
      {.judge = IsMMA16S4PerTensorDataFlow9201,
       .checks = {CheckWeight<DIMS_2>, CheckOutWeight<false>,
                  CheckOutWeightNz<NZ_C0_16, op::Format::FORMAT_FRACTAL_NZ, false>, CheckWeightScale<DIMS_1>,
                  CheckOutWeightScale<false>, CheckWeightOffset<DIMS_1>, CheckOutWeightOffset<false>,
                  CheckOutBiasOptionalSameAsInput},
       .processes = {ProcessWeightNd2NzNoShapeLimit<false, true>, ProcessWeightScaleToOutput<false>,
                     ProcessWeightOffsetToOutput<false>, ProcessBiasDirectCopy}},
      {.judge = IsMMA16S4PerChannelDataFlow,
       .checks = {CheckWeight<DIMS_2>, CheckOutWeight<false>,
                  CheckOutWeightNz<NZ_C0_16, op::Format::FORMAT_FRACTAL_NZ, false>, CheckWeightScale<DIMS_1, DIMS_2>,
                  CheckOutWeightScale<false>, CheckWeightOffset<DIMS_1, DIMS_2>, CheckOutWeightOffset<false>,
                  CheckOutBiasOptionalSameAsInput},
       .processes = {ProcessWeightNd2NzNoShapeLimit<false, true>, ProcessWeightScaleToOutput<false>,
                     ProcessWeightOffsetToOutput<false>, ProcessBiasDirectCopy}},
      {.judge = IsMMC064PerGroupA16W4DataFlow,
       .checks = {CheckWeight<DIMS_2>, CheckOutWeight<true>,
                  CheckOutWeightNz<NZ_C0_64, op::Format::FORMAT_FRACTAL_NZ, true>, CheckWeightScale<DIMS_2>,
                  CheckOutWeightScale<false>, CheckWeightOffsetOptionalNull, CheckOutBiasOptionalSameAsInput},
       .processes = {ProcessWeightNd2NzNoShapeLimit<true, false>, ProcessWeightScaleToOutput<false>,
                     ProcessBiasDirectCopy}},
      {.judge = IsMMA16S4PerGroupDataFlow,
       .checks = {CheckWeight<DIMS_2>, CheckOutWeight<false>,
                  CheckOutWeightNz<NZ_C0_16, op::Format::FORMAT_FRACTAL_NZ, false>, CheckWeightScale<DIMS_2>,
                  CheckOutWeightScale<false>, CheckWeightOffset<DIMS_2>, CheckOutWeightOffset<false>,
                  CheckOutBiasOptionalSameAsInput},
       .processes = {ProcessWeightNd2NzNoShapeLimit<false, true>, ProcessWeightScaleToOutput<false>,
                     ProcessWeightOffsetToOutput<false>, ProcessBiasDirectCopy}},
      {.judge = IsMMA16Fp4PerGroupDataFlow,
       .checks = {CheckWeight<DIMS_2>, CheckOutWeight<false>,
                  CheckOutWeightNz<NZ_C0_16, op::Format::FORMAT_FRACTAL_NZ, false>, CheckWeightScale<DIMS_2>,
                  CheckOutWeightScale<false>, CheckWeightOffsetOptionalNull, CheckOutBiasOptionalSameAsInput},
       .processes = {ProcessWeightNd2NzNoShapeLimit<false, true>, ProcessWeightScaleToOutput<false>,
                     ProcessBiasDirectCopy}},
      {.judge = IsMMMxA16W4DataFlow,
       .checks = {CheckWeight<DIMS_2>, CheckOutWeight<true>,
                  CheckOutWeightNz<NZ_C0_64, op::Format::FORMAT_FRACTAL_NZ, true>, CheckWeightScale<DIMS_2, DIMS_3>,
                  CheckOutWeightScale<true, op::DataType::DT_FLOAT8_E8M0>, CheckWeightOffsetOptionalNull,
                  CheckOutBiasOptionalSameAsInput},
       .processes = {ProcessWeightNd2NzNoShapeLimit<true, false>, ProcessMxScale, ProcessBiasDirectCopy}}}}};
