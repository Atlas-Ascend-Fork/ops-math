/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef WEIGHT_QUANT_PREPROCESS_CHECK_H_
#define WEIGHT_QUANT_PREPROCESS_CHECK_H_

#include "weight_quant_preprocess_registry.h"

#include <string>
#include <utility>

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "log/log.h"

namespace WeightQuantPreprocess {

static aclnnStatus CheckTensorEmpty(const aclTensor* tensorLocal, std::string tensorName, const QuantContext& ctx)
{
    OP_CHECK(tensorLocal != nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnWeightQuantPreprocessGetWorkspaceSize", tensorName.c_str(),
                                                      "the tensor can not be null"),
             return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK(!tensorLocal->IsEmpty(),
             OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                 "aclnnWeightQuantPreprocessGetWorkspaceSize", tensorName.c_str(),
                 std::to_string(tensorLocal->GetViewShape().GetShapeSize()).c_str(),
                 (std::string("the tensor can not be empty when dataFlow is ") + QuantDataFlowToString(ctx.dataFlow))
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

template <size_t... allowedSizes>
static aclnnStatus CheckTensorViewStrides(const aclTensor* tensorLocal, std::string tensorName)
{
    auto viewShape = tensorLocal->GetViewShape();
    size_t dimNum = viewShape.GetDimNum();
    op::Shape allowedDims;
    (allowedDims.AppendDim(static_cast<int64_t>(allowedSizes)), ...);
    OP_CHECK(
        ((dimNum == allowedSizes) || ...),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            "aclnnWeightQuantPreprocessGetWorkspaceSize", tensorName.c_str(), (std::to_string(dimNum) + "D").c_str(),
            (std::string("the tensor rank must be one of ") + op::ToString(allowedDims).GetString() + " dimensions")
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);
    if (dimNum == DIMS_2) {
        OP_CHECK((IsTensor2DTransposed(tensorLocal) || IsTensor2DNonTransposed(tensorLocal)),
                 OP_LOGE_FOR_INVALID_STRIDE("aclnnWeightQuantPreprocessGetWorkspaceSize", tensorName.c_str(),
                                            op::ToString(tensorLocal->GetViewStrides()).GetString(),
                                            (std::string("[1, ") + std::to_string(viewShape.GetDim(IDX_0)) + "] or [" +
                                             std::to_string(viewShape.GetDim(IDX_1)) + ", 1]")
                                                .c_str()),
                 return ACLNN_ERR_PARAM_INVALID);
    } else if (dimNum == DIMS_3) {
        OP_CHECK((IsTensor3DTransposed(tensorLocal) || IsTensor3DNonTransposed(tensorLocal)),
                 OP_LOGE_FOR_INVALID_STRIDE("aclnnWeightQuantPreprocessGetWorkspaceSize", tensorName.c_str(),
                                            op::ToString(tensorLocal->GetViewStrides()).GetString(),
                                            "[S2, S0 * S2, 1] or [S1 * S2, S2, 1], where Si is viewShape[i]"),
                 return ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckTensorFormatND(const aclTensor* tensorLocal, std::string tensorName)
{
    auto tensorFormat = tensorLocal->GetStorageFormat();
    size_t dimNum = tensorLocal->GetViewShape().GetDimNum();
    if (dimNum == DIMS_3) {
        OP_CHECK(tensorFormat == op::Format::FORMAT_ND || tensorFormat == op::Format::FORMAT_NCL,
                 OP_LOGE_FOR_INVALID_FORMAT("aclnnWeightQuantPreprocessGetWorkspaceSize", tensorName.c_str(),
                                            op::ToString(tensorFormat).GetString(), "ND or NCL"),
                 return ACLNN_ERR_PARAM_INVALID);
    } else {
        OP_CHECK(tensorFormat == op::Format::FORMAT_ND,
                 OP_LOGE_FOR_INVALID_FORMAT("aclnnWeightQuantPreprocessGetWorkspaceSize", tensorName.c_str(),
                                            op::ToString(tensorFormat).GetString(), "ND"),
                 return ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckTensorNeedDtype(const aclTensor* tensorLocal, op::DataType needDataType, std::string tensorName)
{
    auto tensorLocalDtype = tensorLocal->GetDataType();
    OP_CHECK(
        tensorLocalDtype == needDataType,
        OP_LOGE_FOR_INVALID_DTYPE("aclnnWeightQuantPreprocessGetWorkspaceSize", tensorName.c_str(),
                                  op::ToString(tensorLocalDtype).GetString(), op::ToString(needDataType).GetString()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

template <bool isTransposedOutput = true>
static aclnnStatus CheckOutTensorViewStrides(const aclTensor* tensorLocal, std::string tensorName)
{
    auto viewShape = tensorLocal->GetViewShape();
    auto viewStrides = tensorLocal->GetViewStrides();
    size_t outTensorDim = viewShape.GetDimNum();
    op::Strides expectStrides(outTensorDim);
    if (outTensorDim == IDX_2) {
        expectStrides[IDX_0] = isTransposedOutput ? SINGLE : viewShape.GetDim(IDX_1);
        expectStrides[IDX_1] = isTransposedOutput ? viewShape.GetDim(IDX_0) : SINGLE;
    } else if (outTensorDim == IDX_3) {
        expectStrides[IDX_0] = isTransposedOutput ? viewShape.GetDim(IDX_2) :
                                                    viewShape.GetDim(IDX_2) * viewShape.GetDim(IDX_1);
        expectStrides[IDX_1] = isTransposedOutput ? viewShape.GetDim(IDX_0) * viewShape.GetDim(IDX_2) :
                                                    viewShape.GetDim(IDX_2);
        expectStrides[IDX_2] = SINGLE;
    } else if (outTensorDim == IDX_1) {
        expectStrides[IDX_0] = SINGLE;
    }
    OP_CHECK(viewStrides == expectStrides,
             OP_LOGE_FOR_INVALID_STRIDE("aclnnWeightQuantPreprocessGetWorkspaceSize", tensorName.c_str(),
                                        op::ToString(viewStrides).GetString(), op::ToString(expectStrides).GetString()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutTensorViewShapeSame(const aclTensor* inTensor, const aclTensor* outTensor,
                                               std::string inTensorName, std::string outTensorName,
                                               const QuantContext& ctx)
{
    auto inViewShape = inTensor->GetViewShape();
    auto outViewShape = outTensor->GetViewShape();
    OP_CHECK(inViewShape == outViewShape,
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                 "aclnnWeightQuantPreprocessGetWorkspaceSize", (inTensorName + ", " + outTensorName).c_str(),
                 (std::string(op::ToString(inViewShape).GetString()) + ", " + op::ToString(outViewShape).GetString())
                     .c_str(),
                 (std::string("the input and output must have the same view shape when dataFlow is ") +
                  QuantDataFlowToString(ctx.dataFlow))
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutTensorViewShapeDiffDim(const QuantContext& ctx)
{
    size_t inDim = ctx.weightScale->GetViewShape().GetDimNum();
    size_t outDim = ctx.outWeightScale->GetViewShape().GetDimNum();
    if (inDim == outDim) {
        return CheckOutTensorViewShapeSame(ctx.weightScale, ctx.outWeightScale, "weightScale", "outWeightScale", ctx);
    }
    OP_CHECK(inDim == DIMS_2 && outDim == DIMS_3 && ctx.dataFlow == QuantDataFlow::MM_A16MXF4,
             OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                 "aclnnWeightQuantPreprocessGetWorkspaceSize", "weightScale, outWeightScale",
                 (std::to_string(inDim) + "D, " + std::to_string(outDim) + "D").c_str(),
                 "different ranks are allowed only for MM_A16MXF4 with a 2D weightScale and a 3D outWeightScale"),
             return ACLNN_ERR_PARAM_INVALID);
    auto inViewShape = ctx.weightScale->GetViewShape();
    auto outViewShape = ctx.outWeightScale->GetViewShape();
    int64_t lastDim = outViewShape.GetDim(IDX_2);
    OP_CHECK(lastDim == DOUBLE && outViewShape.GetDim(IDX_0) == CeilDiv(inViewShape.GetDim(IDX_0), lastDim) &&
                 outViewShape.GetDim(IDX_1) == inViewShape.GetDim(IDX_1),
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                 "aclnnWeightQuantPreprocessGetWorkspaceSize", "weightScale, outWeightScale",
                 (std::string(op::ToString(inViewShape).GetString()) + ", " + op::ToString(outViewShape).GetString())
                     .c_str(),
                 "outWeightScale must have shape [ceildiv(G, 2), N, 2] when weightScale has shape [G, N]"),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

template <size_t... allowedSizes>
static aclnnStatus CheckInTensorBase(const aclTensor* inTensor, std::string inTensorName, const QuantContext& ctx)
{
    auto ret = CheckTensorEmpty(inTensor, inTensorName, ctx);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckTensorFormatND(inTensor, inTensorName);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return CheckTensorViewStrides<allowedSizes...>(inTensor, inTensorName);
}

template <size_t... allowedSizes>
static aclnnStatus CheckWeight(const QuantContext& ctx)
{
    return CheckInTensorBase<allowedSizes...>(ctx.weight, "weight", ctx);
}

template <size_t... allowedSizes>
static aclnnStatus CheckWeightScale(const QuantContext& ctx)
{
    return CheckInTensorBase<allowedSizes...>(ctx.weightScale, "weightScale", ctx);
}

template <size_t... allowedSizes>
static aclnnStatus CheckWeightOffset(const QuantContext& ctx)
{
    if (ctx.weightOffsetOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    return CheckInTensorBase<allowedSizes...>(ctx.weightOffsetOptional, "weightOffsetOptional", ctx);
}

template <bool isTransposedOutput = true>
static aclnnStatus CheckOutTensorBase(const aclTensor* outTensor, std::string outTensorName, op::DataType needDataType,
                                      const QuantContext& ctx)
{
    auto ret = CheckTensorEmpty(outTensor, outTensorName, ctx);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckTensorNeedDtype(outTensor, needDataType, outTensorName);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return CheckOutTensorViewStrides<isTransposedOutput>(outTensor, outTensorName);
}

template <bool isTransposedOutput = true>
static aclnnStatus CheckOutWeight(const QuantContext& ctx)
{
    auto ret = CheckOutTensorBase<isTransposedOutput>(ctx.outWeight, "outWeight", ctx.weight->GetDataType(), ctx);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return CheckOutTensorViewShapeSame(ctx.weight, ctx.outWeight, "weight", "outWeight", ctx);
}

template <bool isTransposedOutput = true, op::DataType needDataType = op::DataType::DT_UNDEFINED>
static aclnnStatus CheckOutWeightScale(const QuantContext& ctx)
{
    if (ctx.dataFlow == QuantDataFlow::MM_C064_A16W4) {
        auto viewShape = ctx.outWeightScale->GetViewShape();
        if (viewShape.GetDimNum() == DIMS_2) {
            auto viewStrides = ctx.outWeightScale->GetViewStrides();
            op::Shape realViewShape(viewShape);
            std::swap(realViewShape[IDX_0], realViewShape[IDX_1]);
            ctx.outWeightScale->SetViewShape(realViewShape);
            ctx.outWeightScale->SetViewStrides(viewStrides);
        }
    }
    auto ret = CheckOutTensorBase<isTransposedOutput>(
        ctx.outWeightScale, "outWeightScale",
        needDataType == op::DataType::DT_UNDEFINED ? ctx.weightScale->GetDataType() : needDataType, ctx);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckTensorFormatND(ctx.outWeightScale, "outWeightScale");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return CheckOutTensorViewShapeDiffDim(ctx);
}

template <bool isTransposedOutput = true>
static aclnnStatus CheckOutWeightOffset(const QuantContext& ctx)
{
    if (ctx.weightOffsetOptional == nullptr) {
        OP_CHECK(ctx.outWeightOffsetOptional == nullptr,
                 OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                     "aclnnWeightQuantPreprocessGetWorkspaceSize", "outWeightOffsetOptional",
                     "outWeightOffsetOptional must be null when weightOffsetOptional is null"),
                 return ACLNN_ERR_PARAM_INVALID);
        return ACLNN_SUCCESS;
    }
    auto ret = CheckOutTensorBase<isTransposedOutput>(ctx.outWeightOffsetOptional, "outWeightOffsetOptional",
                                                      ctx.weightOffsetOptional->GetDataType(), ctx);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckTensorFormatND(ctx.outWeightOffsetOptional, "outWeightOffsetOptional");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return CheckOutTensorViewShapeSame(ctx.weightOffsetOptional, ctx.outWeightOffsetOptional, "weightOffsetOptional",
                                       "outWeightOffsetOptional", ctx);
}

static aclnnStatus CheckWeightNotEmpty(const QuantContext& ctx)
{
    OP_CHECK(
        !ctx.weight->IsEmpty(),
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            "aclnnWeightQuantPreprocessGetWorkspaceSize", "weight",
            std::to_string(ctx.weight->GetViewShape().GetShapeSize()).c_str(),
            (std::string("weight can not be an empty tensor when dataFlow is ") + QuantDataFlowToString(ctx.dataFlow))
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckWeightFormatND(const QuantContext& ctx)
{
    auto weightFormat = ctx.weight->GetStorageFormat();
    OP_CHECK(weightFormat == op::Format::FORMAT_ND || weightFormat == op::Format::FORMAT_NCL,
             OP_LOGE_FOR_INVALID_FORMAT("aclnnWeightQuantPreprocessGetWorkspaceSize", "weight",
                                        op::ToString(weightFormat).GetString(), "ND or NCL"),
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
            "aclnnWeightQuantPreprocessGetWorkspaceSize", "weight",
            op::ToString(ctx.weight->GetViewStrides()).GetString(),
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
                 "aclnnWeightQuantPreprocessGetWorkspaceSize", "weight",
                 op::ToString(ctx.weight->GetViewStrides()).GetString(),
                 (std::string("last two strides [") + std::to_string(viewShape.GetDim(lastIdx)) + ", 1]").c_str()),
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
             OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                 "aclnnWeightQuantPreprocessGetWorkspaceSize", "weight",
                 std::string(op::ToString(viewShape).GetString())
                     .substr(1, std::string(op::ToString(viewShape).GetString()).size() - 2)
                     .c_str(),
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
                 "aclnnWeightQuantPreprocessGetWorkspaceSize", "weight", op::ToString(viewStrides).GetString(),
                 (std::string("contiguous or strictly transposed at last two dims when dataFlow is ") +
                  QuantDataFlowToString(ctx.dataFlow))
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(
        ctx.weight->GetViewOffset() % B4_NUMS_PER_BYTE == 0 && ctx.outWeight->GetViewOffset() % B4_NUMS_PER_BYTE == 0,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            "aclnnWeightQuantPreprocessGetWorkspaceSize", "weight.viewOffset, outWeight.viewOffset",
            (std::to_string(ctx.weight->GetViewOffset()) + ", " + std::to_string(ctx.outWeight->GetViewOffset()))
                .c_str(),
            "both view offsets must be even because each byte packs two 4-bit values"),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckWeightOffsetOptionalNull(const QuantContext& ctx)
{
    OP_CHECK(
        ctx.weightOffsetOptional == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            "aclnnWeightQuantPreprocessGetWorkspaceSize", "weightOffsetOptional",
            (std::string("weightOffsetOptional must be null when dataFlow is ") + QuantDataFlowToString(ctx.dataFlow))
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(ctx.outWeightOffsetOptional == nullptr,
             OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                 "aclnnWeightQuantPreprocessGetWorkspaceSize", "outWeightOffsetOptional",
                 (std::string("outWeightOffsetOptional must be null when dataFlow is ") +
                  QuantDataFlowToString(ctx.dataFlow))
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutWeightNzStorageDim(const QuantContext& ctx)
{
    auto weightViewShape = ctx.weight->GetViewShape();
    auto outStorageShape = ctx.outWeight->GetStorageShape();
    OP_CHECK(outStorageShape.GetDimNum() == weightViewShape.GetDimNum() + 2,
             OP_LOGE_FOR_INVALID_SHAPEDIM("aclnnWeightQuantPreprocessGetWorkspaceSize", "outWeight",
                                          (std::to_string(outStorageShape.GetDimNum()) + "D").c_str(),
                                          (std::to_string(weightViewShape.GetDimNum() + 2) + "D").c_str()),
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
             OP_LOGE_FOR_INVALID_FORMAT("aclnnWeightQuantPreprocessGetWorkspaceSize", "outWeight",
                                        op::ToString(outFormat).GetString(), op::ToString(outWeightFormat).GetString()),
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
            "aclnnWeightQuantPreprocessGetWorkspaceSize", "outWeight, weight",
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
    OP_CHECK(
        (input == nullptr) == (output == nullptr),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            "aclnnWeightQuantPreprocessGetWorkspaceSize", outputName,
            (std::string(outputName) + " and " + inputName +
             " must either both be null or both be non-null when dataFlow is " + QuantDataFlowToString(ctx.dataFlow))
                .c_str()),
        return ACLNN_ERR_PARAM_NULLPTR);
    if (input == nullptr) {
        return ACLNN_SUCCESS;
    }
    OP_CHECK(!output->IsEmpty(),
             OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                 "aclnnWeightQuantPreprocessGetWorkspaceSize", outputName,
                 std::to_string(output->GetViewShape().GetShapeSize()).c_str(),
                 (std::string(outputName) + " can not be an empty tensor when dataFlow is " +
                  QuantDataFlowToString(ctx.dataFlow))
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    auto inViewShape = input->GetViewShape();
    auto outViewShape = output->GetViewShape();
    OP_CHECK(outViewShape == inViewShape,
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                 "aclnnWeightQuantPreprocessGetWorkspaceSize", (std::string(outputName) + ", " + inputName).c_str(),
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
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            "aclnnWeightQuantPreprocessGetWorkspaceSize",
            (std::string(outputName) + ".isContiguous, " + inputName + ".isContiguous").c_str(),
            (std::string(IsContiguous(output) ? "true" : "false") + ", " + (IsContiguous(input) ? "true" : "false"))
                .c_str(),
            (std::string("the input and output must have the same contiguity when dataFlow is ") +
             QuantDataFlowToString(ctx.dataFlow))
                .c_str()),
        return ACLNN_ERR_PARAM_INVALID);
    auto inFormat = input->GetStorageFormat();
    auto outFormat = output->GetStorageFormat();
    OP_CHECK(outFormat == inFormat,
             OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
                 "aclnnWeightQuantPreprocessGetWorkspaceSize", (std::string(outputName) + ", " + inputName).c_str(),
                 (op::ToString(outFormat).GetString() + std::string(", ") + op::ToString(inFormat).GetString()).c_str(),
                 (std::string(outputName) + " and " + inputName + " must have the same format when dataFlow is " +
                  QuantDataFlowToString(ctx.dataFlow))
                     .c_str()),
             return ACLNN_ERR_PARAM_INVALID);
    auto inDtype = input->GetDataType();
    auto outDtype = output->GetDataType();
    OP_CHECK(outDtype == inDtype,
             OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                 "aclnnWeightQuantPreprocessGetWorkspaceSize", (std::string(outputName) + ", " + inputName).c_str(),
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
            "aclnnWeightQuantPreprocessGetWorkspaceSize", (std::string(outputName) + ", " + inputName).c_str(),
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
                 "aclnnWeightQuantPreprocessGetWorkspaceSize", "outWeight, weight",
                 (op::ToString(outWeightDtype).GetString() + std::string(", ") + op::ToString(weightDtype).GetString())
                     .c_str(),
                 (std::string("outWeight and weight must have the same dtype when dataFlow is ") +
                  QuantDataFlowToString(ctx.dataFlow))
                     .c_str()),
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

static aclnnStatus CheckKGroupSizeMx(const QuantContext& ctx)
{
    OP_CHECK(ctx.kGroupSize == KGROUP_SIZE_MX,
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnWeightQuantPreprocessGetWorkspaceSize", "kGroupSize",
                                                   std::to_string(ctx.kGroupSize).c_str(), "kGroupSize must be 32"),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static op::Shape GetMmMxScaleNzStorageShape(const op::Shape& viewShape)
{
    op::Shape storageShape;
    storageShape.AppendDim(CeilDiv(viewShape.GetDim(IDX_1), NZ_16));
    storageShape.AppendDim(viewShape.GetDim(IDX_0));
    storageShape.AppendDim(NZ_16);
    storageShape.AppendDim(NZ_C0_2);
    return storageShape;
}

static aclnnStatus CheckMmMxScaleNzStorage(const QuantContext& ctx, const aclTensor* scale, const char* scaleName)
{
    auto storageShape = scale->GetStorageShape();
    auto viewShape = scale->GetViewShape();
    auto expectedShape = GetMmMxScaleNzStorageShape(viewShape);
    const std::string expectedLayout = "storage shape must be [ceildiv(N, " + std::to_string(NZ_16) + "), ceildiv(K, " +
                                       std::to_string(KGROUP_SIZE_MX * DOUBLE) + "), " + std::to_string(NZ_16) + ", " +
                                       std::to_string(NZ_C0_2) + "] with leading dimensions preserved";
    OP_CHECK(
        storageShape == expectedShape,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            "aclnnWeightQuantPreprocessGetWorkspaceSize",
            (std::string(scaleName) + ".storageShape, " + scaleName + ".viewShape").c_str(),
            (std::string(op::ToString(storageShape).GetString()) + ", " + op::ToString(viewShape).GetString()).c_str(),
            (std::string(scaleName) + " " + expectedLayout + " for " + QuantDataFlowToString(ctx.dataFlow)).c_str()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutWeightScaleFormatNz(const QuantContext& ctx)
{
    auto outScaleFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(ctx.outWeightScale->GetStorageFormat()));
    OP_CHECK(outScaleFormat == op::Format::FORMAT_FRACTAL_NZ_C0_2,
             OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON("aclnnWeightQuantPreprocessGetWorkspaceSize", "outWeightScale",
                                                    op::ToString(outScaleFormat).GetString(),
                                                    "the format of outWeightScale must be FRACTAL_NZ_C0_2"),
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

static aclnnStatus CheckOutWeightScaleMxNzStorage(const QuantContext& ctx)
{
    return CheckMmMxScaleNzStorage(ctx, ctx.outWeightScale, "outWeightScale");
}

} // namespace WeightQuantPreprocess

#endif // WEIGHT_QUANT_PREPROCESS_CHECK_H_
