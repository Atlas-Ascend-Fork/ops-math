/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef WEIGHT_QUANT_PREPROCESS_PROCESS_H_
#define WEIGHT_QUANT_PREPROCESS_PROCESS_H_

#include "weight_quant_preprocess_registry.h"

#include <utility>

#include "common/inc/op_host/op_const_def_math.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/pad.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "math/bitwise_and/op_api/bitwiseand.h"
#include "math/bitwise_or/op_api/bitwise_or.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "log/log.h"

#define LOGE_WITH_SCENARIO(err, fmt, ...)                                              \
    OP_LOGE(err, "[NpuArch=%u, DataFlow=%s] " fmt, static_cast<uint32_t>(ctx.npuArch), \
            QuantDataFlowToString(ctx.dataFlow), ##__VA_ARGS__)

namespace WeightQuantPreprocess {

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

template <size_t viewKIdx, bool isTransposedOutput, bool nz_16 = false>
static aclnnStatus ProcessTensorNd2Nz(const aclTensor* weight, QuantContext& ctx)
{
    auto viewShape = ctx.weight->GetViewShape();
    auto dstFormat = nz_16 ? op::Format::FORMAT_FRACTAL_NZ_C0_16 : ctx.outWeight->GetStorageFormat();
    op::Shape storageShape(weight->GetViewShape());
    if (!IsTensor2DNonTransposed(weight) && isTransposedOutput) {
        std::swap(storageShape[viewKIdx], storageShape[viewKIdx + 1]);
    }
    auto weightTensor = ctx.executor->CreateView(weight, storageShape, weight->GetViewOffset());
    OP_CHECK(weightTensor != nullptr,
             LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "Create Weight storageShape before TransData failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    weightTensor->SetViewShape(storageShape);
    weightTensor->SetOriginalShape(storageShape);
    weightTensor->SetStorageShape(storageShape);

    auto outTensor = const_cast<aclTensor*>(l0op::TransData(weightTensor, dstFormat, 0, ctx.executor));
    OP_CHECK(outTensor != nullptr,
             LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR, "TransData Weight failed, storageShape=%s, dstFormat=%d.",
                                op::ToString(storageShape).GetString(), static_cast<int>(dstFormat)),
             return ACLNN_ERR_INNER_NULLPTR);

    outTensor->SetStorageFormat(ctx.outWeight->GetStorageFormat());
    outTensor->SetViewShape(viewShape);
    if constexpr (isTransposedOutput) {
        op::Strides viewStrides(outTensor->GetViewStrides());
        viewStrides[viewKIdx] = SINGLE;
        viewStrides[viewKIdx + IDX_1] = viewShape.GetDim(viewKIdx);
        outTensor->SetViewStrides(viewStrides);
    }

    auto result = l0op::ViewCopy(outTensor, ctx.outWeight, ctx.executor);
    OP_CHECK(result != nullptr,
             LOGE_WITH_SCENARIO(ACLNN_ERR_INNER_NULLPTR,
                                "ViewCopy NZ Weight failed, transposedOutput=%d, viewShape=%s, format=%d.",
                                static_cast<int>(isTransposedOutput), op::ToString(viewShape).GetString(),
                                static_cast<int>(dstFormat)),
             return ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

template <bool isTransposedOutput = false>
static const aclTensor* TranposeB4Tensor2D(const aclTensor* tensorLocal, QuantContext& ctx)
{
    op::Shape storageShape(tensorLocal->GetViewShape());
    if constexpr (!isTransposedOutput) {
        std::swap(storageShape[IDX_0], storageShape[IDX_1]);
    }
    auto tensorView = ctx.executor->CreateView(tensorLocal, storageShape, tensorLocal->GetViewOffset());
    auto fp4View = ctx.executor->CreateView(tensorView, tensorView->GetViewShape(), tensorView->GetViewOffset());
    OP_CHECK(fp4View != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Create FP4 view for B4 Tensor failed."),
             return nullptr);
    fp4View->SetViewStrides(tensorLocal->GetViewStrides());
    fp4View->SetDataType(op::DataType::DT_FLOAT4_E2M1);
    auto bf16Tensor = l0op::Cast(fp4View, op::DataType::DT_BF16, ctx.executor);
    OP_CHECK(bf16Tensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Cast B4 Tensor to BF16 failed."), return nullptr);
    int64_t permData[DIMS_2] = {IDX_1, IDX_0};
    auto perm = ctx.executor->AllocIntArray(permData, DIMS_2);
    OP_CHECK(perm != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Allocate Tensor transpose permutation failed."),
             return nullptr);
    auto transposedBf16 = l0op::Transpose(bf16Tensor, perm, ctx.executor);
    OP_CHECK(transposedBf16 != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Transpose BF16 Tensor failed."),
             return nullptr);
    auto transposedTensor = l0op::Cast(transposedBf16, op::DataType::DT_FLOAT4_E2M1, ctx.executor);
    OP_CHECK(transposedTensor != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Cast transposed BF16 Tensor to FP4 failed."), return nullptr);
    if (tensorLocal->GetDataType() != op::DataType::DT_INT4) {
        return transposedTensor;
    }
    auto int4View = ctx.executor->CreateView(transposedTensor, transposedTensor->GetViewShape(),
                                             transposedTensor->GetViewOffset());
    OP_CHECK(int4View != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Restore INT4 view after Tensor Transpose failed."),
             return nullptr);
    int4View->SetDataType(op::DataType::DT_INT4);
    return int4View;
}

template <bool isTransposedOutput = false>
static const aclTensor* TranposeTensor(const aclTensor* source, QuantContext& ctx)
{
    size_t dimNum = source->GetViewShape().GetDimNum();
    op::Shape storageShape(source->GetViewShape());
    if constexpr (!isTransposedOutput) {
        std::swap(storageShape[IDX_0], storageShape[IDX_1]);
    }
    auto sourceView = ctx.executor->CreateView(source, storageShape, source->GetViewOffset());
    OP_CHECK(sourceView != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Create Tensor2D/3D parameter storage view failed."), return nullptr);
    if (source->GetDataType() == op::DataType::DT_FLOAT8_E8M0) {
        sourceView->SetDataType(op::DataType::DT_INT8);
    }
    int64_t permData[DIMS_3] = {IDX_1, IDX_0};
    if (dimNum == DIMS_3) {
        permData[DIMS_2] = IDX_2;
    }
    auto perm = ctx.executor->AllocIntArray(permData, dimNum);
    OP_CHECK(perm != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Allocate Tensor2D/3D parameter transpose permutation failed."),
             return nullptr);
    auto target = l0op::Transpose(sourceView, perm, ctx.executor);
    OP_CHECK(target != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Transpose Tensor2D/3D failed."), return nullptr);
    return target;
}

template <bool isTransposedOutput = true, bool nz_16 = false>
static aclnnStatus ProcessWeightNd2NzNoShapeLimit(QuantContext& ctx)
{
    const aclTensor* preparedWeight = ctx.weight;
    if (isTransposedOutput == IsTensor2DNonTransposed(ctx.weight)) {
        if (IsB4TypeNormal(ctx.weight)) {
            preparedWeight = TranposeB4Tensor2D<isTransposedOutput>(ctx.weight, ctx);
        } else {
            preparedWeight = TranposeTensor<isTransposedOutput>(ctx.weight, ctx);
        }
        CHECK_RET(preparedWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ProcessTensorNd2Nz<IDX_0, isTransposedOutput, nz_16>(preparedWeight, ctx);
}

static aclnnStatus CopyTensorToTargetShape(const aclTensor* source, aclTensor* target, QuantContext& ctx)
{
    auto sourceView = ctx.executor->CreateView(source, target->GetViewShape(), source->GetViewOffset());
    OP_CHECK(sourceView != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Create source view with target layout failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    auto targetView = ctx.executor->CreateView(target, target->GetViewShape(), target->GetViewOffset());
    OP_CHECK(targetView != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Create target output byte view failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    if (target->GetDataType() == op::DataType::DT_FLOAT8_E8M0) {
        sourceView->SetDataType(op::DataType::DT_INT8);
        targetView->SetDataType(op::DataType::DT_INT8);
    }
    auto result = l0op::ViewCopy(sourceView, targetView, ctx.executor);
    OP_CHECK(result != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ViewCopy source with target layout failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

template <bool isTransposedOutput = false>
static aclnnStatus ProcessTensorToTarget(const aclTensor* tensor3D, aclTensor* target, QuantContext& ctx)
{
    const aclTensor* preparedTensor = tensor3D;
    auto dimNum = preparedTensor->GetViewShape().GetDimNum();
    auto tensorViewShape = tensor3D->GetViewShape();
    bool needProcess = false;
    if (dimNum == DIMS_2) {
        needProcess = IsTensor2DNonTransposed(tensor3D) == isTransposedOutput &&
                      tensorViewShape.GetDim(IDX_0) != SINGLE && tensorViewShape.GetDim(IDX_1) != SINGLE;
    } else if (dimNum == DIMS_3) {
        needProcess = IsTensor3DTransposed(tensor3D) != isTransposedOutput;
    }
    if (needProcess) {
        preparedTensor = TranposeTensor<isTransposedOutput>(tensor3D, ctx);
        CHECK_RET(preparedTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return CopyTensorToTargetShape(preparedTensor, target, ctx);
}

static const aclTensor* TransScaleTensor2DTo3D(const aclTensor* tensor2D, bool isTensorTransposed, QuantContext& ctx)
{
    int64_t groupNum = ctx.weightScale->GetViewShape().GetDim(IDX_0);
    op::Shape storageShape(tensor2D->GetViewShape());
    if (isTensorTransposed) {
        std::swap(storageShape[IDX_0], storageShape[IDX_1]);
    }
    auto sourceView = ctx.executor->CreateView(tensor2D, storageShape, tensor2D->GetViewOffset());
    OP_CHECK(sourceView != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Create Tensor2D parameter storage view failed."),
             return nullptr);
    const aclTensor* target = sourceView;

    // G 为奇数时仅在每个 N 对应的 G 尾部补一个原始零字节。
    if (groupNum % DOUBLE != 0) {
        int64_t paddingData[DIMS_2 * DOUBLE] = {
            0, 0, // N 不补
            0, 1  // G 尾部补 1
        };
        auto paddings = ctx.executor->AllocIntArray(paddingData, DIMS_2 * DOUBLE);
        OP_CHECK(paddings != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Allocate odd-G 3D padding failed."),
                 return nullptr);
        auto paddingTensor = ctx.executor->ConvertToTensor(paddings, op::DataType::DT_INT64);
        OP_CHECK(paddingTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Create odd-G 3D padding tensor failed."),
                 return nullptr);
        target = l0op::Pad(sourceView, paddingTensor, ctx.executor);
        OP_CHECK(target != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Pad odd-G 3D scale with zero failed."),
                 return nullptr);
    }
    return target;
}

static aclnnStatus ProcessMxScale(QuantContext& ctx)
{
    const aclTensor* preparedScale = ctx.weightScale;
    bool isTensorTransposed = IsTensor2DTransposed(preparedScale);
    auto dimNum = preparedScale->GetViewShape().GetDimNum();
    if (dimNum == DIMS_3) {
        return ProcessTensorToTarget<true>(ctx.weightScale, ctx.outWeightScale, ctx);
    }
    if (!isTensorTransposed) {
        preparedScale = TranposeTensor<true>(ctx.weightScale, ctx);
        CHECK_RET(preparedScale != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    preparedScale = TransScaleTensor2DTo3D(preparedScale, isTensorTransposed, ctx);
    CHECK_RET(preparedScale != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return CopyTensorToTargetShape(preparedScale, ctx.outWeightScale, ctx);
}

// C064 FP16/BF16 Scale 编码转为UINT64
static aclnnStatus ProcessF16ToUint64(QuantContext& ctx)
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
    // 0x0000400000000000：转uint64 第34位需要反转
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
    return CopyTensorToTargetShape(encodedScale, ctx.outWeightScale, ctx);
}

template <bool isTransposedOutput = false>
static aclnnStatus ProcessWeightScaleToOutput(QuantContext& ctx)
{
    return ProcessTensorToTarget<isTransposedOutput>(ctx.weightScale, ctx.outWeightScale, ctx);
}

template <bool isTransposedOutput = false>
static aclnnStatus ProcessWeightOffsetToOutput(QuantContext& ctx)
{
    if (ctx.weightOffsetOptional != nullptr) {
        return ProcessTensorToTarget<isTransposedOutput>(ctx.weightOffsetOptional, ctx.outWeightOffsetOptional, ctx);
    }
    return ACLNN_SUCCESS;
}

static std::vector<int64_t> MakeIdentityPermutation(size_t dimNum)
{
    std::vector<int64_t> permutation(dimNum);
    for (size_t i = 0; i < dimNum; ++i) {
        permutation[i] = static_cast<int64_t>(i);
    }
    return permutation;
}

static aclTensor* CreateContiguousByteView(const aclTensor* tensor, const op::Shape& shape, aclOpExecutor* executor)
{
    auto* view = executor->CreateView(tensor, shape, tensor->GetViewOffset());
    if (view == nullptr) {
        return nullptr;
    }
    view->SetDataType(op::DataType::DT_INT8);
    view->SetViewShape(shape);
    view->SetOriginalShape(shape);
    view->SetStorageShape(shape);
    op::Strides strides(view->GetViewStrides());
    int64_t stride = 1;
    for (int64_t i = static_cast<int64_t>(shape.GetDimNum()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape.GetDim(i);
    }
    view->SetViewStrides(strides);
    return view;
}

static aclnnStatus TransposeMxScaleStorageToLogical(const aclTensor* source, const op::Shape& storageShape,
                                                    QuantContext& ctx, const aclTensor** result)
{
    auto physicalByteView = CreateContiguousByteView(source, storageShape, ctx.executor);
    OP_CHECK(physicalByteView != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Create MX scale physical view failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    auto inputPerm = MakeIdentityPermutation(storageShape.GetDimNum());
    std::swap(inputPerm[IDX_0], inputPerm[IDX_1]);
    auto inputPermArray = ctx.executor->AllocIntArray(inputPerm.data(), inputPerm.size());
    OP_CHECK(inputPermArray != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Allocate ScaleNZ input permutation failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    *result = l0op::Transpose(physicalByteView, inputPermArray, ctx.executor);
    OP_CHECK(*result != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Transpose ScaleNZ input failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus PadMxScaleN(const aclTensor* source, size_t nAxis, int64_t n, int64_t alignedN, QuantContext& ctx,
                               const aclTensor** result)
{
    if (alignedN == n) {
        *result = source;
        return ACLNN_SUCCESS;
    }
    auto viewDim = source->GetViewShape().GetDimNum();
    std::vector<int64_t> paddings(viewDim * DOUBLE, 0);
    paddings[nAxis * DOUBLE + IDX_1] = alignedN - n;
    auto paddingsArray = ctx.executor->AllocIntArray(paddings.data(), paddings.size());
    OP_CHECK(paddingsArray != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Allocate ScaleNZ paddings failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    auto paddingsTensor = ctx.executor->ConvertToTensor(paddingsArray, op::DataType::DT_INT64);
    OP_CHECK(paddingsTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Create ScaleNZ padding tensor failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    *result = l0op::Pad(source, paddingsTensor, ctx.executor);
    OP_CHECK(*result != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Pad ScaleNZ input failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static std::vector<int64_t> GetMxScaleNzReshape(const op::Shape& viewShape, size_t scaleKAxis, int64_t nBlocks)
{
    std::vector<int64_t> reshapeShape;
    reshapeShape.reserve(viewShape.GetDimNum() + IDX_1);
    for (size_t i = 0; i < scaleKAxis; ++i) {
        reshapeShape.emplace_back(viewShape.GetDim(i));
    }
    reshapeShape.emplace_back(viewShape.GetDim(scaleKAxis));
    reshapeShape.emplace_back(nBlocks);
    reshapeShape.emplace_back(NZ_16);
    reshapeShape.emplace_back(NZ_C0_2);
    return reshapeShape;
}

static aclnnStatus ReshapeAndTransposeMxScale(const aclTensor* paddedScale, const std::vector<int64_t>& reshapeShape,
                                              size_t scaleKAxis, size_t nAxis, QuantContext& ctx, aclTensor** result)
{
    auto reshapeArray = ctx.executor->AllocIntArray(reshapeShape.data(), reshapeShape.size());
    OP_CHECK(reshapeArray != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Allocate ScaleNZ reshape failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    auto reshapedScale = l0op::Reshape(paddedScale, reshapeArray, ctx.executor);
    OP_CHECK(reshapedScale != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Reshape ScaleNZ input failed."),
             return ACLNN_ERR_INNER_NULLPTR);

    auto perm = MakeIdentityPermutation(reshapeShape.size());
    std::swap(perm[scaleKAxis], perm[nAxis]);
    auto permArray = ctx.executor->AllocIntArray(perm.data(), perm.size());
    OP_CHECK(permArray != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Allocate ScaleNZ transpose failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    *result = const_cast<aclTensor*>(l0op::Transpose(reshapedScale, permArray, ctx.executor));
    OP_CHECK(*result != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Transpose ScaleNZ output failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus CopyMxScaleToNzOutput(aclTensor* source, QuantContext& ctx)
{
    auto* dstScale = ctx.outWeightScale;
    source->SetViewFormat(dstScale->GetViewFormat());
    source->SetOriginalFormat(dstScale->GetOriginalFormat());
    source->SetStorageFormat(dstScale->GetStorageFormat());
    source->SetViewShape(dstScale->GetViewShape());
    source->SetOriginalShape(dstScale->GetOriginalShape());
    source->SetStorageShape(dstScale->GetStorageShape());

    auto* dstCopyTensor = ctx.executor->CreateView(dstScale, dstScale->GetViewShape(), dstScale->GetViewOffset());
    OP_CHECK(dstCopyTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Create ScaleNZ output view failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    dstCopyTensor->SetDataType(op::DataType::DT_INT8);
    dstCopyTensor->SetViewFormat(dstScale->GetViewFormat());
    dstCopyTensor->SetViewShape(dstScale->GetViewShape());
    dstCopyTensor->SetOriginalFormat(dstScale->GetOriginalFormat());
    dstCopyTensor->SetOriginalShape(dstScale->GetOriginalShape());
    dstCopyTensor->SetStorageFormat(dstScale->GetStorageFormat());
    dstCopyTensor->SetStorageShape(dstScale->GetStorageShape());

    auto result = l0op::ViewCopy(source, dstCopyTensor, ctx.executor);
    OP_CHECK(result != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Copy ScaleNZ output failed."),
             return ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static op::Shape GetMxScalePhysicalShape(const op::Shape& viewShape)
{
    op::Shape physicalShape = viewShape;
    const size_t scaleKAxis = viewShape.GetDimNum() - DIMS_3;
    const size_t nAxis = viewShape.GetDimNum() - DIMS_2;
    physicalShape.SetDim(scaleKAxis, viewShape.GetDim(nAxis));
    physicalShape.SetDim(nAxis, viewShape.GetDim(scaleKAxis));
    return physicalShape;
}

static aclnnStatus ProcessWeightScaleNd2Nz(QuantContext& ctx)
{
    auto viewShape = ctx.weightScale->GetViewShape();
    const size_t scaleKAxis = viewShape.GetDimNum() - DIMS_3;
    const size_t nAxis = viewShape.GetDimNum() - DIMS_2;
    const int64_t n = viewShape.GetDim(nAxis);
    const int64_t nBlocks = CeilDiv(n, NZ_16);
    const int64_t alignedN = nBlocks * NZ_16;

    const aclTensor* contiguousScale = nullptr;
    auto physicalShape = GetMxScalePhysicalShape(viewShape);
    auto ret = TransposeMxScaleStorageToLogical(ctx.weightScale, physicalShape, ctx, &contiguousScale);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const aclTensor* paddedScale = nullptr;
    ret = PadMxScaleN(contiguousScale, nAxis, n, alignedN, ctx, &paddedScale);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto reshapeShape = GetMxScaleNzReshape(viewShape, scaleKAxis, nBlocks);
    aclTensor* transposedScale = nullptr;
    ret = ReshapeAndTransposeMxScale(paddedScale, reshapeShape, scaleKAxis, nAxis, ctx, &transposedScale);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return CopyMxScaleToNzOutput(transposedScale, ctx);
}

} // namespace WeightQuantPreprocess

#undef LOGE_WITH_SCENARIO

#endif // WEIGHT_QUANT_PREPROCESS_PROCESS_H_
