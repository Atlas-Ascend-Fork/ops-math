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
 * \file transpose_tiling_strategy_b4.cpp
 * \brief B4(10009) 策略类实现
 *
 * 实现 4bit（HIF4）尾轴转置的双切分计算、TilingData 回填与打印。
 */

#include "transpose_tiling_strategy_b4.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <vector>
#include "util/math_util.h"
#include "log/log.h"

namespace optiling {
using namespace Transpose;

// 4bit 类型判定常量（is4bit_ = eleLenInBytes == kDataTypeSizeBitOffset + 4）
static constexpr int64_t B4_STRATEGY_BIT_SIZE = 4;

/**
 * @brief B4(10009) 准入检查
 *
 * 准入条件：is4bit_ && shapeInfo_.isLastAxisTranspose（不受 related 门控）。
 */
bool TransposeB4Strategy::IsSupport() const
{
    const bool is4bit = (shapeInfo_.eleLenInBytes == ge::kDataTypeSizeBitOffset + B4_STRATEGY_BIT_SIZE);
    // dim>=1 为切分前准入（主类前置已保证，此处保留防御性校验）
    return is4bit && shapeInfo_.isLastAxisTranspose && shapeInfo_.dim >= 1;
}

// ===== B4 双切分私有实现 =====
// ---------------------------------------------------------------------------
// B4AxisCutter::FindCutAxis
//   Scan from innermost axis outward, accumulating product.
//   The first axis whose cumulative product exceeds ubBound is the cut axis.
// ---------------------------------------------------------------------------
void B4AxisCutter::FindCutAxis(int64_t ubBound)
{
    // Handle empty axis list (all axes already in UB)
    if (axisList_.empty()) {
        cutSet_ = 0;
        cutIdx_ = -1;
        cutFinal_ = false;
        innerCutAxis_.Reset(1, 1, 0);
        outerCutAxis_.Reset(1, 1, 0);
        return;
    }

    int32_t idx = static_cast<int32_t>(axisList_.size()) - 1;
    int64_t prod = 1;
    ubSet_ = 0;

    for (; idx >= 0; idx--) {
        prod *= axisList_[idx].dim;
        if (prod > ubBound) {
            break;
        }
        ubSet_ = ubSet_ | axisList_[idx].code;
    }

    cutSet_ = (idx >= 0) ? axisList_[idx].code : 0;
    cutIdx_ = idx;

    // If all axes fit in UB (idx == -1), no cutting needed
    if (idx < 0) {
        cutFinal_ = false;
        innerCutAxis_.Reset(1, 1, 0);
        outerCutAxis_.Reset(1, 1, 0);
        return;
    }

    B4AxisInf* cutAxisPtr = &axisList_[cutIdx_];
    // cutFinal_ = true only if the cut axis is THIS VIEW's final axis.
    // finalAxisIdx_ is set to axes.back().idx in the constructor,
    // which is the original input index of the view's last axis.
    cutFinal_ = (cutAxisPtr->idx == finalAxisIdx_);

    int64_t cutDim = cutAxisPtr->dim;
    // The constructor aligns the final axis up to alignedFactor_.
    // All calculations involving the "original" dimension must use
    // originFinalDim_ instead of the aligned cutDim.
    int64_t effectiveCutDim = cutFinal_ ? originFinalDim_ : cutDim;
    prod = prod / cutDim;

    // Avoid division by zero
    if (prod == 0) {
        prod = 1;
    }

    int64_t innerDim = ubBound / prod;
    // For HIF4: ensure even and at least 2 only for last axis
    if (cutFinal_ && (innerDim % B4_NUM_TWO != 0)) {
        innerDim = (innerDim / B4_NUM_TWO) * B4_NUM_TWO;
    }
    if (innerDim == 0) {
        innerDim = cutFinal_ ? B4_NUM_TWO : 1;
    }
    // innerDim cannot exceed the original (non-aligned) dimension
    if (innerDim > effectiveCutDim) {
        innerDim = effectiveCutDim;
    }

    int64_t outerDim = (effectiveCutDim + innerDim - 1) / innerDim;

    innerCutAxis_.Reset(innerDim, cutAxisPtr->stride, cutAxisPtr->idx);
    outerCutAxis_.Reset(outerDim, innerDim * cutAxisPtr->stride, cutAxisPtr->idx);
}

// ---------------------------------------------------------------------------
// B4AxisCutter::RemoveAxis
//   Remove axes whose bitmask intersects setCode from axisList_.
// ---------------------------------------------------------------------------
void B4AxisCutter::RemoveAxis(uint32_t setCode)
{
    std::vector<B4AxisInf> tmp;
    for (auto& a : axisList_) {
        if ((a.code & setCode) == 0) {
            tmp.push_back(a);
        }
    }
    axisList_ = tmp;
}

// ---------------------------------------------------------------------------
// B4DualCutAxisSeeker::Init
//   inShape:  input tensor shape (original order, in HIF4 elements)
//   outShape: output tensor shape (transposed order, in HIF4 elements)
//   Both views use the same axis indices, no sorting needed.
//   alignedFactor = 64 HIF4 elements (32 bytes alignment)
// ---------------------------------------------------------------------------
void B4DualCutAxisSeeker::Init(const std::vector<int64_t>& inShape, const std::vector<int64_t>& outShape,
                               const std::vector<int64_t>& perm, int32_t dimNums, gert::TilingContext* context,
                               int32_t ubBlockSize)
{
    context_ = context;
    dimNums_ = dimNums;
    ubBlockSize_ = ubBlockSize;
    outShape_ = outShape;
    perm_ = perm;
    // For HIF4: alignedFactor = 64 elements (32 bytes)
    alignedFactor_ = B4_HIF4_ALIGN_ELEMENTS;

    // Build input strides (row-major, element units)
    std::vector<int64_t> inStrides(dimNums, 1);
    for (int32_t i = dimNums - B4_NUM_TWO; i >= 0; i--) {
        inStrides[i] = inStrides[i + 1] * inShape[i + 1];
    }

    // Build output strides (row-major, element units)
    std::vector<int64_t> outStrides(dimNums, 1);
    for (int32_t i = dimNums - B4_NUM_TWO; i >= 0; i--) {
        outStrides[i] = outStrides[i + 1] * outShape[i + 1];
    }

    // Initialize inputAxis_ with input shape and input strides (no sorting)
    inputAxis_.clear();
    for (int32_t i = 0; i < dimNums_; i++) {
        inputAxis_.emplace_back(inShape[i], inStrides[i], i);
    }

    // Initialize outputAxis_ in output order (outputAxis_[i] = output axis i)
    // idx = perm[i] indicates which input axis this output axis corresponds to
    // stride = input stride of the corresponding input axis (for as_strided compatibility)
    outputAxis_.clear();
    for (int32_t i = 0; i < dimNums_; i++) {
        int32_t inputAxisIdx = static_cast<int32_t>(perm[i]);
        int64_t inputStride = inStrides[inputAxisIdx];
        outputAxis_.emplace_back(outShape[i], inputStride, inputAxisIdx);
    }

    inputCutter_.Init(inputAxis_, alignedFactor_);
    outputCutter_.Init(outputAxis_, alignedFactor_);

    OP_LOGD(context_, "[HIF4DualCut] B4DualCutAxisSeeker: dimNums=%d, alignedFactor=%d", dimNums_, alignedFactor_);
}

// ---------------------------------------------------------------------------
// B4DualCutAxisSeeker::ComputeSetAxisProd
// ---------------------------------------------------------------------------
int64_t B4DualCutAxisSeeker::ComputeSetAxisProd(uint32_t axisSet) const
{
    int64_t factor = 1;
    for (const auto& a : outputAxis_) {
        if ((a.code & axisSet) != 0) {
            factor *= a.dim;
            if (factor <= 0) {
                OP_LOGW(context_, "[HIF4DualCut] ComputeSetAxisProd overflow, axisSet=0x%x", axisSet);
                return INT64_MAX;
            }
        }
    }
    return factor;
}

// ---------------------------------------------------------------------------
// B4DualCutAxisSeeker::ComputeRemainUB
//   Compute remaining UB space after placing axes in axisSet.
//   IMPORTANT: Must consider alignment for BOTH input and output final axes.
//   For transpose, input final axis and output final axis are different axes.
// ---------------------------------------------------------------------------
int64_t B4DualCutAxisSeeker::ComputeRemainUB(int64_t ubNum, uint32_t axisSet) const
{
    // Identify input and output final axes
    uint32_t inFinalCode = inputAxis_[dimNums_ - 1].code;   // input axis (dimNums-1)
    uint32_t outFinalCode = outputAxis_[dimNums_ - 1].code; // output axis (dimNums-1) -> input axis perm[dimNums-1]

    // Compute product of all axes in axisSet
    int64_t prod = 1;
    for (const auto& a : outputAxis_) {
        if ((a.code & axisSet) != 0) {
            // Check if this axis needs alignment
            bool needAlign = (a.code == inFinalCode) || (a.code == outFinalCode);
            if (needAlign) {
                int64_t alignedDim = ((a.dim + alignedFactor_ - 1) / alignedFactor_) * alignedFactor_;
                prod *= alignedDim;
            } else {
                prod *= a.dim;
            }
        }
    }

    if (prod <= 0) {
        OP_LOGW(context_, "[HIF4DualCut] ComputeRemainUB: prod overflow detected, axisSet=0x%x, prod=%ld", axisSet,
                prod);
        return 0;
    }
    return ubNum / prod;
}

// ---------------------------------------------------------------------------
// B4DualCutAxisSeeker::FindCommonUbAxis / UpdateCommonAxis
// ---------------------------------------------------------------------------
uint32_t B4DualCutAxisSeeker::FindCommonUbAxis(const B4AxisCutter* c1, const B4AxisCutter* c2) const
{
    uint32_t commonUb = c1->ubSet_ & c2->ubSet_;
    uint32_t cutCommon1 = c1->cutSet_ & c2->ubSet_;
    uint32_t cutCommon2 = c2->cutSet_ & c1->ubSet_;
    return commonUb | cutCommon1 | cutCommon2;
}

void B4DualCutAxisSeeker::UpdateCommonAxis(uint32_t commonSet)
{
    outputCutter_.RemoveAxis(commonSet);
    inputCutter_.RemoveAxis(commonSet);
}

// ---------------------------------------------------------------------------
// B4DualCutAxisSeeker::FindDualCutAxis
//   Main loop: iteratively find common axes between output/input cutters,
//   absorb them into ubAxisSet, then call CutAxis to finalize gmAxis_/ubAxis_.
//   For HIF4: ubSize is in bytes, shapes are in HIF4 elements (2 elements per byte)
// ---------------------------------------------------------------------------
bool B4DualCutAxisSeeker::FindDualCutAxis(int32_t ubSize, int32_t bufferNum)
{
    ubSize_ = ubSize;

    if (bufferNum == 0) {
        OP_LOGW(context_, "[HIF4DualCut] bufferNum == 0, reset to 2");
        bufferNum = B4_NUM_TWO;
    }

    // ubNum: available UB in HIF4 elements
    // ubSize is in bytes, 2 HIF4 elements per byte
    int64_t ubNum = static_cast<int64_t>(ubSize) * B4_HIF4_ELEMENTS_PER_BYTE / bufferNum;
    OP_LOGD(context_,
            "[HIF4DualCut] FindDualCutAxis start: ubSize=%d bytes, bufferNum=%d, ubNum=%ld HIF4 elements, "
            "alignedFactor=%d, lastDim=%ld",
            ubSize, bufferNum, ubNum, alignedFactor_, outputAxis_[dimNums_ - 1].dim);

    // Note: Allow dual-cut even if last dim < alignedFactor
    // The alignment will be handled during axis cutting

    uint32_t ubAxisSet = 0;
    uint32_t joinUbAxisSet = 0;

    for (int32_t findLoops = TP_B4_GM_LEN + TP_B4_UB_LEN; findLoops >= 0; findLoops--) {
        int64_t remainUB = ComputeRemainUB(ubNum, ubAxisSet);
        if (remainUB <= 0) {
            break;
        }
        int64_t ubBound = static_cast<int64_t>(std::floor(std::sqrt(static_cast<double>(remainUB))));
        if (ubBound <= 0) {
            break;
        }

        outputCutter_.FindCutAxis(ubBound);
        inputCutter_.FindCutAxis(ubBound);

        uint32_t commonUbAxisSet = FindCommonUbAxis(&outputCutter_, &inputCutter_);
        joinUbAxisSet = outputCutter_.ubSet_ | inputCutter_.ubSet_;
        if (commonUbAxisSet == 0) {
            // No shared UB axis: two independent cut axes, or both cutters cut the same axis.
            bool sameCutAxis = (inputCutter_.cutSet_ & outputCutter_.cutSet_) != 0;
            OP_LOGD(context_, "[HIF4DualCut] Search stop: %s.", sameCutAxis ? "one cut axis" : "two cut axes");
            break;
        } else {
            ubAxisSet = ubAxisSet | commonUbAxisSet;
            UpdateCommonAxis(commonUbAxisSet);
        }
    }

    ubAxisSet = ubAxisSet | joinUbAxisSet;
    int64_t remainNums = ComputeRemainUB(ubNum, ubAxisSet);
    OP_LOGD(context_, "[HIF4DualCut] FindDualCutAxis done: remainNums=%ld", remainNums);
    return CutAxis(ubAxisSet, remainNums);
}

// ---------------------------------------------------------------------------
// B4DualCutAxisSeeker::CutAxis
//   Orchestrate the cut, then build gmAxis_ / ubAxis_.
// ---------------------------------------------------------------------------
bool B4DualCutAxisSeeker::CutAxis(uint32_t ubAxisSet, int64_t remainNums)
{
    std::vector<B4AxisInf> innerAxis;
    std::vector<B4AxisInf> outerAxis;

    uint32_t cutAxisSet = inputCutter_.cutSet_ | outputCutter_.cutSet_;
    DispatchCut(ubAxisSet, remainNums, cutAxisSet, innerAxis, outerAxis);

    // Filter out "fake" cuts where outerDim == 1 (treat as full axis in UB)
    FilterFakeCuts(ubAxisSet, innerAxis, outerAxis);

    // Partition outputAxis_ into gmAxis_ / ubAxis_
    FillGmUbAxis(ubAxisSet, cutAxisSet);

    // If ub axes + inner cut axes exceed MOVE_ALIGN limit, move axes to gm.
    EvictExcessUbAxis(innerAxis);

    AppendCutAxes(innerAxis, outerAxis);
    SortGmUbAxis();

    // Validate actual UB footprint with final-axis alignment.
    ShrinkInnerCutDims(innerAxis, outerAxis);

    // Handle "all in UB" case: if gmAxis_ is empty, add a dummy axis with dim=1
    if (gmAxis_.empty()) {
        gmAxis_.emplace_back(1, 1, 0);
    }

    return true;
}

// Dispatch to CutTwoAxis/CutOneAxis, falling back to a 0-axis cut on failure.
void B4DualCutAxisSeeker::DispatchCut(uint32_t& ubAxisSet, int64_t remainNums, uint32_t cutAxisSet,
                                      std::vector<B4AxisInf>& innerAxis, std::vector<B4AxisInf>& outerAxis)
{
    bool ok = true;
    if (cutAxisSet == 0) {
        OP_LOGD(context_, "[HIF4DualCut] CutAxis: all axes fit in UB.");
        // No inner/outer axis to add
    } else if (outputCutter_.cutSet_ != inputCutter_.cutSet_) {
        OP_LOGD(context_, "[HIF4DualCut] CutAxis: CutTwoAxis.");
        ok = CutTwoAxis(ubAxisSet, remainNums, innerAxis, outerAxis);
        if (!ok) {
            // Fallback: treat both cut axes as UB axes (0-axis cut)
            OP_LOGW(context_, "[HIF4DualCut] CutTwoAxis failed, treat cut axes as UB axes.");
            ubAxisSet |= cutAxisSet;
            innerAxis.clear();
            outerAxis.clear();
        }
    } else {
        OP_LOGD(context_, "[HIF4DualCut] CutAxis: CutOneAxis.");
        ok = CutOneAxis(ubAxisSet, remainNums, innerAxis, outerAxis);
        if (!ok) {
            // Fallback: treat cut axis as UB axis (0-axis cut)
            OP_LOGW(context_, "[HIF4DualCut] CutOneAxis failed, treat cut axis as UB axis.");
            ubAxisSet |= cutAxisSet;
            innerAxis.clear();
            outerAxis.clear();
        }
    }
}

// Drop "fake" cuts where outerDim == 1 (the axis is essentially uncut).
void B4DualCutAxisSeeker::FilterFakeCuts(uint32_t& ubAxisSet, std::vector<B4AxisInf>& innerAxis,
                                         std::vector<B4AxisInf>& outerAxis)
{
    std::vector<B4AxisInf> filteredInner;
    std::vector<B4AxisInf> filteredOuter;
    for (size_t i = 0; i < innerAxis.size(); i++) {
        if (outerAxis[i].dim == 1) {
            // Fake cut: outer=1 means the axis is essentially uncut
            // Add the original full axis to ubAxisSet instead
            ubAxisSet |= outerAxis[i].code;
        } else {
            filteredInner.push_back(innerAxis[i]);
            filteredOuter.push_back(outerAxis[i]);
        }
    }
    innerAxis = filteredInner;
    outerAxis = filteredOuter;

    cutAxisNums_ = static_cast<int32_t>(innerAxis.size());
}

// Partition outputAxis_ into UB axes (ubAxisSet) and GM axes (not cut).
void B4DualCutAxisSeeker::FillGmUbAxis(uint32_t ubAxisSet, uint32_t cutAxisSet)
{
    for (const auto& a : outputAxis_) {
        if ((a.code & ubAxisSet) != 0) {
            ubAxis_.push_back(a);
        } else if ((a.code & cutAxisSet) == 0) {
            gmAxis_.push_back(a);
        }
    }
}

// If ub axes + inner cut axes exceed the MOVE_ALIGN limit, move axes to gm.
// Priority: evict non-final axes first; always keep the output final axis in UB.
void B4DualCutAxisSeeker::EvictExcessUbAxis(const std::vector<B4AxisInf>& innerAxis)
{
    if (ubAxis_.size() + innerAxis.size() <= static_cast<size_t>(TP_B4_UB_LEN)) {
        return;
    }

    uint32_t inFinalCode = inputAxis_[dimNums_ - 1].code;
    uint32_t outFinalCode = outputAxis_[dimNums_ - 1].code;

    int32_t excess = static_cast<int32_t>(ubAxis_.size() + innerAxis.size()) - TP_B4_UB_LEN;

    // Separate non-final and final axes
    std::vector<B4AxisInf> nonFinal;
    std::vector<B4AxisInf> finals;
    for (auto& a : ubAxis_) {
        if ((a.code == inFinalCode) || (a.code == outFinalCode)) {
            finals.push_back(a);
        } else {
            nonFinal.push_back(a);
        }
    }

    // Evict non-final axes first (largest stride first to keep small strides in UB)
    std::sort(nonFinal.begin(), nonFinal.end(), [](const B4AxisInf& a, const B4AxisInf& b) {
        return (a.stride == b.stride) ? (a.dim > b.dim) : (a.stride < b.stride);
    });
    int32_t evicted = 0;
    while (evicted < excess && !nonFinal.empty()) {
        gmAxis_.push_back(nonFinal.back());
        nonFinal.pop_back();
        evicted++;
    }

    // If still excess, evict final axes (prefer input final over output final)
    while (evicted < excess && !finals.empty()) {
        gmAxis_.push_back(finals.back());
        finals.pop_back();
        evicted++;
    }

    // Rebuild ubAxis_
    ubAxis_.clear();
    ubAxis_.insert(ubAxis_.end(), nonFinal.begin(), nonFinal.end());
    ubAxis_.insert(ubAxis_.end(), finals.begin(), finals.end());
}

void B4DualCutAxisSeeker::AppendCutAxes(const std::vector<B4AxisInf>& innerAxis,
                                        const std::vector<B4AxisInf>& outerAxis)
{
    for (const auto& a : innerAxis) {
        ubAxis_.push_back(a);
    }
    for (const auto& a : outerAxis) {
        gmAxis_.push_back(a);
    }
}

void B4DualCutAxisSeeker::SortGmUbAxis()
{
    // Sort both gmAxis_ and ubAxis_ by input axis index (inShape order)
    std::sort(gmAxis_.begin(), gmAxis_.end(), [](const B4AxisInf& a, const B4AxisInf& b) { return a.idx < b.idx; });
    std::sort(ubAxis_.begin(), ubAxis_.end(), [](const B4AxisInf& a, const B4AxisInf& b) { return a.idx < b.idx; });
}

// Validate the actual UB footprint with final-axis alignment; shrink inner
// cut dims if it overflows.
//   ND-DMA requires both input and output innermost axes to be aligned to
//   alignedFactor_.  Fake-cut axes that end up in UB must also use their
//   aligned dims when checking capacity.
void B4DualCutAxisSeeker::ShrinkInnerCutDims(std::vector<B4AxisInf>& innerAxis, std::vector<B4AxisInf>& outerAxis)
{
    int64_t ubNum = static_cast<int64_t>(ubSize_) * B4_HIF4_ELEMENTS_PER_BYTE / B4_HIF4_BUFFER_NUM;
    int64_t alignedProd = ComputeAlignedUbProd();
    if (alignedProd <= ubNum) {
        return;
    }

    OP_LOGW(context_,
            "[HIF4DualCut] UB overflow after fake cuts: alignedProd=%ld > ubNum=%ld, "
            "shrinking inner cut dims",
            alignedProd, ubNum);
    int64_t ratio = (alignedProd + ubNum - 1) / ubNum; // ceil
    if (ratio < B4_NUM_TWO) {
        ratio = B4_NUM_TWO;
    }

    for (size_t i = 0; i < innerAxis.size(); i++) {
        ShrinkSingleCutAxis(i, ratio, innerAxis, outerAxis);
    }
}

// Aligned product of UB dims (final axes rounded up to alignedFactor_).
int64_t B4DualCutAxisSeeker::ComputeAlignedUbProd() const
{
    int64_t alignedProd = 1;
    for (const auto& a : ubAxis_) {
        bool isInFinal = (a.code == inputAxis_[dimNums_ - 1].code);
        bool isOutFinal = (a.code == outputAxis_[dimNums_ - 1].code);
        int64_t dim = a.dim;
        if (isInFinal || isOutFinal) {
            dim = ((dim + alignedFactor_ - 1) / alignedFactor_) * alignedFactor_;
        }
        alignedProd *= dim;
    }
    return alignedProd;
}

// Shrink one inner cut axis by ratio and sync its dim into ubAxis_ / gmAxis_.
void B4DualCutAxisSeeker::ShrinkSingleCutAxis(size_t i, int64_t ratio, std::vector<B4AxisInf>& innerAxis,
                                              std::vector<B4AxisInf>& outerAxis)
{
    int64_t newDim = innerAxis[i].dim / ratio;
    if (newDim < 1) {
        newDim = 1;
    }

    bool isInFinal = (innerAxis[i].code == inputAxis_[dimNums_ - 1].code);
    bool isOutFinal = (innerAxis[i].code == outputAxis_[dimNums_ - 1].code);
    if (isInFinal || isOutFinal) {
        if (newDim % B4_NUM_TWO != 0) {
            newDim = (newDim / B4_NUM_TWO) * B4_NUM_TWO;
        }
        if (newDim < B4_NUM_TWO) {
            newDim = B4_NUM_TWO;
        }
    }
    innerAxis[i].dim = newDim;

    // Recompute outer dim from original dim
    int64_t origDim = 0;
    if (isInFinal) {
        origDim = inputCutter_.originFinalDim_;
    } else if (isOutFinal) {
        origDim = outputCutter_.originFinalDim_;
    } else {
        for (const auto& oa : outputAxis_) {
            if (oa.code == innerAxis[i].code) {
                origDim = oa.dim;
                break;
            }
        }
    }
    if (origDim == 0) {
        origDim = innerAxis[i].dim;
    }
    outerAxis[i].dim = (origDim + innerAxis[i].dim - 1) / innerAxis[i].dim;
    outerAxis[i].stride = innerAxis[i].dim * innerAxis[i].stride;

    // Sync the dim in ubAxis_ (find by matching code)
    for (auto& ua : ubAxis_) {
        if (ua.code == innerAxis[i].code) {
            ua.dim = innerAxis[i].dim;
            break;
        }
    }
    // Sync the outer dim in gmAxis_ (find by matching code)
    for (auto& ga : gmAxis_) {
        if (ga.code == innerAxis[i].code) {
            ga.dim = outerAxis[i].dim;
            ga.stride = outerAxis[i].stride;
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// B4DualCutAxisSeeker::CutTwoAxis
// ---------------------------------------------------------------------------
bool B4DualCutAxisSeeker::CutTwoAxis(uint32_t ubAxisSet, int64_t remainNums, std::vector<B4AxisInf>& innerAxis,
                                     std::vector<B4AxisInf>& outerAxis)
{
    (void)ubAxisSet;
    B4AxisInf outInner = outputCutter_.innerCutAxis_;
    B4AxisInf outOuter = outputCutter_.outerCutAxis_;
    B4AxisInf inInner = inputCutter_.innerCutAxis_;
    B4AxisInf inOuter = inputCutter_.outerCutAxis_;

    int32_t inCutAxisIdx = (inputCutter_.cutIdx_ >= 0) ? inputCutter_.axisList_[inputCutter_.cutIdx_].idx : -1;
    int32_t outCutAxisIdx = (outputCutter_.cutIdx_ >= 0) ? outputCutter_.axisList_[outputCutter_.cutIdx_].idx : -1;

    // Pre-compute original dimensions (use originFinalDim_ for final axis because
    // B4AxisCutter aligns the final axis up to alignedFactor_)
    int64_t inOrigDim = GetCutterOrigDim(&inputCutter_);
    int64_t outOrigDim = GetCutterOrigDim(&outputCutter_);
    if (inOrigDim == 0) {
        inOrigDim = inInner.dim;
    }
    if (outOrigDim == 0) {
        outOrigDim = outInner.dim;
    }

    // Decide the inner dims (alignment + even/min-2 rules depending on final axes)
    ComputeCutInnerDims(inInner, outInner, inOrigDim, outOrigDim, inCutAxisIdx, outCutAxisIdx, remainNums);

    // Clamp inner dims to original dims (no cut if inner >= original)
    if (inInner.dim > inOrigDim) {
        inInner.dim = inOrigDim;
    }
    if (outInner.dim > outOrigDim) {
        outInner.dim = outOrigDim;
    }

    inOuter.dim = (inOrigDim + inInner.dim - 1) / inInner.dim;
    outOuter.dim = (outOrigDim + outInner.dim - 1) / outInner.dim;

    inOuter.stride = inInner.dim * inInner.stride;
    outOuter.stride = outInner.dim * outInner.stride;

    outerAxis.push_back(outOuter);
    outerAxis.push_back(inOuter);
    innerAxis.push_back(outInner);
    innerAxis.push_back(inInner);
    return true;
}

// True when axisIdx is the last axis or the axis transposed to last position.
bool B4DualCutAxisSeeker::NeedEvenDim(int32_t axisIdx) const
{
    int32_t lastAxis = dimNums_ - 1;
    int32_t transposedLastAxis = static_cast<int32_t>(perm_[lastAxis]);
    return (axisIdx == lastAxis) || (axisIdx == transposedLastAxis);
}

// Ensure even and >= 2 only for the last axis / transposed-last axis.
void B4DualCutAxisSeeker::FixEvenMin2IfNeeded(int64_t& v, int32_t axisIdx) const
{
    if (NeedEvenDim(axisIdx)) {
        if (v % B4_NUM_TWO != 0) {
            v = (v / B4_NUM_TWO) * B4_NUM_TWO;
        }
        if (v <= 0) {
            v = B4_NUM_TWO;
        }
    } else {
        if (v <= 0) {
            v = 1;
        }
    }
}

// Original (non-aligned) dim of the cutter's cut axis.
int64_t B4DualCutAxisSeeker::GetCutterOrigDim(const B4AxisCutter* cutter) const
{
    if (cutter->cutIdx_ < 0) {
        return 0;
    }
    if (cutter->cutFinal_) {
        return cutter->originFinalDim_;
    }
    return cutter->axisList_[cutter->cutIdx_].dim;
}

// Align the inner dim for a final axis, guarding against fake cut / overflow.
void B4DualCutAxisSeeker::AlignInnerDimIfNeeded(int64_t& inner, int64_t orig, int32_t axisIdx, bool isFinal,
                                                int64_t remainNums) const
{
    if (!isFinal) {
        FixEvenMin2IfNeeded(inner, axisIdx);
        return;
    }
    if (inner >= alignedFactor_) {
        inner = (inner / alignedFactor_) * alignedFactor_;
    } else if (orig >= alignedFactor_ && remainNums >= static_cast<int64_t>(alignedFactor_) * B4_NUM_TWO) {
        inner = alignedFactor_;
    }
    FixEvenMin2IfNeeded(inner, axisIdx);
    // If alignment causes fake cut (inner >= original), revert to original
    if (inner >= orig) {
        inner = orig;
        FixEvenMin2IfNeeded(inner, axisIdx);
    }
}

// Compute inInner/outInner dims for CutTwoAxis, honoring which cut axes are final.
void B4DualCutAxisSeeker::ComputeCutInnerDims(B4AxisInf& inInner, B4AxisInf& outInner, int64_t inOrigDim,
                                              int64_t outOrigDim, int32_t inCutAxisIdx, int32_t outCutAxisIdx,
                                              int64_t remainNums)
{
    if (inputCutter_.cutFinal_ || outputCutter_.cutFinal_) {
        if (inputCutter_.cutFinal_) {
            AlignInnerDimIfNeeded(inInner.dim, inOrigDim, inCutAxisIdx, true, remainNums);
            outInner.dim = remainNums / inInner.dim;
            if (outInner.dim <= 0) {
                outInner.dim = B4_NUM_TWO;
                inInner.dim = remainNums / outInner.dim;
                AlignInnerDimIfNeeded(inInner.dim, inOrigDim, inCutAxisIdx, true, remainNums);
            }
            FixEvenMin2IfNeeded(outInner.dim, outCutAxisIdx);
            if (outputCutter_.cutFinal_) {
                int64_t newInInner = remainNums / outInner.dim;
                if (newInInner > 0) {
                    inInner.dim = newInInner;
                    AlignInnerDimIfNeeded(inInner.dim, inOrigDim, inCutAxisIdx, true, remainNums);
                }
            }
        } else {
            // outputCutter_.cutFinal_
            AlignInnerDimIfNeeded(outInner.dim, outOrigDim, outCutAxisIdx, true, remainNums);
            inInner.dim = remainNums / outInner.dim;
            if (inInner.dim <= 0) {
                inInner.dim = B4_NUM_TWO;
                outInner.dim = remainNums / inInner.dim;
                AlignInnerDimIfNeeded(outInner.dim, outOrigDim, outCutAxisIdx, true, remainNums);
            }
            FixEvenMin2IfNeeded(inInner.dim, inCutAxisIdx);
            if (inputCutter_.cutFinal_) {
                int64_t newOutInner = remainNums / inInner.dim;
                if (newOutInner > 0) {
                    outInner.dim = newOutInner;
                    AlignInnerDimIfNeeded(outInner.dim, outOrigDim, outCutAxisIdx, true, remainNums);
                }
            }
        }
    } else {
        FixEvenMin2IfNeeded(inInner.dim, inCutAxisIdx);
        FixEvenMin2IfNeeded(outInner.dim, outCutAxisIdx);
    }
}

// ---------------------------------------------------------------------------
// B4DualCutAxisSeeker::CutOneAxis
// ---------------------------------------------------------------------------
bool B4DualCutAxisSeeker::CutOneAxis(uint32_t ubAxisSet, int64_t remainNums, std::vector<B4AxisInf>& innerAxis,
                                     std::vector<B4AxisInf>& outerAxis)
{
    (void)ubAxisSet;
    int64_t innerDim = inputCutter_.cutFinal_ ? (remainNums / alignedFactor_ * alignedFactor_) : remainNums;

    B4AxisInf* cutPtr = (inputCutter_.cutIdx_ >= 0) ? &inputCutter_.axisList_[inputCutter_.cutIdx_] :
                                                      &inputCutter_.axisList_[0];

    int32_t axisIdx = cutPtr->idx;
    int32_t lastAxis = dimNums_ - 1;
    int32_t transposedLastAxis = static_cast<int32_t>(perm_[lastAxis]);
    bool needEven = (axisIdx == lastAxis) || (axisIdx == transposedLastAxis);

    // For HIF4: ensure even number only for last axis or transposed-last axis
    if (needEven && (innerDim % B4_NUM_TWO != 0)) {
        innerDim = (innerDim / B4_NUM_TWO) * B4_NUM_TWO;
    }
    // Ensure minimum value
    if (innerDim <= 0) {
        innerDim = needEven ? B4_NUM_TWO : 1;
    }

    // Get original dim (originFinalDim_ for final axis because constructor aligns it)
    int64_t origDim = (inputCutter_.cutFinal_) ? inputCutter_.originFinalDim_ : cutPtr->dim;

    // If innerDim >= original dim, treat as no-cut (outerDim will be 1, filtered later)
    if (innerDim >= origDim) {
        innerDim = origDim;
    }

    int64_t innerStride = cutPtr->stride;
    int64_t outerDim = (origDim + innerDim - 1) / innerDim;
    int64_t outerStride = innerStride * innerDim;

    innerAxis.emplace_back(innerDim, innerStride, axisIdx);
    outerAxis.emplace_back(outerDim, outerStride, axisIdx);
    return true;
}

// ---------------------------------------------------------------------------
// B4DualCutAxisSeeker::SetCutAxisIdx
//   After gmAxis_/ubAxis_ are finalized, find which ubAxis slot corresponds
//   to each cut axis and compute the tail size.
// ---------------------------------------------------------------------------
void B4DualCutAxisSeeker::SetCutAxisIdx()
{
    int32_t cutAxisIdx[2] = {-1, -1};
    int32_t cutAxisTail[2] = {-1, -1};

    // gmAxis_ is sorted by input axis index, so cut axes are interleaved with non-cut axes.
    // Scan gmAxis_ in order and find axes whose code also appears in ubAxis_ (cut axes).
    int32_t found = 0;
    for (int32_t k = 0; k < outerAxisNums_ && found < cutAxisNums_; k++) {
        for (int32_t j = 0; j < static_cast<int32_t>(ubAxis_.size()); j++) {
            if (ubAxis_[j].code == gmAxis_[k].code) {
                // Use merged axis id (ubAxis_[j].idx) instead of array position
                cutAxisIdx[found] = ubAxis_[j].idx;
                // tail = original dim - (outerDim - 1) * innerDim
                cutAxisTail[found] = static_cast<int32_t>(inputAxis_[gmAxis_[k].idx].dim -
                                                          (gmAxis_[k].dim - 1) * ubAxis_[j].dim);
                found++;
                break;
            }
        }
    }
    cutAxisIdx01_ = cutAxisIdx[0];
    cutAxisIdx02_ = cutAxisIdx[1];
    cutAxisTail01_ = cutAxisTail[0];
    cutAxisTail02_ = cutAxisTail[1];
}

// ---------------------------------------------------------------------------
// B4DualCutAxisSeeker::GenTilingData
//   Populate gmShape_/gmInStride_/gmOutStride_ and ubShape_/ubInStride_/ubOutStride_.
//   IMPORTANT: All UB arrays must be in INPUT AXIS ORDER for ND-DMA to work correctly:
//   - ubShape: UB axes in input axis order
//   - ubInStride: input stride for each input axis
//   - ubOutStride: output stride for each input axis (where it appears in output)
// ---------------------------------------------------------------------------
void B4DualCutAxisSeeker::GenTilingData()
{
    outerAxisNums_ = static_cast<int32_t>(gmAxis_.size());
    innerAxisNums_ = static_cast<int32_t>(ubAxis_.size());

    FillGmArrays();
    FillUbShapeAndId();
    FillUbStrides();
    FillUbPermAndMoveStride();
    FillCutIds();
    FixGmOutStrideForCuts();

    SetCutAxisIdx();
}

// Output position (index in output order) of the given input axis, or -1.
int32_t B4DualCutAxisSeeker::FindOutputPos(int32_t axisIdx) const
{
    for (int32_t j = 0; j < dimNums_; j++) {
        if (static_cast<int32_t>(perm_[j]) == axisIdx) {
            return j;
        }
    }
    return -1;
}

// Fill gmShape_ / gmInStride_ / gmOutStride_ (right-aligned).
void B4DualCutAxisSeeker::FillGmArrays()
{
    int32_t gmBaseIdx = TP_B4_GM_LEN - outerAxisNums_;
    for (int32_t i = 0; i < outerAxisNums_; i++) {
        int32_t slot = gmBaseIdx + i;
        gmShape_[slot] = static_cast<int32_t>(gmAxis_[i].dim);
        gmInStride_[slot] = gmAxis_[i].stride; // input stride from outputAxis
        // For dummy axis (dim=1), outStride should be 1 (no skipping)
        if (gmAxis_[i].dim == 1) {
            gmOutStride_[slot] = 1;
        } else {
            gmOutStride_[slot] = ComputeOutputStride(gmAxis_[i]); // compute output stride
        }
    }
}

// Fill ubShape_ (input axis order) and ubShapeId_ (-1 for unused slots).
void B4DualCutAxisSeeker::FillUbShapeAndId()
{
    int32_t baseIdx = TP_B4_UB_LEN - innerAxisNums_;
    for (int32_t i = 0; i < innerAxisNums_; i++) {
        ubShape_[baseIdx + i] = static_cast<int32_t>(ubAxis_[i].dim);
    }
    for (int32_t i = 0; i < TP_B4_UB_LEN; i++) {
        ubShapeId_[i] = -1;
    }
    for (int32_t i = 0; i < innerAxisNums_; i++) {
        ubShapeId_[baseIdx + i] = ubAxis_[i].idx;
    }
}

// Fill ubInStride_ / ubOutStride_ aligned with ubShape_.
void B4DualCutAxisSeeker::FillUbStrides()
{
    int32_t baseIdx = TP_B4_UB_LEN - innerAxisNums_;
    for (int32_t i = 0; i < innerAxisNums_; i++) {
        int32_t slot = baseIdx + i;
        int32_t axisIdx = ubAxis_[i].idx;

        // Input stride for this input axis
        ubInStride_[slot] = inputAxis_[axisIdx].stride;

        // Output stride: product of output dims to the right of this axis's output position
        int32_t outPos = FindOutputPos(axisIdx);
        if (outPos < 0) {
            OP_LOGW(context_, "[HIF4DualCut] GenTilingData: axis idx=%d not found in perm!", axisIdx);
        }
        int64_t outStride = 1;
        for (int32_t j = outPos + 1; j < dimNums_; j++) {
            outStride *= outShape_[j];
        }
        ubOutStride_[slot] = outStride;
    }
}

// Fill ubPerm_ / moveUbStride_: ub axes reordered by output position.
void B4DualCutAxisSeeker::FillUbPermAndMoveStride()
{
    int32_t baseIdx = TP_B4_UB_LEN - innerAxisNums_;
    std::vector<std::pair<int32_t, int32_t>> ubOutOrder; // (outPos, ubAxisIndex)
    ubOutOrder.reserve(innerAxisNums_);
    for (int32_t i = 0; i < innerAxisNums_; i++) {
        int32_t axisIdx = ubAxis_[i].idx;
        int32_t outPos = FindOutputPos(axisIdx);
        if (outPos < 0) {
            OP_LOGW(context_, "[HIF4DualCut] GenTilingData: ubAxis idx=%d not found in perm!", axisIdx);
            outPos = 0;
        }
        ubOutOrder.emplace_back(outPos, i);
    }
    std::sort(ubOutOrder.begin(), ubOutOrder.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    // Fill dummy positions (before baseIdx) as identity permutation
    for (int32_t i = 0; i < baseIdx; i++) {
        ubPerm_[i] = i;
    }

    for (int32_t i = 0; i < innerAxisNums_; i++) {
        int32_t slot = baseIdx + i;
        int32_t outPos = ubOutOrder[i].first;
        int32_t ubIdx = ubOutOrder[i].second;

        // ubPerm: output-order axis -> source slot index in ubShape_
        ubPerm_[slot] = baseIdx + ubIdx;

        // moveUbStride: product of output dims to the right of outPos
        int64_t stride = 1;
        for (int32_t j = outPos + 1; j < dimNums_; j++) {
            stride *= outShape_[j];
        }
        moveUbStride_[slot] = stride;
    }
}

// Fill outCutId_ / inCutId_ (position within gmAxis_).
void B4DualCutAxisSeeker::FillCutIds()
{
    // Collect all real cut axes from gmAxis_ (those whose code also appears in ubAxis_).
    std::vector<int32_t> cutAxisPositions;
    for (int32_t k = 0; k < outerAxisNums_; k++) {
        for (const auto& ua : ubAxis_) {
            if (ua.code == gmAxis_[k].code) {
                cutAxisPositions.push_back(k);
                break;
            }
        }
    }
    outCutId_ = -1;
    inCutId_ = -1;
    if (cutAxisNums_ == 1) {
        // Single real cut axis: both outCut and inCut point to it
        if (!cutAxisPositions.empty()) {
            outCutId_ = cutAxisPositions[0];
            inCutId_ = cutAxisPositions[0];
        }
    } else if (cutAxisNums_ == B4_NUM_TWO) {
        // Two real cut axes: match by cutter cutSet_ bitmask
        for (int32_t pos : cutAxisPositions) {
            if ((gmAxis_[pos].code & outputCutter_.cutSet_) != 0) {
                outCutId_ = pos;
            }
            if ((gmAxis_[pos].code & inputCutter_.cutSet_) != 0) {
                inCutId_ = pos;
            }
        }
    }
}

// Multiply gmOutStride_ of cut outer axes by the corresponding inner dim.
void B4DualCutAxisSeeker::FixGmOutStrideForCuts()
{
    int32_t gmBaseIdx = TP_B4_GM_LEN - outerAxisNums_;
    for (int32_t k = 0; k < outerAxisNums_; k++) {
        uint32_t axisCode = gmAxis_[k].code;
        for (const auto& ua : ubAxis_) {
            if (ua.code == axisCode) {
                int32_t slot = gmBaseIdx + k;
                gmOutStride_[slot] *= static_cast<int64_t>(ua.dim);
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// B4DualCutAxisSeeker::ComputeOutputStride
//   Compute output stride for a given axis (similar to as_strided's ComputeOutputShape).
//   For transpose: find the output position of this input axis, then compute stride
//   as the product of all output dims to the right of that position.
// ---------------------------------------------------------------------------
int64_t B4DualCutAxisSeeker::ComputeOutputStride(const B4AxisInf& axis) const
{
    // Find output position of this input axis
    int32_t outPos = FindOutputPos(axis.idx);
    if (outPos < 0) {
        OP_LOGW(context_, "[HIF4DualCut] ComputeOutputStride: axis idx=%d not found in perm!", axis.idx);
        return 1;
    }

    // Compute stride: product of all output dims to the right
    int64_t prod = 1;
    for (int32_t i = outPos + 1; i < dimNums_; i++) {
        prod *= outShape_[i];
    }
    return prod;
}

// ---------------------------------------------------------------------------
// B4DualCutAxisSeeker::ComputeBlockTiling
//   NK10: distribute total outer iterations across coreNum cores.
// ---------------------------------------------------------------------------
void B4DualCutAxisSeeker::ComputeBlockTiling(int32_t coreNum)
{
    int64_t totalOuter = 1;
    for (const auto& a : gmAxis_) {
        totalOuter *= a.dim;
    }
    dimsPerCore_ = static_cast<int32_t>(Ops::Base::CeilDiv(totalOuter, static_cast<int64_t>(coreNum)));
    useCoreNum_ = static_cast<int32_t>((totalOuter + dimsPerCore_ - 1) / dimsPerCore_);
    dimsPerCoreTail_ = static_cast<int32_t>(totalOuter - static_cast<int64_t>(useCoreNum_ - 1) * dimsPerCore_);
}

// ---------------------------------------------------------------------------
// Print helpers shared by PrintTilingData
// ---------------------------------------------------------------------------
namespace {

// Append "\n  <name><v0>,<v1>,..." for a fixed-length integer array.
template <typename T>
void AppendArray(std::ostringstream& oss, const char* name, const T* data, int32_t len)
{
    oss << "\n  " << name;
    for (int32_t i = 0; i < len; i++) {
        oss << data[i] << (i < len - 1 ? "," : "");
    }
}

// Append "\n  <name>  (idx,dim,stride) ..." for an axis list.
void AppendAxisList(std::ostringstream& oss, const char* name, const std::vector<B4AxisInf>& axes)
{
    oss << "\n  " << name << "  (idx,dim,stride): ";
    for (const auto& a : axes) {
        oss << "(" << a.idx << "," << a.dim << "," << a.stride << ") ";
    }
}

} // namespace

// ---------------------------------------------------------------------------
// B4DualCutAxisSeeker::PrintTilingData  (debug helper)
// ---------------------------------------------------------------------------
void B4DualCutAxisSeeker::PrintTilingData() const
{
    int32_t ubSizePerBuf = (ubSize_ / B4_HIF4_BUFFER_NUM) / ubBlockSize_ * ubBlockSize_;
    if (ubSizePerBuf <= 0) {
        ubSizePerBuf = ubBlockSize_;
    }
    std::ostringstream oss;

    oss << "\n[HIF4DualCut] ===== TilingData =====";

    // --- original shape info ---
    AppendAxisList(oss, "[inShape]", inputAxis_);
    AppendArray(oss, "[outShape]   ", outShape_.data(), dimNums_);
    AppendArray(oss, "[perm]       ", perm_.data(), dimNums_);

    // --- multi-core split ---
    oss << "\n  [MultiCore] useCoreNum=" << useCoreNum_ << "  dimsPerCore=" << dimsPerCore_
        << "  dimsPerCoreTail=" << dimsPerCoreTail_;

    // --- UB budget ---
    oss << "\n  [UB]        ubSize=" << ubSizePerBuf << " bytes";

    // --- axis counts ---
    oss << "\n  [AxisNums]  outerAxisNums=" << outerAxisNums_ << "  innerAxisNums=" << innerAxisNums_
        << "  cutAxisNums=" << cutAxisNums_;

    // --- cut axis tail info ---
    oss << "\n  [CutAxis]   cutAxisIdx01=" << cutAxisIdx01_ << "  cutAxisTail01=" << cutAxisTail01_
        << "  cutAxisIdx02=" << cutAxisIdx02_ << "  cutAxisTail02=" << cutAxisTail02_;

    // --- cut axis id in gmShape ---
    oss << "\n  [CutId]     outCutId=" << outCutId_ << "  inCutId=" << inCutId_;

    // --- GM arrays ---
    AppendArray(oss, "[gmShape]    ", gmShape_, TP_B4_GM_LEN);
    AppendArray(oss, "[gmInStride] ", gmInStride_, TP_B4_GM_LEN);
    AppendArray(oss, "[gmOutStride]", gmOutStride_, TP_B4_GM_LEN);

    // --- UB arrays ---
    AppendArray(oss, "[ubShape]    ", ubShape_, TP_B4_UB_LEN);
    AppendArray(oss, "[ubInStride] ", ubInStride_, TP_B4_UB_LEN);
    AppendArray(oss, "[ubOutStride]", ubOutStride_, TP_B4_UB_LEN);
    AppendArray(oss, "[moveUbStride]", moveUbStride_, TP_B4_UB_LEN);
    AppendArray(oss, "[ubPerm]     ", ubPerm_, TP_B4_UB_LEN);
    AppendArray(oss, "[ubShapeId]  ", ubShapeId_, TP_B4_UB_LEN);

    // --- gmAxis / ubAxis detail (for debug) ---
    AppendAxisList(oss, "[gmAxis]", gmAxis_);
    AppendAxisList(oss, "[ubAxis]", ubAxis_);

    oss << "\n[HIF4DualCut] ===== End =====";
    OP_LOGD(context_, "%s", oss.str().c_str());
}

// ===========================================================================
// TransposeB4Strategy
// ===========================================================================

// ---------------------------------------------------------------------------
// TransposeB4Strategy constructor
//   Prepare inShape_/outShape_ from shapeInfo_ and initialize the seeker.
//   For HIF4: use b4 view (HIF4 element count), shapes are NOT divided by 2.
// ---------------------------------------------------------------------------
/**
 * @brief 由精简 shape 生成输入/输出轴视图，并初始化双切分搜索器（原构造函数逻辑）
 */
void TransposeB4Strategy::PrepareShape()
{
    const int32_t dim = static_cast<int32_t>(shapeInfo_.dim);
    if (dim < 1) {
        return;
    }
    std::vector<int64_t> inShape(dim);
    std::vector<int64_t> outShape(dim);
    std::vector<int64_t> reducedPerm(dim);
    for (int32_t i = 0; i < dim; i++) {
        inShape[i] = shapeInfo_.reducedInShape[i];
        reducedPerm[i] = shapeInfo_.reducedPerm[i];
        outShape[i] = shapeInfo_.reducedInShape[reducedPerm[i]];
    }
    seeker_.Init(inShape, outShape, reducedPerm, dim, context_, static_cast<int32_t>(ubBlockSize_));
}

// ---------------------------------------------------------------------------
// RunDualCut
//   For HIF4: shapes are in HIF4 elements (b4 view, no division by 2)
//   UB budget: (ubSize / 10) aligned down to 32B, then * bufferNum
//   Each buffer holds HIF4 elements, 2 elements per byte
//   Alignment: 64 HIF4 elements = 32 bytes
// ---------------------------------------------------------------------------
ge::graphStatus TransposeB4Strategy::RunDualCut()
{
    int32_t dim = static_cast<int32_t>(shapeInfo_.dim);

    static constexpr int32_t HIF4_TOTAL_BUFFER_NUM = 5; // 2 input + 2 output + 1 workspace
    int32_t perBufUbSize = (ubSize_ / (HIF4_TOTAL_BUFFER_NUM * B4_HIF4_BUFFER_NUM)) / ubBlockSize_ * ubBlockSize_;
    if (perBufUbSize <= 0) {
        perBufUbSize = ubBlockSize_;
    }
    int32_t effectiveUbSize = perBufUbSize * B4_HIF4_BUFFER_NUM;

    OP_LOGD(context_,
            "[HIF4DualCut] RunDualCut: dim=%d, ubSize=%d, perBufUbSize=%d (aligned), "
            "effectiveUbSize=%d bytes (%ld HIF4 elements), coreNum=%d",
            dim, ubSize_, perBufUbSize, effectiveUbSize,
            static_cast<int64_t>(effectiveUbSize) * B4_HIF4_ELEMENTS_PER_BYTE / B4_HIF4_BUFFER_NUM, coreNum_);

    // The seeker is initialized in the constructor; run the dual-cut search here.
    bool ok = seeker_.FindDualCutAxis(effectiveUbSize, B4_HIF4_BUFFER_NUM);
    if (!ok) {
        OP_LOGE(context_, "[HIF4DualCut] FindDualCutAxis failed — "
                          "last dim too small or UB budget exhausted.");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context_, "[HIF4DualCut] FindDualCutAxis success, generating tiling data.");
    seeker_.GenTilingData();
    seeker_.ComputeBlockTiling(coreNum_);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// FillTilingData
// ---------------------------------------------------------------------------
bool TransposeB4Strategy::SetTilingData()
{
    TransposeB4TilingData* tilingData = context_->GetTilingData<TransposeB4TilingData>();
    if (tilingData == nullptr) {
        OP_LOGE(context_->GetNodeName(), "get tiling data failed, capacity is not enough!");
        return false;
    }

    std::copy(seeker_.gmShape_, seeker_.gmShape_ + TP_B4_GM_LEN, tilingData->gmShape);
    std::copy(seeker_.gmInStride_, seeker_.gmInStride_ + TP_B4_GM_LEN, tilingData->gmInStride);
    std::copy(seeker_.gmOutStride_, seeker_.gmOutStride_ + TP_B4_GM_LEN, tilingData->gmOutStride);
    std::copy(seeker_.ubShape_, seeker_.ubShape_ + TP_B4_UB_LEN, tilingData->ubShape);
    std::copy(seeker_.ubInStride_, seeker_.ubInStride_ + TP_B4_UB_LEN, tilingData->ubInStride);
    std::copy(seeker_.ubOutStride_, seeker_.ubOutStride_ + TP_B4_UB_LEN, tilingData->ubOutStride);
    std::copy(seeker_.moveUbStride_, seeker_.moveUbStride_ + TP_B4_UB_LEN, tilingData->moveUbStride);
    for (int32_t i = 0; i < TP_B4_UB_LEN; i++) {
        tilingData->ubPerm[i] = static_cast<int8_t>(seeker_.ubPerm_[i]);
        tilingData->ubShapeId[i] = static_cast<int8_t>(seeker_.ubShapeId_[i]);
    }
    tilingData->cutAxisIdx01 = static_cast<int8_t>(seeker_.cutAxisIdx01_);
    tilingData->cutAxisIdx02 = static_cast<int8_t>(seeker_.cutAxisIdx02_);
    tilingData->cutAxisTail01 = seeker_.cutAxisTail01_;
    tilingData->cutAxisTail02 = seeker_.cutAxisTail02_;
    tilingData->outCutId = static_cast<int8_t>(seeker_.outCutId_);
    tilingData->inCutId = static_cast<int8_t>(seeker_.inCutId_);
    tilingData->outerAxisNums = static_cast<int8_t>(seeker_.outerAxisNums_);
    tilingData->innerAxisNums = static_cast<int8_t>(seeker_.innerAxisNums_);
    tilingData->dimsPerCore = seeker_.dimsPerCore_;
    tilingData->dimsPerCoreTail = seeker_.dimsPerCoreTail_;
    int32_t ubSizePerBuf = (seeker_.ubSize_ / B4_HIF4_BUFFER_NUM) / ubBlockSize_ * ubBlockSize_;
    if (ubSizePerBuf <= 0) {
        ubSizePerBuf = ubBlockSize_;
    }
    tilingData->ubSize = ubSizePerBuf;

    // Fill perm info for kernel
    const int32_t permSize = static_cast<int32_t>(shapeInfo_.dim);
    tilingData->permSize = static_cast<int8_t>(permSize);
    for (int32_t i = 0; i < TP_MAX_AXIS_NUM; i++) {
        tilingData->perm[i] = (i < permSize) ? static_cast<int8_t>(shapeInfo_.reducedPerm[i]) : 0;
    }

    return true;
}

// ---------------------------------------------------------------------------
// DoExecute: PrepareShape -> RunDualCut（仅双切分，无单切分回退）
// ---------------------------------------------------------------------------
/**
 * @brief 主体切分（失败即硬错误）
 */
TilingStatus TransposeB4Strategy::DoExecute()
{
    PrepareShape();
    // Dual-cut only (no sole-cut fallback)
    if (RunDualCut() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "[HIF4DualCut] RunDualCut failed.");
        return TilingStatus::FAILURE;
    }
    return TilingStatus::SUCCESS;
}

/**
 * @brief 打印 TilingData
 */
void TransposeB4Strategy::LogTilingData() { seeker_.PrintTilingData(); }

} // namespace optiling
