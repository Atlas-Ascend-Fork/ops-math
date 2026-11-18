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
 * \file transpose_tiling_b4_dual_cut_arch35.h
 * \brief HIF4 dual-cut tiling for Transpose operator on ascend950 (David).
 *
 * Trigger condition (TilingKey = 10009, SplitMode::B4_DOUBLE_SPLIT):
 *   dtype = HIF4 (DT_FLOAT4_E2M1 / DT_FLOAT4_E1M2)
 *   AND (last-axis transpose  OR  non-last-axis transpose AND lastAxis < 32 elements)
 *
 * Dual-cut strategy (inspired by as_strided DualCutAxisSeeker):
 *   Given input shape [A,B,C,D] -> output shape [A,B,D,C] (last two axes swapped):
 *
 *   Step 1  BuildAxisInfo   — build outputAxis (output-order strides) and
 *                             inputAxis (input-order strides, sorted by stride desc)
 *   Step 2  FindDualCutAxis — iteratively find the cut axis for both output and input
 *                             views using sqrt(remainUB) as the UB bound per direction
 *   Step 3  SplitGmUb       — partition axes into gmAxis (multi-core loop) and
 *                             ubAxis (UB inner loop)
 *   Step 4  CalcBlockTiling — distribute gmAxis iterations across cores (NK10)
 *   Step 5  FillTilingData  — write HIF4DualCutTilingData, set TilingKey=10009
 *
 * UB buffer layout:
 *   HIF4 element = 0.5 B (b4 view, shapes NOT divided by 2)
 *   alignedFactor = 64 HIF4 elements = 32 bytes
 *   bufferNum = 2 (double buffer)
 *   ubNum = ubSize * 2 / bufferNum  (2 HIF4 elements per byte)
 *
 * Tiling data layout (kernel side reads in this order):
 *   gmShape[TP_B4_GM_LEN]      — outer loop shape (multi-core)
 *   gmInStride[TP_B4_GM_LEN]   — input  strides for each gm axis
 *   gmOutStride[TP_B4_GM_LEN]  — output strides for each gm axis
 *   ubShape[TP_B4_UB_LEN]      — UB inner loop shape
 *   ubInStride[TP_B4_UB_LEN]   — input  strides for each ub axis
 *   ubOutStride[TP_B4_UB_LEN]  — output strides for each ub axis
 *   cutAxisIdx01 / cutAxisIdx02    — which ub axis is the cut axis (tail handling)
 *   cutAxisTail01 / cutAxisTail02  — tail size for the cut axis
 *   outerAxisNums / innerAxisNums  — number of gm / ub axes actually used
 *   dimsPerCore / useCoreNum / dimsPerCoreTail — multi-core split
 */

#ifndef TRANSPOSE_TILING_B4_DUAL_CUT_ARCH35_H
#define TRANSPOSE_TILING_B4_DUAL_CUT_ARCH35_H

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>

#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "transpose_tiling_base.h"
#include "transpose_tiling_arch35.h"

namespace optiling {
using namespace Transpose;
namespace TransHIF4DualCut {

// -----------------------------------------------------------------------
// Hardware / buffer constants for ascend950 (David)
// -----------------------------------------------------------------------
static constexpr int32_t HIF4_BLOCK_BYTES = 32;      // 32 B per block
static constexpr int32_t HIF4_ELEMENTS_PER_BYTE = 2; // 2 HIF4 elements per byte
static constexpr int32_t HIF4_ALIGN_ELEMENTS = 64;   // 64 HIF4 elements = 32 bytes (alignment unit)
static constexpr int32_t HIF4_BUFFER_NUM = 2;        // double buffer
static constexpr int32_t NUM_TWO = 2;

// -----------------------------------------------------------------------
// Tiling data structure for HIF4 dual-cut (TilingKey = 10009) is defined in
// op_kernel/arch35/transpose_tiling_data.h and shared with the kernel side.
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
// AxisInf: single-axis descriptor
// -----------------------------------------------------------------------
struct AxisInf {
    int64_t dim = 1;
    int64_t stride = 0;
    int32_t idx = 0;   // original axis index in the shape
    uint32_t code = 0; // bitmask: 1 << idx

    AxisInf() = default;
    AxisInf(int64_t dim_, int64_t stride_, int32_t idx_)
        : dim(dim_), stride(stride_), idx(idx_), code(1u << static_cast<uint32_t>(idx_))
    {}

    void Reset(int64_t dim_, int64_t stride_, int32_t idx_)
    {
        dim = dim_;
        stride = stride_;
        idx = idx_;
        code = 1u << static_cast<uint32_t>(idx_);
    }
};

// -----------------------------------------------------------------------
// AxisCutter: find the cut axis for one view (output or input)
//   Scans from the innermost axis outward, accumulating the product.
//   The first axis whose cumulative product exceeds ubBound is the cut axis.
//   The cut axis is split into:
//     innerCutAxis: dim = ubBound / prod_of_inner_axes  (aligned down)
//     outerCutAxis: dim = ceil(cutDim / innerCutAxis.dim)
// -----------------------------------------------------------------------
class AxisCutter {
public:
    AxisCutter() = default;

    // Initialize (or re-initialize) the cutter from one view's axis list.
    void Init(const std::vector<AxisInf>& axes, int32_t alignedFactor)
    {
        alignedFactor_ = alignedFactor;
        initDimNums_ = static_cast<int32_t>(axes.size());
        // The final axis of THIS view is the LAST element in the axes array.
        // For inputAxis_ it's the original final axis; for outputAxis_ it's
        // the axis that appears last in output order (perm[dimNums-1]).
        finalAxisIdx_ = axes.empty() ? -1 : axes.back().idx;
        originFinalDim_ = 0;
        ubSet_ = 0;
        cutSet_ = 0;
        cutIdx_ = -1;
        cutFinal_ = false;
        axisList_.clear();
        for (int32_t i = 0; i < initDimNums_; i++) {
            // Align ONLY the final axis of this view (DMA access boundary).
            if (axes[i].idx == finalAxisIdx_) {
                int64_t alignedDim = ((axes[i].dim + alignedFactor_ - 1) / alignedFactor_) * alignedFactor_;
                originFinalDim_ = static_cast<int32_t>(axes[i].dim);
                AxisInf a = axes[i];
                a.dim = alignedDim;
                axisList_.push_back(a);
            } else {
                axisList_.push_back(axes[i]);
            }
        }
    }

    // Find the cut axis given ubBound = floor(sqrt(remainUB))
    void FindCutAxis(int64_t ubBound);

    // Remove axes whose bitmask intersects setCode from axisList_
    void RemoveAxis(uint32_t setCode);

    int32_t initDimNums_ = 0;
    int32_t alignedFactor_ = 32;
    int32_t finalAxisIdx_ = -1; // original idx of THIS view's final axis
    int32_t originFinalDim_ = 0;
    uint32_t ubSet_ = 0;    // bitmask of axes fully inside UB
    uint32_t cutSet_ = 0;   // bitmask of the cut axis
    int32_t cutIdx_ = -1;   // index in axisList_ of the cut axis
    bool cutFinal_ = false; // true if the cut axis is the final axis
    AxisInf innerCutAxis_;
    AxisInf outerCutAxis_;
    std::vector<AxisInf> axisList_;
};

// -----------------------------------------------------------------------
// DualCutAxisSeeker: orchestrate the dual-cut search
// -----------------------------------------------------------------------
class DualCutAxisSeeker {
public:
    DualCutAxisSeeker() = default;

    // Build the input/output axis views and cutters from shapes and perm.
    // Must be called once before FindDualCutAxis.
    void Init(const std::vector<int64_t>& inShape, const std::vector<int64_t>& outShape,
              const std::vector<int64_t>& perm, int32_t dimNums, gert::TilingContext* context);

    // Main entry: find dual-cut axes and populate gmAxis_ / ubAxis_
    bool FindDualCutAxis(int32_t ubSize, int32_t bufferNum);

    // Distribute gmAxis_ iterations across cores
    void ComputeBlockTiling(int32_t coreNum);

    // Populate tiling data fields
    void GenTilingData();

    // Compute output stride for a given axis (similar to as_strided's ComputeOutputShape)
    int64_t ComputeOutputStride(const AxisInf& axis) const;

    // Print all tiling data fields
    void PrintTilingData() const;

public:
    gert::TilingContext* context_ = nullptr;

    std::vector<AxisInf> inputAxis_;  // axes in input order (for input view)
    std::vector<AxisInf> outputAxis_; // axes in output order (for output view)
    std::vector<AxisInf> gmAxis_;     // outer (multi-core) axes
    std::vector<AxisInf> ubAxis_;     // inner (UB) axes

    AxisCutter inputCutter_;
    AxisCutter outputCutter_;

    int32_t alignedFactor_ = HIF4_ALIGN_ELEMENTS; // 64 HIF4 elements
    int32_t dimNums_ = 0;

    // Original shapes for stride calculation
    std::vector<int64_t> outShape_;
    std::vector<int64_t> perm_;

    // Tiling data arrays (written by GenTilingData, read by FillTilingData)
    int32_t gmShape_[TP_B4_GM_LEN] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t gmInStride_[TP_B4_GM_LEN] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t gmOutStride_[TP_B4_GM_LEN] = {0, 0, 0, 0, 0, 0, 0, 0};

    int32_t ubShape_[TP_B4_UB_LEN] = {1, 1, 1, 1};
    int64_t ubInStride_[TP_B4_UB_LEN] = {0, 0, 0, 0};
    int64_t ubOutStride_[TP_B4_UB_LEN] = {0, 0, 0, 0};
    int64_t moveUbStride_[TP_B4_UB_LEN] = {0, 0, 0, 0};
    int32_t ubPerm_[TP_B4_UB_LEN] = {0, 0, 0, 0};
    int32_t ubShapeId_[TP_B4_UB_LEN] = {-1, -1, -1, -1};

    int32_t cutAxisNums_ = 0;
    int32_t outerAxisNums_ = 0;
    int32_t innerAxisNums_ = 0;
    int32_t cutAxisIdx01_ = -1;
    int32_t cutAxisIdx02_ = -1;
    int32_t cutAxisTail01_ = 0;
    int32_t cutAxisTail02_ = 0;
    int32_t outCutId_ = -1; // output cut axis index in gmShape valid part
    int32_t inCutId_ = -1;  // input cut axis index in gmShape valid part

    int32_t dimsPerCore_ = 1;
    int32_t useCoreNum_ = 1;
    int32_t dimsPerCoreTail_ = 1;
    int32_t ubSize_ = 0; // effective UB size per buffer (bytes)

private:
    // Compute product of axes in axisSet
    int64_t ComputeSetAxisProd(uint32_t axisSet) const;

    // Compute remaining UB capacity after fixing axisSet axes
    int64_t ComputeRemainUB(int64_t ubNum, uint32_t axisSet) const;

    // Find axes common to both cutters' ubSet (and cross-cut membership)
    uint32_t FindCommonUbAxis(const AxisCutter* c1, const AxisCutter* c2) const;

    // Remove commonSet from both cutters
    void UpdateCommonAxis(uint32_t commonSet);

    // Dispatch to CutOneAxis or CutTwoAxis
    bool CutAxis(uint32_t ubAxisSet, int64_t remainNums);

    bool CutTwoAxis(uint32_t ubAxisSet, int64_t remainNums, std::vector<AxisInf>& innerAxis,
                    std::vector<AxisInf>& outerAxis);

    bool CutOneAxis(uint32_t ubAxisSet, int64_t remainNums, std::vector<AxisInf>& innerAxis,
                    std::vector<AxisInf>& outerAxis);

    // --- CutAxis helpers ---
    // Dispatch to CutTwoAxis/CutOneAxis (or no-op) with fallback on failure.
    void DispatchCut(uint32_t& ubAxisSet, int64_t remainNums, uint32_t cutAxisSet, std::vector<AxisInf>& innerAxis,
                     std::vector<AxisInf>& outerAxis);
    // Drop "fake" cuts (outerDim == 1) and update cutAxisNums_.
    void FilterFakeCuts(uint32_t& ubAxisSet, std::vector<AxisInf>& innerAxis, std::vector<AxisInf>& outerAxis);
    // Partition outputAxis_ into gmAxis_ / ubAxis_.
    void FillGmUbAxis(uint32_t ubAxisSet, uint32_t cutAxisSet);
    // Evict UB axes to gmAxis_ when the MOVE_ALIGN limit is exceeded.
    void EvictExcessUbAxis(const std::vector<AxisInf>& innerAxis);
    // Append inner cut axes to ubAxis_ and outer cut axes to gmAxis_.
    void AppendCutAxes(const std::vector<AxisInf>& innerAxis, const std::vector<AxisInf>& outerAxis);
    // Sort gmAxis_ / ubAxis_ by input axis index.
    void SortGmUbAxis();
    // Shrink inner cut dims if the aligned UB footprint overflows.
    void ShrinkInnerCutDims(std::vector<AxisInf>& innerAxis, std::vector<AxisInf>& outerAxis);
    // Aligned product of UB dims used for the UB capacity check.
    int64_t ComputeAlignedUbProd() const;
    // Shrink a single inner cut axis and sync ubAxis_ / gmAxis_.
    void ShrinkSingleCutAxis(size_t i, int64_t ratio, std::vector<AxisInf>& innerAxis, std::vector<AxisInf>& outerAxis);

    // --- CutTwoAxis helpers ---
    bool NeedEvenDim(int32_t axisIdx) const;
    void FixEvenMin2IfNeeded(int64_t& v, int32_t axisIdx) const;
    int64_t GetCutterOrigDim(const AxisCutter* cutter) const;
    void AlignInnerDimIfNeeded(int64_t& inner, int64_t orig, int32_t axisIdx, bool isFinal, int64_t remainNums) const;
    void ComputeCutInnerDims(AxisInf& inInner, AxisInf& outInner, int64_t inOrigDim, int64_t outOrigDim,
                             int32_t inCutAxisIdx, int32_t outCutAxisIdx, int64_t remainNums);

    // --- GenTilingData helpers ---
    int32_t FindOutputPos(int32_t axisIdx) const;
    void FillGmArrays();
    void FillUbShapeAndId();
    void FillUbStrides();
    void FillUbPermAndMoveStride();
    void FillCutIds();
    void FixGmOutStrideForCuts();

    // Set cutAxisIdx01/02 and cutAxisTail01/02 after gmAxis_/ubAxis_ are finalized
    void SetCutAxisIdx();
};

// -----------------------------------------------------------------------
// TransposeTilingHIF4DualCut: top-level tiling class
// -----------------------------------------------------------------------
class TransposeTilingHIF4DualCut {
public:
    explicit TransposeTilingHIF4DualCut(gert::TilingContext* context, const ShapeInfo& shapeInfo, int64_t coreNum,
                                        int64_t ubSize);

    ge::graphStatus DoTiling();

private:
    ge::graphStatus BuildAxisInfo();
    ge::graphStatus RunDualCut();
    ge::graphStatus FillTilingData();

    gert::TilingContext* context_ = nullptr;
    ShapeInfo shapeInfo_;
    int32_t coreNum_ = 0;
    int32_t ubSize_ = 0;

    // Populated by the constructor
    std::vector<int64_t> inShape_;  // reduced input shape (element view)
    std::vector<int64_t> outShape_; // reduced output shape (element view)

    DualCutAxisSeeker seeker_;
};

} // namespace TransHIF4DualCut
} // namespace optiling

#endif // TRANSPOSE_TILING_B4_DUAL_CUT_ARCH35_H
