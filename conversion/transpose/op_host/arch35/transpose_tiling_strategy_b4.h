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
 * \file transpose_tiling_strategy_b4.h
 * \brief B4(10009) 策略类
 *
 * 适用场景：4bit 类型（HIF4）且尾轴转置，使用双切分策略搬运。
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_B4_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_B4_H

#include <vector>
#include "transpose_tiling_strategy.h"

namespace optiling {
using namespace Transpose;

// ===== B4 双切分的硬件常量与内部辅助类（辅助类保留在策略类外）=====
static constexpr int32_t B4_HIF4_ELEMENTS_PER_BYTE = 2; // 2 HIF4 elements per byte
static constexpr int32_t B4_HIF4_ALIGN_ELEMENTS = 64;   // 64 HIF4 elements = 32 bytes (alignment unit)
static constexpr int32_t B4_HIF4_BUFFER_NUM = 2;        // double buffer
static constexpr int32_t B4_NUM_TWO = 2;

// -----------------------------------------------------------------------
// B4AxisInf: single-axis descriptor
// -----------------------------------------------------------------------
struct B4AxisInf {
    int64_t dim = 1;
    int64_t stride = 0;
    int32_t idx = 0;   // original axis index in the shape
    uint32_t code = 0; // bitmask: 1 << idx

    B4AxisInf() = default;
    B4AxisInf(int64_t dim_, int64_t stride_, int32_t idx_)
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
// B4AxisCutter: find the cut axis for one view (output or input)
//   Scans from the innermost axis outward, accumulating the product.
//   The first axis whose cumulative product exceeds ubBound is the cut axis.
//   The cut axis is split into:
//     innerCutAxis: dim = ubBound / prod_of_inner_axes  (aligned down)
//     outerCutAxis: dim = ceil(cutDim / innerCutAxis.dim)
// -----------------------------------------------------------------------
class B4AxisCutter {
public:
    B4AxisCutter() = default;

    // Initialize (or re-initialize) the cutter from one view's axis list.
    void Init(const std::vector<B4AxisInf>& axes, int32_t alignedFactor)
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
                B4AxisInf a = axes[i];
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
    B4AxisInf innerCutAxis_;
    B4AxisInf outerCutAxis_;
    std::vector<B4AxisInf> axisList_;
};

// -----------------------------------------------------------------------
// B4DualCutAxisSeeker: orchestrate the dual-cut search
// -----------------------------------------------------------------------
class B4DualCutAxisSeeker {
public:
    B4DualCutAxisSeeker() = default;

    // Build the input/output axis views and cutters from shapes and perm.
    // Must be called once before FindDualCutAxis.
    void Init(const std::vector<int64_t>& inShape, const std::vector<int64_t>& outShape,
              const std::vector<int64_t>& perm, int32_t dimNums, gert::TilingContext* context, int32_t ubBlockSize);

    // Main entry: find dual-cut axes and populate gmAxis_ / ubAxis_
    bool FindDualCutAxis(int32_t ubSize, int32_t bufferNum);

    // Distribute gmAxis_ iterations across cores
    void ComputeBlockTiling(int32_t coreNum);

    // Populate tiling data fields
    void GenTilingData();

    // Compute output stride for a given axis (similar to as_strided's ComputeOutputShape)
    int64_t ComputeOutputStride(const B4AxisInf& axis) const;

    // Print all tiling data fields
    void PrintTilingData() const;

public:
    gert::TilingContext* context_ = nullptr;

    std::vector<B4AxisInf> inputAxis_;  // axes in input order (for input view)
    std::vector<B4AxisInf> outputAxis_; // axes in output order (for output view)
    std::vector<B4AxisInf> gmAxis_;     // outer (multi-core) axes
    std::vector<B4AxisInf> ubAxis_;     // inner (UB) axes

    B4AxisCutter inputCutter_;
    B4AxisCutter outputCutter_;

    int32_t alignedFactor_ = B4_HIF4_ALIGN_ELEMENTS; // 64 HIF4 elements
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
    int32_t ubSize_ = 0;      // effective UB size per buffer (bytes)
    int32_t ubBlockSize_ = 0; // platform UB block size (bytes)

private:
    // Compute product of axes in axisSet
    int64_t ComputeSetAxisProd(uint32_t axisSet) const;

    // Compute remaining UB capacity after fixing axisSet axes
    int64_t ComputeRemainUB(int64_t ubNum, uint32_t axisSet) const;

    // Find axes common to both cutters' ubSet (and cross-cut membership)
    uint32_t FindCommonUbAxis(const B4AxisCutter* c1, const B4AxisCutter* c2) const;

    // Remove commonSet from both cutters
    void UpdateCommonAxis(uint32_t commonSet);

    // Dispatch to CutOneAxis or CutTwoAxis
    bool CutAxis(uint32_t ubAxisSet, int64_t remainNums);

    bool CutTwoAxis(uint32_t ubAxisSet, int64_t remainNums, std::vector<B4AxisInf>& innerAxis,
                    std::vector<B4AxisInf>& outerAxis);

    bool CutOneAxis(uint32_t ubAxisSet, int64_t remainNums, std::vector<B4AxisInf>& innerAxis,
                    std::vector<B4AxisInf>& outerAxis);

    // --- CutAxis helpers ---
    // Dispatch to CutTwoAxis/CutOneAxis (or no-op) with fallback on failure.
    void DispatchCut(uint32_t& ubAxisSet, int64_t remainNums, uint32_t cutAxisSet, std::vector<B4AxisInf>& innerAxis,
                     std::vector<B4AxisInf>& outerAxis);
    // Drop "fake" cuts (outerDim == 1) and update cutAxisNums_.
    void FilterFakeCuts(uint32_t& ubAxisSet, std::vector<B4AxisInf>& innerAxis, std::vector<B4AxisInf>& outerAxis);
    // Partition outputAxis_ into gmAxis_ / ubAxis_.
    void FillGmUbAxis(uint32_t ubAxisSet, uint32_t cutAxisSet);
    // Evict UB axes to gmAxis_ when the MOVE_ALIGN limit is exceeded.
    void EvictExcessUbAxis(const std::vector<B4AxisInf>& innerAxis);
    // Append inner cut axes to ubAxis_ and outer cut axes to gmAxis_.
    void AppendCutAxes(const std::vector<B4AxisInf>& innerAxis, const std::vector<B4AxisInf>& outerAxis);
    // Sort gmAxis_ / ubAxis_ by input axis index.
    void SortGmUbAxis();
    // Shrink inner cut dims if the aligned UB footprint overflows.
    void ShrinkInnerCutDims(std::vector<B4AxisInf>& innerAxis, std::vector<B4AxisInf>& outerAxis);
    // Aligned product of UB dims used for the UB capacity check.
    int64_t ComputeAlignedUbProd() const;
    // Shrink a single inner cut axis and sync ubAxis_ / gmAxis_.
    void ShrinkSingleCutAxis(size_t i, int64_t ratio, std::vector<B4AxisInf>& innerAxis,
                             std::vector<B4AxisInf>& outerAxis);

    // --- CutTwoAxis helpers ---
    bool NeedEvenDim(int32_t axisIdx) const;
    void FixEvenMin2IfNeeded(int64_t& v, int32_t axisIdx) const;
    int64_t GetCutterOrigDim(const B4AxisCutter* cutter) const;
    void AlignInnerDimIfNeeded(int64_t& inner, int64_t orig, int32_t axisIdx, bool isFinal, int64_t remainNums) const;
    void ComputeCutInnerDims(B4AxisInf& inInner, B4AxisInf& outInner, int64_t inOrigDim, int64_t outOrigDim,
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

/**
 * @brief B4(10009) 策略
 *
 * 准入：is4bit_ && isLastAxisTranspose && dim>=1（不受 related 门控）。
 * 双切分失败返回 FAILURE（确定性硬错误，不切换后续策略）。
 */
class TransposeB4Strategy : public TransposeTilingStrategy {
public:
    using TransposeTilingStrategy::TransposeTilingStrategy;

    // G1：准入检查，纯判定、无副作用
    bool IsSupport() const override;

protected:
    // 主体切分
    TilingStatus DoExecute() override;
    // 写 TransposeB4TilingData（容量不足 → FAILURE）
    bool SetTilingData() override;
    // 打印 TilingData
    void LogTilingData() override;
    uint32_t GetUsedCoreNum() const override { return static_cast<uint32_t>(seeker_.useCoreNum_); }
    uint64_t GetTilingKey() const override { return static_cast<uint64_t>(TK_B4); }

private:
    // ===== B4 双切分私有状态 =====
    void PrepareShape();
    ge::graphStatus RunDualCut();

    std::vector<int64_t> inShape_;  // reduced input shape (element view)
    std::vector<int64_t> outShape_; // reduced output shape (element view)
    B4DualCutAxisSeeker seeker_;    // 双切分搜索器（原内部类）
};

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_B4_H
