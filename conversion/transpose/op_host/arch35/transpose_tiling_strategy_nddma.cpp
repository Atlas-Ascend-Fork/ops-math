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
 * \file transpose_tiling_strategy_nddma.cpp
 * \brief NDDMA 策略类实现（NDDMA_CUT_1(10002)/NDDMA_CUT_2(10003) 共用）
 *
 * 实现 NDDMA 单/双轴切分的切轴判定、分核、UB 内 shape 与区间边界计算；
 * 普通/关联入口分别写入 TransposeNddmaCut1/Cut2TilingData 紧凑结构。
 */

#include "transpose_tiling_strategy_nddma.h"

#include <cmath>

namespace optiling {
namespace {
static constexpr double NDDMA_VEC_CORE_USED_THRES_HOLD = 0.9; ///< 核利用率阈值
} // namespace

bool TransposeNddmaStrategy::IsSupport() const { return shapeInfo_.dim <= TP_NDDMA_MAX_DIM_NUM; }

/**
 * @brief 执行 NDDMA tiling
 *
 * 1. CalcUBSplitInfo：UB 预算 + DoSplitUB 判定单切/双切；
 * 2. 分核 / UB 内 shape / 区间按 tilingKey_ 分派；
 * 3. 收尾（SetTilingData → SetContext → LogTilingData）由通用基类 Execute() 统一完成。
 */
TilingStatus TransposeNddmaStrategy::DoExecute()
{
    CalcUBSplitInfo();
    CalcBlockSplitInfo();
    NDDMADimExpand();
    GetInUbShapeInfo();
    GetIntervalInfo();
    return TilingStatus::SUCCESS;
}

/**
 * @brief NDDMA UB 切分
 *
 * UB 预算按 NDDMA 缓冲数折算后取 sqrt，作为切轴初始预算：
 *   ubElement = ubSize_ / GetNddmaBufferNum() / eleLenInBytes；
 *   inUbElement = sqrt(ubElement)；随后 DoSplitUB 完成切轴判定（单切/双切）。
 */
void TransposeNddmaStrategy::CalcUBSplitInfo()
{
    OP_LOGD(context_->GetNodeName(), "Entering CalcUBSplitInfo");
    splitInfo_.ubElement = ubSize_ / GetNddmaBufferNum() / shapeInfo_.eleLenInBytes;
    splitInfo_.inUbElement = sqrt(splitInfo_.ubElement);
    DoSplitUB();
}
/**
 * @brief 切轴主流程：输入切轴 + 输出扫描 + absorb 重试/revert（判定单切/双切）
 *
 * 先按原始 sqrt 预算切轴（no absorption）；若结果为 NDDMA_CUT_2，则尝试
 * absorption 策略看能否得到 NDDMA_CUT_1；若仍不是 NDDMA_CUT_1 或 UB 越界校验失败
 * 则回退到原始结果。
 */
/**
 * @brief 分核：按 tilingKey_（10002/10003）分派
 */
void TransposeNddmaStrategy::CalcBlockSplitInfo()
{
    switch (tilingKey_) {
        case TK_NDDMA_CUT_1:
            CalcBlockSplitInfoForCutOnce();
            break;
        case TK_NDDMA_CUT_2:
            CalcBlockSplitInfoForCutTwice();
            break;
        // DoSplitUB 之后 tilingKey 必为 10002/10003，default 不可达（防御性保留）
        // LCOV_EXCL_START
        default:
            break;
            // LCOV_EXCL_STOP
    }
}

void TransposeNddmaStrategy::DoSplitUB()
{
    // Save initial state for potential retry with absorption strategy
    SplitInfo initialSplitInfo = splitInfo_;

    // Strategy 1: original (sqrt budget, no absorption)
    int64_t remainingTotalElment = DoSplitUBInput(false);
    DoSplitUBOutputScan(remainingTotalElment);

    // If NDDMA_CUT_2, try absorption strategy to see if NDDMA_CUT_1 is achievable
    if (tilingKey_ == TK_NDDMA_CUT_2) {
        SplitInfo originalResult = splitInfo_;
        uint64_t originalTilingKey = tilingKey_;

        // Reset to initial state and retry with absorption
        splitInfo_ = initialSplitInfo;
        remainingTotalElment = DoSplitUBInput(true);
        DoSplitUBOutputScan(remainingTotalElment);

        // If absorption didn't yield NDDMA_CUT_1, revert to original strategy
        if (tilingKey_ != TK_NDDMA_CUT_1) {
            splitInfo_ = originalResult;
            tilingKey_ = originalTilingKey;
        } else {
            // Validate: CalcBlockSplitInfoForCutOnce will multiply outUbFactor by
            // inUbFactor when inCutIndex == reducedPerm[outCutIndex]. Check UB bounds
            // with the post-modification factor to avoid UB OOB in the kernel.
            int64_t effectiveOutUbFactor = splitInfo_.outUbFactor;
            if (splitInfo_.inCutIndex == shapeInfo_.reducedPerm[splitInfo_.outCutIndex]) {
                effectiveOutUbFactor *= splitInfo_.inUbFactor;
            }
            if (UbOutOfBoundCheck(splitInfo_.outCutIndex, effectiveOutUbFactor, true)) {
                splitInfo_ = originalResult;
                tilingKey_ = originalTilingKey;
            }
        }
    }
}
/**
 * @brief 输入侧切轴：确定输入切分轴 inCutIndex 和切分因子 inUbFactor
 *
 * 切轴逻辑：
 * 从输入最末轴（连续性最好的轴）向前扫描，优先整体"吞掉"较短的靠后轴；
 * 遇到第一根 UB 预算放不下的轴时，将其切为主块+尾块，并停止扫描。
 *
 * @param tryAbsorb 是否尝试用完整 UB 预算吸收切分轴（避免不必要的 NDDMA_CUT_2）
 * @return 切分后剩余的总 block 基数
 */
int64_t TransposeNddmaStrategy::DoSplitUBInput(bool tryAbsorb)
{
    int64_t remainingTotalElment = shapeInfo_.totalVolumeActual;
    bool hasAbsorbed = false;
    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        int64_t currentShapeDim = shapeInfo_.reducedInShape[shapeInfo_.dim - 1 - i];
        if (splitInfo_.inUbElement < currentShapeDim) {
            // 该轴放不下 → 它就是输入切分轴，主块一次搬 inUbElement 个元素
            // Try to absorb this axis using full UB budget and cut on a more major axis instead.
            if (tryAbsorb && !hasAbsorbed && (shapeInfo_.dim - 1 - i) > 0) {
                int64_t fullBudgetRemaining = splitInfo_.ubElement / splitInfo_.inUbActual;
                if (currentShapeDim <= fullBudgetRemaining) {
                    splitInfo_.inUbActual *= currentShapeDim;
                    splitInfo_.inUbElement = fullBudgetRemaining / currentShapeDim;
                    splitInfo_.inUbFactor = currentShapeDim;
                    remainingTotalElment /= currentShapeDim;
                    hasAbsorbed = true;
                    continue;
                }
            }
            splitInfo_.inCutIndex = shapeInfo_.dim - 1 - i;
            splitInfo_.inUbFactor = splitInfo_.inUbElement;
            splitInfo_.inTailFactor = currentShapeDim % splitInfo_.inUbFactor;
            splitInfo_.inUbActual *= splitInfo_.inUbElement;
            remainingTotalElment = remainingTotalElment / currentShapeDim *
                                   Ops::Base::CeilDiv(currentShapeDim, splitInfo_.inUbElement);
            break;
        } else {
            splitInfo_.inUbElement /= currentShapeDim;
            splitInfo_.inUbActual *= currentShapeDim;
            remainingTotalElment /= currentShapeDim;
            splitInfo_.inUbFactor = currentShapeDim;
        }
    }
    // 输出侧剩余预算：UB 总量扣除输入侧已占用的部分
    splitInfo_.outUbElement = splitInfo_.ubElement / splitInfo_.inUbActual;
    return remainingTotalElment;
}
/**
 * @brief 核心切轴逻辑（DoSplitUB 主流程）
 *
 * 完成输入切轴后（DoSplitUBInput），继续在**输出 perm 顺序**上做切轴：
 * 从输出最末轴向前扫描，在 outUbElement 预算内决定是否切分输出侧，
 * 最终依据"输出切分轴 vs 输入切分轴映射位置"判定 NDDMA_CUT_1 还是 NDDMA_CUT_2。
 */
void TransposeNddmaStrategy::DoSplitUBOutputScan(int64_t remainingTotalElment)
{
    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        int64_t currentSplitIndex = shapeInfo_.dim - 1 - i;
        if (shapeInfo_.reducedPerm[currentSplitIndex] > splitInfo_.inCutIndex) { // skip axis full cut by input shape
            continue;
        }
        int64_t currentShapeDim = shapeInfo_.reducedOutShape[currentSplitIndex];
        if (shapeInfo_.reducedPerm[currentSplitIndex] == splitInfo_.inCutIndex) {
            currentShapeDim = Ops::Base::CeilDiv(shapeInfo_.reducedInShape[splitInfo_.inCutIndex],
                                                 splitInfo_.inUbFactor);
        }
        remainingTotalElment /= currentShapeDim;
        int64_t coreNumTmp = remainingTotalElment * Ops::Base::CeilDiv(currentShapeDim, splitInfo_.outUbElement);
        if (splitInfo_.outUbElement < currentShapeDim) {
            if (coreNumTmp > coreNum_) { // use full coreNum
                int64_t coreNumMultiples = Ops::Base::FloorDiv(coreNumTmp, coreNum_);
                FindSplitFactorByMultiplesLast(currentSplitIndex, currentShapeDim, remainingTotalElment,
                                               coreNumMultiples);
            } else {
                splitInfo_.outCutIndex = currentSplitIndex;
                splitInfo_.outUbFactor = splitInfo_.outUbElement;
                splitInfo_.outTailFactor = currentShapeDim % splitInfo_.outUbElement;
            }
            break;
        } else {
            splitInfo_.outUbElement /= currentShapeDim;
            splitInfo_.outCutIndex = currentSplitIndex;
            splitInfo_.outUbFactor = currentShapeDim;
        }
    }
    if (splitInfo_.outCutIndex > FindOutIndex(splitInfo_.inCutIndex)) {
        tilingKey_ = TK_NDDMA_CUT_2;
    } else {
        tilingKey_ = TK_NDDMA_CUT_1;
    }
}
/**
 * @brief 根据输入轴索引查找其在输出 perm 中的位置
 *
 * @param index 输入轴索引
 * @return 输出 perm 中的位置索引
 */
int64_t TransposeNddmaStrategy::FindOutIndex(int64_t index)
{
    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        if (shapeInfo_.reducedPerm[i] == index) {
            return i;
        }
    }
    // index 必为某个 perm 值（调用方只传 inCutIndex），兜底不可达
    // LCOV_EXCL_START
    return 0;
    // LCOV_EXCL_STOP
}
/**
 * @brief UB 越界校验（NDDMA_CUT_1/NDDMA_CUT_2 分核因子回溯时使用）
 *
 * @param currentSplitIndex  当前候选切分轴（输出 perm 视角）
 * @param currentSplitValue  该轴拟使用的块长
 * @param calcIn             是否把输入切分因子 inUbFactor 计入（切输入侧因子回调时传 true）
 * @return true 越界（该因子不可用）；false 通过
 */
bool TransposeNddmaStrategy::UbOutOfBoundCheck(int64_t currentSplitIndex, int64_t currentSplitValue, bool calcIn)
{
    int64_t burstLenBlockAlign = 1;
    for (int64_t i = currentSplitIndex + 1; i < shapeInfo_.dim; i++) {
        burstLenBlockAlign *= shapeInfo_.reducedOutShape[i];
    }
    burstLenBlockAlign *= currentSplitValue;
    if (calcIn && shapeInfo_.reducedPerm[currentSplitIndex] == splitInfo_.inCutIndex) {
        burstLenBlockAlign *= splitInfo_.inUbFactor;
    }
    burstLenBlockAlign = Ops::Base::CeilAlign(burstLenBlockAlign * shapeInfo_.eleLenInBytes, ubBlockSize_) /
                         shapeInfo_.eleLenInBytes;
    int64_t inUbElements = burstLenBlockAlign;
    for (int64_t i = 0; i < currentSplitIndex; i++) {
        if (shapeInfo_.reducedPerm[i] > splitInfo_.inCutIndex) {
            inUbElements *= shapeInfo_.reducedOutShape[i];
        } else if (shapeInfo_.reducedPerm[i] == splitInfo_.inCutIndex) {
            inUbElements *= splitInfo_.inUbFactor;
        }
    }
    if (inUbElements > splitInfo_.ubElement) {
        return true;
    }
    return false;
}
/**
 * @brief NDDMA_CUT_1/NDDMA_CUT_2 分核因子回溯（Multiples 版，输出侧因子求值）
 *
 * 目标是让块总数等于 coreNum 整数倍的同时尽量用满核：
 * 从候选因子 i=1 递增，若潜在块数仍落在目标倍数且不越界 UB，取最小的合法因子；
 * 找不到完全等倍数的因子时，退而求其次选不越界的最大因子 bestI。
 */
void TransposeNddmaStrategy::FindSplitFactorByMultiplesLast(int64_t currentSplitIndex, int64_t currentInShapeDim,
                                                            int64_t remainingTotalElment, int64_t coreNumMultiples)
{
    splitInfo_.outCutIndex = currentSplitIndex;
    int64_t bestI = 1;
    for (int64_t i = 1; i <= splitInfo_.outUbElement; i++) {
        int64_t coreNumNew = remainingTotalElment * Ops::Base::CeilDiv(currentInShapeDim, i);
        if ((Ops::Base::FloorDiv(coreNumNew, coreNum_) == coreNumMultiples) &&
            !UbOutOfBoundCheck(currentSplitIndex, i, true)) {
            splitInfo_.outUbFactor = i;
            splitInfo_.outTailFactor = currentInShapeDim % i;
            return;
        }
        if (!UbOutOfBoundCheck(currentSplitIndex, i, true)) {
            bestI = i;
        }
    }
    splitInfo_.outUbFactor = bestI;
    splitInfo_.outTailFactor = currentInShapeDim % bestI;
}
/**
 * @brief NDDMA_CUT_1 分核逻辑
 *
 * 输入切轴（DoSplitUBInput）＋输出切轴（DoSplitUB）完成后，计算把 block 摊到核上
 * 的分核参数（realCoreNum/blkFactor/blkTailFactor）。核利用率不足时回调输出切分因子
 * （rate>=0.9 且不越界 UB 的最大因子，找不到则取 rate 最大者）。
 */
void TransposeNddmaStrategy::CalcBlockSplitInfoForCutOnce()
{
    OP_LOGD(context_->GetNodeName(), "Entering CalcBlockSplitInfoForCutOnce.");
    // input and output split the same axis, the split factor is subject to the output
    if (splitInfo_.inCutIndex == shapeInfo_.reducedPerm[splitInfo_.outCutIndex]) {
        splitInfo_.outUbFactor *= splitInfo_.inUbFactor;
        splitInfo_.outTailFactor = shapeInfo_.reducedOutShape[splitInfo_.outCutIndex] % splitInfo_.outUbFactor;
    }
    // input and output split different axis, but output split factor is the whole axis
    if (splitInfo_.outUbFactor == shapeInfo_.reducedOutShape[splitInfo_.outCutIndex]) {
        splitInfo_.outCutIndex = FindOutIndex(splitInfo_.inCutIndex);
        splitInfo_.outUbFactor = splitInfo_.inUbFactor;
        splitInfo_.outTailFactor = shapeInfo_.reducedOutShape[splitInfo_.outCutIndex] % splitInfo_.outUbFactor;
    }
    int64_t outUbAxis = Ops::Base::CeilDiv(shapeInfo_.reducedOutShape[splitInfo_.outCutIndex], splitInfo_.outUbFactor);
    int64_t outUbAxisExceptSplitAxis = 1;
    for (int64_t i = 0; i < splitInfo_.outCutIndex; i++) {
        if (shapeInfo_.reducedPerm[i] < splitInfo_.inCutIndex) {
            outUbAxisExceptSplitAxis *= shapeInfo_.reducedOutShape[i];
        }
    }
    if (outUbAxis < coreNum_) {
        // use at least NDDMA_VEC_CORE_USED_THRES_HOLD * coreNum
        int64_t currentSplitIndex = splitInfo_.outCutIndex;
        int64_t currentShapeDim = shapeInfo_.reducedOutShape[currentSplitIndex];
        int64_t bestI = 1;
        double bestRate = 0.0;
        bool foundValidFactor = false;
        for (int64_t i = splitInfo_.outUbFactor; i >= 1; i--) {
            int64_t coreNumNew = Ops::Base::CeilDiv(currentShapeDim, i) * outUbAxisExceptSplitAxis;
            double rate = static_cast<double>(coreNumNew) / coreNum_;
            if ((rate >= NDDMA_VEC_CORE_USED_THRES_HOLD) && !UbOutOfBoundCheck(currentSplitIndex, i, false)) {
                splitInfo_.outUbFactor = i;
                splitInfo_.outTailFactor = currentShapeDim % i;
                foundValidFactor = true;
                break;
            }
            if (!UbOutOfBoundCheck(currentSplitIndex, i, false) && rate > bestRate) {
                bestRate = rate;
                bestI = i;
            }
        }
        if (!foundValidFactor) {
            splitInfo_.outUbFactor = bestI;
            splitInfo_.outTailFactor = currentShapeDim % bestI;
        }
    }
    outUbAxis = Ops::Base::CeilDiv(shapeInfo_.reducedOutShape[splitInfo_.outCutIndex], splitInfo_.outUbFactor) *
                outUbAxisExceptSplitAxis;
    SetRealCoreNumAndBlkFactor(outUbAxis);
}
/**
 * @brief NDDMA_CUT_2 分核逻辑
 *
 * 双切场景 block 总数按"输出外层块数 × 输入轴块数"计算；核利用率不足时回调
 * **输入**切分因子 inUbFactor（输出切分轴已被外层循环枚举，提高并行度只能让输入侧切得更细）。
 */
void TransposeNddmaStrategy::CalcBlockSplitInfoForCutTwice()
{
    OP_LOGD(context_->GetNodeName(), "Entering CalcBlockSplitInfoForCutTwice.");
    int64_t outAxiseExceptSplitInAxis = 1;
    for (int64_t i = 0; i < splitInfo_.outCutIndex; i++) {
        if (shapeInfo_.reducedPerm[i] < splitInfo_.inCutIndex) {
            outAxiseExceptSplitInAxis *= shapeInfo_.reducedOutShape[i];
        }
    }
    outAxiseExceptSplitInAxis *= Ops::Base::CeilDiv(shapeInfo_.reducedOutShape[splitInfo_.outCutIndex],
                                                    splitInfo_.outUbFactor);
    int64_t inUbAxis = Ops::Base::CeilDiv(shapeInfo_.reducedInShape[splitInfo_.inCutIndex], splitInfo_.inUbFactor);
    if (outAxiseExceptSplitInAxis * inUbAxis < coreNum_) {
        // use at least NDDMA_VEC_CORE_USED_THRES_HOLD * coreNum
        int64_t currentSplitIndex = splitInfo_.inCutIndex;
        int64_t currentShapeDim = shapeInfo_.reducedInShape[currentSplitIndex];
        int64_t bestI = 1;
        double bestRate = 0.0;
        bool foundValidFactor = false;
        for (int64_t i = splitInfo_.inUbFactor; i >= 1; i--) {
            int64_t coreNumNew = Ops::Base::CeilDiv(currentShapeDim, i) * outAxiseExceptSplitInAxis;
            double rate = static_cast<double>(coreNumNew) / coreNum_;
            if ((rate >= NDDMA_VEC_CORE_USED_THRES_HOLD) && !UbOutOfBoundCheck(currentSplitIndex, i, true)) {
                splitInfo_.inUbFactor = i;
                splitInfo_.inTailFactor = currentShapeDim % i;
                foundValidFactor = true;
                break;
            }
            if (rate > bestRate && !UbOutOfBoundCheck(currentSplitIndex, i, true)) {
                bestRate = rate;
                bestI = i;
            }
        }
        if (!foundValidFactor) {
            splitInfo_.inUbFactor = bestI;
            splitInfo_.inTailFactor = currentShapeDim % bestI;
        }
    }
    inUbAxis = Ops::Base::CeilDiv(shapeInfo_.reducedInShape[splitInfo_.inCutIndex], splitInfo_.inUbFactor) *
               outAxiseExceptSplitInAxis;
    SetRealCoreNumAndBlkFactor(inUbAxis);
}
void TransposeNddmaStrategy::NDDMADimExpand()
{
    int64_t offset = (shapeInfo_.dim < TP_NDDMA_MAX_DIM_NUM) ? (TP_NDDMA_MAX_DIM_NUM - shapeInfo_.dim) : 0;

    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        expandedPerm_[i + offset] = shapeInfo_.reducedPerm[i] + offset;
        expandedInputShape_[i + offset] = shapeInfo_.reducedInShape[i];
        expandedOutputShape_[i + offset] = shapeInfo_.reducedOutShape[i];
    }
}

/**
 * @brief UB 内 shape 计算入口（按 TilingKey 分派）
 *
 * 生成 kernel 端 NDDMA SetupLoopInfo 所需的 inUb*SrcShape/inUb*DstShape：
 *   - SMALL_SHAPE：CalcInUbShapeInfoForNoNeedCut（无需 UB 切分的简单情形）
 *   - NDDMA_CUT_1：单轴切分，Main/Tail 两组 shape
 *   - NDDMA_CUT_2：双轴切分，Main/InputTail/OutputTail/Tail 四组 shape
 */

void TransposeNddmaStrategy::GetInUbShapeInfo()
{
    switch (tilingKey_) {
        case TK_NDDMA_CUT_1:
            CalcInUbShapeInfoForCutOnce();
            break;
        case TK_NDDMA_CUT_2:
            CalcInUbShapeInfoForCutTwice();
            break;
        // tilingKey 必为 10002/10003，default 不可达（防御性保留）
        // LCOV_EXCL_START
        default:
            break;
            // LCOV_EXCL_STOP
    }
}

/**
 * @brief NDDMA_CUT_2 区间信息计算入口（按 TilingKey 分派）
 *
 * 仅 NDDMA_CUT_2 需要 4 种数据区间边界：
 *   Main / InputTail / OutputTail / Tail（定义见 GetIntervalInfoForCutTwice）。
 */

void TransposeNddmaStrategy::GetIntervalInfo()
{
    switch (tilingKey_) {
        case TK_NDDMA_CUT_2:
            GetIntervalInfoForCutTwice();
            break;
        default:
            break;
    }
}

/**
 * @brief 计算"无需切分"场景的 UB 内 shape（SMALL_SHAPE 等路径）
 *
 * 只有一个切分轴（outCutIndex）需要以 outUbFactor/outTailFactor 切分，
 * 其余进入 block 的轴要么取完整输出 shape、要么置 1（外层循环负责）：
 *   - i > outCutIndexExpand（切分轴右侧）：完整 expandedOutputShape
 *   - i == outCutIndexExpand：Main=outUbFactor，Tail=outTailFactor
 *   - i < outCutIndexExpand：1（不进入单次 block）
 */

void TransposeNddmaStrategy::SetRealCoreNumAndBlkFactor(int64_t coreNum)
{
    if (coreNum >= coreNum_) {
        realCoreNum_ = coreNum_;
        blkFactor_ = coreNum / coreNum_;
        blkTailFactor_ = coreNum % coreNum_;
    } else {
        realCoreNum_ = coreNum;
        blkFactor_ = 1;
        blkTailFactor_ = 0;
    }
}

/**
 * @brief NDDMA_CUT_1 分核逻辑
 *
 * 输入切轴（DoSplitUBInput）＋输出切轴（DoSplitUB）完成后，计算把
 * block 摊到核上的分核参数（realCoreNum/blkFactor/blkTailFactor）。
 *
 * 前置修正（把 NDDMA_CUT_1 的"单轴切分"统一到输入轴视角）：
 *   1. 若输入/输出切的是同一根轴（inCutIndex == perm[outCutIndex]）：
 *      outUbFactor *= inUbFactor —— 一个 block 同时覆盖输入的切分份数与输出的切分份数
 *   2. 若输出切分因子整好切满整个输出轴（outUbFactor == outShape[outCutIndex]）：
 *      实际有效切轴退化为输入轴，outCutIndex 改指 FindOutIndex(inCutIndex)
 *
 * block 总数 = CeilDiv(outShape[outCutIndex], outUbFactor)
 *            × Π outShape[i]（i < outCutIndex 且 perm[i] < inCutIndex 的外层轴）
 *
 * 核利用率不足（outUbAxis < coreNum）时，回调输出切分因子：
 * 从当前因子递减，找"潜在核数占比 ≥ 0.9 且不越界 UB"的最大因子；
 * 找不到则取 rate 最大的合法因子（bestRate/bestI 记录）。
 */

void TransposeNddmaStrategy::CalcInUbShapeInfoForCutOnce()
{
    int64_t outCutIndexExpand = splitInfo_.outCutIndex + TP_NDDMA_MAX_DIM_NUM - shapeInfo_.dim;
    int64_t inCutIndexExpand = splitInfo_.inCutIndex + TP_NDDMA_MAX_DIM_NUM - shapeInfo_.dim;
    for (int64_t i = 0; i < TP_NDDMA_MAX_DIM_NUM; i++) {
        inUbMainDstShape_[i] = expandedOutputShape_[i];
        inUbTailDstShape_[i] = expandedOutputShape_[i];
        if (i == outCutIndexExpand) {
            inUbMainDstShape_[i] = splitInfo_.outUbFactor;
            inUbTailDstShape_[i] = splitInfo_.outTailFactor;
        } else if (i < outCutIndexExpand && expandedPerm_[i] < inCutIndexExpand) {
            inUbMainDstShape_[i] = 1;
            inUbTailDstShape_[i] = 1;
        }
    }
    for (int64_t i = 0; i < TP_NDDMA_MAX_DIM_NUM; i++) {
        inUbMainSrcShape_[expandedPerm_[i]] = inUbMainDstShape_[i];
        inUbTailSrcShape_[expandedPerm_[i]] = inUbTailDstShape_[i];
    }
}

/**
 * @brief 计算 NDDMA_CUT_2 的 4 组 UB 内源/目标 5 维 shape
 *
 * NDDMA_CUT_2 需同时切输入与输出轴，一个 block 的 UB 数据根据"输入/输出
 * 切分轴各自是主块还是尾块"分为 4 种相位，每种相位对应一组 src/dst shape：
 *
 *   inUbMain*         ：输入主块 × 输出主块
 *   inUbInputTail*    ：输入尾块 × 输出主块
 *   inUbOutputTail*   ：输入主块 × 输出尾块
 *   inUbTail*         ：输入尾块 × 输出尾块
 *
 * 规则（以 src 为例）：
 *   - 输入切分轴左侧外层轴：Main/InputTail/Tail = 1（外层循环负责）
 *   - 输入切分轴本身：Main = inUbFactor；InputTail/Tail = inTailFactor
 *   - 输出切分轴（通过 perm 映射到输入侧）：按 outUbFactor/outTailFactor 覆盖对应组
 *   - 其余进入 block 的轴保持完整 expandedInputShape
 * dst 侧 = src 侧按 expandedPerm 重排后得到（NDDMA 搬入后即完成维度重排）。
 */

void TransposeNddmaStrategy::CalcInUbShapeInfoForCutTwice()
{
    // 双切分场景下，对于输入，输入切分轴右侧为UB内的轴；对于输出，输出切分轴右侧非由输入确定的UB内的轴也都为UB内的轴
    for (int64_t idx = 0; idx < TP_NDDMA_MAX_DIM_NUM; idx++) {
        inUbMainSrcShape_[idx] = expandedInputShape_[idx];
        inUbInputTailSrcShape_[idx] = expandedInputShape_[idx];
        inUbOutputTailSrcShape_[idx] = expandedInputShape_[idx];
        inUbTailSrcShape_[idx] = expandedInputShape_[idx];
        if (idx < splitInfo_.inCutIndex + TP_NDDMA_MAX_DIM_NUM - shapeInfo_.dim) {
            inUbMainSrcShape_[idx] = 1;
            inUbInputTailSrcShape_[idx] = 1;
            inUbTailSrcShape_[idx] = 1;
        } else if (idx == splitInfo_.inCutIndex + TP_NDDMA_MAX_DIM_NUM - shapeInfo_.dim) {
            inUbMainSrcShape_[idx] = splitInfo_.inUbFactor;
            inUbInputTailSrcShape_[idx] = splitInfo_.inTailFactor;
            inUbTailSrcShape_[idx] = splitInfo_.inTailFactor;
        }
        inUbOutputTailSrcShape_[idx] = inUbInputTailSrcShape_[idx];
    }
    inUbOutputTailSrcShape_[splitInfo_.inCutIndex + TP_NDDMA_MAX_DIM_NUM - shapeInfo_.dim] = splitInfo_.inUbFactor;
    inUbOutputTailSrcShape_[expandedPerm_[splitInfo_.outCutIndex + TP_NDDMA_MAX_DIM_NUM -
                                          shapeInfo_.dim]] = splitInfo_.outTailFactor;
    inUbMainSrcShape_[expandedPerm_[splitInfo_.outCutIndex + TP_NDDMA_MAX_DIM_NUM - shapeInfo_.dim]] = splitInfo_
                                                                                                           .outUbFactor;
    inUbInputTailSrcShape_[expandedPerm_[splitInfo_.outCutIndex + TP_NDDMA_MAX_DIM_NUM -
                                         shapeInfo_.dim]] = splitInfo_.outUbFactor;
    inUbTailSrcShape_[expandedPerm_[splitInfo_.outCutIndex + TP_NDDMA_MAX_DIM_NUM -
                                    shapeInfo_.dim]] = splitInfo_.outTailFactor;
    for (int64_t idx = splitInfo_.outCutIndex + TP_NDDMA_MAX_DIM_NUM - shapeInfo_.dim + 1; idx < TP_NDDMA_MAX_DIM_NUM;
         idx++) {
        // perm 是排列：满足 perm[j]==inCutIndex 的唯一 j 即 FindOut(inCutIndex) ≤ outCutIndex，
        // 因此 idx>outCutIndex 时该分支不可达（防御性保留）
        // LCOV_EXCL_START
        if (expandedPerm_[idx] == splitInfo_.inCutIndex + TP_NDDMA_MAX_DIM_NUM - shapeInfo_.dim) {
            continue;
        } else {
            // LCOV_EXCL_STOP
            inUbMainSrcShape_[expandedPerm_[idx]] = expandedInputShape_[expandedPerm_[idx]];
            inUbInputTailSrcShape_[expandedPerm_[idx]] = expandedInputShape_[expandedPerm_[idx]];
            inUbOutputTailSrcShape_[expandedPerm_[idx]] = expandedInputShape_[expandedPerm_[idx]];
            inUbTailSrcShape_[expandedPerm_[idx]] = expandedInputShape_[expandedPerm_[idx]];
        }
    }
    for (int64_t idx = 0; idx < TP_NDDMA_MAX_DIM_NUM; idx++) {
        inUbMainDstShape_[idx] = inUbMainSrcShape_[expandedPerm_[idx]];
        inUbInputTailDstShape_[idx] = inUbInputTailSrcShape_[expandedPerm_[idx]];
        inUbOutputTailDstShape_[idx] = inUbOutputTailSrcShape_[expandedPerm_[idx]];
        inUbTailDstShape_[idx] = inUbTailSrcShape_[expandedPerm_[idx]];
    }
}

/**
 * @brief 计算 NDDMA_CUT_2 的 4 种数据区间边界（Main/InputTail/OutputTail/Tail）
 *
 * 双切后所有 block 按「输入切分轴 × 输出切分轴」二维排布，可分为 4 个连续区间。
 * 区间边界以全局 block 索引表示，供 kernel ProcessBlock 与自身核范围求交集。
 *
 * 排布顺序（共 inBlocks × outBlocks × outUbLoop 个 block）：
 *   Main        ：[0,                     inBlocks×outBlocks×outUbLoop-1]
 *   InputTail   ：Main 之后，长度 = outBlocks×outUbLoop（输入轴为尾、输出逐块）
 *   OutputTail  ：InputTail 之后，长度 = inBlocks×outUbLoop（输出轴为尾）
 *   Tail        ：最后 outUbLoop 个（两轴均为尾）
 *
 * 其中：
 *   inBlocks  = expandedInputShape[inCutIdx] / inUbMainSrcShape[inCutIdx]
 *   outBlocks = expandedInputShape[outCutIdx] / inUbMainSrcShape[outCutIdx]
 *   outUbLoop = Π (expandedInputShape[i]/inUbMainSrcShape[i])，i 为非切分轴
 *
 * 若某侧 tailFactor==0，对应区间长度为 0（start/end 保持 0，kernel 跳过）。
 */

void TransposeNddmaStrategy::GetIntervalInfoForCutTwice()
{
    int64_t expandedInputCutIndex = splitInfo_.inCutIndex + TP_NDDMA_MAX_DIM_NUM - shapeInfo_.dim;
    int64_t inputOutputCutIndex = expandedPerm_[splitInfo_.outCutIndex + TP_NDDMA_MAX_DIM_NUM - shapeInfo_.dim];

    int64_t outUbLoop = 1;
    for (int64_t i = TP_NDDMA_MAX_DIM_NUM - 1; i >= 0; i--) {
        if (i != expandedInputCutIndex && i != inputOutputCutIndex) {
            outUbLoop = outUbLoop * (expandedInputShape_[i] / inUbMainSrcShape_[i]);
        }
    }

    offsetRangeMain_.end = (expandedInputShape_[expandedInputCutIndex] / inUbMainSrcShape_[expandedInputCutIndex]) *
                               (expandedInputShape_[inputOutputCutIndex] / inUbMainSrcShape_[inputOutputCutIndex]) *
                               outUbLoop -
                           1;

    if (splitInfo_.inTailFactor != 0 && splitInfo_.outTailFactor != 0) {
        offsetRangeInputTail_.start = offsetRangeMain_.end + 1;
        offsetRangeInputTail_.end = offsetRangeInputTail_.start +
                                    (expandedInputShape_[inputOutputCutIndex] /
                                     inUbMainSrcShape_[inputOutputCutIndex]) *
                                        outUbLoop -
                                    1;
        offsetRangeOutputTail_.start = offsetRangeInputTail_.end + 1;
        offsetRangeOutputTail_.end = offsetRangeOutputTail_.start +
                                     (expandedInputShape_[expandedInputCutIndex] /
                                      inUbMainSrcShape_[expandedInputCutIndex]) *
                                         outUbLoop -
                                     1;
        offsetRangeTail_.start = offsetRangeOutputTail_.end + 1;
        offsetRangeTail_.end = offsetRangeTail_.start + outUbLoop - 1;
    }

    if (splitInfo_.inTailFactor != 0 && splitInfo_.outTailFactor == 0) {
        offsetRangeInputTail_.start = offsetRangeMain_.end + 1;
        offsetRangeInputTail_.end = offsetRangeInputTail_.start +
                                    (expandedInputShape_[inputOutputCutIndex] /
                                     inUbMainSrcShape_[inputOutputCutIndex]) *
                                        outUbLoop -
                                    1;
    }

    if (splitInfo_.inTailFactor == 0 && splitInfo_.outTailFactor != 0) {
        offsetRangeOutputTail_.start = offsetRangeMain_.end + 1;
        offsetRangeOutputTail_.end = offsetRangeOutputTail_.start +
                                     (expandedInputShape_[expandedInputCutIndex] /
                                      inUbMainSrcShape_[expandedInputCutIndex]) *
                                         outUbLoop -
                                     1;
    }
}

bool TransposeNddmaStrategy::SetTilingData()
{
    OP_LOGD(context_->GetNodeName(), "Entering SetTilingData.");
    if (tilingKey_ == TK_NDDMA_CUT_1) {
        auto* tilingData = context_->GetTilingData<Transpose::TransposeNddmaCut1TilingData>();
        if (tilingData == nullptr) {
            OP_LOGE(context_->GetNodeName(), "get cut once tiling data failed, capacity is not enough!");
            return false;
        }
        *tilingData = Transpose::TransposeNddmaCut1TilingData{};
        tilingData->permSize = static_cast<int8_t>(shapeInfo_.dim);
        tilingData->outCutIndex = static_cast<int8_t>(splitInfo_.outCutIndex);
        tilingData->outUbFactor = static_cast<int32_t>(splitInfo_.outUbFactor);
        tilingData->outTailFactor = static_cast<int32_t>(splitInfo_.outTailFactor);
        tilingData->blkFactor = blkFactor_;
        tilingData->blkTailFactor = blkTailFactor_;
        tilingData->ubSize = static_cast<int32_t>(ubSize_);
        for (int64_t i = 0; i < TP_NDDMA_MAX_DIM_NUM; i++) {
            tilingData->expandedPerm[i] = static_cast<int8_t>(expandedPerm_[i]);
        }
        std::copy(expandedInputShape_, expandedInputShape_ + TP_NDDMA_MAX_DIM_NUM, tilingData->expandedInputShape);
        std::copy(expandedOutputShape_, expandedOutputShape_ + TP_NDDMA_MAX_DIM_NUM, tilingData->expandedOutputShape);
        std::copy(inUbMainSrcShape_, inUbMainSrcShape_ + TP_NDDMA_MAX_DIM_NUM, tilingData->inUbMainSrcShape);
        std::copy(inUbMainDstShape_, inUbMainDstShape_ + TP_NDDMA_MAX_DIM_NUM, tilingData->inUbMainDstShape);
        std::copy(inUbTailSrcShape_, inUbTailSrcShape_ + TP_NDDMA_MAX_DIM_NUM, tilingData->inUbTailSrcShape);
        std::copy(inUbTailDstShape_, inUbTailDstShape_ + TP_NDDMA_MAX_DIM_NUM, tilingData->inUbTailDstShape);
        return true;
    }
    if (tilingKey_ == TK_NDDMA_CUT_2) {
        auto* tilingData = context_->GetTilingData<Transpose::TransposeNddmaCut2TilingData>();
        if (tilingData == nullptr) {
            OP_LOGE(context_->GetNodeName(), "get cut twice tiling data failed, capacity is not enough!");
            return false;
        }
        *tilingData = Transpose::TransposeNddmaCut2TilingData{};
        tilingData->permSize = static_cast<int8_t>(shapeInfo_.dim);
        tilingData->inCutIndex = static_cast<int8_t>(splitInfo_.inCutIndex);
        tilingData->outCutIndex = static_cast<int8_t>(splitInfo_.outCutIndex);
        tilingData->inUbFactor = static_cast<int32_t>(splitInfo_.inUbFactor);
        tilingData->outUbFactor = static_cast<int32_t>(splitInfo_.outUbFactor);
        tilingData->inTailFactor = static_cast<int32_t>(splitInfo_.inTailFactor);
        tilingData->outTailFactor = static_cast<int32_t>(splitInfo_.outTailFactor);
        tilingData->blkFactor = blkFactor_;
        tilingData->blkTailFactor = blkTailFactor_;
        tilingData->ubSize = static_cast<int32_t>(ubSize_);
        tilingData->rangeMainEnd = offsetRangeMain_.end;
        tilingData->rangeInputTailStart = offsetRangeInputTail_.start;
        tilingData->rangeInputTailEnd = offsetRangeInputTail_.end;
        tilingData->rangeOutputTailStart = offsetRangeOutputTail_.start;
        tilingData->rangeOutputTailEnd = offsetRangeOutputTail_.end;
        tilingData->rangeTailStart = offsetRangeTail_.start;
        tilingData->rangeTailEnd = offsetRangeTail_.end;
        for (int64_t i = 0; i < TP_NDDMA_MAX_DIM_NUM; i++) {
            tilingData->expandedPerm[i] = static_cast<int8_t>(expandedPerm_[i]);
        }
        std::copy(expandedInputShape_, expandedInputShape_ + TP_NDDMA_MAX_DIM_NUM, tilingData->expandedInputShape);
        std::copy(expandedOutputShape_, expandedOutputShape_ + TP_NDDMA_MAX_DIM_NUM, tilingData->expandedOutputShape);
        std::copy(inUbMainSrcShape_, inUbMainSrcShape_ + TP_NDDMA_MAX_DIM_NUM, tilingData->inUbMainSrcShape);
        std::copy(inUbMainDstShape_, inUbMainDstShape_ + TP_NDDMA_MAX_DIM_NUM, tilingData->inUbMainDstShape);
        std::copy(inUbInputTailSrcShape_, inUbInputTailSrcShape_ + TP_NDDMA_MAX_DIM_NUM,
                  tilingData->inUbInputTailSrcShape);
        std::copy(inUbInputTailDstShape_, inUbInputTailDstShape_ + TP_NDDMA_MAX_DIM_NUM,
                  tilingData->inUbInputTailDstShape);
        std::copy(inUbOutputTailSrcShape_, inUbOutputTailSrcShape_ + TP_NDDMA_MAX_DIM_NUM,
                  tilingData->inUbOutputTailSrcShape);
        std::copy(inUbOutputTailDstShape_, inUbOutputTailDstShape_ + TP_NDDMA_MAX_DIM_NUM,
                  tilingData->inUbOutputTailDstShape);
        std::copy(inUbTailSrcShape_, inUbTailSrcShape_ + TP_NDDMA_MAX_DIM_NUM, tilingData->inUbTailSrcShape);
        std::copy(inUbTailDstShape_, inUbTailDstShape_ + TP_NDDMA_MAX_DIM_NUM, tilingData->inUbTailDstShape);
        return true;
    }
    // DoSplitUB 之后 tilingKey 必为 10002/10003，本策略不产生其他分支
    // LCOV_EXCL_START
    OP_LOGE(context_->GetNodeName(), "unexpected tiling key %lu for nddma strategy", tilingKey_);
    return false;
    // LCOV_EXCL_STOP
}

void TransposeNddmaStrategy::LogTilingData()
{
    OP_LOGI(context_->GetNodeName(), "Entering LogTilingData.");
    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        OP_LOGI(context_->GetNodeName(), "reducedInShape[%ld] is:%ld, reducedOutShape[%ld]:%ld, reducedPerm[%ld]:%ld.",
                i, shapeInfo_.reducedInShape[i], i, shapeInfo_.reducedOutShape[i], i, shapeInfo_.reducedPerm[i]);
    }
    OP_LOGI(context_->GetNodeName(),
            "tilingData is permSize:%ld, inCutIndex:%ld, outCutIndex:%ld, inUbFactor:%ld, outUbFactor:%ld, "
            "inTailFactor:%ld, outTailFactor:%ld, realCoreNum:%ld, blkFactor:%ld, blkTailFactor:%ld, ubSize:%ld, "
            "Tiling4Transpose ends. ",
            shapeInfo_.dim, splitInfo_.inCutIndex, splitInfo_.outCutIndex, splitInfo_.inUbFactor,
            splitInfo_.outUbFactor, splitInfo_.inTailFactor, splitInfo_.outTailFactor, realCoreNum_, blkFactor_,
            blkTailFactor_, ubSize_);
}

} // namespace optiling
