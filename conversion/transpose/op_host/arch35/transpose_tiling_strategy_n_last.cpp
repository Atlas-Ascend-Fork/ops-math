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
 * \file transpose_tiling_strategy_n_last.cpp
 * \brief N_LAST(10004) 策略类实现
 *
 * 实现尾轴不转置路径的 UB 切分、分核因子回溯与退轴处理。
 */

#include "transpose_tiling_strategy_n_last.h"

namespace optiling {
namespace {
// N_LAST 小 shape 体积阈值
static constexpr int64_t N_LAST_SMALL_SHAPE_BYTES_THRES_HOLD = 4000000;
static constexpr int64_t N_LAST_SMALL_SHAPE_BYTES_THRES_HOLD_DAV_5102 = 400000;
static constexpr uint64_t N_LAST_BUFFER_NUM = 2;               ///< 双缓冲
static constexpr double N_LAST_VEC_CORE_USED_THRES_HOLD = 0.9; ///< 核利用率阈值
static constexpr int64_t N_LAST_MOVEALIGN_LAST_MIN_ELE = 32;   ///< 尾轴最小元素数
static constexpr int64_t N_LAST_DIM_TWO = 2;
} // namespace

/**
 * @brief 准入检查：尾轴不转置 && 尾轴元素数 >= 32 && 体积 >= 平台阈值
 *
 */
bool TransposeNLastStrategy::IsSupport() const
{
    if (shapeInfo_.isLastAxisTranspose ||
        shapeInfo_.reducedInShape[shapeInfo_.dim - 1] < N_LAST_MOVEALIGN_LAST_MIN_ELE) {
        return false;
    }
    const int64_t totalBytes = shapeInfo_.totalVolumeActual * shapeInfo_.eleLenInBytes;
    const int64_t threshold = (npuArch_ == Ops::Base::DAV_5102) ? N_LAST_SMALL_SHAPE_BYTES_THRES_HOLD_DAV_5102 :
                                                                  N_LAST_SMALL_SHAPE_BYTES_THRES_HOLD;
    return (totalBytes >= threshold);
}

TilingStatus TransposeNLastStrategy::DoExecute()
{
    tilingKey_ = TK_N_LAST;
    CalcUBSplitInfo();
    CalcBlockSplitInfo();
    if (shapeInfo_.dim <= TP_NDDMA_MAX_DIM_NUM) {
        NDDMADimExpand();
    }
    return TilingStatus::SUCCESS;
}

void TransposeNLastStrategy::CalcBlockSplitInfo() { CalcBlockSplitInfoForNLastTranspose(); }

void TransposeNLastStrategy::NDDMADimExpand()
{
    int64_t offset = (shapeInfo_.dim < TP_NDDMA_MAX_DIM_NUM) ? (TP_NDDMA_MAX_DIM_NUM - shapeInfo_.dim) : 0;
    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        expandedPerm_[i + offset] = shapeInfo_.reducedPerm[i] + offset;
        expandedInputShape_[i + offset] = shapeInfo_.reducedInShape[i];
        expandedOutputShape_[i + offset] = shapeInfo_.reducedOutShape[i];
    }
}

/**
 * @brief N_LAST UB 切分：双缓冲，ubElement = ubSize/2/eleBytes
 *
 * 一半 UB 用于 CopyIn，另一半用于 CopyOut。N_LAST_BUFFER_NUM=2。
 */
void TransposeNLastStrategy::CalcUBSplitInfo() { ubElement_ = ubSize_ / N_LAST_BUFFER_NUM / shapeInfo_.eleLenInBytes; }
/**
 * @brief N_LAST（尾轴不转置）分核逻辑
 *
 * 尾轴不转置时最后维在输入/输出中保持连续，可按"连续行搬移"分块。
 * 从输入最末轴向前做 UB 预算（inUbElement = ubSize/2/eleBytes，双缓冲）：
 *   - 预算放不下该轴（inUbElement < curDim）：
 *       · 若切后块数已足够 ≥coreNum → FindSplitFactorByMultiplesNLast（保整数倍负载均衡）
 *       · 否则 → FindSplitFactorByRateNLast（按 0.9 核利用率阈值回溯因子）
 *       · 若 inUbFactor 无效（0）→ CheckInUbFactorValid 退轴到上一层
 *   - 预算放得下：整根吞掉，预算除以该轴长度，记录 solvedTotalElment 便于退轴恢复
 *
 * 最终分核参数 = SetRealCoreNumAndBlkFactor(CeilDiv(inCutAxis, inUbFactor) × remaining)
 */
void TransposeNLastStrategy::CalcBlockSplitInfoForNLastTranspose()
{
    OP_LOGD(context_->GetNodeName(), "Entering CalcBlockSplitInfoForNLastTranspose.");
    inUbElement_ = ubElement_;
    int64_t remainingTotalElment = shapeInfo_.totalVolumeActual;
    int64_t solvedTotalElment[TP_MAX_AXIS_NUM];
    solvedTotalElment[shapeInfo_.dim - 1] = 1;
    int64_t currentSplitIndex;
    int64_t currentInShapeDim;
    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        currentSplitIndex = shapeInfo_.dim - 1 - i;
        currentInShapeDim = shapeInfo_.reducedInShape[currentSplitIndex];
        remainingTotalElment /= currentInShapeDim;
        int64_t coreNumTmp = remainingTotalElment * Ops::Base::CeilDiv(currentInShapeDim, inUbElement_);
        if (inUbElement_ < currentInShapeDim) {
            if (coreNumTmp < coreNum_) { // use at least N_LAST_VEC_CORE_USED_THRES_HOLD * coreNum
                FindSplitFactorByRateNLast(currentSplitIndex, currentInShapeDim, remainingTotalElment);
                break;
            } else { // use full coreNum
                int64_t coreNumMultiples = Ops::Base::FloorDiv(coreNumTmp, coreNum_);
                FindSplitFactorByMultiplesNLast(currentSplitIndex, currentInShapeDim, remainingTotalElment,
                                                coreNumMultiples);
                /* 检查inUbFactor是否合法，不合法则执行退轴逻辑 */
                CheckInUbFactorValid(currentSplitIndex, currentInShapeDim, remainingTotalElment, coreNumMultiples,
                                     solvedTotalElment);
                break;
            }
        } else if (coreNumTmp < coreNum_) { // use at least N_LAST_VEC_CORE_USED_THRES_HOLD * coreNum
            FindSplitFactorByRateNLast(currentSplitIndex, currentInShapeDim, remainingTotalElment);
            break;
        } else {
            inUbElement_ /= currentInShapeDim;
            solvedTotalElment[currentSplitIndex] = currentSplitIndex == shapeInfo_.dim - 1 ?
                                                       currentInShapeDim :
                                                       (currentInShapeDim * solvedTotalElment[currentSplitIndex + 1]);
        }
    }
    int64_t coreNum = Ops::Base::CeilDiv(shapeInfo_.reducedInShape[inCutIndex_], inUbFactor_) * remainingTotalElment;
    SetRealCoreNumAndBlkFactor(coreNum);
}
/**
 * @brief N_LAST UB 越界校验
 *
 * 校验方法：
 * 校验"当前候选切分轴 currentSplitIndex 取 currentSplitValue 时，进 UB 的元素数
 * 是否超过预算 ubElement_"。
 */
bool TransposeNLastStrategy::UbOutOfBoundCheckNLast(int64_t currentSplitIndex, int64_t currentSplitValue)
{
    int64_t burstLenBlockAlign = 1;
    if (currentSplitIndex == shapeInfo_.dim - 1) {
        burstLenBlockAlign = currentSplitValue;
    } else {
        burstLenBlockAlign = shapeInfo_.reducedInShape[shapeInfo_.dim - 1];
    }
    burstLenBlockAlign = Ops::Base::CeilAlign(burstLenBlockAlign * shapeInfo_.eleLenInBytes, ubBlockSize_) /
                         shapeInfo_.eleLenInBytes;
    int64_t inUbElements = burstLenBlockAlign;
    for (int64_t i = currentSplitIndex; i < shapeInfo_.dim - 1; i++) {
        if (i == currentSplitIndex) {
            inUbElements *= currentSplitValue;
        } else {
            inUbElements *= shapeInfo_.reducedInShape[i];
        }
    }
    if (inUbElements > ubElement_) {
        return true;
    }
    return false;
}
/**
 * @brief N_LAST 场景：按核利用率阈值回溯输入切分因子（Rate 版）
 *
 * 回溯规则：
 * 从预算上限 inUbElement_ 向下递减 i，寻找满足：
 *   - 潜在核数 coreNumNew = remainingTotalElment * CeilDiv(curDim, i) 与 coreNum 的
 *     比值 rate >= N_LAST_VEC_CORE_USED_THRES_HOLD（0.9，即至少用满90%核）
 *   - 且按该因子切后数据不越界 UB（UbOutOfBoundCheckNLast）
 * 的最大因子。
 */
void TransposeNLastStrategy::FindSplitFactorByRateNLast(int64_t currentSplitIndex, int64_t currentInShapeDim,
                                                        int64_t remainingTotalElment)
{
    inCutIndex_ = currentSplitIndex;
    inUbFactor_ = 1;
    inTailFactor_ = 0;
    for (int64_t i = inUbElement_; i >= N_LAST_DIM_TWO; i--) {
        int64_t coreNumNew = remainingTotalElment * Ops::Base::CeilDiv(currentInShapeDim, i);
        double rate = static_cast<double>(coreNumNew) / coreNum_;
        if ((rate >= N_LAST_VEC_CORE_USED_THRES_HOLD) && !UbOutOfBoundCheckNLast(currentSplitIndex, i)) {
            inUbFactor_ = i;
            inTailFactor_ = currentInShapeDim % i;
            inUbActual_ *= i;
            break;
        }
    }
}
/**
 * @brief N_LAST 场景：分核因子回溯（Multiples 版，输入侧因子求值）
 *
 * 回溯规则：
 * 在先满足"块数 ≥ coreNum"的前提下，进一步让块数保持为 coreNum 的整数倍（负载均衡）。
 */
void TransposeNLastStrategy::FindSplitFactorByMultiplesNLast(int64_t currentSplitIndex, int64_t currentInShapeDim,
                                                             int64_t remainingTotalElment, int64_t coreNumMultiples)
{
    inCutIndex_ = currentSplitIndex;
    for (int64_t i = inUbElement_; i >= 1; i--) {
        if (!UbOutOfBoundCheckNLast(currentSplitIndex, i)) {
            inUbFactor_ = i;
            inTailFactor_ = currentInShapeDim % i;
            inUbActual_ *= i;
            break;
        }
    }
    int64_t coreNumTmp = remainingTotalElment * Ops::Base::CeilDiv(currentInShapeDim, inUbFactor_);
    coreNumMultiples = Ops::Base::FloorDiv(coreNumTmp, coreNum_);
    for (int64_t i = 1; i < inUbFactor_; i++) {
        int64_t coreNumNew = remainingTotalElment * Ops::Base::CeilDiv(currentInShapeDim, i);
        if ((Ops::Base::FloorDiv(coreNumNew, coreNum_) == coreNumMultiples) &&
            !UbOutOfBoundCheckNLast(currentSplitIndex, i)) {
            inUbFactor_ = i;
            inTailFactor_ = currentInShapeDim % i;
            inUbActual_ = inUbActual_ / inUbElement_ * i;
            break;
        }
    }
}
/**
 * @brief N_LAST 退轴逻辑（inUbFactor 不可用时回退到上一根轴重切）
 *
 * 处理流程：
 * 当在某根轴上切出的 inUbFactor==0（预算在该轴耗尽或因子查找失败）且
 * 尚未切到最末轴时，把切分位置**上移一层**（切更靠末的轴），并恢复该层
 * 的原始预算，重新调用 FindSplitFactorByMultiplesNLast 求因子，
 * 直到找到合法 inUbFactor 或穷尽。
 *
 * 该路径为策略内部参数回退：在当前策略内重新求因子，
 * 不改变返回状态、不切换后续策略。
 */
void TransposeNLastStrategy::CheckInUbFactorValid(int64_t& currentSplitIndex, int64_t& currentInShapeDim,
                                                  int64_t& remainingTotalElment, int64_t& coreNumMultiples,
                                                  int64_t* solvedTotalElment)
{
    if (inUbFactor_ == 0 && currentSplitIndex < shapeInfo_.dim - 1) {
        while (currentSplitIndex < shapeInfo_.dim - 1) {
            currentSplitIndex++;
            currentInShapeDim = shapeInfo_.reducedInShape[currentSplitIndex];
            inUbElement_ = shapeInfo_.reducedInShape[currentSplitIndex];
            remainingTotalElment = shapeInfo_.totalVolumeActual / solvedTotalElment[currentSplitIndex];
            coreNumMultiples = remainingTotalElment;
            FindSplitFactorByMultiplesNLast(currentSplitIndex, currentInShapeDim, remainingTotalElment,
                                            coreNumMultiples);
            if (inUbFactor_ > 0) {
                break;
            }
        }
    }
}
void TransposeNLastStrategy::SetRealCoreNumAndBlkFactor(int64_t coreNum)
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
 * @brief 填写 N_LAST 消费的注册 TilingData 字段
 *
 * kernel（transpose_n_last.h）消费：ubSize / realCoreNum / blkFactor / blkTailFactor /
 * inCutIndex / inTailFactor / inUbFactor / permSize + inputShape/perm。
 */
bool TransposeNLastStrategy::SetTilingData()
{
    OP_LOGD(context_->GetNodeName(), "Entering SetTilingData.");
    TransposeOpTilingData* tilingData = GetTransposeOpTilingDataBuffer(context_);
    if (tilingData == nullptr) {
        return false;
    }
    *tilingData = TransposeOpTilingData{};
    tilingData->permSize = shapeInfo_.dim;
    tilingData->inCutIndex = inCutIndex_;
    tilingData->inTailFactor = inTailFactor_;
    tilingData->inUbFactor = inUbFactor_;
    tilingData->realCoreNum = realCoreNum_;
    tilingData->blkFactor = blkFactor_;
    tilingData->blkTailFactor = blkTailFactor_;
    tilingData->ubSize = ubSize_;
    tilingData->bigDimBufferNum = 1;
    tilingData->totalNddmaNum = 1;
    tilingData->nddmaIdx[0] = -1;
    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        tilingData->inputShape[i] = shapeInfo_.reducedInShape[i];
        tilingData->outputShape[i] = shapeInfo_.reducedOutShape[i];
        tilingData->perm[i] = shapeInfo_.reducedPerm[i];
    }
    std::copy(expandedPerm_, expandedPerm_ + TP_NDDMA_MAX_DIM_NUM, tilingData->expandedPerm);
    std::copy(expandedInputShape_, expandedInputShape_ + TP_NDDMA_MAX_DIM_NUM, tilingData->expandedInputShape);
    std::copy(expandedOutputShape_, expandedOutputShape_ + TP_NDDMA_MAX_DIM_NUM, tilingData->expandedOutputShape);
    return true;
}

void TransposeNLastStrategy::LogTilingData()
{
    OP_LOGI(context_->GetNodeName(), "Entering LogTilingData.");
    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        OP_LOGI(context_->GetNodeName(), "reducedInShape[%ld] is:%ld, reducedOutShape[%ld]:%ld, reducedPerm[%ld]:%ld.",
                i, shapeInfo_.reducedInShape[i], i, shapeInfo_.reducedOutShape[i], i, shapeInfo_.reducedPerm[i]);
    }
    OP_LOGI(context_->GetNodeName(),
            "tilingData is permSize:%ld, inCutIndex:%ld, inUbFactor:%ld, inTailFactor:%ld, realCoreNum:%ld, "
            "blkFactor:%ld, blkTailFactor:%ld, ubSize:%ld, Tiling4Transpose ends. ",
            shapeInfo_.dim, inCutIndex_, inUbFactor_, inTailFactor_, realCoreNum_, blkFactor_, blkTailFactor_, ubSize_);
}

} // namespace optiling
