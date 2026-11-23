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
 * \file transpose_tiling_strategy_nddma_big_dim.cpp
 * \brief NDDMA_BIG_DIM(10005) 策略类实现
 *
 * 实现 dim>5 场景的 UB 切分、分核与 5 维压缩映射。
 */

#include "transpose_tiling_strategy_nddma_big_dim.h"

namespace optiling {
namespace {
static constexpr double BIG_DIM_VEC_CORE_USED_THRES_HOLD = 0.9; ///< 核利用率阈值
static constexpr int64_t BIG_DIM_BUFFER_NUM_SINGLE_BUF = 1;     ///< 单缓冲
} // namespace

bool TransposeNddmaBigDimStrategy::IsSupport() const { return shapeInfo_.dim > TP_NDDMA_MAX_DIM_NUM; }

/**
 * @brief 执行 NDDMA_BIG_DIM tiling
 *
 * dim>5 不做 5 维扩展（避免数组越界），expanded* 字段保持 TilingData 默认值。
 * 双缓冲 fallback 为策略内部回退。
 */
TilingStatus TransposeNddmaBigDimStrategy::DoExecute()
{
    CalcUBSplitInfo();
    CalcBlockSplitInfoForBigDim();
    return TilingStatus::SUCCESS;
}

/**
 * @brief NDDMA_BIG_DIM UB 切分
 *
 * UB 预算：
 *   bigDimBufferNum_ = GetNddmaBufferNum()（默认按平台架构：DAV_9201/9202 为 2，其他 1）；
 *   ubElement = ubSize_ / bigDimBufferNum_ / eleLenInBytes；
 *   DoSplitUBBigDim 只切输出侧一根轴；
 *   双缓冲不满足（bigDimBufferNum_>1 && reducedPerm[outCutIndex]==(dim-1)）时
 *   fallback 单缓冲重算（策略内回退）。
 */
void TransposeNddmaBigDimStrategy::CalcUBSplitInfo()
{
    bigDimBufferNum_ = GetNddmaBufferNum();
    ubElement_ = ubSize_ / bigDimBufferNum_ / shapeInfo_.eleLenInBytes;
    DoSplitUBBigDim();
    if (bigDimBufferNum_ > BIG_DIM_BUFFER_NUM_SINGLE_BUF &&
        shapeInfo_.reducedPerm[outCutIndex_] == (shapeInfo_.dim - 1)) {
        OP_LOGD(context_->GetNodeName(), "[TK_NDDMA_BIG_DIM] fallback to single buffer");
        bigDimBufferNum_ = BIG_DIM_BUFFER_NUM_SINGLE_BUF;
        ubElement_ = ubSize_ / bigDimBufferNum_ / shapeInfo_.eleLenInBytes;
        DoSplitUBBigDim();
    }
}
/**
 * @brief NDDMA_BIG_DIM（>5维）场景的切轴逻辑
 *
 * 超过 5 维时无法直接使用 NDDMA 5 维格式，切轴简化为只切一根输出轴：
 * 从输出最末轴向前扫描，遇到以下任一条件即确定 outCutIndex：
 *   1. UB 预算放不下该轴（ubElement < outShape[i]）：直接切分
 *   2. 吞掉该轴后累计元素数 ≤ coreNum（剩余维度已足够铺满核，无需再切更外层）
 *   3. 已经吞了 TP_NDDMA_MAX_DIM_NUM-1=4 根轴（只剩最后一根可切的空间）
 */
void TransposeNddmaBigDimStrategy::DoSplitUBBigDim()
{
    OP_LOGD(context_->GetNodeName(), "Entering DoSplitUBBigDim.");
    int64_t dimSize = TP_NDDMA_MAX_DIM_NUM - 1;
    int64_t totalElment = shapeInfo_.totalVolumeActual;
    // search split index and calc base number
    for (int64_t i = shapeInfo_.dim - 1; i >= 0; i--) {
        if (ubElement_ < shapeInfo_.reducedOutShape[i]) {
            outCutIndex_ = i;
            outUbFactor_ = ubElement_;
            outTailFactor_ = shapeInfo_.reducedOutShape[i] % outUbFactor_;
            break;
        } else if (dimSize > 0) {
            totalElment = totalElment / shapeInfo_.reducedOutShape[i];
            if (totalElment <= coreNum_) {
                outCutIndex_ = i;
                outUbFactor_ = ubElement_ <= shapeInfo_.reducedOutShape[i] ? ubElement_ : shapeInfo_.reducedOutShape[i];
                outTailFactor_ = shapeInfo_.reducedOutShape[i] % outUbFactor_;
                break;
            }
            ubElement_ = ubElement_ / shapeInfo_.reducedOutShape[i];
            dimSize--;
        } else if (dimSize == 0) {
            outCutIndex_ = i;
            outUbFactor_ = ubElement_ <= shapeInfo_.reducedOutShape[i] ? ubElement_ : shapeInfo_.reducedOutShape[i];
            outTailFactor_ = shapeInfo_.reducedOutShape[i] % outUbFactor_;
            break;
        }
    }
}
/**
 * @brief NDDMA_BIG_DIM（>5维）分核逻辑
 *
 * 只回调输出切分因子 outUbFactor 来调节并行度：
 *   - coreNum < coreNum_（块不足）：递减 outUbFactor 找 rate>=0.9 的最大因子
 *     （无越界检查，NDDMA_BIG_DIM 无 UB 内切轴）
 *   - block 数充足：找使 FloorDiv(块数,coreNum) 保持相同倍数的更细因子（负载均衡）
 * 最后 FlushBaseNumForBigDim 生成 5 维压缩映射并 SetRealCoreNumAndBlkFactor。
 */
void TransposeNddmaBigDimStrategy::CalcBlockSplitInfoForBigDim()
{
    OP_LOGD(context_->GetNodeName(), "Entering CalcBlockSplitInfoForBigDim.");
    int64_t outUbAxisExceptSplitAxis = 1;
    for (int64_t i = 0; i < outCutIndex_; i++) {
        outUbAxisExceptSplitAxis *= shapeInfo_.reducedOutShape[i];
    }
    int64_t coreNum = Ops::Base::CeilDiv(shapeInfo_.reducedOutShape[outCutIndex_], outUbFactor_) *
                      outUbAxisExceptSplitAxis;
    int64_t currentShapeDim = shapeInfo_.reducedOutShape[outCutIndex_];
    if (coreNum < coreNum_) {
        // use at least BIG_DIM_VEC_CORE_USED_THRES_HOLD * coreNum
        int64_t bestI = 1;
        double bestRate = 0.0;
        bool foundValidFactor = false;
        for (int64_t i = outUbFactor_; i >= 1; i--) {
            int64_t coreNumNew = Ops::Base::CeilDiv(currentShapeDim, i) * outUbAxisExceptSplitAxis;
            double rate = static_cast<double>(coreNumNew) / coreNum_;
            if ((rate >= BIG_DIM_VEC_CORE_USED_THRES_HOLD)) {
                outUbFactor_ = i;
                outTailFactor_ = currentShapeDim % i;
                foundValidFactor = true;
                break;
            }
            if (rate > bestRate) {
                bestRate = rate;
                bestI = i;
            }
        }
        if (!foundValidFactor) {
            outUbFactor_ = bestI;
            outTailFactor_ = currentShapeDim % bestI;
        }
    } else {
        // use full coreNum：i==outUbFactor 时块数必然等于 coreNum，倍数条件必定命中，
        // 因此不存在需要回退的候选因子，直接取首个同道倍数因子。
        int64_t coreNumMultiples = Ops::Base::FloorDiv(coreNum, coreNum_);
        for (int64_t i = 1; i <= outUbFactor_; i++) {
            int64_t coreNumNew = outUbAxisExceptSplitAxis * Ops::Base::CeilDiv(currentShapeDim, i);
            if ((Ops::Base::FloorDiv(coreNumNew, coreNum_) == coreNumMultiples)) {
                outUbFactor_ = i;
                outTailFactor_ = currentShapeDim % i;
                break;
            }
        }
    }
    FlushBaseNumForBigDim();
    coreNum = Ops::Base::CeilDiv(shapeInfo_.reducedOutShape[outCutIndex_], outUbFactor_) * outUbAxisExceptSplitAxis;
    SetRealCoreNumAndBlkFactor(coreNum);
}
/**
 * @brief NDDMA_BIG_DIM 场景：将 >5 维原始 shape 压缩为 NDDMA 5 维表示
 *
 * 生成 kernel 端 NDDMA SetLoopInfo 所需的三个数组：
 *   - nddmaIdx[i]：5 维压缩索引 i 对应的原始输入轴索引（按输出 perm 顺序从高维往低维排）
 *   - baseNddmaShape[i]：压缩后第 i 维的循环基数（该轴在 block 循环中的步长）
 *   - baseInShape_[k]：原始第 k 轴的输入地址基数（右侧轴的乘积）
 * 最后按输出维度顺序 sort 后回填 baseNddmaShape_（kernel 按 nddmaIdx 查找对应基数）。
 */
void TransposeNddmaBigDimStrategy::FlushBaseNumForBigDim()
{
    OP_LOGD(context_->GetNodeName(), "Entering FlushBaseNumForBigDim.");
    int64_t idxNum = TP_NDDMA_MAX_DIM_NUM - 1;
    int64_t baseInNum = 1;
    int64_t tmpNddmaShape[TP_NDDMA_MAX_DIM_NUM] = {0};
    int64_t oriNddmaIdx[TP_NDDMA_MAX_DIM_NUM] = {-1};
    for (int64_t i = shapeInfo_.dim - 1; i >= 0; i--) {
        baseInShape_[i] = baseInNum;
        baseInNum *= shapeInfo_.reducedInShape[i];
        if (i > outCutIndex_) {
            nddmaIdx_[idxNum] = shapeInfo_.reducedPerm[i];
            oriNddmaIdx[idxNum] = shapeInfo_.reducedPerm[i];
            tmpNddmaShape[idxNum] = totalNddmaNum_;
            totalNddmaNum_ *= shapeInfo_.reducedOutShape[i];
            idxNum--;
        } else if (i == outCutIndex_) {
            nddmaIdx_[idxNum] = shapeInfo_.reducedPerm[i];
            oriNddmaIdx[idxNum] = shapeInfo_.reducedPerm[i];
            tmpNddmaShape[idxNum] = totalNddmaNum_;
            totalNddmaNum_ *= outUbFactor_;
            idxNum--;
        } else if (idxNum >= 0) {
            nddmaIdx_[idxNum] = shapeInfo_.reducedPerm[i];
            oriNddmaIdx[idxNum] = shapeInfo_.reducedPerm[i];
            tmpNddmaShape[idxNum] = totalNddmaNum_;
            idxNum--;
        }
    }

    std::sort(std::begin(nddmaIdx_), std::end(nddmaIdx_));
    for (int64_t i = 0; i < TP_NDDMA_MAX_DIM_NUM; i++) {
        for (int64_t j = 0; j < TP_NDDMA_MAX_DIM_NUM; j++) {
            if (nddmaIdx_[i] == oriNddmaIdx[j]) {
                baseNddmaShape_[i] = tmpNddmaShape[j];
            }
        }
    }
}
void TransposeNddmaBigDimStrategy::SetRealCoreNumAndBlkFactor(int64_t coreNum)
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

/**
 * @brief 填写 NDDMA_BIG_DIM 消费的紧凑 TilingData 字段
 *
 * kernel（transpose_nddma_big_dim.h）消费：ubBytes / bufferNum / realCoreNum / blkFactor /
 * blkTailFactor / outCutIndex / outUbFactor / outTailFactor / permSize /
 * inputShape/outputShape/perm / baseInShape / baseNddmaShape / nddmaIdx / totalNddmaNum。
 */
bool TransposeNddmaBigDimStrategy::SetTilingData()
{
    OP_LOGD(context_->GetNodeName(), "Entering SetTilingData.");
    auto* data = context_->GetTilingData<TransposeNddmaBigDimTilingData>();
    if (data == nullptr) {
        OP_LOGE(context_->GetNodeName(), "get tiling data failed, capacity is not enough!");
        return false;
    }
    *data = TransposeNddmaBigDimTilingData{};
    data->permSize = static_cast<int8_t>(shapeInfo_.dim);
    data->outCutIndex = static_cast<int8_t>(outCutIndex_);
    data->outUbFactor = static_cast<uint32_t>(outUbFactor_);
    data->outTailFactor = static_cast<uint32_t>(outTailFactor_);
    data->blkFactor = blkFactor_;
    data->blkTailFactor = blkTailFactor_;
    data->ubBytes = static_cast<uint32_t>(ubSize_ / bigDimBufferNum_);
    data->bufferNum = static_cast<uint32_t>(bigDimBufferNum_);
    data->totalNddmaNum = totalNddmaNum_;
    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        data->inputShape[i] = shapeInfo_.reducedInShape[i];
        data->outputShape[i] = shapeInfo_.reducedOutShape[i];
        data->perm[i] = static_cast<int8_t>(shapeInfo_.reducedPerm[i]);
    }
    std::copy(baseInShape_, baseInShape_ + TP_MAX_AXIS_NUM, data->baseInShape);
    std::copy(baseNddmaShape_, baseNddmaShape_ + TP_NDDMA_MAX_DIM_NUM, data->baseNddmaShape);
    for (int64_t i = 0; i < TP_NDDMA_MAX_DIM_NUM; i++) {
        data->nddmaIdx[i] = static_cast<int8_t>(nddmaIdx_[i]);
    }
    return true;
}

void TransposeNddmaBigDimStrategy::LogTilingData()
{
    OP_LOGI(context_->GetNodeName(), "Entering LogTilingData.");
    for (int64_t i = 0; i < shapeInfo_.dim; i++) {
        OP_LOGI(context_->GetNodeName(), "reducedInShape[%ld] is:%ld, reducedOutShape[%ld]:%ld, reducedPerm[%ld]:%ld.",
                i, shapeInfo_.reducedInShape[i], i, shapeInfo_.reducedOutShape[i], i, shapeInfo_.reducedPerm[i]);
    }
    for (int64_t i = 0; i < TP_NDDMA_MAX_DIM_NUM; i++) {
        OP_LOGI(context_->GetNodeName(), "baseNddmaShape_[%ld] is:%ld, nddmaIdx_[%ld]:%ld", i, baseNddmaShape_[i], i,
                nddmaIdx_[i]);
    }
    OP_LOGI(context_->GetNodeName(),
            "tilingData is permSize:%ld, outCutIndex:%ld, outUbFactor:%ld, outTailFactor:%ld, realCoreNum:%ld, "
            "blkFactor:%ld, blkTailFactor:%ld, ubBytes:%ld, bufferNum:%ld, totalNddmaNum:%ld, "
            "Tiling4Transpose ends. ",
            shapeInfo_.dim, outCutIndex_, outUbFactor_, outTailFactor_, realCoreNum_, blkFactor_, blkTailFactor_,
            ubSize_ / bigDimBufferNum_, bigDimBufferNum_, totalNddmaNum_);
}

} // namespace optiling
