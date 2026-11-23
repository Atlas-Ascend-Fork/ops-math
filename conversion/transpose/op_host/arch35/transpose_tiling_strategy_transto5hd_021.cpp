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
 * \file transpose_tiling_strategy_transto5hd_021.cpp
 * \brief TRANSTO5HD_021(10008) 策略类实现
 *
 * 实现 3D 021 转置的分核与 UB 切分、TilingData 回填与打印。
 */

#include "transpose_tiling_strategy_transto5hd_021.h"
#include <algorithm>
#include <cstdint>
#include "util/math_util.h"
#include "log/log.h"

namespace optiling {
using namespace Transpose;

/**
 * @brief TRANSTO5HD_021(10008) 准入检查（纯判定、无副作用）
 *
 *   !isRelatedTranspose_ && arch==DAV_5102 && IsPermValue({0, 2, 1})
 *   && dtype B8/B16/B32 && HW 形状/对齐效率/总数据量满足门限。
 */
bool TransposeTransTo5HD021Strategy::IsSupport() const
{
    if (isRelatedTranspose_) {
        return false; // 融合（related）场景不尝试 TRANSTO5HD_021
    }
    if (npuArch_ != Ops::Base::DAV_5102) {
        return false; // DAV_5102 专用硬件指令路径
    }
    if (!IsPermValue({0, 2, 1})) {
        return false; // dim==3 且 perm=[0,2,1]（基类统一 perm 判定）
    }
    // dtype: support B8、B16、B32
    if (!(shapeInfo_.eleLenInBytes == B8_BYTES || shapeInfo_.eleLenInBytes == B16_BYTES ||
          shapeInfo_.eleLenInBytes == B32_BYTES)) {
        return false;
    }
    // HW 形状：过小或对齐后利用率过低时该硬件指令路径不划算
    const int64_t H = shapeInfo_.reducedInShape[DIM_ONE];
    const int64_t W = shapeInfo_.reducedInShape[DIM_TWO];
    if (H <= DIM_EIGHT || W <= DIM_EIGHT) {
        return false;
    }
    if (H * W < HW_MIN_PRODUCT) {
        return false;
    }
    const int64_t hAlign = Ops::Base::CeilDiv(H, HW_ALIGN) * HW_ALIGN;
    const int64_t wAlign = Ops::Base::CeilDiv(W, HW_ALIGN) * HW_ALIGN;
    if (H * W <= hAlign * wAlign / DIM_TWO) {
        return false;
    }
    // 总数据量门槛：低于 70KB 不命中
    return shapeInfo_.totalVolumeActual * shapeInfo_.eleLenInBytes >= SMALL_SHAPE_BYTES_THRES_HOLD_DAV_5102_021;
}

// ===== TRANSTO5HD_021 私有实现 =====

void TransposeTransTo5HD021Strategy::CalcBasicInfo()
{
    basicInfo_.AvailableUbSize = ubSize_ / TRANSTO5HD021_NUM_TWO / TRANSTO5HD021_BUFFER_NUM;
    basicInfo_.NLen = shapeInfo_.reducedInShape[0];
    basicInfo_.HLen = shapeInfo_.reducedInShape[1];
    basicInfo_.WLen = shapeInfo_.reducedInShape[2];
    basicInfo_.BlockElem = (shapeInfo_.eleLenInBytes == B8_BYTES)  ? TRANSTO5HD021_BLOCKELEM_8BIT :
                           (shapeInfo_.eleLenInBytes == B32_BYTES) ? TRANSTO5HD021_BLOCKELEM_32BIT :
                                                                     TRANSTO5HD021_BLOCKELEM_16BIT;
    basicInfo_.UseRConv = basicInfo_.HLen >= basicInfo_.WLen;
    basicInfo_.UseHSplit = basicInfo_.NLen < DIM_FIVE && !basicInfo_.UseRConv &&
                           basicInfo_.HLen >= basicInfo_.NLen * TRANSTO5HD021_TRANSELEM;
    if (basicInfo_.UseRConv) {
        int64_t hAlignUnit = (shapeInfo_.eleLenInBytes == 1) ? basicInfo_.BlockElem : TRANSTO5HD021_TRANSELEM;
        int64_t hAlignBlock = Ops::Base::CeilDiv(basicInfo_.HLen, hAlignUnit);
        int64_t wAlignBlock = Ops::Base::CeilDiv(basicInfo_.WLen, basicInfo_.BlockElem);
        basicInfo_.HAlignBlockElem = hAlignBlock * hAlignUnit;
        basicInfo_.WAlignBlockElem = wAlignBlock * basicInfo_.BlockElem;
    } else {
        int64_t hAlignUnit = (shapeInfo_.eleLenInBytes == 1) ? basicInfo_.BlockElem : TRANSTO5HD021_TRANSELEM;
        int64_t wAlignUnit = (shapeInfo_.eleLenInBytes == 1) ? basicInfo_.BlockElem : TRANSTO5HD021_TRANSELEM;
        int64_t hAlignBlock = Ops::Base::CeilDiv(basicInfo_.HLen, hAlignUnit);
        int64_t wAlignBlock = Ops::Base::CeilDiv(basicInfo_.WLen, wAlignUnit);
        basicInfo_.HAlignBlockElem = hAlignBlock * hAlignUnit;
        basicInfo_.WAlignBlockElem = wAlignBlock * wAlignUnit;
    }
}

void TransposeTransTo5HD021Strategy::CalcNSplitInfo()
{
    int64_t coreNum = coreNum_;
    basicInfo_.NPerCore = Ops::Base::CeilDiv(basicInfo_.NLen, coreNum);
    basicInfo_.NTailCore = basicInfo_.NLen - (coreNum - 1) * basicInfo_.NPerCore;
    if (basicInfo_.NTailCore <= 0) {
        basicInfo_.UsedCoreNum = Ops::Base::CeilDiv(basicInfo_.NLen, basicInfo_.NPerCore);
        basicInfo_.NTailCore = basicInfo_.NLen - (basicInfo_.UsedCoreNum - 1) * basicInfo_.NPerCore;
    } else {
        basicInfo_.UsedCoreNum = coreNum;
    }
    if (basicInfo_.NTailCore <= 0) {
        basicInfo_.NTailCore = basicInfo_.NPerCore;
    }
}

void TransposeTransTo5HD021Strategy::CalcHSplitInfo()
{
    int64_t coreNum = coreNum_;
    int64_t hAlignTotal = basicInfo_.HAlignBlockElem;
    int64_t minHPerCore = (shapeInfo_.eleLenInBytes == 4) ? TRANSTO5HD021_TRANSELEM :
                                                            (TRANSTO5HD021_NUM_TWO * TRANSTO5HD021_TRANSELEM);
    int64_t maxUsedCores = Ops::Base::CeilDiv(hAlignTotal, minHPerCore);
    int64_t effectiveCoreNum = std::min(coreNum, maxUsedCores);
    int64_t hPerCore = Ops::Base::CeilDiv(hAlignTotal, effectiveCoreNum);
    hPerCore = Ops::Base::CeilDiv(hPerCore, TRANSTO5HD021_TRANSELEM) * TRANSTO5HD021_TRANSELEM;
    if (shapeInfo_.eleLenInBytes == 1 && hPerCore % (TRANSTO5HD021_NUM_TWO * TRANSTO5HD021_TRANSELEM) != 0) {
        hPerCore = Ops::Base::CeilDiv(hPerCore, TRANSTO5HD021_NUM_TWO * TRANSTO5HD021_TRANSELEM) *
                   TRANSTO5HD021_NUM_TWO * TRANSTO5HD021_TRANSELEM;
    }
    basicInfo_.HPerCore = hPerCore;
    int64_t usedCoreNum = Ops::Base::CeilDiv(hAlignTotal, hPerCore);
    basicInfo_.HTailCore = hAlignTotal - (usedCoreNum - 1) * hPerCore;
    if (basicInfo_.HTailCore <= 0) {
        basicInfo_.HTailCore = hPerCore;
    }
    basicInfo_.UsedCoreNum = usedCoreNum;
    basicInfo_.NPerCore = basicInfo_.NLen;
    basicInfo_.NTailCore = basicInfo_.NLen;
}

ge::graphStatus TransposeTransTo5HD021Strategy::CalcUbSplitInfo()
{
    if (basicInfo_.UseHSplit) {
        return CalcUbSplitHSplit();
    }
    if (basicInfo_.UseRConv) {
        return CalcUbSplitRConv();
    }
    return CalcUbSplitCConv();
}

ge::graphStatus TransposeTransTo5HD021Strategy::CalcUbSplitHSplit()
{
    if (shapeInfo_.eleLenInBytes == 1) {
        OP_LOGD(context_->GetNodeName(), "8-bit H-split not supported for 021 VCONV branch");
        return ge::GRAPH_FAILED;
    }
    int64_t blockElem = basicInfo_.BlockElem;
    int64_t hPerCore = basicInfo_.HPerCore;
    int64_t hTailCore = basicInfo_.HTailCore;
    int64_t wAlignBlock = Ops::Base::CeilDiv(basicInfo_.WLen, blockElem);
    int64_t wAlignBlockElem = wAlignBlock * blockElem;
    if (hPerCore * wAlignBlockElem * shapeInfo_.eleLenInBytes <= basicInfo_.AvailableUbSize) {
        rUbParamInfo_.UbAlignFactor = Ops::Base::CeilDiv(hPerCore, TRANSTO5HD021_TRANSELEM);
        rUbParamInfo_.UbFactor = hPerCore;
        rUbParamInfo_.UbCount = 1;
        rUbParamInfo_.UbTailAlignFactor = Ops::Base::CeilDiv(hTailCore, TRANSTO5HD021_TRANSELEM);
        rUbParamInfo_.UbTailFactor = hTailCore;
        cUbParamInfo_.UbAlignFactor = wAlignBlock;
        cUbParamInfo_.UbFactor = wAlignBlockElem;
        cUbParamInfo_.UbCount = 1;
        cUbParamInfo_.UbTailAlignFactor = cUbParamInfo_.UbAlignFactor;
        cUbParamInfo_.UbTailFactor = cUbParamInfo_.UbFactor;
        basicInfo_.UbLoopCount = 1;
        return ge::GRAPH_SUCCESS;
    }
    int64_t wPerUbAlignBlock = basicInfo_.AvailableUbSize / shapeInfo_.eleLenInBytes / hPerCore / blockElem;
    if (wPerUbAlignBlock <= 0) {
        OP_LOGD(context_->GetNodeName(), "H too large for 021 VCONV H-split branch");
        return ge::GRAPH_FAILED;
    }
    cUbParamInfo_.UbAlignFactor = wPerUbAlignBlock;
    cUbParamInfo_.UbFactor = wPerUbAlignBlock * blockElem;
    cUbParamInfo_.UbCount = Ops::Base::CeilDiv(wAlignBlock, cUbParamInfo_.UbAlignFactor);
    cUbParamInfo_.UbTailAlignFactor = wAlignBlock - (cUbParamInfo_.UbCount - 1) * cUbParamInfo_.UbAlignFactor;
    cUbParamInfo_.UbTailFactor = cUbParamInfo_.UbTailAlignFactor * blockElem;
    rUbParamInfo_.UbAlignFactor = Ops::Base::CeilDiv(hPerCore, TRANSTO5HD021_TRANSELEM);
    rUbParamInfo_.UbFactor = hPerCore;
    rUbParamInfo_.UbCount = 1;
    rUbParamInfo_.UbTailAlignFactor = Ops::Base::CeilDiv(hTailCore, TRANSTO5HD021_TRANSELEM);
    rUbParamInfo_.UbTailFactor = hTailCore;
    basicInfo_.UbLoopCount = cUbParamInfo_.UbCount;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeTransTo5HD021Strategy::CalcUbSplitRConv()
{
    int64_t blockElem = basicInfo_.BlockElem;
    int64_t eleSize = shapeInfo_.eleLenInBytes;
    if (TRANSTO5HD021_TRANSELEM * basicInfo_.WAlignBlockElem * eleSize > basicInfo_.AvailableUbSize) {
        OP_LOGD(context_->GetNodeName(), "W dimension too large for 021 VCONV branch");
        return ge::GRAPH_FAILED;
    }
    if (basicInfo_.HAlignBlockElem * basicInfo_.WAlignBlockElem * eleSize <= basicInfo_.AvailableUbSize) {
        rUbParamInfo_.UbAlignFactor = Ops::Base::CeilDiv(basicInfo_.HLen, TRANSTO5HD021_TRANSELEM);
        rUbParamInfo_.UbFactor = basicInfo_.HAlignBlockElem;
        rUbParamInfo_.UbCount = 1;
        rUbParamInfo_.UbTailAlignFactor = rUbParamInfo_.UbAlignFactor;
        rUbParamInfo_.UbTailFactor = rUbParamInfo_.UbFactor;
        cUbParamInfo_.UbAlignFactor = Ops::Base::CeilDiv(basicInfo_.WLen, blockElem);
        cUbParamInfo_.UbFactor = basicInfo_.WAlignBlockElem;
        cUbParamInfo_.UbCount = 1;
        cUbParamInfo_.UbTailAlignFactor = cUbParamInfo_.UbAlignFactor;
        cUbParamInfo_.UbTailFactor = cUbParamInfo_.UbFactor;
        basicInfo_.UbLoopCount = 1;
        return ge::GRAPH_SUCCESS;
    }
    rUbParamInfo_.UbAlignFactor = basicInfo_.AvailableUbSize / eleSize / basicInfo_.WAlignBlockElem /
                                  TRANSTO5HD021_TRANSELEM;
    if (eleSize == 1 && rUbParamInfo_.UbAlignFactor % TRANSTO5HD021_NUM_TWO != 0) {
        rUbParamInfo_.UbAlignFactor -= 1;
    }
    rUbParamInfo_.UbFactor = rUbParamInfo_.UbAlignFactor * TRANSTO5HD021_TRANSELEM;
    int64_t hAlignBlock = Ops::Base::CeilDiv(basicInfo_.HLen, TRANSTO5HD021_TRANSELEM);
    rUbParamInfo_.UbCount = Ops::Base::CeilDiv(hAlignBlock, rUbParamInfo_.UbAlignFactor);
    int64_t tailAlignBlock = hAlignBlock - (rUbParamInfo_.UbCount - 1) * rUbParamInfo_.UbAlignFactor;
    if (eleSize == 1 && tailAlignBlock % TRANSTO5HD021_NUM_TWO != 0) {
        tailAlignBlock += 1;
    }
    rUbParamInfo_.UbTailAlignFactor = tailAlignBlock;
    rUbParamInfo_.UbTailFactor = tailAlignBlock * TRANSTO5HD021_TRANSELEM;
    cUbParamInfo_.UbAlignFactor = Ops::Base::CeilDiv(basicInfo_.WLen, blockElem);
    cUbParamInfo_.UbFactor = basicInfo_.WAlignBlockElem;
    cUbParamInfo_.UbCount = 1;
    cUbParamInfo_.UbTailAlignFactor = cUbParamInfo_.UbAlignFactor;
    cUbParamInfo_.UbTailFactor = cUbParamInfo_.UbFactor;
    basicInfo_.UbLoopCount = rUbParamInfo_.UbCount;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeTransTo5HD021Strategy::CalcUbSplitCConv()
{
    int64_t eleSize = shapeInfo_.eleLenInBytes;
    if (TRANSTO5HD021_TRANSELEM * basicInfo_.HAlignBlockElem * eleSize > basicInfo_.AvailableUbSize) {
        OP_LOGD(context_->GetNodeName(), "H dimension too large for 021 VCONV branch");
        return ge::GRAPH_FAILED;
    }
    if (basicInfo_.HAlignBlockElem * basicInfo_.WAlignBlockElem * eleSize <= basicInfo_.AvailableUbSize) {
        rUbParamInfo_.UbAlignFactor = Ops::Base::CeilDiv(basicInfo_.HLen, TRANSTO5HD021_TRANSELEM);
        rUbParamInfo_.UbFactor = basicInfo_.HAlignBlockElem;
        rUbParamInfo_.UbCount = 1;
        rUbParamInfo_.UbTailAlignFactor = rUbParamInfo_.UbAlignFactor;
        rUbParamInfo_.UbTailFactor = rUbParamInfo_.UbFactor;
        cUbParamInfo_.UbAlignFactor = Ops::Base::CeilDiv(basicInfo_.WLen, TRANSTO5HD021_TRANSELEM);
        cUbParamInfo_.UbFactor = basicInfo_.WAlignBlockElem;
        cUbParamInfo_.UbCount = 1;
        cUbParamInfo_.UbTailAlignFactor = cUbParamInfo_.UbAlignFactor;
        cUbParamInfo_.UbTailFactor = cUbParamInfo_.UbFactor;
        basicInfo_.UbLoopCount = 1;
        return ge::GRAPH_SUCCESS;
    }
    cUbParamInfo_.UbAlignFactor = basicInfo_.AvailableUbSize / eleSize / basicInfo_.HAlignBlockElem /
                                  TRANSTO5HD021_TRANSELEM;
    if (eleSize == 1 && cUbParamInfo_.UbAlignFactor % TRANSTO5HD021_NUM_TWO != 0) {
        cUbParamInfo_.UbAlignFactor -= 1;
    }
    if (cUbParamInfo_.UbAlignFactor <= 0) {
        OP_LOGD(context_->GetNodeName(), "W dimension too large for 8-bit 021 VCONV CConv branch");
        return ge::GRAPH_FAILED;
    }
    cUbParamInfo_.UbFactor = cUbParamInfo_.UbAlignFactor * TRANSTO5HD021_TRANSELEM;
    int64_t wAlignBlock = Ops::Base::CeilDiv(basicInfo_.WLen, TRANSTO5HD021_TRANSELEM);
    cUbParamInfo_.UbCount = Ops::Base::CeilDiv(wAlignBlock, cUbParamInfo_.UbAlignFactor);
    int64_t tailAlignBlock = wAlignBlock - (cUbParamInfo_.UbCount - 1) * cUbParamInfo_.UbAlignFactor;
    if (eleSize == 1 && tailAlignBlock % TRANSTO5HD021_NUM_TWO != 0) {
        tailAlignBlock += 1;
    }
    cUbParamInfo_.UbTailAlignFactor = tailAlignBlock;
    cUbParamInfo_.UbTailFactor = tailAlignBlock * TRANSTO5HD021_TRANSELEM;
    rUbParamInfo_.UbAlignFactor = Ops::Base::CeilDiv(basicInfo_.HLen, TRANSTO5HD021_TRANSELEM);
    rUbParamInfo_.UbFactor = basicInfo_.HAlignBlockElem;
    rUbParamInfo_.UbCount = 1;
    rUbParamInfo_.UbTailAlignFactor = rUbParamInfo_.UbAlignFactor;
    rUbParamInfo_.UbTailFactor = rUbParamInfo_.UbFactor;
    basicInfo_.UbLoopCount = cUbParamInfo_.UbCount;
    return ge::GRAPH_SUCCESS;
}

void TransposeTransTo5HD021Strategy::FillUbPara(TransposeTransTo5HD021UbSplitPara& ubSplitPara,
                                                TransTo5HD021UbParamInfo& ubPara)
{
    ubSplitPara.UbAlignFactor = static_cast<int32_t>(ubPara.UbAlignFactor);
    ubSplitPara.UbFactor = static_cast<int32_t>(ubPara.UbFactor);
    ubSplitPara.UbTailAlignFactor = static_cast<int32_t>(ubPara.UbTailAlignFactor);
    ubSplitPara.UbTailFactor = static_cast<int32_t>(ubPara.UbTailFactor);
}

bool TransposeTransTo5HD021Strategy::SetTilingData()
{
    TransposeTransTo5HD021TilingData* tiling = context_->GetTilingData<TransposeTransTo5HD021TilingData>();
    if (tiling == nullptr) {
        OP_LOGE(context_->GetNodeName(), "get tiling data failed, capacity is not enough!");
        return false;
    }

    tiling->AvailableUbSize = static_cast<int32_t>(basicInfo_.AvailableUbSize);
    tiling->NLen = basicInfo_.NLen;
    tiling->HLen = basicInfo_.HLen;
    tiling->WLen = basicInfo_.WLen;
    tiling->HAlignBlockElem = basicInfo_.HAlignBlockElem;
    tiling->WAlignBlockElem = basicInfo_.WAlignBlockElem;
    tiling->NPerCore = basicInfo_.NPerCore;
    tiling->NTailCore = basicInfo_.NTailCore;
    tiling->UbLoopCount = basicInfo_.UbLoopCount;
    tiling->UseRConv = basicInfo_.UseRConv;
    tiling->UseHSplit = basicInfo_.UseHSplit;
    tiling->HPerCore = basicInfo_.HPerCore;
    tiling->HTailCore = basicInfo_.HTailCore;
    FillUbPara(tiling->rUbSplitPara, rUbParamInfo_);
    FillUbPara(tiling->cUbSplitPara, cUbParamInfo_);
    return true;
}

/**
 * @brief 主体切分（不写 TilingData、不打印）
 */
TilingStatus TransposeTransTo5HD021Strategy::DoExecute()
{
    CalcBasicInfo();
    if (basicInfo_.UseHSplit) {
        CalcHSplitInfo();
        int64_t nSplitCoreNum = std::min(basicInfo_.NLen, coreNum_);
        if (basicInfo_.UsedCoreNum < std::max(static_cast<int64_t>(TRANSTO5HD021_NUM_TWO), nSplitCoreNum)) {
            basicInfo_.UseHSplit = false;
            CalcNSplitInfo();
        }
    } else {
        CalcNSplitInfo();
    }
    if (CalcUbSplitInfo() != ge::GRAPH_SUCCESS) {
        OP_LOGD(context_->GetNodeName(), "Stop to run transto5hd_021 tiling, UB split failed!");
        return TilingStatus::FALLBACK;
    }
    return TilingStatus::SUCCESS;
}

/**
 * @brief 打印 TilingData
 */
void TransposeTransTo5HD021Strategy::LogTilingData()
{
    OP_LOGI(context_->GetNodeName(),
            "tilingData is AvailableUbSize:%ld, UsedCoreNum:%ld, NLen:%ld, HLen:%ld, WLen:%ld, "
            "HAlignBlockElem:%ld, WAlignBlockElem:%ld, NPerCore:%ld, NTailCore:%ld, UbLoopCount:%ld, "
            "UseRConv:%d, UseHSplit:%d, HPerCore:%ld, HTailCore:%ld, "
            "rUbParam(UbAlignFactor:%ld, UbFactor:%ld, UbCount:%ld, UbTailAlignFactor:%ld, UbTailFactor:%ld), "
            "cUbParam(UbAlignFactor:%ld, UbFactor:%ld, UbCount:%ld, UbTailAlignFactor:%ld, UbTailFactor:%ld).",
            basicInfo_.AvailableUbSize, basicInfo_.UsedCoreNum, basicInfo_.NLen, basicInfo_.HLen, basicInfo_.WLen,
            basicInfo_.HAlignBlockElem, basicInfo_.WAlignBlockElem, basicInfo_.NPerCore, basicInfo_.NTailCore,
            basicInfo_.UbLoopCount, static_cast<int32_t>(basicInfo_.UseRConv),
            static_cast<int32_t>(basicInfo_.UseHSplit), basicInfo_.HPerCore, basicInfo_.HTailCore,
            rUbParamInfo_.UbAlignFactor, rUbParamInfo_.UbFactor, rUbParamInfo_.UbCount, rUbParamInfo_.UbTailAlignFactor,
            rUbParamInfo_.UbTailFactor, cUbParamInfo_.UbAlignFactor, cUbParamInfo_.UbFactor, cUbParamInfo_.UbCount,
            cUbParamInfo_.UbTailAlignFactor, cUbParamInfo_.UbTailFactor);
}

} // namespace optiling
