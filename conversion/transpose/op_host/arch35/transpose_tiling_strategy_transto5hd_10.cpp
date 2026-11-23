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
 * \file transpose_tiling_strategy_transto5hd_10.cpp
 * \brief TRANSTO5HD_10(10007) 策略类实现
 *
 * 实现 2D perm=[1,0] 的 16bit TransDataTo5HD 转置：分核与 UB 切分、TilingData 回填。
 */

#include "transpose_tiling_strategy_transto5hd_10.h"
#include <cstdint>
#include "util/math_util.h"
#include "log/log.h"

namespace optiling {
using namespace Transpose;

// ===== TRANSTO5HD_10 私有实现 =====
namespace {
static constexpr int32_t TRANSTO5HD10_DSIZE = 2;            ///< 16bit
static constexpr int8_t TRANSTO5HD10_BUFFER_NUM = 2;        ///< 输入/输出双缓冲
static constexpr int64_t TRANSTO5HD10_TRANSELEM = 16;       ///< TransDataTo5HD 每次处理元素数
static constexpr int8_t TRANSTO5HD10_BUFFER_TENSOR_NUM = 2; ///< 输入输出各 1 块
static constexpr int64_t TRANSTO5HD10_DIM_FIVE = 5;
} // namespace

/**
 * @brief TRANSTO5HD_10(10007) 准入检查
 *
 *   !isRelatedTranspose_ && arch==DAV_5102 && IsPermValue({1, 0})
 *   && eleLenInBytes==2 && reducedInShape[0]>5。
 */
bool TransposeTransTo5HD10Strategy::IsSupport() const
{
    if (isRelatedTranspose_) {
        return false; // 融合（related）场景不尝试 TRANSTO5HD_10
    }
    if (npuArch_ != Ops::Base::DAV_5102) {
        return false; // DAV_5102 专用硬件指令路径
    }
    if (!IsPermValue({1, 0})) {
        return false; // dim==2 且 perm=[1,0]（基类统一 perm 判定）
    }
    if (shapeInfo_.eleLenInBytes != TRANSTO5HD10_DSIZE) {
        return false; // 16bit
    }
    return shapeInfo_.reducedInShape[0] > TRANSTO5HD10_DIM_FIVE; // shape[0]>5
}

void TransposeTransTo5HD10Strategy::CalcBasicInfo()
{
    AvailableUbSize = ubSize_ / TRANSTO5HD10_BUFFER_TENSOR_NUM / TRANSTO5HD10_BUFFER_NUM;
    basicInfo_.RLen = shapeInfo_.reducedInShape[0];
    basicInfo_.CLen = shapeInfo_.reducedInShape[1];
    basicInfo_.RAlignBlock = Ops::Base::CeilDiv(basicInfo_.RLen, TRANSTO5HD10_TRANSELEM);
    basicInfo_.CAlignBlock = Ops::Base::CeilDiv(basicInfo_.CLen, TRANSTO5HD10_TRANSELEM);
    basicInfo_.RAlignBlockElem = basicInfo_.RAlignBlock * TRANSTO5HD10_TRANSELEM;
    basicInfo_.CAlignBlockElem = basicInfo_.CAlignBlock * TRANSTO5HD10_TRANSELEM;
}

void TransposeTransTo5HD10Strategy::CalcRSplitInfo()
{
    rCoreSplitInfo_.AlignBlockFactor = Ops::Base::CeilDiv(basicInfo_.RAlignBlock, coreNum_);
    rCoreSplitInfo_.BlockFactor = rCoreSplitInfo_.AlignBlockFactor * TRANSTO5HD10_TRANSELEM;
    rCoreSplitInfo_.BlockCount = Ops::Base::CeilDiv(basicInfo_.RAlignBlock, rCoreSplitInfo_.AlignBlockFactor);
    rCoreSplitInfo_.AlignBlockTailFactor = basicInfo_.RAlignBlock -
                                           (rCoreSplitInfo_.BlockCount - 1) * rCoreSplitInfo_.AlignBlockFactor;
    rCoreSplitInfo_.BlockTailFactor = rCoreSplitInfo_.AlignBlockTailFactor * TRANSTO5HD10_TRANSELEM;
    basicInfo_.UsedCoreNum = rCoreSplitInfo_.BlockCount;
    basicInfo_.IsRSplit = true;
}

void TransposeTransTo5HD10Strategy::CalcCSplitInfo()
{
    cCoreSplitInfo_.AlignBlockFactor = Ops::Base::CeilDiv(basicInfo_.CAlignBlock, coreNum_);
    cCoreSplitInfo_.BlockFactor = cCoreSplitInfo_.AlignBlockFactor * TRANSTO5HD10_TRANSELEM;
    cCoreSplitInfo_.BlockCount = Ops::Base::CeilDiv(basicInfo_.CAlignBlock, cCoreSplitInfo_.AlignBlockFactor);
    cCoreSplitInfo_.AlignBlockTailFactor = basicInfo_.CAlignBlock -
                                           (cCoreSplitInfo_.BlockCount - 1) * cCoreSplitInfo_.AlignBlockFactor;
    cCoreSplitInfo_.BlockTailFactor = cCoreSplitInfo_.AlignBlockTailFactor * TRANSTO5HD10_TRANSELEM;
    basicInfo_.UsedCoreNum = cCoreSplitInfo_.BlockCount;
}

void TransposeTransTo5HD10Strategy::FillBlockPara(CoreSplitPara& coreSplitPara, TransTo5HD10CoreSplitInfo& corePara)
{
    coreSplitPara.AlignBlockFactor = corePara.AlignBlockFactor;
    coreSplitPara.BlockFactor = corePara.BlockFactor;
    coreSplitPara.BlockCount = corePara.BlockCount;
    coreSplitPara.AlignBlockTailFactor = corePara.AlignBlockTailFactor;
    coreSplitPara.BlockTailFactor = corePara.BlockTailFactor;
}

void TransposeTransTo5HD10Strategy::FillUbPara(UbSplitPara& ubSplitPara, TransTo5HD10UbParamInfo& ubPara)
{
    ubSplitPara.MainCoreUbAlignFactor = static_cast<int32_t>(ubPara.MainCoreUbAlignFactor);
    ubSplitPara.MainCoreUbFactor = static_cast<int32_t>(ubPara.MainCoreUbFactor);
    ubSplitPara.MainCoreTailUbAlignFactor = static_cast<int32_t>(ubPara.MainCoreTailUbAlignFactor);
    ubSplitPara.MainCoreTailUbFactor = static_cast<int32_t>(ubPara.MainCoreTailUbFactor);
    ubSplitPara.TailCoreUbAlignFactor = static_cast<int32_t>(ubPara.TailCoreUbAlignFactor);
    ubSplitPara.TailCoreUbFactor = static_cast<int32_t>(ubPara.TailCoreUbFactor);
    ubSplitPara.TailCoreTailUbAlignFactor = static_cast<int32_t>(ubPara.TailCoreTailUbAlignFactor);
    ubSplitPara.TailCoreTailUbFactor = static_cast<int32_t>(ubPara.TailCoreTailUbFactor);
}

bool TransposeTransTo5HD10Strategy::SetTilingData()
{
    TransposeTransTo5HD10TilingData* tiling = context_->GetTilingData<TransposeTransTo5HD10TilingData>();
    if (tiling == nullptr) {
        OP_LOGE(context_->GetNodeName(), "get tiling data failed, capacity is not enough!");
        return false;
    }

    tiling->AvailableUbSize = static_cast<int32_t>(AvailableUbSize);
    tiling->MainCoreLoopCount = basicInfo_.MainCoreLoopCount;
    tiling->TailCoreLoopCount = basicInfo_.TailCoreLoopCount;
    tiling->RLen = basicInfo_.RLen;
    tiling->CLen = basicInfo_.CLen;
    tiling->RAlignBlock = basicInfo_.RAlignBlock;
    tiling->CAlignBlock = basicInfo_.CAlignBlock;
    tiling->RAlignBlockElem = basicInfo_.RAlignBlockElem;
    tiling->CAlignBlockElem = basicInfo_.CAlignBlockElem;
    tiling->IsRSplit = basicInfo_.IsRSplit;
    tiling->IsRCSplit = basicInfo_.IsRCSplit;
    FillBlockPara(tiling->rSplitPara, rCoreSplitInfo_);
    FillBlockPara(tiling->cSplitPara, cCoreSplitInfo_);
    FillUbPara(tiling->rUbSplitPara, rUbParamInfo_);
    FillUbPara(tiling->cUbSplitPara, cUbParamInfo_);
    return true;
}

void TransposeTransTo5HD10Strategy::LogTilingData()
{
    OP_LOGD(context_->GetNodeName(), "Entering PrintTilingData.");
    OP_LOGD(context_->GetNodeName(),
            "AvailableUbSize: %ld, UsedCoreNum: %ld, MainCoreLoopCount: %ld, TailCoreLoopCount: %ld, RLen: %ld, \
        CLen: %ld, RAlignBlock: %ld, CAlignBlock: %ld, RAlignBlockElem: %ld, \
        CAlignBlockElem: %ld, IsRSplit: %d, IsRCSplit: %d",
            AvailableUbSize, basicInfo_.UsedCoreNum, basicInfo_.MainCoreLoopCount, basicInfo_.TailCoreLoopCount,
            basicInfo_.RLen, basicInfo_.CLen, basicInfo_.RAlignBlock, basicInfo_.CAlignBlock,
            basicInfo_.RAlignBlockElem, basicInfo_.CAlignBlockElem, basicInfo_.IsRSplit, basicInfo_.IsRCSplit);
    OP_LOGD(
        context_->GetNodeName(),
        "core split info is: rAlignBlockFactor: %ld, rBlockFactor: %ld, rBlockCount: %ld, rAlignBlockTailFactor: %ld, \
        rBlockTailFactor: %ld, cAlignBlockFactor: %ld, cBlockFactor: %ld, cBlockCount: %ld, cAlignBlockTailFactor: %ld, \
        cBlockTailFactor: %ld",
        rCoreSplitInfo_.AlignBlockFactor, rCoreSplitInfo_.BlockFactor, rCoreSplitInfo_.BlockCount,
        rCoreSplitInfo_.AlignBlockTailFactor, rCoreSplitInfo_.BlockTailFactor, cCoreSplitInfo_.AlignBlockFactor,
        cCoreSplitInfo_.BlockFactor, cCoreSplitInfo_.BlockCount, cCoreSplitInfo_.AlignBlockTailFactor,
        cCoreSplitInfo_.BlockTailFactor);

    OP_LOGD(context_->GetNodeName(),
            "r ub params info is: rMainCoreUbAlignFactor: %ld, rMainCoreUbFactor: %ld, rMainCoreUbCount: %ld, \
        rMainCoreTailUbAlignFactor: %ld, rMainCoreTailUbFactor: %ld, rTailCoreUbAlignFactor: %ld, rTailCoreUbFactor: %ld, \
        rTailCoreUbCount: %ld, rTailCoreTailUbAlignFactor: %ld, rTailCoreTailUbFactor: %ld",
            rUbParamInfo_.MainCoreUbAlignFactor, rUbParamInfo_.MainCoreUbFactor, rUbParamInfo_.MainCoreUbCount,
            rUbParamInfo_.MainCoreTailUbAlignFactor, rUbParamInfo_.MainCoreTailUbFactor,
            rUbParamInfo_.TailCoreUbAlignFactor, rUbParamInfo_.TailCoreUbFactor, rUbParamInfo_.TailCoreUbCount,
            rUbParamInfo_.TailCoreTailUbAlignFactor, rUbParamInfo_.TailCoreTailUbFactor);
    OP_LOGD(context_->GetNodeName(),
            "c ub params info is :cMainCoreUbAlignFactor: %ld, cMainCoreUbFactor: %ld, cMainCoreUbCount: %ld, \
        cMainCoreTailUbAlignFactor: %ld, cMainCoreTailUbFactor: %ld, cTailCoreUbAlignFactor: %ld, cTailCoreUbFactor: %ld, \
        cTailCoreUbCount: %ld, cTailCoreTailUbAlignFactor: %ld, cTailCoreTailUbFactor: %ld",
            cUbParamInfo_.MainCoreUbAlignFactor, cUbParamInfo_.MainCoreUbFactor, cUbParamInfo_.MainCoreUbCount,
            cUbParamInfo_.MainCoreTailUbAlignFactor, cUbParamInfo_.MainCoreTailUbFactor,
            cUbParamInfo_.TailCoreUbAlignFactor, cUbParamInfo_.TailCoreUbFactor, cUbParamInfo_.TailCoreUbCount,
            cUbParamInfo_.TailCoreTailUbAlignFactor, cUbParamInfo_.TailCoreTailUbFactor);
}

ge::graphStatus TransposeTransTo5HD10Strategy::CalcCFullLoadRUbSplitInfo()
{
    SetCFullLoadCUbSplitInfo();
    if (rCoreSplitInfo_.BlockFactor * basicInfo_.CAlignBlockElem * shapeInfo_.eleLenInBytes <= AvailableUbSize) {
        OP_LOGD(context_->GetNodeName(), "Entering CalcCFullLoadRUbSplitInfo if branch");
        SetRBlockFactorUbFullLoadSplitInfo();
        basicInfo_.MainCoreLoopCount = 1;
        basicInfo_.TailCoreLoopCount = 1;
    } else {
        OP_LOGD(context_->GetNodeName(), "Entering CalcCFullLoadRUbSplitInfo else branch");
        rUbParamInfo_.MainCoreUbAlignFactor = AvailableUbSize / shapeInfo_.eleLenInBytes / basicInfo_.CAlignBlockElem /
                                              TRANSTO5HD10_TRANSELEM;
        rUbParamInfo_.MainCoreUbFactor = rUbParamInfo_.MainCoreUbAlignFactor * TRANSTO5HD10_TRANSELEM;
        rUbParamInfo_.MainCoreUbCount = Ops::Base::CeilDiv(rCoreSplitInfo_.BlockFactor, rUbParamInfo_.MainCoreUbFactor);
        rUbParamInfo_.MainCoreTailUbAlignFactor = rCoreSplitInfo_.AlignBlockFactor -
                                                  (rUbParamInfo_.MainCoreUbCount - 1) *
                                                      rUbParamInfo_.MainCoreUbAlignFactor;
        rUbParamInfo_.MainCoreTailUbFactor = rUbParamInfo_.MainCoreTailUbAlignFactor * TRANSTO5HD10_TRANSELEM;

        rUbParamInfo_.TailCoreUbAlignFactor = (rUbParamInfo_.MainCoreUbAlignFactor >=
                                               rCoreSplitInfo_.AlignBlockTailFactor) ?
                                                  rCoreSplitInfo_.AlignBlockTailFactor :
                                                  rUbParamInfo_.MainCoreUbAlignFactor;
        rUbParamInfo_.TailCoreUbFactor = rUbParamInfo_.TailCoreUbAlignFactor * TRANSTO5HD10_TRANSELEM;
        rUbParamInfo_.TailCoreUbCount = Ops::Base::CeilDiv(rCoreSplitInfo_.BlockTailFactor,
                                                           rUbParamInfo_.TailCoreUbFactor);
        rUbParamInfo_.TailCoreTailUbAlignFactor = rCoreSplitInfo_.AlignBlockTailFactor -
                                                  (rUbParamInfo_.TailCoreUbCount - 1) *
                                                      rUbParamInfo_.TailCoreUbAlignFactor;
        rUbParamInfo_.TailCoreTailUbFactor = rUbParamInfo_.TailCoreTailUbAlignFactor * TRANSTO5HD10_TRANSELEM;

        basicInfo_.MainCoreLoopCount = rUbParamInfo_.MainCoreUbCount;
        basicInfo_.TailCoreLoopCount = rUbParamInfo_.TailCoreUbCount;
    }
    return ge::GRAPH_SUCCESS;
}

void TransposeTransTo5HD10Strategy::SetCFullLoadCUbSplitInfo()
{
    cUbParamInfo_.MainCoreUbAlignFactor = basicInfo_.CAlignBlock;
    cUbParamInfo_.MainCoreUbFactor = basicInfo_.CAlignBlockElem;
    cUbParamInfo_.MainCoreUbCount = 1;
    cUbParamInfo_.MainCoreTailUbAlignFactor = cUbParamInfo_.MainCoreUbAlignFactor;
    cUbParamInfo_.MainCoreTailUbFactor = cUbParamInfo_.MainCoreUbFactor;

    cUbParamInfo_.TailCoreUbAlignFactor = basicInfo_.CAlignBlock;
    cUbParamInfo_.TailCoreUbFactor = basicInfo_.CAlignBlockElem;
    cUbParamInfo_.TailCoreUbCount = 1;
    cUbParamInfo_.TailCoreTailUbAlignFactor = cUbParamInfo_.TailCoreUbAlignFactor;
    cUbParamInfo_.TailCoreTailUbFactor = cUbParamInfo_.TailCoreUbFactor;
    cCoreSplitInfo_.AlignBlockFactor = basicInfo_.CAlignBlock;
    cCoreSplitInfo_.BlockFactor = basicInfo_.CAlignBlockElem;
    cCoreSplitInfo_.BlockCount = 1;
    cCoreSplitInfo_.AlignBlockTailFactor = cCoreSplitInfo_.AlignBlockFactor;
    cCoreSplitInfo_.BlockTailFactor = cCoreSplitInfo_.BlockFactor;
}

ge::graphStatus TransposeTransTo5HD10Strategy::CalcRFullLoadCUbSplitInfo()
{
    SetRFullLoadCUbSplitInfo();
    if (cCoreSplitInfo_.BlockFactor * basicInfo_.RAlignBlockElem * shapeInfo_.eleLenInBytes <= AvailableUbSize) {
        OP_LOGD(context_->GetNodeName(), "Entering CalcRFullLoadCUbSplitInfo if branch");
        cUbParamInfo_.MainCoreUbAlignFactor = cCoreSplitInfo_.AlignBlockFactor;
        cUbParamInfo_.MainCoreUbFactor = cCoreSplitInfo_.BlockFactor;
        cUbParamInfo_.MainCoreUbCount = 1;
        cUbParamInfo_.MainCoreTailUbAlignFactor = cUbParamInfo_.MainCoreUbAlignFactor;
        cUbParamInfo_.MainCoreTailUbFactor = cUbParamInfo_.MainCoreUbFactor;

        cUbParamInfo_.TailCoreUbAlignFactor = cCoreSplitInfo_.AlignBlockTailFactor;
        cUbParamInfo_.TailCoreUbFactor = cCoreSplitInfo_.BlockTailFactor;
        cUbParamInfo_.TailCoreUbCount = 1;
        cUbParamInfo_.TailCoreTailUbAlignFactor = cUbParamInfo_.TailCoreUbAlignFactor;
        cUbParamInfo_.TailCoreTailUbFactor = cUbParamInfo_.TailCoreUbFactor;
        basicInfo_.MainCoreLoopCount = 1;
        basicInfo_.TailCoreLoopCount = 1;
    } else {
        OP_LOGD(context_->GetNodeName(), "Entering CalcRFullLoadCUbSplitInfo else branch");
        cUbParamInfo_.MainCoreUbAlignFactor = AvailableUbSize / shapeInfo_.eleLenInBytes / basicInfo_.RAlignBlockElem /
                                              TRANSTO5HD10_TRANSELEM;
        cUbParamInfo_.MainCoreUbFactor = cUbParamInfo_.MainCoreUbAlignFactor * TRANSTO5HD10_TRANSELEM;
        cUbParamInfo_.MainCoreUbCount = Ops::Base::CeilDiv(cCoreSplitInfo_.BlockFactor, cUbParamInfo_.MainCoreUbFactor);
        cUbParamInfo_.MainCoreTailUbAlignFactor = cCoreSplitInfo_.AlignBlockFactor -
                                                  (cUbParamInfo_.MainCoreUbCount - 1) *
                                                      cUbParamInfo_.MainCoreUbAlignFactor;
        cUbParamInfo_.MainCoreTailUbFactor = cUbParamInfo_.MainCoreTailUbAlignFactor * TRANSTO5HD10_TRANSELEM;

        cUbParamInfo_.TailCoreUbAlignFactor = (cUbParamInfo_.MainCoreUbAlignFactor >=
                                               cCoreSplitInfo_.AlignBlockTailFactor) ?
                                                  cCoreSplitInfo_.AlignBlockTailFactor :
                                                  cUbParamInfo_.MainCoreUbAlignFactor;
        cUbParamInfo_.TailCoreUbFactor = cUbParamInfo_.TailCoreUbAlignFactor * TRANSTO5HD10_TRANSELEM;
        cUbParamInfo_.TailCoreUbCount = Ops::Base::CeilDiv(cCoreSplitInfo_.BlockTailFactor,
                                                           cUbParamInfo_.TailCoreUbFactor);
        cUbParamInfo_.TailCoreTailUbAlignFactor = cCoreSplitInfo_.AlignBlockTailFactor -
                                                  (cUbParamInfo_.TailCoreUbCount - 1) *
                                                      cUbParamInfo_.TailCoreUbAlignFactor;
        cUbParamInfo_.TailCoreTailUbFactor = cUbParamInfo_.TailCoreTailUbAlignFactor * TRANSTO5HD10_TRANSELEM;

        basicInfo_.MainCoreLoopCount = cUbParamInfo_.MainCoreUbCount;
        basicInfo_.TailCoreLoopCount = cUbParamInfo_.TailCoreUbCount;
    }
    return ge::GRAPH_SUCCESS;
}

void TransposeTransTo5HD10Strategy::SetRFullLoadCUbSplitInfo()
{
    rUbParamInfo_.MainCoreUbAlignFactor = basicInfo_.RAlignBlock;
    rUbParamInfo_.MainCoreUbFactor = basicInfo_.RAlignBlockElem;
    rUbParamInfo_.MainCoreUbCount = 1;
    rUbParamInfo_.MainCoreTailUbAlignFactor = rUbParamInfo_.MainCoreUbAlignFactor;
    rUbParamInfo_.MainCoreTailUbFactor = rUbParamInfo_.MainCoreUbFactor;

    rUbParamInfo_.TailCoreUbAlignFactor = basicInfo_.RAlignBlock;
    rUbParamInfo_.TailCoreUbFactor = basicInfo_.RAlignBlockElem;
    rUbParamInfo_.TailCoreUbCount = 1;
    rUbParamInfo_.TailCoreTailUbAlignFactor = rUbParamInfo_.TailCoreUbAlignFactor;
    rUbParamInfo_.TailCoreTailUbFactor = rUbParamInfo_.TailCoreUbFactor;

    rCoreSplitInfo_.AlignBlockFactor = basicInfo_.RAlignBlock;
    rCoreSplitInfo_.BlockFactor = basicInfo_.RAlignBlockElem;
    rCoreSplitInfo_.BlockCount = 1;
    rCoreSplitInfo_.AlignBlockTailFactor = rCoreSplitInfo_.AlignBlockFactor;
    rCoreSplitInfo_.BlockTailFactor = rCoreSplitInfo_.BlockFactor;
}

ge::graphStatus TransposeTransTo5HD10Strategy::CalcRCNotFullLoadUbSplitInfo()
{
    if (rCoreSplitInfo_.BlockFactor * TRANSTO5HD10_TRANSELEM * shapeInfo_.eleLenInBytes > AvailableUbSize) {
        OP_LOGD(context_->GetNodeName(), "BlockFactor * 16 > AvailableUbSize!");
        return ge::GRAPH_FAILED;
    } else {
        basicInfo_.IsRCSplit = true;
        OP_LOGD(context_->GetNodeName(), "Entering CalcRCNotFullLoadUbSplitInfo branch");
        SetRBlockFactorUbFullLoadSplitInfo();

        cUbParamInfo_.MainCoreUbAlignFactor = AvailableUbSize / shapeInfo_.eleLenInBytes / rCoreSplitInfo_.BlockFactor /
                                              TRANSTO5HD10_TRANSELEM;
        cUbParamInfo_.MainCoreUbFactor = cUbParamInfo_.MainCoreUbAlignFactor * TRANSTO5HD10_TRANSELEM;
        cUbParamInfo_.MainCoreUbCount = Ops::Base::CeilDiv(basicInfo_.CAlignBlockElem, cUbParamInfo_.MainCoreUbFactor);
        cUbParamInfo_.MainCoreTailUbAlignFactor = basicInfo_.CAlignBlock - (cUbParamInfo_.MainCoreUbCount - 1) *
                                                                               cUbParamInfo_.MainCoreUbAlignFactor;
        cUbParamInfo_.MainCoreTailUbFactor = cUbParamInfo_.MainCoreTailUbAlignFactor * TRANSTO5HD10_TRANSELEM;

        cUbParamInfo_.TailCoreUbAlignFactor = cUbParamInfo_.MainCoreUbAlignFactor;
        cUbParamInfo_.TailCoreUbFactor = cUbParamInfo_.MainCoreUbFactor;
        cUbParamInfo_.TailCoreUbCount = cUbParamInfo_.MainCoreUbCount;
        cUbParamInfo_.TailCoreTailUbAlignFactor = cUbParamInfo_.MainCoreTailUbAlignFactor;
        cUbParamInfo_.TailCoreTailUbFactor = cUbParamInfo_.MainCoreTailUbFactor;

        basicInfo_.MainCoreLoopCount = cUbParamInfo_.MainCoreUbCount;
        basicInfo_.TailCoreLoopCount = cUbParamInfo_.TailCoreUbCount;

        cCoreSplitInfo_.AlignBlockFactor = basicInfo_.CAlignBlock;
        cCoreSplitInfo_.BlockFactor = basicInfo_.CAlignBlockElem;
        cCoreSplitInfo_.BlockCount = 1;
        cCoreSplitInfo_.AlignBlockTailFactor = cCoreSplitInfo_.AlignBlockFactor;
        cCoreSplitInfo_.BlockTailFactor = cCoreSplitInfo_.BlockFactor;
        return ge::GRAPH_SUCCESS;
    }
}

void TransposeTransTo5HD10Strategy::SetRBlockFactorUbFullLoadSplitInfo()
{
    rUbParamInfo_.MainCoreUbAlignFactor = rCoreSplitInfo_.AlignBlockFactor;
    rUbParamInfo_.MainCoreUbFactor = rCoreSplitInfo_.BlockFactor;
    rUbParamInfo_.MainCoreUbCount = 1;
    rUbParamInfo_.MainCoreTailUbAlignFactor = rUbParamInfo_.MainCoreUbAlignFactor;
    rUbParamInfo_.MainCoreTailUbFactor = rUbParamInfo_.MainCoreUbFactor;

    rUbParamInfo_.TailCoreUbAlignFactor = rCoreSplitInfo_.AlignBlockTailFactor;
    rUbParamInfo_.TailCoreUbFactor = rCoreSplitInfo_.BlockTailFactor;
    rUbParamInfo_.TailCoreUbCount = 1;
    rUbParamInfo_.TailCoreTailUbAlignFactor = rUbParamInfo_.TailCoreUbAlignFactor;
    rUbParamInfo_.TailCoreTailUbFactor = rUbParamInfo_.TailCoreUbFactor;
}

ge::graphStatus TransposeTransTo5HD10Strategy::CalcBlockAndUbSplitInfo()
{
    if (TRANSTO5HD10_TRANSELEM * basicInfo_.CAlignBlockElem * shapeInfo_.eleLenInBytes <= AvailableUbSize) {
        CalcRSplitInfo();
        return CalcCFullLoadRUbSplitInfo();
    } else {
        if (TRANSTO5HD10_TRANSELEM * basicInfo_.RAlignBlockElem * shapeInfo_.eleLenInBytes <= AvailableUbSize) {
            CalcCSplitInfo();
            return CalcRFullLoadCUbSplitInfo();
        } else {
            CalcRSplitInfo();
            return CalcRCNotFullLoadUbSplitInfo();
        }
    }
}

/**
 * @brief 主体切分（不写 TilingData、不打印）
 */
TilingStatus TransposeTransTo5HD10Strategy::DoExecute()
{
    CalcBasicInfo();
    if (CalcBlockAndUbSplitInfo() != ge::GRAPH_SUCCESS) {
        OP_LOGD(context_->GetNodeName(), "Stop to run transto5hd_10 tiling, block_factor need split!");
        return TilingStatus::FALLBACK;
    }
    return TilingStatus::SUCCESS;
}

} // namespace optiling
