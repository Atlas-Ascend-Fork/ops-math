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
 * \file transpose_tiling_strategy_transto5hd_10.h
 * \brief TRANSTO5HD_10(10007) 策略类
 *
 * 适用场景：DAV_5102 平台、2D perm=[1,0] 且 16bit 的 TransDataTo5HD 转置路径。
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_TRANSTO5HD_10_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_TRANSTO5HD_10_H

#include <cstdint>
#include "transpose_tiling_strategy.h"

namespace optiling {

/**
 * @brief TRANSTO5HD_10(10007) 策略
 *
 * 2D perm=[1,0] 的 16bit TransDataTo5HD 转置；分核/切分失败返回 FALLBACK。
 */
class TransposeTransTo5HD10Strategy : public TransposeTilingStrategy {
public:
    using TransposeTilingStrategy::TransposeTilingStrategy;

    bool IsSupport() const override;

protected:
    // 主体切分
    TilingStatus DoExecute() override;
    // 写 TransposeTransTo5HD10TilingData（容量不足 → FAILURE）
    bool SetTilingData() override;
    // 打印 TilingData
    void LogTilingData() override;
    uint32_t GetUsedCoreNum() const override { return static_cast<uint32_t>(basicInfo_.UsedCoreNum); }
    uint64_t GetTilingKey() const override { return static_cast<uint64_t>(TK_TRANSTO5HD_10); }

private:
    struct TransTo5HD10BasicInfo {
        int64_t AvailableUbSize;
        int64_t UsedCoreNum;
        int64_t MainCoreLoopCount;
        int64_t TailCoreLoopCount;
        int64_t RLen;
        int64_t CLen;
        int64_t RAlignBlock;
        int64_t CAlignBlock;
        int64_t RAlignBlockElem;
        int64_t CAlignBlockElem;
        bool IsRSplit = false;
        bool IsRCSplit = false;
    };
    struct TransTo5HD10CoreSplitInfo {
        // r 方向
        int64_t AlignBlockFactor;
        int64_t BlockFactor;
        int64_t BlockCount;
        int64_t AlignBlockTailFactor;
        int64_t BlockTailFactor;
    };
    struct TransTo5HD10UbParamInfo {
        // 主核
        int64_t MainCoreUbAlignFactor;
        int64_t MainCoreUbFactor;
        int64_t MainCoreUbCount;
        int64_t MainCoreTailUbAlignFactor;
        int64_t MainCoreTailUbFactor;
        // 尾核
        int64_t TailCoreUbAlignFactor;
        int64_t TailCoreUbFactor;
        int64_t TailCoreUbCount;
        int64_t TailCoreTailUbAlignFactor;
        int64_t TailCoreTailUbFactor;
    };

    void CalcBasicInfo();
    void CalcRSplitInfo();
    void CalcCSplitInfo();
    ge::graphStatus CalcBlockAndUbSplitInfo();
    ge::graphStatus CalcCFullLoadRUbSplitInfo();
    ge::graphStatus CalcRFullLoadCUbSplitInfo();
    ge::graphStatus CalcRCNotFullLoadUbSplitInfo();
    void SetCFullLoadCUbSplitInfo();
    void SetRFullLoadCUbSplitInfo();
    void SetRBlockFactorUbFullLoadSplitInfo();
    void FillBlockPara(Transpose::CoreSplitPara& coreSplitPara, TransTo5HD10CoreSplitInfo& corePara);
    void FillUbPara(Transpose::UbSplitPara& ubSplitPara, TransTo5HD10UbParamInfo& ubPara);

    int64_t AvailableUbSize{0};
    TransTo5HD10BasicInfo basicInfo_{};
    TransTo5HD10UbParamInfo rUbParamInfo_{};
    TransTo5HD10UbParamInfo cUbParamInfo_{};
    TransTo5HD10CoreSplitInfo rCoreSplitInfo_{};
    TransTo5HD10CoreSplitInfo cCoreSplitInfo_{};
};

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_TRANSTO5HD_10_H
