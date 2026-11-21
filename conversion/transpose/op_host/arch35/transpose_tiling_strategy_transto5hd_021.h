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
 * \file transpose_tiling_strategy_transto5hd_021.h
 * \brief TRANSTO5HD_021(10008) 策略类
 *
 * 适用场景：DAV_5102 平台、3D perm=[0,2,1] 的 TransDataTo5HD 转置路径。
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_TRANSTO5HD_021_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_TRANSTO5HD_021_H

#include <cstdint>
#include "transpose_tiling_strategy.h"

namespace optiling {

/**
 * @brief TRANSTO5HD_021(10008) 策略
 *
 * 3D perm=[0,2,1] 的 TransDataTo5HD 转置；分核/切分失败返回 FALLBACK。
 */
class TransposeTransTo5HD021Strategy : public TransposeTilingStrategy {
private:
    static constexpr int8_t TRANSTO5HD021_BUFFER_NUM = 2;  ///< 输入/输出双缓冲
    static constexpr int8_t TRANSTO5HD021_NUM_TWO = 2;     ///< 数值 2
    static constexpr int64_t TRANSTO5HD021_TRANSELEM = 16; ///< TransDataTo5HD 每次处理元素数
    static constexpr int64_t TRANSTO5HD021_BLOCKELEM_8BIT = 32;
    static constexpr int64_t TRANSTO5HD021_BLOCKELEM_16BIT = 16;
    static constexpr int64_t TRANSTO5HD021_BLOCKELEM_32BIT = 8;
    static constexpr uint64_t B8_BYTES = 1;  ///< 8bit 类型字节数
    static constexpr uint64_t B16_BYTES = 2; ///< 16bit 类型字节数
    static constexpr uint64_t B32_BYTES = 4; ///< 32bit 类型字节数
    static constexpr int64_t DIM_ONE = 1;
    static constexpr int64_t DIM_TWO = 2;
    static constexpr int64_t DIM_FIVE = 5;
    static constexpr int64_t DIM_EIGHT = 8;
    static constexpr int64_t HW_ALIGN = 16;                                     ///< HW 对齐单位
    static constexpr int64_t HW_MIN_PRODUCT = 448;                              ///< HW 最小乘积阈值
    static constexpr int64_t SMALL_SHAPE_BYTES_THRES_HOLD_DAV_5102_021 = 70000; ///< DAV_5102 021 阈值：70KB

    struct TransTo5HD021BasicInfo {
        int64_t AvailableUbSize = 0;
        int64_t UsedCoreNum = 0;
        int64_t NLen = 0;
        int64_t HLen = 0;
        int64_t WLen = 0;
        int64_t HAlignBlockElem = 0;
        int64_t WAlignBlockElem = 0;
        int64_t NPerCore = 0;
        int64_t NTailCore = 0;
        int64_t UbLoopCount = 0;
        int64_t BlockElem = TRANSTO5HD021_BLOCKELEM_16BIT;
        bool UseRConv = false;
        bool UseHSplit = false;
        int64_t HPerCore = 0;
        int64_t HTailCore = 0;
    };
    struct TransTo5HD021UbParamInfo {
        int64_t UbAlignFactor = 0;
        int64_t UbFactor = 0;
        int64_t UbCount = 0;
        int64_t UbTailAlignFactor = 0;
        int64_t UbTailFactor = 0;
    };

public:
    using TransposeTilingStrategy::TransposeTilingStrategy;

    bool IsSupport() const override;

protected:
    // 主体切分
    TilingStatus DoExecute() override;
    // 写 TransposeTransTo5HD021TilingData（容量不足 → FAILURE）
    bool SetTilingData() override;
    // 打印 TilingData
    void LogTilingData() override;
    uint32_t GetUsedCoreNum() const override { return static_cast<uint32_t>(basicInfo_.UsedCoreNum); }
    uint64_t GetTilingKey() const override { return static_cast<uint64_t>(TK_TRANSTO5HD_021); }

private:
    void CalcBasicInfo();
    void CalcNSplitInfo();
    void CalcHSplitInfo();
    ge::graphStatus CalcUbSplitInfo();
    ge::graphStatus CalcUbSplitHSplit();
    ge::graphStatus CalcUbSplitRConv();
    ge::graphStatus CalcUbSplitCConv();
    void FillUbPara(Transpose::TransposeTransTo5HD021UbSplitPara& ubSplitPara, TransTo5HD021UbParamInfo& ubPara);

    TransTo5HD021BasicInfo basicInfo_;
    TransTo5HD021UbParamInfo rUbParamInfo_;
    TransTo5HD021UbParamInfo cUbParamInfo_;
};

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_STRATEGY_TRANSTO5HD_021_H
