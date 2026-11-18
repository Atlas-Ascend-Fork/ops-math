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
 * \file transpose_tiling_with_021vconv_arch35.h
 * \brief tiling for 3D 021 transpose using TransDataTo5HD
 */

#ifndef _TRANSPOSE_TILING_WITH_021VCONV_ARCH35_H_
#define _TRANSPOSE_TILING_WITH_021VCONV_ARCH35_H_

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "util/math_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "transpose_tiling_base.h"
#include "transpose_tiling_arch35.h"

namespace optiling {
using namespace Transpose;
namespace Transpose021WithVCONV {
static constexpr int8_t BUFFER_NUM = 2;
static constexpr int8_t NUM_TWO = 2;
static constexpr int64_t TRANSELEM = 16;
static constexpr int64_t BLOCKELEM_8BIT = 32;
static constexpr int64_t BLOCKELEM_16BIT = 16;
static constexpr int64_t BLOCKELEM_32BIT = 8;

struct PlatInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

struct BasicInfo {
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
    int64_t BlockElem = BLOCKELEM_16BIT;
    bool UseRConv = false;
    bool UseHSplit = false;
    int64_t HPerCore = 0;
    int64_t HTailCore = 0;
};

struct UbParamInfo {
    int64_t UbAlignFactor = 0;
    int64_t UbFactor = 0;
    int64_t UbCount = 0;
    int64_t UbTailAlignFactor = 0;
    int64_t UbTailFactor = 0;
};

class Transpose021VCONVTiling {
public:
    explicit Transpose021VCONVTiling(gert::TilingContext* context, const PlatInfo& platInfo, const ShapeInfo& shapeInfo)
        : context_(context), platInfo_(platInfo), shapeInfo_(shapeInfo) {};
    ge::graphStatus DoTiling();

private:
    void CalcBasicInfo();
    void CalcNSplitInfo();
    void CalcHSplitInfo();
    ge::graphStatus CalcUbSplitInfo();
    ge::graphStatus CalcUbSplitHSplit();
    ge::graphStatus CalcUbSplitRConv();
    ge::graphStatus CalcUbSplitCConv();
    void FillUbPara(Transpose021UbSplitPara& ubSplitPara, UbParamInfo& ubPara);
    ge::graphStatus WriteTilingData();
    ge::graphStatus SetTilingKeyAndCore();

    gert::TilingContext* context_ = nullptr;
    PlatInfo platInfo_;
    ShapeInfo shapeInfo_;
    BasicInfo basicInfo_;
    UbParamInfo rUbParamInfo_;
    UbParamInfo cUbParamInfo_;
};

} // namespace Transpose021WithVCONV
} // namespace optiling
#endif
