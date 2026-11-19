/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file histogram_v2_tiling.h
 * \brief
 */
#ifndef MATH_HISTOGRAM_V2_H
#define MATH_HISTOGRAM_V2_H
#include "op_common/op_host/util/op_const_def.h"
#include "register/tilingdata_base.h"
#include "platform/platform_ascendc.h"
#include "op_host/tiling_base_class.h"
#include "math/histogram_v2/op_kernel/arch35/histogram_v2_tiling_data.h"
#include "math/histogram_v2/op_kernel/arch35/histogram_v2_tiling_key.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(HistogramV2TilingData)
TILING_DATA_FIELD_DEF(int64_t, bins);
TILING_DATA_FIELD_DEF(int64_t, ubBinsLength);

TILING_DATA_FIELD_DEF(int64_t, formerNum);
TILING_DATA_FIELD_DEF(int64_t, formerLength);
TILING_DATA_FIELD_DEF(int64_t, formerLengthAligned);
TILING_DATA_FIELD_DEF(int64_t, tailLength);
TILING_DATA_FIELD_DEF(int64_t, tailLengthAligned);

TILING_DATA_FIELD_DEF(int64_t, formerTileNum);
TILING_DATA_FIELD_DEF(int64_t, formerTileDataLength);
TILING_DATA_FIELD_DEF(int64_t, formerTileLeftDataLength);
TILING_DATA_FIELD_DEF(int64_t, formerTileLeftDataLengthAligned);

TILING_DATA_FIELD_DEF(int64_t, tailTileNum);
TILING_DATA_FIELD_DEF(int64_t, tailTileDataLength);
TILING_DATA_FIELD_DEF(int64_t, tailTileLeftDataLength);
TILING_DATA_FIELD_DEF(int64_t, tailTileLeftDataLengthAligned);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(HistogramV2, HistogramV2TilingData)

// arch35 uses the pure-struct HistogramV2SimtTilingData (see
// op_kernel/arch35/histogram_v2_tiling_data.h) with a template-parameter tiling
// key (schId, see op_kernel/arch35/histogram_v2_tiling_key.h); the legacy
// numeric keys 101..2117 and their REGISTER_TILING_DATA_CLASS entries are gone.

struct HistogramV2CompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
    int64_t sysWorkspaceSize = 0;
    NpuArch npuArch = Ops::Base::DAV_2201;
};

class HistogramV2BaseClass : public Ops::Base::TilingBaseClass {
public:
    explicit HistogramV2BaseClass(gert::TilingContext* context) : Ops::Base::TilingBaseClass(context) {}

    void Reset(gert::TilingContext* context) override { TilingBaseClass::Reset(context); }

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus DoLibApiTiling() override;

    NpuArch npuArch = Ops::Base::DAV_2201;
};
} // namespace optiling
#endif // MATH_HISTOGRAM_V2_H
