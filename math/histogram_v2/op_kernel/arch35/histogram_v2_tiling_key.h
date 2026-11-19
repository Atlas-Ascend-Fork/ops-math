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
 * \file histogram_v2_tiling_key.h
 * \brief TilingKey template argument declaration for HistogramV2 (arch35).
 *
 * Single template parameter schId = load mode (0..2) plus deterministic-fp32out
 * offset (+3). Input dtype is covered by the DTYPE_X compile-time macro and the
 * output dtype by DTYPE_Y (both generated from the op def tensor names), so
 * neither appears in the tiling key.
 *
 * Legacy numeric keys (101/111/1111/...) are replaced by GET_TPL_TILING_KEY(schId);
 * with this single-parameter declaration the encoded key equals schId itself.
 */
#ifndef HISTOGRAM_V2_TILING_KEY_H
#define HISTOGRAM_V2_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

// Load modes (HistogramV2Simt<..., schId>::kMode == schId % 3).
constexpr uint64_t TPL_SCH_ID_UB_FULL = 0;   // bins fully resident in UB
constexpr uint64_t TPL_SCH_ID_UB_WINDOW = 1; // UB windowed loop over bins
constexpr uint64_t TPL_SCH_ID_GM_ATOMIC = 2; // direct GM atomic add, no UB histogram
// Deterministic fp32-output variants: int32 accumulation + final in-place cast to fp32.
constexpr uint64_t TPL_SCH_ID_UB_FULL_DET = 3;
constexpr uint64_t TPL_SCH_ID_UB_WINDOW_DET = 4;
constexpr uint64_t TPL_SCH_ID_GM_ATOMIC_DET = 5; // closes the legacy 1311/1317 hole (was silently non-deterministic)

// NOTE: the DECL/SEL macros are text-scanned by the opc template_tiling tool, so they
// must contain literal integers only (constexpr/#define references are not resolved).
ASCENDC_TPL_ARGS_DECL(HistogramV2,
                      ASCENDC_TPL_UINT_DECL(schId, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schId, ASCENDC_TPL_UI_RANGE, 1, 0, 5)));

#endif // HISTOGRAM_V2_TILING_KEY_H
