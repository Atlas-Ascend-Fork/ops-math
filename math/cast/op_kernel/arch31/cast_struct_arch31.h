/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cast_struct_arch31.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_CAST_STRUCT_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_CAST_STRUCT_H

#include "ascendc/host_api/tiling/template_argument.h"

#define CAST_TEMPLATE_DIRECT_CAST 1
#define CAST_TEMPLATE_DST_BOOL 2
#define CAST_TEMPLATE_THROUGH 3
#define CAST_TEMPLATE_SRC_UINT1 4
#define CAST_TEMPLATE_TWO_CAST 5
#define CAST_TEMPLATE_MIRCRO_INOUT 6
#define CAST_TEMPLATE_MIRCRO_CAST 7
#define CAST_TEMPLATE_MIRCRO_CAST_INTER 8
#define CAST_TEMPLATE_MIRCRO_CAST_DEINTER 9
#define CAST_TEMPLATE_MIRCRO_CAST_CAST_DEINTER 10
#define CAST_TEMPLATE_MIRCRO_CAST_CAST 11              // f4 <-> f16
#define CAST_TEMPLATE_MIRCRO_CAST_INTER_CAST 12        // f4 -> f32
#define CAST_TEMPLATE_MIRCRO_CAST_DEINTER_CAST 13      // f32 -> f4
#define CAST_TEMPLATE_MIRCRO_CAST_CAST_DEINTER_CAST 14 // f8 -> f4
#define CAST_TEMPLATE_MIRCRO_CAST_INTER_CAST_CAST 15   // f4 -> f8
#define CAST_TEMPLATE_MIRCRO_DEINTER_SHIFT 16          // int32 -> int4

// kernel会映射到 LoadDist/StoreDist
#define CAST_MODE_REG_COPYIN_NORM 0
#define CAST_MODE_REG_COPYIN_DS_B8 1
#define CAST_MODE_REG_COPYIN_DS_B16 2
#define CAST_MODE_REG_COPYIN_UNPACK_B8 3
#define CAST_MODE_REG_COPYIN_UNPACK_B16 4
#define CAST_MODE_REG_COPYIN_UNPACK_B32 5
#define CAST_MODE_REG_COPYIN_UNPACK4_B8 6
#define CAST_MODE_REG_COPYOUT_NORM 7
#define CAST_MODE_REG_COPYOUT_PACK_B16 8
#define CAST_MODE_REG_COPYOUT_PACK_B32 9
#define CAST_MODE_REG_COPYOUT_PACK_B64 10
#define CAST_MODE_REG_COPYOUT_PACK4_B32 11
#define CAST_MODE_BUTT 12

#define CAST_ROUND_MODE_UNKNOWN 0
#define CAST_ROUND_MODE_NONE 1
#define CAST_ROUND_MODE_RINT 2
#define CAST_ROUND_MODE_FLOOR 3
#define CAST_ROUND_MODE_CEIL 4
#define CAST_ROUND_MODE_ROUND 5
#define CAST_ROUND_MODE_TRUNC 6
#define CAST_ROUND_MODE_ODD 7
#define CAST_ROUND_MODE_HYBRID 8

#define CAST_TPL_BF16 1
#define CAST_TPL_FLOAT16 2
#define CAST_TPL_FLOAT 3
#define CAST_TPL_HIFLOAT8 4
#define CAST_TPL_FLOAT8_E5M2 5
#define CAST_TPL_FLOAT8_E4M3FN 6
#define CAST_TPL_BOOL 7
#define CAST_TPL_INT8 8
#define CAST_TPL_INT16 9
#define CAST_TPL_INT32 10
#define CAST_TPL_INT64 11
#define CAST_TPL_UINT1 12
#define CAST_TPL_UINT8 13
#define CAST_TPL_UINT16 14
#define CAST_TPL_UINT32 15
#define CAST_TPL_COMPLEX32 16
#define CAST_TPL_COMPLEX64 17
#define CAST_TPL_FLOAT4_E1M2 18
#define CAST_TPL_FLOAT4_E2M1 19
#define CAST_TPL_DOUBLE 20
#define CAST_TPL_INT4 21
#define CAST_TPL_MAX 100

ASCENDC_TPL_ARGS_DECL(Cast, ASCENDC_TPL_UINT_DECL(id, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 0), );

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(id, ASCENDC_TPL_UI_LIST, 0), ), );

#ifdef __CCE_AICORE__
template <int ST, int DT>
struct GetCastPolicy {
    static constexpr bool isValid_ = false;
};
#endif

#ifdef __CCE_AICORE__
#define CAST_POLICY_DEFINE(STN, DTN, ST, DT, id, mapDtypeX, mapDtypeY, mapDtypeMid, castMode1, castMode2, \
                           regCopyInMode, regCopyOutMode)                                                 \
    template <>                                                                                           \
    struct GetCastPolicy<ST, DT> {                                                                        \
        static constexpr bool isValid_ = true;                                                            \
        static constexpr int id_ = id;                                                                    \
        static constexpr int srcMapType_ = mapDtypeX;                                                     \
        static constexpr int dstMapType_ = mapDtypeY;                                                     \
        static constexpr int midType_ = mapDtypeMid;                                                      \
        static constexpr int castMode1_ = castMode1;                                                      \
        static constexpr int castMode2_ = castMode2;                                                      \
        static constexpr int regCopyInMode_ = regCopyInMode;                                              \
        static constexpr int regCopyOutMode_ = regCopyOutMode;                                            \
        using srcType = STN;                                                                              \
        using dstType = DTN;                                                                              \
    };
#else
#define CAST_POLICY_DEFINE(STN, DTN, ST, DT, id, mapDtypeX, mapDtypeY, mapDtypeMid, castMode1, castMode2,           \
                           regCopyInMode, regCopyOutMode)                                                           \
    optiling::CastMapSt(ge::ST, ge::DT, id, mapDtypeX, mapDtypeY, mapDtypeMid, castMode1, castMode2, regCopyInMode, \
                        regCopyOutMode),
#endif

#ifndef __CCE_AICORE__
const optiling::CastMapSt castMap[] = {
#endif

    // CAST_TEMPLATE_DIRECT_CAST
    CAST_POLICY_DEFINE(half, float, DT_FLOAT16, DT_FLOAT, CAST_TEMPLATE_DIRECT_CAST, CAST_TPL_FLOAT16, CAST_TPL_FLOAT,
                       CAST_TPL_MAX, CAST_ROUND_MODE_NONE, CAST_ROUND_MODE_NONE, CAST_MODE_BUTT, CAST_MODE_BUTT)
        CAST_POLICY_DEFINE(half, int32_t, DT_FLOAT16, DT_INT32, CAST_TEMPLATE_DIRECT_CAST, CAST_TPL_FLOAT16,
                           CAST_TPL_INT32, CAST_TPL_MAX, CAST_ROUND_MODE_TRUNC, CAST_ROUND_MODE_NONE, CAST_MODE_BUTT,
                           CAST_MODE_BUTT) CAST_POLICY_DEFINE(half, int8_t, DT_FLOAT16, DT_INT8,
                                                              CAST_TEMPLATE_DIRECT_CAST, CAST_TPL_FLOAT16,
                                                              CAST_TPL_INT8, CAST_TPL_MAX, CAST_ROUND_MODE_TRUNC,
                                                              CAST_ROUND_MODE_NONE, CAST_MODE_BUTT, CAST_MODE_BUTT)
            CAST_POLICY_DEFINE(
                float, half, DT_FLOAT, DT_FLOAT16, CAST_TEMPLATE_DIRECT_CAST, CAST_TPL_FLOAT, CAST_TPL_FLOAT16,
                CAST_TPL_MAX, CAST_ROUND_MODE_RINT, CAST_ROUND_MODE_NONE, CAST_MODE_BUTT,
                CAST_MODE_BUTT) CAST_POLICY_DEFINE(float, int32_t, DT_FLOAT, DT_INT32, CAST_TEMPLATE_DIRECT_CAST,
                                                   CAST_TPL_FLOAT, CAST_TPL_INT32, CAST_TPL_MAX, CAST_ROUND_MODE_TRUNC,
                                                   CAST_ROUND_MODE_NONE, CAST_MODE_BUTT, CAST_MODE_BUTT)
                CAST_POLICY_DEFINE(float, int8_t, DT_FLOAT, DT_INT8, CAST_TEMPLATE_TWO_CAST, CAST_TPL_FLOAT,
                                   CAST_TPL_INT8, CAST_TPL_FLOAT16, CAST_ROUND_MODE_TRUNC, CAST_ROUND_MODE_TRUNC,
                                   CAST_MODE_BUTT, CAST_MODE_BUTT)
                    CAST_POLICY_DEFINE(int8_t, half, DT_INT8, DT_FLOAT16, CAST_TEMPLATE_DIRECT_CAST, CAST_TPL_INT8,
                                       CAST_TPL_FLOAT16, CAST_TPL_MAX, CAST_ROUND_MODE_NONE, CAST_ROUND_MODE_NONE,
                                       CAST_MODE_BUTT, CAST_MODE_BUTT)
                        CAST_POLICY_DEFINE(int8_t, int32_t, DT_INT8, DT_INT32, CAST_TEMPLATE_DIRECT_CAST, CAST_TPL_INT8,
                                           CAST_TPL_INT32, CAST_TPL_MAX, CAST_ROUND_MODE_NONE, CAST_ROUND_MODE_NONE,
                                           CAST_MODE_BUTT, CAST_MODE_BUTT)
                            CAST_POLICY_DEFINE(int32_t, half, DT_INT32, DT_FLOAT16, CAST_TEMPLATE_DIRECT_CAST,
                                               CAST_TPL_INT32, CAST_TPL_FLOAT16, CAST_TPL_MAX, CAST_ROUND_MODE_RINT,
                                               CAST_ROUND_MODE_NONE, CAST_MODE_BUTT, CAST_MODE_BUTT)
                                CAST_POLICY_DEFINE(int32_t, float, DT_INT32, DT_FLOAT, CAST_TEMPLATE_DIRECT_CAST,
                                                   CAST_TPL_INT32, CAST_TPL_FLOAT, CAST_TPL_MAX, CAST_ROUND_MODE_RINT,
                                                   CAST_ROUND_MODE_NONE, CAST_MODE_BUTT, CAST_MODE_BUTT)
                                    CAST_POLICY_DEFINE(int32_t, int8_t, DT_INT32, DT_INT8, CAST_TEMPLATE_TWO_CAST,
                                                       CAST_TPL_INT32, CAST_TPL_INT8, CAST_TPL_FLOAT16,
                                                       CAST_ROUND_MODE_RINT, CAST_ROUND_MODE_TRUNC, CAST_MODE_BUTT,
                                                       CAST_MODE_BUTT)
                                        CAST_POLICY_DEFINE(uint8_t, half, DT_UINT8, DT_FLOAT16,
                                                           CAST_TEMPLATE_DIRECT_CAST, CAST_TPL_UINT8, CAST_TPL_FLOAT16,
                                                           CAST_TPL_MAX, CAST_ROUND_MODE_NONE, CAST_ROUND_MODE_NONE,
                                                           CAST_MODE_BUTT, CAST_MODE_BUTT)
                                            CAST_POLICY_DEFINE(uint8_t, int32_t, DT_UINT8, DT_INT32,
                                                               CAST_TEMPLATE_DIRECT_CAST, CAST_TPL_UINT8,
                                                               CAST_TPL_UINT32, CAST_TPL_MAX, CAST_ROUND_MODE_NONE,
                                                               CAST_ROUND_MODE_NONE, CAST_MODE_BUTT, CAST_MODE_BUTT)
    // CAST_TEMPLATE_TWO_CAST
    CAST_POLICY_DEFINE(int8_t, float, DT_INT8, DT_FLOAT, CAST_TEMPLATE_TWO_CAST, CAST_TPL_INT8, CAST_TPL_FLOAT,
                       CAST_TPL_FLOAT16, CAST_ROUND_MODE_NONE, CAST_ROUND_MODE_NONE, CAST_MODE_BUTT, CAST_MODE_BUTT)
        CAST_POLICY_DEFINE(uint8_t, float, DT_UINT8, DT_FLOAT, CAST_TEMPLATE_TWO_CAST, CAST_TPL_UINT8, CAST_TPL_FLOAT,
                           CAST_TPL_FLOAT16, CAST_ROUND_MODE_NONE, CAST_ROUND_MODE_NONE, CAST_MODE_BUTT, CAST_MODE_BUTT)
    // CAST_TEMPLATE_DIRECT_CAST (int32→int16新增)
    CAST_POLICY_DEFINE(int32_t, int16_t, DT_INT32, DT_INT16, CAST_TEMPLATE_DIRECT_CAST, CAST_TPL_INT32, CAST_TPL_INT16,
                       CAST_TPL_MAX, CAST_ROUND_MODE_NONE, CAST_ROUND_MODE_NONE, CAST_MODE_BUTT, CAST_MODE_BUTT)
    // int16→int8 (vcvt不支持s16s8， 改走int16->half->int8中转)
    CAST_POLICY_DEFINE(int16_t, int8_t, DT_INT16, DT_INT8, CAST_TEMPLATE_TWO_CAST, CAST_TPL_INT16, CAST_TPL_INT8,
                       CAST_TPL_FLOAT16, CAST_ROUND_MODE_NONE, CAST_ROUND_MODE_TRUNC, CAST_MODE_BUTT, CAST_MODE_BUTT)

    // ====== 后补充的 dtype 类型（基于 vcvt 指令支持列表，如不支持可删除）======
    // DIRECT_CAST (vcvt 直接支持)
    CAST_POLICY_DEFINE(float, int16_t, DT_FLOAT, DT_INT16, CAST_TEMPLATE_DIRECT_CAST, CAST_TPL_FLOAT, CAST_TPL_INT16,
                       CAST_TPL_MAX, CAST_ROUND_MODE_TRUNC, CAST_ROUND_MODE_NONE, CAST_MODE_BUTT,
                       CAST_MODE_BUTT) CAST_POLICY_DEFINE(half, uint8_t, DT_FLOAT16,
                                                          DT_UINT8, CAST_TEMPLATE_DIRECT_CAST, CAST_TPL_FLOAT16,
                                                          CAST_TPL_UINT8, CAST_TPL_MAX, CAST_ROUND_MODE_TRUNC,
                                                          CAST_ROUND_MODE_NONE, CAST_MODE_BUTT, CAST_MODE_BUTT)
        CAST_POLICY_DEFINE(half, int16_t, DT_FLOAT16, DT_INT16, CAST_TEMPLATE_DIRECT_CAST, CAST_TPL_FLOAT16,
                           CAST_TPL_INT16, CAST_TPL_MAX, CAST_ROUND_MODE_TRUNC, CAST_ROUND_MODE_NONE, CAST_MODE_BUTT,
                           CAST_MODE_BUTT) CAST_POLICY_DEFINE(uint8_t, uint16_t, DT_UINT8,
                                                              DT_UINT16, CAST_TEMPLATE_DIRECT_CAST, CAST_TPL_UINT8,
                                                              CAST_TPL_UINT16, CAST_TPL_MAX, CAST_ROUND_MODE_NONE,
                                                              CAST_ROUND_MODE_NONE, CAST_MODE_BUTT, CAST_MODE_BUTT)
            CAST_POLICY_DEFINE(uint16_t, uint8_t, DT_UINT16, DT_UINT8, CAST_TEMPLATE_DIRECT_CAST, CAST_TPL_UINT16,
                               CAST_TPL_UINT8, CAST_TPL_MAX, CAST_ROUND_MODE_NONE, CAST_ROUND_MODE_NONE, CAST_MODE_BUTT,
                               CAST_MODE_BUTT) CAST_POLICY_DEFINE(uint16_t, uint32_t, DT_UINT16,
                                                                  DT_UINT32, CAST_TEMPLATE_DIRECT_CAST, CAST_TPL_UINT16,
                                                                  CAST_TPL_UINT32, CAST_TPL_MAX, CAST_ROUND_MODE_NONE,
                                                                  CAST_ROUND_MODE_NONE, CAST_MODE_BUTT, CAST_MODE_BUTT)
                CAST_POLICY_DEFINE(int16_t, uint8_t, DT_INT16, DT_UINT8, CAST_TEMPLATE_DIRECT_CAST, CAST_TPL_INT16,
                                   CAST_TPL_UINT8, CAST_TPL_MAX, CAST_ROUND_MODE_NONE, CAST_ROUND_MODE_NONE,
                                   CAST_MODE_BUTT, CAST_MODE_BUTT)
                    CAST_POLICY_DEFINE(int16_t, uint32_t, DT_INT16, DT_UINT32, CAST_TEMPLATE_DIRECT_CAST,
                                       CAST_TPL_INT16, CAST_TPL_UINT32, CAST_TPL_MAX, CAST_ROUND_MODE_NONE,
                                       CAST_ROUND_MODE_NONE, CAST_MODE_BUTT, CAST_MODE_BUTT)
                        CAST_POLICY_DEFINE(int16_t, int32_t, DT_INT16, DT_INT32, CAST_TEMPLATE_DIRECT_CAST,
                                           CAST_TPL_INT16, CAST_TPL_INT32, CAST_TPL_MAX, CAST_ROUND_MODE_NONE,
                                           CAST_ROUND_MODE_NONE, CAST_MODE_BUTT, CAST_MODE_BUTT)
                            CAST_POLICY_DEFINE(uint32_t, uint8_t, DT_UINT32, DT_UINT8, CAST_TEMPLATE_DIRECT_CAST,
                                               CAST_TPL_UINT32, CAST_TPL_UINT8, CAST_TPL_MAX, CAST_ROUND_MODE_NONE,
                                               CAST_ROUND_MODE_NONE, CAST_MODE_BUTT, CAST_MODE_BUTT)
                                CAST_POLICY_DEFINE(uint32_t, uint16_t, DT_UINT32, DT_UINT16, CAST_TEMPLATE_DIRECT_CAST,
                                                   CAST_TPL_UINT32, CAST_TPL_UINT16, CAST_TPL_MAX, CAST_ROUND_MODE_NONE,
                                                   CAST_ROUND_MODE_NONE, CAST_MODE_BUTT, CAST_MODE_BUTT)
                                    CAST_POLICY_DEFINE(uint32_t, int16_t, DT_UINT32, DT_INT16,
                                                       CAST_TEMPLATE_DIRECT_CAST, CAST_TPL_UINT32, CAST_TPL_INT16,
                                                       CAST_TPL_MAX, CAST_ROUND_MODE_NONE, CAST_ROUND_MODE_NONE,
                                                       CAST_MODE_BUTT, CAST_MODE_BUTT)
                                        CAST_POLICY_DEFINE(int32_t, uint8_t, DT_INT32, DT_UINT8,
                                                           CAST_TEMPLATE_DIRECT_CAST, CAST_TPL_INT32, CAST_TPL_UINT8,
                                                           CAST_TPL_MAX, CAST_ROUND_MODE_NONE, CAST_ROUND_MODE_NONE,
                                                           CAST_MODE_BUTT, CAST_MODE_BUTT)
                                            CAST_POLICY_DEFINE(int32_t, uint16_t, DT_INT32, DT_UINT16,
                                                               CAST_TEMPLATE_DIRECT_CAST, CAST_TPL_INT32,
                                                               CAST_TPL_UINT16, CAST_TPL_MAX, CAST_ROUND_MODE_NONE,
                                                               CAST_ROUND_MODE_NONE, CAST_MODE_BUTT, CAST_MODE_BUTT)
// ====== 后补充 dtype 结束 ======

#ifndef __CCE_AICORE__
};
#endif

#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_CAST_STRUCT_H
