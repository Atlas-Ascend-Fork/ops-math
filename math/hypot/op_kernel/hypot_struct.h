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
 * \file hypot_struct.h
 * \brief hypot kernel implementation (struct-based): FP32 stable main chain + special-value fixes;
 *        OneDim / MultiDimBroadcast (compact continuous UB layout); FP16/BF16 Cast to FP32 and back
 *        (fp32->bf16 CAST_RINT). Broadcast expansion is vectorized per host mode (ROW/COL static
 *        Broadcast when aligned, padded-window engine otherwise); all UB bases stay 32B aligned.
 *        The shared tiling data struct / scene macros live in hypot_tiling_data.h.
 */

#ifndef HYPOT_STRUCT_H
#define HYPOT_STRUCT_H

#include "kernel_operator.h"
#include "adv_api/broadcast/broadcast.h"
#include "hypot_tiling_data.h"

namespace NsHypot {
using namespace AscendC;

constexpr uint32_t HYPOT_BUFFER_NUM = 2;  // double buffer
constexpr uint32_t HYPOT_BLK_ELEMS = 8;   // floats per 32B block
constexpr uint32_t HYPOT_CMP_ALIGN = 64;  // Compare/Compares count 256B alignment (float: 64 elems)
constexpr uint32_t HYPOT_BUF_SLACK = 128; // per-buffer slack elements absorbing padded compute count
constexpr uint32_t HYPOT_COPY_CHUNK = 64; // max floats per UB-UB Copy repeat (256B)
constexpr uint32_t HYPOT_MAX_DIM = 8;
constexpr uint32_t HYPOT_MAX_STAGE = 5;           // GENERIC: <=4 inner broadcast runs + 1 split-axis repeat
constexpr uint32_t HYPOT_BRC_TMP_BYTES = 8192;    // brcTmp total (static Broadcast tmp + patterns)
constexpr uint32_t HYPOT_PAT_X2_WORD_OFF = 192;   // x2 pattern word offset inside brcTmp (768B)
constexpr uint32_t HYPOT_BRC_TMP_BYTE_OFF = 1536; // static Broadcast tmpBuffer byte offset
constexpr uint32_t HYPOT_GM_CHUNK = 255;          // Copy/GatherMask repeatTimes HW limit
constexpr uint32_t HYPOT_CROP_REP = 8; // GatherMask crop repeats per call (dst advance 8*I stays 32B-aligned)
// small-tile lightweight expansion dispatch threshold (disabled when 0)
constexpr uint32_t HYPOT_SMALL_TILE_ELEMS = 0;
constexpr uint32_t HYPOT_MASK_SLACK_BYTES = 64; // mask buffer byte slack (bytes, unlike HYPOT_BUF_SLACK elements)
constexpr uint32_t HYPOT_SCALAR_BUF_BYTES = 64; // bufScalar staging buffer size in bytes (two 32B slots)
constexpr uint32_t HYPOT_TINY_BYTES = 163840;   // <=4096-element single-core direct path
constexpr uint32_t HYPOT_TINY_MASK_BYTES = 640; // 4096 compare bits + alignment slack
// VECIDX scene workspace carving (inside bufTiny): one slot = 1088 fp32 elements (1024 + 64
// slack, 32B-aligned), 8 fp32 slots, then two raw-dtype staging slots for half dtypes.
constexpr uint32_t HYPOT_VECIDX_MAX_OUT = 1024;
constexpr uint32_t HYPOT_VECIDX_SLOT = 1088;                           // fp32 elements per slot
constexpr uint32_t HYPOT_VECIDX_RAW_OFF = 9U * HYPOT_VECIDX_SLOT * 4U; // byte offset of raw staging
constexpr uint32_t HYPOT_VECIDX_RAW_SLOT = HYPOT_VECIDX_SLOT * 2U;     // bytes per raw slot
constexpr uint32_t HYPOT_TINY_CROSS_CAP = 4096; // padded element capacity of the cross-3D fast path
constexpr uint32_t HYPOT_TINY_CROSS_RAW_OFF = 0;
constexpr uint32_t HYPOT_TINY_CROSS_A_OFF = 16384;
constexpr uint32_t HYPOT_TINY_CROSS_B_OFF = 32768;
constexpr uint32_t HYPOT_TINY_CROSS_SCRATCH_OFF = 49152;
constexpr uint32_t HYPOT_TINY_CROSS_Y_OFF = 65536;
constexpr uint32_t HYPOT_TINY_CROSS_MX_OFF = 81920;
constexpr bool HYPOT_ENABLE_TINY_CROSS3D = true;
constexpr bool HYPOT_ENABLE_TINY_PATTERN = false;
constexpr bool HYPOT_ENABLE_SMALL_TINY_PATTERN = true;
constexpr bool HYPOT_ENABLE_NATIVE_GATHER = false;
constexpr uint32_t HYPOT_GATHER_SRC_MAX = 2048;
constexpr uint32_t HYPOT_GATHER_IDX_CAP = 1024;
constexpr uint32_t HYPOT_PAT_WORD_BITS = 32;          // elements covered by one uint32 GatherMask pattern word
constexpr uint32_t HYPOT_BF16_SHIFT_BITS = 16;        // bf16 is the high 16 bits of fp32 (bit-pattern widening shift)
constexpr uint32_t HYPOT_QNAN_BITS = 0x7FC00000U;     // IEEE fp32 quiet NaN bit pattern
constexpr uint32_t HYPOT_POS_INF_BITS = 0x7F800000U;  // IEEE fp32 +inf bit pattern
constexpr uint32_t HYPOT_FP32_MAX_BITS = 0x7F7FFFFFU; // IEEE fp32 max finite value
constexpr float HYPOT_FP32_MIN_DENORM = 1.401298464324817e-45f;
constexpr float HYPOT_FP16_MIN_DENORM = 5.960464477539063e-08f;

// NaN fix-chain Select trimming: the FP32 main chain already propagates NaN, so the two explicit
// NaN Compare/Select steps are redundant; the +0 and inf fixes are kept. Set false for the full chain.
constexpr bool HYPOT_TRIM_NAN_FIX = true;

// per-input broadcast expansion plan, built once at Init from Host-precomputed mode + [L,B,I]
// (GENERIC stage list derived from strides; all fields launch-constant, R-dependence resolved per tile)
struct HypotBrcPlan {
    uint32_t mode = 0; // HYPOT_BRC_NONE/ROW/COL/GENERIC
    uint32_t L = 1;
    uint32_t B = 1;
    uint32_t I = 1;
    bool fullSplit = false; // strides[ubSplitAxis_] != 0
    // single-run engine (ROW/COL non-aligned + runs==1 GENERIC fold):
    uint32_t np = 8;     // padded row pitch = CeilAlign(I, 8)
    uint32_t window = 0; // padded window rows (multiple of 8); 0 -> degenerate scalar
    bool useEngine = false;
    bool useStatic = false; // static Broadcast fast path
    // GENERIC runs>=2 staged pipeline:
    uint32_t numStages = 0;              // inner runs in process order (innermost first)
    uint32_t stageB[HYPOT_MAX_STAGE];    // repeat count per stage
    uint32_t stageSeg[HYPOT_MAX_STAGE];  // compact segment elems per stage (multiple of tIn)
    uint32_t stageFpre[HYPOT_MAX_STAGE]; // G per tile = (fullSplit ? R : 1) * fpre
    uint32_t tIn = 1;                    // product of dims after the innermost run (stage-1 SEG)
    uint32_t npG = 8;                    // CeilAlign(tIn, 8): padded pitch for the padded route
    uint32_t genericRoute = 0;           // 0=n/a 1=compact 2=padded 3=degenerate scalar
    bool foldUnit = false;               // GENERIC runs==1 (!fullSplit && L>1): unit + split repeat
};

// KernelHypot<T1, T2, T3, SCENE>: T1/T2/T3 are the raw input/output dtypes (may differ); all compute
// runs on the FP32 chain with Cast around it. STRUCT/STREAM/VECIDX are same-dtype only (host-gated).
template <typename T1, typename T2, typename T3, uint32_t SCENE>
class KernelHypot {
public:
    // input-1 view of the dtype flags; in the same-dtype-only regions T1 == T2 == T3 holds, so
    // IS_FP32/IS_HALF/IS_BF16 keep their original meaning there
    static constexpr bool IS_FP32 = IsSameType<T1, float>::value;
    static constexpr bool IS_HALF = IsSameType<T1, half>::value;
    static constexpr bool IS_BF16 = IsSameType<T1, bfloat16_t>::value;
    static constexpr bool IN2_FP32 = IsSameType<T2, float>::value;
    static constexpr bool OUT_FP32 = IsSameType<T3, float>::value;
    static constexpr bool SAME_DTYPE = IsSameType<T1, T2>::value && IsSameType<T2, T3>::value;
    // any input needs an FP32 cast staging buffer
    static constexpr bool ANY_HALF_IN = !IS_FP32 || !IN2_FP32;
    // fp32 -> bf16 requires CAST_RINT (CAST_ODD/CAST_NONE unsupported)
    static constexpr RoundMode CAST_BACK_MODE = IsSameType<T3, bfloat16_t>::value ? RoundMode::CAST_RINT :
                                                                                    RoundMode::CAST_NONE;
    static constexpr uint32_t RAW_BLK_ELEMS = 32 / sizeof(T1); // raw-dtype elements per 32B block
    static constexpr uint32_t RAW_BLK_ELEMS2 = 32 / sizeof(T2);
    // scene family helpers: MULTI_DIM_BRC and MULTI_DIM_STRUCT share the whole MultiDimBroadcast
    // machinery; STRUCT only adds the isolated structured small-shape extensions on top.
    static constexpr bool MULTI_DIM_FAMILY = (SCENE != HYPOT_SCENE_ONE_DIM);
    static constexpr bool STRUCT_EXT = (SCENE == HYPOT_SCENE_MULTI_DIM_STRUCT);
    // STREAM scene: large same-dtype half 2D single-sided broadcast. Compact per-tile pipeline
    // (no expansion engine, no expA/expB/brcTmp buffers) over the same FP32 compute chain.
    static constexpr bool STREAM_EXT = (SCENE == HYPOT_SCENE_MULTI_DIM_STREAM);
    // VECIDX scene: small (<=1024) same-dtype multi-dim broadcast, single core, whole output in
    // UB, Gather expansion over vectorized index construction (no per-element scalar loops).
    static constexpr bool VECIDX_EXT = (SCENE == HYPOT_SCENE_MULTI_DIM_VECIDX);

    __aicore__ inline KernelHypot() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const HypotTilingData* tilingData)
    {
        x1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ T1*>(x1));
        x2Gm.SetGlobalBuffer(reinterpret_cast<__gm__ T2*>(x2));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T3*>(y));

        totalLength_ = tilingData->totalLength;
        blockIdx_ = GetBlockIdx();
        usedCoreNum_ = tilingData->usedCoreNum;
        blockFormer_ = tilingData->blockFormer;
        blockTail_ = tilingData->blockTail;
        ubFormer_ = tilingData->ubFormer;
        ubTail_ = tilingData->ubTail;
        ubOuter_ = tilingData->ubOuter;
        scalarFlag_ = tilingData->scalarFlag;
        x1Numel_ = tilingData->x1Numel;
        x2Numel_ = tilingData->x2Numel;
        x1PreloadLen_ = tilingData->x1PreloadLen;
        x2PreloadLen_ = tilingData->x2PreloadLen;
        nativeHalfEnabled_ = tilingData->nativeHalfFlag != 0U;
        nativeHalfEnabled_ = false;
        tileNum_ = (blockIdx_ == usedCoreNum_ - 1) ? blockTail_ : blockFormer_;
        startTile_ = static_cast<uint64_t>(blockIdx_) * blockFormer_;

        if constexpr (MULTI_DIM_FAMILY) {
            dimNum_ = tilingData->dimNum;
            ubSplitAxis_ = tilingData->ubSplitAxis;
            dimProduct_ = tilingData->dimProduct;
            innerLen_ = tilingData->innerLen;
            // rowPitch is unused (compact continuous layout); field kept for ABI only
            x1BrcMode_ = tilingData->x1BrcMode;
            x2BrcMode_ = tilingData->x2BrcMode;
            for (uint32_t i = 0; i < dimNum_; i++) {
                outDims_[i] = tilingData->outShape[i];
                x1Strides_[i] = tilingData->x1Strides[i];
                x2Strides_[i] = tilingData->x2Strides[i];
                outStrides_[i] = tilingData->outStrides[i];
            }
            uint32_t derivedInnerLen = 1U;
            for (uint32_t i = ubSplitAxis_ + 1; i < dimNum_; i++) {
                derivedInnerLen *= outDims_[i];
            }
            innerLen_ = derivedInnerLen;
            if constexpr (VECIDX_EXT) {
                // VECIDX: the whole workspace lives in bufTiny; return before the tiny192 gate so a 192-element
                // dimNum==3 launch is not hijacked into the scene-1 tiny pattern path.
                pipe.InitBuffer(bufTiny, HYPOT_TINY_BYTES);
                pipe.InitBuffer(bufMask1, HYPOT_TINY_MASK_BYTES);
                pipe.InitBuffer(bufMask2, HYPOT_TINY_MASK_BYTES);
                return;
            }
            // The 192-element single-core 3D family reuses the structured tiny-pattern expander (same-dtype only).
            constexpr uint32_t tiny192Elems = 192U;
            if (SAME_DTYPE && usedCoreNum_ == 1U && scalarFlag_ == 0U && totalLength_ == tiny192Elems &&
                x1Numel_ <= tiny192Elems && x2Numel_ <= tiny192Elems && dimNum_ == 3U &&
                HYPOT_ENABLE_SMALL_TINY_PATTERN && TinyPatternSupported()) {
                tinyMode_ = true;
                pipe.InitBuffer(bufTiny, HYPOT_TINY_BYTES);
                pipe.InitBuffer(bufMask1, HYPOT_TINY_MASK_BYTES);
                pipe.InitBuffer(bufMask2, HYPOT_TINY_MASK_BYTES);
                return;
            }
            if constexpr (STRUCT_EXT) {
                // Cross-3D tiny route (STRUCT scene): [R,1,I] x [1,A,1] double-sided broadcast, totalLength <= 4096.
                if (scalarFlag_ == 0U && HYPOT_ENABLE_TINY_CROSS3D && TinyCross3DSupported()) {
                    tinyMode_ = true;
                    tinyCrossMode_ = true;
                    pipe.InitBuffer(bufTiny, HYPOT_TINY_BYTES);
                    pipe.InitBuffer(bufMask1, HYPOT_TINY_MASK_BYTES);
                    pipe.InitBuffer(bufMask2, HYPOT_TINY_MASK_BYTES);
                    return;
                }
            }
            // compact continuous layout: tileElems = R * innerLen_ <= ubFormer_ * innerLen_
            bufElems_ = CeilAlign(ubFormer_ * innerLen_, HYPOT_CMP_ALIGN) + HYPOT_BUF_SLACK;
        } else {
            bufElems_ = ubFormer_ + HYPOT_BUF_SLACK;
        }

        // Buffer model (compact continuous layout): queues hold raw-dtype flat runs; compute buffers are
        // FP32, sized by bufElems_ = CeilAlign(ubFormer_*innerLen_,64)+64.
        const uint32_t q1Bytes = bufElems_ * sizeof(T1);
        const uint32_t q2Bytes = bufElems_ * sizeof(T2);
        const uint32_t qyBytes = bufElems_ * sizeof(T3);
        const uint32_t compBytes = bufElems_ * sizeof(float);
        const uint32_t oneScalarFlags = scalarFlag_ & (HYPOT_SCALAR_FLAG_X1 | HYPOT_SCALAR_FLAG_X2);
        const bool oneDimOneScalar = SCENE == HYPOT_SCENE_ONE_DIM &&
                                     (oneScalarFlags == HYPOT_SCALAR_FLAG_X1 || oneScalarFlags == HYPOT_SCALAR_FLAG_X2);
        if constexpr (SCENE == HYPOT_SCENE_ONE_DIM) {
            if (oneScalarFlags == HYPOT_SCALAR_FLAG_X1) {
                pipe.InitBuffer(inQueueX2, HYPOT_BUFFER_NUM, q2Bytes);
            } else if (oneScalarFlags == HYPOT_SCALAR_FLAG_X2) {
                pipe.InitBuffer(inQueueX1, HYPOT_BUFFER_NUM, q1Bytes);
            } else {
                pipe.InitBuffer(inQueueX1, HYPOT_BUFFER_NUM, q1Bytes);
                pipe.InitBuffer(inQueueX2, HYPOT_BUFFER_NUM, q2Bytes);
            }
        } else if constexpr (STREAM_EXT) {
            // STREAM: the full side stages through its input queue; the broadcast side needs a queue only for
            // the ROW-periodic variant (COL-single splats the preloaded per-row scalar).
            if (tilingData->x1BrcMode != HYPOT_BRC_COL) {
                pipe.InitBuffer(inQueueX1, tilingData->x1BrcMode == HYPOT_BRC_ROW ? 1 : HYPOT_BUFFER_NUM, q1Bytes);
            }
            if (tilingData->x2BrcMode != HYPOT_BRC_COL) {
                pipe.InitBuffer(inQueueX2, tilingData->x2BrcMode == HYPOT_BRC_ROW ? 1 : HYPOT_BUFFER_NUM, q2Bytes);
            }
        } else {
            pipe.InitBuffer(inQueueX1, HYPOT_BUFFER_NUM, q1Bytes);
            pipe.InitBuffer(inQueueX2, HYPOT_BUFFER_NUM, q2Bytes);
        }
        pipe.InitBuffer(outQueueY, HYPOT_BUFFER_NUM, qyBytes);
        pipe.InitBuffer(bufMx, compBytes);
        if (!oneDimOneScalar) {
            pipe.InitBuffer(bufChain, compBytes);
        }
        // Compare/Compares write one bit per element; round up and keep 32B aligned
        const uint32_t maskBytes = (bufElems_ / HYPOT_BLK_ELEMS) + HYPOT_MASK_SLACK_BYTES;
        pipe.InitBuffer(bufMask1, maskBytes);
        pipe.InitBuffer(bufMask2, maskBytes);
        if constexpr (ANY_HALF_IN) {
            // FP32 cast staging for the non-FP32 inputs (mixed launches allocate both: the budget
            // model prices mixed at the FP32 per-element constants, which cover the extra buffer)
            pipe.InitBuffer(bufA, compBytes);
            // STREAM-COL runs the scalar-b chain and never materializes a b tensor
            const bool streamNeedsBufB = STREAM_EXT && (x1BrcMode_ == HYPOT_BRC_ROW || x2BrcMode_ == HYPOT_BRC_ROW);
            if (!oneDimOneScalar && (!STREAM_EXT || streamNeedsBufB)) {
                pipe.InitBuffer(bufB, compBytes);
            }
        } else if constexpr (SCENE == HYPOT_SCENE_ONE_DIM) {
            if (scalarFlag_ != 0U) {
                pipe.InitBuffer(bufA, compBytes);
            }
            if ((scalarFlag_ & HYPOT_SCALAR_FLAG_X1) != 0U && (scalarFlag_ & HYPOT_SCALAR_FLAG_X2) != 0U) {
                pipe.InitBuffer(bufB, compBytes);
            }
        }
        if constexpr (ANY_HALF_IN && !STREAM_EXT) {
            pipe.InitBuffer(bufDenom, compBytes);
        }
        if constexpr (MULTI_DIM_FAMILY && !STREAM_EXT) {
            pipe.InitBuffer(bufExpA, compBytes);
            pipe.InitBuffer(bufExpB, compBytes);
            // brcTmp: static Broadcast sharedTmpBuffer (explicit overload mandatory)
            //         + GatherMask pattern words for the padded engine (built once below)
            pipe.InitBuffer(bufBrcTmp, HYPOT_BRC_TMP_BYTES);
            LocalTensor<uint8_t> brcAll = bufBrcTmp.Get<uint8_t>();
            brcTmpU8 = brcAll[HYPOT_BRC_TMP_BYTE_OFF];
            LocalTensor<uint32_t> patAll = bufBrcTmp.Get<uint32_t>();
            patX1 = patAll;
            patX2 = patAll[HYPOT_PAT_X2_WORD_OFF];
            BuildBrcPlan<T1>(tilingData->x1BrcMode, tilingData->x1LeadInner, tilingData->x1BrcInner,
                             tilingData->x1SrcInner, x1Strides_, x1Plan_);
            BuildBrcPlan<T2>(tilingData->x2BrcMode, tilingData->x2LeadInner, tilingData->x2BrcInner,
                             tilingData->x2SrcInner, x2Strides_, x2Plan_);
            singleSidedBroadcast_ = (x1Plan_.mode == HYPOT_BRC_NONE) != (x2Plan_.mode == HYPOT_BRC_NONE);
            BuildPatterns();
        }
        pipe.InitBuffer(bufScalar, HYPOT_SCALAR_BUF_BYTES);
        if (x1PreloadLen_ > 0) {
            pipe.InitBuffer(bufPreX1, CeilAlign(x1PreloadLen_ * sizeof(T1), 32U));
        }
        if (x2PreloadLen_ > 0) {
            pipe.InitBuffer(bufPreX2, CeilAlign(x2PreloadLen_ * sizeof(T2), 32U));
        }
        if (x1PreloadLen_ > 0) {
            LoadPreloadSource(x1Gm, bufPreX1.Get<T1>(), x1PreloadLen_);
        }
        if (x2PreloadLen_ > 0) {
            LoadPreloadSource(x2Gm, bufPreX2.Get<T2>(), x2PreloadLen_);
        }
        if (x1PreloadLen_ > 0 || x2PreloadLen_ > 0) {
            PipeBarrier<PIPE_ALL>();
        }

        // scalar input value preload (single GM read at init; scalar strides are all 0)
        bool needScalar = false;
        if ((scalarFlag_ & HYPOT_SCALAR_FLAG_X1) != 0U) {
            CopyScalarToUb(x1Gm, 0U);
            needScalar = true;
        }
        if ((scalarFlag_ & HYPOT_SCALAR_FLAG_X2) != 0U) {
            CopyScalarToUb(x2Gm, RAW_BLK_ELEMS2);
            needScalar = true;
        }
        if (needScalar) {
            PipeBarrier<PIPE_ALL>();
            if ((scalarFlag_ & HYPOT_SCALAR_FLAG_X1) != 0U) {
                scalarX1_ = ReadScalarAsFloat<T1>(0);
            }
            if ((scalarFlag_ & HYPOT_SCALAR_FLAG_X2) != 0U) {
                scalarX2_ = ReadScalarAsFloat<T2>(RAW_BLK_ELEMS2);
            }
        }
    }

    __aicore__ inline void Process()
    {
        if (tileNum_ == 0 || totalLength_ == 0) {
            return;
        }
        // tiny fast paths are same-dtype only; for mixed launches tinyMode_ is never set (see Init)
        if constexpr (SAME_DTYPE) {
            if (tinyMode_) {
                if constexpr (STRUCT_EXT) {
                    // STRUCT scene reaches tinyMode_ only through the cross-3D route in Init
                    ProcessTinyCross3DSharded();
                } else if (tinyCrossMode_) {
                    ProcessTiny2DGeneric();
                } else {
                    ProcessTiny();
                }
                return;
            }
        }
        if constexpr (VECIDX_EXT) {
            ProcessVecIdx();
            return;
        }
        if constexpr (STREAM_EXT) {
            ProcessMultiDimStream();
            return;
        }
        if constexpr (SCENE == HYPOT_SCENE_ONE_DIM) {
            ProcessOneDim();
        } else {
            ProcessMultiDim();
        }
    }

private:
    static constexpr uint32_t COMPUTE_ALIGN = 256U / sizeof(T1); // native fp16 chain only (same-dtype)

    template <typename W1, typename W2>
    __aicore__ inline W1 CeilDiv(W1 a, W2 b) const
    {
        return (a + b - 1) / b;
    }
    template <typename W1, typename W2>
    __aicore__ inline W1 CeilAlign(W1 a, W2 b) const
    {
        return (a + b - 1) / b * b;
    }

    // scalar slots are staged raw in bufScalar at 32B-aligned positions; read back as float.
    // The bisheng backend does not support scalar bf16 type casts, so bf16 -> fp32 goes through
    // the bit pattern (bf16 is the high 16 bits of fp32).
    template <typename Tin>
    __aicore__ inline float ReadScalarAsFloat(uint32_t slot)
    {
        if constexpr (IsSameType<Tin, bfloat16_t>::value) {
            LocalTensor<uint16_t> sbuf16 = bufScalar.Get<uint16_t>();
            union {
                uint32_t u;
                float f;
            } cvt;
            cvt.u = static_cast<uint32_t>(sbuf16.GetValue(slot * (sizeof(Tin) / sizeof(uint16_t))))
                    << HYPOT_BF16_SHIFT_BITS;
            return cvt.f;
        } else {
            LocalTensor<Tin> sbuf = bufScalar.Get<Tin>();
            return static_cast<float>(sbuf.GetValue(slot));
        }
    }

    template <typename Tin>
    __aicore__ inline void CopyScalarToUb(GlobalTensor<Tin>& gm, uint32_t slotElems)
    {
        // stage one element; the two scalars land at 32B-aligned slots (elem 0 / 32B-offset elems —
        // 32B worth of elements for every supported dtype, which DataCopyPad's UB dst alignment
        // requires; HYPOT_BLK_ELEMS=8 would be a 16B offset for 2B dtypes)
        LocalTensor<Tin> sbuf = bufScalar.Get<Tin>();
        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = sizeof(Tin);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(sbuf[slotElems], gm[0], copyParams, {false, 0, 0, 0});
    }

    template <typename Tin>
    __aicore__ inline void LoadPreloadSource(GlobalTensor<Tin>& gm, LocalTensor<Tin> dst, uint32_t len)
    {
        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = len * sizeof(Tin);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(dst, gm[0], copyParams, {false, 0, 0, 0});
    }

    // FP32 stable main chain + order-sensitive special-value fix chain: +0 (mx==0, NaN-safe) -> NaN ->
    // inf (last, so inf overrides NaN). vsel mask bit=1 selects src0; NaN is detected via EQ(a,a)==0.
    // Buffer roles: scratch = chain intermediate, y = FP32 result, constBuf = fix-chain constants.
    __aicore__ inline void ComputeChain(LocalTensor<float>& a, LocalTensor<float>& b, LocalTensor<float>& scratch,
                                        LocalTensor<float>& y, LocalTensor<float>& constBuf, uint32_t count)
    {
        union {
            uint32_t u;
            float f;
        } nanU;
        union {
            uint32_t u;
            float f;
        } infU;
        nanU.u = HYPOT_QNAN_BITS;    // quiet NaN
        infU.u = HYPOT_POS_INF_BITS; // +inf
        [[maybe_unused]] const float nanVal = nanU.f;
        const float infVal = infU.f;

        LocalTensor<uint8_t> mask1 = bufMask1.Get<uint8_t>();
        LocalTensor<uint8_t> mask2 = bufMask2.Get<uint8_t>();

        // main chain: a=|x1|, b=|x2|, mx=max, mn=min, r=mn/mx, t=r*r, u=t+1, s=sqrt(u), y=mx*s
        Abs(a, a, count);
        Abs(b, b, count);
        Max(constBuf, a, b, count); // constBuf doubles as mx until the final Mul (see header comment)
        Min(scratch, a, b, count);
        Div(scratch, scratch, constBuf, count);
        Mul(scratch, scratch, scratch, count);
        Adds(scratch, scratch, 1.0f, count);
        Sqrt(scratch, scratch, count);
        // 1) double-zero mask while mx is still alive: a,b >= 0 so mx == 0 iff both are zero;
        //    EQ(NaN,0)=0 / EQ(inf,0)=0 keep it NaN/inf-safe
        Compares(mask1, constBuf, 0.0f, CMPMODE::EQ, count);
        Mul(y, constBuf, scratch, count); // FP16/BF16 path: y aliases scratch (in-place)

        Duplicate(constBuf, 0.0f, count);
        Select(y, mask1, constBuf, y, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        if constexpr (!HYPOT_TRIM_NAN_FIX) {
            // 2) explicit NaN fix (baseline, kept behind the trim switch): EQ(a,a)=0 exactly at NaN
            //    positions -> bit=0 selects the NaN constant (src1)
            Duplicate(constBuf, nanVal, count);
            Compare(mask1, a, a, CMPMODE::EQ, count);
            Select(y, mask1, y, constBuf, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
            Compare(mask2, b, b, CMPMODE::EQ, count);
            Select(y, mask2, y, constBuf, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        }
        // 3) inf fix executed last so +inf wins over NaN (inf priority semantics)
        Duplicate(constBuf, infVal, count);
        Compares(mask1, a, infVal, CMPMODE::EQ, count);
        Select(y, mask1, constBuf, y, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        Compares(mask2, b, infVal, CMPMODE::EQ, count);
        Select(y, mask2, constBuf, y, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    }

    // Fast stable chain: clamping the ratio denominator to the smallest normal number handles the
    // double-zero case without a Compare/Duplicate/Select triple (0 / min_normal = 0, so y = +0).
    // For nonzero inputs the clamp is the identity because max(|a|, |b|) is already normal.
    __aicore__ inline void ComputeChainFast(LocalTensor<float>& a, LocalTensor<float>& b, LocalTensor<float>& scratch,
                                            LocalTensor<float>& y, LocalTensor<float>& constBuf,
                                            LocalTensor<float>& denom, uint32_t count)
    {
        union {
            uint32_t u;
            float f;
        } maxU;
        maxU.u = HYPOT_FP32_MAX_BITS;

        Abs(a, a, count);
        Abs(b, b, count);
        Max(constBuf, a, b, count);
        Min(scratch, a, b, count);
        Duplicate(y, maxU.f, count);
        Min(y, constBuf, y, count);
        Maxs(y, y, HYPOT_FP32_MIN_DENORM, count);
        Div(scratch, scratch, y, count);
        Mul(scratch, scratch, scratch, count);
        Adds(scratch, scratch, 1.0f, count);
        Sqrt(scratch, scratch, count);
        Mul(y, constBuf, scratch, count);
    }

    // STREAM-scene-local FP32 chain for half dtypes: bit-identical to ComputeChainFastInf; the MIN_DENORM
    // clamp differs only at mx == 0 and is flushed by the cast back to FP16/BF16.
    __aicore__ inline void ComputeChainStreamInf(LocalTensor<float>& a, LocalTensor<float>& b,
                                                 LocalTensor<float>& scratch, LocalTensor<float>& mx, uint32_t count)
    {
        union {
            uint32_t u;
            float f;
        } infU;
        infU.u = HYPOT_POS_INF_BITS;

        Abs(a, a, count);
        Abs(b, b, count);
        Max(mx, a, b, count);
        Min(scratch, a, b, count);
        Maxs(mx, mx, HYPOT_FP32_MIN_DENORM, count);
        Div(scratch, scratch, mx, count);
        Mul(scratch, scratch, scratch, count);
        Adds(scratch, scratch, 1.0f, count);
        Sqrt(scratch, scratch, count);
        Mul(scratch, mx, scratch, count); // y = mx * s, in place over scratch

        // inf fix last so +inf wins over NaN (inf priority semantics), mx is dead after the Mul
        LocalTensor<uint8_t> mask1 = bufMask1.Get<uint8_t>();
        LocalTensor<uint8_t> mask2 = bufMask2.Get<uint8_t>();
        Duplicate(mx, infU.f, count);
        Compares(mask1, a, infU.f, CMPMODE::EQ, count);
        Select(scratch, mask1, mx, scratch, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        Compares(mask2, b, infU.f, CMPMODE::EQ, count);
        Select(scratch, mask2, mx, scratch, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    }

    // STREAM-COL-local FP32 chain: the broadcast side is a per-row scalar, so Maxs/Mins replace the
    // tensor Max/Min and no b tensor is materialized.
    __aicore__ inline void ComputeChainStreamColInf(LocalTensor<float>& a, float absB, bool brcInf,
                                                    LocalTensor<float>& scratch, LocalTensor<float>& mx, uint32_t count)
    {
        union {
            uint32_t u;
            float f;
        } infU;
        infU.u = HYPOT_POS_INF_BITS;
        if (brcInf) {
            Duplicate(scratch, infU.f, count);
            return;
        }
        Abs(a, a, count);
        Maxs(mx, a, absB, count);
        Mins(scratch, a, absB, count);
        Maxs(mx, mx, HYPOT_FP32_MIN_DENORM, count);
        Div(scratch, scratch, mx, count);
        Mul(scratch, scratch, scratch, count);
        Adds(scratch, scratch, 1.0f, count);
        Sqrt(scratch, scratch, count);
        Mul(scratch, mx, scratch, count); // y = mx * s, in place over scratch

        // a-side inf fix last so +inf wins over NaN (b is finite on this path)
        LocalTensor<uint8_t> mask1 = bufMask1.Get<uint8_t>();
        Duplicate(mx, infU.f, count);
        Compares(mask1, a, infU.f, CMPMODE::EQ, count);
        Select(scratch, mask1, mx, scratch, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    }

    // Fast FP32 chain with an inf-priority fix. FP16/BF16 are widened to FP32 before this call,
    // so the arithmetic accuracy remains identical to the normal stable chain while removing the
    // redundant double-zero compare/select sequence.
    __aicore__ inline void ComputeChainFastInf(LocalTensor<float>& a, LocalTensor<float>& b,
                                               LocalTensor<float>& scratch, LocalTensor<float>& y,
                                               LocalTensor<float>& constBuf, LocalTensor<float>& denom, uint32_t count)
    {
        union {
            uint32_t u;
            float f;
        } specialU;
        specialU.u = HYPOT_FP32_MAX_BITS;
        ComputeChainFast(a, b, scratch, y, constBuf, denom, count);

        LocalTensor<uint8_t> mask1 = bufMask1.Get<uint8_t>();
        LocalTensor<uint8_t> mask2 = bufMask2.Get<uint8_t>();
        specialU.u = HYPOT_POS_INF_BITS;
        Duplicate(constBuf, specialU.f, count);
        Compares(mask1, a, specialU.f, CMPMODE::EQ, count);
        Select(y, mask1, constBuf, y, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        Compares(mask2, b, specialU.f, CMPMODE::EQ, count);
        Select(y, mask2, constBuf, y, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    }

    // FP16/BF16 native vector chain: avoids the input Cast-to-FP32 buffers and the output Cast-back;
    // the max/min/ratio formulation and special-value fixes match ComputeChain.
    __aicore__ inline void ComputeChainNative(LocalTensor<T1>& a, LocalTensor<T1>& b, LocalTensor<T1>& scratch,
                                              LocalTensor<T1>& y, LocalTensor<T1>& constBuf, uint32_t count)
    {
        union {
            uint32_t u;
            float f;
        } infU;
        infU.u = HYPOT_POS_INF_BITS;

        const T1 zeroVal = static_cast<T1>(0.0f);
        const T1 infVal = static_cast<T1>(infU.f);
        const T1 minDenorm = static_cast<T1>(HYPOT_FP16_MIN_DENORM);
        union {
            uint16_t u;
            T1 f;
        } nanU;
        nanU.u = 0x7E00U; // IEEE binary16 quiet NaN
        const T1 nanVal = nanU.f;
        LocalTensor<uint8_t> mask1 = bufMask1.Get<uint8_t>();
        LocalTensor<uint8_t> mask2 = bufMask2.Get<uint8_t>();

        Abs(a, a, count);
        Abs(b, b, count);
        Max(constBuf, a, b, count);
        Min(scratch, a, b, count);
        Maxs(y, constBuf, minDenorm, count);
        Div(scratch, scratch, y, count);
        Mul(scratch, scratch, scratch, count);
        Adds(scratch, scratch, static_cast<T1>(1.0f), count);
        Sqrt(scratch, scratch, count);
        Mul(y, constBuf, scratch, count);

        Duplicate(constBuf, nanVal, count);
        Compare(mask1, a, a, CMPMODE::EQ, count);
        Select(y, mask1, y, constBuf, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        Compare(mask2, b, b, CMPMODE::EQ, count);
        Select(y, mask2, y, constBuf, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);

        Duplicate(constBuf, infVal, count);
        Compares(mask1, a, infVal, CMPMODE::EQ, count);
        Select(y, mask1, constBuf, y, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        Compares(mask2, b, infVal, CMPMODE::EQ, count);
        Select(y, mask2, constBuf, y, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    }

    // ---------------- OneDim scene ----------------
    template <typename Tin>
    __aicore__ inline void CopyInOneDim(TQue<QuePosition::VECIN, HYPOT_BUFFER_NUM>& queue, GlobalTensor<Tin>& gm,
                                        uint64_t offset, uint32_t curLen, uint32_t count, uint32_t scalarBit,
                                        LocalTensor<float>& f32Buf, LocalTensor<float>& a)
    {
        // fills `a` (FP32 compute view of this input): FP32 path returns the live queue tensor via
        // `aQueue` semantics handled by the caller; FP16/BF16 path Casts into f32Buf
        if ((scalarFlag_ & scalarBit) != 0U) {
            Duplicate(f32Buf, scalarBit == HYPOT_SCALAR_FLAG_X1 ? scalarX1_ : scalarX2_, count);
            a = f32Buf;
            return;
        }
        LocalTensor<Tin> xLocal = queue.template AllocTensor<Tin>();
        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = curLen * sizeof(Tin);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(xLocal, gm[offset], copyParams, {false, 0, 0, 0});
        queue.EnQue(xLocal);
        xLocal = queue.template DeQue<Tin>();
        if constexpr (IsSameType<Tin, float>::value) {
            a = xLocal;
        } else {
            Cast(f32Buf, xLocal, RoundMode::CAST_NONE, count);
            queue.FreeTensor(xLocal);
            a = f32Buf;
        }
    }

    __aicore__ inline void ProcessOneDim()
    {
        const uint32_t oneScalar = scalarFlag_ & (HYPOT_SCALAR_FLAG_X1 | HYPOT_SCALAR_FLAG_X2);
        if (oneScalar == HYPOT_SCALAR_FLAG_X1 || oneScalar == HYPOT_SCALAR_FLAG_X2) {
            ProcessOneDimScalar();
            return;
        }
        if constexpr (SAME_DTYPE && IS_HALF) {
            if (nativeHalfEnabled_) {
                ProcessOneDimNative();
                return;
            }
        }
        for (uint32_t t = 0; t < tileNum_; t++) {
            uint64_t gTile = startTile_ + t;
            uint32_t curLen = (gTile == ubOuter_ - 1) ? ubTail_ : ubFormer_;
            uint32_t count = CeilAlign(curLen, HYPOT_CMP_ALIGN);
            uint64_t offset = gTile * ubFormer_;

            LocalTensor<float> a;
            LocalTensor<float> b;
            LocalTensor<float> fa;
            LocalTensor<float> fb;
            if constexpr (!IS_FP32 || !IN2_FP32) {
                fa = bufA.Get<float>();
                fb = bufB.Get<float>();
            } else {
                if (scalarFlag_ != 0U) {
                    fa = bufA.Get<float>();
                    fb = (scalarFlag_ == (HYPOT_SCALAR_FLAG_X1 | HYPOT_SCALAR_FLAG_X2)) ? bufB.Get<float>() : fa;
                }
            }
            CopyInOneDim(inQueueX1, x1Gm, offset, curLen, count, HYPOT_SCALAR_FLAG_X1, fa, a);
            CopyInOneDim(inQueueX2, x2Gm, offset, curLen, count, HYPOT_SCALAR_FLAG_X2, fb, b);

            LocalTensor<float> mx = bufMx.Get<float>();
            LocalTensor<float> chain = bufChain.Get<float>();
            LocalTensor<T3> yOut = outQueueY.template AllocTensor<T3>();
            if constexpr (OUT_FP32) {
                LocalTensor<float> yF = yOut;
                ComputeChain(a, b, chain, yF, mx, count);
                if constexpr (IS_FP32) {
                    if ((scalarFlag_ & HYPOT_SCALAR_FLAG_X1) == 0U) {
                        inQueueX1.FreeTensor(a);
                    }
                }
                if constexpr (IN2_FP32) {
                    if ((scalarFlag_ & HYPOT_SCALAR_FLAG_X2) == 0U) {
                        inQueueX2.FreeTensor(b);
                    }
                }
            } else {
                LocalTensor<float> denom = bufDenom.Get<float>();
                ComputeChain(a, b, chain, chain, mx, count);
                Cast(yOut, chain, CAST_BACK_MODE, count);
            }
            outQueueY.EnQue(yOut);

            LocalTensor<T3> yLocal = outQueueY.template DeQue<T3>();
            DataCopyParams copyOutParams;
            copyOutParams.blockCount = 1;
            copyOutParams.blockLen = curLen * sizeof(T3);
            copyOutParams.srcStride = 0;
            copyOutParams.dstStride = 0;
            DataCopyPad(yGm[offset], yLocal, copyOutParams);
            outQueueY.FreeTensor(yLocal);
        }
    }

    // One-scalar OneDim path: the full side streams in its raw dtype, the scalar side is pre-widened to
    // FP32; the result is written in T3.
    template <typename Tfull>
    __aicore__ inline void ProcessOneDimScalarImpl(GlobalTensor<Tfull>& fullGm,
                                                   TQue<QuePosition::VECIN, HYPOT_BUFFER_NUM>& fullQueue,
                                                   float scalarValue)
    {
        const float absScalar = scalarValue < 0.0f ? -scalarValue : scalarValue;

        union {
            uint32_t u;
            float f;
        } infU;
        union {
            uint32_t u;
            float f;
        } scalarU;
        infU.u = HYPOT_POS_INF_BITS;
        scalarU.f = absScalar;
        const bool scalarIsNan = absScalar != absScalar;
        const bool scalarIsInf = scalarU.u == HYPOT_POS_INF_BITS;

        for (uint32_t t = 0; t < tileNum_; t++) {
            uint64_t gTile = startTile_ + t;
            uint32_t curLen = (gTile == ubOuter_ - 1) ? ubTail_ : ubFormer_;
            uint32_t count = CeilAlign(curLen, HYPOT_CMP_ALIGN);
            uint64_t offset = gTile * ubFormer_;

            LocalTensor<T3> yOut = outQueueY.template AllocTensor<T3>();
            if constexpr (IsSameType<Tfull, float>::value) {
                LocalTensor<float> yF = yOut;
                if (scalarIsInf) {
                    Duplicate(yF, infU.f, count);
                } else {
                    LocalTensor<float> full = fullQueue.template AllocTensor<float>();
                    DataCopyParams copyParams;
                    copyParams.blockCount = 1;
                    copyParams.blockLen = curLen * sizeof(Tfull);
                    copyParams.srcStride = 0;
                    copyParams.dstStride = 0;
                    DataCopyPad(full, fullGm[offset], copyParams, {false, 0, 0, 0});
                    fullQueue.EnQue(full);
                    full = fullQueue.template DeQue<float>();

                    LocalTensor<float> mx = bufMx.Get<float>();
                    LocalTensor<float> denom = bufA.Get<float>();
                    Abs(full, full, count);
                    LocalTensor<uint8_t> infMask;
                    if (scalarIsNan) {
                        infMask = bufMask1.Get<uint8_t>();
                        Compares(infMask, full, infU.f, CMPMODE::EQ, count);
                    }
                    Maxs(mx, full, absScalar, count);
                    Mins(full, full, absScalar, count);
                    Maxs(denom, mx, HYPOT_FP32_MIN_DENORM, count);
                    Div(full, full, denom, count);
                    Mul(full, full, full, count);
                    Adds(full, full, 1.0f, count);
                    Sqrt(full, full, count);
                    Mul(yF, mx, full, count);

                    if (scalarIsNan) {
                        Duplicate(denom, infU.f, count);
                        Select(yF, infMask, denom, yF, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
                    }
                    fullQueue.FreeTensor(full);
                }
            } else {
                LocalTensor<Tfull> raw = fullQueue.template AllocTensor<Tfull>();
                DataCopyParams copyParams;
                copyParams.blockCount = 1;
                copyParams.blockLen = curLen * sizeof(Tfull);
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                DataCopyPad(raw, fullGm[offset], copyParams, {false, 0, 0, 0});
                fullQueue.EnQue(raw);
                raw = fullQueue.template DeQue<Tfull>();

                LocalTensor<float> full = bufA.Get<float>();
                LocalTensor<float> mx = bufMx.Get<float>();
                LocalTensor<float> denom = bufDenom.Get<float>();
                LocalTensor<float> yF = mx;
                Cast(full, raw, RoundMode::CAST_NONE, count);
                if (scalarIsInf) {
                    Duplicate(yF, infU.f, count);
                } else {
                    LocalTensor<uint8_t> infMask;
                    Abs(full, full, count);
                    if (scalarIsNan) {
                        infMask = bufMask1.Get<uint8_t>();
                        Compares(infMask, full, infU.f, CMPMODE::EQ, count);
                    }
                    Maxs(mx, full, absScalar, count);
                    Mins(full, full, absScalar, count);
                    Maxs(denom, mx, HYPOT_FP32_MIN_DENORM, count);
                    Div(full, full, denom, count);
                    Mul(full, full, full, count);
                    Adds(full, full, 1.0f, count);
                    Sqrt(full, full, count);
                    Mul(yF, yF, full, count);
                    if (scalarIsNan) {
                        Duplicate(denom, infU.f, count);
                        Select(yF, infMask, denom, yF, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
                    }
                }
                if constexpr (OUT_FP32) {
                    // mixed launch: yF already holds the FP32 result; count is 64-element aligned
                    CopyRepeatParams outRep{1, 1, static_cast<uint16_t>(count / HYPOT_BLK_ELEMS), 0};
                    Copy<float>(yOut, yF, static_cast<uint64_t>(count), 1, outRep);
                } else {
                    Cast(yOut, yF, CAST_BACK_MODE, count);
                }
                fullQueue.FreeTensor(raw);
            }
            outQueueY.EnQue(yOut);

            LocalTensor<T3> yLocal = outQueueY.template DeQue<T3>();
            DataCopyParams copyOutParams;
            copyOutParams.blockCount = 1;
            copyOutParams.blockLen = curLen * sizeof(T3);
            copyOutParams.srcStride = 0;
            copyOutParams.dstStride = 0;
            DataCopyPad(yGm[offset], yLocal, copyOutParams);
            outQueueY.FreeTensor(yLocal);
        }
    }

    __aicore__ inline void ProcessOneDimScalar()
    {
        if ((scalarFlag_ & HYPOT_SCALAR_FLAG_X1) != 0U) {
            ProcessOneDimScalarImpl(x2Gm, inQueueX2, scalarX1_);
        } else {
            ProcessOneDimScalarImpl(x1Gm, inQueueX1, scalarX2_);
        }
    }

    __aicore__ inline LocalTensor<T1> CopyInOneDimNative(TQue<QuePosition::VECIN, HYPOT_BUFFER_NUM>& queue,
                                                         GlobalTensor<T1>& gm, uint64_t offset, uint32_t curLen,
                                                         uint32_t count, uint32_t scalarBit, LocalTensor<T1>& scalarBuf)
    {
        if ((scalarFlag_ & scalarBit) != 0U) {
            Duplicate(scalarBuf, static_cast<T1>(scalarBit == HYPOT_SCALAR_FLAG_X1 ? scalarX1_ : scalarX2_), count);
            return scalarBuf;
        }
        LocalTensor<T1> xLocal = queue.template AllocTensor<T1>();
        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = curLen * sizeof(T1);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(xLocal, gm[offset], copyParams, {false, 0, 0, 0});
        queue.EnQue(xLocal);
        return queue.template DeQue<T1>();
    }

    __aicore__ inline void ProcessOneDimNative()
    {
        for (uint32_t t = 0; t < tileNum_; t++) {
            uint64_t gTile = startTile_ + t;
            uint32_t curLen = (gTile == ubOuter_ - 1) ? ubTail_ : ubFormer_;
            uint32_t count = CeilAlign(curLen, COMPUTE_ALIGN);
            uint64_t offset = gTile * ubFormer_;

            bool ownA = (scalarFlag_ & HYPOT_SCALAR_FLAG_X1) == 0U;
            bool ownB = (scalarFlag_ & HYPOT_SCALAR_FLAG_X2) == 0U;
            LocalTensor<T1> scalarA = bufA.Get<T1>();
            LocalTensor<T1> scalarB = scalarA;
            LocalTensor<T1> a = CopyInOneDimNative(inQueueX1, x1Gm, offset, curLen, count, HYPOT_SCALAR_FLAG_X1,
                                                   scalarA);
            LocalTensor<T1> b = CopyInOneDimNative(inQueueX2, x2Gm, offset, curLen, count, HYPOT_SCALAR_FLAG_X2,
                                                   scalarB);

            LocalTensor<T1> mx = bufMx.Get<T1>();
            LocalTensor<T1> chain = bufChain.Get<T1>();
            LocalTensor<T1> yOut = outQueueY.template AllocTensor<T1>();
            ComputeChainNative(a, b, chain, yOut, mx, count);
            if (ownA) {
                inQueueX1.FreeTensor(a);
            }
            if (ownB) {
                inQueueX2.FreeTensor(b);
            }
            outQueueY.EnQue(yOut);

            LocalTensor<T1> yLocal = outQueueY.template DeQue<T1>();
            DataCopyParams copyOutParams;
            copyOutParams.blockCount = 1;
            copyOutParams.blockLen = curLen * sizeof(T1);
            copyOutParams.srcStride = 0;
            copyOutParams.dstStride = 0;
            DataCopyPad(yGm[offset], yLocal, copyOutParams);
            outQueueY.FreeTensor(yLocal);
        }
    }

    // Tile space is [outDims_[0..ubSplitAxis-1], ubOuter_] row-major: the divisor for axis i includes
    // ubOuter_, otherwise GM offsets run past the tensor on cores with startTile_ > 0.
    __aicore__ inline void BroadcastGetAxesIndices(int64_t axesIndices[], int64_t flatIdx)
    {
        int64_t dp = dimProduct_;
        for (int64_t i = 0; i < static_cast<int64_t>(ubSplitAxis_); i++) {
            dp /= outDims_[i];
            int64_t divisor = dp * static_cast<int64_t>(ubOuter_);
            axesIndices[i] = flatIdx / divisor;
            flatIdx %= divisor;
        }
        axesIndices[ubSplitAxis_] = flatIdx;
    }

    __aicore__ inline void BroadcastUpdateAxesIndices(int64_t axesIndices[])
    {
        axesIndices[ubSplitAxis_]++;
        if (axesIndices[ubSplitAxis_] >= static_cast<int64_t>(ubOuter_)) {
            axesIndices[ubSplitAxis_] = 0;
            for (int64_t i = static_cast<int64_t>(ubSplitAxis_) - 1; i >= 0; i--) {
                axesIndices[i]++;
                if (axesIndices[i] < static_cast<int64_t>(outDims_[i])) {
                    break;
                }
                axesIndices[i] = 0;
            }
        }
    }

    __aicore__ inline int64_t GetGmOffset(const int64_t axesIndices[], const uint32_t strides[])
    {
        int64_t gmOffset = 0;
        for (uint32_t i = 0; i < ubSplitAxis_; i++) {
            gmOffset += axesIndices[i] * static_cast<int64_t>(strides[i]);
        }
        gmOffset += axesIndices[ubSplitAxis_] * static_cast<int64_t>(ubFormer_) *
                    static_cast<int64_t>(strides[ubSplitAxis_]);
        return gmOffset;
    }

    // ---------------- VecIdx scene ----------------
    // Single-core whole-output broadcast for totalLength <= 1024: both inputs staged as FP32, then one
    // Gather each over vectorized index construction (fp32 reciprocal + CAST_FLOOR, exact within the
    // current host gate; do not raise the gate without integer index math).
    __aicore__ inline void ProcessVecIdx()
    {
        const uint32_t total = static_cast<uint32_t>(totalLength_);
        const uint32_t n = CeilAlign(total, HYPOT_CMP_ALIGN);
        LocalTensor<uint8_t> rawAll = bufTiny.Get<uint8_t>();
        LocalTensor<float> ws = bufTiny.Get<float>();
        LocalTensor<float> srcF1 = ws[0];
        LocalTensor<float> srcF2 = ws[HYPOT_VECIDX_SLOT];
        LocalTensor<float> remF = ws[2 * HYPOT_VECIDX_SLOT];
        LocalTensor<float> coordF = ws[3 * HYPOT_VECIDX_SLOT];
        LocalTensor<float> idxF1 = ws[4 * HYPOT_VECIDX_SLOT];
        LocalTensor<float> idxF2 = ws[5 * HYPOT_VECIDX_SLOT];
        LocalTensor<float> cF = ws[6 * HYPOT_VECIDX_SLOT];
        LocalTensor<int32_t> qI = ws[7 * HYPOT_VECIDX_SLOT].template ReinterpretCast<int32_t>();
        LocalTensor<float> tmpF = ws[8 * HYPOT_VECIDX_SLOT];
        LocalTensor<float> expA = cF;                        // gather dst after the index build
        LocalTensor<float> expB = ws[7 * HYPOT_VECIDX_SLOT]; // gather dst after the index build

        // 1) stage both sources as FP32 (MTE2 -> V sync before the Cast / Gather reads); each
        // input stages through its own raw dtype (mixed-dtype launches are supported)
        if constexpr (IS_FP32) {
            DataCopyParams cp;
            cp.blockCount = 1;
            cp.blockLen = x1Numel_ * sizeof(float);
            cp.srcStride = 0;
            cp.dstStride = 0;
            DataCopyPad(srcF1, x1Gm, cp, {false, 0, 0, 0});
        } else {
            LocalTensor<T1> x1Raw = rawAll[HYPOT_VECIDX_RAW_OFF].template ReinterpretCast<T1>();
            DataCopyParams cp;
            cp.blockCount = 1;
            cp.blockLen = x1Numel_ * sizeof(T1);
            cp.srcStride = 0;
            cp.dstStride = 0;
            DataCopyPad(x1Raw, x1Gm, cp, {false, 0, 0, 0});
        }
        if constexpr (IN2_FP32) {
            DataCopyParams cp;
            cp.blockCount = 1;
            cp.blockLen = x2Numel_ * sizeof(float);
            cp.srcStride = 0;
            cp.dstStride = 0;
            DataCopyPad(srcF2, x2Gm, cp, {false, 0, 0, 0});
        } else {
            LocalTensor<T2> x2Raw = rawAll[HYPOT_VECIDX_RAW_OFF + HYPOT_VECIDX_RAW_SLOT].template ReinterpretCast<T2>();
            DataCopyParams cp;
            cp.blockCount = 1;
            cp.blockLen = x2Numel_ * sizeof(T2);
            cp.srcStride = 0;
            cp.dstStride = 0;
            DataCopyPad(x2Raw, x2Gm, cp, {false, 0, 0, 0});
        }
        PipeBarrier<PIPE_ALL>();
        if constexpr (!IS_FP32) {
            LocalTensor<T1> x1Raw = rawAll[HYPOT_VECIDX_RAW_OFF].template ReinterpretCast<T1>();
            Cast(srcF1, x1Raw, RoundMode::CAST_NONE, CeilAlign(x1Numel_, RAW_BLK_ELEMS));
        }
        if constexpr (!IN2_FP32) {
            LocalTensor<T2> x2Raw = rawAll[HYPOT_VECIDX_RAW_OFF + HYPOT_VECIDX_RAW_SLOT].template ReinterpretCast<T2>();
            Cast(srcF2, x2Raw, RoundMode::CAST_NONE, CeilAlign(x2Numel_, RAW_BLK_ELEMS2));
        }

        // 2) fp32-domain index construction: fp32 reciprocal + CAST_FLOOR with eps = 2.5e-4 absorbs both
        //    exact-integer round-down and legit fractions; zero-stride axes are skipped per input.
        constexpr float HYPOT_VECIDX_DIV_EPS = 2.5e-4f;
        ArithProgression(remF, 0.0f, 1.0f, n);
        Duplicate(idxF1, 0.0f, n);
        Duplicate(idxF2, 0.0f, n);
        for (uint32_t k = 0; k + 1 < dimNum_; k++) {
            const uint32_t os = outStrides_[k];
            Muls(cF, remF, 1.0f / static_cast<float>(static_cast<int32_t>(os)), n);
            Adds(cF, cF, HYPOT_VECIDX_DIV_EPS, n);
            Cast(qI, cF, RoundMode::CAST_FLOOR, n);
            Cast(coordF, qI, RoundMode::CAST_NONE, n); // coord_k as exact fp32
            if (x1Strides_[k] != 0U) {
                Muls(tmpF, coordF, static_cast<float>(static_cast<int32_t>(x1Strides_[k])), n);
                Add(idxF1, idxF1, tmpF, n);
            }
            if (x2Strides_[k] != 0U) {
                Muls(tmpF, coordF, static_cast<float>(static_cast<int32_t>(x2Strides_[k])), n);
                Add(idxF2, idxF2, tmpF, n);
            }
            Muls(tmpF, coordF, static_cast<float>(static_cast<int32_t>(os)), n);
            Sub(remF, remF, tmpF, n);
        }
        const uint32_t last = dimNum_ - 1; // innermost axis: coord == rem, stride 1 when present
        if (x1Strides_[last] == 1U) {
            Add(idxF1, idxF1, remF, n);
        } else if (x1Strides_[last] != 0U) {
            Muls(tmpF, remF, static_cast<float>(static_cast<int32_t>(x1Strides_[last])), n);
            Add(idxF1, idxF1, tmpF, n);
        }
        if (x2Strides_[last] == 1U) {
            Add(idxF2, idxF2, remF, n);
        } else if (x2Strides_[last] != 0U) {
            Muls(tmpF, remF, static_cast<float>(static_cast<int32_t>(x2Strides_[last])), n);
            Add(idxF2, idxF2, tmpF, n);
        }
        // Gather wants uint32 byte offsets into the FP32 staged sources: scale in fp32 (exact),
        // then one fp32 -> int32 cast. NOTE the stride is sizeof(float) regardless of dtype - the
        // Gather source is the widened FP32 buffer, not the raw half-dtype input.
        Muls(idxF1, idxF1, static_cast<float>(static_cast<int32_t>(sizeof(float))), n);
        Muls(idxF2, idxF2, static_cast<float>(static_cast<int32_t>(sizeof(float))), n);
        // remF/coordF are dead now; their slots become the integer index vectors
        LocalTensor<int32_t> idxU1 = ws[2 * HYPOT_VECIDX_SLOT].template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> idxU2 = ws[3 * HYPOT_VECIDX_SLOT].template ReinterpretCast<int32_t>();
        // NOTE: fp32 -> int32 must use CAST_FLOOR here - CAST_NONE leaves the fp32 bit pattern
        // unconverted on this platform (idx slots kept the stale fp32 bits). Values are
        // exact non-negative integers, so FLOOR is lossless.
        Cast(idxU1, idxF1, RoundMode::CAST_FLOOR, n);
        Cast(idxU2, idxF2, RoundMode::CAST_FLOOR, n);

        // 3) one Gather per input expands to the full output
        Gather(expA, srcF1, ws[2 * HYPOT_VECIDX_SLOT].template ReinterpretCast<uint32_t>(), 0U, total);
        Gather(expB, srcF2, ws[3 * HYPOT_VECIDX_SLOT].template ReinterpretCast<uint32_t>(), 0U, total);

        // 4) mainline compute chain on the whole output (index slots are dead after the gathers)
        LocalTensor<float> scratch = ws[4 * HYPOT_VECIDX_SLOT];
        LocalTensor<float> y = ws[5 * HYPOT_VECIDX_SLOT];
        LocalTensor<float> constBuf = tmpF;
        if constexpr (OUT_FP32) {
            ComputeChainFast(expA, expB, scratch, y, constBuf, constBuf, n);
            PipeBarrier<PIPE_ALL>();
            DataCopyParams copyOut;
            copyOut.blockCount = 1;
            copyOut.blockLen = total * sizeof(float);
            copyOut.srcStride = 0;
            copyOut.dstStride = 0;
            DataCopyPad(yGm, y, copyOut);
        } else {
            ComputeChainFastInf(expA, expB, scratch, y, constBuf, constBuf, n);
            LocalTensor<T3> outT = rawAll[HYPOT_VECIDX_RAW_OFF].template ReinterpretCast<T3>();
            Cast(outT, y, CAST_BACK_MODE, n);
            PipeBarrier<PIPE_ALL>();
            DataCopyParams copyOut;
            copyOut.blockCount = 1;
            copyOut.blockLen = total * sizeof(T3);
            copyOut.srcStride = 0;
            copyOut.dstStride = 0;
            DataCopyPad(yGm, outT, copyOut);
        }
    }

    __aicore__ inline void ProcessTiny()
    {
        if constexpr (IS_FP32) {
            LocalTensor<float> workspace = bufTiny.Get<float>();
            LocalTensor<float> x1Raw = workspace[0];
            LocalTensor<float> x2Raw = workspace[4096];
            LocalTensor<float> a = workspace[8192];
            LocalTensor<float> b = workspace[12288];
            LocalTensor<float> mx = workspace[16384];
            LocalTensor<float> scratch = workspace[20480];
            LocalTensor<float> y = workspace[24576];
            LocalTensor<float> pattern = workspace[28672];
            LocalTensor<uint8_t> tmp = workspace.template ReinterpretCast<uint8_t>()[131072];

            DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.blockLen = x1Numel_ * sizeof(float);
            DataCopyPad(x1Raw, x1Gm, copyParams, {false, 0, 0, 0});
            copyParams.blockLen = x2Numel_ * sizeof(float);
            DataCopyPad(x2Raw, x2Gm, copyParams, {false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();

            ExpandTinyInput(a, x1Raw, pattern, tmp, x1Strides_, x1Numel_);
            ExpandTinyInput(b, x2Raw, pattern, tmp, x2Strides_, x2Numel_);
            PipeBarrier<PIPE_V>();

            const uint32_t count = CeilAlign(static_cast<uint32_t>(totalLength_), HYPOT_CMP_ALIGN);
            ComputeChainFastInf(a, b, scratch, y, mx, mx, count);
            PipeBarrier<PIPE_ALL>();

            copyParams.blockLen = totalLength_ * sizeof(float);
            DataCopyPad(yGm, y, copyParams);
        } else {
            LocalTensor<uint8_t> raw = bufTiny.Get<uint8_t>();
            LocalTensor<T1> x1Raw = raw[0].template ReinterpretCast<T1>();
            LocalTensor<T2> x2Raw = raw[4096].template ReinterpretCast<T2>();
            LocalTensor<float> workspace = bufTiny.Get<float>();
            LocalTensor<float> x1 = workspace[2048];
            LocalTensor<float> x2 = workspace[4096];
            LocalTensor<float> a = workspace[6144];
            LocalTensor<float> b = workspace[8192];
            LocalTensor<float> mx = workspace[10240];
            LocalTensor<float> scratch = workspace[12288];
            LocalTensor<float> y = workspace[14336];
            LocalTensor<float> pattern = workspace[16384];
            LocalTensor<uint8_t> tmp = raw[81920];

            DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.blockLen = x1Numel_ * sizeof(T1);
            DataCopyPad(x1Raw, x1Gm, copyParams, {false, 0, 0, 0});
            copyParams.blockLen = x2Numel_ * sizeof(T2);
            DataCopyPad(x2Raw, x2Gm, copyParams, {false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();

            Cast(x1, x1Raw, RoundMode::CAST_NONE, CeilAlign(x1Numel_, RAW_BLK_ELEMS));
            Cast(x2, x2Raw, RoundMode::CAST_NONE, CeilAlign(x2Numel_, RAW_BLK_ELEMS2));
            PipeBarrier<PIPE_V>();

            ExpandTinyInput(a, x1, pattern, tmp, x1Strides_, x1Numel_);
            ExpandTinyInput(b, x2, pattern, tmp, x2Strides_, x2Numel_);
            PipeBarrier<PIPE_V>();

            const uint32_t count = CeilAlign(static_cast<uint32_t>(totalLength_), HYPOT_CMP_ALIGN);
            ComputeChain(a, b, scratch, y, mx, count);
            PipeBarrier<PIPE_V>();

            LocalTensor<T3> yOut = raw[0].template ReinterpretCast<T3>();
            Cast(yOut, y, CAST_BACK_MODE, totalLength_);
            PipeBarrier<PIPE_ALL>();
            copyParams.blockLen = totalLength_ * sizeof(T3);
            DataCopyPad(yGm, yOut, copyParams);
        }
    }

    __aicore__ inline bool TinyInner3D(const uint32_t strides[], uint32_t numel) const
    {
        if (strides[2] != 1U) {
            return false;
        }
        const uint32_t outerInner = strides[0] == outDims_[2] && strides[1] == 0U && numel == outDims_[0] * outDims_[2];
        const uint32_t middleInner = strides[0] == 0U && strides[1] == outDims_[2] &&
                                     numel == outDims_[1] * outDims_[2];
        return outerInner || middleInner;
    }

    __aicore__ inline bool TinyBlock3D(const uint32_t strides[], uint32_t numel) const
    {
        if (strides[2] != 0U) {
            return false;
        }
        const bool outerActive = outDims_[0] > 1U && strides[0] != 0U;
        const bool middleActive = outDims_[1] > 1U && strides[1] != 0U;
        if (outerActive && middleActive) {
            return numel == outDims_[0] * outDims_[1];
        }
        if (outerActive) {
            return numel == outDims_[0];
        }
        if (middleActive) {
            return numel == outDims_[1];
        }
        return numel == 1U;
    }

    __aicore__ inline bool TinyCross3DSupported() const
    {
        if (dimNum_ != 3U || totalLength_ == 0U || totalLength_ > HYPOT_TINY_CROSS_CAP) {
            return false;
        }
        const uint32_t rows = outDims_[0] * outDims_[1];
        const uint32_t np = CeilAlign(outDims_[2], 32U / sizeof(T1));
        if (rows == 0U || rows > HYPOT_TINY_CROSS_CAP / np) {
            return false;
        }
        const bool x1OuterInner = TinyInner3D(x1Strides_, x1Numel_) && x1Strides_[0] != 0U;
        const bool x2OuterInner = TinyInner3D(x2Strides_, x2Numel_) && x2Strides_[0] != 0U;
        const bool x1MiddleBlock = x1Strides_[2] == 0U && x1Strides_[0] == 0U && x1Strides_[1] != 0U &&
                                   (x1Numel_ == outDims_[1] || x1Numel_ == 1U);
        const bool x2MiddleBlock = x2Strides_[2] == 0U && x2Strides_[0] == 0U && x2Strides_[1] != 0U &&
                                   (x2Numel_ == outDims_[1] || x2Numel_ == 1U);
        return (x1OuterInner && x2MiddleBlock) || (x2OuterInner && x1MiddleBlock);
    }

    // Correct compact-plane implementation of the cross-3D fast path. Each outer index is one
    // [A,I] plane built with rank-2 Broadcast calls from aligned bases; no padded row pitches are
    // fed into the compute chain and no multi-block unaligned writeback is used.
    __aicore__ inline void ProcessTiny2D()
    {
        if constexpr (IS_FP32) {
            const uint32_t R = outDims_[0];
            const uint32_t A = outDims_[1];
            const uint32_t I = outDims_[2];
            const uint32_t np = CeilAlign(I, 32U / sizeof(T1));
            const uint32_t plane = A * I;
            const uint32_t count = CeilAlign(plane, HYPOT_CMP_ALIGN);
            const bool x1OuterInner = TinyInner3D(x1Strides_, x1Numel_) && x1Strides_[0] != 0U;
            GlobalTensor<T1>& innerGm = x1OuterInner ? x1Gm : x2Gm;
            GlobalTensor<T1>& blockGm = x1OuterInner ? x2Gm : x1Gm;
            const uint32_t blockNumel = x1OuterInner ? x2Numel_ : x1Numel_;

            LocalTensor<uint8_t> rawU8 = bufTiny.Get<uint8_t>();
            LocalTensor<float> innerSrc = rawU8[HYPOT_TINY_CROSS_RAW_OFF].template ReinterpretCast<float>();
            LocalTensor<float> blockSrc = rawU8[HYPOT_TINY_CROSS_SCRATCH_OFF].template ReinterpretCast<float>();
            LocalTensor<float> aPlane = rawU8[HYPOT_TINY_CROSS_A_OFF].template ReinterpretCast<float>();
            LocalTensor<float> bPlane = rawU8[HYPOT_TINY_CROSS_B_OFF].template ReinterpretCast<float>();
            LocalTensor<float> scratch = rawU8[65536].template ReinterpretCast<float>();
            LocalTensor<float> y = rawU8[HYPOT_TINY_CROSS_Y_OFF].template ReinterpretCast<float>();
            LocalTensor<float> mx = rawU8[HYPOT_TINY_CROSS_MX_OFF].template ReinterpretCast<float>();

            DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            for (uint32_t r = 0; r < R; r++) {
                copyParams.blockLen = I * sizeof(T1);
                DataCopyPad(innerSrc[r * np], innerGm[r * I], copyParams, {false, 0, 0, 0});
            }
            copyParams.blockLen = blockNumel * sizeof(T1);
            DataCopyPad(blockSrc, blockGm, copyParams, {false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();

            uint32_t dstShape[2] = {A, I};
            uint32_t srcShape[2] = {1, I};
            uint32_t colDstShape[2] = {A, I};
            uint32_t colSrcShape[2] = {A, 1};
            if (blockNumel == A) {
                Broadcast<float, 2, 1>(bPlane, blockSrc, colDstShape, colSrcShape, brcTmpU8);
            } else {
                Duplicate(bPlane, blockSrc.GetValue(0), plane);
            }
            PipeBarrier<PIPE_V>();

            for (uint32_t r = 0; r < R; r++) {
                Broadcast<float, 2, 0>(aPlane, innerSrc[r * np], dstShape, srcShape, brcTmpU8);
                PipeBarrier<PIPE_V>();
                ComputeChainFast(aPlane, bPlane, scratch, y, mx, mx, count);
                PipeBarrier<PIPE_V>();
                copyParams.blockLen = plane * sizeof(T1);
                DataCopyPad(yGm[r * plane], y, copyParams);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void ProcessTiny2DGeneric()
    {
        const uint32_t R = outDims_[0];
        const uint32_t A = outDims_[1];
        const uint32_t I = outDims_[2];
        const uint32_t np = CeilAlign(I, HYPOT_BLK_ELEMS);
        const uint32_t rawPitch = CeilAlign(I, RAW_BLK_ELEMS);
        const uint32_t plane = A * np;
        const uint32_t count = CeilAlign(plane, HYPOT_CMP_ALIGN);
        const bool x1OuterInner = TinyInner3D(x1Strides_, x1Numel_) && x1Strides_[0] != 0U;
        GlobalTensor<T1>& innerGm = x1OuterInner ? x1Gm : x2Gm;
        GlobalTensor<T1>& blockGm = x1OuterInner ? x2Gm : x1Gm;
        const uint32_t blockNumel = x1OuterInner ? x2Numel_ : x1Numel_;

        LocalTensor<uint8_t> rawU8 = bufTiny.Get<uint8_t>();
        LocalTensor<T1> innerRaw = rawU8[0].template ReinterpretCast<T1>();
        LocalTensor<T1> blockRaw = rawU8[16384].template ReinterpretCast<T1>();
        LocalTensor<float> innerSrc = rawU8[32768].template ReinterpretCast<float>();
        LocalTensor<float> blockSrc = rawU8[49152].template ReinterpretCast<float>();
        LocalTensor<float> aPlane = rawU8[65536].template ReinterpretCast<float>();
        LocalTensor<float> bPlane = rawU8[81920].template ReinterpretCast<float>();
        LocalTensor<float> scratch = rawU8[98304].template ReinterpretCast<float>();
        LocalTensor<float> y = rawU8[114688].template ReinterpretCast<float>();
        LocalTensor<float> mx = rawU8[131072].template ReinterpretCast<float>();

        DataCopyParams rowCopy;
        rowCopy.blockCount = 1;
        rowCopy.srcStride = 0;
        rowCopy.dstStride = 0;
        rowCopy.blockLen = I * sizeof(T1);
        if constexpr (IS_FP32) {
            for (uint32_t r = 0; r < R; r++) {
                DataCopyPad(innerSrc[r * np], innerGm[r * I], rowCopy, {false, 0, 0, 0});
            }
        } else {
            for (uint32_t r = 0; r < R; r++) {
                DataCopyPad(innerRaw[r * rawPitch], innerGm[r * I], rowCopy, {false, 0, 0, 0});
            }
        }
        DataCopyParams blockCopy;
        blockCopy.blockCount = 1;
        blockCopy.srcStride = 0;
        blockCopy.dstStride = 0;
        blockCopy.blockLen = blockNumel * sizeof(T1);
        if constexpr (IS_FP32) {
            DataCopyPad(blockSrc, blockGm, blockCopy, {false, 0, 0, 0});
        } else {
            DataCopyPad(blockRaw, blockGm, blockCopy, {false, 0, 0, 0});
        }
        PipeBarrier<PIPE_ALL>();

        if constexpr (!IS_FP32) {
            for (uint32_t r = 0; r < R; r++) {
                Cast(innerSrc[r * np], innerRaw[r * rawPitch], RoundMode::CAST_NONE, np);
            }
            Cast(blockSrc, blockRaw, RoundMode::CAST_NONE, CeilAlign(blockNumel, RAW_BLK_ELEMS));
            PipeBarrier<PIPE_V>();
        }

        uint32_t planeShape[2] = {A, np};
        uint32_t innerShape[2] = {1, np};
        uint32_t blockShape[2] = {A, 1};
        if (blockNumel == A) {
            Broadcast<float, 2, 1>(bPlane, blockSrc, planeShape, blockShape, brcTmpU8);
        } else {
            Duplicate(bPlane, blockSrc.GetValue(0), plane);
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<T1> outRaw = blockRaw;
        for (uint32_t r = 0; r < R; r++) {
            Broadcast<float, 2, 0>(aPlane, innerSrc[r * np], planeShape, innerShape, brcTmpU8);
            PipeBarrier<PIPE_V>();
            if constexpr (IS_FP32) {
                ComputeChainFast(aPlane, bPlane, scratch, y, mx, mx, count);
            } else {
                ComputeChainFastInf(aPlane, bPlane, scratch, y, mx, mx, count);
            }
            PipeBarrier<PIPE_V>();
            LocalTensor<T1> out;
            if constexpr (IS_FP32) {
                out = y;
            } else {
                Cast(outRaw, y, CAST_BACK_MODE, plane);
                PipeBarrier<PIPE_V>();
                out = outRaw;
            }
            blockCopy.blockLen = I * sizeof(T1);
            for (uint32_t aIdx = 0; aIdx < A; aIdx++) {
                DataCopyPad(yGm[(r * A + aIdx) * I], out[aIdx * np], blockCopy);
            }
            PipeBarrier<PIPE_ALL>();
        }
    }

    // Cross-3D structured plane path (STRUCT only): [R,1,I] x [1,A,1], totalLength <= 4096, built from
    // Copy/Duplicate primitives; outer rows are sharded across cores.
    __aicore__ inline void ProcessTinyCross3DSharded()
    {
        const uint32_t R = outDims_[0];
        const uint32_t A = outDims_[1];
        const uint32_t I = outDims_[2];
        // Plane pitch uses the RAW-dtype block (fp16/bf16: 16 elems) so the per-row writeback base
        // out[a*np] stays 32B aligned on the UB side for every dtype (DataCopyPad hard constraint);
        // this also matches the pitch convention of TinyCross3DSupported's capacity check.
        const uint32_t np = CeilAlign(I, RAW_BLK_ELEMS);
        const uint32_t rawPitch = CeilAlign(I, RAW_BLK_ELEMS);
        const uint32_t plane = A * np;
        const uint32_t count = CeilAlign(plane, HYPOT_CMP_ALIGN);
        const bool x1OuterInner = TinyInner3D(x1Strides_, x1Numel_) && x1Strides_[0] != 0U;
        GlobalTensor<T1>& innerGm = x1OuterInner ? x1Gm : x2Gm;
        GlobalTensor<T1>& blockGm = x1OuterInner ? x2Gm : x1Gm;
        const uint32_t blockNumel = x1OuterInner ? x2Numel_ : x1Numel_;

        LocalTensor<uint8_t> rawU8 = bufTiny.Get<uint8_t>();
        LocalTensor<T1> innerRaw = rawU8[0].template ReinterpretCast<T1>();
        LocalTensor<T1> blockRaw = rawU8[16384].template ReinterpretCast<T1>();
        LocalTensor<float> innerSrc = rawU8[32768].template ReinterpretCast<float>();
        LocalTensor<float> blockSrc = rawU8[49152].template ReinterpretCast<float>();
        LocalTensor<float> aPlane = rawU8[65536].template ReinterpretCast<float>();
        LocalTensor<float> bPlane = rawU8[81920].template ReinterpretCast<float>();
        LocalTensor<float> scratch = rawU8[98304].template ReinterpretCast<float>();
        LocalTensor<float> y = rawU8[114688].template ReinterpretCast<float>();
        LocalTensor<float> mx = rawU8[131072].template ReinterpretCast<float>();

        DataCopyParams rowCopy;
        rowCopy.blockCount = 1;
        rowCopy.srcStride = 0;
        rowCopy.dstStride = 0;
        rowCopy.blockLen = I * sizeof(T1);
        if constexpr (IS_FP32) {
            for (uint32_t r = blockIdx_; r < R; r += usedCoreNum_) {
                DataCopyPad(innerSrc[r * np], innerGm[r * I], rowCopy, {false, 0, 0, 0});
            }
        } else {
            for (uint32_t r = blockIdx_; r < R; r += usedCoreNum_) {
                DataCopyPad(innerRaw[r * rawPitch], innerGm[r * I], rowCopy, {false, 0, 0, 0});
            }
        }
        DataCopyParams blockCopy;
        blockCopy.blockCount = 1;
        blockCopy.srcStride = 0;
        blockCopy.dstStride = 0;
        blockCopy.blockLen = blockNumel * sizeof(T1);
        if constexpr (IS_FP32) {
            DataCopyPad(blockSrc, blockGm, blockCopy, {false, 0, 0, 0});
        } else {
            DataCopyPad(blockRaw, blockGm, blockCopy, {false, 0, 0, 0});
        }
        PipeBarrier<PIPE_ALL>();

        if constexpr (!IS_FP32) {
            for (uint32_t r = blockIdx_; r < R; r += usedCoreNum_) {
                Cast(innerSrc[r * np], innerRaw[r * rawPitch], RoundMode::CAST_NONE, np);
            }
            Cast(blockSrc, blockRaw, RoundMode::CAST_NONE, CeilAlign(blockNumel, RAW_BLK_ELEMS));
            PipeBarrier<PIPE_V>();
        }

        // bPlane[a][:] = blockSrc[a] (column duplicate). Per-row vector Duplicate instead of the
        // static Broadcast<float,2,1> interface: srcShape[0]=A is not 32B aligned for the cross
        // family (A=4/5/7), and A duplicate calls are cheaper than the Broadcast table setup here.
        PipeBarrier<PIPE_ALL>(); // scalar GetValue reads the MTE2/Cast-written block source
        if (blockNumel == A) {
            for (uint32_t a = 0; a < A; a++) {
                Duplicate(bPlane[a * np], blockSrc.GetValue(a), np);
            }
        } else {
            Duplicate(bPlane, blockSrc.GetValue(0), plane);
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<T1> outRaw = blockRaw;
        for (uint32_t r = blockIdx_; r < R; r += usedCoreNum_) {
            // aPlane = inner row r repeated A times (UB-UB Copy repeat-source idiom, np%8==0 keeps
            // every repeat 32B aligned; RepeatBlocks chunks rep<=255 / count<=64 floats).
            LocalTensor<float> srcRow = innerSrc[r * np];
            RepeatBlocks(aPlane, srcRow, 1, A, np);
            PipeBarrier<PIPE_V>();
            if constexpr (IS_FP32) {
                ComputeChainFast(aPlane, bPlane, scratch, y, mx, mx, count);
            } else {
                ComputeChainFastInf(aPlane, bPlane, scratch, y, mx, mx, count);
            }
            PipeBarrier<PIPE_V>();
            LocalTensor<T1> out;
            if constexpr (IS_FP32) {
                out = y;
            } else {
                Cast(outRaw, y, CAST_BACK_MODE, plane);
                PipeBarrier<PIPE_V>();
                out = outRaw;
            }
            // V -> MTE3: the writeback DMA reads the vector-produced plane; a PIPE_V-only barrier
            // is not sufficient here (observed stale/zero patches in the GM output). Drain all
            // pipes before issuing the per-row DataCopyPad, same as ProcessTiny.
            PipeBarrier<PIPE_ALL>();
            blockCopy.blockLen = I * sizeof(T1);
            for (uint32_t aIdx = 0; aIdx < A; aIdx++) {
                DataCopyPad(yGm[(r * A + aIdx) * I], out[aIdx * np], blockCopy);
            }
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ProcessTinyCross3D()
    {
        const uint32_t R = outDims_[0];
        const uint32_t A = outDims_[1];
        const uint32_t I = outDims_[2];
        const uint32_t rows = R * A;
        const uint32_t np = CeilAlign(I, 32U / sizeof(T1));
        const uint32_t padded = rows * np;
        const uint32_t count = CeilAlign(padded, HYPOT_CMP_ALIGN);
        const bool x1Inner = TinyInner3D(x1Strides_, x1Numel_);
        const bool innerIsX1 = x1Inner || (x1Numel_ == totalLength_);
        const bool innerIsFull = !x1Inner && !TinyInner3D(x2Strides_, x2Numel_);
        GlobalTensor<T1>& innerGm = innerIsX1 ? x1Gm : x2Gm;
        GlobalTensor<T1>& blockGm = innerIsX1 ? x2Gm : x1Gm;
        const uint32_t* blockStrides = innerIsX1 ? x2Strides_ : x1Strides_;
        const uint32_t* innerStrides = innerIsX1 ? x1Strides_ : x2Strides_;
        const uint32_t blockNumel = innerIsX1 ? x2Numel_ : x1Numel_;
        const uint32_t innerRows = innerIsFull ? rows : (innerStrides[0] != 0U ? R : A);

        LocalTensor<uint8_t> rawU8 = bufTiny.Get<uint8_t>();
        LocalTensor<float> innerSrc = rawU8[HYPOT_TINY_CROSS_RAW_OFF].template ReinterpretCast<float>();
        LocalTensor<float> a = rawU8[HYPOT_TINY_CROSS_A_OFF].template ReinterpretCast<float>();
        LocalTensor<float> b = rawU8[HYPOT_TINY_CROSS_B_OFF].template ReinterpretCast<float>();
        LocalTensor<float> scratch = rawU8[HYPOT_TINY_CROSS_SCRATCH_OFF].template ReinterpretCast<float>();
        LocalTensor<float> y = rawU8[HYPOT_TINY_CROSS_Y_OFF].template ReinterpretCast<float>();
        LocalTensor<float> mx = rawU8[HYPOT_TINY_CROSS_MX_OFF].template ReinterpretCast<float>();
        LocalTensor<T1> rawT = rawU8[HYPOT_TINY_CROSS_RAW_OFF].template ReinterpretCast<T1>();

        if constexpr (IS_FP32) {
            DataCopyParams rowCopyParams;
            rowCopyParams.blockLen = I * sizeof(T1);
            rowCopyParams.srcStride = 0;
            rowCopyParams.dstStride = 0;
            for (uint32_t row = 0; row < innerRows; row++) {
                rowCopyParams.blockCount = 1;
                DataCopyPad(innerSrc[row * np], innerGm[row * I], rowCopyParams, {false, 0, 0, 0});
            }
        } else {
            DataCopyParams copyParams;
            copyParams.blockCount = static_cast<uint16_t>(innerRows);
            copyParams.blockLen = I * sizeof(T1);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(rawT, innerGm, copyParams, {false, 0, 0, 0});
        }

        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = blockNumel * sizeof(T1);
        if constexpr (IS_FP32) {
            DataCopyPad(scratch, blockGm, copyParams, {false, 0, 0, 0});
        } else {
            PipeBarrier<PIPE_ALL>();
            const uint32_t rawPitch = CeilAlign(I, RAW_BLK_ELEMS);
            for (uint32_t r = 0; r < innerRows; r++) {
                Cast(innerSrc[r * np], rawT[r * rawPitch], RoundMode::CAST_NONE, np);
            }
            PipeBarrier<PIPE_V>();
            DataCopyPad(rawT, blockGm, copyParams, {false, 0, 0, 0});
        }
        PipeBarrier<PIPE_ALL>();

        if constexpr (!IS_FP32) {
            Cast(scratch, rawT, RoundMode::CAST_NONE, CeilAlign(blockNumel, HYPOT_BLK_ELEMS));
            PipeBarrier<PIPE_V>();
        }

        if constexpr (IS_FP32) {
            uint32_t dstShape[2] = {A, I};
            uint32_t srcShape[2] = {1, I};
            if (innerIsFull) {
                for (uint32_t row = 0; row < rows; row++) {
                    CopyRepeatParams params{1, 1, 1, 1};
                    Copy<float>(a[row * np], innerSrc[row * np], static_cast<uint64_t>(np), 1, params);
                }
            } else if (innerStrides[0] != 0U) {
                for (uint32_t r = 0; r < R; r++) {
                    Broadcast<float, 2, 0>(a[r * A * np], innerSrc[r * np], dstShape, srcShape, brcTmpU8);
                }
            } else {
                for (uint32_t r = 0; r < R; r++) {
                    for (uint32_t aIdx = 0; aIdx < A; aIdx++) {
                        CopyRepeatParams params{1, 1, 1, 1};
                        Copy<float>(a[(r * A + aIdx) * np], innerSrc[aIdx * np], static_cast<uint64_t>(np), 1, params);
                    }
                }
            }
        } else {
            for (uint32_t row = 0; row < rows; row++) {
                CopyRepeatParams params{1, 1, 1, 1};
                Copy<float>(a[row * np], innerSrc[(row % innerRows) * np], static_cast<uint64_t>(np), 1, params);
            }
        }
        PipeBarrier<PIPE_ALL>();

        for (uint32_t r = 0; r < R; r++) {
            for (uint32_t aIdx = 0; aIdx < A; aIdx++) {
                uint32_t blockIdx = 0U;
                const bool outerActive = outDims_[0] > 1U && blockStrides[0] != 0U;
                const bool middleActive = outDims_[1] > 1U && blockStrides[1] != 0U;
                if (outerActive && middleActive) {
                    blockIdx = r * A + aIdx;
                } else if (outerActive) {
                    blockIdx = r;
                } else if (middleActive) {
                    blockIdx = aIdx;
                }
                Duplicate(b[(r * A + aIdx) * np], scratch.GetValue(blockIdx), np);
            }
        }
        PipeBarrier<PIPE_V>();

        ComputeChainFast(a, b, scratch, y, mx, mx, count);
        PipeBarrier<PIPE_V>();

        LocalTensor<T1> yOut;
        if constexpr (IS_FP32) {
            yOut = y;
        } else {
            yOut = rawT;
            Cast(yOut, y, CAST_BACK_MODE, padded);
            PipeBarrier<PIPE_V>();
        }
        if (I % HYPOT_BLK_ELEMS != 0U) {
            DataCopyParams rowParams;
            rowParams.blockCount = 1;
            rowParams.blockLen = I * sizeof(T1);
            rowParams.srcStride = 0;
            rowParams.dstStride = 0;
            for (uint32_t row = 0; row < rows; row++) {
                DataCopyPad(yGm[row * I], yOut[row * np], rowParams);
            }
            return;
        }
        DataCopyExtParams outParams;
        outParams.blockCount = static_cast<uint16_t>(rows);
        outParams.blockLen = I * sizeof(T1);
        // reached only when I % HYPOT_BLK_ELEMS == 0, so np == CeilAlign(I, 8) == I and the source
        // stride is always 0; keep the expression for the general padded-row contract
        outParams.srcStride = (np - I) * sizeof(T1);
        outParams.dstStride = 0;
        DataCopyPad(yGm, yOut, outParams);
    }

    __aicore__ inline uint32_t TinyPattern(const uint32_t strides[], uint32_t numel) const
    {
        if (numel == totalLength_) {
            return 0U;
        }
        if (numel == outDims_[1] && strides[0] == 0U && strides[1] == 1U && strides[2] == 0U) {
            return 1U;
        }
        if (numel == outDims_[0] * outDims_[2] && strides[0] == outDims_[2] && strides[1] == 0U && strides[2] == 1U) {
            return 2U;
        }
        if (numel == outDims_[0] * outDims_[1] && strides[0] == outDims_[1] && strides[1] == 1U && strides[2] == 0U) {
            return 3U;
        }
        return 4U;
    }

    __aicore__ inline bool TinyPatternSupported() const
    {
        const uint32_t x1Mode = TinyPattern(x1Strides_, x1Numel_);
        const uint32_t x2Mode = TinyPattern(x2Strides_, x2Numel_);
        return x1Mode < 4U && x2Mode < 4U && x1Mode + x2Mode > 0U && outDims_[2] % HYPOT_BLK_ELEMS == 0U;
    }

    __aicore__ inline void ExpandTinyInput(LocalTensor<float>& dst, LocalTensor<float>& src,
                                           LocalTensor<float>& pattern, LocalTensor<uint8_t>& tmp,
                                           const uint32_t strides[], uint32_t numel)
    {
        const uint32_t rows = outDims_[0];
        const uint32_t groups = outDims_[1];
        const uint32_t inner = outDims_[2];
        const uint32_t rowElems = groups * inner;
        const uint32_t mode = TinyPattern(strides, numel);
        if (totalLength_ == 192U && inner % HYPOT_BLK_ELEMS == 0U && mode < 4U) {
#define HYPOT_COPY_SMALL_BLOCK(BASE)                                                                        \
    do {                                                                                                    \
        const uint32_t sourceOffset = mode == 0U ? (BASE) : ((BASE) / rowElems) * inner + ((BASE) % inner); \
        CopyRepeatParams params{1, 1, 1, 1};                                                                \
        Copy<float>(dst[BASE], src[sourceOffset], static_cast<uint64_t>(HYPOT_BLK_ELEMS), 1, params);       \
    } while (0)
#define HYPOT_DUP_SMALL_BLOCK(BASE)                                                                              \
    do {                                                                                                         \
        const uint32_t sourceOffset = mode == 1U ? ((BASE) / inner) % groups :                                   \
                                                   ((BASE) / rowElems) * groups + (((BASE) % rowElems) / inner); \
        Duplicate(dst[BASE], src.GetValue(sourceOffset), HYPOT_BLK_ELEMS);                                       \
    } while (0)
            if (mode == 0U || mode == 2U) {
                HYPOT_COPY_SMALL_BLOCK(0);
                HYPOT_COPY_SMALL_BLOCK(8);
                HYPOT_COPY_SMALL_BLOCK(16);
                HYPOT_COPY_SMALL_BLOCK(24);
                HYPOT_COPY_SMALL_BLOCK(32);
                HYPOT_COPY_SMALL_BLOCK(40);
                HYPOT_COPY_SMALL_BLOCK(48);
                HYPOT_COPY_SMALL_BLOCK(56);
                HYPOT_COPY_SMALL_BLOCK(64);
                HYPOT_COPY_SMALL_BLOCK(72);
                HYPOT_COPY_SMALL_BLOCK(80);
                HYPOT_COPY_SMALL_BLOCK(88);
                HYPOT_COPY_SMALL_BLOCK(96);
                HYPOT_COPY_SMALL_BLOCK(104);
                HYPOT_COPY_SMALL_BLOCK(112);
                HYPOT_COPY_SMALL_BLOCK(120);
                HYPOT_COPY_SMALL_BLOCK(128);
                HYPOT_COPY_SMALL_BLOCK(136);
                HYPOT_COPY_SMALL_BLOCK(144);
                HYPOT_COPY_SMALL_BLOCK(152);
                HYPOT_COPY_SMALL_BLOCK(160);
                HYPOT_COPY_SMALL_BLOCK(168);
                HYPOT_COPY_SMALL_BLOCK(176);
                HYPOT_COPY_SMALL_BLOCK(184);
            } else {
                HYPOT_DUP_SMALL_BLOCK(0);
                HYPOT_DUP_SMALL_BLOCK(8);
                HYPOT_DUP_SMALL_BLOCK(16);
                HYPOT_DUP_SMALL_BLOCK(24);
                HYPOT_DUP_SMALL_BLOCK(32);
                HYPOT_DUP_SMALL_BLOCK(40);
                HYPOT_DUP_SMALL_BLOCK(48);
                HYPOT_DUP_SMALL_BLOCK(56);
                HYPOT_DUP_SMALL_BLOCK(64);
                HYPOT_DUP_SMALL_BLOCK(72);
                HYPOT_DUP_SMALL_BLOCK(80);
                HYPOT_DUP_SMALL_BLOCK(88);
                HYPOT_DUP_SMALL_BLOCK(96);
                HYPOT_DUP_SMALL_BLOCK(104);
                HYPOT_DUP_SMALL_BLOCK(112);
                HYPOT_DUP_SMALL_BLOCK(120);
                HYPOT_DUP_SMALL_BLOCK(128);
                HYPOT_DUP_SMALL_BLOCK(136);
                HYPOT_DUP_SMALL_BLOCK(144);
                HYPOT_DUP_SMALL_BLOCK(152);
                HYPOT_DUP_SMALL_BLOCK(160);
                HYPOT_DUP_SMALL_BLOCK(168);
                HYPOT_DUP_SMALL_BLOCK(176);
                HYPOT_DUP_SMALL_BLOCK(184);
            }
#undef HYPOT_DUP_SMALL_BLOCK
#undef HYPOT_COPY_SMALL_BLOCK
            return;
        }
        if (mode == 0U) {
            if (totalLength_ % HYPOT_BLK_ELEMS != 0U || totalLength_ / HYPOT_BLK_ELEMS > 255U) {
                uint32_t shape[3] = {rows, groups, inner};
                Broadcast<float, 3, 0>(dst, src, shape, shape, tmp);
            } else {
                CopyRepeatParams params{1, 1, 1, 1};
                Copy<float>(dst, src, static_cast<uint64_t>(HYPOT_BLK_ELEMS),
                            static_cast<uint8_t>(totalLength_ / HYPOT_BLK_ELEMS), params);
            }
        } else if (mode == 1U) {
            uint32_t midShape[3] = {1U, groups, inner};
            uint32_t srcShape[3] = {1U, groups, 1U};
            uint32_t dstShape[3] = {rows, groups, inner};
            Broadcast<float, 3, 2>(pattern, src, midShape, srcShape, tmp);
            PipeBarrier<PIPE_V>();
            Broadcast<float, 3, 0>(dst, pattern, dstShape, midShape, tmp);
        } else if (mode == 2U) {
            if (inner % HYPOT_BLK_ELEMS == 0U) {
                CopyRepeatParams params{1, 1, static_cast<uint16_t>(inner / HYPOT_BLK_ELEMS), 0};
                for (uint32_t r = 0; r < rows; r++) {
                    Copy<float>(dst[r * rowElems], src[r * inner], static_cast<uint64_t>(inner), groups, params);
                }
            } else {
                uint32_t dstShape[3] = {rows, groups, inner};
                uint32_t srcShape[3] = {rows, 1U, inner};
                Broadcast<float, 3, 1>(dst, src, dstShape, srcShape, tmp);
            }
        } else {
            uint32_t dstShape[3] = {rows, groups, inner};
            uint32_t srcShape[3] = {rows, groups, 1U};
            Broadcast<float, 3, 2>(dst, src, dstShape, srcShape, tmp);
        }
    }

    // block-repeat: dst[g*b*seg .. g*b*seg + b*seg) = b copies of src[g*seg .. g*seg + seg),
    // g in [0, G). Requires seg % 8 == 0 (Copy 32B granularity).
    __aicore__ inline void RepeatBlocks(LocalTensor<float>& dst, LocalTensor<float>& src, uint32_t G, uint32_t b,
                                        uint32_t seg)
    {
        for (uint32_t g = 0; g < G; g++) {
            uint32_t done = 0;
            while (done < b) {
                uint32_t rep = (b - done) < HYPOT_GM_CHUNK ? (b - done) : HYPOT_GM_CHUNK;
                uint32_t off = 0;
                while (off < seg) {
                    uint32_t c = (seg - off) < HYPOT_COPY_CHUNK ? (seg - off) : HYPOT_COPY_CHUNK;
                    CopyRepeatParams params{1, 1, static_cast<uint16_t>(seg / HYPOT_BLK_ELEMS), 0};
                    Copy<float>(dst[(g * b + done) * seg + off], src[g * seg + off], static_cast<uint64_t>(c), rep,
                                params);
                    off += c;
                }
                done += rep;
            }
        }
    }

    // ---- launch-constant expansion planning (Init) ----
    template <typename Tin>
    __aicore__ inline void BuildBrcPlan(uint32_t mode, uint32_t L, uint32_t B, uint32_t I, const uint32_t strides[],
                                        HypotBrcPlan& p)
    {
        p.mode = mode;
        p.L = L;
        p.B = B;
        p.I = I;
        p.fullSplit = (strides[ubSplitAxis_] != 0);
        p.np = CeilAlign(I, HYPOT_BLK_ELEMS);
        const bool tileScalar = mode != HYPOT_BRC_GENERIC && !p.fullSplit && p.L * p.I == 1;
        if (tileScalar) {
            p.useStatic = true;
            p.useEngine = false;
            return;
        }
        if (mode == HYPOT_BRC_ROW) {
            // ROW: [G,B,I] single broadcast run; G per tile = (fullSplit ? R : 1) * L
            p.useStatic = (I % HYPOT_BLK_ELEMS == 0);
            p.useEngine = !p.useStatic;
        } else if (mode == HYPOT_BRC_COL) {
            // COL: (G,1)->(G,B); whole-tile static Broadcast<float,2,1> gated per tile on
            // G%8==0 && B%8==0 (G = (fullSplit ? R : 1) * L is tile-dependent), so the engine
            // window is always precomputed for COL
            p.useStatic = p.fullSplit && (p.B % HYPOT_BLK_ELEMS == 0);
            p.useEngine = !p.useStatic;
        } else if (mode == HYPOT_BRC_GENERIC) {
            BuildGenericPlan<Tin>(strides, p);
        }
        if (p.useEngine) {
            p.window = ComputeEngineWindow<Tin>(p);
        }
    }

    // GENERIC stage list: process broadcast runs innermost -> outermost; each stage inserts run b
    // at its axis position, i.e. repeats every SEG-length block b times (SEG = product of all dims
    // after the run, G = product of non-broadcast dims before the run, times R when fullSplit).
    template <typename Tin>
    __aicore__ inline void BuildGenericPlan(const uint32_t strides[], HypotBrcPlan& p)
    {
        // collect inner broadcast runs (outermost -> innermost)
        uint32_t runB[HYPOT_MAX_STAGE];
        uint32_t runSeg[HYPOT_MAX_STAGE];
        uint32_t runFpre[HYPOT_MAX_STAGE];
        uint32_t nRuns = 0;
        uint32_t segSuffix = 1; // product of ALL dims below the current position (broadcast dims
                                // join the suffix once their run has been processed: after expansion
                                // they are part of the buffer content)
        for (int64_t k = static_cast<int64_t>(dimNum_) - 1; k > static_cast<int64_t>(ubSplitAxis_); k--) {
            if (strides[k] == 0) {
                // extend the run upwards
                uint32_t b = outDims_[k];
                int64_t rs = k;
                while (rs - 1 > static_cast<int64_t>(ubSplitAxis_) && strides[rs - 1] == 0) {
                    rs--;
                    b *= outDims_[rs];
                }
                runB[nRuns] = b;
                runSeg[nRuns] = segSuffix;
                uint32_t fpre = 1;
                for (int64_t j = static_cast<int64_t>(ubSplitAxis_) + 1; j < rs; j++) {
                    if (strides[j] != 0) {
                        fpre *= outDims_[j];
                    }
                }
                runFpre[nRuns] = fpre;
                nRuns++;
                segSuffix *= b;
                k = rs; // continue scanning above this run
            } else {
                segSuffix *= outDims_[k];
            }
        }
        // runs were collected innermost -> outermost already (scan from the innermost axis)
        if (nRuns == 1) {
            // Host marks (!fullSplit && L>1, single run) as GENERIC with placeholder fields; the kernel folds it
            // to ROW/COL unit semantics. The static B_eff = R*B fold assumes [l][r][b] order, not the real
            // [r][l][b] layout, so foldUnit always goes through the padded engine (ST aclnnHypot_L0_006).
            p.foldUnit = true;
            p.L = runFpre[0];
            p.B = runB[0];
            p.I = runSeg[0];
            p.np = CeilAlign(p.I, HYPOT_BLK_ELEMS);
            p.useStatic = false;
            p.useEngine = true;
            return;
        }
        p.tIn = runSeg[0]; // innermost stage SEG = product of dims after the innermost run
        p.npG = CeilAlign(p.tIn, HYPOT_BLK_ELEMS);
        bool aligned = true;
        for (uint32_t s = 0; s < nRuns; s++) {
            p.stageB[s] = runB[s];
            p.stageSeg[s] = runSeg[s];
            p.stageFpre[s] = runFpre[s];
            if (runSeg[s] % HYPOT_BLK_ELEMS != 0) {
                aligned = false;
            }
        }
        p.numStages = nRuns;
        if (!p.fullSplit) {
            // split-axis repeat is the outermost stage: repeat the whole C-element row R times
            p.stageB[nRuns] = ubFormer_; // b = R resolved per tile
            p.stageSeg[nRuns] = innerLen_;
            p.stageFpre[nRuns] = 0; // marker: split stage
            p.numStages = nRuns + 1;
            if (innerLen_ % HYPOT_BLK_ELEMS != 0) {
                aligned = false;
            }
        }
        if (aligned) {
            p.genericRoute = 1; // compact block-repeat pipeline
            return;
        }
        // padded whole-buffer route: total padded floats = R * (innerLen_ / tIn) * npG
        uint64_t padded = static_cast<uint64_t>(ubFormer_) * (innerLen_ / p.tIn) * p.npG;
        bool fits = (padded + HYPOT_BUF_SLACK <= bufElems_);
        if (fits && !IsSameType<Tin, float>::value) {
            // FP16/BF16: the raw stage-1 chunk also lives in the queue buffer
            uint32_t srcInnerRows = 1;
            for (uint32_t k = ubSplitAxis_ + 1; k < dimNum_; k++) {
                if (strides[k] != 0) {
                    srcInnerRows *= outDims_[k];
                }
            }
            srcInnerRows /= p.tIn;
            uint64_t rawStage = static_cast<uint64_t>(ubFormer_) * srcInnerRows * CeilAlign(p.tIn, 32U / sizeof(Tin));
            if (rawStage + HYPOT_BUF_SLACK > bufElems_) {
                fits = false;
            }
        }
        p.genericRoute = fits ? 2 : 3;
    }

    // padded-engine window rows (multiple of 8 so the crop dst advance stays 32B aligned)
    template <typename Tin>
    __aicore__ inline uint32_t ComputeEngineWindow(const HypotBrcPlan& p)
    {
        // dst window lives in bufChain (bufElems_ floats); src padded stage lives in the queue
        // (FP32: bufElems_ floats) or bufA/bufB after per-row Cast (FP16/BF16: bufElems_ floats,
        // raw chunk rows * CeilAlign(I,16) <= bufElems_ in the raw queue)
        uint32_t rowsByDst = bufElems_ / p.np;
        uint32_t srcCapRows = bufElems_ / p.np;
        if constexpr (!IsSameType<Tin, float>::value) {
            uint32_t rawPitch = CeilAlign(p.I, 32U / sizeof(Tin));
            uint32_t rawRows = bufElems_ / rawPitch;
            srcCapRows = srcCapRows < rawRows ? srcCapRows : rawRows;
        }
        uint32_t rowsBySrc;
        if (p.fullSplit) {
            // a window of W output rows spans <= W/B + 2*L source rows (unit-boundary slack)
            uint32_t budget = srcCapRows > 2 * p.L ? srcCapRows - 2 * p.L : 0;
            rowsBySrc = budget * p.B;
        } else {
            // !fullSplit: the whole single-row source (L padded rows) is staged per window
            rowsBySrc = (srcCapRows >= p.L) ? rowsByDst : 0;
        }
        uint32_t w = rowsByDst < rowsBySrc ? rowsByDst : rowsBySrc;
        w = w / HYPOT_CROP_REP * HYPOT_CROP_REP;
        return w; // 0 -> caller falls back to the bounded scalar path (defensive, see ExpandPadded)
    }

    // GatherMask crop patterns: word w covers elements [32w, 32w+32) of one padded row;
    // bit i set iff i < I (i.e. crop the first I of np elements per row). Multi-word patterns
    // with mask > 64 are exercised by the UT cases.
    __aicore__ inline void BuildPatterns()
    {
        // COL can fall into the engine per tile (G%8!=0), so it needs a pattern even when the
        // static gate was satisfiable at plan time; likewise a 0-window engine plan may still
        // run the <=8-row single-window path which crops via GatherMask
        const bool needPadded2DCrop = CanUsePadded2D();
        bool need1 = x1Plan_.useEngine || x1Plan_.genericRoute == 2 || needPadded2DCrop;
        bool need2 = x2Plan_.useEngine || x2Plan_.genericRoute == 2;
        if (!need1 && !need2) {
            return;
        }
        LocalTensor<uint32_t> patAll = bufBrcTmp.Get<uint32_t>();
        if (need1) {
            BuildOnePattern(patX1, x1Plan_.genericRoute == 2 ? x1Plan_.npG : x1Plan_.np,
                            x1Plan_.genericRoute == 2 ? x1Plan_.tIn : x1Plan_.I);
        }
        if (need2) {
            BuildOnePattern(patX2, x2Plan_.genericRoute == 2 ? x2Plan_.npG : x2Plan_.np,
                            x2Plan_.genericRoute == 2 ? x2Plan_.tIn : x2Plan_.I);
        }
        // scalar -> vector sync before the first GatherMask
        TEventID evt = static_cast<TEventID>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(evt);
        WaitFlag<HardEvent::S_V>(evt);
    }

    __aicore__ inline void BuildOnePattern(LocalTensor<uint32_t>& pat, uint32_t np, uint32_t I)
    {
        uint32_t words = CeilDiv(np, HYPOT_PAT_WORD_BITS);
        for (uint32_t w = 0; w < words; w++) {
            uint32_t base = w * HYPOT_PAT_WORD_BITS;
            uint32_t v = 0;
            if (base + HYPOT_PAT_WORD_BITS <= I) {
                v = 0xFFFFFFFFU;
            } else if (base < I) {
                v = (1U << (I - base)) - 1U;
            }
            pat.SetValue(w, v);
        }
    }

    // ---- per-tile expansion runtime ----

    // stage the compact flat source (srcLen elements) into the queue and return the FP32 view:
    // FP32 path returns the live queue tensor (caller frees); FP16/BF16 path flat-Casts into f32Buf
    // (broadcast source is Cast to FP32 first, then expanded as float)
    template <typename Tin>
    __aicore__ inline LocalTensor<float> CopyInFlat(GlobalTensor<Tin>& gm, int64_t gmOff, uint32_t srcLen,
                                                    TQue<QuePosition::VECIN, HYPOT_BUFFER_NUM>& queue,
                                                    LocalTensor<float>& f32Buf, bool& ownsQueue)
    {
        LocalTensor<Tin> src = queue.template AllocTensor<Tin>();
        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = srcLen * sizeof(Tin);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(src, gm[gmOff], copyParams, {false, 0, 0, 0});
        queue.EnQue(src);
        src = queue.template DeQue<Tin>();
        if constexpr (IsSameType<Tin, float>::value) {
            ownsQueue = true;
            return src;
        } else {
            Cast(f32Buf, src, RoundMode::CAST_NONE, CeilAlign(srcLen, 32U / sizeof(Tin)));
            PipeBarrier<PIPE_V>();
            queue.FreeTensor(src);
            ownsQueue = false;
            return f32Buf;
        }
    }

    template <typename Tin>
    __aicore__ inline LocalTensor<float> CopyInFlatDeferred(GlobalTensor<Tin>& gm, int64_t gmOff, uint32_t srcLen,
                                                            TQue<QuePosition::VECIN, HYPOT_BUFFER_NUM>& queue,
                                                            LocalTensor<float>& f32Buf, bool& ownsQueue,
                                                            LocalTensor<Tin>& ownedSrc)
    {
        LocalTensor<Tin> src = queue.template AllocTensor<Tin>();
        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = srcLen * sizeof(Tin);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(src, gm[gmOff], copyParams, {false, 0, 0, 0});
        queue.EnQue(src);
        src = queue.template DeQue<Tin>();
        ownsQueue = true;
        ownedSrc = src;
        if constexpr (IsSameType<Tin, float>::value) {
            return src;
        } else {
            Cast(f32Buf, src, RoundMode::CAST_NONE, CeilAlign(srcLen, 32U / sizeof(Tin)));
            return f32Buf;
        }
    }

    template <typename Tin>
    __aicore__ inline float ReadElemAsFloat(LocalTensor<Tin>& tensor, uint32_t idx)
    {
        if constexpr (IsSameType<Tin, bfloat16_t>::value) {
            LocalTensor<uint16_t> t16 = tensor.template ReinterpretCast<uint16_t>();
            union {
                uint32_t u;
                float f;
            } cvt;
            cvt.u = static_cast<uint32_t>(t16.GetValue(idx)) << HYPOT_BF16_SHIFT_BITS;
            return cvt.f;
        } else {
            return static_cast<float>(tensor.GetValue(idx));
        }
    }

    // ROW static fast path (I%8==0): per-group Broadcast<float,2,0>, dst/src offsets are multiples
    // of I elements -> 32B aligned
    __aicore__ inline void ExpandRowStatic(LocalTensor<float>& exp, LocalTensor<float>& srcF, uint32_t G, uint32_t B,
                                           uint32_t I)
    {
        uint32_t dstShape[2] = {B, I};
        uint32_t srcShape[2] = {1, I};
        for (uint32_t g = 0; g < G; g++) {
            Broadcast<float, 2, 0>(exp[g * B * I], srcF[g * I], dstShape, srcShape, brcTmpU8);
        }
    }

    // COL static fast path (G%8==0 && B%8==0): single whole-tile Broadcast<float,2,1>
    __aicore__ inline void ExpandColStatic(LocalTensor<float>& exp, LocalTensor<float>& srcF, uint32_t G, uint32_t B)
    {
        uint32_t dstShape[2] = {G, B};
        uint32_t srcShape[2] = {G, 1};
        Broadcast<float, 2, 1>(exp, srcF, dstShape, srcShape, brcTmpU8);
    }

    __aicore__ inline void ExpandFoldColStatic(LocalTensor<float>& exp, LocalTensor<float>& srcF, uint32_t R,
                                               uint32_t L, uint32_t B)
    {
        uint32_t dstShape[2] = {L, B};
        uint32_t srcShape[2] = {L, 1};
        LocalTensor<float> row = bufMx.Get<float>();
        Broadcast<float, 2, 1>(row, srcF, dstShape, srcShape, brcTmpU8);
        PipeBarrier<PIPE_V>();
        CopyRepeatParams params{1, 1, static_cast<uint16_t>(L * B / HYPOT_BLK_ELEMS), 0};
        // caller guarantees R <= HYPOT_GM_CHUNK (HW repeatTimes is a uint8 field)
        Copy<float>(exp, row, static_cast<uint64_t>(L * B), R, params);
    }

    __aicore__ inline void ExpandFoldColDuplicate(LocalTensor<float>& exp, LocalTensor<float>& srcF, uint32_t R,
                                                  uint32_t L, uint32_t B)
    {
        LocalTensor<float> row = bufMx.Get<float>();
        PipeBarrier<PIPE_ALL>();
        for (uint32_t l = 0; l < L; l++) {
            Duplicate(row[l * B], srcF.GetValue(l), B);
        }
        PipeBarrier<PIPE_V>();
        CopyRepeatParams params{1, 1, static_cast<uint16_t>(L * B / HYPOT_BLK_ELEMS), 0};
        // caller guarantees R <= HYPOT_GM_CHUNK (HW repeatTimes is a uint8 field)
        Copy<float>(exp, row, static_cast<uint64_t>(L * B), R, params);
        PipeBarrier<PIPE_V>();
    }

    // small-tile ROW (tileElems < 2048, I%8==0): skip the adv Broadcast table and Copy per group.
    __aicore__ inline void ExpandRowLight(LocalTensor<float>& exp, LocalTensor<float>& srcF, uint32_t G, uint32_t B,
                                          uint32_t I)
    {
        for (uint32_t g = 0; g < G; g++) {
            uint32_t done = 0;
            while (done < B) {
                uint32_t rep = (B - done) < HYPOT_GM_CHUNK ? (B - done) : HYPOT_GM_CHUNK;
                uint32_t off = 0;
                while (off < I) {
                    uint32_t c = (I - off) < HYPOT_COPY_CHUNK ? (I - off) : HYPOT_COPY_CHUNK;
                    CopyRepeatParams params{1, 1, static_cast<uint16_t>(I / HYPOT_BLK_ELEMS), 0};
                    Copy<float>(exp[(g * B + done) * I + off], srcF[g * I + off], static_cast<uint64_t>(c), rep,
                                params);
                    off += c;
                }
                done += rep;
            }
        }
    }

    // small-tile COL (tileElems < 2048, B%8==0): per-group vector Duplicate after a full barrier.
    __aicore__ inline void ExpandColLight(LocalTensor<float>& exp, LocalTensor<float>& srcF, uint32_t G, uint32_t B)
    {
        PipeBarrier<PIPE_ALL>(); // scalar GetValue reads the vector/MTE2-written source
        for (uint32_t g = 0; g < G; g++) {
            Duplicate(exp[g * B], srcF.GetValue(g), B);
        }
    }

    // stage srcRows padded rows from GM: padded row j (pitch np floats, first I valid) <-
    // GM[gmOff + (srcRow0 + j) * I .. +I). FP32 lands directly in the queue buffer; FP16/BF16 lands
    // raw in the queue (pitch CeilAlign(I,16)) and is per-row Cast to the float padded stage.
    template <typename Tin>
    __aicore__ inline LocalTensor<float> StagePaddedSrc(GlobalTensor<Tin>& gm, int64_t gmOff, uint32_t srcRow0,
                                                        uint32_t srcRows, uint32_t np, uint32_t I,
                                                        TQue<QuePosition::VECIN, HYPOT_BUFFER_NUM>& queue,
                                                        LocalTensor<float>& f32Stage, bool& ownsQueue)
    {
        LocalTensor<Tin> raw = queue.template AllocTensor<Tin>();
        DataCopyParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(srcRows);
        copyParams.blockLen = I * sizeof(Tin);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(raw, gm[gmOff + static_cast<int64_t>(srcRow0) * I], copyParams, {false, 0, 0, 0});
        queue.EnQue(raw);
        raw = queue.template DeQue<Tin>();
        if constexpr (IsSameType<Tin, float>::value) {
            ownsQueue = true;
            return raw; // padded float rows, pitch np = CeilAlign(I,8) from DataCopyPad 32B tail padding
        } else {
            const uint32_t rawPitch = CeilAlign(I, 32U / sizeof(Tin));
            for (uint32_t r = 0; r < srcRows; r++) {
                Cast(f32Stage[r * np], raw[r * rawPitch], RoundMode::CAST_NONE, np);
            }
            PipeBarrier<PIPE_V>();
            queue.FreeTensor(raw);
            ownsQueue = false;
            return f32Stage;
        }
    }

    // padded-window fallback (ROW/COL non-aligned, runs==1 GENERIC fold): expands to compact exp with
    // G = (fullSplit ? R : 1) * L groups, B_eff = fullSplit ? B : R * B; crop via GatherMask.
    template <typename Tin>
    __aicore__ inline void ExpandPadded(LocalTensor<float>& exp, GlobalTensor<Tin>& gm, int64_t gmOff, uint32_t R,
                                        const HypotBrcPlan& p, TQue<QuePosition::VECIN, HYPOT_BUFFER_NUM>& queue,
                                        LocalTensor<float>& f32Buf, LocalTensor<uint32_t>& pat,
                                        const uint32_t strides[])
    {
        // B/I keep the [L,B,I] math notation: B = extent of the broadcast run,
        // I = trailing full product after the run
        const uint32_t B = p.B;
        const uint32_t LB = p.L * B;
        const uint32_t np = p.np;
        const uint32_t I = p.I;
        // Row decomposition [r][l][b][i] (r outermost): r = j/LB, l = (j%LB)/B, b = j%B;
        // srcRow = fullSplit ? r*L + l : l.
        const uint32_t rowsTotal = R * p.L * p.B;
        if (p.window == 0 && (rowsTotal > HYPOT_CROP_REP || rowsTotal * np > bufElems_)) {
            // defensive degenerate path: np too large for an 8-row window (near-impossible: needs
            // I > ~bufElems_/8 with many broadcast rows). Bounded scalar, correctness first.
            ExpandScalarStrides(exp, gm, gmOff, R, queue, f32Buf, strides);
            return;
        }
        LocalTensor<float> dstWin = bufChain.Get<float>();
        LocalTensor<float> f32Stage = f32Buf; // FP32: unused placeholder
        uint32_t row = 0;
        while (row < rowsTotal) {
            uint32_t wr = (rowsTotal - row) < p.window ? (rowsTotal - row) : p.window;
            if (wr == 0) {
                wr = rowsTotal - row; // rowsTotal <= 8 single-window case
            }
            // source rows spanned by this window (contiguous range in GM)
            uint32_t unit0 = row / LB;
            uint32_t unit1 = (row + wr - 1) / LB;
            uint32_t srcRow0 = p.fullSplit ? unit0 * p.L : 0;
            uint32_t srcRows = p.fullSplit ? (unit1 + 1 - unit0) * p.L : p.L;
            bool ownsQueue = false;
            LocalTensor<float> srcStage = StagePaddedSrc(gm, gmOff, srcRow0, srcRows, np, I, queue, f32Stage,
                                                         ownsQueue);
            // fill: runs of consecutive output rows sharing one source row
            uint32_t j = row;
            while (j < row + wr) {
                uint32_t r = j / LB;
                uint32_t rem = j % LB;
                uint32_t l = rem / B;
                uint32_t b = rem % B;
                uint32_t runLen = (B - b) < (row + wr - j) ? (B - b) : (row + wr - j);
                if (LB == 1) {
                    runLen = row + wr - j; // ROW_SPLIT: every row is src row 0
                }
                uint32_t srcRow = (p.fullSplit ? r * p.L + l : l) - srcRow0;
                uint32_t done = 0;
                while (done < runLen) {
                    uint32_t rep = (runLen - done) < HYPOT_GM_CHUNK ? (runLen - done) : HYPOT_GM_CHUNK;
                    uint32_t off = 0;
                    while (off < np) {
                        uint32_t c = (np - off) < HYPOT_COPY_CHUNK ? (np - off) : HYPOT_COPY_CHUNK;
                        CopyRepeatParams params{1, 1, static_cast<uint16_t>(np / HYPOT_BLK_ELEMS), 0};
                        Copy<float>(dstWin[(j - row + done) * np + off], srcStage[srcRow * np + off],
                                    static_cast<uint64_t>(c), rep, params);
                        off += c;
                    }
                    done += rep;
                }
                j += runLen;
            }
            PipeBarrier<PIPE_V>();
            // crop: packed compact stream to exp, 8 repeats per call (dst advance 8*I, 32B aligned)
            uint32_t k = 0;
            while (k < wr) {
                uint32_t rep = (wr - k) < HYPOT_CROP_REP ? (wr - k) : HYPOT_CROP_REP;
                uint64_t rsvdCnt = 0;
                GatherMask<float, uint32_t>(
                    exp[(row + k) * I], dstWin[k * np], pat, true, np,
                    GatherMaskParams{1, static_cast<uint16_t>(rep), static_cast<uint16_t>(np / HYPOT_BLK_ELEMS), 0},
                    rsvdCnt);
                k += rep;
            }
            PipeBarrier<PIPE_V>();
            if (ownsQueue) {
                queue.FreeTensor(srcStage);
            }
            row += wr;
        }
    }

    // GENERIC runs>=2 staged pipeline (SEG % 8 == 0): each stage repeats every SEG_s block b_s times;
    // buffers ping-pong between bufChain and bufMx, the last stage writes exp directly.
    __aicore__ inline void ExpandGenericCompact(LocalTensor<float>& exp, LocalTensor<float>& srcF, uint32_t R,
                                                const HypotBrcPlan& p)
    {
        LocalTensor<float> winA = bufChain.Get<float>();
        LocalTensor<float> winB = bufMx.Get<float>();
        LocalTensor<float> cur = srcF;
        for (uint32_t s = 0; s < p.numStages; s++) {
            bool isSplitStage = (p.stageFpre[s] == 0 && !p.fullSplit);
            // G keeps the [L,B,I] notation: group count of this stage
            uint32_t G = isSplitStage ? 1 : (p.fullSplit ? R : 1) * p.stageFpre[s];
            uint32_t b = isSplitStage ? R : p.stageB[s];
            uint32_t seg = p.stageSeg[s];
            LocalTensor<float> dst = (s == p.numStages - 1) ? exp : ((s % 2 == 0) ? winA : winB);
            RepeatBlocks(dst, cur, G, b, seg);
            PipeBarrier<PIPE_V>();
            cur = dst;
        }
    }

    // GENERIC runs>=2 padded whole-buffer route (some stage SEG % 8 != 0): same staged structure in
    // the padded domain (pitch npG = CeilAlign(tIn,8)); stage 1 sources padded rows from GM
    // (DataCopyPad multi-block, + per-row Cast for 2B dtypes); final crop packs to compact exp.
    template <typename Tin>
    __aicore__ inline void ExpandGenericPadded(LocalTensor<float>& exp, GlobalTensor<Tin>& gm, int64_t gmOff,
                                               uint32_t R, const HypotBrcPlan& p,
                                               TQue<QuePosition::VECIN, HYPOT_BUFFER_NUM>& queue,
                                               LocalTensor<float>& f32Buf, LocalTensor<uint32_t>& pat,
                                               const uint32_t strides[])
    {
        const uint32_t np = p.npG;
        const uint32_t tIn = p.tIn;
        // stage-1 padded source rows = (fullSplit ? R : 1) * (per-row source extent / tIn)
        uint32_t srcInnerRows = 1;
        for (uint32_t k = ubSplitAxis_ + 1; k < dimNum_; k++) {
            if (strides[k] != 0) {
                srcInnerRows *= outDims_[k];
            }
        }
        srcInnerRows /= tIn;
        uint32_t rows = (p.fullSplit ? R : 1) * srcInnerRows;
        bool ownsQueue = false;
        LocalTensor<float> staged = StagePaddedSrc(gm, gmOff, 0, rows, np, tIn, queue, f32Buf, ownsQueue);
        LocalTensor<float> cur = staged;
        LocalTensor<float> winA = bufChain.Get<float>();
        LocalTensor<float> winB = bufMx.Get<float>();
        for (uint32_t s = 0; s < p.numStages; s++) {
            bool isSplitStage = (p.stageFpre[s] == 0 && !p.fullSplit);
            // G keeps the [L,B,I] notation: group count of this stage
            uint32_t G = isSplitStage ? 1 : (p.fullSplit ? R : 1) * p.stageFpre[s];
            uint32_t b = isSplitStage ? R : p.stageB[s];
            uint32_t segRows = p.stageSeg[s] / tIn;
            LocalTensor<float> dst = (s % 2 == 0) ? winA : winB;
            RepeatBlocks(dst, cur, G, b, segRows * np);
            PipeBarrier<PIPE_V>();
            cur = dst;
            rows = G * b * segRows;
        }
        // crop: packed compact stream to exp (rows* tIn = R*C elements)
        uint32_t k = 0;
        while (k < rows) {
            uint32_t rep = (rows - k) < HYPOT_CROP_REP ? (rows - k) : HYPOT_CROP_REP;
            uint64_t rsvdCnt = 0;
            GatherMask<float, uint32_t>(
                exp[k * tIn], cur[k * np], pat, true, np,
                GatherMaskParams{1, static_cast<uint16_t>(rep), static_cast<uint16_t>(np / HYPOT_BLK_ELEMS), 0},
                rsvdCnt);
            k += rep;
        }
        PipeBarrier<PIPE_V>();
        if (ownsQueue) {
            queue.FreeTensor(staged); // free the queue-owned stage tensor, NOT cur (winA/winB are TBuf-backed)
        }
    }

    // Bounded scalar degenerate path (correctness net, never on the perf case list): per-element GM
    // staging via the queue, pure strides index mapping on the compact layout.
    template <typename Tin>
    __aicore__ inline void ExpandScalarStrides(LocalTensor<float>& exp, GlobalTensor<Tin>& gm, int64_t gmOff,
                                               uint32_t R, TQue<QuePosition::VECIN, HYPOT_BUFFER_NUM>& queue,
                                               LocalTensor<float>& f32Buf, const uint32_t strides[])
    {
        // srcLen = (fullSplit ? R : 1) * product of non-broadcast inner dims
        uint32_t srcInner = 1;
        for (uint32_t k = ubSplitAxis_ + 1; k < dimNum_; k++) {
            if (strides[k] != 0) {
                srcInner *= outDims_[k];
            }
        }
        const bool fullSplit = (strides[ubSplitAxis_] != 0);
        const uint32_t srcLen = (fullSplit ? R : 1) * srcInner;
        bool ownsQueue = false;
        LocalTensor<float> srcF = CopyInFlat(gm, gmOff, srcLen, queue, f32Buf, ownsQueue);
        PipeBarrier<PIPE_ALL>(); // scalar GetValue reads the vector/MTE2-written source
        // C is the element count of the inner region
        // (C = product of dims[ubSplitAxis+1..dimNum-1], aliased as innerLen_)
        const uint32_t C = innerLen_;
        uint32_t total = R * C;
        for (uint32_t e = 0; e < total; e++) {
            uint32_t tmp = e;
            uint32_t off = 0;
            for (int64_t k = static_cast<int64_t>(dimNum_) - 1; k > static_cast<int64_t>(ubSplitAxis_); k--) {
                uint32_t d = outDims_[k];
                uint32_t idx = tmp % d;
                tmp /= d;
                if (strides[k] != 0) {
                    off += idx * strides[k];
                }
            }
            if (strides[ubSplitAxis_] != 0) {
                off += tmp * strides[ubSplitAxis_];
            }
            exp.SetValue(e, srcF.GetValue(off));
        }
        PipeBarrier<PIPE_ALL>();
        if (ownsQueue) {
            queue.FreeTensor(srcF);
        }
    }

    // Correctness-first arbitrary broadcast expansion. A small source tensor is staged once in
    // FP32, then Gather builds the expanded tile from explicit per-element source offsets.
    template <typename Tin>
    __aicore__ inline void ExpandGatherBroadcast(LocalTensor<float>& exp, GlobalTensor<Tin>& gm, uint32_t R,
                                                 uint32_t srcLen, uint32_t count,
                                                 TQue<QuePosition::VECIN, HYPOT_BUFFER_NUM>& queue,
                                                 LocalTensor<float>& f32Buf, const uint32_t strides[],
                                                 const int64_t axesIndices[])
    {
        if (HYPOT_ENABLE_NATIVE_GATHER) {
            if constexpr (!IsSameType<Tin, float>::value) {
                LocalTensor<Tin> srcT = queue.template AllocTensor<Tin>();
                DataCopyParams copyParams;
                copyParams.blockCount = 1;
                copyParams.blockLen = srcLen * sizeof(Tin);
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                DataCopyPad(srcT, gm[0], copyParams, {false, 0, 0, 0});
                queue.EnQue(srcT);
                srcT = queue.template DeQue<Tin>();
                PipeBarrier<PIPE_ALL>();
                LocalTensor<uint8_t> scratchBytes = f32Buf.template ReinterpretCast<uint8_t>();
                LocalTensor<Tin> expandedT = scratchBytes.template ReinterpretCast<Tin>();
                LocalTensor<uint32_t> indices = brcTmpU8.template ReinterpretCast<uint32_t>();
                const uint32_t innerLen = innerLen_;
                const uint32_t total = R * innerLen;
                int64_t coords[HYPOT_MAX_DIM] = {0};
                uint32_t base = 0;
                while (base < total) {
                    const uint32_t chunk = (total - base) < HYPOT_GATHER_IDX_CAP ? (total - base) :
                                                                                   HYPOT_GATHER_IDX_CAP;
                    for (uint32_t j = 0; j < chunk; j++) {
                        const uint32_t e = base + j;
                        uint32_t inner = e % innerLen;
                        coords[ubSplitAxis_] = axesIndices[ubSplitAxis_] * static_cast<int64_t>(ubFormer_) +
                                               e / innerLen;
                        for (int64_t k = static_cast<int64_t>(dimNum_) - 1; k > static_cast<int64_t>(ubSplitAxis_);
                             k--) {
                            coords[k] = inner % outDims_[k];
                            inner /= outDims_[k];
                        }
                        for (int64_t k = 0; k < static_cast<int64_t>(ubSplitAxis_); k++) {
                            coords[k] = axesIndices[k];
                        }
                        uint32_t off = 0;
                        for (uint32_t k = 0; k < dimNum_; k++) {
                            if (strides[k] != 0U) {
                                off += static_cast<uint32_t>(coords[k]) * strides[k];
                            }
                        }
                        indices.SetValue(j, off * sizeof(Tin));
                    }
                    TEventID scalarToVector = static_cast<TEventID>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
                    SetFlag<HardEvent::S_V>(scalarToVector);
                    WaitFlag<HardEvent::S_V>(scalarToVector);
                    Gather(expandedT[base], srcT, indices, 0, chunk);
                    if (blockIdx_ == 0U && base == 0U && total == 192U && srcLen == 24U) {
                        LocalTensor<uint16_t> srcBits = srcT.template ReinterpretCast<uint16_t>();
                        LocalTensor<uint16_t> expBits = expandedT.template ReinterpretCast<uint16_t>();
                        printf("[HYPOT_DBG] src0=%d src1=%d src2=%d idx0=%d idx1=%d exp0=%d exp1=%d exp8=%d\n",
                               static_cast<int>(srcBits.GetValue(0)), static_cast<int>(srcBits.GetValue(1)),
                               static_cast<int>(srcBits.GetValue(2)), static_cast<int>(indices.GetValue(0)),
                               static_cast<int>(indices.GetValue(1)), static_cast<int>(expBits.GetValue(0)),
                               static_cast<int>(expBits.GetValue(1)), static_cast<int>(expBits.GetValue(8)));
                    }
                    TEventID vectorToScalar = static_cast<TEventID>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                    SetFlag<HardEvent::V_S>(vectorToScalar);
                    WaitFlag<HardEvent::V_S>(vectorToScalar);
                    base += chunk;
                }
                PipeBarrier<PIPE_V>();
                Cast(exp, expandedT, RoundMode::CAST_NONE, total);
                PipeBarrier<PIPE_V>();
                queue.FreeTensor(srcT);
                return;
            }
        }
        bool ownsQueue = false;
        LocalTensor<float> srcF = CopyInFlat(gm, 0, srcLen, queue, f32Buf, ownsQueue);
        PipeBarrier<PIPE_V>();
        LocalTensor<uint32_t> indices = brcTmpU8.template ReinterpretCast<uint32_t>();
        const uint32_t innerLen = innerLen_;
        const uint32_t total = R * innerLen;
        int64_t coords[HYPOT_MAX_DIM] = {0};
        uint32_t base = 0;
        while (base < total) {
            const uint32_t count = (total - base) < HYPOT_GATHER_IDX_CAP ? (total - base) : HYPOT_GATHER_IDX_CAP;
            for (uint32_t j = 0; j < count; j++) {
                const uint32_t e = base + j;
                uint32_t inner = e % innerLen;
                coords[ubSplitAxis_] = axesIndices[ubSplitAxis_] * static_cast<int64_t>(ubFormer_) + e / innerLen;
                for (int64_t k = static_cast<int64_t>(dimNum_) - 1; k > static_cast<int64_t>(ubSplitAxis_); k--) {
                    coords[k] = inner % outDims_[k];
                    inner /= outDims_[k];
                }
                for (int64_t k = 0; k < static_cast<int64_t>(ubSplitAxis_); k++) {
                    coords[k] = axesIndices[k];
                }
                uint32_t off = 0;
                for (uint32_t k = 0; k < dimNum_; k++) {
                    if (strides[k] != 0U) {
                        off += static_cast<uint32_t>(coords[k]) * strides[k];
                    }
                }
                indices.SetValue(j, off * sizeof(float));
            }
            TEventID scalarToVector = static_cast<TEventID>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(scalarToVector);
            WaitFlag<HardEvent::S_V>(scalarToVector);
            Gather(exp[base], srcF, indices, 0, count);
            TEventID vectorToScalar = static_cast<TEventID>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(vectorToScalar);
            WaitFlag<HardEvent::V_S>(vectorToScalar);
            base += count;
        }
        PipeBarrier<PIPE_V>();
        if (ownsQueue) {
            queue.FreeTensor(srcF);
        }
    }

    __aicore__ inline uint32_t BroadcastSourceOffset(uint32_t e, const uint32_t strides[], const int64_t axesIndices[])
    {
        uint32_t sourceInner = 1U;
        for (uint32_t k = ubSplitAxis_ + 1; k < dimNum_; k++) {
            sourceInner *= outDims_[k];
        }
        uint32_t inner = e % sourceInner;
        int64_t coords[HYPOT_MAX_DIM] = {0};
        coords[ubSplitAxis_] = axesIndices[ubSplitAxis_] * static_cast<int64_t>(ubFormer_) + e / sourceInner;
        for (int64_t k = static_cast<int64_t>(dimNum_) - 1; k > static_cast<int64_t>(ubSplitAxis_); k--) {
            coords[k] = inner % outDims_[k];
            inner /= outDims_[k];
        }
        for (int64_t k = 0; k < static_cast<int64_t>(ubSplitAxis_); k++) {
            coords[k] = axesIndices[k];
        }
        uint32_t off = 0;
        for (uint32_t k = 0; k < dimNum_; k++) {
            if (strides[k] != 0U) {
                off += static_cast<uint32_t>(coords[k]) * strides[k];
            }
        }
        return off;
    }

    __aicore__ inline void InitBroadcastOffsetCursor(uint32_t coords[], uint32_t& off, const uint32_t strides[],
                                                     const int64_t axesIndices[])
    {
        off = 0U;
        for (uint32_t k = 0; k < dimNum_; k++) {
            coords[k] = 0U;
        }
        for (uint32_t k = 0; k < ubSplitAxis_; k++) {
            coords[k] = static_cast<uint32_t>(axesIndices[k]);
            off += coords[k] * strides[k];
        }
        coords[ubSplitAxis_] = static_cast<uint32_t>(axesIndices[ubSplitAxis_]) * ubFormer_;
        off += coords[ubSplitAxis_] * strides[ubSplitAxis_];
    }

    // Equivalent to BroadcastSourceOffset(e), but walks the compact tile incrementally instead of
    // repeating modulo/division chains for every element.
    __aicore__ inline void AdvanceBroadcastOffsetCursor(uint32_t coords[], uint32_t& off, const uint32_t strides[])
    {
        for (int64_t k = static_cast<int64_t>(dimNum_) - 1; k > static_cast<int64_t>(ubSplitAxis_); k--) {
            coords[k]++;
            off += strides[k];
            if (coords[k] < outDims_[k]) {
                return;
            }
            off -= outDims_[k] * strides[k];
            coords[k] = 0U;
        }
        coords[ubSplitAxis_]++;
        off += strides[ubSplitAxis_];
    }

    // Tiny tiles cannot amortize Gather setup or per-block branching. Walk the compact output
    // once with the incremental cursor and fill the FP32 view directly in UB.
    template <typename Tin>
    __aicore__ inline void ExpandScalarLinear(LocalTensor<float>& exp, GlobalTensor<Tin>& gm, uint32_t R,
                                              uint32_t srcLen, TQue<QuePosition::VECIN, HYPOT_BUFFER_NUM>& queue,
                                              LocalTensor<float>& f32Buf, const uint32_t strides[],
                                              const int64_t axesIndices[])
    {
        bool ownsQueue = false;
        LocalTensor<float> srcF = CopyInFlat(gm, 0, srcLen, queue, f32Buf, ownsQueue);
        PipeBarrier<PIPE_ALL>();
        const uint32_t total = R * innerLen_;
        uint32_t coords[HYPOT_MAX_DIM] = {0};
        uint32_t sourceOffset = 0U;
        InitBroadcastOffsetCursor(coords, sourceOffset, strides, axesIndices);
        for (uint32_t e = 0; e < total; e++) {
            exp.SetValue(e, srcF.GetValue(sourceOffset));
            AdvanceBroadcastOffsetCursor(coords, sourceOffset, strides);
        }
        PipeBarrier<PIPE_ALL>();
        if (ownsQueue) {
            queue.FreeTensor(srcF);
        }
    }

    // Small-source fallback: build aligned 8-float blocks using Duplicate for repeated values,
    // UB Copy for contiguous runs, and a short scalar fill only for irregular boundaries.
    template <typename Tin>
    __aicore__ inline void ExpandSmallSourceBlocks(LocalTensor<float>& exp, GlobalTensor<Tin>& gm, uint32_t R,
                                                   uint32_t srcLen, TQue<QuePosition::VECIN, HYPOT_BUFFER_NUM>& queue,
                                                   LocalTensor<float>& f32Buf, const uint32_t strides[],
                                                   const int64_t axesIndices[])
    {
        if constexpr (!IsSameType<Tin, float>::value) {
            const bool usePaddedSource = srcLen <= 32U;
            LocalTensor<Tin> srcT;
            DataCopyParams copyParams;
            if (usePaddedSource) {
                LocalTensor<uint8_t> scratchBytes = f32Buf.template ReinterpretCast<uint8_t>();
                srcT = scratchBytes.template ReinterpretCast<Tin>();
                copyParams.blockCount = static_cast<uint16_t>(srcLen);
                copyParams.blockLen = sizeof(Tin);
            } else {
                srcT = queue.template AllocTensor<Tin>();
                copyParams.blockCount = 1;
                copyParams.blockLen = srcLen * sizeof(Tin);
            }
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(srcT, gm[0], copyParams, {false, 0, 0, 0});
            if (!usePaddedSource) {
                queue.EnQue(srcT);
                srcT = queue.template DeQue<Tin>();
            }
            TEventID mte2ToScalar = static_cast<TEventID>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(mte2ToScalar);
            WaitFlag<HardEvent::MTE2_S>(mte2ToScalar);
            const uint32_t sourceStride = usePaddedSource ? (32U / sizeof(Tin)) : 1U;
            uint32_t sourceInnerLen = 1U;
            for (uint32_t k = ubSplitAxis_ + 1; k < dimNum_; k++) {
                sourceInnerLen *= outDims_[k];
            }
            const uint32_t total = R * sourceInnerLen;
            uint32_t coords[HYPOT_MAX_DIM] = {0};
            uint32_t sourceOffset = 0U;
            InitBroadcastOffsetCursor(coords, sourceOffset, strides, axesIndices);
            for (uint32_t base = 0; base < total; base += HYPOT_BLK_ELEMS) {
                const uint32_t block = (total - base) < HYPOT_BLK_ELEMS ? (total - base) : HYPOT_BLK_ELEMS;
                uint32_t offsets[HYPOT_BLK_ELEMS] = {0};
                bool same = true;
                for (uint32_t j = 0; j < block; j++) {
                    offsets[j] = sourceOffset;
                    AdvanceBroadcastOffsetCursor(coords, sourceOffset, strides);
                    if (j != 0U) {
                        same = same && offsets[j] == offsets[0];
                    }
                }
                if (same) {
                    Duplicate(exp[base], ReadElemAsFloat(srcT, offsets[0] * sourceStride), block);
                } else {
                    for (uint32_t j = 0; j < block; j++) {
                        exp.SetValue(base + j, ReadElemAsFloat(srcT, offsets[j] * sourceStride));
                    }
                }
            }
            PipeBarrier<PIPE_ALL>();
            if (!usePaddedSource) {
                queue.FreeTensor(srcT);
            }
            return;
        }
        bool ownsQueue = false;
        LocalTensor<float> srcF = CopyInFlat(gm, 0, srcLen, queue, f32Buf, ownsQueue);
        PipeBarrier<PIPE_ALL>();
        const uint32_t total = R * innerLen_;
        uint32_t coords[HYPOT_MAX_DIM] = {0};
        uint32_t sourceOffset = 0U;
        InitBroadcastOffsetCursor(coords, sourceOffset, strides, axesIndices);
        for (uint32_t base = 0; base < total; base += HYPOT_BLK_ELEMS) {
            const uint32_t block = (total - base) < HYPOT_BLK_ELEMS ? (total - base) : HYPOT_BLK_ELEMS;
            uint32_t offsets[HYPOT_BLK_ELEMS] = {0};
            bool same = true;
            bool contiguous = true;
            for (uint32_t j = 0; j < block; j++) {
                offsets[j] = sourceOffset;
                AdvanceBroadcastOffsetCursor(coords, sourceOffset, strides);
                if (j != 0U) {
                    same = same && offsets[j] == offsets[0];
                    contiguous = contiguous && offsets[j] == offsets[0] + j;
                }
            }
            const bool alignedCopy = contiguous && block == HYPOT_BLK_ELEMS && (offsets[0] % HYPOT_BLK_ELEMS) == 0U;
            if (same) {
                Duplicate(exp[base], srcF.GetValue(offsets[0]), block);
            } else if (alignedCopy) {
                CopyRepeatParams params{1, 1, 1, 0};
                Copy<float>(exp[base], srcF[offsets[0]], block, 1, params);
            } else {
                for (uint32_t j = 0; j < block; j++) {
                    exp.SetValue(base + j, srcF.GetValue(offsets[j]));
                }
            }
        }
        PipeBarrier<PIPE_ALL>();
        if (ownsQueue) {
            queue.FreeTensor(srcF);
        }
    }

    // per-input expansion dispatcher (decision tree)
    template <typename Tin>
    __aicore__ inline LocalTensor<float> ExpandInput(GlobalTensor<Tin>& gm, int64_t gmOff, uint32_t R,
                                                     const HypotBrcPlan& p, uint32_t scalarBit, TBuf<>& expBuf,
                                                     TBuf<>& f32BufT, TQue<QuePosition::VECIN, HYPOT_BUFFER_NUM>& queue,
                                                     uint32_t count, bool& ownsQueue, LocalTensor<Tin>& ownedSrc,
                                                     LocalTensor<uint32_t>& pat, const uint32_t strides[],
                                                     const int64_t axesIndices[])
    {
        ownsQueue = false;
        LocalTensor<float> exp = expBuf.Get<float>();
        LocalTensor<float> f32Buf = f32BufT.Get<float>(); // FP32: dead placeholder view of bufChain
        if ((scalarFlag_ & scalarBit) != 0U) {
            Duplicate(exp, scalarBit == HYPOT_SCALAR_FLAG_X1 ? scalarX1_ : scalarX2_, count);
            return exp;
        }
        const uint32_t sourceNumel = (scalarBit == HYPOT_SCALAR_FLAG_X1) ? x1Numel_ : x2Numel_;
        const uint32_t cx = p.L * p.I;
        const uint32_t srcLen = (p.fullSplit ? R : 1) * cx;
        const uint32_t preloadLen = (scalarBit == HYPOT_SCALAR_FLAG_X1) ? x1PreloadLen_ : x2PreloadLen_;
        if (p.mode != HYPOT_BRC_NONE && srcLen == 1) {
            if (gmOff >= 0 && static_cast<uint64_t>(gmOff) < preloadLen) {
                LocalTensor<Tin> pre = (scalarBit == HYPOT_SCALAR_FLAG_X1) ? bufPreX1.Get<Tin>() : bufPreX2.Get<Tin>();
                Duplicate(exp, ReadElemAsFloat(pre, static_cast<uint32_t>(gmOff)), count);
                return exp;
            }
            const uint32_t slot = (scalarBit == HYPOT_SCALAR_FLAG_X1) ? 0U : 1U;
            if (gmOff != smallSrcOff_[slot]) {
                LocalTensor<Tin> s = queue.template AllocTensor<Tin>();
                DataCopyParams cp;
                cp.blockCount = 1;
                cp.blockLen = sizeof(Tin);
                cp.srcStride = 0;
                cp.dstStride = 0;
                DataCopyPad(s, gm[gmOff], cp, {false, 0, 0, 0});
                queue.EnQue(s);
                s = queue.template DeQue<Tin>();
                PipeBarrier<PIPE_ALL>();
                smallSrcVal_[slot] = ReadElemAsFloat(s, 0);
                smallSrcOff_[slot] = gmOff;
                PipeBarrier<PIPE_ALL>();
                queue.FreeTensor(s);
            }
            Duplicate(exp, smallSrcVal_[slot], count);
            PipeBarrier<PIPE_V>();
            return exp;
        }
        const uint32_t smallB = p.fullSplit ? p.B : R * p.B;
        const uint32_t staticG = (p.fullSplit ? R : 1U) * p.L;
        const bool smallAlignedStatic = dimNum_ == 2U && singleSidedBroadcast_ && !p.foldUnit &&
                                        ((p.mode == HYPOT_BRC_ROW && p.I % HYPOT_BLK_ELEMS == 0U) ||
                                         (p.mode == HYPOT_BRC_COL && smallB % HYPOT_BLK_ELEMS == 0U));
        // ROW/COL and the single-run GENERIC fold have a fully structured [L,B,I]
        // representation. Use their block-expansion route instead of rebuilding a source offset
        // for every output element. Multi-run GENERIC layouts retain the Gather correctness path.
        const bool smallDimSafe = totalLength_ > 256U;
        const bool unsafeSmallFold = !smallDimSafe && p.foldUnit && p.I == 1U && !p.fullSplit &&
                                     !(p.L <= HYPOT_BLK_ELEMS && p.B <= HYPOT_BLK_ELEMS);
        const bool rowAlignedStatic = dimNum_ <= 4U && p.mode == HYPOT_BRC_ROW && !p.foldUnit &&
                                      p.I % HYPOT_BLK_ELEMS == 0U && (smallDimSafe || p.I <= HYPOT_BLK_ELEMS);
        const bool colAlignedStatic = dimNum_ <= 4U && p.mode == HYPOT_BRC_COL && !p.foldUnit && p.fullSplit &&
                                      p.B % HYPOT_BLK_ELEMS == 0U && staticG % HYPOT_BLK_ELEMS == 0U &&
                                      (smallDimSafe || p.B <= HYPOT_BLK_ELEMS);
        const bool genericBlockExpansion = totalLength_ > 256U && (p.genericRoute == 1 || p.genericRoute == 2);
        // High-rank (dimNum > 4) aligned ROW (STRUCT only): rowAlignedStatic is capped at dimNum <= 4, so
        // without this term the input falls into the per-element Gather index path.
        const bool rowStaticHighRank = STRUCT_EXT && dimNum_ > 4U && smallDimSafe && p.mode == HYPOT_BRC_ROW &&
                                       !p.foldUnit && p.I % HYPOT_BLK_ELEMS == 0U;
        const bool structuredEngine = !unsafeSmallFold && (rowAlignedStatic || colAlignedStatic ||
                                                           genericBlockExpansion || rowStaticHighRank ||
                                                           (p.useEngine && (p.mode == HYPOT_BRC_ROW ||
                                                                            p.mode == HYPOT_BRC_COL || p.foldUnit)));
        const bool useGatherForSmallSource = p.mode != HYPOT_BRC_NONE && sourceNumel > 1U &&
                                             sourceNumel <= HYPOT_GATHER_SRC_MAX && !smallAlignedStatic &&
                                             !structuredEngine;
        if (useGatherForSmallSource) {
            ExpandGatherBroadcast(exp, gm, R, sourceNumel, count, queue, f32Buf, strides, axesIndices);
            return exp;
        }
        constexpr bool kPreloadColSupported = IsSameType<Tin, float>::value;
        if (!p.foldUnit && kPreloadColSupported && p.mode == HYPOT_BRC_COL && srcLen > 1 && gmOff >= 0 &&
            static_cast<uint64_t>(gmOff) + srcLen <= preloadLen) {
            LocalTensor<Tin> pre = (scalarBit == HYPOT_SCALAR_FLAG_X1) ? bufPreX1.Get<Tin>() : bufPreX2.Get<Tin>();
            LocalTensor<float> srcF;
            if constexpr (IsSameType<Tin, float>::value) {
                srcF = pre;
            } else {
                Cast(f32Buf, pre, RoundMode::CAST_NONE, CeilAlign(srcLen, HYPOT_BLK_ELEMS));
                PipeBarrier<PIPE_V>();
                srcF = f32Buf;
            }
            const uint32_t G = (p.fullSplit ? R : 1) * p.L;
            const uint32_t B = p.fullSplit ? p.B : R * p.B;
            const uint32_t I = p.I;
            if (p.mode == HYPOT_BRC_ROW && I % HYPOT_BLK_ELEMS == 0) {
                if ((R * innerLen_) < HYPOT_SMALL_TILE_ELEMS) {
                    ExpandRowLight(exp, srcF, G, B, I);
                } else {
                    ExpandRowStatic(exp, srcF, G, B, I);
                }
                return exp;
            }
            if constexpr (IsSameType<Tin, float>::value) {
                if (p.mode == HYPOT_BRC_COL && B % HYPOT_BLK_ELEMS == 0) {
                    ExpandColStatic(exp, srcF, G, B);
                    return exp;
                }
            }
        }
        if (p.mode == HYPOT_BRC_NONE) {
            // compact GM continuous run, single-block DataCopyPad; FP16/BF16 flat Cast whole tile
            return CopyInFlatDeferred(gm, gmOff, srcLen, queue, f32Buf, ownsQueue, ownedSrc);
        }
        // G/B/I keep the [L,B,I] math notation: G = group count of the tile,
        // B = effective broadcast extent (split-axis broadcast folds in as R * B when !fullSplit),
        // I = trailing full product after the broadcast run
        const uint32_t G = (p.fullSplit ? R : 1) * p.L;
        const uint32_t B = p.fullSplit ? p.B : R * p.B;
        const uint32_t I = p.I;
        const bool rowLike = (p.mode == HYPOT_BRC_ROW) || (p.foldUnit && I > 1);
        const bool colLike = (p.mode == HYPOT_BRC_COL) || (p.foldUnit && I == 1);
        if (totalLength_ <= 256U && p.foldUnit && I == 1U && !p.fullSplit && p.L <= HYPOT_BLK_ELEMS &&
            p.B <= HYPOT_BLK_ELEMS && R <= HYPOT_GM_CHUNK) {
            LocalTensor<float> srcF = CopyInFlatDeferred(gm, gmOff, srcLen, queue, f32Buf, ownsQueue, ownedSrc);
            ExpandFoldColDuplicate(exp, srcF, R, p.L, p.B);
            return exp;
        }
        if (p.foldUnit && I == 1U && !p.fullSplit) {
            if (R <= HYPOT_GM_CHUNK) {
                LocalTensor<float> srcF = CopyInFlatDeferred(gm, gmOff, srcLen, queue, f32Buf, ownsQueue, ownedSrc);
                PipeBarrier<PIPE_V>();
                ExpandFoldColStatic(exp, srcF, R, p.L, p.B);
                return exp;
            }
            // R beyond the HW repeatTimes uint8 limit: the fold fast path cannot chunk, route to
            // the padded engine which decomposes (r,l,b) rows correctly and chunks its repeats
            ExpandPadded(exp, gm, gmOff, R, p, queue, f32Buf, pat, strides);
            return exp;
        }
        if (rowLike || colLike) {
            // small-tile lightweight dispatch (tileElems < 2048): decide by runtime tile parameters instead of
            // the adv Broadcast table; ROW needs I%8==0, COL needs B%8==0, else fall back to ExpandPadded.
            const bool smallTile = (R * innerLen_) < HYPOT_SMALL_TILE_ELEMS;
            // foldUnit (GENERIC runs==1, !fullSplit, L>1) never takes the static path: the R*B
            // fold assumes an [l][r][b] dst order but the real tile order is [r][l][b]
            const bool foldUnitCol = p.foldUnit && p.I == 1U;
            // The newly routed high-rank ROW family (STRUCT scene) uses the lightweight per-group
            // Copy expansion instead of the static Broadcast interface: its tiles are small and
            // the Broadcast table setup cost dominates there.
            const bool lightRowRoute = rowLike && rowStaticHighRank && (I % HYPOT_BLK_ELEMS == 0U);
            const bool colStaticAligned = B % HYPOT_BLK_ELEMS == 0U &&
                                          (!p.fullSplit || staticG % HYPOT_BLK_ELEMS == 0U);
            bool staticOk = (!p.foldUnit || foldUnitCol) && !smallTile &&
                            (rowLike ? (I % HYPOT_BLK_ELEMS == 0) : colStaticAligned);
            if (lightRowRoute) {
                LocalTensor<float> srcF = CopyInFlatDeferred(gm, gmOff, srcLen, queue, f32Buf, ownsQueue, ownedSrc);
                PipeBarrier<PIPE_V>();
                ExpandRowLight(exp, srcF, G, B, I);
                return exp;
            }
            if (staticOk) {
                LocalTensor<float> srcF = CopyInFlatDeferred(gm, gmOff, srcLen, queue, f32Buf, ownsQueue, ownedSrc);
                PipeBarrier<PIPE_V>();
                if (rowLike) {
                    ExpandRowStatic(exp, srcF, G, B, I);
                } else {
                    ExpandColStatic(exp, srcF, G, B);
                }
                return exp;
            }
            const bool lightOk = smallTile && (!p.foldUnit || foldUnitCol) &&
                                 (rowLike ? (I % HYPOT_BLK_ELEMS == 0) : (B % HYPOT_BLK_ELEMS == 0));
            if (lightOk) {
                LocalTensor<float> srcF = CopyInFlatDeferred(gm, gmOff, srcLen, queue, f32Buf, ownsQueue, ownedSrc);
                PipeBarrier<PIPE_V>();
                if (rowLike) {
                    ExpandRowLight(exp, srcF, G, B, I);
                } else {
                    ExpandColLight(exp, srcF, G, B);
                }
                return exp;
            }
            ExpandPadded(exp, gm, gmOff, R, p, queue, f32Buf, pat, strides);
            return exp;
        }
        // GENERIC runs>=2
        if (p.genericRoute == 1) {
            LocalTensor<float> srcF = CopyInFlatDeferred(gm, gmOff, srcLen, queue, f32Buf, ownsQueue, ownedSrc);
            PipeBarrier<PIPE_V>();
            ExpandGenericCompact(exp, srcF, R, p);
            PipeBarrier<PIPE_V>();
            return exp;
        }
        if (p.genericRoute == 2) {
            ExpandGenericPadded(exp, gm, gmOff, R, p, queue, f32Buf, pat, strides);
            return exp;
        }
        // genericRoute == 3: bounded scalar degenerate (padded tile exceeds scratch buffers)
        ExpandScalarStrides(exp, gm, gmOff, R, queue, f32Buf, strides);
        return exp;
    }

    __aicore__ inline void ExpandTinyGeneric(LocalTensor<float>& dst, LocalTensor<float>& src, const uint32_t strides[])
    {
        int64_t axesIndices[HYPOT_MAX_DIM] = {0};
        uint32_t coords[HYPOT_MAX_DIM] = {0};
        uint32_t sourceOffset = 0U;
        InitBroadcastOffsetCursor(coords, sourceOffset, strides, axesIndices);
        for (uint32_t e = 0; e < totalLength_; e++) {
            dst.SetValue(e, src.GetValue(sourceOffset));
            AdvanceBroadcastOffsetCursor(coords, sourceOffset, strides);
        }
    }

    __aicore__ inline void ProcessTinyGeneric()
    {
        const uint32_t count = CeilAlign(static_cast<uint32_t>(totalLength_), HYPOT_CMP_ALIGN);
        LocalTensor<uint8_t> raw = bufTiny.Get<uint8_t>();
        LocalTensor<float> workspace = bufTiny.Get<float>();
        LocalTensor<float> a = workspace[4096];
        LocalTensor<float> b = workspace[5120];
        LocalTensor<float> mx = workspace[6144];
        LocalTensor<float> scratch = workspace[7168];
        LocalTensor<float> y = workspace[8192];
        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;

        if constexpr (IS_FP32) {
            LocalTensor<float> x1 = workspace[0];
            LocalTensor<float> x2 = workspace[1024];
            copyParams.blockLen = x1Numel_ * sizeof(float);
            DataCopyPad(x1, x1Gm, copyParams, {false, 0, 0, 0});
            copyParams.blockLen = x2Numel_ * sizeof(float);
            DataCopyPad(x2, x2Gm, copyParams, {false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
            ExpandTinyGeneric(a, x1, x1Strides_);
            ExpandTinyGeneric(b, x2, x2Strides_);
            PipeBarrier<PIPE_ALL>();
            ComputeChainFast(a, b, scratch, y, mx, mx, count);
        } else {
            LocalTensor<T1> x1Raw = raw[0].template ReinterpretCast<T1>();
            LocalTensor<T2> x2Raw = raw[4096].template ReinterpretCast<T2>();
            LocalTensor<float> x1 = workspace[2048];
            LocalTensor<float> x2 = workspace[3072];
            copyParams.blockLen = x1Numel_ * sizeof(T1);
            DataCopyPad(x1Raw, x1Gm, copyParams, {false, 0, 0, 0});
            copyParams.blockLen = x2Numel_ * sizeof(T2);
            DataCopyPad(x2Raw, x2Gm, copyParams, {false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
            Cast(x1, x1Raw, RoundMode::CAST_NONE, CeilAlign(x1Numel_, RAW_BLK_ELEMS));
            Cast(x2, x2Raw, RoundMode::CAST_NONE, CeilAlign(x2Numel_, RAW_BLK_ELEMS2));
            PipeBarrier<PIPE_ALL>();
            ExpandTinyGeneric(a, x1, x1Strides_);
            ExpandTinyGeneric(b, x2, x2Strides_);
            PipeBarrier<PIPE_ALL>();
            ComputeChainFastInf(a, b, scratch, y, mx, mx, count);
        }

        LocalTensor<T3> out;
        if constexpr (IS_FP32) {
            out = y;
        } else {
            out = raw[0].template ReinterpretCast<T3>();
            Cast(out, y, CAST_BACK_MODE, totalLength_);
            PipeBarrier<PIPE_V>();
        }
        copyParams.blockLen = totalLength_ * sizeof(T3);
        DataCopyPad(yGm, out, copyParams);
    }

    __aicore__ inline bool IsNativeExpansion(const HypotBrcPlan& p, uint32_t scalarBit, uint32_t R) const
    {
        if ((scalarFlag_ & scalarBit) != 0U || p.mode == HYPOT_BRC_NONE) {
            return true;
        }
        const uint32_t srcLen = (p.fullSplit ? R : 1) * p.L * p.I;
        return p.mode != HYPOT_BRC_NONE && srcLen == 1;
    }

    __aicore__ inline LocalTensor<T1> CopyInFlatNative(GlobalTensor<T1>& gm, int64_t gmOff, uint32_t srcLen,
                                                       TQue<QuePosition::VECIN, HYPOT_BUFFER_NUM>& queue)
    {
        LocalTensor<T1> src = queue.template AllocTensor<T1>();
        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = srcLen * sizeof(T1);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(src, gm[gmOff], copyParams, {false, 0, 0, 0});
        queue.EnQue(src);
        return queue.template DeQue<T1>();
    }

    __aicore__ inline LocalTensor<T1> ExpandInputNative(GlobalTensor<T1>& gm, int64_t gmOff, uint32_t R,
                                                        const HypotBrcPlan& p, uint32_t scalarBit, TBuf<>& expBuf,
                                                        TQue<QuePosition::VECIN, HYPOT_BUFFER_NUM>& queue,
                                                        uint32_t count, bool& ownsQueue)
    {
        ownsQueue = false;
        LocalTensor<T1> exp = expBuf.Get<T1>();
        if ((scalarFlag_ & scalarBit) != 0U) {
            Duplicate(exp, static_cast<T1>(scalarBit == HYPOT_SCALAR_FLAG_X1 ? scalarX1_ : scalarX2_), count);
            return exp;
        }
        const uint32_t srcLen = (p.fullSplit ? R : 1) * p.L * p.I;
        if (p.mode == HYPOT_BRC_NONE) {
            ownsQueue = true;
            return CopyInFlatNative(gm, gmOff, srcLen, queue);
        }
        const uint32_t preloadLen = (scalarBit == HYPOT_SCALAR_FLAG_X1) ? x1PreloadLen_ : x2PreloadLen_;
        if (gmOff >= 0 && static_cast<uint64_t>(gmOff) < preloadLen) {
            LocalTensor<T1> pre = (scalarBit == HYPOT_SCALAR_FLAG_X1) ? bufPreX1.Get<T1>() : bufPreX2.Get<T1>();
            Duplicate(exp, pre.GetValue(static_cast<uint32_t>(gmOff)), count);
            return exp;
        }
        // single-value broadcast: stage one native-dtype element and splat it.
        const uint32_t slot = (scalarBit == HYPOT_SCALAR_FLAG_X1) ? 0U : 1U;
        if (gmOff != smallSrcTOff_[slot]) {
            LocalTensor<T1> s = queue.template AllocTensor<T1>();
            DataCopyParams cp;
            cp.blockCount = 1;
            cp.blockLen = sizeof(T1);
            cp.srcStride = 0;
            cp.dstStride = 0;
            DataCopyPad(s, gm[gmOff], cp, {false, 0, 0, 0});
            queue.EnQue(s);
            s = queue.template DeQue<T1>();
            PipeBarrier<PIPE_ALL>();
            smallSrcTVal_[slot] = s.GetValue(0);
            smallSrcTOff_[slot] = gmOff;
            PipeBarrier<PIPE_ALL>();
            queue.FreeTensor(s);
        }
        Duplicate(exp, smallSrcTVal_[slot], count);
        return exp;
    }

    __aicore__ inline void ProcessMultiDimTileNative(int64_t gmOff1, int64_t gmOff2, int64_t gmOffOut, uint32_t R,
                                                     uint32_t tileElems, uint32_t count)
    {
        bool ownsQueue1 = false;
        bool ownsQueue2 = false;
        LocalTensor<T1> a = ExpandInputNative(x1Gm, gmOff1, R, x1Plan_, HYPOT_SCALAR_FLAG_X1, bufExpA, inQueueX1, count,
                                              ownsQueue1);
        LocalTensor<T1> b = ExpandInputNative(x2Gm, gmOff2, R, x2Plan_, HYPOT_SCALAR_FLAG_X2, bufExpB, inQueueX2, count,
                                              ownsQueue2);
        PipeBarrier<PIPE_ALL>();
        LocalTensor<T1> mx = bufMx.Get<T1>();
        LocalTensor<T1> chain = bufChain.Get<T1>();
        LocalTensor<T1> yOut = outQueueY.template AllocTensor<T1>();
        ComputeChainNative(a, b, chain, yOut, mx, count);
        if (ownsQueue1) {
            inQueueX1.FreeTensor(a);
        }
        if (ownsQueue2) {
            inQueueX2.FreeTensor(b);
        }
        outQueueY.EnQue(yOut);

        LocalTensor<T1> yLocal = outQueueY.template DeQue<T1>();
        DataCopyParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = tileElems * sizeof(T1);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        DataCopyPad(yGm[gmOffOut], yLocal, copyOutParams);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline bool CanUsePadded2D() const
    {
        if constexpr (!IS_FP32) {
            return false;
        }
        if constexpr (IS_FP32) {
            // The padded-2D path is unsafe only when it processes multiple split-axis rows on
            // DAV_2201. Keep the single-row path used by rank-8 boundary cases.
            if (ubFormer_ > 1U) {
                return false;
            }
        }
        if constexpr (SCENE != HYPOT_SCENE_MULTI_DIM_BRC) {
            return false;
        }
        const bool multiDim3Axis1 = dimNum_ == 3U && ubSplitAxis_ == 1U;
        const bool multiDim2Axis0 = dimNum_ == 2U && ubSplitAxis_ == 0U;
        const bool unaligned3D = multiDim3Axis1 && innerLen_ % HYPOT_BLK_ELEMS != 0;
        const bool aligned2D = multiDim2Axis0 && innerLen_ % HYPOT_BLK_ELEMS == 0;
        if constexpr (IS_FP32) {
            if (!unaligned3D) {
                return false;
            }
        }
        if (innerLen_ == 0U || innerLen_ >= 128U || (!unaligned3D && !aligned2D)) {
            return false;
        }
        if (x1Plan_.mode == HYPOT_BRC_GENERIC || x2Plan_.mode == HYPOT_BRC_GENERIC || x1Plan_.foldUnit ||
            x2Plan_.foldUnit || x1Plan_.L != 1U || x2Plan_.L != 1U) {
            return false;
        }
        const uint32_t np = CeilAlign(innerLen_, HYPOT_BLK_ELEMS);
        const uint32_t paddedElems = ubFormer_ * np;
        return paddedElems + HYPOT_BUF_SLACK <= bufElems_;
    }

    template <typename Tin>
    __aicore__ inline LocalTensor<float> ExpandPadded2DInput(GlobalTensor<Tin>& gm, int64_t gmOff, uint32_t R,
                                                             const HypotBrcPlan& p, uint32_t scalarBit, TBuf<>& expBuf,
                                                             TBuf<>& f32BufT,
                                                             TQue<QuePosition::VECIN, HYPOT_BUFFER_NUM>& queue,
                                                             uint32_t np, bool& ownsQueue)
    {
        ownsQueue = false;
        LocalTensor<float> exp = expBuf.Get<float>();
        LocalTensor<float> f32Buf = f32BufT.Get<float>();
        if ((scalarFlag_ & scalarBit) != 0U) {
            Duplicate(exp, scalarBit == HYPOT_SCALAR_FLAG_X1 ? scalarX1_ : scalarX2_, R * np);
            return exp;
        }
        if (p.mode == HYPOT_BRC_NONE) {
            return StagePaddedSrc(gm, gmOff, 0, R, np, innerLen_, queue, f32Buf, ownsQueue);
        }
        if (p.mode == HYPOT_BRC_ROW) {
            LocalTensor<float> src = StagePaddedSrc(gm, gmOff, 0, 1, np, innerLen_, queue, f32Buf, ownsQueue);
            uint32_t dstShape[2] = {R, np};
            uint32_t srcShape[2] = {1, np};
            Broadcast<float, 2, 0>(exp, src, dstShape, srcShape, brcTmpU8);
            PipeBarrier<PIPE_V>();
            if (ownsQueue) {
                if (tileNum_ > 1U) {
                    PipeBarrier<PIPE_ALL>();
                    queue.FreeTensor(src);
                }
                ownsQueue = false;
            }
            return exp;
        }

        LocalTensor<Tin> raw = queue.template AllocTensor<Tin>();
        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = R * sizeof(Tin);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(raw, gm[gmOff], copyParams, {false, 0, 0, 0});
        queue.EnQue(raw);
        raw = queue.template DeQue<Tin>();
        LocalTensor<float> srcF;
        if constexpr (IsSameType<Tin, float>::value) {
            srcF = raw;
        } else {
            Cast(f32Buf, raw, RoundMode::CAST_NONE, CeilAlign(R, HYPOT_BLK_ELEMS));
            PipeBarrier<PIPE_V>();
            queue.FreeTensor(raw);
            srcF = f32Buf;
        }
        uint32_t dstShape[2] = {R, np};
        uint32_t srcShape[2] = {R, 1};
        Broadcast<float, 2, 1>(exp, srcF, dstShape, srcShape, brcTmpU8);
        PipeBarrier<PIPE_V>();
        if constexpr (IsSameType<Tin, float>::value) {
            if (tileNum_ > 1U) {
                PipeBarrier<PIPE_ALL>();
                queue.FreeTensor(raw);
            }
            ownsQueue = false;
        }
        return exp;
    }

    __aicore__ inline void ProcessMultiDimPadded2D(int64_t gmOff1, int64_t gmOff2, int64_t gmOffOut, uint32_t R)
    {
        const uint32_t I = innerLen_;
        const uint32_t np = CeilAlign(I, HYPOT_BLK_ELEMS);
        const uint32_t paddedCount = CeilAlign(R * np, HYPOT_CMP_ALIGN);
        const uint32_t tileElems = R * I;

        bool ownsQueue1 = false;
        bool ownsQueue2 = false;
        LocalTensor<float> a = ExpandPadded2DInput(x1Gm, gmOff1, R, x1Plan_, HYPOT_SCALAR_FLAG_X1, bufExpA, bufA,
                                                   inQueueX1, np, ownsQueue1);
        LocalTensor<float> b = ExpandPadded2DInput(x2Gm, gmOff2, R, x2Plan_, HYPOT_SCALAR_FLAG_X2, bufExpB, bufB,
                                                   inQueueX2, np, ownsQueue2);

        LocalTensor<T3> yOut = outQueueY.template AllocTensor<T3>();
        if constexpr (OUT_FP32) {
            LocalTensor<float> scratch = bufMx.Get<float>();
            LocalTensor<float> yPad = bufChain.Get<float>();
            LocalTensor<float> constBuf = bufBrcTmp.Get<float>();
            ComputeChain(a, b, scratch, yPad, constBuf, paddedCount);
            PipeBarrier<PIPE_V>();
            uint32_t k = 0;
            while (k < R) {
                uint32_t rep = (R - k) < HYPOT_CROP_REP ? (R - k) : HYPOT_CROP_REP;
                uint64_t rsvdCnt = 0;
                GatherMask<float, uint32_t>(
                    yOut[k * I], yPad[k * np], patX1, true, np,
                    GatherMaskParams{1, static_cast<uint16_t>(rep), static_cast<uint16_t>(np / HYPOT_BLK_ELEMS), 0},
                    rsvdCnt);
                k += rep;
            }
        } else {
            LocalTensor<float> scratch = bufChain.Get<float>();
            LocalTensor<float> mx = bufMx.Get<float>();
            LocalTensor<float> denom = bufDenom.Get<float>();
            ComputeChain(a, b, scratch, scratch, mx, paddedCount);
            PipeBarrier<PIPE_V>();
            LocalTensor<float> compact = bufMx.Get<float>();
            uint32_t k = 0;
            while (k < R) {
                uint32_t rep = (R - k) < HYPOT_CROP_REP ? (R - k) : HYPOT_CROP_REP;
                uint64_t rsvdCnt = 0;
                GatherMask<float, uint32_t>(
                    compact[k * I], scratch[k * np], patX1, true, np,
                    GatherMaskParams{1, static_cast<uint16_t>(rep), static_cast<uint16_t>(np / HYPOT_BLK_ELEMS), 0},
                    rsvdCnt);
                k += rep;
            }
            PipeBarrier<PIPE_V>();
            Cast(yOut, compact, CAST_BACK_MODE, tileElems);
        }

        if (ownsQueue1) {
            inQueueX1.FreeTensor(a);
        }
        if (ownsQueue2) {
            inQueueX2.FreeTensor(b);
        }
        outQueueY.EnQue(yOut);
        LocalTensor<T3> yLocal = outQueueY.template DeQue<T3>();
        DataCopyParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = tileElems * sizeof(T3);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        DataCopyPad(yGm[gmOffOut], yLocal, copyOutParams);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void ProcessMultiDim()
    {
        uint64_t splitExtent = static_cast<uint64_t>(dimProduct_) * outDims_[ubSplitAxis_];
        if (splitExtent != 0U) {
            innerLen_ = static_cast<uint32_t>(totalLength_ / splitExtent);
        }
        int64_t axesIndices[HYPOT_MAX_DIM] = {0};
        BroadcastGetAxesIndices(axesIndices, static_cast<int64_t>(startTile_));

        for (uint32_t t = 0; t < tileNum_; t++) {
            if (t != 0) {
                BroadcastUpdateAxesIndices(axesIndices);
            }
            // R keeps the tile-row notation: rows of the current tile on ubSplitAxis
            uint32_t R = (axesIndices[ubSplitAxis_] == static_cast<int64_t>(ubOuter_ - 1)) ? ubTail_ : ubFormer_;
            // compact continuous layout: tile = R * innerLen_ flat elements
            uint32_t tileElems = R * innerLen_;
            uint32_t count = CeilAlign(tileElems, HYPOT_CMP_ALIGN);

            int64_t gmOff1 = GetGmOffset(axesIndices, x1Strides_);
            int64_t gmOff2 = GetGmOffset(axesIndices, x2Strides_);
            int64_t gmOffOut = GetGmOffset(axesIndices, outStrides_);

            if (CanUsePadded2D()) {
                ProcessMultiDimPadded2D(gmOff1, gmOff2, gmOffOut, R);
                continue;
            }

            if constexpr (SAME_DTYPE && IS_HALF) {
                if (nativeHalfEnabled_ && IsNativeExpansion(x1Plan_, HYPOT_SCALAR_FLAG_X1, R) &&
                    IsNativeExpansion(x2Plan_, HYPOT_SCALAR_FLAG_X2, R)) {
                    ProcessMultiDimTileNative(gmOff1, gmOff2, gmOffOut, R, tileElems, count);
                    continue;
                }
            }

            bool ownsQueue1 = false;
            bool ownsQueue2 = false;
            LocalTensor<T1> ownedSrc1;
            LocalTensor<T2> ownedSrc2;
            // bufA/bufB exist only when the corresponding input is FP16/BF16; an FP32 input passes
            // bufChain as the (never dereferenced for computation) f32 placeholder
            LocalTensor<float> a;
            LocalTensor<float> b;
            if constexpr (IS_FP32) {
                a = ExpandInput(x1Gm, gmOff1, R, x1Plan_, HYPOT_SCALAR_FLAG_X1, bufExpA, bufChain, inQueueX1, count,
                                ownsQueue1, ownedSrc1, patX1, x1Strides_, axesIndices);
            } else {
                a = ExpandInput(x1Gm, gmOff1, R, x1Plan_, HYPOT_SCALAR_FLAG_X1, bufExpA, bufA, inQueueX1, count,
                                ownsQueue1, ownedSrc1, patX1, x1Strides_, axesIndices);
            }
            if constexpr (IN2_FP32) {
                b = ExpandInput(x2Gm, gmOff2, R, x2Plan_, HYPOT_SCALAR_FLAG_X2, bufExpB, bufChain, inQueueX2, count,
                                ownsQueue2, ownedSrc2, patX2, x2Strides_, axesIndices);
            } else {
                b = ExpandInput(x2Gm, gmOff2, R, x2Plan_, HYPOT_SCALAR_FLAG_X2, bufExpB, bufB, inQueueX2, count,
                                ownsQueue2, ownedSrc2, patX2, x2Strides_, axesIndices);
            }

            LocalTensor<float> mx = bufMx.Get<float>();
            LocalTensor<float> chain = bufChain.Get<float>();
            LocalTensor<T3> yOut = outQueueY.template AllocTensor<T3>();
            if constexpr (OUT_FP32) {
                LocalTensor<float> yF = yOut;
                ComputeChainFast(a, b, chain, yF, mx, yF, count);
            } else {
                LocalTensor<float> denom = bufDenom.Get<float>();
                ComputeChainFastInf(a, b, chain, denom, mx, mx, count);
                Cast(yOut, denom, CAST_BACK_MODE, count);
            }
            outQueueY.EnQue(yOut);
            LocalTensor<T3> yLocal = outQueueY.template DeQue<T3>();
            // compact layout: the tile output is a single GM continuous run of R*innerLen_ elements
            DataCopyParams copyOutParams;
            copyOutParams.blockCount = 1;
            copyOutParams.blockLen = tileElems * sizeof(T3);
            copyOutParams.srcStride = 0;
            copyOutParams.dstStride = 0;
            DataCopyPad(yGm[gmOffOut], yLocal, copyOutParams);
            outQueueY.FreeTensor(yLocal);
            if (ownsQueue1) {
                inQueueX1.FreeTensor(ownedSrc1);
            }
            if (ownsQueue2) {
                inQueueX2.FreeTensor(ownedSrc2);
            }
        }
    }

    // ---------------- Stream scene ----------------
    // Host-guaranteed (see ApplyStreamTiling): output [M, N] contiguous, one full input plus a
    // COL-single [M,1] or ROW-periodic [1,N] broadcast; no expansion engine.
    __aicore__ inline void ProcessMultiDimStream()
    {
        const uint32_t dimM = outDims_[0];
        const uint32_t dimN = outDims_[1];
        const bool x1Full = (x1BrcMode_ == HYPOT_BRC_NONE);
        const bool colStream = x1Full ? (x2BrcMode_ == HYPOT_BRC_COL) : (x1BrcMode_ == HYPOT_BRC_COL);
        GlobalTensor<T1>& gmFull = x1Full ? x1Gm : x2Gm;
        GlobalTensor<T1>& gmBrc = x1Full ? x2Gm : x1Gm;
        TQue<QuePosition::VECIN, HYPOT_BUFFER_NUM>& queueFull = x1Full ? inQueueX1 : inQueueX2;
        TQue<QuePosition::VECIN, HYPOT_BUFFER_NUM>& queueBrc = x1Full ? inQueueX2 : inQueueX1;
        LocalTensor<float> bufFull = bufA.Get<float>(); // FP32 view of the full-side tile
        LocalTensor<float> mx = bufMx.Get<float>();
        LocalTensor<float> chain = bufChain.Get<float>();

        if (colStream) {
            // row-major tile walk; the broadcast value is the preloaded per-row scalar and the
            // chain runs scalar-vector (no b tensor is materialized)
            union {
                uint32_t u;
                float f;
            } infU;
            infU.u = HYPOT_POS_INF_BITS;
            LocalTensor<T1> pre = x1Full ? bufPreX2.Get<T1>() : bufPreX1.Get<T1>();
            int64_t axesIndices[HYPOT_MAX_DIM] = {0};
            BroadcastGetAxesIndices(axesIndices, static_cast<int64_t>(startTile_));
            for (uint32_t t = 0; t < tileNum_; t++) {
                if (t != 0) {
                    BroadcastUpdateAxesIndices(axesIndices);
                }
                const uint32_t curLen = (axesIndices[1] == static_cast<int64_t>(ubOuter_ - 1)) ? ubTail_ : ubFormer_;
                const uint32_t count = CeilAlign(curLen, HYPOT_CMP_ALIGN);
                const int64_t gmOff = axesIndices[0] * static_cast<int64_t>(dimN) +
                                      axesIndices[1] * static_cast<int64_t>(ubFormer_);
                bool ownsQueue = false;
                LocalTensor<T1> ownedSrc;
                LocalTensor<float> a = CopyInFlatDeferred(gmFull, gmOff, curLen, queueFull, bufFull, ownsQueue,
                                                          ownedSrc);
                const float bv = ReadElemAsFloat(pre, static_cast<uint32_t>(axesIndices[0]));
                const float absB = (bv < 0.0f) ? -bv : bv; // NaN passes through (NaN < 0 is false)
                ComputeChainStreamColInf(a, absB, absB == infU.f, chain, mx, count);
                CopyOutStreamTile(gmOff, curLen, count, chain);
                if (ownsQueue) {
                    queueFull.FreeTensor(ownedSrc);
                }
            }
            return;
        }

        // ROW-periodic: inner-major tile walk so consecutive tiles share the broadcast segment; the FP32
        // segment is cached in bufBrc and reloaded only when the inner block changes.
        LocalTensor<float> bufBrc = bufB.Get<float>(); // FP32 broadcast-side segment cache
        uint32_t row = static_cast<uint32_t>(startTile_ % dimM);
        uint32_t innerBlk = static_cast<uint32_t>(startTile_ / dimM);
        int64_t cachedInnerBlk = -1;
        for (uint32_t t = 0; t < tileNum_; t++) {
            const uint32_t curLen = (innerBlk == ubOuter_ - 1) ? ubTail_ : ubFormer_;
            const uint32_t count = CeilAlign(curLen, HYPOT_CMP_ALIGN);
            if (cachedInnerBlk != static_cast<int64_t>(innerBlk)) {
                bool ownsQueueB = false;
                LocalTensor<T1> ownedSrcB;
                CopyInFlatDeferred(gmBrc, static_cast<int64_t>(innerBlk) * ubFormer_, curLen, queueBrc, bufBrc,
                                   ownsQueueB, ownedSrcB);
                if (ownsQueueB) {
                    queueBrc.FreeTensor(ownedSrcB);
                }
                cachedInnerBlk = static_cast<int64_t>(innerBlk);
            }
            const int64_t gmOff = static_cast<int64_t>(row) * dimN + static_cast<int64_t>(innerBlk) * ubFormer_;
            bool ownsQueue = false;
            LocalTensor<T1> ownedSrc;
            LocalTensor<float> a = CopyInFlatDeferred(gmFull, gmOff, curLen, queueFull, bufFull, ownsQueue, ownedSrc);
            ComputeChainStreamInf(a, bufBrc, chain, mx, count);
            CopyOutStreamTile(gmOff, curLen, count, chain);
            if (ownsQueue) {
                queueFull.FreeTensor(ownedSrc);
            }
            row++;
            if (row == dimM) {
                row = 0;
                innerBlk++;
            }
        }
    }

    __aicore__ inline void CopyOutStreamTile(int64_t gmOff, uint32_t tileElems, uint32_t count,
                                             LocalTensor<float>& resultF)
    {
        LocalTensor<T3> yOut = outQueueY.template AllocTensor<T3>();
        Cast(yOut, resultF, CAST_BACK_MODE, count);
        outQueueY.EnQue(yOut);
        LocalTensor<T3> yLocal = outQueueY.template DeQue<T3>();
        DataCopyParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = tileElems * sizeof(T3);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        DataCopyPad(yGm[gmOff], yLocal, copyOutParams);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TBuf<> bufTiny;
    TQue<QuePosition::VECIN, HYPOT_BUFFER_NUM> inQueueX1;
    TQue<QuePosition::VECIN, HYPOT_BUFFER_NUM> inQueueX2;
    TQue<QuePosition::VECOUT, HYPOT_BUFFER_NUM> outQueueY;
    TBuf<> bufExpA;
    TBuf<> bufExpB;
    TBuf<> bufA;
    TBuf<> bufB;
    TBuf<> bufDenom;
    TBuf<> bufMx;
    TBuf<> bufChain;
    TBuf<> bufMask1;
    TBuf<> bufMask2;
    TBuf<> bufScalar;
    TBuf<> bufPreX1;
    TBuf<> bufPreX2;
    TBuf<> bufBrcTmp; // MultiDimBroadcast only: static Broadcast sharedTmpBuffer + crop patterns

    GlobalTensor<T1> x1Gm;
    GlobalTensor<T2> x2Gm;
    GlobalTensor<T3> yGm;

    uint64_t totalLength_ = 0; // broadcast output total element num; from tilingData->totalLength
    uint32_t blockIdx_ = 0;    // current AIV core index; from GetBlockIdx()
    uint32_t usedCoreNum_ = 0; // actual used AIV core num (blockDim); from tilingData->usedCoreNum
    uint32_t blockFormer_ = 0; // tiles per non-last core; from tilingData->blockFormer
    uint32_t blockTail_ = 0;   // tiles of the last core; from tilingData->blockTail
    uint32_t ubFormer_ = 0;    // OneDim: elements per tile; MultiDim: rows per tile; from tilingData->ubFormer
    uint32_t ubTail_ = 0;      // tail tile elements (OneDim) / rows (MultiDim); from tilingData->ubTail
    uint32_t ubOuter_ = 0;     // tile count on the split axis; from tilingData->ubOuter
    uint32_t scalarFlag_ = 0;  // scalar input bitmap (HYPOT_SCALAR_FLAG_*); from tilingData->scalarFlag
    uint32_t x1Numel_ = 0;
    uint32_t x2Numel_ = 0;
    uint32_t x1PreloadLen_ = 0; // small x1 source length staged once at Init
    uint32_t x2PreloadLen_ = 0; // small x2 source length staged once at Init
    bool nativeHalfEnabled_ = false;
    uint32_t tileNum_ = 0;   // tiles of this core; derived from blockFormer_/blockTail_ by blockIdx_
    uint64_t startTile_ = 0; // global index of this core's first tile; blockIdx_ * blockFormer_
    uint32_t bufElems_ = 0;  // float elements of every compute buffer; derived from ubFormer_/innerLen_
    bool tinyMode_ = false;
    bool tinyCrossMode_ = false;
    bool tinyGenericMode_ = false;
    bool singleSidedBroadcast_ = false;

    // MultiDimBroadcast scene fields
    uint32_t dimNum_ = 0;                      // fused dim num; from tilingData->dimNum
    uint32_t ubSplitAxis_ = 0;                 // UB split axis; from tilingData->ubSplitAxis
    uint32_t dimProduct_ = 0;                  // product of dims[0..ubSplitAxis-1]; from tilingData->dimProduct
    uint32_t innerLen_ = 0;                    // product of dims[ubSplitAxis_+1..dimNum-1]; from tilingData->innerLen
    uint32_t x1BrcMode_ = 0;                   // x1 broadcast expansion mode (HYPOT_BRC_*); from tilingData->x1BrcMode
    uint32_t x2BrcMode_ = 0;                   // x2 broadcast expansion mode (HYPOT_BRC_*); from tilingData->x2BrcMode
    uint32_t outDims_[HYPOT_MAX_DIM] = {0};    // fused output dims; from tilingData->outShape
    uint32_t x1Strides_[HYPOT_MAX_DIM] = {0};  // x1 strides in elements; from tilingData->x1Strides
    uint32_t x2Strides_[HYPOT_MAX_DIM] = {0};  // x2 strides in elements; from tilingData->x2Strides
    uint32_t outStrides_[HYPOT_MAX_DIM] = {0}; // output strides in elements; from tilingData->outStrides

    // expansion plans (launch-constant) + crop patterns + static Broadcast tmp view
    HypotBrcPlan x1Plan_;          // x1 expansion plan; built by BuildBrcPlan at Init
    HypotBrcPlan x2Plan_;          // x2 expansion plan; built by BuildBrcPlan at Init
    LocalTensor<uint32_t> patX1;   // x1 GatherMask crop pattern words inside bufBrcTmp; set by BuildPatterns
    LocalTensor<uint32_t> patX2;   // x2 GatherMask crop pattern words inside bufBrcTmp; set by BuildPatterns
    LocalTensor<uint8_t> brcTmpU8; // static Broadcast sharedTmpBuffer view inside bufBrcTmp; set at Init

    float scalarX1_ = 0.0f; // scalar inputs are preloaded and widened to FP32 at init
    float scalarX2_ = 0.0f; // x2 scalar value preloaded at init (see scalarX1_)

    // Per-input value cache for the broadcast-scalar path, keyed by GM offset: re-staging per tile costs
    // a queue round-trip + 2x PipeBarrier<PIPE_ALL> and breaks the double-buffer overlap.
    float smallSrcVal_[2] = {0.0f, 0.0f}; // cached staged values, indexed by input slot (0=x1, 1=x2)
    int64_t smallSrcOff_[2] = {-1, -1};   // GM offset key of each cached value; -1 = cache empty
    T1 smallSrcTVal_[2];                  // native-dtype values for the FP16/BF16 fast path
    int64_t smallSrcTOff_[2] = {-1, -1};  // native fast-path cache keys
};

// Dispatch on the broadcast scene only; dtypes arrive through the per-combination
// DTYPE_SELF/DTYPE_OTHER/DTYPE_OUT aliases. ONE_DIM/MULTI_DIM_BRC/VECIDX support mixed dtypes;
// STRUCT and STREAM are same-dtype only (host-gated).
template <typename T1, typename T2, typename T3, int BROADCAST_SCENE>
__aicore__ inline void HypotKernelImpl(__gm__ uint8_t* x1, __gm__ uint8_t* x2, __gm__ uint8_t* y,
                                       const HypotTilingData* tilingData)
{
    constexpr bool kSameDtype = IsSameType<T1, T2>::value && IsSameType<T2, T3>::value;
    if constexpr (BROADCAST_SCENE == HYPOT_SCENE_ONE_DIM || BROADCAST_SCENE == HYPOT_SCENE_MULTI_DIM_BRC) {
        KernelHypot<T1, T2, T3, BROADCAST_SCENE> op;
        op.Init(x1, x2, y, tilingData);
        op.Process();
    } else if constexpr (BROADCAST_SCENE == HYPOT_SCENE_MULTI_DIM_VECIDX) {
        // VECIDX stages each input in its own raw dtype, so mixed combinations are supported
        KernelHypot<T1, T2, T3, BROADCAST_SCENE> op;
        op.Init(x1, x2, y, tilingData);
        op.Process();
    } else if constexpr (BROADCAST_SCENE == HYPOT_SCENE_MULTI_DIM_STRUCT) {
        if constexpr (kSameDtype) {
            KernelHypot<T1, T2, T3, BROADCAST_SCENE> op;
            op.Init(x1, x2, y, tilingData);
            op.Process();
        }
    } else if constexpr (BROADCAST_SCENE == HYPOT_SCENE_MULTI_DIM_STREAM) {
        if constexpr (kSameDtype && (IsSameType<T1, half>::value || IsSameType<T1, bfloat16_t>::value)) {
            KernelHypot<T1, T2, T3, BROADCAST_SCENE> op;
            op.Init(x1, x2, y, tilingData);
            op.Process();
        }
    }
}

} // namespace NsHypot
#endif // HYPOT_STRUCT_H
