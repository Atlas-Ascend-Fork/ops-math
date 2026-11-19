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
 * \file histogram_v2_simt.h
 * \brief Unified arch35 HistogramV2 SIMT kernel.
 *
 * One template class parameterised by load mode and the deterministic-fp32out
 * scheme (schId, see histogram_v2_tiling_key.h), replacing the legacy six
 * per-variant headers. The bin-index semantics live in exactly one place
 * (HistogramV2Common::ComputeBinIndex, histogram_v2_simt_common.h); all mode
 * differences are compile-time pruned via if constexpr.
 *
 * Deterministic fp32 output (schId >= 3) is a three-phase algorithm with no
 * float atomicAdd on GM:
 *   Phase 1: clearY cores zero the output GM (as int32) -> SyncAll
 *   Phase 2: x cores accumulate int32 (UB histogram + SetAtomicAdd, or direct
 *            GM asc_atomic_add on the GM_ATOMIC mode) -> SyncAll
 *   Phase 3: clearY cores cast their GM int32 slice in place to GM fp32
 * The GM_ATOMIC deterministic variant (schId 5) is new: the legacy 1311/1317
 * tiling keys silently ran the non-deterministic float GM atomic path.
 */
#ifndef HISTOGRAM_V2_SIMT_H
#define HISTOGRAM_V2_SIMT_H

#include "histogram_v2_tiling_key.h"
#include "histogram_v2_simt_base.h"

namespace HistogramV2SIMT {
using namespace AscendC;

// Clear y GM by SIMT threads (all cores enter; cores beyond clearYCoreNum no-op).
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtCleanVf(__gm__ T* yGmAddr, const int32_t blockIdx,
                                                                        const int64_t clearYCoreNum,
                                                                        const int64_t clearYIndexBase,
                                                                        const int64_t clearYDataLength)
{
    if (blockIdx >= clearYCoreNum) {
        return;
    }
    for (int32_t index = static_cast<int32_t>(threadIdx.x); index < clearYDataLength;
         index += static_cast<int32_t>(blockDim.x)) {
        int64_t yIndex = clearYIndexBase + index;
        yGmAddr[yIndex] = static_cast<T>(0);
    }
}

// Accumulate VF, single copy for all load modes. MODE == TPL_SCH_ID_UB_FULL (0):
// UB histogram, no window filter; MODE == TPL_SCH_ID_UB_WINDOW (1): UB histogram
// for the [ubLoop * ubNumCanUse, (ubLoop + 1) * ubNumCanUse) bin window;
// MODE == TPL_SCH_ID_GM_ATOMIC (2): direct GM atomic add (yLocalAddr unused).
// Caller guarantees the core actually participates (blockIdx < needXCoreNum).
template <typename X_TYPE, typename COMPUTE_TYPE, typename ACCUM_TYPE, uint64_t MODE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtAccumulateVf(
    __gm__ X_TYPE* xGmAddr, __ubuf__ ACCUM_TYPE* yLocalAddr, __gm__ ACCUM_TYPE* yGmAddr, const int64_t xIndexBase,
    const int64_t coreDataLength, const COMPUTE_TYPE minValue, const COMPUTE_TYPE maxValue,
    const COMPUTE_TYPE minMaxLength, const int64_t bins, const int64_t ubLoop, const int64_t ubNumCanUse)
{
    for (int32_t index = static_cast<int32_t>(threadIdx.x); index < coreDataLength;
         index += static_cast<int32_t>(blockDim.x)) {
        int64_t xIndex = xIndexBase + index;
        COMPUTE_TYPE value = HistogramV2Common::SimtCastTo<COMPUTE_TYPE>(xGmAddr[xIndex]);
        int32_t indexBin = 0;
        if (!HistogramV2Common::ComputeBinIndex(value, minValue, maxValue, minMaxLength, bins, indexBin)) {
            continue;
        }
        if constexpr (MODE == TPL_SCH_ID_UB_WINDOW) {
            if (indexBin < ubLoop * ubNumCanUse || indexBin >= (ubLoop + 1) * ubNumCanUse) {
                continue;
            }
            indexBin -= static_cast<int32_t>(ubLoop * ubNumCanUse);
        }
        if constexpr (MODE == TPL_SCH_ID_GM_ATOMIC) {
            asc_atomic_add(yGmAddr + indexBin, static_cast<ACCUM_TYPE>(1));
        } else {
            asc_atomic_add(yLocalAddr + indexBin, static_cast<ACCUM_TYPE>(1));
        }
    }
}

// Deterministic Phase 3: cast a GM slice in place from int32 to float.
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void GmCastIntToFp32Vf(__gm__ int32_t* yGmIntAddr,
                                                                              __gm__ float* yGmFloatAddr,
                                                                              const int64_t indexBase,
                                                                              const int64_t dataLength)
{
    for (int32_t index = static_cast<int32_t>(threadIdx.x); index < dataLength;
         index += static_cast<int32_t>(blockDim.x)) {
        int64_t yIndex = indexBase + index;
        int32_t yValue = yGmIntAddr[yIndex];
        yGmFloatAddr[yIndex] = static_cast<float>(yValue);
    }
}

template <typename X_TYPE, typename COMPUTE_TYPE, typename OUT_TYPE, uint64_t SCH_ID>
class HistogramV2Simt : public HistogramV2SimtBase<X_TYPE, COMPUTE_TYPE, OUT_TYPE> {
public:
    static constexpr uint64_t kMode = SCH_ID % 3; // 0=UB_FULL 1=UB_WINDOW 2=GM_ATOMIC
    // Deterministic scheme only applies to fp32 output; int32 output is already
    // deterministic (integer atomic add) and the host never emits det schIds for it.
    static constexpr bool kDetFp32Out = (SCH_ID >= TPL_SCH_ID_UB_FULL_DET) && IsSameType<OUT_TYPE, float>::value;
    using ACCUM_TYPE = typename ConditionalType<kDetFp32Out, int32_t, OUT_TYPE>::Type;

    __aicore__ inline HistogramV2Simt(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR min, GM_ADDR max, GM_ADDR y,
                                const HistogramV2SimtTilingData* __restrict tilingData, TPipe* tPipe)
    {
        this->BaseInit(x, min, max, y, tilingData, tPipe);
        if constexpr (kMode != TPL_SCH_ID_GM_ATOMIC) {
            int64_t ubElems = (kMode == TPL_SCH_ID_UB_FULL) ? this->bins_ : this->ubNumCanUse_;
            this->pipe_->InitBuffer(this->yQue_, 1, ubElems * sizeof(ACCUM_TYPE));
        }
    }

    __aicore__ inline void Process()
    {
        if (this->blockIdx_ >= GetBlockNum()) {
            return;
        }
        this->ProcessMinMaxValue();
        if constexpr (kDetFp32Out) {
            ProcessDetFp32Out();
        } else {
            ProcessNonDet();
        }
    }

private:
    // ---------------- non-deterministic paths (single SyncAll) ----------------
    __aicore__ inline void ProcessNonDet()
    {
        if constexpr (kMode == TPL_SCH_ID_UB_FULL) {
            // One Duplicate serves both as the clear source and as the accumulation
            // buffer zero-init (legacy full-load behaviour).
            yLocal_ = this->yQue_.template AllocTensor<ACCUM_TYPE>();
            Duplicate(yLocal_, static_cast<ACCUM_TYPE>(0), this->bins_);
            ClearYViaUb();
            this->SyncAllIfNotUt();
            AccumulateFullLoad();
            this->yQue_.template FreeTensor<ACCUM_TYPE>(yLocal_);
        } else {
            ClearYViaSimt();
            this->SyncAllIfNotUt();
            if constexpr (kMode == TPL_SCH_ID_UB_WINDOW) {
                AccumulateWindow();
            } else {
                AccumulateGmAtomic();
            }
        }
    }

    // ---------------- deterministic fp32-out paths (three phases) ----------------
    __aicore__ inline void ProcessDetFp32Out()
    {
        // Phase 1: zero the output GM (int32 view)
        if constexpr (kMode == TPL_SCH_ID_GM_ATOMIC) {
            ClearYViaSimt();
        } else {
            ClearYViaUb();
        }
        this->SyncAllIfNotUt();
        // Phase 2: int32 accumulate
        if constexpr (kMode == TPL_SCH_ID_UB_FULL) {
            AccumulateFullLoad();
        } else if constexpr (kMode == TPL_SCH_ID_UB_WINDOW) {
            AccumulateWindow();
        } else {
            AccumulateGmAtomic();
        }
        this->SyncAllIfNotUt();
        // Phase 3: in-place GM int32 -> fp32 cast
        CastGmIntToFp32();
    }

    // Clear via UB: Duplicate + DataCopyPad of this core's slice. FULL mode: yLocal_
    // is the accumulation buffer already zeroed over bins (non-det) or a scratch
    // buffer (det); det mode duplicates only the clear slice, matching the legacy
    // det variants.
    __aicore__ inline void ClearYViaUb()
    {
        if (this->blockIdx_ >= this->clearYCoreNum_) {
            return;
        }
        if constexpr (kDetFp32Out) {
            LocalTensor<ACCUM_TYPE> yLocal = this->yQue_.template AllocTensor<ACCUM_TYPE>();
            Duplicate(yLocal, static_cast<ACCUM_TYPE>(0), this->ClearYDataLength());
            this->yQue_.EnQue(yLocal);
            yLocal = this->yQue_.template DeQue<ACCUM_TYPE>();
            DataCopyExtParams dataCopyExtParams{static_cast<uint16_t>(1),
                                                static_cast<uint32_t>(this->ClearYDataLength() * sizeof(ACCUM_TYPE)), 0,
                                                0, 0};
            DataCopyPad(this->yGmInt_[this->ClearYIndexBase()], yLocal, dataCopyExtParams);
            this->yQue_.template FreeTensor<ACCUM_TYPE>(yLocal);
        } else {
            this->yQue_.EnQue(yLocal_);
            yLocal_ = this->yQue_.template DeQue<ACCUM_TYPE>();
            DataCopyExtParams dataCopyExtParams{static_cast<uint16_t>(1),
                                                static_cast<uint32_t>(this->ClearYDataLength() * sizeof(ACCUM_TYPE)), 0,
                                                0, 0};
            DataCopyPad(this->yGm_[this->ClearYIndexBase()], yLocal_, dataCopyExtParams);
        }
    }

    // Clear via SIMT threads writing GM directly (ACCUM_TYPE view of y).
    __aicore__ inline void ClearYViaSimt()
    {
        __gm__ ACCUM_TYPE* yGmAddr = (__gm__ ACCUM_TYPE*)this->yGm_.GetPhyAddr();
        asc_vf_call<SimtCleanVf<ACCUM_TYPE>>(dim3{THREAD_NUM, 1, 1}, yGmAddr, this->blockIdx_, this->clearYCoreNum_,
                                             this->ClearYIndexBase(), this->ClearYDataLength());
    }

    // UB_FULL accumulate: VF into the (already zeroed) UB histogram, then one
    // atomic DataCopyPad of the whole bins range to GM.
    __aicore__ inline void AccumulateFullLoad()
    {
        if (this->blockIdx_ >= this->needXCoreNum_) {
            return;
        }
        if constexpr (kDetFp32Out) {
            // Det Phase 2 uses a freshly allocated, fully zeroed UB buffer (the
            // Phase 1 scratch tensor was freed).
            yLocal_ = this->yQue_.template AllocTensor<ACCUM_TYPE>();
            Duplicate(yLocal_, static_cast<ACCUM_TYPE>(0), this->bins_);
            this->yQue_.EnQue(yLocal_);
            yLocal_ = this->yQue_.template DeQue<ACCUM_TYPE>();
        }
        __gm__ X_TYPE* xGmAddr = (__gm__ X_TYPE*)this->xGm_.GetPhyAddr();
        __ubuf__ ACCUM_TYPE* yLocalAddr = (__ubuf__ ACCUM_TYPE*)yLocal_.GetPhyAddr();
        asc_vf_call<SimtAccumulateVf<X_TYPE, COMPUTE_TYPE, ACCUM_TYPE, TPL_SCH_ID_UB_FULL>>(
            dim3{THREAD_NUM, 1, 1}, xGmAddr, yLocalAddr, nullptr, this->XIndexBase(), this->CoreDataLength(),
            this->minValue_, this->maxValue_, this->minMaxLength_, this->bins_, 0, 0);
        WaitSimdCompute();
        WritebackUbToGm(0, this->bins_);
        if constexpr (kDetFp32Out) {
            this->yQue_.template FreeTensor<ACCUM_TYPE>(yLocal_);
        }
    }

    // UB_WINDOW accumulate: loop over bin windows, per-window UB histogram +
    // atomic writeback of the window slice.
    __aicore__ inline void AccumulateWindow()
    {
        if (this->blockIdx_ >= this->needXCoreNum_) {
            return;
        }
        __gm__ X_TYPE* xGmAddr = (__gm__ X_TYPE*)this->xGm_.GetPhyAddr();
        for (int64_t ubLoop = 0; ubLoop < this->ubLoopNum_; ubLoop++) {
            int64_t yLocalNum = (ubLoop == this->ubLoopNum_ - 1) ?
                                    (this->bins_ - (this->ubLoopNum_ - 1) * this->ubNumCanUse_) :
                                    this->ubNumCanUse_;
            yLocal_ = this->yQue_.template AllocTensor<ACCUM_TYPE>();
            Duplicate(yLocal_, static_cast<ACCUM_TYPE>(0), yLocalNum);
            this->yQue_.EnQue(yLocal_);
            yLocal_ = this->yQue_.template DeQue<ACCUM_TYPE>();
            __ubuf__ ACCUM_TYPE* yLocalAddr = (__ubuf__ ACCUM_TYPE*)yLocal_.GetPhyAddr();
            asc_vf_call<SimtAccumulateVf<X_TYPE, COMPUTE_TYPE, ACCUM_TYPE, TPL_SCH_ID_UB_WINDOW>>(
                dim3{THREAD_NUM, 1, 1}, xGmAddr, yLocalAddr, nullptr, this->XIndexBase(), this->CoreDataLength(),
                this->minValue_, this->maxValue_, this->minMaxLength_, this->bins_, ubLoop, this->ubNumCanUse_);
            WaitSimdCompute();
            WritebackUbToGm(ubLoop * this->ubNumCanUse_, yLocalNum);
            this->yQue_.template FreeTensor<ACCUM_TYPE>(yLocal_);
        }
    }

    // GM_ATOMIC accumulate: direct GM atomic add, no UB usage.
    __aicore__ inline void AccumulateGmAtomic()
    {
        if (this->blockIdx_ >= this->needXCoreNum_) {
            return;
        }
        __gm__ X_TYPE* xGmAddr = (__gm__ X_TYPE*)this->xGm_.GetPhyAddr();
        __gm__ ACCUM_TYPE* yGmAddr = (__gm__ ACCUM_TYPE*)this->yGm_.GetPhyAddr();
        asc_vf_call<SimtAccumulateVf<X_TYPE, COMPUTE_TYPE, ACCUM_TYPE, TPL_SCH_ID_GM_ATOMIC>>(
            dim3{THREAD_NUM, 1, 1}, xGmAddr, nullptr, yGmAddr, this->XIndexBase(), this->CoreDataLength(),
            this->minValue_, this->maxValue_, this->minMaxLength_, this->bins_, 0, 0);
    }

    // Atomic writeback of a UB histogram slice to GM: int32 view on the
    // deterministic path, OUT_TYPE view otherwise.
    __aicore__ inline void WritebackUbToGm(int64_t gmOffset, int64_t numElems)
    {
        DataCopyExtParams dataCopyExtParamsAdd{static_cast<uint16_t>(1),
                                               static_cast<uint32_t>(numElems * sizeof(ACCUM_TYPE)), 0, 0, 0};
        if constexpr (kDetFp32Out) {
            SetAtomicAdd<int32_t>();
            DataCopyPad(this->yGmInt_[gmOffset], yLocal_, dataCopyExtParamsAdd);
        } else {
            SetAtomicAdd<OUT_TYPE>();
            DataCopyPad(this->yGm_[gmOffset], yLocal_, dataCopyExtParamsAdd);
        }
        SetAtomicNone();
    }

    // Deterministic Phase 3: clearY cores cast their GM slice int32 -> fp32.
    __aicore__ inline void CastGmIntToFp32()
    {
        if (this->blockIdx_ >= this->clearYCoreNum_) {
            return;
        }
        __gm__ int32_t* yGmIntAddr = (__gm__ int32_t*)this->yGmInt_.GetPhyAddr();
        __gm__ float* yGmFloatAddr = (__gm__ float*)this->yGmFloat_.GetPhyAddr();
        asc_vf_call<GmCastIntToFp32Vf>(dim3{THREAD_NUM, 1, 1}, yGmIntAddr, yGmFloatAddr, this->ClearYIndexBase(),
                                       this->ClearYDataLength());
    }

    __aicore__ inline void WaitSimdCompute()
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    }

private:
    // Accumulation buffer for the UB modes. Non-det FULL keeps it alive across
    // clear and accumulate; other paths alloc/free locally per use.
    LocalTensor<ACCUM_TYPE> yLocal_;
};

} // namespace HistogramV2SIMT

#endif // HISTOGRAM_V2_SIMT_H
