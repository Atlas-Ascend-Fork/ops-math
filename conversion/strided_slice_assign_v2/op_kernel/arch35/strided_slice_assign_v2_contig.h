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
 * \file strided_slice_assign_v2_contig.h
 * \brief T2 模板：约简后切片按连续段直通 DMA。全正：TQueBind<VECIN,VECOUT> 绑定双缓冲逐 chunk 直通；
 *        全负：VECIN/VECOUT 两级 TQue 双缓冲 + 寄存器索引反转（统一 Reg::Gather：
 *        2/4/8B 直采，1B 按 16bit lane 采集后 DIST_PACK_B16 打包写回，配方参考 as_strided_gather.h），
 *        全程无手动 SetFlag/WaitFlag。
 */

#ifndef STRIDED_SLICE_ASSIGN_V2_CONTIG_H_
#define STRIDED_SLICE_ASSIGN_V2_CONTIG_H_

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "strided_slice_assign_v2_tiling_data.h"

namespace StridedSliceAssignV2ContigNs {

using namespace AscendC;

constexpr int32_t SSA_KERNEL_BUFFER_NUM = 2;

template <typename T>
class StridedSliceAssignV2Contig {
public:
    __aicore__ inline explicit StridedSliceAssignV2Contig(TPipe* pipe) : Ppipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR inputValue, GM_ADDR varOut,
                                const StridedSliceAssignV2ContigTilingData* tilingData)
    {
        dimNum_ = tilingData->dimNum;
        blockDim_ = tilingData->blockDim;
        chunksPerSeg_ = tilingData->chunksPerSeg;
        chunkLen_ = tilingData->chunkLen;
        chunkRem_ = tilingData->chunkRem;
        totalChunks_ = tilingData->totalChunks;
        chunksPerCore_ = tilingData->chunksPerCore;
        lastCoreChunks_ = tilingData->lastCoreChunks;
        varBase_ = tilingData->varBase;
        contigRun_ = tilingData->contigRun;
        negative_ = (tilingData->directionId != 0);
        ubFactor_ = static_cast<int64_t>(tilingData->ubFactor);
        for (int32_t i = 0; i < SSA_MAX_DIM_NUM; ++i) {
            sliceShape_[i] = tilingData->sliceShape[i];
            axisStep_[i] = tilingData->axisStep[i];
            segCum_[i] = tilingData->segCum[i];
        }

        // in-place：varOut 与 var 同一 buffer，仅写切片覆盖区
        (void)var;
        inputValueGm_.SetGlobalBuffer((__gm__ T*)inputValue);
        varOutGm_.SetGlobalBuffer((__gm__ T*)varOut);

        if (negative_) {
            // 全负：两级 TQue 双缓冲 + 寄存器索引反转
            const int64_t slotBytes = ubFactor_ * static_cast<int64_t>(sizeof(T));
            Ppipe_->InitBuffer(inputQueue_, SSA_KERNEL_BUFFER_NUM, static_cast<uint32_t>(slotBytes));
            Ppipe_->InitBuffer(reverseQueue_, SSA_KERNEL_BUFFER_NUM, static_cast<uint32_t>(slotBytes));
        } else {
            // 全正：绑定双缓冲直通
            Ppipe_->InitBuffer(chunkBind_, SSA_KERNEL_BUFFER_NUM,
                               static_cast<uint32_t>(ubFactor_ * static_cast<int64_t>(sizeof(T))));
        }

        // 本核 chunk 区间：尾核 lastCoreChunks
        const int64_t blk = static_cast<int64_t>(GetBlockIdx());
        startChunk_ = blk * chunksPerCore_;
        chunksThisCore_ = (blk == blockDim_ - 1) ? lastCoreChunks_ : chunksPerCore_;
    }

    __aicore__ inline void Process()
    {
        if (startChunk_ >= totalChunks_ || chunksThisCore_ <= 0) {
            return;
        }
        int64_t s = startChunk_ / chunksPerSeg_;
        int64_t c = startChunk_ % chunksPerSeg_;
        int64_t segVarOff = SegVarHead(s);
        for (int64_t i = 0; i < chunksThisCore_; ++i) {
            const int64_t chunkStart = c * chunkLen_ + ((c < chunkRem_) ? c : chunkRem_);
            const int64_t curLen = chunkLen_ + ((c < chunkRem_) ? 1 : 0);
            // 全负从段尾取 chunk，UB 内反转后升地址写回
            const int64_t inOff = negative_ ? (s * contigRun_ + (contigRun_ - chunkStart - curLen)) :
                                              (s * contigRun_ + chunkStart);
            const int64_t varOff = segVarOff + chunkStart;
            ProcessOneChunk(inOff, varOff, curLen);
            ++c;
            if (c == chunksPerSeg_) {
                c = 0;
                segVarOff = SegVarHead(++s);
            }
        }
    }

private:
    __aicore__ inline void ProcessOneChunk(int64_t inOff, int64_t varOff, int64_t curLen)
    {
        const DataCopyExtParams copyParams = {1u, static_cast<uint32_t>(curLen * static_cast<int64_t>(sizeof(T))), 0u,
                                              0u, 0u};
        if (negative_) {
            ProcessNegativeChunk(inOff, varOff, curLen, copyParams);
        } else {
            ProcessPositiveChunk(inOff, varOff, copyParams);
        }
    }

    // 全正逐 chunk 直通：GM→UB→GM，同步由绑定队列自动管理
    __aicore__ inline void ProcessPositiveChunk(int64_t inOff, int64_t varOff, const DataCopyExtParams& copyParams)
    {
        LocalTensor<T> xLocal = chunkBind_.AllocTensor<T>();
        DataCopyPad(xLocal, inputValueGm_[inOff], copyParams, padParams_);
        chunkBind_.EnQue(xLocal);

        xLocal = chunkBind_.DeQue<T>();

        DataCopyPad(varOutGm_[varOff], xLocal, copyParams);
        chunkBind_.FreeTensor(xLocal);
    }

    // 全负逐 chunk：搬入 → 寄存器索引反转 → 搬出，依赖由两级队列自动管理
    __aicore__ inline void ProcessNegativeChunk(int64_t inOff, int64_t varOff, int64_t curLen,
                                                const DataCopyExtParams& copyParams)
    {
        LocalTensor<T> xLocal = inputQueue_.AllocTensor<T>();
        DataCopyPad(xLocal, inputValueGm_[inOff], copyParams, padParams_);
        inputQueue_.EnQue(xLocal);

        xLocal = inputQueue_.DeQue<T>();

        LocalTensor<T> revLocal = reverseQueue_.AllocTensor<T>();
        GatherReverse(revLocal, xLocal, curLen);
        reverseQueue_.EnQue(revLocal); // 自动保证向量写完成后再被 MTE3 读取

        revLocal = reverseQueue_.DeQue<T>();
        DataCopyPad(varOutGm_[varOff], revLocal, copyParams);
        reverseQueue_.FreeTensor(revLocal);
        inputQueue_.FreeTensor(xLocal);
    }

    // 反转：寄存器内生成反向元素索引 idx = curLen-1-e，统一 Reg::Gather 收集后写回（索引不落 UB）。
    // 1B 无 gather 语义，按 16bit lane 采集后 StoreAlign<DIST_PACK_B16> 打包写回
    __aicore__ inline void GatherReverse(const LocalTensor<T>& revLocal, const LocalTensor<T>& xLocal, int64_t curLen)
    {
        using RangeType = std::conditional_t<sizeof(T) <= sizeof(int16_t), int16_t, int32_t>;
        using IdxType = std::conditional_t<sizeof(T) <= sizeof(int16_t), uint16_t, uint32_t>;
        using CastType = std::conditional_t<sizeof(T) == 1, int16_t, T>;

        const uint32_t cnt = static_cast<uint32_t>(curLen);
        const uint32_t blkElems = VF_OFF_BYTES / static_cast<uint32_t>(sizeof(CastType));
        const uint16_t reps = static_cast<uint16_t>((cnt + blkElems - 1) / blkElems);

        __ubuf__ T* srcPtr = reinterpret_cast<__ubuf__ T*>(xLocal.GetPhyAddr());
        __ubuf__ T* dstPtr = reinterpret_cast<__ubuf__ T*>(revLocal.GetPhyAddr());

        uint32_t rest = cnt;

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<RangeType> idxReg;
            AscendC::Reg::RegTensor<CastType> dataReg;
            uint32_t idxCnt = cnt; // 全量 mask 供索引计算
            AscendC::Reg::MaskReg idxMask = AscendC::Reg::UpdateMask<RangeType>(idxCnt);

            for (uint16_t rep = 0; rep < reps; ++rep) {
                AscendC::Reg::Arange(idxReg, static_cast<RangeType>(rep * blkElems));
                AscendC::Reg::Muls(idxReg, idxReg, static_cast<RangeType>(-1), idxMask);
                AscendC::Reg::Adds(idxReg, idxReg, static_cast<RangeType>(curLen - 1), idxMask);
                AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<CastType>(rest);
                if constexpr (sizeof(T) == 1) {
                    AscendC::Reg::Gather((AscendC::Reg::RegTensor<CastType>&)dataReg, srcPtr,
                                         (AscendC::Reg::RegTensor<IdxType>&)idxReg, mask);
                    __ubuf__ CastType* dstCastPtr = reinterpret_cast<__ubuf__ CastType*>(dstPtr + rep * blkElems);
                    AscendC::Reg::StoreAlign<CastType, AscendC::Reg::StoreDist::DIST_PACK_B16>(dstCastPtr, dataReg,
                                                                                               mask);
                } else {
                    AscendC::Reg::Gather(dataReg, srcPtr, (AscendC::Reg::RegTensor<IdxType>&)idxReg, mask);
                    AscendC::Reg::StoreAlign(dstPtr + rep * blkElems, dataReg, mask);
                }
            }
        }
    }

    // 段序 s 混基反解为 var 段基址
    __aicore__ inline int64_t DecodeSegVarOff(int64_t s) const
    {
        int64_t off = varBase_;
        for (int64_t d = dimNum_ - 2; d >= 0; --d) {
            const int64_t coord = s / segCum_[d + 1] % sliceShape_[d];
            off += coord * axisStep_[d];
        }
        return off;
    }

    // 段升地址头：全负方向段块最低端为 DecodeSegVarOff - (contigRun - 1)
    __aicore__ inline int64_t SegVarHead(int64_t s) const
    {
        return DecodeSegVarOff(s) - (negative_ ? (contigRun_ - 1) : 0);
    }

private:
    TPipe* Ppipe_{nullptr};
    static constexpr uint32_t VF_OFF_BYTES = 256;                 // Reg 每轮处理 256B
    TQue<TPosition::VECIN, SSA_KERNEL_BUFFER_NUM> inputQueue_;    // 全负专用：反转前输入（双缓冲）
    TQue<TPosition::VECOUT, SSA_KERNEL_BUFFER_NUM> reverseQueue_; // 全负专用：反转后输出（双缓冲）
    TQueBind<TPosition::VECIN, TPosition::VECOUT, SSA_KERNEL_BUFFER_NUM> chunkBind_; // 全正方向专用
    GlobalTensor<T> inputValueGm_;
    GlobalTensor<T> varOutGm_; // in-place：varOut ≡ var
    DataCopyPadExtParams<T> padParams_{false, 0, 0, 0};

    int64_t dimNum_{0};
    int64_t blockDim_{0};
    int64_t chunksPerSeg_{0};
    int64_t chunkLen_{0};
    int64_t chunkRem_{0};
    int64_t totalChunks_{0};
    int64_t chunksPerCore_{0};
    int64_t lastCoreChunks_{0};
    int64_t varBase_{0};
    int64_t contigRun_{0};
    bool negative_{false};
    int64_t ubFactor_{0};

    int64_t sliceShape_[SSA_MAX_DIM_NUM] = {};
    int64_t axisStep_[SSA_MAX_DIM_NUM] = {};
    int64_t segCum_[SSA_MAX_DIM_NUM] = {};

    int64_t startChunk_{0};
    int64_t chunksThisCore_{0};
};

} // namespace StridedSliceAssignV2ContigNs

#endif // STRIDED_SLICE_ASSIGN_V2_CONTIG_H_
