/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSPOSE_TRANSDATA_5HD_ARCH31_H
#define TRANSPOSE_TRANSDATA_5HD_ARCH31_H

#include "transpose_base_arch31.h"

namespace Transpose {
using namespace AscendC;
constexpr uint32_t TRANS_DATA_ELEMS = 16;
constexpr uint32_t TRANS_DATA_BUFFER_COUNT = 2;

template <typename T>
class KirinTransposeTransDataTo5HD {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const KirinTransposeVConvOpTilingData* tiling, TPipe* pipe)
    {
        tiling_ = tiling;
        blockIdx_ = GetBlockIdx();
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        fullCoreNum_ = tiling_->usedCoreNum - 1;
        loopCount_ = blockIdx_ < fullCoreNum_ ? tiling_->mainCoreLoopCount : tiling_->tailCoreLoopCount;
        pipe->InitBuffer(inputQueue_, TRANS_DATA_BUFFER_COUNT, static_cast<uint32_t>(tiling_->availableUbSize));
        pipe->InitBuffer(outputQueue_, TRANS_DATA_BUFFER_COUNT, static_cast<uint32_t>(tiling_->availableUbSize));
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= tiling_->usedCoreNum) {
            return;
        }
        for (uint64_t loop = 0; loop < loopCount_; ++loop) {
            SetOffsets(loop);
            const bool lastLoop = loop + 1 == loopCount_;
            uint64_t r = 0;
            uint64_t c = 0;
            uint64_t rAlign = 0;
            uint64_t cAlign = 0;
            GetTileShape(lastLoop, r, c, rAlign, cAlign);
            CopyIn(loop);
            Compute(r, c, rAlign, cAlign);
            CopyOut(loop);
        }
    }

private:
    __aicore__ inline bool IsMainCore() const { return blockIdx_ < fullCoreNum_; }

    __aicore__ inline void GetTileShape(bool lastLoop, uint64_t& r, uint64_t& c, uint64_t& rAlign,
                                        uint64_t& cAlign) const
    {
        if (IsMainCore()) {
            r = lastLoop ? tiling_->rUbSplitPara.mainCoreTailUbFactor : tiling_->rUbSplitPara.mainCoreUbFactor;
            c = lastLoop ? tiling_->cUbSplitPara.mainCoreTailUbFactor : tiling_->cUbSplitPara.mainCoreUbFactor;
            rAlign = lastLoop ? tiling_->rUbSplitPara.mainCoreTailUbAlignFactor :
                                tiling_->rUbSplitPara.mainCoreUbAlignFactor;
            cAlign = lastLoop ? tiling_->cUbSplitPara.mainCoreTailUbAlignFactor :
                                tiling_->cUbSplitPara.mainCoreUbAlignFactor;
        } else {
            r = lastLoop ? tiling_->rUbSplitPara.tailCoreTailUbFactor : tiling_->rUbSplitPara.tailCoreUbFactor;
            c = lastLoop ? tiling_->cUbSplitPara.tailCoreTailUbFactor : tiling_->cUbSplitPara.tailCoreUbFactor;
            rAlign = lastLoop ? tiling_->rUbSplitPara.tailCoreTailUbAlignFactor :
                                tiling_->rUbSplitPara.tailCoreUbAlignFactor;
            cAlign = lastLoop ? tiling_->cUbSplitPara.tailCoreTailUbAlignFactor :
                                tiling_->cUbSplitPara.tailCoreUbAlignFactor;
        }
    }

    __aicore__ inline void SetOffsets(uint64_t loop)
    {
        if (tiling_->isRSplit) {
            copyInCoreOffset_ = blockIdx_ * tiling_->cLen * tiling_->rSplitPara.blockFactor;
            if (tiling_->isRCSplit) {
                gmOffset_ = loop * tiling_->cUbSplitPara.mainCoreUbFactor;
                copyOutCoreOffset_ = blockIdx_ * tiling_->rUbSplitPara.mainCoreUbFactor;
                ubOffset_ = loop * tiling_->rLen * tiling_->cUbSplitPara.mainCoreUbFactor;
            } else {
                gmOffset_ = loop * tiling_->rUbSplitPara.mainCoreUbFactor * tiling_->cLen;
                copyOutCoreOffset_ = blockIdx_ * tiling_->rSplitPara.blockFactor;
                ubOffset_ = loop * tiling_->rUbSplitPara.mainCoreUbFactor;
            }
        } else {
            copyInCoreOffset_ = blockIdx_ * tiling_->cSplitPara.blockFactor;
            gmOffset_ = loop * tiling_->cUbSplitPara.mainCoreUbFactor;
            copyOutCoreOffset_ = blockIdx_ * tiling_->rLen * tiling_->cSplitPara.blockFactor;
            ubOffset_ = loop * tiling_->rLen * tiling_->cUbSplitPara.mainCoreUbFactor;
        }
    }

    __aicore__ inline void CopyIn(uint64_t loop)
    {
        LocalTensor<T> local = inputQueue_.AllocTensor<T>();
        SetCopyInParams(loop);
        DataCopyPad(local, input_[copyInCoreOffset_ + gmOffset_], copyParams_, padParams_);
        inputQueue_.EnQue(local);
    }

    __aicore__ inline void Compute(uint64_t r, uint64_t c, uint64_t rAlign, uint64_t cAlign)
    {
        LocalTensor<T> source = inputQueue_.DeQue<T>();
        LocalTensor<T> destination = outputQueue_.AllocTensor<T>();
        TransDataTo5HDParams params;
        params.dstHighHalf = false;
        params.srcHighHalf = false;

        if (r >= c) {
            params.repeatTimes = static_cast<uint8_t>(r / TRANS_DATA_ELEMS);
            params.dstRepStride = static_cast<uint16_t>(params.repeatTimes == 1 ? 0 : 1);
            params.srcRepStride = static_cast<uint16_t>(params.repeatTimes == 1 ? 0 : cAlign * TRANS_DATA_ELEMS);
            for (uint64_t block = 0; block < cAlign; ++block) {
                uint64_t sourceList[TRANS_DATA_ELEMS];
                uint64_t destinationList[TRANS_DATA_ELEMS];
                for (uint32_t index = 0; index < TRANS_DATA_ELEMS; ++index) {
                    sourceList[index] = source[index * c + block * TRANS_DATA_ELEMS].GetPhyAddr();
                    destinationList[index] = destination[index * r + block * TRANS_DATA_ELEMS * r].GetPhyAddr();
                }
                TransDataTo5HD<T>(destinationList, sourceList, params);
            }
        } else {
            params.repeatTimes = static_cast<uint8_t>(c / TRANS_DATA_ELEMS);
            params.dstRepStride = static_cast<uint16_t>(params.repeatTimes == 1 ? 0 : rAlign * TRANS_DATA_ELEMS);
            params.srcRepStride = static_cast<uint16_t>(params.repeatTimes == 1 ? 0 : 1);
            for (uint64_t block = 0; block < rAlign; ++block) {
                uint64_t sourceList[TRANS_DATA_ELEMS];
                uint64_t destinationList[TRANS_DATA_ELEMS];
                for (uint32_t index = 0; index < TRANS_DATA_ELEMS; ++index) {
                    sourceList[index] = source[index * c + block * TRANS_DATA_ELEMS * c].GetPhyAddr();
                    destinationList[index] = destination[index * r + block * TRANS_DATA_ELEMS].GetPhyAddr();
                }
                TransDataTo5HD<T>(destinationList, sourceList, params);
            }
        }
        outputQueue_.EnQue(destination);
        inputQueue_.FreeTensor(source);
    }

    __aicore__ inline void CopyOut(uint64_t loop)
    {
        LocalTensor<T> local = outputQueue_.DeQue<T>();
        SetCopyOutParams(loop);
        DataCopyPad(output_[copyOutCoreOffset_ + ubOffset_], local, copyParams_);
        outputQueue_.FreeTensor(local);
    }

    __aicore__ inline void SetCopyInParams(uint64_t loop)
    {
        if (tiling_->isRCSplit) {
            SetRCSplitCopyInParams(loop);
        } else if (tiling_->isRSplit) {
            SetRSplitCopyInParams(loop);
        } else {
            SetCSplitCopyInParams(loop);
        }
    }

    __aicore__ inline void SetRCSplitCopyInParams(uint64_t loop)
    {
        if (IsMainCore()) {
            copyParams_.blockCount = loop + 1 < loopCount_ ? tiling_->rUbSplitPara.mainCoreUbFactor :
                                                             tiling_->rUbSplitPara.mainCoreTailUbFactor;
            if (loop + 1 < loopCount_) {
                copyParams_.blockLen = tiling_->cUbSplitPara.mainCoreUbFactor * sizeof(T);
                copyParams_.srcStride = (tiling_->cLen - tiling_->cUbSplitPara.mainCoreUbFactor) * sizeof(T);
            } else {
                copyParams_.blockLen = (tiling_->cLen - loop * tiling_->cUbSplitPara.mainCoreUbFactor) * sizeof(T);
                copyParams_.srcStride = loop * tiling_->cUbSplitPara.mainCoreUbFactor * sizeof(T);
            }
        } else {
            copyParams_.blockCount = tiling_->rLen - blockIdx_ * tiling_->rSplitPara.blockFactor;
            if (loop + 1 < loopCount_) {
                copyParams_.blockLen = tiling_->cUbSplitPara.tailCoreUbFactor * sizeof(T);
                copyParams_.srcStride = (tiling_->cLen - tiling_->cUbSplitPara.tailCoreUbFactor) * sizeof(T);
            } else {
                copyParams_.blockLen = (tiling_->cLen - loop * tiling_->cUbSplitPara.tailCoreUbFactor) * sizeof(T);
                copyParams_.srcStride = loop * tiling_->cUbSplitPara.tailCoreUbFactor * sizeof(T);
            }
        }
        copyParams_.dstStride = 0;
    }

    __aicore__ inline void SetRSplitCopyInParams(uint64_t loop)
    {
        if (tiling_->cLen % TRANS_DATA_ELEMS == 0) {
            copyParams_.blockCount = 1;
            copyParams_.srcStride = 0;
            copyParams_.dstStride = 0;
            if (IsMainCore()) {
                copyParams_.blockLen = (loop + 1 < loopCount_ ? tiling_->rUbSplitPara.mainCoreUbFactor *
                                                                    tiling_->cUbSplitPara.mainCoreUbFactor :
                                                                tiling_->rUbSplitPara.mainCoreTailUbFactor *
                                                                    tiling_->cUbSplitPara.mainCoreTailUbFactor) *
                                       sizeof(T);
            } else if (loop + 1 < loopCount_) {
                copyParams_.blockLen = tiling_->rUbSplitPara.tailCoreUbFactor * tiling_->cUbSplitPara.tailCoreUbFactor *
                                       sizeof(T);
            } else {
                copyParams_.blockLen = (tiling_->rLen - (blockIdx_ * tiling_->rSplitPara.blockFactor +
                                                         loop * tiling_->rUbSplitPara.tailCoreUbFactor)) *
                                       tiling_->cUbSplitPara.tailCoreTailUbFactor * sizeof(T);
            }
            return;
        }

        if (IsMainCore()) {
            copyParams_.blockCount = loop + 1 < loopCount_ ? tiling_->rUbSplitPara.mainCoreUbFactor :
                                                             tiling_->rUbSplitPara.mainCoreTailUbFactor;
        } else if (loop + 1 < loopCount_) {
            copyParams_.blockCount = tiling_->rUbSplitPara.tailCoreUbFactor;
        } else {
            copyParams_.blockCount = tiling_->rLen - (blockIdx_ * tiling_->rSplitPara.blockFactor +
                                                      loop * tiling_->rUbSplitPara.tailCoreUbFactor);
        }
        copyParams_.blockLen = tiling_->cLen * sizeof(T);
        copyParams_.srcStride = 0;
        copyParams_.dstStride = 0;
    }

    __aicore__ inline void SetCSplitCopyInParams(uint64_t loop)
    {
        copyParams_.blockCount = tiling_->rLen;
        uint64_t cFactor = 0;
        if (IsMainCore()) {
            cFactor = loop + 1 < loopCount_ ? tiling_->cUbSplitPara.mainCoreUbFactor :
                                              tiling_->cUbSplitPara.mainCoreTailUbFactor;
        } else if (loop + 1 < loopCount_) {
            cFactor = tiling_->cUbSplitPara.tailCoreUbFactor;
        } else {
            cFactor = tiling_->cLen - blockIdx_ * tiling_->cSplitPara.blockFactor -
                      loop * tiling_->cUbSplitPara.tailCoreUbFactor;
        }
        copyParams_.blockLen = cFactor * sizeof(T);
        copyParams_.srcStride = (tiling_->cLen - cFactor) * sizeof(T);
        copyParams_.dstStride = 0;
    }

    __aicore__ inline void SetCopyOutParams(uint64_t loop)
    {
        if (tiling_->isRCSplit) {
            SetRCSplitCopyOutParams(loop);
        } else if (tiling_->isRSplit) {
            SetRSplitCopyOutParams(loop);
        } else {
            SetCSplitCopyOutParams(loop);
        }
    }

    __aicore__ inline void SetRCSplitCopyOutParams(uint64_t loop)
    {
        uint64_t rFactor = 0;
        if (IsMainCore()) {
            copyParams_.blockCount = loop + 1 < loopCount_ ?
                                         tiling_->cUbSplitPara.mainCoreUbFactor :
                                         tiling_->cLen - loop * tiling_->cUbSplitPara.mainCoreUbFactor;
            rFactor = loop + 1 < loopCount_ ? tiling_->rUbSplitPara.mainCoreUbFactor :
                                              tiling_->rUbSplitPara.mainCoreTailUbFactor;
        } else {
            copyParams_.blockCount = loop + 1 < loopCount_ ?
                                         tiling_->cUbSplitPara.tailCoreUbFactor :
                                         tiling_->cLen - loop * tiling_->cUbSplitPara.tailCoreUbFactor;
            rFactor = tiling_->rLen - blockIdx_ * tiling_->rSplitPara.blockFactor;
        }
        copyParams_.blockLen = rFactor * sizeof(T);
        copyParams_.srcStride = 0;
        copyParams_.dstStride = (tiling_->rLen - rFactor) * sizeof(T);
    }

    __aicore__ inline void SetRSplitCopyOutParams(uint64_t loop)
    {
        copyParams_.blockCount = tiling_->cLen;
        uint64_t rFactor = 0;
        if (IsMainCore()) {
            rFactor = loop + 1 < loopCount_ ? tiling_->rUbSplitPara.mainCoreUbFactor :
                                              tiling_->rUbSplitPara.mainCoreTailUbFactor;
        } else if (loop + 1 < loopCount_) {
            rFactor = tiling_->rUbSplitPara.tailCoreUbFactor;
        } else {
            rFactor = tiling_->rLen - blockIdx_ * tiling_->rSplitPara.blockFactor -
                      loop * tiling_->rUbSplitPara.tailCoreUbFactor;
        }
        copyParams_.blockLen = rFactor * sizeof(T);
        copyParams_.srcStride = 0;
        copyParams_.dstStride = (tiling_->rLen - rFactor) * sizeof(T);
    }

    __aicore__ inline void SetCSplitCopyOutParams(uint64_t loop)
    {
        uint64_t rFactor = 0;
        uint64_t cFactor = 0;
        if (IsMainCore()) {
            rFactor = loop + 1 < loopCount_ ? tiling_->rUbSplitPara.mainCoreUbFactor :
                                              tiling_->rUbSplitPara.mainCoreTailUbFactor;
            cFactor = loop + 1 < loopCount_ ? tiling_->cUbSplitPara.mainCoreUbFactor :
                                              tiling_->cUbSplitPara.mainCoreTailUbFactor;
        } else if (loop + 1 < loopCount_) {
            rFactor = tiling_->rUbSplitPara.tailCoreUbFactor;
            cFactor = tiling_->cUbSplitPara.tailCoreUbFactor;
        } else {
            rFactor = tiling_->rUbSplitPara.tailCoreTailUbFactor;
            cFactor = tiling_->cLen - blockIdx_ * tiling_->cSplitPara.blockFactor -
                      loop * tiling_->cUbSplitPara.tailCoreUbFactor;
        }
        if (tiling_->rLen % TRANS_DATA_ELEMS == 0) {
            copyParams_.blockCount = 1;
            copyParams_.blockLen = rFactor * cFactor * sizeof(T);
        } else {
            copyParams_.blockCount = cFactor;
            copyParams_.blockLen = tiling_->rLen * sizeof(T);
        }
        copyParams_.srcStride = 0;
        copyParams_.dstStride = 0;
    }

private:
    const KirinTransposeVConvOpTilingData* tiling_ = nullptr;
    TQue<TPosition::VECIN, 1> inputQueue_;
    TQue<TPosition::VECOUT, 1> outputQueue_;
    GlobalTensor<T> input_;
    GlobalTensor<T> output_;
    uint64_t fullCoreNum_ = 0;
    uint64_t blockIdx_ = 0;
    uint64_t loopCount_ = 0;
    uint64_t copyInCoreOffset_ = 0;
    uint64_t copyOutCoreOffset_ = 0;
    uint64_t gmOffset_ = 0;
    uint64_t ubOffset_ = 0;
    DataCopyPadExtParams<T> padParams_{false, 0, 0, 0};
    DataCopyExtParams copyParams_{};
};
} // namespace Transpose

#endif // TRANSPOSE_TRANSDATA_5HD_ARCH31_H
