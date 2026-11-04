/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stack_ball_query.h
 * \brief
 */
#ifndef _SRC_STACK_BALL_QUERY_H_
#define _SRC_STACK_BALL_QUERY_H_
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t ALIGN_32 = 32;
constexpr int32_t XYZ_NUM = 3;
constexpr int32_t XYZ_GM_OFFSET = 2;

template <typename INPUT_T, typename COUNT_T>
class KernelStackBallQuery {
public:
    __aicore__ inline KernelStackBallQuery(AscendC::TPipe* p) : pipe(p){};

    __aicore__ inline void Init(GM_ADDR xyz, GM_ADDR center_xyz, GM_ADDR xyz_batch_cnt, GM_ADDR center_xyz_batch_cnt,
                                GM_ADDR idx, StackBallQueryTilingData tilingData)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero !");
        this->batchSize = tilingData.batchSize;
        this->totalLengthCenterXyz = tilingData.totalLengthCenterXyz;
        this->totalLengthXyz = tilingData.totalLengthXyz;
        this->totalIdxLength = tilingData.totalIdxLength;
        this->coreNum = tilingData.coreNum;
        this->centerXyzPerCore = tilingData.centerXyzPerCore;
        this->tailCenterXyzPerCore = tilingData.tailCenterXyzPerCore;
        float radiusSq = tilingData.maxRadius * tilingData.maxRadius;
        if constexpr (sizeof(INPUT_T) == sizeof(half)) {
            // fp16 的半径平方必须与 Golden 一致：先按 float 计算平方，再舍入到 half。
            // 直接使用未舍入的 float 阈值，会使“距离==半径”的边界点与参考产生 1 ULP 偏差。
            this->maxRadius = static_cast<float>(static_cast<half>(radiusSq));
        } else {
            this->maxRadius = radiusSq;
        }
        this->sampleNum = tilingData.sampleNum;
        this->typeXyzBlockSize = ALIGN_32 / (sizeof(INPUT_T));
        this->typeIntBlockSize = ALIGN_32 / sizeof(COUNT_T);
        this->centerXyzEachSegmentLength = this->centerXyzEachSegmentLength / BUFFER_NUM;
        this->xyzEachSegmentLength = this->xyzEachSegmentLength / BUFFER_NUM;
        this->idxEachSegmentLength = this->idxEachSegmentLength / BUFFER_NUM;

        // 本核实际处理的中心点数量：尾核为 tailCenterXyzPerCore，其余为 centerXyzPerCore。
        // 原实现按 Ceil(total, coreNum) 声明 GM 长度，会高估尾核长度，配合 32B 对齐搬运造成越界读。
        int32_t currentCoreCenterXyz = this->centerXyzPerCore;
        if (this->tailCenterXyzPerCore != 0 && GetBlockIdx() == this->coreNum - 1) {
            currentCoreCenterXyz = this->tailCenterXyzPerCore;
        }
        this->centerXyzCoreCount = static_cast<uint32_t>(currentCoreCenterXyz);
        centerXyzGm.SetGlobalBuffer((__gm__ INPUT_T*)center_xyz + 3 * this->centerXyzPerCore * GetBlockIdx(),
                                    3 * this->centerXyzCoreCount);
        xyzGm.SetGlobalBuffer((__gm__ INPUT_T*)xyz, 3 * this->totalLengthXyz);
        idxGm.SetGlobalBuffer((__gm__ int32_t*)idx, this->totalIdxLength);
        centerXyzBatchCntGm.SetGlobalBuffer((__gm__ COUNT_T*)center_xyz_batch_cnt, this->batchSize);
        xyzBatchCntGm.SetGlobalBuffer((__gm__ COUNT_T*)xyz_batch_cnt, this->batchSize);

        pipe->InitBuffer(inQueueCenterXyz, BUFFER_NUM, XYZ_NUM * this->centerXyzEachSegmentLength * sizeof(INPUT_T));
        pipe->InitBuffer(inQueueX, BUFFER_NUM, this->xyzEachSegmentLength * sizeof(INPUT_T));
        pipe->InitBuffer(inQueueY, BUFFER_NUM, this->xyzEachSegmentLength * sizeof(INPUT_T));
        pipe->InitBuffer(inQueueZ, BUFFER_NUM, this->xyzEachSegmentLength * sizeof(INPUT_T));

        pipe->InitBuffer(resultBuf, this->idxEachSegmentLength * sizeof(int32_t));
        this->resultOut = resultBuf.Get<int32_t>();
        pipe->InitBuffer(calcBufCenterX, this->xyzEachSegmentLength * sizeof(INPUT_T));
        pipe->InitBuffer(calcBufCenterY, this->xyzEachSegmentLength * sizeof(INPUT_T));
        pipe->InitBuffer(calcBufCenterZ, this->xyzEachSegmentLength * sizeof(INPUT_T));

        pipe->InitBuffer(calcBufDistanceResult, this->xyzEachSegmentLength * sizeof(INPUT_T));
        pipe->InitBuffer(calcBufCenterDistanceX, this->xyzEachSegmentLength * sizeof(INPUT_T));
        pipe->InitBuffer(calcBufCenterDistanceY, this->xyzEachSegmentLength * sizeof(INPUT_T));
        pipe->InitBuffer(calcBufCenterDistanceZ, this->xyzEachSegmentLength * sizeof(INPUT_T));
        pipe->InitBuffer(xyzBatchValue, this->GetAlignValue(this->batchSize, this->typeIntBlockSize) * sizeof(COUNT_T));
        pipe->InitBuffer(centerXyzBatchValue,
                         this->GetAlignValue(this->batchSize, this->typeIntBlockSize) * sizeof(COUNT_T));
        PipeBarrier<PIPE_ALL>();
        ;
    }

    __aicore__ inline void Process()
    {
        // 空 center_xyz（M=0）时输出张量长度为 0，无任何计算与搬运，直接返回。
        if (this->totalIdxLength == 0) {
            return;
        }
        this->CopyInBatchCnt();
        int currentCoreCenterXyz = this->centerXyzPerCore;
        if (this->tailCenterXyzPerCore != 0 && GetBlockIdx() == this->coreNum - 1) {
            currentCoreCenterXyz = this->tailCenterXyzPerCore;
        }

        int32_t centerXyzLoopCount = currentCoreCenterXyz / this->centerXyzEachSegmentLength;
        int32_t centerXyzLoopTail = currentCoreCenterXyz % this->centerXyzEachSegmentLength;

        this->offsetCenterXyzStart = this->centerXyzPerCore * GetBlockIdx();

        for (int32_t i = 0; i < centerXyzLoopCount; i++) {
            CopyInCenterXyz(i, this->centerXyzEachSegmentLength);

            this->centerXyzLocal = inQueueCenterXyz.DeQue<INPUT_T>();
            for (int j = 0; j < this->centerXyzEachSegmentLength; j++) {
                RunPerCluster(i, j);
            }
            PipeBarrier<PIPE_ALL>();
            ;
            inQueueCenterXyz.FreeTensor(this->centerXyzLocal);
        }
        PipeBarrier<PIPE_ALL>();
        ;

        if (centerXyzLoopTail != 0) {
            CopyInCenterXyz(centerXyzLoopCount, centerXyzLoopTail);

            this->centerXyzLocal = inQueueCenterXyz.DeQue<INPUT_T>();

            for (int j = 0; j < centerXyzLoopTail; j++) {
                RunPerCluster(centerXyzLoopCount, j);
            }
            PipeBarrier<PIPE_ALL>();
            ;
            inQueueCenterXyz.FreeTensor(this->centerXyzLocal);
        }
        PipeBarrier<PIPE_ALL>();
        ;
        this->SendResultToGm(true);
    }

private:
    __aicore__ inline int GetAlignValue(const uint32_t calCount, const uint32_t blockSize)
    {
        if (blockSize == 0) {
            return calCount;
        }
        uint32_t tail = calCount % blockSize;
        if (tail == 0) {
            return calCount;
        }
        uint32_t alignVal = blockSize - tail;
        return calCount + alignVal;
    }

    template <typename Type>
    __aicore__ inline void DataCopyGm2UbAlign32(const LocalTensor<Type>& dstLocal, const GlobalTensor<Type>& srcGlobal,
                                                const uint32_t calCount, const uint32_t blockSize,
                                                const uint32_t remaining)
    {
        if (g_coreType == AIC) {
            return;
        }
        uint32_t tail = calCount % blockSize;
        uint32_t alignVal = (tail == 0) ? 0 : (blockSize - tail);
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 100)
        // Ascend910A(dav_c100) 没有 DataCopyPad 的 GM->UB 重载：
        // 整块部分仍用 DataCopy；张量剩余长度不足一个 32B 对齐块时，尾段逐元素搬运，
        // 既避免对齐放大越界读，也不依赖 DataCopyPad。
        if (calCount + alignVal <= remaining) {
            DataCopy(dstLocal, srcGlobal, calCount + alignVal);
            return;
        }
        uint32_t fullCount = calCount - tail;
        if (fullCount > 0) {
            DataCopy(dstLocal, srcGlobal, fullCount);
            PipeBarrier<PIPE_ALL>();
        }
        for (uint32_t i = fullCount; i < calCount; ++i) {
            dstLocal.SetValue(i, srcGlobal.GetValue(i));
        }
#else
        if (calCount + alignVal <= remaining) {
            DataCopy(dstLocal, srcGlobal, calCount + alignVal);
        } else {
            // 张量剩余长度不足一个 32B 对齐块时，按真实长度搬运，避免对齐放大造成越界读。
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(calCount * sizeof(Type)), 0, 0, 0};
            DataCopyPadExtParams<Type> padParams{false, 0, 0, static_cast<Type>(0)};
            DataCopyPad(dstLocal, srcGlobal, copyParams, padParams);
        }
#endif
    }

    __aicore__ inline void CopyInBatchCnt()
    {
        this->centerXyzBatchLocal = centerXyzBatchValue.Get<COUNT_T>();
        DataCopyGm2UbAlign32(this->centerXyzBatchLocal, centerXyzBatchCntGm, this->batchSize, typeIntBlockSize,
                             this->batchSize);

        this->xyzBatchLocal = xyzBatchValue.Get<COUNT_T>();
        DataCopyGm2UbAlign32(this->xyzBatchLocal, xyzBatchCntGm, this->batchSize, typeIntBlockSize, this->batchSize);
    }

    __aicore__ inline void CopyInCenterXyz(int32_t segmentLoopIndex, int lenCenterXyzSegment)
    {
        int64_t offset = segmentLoopIndex * this->centerXyzEachSegmentLength * 3;
        this->centerXyzLocal = inQueueCenterXyz.AllocTensor<INPUT_T>();
        uint32_t remaining = static_cast<uint32_t>(XYZ_NUM * this->centerXyzCoreCount - offset);
        DataCopyGm2UbAlign32(centerXyzLocal, centerXyzGm[offset], XYZ_NUM * lenCenterXyzSegment, typeXyzBlockSize,
                             remaining);
        inQueueCenterXyz.EnQue(centerXyzLocal);
    }

    __aicore__ inline void CopyInXyz(int offsetXyzStart, int xyzSegmentLoopIndex, int xyzSegmentLen)
    {
        xLocal = inQueueX.AllocTensor<INPUT_T>();
        yLocal = inQueueY.AllocTensor<INPUT_T>();
        zLocal = inQueueZ.AllocTensor<INPUT_T>();

        uint32_t xyzOffsetInBatch = static_cast<uint32_t>(offsetXyzStart +
                                                          xyzSegmentLoopIndex * this->xyzEachSegmentLength);
        uint32_t remaining = static_cast<uint32_t>(this->totalLengthXyz) - xyzOffsetInBatch;

        DataCopyGm2UbAlign32(xLocal, xyzGm[xyzOffsetInBatch], xyzSegmentLen, typeXyzBlockSize, remaining);
        DataCopyGm2UbAlign32(yLocal, xyzGm[totalLengthXyz + xyzOffsetInBatch], xyzSegmentLen, typeXyzBlockSize,
                             remaining);
        DataCopyGm2UbAlign32(zLocal, xyzGm[XYZ_GM_OFFSET * totalLengthXyz + xyzOffsetInBatch], xyzSegmentLen,
                             typeXyzBlockSize, remaining);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
        inQueueZ.EnQue(zLocal);
    }

    __aicore__ inline void SendResultToGm(bool forceSend)
    {
        PipeBarrier<PIPE_ALL>();
        ;

        int tailLen = this->resultOffset % this->idxEachSegmentLength;
        if (forceSend or tailLen == 0) {
            int lenToSend = this->idxEachSegmentLength;
            if (tailLen != 0) {
                lenToSend = tailLen;
            }

            int sendTail = lenToSend % 8;
            int64_t gmOffset = this->sampleNum * this->offsetCenterXyzStart + this->resultOffset - lenToSend;
            if (sendTail == 0) {
                if (g_coreType == AIC) {
                    return;
                }
                DataCopy(idxGm[gmOffset], resultOut, lenToSend);
            } else {
                if (g_coreType == AIC) {
                    return;
                }
                // 非 32B 对齐尾段：按真实长度搬运。原实现为凑齐 32B 块而在 idx 起始处
                // 产生负偏移读写（越界），此处不再使用对齐块读写。
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 100)
                // Ascend910A(dav_c100) 没有 DataCopyPad 的 UB->GM 重载：逐元素写回。
                for (int k = 0; k < lenToSend; ++k) {
                    idxGm.SetValue(gmOffset + k, resultOut.GetValue(k));
                }
#else
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(lenToSend * sizeof(int32_t)), 0, 0, 0};
                DataCopyPad(idxGm[gmOffset], resultOut, copyParams);
#endif
            }
            PipeBarrier<PIPE_ALL>();
            ;
        }
    }

    __aicore__ inline void SetResultAndTrySend(int currentN)
    {
        this->resultOut.SetValue(this->resultOffset % this->idxEachSegmentLength, currentN);
        this->resultOffset += 1;
        SendResultToGm(false);
    }

    __aicore__ inline void ComputeBallQueryFp32(int currentNStart, int currentSegmentLen)
    {
        for (int i = 0; i < currentSegmentLen; i++) {
            if (this->resultNum >= this->sampleNum) {
                break;
            }
            auto currentDistance = this->distanceEachSegment.GetValue(i);
            if (float(currentDistance) < this->maxRadius) {
                if (this->resultNum == 0) {
                    this->firstResult = currentNStart + i;
                }
                this->resultNum += 1;

                int currentN = currentNStart + i;
                SetResultAndTrySend(currentN);
            }
        }
    }

    __aicore__ inline void CalculateDistance()
    {
        this->xLocal = inQueueX.DeQue<INPUT_T>();
        this->yLocal = inQueueY.DeQue<INPUT_T>();
        this->zLocal = inQueueZ.DeQue<INPUT_T>();

        LocalTensor<INPUT_T> centerXList = calcBufCenterX.Get<INPUT_T>(this->xyzEachSegmentLength);
        LocalTensor<INPUT_T> centerYList = calcBufCenterY.Get<INPUT_T>(this->xyzEachSegmentLength);
        LocalTensor<INPUT_T> centerZList = calcBufCenterZ.Get<INPUT_T>(this->xyzEachSegmentLength);

        LocalTensor<INPUT_T> distanceX = calcBufCenterDistanceX.Get<INPUT_T>(this->xyzEachSegmentLength);
        LocalTensor<INPUT_T> distanceY = calcBufCenterDistanceY.Get<INPUT_T>(this->xyzEachSegmentLength);
        LocalTensor<INPUT_T> distanceZ = calcBufCenterDistanceZ.Get<INPUT_T>(this->xyzEachSegmentLength);
        distanceEachSegment = calcBufDistanceResult.Get<INPUT_T>(this->xyzEachSegmentLength);

        Duplicate(centerXList, this->centerX, this->xyzEachSegmentLength);
        Duplicate(centerYList, this->centerY, this->xyzEachSegmentLength);
        Duplicate(centerZList, this->centerZ, this->xyzEachSegmentLength);

        Sub(distanceX, centerXList, xLocal, this->xyzEachSegmentLength);
        Mul(distanceX, distanceX, distanceX, this->xyzEachSegmentLength);
        Sub(distanceY, centerYList, yLocal, this->xyzEachSegmentLength);
        Mul(distanceY, distanceY, distanceY, this->xyzEachSegmentLength);
        Sub(distanceZ, centerZList, zLocal, this->xyzEachSegmentLength);
        Mul(distanceZ, distanceZ, distanceZ, this->xyzEachSegmentLength);

        Add(distanceEachSegment, distanceX, distanceY, this->xyzEachSegmentLength);
        Add(distanceEachSegment, distanceEachSegment, distanceZ, this->xyzEachSegmentLength);
    }

    __aicore__ inline void GetXyzSliceAndCalDis(int currentBIndex)
    {
        this->resultNum = 0;
        this->firstResult = 0;

        int currentN = this->xyzBatchLocal.GetValue(currentBIndex);

        int xyzSegmentLoop = currentN / this->xyzEachSegmentLength;
        int xyzSegmentTail = currentN % this->xyzEachSegmentLength;

        int offsetXyzStart = 0;
        for (int i = 0; i < currentBIndex; i++) {
            offsetXyzStart += this->xyzBatchLocal.GetValue(i);
        }

        for (int i = 0; i < xyzSegmentLoop; i++) {
            if (this->resultNum >= this->sampleNum) {
                break;
            }
            int segmentLen = this->xyzEachSegmentLength;
            int currentNStart = i * this->xyzEachSegmentLength;

            CopyInXyz(offsetXyzStart, i, segmentLen);
            PipeBarrier<PIPE_ALL>();
            ;
            this->CalculateDistance();
            ComputeBallQueryFp32(currentNStart, this->xyzEachSegmentLength);

            inQueueX.FreeTensor(this->xLocal);
            inQueueY.FreeTensor(this->yLocal);
            inQueueZ.FreeTensor(this->zLocal);
        }

        if (xyzSegmentTail != 0 && this->resultNum < this->sampleNum) {
            int segmentLen = xyzSegmentTail;
            int currentNStart = xyzSegmentLoop * this->xyzEachSegmentLength;

            CopyInXyz(offsetXyzStart, xyzSegmentLoop, segmentLen);
            PipeBarrier<PIPE_ALL>();
            ;
            this->CalculateDistance();
            ComputeBallQueryFp32(currentNStart, xyzSegmentTail);
            inQueueX.FreeTensor(this->xLocal);
            inQueueY.FreeTensor(this->yLocal);
            inQueueZ.FreeTensor(this->zLocal);
        }

        if (resultNum == 0) {
            // 无任何候选点：按参考语义整行保持 -1，剩余槽位不得填 0。
            this->firstResult = -1;
            this->resultNum += 1;
            this->SetResultAndTrySend(-1);
        }

        for (int i = resultNum; i < sampleNum; i++) {
            this->SetResultAndTrySend(this->firstResult);
            PipeBarrier<PIPE_ALL>();
            ;
        }
    }

    __aicore__ inline void RunPerCluster(int segmentLoopIndex, int clusterIndex)
    {
        this->centerX = centerXyzLocal.GetValue(XYZ_NUM * clusterIndex + 0);
        this->centerY = centerXyzLocal.GetValue(XYZ_NUM * clusterIndex + 1);
        this->centerZ = centerXyzLocal.GetValue(XYZ_NUM * clusterIndex + XYZ_GM_OFFSET);

        int currentIdx = segmentLoopIndex * this->centerXyzEachSegmentLength + clusterIndex + offsetCenterXyzStart;
        int currentBIndex = 0;
        int tmpB = 0;

        for (int i = 0; i < this->batchSize; i++) {
            tmpB += centerXyzBatchLocal.GetValue(i);
            if (tmpB > currentIdx) {
                currentBIndex = i;
                break;
            }
        }

        GetXyzSliceAndCalDis(currentBIndex);
    }

    TPipe* pipe{nullptr};
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueCenterXyz;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueZ;

    LocalTensor<COUNT_T> centerXyzBatchLocal;
    LocalTensor<COUNT_T> xyzBatchLocal;
    LocalTensor<int32_t> resultOut;
    LocalTensor<INPUT_T> distanceEachSegment;
    LocalTensor<INPUT_T> centerXyzLocal;
    LocalTensor<INPUT_T> xLocal, yLocal, zLocal;

    GlobalTensor<INPUT_T> centerXyzGm, xyzGm;
    GlobalTensor<int32_t> idxGm;
    GlobalTensor<COUNT_T> xyzBatchCntGm, centerXyzBatchCntGm;

    TBuf<TPosition::VECCALC> calcBufCenterX, calcBufCenterY, calcBufCenterZ, calcBufDistanceResult;
    TBuf<TPosition::VECCALC> calcBufCenterDistanceX, calcBufCenterDistanceY, calcBufCenterDistanceZ;
    TBuf<TPosition::VECCALC> resultBuf;
    TBuf<TPosition::VECCALC> xyzBatchValue, centerXyzBatchValue;

    INPUT_T centerX, centerY, centerZ;

    int resultNum;
    int offsetCenterXyzStart;
    int resultOffset{0};
    int firstResult;

    int32_t batchSize;
    int32_t totalLengthCenterXyz;
    int32_t totalLengthXyz;
    int32_t totalIdxLength;

    int32_t coreNum;
    int32_t centerXyzPerCore;
    int32_t tailCenterXyzPerCore;

    uint32_t centerXyzCoreCount{0};
    int centerXyzEachSegmentLength = 2048;
    int xyzEachSegmentLength = 2048;
    int idxEachSegmentLength = 2048;

    float maxRadius{};
    int32_t sampleNum{};

    int typeXyzBlockSize = 8;
    int typeIntBlockSize = 8;
};

#endif // _SRC_STACK_BALL_QUERY_H_
