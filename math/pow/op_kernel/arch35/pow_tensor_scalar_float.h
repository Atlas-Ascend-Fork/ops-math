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
 * \file pow_tensor_scalar_float.h
 * \brief
 */
#ifndef ASCENDC_POW_POWS_FLOAT_H_
#define ASCENDC_POW_POWS_FLOAT_H_

#include "vf/pow_tensor_scalar_vf.h"

namespace Pow {
using namespace AscendC;
template <typename T1, typename T2>
class PowTensorScalarFloat {
public:
    __aicore__ inline PowTensorScalarFloat(){};

    __aicore__ inline void Init(GM_ADDR base, GM_ADDR exponent, GM_ADDR pow,
                                const PowTensorScalarTilingData* __restrict ordTilingData, TPipe* pipeIn)
    {
        cBlockIdx = GetBlockIdx();
        tilingData = ordTilingData;

        // init pipe
        pipe = pipeIn;

        // init gm
        baseGm.SetGlobalBuffer((__gm__ T1*)base);
        expGm.SetGlobalBuffer((__gm__ T2*)exponent);
        powGm.SetGlobalBuffer((__gm__ T1*)pow);

        // init ub buffer
        constexpr static int64_t EXP_UB_SIZE = 256;
        constexpr static int64_t DB_BUFFER = 2;
        pipe->InitBuffer(vecBaseQue, DB_BUFFER, tilingData->ubSize);
        pipe->InitBuffer(vecExpQue, 1, EXP_UB_SIZE);
        pipe->InitBuffer(vecPowQue, 1, tilingData->ubSize);
        pipe->InitBuffer(tmp1Tensor, tilingData->ubSize);
        pipe->InitBuffer(tmp2Tensor, tilingData->ubSize);
        pipe->InitBuffer(tmpExpTensor, EXP_UB_SIZE);

        // get exp value
        if constexpr (IsSameType<T2, bfloat16_t>::value) {
            LocalTensor<T2> expLocal = vecExpQue.AllocTensor<T2>();
            DataCopyPad(expLocal, expGm, {static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(T2)), 0, 0, 0},
                        {false, 0, 0, 0});
            vecExpQue.EnQue(expLocal);
            vecExpQue.DeQue<T2>();
            LocalTensor<float> expCastTensor = tmpExpTensor.Get<float>();
            Cast(expCastTensor, expLocal, RoundMode::CAST_NONE, 1);
            vecExpQue.FreeTensor(expLocal);
            // add sync scalar wait for vector
            TEventID eventID_V_S = GetTPipePtr()->FetchEventID(HardEvent::V_S);
            SetFlag<HardEvent::V_S>(eventID_V_S);
            WaitFlag<HardEvent::V_S>(eventID_V_S);
            expValue = expCastTensor(0);
        } else {
            expValue = static_cast<float>(expGm(0));
        }

        // add sync v wait for s
        TEventID eventID_S_V = GetTPipePtr()->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(eventID_S_V);
        WaitFlag<HardEvent::S_V>(eventID_S_V);
    }

    __aicore__ inline void Process()
    {
        int64_t beginIdx = cBlockIdx * tilingData->blockFactor;
        int64_t endIdx = (cBlockIdx + 1) * tilingData->blockFactor;
        endIdx = endIdx > tilingData->ubOuter ? tilingData->ubOuter : endIdx;
        int64_t realEndIdx = tilingData->ubOuter - 1;
        int64_t curDataLength = tilingData->ubFactor;
        int64_t gmOffset = beginIdx * curDataLength;

        if (expValue == 0.0f) {
            LocalTensor<T1> dstLocal = vecPowQue.AllocTensor<T1>();
            ComputeZeros(dstLocal, tilingData->ubFactor);
            vecPowQue.EnQue(dstLocal);
            vecPowQue.DeQue<T1>();
            for (int64_t idx = beginIdx; idx < endIdx; idx++) {
                if (unlikely(idx == realEndIdx)) {
                    curDataLength = tilingData->ubTail;
                }
                CopyPow(dstLocal, gmOffset, curDataLength);
                gmOffset = gmOffset + tilingData->ubFactor;
            }
            vecPowQue.FreeTensor(dstLocal);
        } else if (expValue == 1.0f) {
            for (int64_t idx = beginIdx; idx < endIdx; idx++) {
                if (unlikely(idx == realEndIdx)) {
                    curDataLength = tilingData->ubTail;
                }
                LocalTensor<T1> srcLocal = vecBaseQue.AllocTensor<T1>();
                CopyBase(srcLocal, gmOffset, curDataLength);
                vecBaseQue.EnQue(srcLocal);
                vecBaseQue.DeQue<T1>();
                LocalTensor<T1> dstLocal = vecPowQue.AllocTensor<T1>();
                ComputeCopy(dstLocal, srcLocal, curDataLength);
                vecBaseQue.FreeTensor(srcLocal);
                vecPowQue.EnQue(dstLocal);
                vecPowQue.DeQue<T1>();
                CopyPow(dstLocal, gmOffset, curDataLength);
                vecPowQue.FreeTensor(dstLocal);
                gmOffset = gmOffset + tilingData->ubFactor;
            }
        } else if (expValue == 0.5f) {
            for (int64_t idx = beginIdx; idx < endIdx; idx++) {
                if (unlikely(idx == realEndIdx)) {
                    curDataLength = tilingData->ubTail;
                }
                LocalTensor<T1> srcLocal = vecBaseQue.AllocTensor<T1>();
                CopyBase(srcLocal, gmOffset, curDataLength);
                vecBaseQue.EnQue(srcLocal);
                vecBaseQue.DeQue<T1>();
                LocalTensor<T1> dstLocal = vecPowQue.AllocTensor<T1>();
                __ubuf__ T1* srcAddr = (__ubuf__ T1*)srcLocal.GetPhyAddr();
                __ubuf__ T1* dstAddr = (__ubuf__ T1*)dstLocal.GetPhyAddr();
                asc_vf_call<PowScalarSqrtVF<T1>>(dstAddr, srcAddr, curDataLength);
                vecBaseQue.FreeTensor(srcLocal);
                vecPowQue.EnQue(dstLocal);
                vecPowQue.DeQue<T1>();
                CopyPow(dstLocal, gmOffset, curDataLength);
                vecPowQue.FreeTensor(dstLocal);
                gmOffset = gmOffset + tilingData->ubFactor;
            }
        } else if (expValue == -0.5f) {
            for (int64_t idx = beginIdx; idx < endIdx; idx++) {
                if (unlikely(idx == realEndIdx)) {
                    curDataLength = tilingData->ubTail;
                }
                LocalTensor<T1> srcLocal = vecBaseQue.AllocTensor<T1>();
                CopyBase(srcLocal, gmOffset, curDataLength);
                vecBaseQue.EnQue(srcLocal);
                vecBaseQue.DeQue<T1>();
                LocalTensor<T1> dstLocal = vecPowQue.AllocTensor<T1>();
                __ubuf__ T1* srcAddr = (__ubuf__ T1*)srcLocal.GetPhyAddr();
                __ubuf__ T1* dstAddr = (__ubuf__ T1*)dstLocal.GetPhyAddr();
                asc_vf_call<PowScalarRsqrtVF<T1>>(dstAddr, srcAddr, curDataLength);
                vecBaseQue.FreeTensor(srcLocal);
                vecPowQue.EnQue(dstLocal);
                vecPowQue.DeQue<T1>();
                CopyPow(dstLocal, gmOffset, curDataLength);
                vecPowQue.FreeTensor(dstLocal);
                gmOffset = gmOffset + tilingData->ubFactor;
            }
        } else if (expValue == 2.0f) {
            for (int64_t idx = beginIdx; idx < endIdx; idx++) {
                if (unlikely(idx == realEndIdx)) {
                    curDataLength = tilingData->ubTail;
                }
                LocalTensor<T1> srcLocal = vecBaseQue.AllocTensor<T1>();
                CopyBase(srcLocal, gmOffset, curDataLength);
                vecBaseQue.EnQue(srcLocal);
                vecBaseQue.DeQue<T1>();
                LocalTensor<T1> dstLocal = vecPowQue.AllocTensor<T1>();
                __ubuf__ T1* srcAddr = (__ubuf__ T1*)srcLocal.GetPhyAddr();
                __ubuf__ T1* dstAddr = (__ubuf__ T1*)dstLocal.GetPhyAddr();
                asc_vf_call<PowScalarSquareVF<T1>>(dstAddr, srcAddr, curDataLength);
                vecBaseQue.FreeTensor(srcLocal);
                vecPowQue.EnQue(dstLocal);
                vecPowQue.DeQue<T1>();
                CopyPow(dstLocal, gmOffset, curDataLength);
                vecPowQue.FreeTensor(dstLocal);
                gmOffset = gmOffset + tilingData->ubFactor;
            }
        } else if (expValue == 3.0f) {
            for (int64_t idx = beginIdx; idx < endIdx; idx++) {
                if (unlikely(idx == realEndIdx)) {
                    curDataLength = tilingData->ubTail;
                }
                LocalTensor<T1> srcLocal = vecBaseQue.AllocTensor<T1>();
                CopyBase(srcLocal, gmOffset, curDataLength);
                vecBaseQue.EnQue(srcLocal);
                vecBaseQue.DeQue<T1>();
                LocalTensor<T1> dstLocal = vecPowQue.AllocTensor<T1>();
                __ubuf__ T1* srcAddr = (__ubuf__ T1*)srcLocal.GetPhyAddr();
                __ubuf__ T1* dstAddr = (__ubuf__ T1*)dstLocal.GetPhyAddr();
                asc_vf_call<PowScalarCubeVF<T1>>(dstAddr, srcAddr, curDataLength);
                vecBaseQue.FreeTensor(srcLocal);
                vecPowQue.EnQue(dstLocal);
                vecPowQue.DeQue<T1>();
                CopyPow(dstLocal, gmOffset, curDataLength);
                vecPowQue.FreeTensor(dstLocal);
                gmOffset = gmOffset + tilingData->ubFactor;
            }
        } else if (expValue == -1.0f) {
            for (int64_t idx = beginIdx; idx < endIdx; idx++) {
                if (unlikely(idx == realEndIdx)) {
                    curDataLength = tilingData->ubTail;
                }
                LocalTensor<T1> srcLocal = vecBaseQue.AllocTensor<T1>();
                CopyBase(srcLocal, gmOffset, curDataLength);
                vecBaseQue.EnQue(srcLocal);
                vecBaseQue.DeQue<T1>();
                LocalTensor<T1> dstLocal = vecPowQue.AllocTensor<T1>();
                __ubuf__ T1* srcAddr = (__ubuf__ T1*)srcLocal.GetPhyAddr();
                __ubuf__ T1* dstAddr = (__ubuf__ T1*)dstLocal.GetPhyAddr();
                asc_vf_call<PowScalarRecVF<T1>>(dstAddr, srcAddr, curDataLength);
                vecBaseQue.FreeTensor(srcLocal);
                vecPowQue.EnQue(dstLocal);
                vecPowQue.DeQue<T1>();
                CopyPow(dstLocal, gmOffset, curDataLength);
                vecPowQue.FreeTensor(dstLocal);
                gmOffset = gmOffset + tilingData->ubFactor;
            }
        } else if (expValue == -2.0f) {
            for (int64_t idx = beginIdx; idx < endIdx; idx++) {
                if (unlikely(idx == realEndIdx)) {
                    curDataLength = tilingData->ubTail;
                }
                LocalTensor<T1> srcLocal = vecBaseQue.AllocTensor<T1>();
                CopyBase(srcLocal, gmOffset, curDataLength);
                vecBaseQue.EnQue(srcLocal);
                vecBaseQue.DeQue<T1>();
                LocalTensor<T1> dstLocal = vecPowQue.AllocTensor<T1>();
                __ubuf__ T1* srcAddr = (__ubuf__ T1*)srcLocal.GetPhyAddr();
                __ubuf__ T1* dstAddr = (__ubuf__ T1*)dstLocal.GetPhyAddr();
                asc_vf_call<PowScalarSquareRecVF<T1>>(dstAddr, srcAddr, curDataLength);
                vecBaseQue.FreeTensor(srcLocal);
                vecPowQue.EnQue(dstLocal);
                vecPowQue.DeQue<T1>();
                CopyPow(dstLocal, gmOffset, curDataLength);
                vecPowQue.FreeTensor(dstLocal);
                gmOffset = gmOffset + tilingData->ubFactor;
            }
        } else {
            for (int64_t idx = beginIdx; idx < endIdx; idx++) {
                if (unlikely(idx == realEndIdx)) {
                    curDataLength = tilingData->ubTail;
                }

                if constexpr (sizeof(T1) == sizeof(float)) {
                    LocalTensor<T1> srcLocal = vecBaseQue.AllocTensor<T1>();
                    CopyBase(srcLocal, gmOffset, curDataLength);
                    vecBaseQue.EnQue(srcLocal);
                    vecBaseQue.DeQue<T1>();
                    LocalTensor<T1> dstLocal = vecPowQue.AllocTensor<T1>();
                    Power<T1, false, powsConfig_>(dstLocal, srcLocal, expValue, static_cast<uint32_t>(curDataLength));
                    vecBaseQue.FreeTensor(srcLocal);
                    vecPowQue.EnQue(dstLocal);
                    vecPowQue.DeQue<T1>();
                    CopyPow(dstLocal, gmOffset, curDataLength);
                    vecPowQue.FreeTensor(dstLocal);
                } else {
                    LocalTensor<T1> srcLocal = vecBaseQue.AllocTensor<T1>();
                    CopyBase(srcLocal, gmOffset, curDataLength);
                    vecBaseQue.EnQue(srcLocal);
                    vecBaseQue.DeQue<T1>();
                    LocalTensor<float> castTensor = tmp1Tensor.Get<float>();
                    Cast(castTensor, srcLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(curDataLength));
                    vecBaseQue.FreeTensor(srcLocal);
                    LocalTensor<float> powerTensor = tmp2Tensor.Get<float>();
                    Power<float, false, powsConfig_>(powerTensor, castTensor, expValue,
                                                     static_cast<uint32_t>(curDataLength));
                    LocalTensor<T1> dstLocal = vecPowQue.AllocTensor<T1>();
                    RoundMode b16RoundMode = IsSameType<T1, bfloat16_t>::value ? RoundMode::CAST_RINT :
                                                                                 RoundMode::CAST_NONE;
                    Cast(dstLocal, powerTensor, b16RoundMode, static_cast<uint32_t>(curDataLength));
                    vecPowQue.EnQue(dstLocal);
                    vecPowQue.DeQue<T1>();
                    CopyPow(dstLocal, gmOffset, curDataLength);
                    vecPowQue.FreeTensor(dstLocal);
                }
                gmOffset = gmOffset + tilingData->ubFactor;
            }
        }
    }

private:
    __aicore__ inline void CopyBase(LocalTensor<T1>& dstTensor, int64_t offset, int64_t gatherLength)
    {
        DataCopyPad(dstTensor, baseGm[offset],
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(gatherLength * sizeof(T1)), 0, 0, 0},
                    {false, 0, 0, 0});
    }

    __aicore__ inline void CopyPow(LocalTensor<T1>& srcTensor, int64_t offset, int64_t gatherLength)
    {
        DataCopyPad(powGm[offset], srcTensor,
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(gatherLength * sizeof(T1)), 0, 0, 0});
    }

    __aicore__ inline void ComputeZeros(LocalTensor<T1>& dstTensor, int64_t dataLength)
    {
        T1 inputVal(1.0);
        Duplicate<T1>(dstTensor, inputVal, static_cast<int32_t>(dataLength));
    }

    __aicore__ inline void ComputeCopy(LocalTensor<T1>& dstTensor, LocalTensor<T1>& srcTensor, int64_t dataLength)
    {
        if constexpr (sizeof(T1) == sizeof(float)) {
            uint32_t scalarValue = 0;
            ShiftLeft(dstTensor.template ReinterpretCast<uint32_t>(), srcTensor.template ReinterpretCast<uint32_t>(),
                      scalarValue, static_cast<int32_t>(dataLength));
        } else {
            uint16_t scalarValue = 0;
            ShiftLeft(dstTensor.template ReinterpretCast<uint16_t>(), srcTensor.template ReinterpretCast<uint16_t>(),
                      scalarValue, static_cast<int32_t>(dataLength));
        }
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> vecBaseQue;
    TQue<QuePosition::VECIN, 1> vecExpQue;
    TQue<QuePosition::VECOUT, 1> vecPowQue;
    TBuf<> tmp1Tensor;
    TBuf<> tmp2Tensor;
    TBuf<> tmpExpTensor;

    int64_t cBlockIdx;
    const PowTensorScalarTilingData* __restrict tilingData;

    // input
    GlobalTensor<T1> baseGm;
    GlobalTensor<T2> expGm;

    // output
    GlobalTensor<T1> powGm;

    // exp value
    float expValue;

    // cast info
    constexpr static PowerConfig powsConfig_ = {PowerAlgo::DOUBLE_FLOAT_TECH};
};
} // namespace Pow
#endif // ASCENDC_POW_POWS_FLOAT_H_
