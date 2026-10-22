/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "fused_exponential_tiling_data.h"

using namespace AscendC;
// Keep each power-of-two multiplier normal in FP32, including negative steps.
constexpr int32_t MAX_EXPONENT_STEP = 120;
constexpr int32_t FLOAT_EXPONENT_BIAS = 127;
constexpr uint32_t FLOAT_MANTISSA_BITS = 23;
template <typename T>
class FusedExponentialKernel {
public:
    __aicore__ inline void Init(GM_ADDR u, GM_ADDR out, const FusedExponentialTilingData& data)
    {
        data_ = data;
        offset_ = static_cast<uint64_t>(GetBlockIdx()) * data_.coreLength;
        const uint64_t remaining = offset_ < data_.length ? data_.length - offset_ : 0;
        length_ = remaining < data_.coreLength ? remaining : data_.coreLength;
        u_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(u));
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
        pipe_.InitBuffer(inQueue_, 1, data_.tileLength * sizeof(T));
        pipe_.InitBuffer(outQueue_, 1, data_.tileLength * sizeof(T));
        pipe_.InitBuffer(vBuffer_, data_.tileLength * sizeof(float));
        pipe_.InitBuffer(tBuffer_, data_.tileLength * sizeof(float));
        pipe_.InitBuffer(dBuffer_, data_.tileLength * sizeof(float));
        pipe_.InitBuffer(yBuffer_, data_.tileLength * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        for (uint64_t start = 0; start < length_; start += data_.tileLength) {
            const uint64_t remaining = length_ - start;
            const uint32_t n = remaining < data_.tileLength ? static_cast<uint32_t>(remaining) : data_.tileLength;
            if (!data_.zero) {
                auto input = inQueue_.AllocTensor<T>();
                DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
                DataCopyPad(input, u_[offset_ + start], copy, DataCopyPadExtParams<T>{false, 0, 0, 0});
                inQueue_.EnQue(input);
            }
            Compute(n);
            auto output = outQueue_.DeQue<T>();
            DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
            DataCopyPad(out_[offset_ + start], output, copy);
            outQueue_.FreeTensor(output);
        }
    }

private:
    __aicore__ inline void Compute(uint32_t n)
    {
        auto output = outQueue_.AllocTensor<T>();
        auto v = vBuffer_.Get<float>();
        auto t = tBuffer_.Get<float>();
        auto d = dBuffer_.Get<float>();
        auto y = yBuffer_.Get<float>();
        if (data_.zero) {
            Duplicate(y, 0.0f, n);
        } else {
            auto input = inQueue_.DeQue<T>();
            if constexpr (IsSameType<T, float>::value) {
                Adds(v, input, 0.0f, n);
            } else {
                Cast(v, input, RoundMode::CAST_NONE, n);
            }
            inQueue_.FreeTensor(input);
            Maxs(v, v, data_.epsilon, n);
            Muls(t, v, -1.0f, n);
            Adds(t, t, 1.0f, n);
            // For FP16/BF16, clamping to epsilon makes 1-v exact in FP32.
            // Only FP32 input needs compensation for the rounded subtraction.
            if constexpr (IsSameType<T, float>::value) {
                Muls(d, t, -1.0f, n);
                Adds(d, d, 1.0f, n);
                Div(d, v, d, n);
            }
            Ln(y, t, n);
            if constexpr (IsSameType<T, float>::value) {
                Mul(y, y, d, n);
            }
            Muls(y, y, data_.scale, n);
            int32_t exponent = data_.exponent;
            while (exponent != 0) {
                const int32_t step = exponent > MAX_EXPONENT_STEP ?
                                         MAX_EXPONENT_STEP :
                                         (exponent < -MAX_EXPONENT_STEP ? -MAX_EXPONENT_STEP : exponent);
                union {
                    uint32_t bits;
                    float value;
                } power{};
                power.bits = static_cast<uint32_t>(step + FLOAT_EXPONENT_BIAS) << FLOAT_MANTISSA_BITS;
                Muls(y, y, power.value, n);
                exponent -= step;
            }
        }
        if constexpr (IsSameType<T, float>::value) {
            Adds(output, y, 0.0f, n);
        } else {
            Cast(output, y, RoundMode::CAST_RINT, n);
        }
        outQueue_.EnQue(output);
    }
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TBuf<QuePosition::VECCALC> vBuffer_;
    TBuf<QuePosition::VECCALC> tBuffer_;
    TBuf<QuePosition::VECCALC> dBuffer_;
    TBuf<QuePosition::VECCALC> yBuffer_;
    GlobalTensor<T> u_;
    GlobalTensor<T> out_;
    FusedExponentialTilingData data_{};
    uint64_t offset_ = 0;
    uint64_t length_ = 0;
};

extern "C" __global__ __aicore__ void fused_exponential(GM_ADDR u, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(FusedExponentialTilingData);
    GET_TILING_DATA_WITH_STRUCT(FusedExponentialTilingData, data, tiling);
    FusedExponentialKernel<DTYPE_U> op;
    op.Init(u, out, data);
    op.Process();
}
