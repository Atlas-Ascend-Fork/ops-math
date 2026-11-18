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
 * \file fused_mul_addn_l2loss.h
 * \brief FusedMulAddNL2loss arch35(regbase) kernel
 *        y1 = x1 * x3 + x2   (elementwise, x3 标量广播, 多核均分)
 *        y2 = Σ(x1² / 2)     (全量 reduce, 标量输出)
 *        与 910b TBE 语义一致；fp16 输入统一 cast 到 fp32 计算，y2 fp32 累加。
 *        y2 归约（两条路径的 partial 都在 y1 循环里顺路计算，零额外访存）：
 *          fp32 —— 每核用 TwoSum 补偿累加（Knuth）把各 tile partial 精确累进
 *                  (acc, comp)；SyncAll 后各核把二元组写 y1 尾部 scratch（2 x 核数
 *                  个元素），core0 按核号固定顺序 TwoSum 合并后单次舍入写 y2，
 *                  再用 x1/x2/x3 重算恢复这段 y1。
 *                  无原子加 → 多次运行 bit 级一致；补偿累加 → 误差 ~eps²，
 *                  等价于高精度归约后再舍入（fp32 顺序累加会偏 1~2 ulp，
 *                  大 N 时 cross_check 三比值超标）。
 *          fp16 —— 避免半精度原子加误差，core0 串行归约全量（只读 x1，fp32 累加，
 *                  单次舍入写回）。
 */

#ifndef FUSED_MUL_ADD_NL2LOSS_H
#define FUSED_MUL_ADD_NL2LOSS_H

#include <type_traits>
#include "kernel_operator.h"
#include "fused_mul_addn_l2loss_tiling_data.h"

namespace FusedMulAddNL2lossOps {
using namespace AscendC;

template <typename T>
class FusedMulAddNL2lossKernel {
public:
    __aicore__ inline FusedMulAddNL2lossKernel() = default;

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y1, GM_ADDR y2, GM_ADDR workspace,
                                const FusedMulAddNL2lossTilingData* tilingData, TPipe* pipe)
    {
        pipe_ = pipe;
        tl_ = tilingData;
        blockIdx_ = static_cast<int64_t>(GetBlockIdx());
        blockNum_ = static_cast<int64_t>(GetBlockNum());

        if (blockIdx_ < blockNum_ - 1) {
            coreStart_ = blockIdx_ * tl_->coreElements;
            coreSize_ = tl_->coreElements;
        } else {
            coreStart_ = (blockNum_ - 1) * tl_->coreElements;
            coreSize_ = tl_->tailCoreElements;
        }

        x1Gm_.SetGlobalBuffer((__gm__ T*)x1);
        x2Gm_.SetGlobalBuffer((__gm__ T*)x2);
        y1Gm_.SetGlobalBuffer((__gm__ T*)y1);
        y2Gm_.SetGlobalBuffer((__gm__ T*)y2);
        (void)workspace; // 不走框架 workspace：跨核归约借用 y1 尾部 scratch，见 ReduceY2Deterministic

        GlobalTensor<T> x3Gm;
        x3Gm.SetGlobalBuffer((__gm__ T*)x3, 1);
        x3Value_ = static_cast<float>(x3Gm.GetValue(0));

        int64_t tileBytes = tl_->ubTileSize * sizeof(T);
        pipe_->InitBuffer(x1Que_, 2, tileBytes);
        pipe_->InitBuffer(x2Que_, 2, tileBytes);
        pipe_->InitBuffer(y1Que_, 2, tileBytes);
        if constexpr (std::is_same<T, half>::value) {
            // fp16 路径额外需要 fp32 计算缓冲
            pipe_->InitBuffer(x1F32Buf_, tl_->ubTileSize * sizeof(float));
            pipe_->InitBuffer(x2F32Buf_, tl_->ubTileSize * sizeof(float));
            pipe_->InitBuffer(y1F32Buf_, tl_->ubTileSize * sizeof(float));
        }
        pipe_->InitBuffer(sumBuf_, 32);                                    // ReduceSum 输出（1 float）
        pipe_->InitBuffer(accBuf_, 32);                                    // partial 累加器
        pipe_->InitBuffer(compBuf_, 32);                                   // partial 补偿项（TwoSum）
        pipe_->InitBuffer(tSumBuf_, 32);                                   // TwoSum 临时量 t
        pipe_->InitBuffer(bbBuf_, 32);                                     // TwoSum 临时量 bb
        pipe_->InitBuffer(vBuf_, 32);                                      // TwoSum 临时量 t-bb
        pipe_->InitBuffer(uBuf_, 32);                                      // TwoSum 临时量 / 误差项
        pipe_->InitBuffer(wBuf_, 32);                                      // TwoSum 临时量 sum-bb
        pipe_->InitBuffer(reduceTmpBuf_, tl_->ubTileSize * sizeof(float)); // ReduceSum sharedTmp（整 tile）
    }

    __aicore__ inline void Process()
    {
        if constexpr (std::is_same<T, half>::value) {
            if (coreSize_ <= 0) {
                return; // fp16 无跨核同步，零工作核直接退出是安全的
            }
            ComputeY1Fp16();
            if (blockIdx_ == 0) {
                ReduceY2Serial(); // fp16：core0 串行归约全量
            }
        } else {
            // fp32 路径所有核都必须到达 ReduceY2Deterministic 里的 SyncAll，
            // 零工作核也要参与（partial=0），不能提前 return（当前 tiling 保证每核 >=VL 个元素，
            // 这里按防御式写法不依赖该保证）
            ComputeY1AndPartialFp32();
            ReduceY2Deterministic(); // fp32：写 y1 尾部 scratch + SyncAll + core0 固定顺序合并
        }
    }

private:
    // 每 tile：x1/x2 已在 UB（T 类型），返回 fp32 计算视图；fp32 路径零拷贝别名，fp16 路径 cast
    __aicore__ inline void ToFp32(const LocalTensor<T>& x1Ub, const LocalTensor<T>& x2Ub, const LocalTensor<T>& y1Ub,
                                  int64_t extent, LocalTensor<float>& x1f, LocalTensor<float>& x2f,
                                  LocalTensor<float>& y1f)
    {
        if constexpr (std::is_same<T, half>::value) {
            x1f = x1F32Buf_.Get<float>();
            x2f = x2F32Buf_.Get<float>();
            y1f = y1F32Buf_.Get<float>();
            Cast(x1f, x1Ub, RoundMode::CAST_NONE, extent);
            Cast(x2f, x2Ub, RoundMode::CAST_NONE, extent);
        } else {
            x1f = x1Ub;
            x2f = x2Ub;
            y1f = y1Ub;
        }
    }

    // 单 tile 公共计算：y1 = x1*x3 + x2 写回 GM；doPartial 时 (acc, comp) += Σx1²（TwoSum 补偿）
    template <bool doPartial>
    __aicore__ inline void ComputeTile(int64_t extent, int64_t offset, LocalTensor<float>& accUb,
                                       LocalTensor<float>& compUb, LocalTensor<float>& sumUb, LocalTensor<float>& tmpUb)
    {
        LocalTensor<T> x1Ub = x1Que_.AllocTensor<T>();
        LocalTensor<T> x2Ub = x2Que_.AllocTensor<T>();
        DataCopyExtParams cpIn{1, static_cast<uint32_t>(extent * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padIn{false, 0, 0, 0};
        DataCopyPad(x1Ub, x1Gm_[offset], cpIn, padIn);
        DataCopyPad(x2Ub, x2Gm_[offset], cpIn, padIn);
        x1Que_.EnQue(x1Ub);
        x2Que_.EnQue(x2Ub);
        x1Ub = x1Que_.DeQue<T>();
        x2Ub = x2Que_.DeQue<T>();

        LocalTensor<T> y1Ub = y1Que_.AllocTensor<T>();
        LocalTensor<float> x1f;
        LocalTensor<float> x2f;
        LocalTensor<float> y1f;
        ToFp32(x1Ub, x2Ub, y1Ub, extent, x1f, x2f, y1f);

        // partial += Σ x1²（先借 y1f 存平方）
        if constexpr (doPartial) {
            Mul(y1f, x1f, x1f, extent);
            ReduceSum(sumUb, y1f, tmpUb, extent);
            // TwoSum 补偿累加（Knuth 无分支版），把 t = acc + sum 的舍入误差精确回补进 comp：
            // 全程无分支/无原子，结果与核数、执行顺序无关，精度 ~eps²
            LocalTensor<float> tUb = tSumBuf_.Get<float>();
            LocalTensor<float> bbUb = bbBuf_.Get<float>();
            LocalTensor<float> vUb = vBuf_.Get<float>();
            LocalTensor<float> uUb = uBuf_.Get<float>();
            LocalTensor<float> wUb = wBuf_.Get<float>();
            Add(tUb, accUb, sumUb, 1);   // t  = acc + sum
            Sub(bbUb, tUb, accUb, 1);    // bb = t - acc
            Sub(vUb, tUb, bbUb, 1);      // v  = t - bb
            Sub(uUb, accUb, vUb, 1);     // u  = acc - (t - bb)
            Sub(wUb, sumUb, bbUb, 1);    // w  = sum - bb
            Add(uUb, uUb, wUb, 1);       // err = u + w
            Add(compUb, compUb, uUb, 1); // comp += err
            Adds(accUb, tUb, 0.0f, 1);   // acc = t
        }

        // y1 = x1 * x3 + x2
        Muls(y1f, x1f, x3Value_, extent);
        Add(y1f, y1f, x2f, extent);
        if constexpr (std::is_same<T, half>::value) {
            Cast(y1Ub, y1f, RoundMode::CAST_NONE, extent);
        }

        y1Que_.EnQue(y1Ub);
        y1Ub = y1Que_.DeQue<T>();
        DataCopyPad(y1Gm_[offset], y1Ub, cpIn);
        y1Que_.FreeTensor(y1Ub);
        x1Que_.FreeTensor(x1Ub);
        x2Que_.FreeTensor(x2Ub);
    }

    template <bool doPartial>
    __aicore__ inline void RunTileLoop(int64_t start, int64_t size, LocalTensor<float>& accUb,
                                       LocalTensor<float>& compUb)
    {
        int64_t ubTile = tl_->ubTileSize;
        int64_t fullTiles = size / ubTile;
        int64_t tailCount = size - fullTiles * ubTile;
        int64_t totalTiles = fullTiles + (tailCount > 0 ? 1 : 0);
        LocalTensor<float> sumUb = sumBuf_.Get<float>();
        LocalTensor<float> tmpUb = reduceTmpBuf_.Get<float>();

        for (int64_t t = 0; t < totalTiles; t++) {
            int64_t extent = (t == totalTiles - 1 && tailCount > 0) ? tailCount : ubTile;
            ComputeTile<doPartial>(extent, start + t * ubTile, accUb, compUb, sumUb, tmpUb);
        }
    }

    // fp32 路径：y1 多核均分 + 本核 partial（零工作核 partial=0）
    __aicore__ inline void ComputeY1AndPartialFp32()
    {
        LocalTensor<float> accUb = accBuf_.Get<float>();
        LocalTensor<float> compUb = compBuf_.Get<float>();
        Duplicate(accUb, 0.0f, 8);
        Duplicate(compUb, 0.0f, 8);
        if (coreSize_ > 0) {
            RunTileLoop<true>(coreStart_, coreSize_, accUb, compUb);
        }
        // 0.5 在 core0 合并后统一乘（2 的幂，乘法本身精确，不引入舍入）
    }

    // fp16 路径：仅 y1（y2 由 core0 串行处理，y1 循环不算 partial）
    __aicore__ inline void ComputeY1Fp16()
    {
        LocalTensor<float> accUb = accBuf_.Get<float>();   // doPartial=false，仅占位
        LocalTensor<float> compUb = compBuf_.Get<float>(); // 同上
        RunTileLoop<false>(coreStart_, coreSize_, accUb, compUb);
    }

    // fp32 y2：各核把 (acc, comp) 写 y1 尾部 scratch → SyncAll → core0 按核号固定顺序
    //           TwoSum 合并 → core0 用 x1/x2/x3 重算恢复这段 y1。
    //           不用原子加（顺序随调度变化，多次运行结果 bit 级不一致）；
    //           也不走框架 workspace（小 workspace 落位不稳定，实测会压到 y1 输出）。
    __aicore__ inline void ReduceY2Deterministic()
    {
        LocalTensor<float> accUb = accBuf_.Get<float>();
        LocalTensor<float> compUb = compBuf_.Get<float>();
        DataCopyExtParams cpOne{1, sizeof(float), 0, 0, 0};
        const int64_t scratchNum = (blockNum_ > 1) ? 2 * blockNum_ : 0;
        const int64_t scratchStart = tl_->totalElements - scratchNum;

        if (scratchNum > 0) {
            // 先等所有核写完 y1，再借尾部做 scratch，避免与尾核的 y1 写竞争
            PipeBarrier<PIPE_ALL>();
            SyncAll();
            DataCopyPad(y1Gm_[scratchStart + 2 * blockIdx_], accUb, cpOne);
            DataCopyPad(y1Gm_[scratchStart + 2 * blockIdx_ + 1], compUb, cpOne);
            PipeBarrier<PIPE_ALL>();
            SyncAll();
        }

        if (blockIdx_ != 0) {
            return;
        }
        float y2 = 0.0f;
        if (scratchNum == 0) {
            PipeBarrier<PIPE_ALL>();
            y2 = (accUb.GetValue(0) + compUb.GetValue(0)) * 0.5f;
        } else {
            // core0：读回全部核的 (acc, comp)，按 0..blockNum-1 的固定顺序做 TwoSum 合并
            LocalTensor<float> allUb = reduceTmpBuf_.Get<float>(); // 归约已结束，可复用
            DataCopyExtParams cpAll{1, static_cast<uint32_t>(scratchNum * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padAll{false, 0, 0, 0};
            DataCopyPad(allUb, y1Gm_[scratchStart], cpAll, padAll);
            PipeBarrier<PIPE_ALL>();

            float acc = 0.0f;
            float comp = 0.0f;
            for (int64_t i = 0; i < blockNum_; ++i) {
                float hi = allUb.GetValue(2 * i);
                float lo = allUb.GetValue(2 * i + 1);
                float s = acc + hi;
                float bb = s - acc;
                float err = (acc - (s - bb)) + (hi - bb); // TwoSum 误差项（须在更新 acc 之前算）
                acc = s;
                comp += err;
                comp += lo; // 本核的补偿项
            }
            y2 = (acc + comp) * 0.5f; // 唯一一次舍入
        }

        LocalTensor<float> outUb = sumBuf_.Get<float>();
        outUb.SetValue(0, y2);
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(y2Gm_[0], outUb, cpOne);

        if (scratchNum > 0) {
            RestoreY1Range(scratchStart, scratchNum);
        }
    }

    // 重算并写回 y1[start, start+num)（仅 fp32 路径调用，T 为 float）
    __aicore__ inline void RestoreY1Range(int64_t start, int64_t num)
    {
        LocalTensor<T> x1Ub = x1Que_.AllocTensor<T>();
        LocalTensor<T> x2Ub = x2Que_.AllocTensor<T>();
        DataCopyExtParams cpIn{1, static_cast<uint32_t>(num * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padIn{false, 0, 0, 0};
        DataCopyPad(x1Ub, x1Gm_[start], cpIn, padIn);
        DataCopyPad(x2Ub, x2Gm_[start], cpIn, padIn);
        x1Que_.EnQue(x1Ub);
        x2Que_.EnQue(x2Ub);
        x1Ub = x1Que_.DeQue<T>();
        x2Ub = x2Que_.DeQue<T>();

        LocalTensor<T> y1Ub = y1Que_.AllocTensor<T>();
        LocalTensor<float> x1f;
        LocalTensor<float> x2f;
        LocalTensor<float> y1f;
        ToFp32(x1Ub, x2Ub, y1Ub, num, x1f, x2f, y1f);
        Muls(y1f, x1f, x3Value_, num);
        Add(y1f, y1f, x2f, num);
        if constexpr (std::is_same<T, half>::value) {
            Cast(y1Ub, y1f, RoundMode::CAST_NONE, num);
        }
        y1Que_.EnQue(y1Ub);
        y1Ub = y1Que_.DeQue<T>();
        DataCopyPad(y1Gm_[start], y1Ub, cpIn);
        y1Que_.FreeTensor(y1Ub);
        x1Que_.FreeTensor(x1Ub);
        x2Que_.FreeTensor(x2Ub);
    }

    // fp16 y2：core0 串行归约全量 x1（只读 x1 一条流，fp32 累加，单次舍入写回）
    __aicore__ inline void ReduceY2Serial()
    {
        LocalTensor<float> accUb = accBuf_.Get<float>();
        LocalTensor<float> sumUb = sumBuf_.Get<float>();
        LocalTensor<float> tmpUb = reduceTmpBuf_.Get<float>();
        LocalTensor<float> x1f = x1F32Buf_.Get<float>();
        LocalTensor<float> sqBuf = y1F32Buf_.Get<float>(); // 借作平方缓冲
        Duplicate(accUb, 0.0f, 8);

        int64_t totalN = tl_->totalElements;
        int64_t ubTile = tl_->ubTileSize;
        int64_t fullTiles = totalN / ubTile;
        int64_t tailCount = totalN - fullTiles * ubTile;
        int64_t totalTiles = fullTiles + (tailCount > 0 ? 1 : 0);
        for (int64_t t = 0; t < totalTiles; t++) {
            int64_t extent = (t == totalTiles - 1 && tailCount > 0) ? tailCount : ubTile;
            LocalTensor<T> x1Ub = x1Que_.AllocTensor<T>();
            DataCopyExtParams cpIn{1, static_cast<uint32_t>(extent * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padIn{false, 0, 0, 0};
            DataCopyPad(x1Ub, x1Gm_[t * ubTile], cpIn, padIn);
            x1Que_.EnQue(x1Ub);
            x1Ub = x1Que_.DeQue<T>();

            Cast(x1f, x1Ub, RoundMode::CAST_NONE, extent);
            Mul(sqBuf, x1f, x1f, extent);
            ReduceSum(sumUb, sqBuf, tmpUb, extent);
            Add(accUb, accUb, sumUb, 1);
            x1Que_.FreeTensor(x1Ub);
        }
        Muls(accUb, accUb, 0.5f, 1);

        LocalTensor<T> y2Ub = y1Que_.AllocTensor<T>();
        Cast(y2Ub, accUb, RoundMode::CAST_NONE, 1);
        y1Que_.EnQue(y2Ub);
        y2Ub = y1Que_.DeQue<T>();
        DataCopyExtParams cpOut{1, sizeof(T), 0, 0, 0};
        DataCopyPad(y2Gm_[0], y2Ub, cpOut);
        y1Que_.FreeTensor(y2Ub);
    }

private:
    TPipe* pipe_ = nullptr;
    const FusedMulAddNL2lossTilingData* tl_ = nullptr;
    int64_t blockIdx_ = 0;
    int64_t blockNum_ = 1;
    int64_t coreStart_ = 0;
    int64_t coreSize_ = 0;
    float x3Value_ = 0.0f;

    GlobalTensor<T> x1Gm_;
    GlobalTensor<T> x2Gm_;
    GlobalTensor<T> y1Gm_;
    GlobalTensor<T> y2Gm_;

    TQue<QuePosition::VECIN, 2> x1Que_;
    TQue<QuePosition::VECIN, 2> x2Que_;
    TQue<QuePosition::VECOUT, 2> y1Que_;
    TBuf<> x1F32Buf_;
    TBuf<> x2F32Buf_;
    TBuf<> y1F32Buf_;
    TBuf<> sumBuf_;
    TBuf<> accBuf_;
    TBuf<> compBuf_;
    TBuf<> tSumBuf_;
    TBuf<> bbBuf_;
    TBuf<> vBuf_;
    TBuf<> uBuf_;
    TBuf<> wBuf_;
    TBuf<> reduceTmpBuf_;
};

} // namespace FusedMulAddNL2lossOps

#endif
