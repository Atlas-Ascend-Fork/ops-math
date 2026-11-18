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
 * \file radix_top_k_select.h
 * \brief Radix TopK — SelectTopK 单遍扫描快路径（块扫描 + 阈值过滤 + 水位复选 + 16bit 键域二分）
 *
 * DAV_2201：向量操作数基址须 32B 对齐，候选写指针按 8 元素向上取整，间隙写 NaN 空槽。
 * NaN 与现网 radix 对齐：视为最大键，同组按原始索引升序，输出统一 0x7FFF。
 */

#ifndef RADIX_TOPK_SELECT_H
#define RADIX_TOPK_SELECT_H

#include "radix_top_k_common.h"

namespace RadixTopK {
using namespace AscendC;

static constexpr uint32_t STK_REPEAT_F32 = 64; // fp32 比较 repeat 元素数（256B / 4B）
static constexpr uint32_t STK_QUE_DEPTH = 3;   // staging TQue 编译期深度上限（运行期级数 <= 该值）
// 16bit 键域二分固定 17 轮（最坏 16 轮）；不收敛会让 tStar 落在 false 键，压实数 a > k（无符号下溢）
static constexpr int32_t STK_KEY16_BITS_LARGEST = 17;
static constexpr int32_t STK_KEY16_BITS_MIN = 17;
static constexpr uint32_t STK_SLOT_ALIGN = 64;        // GM 归并槽位步长对齐（64 元素 = 256B）
static constexpr uint32_t STK_EQ_DIRECT_VEC_MIN = 32; // 已对齐 EQ 启用直接向量写回的最短长度
static constexpr uint32_t STK_EQ_GATHER_VEC_MIN = 32; // 非对齐 EQ 启用 GatherMask 直接写回的最短长度
static constexpr uint32_t STK_MASK_BITS_PER_BYTE = 8; // 1 个掩码字节承载 8 个元素的比较结果
static constexpr uint32_t STK_KEY16_HIGH_SHIFT = 16;  // 16bit 键左移至 32bit 单调键高半区的位偏移

/** @brief 向上取整到 32 字节（DAV_2201 向量基址对齐要求） */
#define STK_ALIGN32BYTE(x) ((((x) + 31u) / 32u) * 32u)

__aicore__ inline uint32_t StkAlignUp(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }

/** @brief 逆 Twiddle：32bit 单调键 -> 原始 fp32 位模式 -> 浮点值 */
__aicore__ inline float StkKeyToFloat(uint32_t key)
{
    uint32_t raw = (key >= 0x80000000u) ? (key & 0x7FFFFFFFu) : (~key);
    return *reinterpret_cast<float*>(&raw);
}

/**
 * @brief 16bit 单调键 -> 浮点值（bf16）：32bit 键低 16 位恒为 0x0000/0xFFFF，可无损补回
 */
__aicore__ inline float StkKey16ToFloatBf16(uint32_t key16)
{
    const uint32_t lowBits = ((key16 & 0x8000u) != 0U) ? 0U : 0x0000FFFFu;
    return StkKeyToFloat((key16 << STK_KEY16_HIGH_SHIFT) | lowBits);
}

/**
 * @brief 16bit 单调键 -> 浮点值（fp16）：前缀仅 bf16 粒度会混淆同桶 8 值并越界写 ltV，故用 fp16 码还原
 */
__aicore__ inline float StkKey16ToFloatFp16(uint32_t key16)
{
    const uint16_t k = static_cast<uint16_t>(key16);
    // 逆 Twiddle：最高位为 1 表示正值（键 = 原始位 | 0x8000），异或还原；否则整体取反。
    // 勿用 (~(k >> 15))：uint16 逻辑右移会得到错误掩码。
    const uint16_t mask = ((k & 0x8000u) != 0U) ? static_cast<uint16_t>(0x8000u) : static_cast<uint16_t>(0xFFFFu);
    const uint16_t bits = static_cast<uint16_t>(k ^ mask);
    return static_cast<float>(*reinterpret_cast<const half*>(&bits));
}

template <typename T>
__aicore__ inline float StkKey16ToFloat(uint32_t key16)
{
    if constexpr (IsSameType<T, half>::value) {
        return StkKey16ToFloatFp16(key16);
    } else {
        return StkKey16ToFloatBf16(key16);
    }
}

/** @brief 16bit 键域二分下界：-inf 的单调键（bf16 0xFF80 -> 0x007F；fp16 0xFC00 -> 0x03FF） */
template <typename T>
__aicore__ inline uint32_t StkKey16NegInf()
{
    if constexpr (IsSameType<T, half>::value) {
        return 0x03FFu;
    } else {
        return 0x007Fu;
    }
}

/** @brief 16bit 键域二分上界：+inf 的单调键（bf16 0x7F80 -> 0xFF80；fp16 0x7C00 -> 0xFC00） */
template <typename T>
__aicore__ inline uint32_t StkKey16PosInf()
{
    if constexpr (IsSameType<T, half>::value) {
        return 0xFC00u;
    } else {
        return 0xFF80u;
    }
}

/** @brief 安静 NaN：在 LT / LE / GT / GE / EQ 比较下恒 false ⇒ 可作空槽哨兵 */
__aicore__ inline float StkNanValue()
{
    uint32_t bits = 0x7FC00000u;
    return *reinterpret_cast<float*>(&bits);
}

/** @brief 与现网 radix 的 NaN 规范化结果一致：fp16/bf16 均为 0x7FFF */
template <typename T>
__aicore__ inline T StkCanonicalNan()
{
    const uint16_t bits = 0x7FFFu;
    return *reinterpret_cast<const T*>(&bits);
}

/** @brief +Inf（min 方向的行内多核空槽哨兵：永不入选最小 k） */
__aicore__ inline float StkPosInfValue()
{
    uint32_t bits = 0x7F800000u;
    return *reinterpret_cast<float*>(&bits);
}

/** @brief -Inf（max 方向的行内多核空槽哨兵：永不入选最大 k） */
__aicore__ inline float StkNegInfValue()
{
    uint32_t bits = 0xFF800000u;
    return *reinterpret_cast<float*>(&bits);
}

template <typename DT>
__aicore__ inline void StkFillScalar(LocalTensor<DT> v, uint32_t from, uint32_t to, DT val)
{
    for (uint32_t i = from; i < to; ++i) {
        v.SetValue(i, val);
    }
}

__aicore__ inline bool StkIsNan(float v)
{
    const uint32_t bits = *reinterpret_cast<uint32_t*>(&v);
    return (bits & 0x7FFFFFFFu) > 0x7F800000u;
}

#define STK_MASK_BINARY(OP, dst, a, b, cnt)                                                \
    OP((dst).template ReinterpretCast<int16_t>(), (a).template ReinterpretCast<int16_t>(), \
       (b).template ReinterpretCast<int16_t>(),                                            \
       static_cast<uint32_t>(((cnt) / STK_MASK_BITS_PER_BYTE) / sizeof(int16_t)))

/** @brief SelectTopK 单遍扫描快路径；stkCoreMode=0 行-核映射逐行写回，=1 行内多核树形归并 */
template <typename T, bool largest>
class RadixTopKSelectTopK {
public:
    __aicore__ inline RadixTopKSelectTopK(TPipe& tpipe, const RadixTopKTilingData& tilingData)
        : tpipe_(tpipe), tiling_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR k, GM_ADDR values, GM_ADDR indices, GM_ADDR workspace)
    {
        blockIdx_ = GetBlockIdx();
        blockDim_ = tiling_.coreNum > 0 ? tiling_.coreNum : static_cast<uint64_t>(GetBlockNum());
        sortLen_ = tiling_.sortLen;
        batch_ = tiling_.batch;
        kValue_ = static_cast<uint32_t>(tiling_.kValue);
        blockSize_ = static_cast<uint32_t>(tiling_.stkBlockSize);
        numStages_ = static_cast<uint32_t>(tiling_.stkNumStages);
        cap_ = static_cast<uint32_t>(tiling_.stkCandidateCapacity);
        safeTrigger_ = static_cast<uint32_t>(tiling_.stkSafeTrigger);
        coreMode_ = tiling_.stkCoreMode;
        slotStride_ = StkAlignUp(kValue_, STK_SLOT_ALIGN);
        mergeArity_ = (tiling_.stkMergeArity == 0U) ? 0U : ((tiling_.stkMergeArity >= 2U) ? tiling_.stkMergeArity : 2U);

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), batch_ * sortLen_);
        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(values), batch_ * kValue_);
        indexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices), batch_ * kValue_);
        // 树形归并槽位：每核 k 个 fp32 值 + k 个 int32 索引，仅 stkCoreMode==1 使用
        wsV_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace), blockDim_ * slotStride_);
        wsI_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace) + blockDim_ * slotStride_,
                             blockDim_ * slotStride_);

        InitBuffers();
    }

    __aicore__ inline void Process();

private:
    __aicore__ inline void InitBuffers()
    {
        // 掩码按「对齐到 STK_REPEAT_F32 的元素数 / 8」取字节再向 32B 对齐（复选 Compares 的 cnt 可略超 cap）
        uint32_t maskBytes = static_cast<uint32_t>(
            STK_ALIGN32BYTE((static_cast<uint64_t>(cap_) + STK_REPEAT_F32) / STK_MASK_BITS_PER_BYTE));
        // fp32Buf_ 兼作扫描工作区与 ReduceToK 的 NaN 索引暂存，容量 ≥ k，与 tiling 的 UB 核算一致。
        const uint32_t kA = StkAlignUp(kValue_, 8u);
        uint32_t mergeSides = 0;
        if (mergeArity_ == 0U) {
            mergeSides = (cap_ > kValue_) ? ((cap_ - kValue_) / kA) : 0U;
            // 单批最多合并 blockDim-1 个兄弟槽位（与 tiling 的 UB 核算同口径）
            const uint32_t maxSides = (blockDim_ > 0U) ? static_cast<uint32_t>(blockDim_ - 1U) : 0U;
            if (mergeSides > maxSides) {
                mergeSides = maxSides;
            }
        } else if (mergeArity_ > 2U) {
            mergeSides = mergeArity_ - 1U;
        }
        const uint32_t mergeElems = (mergeArity_ != 2U) ? (mergeSides * kA + kValue_) : blockSize_;
        const uint32_t fp32Elems = (mergeElems > blockSize_) ? mergeElems : blockSize_;
        tpipe_.InitBuffer(inQue_, numStages_ > STK_QUE_DEPTH ? STK_QUE_DEPTH : numStages_,
                          static_cast<uint32_t>(blockSize_ * sizeof(T)));
        tpipe_.InitBuffer(fp32Buf_, static_cast<uint32_t>(fp32Elems * sizeof(float)));
        tpipe_.InitBuffer(candVBuf_, static_cast<uint32_t>(cap_ * sizeof(float)));
        tpipe_.InitBuffer(candIBuf_, static_cast<uint32_t>(cap_ * sizeof(int32_t)));
        tpipe_.InitBuffer(scratchBuf_, static_cast<uint32_t>(cap_ * sizeof(int32_t)));
        tpipe_.InitBuffer(ltVBuf_, static_cast<uint32_t>(kValue_ * sizeof(float)));
        tpipe_.InitBuffer(ltIBuf_, static_cast<uint32_t>(kValue_ * sizeof(int32_t)));
        tpipe_.InitBuffer(maskLBuf_, maskBytes);
        tpipe_.InitBuffer(maskEBuf_, maskBytes);
    }

    /** @brief 单遍扫描 [start, start+len) 并压缩候选到 candV/candI，返回 nCand 与实时复合阈值 */
    __aicore__ inline void ScanAndSelect(uint64_t rowBase, uint64_t start, uint64_t len, int32_t& nCand, float& thr,
                                         int32_t& thrIdx)
    {
        LocalTensor<float> curV = candVBuf_.Get<float>();
        LocalTensor<int32_t> curI = candIBuf_.Get<int32_t>();
        LocalTensor<uint8_t> mf = maskLBuf_.Get<uint8_t>();
        LocalTensor<uint32_t> mf32 = mf.template ReinterpretCast<uint32_t>();
        LocalTensor<uint8_t> mE = maskEBuf_.Get<uint8_t>();
        LocalTensor<int32_t> idxT = scratchBuf_.Get<int32_t>();
        LocalTensor<uint8_t> mi = idxT[blockSize_].template ReinterpretCast<uint8_t>();

        uint64_t envEnd = start + len;
        uint64_t off = start;
        nCand = 0;
        while (off < envEnd) {
            uint32_t n = static_cast<uint32_t>(envEnd - off);
            if (n > blockSize_) {
                n = blockSize_;
            }
            LocalTensor<T> st = inQue_.AllocTensor<T>();
            DataCopyPad(st, xGm_[rowBase + off], DataCopyExtParams{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0},
                        DataCopyPadExtParams<T>{false, 0, 0, static_cast<T>(0)});
            inQue_.EnQue(st);
            st = inQue_.DeQue<T>();

            LocalTensor<float> f32 = fp32Buf_.Get<float>();
            Cast(f32, st, RoundMode::CAST_NONE, n);
            CreateVecIndexPerf<int32_t>(idxT, static_cast<int32_t>(off), n);

            uint32_t cmpCnt = StkAlignUp(n, STK_REPEAT_F32);
            // 尾部 lane 同时填充「永不入选」的值和索引，避免 EQ+索引复合掩码误选旧数据。
            if (cmpCnt > n) {
                // 标量写 UB 是 32B 块读改写，须先等向量写完，否则会带回上一块同 lane 的旧值。
                VToSSync();
                if constexpr (largest) {
                    StkFillScalar(f32, n, cmpCnt, StkNegInfValue());
                } else {
                    StkFillScalar(f32, n, cmpCnt, StkNanValue());
                }
                StkFillScalar(idxT, n, cmpCnt, static_cast<int32_t>(0x7FFFFFFF));
                SToVSync();
            }
            PipeBarrier<PIPE_V>();
            if (off == start) {
                // 首块无需阈值过滤；NaN 也必须进入候选，由 ReduceToK 按最大键语义统一处理。
                StkFillScalar(mf, 0, cmpCnt / STK_MASK_BITS_PER_BYTE, static_cast<uint8_t>(0xFF));
                SToVSync();
            } else if (StkIsNan(thr)) {
                // 边界是 NaN：largest 只让更小索引的 NaN 竞争；smallest 允许正常值替换 NaN
                if constexpr (largest) {
                    Compare<float, uint8_t>(mf, f32, f32, CMPMODE::EQ, cmpCnt);
                    PipeBarrier<PIPE_V>();
                    Not(mf.template ReinterpretCast<int16_t>(), mf.template ReinterpretCast<int16_t>(),
                        static_cast<uint32_t>((cmpCnt / STK_MASK_BITS_PER_BYTE) / sizeof(int16_t)));
                } else {
                    Compare<float, uint8_t>(mE, f32, f32, CMPMODE::EQ, cmpCnt); // 非 NaN
                    Compare<float, uint8_t>(mf, f32, f32, CMPMODE::EQ, cmpCnt);
                    PipeBarrier<PIPE_V>();
                    Not(mf.template ReinterpretCast<int16_t>(), mf.template ReinterpretCast<int16_t>(),
                        static_cast<uint32_t>((cmpCnt / STK_MASK_BITS_PER_BYTE) / sizeof(int16_t)));
                }
                Compares<int32_t, uint8_t>(mi, idxT, thrIdx, CMPMODE::LT, cmpCnt);
                PipeBarrier<PIPE_V>();
                STK_MASK_BINARY(And, mf, mf, mi, cmpCnt);
                if constexpr (!largest) {
                    PipeBarrier<PIPE_V>();
                    STK_MASK_BINARY(Or, mf, mf, mE, cmpCnt);
                }
            } else {
                if constexpr (largest) {
                    // 值掩码 = (x > T) ∪ isNaN ← NOT(LE(T))；正常边界下任意 NaN 都优于正常值。
                    Compares<float, uint8_t>(mf, f32, thr, CMPMODE::LE, cmpCnt);
                    PipeBarrier<PIPE_V>();
                    Not(mf.template ReinterpretCast<int16_t>(), mf.template ReinterpretCast<int16_t>(),
                        static_cast<uint32_t>((cmpCnt / STK_MASK_BITS_PER_BYTE) / sizeof(int16_t)));
                } else {
                    Compares<float, uint8_t>(mf, f32, thr, CMPMODE::LT, cmpCnt);
                }
                // 当前阈值的同值元素也参与竞争，但仅当原始索引优于已选最差 EQ 索引。
                Compares<float, uint8_t>(mE, f32, thr, CMPMODE::EQ, cmpCnt);
                Compares<int32_t, uint8_t>(mi, idxT, thrIdx, CMPMODE::LT, cmpCnt);
                PipeBarrier<PIPE_V>();
                STK_MASK_BINARY(And, mE, mE, mi, cmpCnt);
                PipeBarrier<PIPE_V>();
                STK_MASK_BINARY(Or, mf, mf, mE, cmpCnt);
            }
            PipeBarrier<PIPE_V>();

            // dst 基址须 32B 对齐；间隙是空槽而非输入 NaN，统一放在复合序末尾
            uint32_t dstOff = StkAlignUp(static_cast<uint32_t>(nCand), 8u);
            if (dstOff > static_cast<uint32_t>(nCand)) {
                const float empty = largest ? StkNegInfValue() : StkPosInfValue();
                // 间隙起点不一定 8 对齐，先等上一轮向量压实落地
                VToSSync();
                StkFillScalar(curV, static_cast<uint32_t>(nCand), dstOff, empty);
                StkFillScalar(curI, static_cast<uint32_t>(nCand), dstOff, static_cast<int32_t>(0x7FFFFFFF));
                SToVSync();
            }
            uint64_t cntV = 0;
            GatherMask(curV[dstOff], f32, mf32, true, n, {1, 1, 0, 0}, cntV);
            uint64_t cntI = 0;
            GatherMask(curI[dstOff], idxT, mf32, true, n, {1, 1, 0, 0}, cntI);

            inQue_.FreeTensor(st);
            VToSSync();
            nCand = static_cast<int32_t>(dstOff + static_cast<uint32_t>(cntV));

            if (nCand >= static_cast<int32_t>(safeTrigger_)) {
                Refine(curV, curI, nCand, thr, thrIdx);
            }
            off += n;
        }
    }

    /**
     * @brief 复选收敛：二分定界 → 外侧压实 → EQ 索引前缀补齐，就地保留恰好 k 个并给出阈值 (thr, thrIdx)
     */
    __aicore__ inline void ReduceToK(LocalTensor<float>& curV, LocalTensor<int32_t>& curI, int32_t& nCand, float& thr,
                                     int32_t& thrIdx)
    {
        const uint32_t n = static_cast<uint32_t>(nCand);
        const uint32_t cnt = StkAlignUp(n, STK_REPEAT_F32);
        // 比较/压实按 64 元素 repeat 覆盖 [n, cnt)：值填 NaN（任何比较恒 false），索引填 INT32_MAX
        if (cnt > n) {
            // 标量写 UB 是 32B 块读改写，n 非 8 的倍数时会带回 lane n-1 的旧内容，须先等向量完成
            VToSSync();
            StkFillScalar(curV, n, cnt, StkNanValue());
            StkFillScalar(curI, n, cnt, static_cast<int32_t>(0x7FFFFFFF));
            SToVSync();
        }
        LocalTensor<uint8_t> mL = maskLBuf_.Get<uint8_t>();
        LocalTensor<uint32_t> mL32 = mL.template ReinterpretCast<uint32_t>();
        LocalTensor<uint8_t> mE = maskEBuf_.Get<uint8_t>();
        LocalTensor<uint32_t> mE32 = mE.template ReinterpretCast<uint32_t>();
        LocalTensor<float> scratchV = scratchBuf_.Get<float>();
        LocalTensor<int32_t> scratchI = scratchBuf_.Get<int32_t>();
        // NaN 索引暂存复用扫描 fp32 工作区（容量 ≥ k，ReduceToK 期间扫描已暂停）
        LocalTensor<int32_t> nanI = fp32Buf_.Get<int32_t>();
        LocalTensor<float> ltV = ltVBuf_.Get<float>();
        LocalTensor<int32_t> ltI = ltIBuf_.Get<int32_t>();

        // --- 0. NaN 预分类：self-EQ 取反得到 isNaN，保序压实原始索引 ---
        // 先压到 cap 大小的 scratchBuf_（n 恒 < cap），再取索引最小的前 k 个搬到 nanI，避免写穿 fp32Buf_。
        Compare<float, uint8_t>(mE, curV, curV, CMPMODE::EQ, cnt);
        PipeBarrier<PIPE_V>();
        Not(mE.template ReinterpretCast<int16_t>(), mE.template ReinterpretCast<int16_t>(),
            static_cast<uint32_t>((cnt / STK_MASK_BITS_PER_BYTE) / sizeof(int16_t)));
        PipeBarrier<PIPE_V>();
        uint64_t nanCnt64 = 0;
        GatherMask(scratchI, curI, mE32, true, n, {1, 1, 0, 0}, nanCnt64);
        VToSSync();
        const uint32_t nanCnt = (nanCnt64 < n) ? static_cast<uint32_t>(nanCnt64) : n;
        const uint32_t nanKeep = (nanCnt < kValue_) ? nanCnt : kValue_;
        for (uint32_t i = 0; i < nanKeep; ++i) {
            nanI.SetValue(i, scratchI.GetValue(i));
        }

        if constexpr (largest) {
            // NaN 是最大键：k 个 NaN 已足够时，正常值无需再参与二分。
            if (nanCnt >= kValue_) {
                for (uint32_t i = 0; i < kValue_; ++i) {
                    curV.SetValue(i, StkNanValue());
                    curI.SetValue(i, nanI.GetValue(i));
                }
                SToVSync();
                nCand = static_cast<int32_t>(kValue_);
                thr = StkNanValue();
                thrIdx = nanI.GetValue(kValue_ - 1);
                return;
            }
        } else {
            // 正常值必须同时满足「非 NaN」和「索引有效」，排除 8 元素对齐引入的 ±Inf/INT32_MAX 空槽。
            Compare<float, uint8_t>(mL, curV, curV, CMPMODE::EQ, cnt);
            Compares<int32_t, uint8_t>(mE, curI, static_cast<int32_t>(0x7FFFFFFF), CMPMODE::EQ, cnt);
            PipeBarrier<PIPE_V>();
            Not(mE.template ReinterpretCast<int16_t>(), mE.template ReinterpretCast<int16_t>(),
                static_cast<uint32_t>((cnt / STK_MASK_BITS_PER_BYTE) / sizeof(int16_t)));
            PipeBarrier<PIPE_V>();
            STK_MASK_BINARY(And, mL, mL, mE, cnt);
            PipeBarrier<PIPE_V>();
            uint64_t nvCnt = 0;
            GatherMask(scratchV, curV, mL32, true, n, {1, 1, 0, 0}, nvCnt);
            VToSSync();
            const uint32_t normalCnt = (nvCnt < n) ? static_cast<uint32_t>(nvCnt) : n;
            // 正常值不足 k 时全部保留，再用原始索引最小的 NaN 补足，杜绝 INT32_MAX 伪索引。
            if (normalCnt < kValue_) {
                uint64_t nvc = 0;
                GatherMask(ltV, curV, mL32, true, n, {1, 1, 0, 0}, nvc);
                uint64_t nic = 0;
                GatherMask(ltI, curI, mL32, true, n, {1, 1, 0, 0}, nic);
                VToSSync();
                for (uint32_t i = 0; i < normalCnt; ++i) {
                    curV.SetValue(i, ltV.GetValue(i));
                    curI.SetValue(i, ltI.GetValue(i));
                }
                for (uint32_t i = normalCnt; i < kValue_; ++i) {
                    curV.SetValue(i, StkNanValue());
                    curI.SetValue(i, nanI.GetValue(i - normalCnt));
                }
                SToVSync();
                nCand = static_cast<int32_t>(kValue_);
                thr = StkNanValue();
                thrIdx = nanI.GetValue(kValue_ - normalCnt - 1);
                return;
            }
        }

        // largest 先给 NaN 分配配额，正常值只需竞争 normalK 个槽位；smallest 此处正常值不少于 k。
        const uint32_t normalK = largest ? (kValue_ - nanCnt) : kValue_;

        // --- 1. 16bit 单调键域二分定位正常值边界键 K*（NaN 比较恒 false，天然不计数） ---
        const uint32_t keyNegInf = StkKey16NegInf<T>();
        const uint32_t keyPosInf = StkKey16PosInf<T>();
        uint32_t lo;
        uint32_t hi;
        if constexpr (largest) {
            lo = keyNegInf;
            hi = keyPosInf;
            for (int32_t it = 0; it < STK_KEY16_BITS_LARGEST; ++it) {
                uint32_t mid = lo + ((hi - lo + 1u) >> 1);
                float tm = StkKey16ToFloat<T>(mid);
                Compares<float, uint8_t>(mL, curV, tm, CMPMODE::GE, cnt);
                PipeBarrier<PIPE_V>();
                uint64_t rc = 0;
                GatherMask(scratchV, curV, mL32, true, n, {1, 1, 0, 0}, rc);
                VToSSync();
                if (static_cast<uint32_t>(rc) >= normalK) {
                    lo = mid;
                } else {
                    hi = mid - 1u;
                }
            }
        } else {
            lo = 0u;
            hi = keyPosInf;
            for (int32_t it = 0; it < STK_KEY16_BITS_MIN; ++it) {
                uint32_t mid = lo + ((hi - lo) >> 1);
                float tm = StkKey16ToFloat<T>(mid);
                Compares<float, uint8_t>(mL, curV, tm, CMPMODE::LE, cnt);
                PipeBarrier<PIPE_V>();
                uint64_t rc = 0;
                GatherMask(scratchV, curV, mL32, true, n, {1, 1, 0, 0}, rc);
                VToSSync();
                if (static_cast<uint32_t>(rc) >= normalK) {
                    hi = mid;
                } else {
                    lo = mid + 1u;
                }
            }
        }
        const float tStar = StkKey16ToFloat<T>(lo);

        // --- 2. 正常值严格外侧与 EQ 分类 ---
        PipeBarrier<PIPE_V>();
        if constexpr (largest) {
            Compares<float, uint8_t>(mL, curV, tStar, CMPMODE::GT, cnt);
        } else {
            Compares<float, uint8_t>(mL, curV, tStar, CMPMODE::LT, cnt);
        }
        Compares<float, uint8_t>(mE, curV, tStar, CMPMODE::EQ, cnt);
        PipeBarrier<PIPE_V>();

        // --- 3. 保序压实正常值严格侧与 EQ 索引 ---
        // 只压实 EQ 的索引，值由 tStar 再生；EQ 索引写入 scratchBuf_[0,eCnt)，eCnt ≤ n ≤ cap 不越界。
        // 勿改成压实 EQ 值：容量不足时会覆写 ltVBuf_，把外侧值污染成 tStar。
        uint64_t aCnt = 0;
        GatherMask(ltV, curV, mL32, true, n, {1, 1, 0, 0}, aCnt);
        uint64_t aiCnt = 0;
        GatherMask(ltI, curI, mL32, true, n, {1, 1, 0, 0}, aiCnt);
        uint64_t eCnt = 0;
        GatherMask(scratchI, curI, mE32, true, n, {1, 1, 0, 0}, eCnt);
        VToSSync();

        const uint32_t aU = (static_cast<uint32_t>(aCnt) < normalK) ? static_cast<uint32_t>(aCnt) : normalK;
        const uint32_t eqNeed = normalK - aU;
        uint32_t eqHave = (static_cast<uint32_t>(eCnt) < eqNeed) ? static_cast<uint32_t>(eCnt) : eqNeed;
        const bool eqDegraded = eqHave < eqNeed;
        if (eqDegraded) {
            // scratchI 刚由向量写满 [0,eCnt)，标量补空槽按 32B 块读改写，需先等向量落地。
            VToSSync();
            StkFillScalar(scratchI, eqHave, eqNeed, static_cast<int32_t>(0x7FFFFFFF));
            SToVSync();
            eqHave = eqNeed;
        }

        // --- 3.5 EQ 段值的再生与写回 [aU,normalK)，须在步骤 4 覆写尾部之前完成；±0 按原始位模式回填 ---
        if (eqNeed > 0) {
            if (tStar == 0.0f) {
                // 先搬到 fp32Buf_（与候选区分离，避免边读边写覆盖），再整体写回。
                LocalTensor<float> eqValTmp = fp32Buf_.Get<float>();
                uint32_t slot = 0;
                for (uint32_t i = 0; i < n && slot < eqNeed; ++i) {
                    const float v = curV.GetValue(i);
                    if (v == tStar) {
                        eqValTmp.SetValue(slot, v);
                        ++slot;
                    }
                }
                for (uint32_t j = slot; j < eqNeed; ++j) {
                    eqValTmp.SetValue(j, tStar);
                }
                SToVSync();
                for (uint32_t j = 0; j < eqNeed; ++j) {
                    curV.SetValue(aU + j, eqValTmp.GetValue(j));
                }
                SToVSync();
            } else {
                const uint32_t vhead = (((8u - (aU & 7u)) & 7u) < eqNeed) ? ((8u - (aU & 7u)) & 7u) : eqNeed;
                for (uint32_t t = 0; t < vhead; ++t) {
                    curV.SetValue(aU + t, tStar);
                }
                SToVSync();
                if (eqNeed > vhead) {
                    Duplicate(curV[aU + vhead], tStar, eqNeed - vhead);
                }
                VToSSync();
            }
        }

        // --- 4. 正常值严格侧 [0,aU) ---
        const uint32_t kA8 = normalK & ~7u;
        const uint32_t aV = (aU < kA8 ? aU : kA8) & ~7u;
        if (aV > 0) {
            PipeBarrier<PIPE_V>();
            CopyData<float>(curV, ltV, aV);
            CopyData<int32_t>(curI, ltI, aV);
        }
        VToSSync();
        for (uint32_t i = aV; i < aU; ++i) {
            curV.SetValue(i, ltV.GetValue(i));
            curI.SetValue(i, ltI.GetValue(i));
        }

        // largest 的 NaN 配额置于正常值之后；此后 fp32Buf_ 才可复用为临时索引区
        if constexpr (largest) {
            for (uint32_t i = normalK; i < kValue_; ++i) {
                curV.SetValue(i, StkNanValue());
                curI.SetValue(i, nanI.GetValue(i - normalK));
            }
        }
        SToVSync();

        // --- 5. 正常值 EQ 段索引 [aU,normalK)：对齐走 CopyData，非对齐头标量 + GatherMask，尾部标量 ---
        const uint32_t pad = (8u - (aU & 7u)) & 7u;
        const uint32_t head = (pad < eqNeed) ? pad : eqNeed;
        const uint32_t remain = eqNeed - head;
        const uint32_t vecN = remain & ~7u;
        const bool directVec = !eqDegraded && ((aU & 7u) == 0u) && (eqNeed >= STK_EQ_DIRECT_VEC_MIN);
        const bool gatherVec = !eqDegraded && ((aU & 7u) != 0u) && (remain >= STK_EQ_GATHER_VEC_MIN);
        if (directVec) {
            if (vecN > 0) {
                PipeBarrier<PIPE_V>();
                CopyData<int32_t>(curI[aU], scratchI, vecN);
            }
            VToSSync();
            for (uint32_t t = vecN; t < eqNeed; ++t) {
                curI.SetValue(aU + t, scratchI.GetValue(t));
            }
        } else if (gatherVec) {
            for (uint32_t t = 0; t < head; ++t) {
                curI.SetValue(aU + t, scratchI.GetValue(t));
            }
            SToVSync();
            const uint32_t srcN = head + vecN;
            // 用 fp32 位置索引，避开本平台异常的 int32 边界比较路径；srcN ≤ k ≤ 4096 可精确表示。
            LocalTensor<float> idxTmp = fp32Buf_.Get<float>();
            CreateVecIndex<float>(idxTmp, 0.0f, srcN);
            PipeBarrier<PIPE_V>();
            const uint32_t cmpCnt = StkAlignUp(srcN, STK_REPEAT_F32);
            Compares<float, uint8_t>(mL, idxTmp, static_cast<float>(static_cast<int32_t>(head)), CMPMODE::GE, cmpCnt);
            PipeBarrier<PIPE_V>();
            uint64_t gatheredI = 0;
            GatherMask(curI[aU + head], scratchI, mL32, true, srcN, {1, 1, 0, 0}, gatheredI);
            VToSSync();
            for (uint32_t t = head + vecN; t < eqNeed; ++t) {
                curI.SetValue(aU + t, scratchI.GetValue(t));
            }
        } else {
            for (uint32_t t = 0; t < eqNeed; ++t) {
                curI.SetValue(aU + t, scratchI.GetValue(t));
            }
        }
        SToVSync();

        nCand = static_cast<int32_t>(kValue_);
        thr = tStar;
        thrIdx = (eqHave > 0) ? scratchI.GetValue(eqHave - 1) : 0;
    }

    __aicore__ inline void Refine(LocalTensor<float>& curV, LocalTensor<int32_t>& curI, int32_t& nCand, float& thr,
                                  int32_t& thrIdx)
    {
        ReduceToK(curV, curI, nCand, thr, thrIdx);
    }

    /** @brief 把本核本地候选（不足 k）用哨兵补齐到恰好 k 个（行内多核归并用） */
    __aicore__ inline void PadLocalToK(LocalTensor<float> curV, LocalTensor<int32_t> curI, int32_t nCand)
    {
        float sentinel = largest ? StkNegInfValue() : StkPosInfValue();
        // 候选 [0,nCand) 由向量压实写入，标量补位按 32B 块读改写，先等向量落地
        VToSSync();
        for (uint32_t i = static_cast<uint32_t>(nCand); i < kValue_; ++i) {
            curV.SetValue(i, sentinel);
            curI.SetValue(i, static_cast<int32_t>(0x7FFFFFFF));
        }
        SToVSync();
    }

    /** @brief 把 UB 中恰好 k 个结果写回该行输出（values: 就地 Cast bf16；indices: int32） */
    __aicore__ inline void WriteRow(uint64_t rowOff, LocalTensor<float> srcV, LocalTensor<int32_t> srcI)
    {
        // 就地 Cast 会覆盖 srcV 的前半区，必须在其之前生成 NaN 位置掩码。
        LocalTensor<uint8_t> mN = maskLBuf_.Get<uint8_t>();
        LocalTensor<uint32_t> mN32 = mN.template ReinterpretCast<uint32_t>();
        LocalTensor<float> scratchV = scratchBuf_.Get<float>();
        const uint32_t cmpCnt = StkAlignUp(kValue_, STK_REPEAT_F32);
        Compare<float, uint8_t>(mN, srcV, srcV, CMPMODE::EQ, cmpCnt);
        PipeBarrier<PIPE_V>();
        Not(mN.template ReinterpretCast<int16_t>(), mN.template ReinterpretCast<int16_t>(),
            static_cast<uint32_t>((cmpCnt / STK_MASK_BITS_PER_BYTE) / sizeof(int16_t)));
        PipeBarrier<PIPE_V>();
        uint64_t nanCnt = 0;
        GatherMask(scratchV, srcV, mN32, true, kValue_, {1, 1, 0, 0}, nanCnt);
        VToSSync();
        LocalTensor<T> ov = srcV.template ReinterpretCast<T>();
        PipeBarrier<PIPE_V>();
        Cast(ov, srcV, RoundMode::CAST_RINT, kValue_);
        if (nanCnt > 0) {
            // 与现网 radix 的 Muls(+1) 规范化结果对齐：所有入选 NaN 统一写 0x7FFF。
            VToSSync();
            for (uint32_t i = 0; i < kValue_; ++i) {
                const uint8_t bit = static_cast<uint8_t>(1u << (i & 7u));
                if ((mN.GetValue(i / STK_MASK_BITS_PER_BYTE) & bit) != 0) {
                    ov.SetValue(i, StkCanonicalNan<T>());
                }
            }
            SToVSync();
        }
        VToMTE3Sync();
        DataCopyPad(valueGm_[rowOff], ov, DataCopyExtParams{1, static_cast<uint32_t>(kValue_ * sizeof(T)), 0, 0, 0});
        DataCopyPad(indexGm_[rowOff], srcI,
                    DataCopyExtParams{1, static_cast<uint32_t>(kValue_ * sizeof(int32_t)), 0, 0, 0});
        // 输出搬运读的是 candV/candI，须等 MTE3 读完才能复用该缓冲，否则下一行会覆盖仍在读的尾部。
        MTE3ToVSync();
        MTE3ToSSync();
    }

    /** @brief 扫完该行/段后，把候选收敛为恰好 k 个（保持索引升序，有序性由 aclnn 复合链承接） */
    __aicore__ inline void FinalizeLocal(int32_t& nCand, float& thr, int32_t& thrIdx)
    {
        LocalTensor<float> curV = candVBuf_.Get<float>();
        LocalTensor<int32_t> curI = candIBuf_.Get<int32_t>();
        if (nCand > static_cast<int32_t>(kValue_)) {
            ReduceToK(curV, curI, nCand, thr, thrIdx);
        } else if (nCand < static_cast<int32_t>(kValue_)) {
            PadLocalToK(curV, curI, nCand);
            nCand = static_cast<int32_t>(kValue_);
        }
    }

    /** @brief stkCoreMode==0：行-核映射，整行主体 + 余数行块内段化 */
    __aicore__ inline void ProcessRowMode()
    {
        uint64_t rowsPerCore = tiling_.stkRowsPerCore;
        uint64_t formerCores = tiling_.stkFormerCoreNum;
        uint64_t rowStart;
        uint64_t rowNum;
        if (blockIdx_ < formerCores) {
            rowStart = blockIdx_ * rowsPerCore;
            rowNum = rowsPerCore;
        } else {
            rowStart = formerCores * rowsPerCore + (blockIdx_ - formerCores) * (rowsPerCore - 1);
            rowNum = (rowsPerCore > 0) ? (rowsPerCore - 1) : 0;
        }
        LocalTensor<float> curV = candVBuf_.Get<float>();
        LocalTensor<int32_t> curI = candIBuf_.Get<int32_t>();
        for (uint64_t r = rowStart; r < rowStart + rowNum; ++r) {
            // smallest 初始以 NaN 为最大键边界，保证首次复选前 NaN 也能进入候选。
            float thr = largest ? StkNegInfValue() : StkNanValue();
            int32_t thrIdx = static_cast<int32_t>(sortLen_);
            int32_t nCand = 0;
            ScanAndSelect(r * sortLen_, 0, sortLen_, nCand, thr, thrIdx);
            if (nCand > static_cast<int32_t>(kValue_)) {
                ReduceToK(curV, curI, nCand, thr, thrIdx);
            } else if (nCand < static_cast<int32_t>(kValue_)) {
                PadLocalToK(curV, curI, nCand);
            }
            WriteRow(r * kValue_, curV, curI);
        }
        // 余数行块内段化：主体整行之后，stkResidualRows 行由全核协作（复用段扫描 + 树归并）
        for (uint64_t r = tiling_.stkFullRows; r < tiling_.stkFullRows + tiling_.stkResidualRows; ++r) {
            ProcessSegRow(r);
        }
    }

    /** @brief 单行段化处理：本核扫该行一段 → GM 槽位树形归并 → 核0写回（含全部 SyncAll 栅栏） */
    __aicore__ inline void ProcessSegRow(uint64_t r)
    {
        uint64_t segLen = tiling_.stkSegLen;
        uint64_t segStart = blockIdx_ * segLen;
        uint64_t segLenThis = (segStart < sortLen_) ?
                                  ((sortLen_ - segStart < segLen) ? (sortLen_ - segStart) : segLen) :
                                  0;
        uint64_t rowBase = r * sortLen_;
        uint64_t rowOut = r * kValue_;
        LocalTensor<float> curV = candVBuf_.Get<float>();
        LocalTensor<int32_t> curI = candIBuf_.Get<int32_t>();

        // segLenThis == 0 也必须走完全部 SyncAll，否则栅栏计数不一致会挂起；本地候选用哨兵补齐。
        float thr = largest ? StkNegInfValue() : StkNanValue();
        int32_t thrIdx = static_cast<int32_t>(0x7FFFFFFF);
        int32_t nCand = 0;
        if (segLenThis > 0) {
            ScanAndSelect(rowBase, segStart, segLenThis, nCand, thr, thrIdx);
        }
        FinalizeLocal(nCand, thr, thrIdx);

        // 第 0 层：各核把本地候选（恰好 k）写入自己的 GM 槽位
        WriteSlot(blockIdx_, curV, curI);
        SyncAll();

        // 单核归并（arity=0，k≤128）：核 0 按 cap 分批串行合并全部槽位，只留两次 SyncAll。
        if (mergeArity_ == 0U) {
            if (blockIdx_ == 0) {
                const uint32_t kA = StkAlignUp(kValue_, 8u);
                const uint32_t batchSides = (cap_ > kValue_) ? ((cap_ - kValue_) / kA) : 0U;
                uint64_t base = 0;
                while ((batchSides > 0U) && (base + 1U < blockDim_)) {
                    const uint32_t remain = static_cast<uint32_t>(blockDim_ - 1U - base);
                    const uint32_t sides = (batchSides < remain) ? batchSides : remain;
                    MergeSlots(base, 1U, sides + 1U, curV, curI);
                    base += sides;
                }
            }
            SyncAll();
            if (blockIdx_ == 0) {
                WriteRow(rowOut, curV, curI);
            }
            return;
        }

        // 其余层：组内合并最多 arity 个槽位，按段号升序读入（本段 → 兄弟段），同层各组并行。
        for (uint64_t step = 1; step < blockDim_; step *= mergeArity_) {
            const uint64_t span = static_cast<uint64_t>(mergeArity_) * step;
            if ((blockIdx_ % span) == 0) {
                if (MergeSlots(blockIdx_, step, mergeArity_, curV, curI)) {
                    WriteSlot(blockIdx_, curV, curI);
                }
            }
            SyncAll();
        }

        // 末层归并结果（核 0 UB）即全行 TopK，直接写回输出
        if (blockIdx_ == 0) {
            WriteRow(rowOut, curV, curI);
        }
    }

    /**
     * @brief stkCoreMode==1：行内多核，各核扫一段 + GM 槽位树形归并；每层一次 SyncAll，末层核 0 写回
     */
    __aicore__ inline void ProcessSegMode()
    {
        for (uint64_t r = 0; r < batch_; ++r) {
            ProcessSegRow(r);
        }
    }

    /** @brief 把本核恰好 k 个候选写入 workspace 槽位 */
    __aicore__ inline void WriteSlot(uint64_t slot, LocalTensor<float> v, LocalTensor<int32_t> idx)
    {
        const uint64_t off = slot * slotStride_;
        VToMTE3Sync();
        DataCopyPad(wsV_[off], v, DataCopyExtParams{1, static_cast<uint32_t>(kValue_ * sizeof(float)), 0, 0, 0});
        DataCopyPad(wsI_[off], idx, DataCopyExtParams{1, static_cast<uint32_t>(kValue_ * sizeof(int32_t)), 0, 0, 0});
        // 同 WriteRow：槽位搬运读的也是 candV/candI，须等 MTE3 完成，否则下一层归并读到脏数据。
        MTE3ToVSync();
        MTE3ToSSync();
    }

    /** @brief 一轮内合并组内最多 (arity-1) 个兄弟槽位（追加到 [j·kA, j·kA+k)）并收敛到 k；返回是否有兄弟被合并 */
    __aicore__ inline bool MergeSlots(uint64_t base, uint64_t step, uint32_t arity, LocalTensor<float>& curV,
                                      LocalTensor<int32_t>& curI)
    {
        const uint32_t kA = StkAlignUp(kValue_, 8u);
        uint32_t sides = 0;
        for (uint32_t j = 1; j < arity; ++j) {
            const uint64_t sibling = base + static_cast<uint64_t>(j) * step;
            if (sibling >= blockDim_) {
                break;
            }
            const uint64_t off = sibling * slotStride_;
            const uint32_t dst = j * kA;
            DataCopyPad(curV[dst], wsV_[off],
                        DataCopyExtParams{1, static_cast<uint32_t>(kValue_ * sizeof(float)), 0, 0, 0},
                        DataCopyPadExtParams<float>{false, 0, 0, 0});
            DataCopyPad(curI[dst], wsI_[off],
                        DataCopyExtParams{1, static_cast<uint32_t>(kValue_ * sizeof(int32_t)), 0, 0, 0},
                        DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
            ++sides;
        }
        if (sides == 0) {
            return false;
        }
        MTE2ToVSync();
        MTE2ToSSync();
        // 对齐间隙 [j·kA+k, (j+1)·kA)：用「永不入选」哨兵占位，避免被计入候选
        if (kA > kValue_) {
            const float sentinel = largest ? StkNegInfValue() : StkPosInfValue();
            for (uint32_t j = 0; j <= sides; ++j) {
                StkFillScalar(curV, j * kA + kValue_, (j + 1) * kA, sentinel);
                StkFillScalar(curI, j * kA + kValue_, (j + 1) * kA, static_cast<int32_t>(0x7FFFFFFF));
            }
            SToVSync();
        }
        // 各路集合（本段在前、兄弟段按段号升序在后，天然索引升序）归并为前 k
        float thr = largest ? StkNegInfValue() : StkPosInfValue();
        int32_t thrIdx = 0;
        int32_t nCand = static_cast<int32_t>(sides * kA + kValue_);
        ReduceToK(curV, curI, nCand, thr, thrIdx);
        return true;
    }

private:
    TPipe& tpipe_;
    const RadixTopKTilingData& tiling_;

    uint64_t blockIdx_{0};
    uint64_t blockDim_{0};
    uint64_t sortLen_{0};
    uint64_t batch_{0};
    uint32_t kValue_{0};
    uint32_t blockSize_{0};
    uint32_t numStages_{1};
    uint32_t cap_{0};
    uint32_t safeTrigger_{0};
    uint32_t coreMode_{0};
    uint32_t slotStride_{0};
    uint32_t mergeArity_{2};

    GlobalTensor<T> xGm_;
    GlobalTensor<T> valueGm_;
    GlobalTensor<int32_t> indexGm_;
    GlobalTensor<float> wsV_;
    GlobalTensor<int32_t> wsI_;

    TQue<TPosition::VECIN, STK_QUE_DEPTH> inQue_;
    TBuf<TPosition::VECCALC> fp32Buf_;
    TBuf<TPosition::VECCALC> candVBuf_;
    TBuf<TPosition::VECCALC> candIBuf_;
    TBuf<TPosition::VECCALC> scratchBuf_;
    TBuf<TPosition::VECCALC> ltVBuf_;
    TBuf<TPosition::VECCALC> ltIBuf_;
    TBuf<TPosition::VECCALC> maskLBuf_;
    TBuf<TPosition::VECCALC> maskEBuf_;
};

template <typename T, bool largest>
__aicore__ inline void RadixTopKSelectTopK<T, largest>::Process()
{
    if (coreMode_ == 1) {
        ProcessSegMode();
    } else {
        ProcessRowMode();
    }
}

} // namespace RadixTopK

#endif // RADIX_TOPK_SELECT_H
