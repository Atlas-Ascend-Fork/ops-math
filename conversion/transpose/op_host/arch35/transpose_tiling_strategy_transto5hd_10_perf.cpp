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
 * \file transpose_tiling_strategy_transto5hd_10_perf.cpp
 * \brief TRANSTO5HD_10_PERF(10017) 策略类实现
 *
 * 实现 2D perm=[1,0] 全 tile 转置的 tile 几何计算、TilingData 回填与打印。
 */

#include "transpose_tiling_strategy_transto5hd_10_perf.h"

#include <limits>

namespace optiling {
namespace {
static constexpr uint64_t TRANSDATA_B16_BYTES = 2;             ///< 16bit 类型字节数
static constexpr uint64_t TRANSDATA_B32_BYTES = 4;             ///< 32bit 类型字节数
static constexpr int64_t TRANSDATA_TILE = 16;                  ///< TransDataTo5HD 16x16 块
static constexpr int64_t TRANSDATA_UB_RESERVE = 4096;          ///< UB 预留（字节）
static constexpr int64_t TRANSDATA_MAX_M_16BIT = 256;          ///< 16bit 单块行数上限
static constexpr int64_t TRANSDATA_MAX_M_32BIT = 128;          ///< 32bit 单块行数上限
static constexpr int64_t TRANSDATA_MAX_REPEAT = 255;           ///< 单次最大 repeat
static constexpr int64_t TRANSDATA_MIN_BYTES = 512 * 1024;     ///< DAV_3510 数据量门槛
static constexpr int64_t TRANSDATA_MIN_BYTES_LEGACY = 4000000; ///< 其它代际数据量门槛
// TilingData 中 m/n 为 int32, 准入时拒绝超范围 shape, 避免下发时静默截断
static constexpr int64_t TRANSDATA_MAX_DIM_ELEMS = std::numeric_limits<int32_t>::max();
} // namespace

/**
 * @brief 准入检查：本特化模板的全部切分前条件（纯判定、无副作用）
 *
 * 1. 非融合内联（related）场景；
 * 2. 架构为 DAV_3510 / DAV_5102（kernel 仅在这两代实例化）；
 * 3. perm 取值 [1,0]（基类 IsPermValue：同时要求 dim==2。RemoveAxis/MergeAxis 后
 *    identity 已归并为 1 维，因此 dim==2 且 perm=[1,0] 即为真正的二维转置）；
 * 4. dtype 位宽 16bit/32bit（kernel 仅实例化 int16_t/int32_t）；
 * 5. m、n 均 16 对齐且不超过 int32（TransDataTo5HD 按 16x16 块转置，TilingData 为 int32）；
 * 6. 总数据量达到门槛：DAV_3510 为 512KB（实测对 SIMT/gather 交叉点），其它代际 4MB。
 */
bool TransposeTransTo5HD10PerfStrategy::IsSupport() const
{
    if (isRelatedTranspose_) {
        return false;
    }
    if (npuArch_ != Ops::Base::DAV_3510 && npuArch_ != Ops::Base::DAV_5102) {
        return false;
    }
    if (!IsPermValue({1, 0})) {
        return false; // dim==2 且 perm=[1,0]（基类统一 perm 判定）
    }
    const int64_t eleLen = shapeInfo_.eleLenInBytes;
    if (eleLen < static_cast<int64_t>(TRANSDATA_B16_BYTES) || eleLen > static_cast<int64_t>(TRANSDATA_B32_BYTES)) {
        return false;
    }
    const int64_t m = shapeInfo_.reducedInShape[0];
    const int64_t n = shapeInfo_.reducedInShape[1];
    if (m <= 0 || n <= 0 || m > TRANSDATA_MAX_DIM_ELEMS || n > TRANSDATA_MAX_DIM_ELEMS) {
        return false;
    }
    if (m % TRANSDATA_TILE != 0 || n % TRANSDATA_TILE != 0) {
        return false;
    }
    // 用 m*n >= CeilDiv(minBytes, eleLen) 判总量，避免 m*n*eleLen 溢出
    const int64_t minBytes = (npuArch_ == Ops::Base::DAV_3510) ? TRANSDATA_MIN_BYTES : TRANSDATA_MIN_BYTES_LEGACY;
    return m * n >= Ops::Base::CeilDiv(minBytes, eleLen);
}

/**
 * @brief 执行切分（计算 tile 几何）
 */
TilingStatus TransposeTransTo5HD10PerfStrategy::DoExecute() { return CalcTileGeometry(); }

/**
 * @brief 计算 tile 几何（只计算，不写 TilingData、不打印）
 *
 * 准入已在 IsSupport 完成，这里只做 UB 预算相关的几何切分：
 *   - tileM 不超过 m（m<=cap 时保证整行搬出，不把连续输出行切碎），不超过位宽对应的单块上限；
 *   - tileN 按 UB 预算（tileM*(tileN+pad) + 2*tileN*(tileM+pad) 个元素）确定，
 *     避免 m 很小时 tile 退化成 16x128 的 4KB 小块。
 * UB 预算装不下一个 16x16 tile 时返回 FALLBACK。
 */
TilingStatus TransposeTransTo5HD10PerfStrategy::CalcTileGeometry()
{
    m_ = shapeInfo_.reducedInShape[0];
    n_ = shapeInfo_.reducedInShape[1];
    const int64_t eleLen = shapeInfo_.eleLenInBytes;

    int64_t pad = ubBlockSize_ / eleLen;
    int64_t maxTileM = (eleLen == static_cast<int64_t>(TRANSDATA_B16_BYTES)) ? TRANSDATA_MAX_M_16BIT :
                                                                               TRANSDATA_MAX_M_32BIT;
    int64_t tileM = std::min(m_, maxTileM);
    int64_t ubBudgetElems = (ubSize_ - TRANSDATA_UB_RESERVE) / eleLen;
    int64_t maxTileN = (ubBudgetElems - tileM * pad) / (3 * tileM + 2 * pad);
    maxTileN = std::min(maxTileN, TRANSDATA_TILE * TRANSDATA_MAX_REPEAT);
    if (maxTileN < TRANSDATA_TILE) {
        return TilingStatus::FALLBACK; // UB 预算装不下一个 16x16 tile
    }
    // tileM 达到单块行数上限 (m 需要按 cap 行分块) 时, tileN 取 2 的幂次:
    // 此时 tile 已足够大, 2 的幂次行宽对齐收益更高 (实测 m=256/1024 更快);
    // 小 m 场景则取 UB 预算允许的最大 16 倍数, 用更少的 tile 摊薄固定开销。
    int64_t tileN = TRANSDATA_TILE;
    if (tileM == maxTileM) {
        while (tileN * 2 <= maxTileN) {
            tileN *= 2;
        }
    } else {
        tileN = (maxTileN / TRANSDATA_TILE) * TRANSDATA_TILE;
    }
    // 窄列场景 (n < tileN 预算, 如 131072x16): 每 tile 实际只用 n 列, 按 n 收紧
    // tileN, 改用行方向切分。tileM 取能满足 "行块数≈k*coreNum" 的最大值 (k 从小到大),
    // 既放大单 tile 又保证多核负载均衡 (避免行块数落在 coreNum 与 2*coreNum 之间)。
    if (n_ < tileN) {
        tileN = n_;
        // 列宽只有一个 32B block 的整数倍时无需 padding 即可让 16 个并发源地址
        // 落在不同 bank; 此时去掉源侧 padding, tile 在 UB 内保持连续, 搬入可退化为
        // 单次连续 DataCopyPad。
        int64_t blockElems = ubBlockSize_ / eleLen;
        int64_t srcPad = ((tileN / blockElems) % 2 == 1) ? 0 : pad;
        int64_t bestTileM = tileM;
        for (int64_t k = 1; k <= coreNum_; ++k) {
            int64_t cand = Ops::Base::CeilDiv(m_, k * coreNum_);
            cand = Ops::Base::CeilAlign(cand, TRANSDATA_TILE);
            cand = std::max(cand, tileM);
            cand = std::min(cand, m_);
            int64_t inElems = cand * (tileN + srcPad);
            int64_t outElems = tileN * (cand + pad);
            if (inElems + 2 * outElems <= ubBudgetElems) {
                bestTileM = cand;
                break;
            }
        }
        tileM = bestTileM;
    }
    tileM_ = tileM;
    tileN_ = tileN;
    return TilingStatus::SUCCESS;
}

/**
 * @brief 写 TransposeTransTo5HD10PerfTilingData（容量不足 → FAILURE）
 */
bool TransposeTransTo5HD10PerfStrategy::SetTilingData()
{
    TransposeTransTo5HD10PerfTilingData* tilingData = context_->GetTilingData<TransposeTransTo5HD10PerfTilingData>();
    if (tilingData == nullptr) {
        OP_LOGE(context_->GetNodeName(), "get tiling data failed, capacity is not enough!");
        return false;
    }
    *tilingData = TransposeTransTo5HD10PerfTilingData{};
    tilingData->m = static_cast<int32_t>(m_);
    tilingData->n = static_cast<int32_t>(n_);
    tilingData->coreNum = static_cast<int32_t>(coreNum_);
    tilingData->tileM = static_cast<int32_t>(tileM_);
    tilingData->tileN = static_cast<int32_t>(tileN_);
    return true;
}

/**
 * @brief 打印 TilingData
 */
void TransposeTransTo5HD10PerfStrategy::LogTilingData()
{
    OP_LOGI(context_->GetNodeName(),
            "tilingData is m:%ld, n:%ld, coreNum:%ld, tileM:%ld, tileN:%ld, Tiling4Transpose ends.", m_, n_, coreNum_,
            tileM_, tileN_);
}

} // namespace optiling
