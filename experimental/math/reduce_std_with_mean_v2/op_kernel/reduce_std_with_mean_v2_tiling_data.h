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
 * \file reduce_std_with_mean_v2_tiling_data.h
 * \brief TilingData struct for Welford one-pass kernel
 *
 * Data layout: reduce-dim elements MUST be contiguous (reduce dims are
 * innermost after transpose in the L2 API layer). No srcStride needed.
 */

#ifndef _REDUCE_STD_WITH_MEAN_WELFORD_TILING_DATA_H_
#define _REDUCE_STD_WITH_MEAN_WELFORD_TILING_DATA_H_

struct ReduceStdWithMeanV2TilingData {
    int64_t totalNonReduce = 0; // total non-reduce elements across all cores
    int64_t reduceLength = 0;   // reduce dimension length
    int64_t blockFactor = 0;    // max non-reduce elements per core
    int64_t ubLength = 0;       // UB tile length for reduce dimension (serial path)
    int64_t correction = 0;     // Bessel correction (0 or 1)
    float eps = 0.0f;           // numerical stability epsilon
    bool invert = false;        // if true, output 1/std instead of std
    bool vectorize = false;     // use the M-vectorized path (ProcessVectorized)
    int64_t ubM = 0;            // rows per vectorized batch (M_tile)
    // two-stage cross-core reduction (PARALLEL_PLAN.md two-stage):
    //   stage 1 (Partial): one core per R segment; each core reduces every row's
    //     segment [k*partialSeg, k*partialSeg+segLen) and writes (n, mean, m2)
    //     triples to the partial output [M, K, 3].
    //   stage 2 (Merge): Chan-merges the K triples of each row -> std/mean.
    int64_t stage = 0;      // 0=default single-kernel, 1=Partial, 2=Merge
    int64_t partialK = 0;   // two-stage: number of R segments (= Partial blockDim)
    int64_t partialSeg = 0; // two-stage: nominal segment length ceil(R/K); last one short
};

#endif // _REDUCE_STD_WITH_MEAN_WELFORD_TILING_DATA_H_
