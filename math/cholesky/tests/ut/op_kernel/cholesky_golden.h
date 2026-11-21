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
 * \file cholesky_golden.h
 * \brief Cholesky Kernel UT 专用的 CPU 侧 FP64 独立参考实现、SPD 造数与误差度量。
 *
 * 独立性（非循环证据）：本头文件只包含 <cmath>/<cstdint>/<vector>，不 include 任何
 * op_kernel/op_host 头文件，不调用被测 kernel 的任何函数；golden 由本文件内的 FP64
 * 递推独立产生。被测实现是 FP32（A2/A3 逐列逐行、A5 SIMT/RegBase 协同面板），两者
 * 不共享累加精度、不共享运算次序、不共享代码。
 *
 * tril 与 triu 各用一套独立递推（列前进的 Cholesky-Banachiewicz 与行前进的
 * Cholesky-Crout），不由转置互推，因此 kernel 侧的转置类缺陷（例如 A5 tril 的
 * 「先算上三角再原位转置」漏转或转错方向）会被 golden 比对直接鉴别。
 *
 * golden 的输入是 kernel 实际收到的 FP32 数值（原样提升为 FP64），不是造数过程中的
 * FP64 中间量，避免把造数舍入误算成 kernel 误差。
 */

#ifndef CHOLESKY_GOLDEN_H
#define CHOLESKY_GOLDEN_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace cholesky_golden {

/*!
 * 精度判据口径：
 *
 * 绝对门限（FLOAT32）：MERE < 2^-13 且 MARE < 10 * 2^-13，与仓内
 * experimental/math 下已有算子 UT 的 compare 口径一致。本 UT 不做竞品对标，
 * ErrorCount 判据的分母取 max(0, 1)=1。
 *
 * 小值域：|golden| < 2^-14 时相对误差不可判，不计入 maxRelErrNormal，改按
 * 小值域 error 指标 2^-30 与 ErrorCount <= 2 判定；MARE/MERE 的相对误差
 * 分母保护项为 1e-7。
 *
 * 下文 [S1] 指外部《算子精度标准》2.1（不随本仓分发），[S2] 指仓内公开的
 * FLOAT32 绝对门限口径；两者出处与定义不同，不得混用。
 */
// IEEE-754 binary32 round-to-nearest 单位舍入 u = 2^-24
constexpr double FP32_UNIT_ROUNDOFF = 5.9604644775390625e-8;
// IEEE-754 binary64 单位舍入 u64 = 2^-53，只用于 golden 自洽残差的量级界定
constexpr double FP64_UNIT_ROUNDOFF = 1.1102230246251565e-16;
// [S1] §4.5.3 FLOAT32 Small Value Threshold 2^-14：|golden| 低于该值时相对误差不可判，
// 不计入 maxRelErrNormal，改按小值域 ErrorCount 判据处理
constexpr double FP32_SMALL_VALUE = 6.103515625e-5;
// [S1] §4.5.3 FLOAT32 小值域 error 指标 2^-30
constexpr double FP32_SMALL_VALUE_ERR = 9.313225746154785e-10;
// [S1] §4.5.3 ErrorCount_npu / max(ErrorCount_gpu, 1) <= 2；[S1] §4.5 无竞品 => 分母为 1
constexpr uint64_t SMALL_VALUE_ERROR_COUNT_LIMIT = 2;
// [S1] §4.1 MARE/MERE 分母保护项
constexpr double MARE_DENOM_EPS = 1e-7;
// [S2] FLOAT32 绝对门限
constexpr double FP32_MERE_THRESHOLD = 1.220703125e-4; // 2^-13
constexpr double FP32_MARE_THRESHOLD = 1.220703125e-3; // 10 * 2^-13

/*!
 * 容差模型（推导见 test_cholesky.cpp 顶部注释）：
 *   relTol(n) = C_REL * n^2 * u
 *     n 的一次方来自递推深度（主元逐层依赖前一层结果），另一次方来自 Schur 补
 *     A[i][j] - Sigma 的抵消放大：点积的绝对舍入为 O(n*u*sum|terms|)，而 sum|terms|
 *     可比 |Sigma| 大 O(n) 倍。
 *   absTol(n) = C_ABS * n * u * maxAbsA / minPivot
 *     Schur 补的绝对误差 O(n*u*maxAbsA) 经 1/L[j][j] <= 1/minPivot 放大；
 *     maxAbsA/minPivot 即因子元素的最大量级，故 absTol 是「normwise 绝对误差」。
 *   逐元素判据 |actual - golden| <= absTol + relTol * |golden|（allclose 形式）。
 *
 * C_REL / C_ABS 的来源（标定，不是解析上界）：n 的幂次由上述推导固定，两个常数吸收全部
 * 与 n 无关的因子——kernel 以 T(1 / sqrt(pivot)) 计算缩放因子的三次舍入、A5 面板路径逐
 * 面板重写 outGM 的额外 n/PANEL_WIDTH 次舍入、以及 Schur 补抵消的常数项。二者按三平台
 * CPU 仿真实测最坏值标定：n >= 3 时实测最坏 maxRelErrNormal/relTol 与 maxAbsErr/absTol
 * 均 >= 20x 余量；n = 1/2 更紧（最坏 8x / 7x 与 21x / 11x），因为 relTol(1) = 4u、
 * relTol(2) = 16u 已接近单次 sqrt 与单次乘法的 0.5 ulp 固有下限，而 n <= 2 无三角结构可错。
 * 逐条实测值见 test_cholesky.cpp 顶部注释的余量表。
 *
 * 不再以 kappa(A) 有界作为 C_REL / C_ABS 的依据：MakeSpdMatrix 注释里给出的实测 kappa
 * 表明 kappa(A) 既不是 <= 2、也不是与 n 无关的常数（对角型固定用例的 kappa 高达 128），
 * 任何 kappa 论证都无法支撑「与 n 无关的常数」。
 */
constexpr double C_REL = 4.0;
constexpr double C_ABS = 4.0;

inline double RelativeTolerance(uint32_t n)
{
    const double nd = static_cast<double>(n);
    return C_REL * nd * nd * FP32_UNIT_ROUNDOFF;
}

inline double AbsoluteTolerance(uint32_t n, double maxAbsA, double minPivot)
{
    return C_ABS * static_cast<double>(n) * FP32_UNIT_ROUNDOFF * maxAbsA / minPivot;
}

/*!
 * golden 自洽残差 ||A - L*L^T||_F / ||A||_F 的上界，L 为 FP64 参考因子。
 *
 * 该判据只鉴别「参考实现自身是否真的把 A 分解了」（错下标、错三角归属、错算法给 O(1)），
 * 度量的是 FP64 舍入自洽性，**不鉴别被测 kernel**——kernel 输出不参与该残差。
 * 推导：n 层主元递推、每层 O(1) 次 FP64 舍入 => 因子相对误差 eps64 <= n*u64；
 * 代入 ||A - L*L^T||_F/||A||_F <= C*(2*eps64 + eps64^2) <= C_SELF*n*u64。
 * C_SELF = 128 覆盖抵消放大与 FP64 求和次序。三平台实测：最坏绝对值 4.19e-16（n=130），
 * 最紧余量出现在 n=1（2.20e-16 / 1.42e-14 = 1/65），n >= 8 时余量 >= 1/500。
 * 结构性缺陷（错下标/错三角/错算法）给 O(1)，即比该容差大 >= 1e12 倍，鉴别力充分。
 */
constexpr double C_SELF = 128.0;

inline double GoldenSelfConsistencyTolerance(uint32_t n)
{
    return C_SELF * static_cast<double>(n) * FP64_UNIT_ROUNDOFF;
}

// splitmix64：完全由整数运算定义，逐位可复现，不依赖 <random> 的库实现版本
class SplitMix64 {
public:
    explicit SplitMix64(uint64_t seed) : state_(seed) {}
    uint64_t Next()
    {
        state_ += 0x9E3779B97F4A7C15ULL;
        uint64_t z = state_;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    // [0, 1) 内 53 位精度均匀分布
    double Uniform01() { return static_cast<double>(Next() >> 11) / 9007199254740992.0; }
    // [-1, 1)：符号混合，保证 A = M*M^T + lambda*I 的非对角元素同时出现正负值
    double UniformSymmetric() { return Uniform01() * 2.0 - 1.0; }

private:
    uint64_t state_;
};

/*!
 * \brief 可复现的 SPD 造数：A = M*M^T + lambda*I，M 的元素为 U(-1,1)，FP64 累加后落 FP32。
 *
 * M 含正负元素 => A 的非对角元素 A[i][j] = sum_k M[i][k]*M[j][k] 符号混合，
 * 覆盖「SPD 但非对角为负」的合法场景（正式 static finding F-CHOLESKY-VO-005）。
 * 结果严格对称：A[i][j] 与 A[j][i] 的 FP64 求和项序完全相同，乘法可交换 => 逐位相等。
 *
 * 条件数（lambda 取 n，即 UT 实际用法；FP64 eigvalsh 实测，覆盖 UT 全部 n 与 seed）：
 *   成立的解析界：M*M^T >= 0 => lambda_min(A) >= lambda = n（实测 lambda_min 落在
 *     [n, 1.09n]，n=1 时因 U(-1,1) 单样本而偏大）；
 *     lambda_max(A) <= ||M||_F^2 + lambda < n^2 + n => kappa(A) < n + 1。
 *   不成立的界（曾写在 C_REL/C_ABS 注释里，已删除）：lambda_max <= 2n、kappa(A) <= 2。
 *     实测反例：n=32 seed 0x5678 lambda_max=71.44 > 2n=64，kappa=2.2323；
 *     n=8 seed 0x9ABC 批次族 kappa 最大 2.4436；n=130 seed 0xABCDEF lambda_max=294.60 > 260。
 *     UT 实际使用的 17 个 seeded 矩阵中，10 个 lambda_max > 2n、同样 10 个 kappa > 2，
 *     即该界系统性失效，不是边角巧合。
 *   实测范围：本造数族 kappa(A) 属于 [1.00, 2.45]，与 Marchenko-Pastur 边缘
 *     lambda_max(M*M^T) ~ 4n/3（Var[m]=1/3）一致，即 kappa -> (4n/3 + n)/n = 7/3。
 *     但这是概率性结论、不是解析上界，唯一解析上界是上面的 kappa < n + 1（随 n 增长）。
 *   另有三个固定用例不受本造数约束：diagonal_spd_batch2_n8 的第二块 kappa=128、
 *     diagonal_spd_n8 kappa=64、negative_offdiag_n3 kappa=6.14。
 *   结论：kappa(A) 既无 <= 2 的界、也无与 n 无关的界，故容差常数只能按实测标定
 *     （见 C_REL/C_ABS 注释），不得引用 kappa 论证。造数本身未改，覆盖意图不变。
 */
inline void MakeSpdMatrix(std::vector<float>& a, uint32_t n, uint64_t seed, double lambda)
{
    const size_t nn = static_cast<size_t>(n) * n;
    std::vector<double> m(nn);
    SplitMix64 rng(seed);
    for (double& v : m) {
        v = rng.UniformSymmetric();
    }
    a.assign(nn, 0.0f);
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (uint32_t k = 0; k < n; ++k) {
                sum += m[static_cast<size_t>(i) * n + k] * m[static_cast<size_t>(j) * n + k];
            }
            if (i == j) {
                sum += lambda;
            }
            a[static_cast<size_t>(i) * n + j] = static_cast<float>(sum);
        }
    }
}

// 严格对称性前置：造数与 FP32 落地都不得破坏对称，否则 kernel 与 golden 的输入语义不同
inline bool IsExactlySymmetric(const float* a, uint32_t n)
{
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < i; ++j) {
            if (a[static_cast<size_t>(i) * n + j] != a[static_cast<size_t>(j) * n + i]) {
                return false;
            }
        }
    }
    return true;
}

/*!
 * \brief 独立 FP64 下三角参考实现（Cholesky-Banachiewicz，按列 j 前进，列内自 i=j 向下）。
 *
 * L[j][j] = sqrt(A[j][j] - sum_{k<j} L[j][k]^2)
 * L[i][j] = (A[i][j] - sum_{k<j} L[i][k]*L[j][k]) / L[j][j],  i > j
 * 上三角置 0。返回 false 表示某个主元 <= 0（输入非正定），minPivot 输出最小主元。
 */
inline bool GoldenTril(const float* a, uint32_t n, std::vector<double>& l, double* minPivot)
{
    const size_t nn = static_cast<size_t>(n) * n;
    l.assign(nn, 0.0);
    *minPivot = 0.0;
    for (uint32_t j = 0; j < n; ++j) {
        for (uint32_t i = j; i < n; ++i) {
            double sum = static_cast<double>(a[static_cast<size_t>(i) * n + j]);
            for (uint32_t k = 0; k < j; ++k) {
                sum -= l[static_cast<size_t>(i) * n + k] * l[static_cast<size_t>(j) * n + k];
            }
            if (i == j) {
                if (!(sum > 0.0) || std::isinf(sum)) {
                    *minPivot = sum;
                    return false;
                }
                const double pivot = std::sqrt(sum);
                l[static_cast<size_t>(j) * n + j] = pivot;
                if (*minPivot == 0.0 || pivot < *minPivot) {
                    *minPivot = pivot;
                }
            } else {
                l[static_cast<size_t>(i) * n + j] = sum / l[static_cast<size_t>(j) * n + j];
            }
        }
    }
    return true;
}

/*!
 * \brief 独立 FP64 上三角参考实现（Cholesky-Crout，按行 i 前进，行内自 j=i 向右）。
 *
 * U[i][i] = sqrt(A[i][i] - sum_{k<i} U[k][i]^2)
 * U[i][j] = (A[i][j] - sum_{k<i} U[k][i]*U[k][j]) / U[i][i],  j > i
 * 下三角置 0。与 GoldenTril 的循环嵌套次序、归约下标角色均不同，非转置推导。
 */
inline bool GoldenTriu(const float* a, uint32_t n, std::vector<double>& u, double* minPivot)
{
    const size_t nn = static_cast<size_t>(n) * n;
    u.assign(nn, 0.0);
    *minPivot = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = i; j < n; ++j) {
            double sum = static_cast<double>(a[static_cast<size_t>(i) * n + j]);
            for (uint32_t k = 0; k < i; ++k) {
                sum -= u[static_cast<size_t>(k) * n + i] * u[static_cast<size_t>(k) * n + j];
            }
            if (i == j) {
                if (!(sum > 0.0) || std::isinf(sum)) {
                    *minPivot = sum;
                    return false;
                }
                const double pivot = std::sqrt(sum);
                u[static_cast<size_t>(i) * n + i] = pivot;
                if (*minPivot == 0.0 || pivot < *minPivot) {
                    *minPivot = pivot;
                }
            } else {
                u[static_cast<size_t>(i) * n + j] = sum / u[static_cast<size_t>(i) * n + i];
            }
        }
    }
    return true;
}

inline bool GoldenFactor(const float* a, uint32_t n, bool upper, std::vector<double>& factor, double* minPivot)
{
    return upper ? GoldenTriu(a, n, factor, minPivot) : GoldenTril(a, n, factor, minPivot);
}

// trace(A) = ||L||_F^2（因 A = L*L^T），用作绝对容差的天然量纲
inline double TraceA(const float* a, uint32_t n)
{
    double trace = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        trace += static_cast<double>(a[static_cast<size_t>(i) * n + i]);
    }
    return trace;
}

struct GoldenReport {
    bool inputSpd = false;
    double minPivot = 0.0;
    double traceA = 0.0;
    double maxAbsA = 0.0;
    double maxAbsGolden = 0.0;
    uint64_t activeCount = 0;
    uint64_t smallValueCount = 0;
    // [S1] §4.5.3 ErrorCount：|golden| < 2^-14 且 |actual-golden| > 2^-30 的活动三角元素数
    uint64_t smallValueErrorCount = 0;
    double maxAbsErr = 0.0;
    // |golden| >= 2^-14 的元素上的最大相对误差（分母 |golden|，无保护项）
    double maxRelErrNormal = 0.0;
    // [S1] §4.1 定义：max/avg |actual-golden| / (|golden| + 1e-7)
    // mare 覆盖全部活动三角元素，含 [S1] §4.5.3 小值域；mareNormal 只覆盖正常值域，
    // 是 [S2] 绝对门限的可断言口径（理由见 test_cholesky.cpp 顶部注释 (2)）
    double mare = 0.0;
    double mareNormal = 0.0;
    double mere = 0.0;
    double rmse = 0.0;
    /*!
     * 被测输出重构残差 R = A - Lhat*Lhat^T（tril）或 A - Uhat^T*Uhat（triu）。
     * Lhat/Uhat 只取 kernel 输出的活动三角：求和下标恒为 k <= min(i,j)，落在活动三角内，
     * 补三角（kernel 不写、由框架 memset 保证）不参与，故其初值不污染残差。
     * 该判据使用被测输出，能鉴别 kernel 的转置/错三角/错算法/数值错误。
     */
    double residMaxAbs = 0.0;
    double residFroRel = 0.0;
    // 推导上界，见 Evaluate 内注释
    double residFroAbsTol = 0.0;
    double residFroRelTol = 0.0;
    /*!
     * golden 自洽残差 ||A - L*L^T||_F/||A||_F，L 为 FP64 参考因子。
     * kernel 输出不参与，只鉴别参考实现自身是否真的分解了 A，**不是 kernel 正确性判据**。
     */
    double goldenSelfFroRel = 0.0;
    // 逐元素判据 |a-g| <= absTol + relTol*|g| 的失败个数
    uint64_t elementFailures = 0;
    double worstAbsErr = 0.0;
    double worstGolden = 0.0;
    double worstLimit = 0.0;
    uint32_t worstRow = 0;
    uint32_t worstCol = 0;
};

/*!
 * \brief 对单个矩阵的 kernel 输出做 golden 比对 + 两套残差，全部在 FP64 下累加。
 *
 * 两套残差各自鉴别什么（不可互相替代）：
 *   residFroRel / residMaxAbs —— 用**被测输出**构造，鉴别 kernel；上界推导：
 *     记 L 为 FP64 参考因子、Lhat = L + E 为被测输出（活动三角外均为 0），则
 *       A - Lhat*Lhat^T = -(L*E^T + E*L^T + E*E^T)
 *       ||A - Lhat*Lhat^T||_F <= 2*||L||_2*||E||_F + ||E||_F^2
 *     由逐元素契约 |E_ij| <= absTol + relTol*|L_ij| 与 Minkowski 不等式：
 *       ||E||_F <= relTol*||L||_F + sqrt(m)*absTol,  m = n(n+1)/2
 *     ||L||_2^2 = lambda_max(L*L^T) = lambda_max(A) <= min(||L||_F^2, max_i sum_j |A_ij|)
 *       （前者因 A 半正定 => lambda_max <= trace；后者为对称阵的行和范数界）
 *     相对化用实测 ||A||_F（不再用松的 ||A||_F >= ||L||_F^2/sqrt(n)）。
 *     该上界由 **FP32 kernel 输出**的误差契约导出，量级远大于 FP64 自洽残差。三平台实测
 *     上界值：n=1 9.5e-7、n=8 5.1e-5、n=32 8.4e-4、n=100 9.3e-3、n=130 1.6e-2；
 *     对应实测残差恒在 5.7e-8 ~ 7.9e-8（余量 1/17(n=1) ~ 1/2.1e5(n=130)），
 *     而结构性缺陷给 O(1)：实测错误转置 0.209（超上界 4074 倍）、错误算法 0.0201（391 倍）。
 *     upper 情形同界成立：A = U^T*U => ||U||_2 = sqrt(lambda_max(A))、||U||_F^2 = trace(A)。
 *   goldenSelfFroRel —— 用 **FP64 golden** 构造，只鉴别参考实现自身，对任何 kernel 输出
 *     恒为 FP64 舍入量级，因此**不得**当作 kernel 正确性判据（上界见
 *     GoldenSelfConsistencyTolerance）。保留它的价值是：golden 递推若写错下标/错三角，
 *     逐元素比对会与被测实现「一致地错」，而自洽残差给 O(1)。
 */
inline GoldenReport Evaluate(const float* actual, const float* a, const std::vector<double>& golden, uint32_t n,
                             bool upper, double relTol, double absTol)
{
    GoldenReport r;
    r.traceA = TraceA(a, n);
    double sumSq = 0.0;
    double sumRel = 0.0;
    double froA = 0.0;
    double froL = 0.0;
    double maxRowSumAbsA = 0.0;
    double froResid = 0.0;
    double froSelf = 0.0;

    for (uint32_t i = 0; i < n; ++i) {
        double rowSumAbsA = 0.0;
        for (uint32_t j = 0; j < n; ++j) {
            const double av = static_cast<double>(a[static_cast<size_t>(i) * n + j]);
            const double absAv = std::abs(av);
            r.maxAbsA = std::max(r.maxAbsA, absAv);
            rowSumAbsA += absAv;
            froA += av * av;

            const uint32_t kLimit = i < j ? i : j;
            double reconActual = 0.0;
            double reconGolden = 0.0;
            for (uint32_t k = 0; k <= kLimit; ++k) {
                if (upper) {
                    reconActual += static_cast<double>(actual[static_cast<size_t>(k) * n + i]) *
                                   static_cast<double>(actual[static_cast<size_t>(k) * n + j]);
                    reconGolden += golden[static_cast<size_t>(k) * n + i] * golden[static_cast<size_t>(k) * n + j];
                } else {
                    reconActual += static_cast<double>(actual[static_cast<size_t>(i) * n + k]) *
                                   static_cast<double>(actual[static_cast<size_t>(j) * n + k]);
                    reconGolden += golden[static_cast<size_t>(i) * n + k] * golden[static_cast<size_t>(j) * n + k];
                }
            }
            const double resid = av - reconActual;
            froResid += resid * resid;
            r.residMaxAbs = std::max(r.residMaxAbs, std::abs(resid));
            const double selfResid = av - reconGolden;
            froSelf += selfResid * selfResid;

            const bool active = upper ? (j >= i) : (j <= i);
            if (!active) {
                continue;
            }
            const double gold = golden[static_cast<size_t>(i) * n + j];
            r.maxAbsGolden = std::max(r.maxAbsGolden, std::abs(gold));
            froL += gold * gold;
            const double act = static_cast<double>(actual[static_cast<size_t>(i) * n + j]);
            const double absErr = std::abs(act - gold);
            const double relErr = absErr / (std::abs(gold) + MARE_DENOM_EPS);
            ++r.activeCount;
            sumSq += absErr * absErr;
            sumRel += relErr;
            r.mare = std::max(r.mare, relErr);
            r.maxAbsErr = std::max(r.maxAbsErr, absErr);
            if (std::abs(gold) >= FP32_SMALL_VALUE) {
                r.maxRelErrNormal = std::max(r.maxRelErrNormal, absErr / std::abs(gold));
                r.mareNormal = std::max(r.mareNormal, relErr);
            } else {
                ++r.smallValueCount;
                if (absErr > FP32_SMALL_VALUE_ERR) {
                    ++r.smallValueErrorCount;
                }
            }
            const double limit = absTol + relTol * std::abs(gold);
            if (!(absErr <= limit)) {
                ++r.elementFailures;
                if (absErr > r.worstAbsErr) {
                    r.worstAbsErr = absErr;
                    r.worstGolden = gold;
                    r.worstLimit = limit;
                    r.worstRow = i;
                    r.worstCol = j;
                }
            }
        }
        maxRowSumAbsA = std::max(maxRowSumAbsA, rowSumAbsA);
    }
    r.mere = r.activeCount > 0 ? sumRel / static_cast<double>(r.activeCount) : 0.0;
    r.rmse = r.activeCount > 0 ? std::sqrt(sumSq / static_cast<double>(r.activeCount)) : 0.0;
    r.residFroRel = froA > 0.0 ? std::sqrt(froResid / froA) : 0.0;
    r.goldenSelfFroRel = froA > 0.0 ? std::sqrt(froSelf / froA) : 0.0;

    const double froLNorm = std::sqrt(froL);
    const double l2Bound = std::sqrt(std::min(froL, maxRowSumAbsA));
    const double errFroBound = relTol * froLNorm + std::sqrt(static_cast<double>(r.activeCount)) * absTol;
    r.residFroAbsTol = 2.0 * l2Bound * errFroBound + errFroBound * errFroBound;
    r.residFroRelTol = froA > 0.0 ? r.residFroAbsTol / std::sqrt(froA) : 0.0;
    return r;
}

} // namespace cholesky_golden

#endif // CHOLESKY_GOLDEN_H
