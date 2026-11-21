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
 * \file test_cholesky.cpp
 * \brief Cholesky Kernel UT（CPU 仿真）。
 *
 * 证据边界：本文件全部结论均为 ASCENDC_CPU_DEBUG CPU 仿真结果，不是设备证据。
 * 结构上无法在 CPU 仿真覆盖、必须由设备 ST 承担的项：
 *   - GetUserWorkspace() 偏移类缺陷：CPU 仿真分支原样返回入参，漏偏移不可暴露；
 *   - 多核 SyncAll 时序/死锁：CPU 仿真按序执行，不建模屏障到达时序；
 *   - 非法主元的设备异常返回码和同步行为必须由设备 ST 验证；CPU 仿真只核对断言诊断。
 *   - S_V / V_S 等 HardEvent 流水同步缺失：CPU 仿真按序执行，
 *     同步缺陷不产生数值差异，本文件只覆盖该路径的数值正确性，不覆盖同步本身；
 *   - RegBase 整寄存器（256B）无掩码越读：CPU 仿真不做物理寄存器装载，
 *     越读不可见，本文件只覆盖 count==1 路径的数值正确性；
 *   - blockSize < matSizeN 的多分块循环：host tiling 仅在 matSizeN > 256 时产生该配置，
 *     CPU 仿真下 O(n^3) 逐元素同步开销使 n > 256 不可承受。
 *
 * 精度判据的口径与阈值定义见 cholesky_golden.h 顶部注释。
 *
 * 判据分四组，各自鉴别的东西不同，不可互相替代：
 *
 * (1) 逐元素前向误差（n 放缩，解析推导 + 常数标定）。FP32，u = 2^-24 = 5.96e-8：
 *       |actual - golden| <= absTol + relTol * |golden|
 *       relTol(n) = 4 * n^2 * u       n=8 -> 1.5e-5, n=32 -> 2.4e-4, n=130 -> 4.0e-3
 *       absTol(n) = 4 * n * u * max|A| / minPivot
 *     推导见 cholesky_golden.h 的 C_REL/C_ABS 注释。n^2 来自递推深度 O(n) 与 Schur 补
 *     抵消放大 O(n)，是 Cholesky 前向误差的固有增长，不是为放宽而引入的自由参数。
 *
 * (2) 仓内公开绝对门限 [S2]（与 n 无关）：MERE < 2^-13 = 1.22e-4，MARE < 10*2^-13 = 1.22e-3。
 *     relTol 与 2^-13 的关系：二者度量对象不同，不构成「relTol 放宽了仓内标准」。
 *       - 2^-13 在 [S2] 中是 **MERE（平均相对误差）** 的门限，本文件对 MERE 同时断言
 *         relTol 与 2^-13，取更严者生效：n <= 16 时 relTol < 2^-13（n=8: 1.5e-5），
 *         n >= 32 时 2^-13 生效（relTol(32) = 2.4e-4 > 1.22e-4）。实测 MERE 最坏
 *         3.87e-7（n=100），比 2^-13 低 316 倍，仓内门限被真实满足而非绕过。
 *       - relTol 是**单元素最坏情况**的 normwise 前向误差界，[S2] 未对该量设门限；
 *         若强行令 relTol <= 2^-13，等价于要求 n=130 的单元素最坏相对误差不超过
 *         1.22e-4 = 2048*u，而 O(n)-深度递推的可达界为 O(n^2*u) = 6.8e4*u，会把
 *         判据变成必然失败的假门限。故保留 n 放缩，并由 (2) 的 MERE/MARE 承接仓内口径。
 *     MARE 此前只计算并打印、从未断言，现按 [S2] 补绝对门限断言。它是绝对门限而非仅诊断量：
 *     [S2] 不依赖竞品，本 UT 无 GPU 对标（[S1] §4.5 允许用自构造 CPU 实现作标杆），
 *     故 [S1] §4.5.1 的比值判据不可用，绝对门限是唯一可判定口径。
 *
 *     MARE 的取值域必须是**正常值域**（|golden| >= 2^-14），即断言 mareNormal 而非 mare：
 *       - [S1] §4.5.3 明文规定「当算子输出结果为接近 0 的极小值时，相对误差计算可能不稳定，
 *         需要使用小值域通过标准」，小值域改由 ErrorCount 判据承载（见 (3)）。把 [S2] 的
 *         相对门限套到小值域上，与 [S1] §4.5.3 直接冲突。
 *       - 全域 mare 不存在有意义的绝对门限：小值域元素的分母被 [S1] §4.1 的保护项钳到
 *         >= 1e-7，而分子只受 absTol 约束，故 mare <= absTol/1e-7（n=100 时为 3.0e3），
 *         该上界无鉴别力；任何按 mare 设的常数门限都只能是拍脑袋值。
 *       - 实测支撑（ascend950 golden_panel_batchcoop_triu_n100 batch 0，唯一小值域元素
 *         U[67,71]，golden = -4.0186e-6，|golden|/2^-14 = 0.066）：
 *           被测 kernel            |err| = 5.80e-9   MARE 项 = 1.41e-3
 *           FP32 竞品（numpy Crout 同序直算）|err| = 2.48e-8   MARE 项 = 6.02e-3
 *           FP32 可达最优（round(FP64 golden)）|err| = 7.0e-14  MARE 项 = 1.70e-8
 *         被测 kernel 在该元素上比 FP32 竞品**精确 4.3 倍**；竞品全域 MARE = 6.02e-3，
 *         同样越过 10*2^-13 且差 4.9 倍。即「全域 MARE <= 10*2^-13」对任何基于 Schur 补
 *         抵消的 FP32 Cholesky 都不可达（该元素由 O(10) 量级的项相消到 4e-6，损失约 6 位
 *         十进制精度，与实现无关），只有逐元素独立、无相消的运算才可能满足。
 *       - 正常值域口径不因此变松：n=100 时 mareNormal 实测 2.51e-5，比 10*2^-13 严 48 倍；
 *         且 relTol(100) = 2.38e-3 > 10*2^-13，故 [S2] 门限在 n >= 71 时是更严的那一条，
 *         真实收紧了判据。同一输入下 FP32 竞品的正常值域 MARE = 5.37e-5，与被测同量级。
 *     MERE 不需要该排除：它是均值，单个小值域元素被 1/m 稀释（n=100 时 m=5050），
 *     实测 3.87e-7 比 2^-13 低 316 倍，[S1] §4.1 的 MERE 定义也无排除项，故按全域断言。
 *
 * (3) 小值域 [S1] §4.5.3：|golden| < 2^-14 的活动三角元素不计入 maxRelErrNormal，改按
 *       ErrorCount = #{|golden| < 2^-14 且 |actual-golden| > 2^-30}
 *       ErrorCount / max(ErrorCount_gpu, 1) <= 2；无竞品 => 分母为 1 => ErrorCount <= 2。
 *
 * (4) 两套残差（推导见 cholesky_golden.h 的 Evaluate 注释）：
 *     (4a) 被测输出重构残差 R = A - factor(actual)*factor(actual)^T（upper=false）或
 *         A - factor(actual)^T*factor(actual)（upper=true），factor(actual) 只取 kernel
 *         输出的活动三角。判据 ||R||_F/||A||_F <= residFroRelTol，上界由 (1) 的 FP32 误差
 *         契约经 ||A - Lhat*Lhat^T||_F <= 2*||L||_2*||E||_F + ||E||_F^2 导出，
 *         **使用被测输出**，能鉴别 kernel 的转置/错三角/错算法/数值错误。
 *         max|R| 只报告不断言：max|R| <= ||R||_F，该判据被上式严格蕴含，单独写出来只会
 *         成为一条永不独立失败的假断言（与本次修复的缺陷同类）。
 *     (4b) golden 自洽残差 ||A - L*L^T||_F/||A||_F <= GoldenSelfConsistencyTolerance(n)。
 *         kernel 输出不参与，只鉴别 FP64 参考实现自身是否真的分解了 A（golden 递推写错
 *         下标/错三角时，(1) 会与被测实现「一致地错」而失效，该判据给 O(1)）。
 *         它**不是** kernel 正确性判据——对任何 kernel 输出都恒为 FP64 舍入量级。
 *
 * 鉴别力余量（三平台 CPU 仿真实测最坏值 / 对应容差；1/X 表示还有 X 倍余量，取同一 n 下
 * 全部用例与三平台的最坏者）：
 *   n     maxRelErrNormal/relTol  maxAbsErr/absTol  MAREnorm/(10*2^-13)  MERE/2^-13
 *   1     1/8                     1/7               1/42762              1/4276
 *   2     1/21                    1/11              1/26497              1/3794
 *   3     1/64                    1/34              1/36658              1/21995
 *   8     1/70                    1/28              1/5627               1/2776
 *   32    1/22                    1/105             1/109                1/1223
 *   100   1/95                    1/250             1/49                 1/316
 *   130   1/92                    1/393             1/28                 1/1004
 *   residFroRel/residFroRelTol    1/17(n=1) 1/51(n=2) 1/646(n=8) 1/1.2e4(n=32)
 *                                 1/1.2e5(n=100) 1/2.1e5(n=130)
 *   goldenSelfFroRel/Tol          1/65(n=1) 1/210(n=2) 1/564(n=8) 1/4410(n=130)
 *   smallValueErrorCount          最坏 1（限 2，出现在 n=100 的唯一小值域元素）
 *   n=1/2 余量最小是固有的：relTol(1)=4u、relTol(2)=16u 已接近单次 sqrt 与单次乘法的
 *   0.5 ulp 下限，absTol 同理由 O(n*u) 收缩；而 n<=2 无（或仅有 1 个）非对角元素，
 *   无三角结构可错。结构性鉴别力由 n >= 3 的用例承载（两项余量均 >= 1/34）。
 * 鉴别力对照实验（一次性验证代码，不入库；n=8 tril，ascend910b 与 ascend950 结果逐位相同）：
 * 对 kernel 输出注入扰动后，比较 (4a) 新判据与修复前的旧判据（残差全部由 golden 构造）。
 * 旧判据实测值在全部 6 个场景下恒为 2.01609e-16、恒 PASS，与其容差 8.63e-5 无关，
 * 即旧判据与被测输出完全无关（假断言）。新判据容差 5.14e-5：
 *   S0 未扰动                        7.95e-8  PASS（余量 1/646）
 *   S1 单个活动元素 += 10*absTol      1.18e-5  PASS（由 (1) 拦截：elementFailures=1）
 *   S2 单个活动元素 *= 1.001          1.05e-5  PASS（由 (1) 拦截：elementFailures=1）
 *   S3 错误转置（tril <- golden^T）   0.209    FAIL（超容差 4074 倍）
 *   S4 错误算法（活动三角 *= 1.01）   0.0201   FAIL（超容差 391 倍）
 *   S5 主元 *= (1 + 100*relTol)       0.00294  FAIL（超容差 57 倍）
 * S1/S2 说明 (4a) 与 (1) 互补而非冗余：单元素扰动对 64 项 Frobenius 范数贡献小，由 (1) 拦截；
 * 结构性缺陷（S3/S4/S5）由 (4a) 以 1e1~1e3 倍裕度拦截，旧判据对三者全部漏判。
 *
 * 补三角断言单独校验 kernel 是否写入契约区域之外。
 */
#include <array>
#include <vector>
#include <gtest/gtest.h>

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "cholesky_golden.h"

using namespace std;

// 与 kernel 入口（op_kernel/arch35/cholesky_arch35.cpp / op_kernel/arch22/cholesky_arch22.cpp）保持同一模板签名：
// 第一参数 upper，第二参数 cube（仅 arch35 生效；CPU 仿真用例取默认 false）
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
template <bool upper, bool cube = false>
#else
template <bool upper>
#endif
__global__ __aicore__ void cholesky(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling);

// 针对构建生成/两侧共享 TilingData ABI 的机器可核对断言：CANN 字段宏在 uint64 字段
// matrixNumCount 前插入 4 字节对齐占位（matrixNumCountPH），实际序列化布局为 24 字节
static_assert(sizeof(CholeskyTilingData) == 24, "Cholesky tiling ABI must remain 24 bytes");
static_assert(offsetof(CholeskyTilingData, matSizeN) == 0, "matSizeN must be at offset 0");
static_assert(offsetof(CholeskyTilingData, matrixNumCount) == 8, "matrixNumCount must be at offset 8");
static_assert(offsetof(CholeskyTilingData, blockSize) == 16, "blockSize must be at offset 16");
static_assert(offsetof(CholeskyTilingData, blockNum) == 20, "blockNum must be at offset 20");

namespace {
// 可辨识的 out 初值：非 0、非 NaN、非任何合法因子取值，用于判定 kernel 是否写过某元素
constexpr float OUT_SENTINEL = -7.5f;
// 与仓内 host UT（tests/ut/op_host/arch{22,35}）的 CholeskyCompileInfo{48} 一致
constexpr uint32_t PLATFORM_CORE_NUM = 48;
constexpr uint32_t MAX_BLOCK_SIZE = 256;
constexpr uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
constexpr uint32_t WS_USER_ALIGN = 32U;
#if __NPU_ARCH__ == 3510
// 必须与 op_kernel/arch35/cholesky_arch35.h 及 op_host/arch35/cholesky_tiling_arch35.cpp 的同名常量一致
constexpr uint32_t BATCH_COOP_MIN_M = 96;
constexpr uint32_t SINGLE_MATRIX_SINGLE_CORE_MAX = 4;
#endif

struct LaunchConfig {
    uint32_t blockDim = 1;
    uint32_t blockSize = 1;
    uint32_t blockNum = 1;
    size_t workspaceBytes = WS_SYS_SIZE;
};

// 镜像 host tiling 的 needCoreNum/blockSize/blockNum/workspace 公式，使 UT 的 launch
// 配置与设备真实下发一致（arch22: op_host/arch22/cholesky_tiling_arch22.cpp:138-161；
// arch35: op_host/arch35/cholesky_tiling_arch35.cpp:148-179）
inline LaunchConfig MakeLaunchConfig(uint32_t n, uint64_t batch)
{
    LaunchConfig cfg;
    cfg.blockSize = n <= MAX_BLOCK_SIZE ? n : MAX_BLOCK_SIZE;
    cfg.blockNum = n <= MAX_BLOCK_SIZE ? 1U : (n + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
    const uint64_t perMatrixCores = batch == 1 ? cfg.blockNum : batch;
#if __NPU_ARCH__ == 3510
    const bool cooperative = batch == 1 || n > BATCH_COOP_MIN_M;
    cfg.blockDim = cooperative ? ((batch == 1 && n <= SINGLE_MATRIX_SINGLE_CORE_MAX) ? 1U : PLATFORM_CORE_NUM) :
                                 static_cast<uint32_t>(std::min<uint64_t>(PLATFORM_CORE_NUM, perMatrixCores));
    const uint64_t voteBytes = static_cast<uint64_t>(cfg.blockDim) * sizeof(float);
    const uint64_t userBytes = (voteBytes + WS_USER_ALIGN - 1) / WS_USER_ALIGN * WS_USER_ALIGN;
    cfg.workspaceBytes = static_cast<size_t>(WS_SYS_SIZE + userBytes);
#else
    cfg.blockDim = static_cast<uint32_t>(std::min<uint64_t>(PLATFORM_CORE_NUM, perMatrixCores));
    cfg.workspaceBytes = WS_SYS_SIZE;
#endif
    return cfg;
}

inline void SetTilingFields(uint8_t* tiling, uint32_t n, uint64_t batch, const LaunchConfig& cfg)
{
    auto* tilingData = reinterpret_cast<CholeskyTilingData*>(tiling);
    tilingData->matSizeN = n;
    tilingData->matrixNumCount = batch;
    tilingData->blockSize = cfg.blockSize;
    tilingData->blockNum = cfg.blockNum;
}

// 设备上由框架按 TilingKey 选中构建系统实例化的 cholesky<upper>；CPU 仿真下由本函数
// 取同一实例的地址，两条路径的入参与执行体完全一致
inline void RunCholeskyKernel(bool upper, uint32_t blockDim, uint8_t* selfGM, uint8_t* outGM, uint8_t* workspace,
                              uint8_t* tiling)
{
    if (upper) {
        ICPU_RUN_KF(cholesky<true>, blockDim, selfGM, outGM, workspace, tiling);
    } else {
        ICPU_RUN_KF(cholesky<false>, blockDim, selfGM, outGM, workspace, tiling);
    }
}

inline double MaxAbsA(const float* a, uint32_t n)
{
    double m = 0.0;
    for (size_t i = 0; i < static_cast<size_t>(n) * n; ++i) {
        m = std::max(m, std::abs(static_cast<double>(a[i])));
    }
    return m;
}

// SPD 前置：输入必须真的对称正定，否则「输出无意义但用例仍 PASS」。
// 判据为 FP64 下两套独立递推均成功且全部主元 > 0（Sylvester 判据的等价形式）。
inline void CheckSpdPrecondition(const float* a, uint32_t n, const char* tag, uint64_t batchIndex)
{
    ASSERT_TRUE(cholesky_golden::IsExactlySymmetric(a, n))
        << tag << " batch " << batchIndex << ": input is not exactly symmetric, SPD precondition invalid";
    std::vector<double> factor;
    double minPivot = 0.0;
    ASSERT_TRUE(cholesky_golden::GoldenTril(a, n, factor, &minPivot))
        << tag << " batch " << batchIndex << ": FP64 tril failed, input is not SPD (pivot=" << minPivot << ")";
    ASSERT_GT(minPivot, 0.0) << tag << " batch " << batchIndex << ": non-positive FP64 pivot (tril)";
    ASSERT_TRUE(cholesky_golden::GoldenTriu(a, n, factor, &minPivot))
        << tag << " batch " << batchIndex << ": FP64 triu failed, input is not SPD (pivot=" << minPivot << ")";
    ASSERT_GT(minPivot, 0.0) << tag << " batch " << batchIndex << ": non-positive FP64 pivot (triu)";
}

struct RunSummary {
    uint64_t complementZero = 0;
    uint64_t complementUntouched = 0;
    uint64_t complementViolation = 0;
    uint64_t negativeOffDiag = 0;
    double worstRelErr = 0.0;
    double worstAbsErr = 0.0;
    double worstMare = 0.0;
    double worstMareNormal = 0.0;
    double worstMere = 0.0;
    double worstRmse = 0.0;
    double worstResidFroRel = 0.0;
    double worstResidMaxAbs = 0.0;
    double tightestResidFroRelTol = std::numeric_limits<double>::max();
    double tightestResidFroAbsTol = std::numeric_limits<double>::max();
    double worstGoldenSelfFroRel = 0.0;
    uint64_t worstSmallValueErrorCount = 0;
    uint64_t smallValueCount = 0;
    double minRelTol = std::numeric_limits<double>::max();
    double minAbsTol = std::numeric_limits<double>::max();
    double minPivot = std::numeric_limits<double>::max();
};

inline void RunAndVerify(const char* tag, const std::vector<std::vector<float>>& mats, uint32_t n, uint64_t batch,
                         bool upper, bool requireNegativeOffDiag = true)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    const size_t matBytes = static_cast<size_t>(n) * n * sizeof(float);
    const size_t totalBytes = matBytes * batch;
    for (uint64_t b = 0; b < batch; ++b) {
        ASSERT_NO_FATAL_FAILURE(CheckSpdPrecondition(mats[b].data(), n, tag, b));
    }

    const LaunchConfig cfg = MakeLaunchConfig(n, batch);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(cfg.workspaceBytes);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(CholeskyTilingData));
    uint8_t* selfGM = (uint8_t*)AscendC::GmAlloc(totalBytes);
    uint8_t* outGM = (uint8_t*)AscendC::GmAlloc(totalBytes);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);
    ASSERT_NE(selfGM, nullptr);
    ASSERT_NE(outGM, nullptr);

    for (uint64_t b = 0; b < batch; ++b) {
        memcpy(selfGM + b * matBytes, mats[b].data(), matBytes);
    }
    auto* out = reinterpret_cast<float*>(outGM);
    for (size_t i = 0; i < batch * static_cast<size_t>(n) * n; ++i) {
        out[i] = OUT_SENTINEL;
    }

    SetTilingFields(tiling, n, batch, cfg);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    RunCholeskyKernel(upper, cfg.blockDim, selfGM, outGM, workspace, tiling);

    RunSummary s;
    for (uint64_t b = 0; b < batch; ++b) {
        const float* a = mats[b].data();
        const float* actual = out + b * static_cast<size_t>(n) * n;
        for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t j = 0; j < n; ++j) {
                if (i != j && a[static_cast<size_t>(i) * n + j] < 0.0f) {
                    ++s.negativeOffDiag;
                }
            }
        }

        std::vector<double> golden;
        double minPivot = 0.0;
        ASSERT_TRUE(cholesky_golden::GoldenFactor(a, n, upper, golden, &minPivot))
            << tag << " batch " << b << ": golden factorization failed";
        s.minPivot = std::min(s.minPivot, minPivot);

        const double maxAbsA = MaxAbsA(a, n);
        const double relTol = cholesky_golden::RelativeTolerance(n);
        const double absTol = cholesky_golden::AbsoluteTolerance(n, maxAbsA, minPivot);
        s.minRelTol = std::min(s.minRelTol, relTol);
        s.minAbsTol = std::min(s.minAbsTol, absTol);

        const cholesky_golden::GoldenReport rep = cholesky_golden::Evaluate(actual, a, golden, n, upper, relTol,
                                                                            absTol);
        s.worstRelErr = std::max(s.worstRelErr, rep.maxRelErrNormal);
        s.worstAbsErr = std::max(s.worstAbsErr, rep.maxAbsErr);
        s.worstMare = std::max(s.worstMare, rep.mare);
        s.worstMareNormal = std::max(s.worstMareNormal, rep.mareNormal);
        s.worstMere = std::max(s.worstMere, rep.mere);
        s.worstRmse = std::max(s.worstRmse, rep.rmse);
        s.worstResidFroRel = std::max(s.worstResidFroRel, rep.residFroRel);
        s.worstResidMaxAbs = std::max(s.worstResidMaxAbs, rep.residMaxAbs);
        s.tightestResidFroRelTol = std::min(s.tightestResidFroRelTol, rep.residFroRelTol);
        s.tightestResidFroAbsTol = std::min(s.tightestResidFroAbsTol, rep.residFroAbsTol);
        s.worstGoldenSelfFroRel = std::max(s.worstGoldenSelfFroRel, rep.goldenSelfFroRel);
        s.worstSmallValueErrorCount = std::max(s.worstSmallValueErrorCount, rep.smallValueErrorCount);
        s.smallValueCount += rep.smallValueCount;

        EXPECT_EQ(rep.elementFailures, 0ULL)
            << tag << " batch " << b << ": golden mismatch, worst |actual-golden|=" << rep.worstAbsErr << " at ("
            << rep.worstRow << "," << rep.worstCol << ") golden=" << rep.worstGolden << " limit=" << rep.worstLimit;
        EXPECT_LE(rep.maxRelErrNormal, relTol) << tag << " batch " << b << ": normal-domain relative error over relTol";
        EXPECT_LE(rep.maxAbsErr, absTol) << tag << " batch " << b << ": normwise absolute error over absTol";
        EXPECT_LE(rep.rmse, absTol) << tag << " batch " << b << ": RMSE over absTol";
        // [S2] 仓内公开绝对门限，与 n 无关：MERE < 2^-13、MARE < 10*2^-13。
        // MARE 取正常值域口径 mareNormal，理由见本文件顶部注释 (2)。
        EXPECT_LT(rep.mere, cholesky_golden::FP32_MERE_THRESHOLD)
            << tag << " batch " << b << ": MERE over in-repo FLOAT32 threshold 2^-13";
        EXPECT_LT(rep.mareNormal, cholesky_golden::FP32_MARE_THRESHOLD)
            << tag << " batch " << b << ": normal-domain MARE over in-repo FLOAT32 threshold 10*2^-13";
        // relTol 只在 n <= 16 时比 2^-13 更严；两条都断言，取更严者生效
        EXPECT_LE(rep.mere, relTol) << tag << " batch " << b << ": MERE over relTol";
        // [S1] §4.5.3 小值域 ErrorCount 判据，无竞品 => 分母 max(ErrorCount_gpu,1)=1
        EXPECT_LE(rep.smallValueErrorCount, cholesky_golden::SMALL_VALUE_ERROR_COUNT_LIMIT)
            << tag << " batch " << b << ": small-value ErrorCount over [S1] 4.5.3 limit";
        // (4a) 被测输出重构残差：使用 kernel 输出，鉴别 kernel。
        // max|R| 不单独断言：max|R| <= ||R||_F，该判据被上式严格蕴含，单独写出来只会是
        // 一条永不独立失败的假断言；其实测值仍打到 [GOLDEN] 行作诊断。
        EXPECT_LE(rep.residFroRel, rep.residFroRelTol)
            << tag << " batch " << b << ": ||A - factor(actual)*factor(actual)^T||_F/||A||_F over derived bound";
        // (4b) golden 自洽残差：只鉴别 FP64 参考实现自身，不是 kernel 正确性判据
        EXPECT_LE(rep.goldenSelfFroRel, cholesky_golden::GoldenSelfConsistencyTolerance(n))
            << tag << " batch " << b << ": FP64 golden is not a factorization of A (reference impl defect)";

        // 补三角：kernel 不写补三角。「补三角为 0」的对外契约不由 kernel 承载，而由 OpDef
        // 输出声明 .InitValue(0)（op_host/cholesky_def.cpp）+ 框架在 aclnn 通路对输出执行
        // memset（libnnopbase.so 的 NnopbaseMemsetTilingContextInit）保证。Kernel UT 直接
        // 调用 kernel、绕过框架 memset，故调用前初值不会被清。本断言因此只鉴别「契约外
        // 越界写」：补三角取值必须落在 {调用前初值, 0.0f}，其余值一律判为越界写。
        for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t j = 0; j < n; ++j) {
                if (upper ? (j >= i) : (j <= i)) {
                    continue;
                }
                const float v = actual[static_cast<size_t>(i) * n + j];
                if (v == OUT_SENTINEL) {
                    ++s.complementUntouched;
                } else if (v == 0.0f) {
                    ++s.complementZero;
                } else {
                    ++s.complementViolation;
                    ADD_FAILURE() << tag << " batch " << b << ": out-of-contract write at (" << i << "," << j
                                  << ") value=" << v;
                }
            }
        }
    }
    EXPECT_EQ(s.complementViolation, 0ULL) << tag << ": kernel wrote outside the active triangle";
    if (requireNegativeOffDiag && n >= 2) {
        EXPECT_GT(s.negativeOffDiag, 0ULL) << tag << ": SPD input degenerated to all-non-negative off-diagonals";
    }

    cout << "[GOLDEN] tag=" << tag << " n=" << n << " batch=" << batch << " upper=" << (upper ? 1 : 0)
         << " blockDim=" << cfg.blockDim << " blockSize=" << cfg.blockSize << " blockNum=" << cfg.blockNum
         << " wsBytes=" << cfg.workspaceBytes << " minPivot=" << s.minPivot << " relTol=" << s.minRelTol
         << " absTol=" << s.minAbsTol << " maxRelErr=" << s.worstRelErr << " maxAbsErr=" << s.worstAbsErr
         << " MARE=" << s.worstMare << " MAREnorm=" << s.worstMareNormal << " MERE=" << s.worstMere
         << " RMSE=" << s.worstRmse << " residFroRel=" << s.worstResidFroRel
         << " residFroRelTol=" << s.tightestResidFroRelTol << " residMaxAbs=" << s.worstResidMaxAbs
         << " residFroAbsTol=" << s.tightestResidFroAbsTol << " goldenSelfFroRel=" << s.worstGoldenSelfFroRel
         << " goldenSelfTol=" << cholesky_golden::GoldenSelfConsistencyTolerance(n) << " smallVal=" << s.smallValueCount
         << " smallValErrCnt=" << s.worstSmallValueErrorCount << " negOffDiag=" << s.negativeOffDiag
         << " compZero=" << s.complementZero << " compUntouched=" << s.complementUntouched << endl;

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)selfGM);
    AscendC::GmFree((void*)outGM);
}

// A = M*M^T + n*I，M ~ U(-1,1) 由 splitmix64(seed) 产生；逐批次 seed 不同 => 矩阵互不相同。
// 非对角元素 A[i][j] = sum_k M[i][k]*M[j][k] 符号混合，覆盖「SPD 但非对角为负」的合法场景。
inline void RunSeededCase(const char* tag, uint32_t n, uint64_t batch, bool upper, uint64_t seed)
{
    std::vector<std::vector<float>> mats(batch);
    for (uint64_t b = 0; b < batch; ++b) {
        cholesky_golden::MakeSpdMatrix(mats[b], n, seed + b * 0x9E3779B97F4A7C15ULL, static_cast<double>(n));
    }
    RunAndVerify(tag, mats, n, batch, upper);
}

inline std::vector<std::vector<float>> Replicate(const std::vector<float>& a, uint64_t batch)
{
    return std::vector<std::vector<float>>(batch, a);
}

inline std::vector<float> MakeDiagonalMatrix(const std::vector<float>& diag)
{
    const uint32_t n = static_cast<uint32_t>(diag.size());
    std::vector<float> a(static_cast<size_t>(n) * n, 0.0f);
    for (uint32_t i = 0; i < n; ++i) {
        a[static_cast<size_t>(i) * n + i] = diag[i];
    }
    return a;
}

// CPU 仿真核由独立子进程承载，断言诊断由父进程收集。
// 非法输入失败后不读取输出；设备侧返回码由上板测试核对。
inline void RunIllegalPivotCase(const char* tag, const std::vector<std::vector<float>>& mats, uint32_t n,
                                uint64_t batch, bool upper)
{
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    const size_t matElems = static_cast<size_t>(n) * n;
    const size_t matBytes = matElems * sizeof(float);
    const LaunchConfig cfg = MakeLaunchConfig(n, batch);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(cfg.workspaceBytes);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(CholeskyTilingData));
    uint8_t* selfGM = (uint8_t*)AscendC::GmAlloc(matBytes * batch);
    uint8_t* outGM = (uint8_t*)AscendC::GmAlloc(matBytes * batch);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);
    ASSERT_NE(selfGM, nullptr);
    ASSERT_NE(outGM, nullptr);

    for (uint64_t b = 0; b < batch; ++b) {
        memcpy(selfGM + b * matBytes, mats[b].data(), matBytes);
        std::vector<double> factor;
        double minPivot = 0.0;
        EXPECT_FALSE(cholesky_golden::GoldenFactor(mats[b].data(), n, upper, factor, &minPivot))
            << tag << " batch " << b << ": input is actually SPD in FP64";
    }
    std::fill_n(reinterpret_cast<float*>(outGM), matElems * batch, OUT_SENTINEL);
    SetTilingFields(tiling, n, batch, cfg);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    RunCholeskyKernel(upper, cfg.blockDim, selfGM, outGM, workspace, tiling);
    const auto standardError = testing::internal::GetCapturedStderr();
    const auto standardOutput = testing::internal::GetCapturedStdout();
    const auto diagnostic = standardOutput + standardError;
    EXPECT_NE(diagnostic.find("MAX_FINITE_FLOAT"), std::string::npos)
        << tag << ": kernel did not report the finite-positive pivot assertion";
    EXPECT_NE(diagnostic.find("failed"), std::string::npos) << tag << ": kernel did not report assertion failure";

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)selfGM);
    AscendC::GmFree((void*)outGM);
}

inline void RunDiagonalIllegalPivotCase(const char* tag, const std::vector<float>& diag, bool upper)
{
    const uint32_t n = static_cast<uint32_t>(diag.size());
    RunIllegalPivotCase(tag, Replicate(MakeDiagonalMatrix(diag), 1), n, 1, upper);
}

inline void RunDenseIndefiniteCase(const char* tag, bool upper)
{
    const std::vector<float> a{1.0f, 3.0f, 0.0f, 0.0f, 3.0f, 1.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    RunIllegalPivotCase(tag, Replicate(a, 1), 4, 1, upper);
}

inline void RunAllZeroBatchCase(const char* tag, uint32_t n, uint64_t batch, bool upper)
{
    const std::vector<float> zeroMat(static_cast<size_t>(n) * n, 0.0f);
    RunIllegalPivotCase(tag, Replicate(zeroMat, batch), n, batch, upper);
}
} // namespace

class cholesky_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "cholesky_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "cholesky_test TearDown\n" << endl; }
};

// ---------------------------------------------------------------------------
// 单矩阵稠密 SPD：A2/A3 走逐列/逐行路径（needCoreNum=1），A5 走协同面板路径
// （matrixNumCount==1 => cooperative_，needCoreNum=48）。
// ---------------------------------------------------------------------------

TEST_F(cholesky_test, golden_tril_single_n8) { RunSeededCase("tril_single_n8", 8, 1, false, 0x1234); }

TEST_F(cholesky_test, golden_triu_single_n8) { RunSeededCase("triu_single_n8", 8, 1, true, 0x1234); }

TEST_F(cholesky_test, golden_tril_single_n32) { RunSeededCase("tril_single_n32", 32, 1, false, 0x5678); }

TEST_F(cholesky_test, golden_triu_single_n32) { RunSeededCase("triu_single_n32", 32, 1, true, 0x5678); }

// ---------------------------------------------------------------------------
// 批量：matrixNumCount > 1 且 matSizeN <= BATCH_COOP_MIN_M(96)，三平台均为每核一矩阵。
// upper=true 且 batch>1、N<=96 覆盖 ProcessRowDotProduct 标量 SetValue 累加后
// 的 S->V 同步路径（arch35/cholesky_arch35.h 的 SyncPipeSToV）。
// ---------------------------------------------------------------------------

TEST_F(cholesky_test, golden_tril_batch3_n8) { RunSeededCase("tril_batch3_n8", 8, 3, false, 0x9ABC); }

TEST_F(cholesky_test, golden_triu_batch3_n8) { RunSeededCase("triu_batch3_n8", 8, 3, true, 0x9ABC); }

TEST_F(cholesky_test, golden_tril_batch2_n32) { RunSeededCase("tril_batch2_n32", 32, 2, false, 0xDEF0); }

TEST_F(cholesky_test, golden_triu_batch2_n32) { RunSeededCase("triu_batch2_n32", 32, 2, true, 0xDEF0); }

// ---------------------------------------------------------------------------
// count == 1 的 ScaleCholesky 路径，覆盖 RegBase 整寄存器装载改为 Muls 后的行为
// （arch35/cholesky_arch35.h:70-75）。batch shape [4,2,2] 使 SecondToNRow 的
// count 落到 1，[3,1,1] 使 FirstRow/FirstColumn 只有 count==1 的分块。
// ---------------------------------------------------------------------------

TEST_F(cholesky_test, golden_triu_batch4_n2) { RunSeededCase("triu_batch4_n2", 2, 4, true, 0x2468); }

TEST_F(cholesky_test, golden_tril_batch4_n2) { RunSeededCase("tril_batch4_n2", 2, 4, false, 0x2468); }

TEST_F(cholesky_test, golden_triu_batch3_n1) { RunSeededCase("triu_batch3_n1", 1, 3, true, 0x1357); }

TEST_F(cholesky_test, golden_tril_batch3_n1) { RunSeededCase("tril_batch3_n1", 1, 3, false, 0x1357); }

// ---------------------------------------------------------------------------
// 对角 SPD：三平台走通用路径的对角输入情形，补三角必为 0。
// ---------------------------------------------------------------------------

TEST_F(cholesky_test, golden_diagonal_spd_n8_tril)
{
    const std::vector<float> diag{1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f, 64.0f};
    RunAndVerify("diagonal_spd_n8_tril", Replicate(MakeDiagonalMatrix(diag), 1), 8, 1, false, false);
}

TEST_F(cholesky_test, golden_diagonal_spd_n8_triu)
{
    const std::vector<float> diag{1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f, 64.0f};
    RunAndVerify("diagonal_spd_n8_triu", Replicate(MakeDiagonalMatrix(diag), 1), 8, 1, true, false);
}

TEST_F(cholesky_test, golden_diagonal_spd_batch2_n8_tril)
{
    const std::vector<float> diag0{1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 13.0f, 15.0f};
    const std::vector<float> diag1{2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f, 256.0f};
    const std::vector<std::vector<float>> mats{MakeDiagonalMatrix(diag0), MakeDiagonalMatrix(diag1)};
    RunAndVerify("diagonal_spd_batch2_n8_tril", mats, 8, 2, false, false);
}

// ---------------------------------------------------------------------------
// 负非对角元素的合法 SPD（正式 static finding F-CHOLESKY-VO-005）：
// A = [[4,-2,0],[-2,5,-3],[0,-3,6]]，顺序主子式 4 / 16 / 60 均 > 0。
// 解析 golden：L = [[2,0,0],[-1,2,0],[0,-1.5,sqrt(3.75)]]，
//               U = [[2,-1,0],[0,2,-1.5],[0,0,sqrt(3.75)]]。
// n=3 <= SINGLE_MATRIX_SINGLE_CORE_MAX(4)，三平台 blockDim 均为 1。
// ---------------------------------------------------------------------------

TEST_F(cholesky_test, golden_negative_offdiag_n3_tril)
{
    const std::vector<float> a{4.0f, -2.0f, 0.0f, -2.0f, 5.0f, -3.0f, 0.0f, -3.0f, 6.0f};
    RunAndVerify("negative_offdiag_n3_tril", Replicate(a, 1), 3, 1, false);
}

TEST_F(cholesky_test, golden_negative_offdiag_n3_triu)
{
    const std::vector<float> a{4.0f, -2.0f, 0.0f, -2.0f, 5.0f, -3.0f, 0.0f, -3.0f, 6.0f};
    RunAndVerify("negative_offdiag_n3_triu", Replicate(a, 1), 3, 1, true);
}

// ---------------------------------------------------------------------------
// 零、负数及非有限主元均应触发有限正主元断言。
// 不修改 ACLNN 原型、不新增 info、不在第一段扫描 Device 数据、不新增 161002。

// ---------------------------------------------------------------------------

TEST_F(cholesky_test, pivot_assert_diagonal_negative_pivot_tril)
{
    RunDiagonalIllegalPivotCase("diagonal_negative_pivot_tril", {4.0f, -1.0f, 9.0f, 16.0f}, false);
}

TEST_F(cholesky_test, pivot_assert_diagonal_zero_pivot_tril)
{
    RunDiagonalIllegalPivotCase("diagonal_zero_pivot_tril", {4.0f, 0.0f, 9.0f, 16.0f}, false);
}

TEST_F(cholesky_test, pivot_assert_diagonal_nan_pivot_tril)
{
    RunDiagonalIllegalPivotCase("diagonal_nan_pivot_tril", {4.0f, std::numeric_limits<float>::quiet_NaN(), 9.0f, 16.0f},
                                false);
}

TEST_F(cholesky_test, pivot_assert_diagonal_posinf_pivot_triu)
{
    RunDiagonalIllegalPivotCase("diagonal_posinf_pivot_triu",
                                {4.0f, std::numeric_limits<float>::infinity(), 9.0f, 16.0f}, true);
}

// 非对角型非法输入也应触发主元断言。
TEST_F(cholesky_test, pivot_assert_dense_indefinite_tril) { RunDenseIndefiniteCase("dense_indefinite_tril", false); }

TEST_F(cholesky_test, pivot_assert_dense_indefinite_triu) { RunDenseIndefiniteCase("dense_indefinite_triu", true); }

// ---------------------------------------------------------------------------
// 全零矩阵的单矩阵与 batch 路径均应触发有限正主元断言。
// CPU 仿真核对断言诊断，设备 ST 核对返回码。
// ---------------------------------------------------------------------------

TEST_F(cholesky_test, pivot_assert_diagonal_all_zero_single_tril)
{
    RunDiagonalIllegalPivotCase("diagonal_all_zero_single_tril", {0.0f, 0.0f, 0.0f, 0.0f}, false);
}

TEST_F(cholesky_test, pivot_assert_all_zero_batch2_tril) { RunAllZeroBatchCase("all_zero_batch2_tril", 4, 2, false); }

TEST_F(cholesky_test, pivot_assert_all_zero_batch2_triu) { RunAllZeroBatchCase("all_zero_batch2_triu", 4, 2, true); }

#if __NPU_ARCH__ == 3510
// ---------------------------------------------------------------------------
// A5 协同面板路径（wenti.md 第 35 节第 7 项：此前零覆盖）。
// 触发条件（op_kernel/arch35/cholesky_arch35.h 的 cooperative_ 与
// op_host/arch35/cholesky_tiling_arch35.cpp:162-167 与 :206-209 完全一致）：
//   matrixNumCount == 1 || matSizeN > BATCH_COOP_MIN_M(96)
// PANEL_WIDTH = 64，故 matSizeN = 130 产生 3 个面板 [0,64) [64,128) [128,130)，
// 覆盖 FactorPanelTriu 的 firstPanel=true/false 与 UpdateTrailingTriu 的
// firstPanel=true/false（srcGM 在 matAGM_ 与 outGM_ 之间切换）。
// ---------------------------------------------------------------------------

TEST_F(cholesky_test, golden_panel_triu_n130_single) { RunSeededCase("panel_triu_n130", 130, 1, true, 0xABCDEF); }

TEST_F(cholesky_test, golden_panel_tril_n130_single) { RunSeededCase("panel_tril_n130", 130, 1, false, 0xABCDEF); }

// 批量协同分支：matrixNumCount(2) > 1 且 matSizeN(100) > BATCH_COOP_MIN_M(96)，
// 全核逐矩阵推进 RunUpperPanels，覆盖 ProcessTriu 中 for m in matrixNumCount 的
// SetGlobalBuffer 重定位与 2 个面板 [0,64) [64,100)。
TEST_F(cholesky_test, golden_panel_batchcoop_triu_n100)
{
    RunSeededCase("panel_batchcoop_triu_n100", 100, 2, true, 0xF00D);
}
#endif
