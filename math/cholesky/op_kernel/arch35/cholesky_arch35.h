/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHOLESKY_H
#define CHOLESKY_H

#include "kernel_operator.h"
#include "../cholesky_shared_const.h"
#if __NPU_ARCH__ == 3510
#include "simt_api/common_functions.h"
#endif

using namespace AscendC;

namespace Cholesky {
// 单缓冲：当前流水为同步串行（搬运/计算逐级等待），双缓冲无重叠收益，仅保留 1 份 UB 分配
constexpr uint32_t BUFFER_NUM = 1;
// [general] 32B 对齐为 Ascend 通用约束
constexpr uint32_t BASIC_BLOCK = 32;
constexpr uint32_t DIAGONAL_SCAN_THREADS = 512;

#if __NPU_ARCH__ == 3510
__simt_vf__ __aicore__ __launch_bounds__(DIAGONAL_SCAN_THREADS) inline void CholeskyDiagonalVf(uint64_t elementCount,
                                                                                               uint32_t matrixSize,
                                                                                               __gm__ float* input,
                                                                                               __gm__ float* output)
{
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elementCount;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        output[index] = index / matrixSize == index % matrixSize ? sqrt(input[index]) : 0.0f;
    }
}

#endif
// 正定性检查上界：拦截 +inf（inf > 0 为真，需显式排除）；NaN 与任何比较均为假，天然被拦截
constexpr float MAX_FINITE_FLOAT = 3.402823466e+38f;
// 单矩阵面板宽度与批量协同阈值由 ../cholesky_shared_const.h 提供（host tiling 与本文件共享的 ABI）：
// CHOLESKY_PANEL_WIDTH(64) 为右看面板算法的串行段宽度，8 的倍数保证面板行 32B 对齐；
// CHOLESKY_BATCH_COOP_MIN_M(96) 为批量模式进入协同面板的矩阵维度阈值
// 批量模式点积 2D 搬运分块缓冲上限（与 InitBuffer 的 matRightQueue_ 容量一致）
constexpr uint64_t CHUNK_BUFFER_BYTES = 32UL * 1024UL;
// R4 尾随更新行块上限：UB 复用粒度（matA/matPivot 容量与 U 面板复用倍数的折中）
constexpr uint32_t R4_TRAILING_ROW_BLOCK = 16;
// Schur 更新列块宽度：仅用于并行尾随更新（面板分解仍按 tiling 的 blockSize_）。
// 实测 VEC 忙 69.6%/SCALAR 35.7%/MTE2 2.1% ⇒ 瓶颈是每 k 的 UB 往返 + 地址开销，
// 加宽列块把每 k 的开销摊到 512 个元素上。
constexpr uint32_t CHOLESKY_SCHUR_COLS = 512U;
// ---------------------------------------------------------------------------
// Cube（AIC）Schur 更新参数：把 C[i,j] -= Σ_{k<64} U[k,i]·U[k,j] 做成一次
// Mmad(A=(-U)ᵀ, B=U)，K = 面板宽度 64 全部驻留 L0（singleCoreK=64，无 k-tail）。
//
// L0 容量建模（Ascend950PR_9599.ini [AICoreSpec]：l0_a_size=l0_b_size=65536、
// l0_c_size=262144、l1_size=524288；[DtypeMKN] DT_FLOAT=16,8,16 ⇒ m0=16、n0=8）：
//   baseK = min(singleCoreK, FloorAlign(l0aSize/2/sizeof(T)/max(baseM,baseN), align))
//         = min(64, FloorAlign(65536/2/4/128, 8)) = min(64, 64) = 64
//   ⇒ L0A = baseM×baseK×4 = 64×64×4  = 16 KiB（64 KiB 的 25%）
//      L0B = baseN×baseK×4 = 128×64×4 = 32 KiB（64 KiB 的 50%）
//      L0C = baseM×baseN×4 = 64×128×4 = 32 KiB（256 KiB 的 12.5%）
//   L1 只需容纳 A/B 两个 ND2NZ 块（16+32 = 48 KiB / 512 KiB）。
// baseN 取 128 而不是 256：MTE2 重复搬运系数 = (1/baseM + 1/baseN) 从 3/256 抬到
// 5/256，但换来 L0B 留出 32 KiB 给后续双缓冲（K=64 时单块已占满 L0A 无法 ping-pong）。
constexpr uint32_t CUBE_BASE_M = 64;
constexpr uint32_t CUBE_BASE_N = 128;
constexpr uint32_t CUBE_BASE_K = CHOLESKY_PANEL_WIDTH;
// 阶段 A 的列块宽度：与 Cube 的行块同宽，使 -Uᵀ 的转置块恒为 64×64 稠密块
constexpr uint32_t CUBE_PHASE_COLS = CUBE_BASE_M;
constexpr uint32_t CUBE_L1_A_ELEMS = CUBE_BASE_M * CUBE_BASE_K;
constexpr uint32_t CUBE_L1_B_ELEMS = CUBE_BASE_N * CUBE_BASE_K;
constexpr uint32_t CUBE_L0C_ELEMS = CUBE_BASE_M * CUBE_BASE_N;
constexpr uint32_t CUBE_UT_TILE_ELEMS = CUBE_PHASE_COLS * CUBE_BASE_K;
constexpr int32_t CUBE_EVENT_ID = 0;

// 3510 向量寄存器宽度：asc/impl/basic_api/utils/kernel_utils_constants.h:199 VECTOR_REG_WIDTH=256，
// 经 asc/include/basic_api/kernel_operator_sys_var_intf.h:76 GetVecLen() 暴露
constexpr uint32_t VEC_REG_BYTES = 256;
#if __NPU_ARCH__ == 3510
// [RegBase-native] SIMT __launch_bounds__ 线程数约束，与 asc_vf_call(dim3(...)) 一致（DAV_3510）

// [RegBase-native] SIMT 路径（__simt_vf__ + asc_vf_call(dim3)），仅 DAV_3510 生效

// [RegBase-native] 原地转置输出：上三角值搬移到下三角（L = U^T），上三角清零（对齐
// 旧 tril 路径只写下三角、golden 上三角为零的输出契约）。全核 grid-stride 并行，索引公式与
// grid-stride 写法：blockIdx.x*blockDim.x+threadIdx.x 起、blockDim.x*gridDim.x 跨步。
//
// 跨核无数据竞争的证明（grid-stride 把 [0, elementCount) 恰好划分给 gridDim.x*blockDim.x 个线程，
// 每个 index 被访问一次；守卫 row < col 使迭代集恰为严格上三角，不含对角线、不含下三角）：
// 线程处理 i=(r,c)（r<c）时只做三件事——① 读 mat[i]（上三角）；② 写 mat[(c,r)]（下三角）；
// ③ 写 mat[i]=0（上三角，即 ① 刚读过的同一地址）。
// · 读写冲突：全部读都落在上三角，而 ③ 的目标 i 与读它的是同一线程（同一 index 只属于一个线程）；
//   ② 的目标 (c,r) 落在下三角，不在任何线程的读集内 ⇒ 无跨线程读写冲突。
// · 写写冲突：两不同线程 i=(r,c)、i'=(r',c')（均 r<c、r'<c'）的写集 {(c,r),(r,c)} 与
//   {(c',r'),(r',c')} 相交要求 (c,r)==(c',r')（⇒ i==i'，矛盾）、(r,c)==(r',c')（⇒ i==i'，矛盾）
//   或 (c,r)==(r',c')（⇒ c==r' 且 r==c'，与 r<c、r'<c' 联立得 r<r，矛盾）⇒ 写集两两不相交。
// · 对角线：守卫 row<col 排除 r==c，故对角线既不读也不写，③ 不可能把对角线清零
//   （转置本就不改变对角线，其值由 FactorPanelTriu 的主元开方写入并原样保留）。
// 因此每个元素的最终值只由它自身决定、与线程/核的调度顺序无关 ⇒ 与单核版逐位相同（bit-exact），
// 纯数据搬移 + 常量清零，无算术、无归约、无舍入，不改变任何数值序。
// 原地清零严格下三角：upper 输出的非三角区必须逐位为 0（由 op_api MemSet 保证，kernel 不得写脏）。
// 尾随更新的对角行块按整块写出，会覆盖行 i0+r 的列 [i0, i0+r)；收尾统一清零，无逐块边界推断。
__simt_vf__ __aicore__ __launch_bounds__(DIAGONAL_SCAN_THREADS) inline void CholeskyZeroLowerVf(
    uint64_t elementCount, uint64_t elementsPerMatrix, uint32_t matrixSize, __gm__ float* mat)
{
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elementCount;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        const uint64_t local = index % elementsPerMatrix;
        if (local / matrixSize > local % matrixSize) {
            mat[index] = 0.0f;
        }
    }
}

__simt_vf__ __aicore__ __launch_bounds__(DIAGONAL_SCAN_THREADS) inline void CholeskyTransposeSwapVf(
    uint64_t elementCount, uint32_t matrixSize, __gm__ float* mat)
{
    for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elementCount;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        uint64_t row = index / matrixSize;
        uint64_t col = index % matrixSize;
        if (row < col) {
            mat[col * matrixSize + row] = mat[index];
            mat[index] = 0.0f;
        }
    }
}
#endif

// UB 队列缓冲的物理尺寸：逻辑尺寸按 VEC_REG_BYTES 向上取整后再加一个寄存器尾护。
// 3510 的 Level-2 向量 API 每 repeat 以「无掩码整寄存器」装载源操作数：
// asc/impl/basic_api/dav_3510/kernel_operator_vec_binary_scalar_impl.h:38 的
// Reg::LoadAlign(srcReg, src + i * repeatStride) 命中 asc/include/basic_api/reg_compute/
// kernel_reg_compute_datacopy_intf.h:35 的无掩码重载（带掩码重载在同文件 :72），
// 装载宽度 repeatStride*sizeof(T) = GetVecLen()/sizeof(T)*REG_NUM*sizeof(T) = VEC_REG_BYTES
// （本算子仅实例化 T=float，走 RegTraitNumOne 即 REG_NUM=1）；计算(:39)与存储(:40)才带掩码。
// 故源缓冲必须容纳 (元素偏移字节 + AlignUp(count*sizeof(T), VEC_REG_BYTES))；
// 各调用点均满足 偏移字节+count*sizeof(T) <= 逻辑尺寸，尾护一个寄存器即覆盖该上界
__aicore__ inline uint64_t VecRegGuardedBytes(uint64_t logicalBytes)
{
    return (logicalBytes + VEC_REG_BYTES - 1) / VEC_REG_BYTES * VEC_REG_BYTES + VEC_REG_BYTES;
}

// count==1 同样走普通向量 Muls：源缓冲尺寸已由 VecRegGuardedBytes 保证容纳整寄存器无掩码装载
__aicore__ inline void ScaleCholesky(LocalTensor<float>& dst, float scale, uint32_t count)
{
    Muls(dst, dst, scale, count);
}

template <typename T>
class Cholesky {
public:
    __aicore__ inline Cholesky(){};
    __aicore__ inline void InitTril(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, const CholeskyTilingData* tilingData,
                                    TPipe* pipe, bool cube = false);
    __aicore__ inline void InitTriu(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, const CholeskyTilingData* tilingData,
                                    TPipe* pipe, bool cube = false);
    __aicore__ inline void ProcessTril();
    __aicore__ inline void ProcessTriu();
    // Cube（AIC Schur）路径：入口与实现只在 TilingKey.cube==true 的实例化里被调用
    __aicore__ inline void ProcessTriuCube();
    __aicore__ inline void ProcessTrilCube();

private:
    __aicore__ inline void SyncPipeVToS();
    __aicore__ inline void SyncPipeSToV();
    __aicore__ inline void SyncPipeMte2ToV();
    __aicore__ inline void SyncPipeMte2ToS();
    __aicore__ inline void SyncPipeMte3ToS();
    __aicore__ inline void SyncPipeVToMte3();
    __aicore__ inline void GetTilingData(const CholeskyTilingData* tilingData);
    __aicore__ inline void GetGroupGeometry(uint64_t remaining);
    __aicore__ inline void FirstColumn(uint64_t offset);
    __aicore__ inline void SecondToNColumn(uint32_t index, uint64_t offset);
    __aicore__ inline void FirstRow(uint64_t offset);
    __aicore__ inline void SecondToNRow(uint32_t index, uint64_t offset);
    __aicore__ inline void FactorPanelTriu(uint32_t panelStart, uint32_t panelEnd, bool firstPanel, bool narrow);
    __aicore__ inline void UpdatePanelRowTrailing(uint32_t panelStart, uint32_t panelEnd, bool firstPanel, bool active);
    __aicore__ inline void UpdateTrailingTriu(uint32_t panelStart, uint32_t panelEnd, bool firstPanel, bool active,
                                              bool narrow);
#if __NPU_ARCH__ == 3510
    __aicore__ inline bool ProcessDiagonalMatrix();
    __aicore__ inline void RunUpperPanels(bool active);
    __aicore__ inline void TransposeAllMatrices();
    __aicore__ inline void TransposeUpperToLowerTiles();
    __aicore__ inline void ZeroLowerTriangles(bool needed);
    __aicore__ inline void RunUpperPanelsCube();
    __aicore__ inline void InitUpperTriangleFromInput();
    __aicore__ inline void InitUtTransposeIndex();
    __aicore__ inline void UpdatePanelRowTrailingCube(uint32_t panelStart, uint32_t panelEnd, bool firstPanel);
    __aicore__ inline void CubeSchurPanel(uint32_t panelStart, uint32_t panelEnd);
    __aicore__ inline void InitCubeBuffers(TPipe* pipe);
    __aicore__ inline void CubeUtCopyIn(const GlobalTensor<T>& src, uint32_t rowOffset, uint32_t rows,
                                        const LocalTensor<T>& l1Dst);
    __aicore__ inline void CubeLoadA(const LocalTensor<T>& l1Src, uint32_t mExt, const LocalTensor<T>& l0Dst);
    __aicore__ inline void CubeLoadB(const LocalTensor<T>& l1Src, uint32_t nExt, const LocalTensor<T>& l0Dst);
#endif

    template <typename T1, typename T2>
    __aicore__ inline T1 CeilDiv(T1 a, T2 b)
    {
        return b == 0 ? a : (a + b - 1) / b;
    }

    __aicore__ inline uint32_t AlignToSlot(uint32_t elements)
    {
        constexpr uint32_t SLOT = BASIC_BLOCK / sizeof(float);
        return (elements + SLOT - 1) / SLOT * SLOT;
    }

private:
    uint32_t blockIdx_ = 0;
    uint32_t blockDim_ = 0;
    // 批量分组调度（仅 cooperative_）：每个矩阵由 groupCores_ 个核协同（lane_ 为组内序号），
    // 全核被切成 groupNum_ 个组，组 g 处理矩阵 {g, g+groupNum_, ...}。groupCores_==1 时
    // 退化为「每核一矩阵」，groupCores_==blockDim_ 时退化为「全核逐矩阵串行」（原行为）。
    uint32_t groupCores_ = 1;
    uint32_t groupNum_ = 1;
    uint32_t lane_ = 0;
    uint32_t group_ = 0;
    uint32_t matSizeN_ = 0;
    uint64_t matrixNumCount_ = 0;
    uint32_t blockSize_ = 0;
    // 协同面板模式（仅 3510 恒为 true；非 3510 恒为 false）
    bool cooperative_ = false;
    __gm__ T* selfBase_ = nullptr;
    __gm__ T* outBase_ = nullptr;
    T invSqrtA11_ = 0.0f; // 存储首主元缩放因子，避免重复计算和直接访问GM内存
    T zero_ = 0.0f;
    T one_ = 1.0f;

    TQue<QuePosition::VECIN, BUFFER_NUM> matAQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> matLeftQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> matRightQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> matResultQueue_;

    // Cube 路径：-Uᵀ / Uᵀ workspace 与 AIC 侧 L1/L0 缓冲
    GlobalTensor<T> cubeUtNegGm_;
    GlobalTensor<T> cubeUtGm_;
    bool cubeActive_ = false;
    uint32_t cubeNum_ = 0;
    uint32_t cubeIdx_ = 0;
    TBuf<TPosition::A1> cubeL1ABuf_;
    TBuf<TPosition::B1> cubeL1BBuf_;
    TBuf<TPosition::A2> cubeL0ABuf_;
    TBuf<TPosition::B2> cubeL0BBuf_;
    TBuf<TPosition::CO1> cubeL0CBuf_;
    TBuf<TPosition::VECCALC> cubeUtBuf0_;
    TBuf<TPosition::VECCALC> cubeUtBuf1_;
    TBuf<TPosition::VECCALC> cubeUtIdxBuf_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> matLQueue_;

    GlobalTensor<T> matAGM_;
    GlobalTensor<T> outGM_;
    GlobalTensor<T> workspaceFlagGM_;
    // 面板分解（串行段）→ 并行阶段 A 的通道：每面板每行一个 1/U[i,i]，按组隔离
    GlobalTensor<T> workspaceScaleGM_;

    // 辅助函数声明
    __aicore__ inline void ProcessColumnDotProduct(LocalTensor<T>& matLLocal, LocalTensor<T>& matLeftLocal,
                                                   LocalTensor<T>& matRightLocal, LocalTensor<T>& matResultLocal,
                                                   uint32_t index, uint64_t offset, uint32_t blockStart,
                                                   uint32_t count);

    __aicore__ inline void ProcessRowDotProduct(LocalTensor<T>& matLLocal, LocalTensor<T>& matLeftLocal,
                                                LocalTensor<T>& matRightLocal, LocalTensor<T>& matResultLocal,
                                                uint32_t index, uint64_t offset, uint32_t blockStart, uint32_t count);

    __aicore__ inline T ComputeScaleFactor(const LocalTensor<T>& matLLocal);
    __aicore__ inline void SyncSingleMatrix();
};

template <typename T>
__aicore__ inline void Cholesky<T>::SyncPipeVToS()
{
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
}

template <typename T>
__aicore__ inline void Cholesky<T>::SyncPipeSToV()
{
    event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIDSToV);
    WaitFlag<HardEvent::S_V>(eventIDSToV);
}

template <typename T>
__aicore__ inline void Cholesky<T>::SyncPipeMte2ToV()
{
    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
}

template <typename T>
__aicore__ inline void Cholesky<T>::SyncPipeMte2ToS()
{
    event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
}

template <typename T>
__aicore__ inline void Cholesky<T>::SyncPipeMte3ToS()
{
    event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
    WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
}

template <typename T>
__aicore__ inline void Cholesky<T>::SyncPipeVToMte3()
{
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
}

template <typename T>
__aicore__ inline void Cholesky<T>::GetTilingData(const CholeskyTilingData* tilingData)
{
    matSizeN_ = tilingData->matSizeN;
    matrixNumCount_ = tilingData->matrixNumCount;
    blockSize_ = tilingData->blockSize;
}

// 分组几何：C = max(1, blockDim / B)，G = blockDim / C。
// B >= blockDim 时 C=1（每核一矩阵、ceil(B/blockDim) 波）；B < blockDim 时 G*C <= blockDim 且
// G >= B（一批内所有矩阵同时推进，剩余核作为空组只参与 SyncAll）。
template <typename T>
__aicore__ inline void Cholesky<T>::GetGroupGeometry(uint64_t remaining)
{
    if (remaining == 0 || blockDim_ == 0) {
        groupCores_ = 1;
        groupNum_ = 1;
        lane_ = blockIdx_;
        group_ = 0;
        return;
    }
    uint32_t cores = remaining >= blockDim_ ? 1U : blockDim_ / static_cast<uint32_t>(remaining);
    groupCores_ = cores == 0 ? 1 : cores;
    groupNum_ = blockDim_ / groupCores_;
    lane_ = blockIdx_ % groupCores_;
    group_ = blockIdx_ / groupCores_;
}

template <typename T>
__aicore__ inline void Cholesky<T>::SyncSingleMatrix()
{
    // 原版语义：仅单矩阵（逐列路径）跨核同步；批量每核独立禁止 SyncAll（到达数不匹配会死锁）
    if (matrixNumCount_ == 1) {
        AscendC::SyncAll();
    }
}

#if __NPU_ARCH__ == 3510
template <typename T>
__aicore__ inline bool Cholesky<T>::ProcessDiagonalMatrix()
{
    if (matrixNumCount_ != 1) {
        return false;
    }
    bool allPositiveDiagonal = true;
    if ASCEND_IS_AIV {
        bool localPositiveDiagonal = true;
        LocalTensor<T> scanLocal = matAQueue_.AllocTensor<T>();
        for (uint32_t row = blockIdx_; row < matSizeN_ && localPositiveDiagonal; row += blockDim_) {
            for (uint32_t column = 0; column < matSizeN_ && localPositiveDiagonal; column += blockSize_) {
                uint32_t count = matSizeN_ - column > blockSize_ ? blockSize_ : matSizeN_ - column;
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
                DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                DataCopyPad(scanLocal, matAGM_[static_cast<uint64_t>(row) * matSizeN_ + column], copyParams, padParams);
                SyncPipeMte2ToS();
                for (uint32_t offset = 0; offset < count; ++offset) {
                    const T value = scanLocal.GetValue(offset);
                    const bool valid = column + offset == row ? (value > zero_ && value <= MAX_FINITE_FLOAT) :
                                                                value == zero_;
                    if (!valid) {
                        localPositiveDiagonal = false;
                        break;
                    }
                }
            }
        }
        matAQueue_.FreeTensor(scanLocal);
        LocalTensor<T> flagLocal = matAQueue_.AllocTensor<T>();
        Duplicate(flagLocal, localPositiveDiagonal ? zero_ : one_, 1);
        SyncPipeVToMte3();
        DataCopyExtParams flagCopyParams{1, sizeof(T), 0, 0, 0};
        DataCopyPad(workspaceFlagGM_[blockIdx_], flagLocal, flagCopyParams);
        SyncPipeMte3ToS();
        matAQueue_.FreeTensor(flagLocal);
    }
    // MIX launches include AIC participants in the same vote and early-return decision.
    if (cubeActive_) {
        AscendC::SyncAll<false>();
    } else {
        SyncSingleMatrix();
    }
    if ASCEND_IS_AIV {
        LocalTensor<T> flagLocal = matAQueue_.AllocTensor<T>();
        DataCopyExtParams oneVoteParams{1, sizeof(T), 0, 0, 0};
        DataCopyPadExtParams<T> oneVotePadParams{false, 0, 0, 0};
        for (uint32_t core = 0; core < blockDim_; ++core) {
            DataCopyPad(flagLocal, workspaceFlagGM_[core], oneVoteParams, oneVotePadParams);
            SyncPipeMte2ToS();
            if (flagLocal.GetValue(0) != zero_) {
                allPositiveDiagonal = false;
                break;
            }
        }
        matAQueue_.FreeTensor(flagLocal);
    }
    if ASCEND_IS_AIC {
        for (uint32_t core = 0; core < blockDim_; ++core) {
            if (workspaceFlagGM_.GetValue(core) != zero_) {
                allPositiveDiagonal = false;
                break;
            }
        }
    }
    if (!allPositiveDiagonal) {
        return false;
    }
    if ASCEND_IS_AIV {
        const uint64_t elementCount = static_cast<uint64_t>(matSizeN_) * matSizeN_;
        asc_vf_call<CholeskyDiagonalVf>(dim3(DIAGONAL_SCAN_THREADS), elementCount, matSizeN_,
                                        (__gm__ float*)matAGM_.GetPhyAddr(), (__gm__ float*)outGM_.GetPhyAddr());
    }
    if (cubeActive_) {
        AscendC::SyncAll<false>();
    }
    return true;
}
#endif

template <typename T>
__aicore__ inline void Cholesky<T>::InitTril(GM_ADDR self, GM_ADDR out, GM_ADDR workspace,
                                             const CholeskyTilingData* tilingData, TPipe* pipe, bool cube)
{
    blockIdx_ = GetBlockIdx();
    blockDim_ = GetBlockNum();
#if __NPU_ARCH__ == 3510
    // MIX(AIC:AIV = 1:2) 下 GetBlockNum() 返回 AIC 数，AIV 的 BlockIdx 空间是 aicNum*2；
    // cube 键 2/3 要求 MIX(AIC:AIV = 1:2)；AIV_ONLY 不属于其支持的启动配置。
    // 此处按实际 task ratio 设置资源，不提供 AIV_ONLY 的 Cube 计算降级路径。
    bool mixLaunch = true;
    if ASCEND_IS_AIV {
        mixLaunch = GetTaskRatio() >= 2; // MIX_AIC_1_2 下 AIV 的 GetTaskRatio()==2
    }
    cubeActive_ = cube && mixLaunch;
    if (cubeActive_) {
        // MIX(1:2)：GetBlockNum() 是 AIC 数，AIV 数为 2 倍（GetBlockIdx 在 AIV 上即 0..2*aicNum-1）
        blockDim_ = static_cast<uint32_t>(GetBlockNum()) * 2U;
        cubeNum_ = static_cast<uint32_t>(GetBlockNum());
        cubeIdx_ = static_cast<uint32_t>(GetBlockIdx());
    }
#endif
    GetTilingData(tilingData);
    // 3510：面板算法是唯一路径（批量也走分组并行面板），旧逐列算法仅保留给非 3510 代际
    cooperative_ = true;
    selfBase_ = (__gm__ T*)self;
    outBase_ = (__gm__ T*)out;

    matAGM_.SetGlobalBuffer(selfBase_, matSizeN_ * matSizeN_ * matrixNumCount_);
    outGM_.SetGlobalBuffer(outBase_, matSizeN_ * matSizeN_ * matrixNumCount_);
    // kernel 入参 workspace 是整块 workspace 基址，前 RESERVED_WORKSPACE(3510=16MB) 为系统/lib-API
    // 保留区，必须经 GetUserWorkspace() 偏移后才能放算子私有数据；Host 侧申报 保留区+用户区
    // （blockDim 个 float，32B 对齐）。CPU 仿真下 GetUserWorkspace() 原样返回，故 UT 无法暴露漏偏移。
    workspaceFlagGM_.SetGlobalBuffer((__gm__ T*)GetUserWorkspace(workspace), blockDim_);
    // 用户区布局：[0, blockDim) 保留 | [blockDim, blockDim + blockDim*PANEL_WIDTH) 面板缩放因子
    // 下标 = group_ * CHOLESKY_PANEL_WIDTH + (行在面板内的序号)；host 侧按同一公式申报
    workspaceScaleGM_.SetGlobalBuffer((__gm__ T*)GetUserWorkspace(workspace) + AlignToSlot(blockDim_),
                                      static_cast<uint64_t>(blockDim_) * CHOLESKY_PANEL_WIDTH);

#if __NPU_ARCH__ == 3510
    if (cubeActive_) {
        // Cube 路径 UB：阶段 A 列块固定 CUBE_PHASE_COLS(64)，accTile / 转置块恒为 64×64；
        // 相位 B（UpdateTrailingTriu）不再调用，故 matRight 从 512 列的 128 KiB 降到 16 KiB。
        const uint64_t cubeTileBytes = static_cast<uint64_t>(CUBE_UT_TILE_ELEMS) * sizeof(T);
        const uint64_t cubeSlotBytes = static_cast<uint64_t>(CHOLESKY_PANEL_WIDTH + 1U) * BASIC_BLOCK;
        const uint64_t cubePanelBytes = cubeSlotBytes > cubeTileBytes ? cubeSlotBytes : cubeTileBytes;
        const uint64_t cubeTmpBytes = static_cast<uint64_t>(CUBE_PHASE_COLS) * sizeof(T);
        pipe->InitBuffer(matAQueue_, BUFFER_NUM, VecRegGuardedBytes(cubeTileBytes));
        pipe->InitBuffer(matLQueue_, BUFFER_NUM, VecRegGuardedBytes(cubePanelBytes));
        pipe->InitBuffer(matLeftQueue_, BUFFER_NUM, VecRegGuardedBytes(cubePanelBytes));
        pipe->InitBuffer(matRightQueue_, BUFFER_NUM, VecRegGuardedBytes(cubeTileBytes));
        pipe->InitBuffer(matResultQueue_, BUFFER_NUM, VecRegGuardedBytes(cubeTmpBytes));
        pipe->InitBuffer(cubeUtBuf0_, VecRegGuardedBytes(cubeTileBytes));
        pipe->InitBuffer(cubeUtBuf1_, VecRegGuardedBytes(cubeTileBytes));
        pipe->InitBuffer(cubeUtIdxBuf_, static_cast<uint64_t>(CUBE_UT_TILE_ELEMS) * sizeof(uint32_t));
        if ASCEND_IS_AIC {
            InitCubeBuffers(pipe);
        }
        // 用户区尾部：[.. , +64*M) 为 -Uᵀ，[+64*M, +128*M) 为 Uᵀ（host 按同一公式申报）
        const uint64_t utElems = static_cast<uint64_t>(CHOLESKY_PANEL_WIDTH) * matSizeN_;
        __gm__ T* userBase = (__gm__ T*)GetUserWorkspace(workspace);
        const uint64_t utOff = AlignToSlot(blockDim_) + static_cast<uint64_t>(blockDim_) * CHOLESKY_PANEL_WIDTH;
        cubeUtGm_.SetGlobalBuffer(userBase + utOff, utElems);
        cubeUtNegGm_.SetGlobalBuffer(userBase + utOff + utElems, utElems);
        return;
    }
    if (cooperative_) {
        // 协同面板路径（仅 3510）：Schur 补驻留 outGM 尾部区域，2D 面板块需要
        // 更大搬运缓冲。matLeft 需容纳面板 slotted 主元拷贝（CHOLESKY_PANEL_WIDTH × slot 元素）。
        // 非 3510 走原版逐列路径，保持原版批量缓冲布局（matLeft/matResult = columnBufferSize）
        // 字节口径：matA 需容纳 R4_TRAILING_ROW_BLOCK 行 × blockSize 列；matL(matPivot) 需容纳
        // 面板列槽位((P+1)*slot) 或 U 面板列块(P × R4_TRAILING_ROW_BLOCK)，取大者
        const uint64_t aBufferSize = static_cast<uint64_t>(R4_TRAILING_ROW_BLOCK) * CHOLESKY_SCHUR_COLS * sizeof(T);
        const uint64_t panelSlotBytes = static_cast<uint64_t>(CHOLESKY_PANEL_WIDTH + 1U) * BASIC_BLOCK;
        const uint64_t panelTileBytes = static_cast<uint64_t>(CHOLESKY_PANEL_WIDTH) * R4_TRAILING_ROW_BLOCK * sizeof(T);
        const uint64_t panelBufferSize = panelSlotBytes > panelTileBytes ? panelSlotBytes : panelTileBytes;
        const uint64_t rightBufferSize = static_cast<uint64_t>(CHOLESKY_SCHUR_COLS) * CHOLESKY_PANEL_WIDTH * sizeof(T);
        const uint64_t tmpBufferSize = static_cast<uint64_t>(CHOLESKY_SCHUR_COLS) * sizeof(T);
        pipe->InitBuffer(matAQueue_, BUFFER_NUM, VecRegGuardedBytes(aBufferSize));
        pipe->InitBuffer(matLQueue_, BUFFER_NUM, VecRegGuardedBytes(panelBufferSize));
        pipe->InitBuffer(matLeftQueue_, BUFFER_NUM, VecRegGuardedBytes(panelBufferSize));
        pipe->InitBuffer(matRightQueue_, BUFFER_NUM, VecRegGuardedBytes(rightBufferSize));
        pipe->InitBuffer(matResultQueue_, BUFFER_NUM, VecRegGuardedBytes(tmpBufferSize));
        return;
    }
#endif

    // 使用分块大小计算buffer，减少UB内存使用
    uint64_t columnBufferSize = VecRegGuardedBytes(blockSize_ * BASIC_BLOCK);
    uint64_t rowBufferSize = VecRegGuardedBytes(CeilDiv(blockSize_ * sizeof(T), BASIC_BLOCK) * BASIC_BLOCK);

    pipe->InitBuffer(matAQueue_, BUFFER_NUM, columnBufferSize);
    pipe->InitBuffer(matLQueue_, BUFFER_NUM, columnBufferSize);
    pipe->InitBuffer(matLeftQueue_, BUFFER_NUM, columnBufferSize);
    pipe->InitBuffer(matRightQueue_, BUFFER_NUM, VecRegGuardedBytes(CHUNK_BUFFER_BYTES));
    pipe->InitBuffer(matResultQueue_, BUFFER_NUM, rowBufferSize);
}

// 协同模式：对 base 偏移处的单个矩阵执行上三角右看面板算法
// 协同模式：对 base 偏移处的单个矩阵执行上三角右看面板算法（仅 3510）。
// 面板间同步无条件 SyncAll：调用方保证所有核执行同一矩阵序列（单矩阵或协同批量）
#if __NPU_ARCH__ == 3510
template <typename T>
__aicore__ inline void Cholesky<T>::RunUpperPanels(bool active)
{
    for (uint32_t panelStart = 0; panelStart < matSizeN_; panelStart += CHOLESKY_PANEL_WIDTH) {
        const uint32_t panelEnd = panelStart + CHOLESKY_PANEL_WIDTH < matSizeN_ ? panelStart + CHOLESKY_PANEL_WIDTH :
                                                                                  matSizeN_;
        const bool firstPanel = panelStart == 0;
        // 窄化前提：组内至少 2 核（否则面板行段拆并行无收益，只多付 1 次屏障 + 主元重算）
        // 且尾随宽度跨多个列块（否则原逐行实现本就没有 MTE2 放大）
        const bool narrow = groupCores_ >= 2U && (matSizeN_ - panelEnd) > blockSize_;
        // 面板属主 = 组内轮转（原实现为全体核轮转，等价于 groupCores_==blockDim_ 且单组）
        if (active && lane_ == (panelStart / CHOLESKY_PANEL_WIDTH) % groupCores_) {
            FactorPanelTriu(panelStart, panelEnd, firstPanel, narrow);
        }
        // SyncAll 是全体核屏障：所有核（含空组）都必须无条件到达，次数只由 matSizeN_ 决定
        AscendC::SyncAll();
        if (panelEnd < matSizeN_) {
            if (narrow) {
                // 阶段 A 的 U 尾随段是阶段 B 的输入 ⇒ 必须再插一次全核屏障
                if (active) {
                    UpdatePanelRowTrailing(panelStart, panelEnd, firstPanel, active);
                }
                AscendC::SyncAll();
            }
            if (active) {
                UpdateTrailingTriu(panelStart, panelEnd, firstPanel, active, narrow);
            }
            AscendC::SyncAll();
        }
    }
}

// 全部矩阵的原地转置（一次性、全核共享）。VF 的 grid-stride 由 gridDim.x = 本次 launch 核数
// 决定，无法按组裁剪，故把转置从「每矩阵紧跟面板」改为「全部分解完成后统一执行」；
// 各矩阵区域互不相交，任意核数/任意顺序都写出同一结果（与逐矩阵版本逐位相同）。
template <typename T>
__aicore__ inline void Cholesky<T>::ZeroLowerTriangles(bool needed)
{
    if (!needed) {
        return;
    }
    // 一次调用覆盖整批（逐矩阵多次 asc_vf_call 的启动开销对小批量 upT 用例有 3–11% 回退）
    const uint64_t elementsPerMatrix = static_cast<uint64_t>(matSizeN_) * matSizeN_;
    const uint64_t total = elementsPerMatrix * matrixNumCount_;
    if (total > 0) {
        asc_vf_call<CholeskyZeroLowerVf>(dim3(DIAGONAL_SCAN_THREADS), total, elementsPerMatrix, matSizeN_,
                                         (__gm__ float*)outBase_);
    }
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventId);
    WaitFlag<HardEvent::V_MTE3>(eventId);
    AscendC::SyncAll();
}

// Cube 路径的下三角输出：按 64×64 tile 做「上三角→下三角」成对搬运。
// 不能用 TransposeAllMatrices 的 SIMT VF：MIX launch 下 AIC 的 Fixpipe 原子加写入不经过 AIV 的
// D-cache，VF 的普通 global load 实测读到 0（导致下三角 93% 为 0；对照：同 seed 的 upper=True
// 输出逐位正确、且跳过转置时 upper 输出与 upper=True 逐位相同 ⇒ 计算无误，只有 VF 读失效）。
// MTE2 的 GM→UB 读路径与 AIC 写入是一致的（阶段 A 就靠它读上一面板的 Schur 结果），故改为
// MTE2 读 tile → UB 标量转置 → MTE3 写回下三角 tile + 清零上三角 tile。
// 前提：cube 路径由 host tiling 保证 M % 64 == 0（cholesky_tiling_arch35.cpp CUBE_MIN_MATRIX_SIZE 段）。
template <typename T>
__aicore__ inline void Cholesky<T>::TransposeUpperToLowerTiles()
{
    if (matrixNumCount_ == 0 || matSizeN_ % CUBE_BASE_M != 0) {
        return;
    }
    const uint32_t nBlk = matSizeN_ / CUBE_BASE_M;
    const uint64_t tilesPerMatrix = static_cast<uint64_t>(nBlk) * (nBlk + 1U) / 2U;
    const uint64_t totalTiles = tilesPerMatrix * matrixNumCount_;
    LocalTensor<T> srcTile = cubeUtBuf0_.GetWithOffset<T>(CUBE_UT_TILE_ELEMS, 0);
    LocalTensor<T> dstTile = cubeUtBuf1_.GetWithOffset<T>(CUBE_UT_TILE_ELEMS, 0);
    const uint32_t rowGapBytes = static_cast<uint32_t>((matSizeN_ - CUBE_BASE_M) * sizeof(T));
    uint64_t idx = 0;
    for (uint64_t mat = 0; mat < matrixNumCount_; ++mat) {
        const uint64_t matBase = mat * static_cast<uint64_t>(matSizeN_) * matSizeN_;
        for (uint32_t bi = 0; bi < nBlk; ++bi) {
            for (uint32_t bj = bi; bj < nBlk; ++bj, ++idx) {
                if (idx % static_cast<uint64_t>(blockDim_) != static_cast<uint64_t>(blockIdx_)) {
                    continue;
                }
                const uint64_t upOff = matBase + static_cast<uint64_t>(bi * CUBE_BASE_M) * matSizeN_ +
                                       static_cast<uint64_t>(bj) * CUBE_BASE_M;
                const uint64_t loOff = matBase + static_cast<uint64_t>(bj * CUBE_BASE_M) * matSizeN_ +
                                       static_cast<uint64_t>(bi) * CUBE_BASE_M;
                DataCopyExtParams readCp{CUBE_BASE_M, static_cast<uint32_t>(CUBE_BASE_M * sizeof(T)), rowGapBytes, 0,
                                         0};
                DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                DataCopyPad(srcTile, outGM_[upOff], readCp, padParams);
                SyncPipeMte2ToS();
                for (uint32_t r = 0; r < CUBE_BASE_M; ++r) {
                    for (uint32_t c = 0; c < CUBE_BASE_M; ++c) {
                        // dstTile[r][c] = srcTile[c][r]；对角 tile 的严格上三角（r<c）清零
                        T v = (bi == bj && r < c) ? T(0) : srcTile.GetValue(c * CUBE_BASE_M + r);
                        dstTile.SetValue(r * CUBE_BASE_M + c, v);
                    }
                }
                SetFlag<HardEvent::S_MTE3>(CUBE_EVENT_ID);
                WaitFlag<HardEvent::S_MTE3>(CUBE_EVENT_ID);
                if (bi != bj) {
                    SetFlag<HardEvent::S_V>(CUBE_EVENT_ID);
                    WaitFlag<HardEvent::S_V>(CUBE_EVENT_ID);
                    Duplicate(srcTile, T(0), CUBE_UT_TILE_ELEMS);
                    SyncPipeVToMte3();
                }
                DataCopyExtParams writeCp{CUBE_BASE_M, static_cast<uint32_t>(CUBE_BASE_M * sizeof(T)), 0, rowGapBytes,
                                          0};
                if (bi == bj) {
                    DataCopyPad(outGM_[upOff], dstTile, writeCp);
                } else {
                    DataCopyPad(outGM_[loOff], dstTile, writeCp);
                    DataCopyPad(outGM_[upOff], srcTile, writeCp);
                }
                SyncPipeMte3ToS();
            }
        }
    }
}

template <typename T>
__aicore__ inline void Cholesky<T>::TransposeAllMatrices()
{
    const uint64_t elementsPerMatrix = static_cast<uint64_t>(matSizeN_) * matSizeN_;
    for (uint64_t m = 0; m < matrixNumCount_; ++m) {
        asc_vf_call<CholeskyTransposeSwapVf>(dim3(DIAGONAL_SCAN_THREADS), elementsPerMatrix, matSizeN_,
                                             (__gm__ float*)(outBase_ + m * elementsPerMatrix));
    }
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventId);
    WaitFlag<HardEvent::V_MTE3>(eventId);
    AscendC::SyncAll();
}
#endif

template <typename T>
__aicore__ inline void Cholesky<T>::ProcessTril()
{
#if __NPU_ARCH__ == 3510
    if (ProcessDiagonalMatrix()) {
        return;
    }
    if (cooperative_) {
        // 协同模式（单矩阵或大维度批量）：下三角 = 上三角分解结果的转置（A 对称，SPD 契约
        // 保证）。转置经 SIMT 成对交换完成，免去逐行点积慢路径。
        // 批量为 0（空 batch）时整段跳过，不产生任何 SyncAll。
        if (matrixNumCount_ == 0) {
            return;
        }
        uint64_t done = 0;
        while (done < matrixNumCount_) {
            GetGroupGeometry(matrixNumCount_ - done);
            const uint64_t m = static_cast<uint64_t>(group_) + done;
            done += groupNum_;
            const bool active = group_ < groupNum_ && m < matrixNumCount_;
            if (active) {
                const uint64_t offset = m * matSizeN_ * matSizeN_;
                matAGM_.SetGlobalBuffer(selfBase_ + offset, matSizeN_ * matSizeN_);
                outGM_.SetGlobalBuffer(outBase_ + offset, matSizeN_ * matSizeN_);
            }
            RunUpperPanels(active);
        }
        TransposeAllMatrices();
        return;
    }
#endif
    // 非 3510 / 非协同批量：原版每核一矩阵（单矩阵 offsetPrefix=0），单矩阵逐列 SyncAll
    if (blockIdx_ < blockDim_) {
        auto loopTimes = matrixNumCount_ == 1 ? 0 : matrixNumCount_ / blockDim_;
        for (uint64_t loopIndex = 0; loopIndex <= loopTimes; loopIndex++) {
            uint64_t offsetPrefix = matrixNumCount_ == 1 ? 0 : blockIdx_ + blockDim_ * loopIndex;
            if (offsetPrefix < matrixNumCount_) {
                uint64_t offset = offsetPrefix * matSizeN_ * matSizeN_;
                FirstColumn(offset);
                SyncSingleMatrix();
                for (uint32_t index = 1; index < matSizeN_; index++) {
                    SecondToNColumn(index, offset);
                    SyncSingleMatrix();
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void Cholesky<T>::FirstColumn(uint64_t offset)
{
    LocalTensor<T> matALocal = matAQueue_.AllocTensor<T>();
    if (matrixNumCount_ == 1) {
        // 单元素 DataCopyPad 搬入首主元后经 MTE2_S 同步再标量读取，避免逐元素 GM 访问
        DataCopyExtParams pivotCopyParams{1, sizeof(T), 0, 0, 0};
        DataCopyPadExtParams<T> pivotPadParams{false, 0, 0, 0};
        DataCopyPad(matALocal, matAGM_[offset], pivotCopyParams, pivotPadParams);
        SyncPipeMte2ToS();
        T a11 = matALocal.GetValue(0);
        ascendc_assert(a11 > 0.0f && a11 <= MAX_FINITE_FLOAT,
                       "The factorization could not be completed because the input is not positive-definite.");
        invSqrtA11_ = T(1 / sqrt(a11));
    }

    // 核内分块处理，每次处理blockSize大小的数据
    for (uint32_t blockStart = 0; blockStart < matSizeN_; blockStart += blockSize_) {
        if (matrixNumCount_ == 1 && (blockStart / blockSize_) % blockDim_ != blockIdx_) {
            continue;
        }
        uint32_t count = (matSizeN_ - blockStart) > blockSize_ ? blockSize_ : (matSizeN_ - blockStart);

        DataCopyExtParams copyParamsMatALocal{static_cast<uint16_t>(count), sizeof(T),
                                              static_cast<uint32_t>((matSizeN_ - 1) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> padParamsMatALocal{true, 0, BASIC_BLOCK / sizeof(T) - 1, 0};
        DataCopyPad(matALocal, matAGM_[offset + blockStart * matSizeN_], copyParamsMatALocal, padParamsMatALocal);
        SyncPipeMte2ToV();

        // 只在处理第一个元素时计算平方根并存储缩放因子
        if (blockStart == 0 && matrixNumCount_ > 1) {
            SyncPipeMte2ToS();
            T a11 = matALocal.GetValue(0);
            ascendc_assert(a11 > 0.0f && a11 <= MAX_FINITE_FLOAT,
                           "The factorization could not be completed because the input is not positive-definite.");
            invSqrtA11_ = T(1 / sqrt(a11));
        }
        ScaleCholesky(matALocal, static_cast<float>(invSqrtA11_), count * BASIC_BLOCK / sizeof(T));

        SyncPipeVToMte3();
        DataCopyExtParams dataCopyOutParams{static_cast<uint16_t>(count), sizeof(T), 0,
                                            static_cast<uint32_t>((matSizeN_ - 1) * sizeof(T)), 0};
        DataCopyPad(outGM_[offset + blockStart * matSizeN_], matALocal, dataCopyOutParams);
        SyncPipeMte3ToS();
    }

    matAQueue_.FreeTensor(matALocal);
}

// 辅助函数：批量模式列点积（行块 2D 分块搬运 + 批量同步，消除逐行 MTE2→V/V→S 同步）
template <typename T>
__aicore__ inline void Cholesky<T>::ProcessColumnDotProduct(LocalTensor<T>& matLLocal, LocalTensor<T>& matLeftLocal,
                                                            LocalTensor<T>& matRightLocal,
                                                            LocalTensor<T>& matResultLocal, uint32_t index,
                                                            uint64_t offset, uint32_t blockStart, uint32_t count)
{
    const uint32_t slot = BASIC_BLOCK / sizeof(T);
    const uint32_t leftBlockNum = (index + blockSize_ - 1) / blockSize_;

    for (uint32_t leftBlockIdx = 0; leftBlockIdx < leftBlockNum; leftBlockIdx++) {
        const uint32_t leftBlockStart = leftBlockIdx * blockSize_;
        const uint32_t leftBlockSize = (index - leftBlockStart) > blockSize_ ? blockSize_ : (index - leftBlockStart);
        const uint32_t leftPad = AlignToSlot(leftBlockSize);

        // 主元行段 L[index, leftBlockStart..+leftBlockSize)
        DataCopyExtParams copyLeft{1, static_cast<uint32_t>(leftBlockSize * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padLeft{false, 0, 0, 0};
        DataCopyPad(matLeftLocal, outGM_[offset + index * matSizeN_ + leftBlockStart], copyLeft, padLeft);

        uint32_t chunkRows = static_cast<uint32_t>(CHUNK_BUFFER_BYTES / (leftPad * sizeof(T)));
        if (chunkRows == 0) {
            chunkRows = 1;
        }
        for (uint32_t rowBase = 0; rowBase < count; rowBase += chunkRows) {
            const uint32_t rows = count - rowBase > chunkRows ? chunkRows : count - rowBase;
            // 行块 [blockStart+rowBase, +rows) × [leftBlockStart, +leftBlockSize)：一次 2D 搬入
            // （padTail 扩展行距，dstStride=0，禁止叠加）
            DataCopyExtParams copyRight{static_cast<uint16_t>(rows), static_cast<uint32_t>(leftBlockSize * sizeof(T)),
                                        static_cast<uint32_t>((matSizeN_ - leftBlockSize) * sizeof(T)), 0, 0};
            DataCopyPadExtParams<T> padRight{true, 0, static_cast<uint8_t>(leftPad - leftBlockSize), 0};
            DataCopyPad(matRightLocal, outGM_[offset + (index + blockStart + rowBase) * matSizeN_ + leftBlockStart],
                        copyRight, padRight);
            SyncPipeMte2ToV();
            for (uint32_t r = 0; r < rows; ++r) {
                Mul(matResultLocal, matLeftLocal, matRightLocal[r * leftPad], leftBlockSize);
                ReduceSum<T>(matResultLocal, matResultLocal, matRightLocal[r * leftPad], leftBlockSize);
                Add(matLLocal[(rowBase + r) * slot], matLLocal[(rowBase + r) * slot], matResultLocal, slot);
            }
        }
    }
}

// 检查主元后计算缩放因子。
template <typename T>
__aicore__ inline T Cholesky<T>::ComputeScaleFactor(const LocalTensor<T>& matLLocal)
{
    // 调用方 Sub（V 写）与本次标量读取之间同步
    SyncPipeVToS();
    T b1 = matLLocal.GetValue(0);
    ascendc_assert(b1 > 0.0f && b1 <= MAX_FINITE_FLOAT,
                   "The factorization could not be completed because the input is not positive-definite.");
    return T(1 / sqrt(b1));
}

// 辅助函数：批量模式行点积。保持逐列 Mul+ReduceSum 树归约的原始数值序（Add 链在病态
// 输入下累积舍入会使主元变负触发正定性断言），仅将同步从 PipeBarrier 收敛为事件同步
template <typename T>
__aicore__ inline void Cholesky<T>::ProcessRowDotProduct(LocalTensor<T>& matLLocal, LocalTensor<T>& matLeftLocal,
                                                         LocalTensor<T>& matRightLocal, LocalTensor<T>& matResultLocal,
                                                         uint32_t index, uint64_t offset, uint32_t blockStart,
                                                         uint32_t count)
{
    const uint32_t leftBlockNum = (index + blockSize_ - 1) / blockSize_;

    for (uint32_t leftBlockIdx = 0; leftBlockIdx < leftBlockNum; leftBlockIdx++) {
        const uint32_t leftBlockStart = leftBlockIdx * blockSize_;
        const uint32_t leftBlockSize = (index - leftBlockStart) > blockSize_ ? blockSize_ : (index - leftBlockStart);

        // 搬运当前块的matLeftLocal数据（槽位布局，padding×padding=0 保证槽位乘积可整体归约）
        DataCopyExtParams copyParamsLeftLocal{static_cast<uint16_t>(leftBlockSize), sizeof(T),
                                              static_cast<uint32_t>((matSizeN_ - 1) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> padParamsLeftLocal{true, 0, BASIC_BLOCK / sizeof(T) - 1, 0};
        DataCopyPad(matLeftLocal, outGM_[offset + index + leftBlockStart * matSizeN_], copyParamsLeftLocal,
                    padParamsLeftLocal);

        // 对当前块中的每一列，分块搬运matRightLocal并计算
        for (uint32_t colInBlock = 0; colInBlock < count; colInBlock++) {
            uint32_t columnRightPivot = blockStart + colInBlock;

            // 搬运当前列的matRightLocal数据
            DataCopyPad(matRightLocal, outGM_[offset + (index + columnRightPivot) + leftBlockStart * matSizeN_],
                        copyParamsLeftLocal, padParamsLeftLocal);
            SyncPipeMte2ToV();

            Mul(matResultLocal, matLeftLocal, matRightLocal, leftBlockSize * BASIC_BLOCK / sizeof(T));
            ReduceSum<T>(matResultLocal, matResultLocal, matRightLocal, leftBlockSize * BASIC_BLOCK / sizeof(T));
            // V 写与 S 读之间同步：先等向量归约完成，再标量读取累加
            SyncPipeVToS();
            T currentSum = matResultLocal.GetValue(0);
            T existingSum = matLLocal.GetValue(colInBlock);
            matLLocal.SetValue(colInBlock, existingSum + currentSum);
        }
    }
    // S→V 事件同步：标量 SetValue 累加落 UB 后，调用方的向量 Sub/复用才可读取
    SyncPipeSToV();
}

template <typename T>
__aicore__ inline void Cholesky<T>::SecondToNColumn(uint32_t index, uint64_t offset)
{
    LocalTensor<T> matALocal = matAQueue_.AllocTensor<T>();
    LocalTensor<T> matLLocal = matLQueue_.AllocTensor<T>();
    LocalTensor<T> matLeftLocal = matLeftQueue_.AllocTensor<T>();
    LocalTensor<T> matRightLocal = matRightQueue_.AllocTensor<T>();
    LocalTensor<T> matResultLocal = matResultQueue_.AllocTensor<T>();

    // 存储当前列的缩放因子，所有分块共享同一个缩放因子
    T columnScaleFactor = 0.0f;
    if (matrixNumCount_ == 1) {
        DataCopyExtParams pivotCopyParams{1, sizeof(T), 0, 0, 0};
        DataCopyPadExtParams<T> pivotPadParams{false, 0, 0, 0};
        DataCopyPad(matALocal, matAGM_[offset + index * matSizeN_ + index], pivotCopyParams, pivotPadParams);
        SyncPipeMte2ToV();
        Duplicate(matLLocal, zero_, BASIC_BLOCK / sizeof(T));
        ProcessColumnDotProduct(matLLocal, matLeftLocal, matRightLocal, matResultLocal, index, offset, 0, 1);
        Sub(matLLocal, matALocal, matLLocal, BASIC_BLOCK / sizeof(T));
        columnScaleFactor = ComputeScaleFactor(matLLocal);
    }

    // 对当前列的所有元素进行分块处理
    for (uint32_t blockStart = 0; blockStart < (matSizeN_ - index); blockStart += blockSize_) {
        if (matrixNumCount_ == 1 && (blockStart / blockSize_) % blockDim_ != blockIdx_) {
            continue;
        }
        // 计算当前块的大小
        uint32_t count = (matSizeN_ - index - blockStart) > blockSize_ ? blockSize_ : (matSizeN_ - index - blockStart);

        // 搬运当前块的count个A元素
        DataCopyExtParams copyParamsMatALocal{static_cast<uint16_t>(count), sizeof(T),
                                              static_cast<uint32_t>((matSizeN_ - 1) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> padParamsMatALocal{true, 0, BASIC_BLOCK / sizeof(T) - 1, 0};
        DataCopyPad(matALocal, matAGM_[offset + index * matSizeN_ + index + blockStart * matSizeN_],
                    copyParamsMatALocal, padParamsMatALocal);
        SyncPipeMte2ToV();

        // 初始化当前块的L结果为0
        Duplicate(matLLocal, zero_, count * BASIC_BLOCK / sizeof(T));

        // 调用辅助函数处理点积计算
        ProcessColumnDotProduct(matLLocal, matLeftLocal, matRightLocal, matResultLocal, index, offset, blockStart,
                                count);

        // 执行计算操作
        Sub(matLLocal, matALocal, matLLocal, count * BASIC_BLOCK / sizeof(T));

        // 只在第一次分块时计算缩放因子
        if (blockStart == 0 && matrixNumCount_ > 1) {
            columnScaleFactor = ComputeScaleFactor(matLLocal);
        }

        // 对当前块的所有元素应用同一个缩放因子
        ScaleCholesky(matLLocal, static_cast<float>(columnScaleFactor), count * BASIC_BLOCK / sizeof(T));

        // 得到count个L元素并搬出
        SyncPipeVToMte3();
        DataCopyExtParams dataCopyOutParams{static_cast<uint16_t>(count), sizeof(T), 0,
                                            static_cast<uint32_t>((matSizeN_ - 1) * sizeof(T)), 0};
        DataCopyPad(outGM_[offset + index * matSizeN_ + index + blockStart * matSizeN_], matLLocal, dataCopyOutParams);
        SyncPipeMte3ToS();
    }

    // 释放张量资源
    matResultQueue_.FreeTensor(matResultLocal);
    matRightQueue_.FreeTensor(matRightLocal);
    matLeftQueue_.FreeTensor(matLeftLocal);
    matLQueue_.FreeTensor(matLLocal);
    matAQueue_.FreeTensor(matALocal);
}

template <typename T>
__aicore__ inline void Cholesky<T>::InitTriu(GM_ADDR self, GM_ADDR out, GM_ADDR workspace,
                                             const CholeskyTilingData* tilingData, TPipe* pipe, bool cube)
{
    blockIdx_ = GetBlockIdx();
    blockDim_ = GetBlockNum();
#if __NPU_ARCH__ == 3510
    bool mixLaunch = true;
    if ASCEND_IS_AIV {
        mixLaunch = GetTaskRatio() >= 2; // MIX_AIC_1_2 下 AIV 的 GetTaskRatio()==2
    }
    cubeActive_ = cube && mixLaunch;
    if (cubeActive_) {
        // MIX(1:2)：GetBlockNum() 是 AIC 数，AIV 数为 2 倍（GetBlockIdx 在 AIV 上即 0..2*aicNum-1）
        blockDim_ = static_cast<uint32_t>(GetBlockNum()) * 2U;
        cubeNum_ = static_cast<uint32_t>(GetBlockNum());
        cubeIdx_ = static_cast<uint32_t>(GetBlockIdx());
    }
#endif
    GetTilingData(tilingData);
    // 3510：面板算法是唯一路径（批量也走分组并行面板），旧逐列算法仅保留给非 3510 代际
    cooperative_ = true;
    selfBase_ = (__gm__ T*)self;
    outBase_ = (__gm__ T*)out;

    matAGM_.SetGlobalBuffer(selfBase_, matSizeN_ * matSizeN_ * matrixNumCount_);
    outGM_.SetGlobalBuffer(outBase_, matSizeN_ * matSizeN_ * matrixNumCount_);
    // kernel 入参 workspace 是整块 workspace 基址，前 RESERVED_WORKSPACE(3510=16MB) 为系统/lib-API
    // 保留区，必须经 GetUserWorkspace() 偏移后才能放算子私有数据；Host 侧申报 保留区+用户区
    // （blockDim 个 float，32B 对齐）。CPU 仿真下 GetUserWorkspace() 原样返回，故 UT 无法暴露漏偏移。
    workspaceFlagGM_.SetGlobalBuffer((__gm__ T*)GetUserWorkspace(workspace), blockDim_);
    // 用户区布局：[0, blockDim) 保留 | [blockDim, blockDim + blockDim*PANEL_WIDTH) 面板缩放因子
    // 下标 = group_ * CHOLESKY_PANEL_WIDTH + (行在面板内的序号)；host 侧按同一公式申报
    workspaceScaleGM_.SetGlobalBuffer((__gm__ T*)GetUserWorkspace(workspace) + AlignToSlot(blockDim_),
                                      static_cast<uint64_t>(blockDim_) * CHOLESKY_PANEL_WIDTH);

#if __NPU_ARCH__ == 3510
    if (cubeActive_) {
        // Cube 路径 UB（与 InitTril 相同）：阶段 A 列块固定 CUBE_PHASE_COLS(64)
        const uint64_t cubeTileBytes = static_cast<uint64_t>(CUBE_UT_TILE_ELEMS) * sizeof(T);
        const uint64_t cubeSlotBytes = static_cast<uint64_t>(CHOLESKY_PANEL_WIDTH + 1U) * BASIC_BLOCK;
        const uint64_t cubePanelBytes = cubeSlotBytes > cubeTileBytes ? cubeSlotBytes : cubeTileBytes;
        const uint64_t cubeTmpBytes = static_cast<uint64_t>(CUBE_PHASE_COLS) * sizeof(T);
        pipe->InitBuffer(matAQueue_, BUFFER_NUM, VecRegGuardedBytes(cubeTileBytes));
        pipe->InitBuffer(matLQueue_, BUFFER_NUM, VecRegGuardedBytes(cubePanelBytes));
        pipe->InitBuffer(matLeftQueue_, BUFFER_NUM, VecRegGuardedBytes(cubePanelBytes));
        pipe->InitBuffer(matRightQueue_, BUFFER_NUM, VecRegGuardedBytes(cubeTileBytes));
        pipe->InitBuffer(matResultQueue_, BUFFER_NUM, VecRegGuardedBytes(cubeTmpBytes));
        pipe->InitBuffer(cubeUtBuf0_, VecRegGuardedBytes(cubeTileBytes));
        pipe->InitBuffer(cubeUtBuf1_, VecRegGuardedBytes(cubeTileBytes));
        pipe->InitBuffer(cubeUtIdxBuf_, static_cast<uint64_t>(CUBE_UT_TILE_ELEMS) * sizeof(uint32_t));
        if ASCEND_IS_AIC {
            InitCubeBuffers(pipe);
        }
        const uint64_t utElems = static_cast<uint64_t>(CHOLESKY_PANEL_WIDTH) * matSizeN_;
        __gm__ T* userBase = (__gm__ T*)GetUserWorkspace(workspace);
        const uint64_t utOff = AlignToSlot(blockDim_) + static_cast<uint64_t>(blockDim_) * CHOLESKY_PANEL_WIDTH;
        cubeUtGm_.SetGlobalBuffer(userBase + utOff, utElems);
        cubeUtNegGm_.SetGlobalBuffer(userBase + utOff + utElems, utElems);
        return;
    }
    if (cooperative_) {
        // 协同面板路径（仅 3510）：与 InitTril 相同的面板缓冲布局（tril/triu 仅
        // GM 索引不同）。matLeft 需容纳面板 slotted 主元拷贝（CHOLESKY_PANEL_WIDTH × slot 元素）。
        // 非 3510 走原版逐列路径，保持原版批量缓冲布局
        // 字节口径：matA 需容纳 R4_TRAILING_ROW_BLOCK 行 × blockSize 列；matL(matPivot) 需容纳
        // 面板列槽位((P+1)*slot) 或 U 面板列块(P × R4_TRAILING_ROW_BLOCK)，取大者
        const uint64_t aBufferSize = static_cast<uint64_t>(R4_TRAILING_ROW_BLOCK) * CHOLESKY_SCHUR_COLS * sizeof(T);
        const uint64_t panelSlotBytes = static_cast<uint64_t>(CHOLESKY_PANEL_WIDTH + 1U) * BASIC_BLOCK;
        const uint64_t panelTileBytes = static_cast<uint64_t>(CHOLESKY_PANEL_WIDTH) * R4_TRAILING_ROW_BLOCK * sizeof(T);
        const uint64_t panelBufferSize = panelSlotBytes > panelTileBytes ? panelSlotBytes : panelTileBytes;
        const uint64_t rightBufferSize = static_cast<uint64_t>(CHOLESKY_SCHUR_COLS) * CHOLESKY_PANEL_WIDTH * sizeof(T);
        const uint64_t tmpBufferSize = static_cast<uint64_t>(CHOLESKY_SCHUR_COLS) * sizeof(T);
        pipe->InitBuffer(matAQueue_, BUFFER_NUM, VecRegGuardedBytes(aBufferSize));
        pipe->InitBuffer(matLQueue_, BUFFER_NUM, VecRegGuardedBytes(panelBufferSize));
        pipe->InitBuffer(matLeftQueue_, BUFFER_NUM, VecRegGuardedBytes(panelBufferSize));
        pipe->InitBuffer(matRightQueue_, BUFFER_NUM, VecRegGuardedBytes(rightBufferSize));
        pipe->InitBuffer(matResultQueue_, BUFFER_NUM, VecRegGuardedBytes(tmpBufferSize));
        return;
    }
#endif

    // 使用分块大小计算buffer，减少UB内存使用。
    // matLeft/matResult 需容纳逐列槽位乘加（leftBlockSize × slot 元素，原版行点积数值序）
    uint64_t columnBufferSize = VecRegGuardedBytes(blockSize_ * BASIC_BLOCK);
    uint64_t rowBufferSize = VecRegGuardedBytes(CeilDiv(blockSize_ * sizeof(T), BASIC_BLOCK) * BASIC_BLOCK);

    pipe->InitBuffer(matAQueue_, BUFFER_NUM, rowBufferSize);
    pipe->InitBuffer(matLQueue_, BUFFER_NUM, rowBufferSize);
    pipe->InitBuffer(matLeftQueue_, BUFFER_NUM, columnBufferSize);
    pipe->InitBuffer(matRightQueue_, BUFFER_NUM, columnBufferSize);
    pipe->InitBuffer(matResultQueue_, BUFFER_NUM, columnBufferSize);
}

template <typename T>
__aicore__ inline void Cholesky<T>::ProcessTriu()
{
#if __NPU_ARCH__ == 3510
    if (ProcessDiagonalMatrix()) {
        return;
    }
    if (cooperative_) {
        // 协同模式（单矩阵或大维度批量）：上三角右看面板算法，面板间多核并行更新 Schur 补，
        // SyncAll 次数从逐列 M 次降为逐面板 2*ceil(M/CHOLESKY_PANEL_WIDTH) 次。
        // 批量为 0（空 batch）时整段跳过，不产生任何 SyncAll。
        if (matrixNumCount_ == 0) {
            return;
        }
        uint64_t done = 0;
        bool zeroNeeded = false;
        while (done < matrixNumCount_) {
            GetGroupGeometry(matrixNumCount_ - done);
            const uint64_t m = static_cast<uint64_t>(group_) + done;
            done += groupNum_;
            const bool active = group_ < groupNum_ && m < matrixNumCount_;
            if (active) {
                const uint64_t offset = m * matSizeN_ * matSizeN_;
                matAGM_.SetGlobalBuffer(selfBase_ + offset, matSizeN_ * matSizeN_);
                outGM_.SetGlobalBuffer(outBase_ + offset, matSizeN_ * matSizeN_);
            }
            // 仅当某个面板的尾随行块可能出现 rows>=2（对角块整块写出会覆盖严格下三角）时才需要收尾清零；
            // ib = (M-panelEnd)/(groupCores*4) 在首面板取最大，故按 (M-64)/(groupCores*4) >= 2 判定
            if (matSizeN_ > CHOLESKY_PANEL_WIDTH && (matSizeN_ - CHOLESKY_PANEL_WIDTH) / (groupCores_ * 4U) >= 2U) {
                zeroNeeded = true;
            }
            RunUpperPanels(active);
        }
        ZeroLowerTriangles(zeroNeeded);
        return;
    }
#endif
    // 非 3510 / 非协同批量：原版每核一矩阵（单矩阵 offsetPrefix=0），单矩阵逐列 SyncAll
    if (blockIdx_ < blockDim_) {
        auto loopTimes = matrixNumCount_ == 1 ? 0 : matrixNumCount_ / blockDim_;
        for (uint64_t loopIndex = 0; loopIndex <= loopTimes; loopIndex++) {
            uint64_t offsetPrefix = matrixNumCount_ == 1 ? 0 : blockIdx_ + blockDim_ * loopIndex;
            if (offsetPrefix < matrixNumCount_) {
                uint64_t offset = offsetPrefix * matSizeN_ * matSizeN_;
                FirstRow(offset);
                SyncSingleMatrix();
                for (uint32_t index = 1; index < matSizeN_; index++) {
                    SecondToNRow(index, offset);
                    SyncSingleMatrix();
                }
            }
        }
    }
}

// 单矩阵面板分解（上三角）：面板行 [panelStart, panelEnd) 由属主核串行分解，
// 行 j 的点积仅覆盖面板内列 [panelStart, j)，按 k 广播乘加（连续向量运算）
// 只写面板自身列 [j, panelEnd)：尾随段 [panelEnd, matSizeN) 由 UpdatePanelRowTrailing（并行）
// + UpdateTrailingTriu（并行）完成，串行段由 O(P*M) 降为 O(P^2)
template <typename T>
__aicore__ inline void Cholesky<T>::FactorPanelTriu(uint32_t panelStart, uint32_t panelEnd, bool firstPanel,
                                                    bool narrow)
{
    const GlobalTensor<T>& srcGM = firstPanel ? matAGM_ : outGM_;
    LocalTensor<T> matPivot = matLQueue_.AllocTensor<T>();
    LocalTensor<T> matA = matAQueue_.AllocTensor<T>();
    LocalTensor<T> matRight = matRightQueue_.AllocTensor<T>();
    LocalTensor<T> matTmp = matResultQueue_.AllocTensor<T>();
    LocalTensor<T> matScale = matLeftQueue_.AllocTensor<T>(); // 本面板各行 1/U[i,i]（narrow 时导出给阶段 A）
    const uint32_t slot = BASIC_BLOCK / sizeof(T);

    for (uint32_t j = panelStart; j < panelEnd; ++j) {
        const uint32_t width = j - panelStart;
        if (width > 0) {
            // 主元列 out[panelStart..j, j) → 槽位（S 侧逐 k 标量读取）
            DataCopyExtParams pivotCopy{static_cast<uint16_t>(width), sizeof(T),
                                        static_cast<uint32_t>((matSizeN_ - 1) * sizeof(T)), 0, 0};
            DataCopyPadExtParams<T> pivotPad{true, 0, static_cast<uint8_t>(slot - 1), 0};
            DataCopyPad(matPivot, outGM_[static_cast<uint64_t>(panelStart) * matSizeN_ + j], pivotCopy, pivotPad);
            SyncPipeMte2ToS();
        }
        T scaleFactor = one_;
        // narrow: 只算面板自身列（尾随段交给两个全核阶段）；否则保持原「整行尾随」语义
        const uint32_t colLimit = narrow && panelEnd < matSizeN_ ? panelEnd : matSizeN_;
        for (uint32_t colStart = j; colStart < colLimit; colStart += blockSize_) {
            // cols 必须以 colLimit 为界：narrow 下若按 matSizeN_ 取块，会把 [panelEnd, j+blockSize) 一并写掉，
            // 而那些列随后由阶段 A 再乘一次 1/U[i,i]（panel>=1 的 srcGM 就是 outGM）⇒ 重复缩放
            const uint32_t cols = colLimit - colStart > blockSize_ ? blockSize_ : colLimit - colStart;
            const uint32_t colsPad = AlignToSlot(cols);
            // A 行段（连续）→ matA
            DataCopyExtParams copyA{1, static_cast<uint32_t>(cols * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padA{false, 0, 0, 0};
            DataCopyPad(matA, srcGM[static_cast<uint64_t>(j) * matSizeN_ + colStart], copyA, padA);
            if (width > 0) {
                // out[panelStart..j, colStart..+cols)：k 行 × cols 列 2D 搬入（padTail 扩展行距，dstStride=0）
                DataCopyExtParams copyRight{static_cast<uint16_t>(width), static_cast<uint32_t>(cols * sizeof(T)),
                                            static_cast<uint32_t>((matSizeN_ - cols) * sizeof(T)), 0, 0};
                DataCopyPadExtParams<T> padRight{true, 0, static_cast<uint8_t>(colsPad - cols), 0};
                DataCopyPad(matRight, outGM_[static_cast<uint64_t>(panelStart) * matSizeN_ + colStart], copyRight,
                            padRight);
                SyncPipeMte2ToV();
                Duplicate(matTmp, zero_, cols);
                for (uint32_t k = 0; k < width; ++k) {
                    // 主元列拷贝为 4B 块 × padTail 的 2D 布局，每块落位 32B 槽距，需按 slot 距读取
                    T pivotValue = matPivot.GetValue(k * slot);
                    Muls(matRight[k * colsPad], matRight[k * colsPad], static_cast<float>(pivotValue), cols);
                    Add(matTmp, matTmp, matRight[k * colsPad], cols);
                }
                Sub(matA, matA, matTmp, cols);
            } else {
                SyncPipeMte2ToV();
            }
            if (colStart == j) {
                if (width > 0) {
                    SyncPipeVToS();
                } else {
                    SyncPipeMte2ToS();
                }
                T pivot = matA.GetValue(0);
                ascendc_assert(pivot > 0.0f && pivot <= MAX_FINITE_FLOAT,
                               "The factorization could not be completed because the input is not positive-definite.");
                scaleFactor = T(1 / sqrt(pivot));
                if (narrow) {
                    // 逐位导出给阶段 A：那里不再重算 1/sqrt(A[i,i]-Σ)，避免与本次结果差 1 ulp
                    Duplicate(matScale[(j - panelStart) * slot], scaleFactor, 1);
                }
            }
            Muls(matA, matA, static_cast<float>(scaleFactor), cols);
            SyncPipeVToMte3();
            DataCopyExtParams copyOut{1, static_cast<uint32_t>(cols * sizeof(T)), 0, 0, 0};
            DataCopyPad(outGM_[static_cast<uint64_t>(j) * matSizeN_ + colStart], matA, copyOut);
            SyncPipeMte3ToS();
        }
    }

    if (narrow && panelEnd < matSizeN_) {
        // UB→GM 的 srcStride 以 32B 为单位且 srcStride*32+blockLen 再向上对齐 32B ⇒ blockLen=4B 时
        // 传 0 恰好得到 32B 间距（= slot），与 matScale 的 slot 落位一致；dstStride=0 ⇒ GM 侧连续
        SyncPipeVToMte3();
        DataCopyExtParams scaleCopy{static_cast<uint16_t>(panelEnd - panelStart), sizeof(T), 0, 0, 0};
        DataCopyPad(workspaceScaleGM_[static_cast<uint64_t>(group_) * CHOLESKY_PANEL_WIDTH], matScale, scaleCopy);
        SyncPipeMte3ToS();
    }
    matLeftQueue_.FreeTensor(matScale);
    matResultQueue_.FreeTensor(matTmp);
    matRightQueue_.FreeTensor(matRight);
    matAQueue_.FreeTensor(matA);
    matLQueue_.FreeTensor(matPivot);
}

// 阶段 A：面板自身行 i∈[panelStart,panelEnd) 的尾随段 [panelEnd, M)。
// 行方向存在三角依赖（行 i 的 k 累积需要 U[k<i, j]，原实现靠单核升序天然满足），故按「列块」并行：
// 每核认领一个列块，块内对 width 行按 i 升序做一次 64×jc 的三角求解，U[k, j] 直接取自已就地改写的
// ACC 行（不重搬 GM）。列块之间彼此独立，行链长度 = 面板宽度（与 M 无关）。
// 缩放因子 1/U[i,i] 由面板分解经 workspace 逐位导出，故数值序与串行分解完全一致（bit-exact）。
template <typename T>
__aicore__ inline void Cholesky<T>::UpdatePanelRowTrailing(uint32_t panelStart, uint32_t panelEnd, bool firstPanel,
                                                           bool active)
{
    if (panelEnd >= matSizeN_) {
        return;
    }
    const GlobalTensor<T>& srcGM = firstPanel ? matAGM_ : outGM_;
    LocalTensor<T> matScale = matLQueue_.AllocTensor<T>();
    LocalTensor<T> matAcc = matLeftQueue_.AllocTensor<T>();
    LocalTensor<T> matTmp = matResultQueue_.AllocTensor<T>();
    LocalTensor<T> matBuf = matRightQueue_.AllocTensor<T>();
    const uint32_t width = panelEnd - panelStart;
    const uint32_t panelColsPad = AlignToSlot(width);
    // 阶段 A 列块宽：UB 上界 = 64 行 × 128 列（32KB），使 M=8192 时 64 个列块可铺满 72 核
    constexpr uint32_t PHASE_A_COLS = 256U;
    constexpr uint32_t ACC_SPAN = CHOLESKY_PANEL_WIDTH * PHASE_A_COLS;
    LocalTensor<T> accTile = matBuf;
    LocalTensor<T> panelTile = matAQueue_.AllocTensor<T>(); // 面板对角块（64×64）单独占 matA，ACC 独占 matBuf

    // 本面板各行缩放因子 + 面板对角块（各列块共享，只搬一次）
    DataCopyExtParams scaleCopy{1, static_cast<uint32_t>(width * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> scalePad{false, 0, 0, 0};
    DataCopyPad(matScale, workspaceScaleGM_[static_cast<uint64_t>(group_) * CHOLESKY_PANEL_WIDTH], scaleCopy, scalePad);
    DataCopyExtParams panelCopy{static_cast<uint16_t>(width), static_cast<uint32_t>(width * sizeof(T)),
                                static_cast<uint32_t>((matSizeN_ - width) * sizeof(T)), 0, 0};
    DataCopyPadExtParams<T> panelPad{true, 0, static_cast<uint8_t>(panelColsPad - width), 0};
    DataCopyPad(panelTile, outGM_[static_cast<uint64_t>(panelStart) * matSizeN_ + panelStart], panelCopy, panelPad);
    SyncPipeMte2ToS();

    const uint32_t nColBlocks = (matSizeN_ - panelEnd + PHASE_A_COLS - 1) / PHASE_A_COLS;
    for (uint32_t jb = lane_; jb < nColBlocks; jb += groupCores_) {
        const uint32_t j0 = panelEnd + jb * PHASE_A_COLS;
        const uint32_t jc = matSizeN_ - j0 > PHASE_A_COLS ? PHASE_A_COLS : matSizeN_ - j0;
        const uint32_t colsPad = AlignToSlot(jc);
        // A 列块 [panelStart..panelEnd) × [j0..j0+jc)：一次 2D 搬入，随后就地改写为 U
        DataCopyExtParams copyA{static_cast<uint16_t>(width), static_cast<uint32_t>(jc * sizeof(T)),
                                static_cast<uint32_t>((matSizeN_ - jc) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> padA{true, 0, static_cast<uint8_t>(colsPad - jc), 0};
        DataCopyPad(accTile, srcGM[static_cast<uint64_t>(panelStart) * matSizeN_ + j0], copyA, padA);
        SyncPipeMte2ToV();
        for (uint32_t r = 0; r < width; ++r) {
            // 行 i = panelStart + r：acc = Σ_{k<r} U[k,i]·U[k,j]，t 升序、单次 Sub、再乘 1/U[i,i]
            Duplicate(matAcc, zero_, jc);
            for (uint32_t k = 0; k < r; ++k) {
                T pivotValue = panelTile.GetValue(k * panelColsPad + r);
                Muls(matTmp, accTile[k * colsPad], static_cast<float>(pivotValue), jc);
                Add(matAcc, matAcc, matTmp, jc);
            }
            Sub(accTile[r * colsPad], accTile[r * colsPad], matAcc, jc);
            Muls(accTile[r * colsPad], accTile[r * colsPad], matScale.GetValue(r), jc);
        }
        SyncPipeVToMte3();
        if (colsPad == jc) {
            // UB 侧连续（srcStride=0）；GM 侧行距必须是矩阵行距，dstStride=(M-jc)*sizeof(T)
            // （dstStride=0 会退化成 burstLength 行距，把 64 行写到连续地址上）
            DataCopyExtParams copyOut{static_cast<uint16_t>(width), static_cast<uint32_t>(jc * sizeof(T)), 0,
                                      static_cast<uint32_t>((matSizeN_ - jc) * sizeof(T)), 0};
            DataCopyPad(outGM_[static_cast<uint64_t>(panelStart) * matSizeN_ + j0], accTile, copyOut);
        } else {
            for (uint32_t r = 0; r < width; ++r) {
                DataCopyExtParams copyOut{1, static_cast<uint32_t>(jc * sizeof(T)), 0, 0, 0};
                DataCopyPad(outGM_[static_cast<uint64_t>(panelStart + r) * matSizeN_ + j0], accTile[r * colsPad],
                            copyOut);
            }
        }
        SyncPipeMte3ToS();
    }
    matAQueue_.FreeTensor(panelTile);
    matRightQueue_.FreeTensor(matBuf);
    matResultQueue_.FreeTensor(matTmp);
    matLeftQueue_.FreeTensor(matAcc);
    matLQueue_.FreeTensor(matScale);
}

// 阶段 B：尾随行 i∈[panelEnd, matSizeN) 的 Schur 更新（上三角镜像）
// A[i, j] -= Σ_{k∈面板} out[i, k]·out[j, k]（i <= j）
// B0 对角行块逐行（j0 从 i 起，绝不写下三角）；B1 其余行块按 IB 行分块，块内复用 U 面板行块，
// 把原逐行实现的 MTE2 放大（每行重搬 width×jc）降为 1/IB。
template <typename T>
__aicore__ inline void Cholesky<T>::UpdateTrailingTriu(uint32_t panelStart, uint32_t panelEnd, bool firstPanel,
                                                       bool active, bool narrow)
{
    if (panelEnd >= matSizeN_) {
        return;
    }
    // A 行取未更新的 srcGM；U 面板行块取 outGM（阶段 A 刚补齐全部尾随段）
    const GlobalTensor<T>& srcGM = firstPanel ? matAGM_ : outGM_;
    LocalTensor<T> matPivot = matLQueue_.AllocTensor<T>();
    LocalTensor<T> matA = matAQueue_.AllocTensor<T>();
    LocalTensor<T> matAcc = matLeftQueue_.AllocTensor<T>();
    LocalTensor<T> matRight = matRightQueue_.AllocTensor<T>();
    LocalTensor<T> matTmp = matResultQueue_.AllocTensor<T>();
    const uint32_t slot = BASIC_BLOCK / sizeof(T);
    const uint32_t width = panelEnd - panelStart;
    (void)narrow; // 面板行段由调用方按同一 narrow 判据决定是否已由阶段 A 完成

    uint32_t ib = (matSizeN_ - panelEnd) / (groupCores_ * 4U);
    if (ib > R4_TRAILING_ROW_BLOCK) {
        ib = R4_TRAILING_ROW_BLOCK;
    }
    if (ib < 1U) {
        ib = 1U;
    }

    // ---- B0 对角行块 [panelEnd, panelEnd+ib)：逐行，j0 从 i 起（UB 侧始终 32B 对齐）----
    // B0 只覆盖尾随区的对角行块（逐行、j0 从 i 起，绝不写下三角），与 wide/narrow 无关；
    // 面板行 [panelStart,panelEnd) 的尾随段在 narrow 由阶段 A、在 wide 由面板分解完成，两者都不在此更新
    const uint32_t diagEnd = panelEnd + ib < matSizeN_ ? panelEnd + ib : matSizeN_;
    for (uint32_t i = panelEnd; i < diagEnd; ++i) {
        if (!active || (i - panelEnd) % groupCores_ != lane_) {
            continue;
        }
        DataCopyExtParams pivotCopy{static_cast<uint16_t>(width), sizeof(T),
                                    static_cast<uint32_t>((matSizeN_ - 1) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> pivotPad{true, 0, static_cast<uint8_t>(slot - 1), 0};
        DataCopyPad(matPivot, outGM_[static_cast<uint64_t>(panelStart) * matSizeN_ + i], pivotCopy, pivotPad);
        SyncPipeMte2ToS();
        for (uint32_t j0 = i; j0 < matSizeN_; j0 += CHOLESKY_SCHUR_COLS) {
            const uint32_t jc = matSizeN_ - j0 > CHOLESKY_SCHUR_COLS ? CHOLESKY_SCHUR_COLS : matSizeN_ - j0;
            const uint32_t colsPad = AlignToSlot(jc);
            DataCopyExtParams copyA{1, static_cast<uint32_t>(jc * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padA{false, 0, 0, 0};
            DataCopyPad(matA, srcGM[static_cast<uint64_t>(i) * matSizeN_ + j0], copyA, padA);
            DataCopyExtParams copyRight{static_cast<uint16_t>(width), static_cast<uint32_t>(jc * sizeof(T)),
                                        static_cast<uint32_t>((matSizeN_ - jc) * sizeof(T)), 0, 0};
            DataCopyPadExtParams<T> padRight{true, 0, static_cast<uint8_t>(colsPad - jc), 0};
            DataCopyPad(matRight, outGM_[static_cast<uint64_t>(panelStart) * matSizeN_ + j0], copyRight, padRight);
            SyncPipeMte2ToV();
            Duplicate(matAcc, zero_, jc);
            for (uint32_t k = 0; k < width; ++k) {
                T pivotValue = matPivot.GetValue(k * slot);
                Muls(matTmp, matRight[k * colsPad], static_cast<float>(pivotValue), jc);
                Add(matAcc, matAcc, matTmp, jc);
            }
            Sub(matA, matA, matAcc, jc);
            SyncPipeVToMte3();
            DataCopyExtParams copyOut{1, static_cast<uint32_t>(jc * sizeof(T)), 0, 0, 0};
            DataCopyPad(outGM_[static_cast<uint64_t>(i) * matSizeN_ + j0], matA, copyOut);
            SyncPipeMte3ToS();
        }
    }

    // ---- B1 其余尾随行：IB 行块，U 面板行块块内复用 ----
    const uint32_t nBlocks = diagEnd < matSizeN_ ? (matSizeN_ - diagEnd + ib - 1) / ib : 0;
    for (uint32_t blk = 0; blk < nBlocks; ++blk) {
        if (!active || blk % groupCores_ != lane_) {
            continue;
        }
        const uint32_t i0 = diagEnd + blk * ib;
        const uint32_t rows = matSizeN_ - i0 > ib ? ib : matSizeN_ - i0;
        const uint32_t rowsPad = AlignToSlot(rows);
        DataCopyExtParams uiCopy{static_cast<uint16_t>(width), static_cast<uint32_t>(rows * sizeof(T)),
                                 static_cast<uint32_t>((matSizeN_ - rows) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> uiPad{true, 0, static_cast<uint8_t>(rowsPad - rows), 0};
        DataCopyPad(matPivot, outGM_[static_cast<uint64_t>(panelStart) * matSizeN_ + i0], uiCopy, uiPad);
        SyncPipeMte2ToS();
        for (uint32_t j0 = i0; j0 < matSizeN_; j0 += CHOLESKY_SCHUR_COLS) {
            const uint32_t jc = matSizeN_ - j0 > CHOLESKY_SCHUR_COLS ? CHOLESKY_SCHUR_COLS : matSizeN_ - j0;
            const uint32_t colsPad = AlignToSlot(jc);
            DataCopyExtParams copyA{static_cast<uint16_t>(rows), static_cast<uint32_t>(jc * sizeof(T)),
                                    static_cast<uint32_t>((matSizeN_ - jc) * sizeof(T)), 0, 0};
            DataCopyPadExtParams<T> padA{true, 0, static_cast<uint8_t>(colsPad - jc), 0};
            DataCopyPad(matA, srcGM[static_cast<uint64_t>(i0) * matSizeN_ + j0], copyA, padA);
            DataCopyExtParams copyRight{static_cast<uint16_t>(width), static_cast<uint32_t>(jc * sizeof(T)),
                                        static_cast<uint32_t>((matSizeN_ - jc) * sizeof(T)), 0, 0};
            DataCopyPadExtParams<T> padRight{true, 0, static_cast<uint8_t>(colsPad - jc), 0};
            DataCopyPad(matRight, outGM_[static_cast<uint64_t>(panelStart) * matSizeN_ + j0], copyRight, padRight);
            SyncPipeMte2ToV();
            for (uint32_t r = 0; r < rows; ++r) {
                Duplicate(matAcc, zero_, jc);
                for (uint32_t k = 0; k < width; ++k) {
                    T pivotValue = matPivot.GetValue(k * rowsPad + r);
                    Muls(matTmp, matRight[k * colsPad], static_cast<float>(pivotValue), jc);
                    Add(matAcc, matAcc, matTmp, jc);
                }
                Sub(matA[r * colsPad], matA[r * colsPad], matAcc, jc);
            }
            SyncPipeVToMte3();
            for (uint32_t r = 0; r < rows; ++r) {
                DataCopyExtParams copyOut{1, static_cast<uint32_t>(jc * sizeof(T)), 0, 0, 0};
                DataCopyPad(outGM_[static_cast<uint64_t>(i0 + r) * matSizeN_ + j0], matA[r * colsPad], copyOut);
            }
            SyncPipeMte3ToS();
        }
    }

    matResultQueue_.FreeTensor(matTmp);
    matRightQueue_.FreeTensor(matRight);
    matLeftQueue_.FreeTensor(matAcc);
    matAQueue_.FreeTensor(matA);
    matLQueue_.FreeTensor(matPivot);
}

#if __NPU_ARCH__ == 3510
// ===========================================================================
// Cube（AIC）Schur 更新：out[i,j] -= Σ_{k<64} U[k,i]·U[k,j]（i <= j，i,j ∈ [panelEnd, M)）
//
// 数据流：
//   阶段 A（AIV，列块 64）已把 U 面板行写回 outGM，并就地转置出
//     UT[j][k]  =  U[k,j]        （UT  workspace，稠密 L×64）
//     UTn[j][k] = -U[k,j]        （UTn workspace，供 A 侧做减法：atomic add 只能加）
//   AIC：A = UTn 行块 [i0, i0+64)，B = UT 行块 [j0, j0+nc)
//        ⇒ C' = A·Bᵀ = Σ_k (-U[k,i])·U[k,j] = -C，Fixpipe 以 SetAtomicAdd 写回 outGM
//        ⇒ out = A - C。对角块一并整块写（j<i 部分写脏，见下）
//   对角块整块 atomic add 会把严格下三角写成累计脏值；该区域此后只被
//   ZeroLowerTriangles（upper=true）或 CholeskyTransposeSwapVf（upper=false 时
//   下三角整体被上三角覆盖）抹掉，算法自身只读 j>=i 区域，故不影响正确性。
// ===========================================================================

// 面板行 [panelStart,panelEnd) 的尾随段 + 转置导出（AIV）。
// 数值序与 UpdatePanelRowTrailing 完全一致（k 升序 Muls→Add 累积、单次 Sub、再 Muls）
// ⇒ 面板 U 值逐位不变；只有后续 Schur 更新的 K 维归约序被 Cube 替换。
template <typename T>
__aicore__ inline void Cholesky<T>::UpdatePanelRowTrailingCube(uint32_t panelStart, uint32_t panelEnd, bool firstPanel)
{
    if (panelEnd >= matSizeN_) {
        return;
    }
    const GlobalTensor<T>& srcGM = firstPanel ? matAGM_ : outGM_;
    LocalTensor<T> matScale = matLQueue_.AllocTensor<T>();
    LocalTensor<T> matAcc = matLeftQueue_.AllocTensor<T>();
    LocalTensor<T> matTmp = matResultQueue_.AllocTensor<T>();
    LocalTensor<T> accTile = matRightQueue_.AllocTensor<T>();
    LocalTensor<T> panelTile = matAQueue_.AllocTensor<T>();
    // cube 路径专用 UB：UT / UTn 转置块 + 转置索引（每块 64×64）
    LocalTensor<T> utTile = cubeUtBuf0_.GetWithOffset<T>(CUBE_UT_TILE_ELEMS, 0);
    LocalTensor<T> utNegTile = cubeUtBuf1_.GetWithOffset<T>(CUBE_UT_TILE_ELEMS, 0);
    LocalTensor<uint32_t> utIdx = cubeUtIdxBuf_.GetWithOffset<uint32_t>(CUBE_UT_TILE_ELEMS, 0);

    const uint32_t width = panelEnd - panelStart;
    const uint32_t panelColsPad = AlignToSlot(width);

    DataCopyExtParams scaleCopy{1, static_cast<uint32_t>(width * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> scalePad{false, 0, 0, 0};
    DataCopyPad(matScale, workspaceScaleGM_[static_cast<uint64_t>(group_) * CHOLESKY_PANEL_WIDTH], scaleCopy, scalePad);
    DataCopyExtParams panelCopy{static_cast<uint16_t>(width), static_cast<uint32_t>(width * sizeof(T)),
                                static_cast<uint32_t>((matSizeN_ - width) * sizeof(T)), 0, 0};
    DataCopyPadExtParams<T> panelPad{true, 0, static_cast<uint8_t>(panelColsPad - width), 0};
    DataCopyPad(panelTile, outGM_[static_cast<uint64_t>(panelStart) * matSizeN_ + panelStart], panelCopy, panelPad);
    SyncPipeMte2ToS();

    const uint32_t nColBlocks = (matSizeN_ - panelEnd + CUBE_PHASE_COLS - 1) / CUBE_PHASE_COLS;
    for (uint32_t jb = lane_; jb < nColBlocks; jb += groupCores_) {
        const uint32_t j0 = panelEnd + jb * CUBE_PHASE_COLS;
        const uint32_t jc = matSizeN_ - j0 > CUBE_PHASE_COLS ? CUBE_PHASE_COLS : matSizeN_ - j0;
        const uint32_t colsPad = AlignToSlot(jc);
        DataCopyExtParams copyA{static_cast<uint16_t>(width), static_cast<uint32_t>(jc * sizeof(T)),
                                static_cast<uint32_t>((matSizeN_ - jc) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> padA{true, 0, static_cast<uint8_t>(colsPad - jc), 0};
        DataCopyPad(accTile, srcGM[static_cast<uint64_t>(panelStart) * matSizeN_ + j0], copyA, padA);
        SyncPipeMte2ToV();
        for (uint32_t r = 0; r < width; ++r) {
            Duplicate(matAcc, zero_, jc);
            for (uint32_t k = 0; k < r; ++k) {
                T pivotValue = panelTile.GetValue(k * panelColsPad + r);
                Muls(matTmp, accTile[k * colsPad], static_cast<float>(pivotValue), jc);
                Add(matAcc, matAcc, matTmp, jc);
            }
            Sub(accTile[r * colsPad], accTile[r * colsPad], matAcc, jc);
            Muls(accTile[r * colsPad], accTile[r * colsPad], matScale.GetValue(r), jc);
        }
        SyncPipeVToMte3();
        DataCopyExtParams copyOut{static_cast<uint16_t>(width), static_cast<uint32_t>(jc * sizeof(T)), 0,
                                  static_cast<uint32_t>((matSizeN_ - jc) * sizeof(T)), 0};
        DataCopyPad(outGM_[static_cast<uint64_t>(panelStart) * matSizeN_ + j0], accTile, copyOut);
        if (jc == CUBE_PHASE_COLS && colsPad == CUBE_PHASE_COLS) {
            // 转置导出：UT[i][k] = accTile[k][i]（Gather 索引 = (p%64)*64 + p/64）。
            // 行号必须是**相对尾随区原点 panelEnd** 的偏移（与 CubeSchurPanel 的 i0/j0 同一坐标系），
            // 用绝对列号 j0 会整体错位 panelEnd*64 个元素 ⇒ Cube 读到上一面板的陈旧 -Uᵀ
            const uint32_t rel = j0 - panelEnd;
            // 标量转置：UT[i][k] = accTile[k][i]。Gather 在本平台的 Reg 微指令语义未被证实
            // （实测只填出前 128 个元素，其余为垃圾），改为纯标量循环，语义确定。
            SyncPipeVToS();
            for (uint32_t i = 0; i < CUBE_PHASE_COLS; ++i) {
                for (uint32_t k = 0; k < CUBE_BASE_K; ++k) {
                    utTile.SetValue(i * CUBE_BASE_K + k, accTile.GetValue(k * CUBE_PHASE_COLS + i));
                }
            }
            SetFlag<HardEvent::S_V>(CUBE_EVENT_ID);
            WaitFlag<HardEvent::S_V>(CUBE_EVENT_ID);
            Muls(utNegTile, utTile, static_cast<float>(-1.0), CUBE_UT_TILE_ELEMS);
            SyncPipeVToMte3();
            DataCopyExtParams utCopy{1, static_cast<uint32_t>(CUBE_UT_TILE_ELEMS * sizeof(T)), 0, 0, 0};
            DataCopyPad(cubeUtGm_[static_cast<uint64_t>(rel) * CUBE_BASE_K], utTile, utCopy);
            DataCopyPad(cubeUtNegGm_[static_cast<uint64_t>(rel) * CUBE_BASE_K], utNegTile, utCopy);
        }
        SyncPipeMte3ToS();
    }
    matAQueue_.FreeTensor(panelTile);
    matRightQueue_.FreeTensor(accTile);
    matResultQueue_.FreeTensor(matTmp);
    matLeftQueue_.FreeTensor(matAcc);
    matLQueue_.FreeTensor(matScale);
}

// AIC 侧 L1/L0 缓冲（仅每 AIC 单份；K=64 一次进 L0，暂不做 ping-pong）
template <typename T>
__aicore__ inline void Cholesky<T>::InitCubeBuffers(TPipe* pipe)
{
    pipe->InitBuffer(cubeL1ABuf_, static_cast<uint64_t>(CUBE_L1_A_ELEMS) * sizeof(T));
    pipe->InitBuffer(cubeL1BBuf_, static_cast<uint64_t>(CUBE_L1_B_ELEMS) * sizeof(T));
    pipe->InitBuffer(cubeL0ABuf_, static_cast<uint64_t>(CUBE_L1_A_ELEMS) * sizeof(T));
    pipe->InitBuffer(cubeL0BBuf_, static_cast<uint64_t>(CUBE_L1_B_ELEMS) * sizeof(T));
    pipe->InitBuffer(cubeL0CBuf_, static_cast<uint64_t>(CUBE_L0C_ELEMS) * sizeof(T));
}

// GM(稠密 (rows,64) 行主序) → L1 NZ
// Nd2NzParams 取值与本仓 math/add_lora/op_kernel/add_lora_base.h:340-356 及
// asc/impl/adv_api/detail/matmul/stage/copy_cube_in/copy_tile_to_cube/data_copy_wrapper_nd.h:99 一致
template <typename T>
__aicore__ inline void Cholesky<T>::CubeUtCopyIn(const GlobalTensor<T>& src, uint32_t rowOffset, uint32_t rows,
                                                 const LocalTensor<T>& l1Dst)
{
    Nd2NzParams params{1,
                       static_cast<uint16_t>(rows),
                       static_cast<uint32_t>(CUBE_BASE_K),
                       0,
                       static_cast<uint64_t>(CUBE_BASE_K),
                       static_cast<uint16_t>(rows),
                       1,
                       0};
    DataCopy(l1Dst, src[static_cast<uint64_t>(rowOffset) * CUBE_BASE_K], params);
}

// L1(Nz, (m,64)) → L0A：3510 上 fp32 matmul 的 L1→L0 实际走 Load2Dv2（LoadData3D 的 FMatrix
// SPR 只在 V2 结构体内部按 l1H/l1W 设置；V2Pro 无这两个字段，外部 SetFmatrix 前 SPR 未初始化 ⇒
// MTE1 整块不搬、L0C 恒 0）。逐行依据：
//   asc/impl/adv_api/detail/matmul/stage/split/load_to_l0a/load_to_l0a_load2dV2.h:142-155
//   （非转置分支 fp32：mStartPosition/kStartPosition 强制 0；mStep=CeilDiv(madM,16)、
//     kStep=CeilDiv(madK,8)、srcStride=CeilDiv(aL1M,16)、dstStride=CeilDiv(madM,16)、ifTranspose=false）
// 以及 asc-devkit 官方样例 load_data_2dv2_l12l0 场景 5（float, A=[m,k], B=[n,k]）。
template <typename T>
__aicore__ inline void Cholesky<T>::CubeLoadA(const LocalTensor<T>& l1Src, uint32_t mExt, const LocalTensor<T>& l0Dst)
{
    LoadData2DParamsV2 param;
    param.mStartPosition = 0;
    param.kStartPosition = 0;
    param.mStep = static_cast<uint16_t>(mExt / 16U);
    param.kStep = static_cast<uint16_t>(CUBE_BASE_K / 8U);
    param.srcStride = static_cast<uint16_t>(mExt / 16U);
    param.dstStride = static_cast<uint16_t>(mExt / 16U);
    param.ifTranspose = false;
    param.sid = 0;
    LoadData(l0Dst, l1Src, param);
}

// L1(Nz, (n,64)) → L0B：同一 2D 通道（L1 的 NZ(n,k) 与 L0B 所需 Zn 字节等价）。
// 依据 load_to_l0b_load2dV2.h:95-156 非转置分支（B 输入已按 [n,k] 存 GM）：
//   mStep=CeilDiv(nAlignL1,16)、kStep=CeilDiv(kbAlignL1,8)、
//   srcStride=CeilDiv(nAlignL1,16)、dstStride=CeilDiv(nAlignL0,16)、ifTranspose=false
template <typename T>
__aicore__ inline void Cholesky<T>::CubeLoadB(const LocalTensor<T>& l1Src, uint32_t nExt, const LocalTensor<T>& l0Dst)
{
    LoadData2DParamsV2 param;
    param.mStartPosition = 0;
    param.kStartPosition = 0;
    param.mStep = static_cast<uint16_t>(nExt / 16U);
    param.kStep = static_cast<uint16_t>(CUBE_BASE_K / 8U);
    param.srcStride = static_cast<uint16_t>(nExt / 16U);
    param.dstStride = static_cast<uint16_t>(nExt / 16U);
    param.ifTranspose = false;
    param.sid = 0;
    LoadData(l0Dst, l1Src, param);
}

// AIC：本面板尾随区（L = M - panelEnd）的 Schur 更新，按行块轮转分给各 AIC
template <typename T>
__aicore__ inline void Cholesky<T>::CubeSchurPanel(uint32_t panelStart, uint32_t panelEnd)
{
    (void)panelStart;
    const uint32_t base = panelEnd;
    const uint32_t lTrail = matSizeN_ - base;
    if (lTrail == 0U || cubeNum_ == 0U) {
        return;
    }
    LocalTensor<T> l1A = cubeL1ABuf_.Get<T>();
    LocalTensor<T> l1B = cubeL1BBuf_.Get<T>();
    LocalTensor<T> l0A = cubeL0ABuf_.Get<T>();
    LocalTensor<T> l0B = cubeL0BBuf_.Get<T>();
    LocalTensor<float> l0C = cubeL0CBuf_.Get<float>();
    const uint32_t nRowBlocks = lTrail / CUBE_BASE_M; // cube 路径要求 M % 64 == 0
    for (uint32_t ri = cubeIdx_; ri < nRowBlocks; ri += cubeNum_) {
        const uint32_t i0 = ri * CUBE_BASE_M;
        CubeUtCopyIn(cubeUtNegGm_, i0, CUBE_BASE_M, l1A);
        SetFlag<HardEvent::MTE2_MTE1>(CUBE_EVENT_ID);
        WaitFlag<HardEvent::MTE2_MTE1>(CUBE_EVENT_ID);
        CubeLoadA(l1A, CUBE_BASE_M, l0A);
        SetFlag<HardEvent::MTE1_M>(CUBE_EVENT_ID);
        WaitFlag<HardEvent::MTE1_M>(CUBE_EVENT_ID);
        for (uint32_t j0 = i0; j0 < lTrail; j0 += CUBE_BASE_N) {
            const uint32_t nc = lTrail - j0 > CUBE_BASE_N ? CUBE_BASE_N : lTrail - j0;
            CubeUtCopyIn(cubeUtGm_, j0, nc, l1B);
            SetFlag<HardEvent::MTE2_MTE1>(CUBE_EVENT_ID);
            WaitFlag<HardEvent::MTE2_MTE1>(CUBE_EVENT_ID);
            CubeLoadB(l1B, nc, l0B);
            SetFlag<HardEvent::MTE1_M>(CUBE_EVENT_ID);
            WaitFlag<HardEvent::MTE1_M>(CUBE_EVENT_ID);
            MmadParams mmadParams;
            mmadParams.m = static_cast<uint16_t>(CUBE_BASE_M);
            mmadParams.n = static_cast<uint16_t>(nc);
            mmadParams.k = static_cast<uint16_t>(CUBE_BASE_K);
            mmadParams.cmatrixSource = false;
            mmadParams.cmatrixInitVal = true;
            Mmad(l0C, l0A, l0B, mmadParams);
            SetFlag<HardEvent::M_FIX>(CUBE_EVENT_ID);
            WaitFlag<HardEvent::M_FIX>(CUBE_EVENT_ID);
            FixpipeParamsV220 fixpipeParams(static_cast<uint16_t>(nc), static_cast<uint16_t>(CUBE_BASE_M),
                                            static_cast<uint16_t>(CUBE_BASE_M), static_cast<uint32_t>(matSizeN_),
                                            false);
            fixpipeParams.quantPre = QuantMode_t::NoQuant;
            SetAtomicAdd<float>();
            Fixpipe(outGM_[static_cast<uint64_t>(base + i0) * matSizeN_ + base + j0], l0C, fixpipeParams);
            SetAtomicNone();
            SetFlag<HardEvent::FIX_MTE2>(CUBE_EVENT_ID);
            WaitFlag<HardEvent::FIX_MTE2>(CUBE_EVENT_ID);
        }
    }
}

// 面板循环（AIC/AIV 共享同一份控制流 ⇒ SyncAll<false> 到达次数两侧恒等）
// 每面板 3 次全核屏障：B1(面板分解完成) → 阶段A + 转置导出 → B2 → Cube Schur → B3
template <typename T>
__aicore__ inline void Cholesky<T>::RunUpperPanelsCube()
{
    // 单矩阵 = 一个组：groupCores_ = AIV 核数、lane_ = AIV 序号。
    // 缺这句会让 groupCores_ 保持默认 1 ⇒ 所有 AIV 都跑全部列块/都当面板属主（72 倍冗余）
    GetGroupGeometry(1);
    for (uint32_t panelStart = 0; panelStart < matSizeN_; panelStart += CHOLESKY_PANEL_WIDTH) {
        const uint32_t panelEnd = panelStart + CHOLESKY_PANEL_WIDTH < matSizeN_ ? panelStart + CHOLESKY_PANEL_WIDTH :
                                                                                  matSizeN_;
        const bool firstPanel = panelStart == 0;
        bool owner = false;
        if ASCEND_IS_AIV {
            owner = lane_ == (panelStart / CHOLESKY_PANEL_WIDTH) % groupCores_;
        }
        if (owner) {
            FactorPanelTriu(panelStart, panelEnd, firstPanel, panelEnd < matSizeN_);
        }
        AscendC::SyncAll<false>();
        if (panelEnd >= matSizeN_) {
            continue;
        }
        if ASCEND_IS_AIV {
            UpdatePanelRowTrailingCube(panelStart, panelEnd, firstPanel);
        }
        AscendC::SyncAll<false>();
        if ASCEND_IS_AIC {
            CubeSchurPanel(panelStart, panelEnd);
        }
        AscendC::SyncAll<false>();
    }
}

// UT[i][k] = accTile[k][i]（64×64 块转置）的 Gather 索引：idx[p] = (p%64)*64 + p/64
// 与 jc 无关（cube 路径阶段 A 列块恒为 CUBE_PHASE_COLS==CUBE_BASE_M==64），只建一次
template <typename T>
__aicore__ inline void Cholesky<T>::InitUtTransposeIndex()
{
    LocalTensor<uint32_t> idx = cubeUtIdxBuf_.GetWithOffset<uint32_t>(CUBE_UT_TILE_ELEMS, 0);
    // 4096 个索引只在启动时算一次（cube 路径 M>=2048，~5us 可忽略）；走标量写避免
    // Ands/Shift 在 dav_3510 上不支持 int32 的实例化限制
    for (uint32_t p = 0; p < CUBE_UT_TILE_ELEMS; ++p) {
        idx.SetValue(p, (p % CUBE_BASE_M) * CUBE_BASE_M + (p / CUBE_BASE_M));
    }
    SetFlag<HardEvent::S_V>(CUBE_EVENT_ID);
    WaitFlag<HardEvent::S_V>(CUBE_EVENT_ID);
}

// outGM 上三角 = 输入 A 上三角（cube 路径所有面板统一读 outGM 后 atomic add）
template <typename T>
__aicore__ inline void Cholesky<T>::InitUpperTriangleFromInput()
{
    LocalTensor<T> buf = matRightQueue_.AllocTensor<T>();
    const uint32_t chunk = CUBE_UT_TILE_ELEMS;
    for (uint32_t i = blockIdx_; i < matSizeN_; i += blockDim_) {
        for (uint32_t c = i; c < matSizeN_; c += chunk) {
            const uint32_t n = matSizeN_ - c > chunk ? chunk : matSizeN_ - c;
            DataCopyExtParams params{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(buf, matAGM_[static_cast<uint64_t>(i) * matSizeN_ + c], params, padParams);
            SetFlag<HardEvent::MTE2_MTE3>(CUBE_EVENT_ID);
            WaitFlag<HardEvent::MTE2_MTE3>(CUBE_EVENT_ID);
            DataCopyPad(outGM_[static_cast<uint64_t>(i) * matSizeN_ + c], buf, params);
            SetFlag<HardEvent::MTE3_MTE2>(CUBE_EVENT_ID);
            WaitFlag<HardEvent::MTE3_MTE2>(CUBE_EVENT_ID);
        }
    }
    matRightQueue_.FreeTensor(buf);
}

template <typename T>
__aicore__ inline void Cholesky<T>::ProcessTriuCube()
{
    if (ProcessDiagonalMatrix()) {
        return;
    }
    if (matrixNumCount_ == 0) {
        return;
    }
    if ASCEND_IS_AIV {
        InitUtTransposeIndex();
        InitUpperTriangleFromInput();
    }
    AscendC::SyncAll<false>();
    RunUpperPanelsCube();
    if ASCEND_IS_AIV {
        // 对角 tile 整块 atomic add 会写脏严格下三角 ⇒ upper=true 恒需收尾清零
        ZeroLowerTriangles(true);
    }
}

template <typename T>
__aicore__ inline void Cholesky<T>::ProcessTrilCube()
{
    if (ProcessDiagonalMatrix()) {
        return;
    }
    if (matrixNumCount_ == 0) {
        return;
    }
    if ASCEND_IS_AIV {
        InitUtTransposeIndex();
        InitUpperTriangleFromInput();
    }
    AscendC::SyncAll<false>();
    RunUpperPanelsCube();
    if ASCEND_IS_AIV {
        // 下三角 = 上三角的逐 tile 转置（见 TransposeUpperToLowerTiles 的注释）
        TransposeUpperToLowerTiles();
    }
}
#endif

template <typename T>
__aicore__ inline void Cholesky<T>::FirstRow(uint64_t offset)
{
    LocalTensor<T> matALocal = matAQueue_.AllocTensor<T>();
    if (matrixNumCount_ == 1) {
        // 单元素 DataCopyPad 搬入首主元后经 MTE2_S 同步再标量读取，避免逐元素 GM 访问
        DataCopyExtParams pivotCopyParams{1, sizeof(T), 0, 0, 0};
        DataCopyPadExtParams<T> pivotPadParams{false, 0, 0, 0};
        DataCopyPad(matALocal, matAGM_[offset], pivotCopyParams, pivotPadParams);
        SyncPipeMte2ToS();
        T a11 = matALocal.GetValue(0);
        ascendc_assert(a11 > 0.0f && a11 <= MAX_FINITE_FLOAT,
                       "The factorization could not be completed because the input is not positive-definite.");
        invSqrtA11_ = T(1 / sqrt(a11));
    }

    // 核内分块处理，每次处理blockSize大小的数据
    for (uint32_t blockStart = 0; blockStart < matSizeN_; blockStart += blockSize_) {
        if (matrixNumCount_ == 1 && (blockStart / blockSize_) % blockDim_ != blockIdx_) {
            continue;
        }
        uint32_t count = (matSizeN_ - blockStart) > blockSize_ ? blockSize_ : (matSizeN_ - blockStart);

        DataCopyExtParams copyParamsMatALocal{1, static_cast<uint32_t>(sizeof(T) * count), 0, 0, 0};
        DataCopyPadExtParams<T> padParamsMatALocal{false, 0, 0, 0};
        DataCopyPad(matALocal, matAGM_[offset + blockStart], copyParamsMatALocal, padParamsMatALocal);
        SyncPipeMte2ToV();

        // 只在处理第一个元素时计算平方根并存储缩放因子
        if (blockStart == 0 && matrixNumCount_ > 1) {
            SyncPipeMte2ToS();
            T a11 = matALocal.GetValue(0);
            ascendc_assert(a11 > 0.0f && a11 <= MAX_FINITE_FLOAT,
                           "The factorization could not be completed because the input is not positive-definite.");
            invSqrtA11_ = T(1 / sqrt(a11));
        }
        ScaleCholesky(matALocal, static_cast<float>(invSqrtA11_), count);

        // 搬出当前块的结果
        SyncPipeVToMte3();
        DataCopyExtParams dataCopyOutParams{1, static_cast<uint32_t>(sizeof(T) * count), 0, 0, 0};
        DataCopyPad(outGM_[offset + blockStart], matALocal, dataCopyOutParams);
        SyncPipeMte3ToS();
    }

    matAQueue_.FreeTensor(matALocal);
}

template <typename T>
__aicore__ inline void Cholesky<T>::SecondToNRow(uint32_t index, uint64_t offset)
{
    LocalTensor<T> matALocal = matAQueue_.AllocTensor<T>();
    LocalTensor<T> matLLocal = matLQueue_.AllocTensor<T>();
    LocalTensor<T> matLeftLocal = matLeftQueue_.AllocTensor<T>();
    LocalTensor<T> matRightLocal = matRightQueue_.AllocTensor<T>();
    LocalTensor<T> matResultLocal = matResultQueue_.AllocTensor<T>();

    // 存储当前行的缩放因子，所有分块共享同一个缩放因子
    T rowScaleFactor = 0.0f;
    if (matrixNumCount_ == 1) {
        DataCopyExtParams pivotCopyParams{1, sizeof(T), 0, 0, 0};
        DataCopyPadExtParams<T> pivotPadParams{false, 0, 0, 0};
        DataCopyPad(matALocal, matAGM_[offset + index * matSizeN_ + index], pivotCopyParams, pivotPadParams);
        SyncPipeMte2ToV();
        Duplicate(matLLocal, zero_, BASIC_BLOCK / sizeof(T));
        ProcessRowDotProduct(matLLocal, matLeftLocal, matRightLocal, matResultLocal, index, offset, 0, 1);
        Sub(matLLocal, matALocal, matLLocal, 1);
        rowScaleFactor = ComputeScaleFactor(matLLocal);
    }

    // 对当前行的所有元素进行分块处理
    for (uint32_t blockStart = 0; blockStart < (matSizeN_ - index); blockStart += blockSize_) {
        if (matrixNumCount_ == 1 && (blockStart / blockSize_) % blockDim_ != blockIdx_) {
            continue;
        }
        // 计算当前块的大小
        uint32_t count = (matSizeN_ - index - blockStart) > blockSize_ ? blockSize_ : (matSizeN_ - index - blockStart);

        // 搬运当前块的count个A元素
        DataCopyExtParams copyParamsMatALocal{1, static_cast<uint32_t>(sizeof(T) * count), 0, 0, 0};
        DataCopyPadExtParams<T> padParamsMatALocal{false, 0, 0, 0};
        DataCopyPad(matALocal, matAGM_[offset + index * matSizeN_ + index + blockStart], copyParamsMatALocal,
                    padParamsMatALocal);
        SyncPipeMte2ToV();

        // 初始化当前块的L结果为0
        Duplicate(matLLocal, zero_, count);

        // 调用辅助函数处理点积计算
        ProcessRowDotProduct(matLLocal, matLeftLocal, matRightLocal, matResultLocal, index, offset, blockStart, count);

        // 执行计算操作
        Sub(matLLocal, matALocal, matLLocal, count);

        // 只在第一次分块时计算缩放因子
        if (blockStart == 0 && matrixNumCount_ > 1) {
            rowScaleFactor = ComputeScaleFactor(matLLocal);
        }

        // 对当前块的所有元素应用同一个缩放因子
        ScaleCholesky(matLLocal, static_cast<float>(rowScaleFactor), count);

        // 得到count个L元素并搬出
        SyncPipeVToMte3();
        DataCopyExtParams dataCopyOutParams{1, static_cast<uint32_t>(sizeof(T) * count), 0, 0, 0};
        DataCopyPad(outGM_[offset + index * matSizeN_ + index + blockStart], matLLocal, dataCopyOutParams);
        SyncPipeMte3ToS();
    }

    // 释放张量资源
    matResultQueue_.FreeTensor(matResultLocal);
    matRightQueue_.FreeTensor(matRightLocal);
    matLeftQueue_.FreeTensor(matLeftLocal);
    matLQueue_.FreeTensor(matLLocal);
    matAQueue_.FreeTensor(matALocal);
}

} // namespace Cholesky
#endif
