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
 * \file test_space_to_batch_nd.cpp
 * \brief kernel UT for SpaceToBatchND (ascend950 / arch35, CPU simulation)
 *
 * 覆盖刚修复的“固定外层轴落在纯 padding 时 CopyInKey3/4/5Plus 越界读”问题：
 *   - Key3   (M=3, rank=5, x=[1,4,2,2,1],     block=[2,2,1],   pads0=[1,1])
 *   - Key4   (M=4, rank=6, x=[1,4,2,2,2,1],   block=[2,2,1,1], pads0=[1,1])
 *   - Key5Plus(M=5, rank=7, x=[1,4,2,2,2,2,1], block=[2,2,1,1,1], pads0=[1,1])
 *
 * 三个用例都令 ubAxis=2、ubFactor=outShape[2]=1、totalCount=12、numBlocks=1、
 * perCoreCount=totalCount、bufferSize=4096。第一个空间维 padTop=padBottom=1，
 * 因此部分 tile 的固定外层轴（axis1）实际输入坐标 inPos=-1（完全落在 padding），
 * 修复前 CopyInKey3/4 会继续以负偏移发起搬运，修复后应跳过该 tile 且输出全 0。
 * 其余 tile 为正常数据搬运；全部输出都与逐元素参考实现精确比较。
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "tikicpulib.h"
// 先完整引入 AscendC 接口实现，避免下面的测试兼容层宏影响框架内部头文件。
#include "kernel_operator.h"

// ---------------------------------------------------------------------------
// CPU 仿真器兼容层：当前 CANN CPU 仿真（tikicpulib）未实现 DataCopyPad 的
// LoopMode 语义（SetLoopModePara 设置后仅执行第一轮，后续 loop 迭代丢失）。
// CopyInKey4/Key5Plus 依赖 LoopMode 完成多维搬运，若直接仿真会导致部分输出
// 恒为 0，从而无法验证 golden。这里仅在测试翻译单元内拦截对外调用，按硬件
// 语义把一次带 LoopMode 的 DataCopyPad 展开为多次普通 DataCopyPad；不修改
// 生产 Kernel，也不改变被验证 Kernel 的索引/ stride 计算逻辑。
// ---------------------------------------------------------------------------
namespace stb_cpu_loop_shim {

inline AscendC::LoopModeParams g_loopParams{};
inline bool g_loopActive = false;

inline void SetLoopModePara(const AscendC::LoopModeParams& params, AscendC::DataCopyMVType /* type */)
{
    g_loopParams = params;
    g_loopActive = true;
}

inline void ResetLoopModePara(AscendC::DataCopyMVType /* type */) { g_loopActive = false; }

template <typename T>
void DataCopyPad(AscendC::LocalTensor<T> dst, AscendC::GlobalTensor<T> src, const AscendC::DataCopyExtParams& params,
                 const AscendC::DataCopyPadExtParams<T>& padParams)
{
    if (!g_loopActive) {
        AscendC::DataCopyPad(dst, src, params, padParams);
        return;
    }
    const int64_t elemSize = static_cast<int64_t>(sizeof(T));
    for (int64_t l2 = 0; l2 < g_loopParams.loop2Size; ++l2) {
        for (int64_t l1 = 0; l1 < g_loopParams.loop1Size; ++l1) {
            const int64_t srcOff = (l1 * g_loopParams.loop1SrcStride + l2 * g_loopParams.loop2SrcStride) / elemSize;
            const int64_t dstOff = (l1 * g_loopParams.loop1DstStride + l2 * g_loopParams.loop2DstStride) / elemSize;
            AscendC::DataCopyPad(dst[dstOff], src[srcOff], params, padParams);
        }
    }
}

template <typename T>
void DataCopyPad(AscendC::GlobalTensor<T> dst, AscendC::LocalTensor<T> src, const AscendC::DataCopyExtParams& params)
{
    if (!g_loopActive) {
        AscendC::DataCopyPad(dst, src, params);
        return;
    }
    const int64_t elemSize = static_cast<int64_t>(sizeof(T));
    for (int64_t l2 = 0; l2 < g_loopParams.loop2Size; ++l2) {
        for (int64_t l1 = 0; l1 < g_loopParams.loop1Size; ++l1) {
            const int64_t srcOff = (l1 * g_loopParams.loop1SrcStride + l2 * g_loopParams.loop2SrcStride) / elemSize;
            const int64_t dstOff = (l1 * g_loopParams.loop1DstStride + l2 * g_loopParams.loop2DstStride) / elemSize;
            AscendC::DataCopyPad(dst[dstOff], src[srcOff], params);
        }
    }
}

} // namespace stb_cpu_loop_shim

#define SetLoopModePara stb_cpu_loop_shim::SetLoopModePara
#define ResetLoopModePara stb_cpu_loop_shim::ResetLoopModePara
#define DataCopyPad stb_cpu_loop_shim::DataCopyPad

// kernel 入口为函数模板，包含后获得 space_to_batch_nd<TilingKey> 与 TilingData 定义。
#include "../../../op_kernel/space_to_batch_nd.cpp"

#undef DataCopyPad
#undef ResetLoopModePara
#undef SetLoopModePara

namespace {

// ---------------------------------------------------------------------------
// 行主序 linear <-> 多维权坐标辅助函数
// ---------------------------------------------------------------------------
std::vector<int64_t> LinearToCoords(int64_t linear, const std::vector<int64_t>& shape)
{
    std::vector<int64_t> coords(shape.size(), 0);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
        coords[i] = linear % shape[i];
        linear /= shape[i];
    }
    return coords;
}

int64_t CoordsToLinear(const std::vector<int64_t>& coords, const std::vector<int64_t>& shape)
{
    int64_t linear = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
        linear = linear * shape[i] + coords[i];
    }
    return linear;
}

int64_t Product(const std::vector<int64_t>& shape)
{
    int64_t p = 1;
    for (int64_t d : shape) {
        p *= d;
    }
    return p;
}

// 参考实现：space_to_batch_nd，末尾 inner 维参与坐标映射。
// 输入 [xBatch, xSpatial..., innerSize] -> 输出 [outBatch, outSpatial..., innerSize]，
// 每个输出元素按 TF 定义映射回输入，padding（或越界）区域写 0。
std::vector<float> BuildGolden(const std::vector<float>& x, int64_t xBatch, const std::vector<int64_t>& xSpatial,
                               const std::vector<int64_t>& block, const std::vector<int64_t>& padTop,
                               const std::vector<int64_t>& outSpatial, int64_t innerSize)
{
    const int64_t M = static_cast<int64_t>(block.size());
    std::vector<int64_t> batchShape(block);
    const int64_t outBatch = xBatch * Product(batchShape);

    std::vector<int64_t> inShape;
    inShape.push_back(xBatch);
    inShape.insert(inShape.end(), xSpatial.begin(), xSpatial.end());
    inShape.push_back(innerSize);

    std::vector<int64_t> outShape;
    outShape.push_back(outBatch);
    outShape.insert(outShape.end(), outSpatial.begin(), outSpatial.end());
    outShape.push_back(innerSize);

    const int64_t outCount = Product(outShape);
    std::vector<float> golden(static_cast<size_t>(outCount), 0.0f);

    for (int64_t linear = 0; linear < outCount; ++linear) {
        std::vector<int64_t> outCoords = LinearToCoords(linear, outShape);

        const int64_t ob = outCoords[0];
        const int64_t origBatch = ob % xBatch;
        int64_t blockLinear = ob / xBatch;

        std::vector<int64_t> blockIdx(static_cast<size_t>(M), 0);
        for (int64_t i = M - 1; i >= 0; --i) {
            blockIdx[i] = blockLinear % block[i];
            blockLinear /= block[i];
        }

        std::vector<int64_t> inCoords;
        inCoords.push_back(origBatch);
        bool valid = true;
        for (int64_t i = 0; i < M; ++i) {
            const int64_t inCoord = outCoords[i + 1] * block[i] + blockIdx[i] - padTop[i];
            if (inCoord < 0 || inCoord >= xSpatial[i]) {
                valid = false;
            }
            inCoords.push_back(inCoord);
        }
        inCoords.push_back(outCoords[M + 1]);

        if (valid) {
            golden[static_cast<size_t>(linear)] = x[static_cast<size_t>(CoordsToLinear(inCoords, inShape))];
        }
    }
    return golden;
}

// ---------------------------------------------------------------------------
// 手工构造 TilingData 并执行 kernel，逐元素对比参考实现
// ---------------------------------------------------------------------------
template <int TilingKey>
void RunAndVerify(int64_t xBatch, const std::vector<int64_t>& xSpatial, const std::vector<int64_t>& block,
                  const std::vector<int64_t>& padTop, const std::vector<int64_t>& padBottom,
                  const std::vector<int64_t>& expectedOutSpatial, int64_t ubAxis, int64_t forcedUbFactor = -1,
                  int64_t innerSize = 1)
{
    const int64_t M = static_cast<int64_t>(xSpatial.size());
    ASSERT_EQ(M, static_cast<int64_t>(block.size()));
    ASSERT_EQ(M, static_cast<int64_t>(padTop.size()));
    ASSERT_EQ(M, static_cast<int64_t>(padBottom.size()));

    int64_t batchMul = 1;
    for (int64_t b : block) {
        batchMul *= b;
    }
    const int64_t outBatch = xBatch * batchMul;

    std::vector<int64_t> outSpatial(static_cast<size_t>(M), 0);
    for (int64_t i = 0; i < M; ++i) {
        outSpatial[i] = (xSpatial[i] + padTop[i] + padBottom[i]) / block[i];
    }
    EXPECT_EQ(outSpatial, expectedOutSpatial) << "outSpatial derived from x/block/paddings must match the case spec";

    std::vector<int64_t> inShape;
    inShape.push_back(xBatch);
    inShape.insert(inShape.end(), xSpatial.begin(), xSpatial.end());
    inShape.push_back(innerSize);

    std::vector<int64_t> outShape;
    outShape.push_back(outBatch);
    outShape.insert(outShape.end(), outSpatial.begin(), outSpatial.end());
    outShape.push_back(innerSize);

    const int64_t xCount = Product(inShape);
    const int64_t outCount = Product(outShape);

    // ---- value-initialize tiling data, then fill per case spec ----
    SpaceToBatchNDTilingData td{};
    td.rank = M + 2;
    td.numSpatialDims = M;
    td.inShape[0] = xBatch;
    for (int64_t i = 0; i < M; ++i) {
        td.inShape[i + 1] = xSpatial[i];
    }
    td.inShape[M + 1] = innerSize;
    td.outShape[0] = outBatch;
    for (int64_t i = 0; i < M; ++i) {
        td.outShape[i + 1] = outSpatial[i];
    }
    td.outShape[M + 1] = innerSize;
    for (int64_t i = 0; i < M; ++i) {
        td.blockShape[i] = block[i];
        td.padTop[i] = padTop[i];
        td.padBottom[i] = padBottom[i];
    }
    td.ubAxis = ubAxis;
    if (forcedUbFactor > 0) {
        td.ubFactor = forcedUbFactor;
    } else {
        td.ubFactor = td.outShape[ubAxis];
    }
    td.bufferSize = 4096;

    int64_t totalCount = 1;
    for (int64_t axis = 0; axis < ubAxis; ++axis) {
        totalCount *= td.outShape[axis];
    }
    totalCount *= (td.outShape[ubAxis] + td.ubFactor - 1) / td.ubFactor;
    td.totalCount = totalCount;
    td.perCoreCount = totalCount;

    // ---- host inputs ----
    std::vector<float> xHost(static_cast<size_t>(xCount));
    for (int64_t i = 0; i < xCount; ++i) {
        xHost[static_cast<size_t>(i)] = static_cast<float>(i + 1);
    }

    std::vector<int64_t> blockShapeHost = block;
    std::vector<int64_t> paddingsHost;
    for (int64_t i = 0; i < M; ++i) {
        paddingsHost.push_back(padTop[i]);
        paddingsHost.push_back(padBottom[i]);
    }

    const size_t xBytes = static_cast<size_t>(xCount) * sizeof(float);
    const size_t yBytes = static_cast<size_t>(outCount) * sizeof(float);
    const size_t bsBytes = blockShapeHost.size() * sizeof(int64_t);
    const size_t padsBytes = paddingsHost.size() * sizeof(int64_t);
    constexpr size_t kWorkspaceSize = 1024; // kernel 不使用，仅满足签名

    uint8_t* x = static_cast<uint8_t*>(AscendC::GmAlloc(std::max<size_t>(xBytes, 1)));
    uint8_t* bs = static_cast<uint8_t*>(AscendC::GmAlloc(bsBytes));
    uint8_t* pads = static_cast<uint8_t*>(AscendC::GmAlloc(padsBytes));
    uint8_t* y = static_cast<uint8_t*>(AscendC::GmAlloc(yBytes));
    uint8_t* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(kWorkspaceSize));
    uint8_t* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(SpaceToBatchNDTilingData)));

    if (xBytes > 0) {
        std::memcpy(x, xHost.data(), xBytes);
    }
    std::memcpy(bs, blockShapeHost.data(), bsBytes);
    std::memcpy(pads, paddingsHost.data(), padsBytes);
    std::memcpy(tiling, &td, sizeof(SpaceToBatchNDTilingData));

    // 输出预填毒饵：若 kernel 漏写某处，会以明显不同的值暴露出来。
    float* yHost = reinterpret_cast<float*>(y);
    std::fill(yHost, yHost + outCount, -12345.0f);

    ICPU_SET_TILING_KEY(TilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(space_to_batch_nd<TilingKey>, 1, x, bs, pads, y, workspace, tiling);

    // ---- compare against golden ----
    std::vector<float> golden = BuildGolden(xHost, xBatch, xSpatial, block, padTop, outSpatial, innerSize);
    for (int64_t i = 0; i < outCount; ++i) {
        EXPECT_FLOAT_EQ(yHost[static_cast<size_t>(i)], golden[static_cast<size_t>(i)]) << "output index " << i;
    }

    AscendC::GmFree(x);
    AscendC::GmFree(bs);
    AscendC::GmFree(pads);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

} // namespace

class SpaceToBatchNDKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

// Key3: M=3, rank=5, outShape=[4,3,1,2,1], ubAxis=2
TEST_F(SpaceToBatchNDKernelTest, CopyInKey3_FixedAxisInPadding)
{
    RunAndVerify<3>(/*xBatch=*/1, /*xSpatial=*/{4, 2, 2}, /*block=*/{2, 2, 1},
                    /*padTop=*/{1, 0, 0}, /*padBottom=*/{1, 0, 0},
                    /*expectedOutSpatial=*/{3, 1, 2}, /*ubAxis=*/2);
}

// Key4: M=4, rank=6, outShape=[4,3,1,2,2,1], ubAxis=2
TEST_F(SpaceToBatchNDKernelTest, CopyInKey4_FixedAxisInPadding)
{
    RunAndVerify<4>(/*xBatch=*/1, /*xSpatial=*/{4, 2, 2, 2}, /*block=*/{2, 2, 1, 1},
                    /*padTop=*/{1, 0, 0, 0}, /*padBottom=*/{1, 0, 0, 0},
                    /*expectedOutSpatial=*/{3, 1, 2, 2}, /*ubAxis=*/2);
}

// Key5Plus: M=5, rank=7, outShape=[4,3,1,2,2,2,1], ubAxis=2
TEST_F(SpaceToBatchNDKernelTest, CopyInKey5Plus_FixedAxisInPadding)
{
    RunAndVerify<5>(/*xBatch=*/1, /*xSpatial=*/{4, 2, 2, 2, 2}, /*block=*/{2, 2, 1, 1, 1},
                    /*padTop=*/{1, 0, 0, 0, 0}, /*padBottom=*/{1, 0, 0, 0, 0},
                    /*expectedOutSpatial=*/{3, 1, 2, 2, 2}, /*ubAxis=*/2);
}

// Key4: M=4, rank=6, outShape=[4,2,1,4,2,1], ubAxis=2; loop1 axis (out axis3) has padding on os=0/os=3
TEST_F(SpaceToBatchNDKernelTest, CopyInKey4_LoopAxisInPadding)
{
    RunAndVerify<4>(/*xBatch=*/1, /*xSpatial=*/{4, 2, 2, 2}, /*block=*/{2, 2, 1, 1},
                    /*padTop=*/{0, 0, 1, 0}, /*padBottom=*/{0, 0, 1, 0},
                    /*expectedOutSpatial=*/{2, 1, 4, 2}, /*ubAxis=*/2);
}

// Key5Plus: M=5, rank=7, outShape=[4,2,1,2,4,2,1], ubAxis=2; loop1 axis (out axis4) has padding on os=0/os=3
TEST_F(SpaceToBatchNDKernelTest, CopyInKey5Plus_LoopAxisInPadding)
{
    RunAndVerify<5>(/*xBatch=*/1, /*xSpatial=*/{4, 2, 2, 2, 2}, /*block=*/{2, 2, 1, 1, 1},
                    /*padTop=*/{0, 0, 0, 1, 0}, /*padBottom=*/{0, 0, 0, 1, 0},
                    /*expectedOutSpatial=*/{2, 1, 2, 4, 2}, /*ubAxis=*/2);
}

// Key1: M=2, rank=4, in spatial dim 0 is empty but padding yields a non-empty output;
// TilingKey=1 = rank(4)-ubAxis(3). Output must be entirely zero.
TEST_F(SpaceToBatchNDKernelTest, CopyInKey1_EmptySpatialWithPadding)
{
    RunAndVerify<1>(/*xBatch=*/2, /*xSpatial=*/{0, 4}, /*block=*/{2, 2},
                    /*padTop=*/{1, 1}, /*padBottom=*/{1, 1},
                    /*expectedOutSpatial=*/{1, 3}, /*ubAxis=*/3);
}

// Key5Plus: M=6, rank=8, outShape=[4,1,2,2,2,2,2,1], ubAxis=2, ubFactor forced to 1 so
// outShape[2]=2 yields a second tile with tileStart=1 (exercises space_to_batch_nd.h:518).
// TilingKey=8-2=6.
TEST_F(SpaceToBatchNDKernelTest, CopyInKey5Plus_MultiTileOnUbAxis)
{
    RunAndVerify<6>(/*xBatch=*/1, /*xSpatial=*/{2, 4, 2, 2, 2, 2}, /*block=*/{2, 2, 1, 1, 1, 1},
                    /*padTop=*/{0, 0, 0, 0, 0, 0}, /*padBottom=*/{0, 0, 0, 0, 0, 0},
                    /*expectedOutSpatial=*/{1, 2, 2, 2, 2, 2}, /*ubAxis=*/2, /*forcedUbFactor=*/1);
}

// ---------------------------------------------------------------------------
// 定向补充：innerSize > 1 / ubAxis < rank-1（CODE.06/CODE.08 风险定向验证）
// ---------------------------------------------------------------------------

// A. Key2 + innerSize>1: M=2, rank=4, outShape=[8,2,2,3], ubAxis=2.
//    ubAxis 是最后空间轴，inner 维（=3）参与映射；验证 CopyInKey2 在 inner>1 下的搬运。
TEST_F(SpaceToBatchNDKernelTest, CopyInKey2_InnerGreaterThanOne)
{
    RunAndVerify<2>(/*xBatch=*/2, /*xSpatial=*/{4, 4}, /*block=*/{2, 2},
                    /*padTop=*/{0, 0}, /*padBottom=*/{0, 0},
                    /*expectedOutSpatial=*/{2, 2}, /*ubAxis=*/2,
                    /*forcedUbFactor=*/-1, /*innerSize=*/3);
}

// B. Key3 + ubAxis<rank-1 + innerSize>1 + 多 tile:
//    M=2, rank=4, outShape=[4,2,3,2], ubAxis=1, ubFactor=1。
//    ubAxis 固定轴不在最后空间轴，且 inner 维（=2）参与映射。
TEST_F(SpaceToBatchNDKernelTest, CopyInKey3_UbAxisBeforeLast_InnerGreaterThanOne)
{
    RunAndVerify<3>(/*xBatch=*/1, /*xSpatial=*/{4, 4}, /*block=*/{2, 2},
                    /*padTop=*/{0, 1}, /*padBottom=*/{0, 1},
                    /*expectedOutSpatial=*/{2, 3}, /*ubAxis=*/1,
                    /*forcedUbFactor=*/1, /*innerSize=*/2);
}

// C. Key2 + 多 tile + innerSize>1:
//    M=2, rank=4, outShape=[4,2,4,2], ubAxis=2, ubFactor=1（多 tile）。
TEST_F(SpaceToBatchNDKernelTest, CopyInKey2_MultiTile_InnerGreaterThanOne)
{
    RunAndVerify<2>(/*xBatch=*/1, /*xSpatial=*/{4, 6}, /*block=*/{2, 2},
                    /*padTop=*/{0, 1}, /*padBottom=*/{0, 1},
                    /*expectedOutSpatial=*/{2, 4}, /*ubAxis=*/2,
                    /*forcedUbFactor=*/1, /*innerSize=*/2);
}

// D. Key3 + 大 inner（接近 UB 容量边界）:
//    M=2, rank=4, outShape=[4,2,2,128], ubAxis=1, ubFactor=1。
//    inner=128 float（512B/行），单 tile 2 行 = 1KB；验证不越界且输出正确。
TEST_F(SpaceToBatchNDKernelTest, LargeInner_CapacityBoundary)
{
    RunAndVerify<3>(/*xBatch=*/1, /*xSpatial=*/{4, 4}, /*block=*/{2, 2},
                    /*padTop=*/{0, 0}, /*padBottom=*/{0, 0},
                    /*expectedOutSpatial=*/{2, 2}, /*ubAxis=*/1,
                    /*forcedUbFactor=*/1, /*innerSize=*/128);
}
