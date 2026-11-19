#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import numpy

try:
    import ml_dtypes  # bfloat16 golden 支持(astype('bfloat16') 需要)
except ImportError:
    ml_dtypes = None

__golden__ = {"kernel": {"histogram_v2": "histogram_v2_golden"}}

# 计数输出逐位比对: binary_equal 不属于 TTK Promote 判据集合(mixed/mix_tolerance/
# cross_check), golden 输入保持原生 dtype(否则 fp32 用例被升精度为 float64 副本,
# golden 分支路由错乱); 整数计数语义上也应位级一致。
_BIN_EQUAL = {
    dt: {"standard": "binary_equal"}
    for dt in (
        "int8",
        "uint8",
        "int16",
        "int32",
        "int64",
        "float16",
        "bfloat16",
        "float32",
    )
}

__spec__ = {
    "histogram_v2": "HistogramV2KernelSpec",
    "histogram": "HistogramTestSpec",
    "aclnnHistc": "HistcTorchTestSpec",
    "torch.histc": "HistcTorchTestSpec",
}


class HistogramV2KernelSpec:
    """kernel 通路 histogram_v2 的 tolerance 声明(golden 仍走 __golden__ 函数约定)。"""

    tolerance = _BIN_EQUAL


class HistcTorchTestSpec:
    """ACLNN(aclnnHistc)/E2E(torch.histc) 通路 golden, CPU 侧 torch.Tensor 入参。

    浮点(bf16/fp16/fp32/fp64): 一律转 fp32 后用 torch.histc CPU——NPU kernel 在 float
    域分 bin; CPU bf16/fp16 histc 的 bin_edges 为低精度 linspace 且分 bin 算术每步舍回
    低精度(HistogramKernel.cpp input_t 模板), 与 NPU 语义不同族, 不可作标杆(docs §8.5)。
    输出计数转回输入 dtype(bf16/fp16 计数 <=256 精确)。

    整数: CPU histc 不支持整数输入, 手写 int64 域计数(与 kernel COMPUTE_TYPE 同口径);
    min==max 时按 CUDA/aclnn 语义 aminmax 重算, 仍相等则整数裸 +/-1 展开(kernel 整数路径)。
    min/max 保持 Python int 不进 float, int64 + >2^53 标量(M6a)全程精确。
    """

    tolerance = _BIN_EQUAL

    @staticmethod
    def golden(x, *args, **kwargs):
        import torch

        # kwargs 优先; aclnn 通路按签名顺序 (bins, min, max) 位置传参
        bins = kwargs.get("bins", args[0] if len(args) > 0 else 100)
        min_v = kwargs.get("min", args[1] if len(args) > 1 else 0.0)
        max_v = kwargs.get("max", args[2] if len(args) > 2 else 0.0)

        # 输出计数 dtype: 按用例声明的 out dtype(torch.histc 语义 out==x dtype;
        # aclnn fp32out 变体显式声明 fp32)。TTK 装载层会把 bf16/fp16 输入转成
        # fp32 tensor 喂 golden(声明 dtype 不可恢复), 故回落序: output_dtypes 声明
        # > input_dtypes 声明(csv 声明的 x dtype, histc 语义 out==x) > x.dtype 兜底
        def _to_torch_dtype(name):
            return name if isinstance(name, torch.dtype) else getattr(torch, str(name))

        # 键形态随通路: kernel 注入 output_dtypes(输出声明); aclnn/e2e 注入
        # tensor_dtypes —— aclnn=(x,...,out) 末位即 out 声明, e2e=(x,) 单元素
        # 即 x(histc 语义 out==x); input_dtypes 为 kernel 兜底(全输入, 取首= x)。
        # 统一取 [-1]: aclnn 得 out 声明, e2e 得 x, 语义各自正确。
        out_decl = kwargs.get("output_dtypes")
        td_decl = kwargs.get("tensor_dtypes")
        in_decl = kwargs.get("input_dtypes")
        if isinstance(td_decl, (tuple, list)) and td_decl:
            out_dtype = _to_torch_dtype(td_decl[-1])
        elif isinstance(out_decl, (tuple, list)) and out_decl:
            out_dtype = _to_torch_dtype(out_decl[-1])
        elif isinstance(in_decl, (tuple, list)) and in_decl:
            out_dtype = _to_torch_dtype(in_decl[0])
        else:
            out_dtype = x.dtype

        if x.is_floating_point():
            min_f, max_f = float(min_v), float(max_v)
            if x.dtype in (torch.bfloat16, torch.float16):
                # kernel 收到的 min/max 是 x dtype 张量(aclnn/torch_npu 把标量舍入到 x dtype);
                # M3a 超值域提升 fp32 时除外(标量不经 x dtype 舍入)
                finfo = torch.finfo(x.dtype)
                if abs(min_f) <= finfo.max and abs(max_f) <= finfo.max:

                    def _cast_via_xdtype(v):
                        return float(torch.tensor(v, dtype=x.dtype).float())

                    min_f, max_f = _cast_via_xdtype(min_f), _cast_via_xdtype(max_f)
            # 注意: TTK 装载层会把 bf16/fp16 输入转成 fp32 tensor 喂 golden(数值位级
            # 精确, 转换后 dtype 同步进 tensor_dtypes[0]), 声明 dtype 不可恢复——但
            # M3a 提升与 fp32 直算在"标量不经低精度舍入"上同义, 无需区分。
            if min_f == max_f:
                # aclnn NeedComputeMinMax 精确相等(含同号 ±inf): 先按数据 aminmax
                # 重算消化(±inf 标量被数据范围替换)
                xf = x.float()
                if xf.numel() > 0:
                    min_f = float(xf.min())
                    max_f = float(xf.max())
                if min_f == max_f:
                    # 重算后仍相等(常数数据/单元素): 与 kernel ExpandMinMaxIfEqual
                    # 同口径(CPU #163506): fp32 域取 nextafter 邻接值与 ±1 更远端,
                    # 参数名遮蔽内建, 用 numpy.minimum/maximum 聚合
                    m = numpy.float32(min_f)
                    min_f = float(
                        numpy.minimum(
                            numpy.nextafter(m, numpy.float32(-numpy.inf)),
                            numpy.float32(m - numpy.float32(1.0)),
                        )
                    )
                    max_f = float(
                        numpy.maximum(
                            numpy.nextafter(m, numpy.float32(numpy.inf)),
                            numpy.float32(m + numpy.float32(1.0)),
                        )
                    )
            with numpy.errstate(over="ignore"):
                length32 = numpy.float32(numpy.float32(max_f) - numpy.float32(min_f))
            # 逐点复刻 kernel ComputeBinIndex 的 fp32 算术序(subtract->multiply->
            # divide->trunc, idx==bins 钳末 bin), int64 精确计数。不再用 CPU
            # torch.histc 作标杆, 三处不可靠:
            # 1) 大值域: CPU linspace 边界与 kernel 的 mul/div 序在 fp32 中间舍入
            #    上不同(wide ±1e30 实测边界差 1 桶);
            # 2) 大计数: CPU 对 fp32 输出逐 +1 累计, 单桶过 2^24 后舍入卡死;
            # 3) max-min fp32 溢出(bf16 M3a 对称超值域): linspace 退化; (v-min)*bins
            #    亦溢出 -> inf/inf=NaN, NPU fp32->int64 对 NaN/±inf 归 0(实测),
            #    全落 bin0。|v| 大到影响溢出形态的场景不在覆盖内。
            v32 = x.float().cpu().numpy().reshape(-1)  # bf16/fp16->fp32 位级精确
            mn = numpy.float32(min_f)
            mx = numpy.float32(max_f)
            res = numpy.zeros((int(bins),), dtype=numpy.int64)
            with numpy.errstate(over="ignore", invalid="ignore", divide="ignore"):
                in_range = v32[(v32 >= mn) & (v32 <= mx)]
                if in_range.size:
                    idx_f = (in_range - mn) * numpy.float32(int(bins)) / length32
                    idx_f = numpy.where(
                        numpy.isfinite(idx_f), idx_f, numpy.float32(0.0)
                    )
                    idx = numpy.clip(idx_f.astype(numpy.int64), 0, int(bins) - 1)
                    res = numpy.bincount(idx, minlength=int(bins)).astype(numpy.int64)
            return [torch.from_numpy(res).to(out_dtype)]

        xv = x.cpu().numpy().astype(numpy.int64).reshape(-1)
        lo, hi = int(min_v), int(max_v)
        if lo == hi and xv.size > 0:
            lo, hi = int(xv.min()), int(xv.max())
            if lo == hi:
                lo -= 1
                hi += 1
        res = numpy.zeros((int(bins),), dtype=numpy.int64)
        if hi > lo and xv.size > 0:
            in_range = xv[(xv >= lo) & (xv <= hi)]
            idx = (
                (in_range - lo) * int(bins) // (hi - lo)
            )  # 非负域 floor==trunc, 同 C++ int64 除法
            idx = numpy.clip(idx, 0, int(bins) - 1)
            res = numpy.bincount(idx, minlength=int(bins)).astype(numpy.int64)
        return [torch.from_numpy(res).to(out_dtype)]


class HistogramTestSpec:
    """GEIR 老 Histogram(attr min/max) 经 HistogramFusionPass -> HistogramV2 的 golden。

    CUDA 语义: min==max(含默认 0/0) 时按数据 aminmax 重算范围; 重算后仍相等
    (常数数据) 则按 CPU #163506 展开(fp32 取 nextafter 邻接值与 ±1 更远者,
    整数 int64 裸 ±1, 与 kernel ExpandMinMaxIfEqual 逐点一致); 空输入输出全零。
    min<max 时直接用给定范围(M3a 提升对计数结果透明)。
    """

    tolerance = _BIN_EQUAL

    @staticmethod
    def golden(*args, **kwargs):
        x = numpy.asarray(args[0])
        bins = int(kwargs.get("bins", 100))
        min_v = float(kwargs.get("min", 0.0))
        max_v = float(kwargs.get("max", 0.0))
        out_dtype = str(kwargs.get("output_dtypes", ("int32",))[0])
        if out_dtype == "bfloat16" and ml_dtypes is not None:
            out_dtype = ml_dtypes.bfloat16
        # bf16 与 fp16/fp32 同走浮点分支(astype(float32) 对 ml_dtypes.bfloat16 精确)
        x_is_float = x.dtype.name in ("float16", "float32", "bfloat16")

        # 老 IR min/max 经 float32 attr 传递: int 声明标量先过 fp32 舍入(与 pass
        # 建张量一致, 2^31-1 等不可表示值在 kernel 侧实为 fp32 最近值);
        # autorange 重算/展开的值不经 attr 通道, 不二次舍入
        if not x_is_float:
            min_v = float(numpy.float32(min_v))
            max_v = float(numpy.float32(max_v))

        if min_v == max_v:
            if x.size == 0:
                return [numpy.zeros((bins,), dtype=out_dtype)]
            if x_is_float:
                xc = x.astype(numpy.float32)
                lo = float(xc.min())
                hi = float(xc.max())
                if lo == hi:
                    m = numpy.float32(lo)
                    lo = float(
                        min(
                            numpy.nextafter(m, numpy.float32(-numpy.inf)),
                            numpy.float32(m - numpy.float32(1.0)),
                        )
                    )
                    hi = float(
                        max(
                            numpy.nextafter(m, numpy.float32(numpy.inf)),
                            numpy.float32(m + numpy.float32(1.0)),
                        )
                    )
            else:
                xc = x.astype(numpy.int64)
                lo = int(xc.min())
                hi = int(xc.max())
                if lo == hi:
                    lo -= 1
                    hi += 1
            min_v, max_v = lo, hi
        elif x_is_float and x.dtype.name in ("float16", "bfloat16"):
            # min<max 透传: fusion pass 把 float attr 舍入到 x dtype 建 min/max 张量,
            # golden 同步舍入(M3a 超值域提升时不经 x dtype 舍入, 与 aclnn 一致)
            # numpy.finfo 不支持 ml_dtypes.bfloat16, 用 ml_dtypes 自己的 finfo
            finfo_max = float(
                ml_dtypes.finfo(x.dtype).max
                if x.dtype.name == "bfloat16" and ml_dtypes is not None
                else numpy.finfo(x.dtype).max
            )
            if abs(min_v) <= finfo_max and abs(max_v) <= finfo_max:
                min_v = float(numpy.array(min_v, dtype=x.dtype).astype(numpy.float32))
                max_v = float(numpy.array(max_v, dtype=x.dtype).astype(numpy.float32))

        # 与 kernel 同口径: 仅统计 [min,max] 内元素, idx = trunc((v-min)*bins/(max-min)),
        # idx==bins 收敛到末 bin; 计算类型 fp32->float32, 整数->int64(同 kernel COMPUTE_TYPE)
        res = numpy.zeros((bins,), dtype=numpy.int64)
        if x_is_float:
            work = x.astype(numpy.float32).reshape(-1)
            mn = numpy.float32(min_v)
            mx = numpy.float32(max_v)
            b = numpy.float32(bins)
        else:
            work = x.astype(numpy.int64).reshape(-1)
            mn = numpy.int64(int(min_v))
            mx = numpy.int64(int(max_v))
            b = numpy.int64(bins)
        with numpy.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for value in work:
                if mn <= value <= mx:
                    if x_is_float:
                        idx_f = (value - mn) * b / (mx - mn)
                        # M3a bf16 对称超值域: (v-min)*bins 与 mx-mn 先后溢出 inf,
                        # inf/inf=NaN; NPU fp32->int64 对 NaN/±inf 归 0(实测), 落 bin0
                        index = int(idx_f) if numpy.isfinite(idx_f) else 0
                    else:
                        # int 域与 kernel 同为 int64 截断除法(非负域 floor==trunc)
                        index = int((value - mn) * b // (mx - mn))
                    if index == bins:
                        index -= 1
                    res[index] += 1

        return [res.astype(out_dtype)]


def histogram_v2_golden(x, min, max, *, bins=100, y_dtype=3, **kwargs):
    """
    Kernel golden for histogram_v2.
    All the parameters follow @histogram_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
            input_formats, output_formats, input_ori_formats, output_ori_formats,
            input_dtypes, output_dtypes.
    """
    output_dtypes = kwargs.get("output_dtypes", ["int32"])
    out_dtype = output_dtypes[0] if output_dtypes else "int32"

    if out_dtype not in ("float32", "int32"):
        print("out dtype error")

    min_data = min[0]
    max_data = max[0]
    x_dtype = x.dtype.name

    if x_dtype in ("float16", "float32", "bfloat16"):
        min_data = numpy.float32(min_data)
        max_data = numpy.float32(max_data)
        x = x.astype(numpy.float32)
        bins_val = numpy.float32(bins)
        if min_data == max_data:
            # 与 kernel ExpandMinMaxIfEqual 同口径(CPU #163506 语义):
            # 取 nextafter 邻接值与 +/-1 中更远者; +/-0 展开到最小非规格化数
            # (注意本函数参数名 min/max 遮蔽内建, 用 numpy 聚合函数)
            m = min_data
            lo = numpy.minimum(
                numpy.nextafter(m, numpy.float32(-numpy.inf)),
                m - numpy.float32(1.0),
            )
            hi = numpy.maximum(
                numpy.nextafter(m, numpy.float32(numpy.inf)),
                m + numpy.float32(1.0),
            )
            min_data, max_data = numpy.float32(lo), numpy.float32(hi)
    elif x_dtype in ("int8", "uint8", "int16"):
        min_data = numpy.int32(min_data)
        max_data = numpy.int32(max_data)
        x = x.astype(numpy.int32)
        bins_val = numpy.int32(bins)
        if min_data == max_data:
            # kernel 整数路径: COMPUTE 域(int32)裸 +/-1, 无环绕
            min_data, max_data = min_data - 1, max_data + 1
    else:
        min_data = numpy.int64(min_data)
        max_data = numpy.int64(max_data)
        x = x.astype(numpy.int64)
        bins_val = numpy.int32(bins)
        if min_data == max_data:
            min_data, max_data = min_data - 1, max_data + 1

    # 计数一律 int64 累计: res 若直接用 out_dtype(float32)逐 +1, 单桶计数过
    # 2^24 后舍入卡死(实测停在 16777216), 大计数标杆失真
    res = numpy.zeros((bins,), dtype=numpy.int64)
    for value in x.reshape(-1):
        if min_data <= value <= max_data:
            index = int((value - min_data) * bins_val / (max_data - min_data))
            if index == bins:
                index -= 1
            res[index] += 1

    return res.astype(out_dtype)
