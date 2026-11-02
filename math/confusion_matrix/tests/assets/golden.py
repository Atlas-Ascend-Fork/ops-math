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
"""ConfusionMatrix 多通路 golden（TestSpec 范式）。

通路支持表（本算子交付面）：
| 通路       | 支持 | 依据 |
|------------|------|------|
| kernel     | ✅   | op_kernel/arch35 有实现 |
| geir       | ✅   | op_graph/confusion_matrix_proto.h 有 REG_OP(ConfusionMatrix) |
| aclnn      | ❌   | Ascend910B/910C built-in 无 aclnn 接口，本算子对齐该支持面不交付 |
| e2e (tf)   | ✅   | tf.math.confusion_matrix 经 npu_device（Ascend TF adapter）入 GE 图执行 |
| e2e (torch)| ❌   | torch_npu 无绑定（无 aclnn 接口 → 无可调用入口） |

int8/uint8 输出：整型计数语义、差 1 就是错 → binary_equal（非量化场景，输入含整型，
不适用 quant 的浮点输入前提）。int8/uint8 以 int32 中转累加、末端窄回：纯加法下
模 2^n 环同态，与 NPU 窄整型逐窄回绕累加等价（div 类算子才需每步窄回，本算子无除法）。
"""

__spec__ = {
    "confusion_matrix": "ConfusionMatrixKernelSpec",
    # e2e(tf) 通路：tf 仅为驱动接口，精度标杆仍为算子语义（同 kernel/geir 共用 torch 真值）
    "tf.math.confusion_matrix": "ConfusionMatrixTfE2eSpec",
}

import numpy as np
import torch

_DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "int32": torch.int32,
    "int8": torch.int8,
    "uint8": torch.uint8,
}

_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    # 整型输出：计数语义、差 1 就是错，逐 bit 相等（与上文说明一致，显式落地防默认容差漂移）
    "int32": {"standard": "binary_equal"},
    "int8": {"standard": "binary_equal"},
    "uint8": {"standard": "binary_equal"},
}


def _attr(kwargs, name, default):
    """CSV 里的 attributes 都是字符串，统一转成 default 的类型。"""
    v = kwargs.get(name, default)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "false", "yes", "no", "1", "0"):
            return s in ("true", "yes", "1")
        try:
            return type(default)(v)
        except Exception:
            return default
    return v


def _normalize(val):
    if isinstance(val, (np.ndarray, np.generic)):
        return val.item() if val.ndim == 0 else val.tolist()
    if isinstance(val, bytes):
        return val.decode()
    return val


def _resolve_compute_dtype(**kwargs):
    """累加精度基准：取 TTK 下发的 output_dtypes（Promote 后的值）。

    三方比对（cross_check）下 TTK 已按 Promote 抬高 output_dtypes（fp16->fp32、
    fp32->fp64），golden 据此在抬高后的精度上累加出高精度真值；两方比对未抬高，
    则跟随算子声明精度。取不到 output_dtypes 时回落算子声明的 dtype。
    返回 (compute_dtype, final_dtype)：int8/uint8 时 compute 为 int32 中转、
    final 为目标 dtype；其余两者相同。
    """
    dtype_str = _normalize(kwargs.get("dtype", "float32"))
    final_dtype = _DTYPE_MAP.get(dtype_str, torch.float32)
    od = kwargs.get("output_dtypes") or []
    if od:
        first = od[0]
        if isinstance(first, (list, tuple)):
            first = first[0] if first else None
        if first is not None:
            final_dtype = _DTYPE_MAP.get(str(first), final_dtype)
    compute_dtype = (
        torch.int32 if final_dtype in (torch.int8, torch.uint8) else final_dtype
    )
    return compute_dtype, final_dtype


def _compute(labels, predictions, weights, **kwargs):
    """计算核（torch 进 torch 出）。golden 用 torch.scatter_add 竞品接口拼接。

    累加精度按 _resolve_compute_dtype：三方比对下在 Promote 抬高后的精度上算
    高精度真值；两方比对跟随算子声明精度。int8/uint8 输出用 int32 中间累加再
    cast 回（加法环同态下与窄整型累加等价，规避 torch 窄整型 scatter_add 的
    中间量行为差异）。
    """
    num_classes = _attr(kwargs, "num_classes", 0)
    num_classes = _normalize(num_classes)

    labels_int = labels.to(torch.int64)
    preds_int = predictions.to(torch.int64)
    flat_index = labels_int * num_classes + preds_int

    compute_dtype, final_dtype = _resolve_compute_dtype(**kwargs)

    y_flat = torch.zeros(num_classes * num_classes, dtype=compute_dtype)
    if weights is not None and weights.numel() > 0:
        y_flat.scatter_add_(0, flat_index, weights.to(compute_dtype))
    else:
        y_flat.scatter_add_(
            0, flat_index, torch.ones(labels_int.shape[0], dtype=compute_dtype)
        )

    if compute_dtype != final_dtype:
        y_flat = y_flat.to(final_dtype)
    return [y_flat.view(num_classes, num_classes)]


class _Compose:
    """三方标杆（远端 GPU server 执行）。属性喂 __init__、输入喂 __call__。

    参数名与 proto REG_OP 注册名逐字一致（labels/predictions/weights/num_classes/dtype）。
    实现与 _compute 同为 torch.scatter_add 拼接（本算子即竞品拼接型，无更高层 API）。
    按 NPU 加宽行为同步：NPU 浮点窄累加不宽 → 三方腿直接在声明 dtype 上计算，
    出口无需 cast（ratio 由「同为窄精度独立实现」的双标杆结构保证）；
    int8/uint8 以 int32 中转累加后窄回，与 NPU 窄宽回绕等价。
    """

    def __init__(self, **kwargs):
        self.num_classes = _attr(kwargs, "num_classes", 0)
        self.num_classes = _normalize(self.num_classes)
        self.dtype = _normalize(kwargs.get("dtype", "float32"))
        self.out_dtype = _DTYPE_MAP.get(self.dtype, torch.float32)

    def __call__(self, labels, predictions, weights, **kwargs):
        nc = self.num_classes
        flat_index = labels.to(torch.int64) * nc + predictions.to(torch.int64)
        compute_dtype = self.out_dtype
        if compute_dtype in (torch.int8, torch.uint8):
            compute_dtype = torch.int32
        y_flat = torch.zeros(nc * nc, dtype=compute_dtype, device=labels.device)
        if weights is not None and weights.numel() > 0:
            y_flat.scatter_add_(0, flat_index, weights.to(compute_dtype))
        else:
            y_flat.scatter_add_(
                0,
                flat_index,
                torch.ones(
                    flat_index.shape[0], dtype=compute_dtype, device=labels.device
                ),
            )
        if compute_dtype != self.out_dtype:
            y_flat = y_flat.to(self.out_dtype)
        return [y_flat.view(nc, nc)]


class ConfusionMatrixKernelSpec:
    """kernel + geir 共用。golden 收 numpy.ndarray，返 numpy.ndarray。
    参数名照 op_host/confusion_matrix_def.cpp（labels/predictions/weights + num_classes/dtype）。
    """

    @staticmethod
    def golden(*inputs, **kwargs):
        """出口零 cast：_compute 已按 TTK 下发的 output_dtypes（Promote 后的值）
        决定累加/窄回精度，torch 转 numpy 时 dtype 随身携带。原先按 output_dtypes
        二次 astype 在 Promote 下是 no-op、且隐式依赖 TTK 传递行为，去除后
        「浮点+三方」场景字面满足入口出口零 cast，整型窄回仍由 _compute 完成。"""
        t = [
            torch.from_numpy(np.ascontiguousarray(a)) if a is not None else None
            for a in inputs
        ]
        outs = _compute(*t, **kwargs)
        return [o.numpy() for o in outs]

    third_party = {"torch": _Compose}
    tolerance = _TOL


class ConfusionMatrixTfE2eSpec:
    """E2E (tf) 通路：api_name ``tf.math.confusion_matrix`` —— 驱动接口，非标杆。

    路由：TF adapter（npu_device）→ GE 图执行。tf.math.confusion_matrix 为 Python
    组合 API，图分解为 Cast/Pack/Transpose/SparseTensorDenseAdd 原语逐算子执行
    （实测：CANN 9.2.0 GE 的 support_fusion_pass.json 白名单不含
    ConfusionMatrixFusionPass，950 上不触发融合，本算子不参与该通路执行）；
    故本通路验证的是 **tf 前端 API 语义在 NPU 上的正确性**。命中本算子的通路为
    GE 图模式（geir，proto + plugin 映射）。输入 torch.Tensor，返回 torch.Tensor。

    精度标杆对标**算子语义**，而非 tf 接口：期望输出始终是 ConfusionMatrix
    算子的语义结果，真值与 kernel/geir 共用同一套 torch 标杆 ``_compute``。

    实参解析（TTK 依 API 签名顺序组装，标量属性 const-fold 为位置实参，可选位
    补 None）：labels(0)、predictions(1)、num_classes(2)、weights(3)、dtype(4)，
    kwargs 同名参数兜底。每个位置按类型校验，形态不符时报错而非静默错位：
    - num_classes 缺省时按 tf 语义由数据推导（max(labels, predictions) + 1）；
    - dtype 缺省时用 tf 接口默认 ``int32``（签名 dtype=dtypes.int32）；
    - dtype 归一优先取 ``.name``（tf.DType），未命中算子五类 dtype 白名单时
      报错而非静默回落 float32。
    """

    _TF_DEFAULT_DTYPE = "int32"

    @staticmethod
    def golden(*inputs, **kwargs):
        def _to_tensor(a, slot):
            """torch/np/tf 张量统一转 torch；非数值形态响亮报错不静默吞掉。"""
            if a is None:
                return None
            if isinstance(a, torch.Tensor):
                return a
            try:
                arr = np.asarray(a)  # ndarray / tf.Tensor(Eager) 等均可 numpy 中转
            except Exception as exc:
                raise ValueError(
                    f"{slot} slot cannot convert to tensor: {a!r}"
                ) from exc
            if arr.dtype.kind not in "fiub":
                raise ValueError(
                    f"{slot} slot got non-numeric dtype {arr.dtype} (positional drift?)"
                )
            return torch.as_tensor(arr)

        def _scalar_int(v):
            if isinstance(v, str):
                return int(v.strip())
            if isinstance(v, torch.Tensor):
                if v.numel() != 1:
                    raise ValueError(
                        f"num_classes must be scalar, got shape {tuple(v.shape)}"
                    )
                v = v.item()
            else:
                arr = np.asarray(v)  # np.generic / ndarray / tf.Tensor 标量均可中转
                if arr.size != 1:
                    raise ValueError(
                        f"num_classes must be scalar, got shape {arr.shape}"
                    )
                v = arr.reshape(-1)[0].item()
            return int(v)

        labels = _to_tensor(inputs[0], "labels") if len(inputs) > 0 else None
        predictions = _to_tensor(inputs[1], "predictions") if len(inputs) > 1 else None
        if labels is None or predictions is None:
            raise ValueError(
                "tf.math.confusion_matrix requires labels/predictions tensors"
            )

        # num_classes：kwargs 优先；第 3 位为 const-fold 标量/标量张量；
        # 缺省时按 tf 语义由数据推导（maximum(reduce_max(l), reduce_max(p)) + 1）
        nc = kwargs.get("num_classes")
        if nc is None and len(inputs) > 2 and inputs[2] is not None:
            nc = inputs[2]
        if nc is None:
            nc = max(int(labels.max().item()), int(predictions.max().item())) + 1
        else:
            nc = _scalar_int(nc)

        # weights：第 4 位张量（可选）
        weights = _to_tensor(inputs[3], "weights") if len(inputs) > 3 else None

        # dtype：签名第 5 位优先（const-fold 为原始字符串/类 dtype 对象），kwargs 兜底；
        # 归一优先 .name（tf.DType）；缺省用 tf 接口默认 int32；白名单外报错不静默回落
        dt = (
            inputs[4]
            if len(inputs) > 4 and inputs[4] is not None
            else kwargs.get("dtype")
        )
        if dt is None:
            dtype_str = ConfusionMatrixTfE2eSpec._TF_DEFAULT_DTYPE
        else:
            name = getattr(dt, "name", None)
            dtype_str = name if isinstance(name, str) else str(_normalize(dt))
            if dtype_str not in _DTYPE_MAP:
                raise ValueError(
                    f"unsupported dtype {dtype_str!r} for confusion_matrix tf e2e "
                    f"(expected one of {sorted(_DTYPE_MAP)})"
                )

        kwargs = dict(kwargs)
        kwargs["num_classes"] = nc
        kwargs["dtype"] = dtype_str
        outs = _compute(labels, predictions, weights, **kwargs)
        return outs  # torch 入/出（e2e 契约，参照 left_shift 的 LeftShiftTfSpec）

    third_party = {"torch": _Compose}
    tolerance = _TOL


# 【不存在】aclnn 通路：Ascend910B/910C built-in 无 aclnn 源码 → 对齐该支持面，本算子不交付 aclnn。
# 【不存在】e2e(torch) 通路：torch_npu 无本算子绑定。无 aclnn 源码 → 无编译产物 → torch_npu 无可
# 调用入口；验证环境 torch_npu dispatcher 全表 grep "confusion" 无命中。
# e2e(tf) 通路存在：tf.math.confusion_matrix 为组合 API，经 npu_device adapter 入 GE 图执行
# （见 ConfusionMatrixTfE2eSpec），非 torch_npu 绑定形式。
