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

import numpy as np


def _tf_pad_compose(arr, pad_width, mode):
    """tf 原语手写 edge(=replicate)/circular(=wrap) 任意维 pad（tf.pad 无这两种模式）。
    非 eager 进程（被其他 golden 置为图模式时）用独立 Graph 兜底。"""
    import tensorflow as tf

    def build(x):
        for i, (lp, r) in enumerate(pad_width):
            lp, r = int(lp), int(r)
            if lp == 0 and r == 0:
                continue
            n = tf.shape(x)[i]
            if mode == "edge":
                if lp > 0:
                    head = tf.gather(x, [0], axis=i)
                    mult = tf.stack([lp if j == i else 1 for j in range(x.shape.rank)])
                    x = tf.concat([tf.tile(head, mult), x], axis=i)
                if r > 0:
                    tail = tf.gather(x, [tf.shape(x)[i] - 1], axis=i)
                    mult = tf.stack([r if j == i else 1 for j in range(x.shape.rank)])
                    x = tf.concat([x, tf.tile(tail, mult)], axis=i)
            else:  # circular
                if lp > 0 or r > 0:
                    reps = tf.stack(
                        [
                            (n + lp + r + n - 1) // n if j == i else 1
                            for j in range(x.shape.rank)
                        ]
                    )
                    tiled = tf.tile(x, reps)
                    begin = tf.stack(
                        [
                            tf.math.floormod(-lp, n) if j == i else 0
                            for j in range(x.shape.rank)
                        ]
                    )
                    size = tf.stack(
                        [
                            n + lp + r if j == i else tf.shape(x)[j]
                            for j in range(x.shape.rank)
                        ]
                    )
                    x = tf.slice(tiled, begin, size)
        return x

    if tf.executing_eagerly():
        return build(tf.convert_to_tensor(arr)).numpy()
    graph = tf.Graph()
    with graph.as_default():
        out_g = build(tf.convert_to_tensor(np.asarray(arr)))
    with tf.compat.v1.Session(graph=graph) as sess:
        return np.asarray(sess.run(out_g))


def _framework_pad(arr, pad_width, mode):
    """非 constant 模式分派: reflect/symmetric/circular 统一实现。
    快路径: 各维 pad < size 时 tf.pad 单次反射; 否则三角波索引 gather
    支持任意宽度多重反射(np.pad 等价语义, tf.pad 有 pad<size 限制)。
    edge 走 torch F.pad(仅尾部维), dim0 有 pad 或维数不支持时
    走 _tf_pad_compose(tf 原语拼接)。"""
    import tensorflow as tf
    import torch

    ndim = arr.ndim
    mode_l = mode.lower() if isinstance(mode, str) else mode
    if mode_l in ("reflect", "symmetric", "circular"):
        fast = all(
            (int(lp) == 0 and int(r) == 0) or max(int(lp), int(r)) < int(arr.shape[i])
            for i, (lp, r) in enumerate(pad_width)
        )
        if (
            fast and mode_l != "circular"
        ):  # torch 缺 symmetric, 快路径仅 reflect/symmetric
            paddings = [[0, 0]] * ndim
            for i, (lp, r) in enumerate(pad_width):
                paddings[i] = [int(lp), int(r)]
            tf_mode = "REFLECT" if mode_l == "reflect" else "SYMMETRIC"
            if tf.executing_eagerly():
                return tf.pad(tf.convert_to_tensor(arr), paddings, mode=tf_mode).numpy()
            graph = tf.Graph()
            with graph.as_default():
                out_g = tf.pad(tf.convert_to_tensor(arr), paddings, mode=tf_mode)
            with tf.compat.v1.Session(graph=graph) as sess:
                return np.asarray(sess.run(out_g))

        def build(x):
            for i, (lp, r) in enumerate(pad_width):
                lp, r = int(lp), int(r)
                if lp == 0 and r == 0:
                    continue
                n = tf.shape(x)[i]
                total = n + lp + r
                ks = tf.range(total)
                if mode_l == "circular" or arr.shape[i] == 1:
                    idx = tf.math.floormod(ks - lp, n)
                elif mode_l == "reflect":
                    t = tf.math.floormod(ks - lp, 2 * n - 2)
                    idx = tf.where(t < n, t, 2 * n - 2 - t)
                else:  # symmetric
                    t = tf.math.floormod(ks - lp, 2 * n)
                    idx = tf.where(t < n, t, 2 * n - 1 - t)
                x = tf.gather(x, idx, axis=i)
            return x

        if tf.executing_eagerly():
            return build(tf.convert_to_tensor(arr)).numpy()
        graph = tf.Graph()
        with graph.as_default():
            out_g = build(tf.convert_to_tensor(arr))
        with tf.compat.v1.Session(graph=graph) as sess:
            return np.asarray(sess.run(out_g))
    nz = [i for i, (lp, r) in enumerate(pad_width) if lp or r]
    if not nz:
        return arr.copy()
    if ndim not in (2, 3, 4) or nz[0] == 0:
        return _tf_pad_compose(arr, pad_width, mode_l)
    # Remaining cases pad trailing axes of a 2D/3D/4D tensor.
    torch_mode = {"edge": "replicate", "circular": "circular"}[mode_l]
    flat = [0] * (2 * (ndim - 1))
    for i in range(1, ndim):
        flat[2 * (ndim - 1 - i)] = int(pad_width[i][0])
        flat[2 * (ndim - 1 - i) + 1] = int(pad_width[i][1])
    return torch.nn.functional.pad(
        torch.from_numpy(np.ascontiguousarray(arr)), flat, mode=torch_mode
    ).numpy()


__golden__ = {"kernel": {"mirror_pad": "mirror_pad_golden"}}

__spec__ = {"mirror_pad": "MirrorPadTestSpec"}


def _numpy_bfloat16():
    try:
        from ml_dtypes import bfloat16
    except ModuleNotFoundError:
        try:
            import tensorflow

            bfloat16 = tensorflow.bfloat16.as_numpy_dtype
        except ModuleNotFoundError:
            raise RuntimeError(
                "ml-dtypes or tensorflow is needed to support bfloat16 dtype!!! "
                "Please install with `pip3 install ml-dtypes` or `pip3 install tensorflow`"
            )
    return bfloat16


def _numpy_to_torch_tensor(np_array):
    import torch

    if np_array is None:
        return None
    np_dtype = np_array.dtype.name
    if "bfloat16" in np_dtype:
        np_int16 = np_array.view(dtype=np.int16)
        t_int16 = torch.from_numpy(np_int16)
        return t_int16.view(torch.bfloat16)
    else:
        return torch.from_numpy(np_array)


def _torch_to_numpy_tensor(torch_tensor):
    import torch

    if torch_tensor is None:
        return None
    if not isinstance(torch_tensor, torch.Tensor):
        raise RuntimeError(f"Only support torch.Tensor. But got {type(torch_tensor)}")
    torch_dtype = torch_tensor.dtype
    if torch_dtype == torch.bfloat16:
        t_int16 = torch_tensor.view(torch.int16)
        np_int16 = t_int16.numpy()
        return np_int16.view(dtype=_numpy_bfloat16())
    else:
        return torch_tensor.numpy()


def _pad_and_slice(arr, pad_width, mode="reflect"):
    pad_width = tuple(pad_width)
    if len(pad_width) != arr.ndim:
        raise ValueError(
            f"pad_width length ({len(pad_width)}) must match array ndim ({arr.ndim})"
        )

    result = arr

    for i, (left, right) in enumerate(pad_width):
        if left == 0 and right == 0:
            continue

        if left > 0 or right > 0:
            pad_tuple = [(0, 0)] * arr.ndim
            lv = left if left > 0 else 0
            rv = right if right > 0 else 0
            pad_tuple[i] = (lv, rv)
            result = _framework_pad(result, pad_tuple, mode)

        if left < 0 or right < 0:
            crop_left = abs(left) if left < 0 else 0
            crop_right = abs(right) if right < 0 else 0
            slice_start = crop_left
            slice_end = -crop_right if crop_right > 0 else None
            slices = [slice(None)] * arr.ndim
            slices[i] = slice(slice_start, slice_end)
            result = result[tuple(slices)]

    return result


def mirror_pad_golden(x, paddings, mode="REFLECT", **kwargs):
    """
    Kernel golden for mirror_pad.
    All the parameters follow @mirror_pad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    import torch

    pad_shape = np.reshape(paddings, (len(x.shape), 2))
    if mode == "REFLECT" or mode == "SYMMETRIC":
        if mode == "SYMMETRIC":
            np_mode = "symmetric"
        else:
            np_mode = "reflect"

        orig_dtype = x.dtype
        if orig_dtype.name == "bfloat16":
            x = _numpy_to_torch_tensor(x).to(torch.float32).numpy()

        neg_pad = _numpy_to_torch_tensor(np.array(pad_shape))
        neg_mask = neg_pad < 0
        indices = torch.nonzero(neg_mask, as_tuple=False)

        if indices.size(0) == 0:
            golden = _framework_pad(x, pad_shape, np_mode)
        else:
            golden = _pad_and_slice(x, pad_shape, mode=np_mode)

        if orig_dtype.name == "bfloat16":
            golden = _torch_to_numpy_tensor(
                _numpy_to_torch_tensor(golden).to(torch.bfloat16)
            )
    else:
        raise ValueError(f"Unsupported pad mode: {mode}")

    return golden


class MirrorPadImpl:
    def __call__(self, x, paddings, mode="REFLECT"):
        import torch

        mode = mode.lower()
        if mode != "reflect":
            raise NotImplementedError("PyTorch mirror pad only supports REFLECT mode")
        x_t = (
            x if isinstance(x, torch.Tensor) else _numpy_to_torch_tensor(np.asarray(x))
        )
        pads = torch.as_tensor(paddings).reshape(-1).tolist()
        pads = [[int(pads[2 * i]), int(pads[2 * i + 1])] for i in range(len(pads) // 2)]
        ndim = x_t.dim()
        offset = ndim - len(pads)

        # PyTorch 非 constant pad 仅支持指定 rank 的尾部 1～3 维。
        pad_width = [(0, 0)] * offset + pads
        active = [i for i, pair in enumerate(pad_width) if any(pair)]
        if not active:
            return x_t.clone()
        count = ndim - active[0]
        if count not in (1, 2, 3) or ndim not in (count + 1, count + 2):
            raise NotImplementedError(
                "PyTorch non-constant pad requires trailing 1-3 dimensions "
                "with one or two leading dimensions"
            )
        result = x_t
        for axis, (left, right) in enumerate(pad_width):
            if left > 0 or right > 0:
                flat = [0] * (2 * count)
                pos = 2 * (ndim - 1 - axis)
                flat[pos : pos + 2] = [max(0, left), max(0, right)]
                result = torch.nn.functional.pad(result, flat, mode="reflect")
            if left < 0 or right < 0:
                slices = [slice(None)] * ndim
                slices[axis] = slice(max(0, -left), right if right < 0 else None)
                result = result[tuple(slices)]
        return result


class MirrorPadTFImpl:
    def __call__(self, x, paddings, mode="REFLECT"):
        import tensorflow as tf
        import torch

        mode = mode.lower()
        if mode not in ("reflect", "symmetric"):
            raise NotImplementedError(f"TensorFlow pad does not support mode: {mode}")
        x_np = (
            _torch_to_numpy_tensor(x.detach().cpu())
            if isinstance(x, torch.Tensor)
            else np.asarray(x)
        )
        pads_np = (
            paddings.detach().cpu().numpy()
            if isinstance(paddings, torch.Tensor)
            else np.asarray(paddings)
        )
        pads = pads_np.astype(np.int64)
        pads = pads.reshape(-1, 2)
        if len(pads) > x_np.ndim:
            raise ValueError("paddings length exceeds input rank")
        pads = [(0, 0)] * (x_np.ndim - len(pads)) + pads.tolist()

        def build():
            result = tf.convert_to_tensor(x_np)
            for axis, (left, right) in enumerate(pads):
                slices = [slice(None)] * x_np.ndim
                slices[axis] = slice(max(0, -left), right if right < 0 else None)
                if left > 0 or right > 0:
                    widths = [[0, 0] for _ in range(x_np.ndim)]
                    widths[axis] = [max(0, left), max(0, right)]
                    result = tf.pad(
                        result,
                        widths,
                        mode=mode.upper(),
                    )
                result = result[tuple(slices)]
            return result

        if tf.executing_eagerly():
            out_np = build().numpy()
        else:
            graph = tf.Graph()
            with graph.as_default():
                out_g = build()
            with tf.compat.v1.Session(graph=graph) as sess:
                out_np = sess.run(out_g)
        return np.asarray(out_np)


class MirrorPadTestSpec:
    def golden(x, paddings, mode="REFLECT", **kwargs):
        return [mirror_pad_golden(x, paddings, mode, **kwargs)]

    third_party = {"torch": MirrorPadImpl, "tf": MirrorPadTFImpl}

    tolerance = {
        "float32": {"standard": "binary_equal"},
        "float16": {"standard": "binary_equal"},
        "bfloat16": {"standard": "binary_equal"},
    }
