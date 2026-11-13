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

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None


__golden__ = {"kernel": {"pad_v3_grad_replication": "pad_v3_grad_replication_golden"}}

__spec__ = {
    "pad_v3_grad_replication": "PadV3GradReplicationTestSpec",
    "aclnnReplicationPad3dBackward": "AclnnReplicationPad3dBackwardPublicSpec",
}


def _cal_out_shape(in_shape, paddings):
    out_shape = []
    offset = len(in_shape) - len(paddings)
    for i in range(len(in_shape)):
        if i < len(in_shape) - len(paddings):
            out_shape.append(in_shape[i])
        else:
            out_shape.append(
                in_shape[i] - paddings[i - offset][0] - paddings[i - offset][1]
            )
    return out_shape


def _np_to_torch(arr):
    import torch

    if isinstance(arr, torch.Tensor):
        return arr
    arr = np.asarray(arr)
    if ml_dtypes is not None and arr.dtype == ml_dtypes.bfloat16:
        return torch.from_numpy(np.ascontiguousarray(arr).view(np.int16)).view(
            torch.bfloat16
        )
    return torch.from_numpy(arr)


def _torch_to_np(t):
    import torch

    if not isinstance(t, torch.Tensor):
        return np.asarray(t)
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        if ml_dtypes is not None:
            return t.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
        return t.float().numpy()
    return t.numpy()


def _norm_padding(padding):
    import torch

    if padding is None:
        return [0, 0]
    if isinstance(padding, np.ndarray):
        padding = padding.tolist()
    if isinstance(padding, torch.Tensor):
        padding = padding.detach().cpu().tolist()
    return [int(p) for p in padding]


def _kernel_paddings(paddings):
    import torch

    if isinstance(paddings, torch.Tensor):
        paddings = paddings.detach().cpu().numpy()
    return np.array(paddings).astype(np.int64).reshape(-1, 2)


def _calc_dtype(dtype):
    import torch

    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _run_public_backward(grad_output, self_tensor, pad_whd):
    """Compute replication-pad backward with the public F.pad + autograd API."""
    import torch.nn.functional as F

    grad_output = _np_to_torch(grad_output)
    self_tensor = _np_to_torch(self_tensor)
    if not (grad_output.is_floating_point() or grad_output.is_complex()) or not (
        self_tensor.is_floating_point() or self_tensor.is_complex()
    ):
        raise NotImplementedError(
            "replication pad backward requires floating-point or complex tensors"
        )

    calc_dtype = _calc_dtype(self_tensor.dtype)
    self_calc = self_tensor.detach().clone().to(calc_dtype)
    self_calc.requires_grad_(True)
    out = F.pad(self_calc, tuple(int(v) for v in pad_whd), mode="replicate")
    out.backward(grad_output.to(calc_dtype))
    return self_calc.grad.to(self_tensor.dtype)


def _kernel_self_shape(grad_output, paddings_arr):
    return _cal_out_shape(list(grad_output.shape), paddings_arr.tolist())


def _trim_leading_zero_pairs(paddings_arr):
    i = 0
    while i < len(paddings_arr) and not paddings_arr[i].any():
        i += 1
    return paddings_arr[i:]


def _check_kernel_paddings(grad_output, paddings_arr):
    pad_dim = len(paddings_arr)
    if pad_dim not in (1, 2, 3) or grad_output.dim() not in (pad_dim + 1, pad_dim + 2):
        raise NotImplementedError(
            "PyTorch replication padding backward requires trailing 1-3 "
            "dimensions with one or two leading dimensions"
        )
    return pad_dim


def pad_v3_grad_replication_golden(x, paddings, **kwargs):
    """
    Kernel golden for pad_v3_grad_replication using public torch APIs.

    All the parameters follow @pad_v3_grad_replication_def.cpp without outputs.
    Inputs may be numpy.ndarray or torch.Tensor. The kernel path receives
    paddings in D/H/W pair order; F.pad consumes W/H/D order, so the pairs are
    reversed before the public API call.
    """
    import torch

    grad_output = _np_to_torch(x)
    paddings_arr = _kernel_paddings(paddings)
    y_shape = _kernel_self_shape(grad_output, paddings_arr)
    pad_pairs = _trim_leading_zero_pairs(paddings_arr)
    if not len(pad_pairs):
        return _torch_to_np(grad_output.clone())
    _check_kernel_paddings(grad_output, pad_pairs)
    pad_whd = [p for pair in reversed(pad_pairs) for p in pair]
    self_t = torch.zeros(y_shape, dtype=grad_output.dtype, device=grad_output.device)
    grad = _run_public_backward(grad_output, self_t, pad_whd)
    return _torch_to_np(grad)


class PadV3GradReplicationImpl:
    def __call__(self, x, paddings, **kwargs):
        import torch

        grad_output = _np_to_torch(x)
        paddings_arr = _kernel_paddings(paddings)
        y_shape = _kernel_self_shape(grad_output, paddings_arr)
        pad_pairs = _trim_leading_zero_pairs(paddings_arr)
        if not len(pad_pairs):
            return grad_output.clone()
        _check_kernel_paddings(grad_output, pad_pairs)
        pad_whd = [p for pair in reversed(pad_pairs) for p in pair]
        self_t = torch.zeros(
            y_shape, dtype=grad_output.dtype, device=grad_output.device
        )
        return _run_public_backward(grad_output, self_t, pad_whd)


class PadV3GradReplicationTestSpec:
    def golden(x, paddings, **kwargs):
        return [pad_v3_grad_replication_golden(x, paddings, **kwargs)]

    third_party = {"torch": PadV3GradReplicationImpl}


class ReplicationPad3dBackwardPublicImpl:
    """third_party for aclnnReplicationPad3dBackward using F.pad + autograd."""

    def __call__(self, gradOutput, padding=None, *args, **kwargs):
        self_tensor = args[0] if args else kwargs.get("self")
        if self_tensor is None:
            raise ValueError("self tensor is required")
        pad = _norm_padding(padding)
        if len(pad) not in (2, 4, 6):
            raise ValueError(f"Unsupported padding length: {len(pad)}")
        return _run_public_backward(gradOutput, self_tensor, pad)


class AclnnReplicationPad3dBackwardPublicSpec:
    """aclnn golden using F.pad + autograd (public API)."""

    def golden(gradOutput, self, padding=None, gradInput=None, **kwargs):
        pad = _norm_padding(padding)
        if len(pad) not in (2, 4, 6):
            raise ValueError(f"Unsupported padding length: {len(pad)}")
        return [_run_public_backward(gradOutput, self, pad)]

    third_party = {"torch": ReplicationPad3dBackwardPublicImpl}
