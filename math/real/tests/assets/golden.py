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
"""TestSpec definitions for Real on GEIR/Kernel, ACLNN, Torch Single, and Torch Graph paths."""

import numpy
import torch

__spec__ = {
    "real": "RealKernelTestSpec",
    "Real": "RealKernelTestSpec",
    "aclnnReal": "AclnnRealTestSpec",
    "torch.real": "TorchRealTestSpec",
}


class RealGraph(torch.nn.Module):
    """Graph module consumed by TTK E2E torch.compile modes."""

    def forward(self, input):
        return torch.real(input)


class TorchRealReference:
    """Third-party reference absorbing TTK attrs (e.g. Tout).

    torch.real is a C builtin: the remote three-phase invoke cannot bind extra
    kwargs (Tout) or inspect a signature, returning HTTP 400 GOLDEN_FAILURE.
    A callable class absorbs them and resolves the input tensor by name.
    """

    def __call__(self, *args, input=None, **kwargs):
        value = input if input is not None else kwargs.get("self")
        if value is None:
            value = kwargs.get("selfT")
        if value is None and args:
            value = args[0]
        return [torch.real(value)]


class RealKernelTestSpec:
    """Shared GEIR/Kernel spec registered by raw CSV ``op_name`` variants."""

    third_party = {"torch": TorchRealReference}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L2"},
        "float32": {"standard": "cross_check", "level": "L2"},
    }

    def golden(x, **kwargs):
        """Explicit golden: unpack complex32 fp16 ``[shape..., 2]`` storage.

        GEIR/Kernel golden receives numpy arrays. complex32 has no numpy
        native dtype and ships as float16 ``[shape..., 2]`` (real/imag
        interleaved); under Promote (cross_check) the framework has already
        lifted it to native complex64/complex128. The string form
        ``"torch.real"`` dispatches through ``__call_torch_api`` without the
        ``is_complex32`` reinterpret flag, so ``torch.real`` sees a plain
        float16 tensor (a no-op on real dtypes) and returns the packed array
        unchanged — shape_size doubles and imag leaks in. Discriminate by the
        framework-injected ``input_dtypes`` kwarg (not by shape alone: real
        float16 inputs with a trailing 2 are legal for Real) and take the
        real part explicitly.
        """
        dtypes = kwargs.get("input_dtypes") or ()
        dtype = dtypes[0] if isinstance(dtypes, (list, tuple)) and dtypes else None
        if (
            dtype == "complex32"
            and getattr(x, "dtype", None) == numpy.dtype("float16")
            and getattr(x, "shape", ())[-1:] == (2,)
        ):
            return [numpy.ascontiguousarray(x[..., 0])]
        return [numpy.real(x)]


class AclnnRealTestSpec:
    """ACLNN spec registered by CSV ``api_name``."""

    golden = "torch.real"
    third_party = {"torch": TorchRealReference}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L2"},
        "float32": {"standard": "cross_check", "level": "L2"},
    }


class TorchRealTestSpec:
    """Shared Torch Single/Graph spec registered by CSV ``api_name``."""

    def golden(input, **kwargs):
        return [torch.real(input)]

    third_party = {"torch": TorchRealReference}
    torch_graph = RealGraph
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L2"},
        "float32": {"standard": "cross_check", "level": "L2"},
    }
