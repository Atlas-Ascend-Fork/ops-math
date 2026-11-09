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
"""CPU regression tests; ATK transport is stubbed, operator callbacks are real."""

import ctypes
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AtkBaseStub:
    def init_by_input_data(self, data):
        return list(data.args), data.outputs


def load_atk_callbacks():
    registry = {}

    def register(name):
        def decorate(cls):
            registry[name] = cls
            return cls

        return decorate

    bindings = {
        "atk.configs.dataset_config": {"InputDataset": object},
        "atk.configs.results_config": {"TaskResult": object},
        "atk.tasks.api_execute": {"register": register},
        "atk.tasks.api_execute.base_api": {"BaseApi": object},
        "atk.tasks.api_execute.aclnn_base_api": {"AclnnBaseApi": AtkBaseStub},
        "atk.tasks.backends.lib_interface.acl_wrapper": {"TensorPtr": ctypes.c_void_p},
    }
    modules = {}
    for name, values in bindings.items():
        module = types.ModuleType(name)
        module.__dict__.update(values)
        modules[name] = module
    with patch.dict(sys.modules, modules):
        load_module(
            "scale_atk_under_test", ROOT / "tests/st/aclnnScale/executor_aclnnScale.py"
        )
    return registry


class ScaleReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.golden = staticmethod(
            load_module(
                "scale_golden_under_test", ROOT / "tests/assets/golden.py"
            ).aclnn_scale_golden
        )
        cls.callbacks = load_atk_callbacks()

    def evaluate(self, x, scale, bias, axis=0, axes=0):
        data = types.SimpleNamespace(
            kwargs=dict(
                x=x, scale=scale, bias=bias, axis=axis, numAxes=axes, scaleFromBlob=True
            )
        )
        return [
            # Simulate TTK's CPU promotion at the caller, not inside the Golden.
            self.golden(
                x.double(),
                scale.double(),
                bias.double() if bias is not None else None,
                axis,
                axes,
                True,
            )[0].to(x.dtype),
            self.callbacks["function_aclnn_scale"]()(data),
        ]

    def test_finite_result_after_large_product_cancellation(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = torch.full((2, 3), 2.0**64, dtype=dtype)
                scale = torch.tensor(2.0**64, dtype=dtype)
                bias = torch.tensor(-(2.0**127), dtype=dtype)
                # Only TTK promotes the CPU reference; ATK keeps its existing FP32 path.
                result = self.golden(
                    x.double(), scale.double(), bias.double(), 0, 0, True
                )[0].to(dtype)
                self.assertEqual(result.dtype, dtype)
                self.assertTrue(torch.equal(result, torch.full_like(x, 2.0**127)))

    def test_scalar_bias_and_omitted_bias(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            x = torch.tensor([1, 2, 3], dtype=dtype)
            scale = torch.tensor(2, dtype=dtype)
            for bias in (None, torch.tensor(3, dtype=dtype)):
                with self.subTest(dtype=dtype, omitted=bias is None):
                    expected = torch.tensor(
                        [2, 4, 6] if bias is None else [5, 7, 9], dtype=dtype
                    )
                    for result in self.evaluate(x, scale, bias):
                        self.assertTrue(torch.equal(result, expected))

    def test_axis_broadcast_and_empty_output(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for empty in (False, True):
                with self.subTest(dtype=dtype, empty=empty):
                    x = torch.ones((0 if empty else 2, 3, 4), dtype=dtype)
                    scale = torch.tensor([1, 2, 3, 4], dtype=dtype)
                    bias = torch.tensor([4, 3, 2, 1], dtype=dtype)
                    for result in self.evaluate(x, scale, bias, axis=-1, axes=1):
                        self.assertEqual(result.shape, x.shape)
                        self.assertTrue(torch.equal(result, torch.full_like(x, 5)))

    def test_nan_inf_and_real_overflow_still_propagate(self):
        x = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), torch.finfo(torch.float32).max]
        )
        for result in self.evaluate(x, torch.tensor(2.0), None):
            self.assertTrue(torch.isnan(result[0]))
            self.assertTrue(
                torch.equal(
                    result[1:],
                    torch.tensor([float("inf"), -float("inf"), float("inf")]),
                )
            )

    def test_three_party_specs_preserve_supplied_reference_dtype(self):
        module = load_module("scale_specs_under_test", ROOT / "tests/assets/golden.py")
        self.assertEqual(set(module.__spec__), {"scale", "aclnnScale"})
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            x = torch.ones(3, dtype=dtype)
            scale = torch.tensor(2, dtype=dtype)
            golden = module.ScaleAclnnSpec.golden(x, scale, None, 0, 0, True)[0]
            third = module.ScaleAclnnSpec.third_party["torch"]()(
                x, scale, None, 0, 0, True
            )[0]
            self.assertEqual(golden.dtype, dtype)
            self.assertEqual(third.dtype, dtype)
            self.assertEqual(
                module.ScaleAclnnSpec.tolerance[str(dtype).split(".")[-1]]["standard"],
                "cross_check",
            )

    def test_reference_broadcast_uses_axis(self):
        module = load_module("scale_window_specs", ROOT / "tests/assets/golden.py")
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            x = (torch.arange(120) % 11 - 5).reshape(2, 3, 4, 5).to(dtype)
            scale = torch.tensor([[1], [2], [3]], dtype=dtype)
            for spec in (module.ScaleKernelSpec, module.ScaleAclnnSpec):
                for compute in (spec.golden, spec.third_party["torch"]()):
                    actual = compute(x, scale, scale, 1, 1, False)[0]
                    expected = x * scale.reshape(1, 3, 1, 1) + scale.reshape(1, 3, 1, 1)
                    self.assertTrue(torch.equal(actual, expected))
                    scale = scale.reshape(1, 3, 1, 1)
                    expanded = compute(x, scale, scale, 1, 1, False)[0]
                    self.assertTrue(torch.equal(expanded, expected))
                    scale = scale.reshape(3, 1)

    def test_reference_does_not_duplicate_input_validation(self):
        module = load_module("scale_math_specs", ROOT / "tests/assets/golden.py")
        for spec in (module.ScaleKernelSpec, module.ScaleAclnnSpec):
            for compute in (spec.golden, spec.third_party["torch"]()):
                # Device rank limits and bias-shape contracts are tested at the
                # operator entry, rather than imposed on the numerical reference.
                for shape in ((2,) * 8, (2,) * 9, (0,) + (2,) * 8):
                    x = torch.ones(shape)
                    result = compute(
                        x, torch.full((2,), 3.0), None, len(shape) - 1, 1, True
                    )[0]
                    self.assertTrue(torch.equal(result, x * 3))
                x = torch.ones(2, 3)
                result = compute(x, torch.ones(3), torch.ones(1, 3), 1, 1, True)[0]
                self.assertTrue(torch.equal(result, x + 1))
                result = compute(x, torch.ones(3), None, 1, -2, False)[0]
                self.assertTrue(torch.equal(result, x))

    def test_scalar_shapes_and_empty_bias(self):
        module = load_module("scale_scalar_specs", ROOT / "tests/assets/golden.py")
        x = torch.ones((2, 3))
        for spec in (module.ScaleKernelSpec, module.ScaleAclnnSpec):
            for compute in (spec.golden, spec.third_party["torch"]()):
                for shape in ((), (1,), (1, 1, 1), (1,) * 8):
                    scale = torch.full(shape, 2.0)
                    self.assertTrue(
                        torch.equal(compute(x, scale, None, 1, 0, True)[0], x * 2)
                    )
                self.assertTrue(
                    torch.equal(
                        compute(x, torch.ones(3), torch.empty(0), 1, 1, True)[0], x
                    )
                )

    def test_device_bias_identity_is_preserved(self):
        for bias in (
            torch.tensor(3.0),
            torch.tensor([3.0]),
            torch.empty(0),
            torch.empty(0, 4),
        ):
            with self.subTest(shape=tuple(bias.shape)):
                handle = object()
                outputs = object()
                data = types.SimpleNamespace(
                    kwargs={"bias": bias},
                    args=[object(), object(), handle],
                    outputs=outputs,
                )
                args, actual_outputs = self.callbacks[
                    "aclnn_scale"
                ]().init_by_input_data(data)
                self.assertIs(args[2], handle)
                self.assertIs(actual_outputs, outputs)

    def test_omitted_bias_gets_null_handle(self):
        data = types.SimpleNamespace(
            kwargs={"bias": None}, args=[object(), object(), None], outputs=object()
        )
        args, _ = self.callbacks["aclnn_scale"]().init_by_input_data(data)
        self.assertIsInstance(args[2], ctypes.c_void_p)
        self.assertIsNone(args[2].value)


if __name__ == "__main__":
    unittest.main()
