# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Bernoulli operator-side contract with ops-test-kit PR #578. No NPU is required."""

import ast
import csv
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch


def _asset(name):
    file = Path(__file__).resolve().parents[1] / "assets" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"bernoulli_generator_{name}", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


golden = _asset("golden")
try:
    from ttk.core_modules.common.testcase.api_param import (
        GeneratorSpec,
        _normalize_generator_spec,
    )
    from ttk.core_modules.framework_api.object_spec import (
        materialize_specs,
        strip_specs,
    )
    from ttk.core_modules.framework_api.testcase import TestcaseE2e
except ImportError:
    GeneratorSpec = None


class BernoulliE2EGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.remote_env = mock.patch.dict(os.environ, {"BERNOULLI_GOLDEN_HOST": ""})
        self.remote_env.start()
        self.addCleanup(self.remote_env.stop)
        self.target = torch.zeros((4, 8), dtype=torch.float16)
        self.prob = torch.full((4, 1), 0.5, dtype=torch.float16)

    def _result(self, spec, tensor=False, **kwargs):
        return golden.e2e_bernoulli_golden(
            self.target, self.prob if tensor else 0.5, generator=spec, **kwargs
        )

    def test_seed_only_dict_defaults_offset_to_zero(self):
        for tensor in (False, True):
            self.assertTrue(
                torch.equal(
                    self._result({"seed": 603}, tensor),
                    self._result({"seed": 603, "offset": 0}, tensor),
                )
            )

    def test_legacy_random_flow_is_preserved_for_declared_spec(self):
        for tensor in (False, True):
            spec = {"seed": 53073689, "offset": 4}
            actual = self._result(spec, tensor)
            expected = (
                golden.aclnn_bernoulli_tensor_golden(
                    self.target, self.prob, 53073689, 4
                )
                if tensor
                else golden.aclnn_bernoulli_golden(self.target, 0.5, 53073689, 4)
            )
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(spec, {"seed": 53073689, "offset": 4})

    def test_full_uint64_seed_domain(self):
        for seed in (-(2**63), -1, 0, 2**63, 2**64 - 1):
            with self.subTest(seed=seed):
                result = self._result({"seed": seed, "offset": 4}, tensor=True)
                self.assertEqual(tuple(result.shape), (4, 8))
                self.assertEqual(
                    golden._e2e_generator_seed_offset({"seed": seed}), (seed % 2**64, 0)
                )
        self.assertTrue(
            torch.equal(
                self._result({"seed": -1}, True),
                self._result({"seed": 2**64 - 1}, True),
            )
        )

    def test_missing_or_materialized_generator_does_not_silently_use_seed_zero(self):
        for spec in (None, "None", torch.Generator()):
            with self.subTest(spec=type(spec).__name__):
                with self.assertRaisesRegex(ValueError, "generator spec"):
                    self._result(spec)

    def test_top_level_seed_offset_are_rejected(self):
        for extra in ({"seed": 42}, {"offset": 4}, {"seed": 99, "offset": 256}):
            with self.assertRaisesRegex(ValueError, "attributes.generator"):
                self._result({"seed": 42, "offset": 4}, **extra)

    def test_invalid_specs_fail_in_golden(self):
        invalid = (
            603,
            True,
            {},
            {"seed": True},
            {"seed": 1.5},
            {"seed": -(2**63) - 1},
            {"seed": 2**64},
            {"seed": 42, "offset": -4},
            {"seed": 42, "offset": 1},
            {"seed": 42, "offset": True},
            {"seed": 42, "offset": 4.0},
            {"seed": 42, "extra": 1},
        )
        for spec in invalid:
            with self.subTest(spec=spec):
                with self.assertRaises(ValueError):
                    self._result(spec)

    def test_golden_does_not_materialize_or_advance_a_generator(self):
        before = torch.random.get_rng_state().clone()
        with mock.patch.object(
            torch,
            "Generator",
            side_effect=AssertionError("golden must not create a Generator"),
        ):
            self._result({"seed": 42, "offset": 4})
            self._result({"seed": 42, "offset": 4}, tensor=True)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_rank0_and_empty_outputs(self):
        spec = {"seed": 42, "offset": 4}
        scalar = golden.e2e_bernoulli_golden(
            torch.empty(()), torch.tensor(0.5), generator=spec
        )
        self.assertEqual(scalar.shape, torch.Size([]))
        self.assertIn(scalar.item(), (0, 1))
        empty = golden.e2e_bernoulli_golden(torch.empty((0, 8)), generator=spec)
        self.assertEqual(tuple(empty.shape), (0, 8))

    def test_no_operator_pre_round_rng_hook_remains(self):
        self.assertNotIn("torch.ops.aten.bernoulli_.float", golden.__spec__)
        self.assertNotIn("torch.ops.aten.bernoulli_.Tensor", golden.__spec__)
        self.assertFalse(hasattr(golden, "BernoulliE2ETestSpec"))

    def test_all_csv_cases_use_generator_specs(self):
        path = (
            Path(__file__).resolve().parents[1] / "st/arch35/ttk_e2e_bernoulli_st.csv"
        )
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertGreaterEqual(len(rows), 10)
        for row in rows:
            self.assertNotIn(None, row)
            attrs = ast.literal_eval(row["attributes"])
            self.assertNotIn("seed", attrs)
            self.assertNotIn("offset", attrs)
            self.assertEqual(ast.literal_eval(row["inplace_input_indexes"]), (0,))
            if row["input_data_ranges"]:
                for value_range in ast.literal_eval(row["input_data_ranges"]):
                    self.assertIsInstance(value_range, (tuple, list))
                    self.assertEqual(len(value_range), 2)
            golden._e2e_generator_seed_offset(attrs["generator"])


@unittest.skipUnless(GeneratorSpec is not None, "requires ops-test-kit PR #578")
class BernoulliGeneratorFrameworkContractTest(unittest.TestCase):
    def test_merged_framework_and_operator_reject_integer_sugar(self):
        with self.assertRaisesRegex(ValueError, "GENERATOR_SPEC_INVALID"):
            _normalize_generator_spec(603)
        with self.assertRaises(ValueError):
            golden._e2e_generator_seed_offset(603)

    def test_real_framework_plugin_dispatch_uses_generator_spec(self):
        from ttk.core_modules.framework_api.golden import generate_golden

        assets = Path(__file__).resolve().parents[1] / "assets"
        for tensor_prob in (False, True):
            case = TestcaseE2e()
            case.testcase_name = "generator_plugin_dispatch"
            case.api_name = (
                "torch.ops.aten.bernoulli_.Tensor"
                if tensor_prob
                else "torch.ops.aten.bernoulli_.float"
            )
            case.tensor_view_shapes = ((4, 8), (4, 1)) if tensor_prob else ((4, 8),)
            case.tensor_dtypes = ("float32",) * len(case.tensor_view_shapes)
            case.tensor_formats = ("ND",) * len(case.tensor_view_shapes)
            case.output_tensor_indexes = ()
            case.inplace_input_indexes = (0,)
            case.golden_api = ""
            case.attributes = {"generator": {"seed": 603, "offset": 64}}
            if not tensor_prob:
                case.attributes["p"] = 0.5
            case.tensors = [torch.zeros(shape) for shape in case.tensor_view_shapes]
            if tensor_prob:
                case.tensors[1].fill_(0.5)
            result = generate_golden(
                case,
                [t.numpy() for t in case.tensors],
                plugin_path=str(assets),
                switches=SimpleNamespace(golden_mode="Enable"),
            )
            expected = golden.e2e_bernoulli_golden(
                case.tensors[0],
                case.tensors[1] if tensor_prob else 0.5,
                generator={"seed": 603, "offset": 64},
            )
            np.testing.assert_array_equal(result[0], expected.numpy())

    def test_normalized_generator_spec_reaches_plugin_unchanged(self):
        for api, tensors in (
            ("torch.ops.aten.bernoulli_.float", [torch.zeros((4, 8))]),
            (
                "torch.ops.aten.bernoulli_.Tensor",
                [torch.zeros((4, 8)), torch.ones((4, 1))],
            ),
        ):
            case = TestcaseE2e()
            case.testcase_name, case.api_name = "generator_contract", api
            case.tensor_view_shapes = tuple(tuple(t.shape) for t in tensors)
            case.tensor_dtypes = ("float32",) * len(tensors)
            case.output_tensor_indexes = ()
            case.attributes = {"generator": {"seed": 42, "offset": 256}}
            if api.endswith(".float"):
                case.attributes["p"] = 0.5
            plan = case.get_param_plan()
            self.assertIsNotNone(plan)
            args, kwargs, extra = plan.build_args(tensors)
            self.assertIsInstance(kwargs["generator"], GeneratorSpec)
            self.assertEqual(kwargs["generator"], {"seed": 42, "offset": 256})
            with mock.patch.object(
                golden, "_golden_bool", return_value=np.zeros(32, dtype=bool)
            ) as sample:
                result = golden.e2e_bernoulli_golden(*args, **kwargs, **extra)
            self.assertEqual(tuple(result.shape), (4, 8))
            self.assertEqual(sample.call_args.args[1:3], (42, 256))
            self.assertEqual(case.attributes["generator"], {"seed": 42, "offset": 256})

    def test_framework_materializes_and_resets_only_execution_side(self):
        spec = _normalize_generator_spec({"seed": 603, "offset": 64})
        kwargs = {"generator": spec}
        backend = SimpleNamespace(is_npu=lambda: True, device_type=lambda: "npu")
        made = []

        class FakeGenerator:
            def __init__(self, device):
                self.device = device
                made.append(self)

            def manual_seed(self, value):
                self.seed = value

            def set_offset(self, value):
                self.offset = value

        with mock.patch.object(torch, "Generator", FakeGenerator):
            reset = materialize_specs([], kwargs, backend)
            for _ in range(4):
                self.assertEqual(
                    (kwargs["generator"].seed, kwargs["generator"].offset), (603, 64)
                )
                kwargs["generator"].offset += 100
                reset()
        self.assertEqual(len(made), 5)
        self.assertEqual(spec, {"seed": 603, "offset": 64})
        self.assertEqual(golden._e2e_generator_seed_offset(spec), (603, 64))

    def test_graph_strip_preserves_declared_spec_for_golden(self):
        spec = _normalize_generator_spec({"seed": 603, "offset": 64})
        kwargs = {"generator": spec}
        self.assertEqual(strip_specs([], kwargs), [spec])
        self.assertIsNone(kwargs["generator"])
        self.assertEqual(golden._e2e_generator_seed_offset(spec), (603, 64))


if __name__ == "__main__":
    unittest.main()
