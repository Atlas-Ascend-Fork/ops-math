# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run with python -m unittest discover -s random/stateless_bernoulli/tests/ut -p test_bernoulli_output_overlap.py.

These host tests validate golden rejection and output-view transport. Actual
ACLNN rejection is asserted separately by test_aclnn_bernoulli_l2.cpp.
"""

import importlib.util
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import torch

try:
    from ttk.core_modules.common.xpu.binding import invoke_third_party
except ImportError:
    invoke_third_party = None


def _load_asset(name):
    path = Path(__file__).resolve().parents[1] / "assets" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"bernoulli_overlap_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


golden = _load_asset("golden")
remote = _load_asset("bernoulli_remote_golden")
OVERLAP_ERROR = "more than one element of the written-to tensor"


class BernoulliOutputOverlapTest(unittest.TestCase):
    def setUp(self):
        self.self_t = torch.full((875, 205), -(2**-14), dtype=torch.float16)
        self.prob = torch.full((875, 1), 0.5, dtype=torch.float16)

    def test_original_out_rejected_before_local_or_remote_sampling(self):
        out = torch.empty(1, dtype=torch.float16).expand(875, 205)
        for func, prob in (
            (golden.aclnn_bernoulli_tensor_golden, self.prob),
            (golden.aclnn_bernoulli_golden, 0.5),
        ):
            with self.subTest(func=func.__name__):
                with mock.patch.object(golden, "_golden_bool") as sample:
                    with self.assertRaisesRegex(RuntimeError, OVERLAP_ERROR):
                        func(self.self_t, prob, 53073689, 4, out=out)
                    sample.assert_not_called()
        self.assertEqual(out.stride(), (0, 0))
        self.assertEqual(out.untyped_storage().nbytes(), 2)

    def test_inplace_and_e2e_check_self_before_sampling(self):
        out = torch.empty(1, dtype=torch.float16).expand(875, 205)
        calls = (
            lambda: golden.aclnn_inplace_bernoulli_golden(out, 0.5, 53073689, 4),
            lambda: golden.aclnn_inplace_bernoulli_tensor_golden(
                out, self.prob, 53073689, 4
            ),
            lambda: golden.e2e_bernoulli_golden(out, p=0.5),
            lambda: golden.e2e_bernoulli_golden(out, p=self.prob),
        )
        for index, call in enumerate(calls):
            with self.subTest(index=index):
                with mock.patch.object(golden, "_golden_bool") as sample:
                    with self.assertRaisesRegex(RuntimeError, OVERLAP_ERROR):
                        call()
                    sample.assert_not_called()

    def test_partial_broadcast_outputs_rejected(self):
        for shape, strides, storage_size in (((4, 5), (0, 1), 5), ((4, 5), (1, 0), 4)):
            with self.subTest(strides=strides):
                out = torch.empty(storage_size).as_strided(shape, strides)
                with self.assertRaisesRegex(RuntimeError, OVERLAP_ERROR):
                    golden._check_writable_tensor(out)

    def test_valid_layouts_and_empty_outputs_are_not_rejected(self):
        layouts = (
            ((5, 4), (1, 5), 20, 0),
            ((4, 5), (11, 2), 44, 1),
            ((1, 5), (0, 1), 5, 0),
            ((5, 1), (1, 0), 5, 0),
            ((0, 5), (0, 0), 0, 0),
            ((), (), 1, 0),
        )
        for shape, strides, size, offset in layouts:
            with self.subTest(shape=shape, strides=strides):
                out = torch.empty(size).as_strided(shape, strides, offset)
                prob = torch.ones(())
                result = golden.aclnn_bernoulli_tensor_golden(out, prob, out=out)
                self.assertEqual(tuple(result.shape), shape)
                self.assertTrue(torch.all(result == 1).item())
                golden.aclnn_bernoulli_golden(out, 1.0, out=out)
                golden.aclnn_inplace_bernoulli_golden(out, 1.0)
                golden.aclnn_inplace_bernoulli_tensor_golden(out, prob)
                golden.e2e_bernoulli_golden(out, p=1.0, generator={"seed": 0})

    def test_read_only_broadcasts_remain_valid(self):
        self_t = torch.empty(1).expand(5, 4)
        prob = torch.ones(1).expand(5, 4)
        out = torch.empty((4, 5)).t()
        result = golden.aclnn_bernoulli_tensor_golden(self_t, prob, out=out)
        self.assertTrue(torch.all(result == 1).item())
        # An omitted out is newly allocated, not an inplace write to self.
        result = golden.aclnn_bernoulli_tensor_golden(self_t, prob)
        self.assertTrue(torch.all(result == 1).item())

    def test_registered_inplace_goldens_are_distinct(self):
        for api in ("aclnnInplaceBernoulli", "aclnnInplaceBernoulliTensor"):
            function = getattr(golden, golden.__golden__["aclnn"][api])
            prob = self.prob if api.endswith("Tensor") else 0.5
            out = torch.empty(1, dtype=torch.float16).expand(875, 205)
            with self.assertRaisesRegex(RuntimeError, OVERLAP_ERROR):
                function(out, prob)

    def test_xpu_adapters_do_not_hide_overlap(self):
        out = torch.empty(1).expand(5, 4)
        self_t = torch.empty((5, 4))
        prob = torch.ones((5, 1))
        for cls, p in (
            (golden.AclnnBernoulliImpl, 0.5),
            (golden.AclnnBernoulliTensorImpl, prob),
        ):
            with self.subTest(cls=cls.__name__):
                with self.assertRaisesRegex(RuntimeError, OVERLAP_ERROR):
                    cls(self_t, p, out=out, device="cpu")
        for cls, p in (
            (golden.AclnnInplaceBernoulliImpl, 0.5),
            (golden.AclnnInplaceBernoulliTensorImpl, prob),
        ):
            with self.subTest(cls=cls.__name__):
                with self.assertRaisesRegex(RuntimeError, OVERLAP_ERROR):
                    cls(out, p, device="cpu")

    def test_xpu_adapter_preserves_legal_output_geometry(self):
        out = torch.empty(44).as_strided((4, 5), (11, 2), 1)
        impl = golden.AclnnBernoulliTensorImpl(
            torch.empty((4, 5)), torch.ones((4, 1)), out=out, device="cpu"
        )
        self.assertEqual(impl.out.stride(), out.stride())
        self.assertEqual(impl.out.storage_offset(), 1)
        self.assertNotEqual(impl.out.data_ptr(), out.data_ptr())
        result = impl(torch.empty((4, 5)), torch.ones((4, 1)))[0]
        self.assertTrue(torch.all(result == 1).item())

    def test_remote_output_allocator_preserves_view(self):
        request = {
            "shape": [875, 205],
            "out_strides": [0, 0],
            "out_storage_size": 1,
            "out_storage_offset": 0,
        }
        out = remote._allocate_output(request, torch.float16, device="cpu")
        self.assertEqual(out.stride(), (0, 0))
        self.assertEqual(out.untyped_storage().nbytes(), 2)
        with self.assertRaisesRegex(RuntimeError, OVERLAP_ERROR):
            out.bernoulli_(self.prob)

    @unittest.skipUnless(
        invoke_third_party is not None, "ops-test-kit required for binding test"
    )
    def test_xpu_registered_classes_bind_actual_api_parameter_names(self):
        for api in (
            "aclnnBernoulli",
            "aclnnBernoulliTensor",
            "aclnnInplaceBernoulli",
            "aclnnInplaceBernoulliTensor",
        ):
            with self.subTest(api=api):
                spec = getattr(golden, golden.__spec__[api])
                cls = spec.third_party["torch"]
                self_t = torch.empty((5, 4))
                named = {
                    "selfRef" if "Inplace" in api else "self": self_t,
                    "prob": torch.ones((5, 1)) if api.endswith("Tensor") else 1.0,
                }
                result = invoke_third_party(
                    cls, named, {"seed": 42, "offset": 0}, "cpu"
                )[0]
                self.assertEqual(result.shape, self_t.shape)
                self.assertTrue(torch.all(result == 1).item())
                target = torch.empty(1).expand(5, 4)
                named["selfRef" if "Inplace" in api else "out"] = target
                with self.assertRaisesRegex(RuntimeError, OVERLAP_ERROR):
                    invoke_third_party(cls, named, {"seed": 42, "offset": 0}, "cpu")

    def test_dense_golden_random_stream_unchanged(self):
        out = torch.empty_like(self.self_t)
        result = golden.aclnn_bernoulli_tensor_golden(
            self.self_t, self.prob, 53073689, 4, out=out
        )
        self.assertEqual(int((result == 0).sum()), 89545)
        self.assertEqual(int((result == 1).sum()), 89830)
        self.assertEqual(result.dtype, torch.float16)

    def test_numpy_write_target_overlap(self):
        out = np.broadcast_to(np.empty(1, np.float16), (4, 5))
        with self.assertRaisesRegex(RuntimeError, OVERLAP_ERROR):
            golden._check_writable_tensor(out)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    def test_cuda_rejects_original_output_layout(self):
        prob = self.prob.cuda()
        for mode in ("scalar", "tensor", "out"):
            with self.subTest(mode=mode):
                out = torch.empty(1, device="cuda", dtype=torch.float16).expand(
                    875, 205
                )
                with self.assertRaisesRegex(RuntimeError, OVERLAP_ERROR):
                    if mode == "out":
                        torch.bernoulli(prob.expand_as(out), out=out)
                    else:
                        out.bernoulli_(0.5 if mode == "scalar" else prob)
                    torch.cuda.synchronize()
                self.assertEqual(out.stride(), (0, 0))
                self.assertEqual(out.untyped_storage().nbytes(), 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    def test_remote_cuda_uses_original_output_view(self):
        request = {
            "mode": "tensor",
            "shape": [875, 205],
            "dtype": "float16",
            "seed": 53073689,
            "offset": 4,
            "prob_data": np.full((875, 1), 0.5, np.float16).tobytes(),
            "prob_dtype": "float16",
            "prob_shape": [875, 1],
            "out_strides": [0, 0],
            "out_storage_size": 1,
            "out_storage_offset": 0,
        }
        with self.assertRaisesRegex(RuntimeError, OVERLAP_ERROR):
            remote.gpu_golden_full(request)
        request.update(out_strides=[1, 875], out_storage_size=875 * 205)
        response = remote.gpu_golden_full(request)
        self.assertTrue(response["ok"])
        self.assertEqual(len(response["data"]), 875 * 205 * 2)


if __name__ == "__main__":
    unittest.main()
