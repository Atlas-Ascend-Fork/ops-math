# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Supplementary ACLNN tests. The official ATK matrix is run separately."""

import ctypes as ct
from contextlib import ExitStack
import os
import unittest

import torch
import torch_npu  # noqa: F401 - registers the NPU backend used by torch below.


class AclnnRunner:
    def __init__(self):
        self.lib = ct.CDLL(os.environ["ATK_CUSTOM_OPP_PATH"], mode=ct.RTLD_GLOBAL)
        self.acl = ct.CDLL("libopapi.so", mode=ct.RTLD_GLOBAL)
        self.runtime = ct.CDLL("libascendcl.so")
        self.runtime.aclGetRecentErrMsg.restype = ct.c_char_p
        ptr = ct.c_void_p
        i64p = ct.POINTER(ct.c_int64)
        self.acl.aclCreateTensor.argtypes = [
            i64p,
            ct.c_uint64,
            ct.c_int,
            i64p,
            ct.c_int64,
            ct.c_int,
            i64p,
            ct.c_uint64,
            ptr,
        ]
        self.acl.aclCreateTensor.restype = ptr
        self.acl.aclCreateScalar.argtypes = [ptr, ct.c_int]
        self.acl.aclCreateScalar.restype = ptr
        self.acl.aclDestroyTensor.argtypes = [ptr]
        self.acl.aclDestroyScalar.argtypes = [ptr]
        self.acl.aclSetAclOpExecutorRepeatable.argtypes = [ptr]
        self.acl.aclSetAclOpExecutorRepeatable.restype = ct.c_int
        self.acl.aclDestroyAclOpExecutor.argtypes = [ptr]
        self.stage1 = self.lib.aclnnFusedExponentialGetWorkspaceSize
        self.stage1.argtypes = [ptr, ptr, ptr, ct.POINTER(ct.c_uint64), ct.POINTER(ptr)]
        self.stage1.restype = ct.c_int
        self.stage2 = self.lib.aclnnFusedExponential
        self.stage2.argtypes = [ptr, ct.c_uint64, ptr, ptr]
        self.stage2.restype = ct.c_int

    def tensor(self, value):
        dtype = {
            torch.float32: 0,
            torch.float16: 1,
            torch.int32: 3,
            torch.float64: 11,
            torch.bfloat16: 27,
        }[value.dtype]
        shape = (ct.c_int64 * value.ndim)(*value.shape)
        strides = (ct.c_int64 * value.ndim)(*value.stride())
        storage = value.untyped_storage()
        storage_shape = (ct.c_int64 * 1)(storage.nbytes() // value.element_size())
        result = self.acl.aclCreateTensor(
            shape,
            value.ndim,
            dtype,
            strides,
            value.storage_offset(),
            2,
            storage_shape,
            1,
            storage.data_ptr(),
        )
        if not result:
            raise RuntimeError("aclCreateTensor failed")
        return result

    def scalar(self, value, dtype=11):
        result = self.acl.aclCreateScalar(ct.byref(value), dtype)
        if not result:
            raise RuntimeError("aclCreateScalar failed")
        return result

    def run(self, source, rate, output, expected_status=0):
        with ExitStack() as resources:
            u = self.tensor(source)
            resources.callback(self.acl.aclDestroyTensor, u)
            out = self.tensor(output)
            resources.callback(self.acl.aclDestroyTensor, out)
            scalar_value = ct.c_double(rate)
            scalar = self.scalar(scalar_value)
            resources.callback(self.acl.aclDestroyScalar, scalar)
            size, executor = ct.c_uint64(), ct.c_void_p()
            status = self.stage1(u, scalar, out, ct.byref(size), ct.byref(executor))
            if status == 0:
                resources.callback(self.acl.aclDestroyAclOpExecutor, executor)
                if self.acl.aclSetAclOpExecutorRepeatable(executor) != 0:
                    raise RuntimeError("Failed to retain executor for explicit cleanup")
            if status != expected_status:
                detail = self.runtime.aclGetRecentErrMsg()
                raise AssertionError(
                    f"GetWorkspaceSize: {status}, expected {expected_status}: {detail}"
                )
            if status != 0:
                return size.value
            workspace = torch.empty(size.value, dtype=torch.uint8, device=source.device)
            stream = torch.npu.current_stream(source.device)
            # Registered last: synchronize before descriptors are released, even on failure.
            resources.callback(stream.synchronize)
            status = self.stage2(
                workspace.data_ptr(), size.value, executor, stream.npu_stream
            )
            if status != 0:
                raise AssertionError(
                    f"Execute: {status}: {self.runtime.aclGetRecentErrMsg()}"
                )
            stream.synchronize()
            return size.value


class TestFusedExponential(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(20261014)
        torch.npu.set_device(int(os.environ.get("DEVICE", "0")))
        torch.zeros(1, device="npu")
        cls.runner = AclnnRunner()

    def check_case(self, source, rate=2.0, output=None):
        saved = source.cpu().clone()
        if output is None:
            output = torch.empty_like(source)
        workspace = self.runner.run(source, rate, output)
        epsilon = torch.finfo(source.dtype).eps / 2
        reference = -torch.log(torch.clamp(1 - saved.double(), max=1 - epsilon)) / rate
        if rate == float("inf"):
            reference.zero_()
        tolerance = {
            torch.float16: 2**-10,
            torch.bfloat16: 2**-7,
            torch.float32: 2**-13,
        }[source.dtype]
        torch.testing.assert_close(
            output.cpu(),
            reference.to(source.dtype),
            rtol=tolerance,
            atol=tolerance,
            equal_nan=True,
        )
        if source.is_contiguous() and output.is_contiguous():
            # The official memory gate is checked separately without changes.
            # CANN 9.1 composite executor aligns a zero-byte workspace to 512 B.
            self.assertLessEqual(workspace, 512, "unexpected tensor-sized temporary")
        return output

    def test_rank_dtype_empty_and_tail(self):
        shapes = [(), (0,), (3, 0, 7), (1,), (7,), (65,), (1025,), (4097,)]
        shapes += [(2,) * rank for rank in range(2, 9)]
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for shape in shapes:
                with self.subTest(dtype=dtype, shape=shape):
                    self.check_case((torch.rand(shape) * 0.99).to(dtype).npu())

    def test_views_inplace_and_determinism(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for layout in ("transpose", "slice", "expand"):
                with self.subTest(dtype=dtype, layout=layout):
                    base = (torch.rand(9, 17) * 0.99).to(dtype).npu()
                    source = {
                        "transpose": base.T,
                        "slice": base[1::2, 1::2],
                        "expand": base[:1].expand(5, 17),
                    }[layout]
                    storage = torch.full(
                        (*source.shape, 2), -7, dtype=dtype, device="npu"
                    )
                    result = self.check_case(source, output=storage[..., 1])
                    self.assertTrue(
                        torch.equal(
                            storage[..., 0].cpu(),
                            torch.full(source.shape, -7, dtype=dtype),
                        )
                    )
                    repeated = self.check_case(source)
                    self.assertTrue(torch.equal(result.cpu(), repeated.cpu()))
                    if layout != "expand":
                        self.check_case(source, output=source)

    def test_rank_eight_output_stride_order(self):
        shape = (2,) * 8
        permutations = (
            tuple(range(8)),
            tuple(reversed(range(8))),
            (3, 0, 7, 2, 5, 1, 6, 4),
        )
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            source = (torch.rand(shape) * 0.99).to(dtype).npu()
            for permutation in permutations:
                with self.subTest(dtype=dtype, permutation=permutation):
                    storage = torch.full((2,) * 7 + (4,), -7, dtype=dtype, device="npu")
                    output = storage[..., 1::2].permute(permutation)
                    self.check_case(source, output=output)
                    self.assertTrue(torch.all(storage[..., ::2].cpu() == -7).item())
            overlap = torch.as_strided(
                torch.empty(9, dtype=dtype, device="npu"), shape, (1,) * 8
            )
            self.runner.run(source, 1.0, overlap, expected_status=161002)

    def test_noncontiguous_partial_alias_uses_snapshot(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for layout in ("input", "output", "both"):
                with self.subTest(dtype=dtype, layout=layout):
                    base = (torch.rand(32) * 0.99).to(dtype).npu()
                    source = base[:12] if layout == "output" else base[:24:2]
                    output = base[:12] if layout == "input" else base[2:26:2]
                    self.check_case(source.reshape(3, 4), output=output.reshape(3, 4))

    def test_boundary_rates_and_nondefault_stream(self):
        with torch.npu.stream(torch.npu.Stream()):
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                eps = torch.finfo(dtype).eps / 2
                values = torch.tensor(
                    [0, eps / 2, eps, eps * 2, 0.5, 1 - 2**-10, 1], dtype=dtype
                ).npu()
                for rate in (0.002, 0.1, 1, 1e4, 1e-300, 1e300, float("inf")):
                    with self.subTest(dtype=dtype, rate=rate):
                        self.check_case(values, rate)

    def test_extreme_finite_double_rates(self):
        rates = (
            float.fromhex("0x0.0000000000001p-1022"),
            float.fromhex("0x1p-1026"),
            float.fromhex("0x1p-1025"),
            float.fromhex("0x1p-1022"),
            float.fromhex("0x1.fffffffffffffp+1023"),
        )
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            values = torch.tensor([0, 0.25, 0.5, 1], dtype=dtype).npu()
            for rate in rates:
                with self.subTest(dtype=dtype, rate=rate.hex()):
                    self.check_case(values, rate)

    def test_invalid_metadata(self):
        source = torch.zeros(2, device="npu")
        for output in (
            torch.empty(3, device="npu"),
            torch.empty(2, device="npu", dtype=torch.float16),
        ):
            self.runner.run(source, 1.0, output, expected_status=161002)
        source = torch.zeros((1,) * 9, device="npu")
        self.runner.run(source, 1.0, torch.empty_like(source), expected_status=161002)
        base = torch.zeros(18, device="npu")
        self.runner.run(base[:-1], 1.0, base[1:], expected_status=161002)
        self.runner.run(base[:4], 1.0, base[:1].expand(4), expected_status=161002)
        overlap = torch.as_strided(base, (2, 2), (1, 1))
        self.runner.run(
            torch.zeros((2, 2), device="npu"), 1.0, overlap, expected_status=161002
        )
        source = torch.zeros(2, dtype=torch.int32, device="npu")
        self.runner.run(source, 1.0, torch.empty_like(source), expected_status=161002)

    def test_null_parameters_and_scalar_type(self):
        source = torch.zeros(2, device="npu")
        output = torch.empty_like(source)
        with ExitStack() as resources:
            u = self.runner.tensor(source)
            resources.callback(self.runner.acl.aclDestroyTensor, u)
            out = self.runner.tensor(output)
            resources.callback(self.runner.acl.aclDestroyTensor, out)
            value = ct.c_double(1.0)
            scalar = self.runner.scalar(value)
            resources.callback(self.runner.acl.aclDestroyScalar, scalar)
            float_value = ct.c_float(1.0)
            wrong_scalar = self.runner.scalar(float_value, 0)
            resources.callback(self.runner.acl.aclDestroyScalar, wrong_scalar)
            for position in range(5):
                with self.subTest(null_argument=position):
                    size, executor = ct.c_uint64(), ct.c_void_p()
                    args = [u, scalar, out, ct.byref(size), ct.byref(executor)]
                    args[position] = None
                    self.assertEqual(self.runner.stage1(*args), 161001)
            size, executor = ct.c_uint64(), ct.c_void_p()
            self.assertEqual(
                self.runner.stage1(
                    u, wrong_scalar, out, ct.byref(size), ct.byref(executor)
                ),
                161002,
            )

    def test_torch_layer_rejects_invalid_rates(self):
        # The task assigns value-domain checks to the existing torch adapter,
        # not the raw ACLNN interface. This tests that layer independently.
        for rate in (0.0, -1.0, float("nan")):
            with self.subTest(rate=rate):
                with self.assertRaisesRegex(RuntimeError, "lambda|lambd|exponential"):
                    torch.empty(7, device="npu").exponential_(rate)


if __name__ == "__main__":
    unittest.main(verbosity=2)
