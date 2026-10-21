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


import torch
import numpy as np

__golden__ = {"kernel": {"sim_thread_exponential": "sim_thread_exponential_golden"}}

__spec__ = {
    "aclnnSimThreadExponential": "aclnnSimThreadExponentialSpec",
}

_GPU_SERVER_IP = "127.0.0.1"
_GPU_SERVER_PORT = 8888


class MaxPool3DGradGoldenGpuClient:
    def __init__(self, server_ip, server_port=8888):
        self.server_ip = server_ip
        self.server_port = server_port
        self._deps_loaded = False
        self._load_dependencies()

    def _load_dependencies(self):
        if not self._deps_loaded:
            global socket, pickle, struct, torch, np
            import socket
            import pickle
            import struct
            import torch
            import numpy as np

            self._deps_loaded = True

    def _recv_all(self, sock, n):
        data = b""
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def _send_msg(self, sock, msg):
        msg = pickle.dumps(msg)
        msg = struct.pack(">I", len(msg)) + msg
        sock.sendall(msg)

    def _recv_msg(self, sock):
        raw_msglen = self._recv_all(sock, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack(">I", raw_msglen)[0]
        return pickle.loads(self._recv_all(sock, msglen))

    def compute_on_gpu(self, attr_count, attr_seed, attr_offset, attr_lambd, dtype):
        request = {
            "attr_count": attr_count,
            "attr_seed": attr_seed,
            "attr_offset": attr_offset,
            "attr_lambd": attr_lambd,
            "dtype": dtype,
        }
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(3000)
                s.connect((self.server_ip, self.server_port))
                self._send_msg(s, request)
                result = self._recv_msg(s)
                return result
        except Exception as e:
            print(f"Connection error: {e}")


def sim_thread_exponential_golden(self, count, lambd=1.0, seed=0, offset=0, **kwargs):
    """
    Kernel golden for sim_thread_exponential.
    All the parameters follow @sim_thread_exponential_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    input_dtypes = kwargs.get("input_dtypes", [])
    dtype = input_dtypes[0] if input_dtypes else "float32"

    client = MaxPool3DGradGoldenGpuClient(_GPU_SERVER_IP, _GPU_SERVER_PORT)
    result = client.compute_on_gpu(count, seed, offset, lambd, dtype)

    return result


class AclnnSimThreadExponentialThirdPartyImpl:
    """对标 PyTorch：lambd 以 Python float(double) 直传 torch CUDA exponential_，host 侧按 double 校验。"""

    def __init__(self, *args, **kwargs):
        self_t, count, lambd, seed, offset = (list(args) + [None] * 5)[:5]
        self.count = int(count) if count is not None else self_t.numel()
        self.lambd = lambd if lambd is not None else 1.0
        self.seed = int(seed) if seed is not None else 0
        self.offset = int(offset) if offset is not None else 0
        self.dtype = self_t.dtype

    def __call__(self, **kwargs):
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(self.seed)
        default_gen = torch.cuda.default_generators[0]
        if hasattr(default_gen, "set_offset"):
            default_gen.set_offset(self.offset)
        return torch.empty(self.count, device=device, dtype=self.dtype).exponential_(
            self.lambd
        )


class aclnnSimThreadExponentialSpec:
    """ACLNN 流程 — golden / third_party 均收到 torch.Tensor（已在设备上）。"""

    def golden(self, count, lambd=1.0, seed=0, offset=0, **kwargs):
        output_dtypes = kwargs.get("output_dtypes") or kwargs.get("input_dtypes") or []
        dtype = output_dtypes[0] if output_dtypes else "float32"
        client = MaxPool3DGradGoldenGpuClient(_GPU_SERVER_IP, _GPU_SERVER_PORT)
        result = client.compute_on_gpu(int(count), int(seed), int(offset), lambd, dtype)
        if result is None:
            raise RuntimeError(
                "gpu golden server unavailable for aclnnSimThreadExponential"
            )
        tensor = (
            result
            if isinstance(result, torch.Tensor)
            else torch.from_numpy(np.asarray(result))
        )
        return tensor.reshape(tuple(self.shape))

    third_party = {"torch": AclnnSimThreadExponentialThirdPartyImpl}
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }
