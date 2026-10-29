#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""bernoulli GPU 远程 golden 服务（两方精度对拍：NPU 输出 vs GPU 真值 bit 级比对）

GPU 侧启动（对拍机器上）：
  python bernoulli_remote_golden.py --serve --port 9999

NPU 侧配套：golden.py 设 BERNOULLI_GOLDEN_HOST=<GPU_IP>（可选 PORT，默认 9999），
golden 直接取本服务 GPU 真值，ttk --compare binary 即两方比对；
未配置环境变量时 golden.py 走 CPU 复刻。

协议：4 字节大端长度 + pickle 载荷
  cmd=hello   → {"ok", "sm", "torch"}（客户端校验 SM 与标量 T 封顶一致）
  cmd=golden  → {"ok", "data": 原始字节, "dtype", "shape"}
"""

import argparse
import hashlib
import pickle
import socket
import struct
import sys
import time

import numpy as np


def gpu_golden_full(req):
    """在 GPU 上执行 bernoulli，返回完整输出原始字节（供 golden.py 远程模式逐 bit 比对）

    req: {"mode": "scalar"|"tensor", "shape": [...], "p": float,        # scalar 模式
          "prob_data": bytes, "prob_shape": [...], "prob_dtype": str,   # tensor 模式
          "seed": int, "offset": int, "dtype": str}
    """
    import torch

    mode, seed, offset = req["mode"], req["seed"], req["offset"]
    dtype_str, shape = req["dtype"], list(req["shape"])
    if mode == "scalar":
        print(
            f"[server] req: mode=scalar shape={shape} p={req.get('p')} seed={seed} offset={offset} dtype={dtype_str}"
        )
    else:
        print(
            f"[server] req: mode=tensor shape={shape} prob_shape={list(req.get('prob_shape', []))} "
            f"prob_dtype={req.get('prob_dtype')} prob_bytes={len(req.get('prob_data', b''))} "
            f"seed={seed} offset={offset} dtype={dtype_str}"
        )

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
    }
    # 竞品 CUDA bernoulli 的 dispatch 宏不含宽无符号整型（AT_DISPATCH_ALL_TYPES
    # 无 UInt16/32/64，运行时 not implemented）。但输出仅 0/1、随机流与输出
    # dtype 无关（u 生成与比较均在 fp32 域）——uint 系以 float32 代跑，返回前
    # numpy 侧 cast 到目标位宽，bit 级等价（torch 对 uint 宽整型算子支持不
    # 完整，不走 .to()）。
    # torch 无 uint16/32/64 dtype, 按字节宽度对齐选择代跑 dtype(而非统一 float32):
    #   uint16 -> float16(2B)、uint32 -> float32(4B)、uint64 -> int64(8B): 代跑宽度
    #     与真实输出宽度一致, 超界(n*elemSize > 2^31-1)时 TensorIterator 的切分/级联
    #     与 NPU 侧同式——若宽度不一致(如 uint16 用 float32 代跑), GPU 按 6GB 切 4 块
    #     而 NPU 按 3GB 切 2 块, 随机流全域错位(p=1.0 时被掩盖成少量临界位差异, 实测踩坑);
    #   代跑 dtype 的随机流与整型输出完全同构: 非 double 输出的 accscalar 恒为 float
    #     (DiscreteDistributionType 仅 double 特化为 double, int64/fp16 均走 float 路径),
    #     curand_uniform4/unroll=4/calc_execution_policy 均与输出 dtype 无关, 仅最后
    #     static_cast 目标不同, 0/1 值域经 numpy astype 无损——float64 不可用(其
    #     accscalar 为 double: 2^-53 缩放、curand_uniform2_double、unroll=2, 流语义不同)
    proxy_exec_dt = {"uint16": "float16", "uint32": "float32", "uint64": "int64"}
    uint_proxy = dtype_str in proxy_exec_dt
    dt = dtype_map.get(
        proxy_exec_dt[dtype_str] if uint_proxy else dtype_str, torch.float32
    )

    # 精度语义关键：rng state 按 seed + offset 设置（与竞品 philox_cuda_state 同构）
    torch.cuda.manual_seed(seed)
    st = torch.cuda.get_rng_state().clone()
    st[8:16] = torch.tensor(
        list(struct.pack("<Q", offset & 0xFFFFFFFFFFFFFFFF)), dtype=torch.uint8
    )
    torch.cuda.set_rng_state(st)

    if mode == "scalar":
        out = torch.empty(shape, device="cuda", dtype=dt).bernoulli_(float(req["p"]))
    else:
        np_dt = np.dtype(
            {
                "float32": "float32",
                "float16": "float16",
                "bfloat16": "uint16",
                "float64": "float64",
            }.get(req["prob_dtype"], "float32")
        )
        prob = torch.from_numpy(np.frombuffer(req["prob_data"], dtype=np_dt))
        if req["prob_dtype"] == "bfloat16":
            prob = prob.view(torch.bfloat16)
        prob = prob.reshape(list(req["prob_shape"])).cuda()
        out = torch.empty(shape, device="cuda", dtype=dt).bernoulli_(p=prob)

    tc = out.cpu()
    if uint_proxy:
        # float32 的 0.0/1.0 → 目标 uint 位宽（numpy astype，数值无损）
        raw = tc.numpy().astype(np.dtype(dtype_str)).tobytes()
    elif dtype_str == "bfloat16":
        raw = tc.view(torch.int16).numpy().tobytes()
    else:
        raw = tc.numpy().tobytes()
    ones = int(tc.to(torch.int64).sum().item()) if tc.numel() > 0 else 0
    print(
        f"[server] rsp: dtype={dtype_str} shape={shape} nbytes={len(raw)} "
        f"md5={hashlib.md5(raw).hexdigest()[:12]} ones={ones}"
    )
    return {"ok": True, "data": raw, "dtype": dtype_str, "shape": shape}


def _recv_all(conn, n):
    data = b""
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def _send_msg(conn, obj):
    msg = pickle.dumps(obj)
    conn.sendall(struct.pack(">I", len(msg)) + msg)


def _recv_msg(conn):
    raw = _recv_all(conn, 4)
    if raw is None:
        return None
    (ln,) = struct.unpack(">I", raw)
    return pickle.loads(_recv_all(conn, ln))


def run_server(args):
    import torch

    props = torch.cuda.get_device_properties(0)
    print(
        f"[server] GPU: {torch.cuda.get_device_name(0)} SM={props.multi_processor_count} "
        f"torch={torch.__version__} listening {args.bind}:{args.port}"
    )

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.bind, args.port))
    srv.listen(4)
    while True:
        conn, addr = srv.accept()
        print(f"[server] connected: {addr[0]} {time.strftime('%F %T')}")
        conn.settimeout(600)
        while True:
            try:
                req = _recv_msg(conn)
            except Exception:
                break
            if req is None:
                break
            cmd = req.get("cmd")
            if cmd == "hello":
                props = torch.cuda.get_device_properties(0)
                resp = {
                    "ok": True,
                    "sm": props.multi_processor_count,
                    "torch": torch.__version__,
                }
            elif cmd == "golden":
                try:
                    resp = gpu_golden_full(req)
                except Exception as e:
                    print(f"[server] FAILED: {type(e).__name__}: {e}")
                    resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            elif cmd == "shutdown":
                _send_msg(conn, {"ok": True})
                break
            else:
                resp = {"ok": False, "error": f"unknown cmd {cmd}"}
            _send_msg(conn, resp)
        conn.close()
        print("[server] disconnected")


def main():
    ap = argparse.ArgumentParser(description="bernoulli GPU 远程 golden 服务")
    ap.add_argument("--serve", action="store_true", help="启动服务端")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9999)
    args = ap.parse_args()
    if not args.serve:
        ap.error(
            "仅本地使用：GPU 侧 --serve 启动，NPU 侧经 golden.py 的 BERNOULLI_GOLDEN_HOST 调用"
        )
    if args.serve:
        run_server(args)


if __name__ == "__main__":
    sys.exit(main())
