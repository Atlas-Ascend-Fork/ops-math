# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""AccumulateNV2 TestSpec：GEIR/Kernel、ACLNN。

数学公式: y = Σ x_i

精度策略:
- CPU Golden 真调用 ``tf.raw_ops.AccumulateNV2``（图模式——eager 下该 op 无 kernel，
  唯一执行路径），与三方竞品同 op 同路由；图运行时链顺序逐轮波动的特性见
  ``_raw_accumulate_nv2`` docstring，按统计意义解读判定。
- fp16 输入统一升精度到 fp32 累加后单次 cast 回（promote 域内同样真调用
  raw_ops）：kernel 直连路径在 Vector 管线内实现（Cast→Add(fp32)→Cast），
  aclnnSum 路径与 op_api 的 promoteFp16 路径一致。
- 三方结果 ``b`` 使用 TensorFlow XPU 原始 dtype 计算，不做 cast。
- L1 比较 NPU 相对 CPU 的误差与 TensorFlow XPU 相对 CPU 的误差之比。
- 容差按 dtype 分级声明于 TestSpec ``tolerance``（浮点 cross_check L1、整数 binary_equal L2）。
- Kernel/GEIR 直连与 TF E2E 为同 shape 契约（算子本体不支持 broadcast，不做对齐，
  异 shape 输入由竞品自身的拒绝语义暴露）；aclnnSum 路由由 op_api 的 BroadcastTo
  完成对齐（shape 可不同），golden 与三方参考同步该行为。三方分两个实现：
  ``ThirdPartyTF``（同 shape 直传，kernel/GEIR/TF E2E）与 ``ThirdPartyTFBroadcast``
  （numpy 对齐并回传原设备，覆盖 rank≤8 全域，aclnnSum 专用）。
"""

import numpy as np

__golden__ = {
    "kernel": {"accumulate_nv2": "accumulate_nv2_golden"},
    "aclnn": {"aclnnSum": "aclnn_sum_golden"},
}

__spec__ = {
    "accumulate_nv2": "AccumulateNv2KernelTestSpec",
    "AccumulateNV2": "AccumulateNv2KernelTestSpec",
    "aclnnSum": "AclnnSumTestSpec",
    "tf.raw_ops.AccumulateNV2": "TensorFlowAccumulateNv2TestSpec",
}


def _as_tensor_list(value):
    tensors = list(value) if isinstance(value, (list, tuple)) else [value]
    if not tensors:
        raise ValueError("AccumulateNV2 requires at least one input tensor")
    return tensors


def _is_fp16_inputs(values, input_dtypes):
    """Whether the case is fp16-origin.

    TTK passes the ORIGINAL ``input_dtypes`` (pre-elevation) to custom golden
    functions: under cross_check the framework promotes fp16 inputs to fp32
    before invoking the golden, so the array dtype alone is unreliable —
    ``input_dtypes`` is authoritative when present.
    """
    if input_dtypes is not None:

        def _flatten(items):
            for item in items:
                if isinstance(item, (list, tuple)):
                    yield from _flatten(item)
                else:
                    yield str(item)

        dtypes = list(_flatten(input_dtypes))
        return bool(dtypes) and dtypes[0] == "float16"
    return np.asarray(values[0]).dtype == np.float16


def _fp16_promote_reference(values):
    """fp16 promote reference: fp32-domain raw_ops accumulation, one cast back.

    Promote semantics for both invocation forms: raw fp16 inputs (cast up,
    accumulate in the fp32 domain, cast back once) and TTK's Promote-elevated
    fp32 inputs (fp16→fp32 is exact, so accumulate directly and cast back
    once) — both model the kernel's fp32 intermediate accumulation in the
    Vector pipeline (Cast→Add(fp32)→Cast) and the aclnnSum promoteFp16 path.
    The fp32-domain accumulation itself calls ``tf.raw_ops.AccumulateNV2``
    (the operator's own semantics, same op as the GPU third-party).
    """
    import tensorflow as tf

    tensors = [_to_tensorflow_tensor(v, tf) for v in _as_tensor_list(values)]
    promoted = [t if t.dtype == tf.float32 else tf.cast(t, tf.float32) for t in tensors]
    result = _raw_accumulate_nv2(promoted)
    return tf.cast(result, tf.float16)


def _to_tensorflow_tensor(value, tf):
    """Convert a CPU Golden input to TensorFlow without changing its dtype."""
    if tf.is_tensor(value):
        return value
    if hasattr(value, "detach") and hasattr(value, "__dlpack__"):
        value = value.detach().contiguous().clone()
        return tf.experimental.dlpack.from_dlpack(value.__dlpack__())
    return tf.convert_to_tensor(value)


def _to_kernel_array(value):
    """Convert a TensorFlow CPU Golden result to TTK's ndarray representation."""
    return value.numpy()


_RAW_GRAPH_FN = None


def _raw_accumulate_nv2(values):
    """TF reference calling the operator's own AccumulateNV2.

    ``tf.raw_ops.AccumulateNV2`` is a kernel-less stub in eager mode (fails
    with "<no registered kernels>"); it only executes in graph mode, where TF
    rewrites it via AccumulateNV2RemovePass (PRE_PLACEMENT) into a
    TemporaryVariable + AssignAdd chain — sequential accumulation at the
    input dtype. The reference therefore wraps the raw op in ``tf.function``.

    Known behavior (measured): the graph runtime is not bit-reproducible —
    repeated executions inside one process and executions across processes
    both reorder the chain (pinning inter/intra-op threads to 1 only fixes
    the within-process case), so small-value boundary cases may fluctuate
    run to run. This is the cost of calling the real operator; it is
    accepted deliberately to keep the golden source-symmetric with the GPU
    third-party (``ThirdPartyTF``), which runs the same op.
    """
    global _RAW_GRAPH_FN
    import tensorflow as tf

    if _RAW_GRAPH_FN is None:

        @tf.function
        def _graph_fn(vs, shape):
            return tf.raw_ops.AccumulateNV2(inputs=vs, shape=shape)

        _RAW_GRAPH_FN = _graph_fn
    tensors = [_to_tensorflow_tensor(v, tf) for v in _as_tensor_list(values)]
    shape = [int(d) for d in tensors[0].shape]
    return _RAW_GRAPH_FN(list(tensors), shape)


def accumulate_nv2_golden(x, **kwargs):
    """GEIR/Kernel CPU Golden.

    fp16: promote semantics — fp32 accumulation with a single cast back (see
    ``_fp16_promote_reference``), matching the kernel's fp32 intermediate
    accumulation. The fp16-origin test uses the original ``input_dtypes``
    kwarg (TTK elevates fp16 inputs to fp32 under cross_check's golden
    Promote). Other dtypes: ``tf.raw_ops.AccumulateNV2`` graph-mode reference
    (the operator's own semantics, same route as the GPU third-party).
    """
    tensors = _as_tensor_list(x)
    if _is_fp16_inputs(tensors, kwargs.get("input_dtypes")):
        return [_to_kernel_array(_fp16_promote_reference(tensors))]
    return [_to_kernel_array(_raw_accumulate_nv2(tensors))]


def tensorflow_accumulate_nv2_golden(inputs, shape=None, tensor_dtype=None, **kwargs):
    """TF E2E CPU Golden for tf.raw_ops.AccumulateNV2.

    Same-shape contract (the operator rejects mismatched shapes); fp16 uses
    the promote reference, consistent with the operator's fp32 intermediate
    accumulation contract; other dtypes call ``tf.raw_ops.AccumulateNV2``
    directly at the original dtype.
    """
    del shape, tensor_dtype, kwargs
    arrays = _as_tensor_list(inputs)
    if _is_fp16_inputs(arrays, None):
        return [_to_kernel_array(_fp16_promote_reference(arrays))]
    return [_to_kernel_array(_raw_accumulate_nv2(arrays))]


def aclnn_sum_golden(tensors, out=None, **kwargs):
    """ACLNN CPU Golden: mirrors aclnnSum's BroadcastTo + fp16→fp32 promote.

    numpy performs the BroadcastTo alignment (TF CPU's broadcast_to rejects
    rank>=7 shape pairs; np.broadcast_shapes/np.broadcast_to cover the full
    rank<=8 domain — pure shape replication, semantically identical). The
    aligned accumulation calls ``tf.raw_ops.AccumulateNV2`` (fp16 via the
    fp32 promote domain with one cast back, matching aclnn_sum.cpp
    promoteFp16; other dtypes at the original dtype).
    """
    del out, kwargs
    arrays = [np.asarray(v) for v in _as_tensor_list(tensors)]
    target_shape = np.broadcast_shapes(*[a.shape for a in arrays])
    arrays = [np.broadcast_to(a, target_shape) for a in arrays]
    if arrays[0].dtype == np.float16:
        return [_to_kernel_array(_fp16_promote_reference(arrays))]
    return [_to_kernel_array(_raw_accumulate_nv2(arrays))]


class ThirdPartyTF:
    """TensorFlow XPU competitor: AccumulateNV2 on the XPU server.

    ``tf.raw_ops.AccumulateNV2`` is a kernel-less stub op in every TF build
    (OPS_WITHOUT_KERNEL_ALLOWLIST): eager raw calls always fail with
    "<no registered kernels>". In graph mode TF rewrites the stub via
    AccumulateNV2RemovePass (accumulate_n_optimizer.cc, PRE_PLACEMENT) into
    a TemporaryVariable + AssignAdd chain — sequential accumulation at the
    original input dtype, TF's official execution semantics for this op.
    The tf.function wrapper below routes the competitor call through that
    graph path on the XPU server.

    Same-shape contract: inputs pass through untouched and stay on the
    dispatch device — no broadcast alignment is performed, so mixed-shape
    inputs surface the competitor's own rejection semantics. Serves the
    kernel/GEIR/TF E2E TestSpecs (the operator itself requires same-shape
    inputs on these routes).
    """

    def __init__(self):
        import tensorflow as tf

        self._tf = tf

        @tf.function
        def _accumulate_nv2(values, shape):
            return tf.raw_ops.AccumulateNV2(inputs=values, shape=shape)

        self._accumulate = _accumulate_nv2

    def __call__(self, inputs=None, x=None, tensors=None, **kwargs):
        del kwargs
        values = inputs if inputs is not None else x
        if values is None:
            values = tensors
        values = _as_tensor_list(values)
        # Same-shape inputs: the first input's shape is the common shape.
        # Mixed shapes reach the competitor unaligned and are rejected by
        # the op itself (no broadcast logic on this route).
        return [self._accumulate(list(values), list(values[0].shape))]


class ThirdPartyTFBroadcast(ThirdPartyTF):
    """aclnnSum-route competitor: broadcast-aligns inputs before the call.

    The aclnn interface layer supports broadcast (op_api BroadcastTo), so
    this route may pass broadcastable different-shaped inputs — the
    comparison target is "the correct sum result", not "the competitor's
    rejection behavior". numpy performs the alignment, covering the
    operator's full rank<=8 domain: TF's broadcast_to raises
    UnimplementedError for rank>=7 multi-axis expansion, an OpError
    subclass that escapes (ValueError, InvalidArgumentError) handlers.
    Aligned tensors are re-uploaded to the original dispatch device so
    the competitor computation stays on the XPU.
    """

    def __call__(self, inputs=None, x=None, tensors=None, **kwargs):
        del kwargs
        values = inputs if inputs is not None else x
        if values is None:
            values = tensors
        values = _as_tensor_list(values)
        # Shape-only inspection: no host round-trip, tensors stay on the
        # dispatch device (both TF tensors and numpy arrays expose .shape).
        shapes = [tuple(v.shape) for v in values]
        target_shape = tuple(np.broadcast_shapes(*shapes))
        if len(set(shapes)) > 1:
            # numpy alignment; re-upload to the original device so the
            # competitor still computes on the XPU.
            device = values[0].device if self._tf.is_tensor(values[0]) else None
            arrays = [np.broadcast_to(np.asarray(v), target_shape) for v in values]
            if device is not None:
                with self._tf.device(device):
                    values = [self._tf.convert_to_tensor(a) for a in arrays]
            else:
                values = [_to_tensorflow_tensor(a, self._tf) for a in arrays]
        return [self._accumulate(list(values), list(target_shape))]


_L1_FLOAT_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
}

_INTEGER_TOLERANCE = {
    "int32": {"standard": "binary_equal", "level": "L2"},
    "int8": {"standard": "binary_equal", "level": "L2"},
    "uint8": {"standard": "binary_equal", "level": "L2"},
}

_ACCUMULATE_NV2_TOLERANCE = {
    **_L1_FLOAT_TOLERANCE,
    **_INTEGER_TOLERANCE,
}


class AccumulateNv2KernelTestSpec:
    golden = staticmethod(accumulate_nv2_golden)
    third_party = {"tf": ThirdPartyTF}
    tolerance = _ACCUMULATE_NV2_TOLERANCE


class AclnnSumTestSpec:
    golden = staticmethod(aclnn_sum_golden)
    third_party = {"tf": ThirdPartyTFBroadcast}
    tolerance = _ACCUMULATE_NV2_TOLERANCE


class TensorFlowAccumulateNv2TestSpec:
    golden = staticmethod(tensorflow_accumulate_nv2_golden)
    third_party = {"tf": ThirdPartyTF}
    tolerance = _ACCUMULATE_NV2_TOLERANCE
