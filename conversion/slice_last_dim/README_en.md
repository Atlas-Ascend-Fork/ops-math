# SliceLastDim

English | [简体中文](./README.md)

## Product Support Status

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                              |    ×     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: slices the last dimension of the input tensor, with the slice range specified by `start` and `end` and the step by `stride`.

- Computation example:
  - Input x shape = (2, 3, 10), start = 0, end = 10, stride = 2
  - Output y shape = (2, 3, 5)
  - y[i, j, k] = x[i, j, start + k \* stride] = x[i, j, 2 \* k]

- Computation formula:
  The shape of the output y is the same as that of the input x, except that only the last dimension becomes:

  ```text
  y.shape[-1] = ceil((end - start) / stride)
  y[..., k] = x[..., start + k * stride]
  ```

  `start` and `end` support negative indexing (relative to the length of the last dimension); `start` is clamped to [0, lastDim], and `end` is clamped to [0, lastDim].

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
| :--- | :--- | :--- | :--- | :--- |
| x | Input | Input tensor. | INT8, INT16, INT32, INT64, FLOAT16, FLOAT, BFLOAT16 | ND |
| y | Output | Output tensor, with the same data type and format as `x`. | Same as `x` | ND |
| start | Attribute | Start index of the last-dimension slice; negative indexing is supported. | INT | - |
| end | Attribute | End index of the last-dimension slice; negative indexing is supported. | INT | - |
| stride | Optional Attribute | Slice step, which defaults to 1 and must be >= 1. | INT | - |

## Constraint Description

- The rank of the input tensor `x` is in the range [1, 8].
- `stride` must be >= 1; when the output is non-empty, `stride` must additionally satisfy stride < 2^40, otherwise it is rejected during the Host Tiling stage.
- `start` and `end` support negative indexing; a negative value is converted to a positive index by adding the length of the last dimension; `start` is clamped to [0, lastDim], and `end` is clamped to [0, lastDim].
- Empty-axis input is supported: when a non-last dimension of `x` is 0, or the slice result is empty (e.g. end <= start), the output is an empty tensor of the corresponding shape: the Host-side Tiling returns via a short-circuit path, and the Kernel directly skips the copy.
- Row-stride upper bound: the row stride of the continuous copy, i.e. (input last-dimension length - output last-dimension length) x the number of bytes of the data type, must be less than 2^40 bytes.
- Dynamic shape support: dynamic Shape (dimensions marked as -1) and dynamic-compile-static are supported: a non-last -1 dimension is propagated to the output as-is, and when the last dimension is -1, the output last dimension is inferred as -1. An unknown-rank input (shape marked as -2) is passed through with an unknown-rank output, and the final shape is derived at runtime from the actual input.

### Exception Validation

| Violation | Validation Stage | Error Behavior |
| :------- | :------- | :------- |
| The rank (number of dimensions) of `x` is not in [1, 8] | Graph-compilation inference / Host Tiling | Returns GRAPH_FAILED, presenting `EZ0012` and recording the shape-dim validation log |
| stride < 1, or stride >= 2^40 when the output is non-empty | Graph-compilation inference / Host Tiling | Returns GRAPH_FAILED, presenting `EZ0026` and recording the stride validation log |
| The data type of `x` is not in the supported set (INT8, INT16, INT32, INT64, FLOAT16, FLOAT, BFLOAT16) | Graph-compilation inference / Host Tiling | Returns GRAPH_FAILED, presenting `EZ0020` and recording the dtype validation log |

> Note: Error codes are wrapped by graph compilation or the running framework upon validation failure; different integration paths may only return GRAPH_FAILED while keeping the detailed logs. Some invalid inputs may be intercepted by framework-level checks before reaching operator validation (e.g. `E80012` on some paths for rank out-of-range), in which case the framework-level code takes precedence.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| :--- | :--- | :--- |
| Graph-mode invocation | [test_geir_slice_last_dim](examples/test_geir_slice_last_dim.cpp) | Invokes the SliceLastDim operator by building a graph through the [operator IR](./op_graph/slice_last_dim_proto.h). |
