# SliceLastDim

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                              |    ×     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: slices the last dimension of the input tensor, where start and end specify the slice range and stride specifies the step.

- Computation example:
  - Input x shape = (2, 3, 10), start = 0, end = 10, stride = 2
  - Output y shape = (2, 3, 5)
  - y[i, j, k] = x[i, j, start + k \* stride] = x[i, j, 2 \* k]

- Computation formula:
  The shape of the output y is the same as that of the input x, except that only the last dimension changes:

  ```text
  y.shape[-1] = ceil((end - start) / stride)
  y[..., k] = x[..., start + k * stride]
  ```

  Negative indices relative to the length of the last dimension are supported for start and end. start is truncated to [0, lastDim], and end is truncated to [0, lastDim].

## Parameter Description

| Parameter Name | Input/Output/Attribute | Description | Data Type | Data Format |
|--------|---------------|------|----------|----------|
| x      | Input          | Input tensor | INT8, INT16, INT32, INT64, FLOAT16, FLOAT, BF16 | ND |
| start  | Attribute (Required)   | Start index of the slice on the last dimension; negative indices are supported | INT | - |
| end    | Attribute (Required)   | End index of the slice on the last dimension; negative indices are supported | INT | - |
| stride | Attribute (Optional)   | Slice step; the default value is 1 and the value must be >= 1 | INT | - |
| y      | Output          | Output tensor, which has the same data type and format as x | Same as x | ND |

## Constraint Description

- The dimension of the input tensor x ranges from [1, 8].
- stride must be >= 1.
- Negative indices are supported for start and end. A negative index is converted to a positive index by adding the length of the last dimension. start is truncated to [0, lastDim], and end is truncated to [0, lastDim].
- If end <= start, the length of the last dimension of the output is 0.
- Dynamic shape, dynamic rank, and dynamic compilation with static shapes are supported.

## Invocation Description

| Invocation Method | Sample Code | Description |
|----------|----------|------|
| Graph-mode invocation | [test_geir_slice_last_dim](examples/test_geir_slice_last_dim.cpp) | Invokes the SliceLastDim operator by building a graph through the [operator IR](./op_graph/slice_last_dim_proto.h). |
