# SliceWithAxes

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                              |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: extracts slices from the input tensor along the specified axes, where offsets specifies the start position of each axis and size specifies the slice length of each axis.

- Computation example:
  - Input x shape = (10, 20, 30), axes = [0, 2], offsets = [1, 5], size = [5, 10]
  - Output y shape = (5, 20, 10)
  - y[i, j, k] = x[1 + i, j, 5 + k]

- Computation formula:
  The shape of the output y is the same as that of the input x, except that the dimensions specified in axes are replaced with the corresponding sizes. For the k-th axis axis_k in axes:

  ```text
  y[..., axis_k, ...] = x[..., offsets[k] : offsets[k] + size[k], ...]

  If size[k] == -1, the slice starts at offsets[k] and extends to the end of the axis, that is, size[k] = x.shape[axis_k] - offsets[k]
  ```

## Parameter Description

| Parameter Name  | Input/Output/Attribute | Description | Data Type | Data Format |
|--------|---------------|------|----------|----------|
| x      | Input          | Input tensor | INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, FLOAT, FLOAT16, BF16, BOOL | ND |
| offsets| Input          | Start position of the slice on each axis. This parameter is a 1D tensor whose length equals the length of axes | INT32, INT64 | ND |
| size   | Input          | Slice length of each axis. This parameter is a 1D tensor whose length equals the length of axes. The value -1 indicates that the slice starts at the offset and extends to the end of the axis | INT32, INT64 | ND |
| axes   | Attribute (Required)   | List of axes to be sliced | ListInt | - |
| y      | Output          | Output tensor, which has the same data type and format as x | Same as x | ND |

## Constraint Description

- The dimension of the input tensor x ranges from [1, 8].
- Each axis in axes must satisfy 0 <= axis < rank.
- For each sliced axis, offsets[k] >= 0 and offsets[k] + size[k] <= x.shape[axis_k] must be satisfied (if size[k] is -1, the slice automatically extends to the end of the axis).
- The size of each dimension of the output must be >= 0.
- Dynamic shape, dynamic rank, and dynamic compilation with static shapes are supported.

## Invocation Description

| Invocation Method | Sample Code | Description |
|----------|----------|------|
| Graph-mode invocation | [test_geir_slice_with_axes](examples/test_geir_slice_with_axes.cpp) | Invokes the SliceWithAxes operator by building a graph through the [operator IR](./op_graph/slice_with_axes_proto.h). |
