# StridedSliceAssignV2

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |
| <term>Kirin X90 processor series products</term>                          |    √     |
| <term>Kirin 9030 processor series products</term>                         |    √     |

## Function Description

- Operator function: StridedSliceAssignV2 is a tensor slice assignment operation, which assigns the content of the input tensor input_value to the slice position of the target tensor var specified by begin, end, and strides. The region outside the slice keeps the original value of var unchanged. The output var has the same name as the input var, which is in-place semantics, and the output shape is the same as the shape of the input var.

- Computation formula:

$$
out[b_0:e_0:s_0, b_1:e_1:s_1, \dots] = input\_value
$$

The length of the i-th dimension of the slice is $sliceLen[i] = \max(0, \lceil\frac{end[i] - begin[i]}{strides[i]}\rceil)$, where $\lceil x \rceil$ denotes the ceiling of $x$. $begin$ and $end$ are values after negative index normalization and boundary truncation: if $x < 0$, then $x = x + dim$. For a positive stride, the value is truncated to the $[0, dim]$ interval. For a negative stride (supported only by the Ascend 950 products), $begin$ is truncated to the $[0, dim-1]$ interval, and $end$ is truncated to the $[-1, dim]$ interval.

## Parameter Description

<table style="table-layout: fixed; width: 1162px"><colgroup>
<col style="width: 139px">
<col style="width: 148px">
<col style="width: 532px">
<col style="width: 267px">
<col style="width: 76px">
</colgroup>
<thead>
  <tr>
    <th>Parameter Name</th>
    <th>Input/Output/Attribute</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Data Format</th>
  </tr></thead>
<tbody>
  <tr>
    <td>var</td>
    <td>Input</td>
    <td>Target tensor for slice assignment.</td>
    <td>FLOAT16, FLOAT, BFLOAT16, INT32, INT64, DOUBLE, INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>input_value</td>
    <td>Input</td>
    <td>Tensor that replaces the slice content. Its dtype must be the same as that of var, and its shape must strictly equal the computed slice shape.</td>
    <td>FLOAT16, FLOAT, BFLOAT16, INT32, INT64, DOUBLE, INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>begin</td>
    <td>Input</td>
    <td>Start index of the slice position. This parameter is a 1D constant tensor whose length does not exceed the number of dimensions of var.</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>end</td>
    <td>Input</td>
    <td>End index of the slice position. This parameter is a 1D constant tensor whose length does not exceed the number of dimensions of var.</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>strides</td>
    <td>Input</td>
    <td>Step of the slice. This parameter is a 1D constant tensor whose length does not exceed the number of dimensions of var.</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>axes</td>
    <td>Input</td>
    <td>Optional parameter, which specifies the axes on which the slice takes effect. This parameter is a 1D constant tensor whose length is the same as that of begin.</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>var</td>
    <td>Output</td>
    <td>Result tensor after slice assignment. It has the same name as the input var, and its shape and dtype are the same as those of the input var.</td>
    <td>FLOAT16, FLOAT, BFLOAT16, INT32, INT64, DOUBLE, INT8</td>
    <td>ND</td>
  </tr>
</tbody></table>

## Constraint Description

- begin, end, and strides are value-dependent inputs, axes is an optional value-dependent input, and their dtypes are all INT64. Their concrete values must be obtainable on the host side.
- The input var supports 1D to 8D, and the rank of var must be the same as the rank of input_value. The Atlas A2/A3 and Kirin products require each dimension of var to be greater than 0. The Ascend 950 products allow var to contain zero-length dimensions (which are handled as empty-slice no-ops, and the output equals the input var).
- The lengths of begin, end, and strides must be the same, and must not exceed the number of dimensions of var. The length of axes must be the same as that of begin.
- The value of each dimension of strides must not be 0.
- The dtype of input_value must be the same as that of var, and its shape must strictly equal the shape after slice computation. Broadcasting is not supported.
- Each axis in axes must be in the range [-dim, dim), and axes must not contain duplicates. If axes is absent, the slice applies to the first len(begin) axes (equivalent to axes=[0, 1, ..., len(begin)-1]), and the remaining unlisted axes are treated as full-axis slices (begin=0, end=dim, strides=1).
- The output var and the input var are in-place semantics, and the output shape equals the shape of the input var.
- This operator is a pure data movement operator. The result is bit-exact, with no precision loss.
- Atlas A2/A3 and Kirin products: strides must be positive (negative strides are not supported), and the strides corresponding to the innermost dimension (the last dimension) must be 1.
- Ascend 950 products: strides can be negative, negative-stride reverse slicing is supported, and the strides of axes can be a mix of positive and negative values. These products do not require the innermost-dimension strides to be 1.
- Kirin X90/Kirin 9030 processor series products: BFLOAT16 is not supported.

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn invocation | [test_aclnn_strided_slice_assign_v2](./examples/test_aclnn_strided_slice_assign_v2.cpp) | Invokes the StridedSliceAssignV2 operator through the [aclnnStridedSliceAssignV2](./docs/aclnnStridedSliceAssignV2_en.md) API. |
| Graph-mode invocation | [test_geir_strided_slice_assign_v2](./examples/test_geir_strided_slice_assign_v2.cpp) | Invokes the StridedSliceAssignV2 operator by building a graph through the [operator IR](./op_graph/strided_slice_assign_v2_proto.h). |
