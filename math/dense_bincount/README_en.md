# DenseBincount

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| --- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 training products/Atlas A3 inference products</term> | √ |
| <term>Atlas A2 training products/Atlas A2 inference products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | √ |

## Function Description

- **Operator function**: DenseBincount counts the number of occurrences or the weight sum of each bin row by row for one-dimensional or two-dimensional integer input.

- **Computation formula**:

  Let the unique element value in `size` be the number of bins `M`. For one-dimensional input, the output shape is `[M]`; for two-dimensional input, the output shape is `[N, M]`, where `N = input.shape[0]`. For a valid row number `r` and bin subscript `m`, the non-binary output is:

  $$
  output[r,m] = \sum_{i \in row(r)} \mathbf{1}(input_i=m) \times
  \begin{cases}
  weights_i, & numel(weights)>0 \\
  1, & numel(weights)=0
  \end{cases}
  $$

  When `binary_output` is `true`, the output is:

  $$
  output[r,m] = \mathbf{1}(\exists i \in row(r), input_i=m)
  $$

  Input values greater than or equal to `M` are ignored. Negative values in one-dimensional input are ignored; negative values in two-dimensional input are wrapped to the previous row according to the TensorFlow DenseBincount semantics, and are ignored if the row number or bin subscript is still out of bounds after wrapping. When `binary_output` is `true`, the values of `weights` are not used.

## Parameter Description

<table style="table-layout: fixed; width: 100%">
<colgroup>
<col style="width: 14%">
<col style="width: 14%">
<col style="width: 12%">
<col style="width: 34%">
<col style="width: 16%">
<col style="width: 10%">
</colgroup>
<thead>
  <tr>
    <th>Parameter Name</th>
    <th>Input/Output/Attribute</th>
    <th>Required</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Data Format</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>input</td>
    <td>Input</td>
    <td>Yes</td>
    <td>A one-dimensional or two-dimensional tensor of bin subscripts.</td>
    <td>INT32, INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>size</td>
    <td>Input</td>
    <td>Yes</td>
    <td>A scalar or a one-dimensional single-element tensor containing one non-negative constant value, where the element value indicates the number of bins. The data type must be the same as that of <code>input</code>.</td>
    <td>INT32, INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weights</td>
    <td>Input</td>
    <td>Yes</td>
    <td>The weight tensor. This input cannot be omitted; when no weights exist, pass a zero-element tensor to indicate counting by occurrence. When it is not empty, the number of elements must be the same as that of <code>input</code>, and elements correspond one to one by flattened subscripts.</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>binary_output</td>
    <td>Attribute</td>
    <td>No</td>
    <td>Whether to output only bin hit markers. The default value is <code>false</code>.</td>
    <td>Bool</td>
    <td>-</td>
  </tr>
  <tr>
    <td>output</td>
    <td>Output</td>
    <td>Yes</td>
    <td>The counting result. It corresponds to shape <code>[M]</code> for one-dimensional input, and to shape <code>[input.shape[0], M]</code> for two-dimensional input, where <code>M</code> is the unique element value in <code>size</code>.</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## Constraint Description

- The rank of `input` can only be 1 or 2.
- `size` must be a scalar or a one-dimensional ND tensor containing only one non-negative constant value, and its data type must be the same as that of `input`.
- `weights` cannot be omitted, and its rank cannot exceed 2. Pass a zero-element tensor when no weights exist. When it is not empty, the number of elements must be the same as that of `input`, elements correspond one by one in the flattened order, and the shape is not required to be the same as that of `input` dimension by dimension.
- <term>Ascend 950PR&950DT products</term>:
  - `weights` is a FLOAT32 ND tensor whose rank cannot exceed 2. A value of 0 for the number of elements indicates that weights are not used; when it is not empty, the number of elements must be the same as that of `input`.
  - `input` with any dimension of size 0 or with `M` of 0 is supported. When the shape of `input` is `[0]`, the output shape is `[M]`; when the shape is `[N, 0]`, the output shape is `[N, M]`; when the shape is `[0, N]` or `[0, 0]`, the output shape is `[0, M]`. When `M` is 0, the last dimension of the output has size 0.
  - The number of output elements `rows * M` cannot exceed the maximum number of FLOAT32 elements representable by INT64 indexes, and the storage required by the output tensor must be allocatable by the running environment. For one-dimensional input, `rows` is 1.
- <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>:
  - The empty Tensor handling boundary of the existing implementation is retained.
  - The rank of `weights` cannot exceed 2. A value of 0 for the number of elements indicates that weights are not used; when it is not empty, the number of elements must be the same as that of `input`, and the shape is not required to be the same as that of `input` dimension by dimension.
  - `M`, input dimension sizes, and the number of input elements must be within the INT32 range.
- This operator supports only the GE graph mode and the TensorFlow graph parsing path, and does not provide a public aclnn API.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
| --- | --- | --- |
| Graph-mode invocation | [test_geir_dense_bincount.cpp](./examples/test_geir_dense_bincount.cpp) | Invokes the DenseBincount operator by building a graph through the [operator IR](./op_graph/dense_bincount_proto.h). |
