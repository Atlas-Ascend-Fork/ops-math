# DiagFlat

English | [简体中文](./README.md)

## Product Support Status

| Product                                           | Supported |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT products</term>          |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>             |    ×     |
| <term>Atlas inference products</term>                       |    √     |
| <term>Atlas training products</term>                       |    √     |
| <term>Kirin X90 processor series products</term> | √ |
| <term>Kirin 9030 processor series products</term> | √ |

## Function Description

- Operator function: generates a diagonal tensor. If the input self is a one-dimensional tensor, a two-dimensional tensor is returned, where the elements of self are the diagonal values. If the input self is a tensor with two or more dimensions, it is first flattened (reduced to a one-dimensional tensor) and then processed as in the first scenario.

## Parameter Description

<table style="table-layout: fixed; width: 1043px"><colgroup>
<col style="width: 139px">
<col style="width: 146px">
<col style="width: 342px">
<col style="width: 320px">
<col style="width: 96px">
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
    <td>self</td>
    <td>Input tensor</td>
    <td>Vector to be filled into the diagonal. At most 8 dimensions are supported, and non-contiguous Tensors are supported.</td>
    <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX64, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>diagonal</td>
    <td>Optional attribute</td>
    <td>Specifies the diagonal position. The default value is 0. diagonal = 0 indicates the main diagonal, diagonal &gt; 0 indicates a diagonal above the main diagonal, and diagonal &lt; 0 indicates a diagonal below the main diagonal.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>out</td>
    <td>Output tensor</td>
    <td>Output diagonal tensor.</td>
    <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX64, BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

- Kirin X90/Kirin 9030 processor series products: the self and out data types do not support COMPLEX or BFLOAT16.
- Atlas training products and Atlas inference products: BFLOAT16 is not supported.

## Constraint Description

- When the input self is a one-dimensional tensor, the output is a two-dimensional square matrix with side length n = numel(self) + |diagonal|. When self is a tensor with two or more dimensions, it is first flattened and then processed as one-dimensional, and the output is also a two-dimensional square matrix.
- Empty Tensors (numel(self) = 0) are supported: if |diagonal| > 0, the output is an all-zero square matrix of |diagonal| × |diagonal|; if |diagonal| = 0, the output is 0 × 0.
- The value of diagonal must satisfy that numel(self) + |diagonal| does not exceed the upper limit of the int64 output size. If diagonal is INT64_MIN, the input is rejected as an invalid parameter.

## Invocation Description

| Invocation Method  | Sample Code                                                    | Description                                                         |
| --------- | ----------------------------------------------------------- | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_diag_flat](./examples/test_aclnn_diag_flat.cpp) | Invokes the DiagFlat operator through the [aclnnDiagFlat](docs/aclnnDiagFlat_en.md) API. |
| GE graph-mode invocation | - | Invokes the DiagFlat operator by building a graph through the [operator IR](./op_graph/diag_flat_proto.h). |
