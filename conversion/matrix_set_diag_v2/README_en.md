# MatrixSetDiagV2

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
|<term>Ascend 950PR&950DT products</term>|√|
|<term>Atlas A3 products</term>|×|
|<term>Atlas A2 products</term>|×|
|<term>Atlas 200I/500 A2 inference products</term>|×|
|<term>Atlas inference products</term>|×|
|<term>Atlas training products</term>|×|

## Function Description

- Operator function: replaces the diagonal elements of the input tensor with the values of the diagonal tensor.

## Parameter Description

<table style="table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>input</td>
      <td>Input</td>
      <td>Original tensor to be replaced.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, BOOL, COMPLEX64, DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>diagonal</td>
      <td>Input</td>
      <td>Diagonal tensor.</td>
      <td>Same as the data type of input.</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k</td>
      <td>Input</td>
      <td>Value range of diagonals.</td>
      <td>INT32.</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output</td>
      <td>Output</td>
      <td>Tensor after replacement.</td>
      <td>Same as the data type of input.</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- The number of dimensions of `input` is at least 2 and at most 8.
- `k` is a scalar or a vector of length 2 with the data type INT32.
- When `k` is a vector of length 2, `k[1] >= k[0]` must be satisfied.
- When `k` is a scalar or `k[0] = k[1]` (single diagonal), the number of dimensions of `diagonal` is 1 less than that of `input`: the length of the last dimension is `maxDiagLen = min(row + min(k[1], 0), col - max(k[0], 0))`, where `row` and `col` are the lengths of the last two dimensions of `input` respectively; the other dimensions are one-to-one equal to the corresponding dimensions of `input`.
- When `k` is not a scalar and `k[0] != k[1]` (multiple diagonals), the number of dimensions of `diagonal` is the same as that of `input`: the lengths of the last two dimensions are `numDiags = k[1] - k[0] + 1` and `maxDiagLen = min(row + min(k[1], 0), col - max(k[0], 0))` respectively, where `row` and `col` are the lengths of the last two dimensions of `input`, and `numDiags` is the number of diagonals; except for the last two dimensions, the other dimensions are one-to-one equal to the corresponding dimensions of `input`.

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| :-------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| Graph-mode invocation | [test_geir_matrix_set_diag_v2](examples/test_geir_matrix_set_diag_v2.cpp) | Invokes the MatrixSetDiagV2 operator by building a graph through the [operator IR](op_graph/matrix_set_diag_v2_proto.h). |
