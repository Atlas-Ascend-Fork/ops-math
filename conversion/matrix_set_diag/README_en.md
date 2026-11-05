# MatrixSetDiag

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| ---- | :----:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term>     | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

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
      <td>x</td>
      <td>Input</td>
      <td>Original tensor to be replaced.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, BOOL, COMPLEX64, DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>diagonal</td>
      <td>Input</td>
      <td>Diagonal tensor.</td>
      <td>Same as the data type of x.</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Tensor after replacement.</td>
      <td>Same as the data type of x.</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- The number of dimensions of x is at least 2 and at most 8.
- The number of dimensions of diagonal is 1 less than that of x, and the value of the last dimension is the smaller of the last two dimensions of x.
- Except for the last dimension, the other dimensions of diagonal must correspond one-to-one and be equal to the dimensions of x.

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_matrix_set_diag](./examples/test_geir_matrix_set_diag.cpp)   | Invokes the MatrixSetDiag operator by building a graph through the [operator IR](./op_graph/matrix_set_diag_proto.h). |
