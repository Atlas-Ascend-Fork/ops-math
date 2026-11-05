# Diag

English | [简体中文](./README.md)

## Product Support Status

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: expands the diagonal elements of the input tensor (flattened and viewed as 1D) into a 2D diagonal matrix

- Computation formula:

    Suppose:
    $$
    \mathbf{x} \in \mathbb{R}^n \text{ : input vector}
    $$
    $$
    \mathbf{y} \in \mathbb{R}^{n \times n} \text{ : output diagonal matrix}
    $$

    where:
    $$
    \mathbf{y}_{i,i} = x_i
    $$
    $$
    \mathbf{y}_{i,j} = 0 \text{ when } i \ne j
    $$

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
      <td>x in the formula.</td>
      <td>INT64, INT32, FLOAT, FLOAT16, DOUBLE, BF16, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>y in the formula.</td>
      <td>INT64, INT32, FLOAT, FLOAT16, DOUBLE, BF16, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas training products</term>, <term>Atlas inference products</term>, <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas A2 products</term>, and <term>Atlas A3 products</term>: BF16 is not supported.

## Constraint Description

Only input shapes with 1 to 4 dimensions are supported, which correspond to outputs with 2 to 8 dimensions. Scalar inputs are not supported.

## Invocation Description

| Invocation Method | Invocation Sample                                             | Description                                                                                         |
|---------|----------------------------------------------------|----------------------------------------------------------------------------------------------|
| Graph-mode invocation | NA  | Invokes the diag operator by building a graph through the [operator IR](./op_graph/diag_proto.h)
