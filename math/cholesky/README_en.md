# Cholesky

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: computes the Cholesky decomposition of one or more real symmetric positive-definite matrices.
- When upper is false, out is lower triangular and satisfies:

  $$
  self = out \times out^{H}
  $$

- When upper is true, out is upper triangular and satisfies:

  $$
  self = out^{H} \times out
  $$

## Parameter Description

<table style="table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>self</td>
      <td>Input</td>
      <td>Input tensor, which is self in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output tensor, which is out in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>upper</td>
      <td>Attribute</td>
      <td>Optional attribute with a default value of false. true selects an upper-triangular result, and false selects a lower-triangular result.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraint Description

- The rank of self must be in [2, 8], the last two dimensions must be equal to form a square matrix, and the last dimension must not exceed 8192.
- self must be a real symmetric positive-definite matrix.

## Invocation Description

| Invocation Method | Sample Code                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn API | [test_aclnn_linalg_cholesky](./examples/test_aclnn_linalg_cholesky.cpp) | Invokes the Cholesky operator through the [aclnnLinalgCholesky](./docs/aclnnLinalgCholesky_en.md) API.    |
