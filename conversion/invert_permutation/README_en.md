# InvertPermutation

English | [简体中文](./README.md)

## Product Support Status

| Product                                                     | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: computes the inverse permutation of the input tensor x.

- Computation formula: y[x[i]] = i, where 0 <= i < len(x)

## Parameter Description

<table style="table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
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
      <td>Input tensor.</td>
      <td>INT32, INT64</td>
      <td>1D</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Inverse permutation of the input tensor x</td>
      <td>INT32, INT64</td>
      <td>1D</td>
    </tr>

  </tbody></table>

## Constraint Description

- None.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_invert_permutation](./examples/test_geir_invert_permutation.cpp)   | Invokes the invert_permutation operator by building a graph through the [operator IR](./op_graph/invert_permutation_proto.h). |
