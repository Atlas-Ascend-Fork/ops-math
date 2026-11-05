# Xdivy

English | [简体中文](./README.md)

## Product Support Status

| Product                                              | Supported |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT products</term>          |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>             |    √     |
| <term>Atlas inference products</term>                       |    √     |
| <term>Atlas training products</term>                       |    √     |

## Function Description

- Operator function: computes x / y, and the result is 0 when x == 0.

- Computation formula:

  $$out = \begin{cases} 0 & \text{if } x = 0 \\ \frac{x}{y} & \text{otherwise} \end{cases}$$

- Supports broadcast: the shapes of x and y must satisfy the broadcast relationship, and the output shape is the maximum shape after broadcasting.

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
      <td>x1</td>
      <td>Input</td>
      <td>x in the formula, which is the dividend. The shape of x1 must satisfy the broadcast relationship with x2.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>y in the formula, which is the divisor. The shape of x2 must satisfy the broadcast relationship with x1.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>out in the formula, which is the computation result. The shape is the maximum shape of x1 and x2 after broadcasting.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- The input shape has a maximum of 8 dimensions.
- The data types of the inputs x1 and x2 and the output y must be the same.

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_xdivy](./examples/test_geir_xdivy.cpp)   | Invokes the Xdivy operator by building a graph through the [operator IR](./op_graph/xdivy_proto.h). |
