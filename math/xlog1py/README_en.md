# Xlog1py

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

- Operator function: computes x * log(1 + y), and the result is 0 when x == 0.
- Computation formula:

$$
z_i =
\begin{cases}
0,                                & x_i = 0 \\
x_i \cdot \log(1 + y_i),          & x_i \neq 0
\end{cases}
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
      <td>x in the formula, which is the multiplier. The shape of x must satisfy the broadcast relationship with y.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Input</td>
      <td>y in the formula, which is the independent variable of log1p. The shape of y must satisfy the broadcast relationship with x.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>z</td>
      <td>Output</td>
      <td>z in the formula, which is the computation result. The shape is the maximum shape of x and y after broadcasting.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

- The input shape has a maximum of 8 dimensions.
- The data types of the inputs x and y and the output z must be the same.

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_xlog1py](./examples/arch35/test_aclnn_xlog1py.cpp) | Invokes the Xlog1py operator through the [aclnnXlog1py](./docs/aclnnXlog1py_en.md) API. |
| Graph-mode invocation | [test_geir_xlog1py](./examples/test_geir_xlog1py.cpp)   | Invokes the Xlog1py operator by building a graph through the [operator IR](./op_graph/xlog1py_proto.h). |
