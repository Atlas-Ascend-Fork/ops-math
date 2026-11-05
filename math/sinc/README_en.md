# Sinc

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    x     |
| <term>Atlas A2 products</term> |    x    |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    ×    |
| <term>Atlas training products</term>                              |    ×     |

## Function Description

- Operator function: performs the sinc computation on the input Tensor.

- Computation formula:
  $$
  out_i =
  \begin{cases}
  1,\quad self_i = 0\\
  \sin(\pi*self_i)/(\pi*self_i),\quad otherwise\\
  \end{cases}
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
      <td>x</td>
      <td>Input</td>
      <td>The input tensor x in the formula.</td>
      <td>FLOAT, BFLOAT16, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>The output tensor y in the formula.</td>
      <td>FLOAT, BFLOAT16, FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

The input and output must have the same shape and type.

## Invocation Description

| Invocation Method | Invocation Sample                                              | Description                                                               |
|---------|---------------------------------------------------|------------------------------------------------------------------|
| aclnn invocation | [test_aclnn_sinc](./examples/test_aclnn_sinc.cpp) | Invokes the Sinc operator through the [aclnnSinc&aclnnInplaceSinc]( ./docs/aclnnSinc&aclnnInplaceSinc_en.md) API. |
| Graph-mode invocation | [test_geir_sinc](./examples/test_geir_sinc.cpp)   | Invokes the Sinc operator by building a graph through the [operator IR](./op_graph/sinc_proto.h).                   |
