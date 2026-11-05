# InvGrad

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: computes the backward gradient of the Inv (reciprocal) operator, corresponding to the gradient computation of the forward `y = 1 / original_x`.
- Operator formula:

  $$
  y_i = -1 \cdot x_i \cdot x_i \cdot grad_i
  $$

  where:
  - $x_i$ is the output of the forward Inv operator (that is, $1 / \mathrm{original\_x}_i$);
  - $grad_i$ is the gradient passed from upstream;
  - $y_i$ is the gradient of the original input $\mathrm{original\_x}_i$.

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
      <td>FLOAT16, FLOAT32, BFLOAT16, INT32, INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>Input</td>
      <td>Corresponding input gradient.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16, INT32, INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Derivative gradient of the input tensor x</td>
      <td>FLOAT16, FLOAT32, BFLOAT16, INT32, INT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_inv_grad](./examples/test_geir_inv_grad.cpp)   | Invokes the inv_grad operator by building a graph through the [operator IR](./op_graph/inv_grad_proto.h). |
