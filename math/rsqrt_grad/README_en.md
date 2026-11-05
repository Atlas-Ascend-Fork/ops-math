# RsqrtGrad

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

- Operator function: computes the backward gradient of the Rsqrt operator.

- Computation formula:

$$
z = -0.5 \times dy \times y^{3}
$$

Here, y is the forward output of the Rsqrt operator.

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
      <td>y</td>
      <td>Input</td>
      <td>Forward output of the Rsqrt operator, which is y in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT8, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>Input</td>
      <td>Output gradient, which is dy in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT8, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>z</td>
      <td>Output</td>
      <td>Input gradient, which is z in the formula. Its shape is the same as that of y.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT8, INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                      | Description                                                      |
|--------------|-----------------------------------------------------------|---------------------------------------------------------|
| Graph-mode invocation | [test_geir_rsqrt_grad](./examples/test_geir_rsqrt_grad.cpp) | Invokes the RsqrtGrad operator by building a graph through the [operator IR](./op_graph/rsqrt_grad_proto.h). |
