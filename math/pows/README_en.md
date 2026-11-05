# Pows

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
| <term>Kirin X90 processor series products</term>                         |    √     |
| <term>Kirin 9030 processor series products</term> | √ |

## Function Description

- Operator function: applies the power operation with the exponent to each element in input.
- Computation formula:

$$
out_i = self_i^{exponent_i}
$$

## Parameter Description

<table style="table-layout: fixed; width: 1030px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 290px">
  <col style="width: 370px">
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
      <td>Input `self` in the formula, an aclTensor on the Device side.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>exponent</td>
      <td>Input</td>
      <td>Input `exponent` in the formula, an aclScalar on the Device side.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output `out` in the formula, an aclTensor on the Device side.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Kirin X90/Kirin 9030 processor series products, Atlas training products, and Atlas inference products: BFLOAT16 is not supported.

## Constraint Description

None

## Invocation Description

| Invocation Method | Invocation Sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_pows](./examples/test_geir_pows.cpp)   | Invokes the Pows operator by building a graph through the [operator IR](./op_graph/pows_proto.h). |
