# ReciprocalGrad

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √    |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: computes the gradient of the Reciprocal operator. `y = Reciprocal(x) = 1/x`, `dy` is the upstream gradient, and `z` is the gradient with respect to the input `x`.
- Computation formula:

$$
z = -y \times y \times dy
$$

## Parameter Description

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
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
      <td>y in the formula, the output of the forward Reciprocal operator.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>Input</td>
      <td>dy in the formula, the upstream gradient.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>z</td>
      <td>Output</td>
      <td>z in the formula, the gradient with respect to the input x.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas training products</term> and <term>Atlas inference products</term>: BFLOAT16 is not supported.

## Constraint Description

- The data types of y, dy, and z must be the same.
- The shapes of y, dy, and z must be the same.
- Only the ND data format is supported.
- The shape has no more than 8 dimensions.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
|---------|---------|------|
| Graph-mode invocation | [test_geir_reciprocal_grad](./examples/test_geir_reciprocal_grad.cpp) | Invokes the ReciprocalGrad operator by building a graph through the [operator IR](./op_graph/reciprocal_grad_proto.h). |
