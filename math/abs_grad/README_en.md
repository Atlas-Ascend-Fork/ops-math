# AbsGrad

English | [简体中文](./README.md)

## Product Support Status

| Product                                              | Supported |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT products</term>          |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>             |    ×     |
| <term>Atlas inference products</term>                       |    √     |
| <term>Atlas training products</term>                       |    √     |

## Function Description

- Operator function: computes the gradient of the absolute value function (Abs). It computes the downstream gradient z based on the forward input tensor y and the upstream gradient dy.

- Computation formula:

$$z_i=dy_i \times sign(y_i)$$

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
      <td>Input of the original Abs operator, which is y<sub>i</sub> in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>Input</td>
      <td>Upstream gradient of the output of the Abs operator, which is dy<sub>i</sub> in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>z</td>
      <td>Output</td>
      <td>Downstream gradient of the input of the Abs operator, which is z<sub>i</sub> in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas training products and Atlas inference products: BFLOAT16 is not supported.

## Constraint Description

- The data types of y, dy, and z must be the same.
- The shapes of all inputs and outputs cannot contain 0 (empty tensors are not supported).

## Third-Party Framework Compatibility

- Compatible with the AbsGrad operator of TensorFlow.

## Invocation Description

| Invocation Method | Invocation Sample | Description |
|--------------|--------|------|
| Graph-mode invocation | - | Invokes the AbsGrad operator by building a graph through the [operator IR](./op_graph/abs_grad_proto.h). |
