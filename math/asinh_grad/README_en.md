# AsinhGrad

English | [简体中文](./README.md)

## Product Support Status

| Product                                                         | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                              |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function Description

- Operator function: computes asinh_grad(y, dy), that is, the gradient of the inverse hyperbolic sine function.
- Computation formula:

  $$
  dx = \frac{dy}{\cosh(y)}
  $$

## Parameter Description

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 400px">
<col style="width: 200px">
<col style="width: 170px">
</colgroup>
<thead>
  <tr>
    <th>Parameter Name</th>
    <th>Input/Output</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Data Format</th>
  </tr></thead>
<tbody>
  <tr>
    <td>y</td>
    <td>Input</td>
    <td>The input tensor of the forward computation, which corresponds to y in the formula and is the independent variable of the asinh function.</td>
    <td>FLOAT16, FLOAT, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dy</td>
    <td>Input</td>
    <td>The gradient tensor passed from upstream, which corresponds to dy in the formula. The data type must be exactly the same as that of y.</td>
    <td>FLOAT16, FLOAT, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dx</td>
    <td>Output</td>
    <td>The output input-gradient tensor, which corresponds to dx in the formula. The data type must be exactly the same as that of y.</td>
    <td>FLOAT16, FLOAT, BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## Constraint Description

None

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode  | [test_geir_asinh_grad.cpp](examples/test_geir_asinh_grad.cpp) | Invokes the AsinhGrad operator through the graph mode. |
